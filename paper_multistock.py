"""观察账户 8 号：M2 低波 top30 月调组合（paper_multistock）。

规则（R11 预声明原文，不得修改）：中证800 当前成分股 717 只池，上市≥250 交易日门槛，
每月首个交易日收盘按 60 日波动升序取 top30 等权；信号 T 收盘 → T+1 开盘成交；
个股万2.5+印花税0.05%+滑点0.1%；100股整手；涨跌停跳过顺延；账户 ¥1,000,000。
**幸存者偏差立场**：回测含偏差（当前成分股回填），本台账是无偏差前向证据。

机制复刻既有台账：幂等、无新交易日跳过、多日缺跑补记、非调仓日只估值不交易。
性能：调仓日全量刷新 717 只（qfq 尾部拉取+重叠区校验，不一致则全量重拉）；
非调仓日只刷新当前持仓（~30 只）。
存储 paper_multistock/（account/trades/nav/signals），文件缺失视为新账户。
"""
import json
import os
import sys
import time
from datetime import datetime

import numpy as np
import pandas as pd
import requests

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

NAME = "lowvol30"
INITIAL = 1_000_000.0
VOL_W, TOP_N, MIN_BARS = 60, 30, 250
LOT = 100
COMM, STAMP, SLIP = 0.00025, 0.0005, 0.001
TAIL_DAYS = 130          # 非全量刷新的尾部拉取窗口（覆盖 60 日波动窗口+缓冲）

TRADE_COLUMNS = ["date", "code", "side", "shares", "price", "amount", "fee",
                 "pnl", "win", "signal_date", "signal_close", "dev_bps"]
NAV_COLUMNS = ["date", "nav"]
SIGNAL_COLUMNS = ["date", "target", "reason"]


# ---------- 数据（腾讯源，尾部合并 + 重叠区校验） ----------

def _tx_symbol(code):
    return ("sh" if code.startswith(("5", "6")) else "sz") + code


def _fetch_tx(code, start, end=None):
    import akshare as ak
    end = end or pd.Timestamp.now().strftime("%Y-%m-%d")
    df = ak.stock_zh_a_hist_tx(symbol=_tx_symbol(code),
                               start_date=start.replace("-", ""),
                               end_date=end.replace("-", ""), adjust="qfq")
    if df is None or df.empty:
        raise ValueError(f"{code} 空数据")
    df = df.rename(columns={"date": "date"})
    df["date"] = pd.to_datetime(df["date"])
    df = df[["date", "open", "high", "low", "close", "volume"]].copy()
    for c in ("open", "high", "low", "close", "volume"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df.sort_values("date").drop_duplicates("date").reset_index(drop=True)


def refresh_code(code, cache_dir, full=False):
    """尾部刷新 ledger 缓存；重叠区不一致则全量重拉（防 qfq 接缝）。"""
    path = os.path.join(cache_dir, f"{code}.parquet")
    if not os.path.exists(path) or full:
        df = _fetch_tx(code, "2017-06-01")
        df.to_parquet(path, index=False)
        return df, "full"
    old = pd.read_parquet(path)
    tail_start = (old["date"].max() - pd.Timedelta(days=TAIL_DAYS)).strftime("%Y-%m-%d")
    new = _fetch_tx(code, tail_start)
    overlap = old.merge(new, on="date", suffixes=("_o", "_n"))
    if not overlap.empty:
        mismatch = (np.abs(overlap["close_o"] / overlap["close_n"] - 1) > 1e-4).mean()
        if mismatch > 0.2:  # qfq 基准漂移 → 全量重拉
            df = _fetch_tx(code, "2017-06-01")
            df.to_parquet(path, index=False)
            return df, "full_refetch(复权漂移)"
    merged = pd.concat([old[old["date"] < new["date"].min()], new], ignore_index=True)
    merged = merged.drop_duplicates("date").sort_values("date").reset_index(drop=True)
    merged.to_parquet(path, index=False)
    return merged, "tail"


def load_panel(codes, cache_dir):
    return {c: pd.read_parquet(os.path.join(cache_dir, f"{c}.parquet"))
            for c in codes if os.path.exists(os.path.join(cache_dir, f"{c}.parquet"))}


# ---------- 信号 ----------

def compute_target(data, date, vol_w=VOL_W, top_n=TOP_N, min_bars=MIN_BARS):
    """60 日波动升序 top30 等权（R11 原文）。返回 ({code: w}, n_eligible)。"""
    closes = {c: df.set_index("date")["close"] for c, df in data.items()}
    panel = pd.DataFrame(closes).sort_index()
    panel = panel[panel.index <= pd.Timestamp(date)]
    if len(panel) < min_bars:
        return {}, 0
    vol = panel.pct_change().rolling(vol_w).std() * np.sqrt(252)
    i = len(panel) - 1
    elig = panel.columns[(panel.notna().cumsum().iloc[i] >= min_bars)
                         & panel.iloc[i].notna() & vol.iloc[i].notna()]
    if len(elig) < top_n:
        return {}, len(elig)
    winners = vol.iloc[i][elig].nsmallest(top_n).index
    return {c: 1.0 / top_n for c in winners}, len(elig)


def is_rebalance_day(calendar, date):
    """每月首个交易日（相位=1，R11 生产口径）。"""
    dates = pd.DatetimeIndex(calendar)
    firsts = set(pd.Series(dates).groupby([dates.year, dates.month]).nth(0))
    return pd.Timestamp(date) in firsts


# ---------- 成交（与 R11 引擎同规则） ----------

def _limit_pct(code):
    return 0.20 if code.startswith(("30", "68")) else 0.10


def _execute_target(state, tgt, date, open_row, prev_close, paper_dir, actions):
    """按开盘价把组合调到目标权重；涨跌停/无开盘价顺延到 state['blocked']。
    返回 True=调仓完成（可清 pending），False=仍有顺延/缺价。"""
    cash, shares = state["cash"], state["shares"]
    nav_open = cash + sum(n * open_row.get(c) for c, n in shares.items()
                          if n > 0 and pd.notna(open_row.get(c)))
    deferred = False
    desired = {}
    for c, w in tgt.items():
        p = open_row.get(c)
        desired[c] = int(w * nav_open / (p * (1 + SLIP)) / LOT) * LOT \
            if pd.notna(p) and p > 0 else None
    blocked = state.get("blocked", [])
    retry_codes = {o["code"] for o in blocked}
    new_blocked = []

    def do_sell(c, n):
        p = open_row.get(c)
        if pd.isna(p) or p <= 0:
            new_blocked.append({"code": c, "side": "sell", "shares": n})
            return
        lim = _limit_pct(c)
        pc = prev_close.get(c)
        if pc is not None and pd.notna(pc) and p <= pc * (1 - lim + 0.001):
            new_blocked.append({"code": c, "side": "sell", "shares": n})
            return
        price = p * (1 - SLIP)
        amount = n * price
        fee = amount * (COMM + STAMP)
        cost = state["cost_basis"].get(c, price)
        pnl = (price - cost) * n - fee
        _append_csv(_csv_path(paper_dir, "trades"), {
            "date": date, "code": c, "side": "sell", "shares": n,
            "price": round(price, 4), "amount": round(amount, 2), "fee": round(fee, 2),
            "pnl": round(pnl, 2), "win": pnl > 0,
            "signal_date": state["pending"]["signal_date"] if state.get("pending") else "",
            "signal_close": "", "dev_bps": ""}, TRADE_COLUMNS)
        nonlocal_cash[0] += amount - fee
        shares[c] = shares[c] - n
        if shares[c] <= 0:
            shares.pop(c, None)
            state["cost_basis"].pop(c, None)

    nonlocal_cash = [cash]
    # 先卖（顺延订单 + 超出目标部分）
    for od in [o for o in blocked if o["side"] == "sell"]:
        do_sell(od["code"], od["shares"])
    for c in list(shares):
        if c in retry_codes:
            continue
        cur = shares[c]
        want = desired.get(c, 0) or 0
        if cur > want:
            do_sell(c, cur - want)

    # 后买
    buys = []
    for od in [o for o in blocked if o["side"] == "buy"]:
        buys.append((od["code"], od["shares"]))
    for c, want in desired.items():
        if c in retry_codes:
            continue
        if want is None:
            # 无开盘价：不挂 0 股垃圾单，整只延期（pending 保留，次日重算目标股数）
            deferred = True
            continue
        cur = shares.get(c, 0)
        if want > cur:
            buys.append((c, want - cur))
    # 涨停检查 + 现金缩仓
    real_buys = []
    for c, n in buys:
        p = open_row.get(c)
        if pd.isna(p) or p <= 0:
            new_blocked.append({"code": c, "side": "buy", "shares": n})
            continue
        lim = _limit_pct(c)
        pc = prev_close.get(c)
        if pc is not None and pd.notna(pc) and p >= pc * (1 + lim - 0.001):
            new_blocked.append({"code": c, "side": "buy", "shares": n})
            continue
        real_buys.append((c, n, p))
    need = sum(n * p * (1 + SLIP) * (1 + COMM) for c, n, p in real_buys)
    ratio = min(1.0, nonlocal_cash[0] / need) if need > 0 else 1.0
    for c, n, p in real_buys:
        n2 = int(n * ratio / LOT) * LOT
        if n2 <= 0:
            continue
        price = p * (1 + SLIP)
        amount = n2 * price
        fee = amount * COMM
        old_n = shares.get(c, 0)
        old_cost = state["cost_basis"].get(c, 0.0)
        state["cost_basis"][c] = (old_cost * old_n + price * n2) / (old_n + n2)
        sig_close = (state["pending"]["signal_close"].get(c)
                     if state.get("pending") else None)
        dev = (price / sig_close - 1) * 1e4 if sig_close else None
        _append_csv(_csv_path(paper_dir, "trades"), {
            "date": date, "code": c, "side": "buy", "shares": n2,
            "price": round(price, 4), "amount": round(amount, 2), "fee": round(fee, 2),
            "pnl": 0.0, "win": None,
            "signal_date": state["pending"]["signal_date"] if state.get("pending") else "",
            "signal_close": sig_close,
            "dev_bps": round(dev, 1) if dev is not None else None}, TRADE_COLUMNS)
        nonlocal_cash[0] -= amount + fee
        shares[c] = old_n + n2
        # 现金不足残额放弃（缩仓即完成，不顺延）
    state["cash"] = nonlocal_cash[0]
    state["blocked"] = new_blocked
    if real_buys or blocked or deferred:
        actions.append(f"{NAME}: {date:%Y-%m-%d} 调仓执行（买 {len(real_buys)} 只，"
                       f"顺延 {len(new_blocked)} 笔"
                       f"{'，含缺价延期' if deferred else ''}）")
    return (not new_blocked) and (not deferred)


# ---------- 状态与流水 ----------

def _state_path(paper_dir):
    return os.path.join(paper_dir, f"account_{NAME}.json")


def _csv_path(paper_dir, kind):
    return os.path.join(paper_dir, f"{kind}_{NAME}.csv")


def load_state(paper_dir):
    try:
        with open(_state_path(paper_dir), encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return {"cash": INITIAL, "shares": {}, "cost_basis": {}, "blocked": [],
                "pending": None, "last_signal_date": None}


def save_state(paper_dir, state):
    with open(_state_path(paper_dir), "w", encoding="utf-8") as f:
        json.dump(state, f, ensure_ascii=False, indent=2)


def _append_csv(path, row, columns):
    pd.DataFrame([row], columns=columns).to_csv(
        path, mode="a", header=not os.path.exists(path), index=False)


def read_csv(path):
    try:
        return pd.read_csv(path, parse_dates=["date"])
    except (OSError, ValueError):
        return pd.DataFrame()


# ---------- 每日更新 ----------

def load_universe_codes(project_dir):
    uni = pd.read_csv(os.path.join(project_dir, "research_r11", "results",
                                   "universe.csv"), dtype={"code": str})
    return uni["code"].str.zfill(6).tolist()


def refresh_data(codes, cache_dir, progress=False):
    """逐只尾部刷新（重叠区校验）。返回失败清单。"""
    failed = []
    for i, code in enumerate(codes, 1):
        try:
            refresh_code(code, cache_dir)
        except Exception as e:  # noqa: BLE001
            failed.append((code, str(e)[:60]))
        if progress and (i % 100 == 0 or i == len(codes)):
            print(f"  数据刷新 {i}/{len(codes)}（失败 {len(failed)}）", flush=True)
        time.sleep(0.2)
    return failed


def update_account(paper_dir, data, today=None):
    """data: {code: df}（当前持仓+调仓所需的全池数据）。today 可注入（测试用）。"""
    state = load_state(paper_dir)
    actions = []
    cache_dir = os.path.join(paper_dir, "cache")
    os.makedirs(cache_dir, exist_ok=True)

    calendar = sorted(set().union(*[set(df["date"]) for df in data.values()]))
    calendar = pd.DatetimeIndex(calendar)
    nav_df = read_csv(_csv_path(paper_dir, "nav"))
    watermark = nav_df["date"].max() if not nav_df.empty else None
    # 新账户只处理最新一天；若当月调仓日已过，空仓等待下一调仓日（不伪造历史持仓）
    new_dates = [calendar[-1]] if watermark is None else list(calendar[calendar > watermark])
    if len(new_dates) == 0:
        return [f"{NAME}: 无新交易日，跳过"]

    for date in new_dates:
        # ① 执行待成交目标/顺延订单（当日开盘价）
        if (state["pending"] and date > pd.Timestamp(state["pending"]["signal_date"])) \
                or state.get("blocked"):
            open_row = {c: df.set_index("date")["open"].get(date) for c, df in data.items()}
            prev_close = {c: df.set_index("date")["close"].shift(1).get(date)
                          for c, df in data.items()}
            done = _execute_target(state,
                                   state["pending"]["target"] if state["pending"] else {},
                                   date, open_row, prev_close, paper_dir, actions)
            if done and not state["blocked"]:
                state["pending"] = None

        # ② 调仓日：计算新目标
        if is_rebalance_day(calendar, date):
            tgt, n_elig = compute_target(data, date)
            if tgt:
                sig_close = {c: round(float(df.set_index("date")["close"].loc[date]), 4)
                             for c, df in data.items() if c in tgt}
                state["pending"] = {"signal_date": f"{date:%Y-%m-%d}", "target": tgt,
                                    "signal_close": sig_close}
                state["last_signal_date"] = f"{date:%Y-%m-%d}"
                _append_csv(_csv_path(paper_dir, "signals"),
                            {"date": f"{date:%Y-%m-%d}", "target": json.dumps(tgt),
                             "reason": f"低波 top{TOP_N}（合格池 {n_elig} 只）"},
                            SIGNAL_COLUMNS)
                actions.append(f"{NAME}: {date:%Y-%m-%d} 调仓日，目标 {len(tgt)} 只，"
                               f"待次日开盘成交")

        # ③ 收盘估值
        nav = state["cash"]
        for c, n in state["shares"].items():
            d = data.get(c)
            if d is not None:
                p = d.set_index("date")["close"].get(date)
                if pd.notna(p):
                    nav += n * p
        _append_csv(_csv_path(paper_dir, "nav"),
                    {"date": f"{date:%Y-%m-%d}", "nav": round(nav, 2)}, NAV_COLUMNS)
        actions.append(f"{NAME}: 净值 {nav:,.0f}（{date:%Y-%m-%d}）"
                       + ("（调仓日）" if is_rebalance_day(calendar, date) else ""))

    save_state(paper_dir, state)
    return actions
