"""美股 donchian 模拟盘台账（paper trading）——与 A股台账（paper.py）并列但完全独立。

两个账户：donchian_qqq / donchian_spy，各 $50,000 起步。
策略：收盘价突破前 20 日最高价买、跌破前 10 日最低价卖（us_equities.research
的 Donchian 实现），t 收盘信号 → t+1 开盘成交，US 成本口径
（$0.005/股 min$1 + 卖出 0.5bps + 滑点 10bps，复用 research 的 EtfBacktester）。

每日流程（与 A股台账同机制）：
① 昨日待成交目标按今日真实开盘价成交（含成本，缺开盘价顺延）；
② 记录今日收盘信号为待成交；③ 收盘估值（前向填充价）追加净值。
幂等、非交易日跳过、多日缺跑逐日补记。
存储 paper_us/：account_{qqq,spy}.json + trades/nav/signals_{qqq,spy}.csv。

数据：新浪美股日线+复权因子。us_equities.data.load 的缓存不刷新、不适合
日更台账，因此本模块每日强制拉新（失败则降级用本地缓存并 WARN）。
"""
import json
import os
import sys
from datetime import datetime
from zoneinfo import ZoneInfo

import pandas as pd
import requests

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from us_equities.data import DAILY, FACTOR, parse, normalize  # noqa: E402
from us_equities.research.etf_strategies import donchian  # noqa: E402
from us_equities.research.etf_engine import EtfBacktester, DEFAULT_COST  # noqa: E402

TRACKED = {"qqq": "QQQ", "spy": "SPY"}
INITIAL_USD = 50_000.0
ENTRY, EXIT = 20, 10

TRADE_COLUMNS = ["date", "code", "side", "shares", "price", "amount", "fee",
                 "pnl", "win", "signal_date", "signal_close", "dev_bps"]
NAV_COLUMNS = ["date", "nav"]
SIGNAL_COLUMNS = ["date", "target", "reason"]


# ---------- 数据（每日强制刷新 + 缓存兜底） ----------

def _ny_today():
    return pd.Timestamp(datetime.now(ZoneInfo("America/New_York")).date())


def fetch_fresh(symbol, cache_dir):
    """从新浪拉最新数据并更新缓存。返回 df（截至最近已完成纽约交易日）。"""
    last_err = None
    for attempt in range(2):
        try:
            r = requests.get(DAILY, params={"symbol": symbol.lower(), "___qn": 3},
                             timeout=25)
            r.raise_for_status()
            f = requests.get(FACTOR.format(symbol), timeout=25)
            f.raise_for_status()
            df = parse(r.json(), f.text)
            df = df[df["date"] < _ny_today()]  # 只保留已完成纽约交易日
            os.makedirs(cache_dir, exist_ok=True)
            df.to_csv(os.path.join(cache_dir, f"{symbol}.csv"), index=False)
            return df
        except Exception as e:  # noqa: BLE001
            last_err = e
    raise RuntimeError(f"{symbol} 拉取失败: {last_err}")


def load_symbol(symbol, cache_dir):
    """先尝试刷新，失败则降级用缓存（WARN），缓存也没有才 raise。"""
    try:
        return fetch_fresh(symbol, cache_dir)
    except RuntimeError as e:
        path = os.path.join(cache_dir, f"{symbol}.csv")
        if os.path.exists(path):
            df = normalize(pd.read_csv(path))
            print(f"警告：数据源不可达，{symbol} 使用截至 "
                  f"{df['date'].max().date()} 的缓存（{e}）", file=sys.stderr)
            return df
        raise


def load_universe(paper_dir):
    """台账所需标的（QQQ/SPY）+ 基准（SPY）。"""
    cache_dir = os.path.join(paper_dir, "cache")
    return {sym: load_symbol(sym, cache_dir) for sym in set(TRACKED.values())}


# ---------- 状态与流水 ----------

def _state_path(paper_dir, name):
    return os.path.join(paper_dir, f"account_{name}.json")


def _csv_path(paper_dir, kind, name):
    return os.path.join(paper_dir, f"{kind}_{name}.csv")


def load_state(paper_dir, name):
    try:
        with open(_state_path(paper_dir, name), encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return {"cash": INITIAL_USD, "shares": {}, "cost_basis": {},
                "pending": None, "last_signal_date": None, "executed_target": {}}


def save_state(paper_dir, name, state):
    with open(_state_path(paper_dir, name), "w", encoding="utf-8") as f:
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

def update_account(data, name, symbol, paper_dir):
    """更新一个美股台账账户。data: {symbol: df}（只含已完成交易日）。"""
    state = load_state(paper_dir, name)
    state.setdefault("executed_target", {})
    actions = []

    signals = donchian(data, symbol, ENTRY, EXIT)
    d = data[symbol].set_index("date")
    adj_o, adj_c = d["open"] * d["factor"], d["close"] * d["factor"]
    calendar = adj_c.index
    aligned = (signals.reindex(calendar.union(signals.index)).sort_index().ffill()
               .reindex(calendar).fillna(0.0)) if not signals.empty else \
        pd.DataFrame(0.0, index=calendar, columns=[symbol])

    nav_df = read_csv(_csv_path(paper_dir, "nav", name))
    watermark = nav_df["date"].max() if not nav_df.empty else None
    new_dates = [calendar[-1]] if watermark is None else calendar[calendar > watermark]
    if len(new_dates) == 0:
        return [f"donchian_{name}: 无新交易日，跳过"]

    bt = EtfBacktester(cost=DEFAULT_COST, initial=INITIAL_USD)
    close_ffill = adj_c.ffill()
    for date in new_dates:
        deferred = False

        # ---- ① 执行上一待成交目标（当日真实开盘价） ----
        pend = state["pending"]
        if pend and date > pd.Timestamp(pend["signal_date"]):
            tgt = pend["target"]  # {symbol: 1.0} 或 {}（空仓）
            if tgt == state["executed_target"]:
                actions.append(f"donchian_{name}: {date:%m-%d} 目标未变，无需调仓")
            elif pd.isna(adj_o.get(date)) or adj_o[date] <= 0:
                deferred = True
                actions.append(f"donchian_{name}: {date:%Y-%m-%d} 开盘价缺失，待成交顺延")
            else:
                w = pd.Series({symbol: tgt.get(symbol, 0.0)})
                open_row = pd.Series({symbol: adj_o[date]})
                shares = {symbol: state["shares"].get(symbol, 0)}
                trades = []
                cash = bt._rebalance(date, w, open_row, state["cash"], shares, trades)
                for t in trades:
                    sig_close = pend["signal_close"].get(symbol)
                    dev = ((t["price"] / sig_close - 1) * 1e4
                           if sig_close and sig_close > 0 else None)
                    n = t["shares"]
                    if t["side"] == "buy":
                        old_n = state["shares"].get(symbol, 0)
                        old_cost = state["cost_basis"].get(symbol, 0.0)
                        state["cost_basis"][symbol] = (
                            (old_cost * old_n + t["price"] * n) / (old_n + n))
                        pnl, win = 0.0, None
                    else:
                        pnl = ((t["price"] - state["cost_basis"].get(symbol, t["price"]))
                               * n - t["fee"])
                        win = pnl > 0
                    _append_csv(_csv_path(paper_dir, "trades", name), {
                        "date": t["date"], "code": t["code"], "side": t["side"],
                        "shares": n, "price": round(t["price"], 4),
                        "amount": round(t["amount"], 2), "fee": round(t["fee"], 2),
                        "pnl": round(pnl, 2), "win": win,
                        "signal_date": pend["signal_date"],
                        "signal_close": sig_close,
                        "dev_bps": round(dev, 1) if dev is not None else None},
                        TRADE_COLUMNS)
                if shares.get(symbol, 0) == 0:
                    state["cost_basis"].pop(symbol, None)
                state["cash"] = cash
                state["shares"] = {c: q for c, q in shares.items() if q > 0}
                state["executed_target"] = dict(tgt)
                n_buy = sum(1 for t in trades if t["side"] == "buy")
                n_sell = sum(1 for t in trades if t["side"] == "sell")
                actions.append(f"donchian_{name}: {pend['signal_date']} 信号已按 "
                               f"{date:%Y-%m-%d} 开盘价成交（买{n_buy}笔/卖{n_sell}笔）")

        # ---- ② 记录当日信号为待成交（顺延时不覆盖旧目标） ----
        if not deferred:
            w_now = float(aligned.loc[date].get(symbol, 0.0))
            tgt_now = {symbol: w_now} if w_now > 0 else {}
            if date == calendar[-1]:
                holding = "持有" if w_now > 0 else "空仓"
                reason = (f"donchian {ENTRY}/{EXIT}：收盘价 {adj_c[date]:.2f}，"
                          f"当前状态：{holding} {symbol}")
            else:
                reason = "（补记历史交易日信号）"
            state["pending"] = {
                "signal_date": f"{date:%Y-%m-%d}", "target": tgt_now,
                "reason": reason,
                "signal_close": ({symbol: round(float(adj_c[date]), 4)} if tgt_now else {}),
            }
            state["last_signal_date"] = f"{date:%Y-%m-%d}"
            _append_csv(_csv_path(paper_dir, "signals", name),
                        {"date": f"{date:%Y-%m-%d}", "target": json.dumps(tgt_now),
                         "reason": reason}, SIGNAL_COLUMNS)
            actions.append(f"donchian_{name}: {date:%m-%d} 信号 "
                           f"{tgt_now or '空仓'}，待下一交易日开盘成交")

        # ---- ③ 收盘估值，追加净值 ----
        nav = state["cash"]
        for c, q in state["shares"].items():
            p = close_ffill.loc[date]
            if pd.notna(p):
                nav += q * p
        _append_csv(_csv_path(paper_dir, "nav", name),
                    {"date": f"{date:%Y-%m-%d}", "nav": round(nav, 2)}, NAV_COLUMNS)
        actions.append(f"donchian_{name}: 净值 ${nav:,.0f}（{date:%Y-%m-%d}）")

    save_state(paper_dir, name, state)
    return actions


# ---------- 报表 ----------

def _adjusted_frame(data, symbol):
    """研究引擎口径的复权日线（OHLC 已乘 factor），供理想净值使用。"""
    d = data[symbol].copy()
    for f in ("open", "high", "low", "close"):
        d[f] = d[f] * d["factor"]
    return d


def build_report(data, bench, paper_dir):
    """生成中文报表。返回 (text, nav_curves, ideal_curves)。"""
    import paper as a_paper  # 复用 A股台账的理想净值/绘图工具（只读调用）
    from metrics import calc_metrics

    lines = [f"===== 美股 donchian 模拟盘台账报表（{datetime.now():%Y-%m-%d %H:%M} 北京）====="]
    nav_curves, ideal_curves = {}, {}
    for name, symbol in TRACKED.items():
        state = load_state(paper_dir, name)
        nav_df = read_csv(_csv_path(paper_dir, "nav", name))
        trades_df = read_csv(_csv_path(paper_dir, "trades", name))
        lines.append(f"\n----- 策略：donchian_{name}（{symbol} {ENTRY}/{EXIT}） -----")
        if nav_df.empty:
            lines.append("台账为空，请先运行 paper-us-update 开户")
            continue
        nav = nav_df.set_index("date")["nav"].sort_index()
        nav_curves[f"donchian_{name}"] = nav

        d = data[symbol].set_index("date")
        adj_c = d["close"] * d["factor"]
        if state["shares"]:
            q = state["shares"][symbol]
            mv = q * adj_c.iloc[-1]
            lines.append(f"当前持仓：{symbol} {q}股（市值约 ${mv:,.0f}）；"
                         f"现金 ${state['cash']:,.0f}")
        else:
            lines.append(f"当前持仓：空仓；现金 ${state['cash']:,.0f}")
        pend = state["pending"]
        if pend:
            lines.append(f"待成交目标（{pend['signal_date']} 信号）："
                         f"{pend['target'] or '空仓'}")

        trades = trades_df.to_dict("records") if not trades_df.empty else []
        m = calc_metrics(nav, trades, initial=INITIAL_USD)
        sharpe_s = f"{m['夏普比率']:.2f}" if pd.notna(m["夏普比率"]) else "N/A"
        lines.append(f"累计收益 {m['总收益率']:+.2%}  年化 {m['年化收益率']:+.2%}"
                     f"  最大回撤 {m['最大回撤']:.2%}  夏普 {sharpe_s}"
                     f"  交易 {m['交易次数']} 笔")

        # 理想净值对比（信号当日收盘成交/无成本）
        adj_data = {symbol: _adjusted_frame(data, symbol)}
        signals = donchian(adj_data, symbol, ENTRY, EXIT)
        ideal = a_paper.ideal_nav_series(signals, adj_data, nav.index[0], INITIAL_USD)
        ideal_curves[f"donchian_{name}"] = ideal
        drag = nav.iloc[-1] / INITIAL_USD - 1 - (ideal.iloc[-1] / INITIAL_USD - 1)
        lines.append(f"理想净值（收盘成交/无成本）累计 {ideal.iloc[-1] / INITIAL_USD - 1:+.2%}"
                     f"  vs  台账 {nav.iloc[-1] / INITIAL_USD - 1:+.2%}  → 执行损耗 {drag:+.2%}")

        # 执行偏差
        lines.append("\n执行偏差（信号日收盘 → 实际开盘成交价，含滑点与隔夜跳空，bps）")
        if trades_df.empty or trades_df["dev_bps"].dropna().empty:
            lines.append("  暂无成交记录")
        else:
            for side, label in (("buy", "买入"), ("sell", "卖出")):
                g = trades_df[trades_df["side"] == side]["dev_bps"].dropna()
                if g.empty:
                    continue
                lines.append(f"  {label}: n={len(g)}  均值 {g.mean():+.1f}  "
                             f"中位 {g.median():+.1f}  P25 {g.quantile(0.25):+.1f}  "
                             f"P75 {g.quantile(0.75):+.1f}")

        monthly = nav.resample("ME").last().pct_change().dropna()
        lines.append("\n月度收益：")
        if monthly.empty:
            lines.append("  （净值数据不足一个月）")
        else:
            lines += [f"  {dt:%Y-%m}  {r:+.2%}" for dt, r in monthly.items()]
        if not trades_df.empty:
            lines.append("\n最近 10 笔成交：")
            for _, t in trades_df.tail(10).iterrows():
                dev_s = f"{t['dev_bps']:+.1f}bps" if pd.notna(t["dev_bps"]) else "N/A"
                lines.append(f"  {t['date']:%Y-%m-%d} {t['code']} "
                             f"{'买入' if t['side'] == 'buy' else '卖出'} {int(t['shares'])}股"
                             f" @ ${t['price']:.2f}（偏差 {dev_s}）")
    if bench is not None and nav_curves:
        start = min(n.index[0] for n in nav_curves.values())
        b = bench[bench.index >= start]
        if not b.empty:
            lines.append(f"\n同期基准 SPY 持有：{b.iloc[-1] / b.iloc[0] - 1:+.2%}")
    return "\n".join(lines), nav_curves, ideal_curves
