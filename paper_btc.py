"""BTC×F1 模拟盘台账（观察账户，与 A股/美股台账并列、完全独立）。

规则（docs/R4_CRYPTO_DIVIDEND.md 判定"挂模拟观察"）：
BTC 唐奇安 20/10 + F1 量能确认（突破日成交量 > 前 20 日均量 ×1.5 才入场），
币安日线 UTC 收盘（北京 08:00）出信号 → 下一根 K 线开盘成交，
taker 10bps + 滑点 10bps，账户 $50,000。

机制复刻 A股台账：幂等、非交易日不存在（365 日历）、多日缺跑逐日补记、
dev_bps 执行偏差统计。存储 paper_btc/（JSON + CSV），文件缺失视为新账户。
"""
import json
import os
import sys
from datetime import datetime

import pandas as pd
import requests

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from macro_research.data_layer import load_btc  # noqa: E402
from research_r4.runner import btc_donchian_filtered, f1_volume_mask  # noqa: E402

TRACKED_NAME = "btc_f1"
INITIAL_USD = 50_000.0
TAKER = 0.001     # 佣金（双边）
SLIP = 0.001      # 滑点（单边）
LOT = 1e-4

TRADE_COLUMNS = ["date", "code", "side", "shares", "price", "amount", "fee",
                 "pnl", "win", "signal_date", "signal_close", "dev_bps"]
NAV_COLUMNS = ["date", "nav"]
SIGNAL_COLUMNS = ["date", "target", "reason"]


# ---------- 数据 ----------

def load_btc_with_today_open():
    """已完成 K 线（至昨日）+ 进行中 K 线的开盘价（开盘即定，可安全用于成交）。"""
    df, _note = load_btc()
    r = requests.get("https://data-api.binance.vision/api/v3/klines",
                     params={"symbol": "BTCUSDT", "interval": "1d", "limit": 2},
                     timeout=25)
    r.raise_for_status()
    last = r.json()[-1]
    today_date = pd.to_datetime(last[0], unit="ms").normalize()
    return df, (today_date, float(last[1]))


# ---------- 状态与流水 ----------

def _state_path(paper_dir):
    return os.path.join(paper_dir, f"account_{TRACKED_NAME}.json")


def _csv_path(paper_dir, kind):
    return os.path.join(paper_dir, f"{kind}_{TRACKED_NAME}.csv")


def load_state(paper_dir):
    try:
        with open(_state_path(paper_dir), encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return {"cash": INITIAL_USD, "shares": 0.0, "cost_basis": 0.0,
                "pending": None, "last_signal_date": None, "executed": False}


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


# ---------- 成交 ----------

def _execute(state, exec_date, exec_open, paper_dir, actions, tag=""):
    """按开盘价执行 pending 目标（含 taker + 滑点）。返回是否成交。"""
    pend = state["pending"]
    if not pend or state["executed"]:
        return False
    tgt = pend["target"].get("BTC", 0.0)
    sig_close = pend["signal_close"].get("BTC")
    shares, cash = state["shares"], state["cash"]
    if tgt > 0 and shares <= 0:  # 买入（全仓）
        price = exec_open * (1 + SLIP)
        n = int(cash / (price * (1 + TAKER)) / LOT) * LOT
        if n <= 0:
            return False
        amount = n * price
        fee = amount * TAKER
        state["cost_basis"] = price
        state["shares"], state["cash"] = n, cash - amount - fee
        side, pnl, win = "buy", 0.0, None
    elif tgt == 0 and shares > 0:  # 卖出（清仓）
        price = exec_open * (1 - SLIP)
        n = shares
        amount = n * price
        fee = amount * TAKER
        pnl = (price - state["cost_basis"]) * n - fee
        win = pnl > 0
        state["cash"], state["shares"], state["cost_basis"] = cash + amount - fee, 0.0, 0.0
        side = "sell"
    else:
        state["executed"] = True
        return False
    dev = (price / sig_close - 1) * 1e4 if sig_close and sig_close > 0 else None
    _append_csv(_csv_path(paper_dir, "trades"), {
        "date": exec_date, "code": "BTC", "side": side, "shares": round(n, 6),
        "price": round(price, 2), "amount": round(amount, 2), "fee": round(fee, 2),
        "pnl": round(pnl, 2), "win": win,
        "signal_date": pend["signal_date"], "signal_close": sig_close,
        "dev_bps": round(dev, 1) if dev is not None else None}, TRADE_COLUMNS)
    state["executed"] = True
    actions.append(f"{TRACKED_NAME}: {pend['signal_date']} 信号已按 "
                   f"{pd.Timestamp(exec_date):%Y-%m-%d} 开盘价成交（{side} "
                   f"{n:.4f} BTC @ ${price:,.0f}）{tag}")
    return True


# ---------- 每日更新 ----------

def update_account(data, today_open_info, paper_dir):
    """data: 已完成 K 线（至昨日）；today_open_info: (进行中K线日期, 开盘价)。"""
    state = load_state(paper_dir)
    actions = []
    signals = btc_donchian_filtered(data, allow_entry=f1_volume_mask(data))
    d = data.set_index("date")
    calendar = d.index
    aligned = (signals.reindex(calendar.union(signals.index)).sort_index().ffill()
               .reindex(calendar).fillna(0.0)) if not signals.empty else \
        pd.Series(0.0, index=calendar)

    nav_df = read_csv(_csv_path(paper_dir, "nav"))
    watermark = nav_df["date"].max() if not nav_df.empty else None
    new_dates = [calendar[-1]] if watermark is None else calendar[calendar > watermark]
    if len(new_dates) == 0 and (state["pending"] is None or state["executed"]):
        return [f"{TRACKED_NAME}: 无新交易日，跳过"]

    for date in new_dates:
        # ① 执行上一信号（今日开盘价）
        _execute(state, date, d["open"].loc[date], paper_dir, actions)
        # ② 记录当日收盘信号为待成交
        w_now = float(aligned.loc[date])
        state["pending"] = {
            "signal_date": f"{date:%Y-%m-%d}",
            "target": {"BTC": w_now} if w_now > 0 else {},
            "reason": (f"BTC 唐奇安20/10+F1：收盘 ${d['close'].loc[date]:,.0f}，"
                       f"{'持有' if w_now > 0 else '空仓'}"),
            "signal_close": {"BTC": round(float(d["close"].loc[date]), 2)},
        }
        state["last_signal_date"] = f"{date:%Y-%m-%d}"
        state["executed"] = False  # 新信号待次日开盘
        _append_csv(_csv_path(paper_dir, "signals"),
                    {"date": f"{date:%Y-%m-%d}",
                     "target": json.dumps(state["pending"]["target"]),
                     "reason": state["pending"]["reason"]}, SIGNAL_COLUMNS)
        actions.append(f"{TRACKED_NAME}: {date:%m-%d} 信号 "
                       f"{state['pending']['target'] or '空仓'}，待下一根K线开盘成交")
        # ③ 收盘估值追加净值
        nav = state["cash"] + state["shares"] * d["close"].loc[date]
        _append_csv(_csv_path(paper_dir, "nav"),
                    {"date": f"{date:%Y-%m-%d}", "nav": round(nav, 2)}, NAV_COLUMNS)
        actions.append(f"{TRACKED_NAME}: 净值 ${nav:,.0f}（{date:%Y-%m-%d}）")

    # ④ 进行中 K 线：昨日信号按本根开盘价成交（开盘价已确定）
    today_date, today_open = today_open_info
    if state["pending"] and not state["executed"] \
            and today_date > pd.Timestamp(state["pending"]["signal_date"]):
        _execute(state, today_date, today_open, paper_dir, actions, tag="（本根K线开盘）")

    if not actions:
        actions.append(f"{TRACKED_NAME}: 无新交易日，跳过")
    save_state(paper_dir, state)
    return actions


# ---------- 报表 ----------

def build_report(data, paper_dir):
    from metrics import calc_metrics

    state = load_state(paper_dir)
    nav_df = read_csv(_csv_path(paper_dir, "nav"))
    trades_df = read_csv(_csv_path(paper_dir, "trades"))
    lines = [f"===== BTC×F1 模拟盘台账报表（{datetime.now():%Y-%m-%d %H:%M}）====="]
    if nav_df.empty:
        return "\n".join(lines + ["台账为空，请先运行 paper-btc-update 开户"]), None, None
    nav = nav_df.set_index("date")["nav"].sort_index()
    d = data.set_index("date")
    last_close = d["close"].iloc[-1]
    if state["shares"] > 0:
        lines.append(f"当前持仓：BTC {state['shares']:.4f} 个"
                     f"（市值约 ${state['shares'] * last_close:,.0f}）；"
                     f"现金 ${state['cash']:,.0f}")
    else:
        lines.append(f"当前持仓：空仓；现金 ${state['cash']:,.0f}")
    pend = state["pending"]
    if pend:
        lines.append(f"待成交目标（{pend['signal_date']} 信号）：{pend['target'] or '空仓'}"
                     + ("（已成交）" if state["executed"] else "（待下一根K线开盘）"))
    nav_curves = {TRACKED_NAME: nav}
    trades = trades_df.to_dict("records") if not trades_df.empty else []
    m = calc_metrics(nav, trades, initial=INITIAL_USD)
    sharpe_s = f"{m['夏普比率']:.2f}" if pd.notna(m["夏普比率"]) else "N/A"
    lines.append(f"累计收益 {m['总收益率']:+.2%}  年化 {m['年化收益率']:+.2%}"
                 f"  最大回撤 {m['最大回撤']:.2%}  夏普 {sharpe_s}  交易 {m['交易次数']} 笔")

    # 理想净值（信号当日收盘成交/无成本）
    signals = btc_donchian_filtered(data, allow_entry=f1_volume_mask(data))
    close = d["close"]
    ideal_rows, holding, cash, units = [], False, INITIAL_USD, 0.0
    for date in close.index[close.index >= nav.index[0]]:
        w = float(signals.reindex(close.index.union(signals.index)).sort_index()
                  .ffill().loc[date]) if not signals.empty else 0.0
        if w > 0 and not holding:
            units, cash = cash / close.loc[date], 0.0
            holding = True
        elif w == 0 and holding:
            cash, units = units * close.loc[date], 0.0
            holding = False
        ideal_rows.append((date, cash + units * close.loc[date]))
    ideal = pd.Series(dict(ideal_rows)).sort_index()
    drag = nav.iloc[-1] / INITIAL_USD - 1 - (ideal.iloc[-1] / INITIAL_USD - 1)
    lines.append(f"理想净值（收盘成交/无成本）累计 {ideal.iloc[-1] / INITIAL_USD - 1:+.2%}"
                 f"  vs  台账 {nav.iloc[-1] / INITIAL_USD - 1:+.2%}  → 执行损耗 {drag:+.2%}")

    lines.append("\n执行偏差（信号日收盘 → 下一根K线开盘成交价，含费用滑点与隔夜，bps）")
    if trades_df.empty or trades_df["dev_bps"].dropna().empty:
        lines.append("  暂无成交记录")
    else:
        for side, label in (("buy", "买入"), ("sell", "卖出")):
            g = trades_df[trades_df["side"] == side]["dev_bps"].dropna()
            if g.empty:
                continue
            lines.append(f"  {label}: n={len(g)}  均值 {g.mean():+.1f}  中位 {g.median():+.1f}"
                         f"  P25 {g.quantile(0.25):+.1f}  P75 {g.quantile(0.75):+.1f}")
    monthly = nav.resample("ME").last().pct_change().dropna()
    lines.append("\n月度收益：")
    lines += ([f"  {dt:%Y-%m}  {r:+.2%}" for dt, r in monthly.items()]
              if not monthly.empty else ["  （净值数据不足一个月）"])
    if not trades_df.empty:
        lines.append("\n最近 10 笔成交：")
        for _, t in trades_df.tail(10).iterrows():
            dev_s = f"{t['dev_bps']:+.1f}bps" if pd.notna(t["dev_bps"]) else "N/A"
            lines.append(f"  {t['date']:%Y-%m-%d} BTC "
                         f"{'买入' if t['side'] == 'buy' else '卖出'} {t['shares']:.4f}个"
                         f" @ ${t['price']:,.0f}（偏差 {dev_s}）")
    # 基准：BTC 持有
    b = close[close.index >= nav.index[0]]
    lines.append(f"\n同期 BTC 买入持有：{b.iloc[-1] / b.iloc[0] - 1:+.2%}")
    return "\n".join(lines), nav_curves, ideal
