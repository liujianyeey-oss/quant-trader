"""观察账户 6/7（paper_obs）：QQQ×H2 利率过滤 + 159915×RSI2 均值回归。

账户 6 qqq_h2（$50,000，us 成本）：donchian_QQQ 20/10 原样 + H2 过滤——
美债10Y 20 日变化 > +25bp 的宏观窗口内禁止新开多仓（已持有的退出仍按唐奇安）。
宏观数据 T-1 规则（macro_research bond_zh_us_rate 通道与口径）。
账户 7 hs_rsi2（¥1,000,000，ETF 万1+滑点0.1%）：159915 收盘>MA200 且 RSI(2)<10
→ 次日开盘买；收盘>MA5 → 次日开盘卖；单标的全进全出（R2 预声明原文）。

机制复刻既有台账：幂等、无新交易日跳过、多日缺跑补记、dev_bps。
存储 paper_obs/（account/trades/nav/signals 两账户各一套）。
"""
import json
import os
import sys
from datetime import datetime

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from macro_research.strategies import macro_flags  # noqa: E402
from macro_research.data_layer import load_us10y  # noqa: E402
from research_r2.runner import sig_mr1  # noqa: E402
from us_equities.research.etf_strategies import _rsi  # noqa: E402

TRADE_COLUMNS = ["date", "code", "side", "shares", "price", "amount", "fee",
                 "pnl", "win", "signal_date", "signal_close", "dev_bps"]
NAV_COLUMNS = ["date", "nav"]
SIGNAL_COLUMNS = ["date", "target", "reason"]


# ---------- 策略信号（各自规则，返回 weight Series，仅状态变化日） ----------

def sig_qqq_h2(data, us10y, symbol="QQQ"):
    """donchian 20/10 + H2 宏观窗口禁开新仓（T-1）。返回 (weight Series, H2 flags)。"""
    d = data[symbol].set_index("date")
    adj_c = _adj(d, "close")
    adj_h = _adj(d, "high")
    adj_l = _adj(d, "low")
    upper = adj_h.rolling(20).max().shift(1)
    lower = adj_l.rolling(10).min().shift(1)
    flags = macro_flags(pd.DatetimeIndex(d.index), None if us10y is None else
                        pd.DataFrame({"date": d.index, "dxy": 0.0}), us10y) \
        if us10y is not None else None
    h2 = flags["H2"] if flags is not None else pd.Series(False, index=d.index)

    holding, rows = False, []
    for i in range(21, len(d)):
        date = d.index[i]
        if not holding and adj_c.iloc[i] > upper.iloc[i] and not bool(h2.iloc[i]):
            holding = True
        elif holding and adj_c.iloc[i] < lower.iloc[i]:
            holding = False
        else:
            continue
        rows.append((date, 1.0 if holding else 0.0))
    return pd.Series(dict(rows)).sort_index(), h2


def sig_hs_rsi2(df):
    """159915 RSI(2)<10 Connors 式（R2 原文实现复用）。"""
    return sig_mr1(df, 10)


def _adj(d, field):
    """复权价序列；无 factor 列（A股 qfq 数据）按 1.0 计。"""
    f = d["factor"] if "factor" in d.columns else 1.0
    return d[field] * f


# ---------- 成本模型 ----------

def _fee_us(n, price, sell=False):
    f = max(1.0, n * 0.005)
    if sell:
        f += n * price * 0.5 / 10000
    return f


def _fee_cn(amount):
    return amount * 0.0001


ACCOUNTS = {
    "qqq_h2": {"symbol": "QQQ", "initial": 50_000.0, "lot": 1, "slip": 0.001,
               "fee": "us", "currency": "USD"},
    "hs_rsi2": {"symbol": "159915", "initial": 1_000_000.0, "lot": 100,
                "slip": 0.001, "fee": "cn", "currency": "CNY"},
}


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
        return {"cash": ACCOUNTS[name]["initial"], "shares": 0, "cost_basis": 0.0,
                "pending": None, "last_signal_date": None, "executed": True}


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


# ---------- 成交 ----------

def _execute(state, name, exec_date, exec_open, paper_dir, actions):
    pend = state["pending"]
    if not pend or state["executed"]:
        return False
    cfg = ACCOUNTS[name]
    symbol = cfg["symbol"]
    tgt = pend["target"].get(symbol, 0.0)
    sig_close = pend["signal_close"].get(symbol)
    slip = cfg["slip"]
    shares, cash = state["shares"], state["cash"]
    if tgt > 0 and shares <= 0:
        price = exec_open * (1 + slip)
        n = int(cash / price / cfg["lot"]) * cfg["lot"]
        if n <= 0:
            return False
        fee = _fee_us(n, price) if cfg["fee"] == "us" else _fee_cn(n * price)
        if n * price + fee > cash:
            n = int(cash / (price * (1 + 0.0001)) / cfg["lot"]) * cfg["lot"]
            if n <= 0:
                return False
            fee = _fee_us(n, price) if cfg["fee"] == "us" else _fee_cn(n * price)
        amount = n * price
        state["cost_basis"] = price
        state["shares"], state["cash"] = n, cash - amount - fee
        side, pnl, win = "buy", 0.0, None
    elif tgt == 0 and shares > 0:
        price = exec_open * (1 - slip)
        n = shares
        amount = n * price
        fee = _fee_us(n, price, sell=True) if cfg["fee"] == "us" else _fee_cn(amount)
        pnl = (price - state["cost_basis"]) * n - fee
        win = pnl > 0
        state["cash"], state["shares"], state["cost_basis"] = cash + amount - fee, 0, 0.0
        side = "sell"
    else:
        state["executed"] = True
        return False
    dev = (price / sig_close - 1) * 1e4 if sig_close and sig_close > 0 else None
    _append_csv(_csv_path(paper_dir, "trades", name), {
        "date": exec_date, "code": symbol, "side": side, "shares": n,
        "price": round(price, 4), "amount": round(n * price, 2), "fee": round(fee, 2),
        "pnl": round(pnl, 2), "win": win,
        "signal_date": pend["signal_date"], "signal_close": sig_close,
        "dev_bps": round(dev, 1) if dev is not None else None}, TRADE_COLUMNS)
    state["executed"] = True
    cur = "USD" if cfg["currency"] == "USD" else "CNY"
    actions.append(f"{name}: {pend['signal_date']} 信号已按 {exec_date:%Y-%m-%d} "
                   f"开盘价成交（{side} {n} {symbol} @ {price:.3f} {cur}）")
    return True


# ---------- 每日更新 ----------

def update_account(data, name, paper_dir, extra=None):
    """data: {symbol: df}（只含已完成交易日）。extra: qqq_h2 传 us10y DataFrame。"""
    cfg = ACCOUNTS[name]
    symbol = cfg["symbol"]
    state = load_state(paper_dir, name)
    actions = []

    if name == "qqq_h2":
        w, h2_flags = sig_qqq_h2(data, extra, symbol)
    else:
        w, h2_flags = sig_hs_rsi2(data[symbol]), None

    d = data[symbol].set_index("date")
    adj_o = _adj(d, "open")
    adj_c = _adj(d, "close")
    calendar = d.index
    aligned = (w.reindex(calendar.union(w.index)).sort_index().ffill()
               .reindex(calendar).fillna(0.0)) if not w.empty else \
        pd.Series(0.0, index=calendar)

    nav_df = read_csv(_csv_path(paper_dir, "nav", name))
    watermark = nav_df["date"].max() if not nav_df.empty else None
    new_dates = [calendar[-1]] if watermark is None else calendar[calendar > watermark]
    if len(new_dates) == 0 and (state["pending"] is None or state["executed"]):
        return [f"{name}: 无新交易日，跳过"]

    for date in new_dates:
        # ① 执行上一信号（当日开盘价）
        if state["pending"] and not state["executed"] \
                and date > pd.Timestamp(state["pending"]["signal_date"]) \
                and pd.notna(adj_o.get(date)) and adj_o[date] > 0:
            _execute(state, name, date, adj_o[date], paper_dir, actions)
        # ② 记录当日收盘信号为待成交
        w_now = float(aligned.loc[date])
        tgt_now = {symbol: w_now} if w_now > 0 else {}
        if date == calendar[-1]:
            reason = _describe(name, data, extra, date, w_now, h2_flags)
        else:
            reason = "（补记历史交易日信号）"
        state["pending"] = {
            "signal_date": f"{date:%Y-%m-%d}", "target": tgt_now,
            "reason": reason,
            "signal_close": {symbol: round(float(adj_c[date]), 4)} if tgt_now else {},
        }
        state["last_signal_date"] = f"{date:%Y-%m-%d}"
        state["executed"] = False
        _append_csv(_csv_path(paper_dir, "signals", name),
                    {"date": f"{date:%Y-%m-%d}", "target": json.dumps(tgt_now),
                     "reason": reason}, SIGNAL_COLUMNS)
        actions.append(f"{name}: {date:%m-%d} 信号 {tgt_now or '空仓'}，待下一交易日开盘成交")
        # ③ 收盘估值
        nav = state["cash"] + (state["shares"] * adj_c[date] if state["shares"] else 0)
        _append_csv(_csv_path(paper_dir, "nav", name),
                    {"date": f"{date:%Y-%m-%d}", "nav": round(nav, 2)}, NAV_COLUMNS)
        actions.append(f"{name}: 净值 {nav:,.0f}（{date:%Y-%m-%d}）")

    if not actions:
        actions.append(f"{name}: 无新交易日，跳过")
    save_state(paper_dir, name, state)
    return actions


def _describe(name, data, us10y, date, w_now, h2_flags):
    symbol = ACCOUNTS[name]["symbol"]
    d = data[symbol].set_index("date")
    adj_c = _adj(d, "close")
    if name == "qqq_h2":
        h2_on = bool(h2_flags.loc[date]) if h2_flags is not None and date in h2_flags.index else False
        return (f"donchian 20/10+H2：收盘 {adj_c[date]:.2f}，"
                f"{'持有' if w_now > 0 else '空仓'}，H2窗口{'开启' if h2_on else '关闭'}")
    rsi = _rsi(adj_c, 2).loc[date]
    ma200 = adj_c.rolling(200).mean().loc[date]
    ma5 = adj_c.rolling(5).mean().loc[date]
    return (f"RSI2={rsi:.1f}，MA200={ma200:.2f}，MA5={ma5:.2f}，"
            f"收盘={adj_c[date]:.2f} → {'持有' if w_now > 0 else '空仓'}")


# ---------- 报表 ----------

def build_report(data_map, paper_dir, extras=None):
    from metrics import calc_metrics

    lines = [f"===== 观察账户 6/7 台账报表（{datetime.now():%Y-%m-%d %H:%M}）====="]
    for name, cfg in ACCOUNTS.items():
        symbol = cfg["symbol"]
        state = load_state(paper_dir, name)
        nav_df = read_csv(_csv_path(paper_dir, "nav", name))
        trades_df = read_csv(_csv_path(paper_dir, "trades", name))
        lines.append(f"\n----- 账户：{name}（{symbol}，{cfg['currency']}） -----")
        if nav_df.empty:
            lines.append("台账为空，请先运行 paper-obs-update 开户")
            continue
        nav = nav_df.set_index("date")["nav"].sort_index()
        d = data_map[symbol].set_index("date")
        adj_c = _adj(d, "close")
        if state["shares"]:
            mv = state["shares"] * adj_c.iloc[-1]
            lines.append(f"当前持仓：{symbol} {state['shares']}（市值约 {mv:,.0f}）；"
                         f"现金 {state['cash']:,.0f}")
        else:
            lines.append(f"当前持仓：空仓；现金 {state['cash']:,.0f}")
        pend = state["pending"]
        if pend:
            lines.append(f"待成交目标（{pend['signal_date']} 信号）："
                         f"{pend['target'] or '空仓'}"
                         + ("（已成交）" if state["executed"] else "（待开盘）"))
            lines.append(f"信号摘要：{pend['reason']}")
        trades = trades_df.to_dict("records") if not trades_df.empty else []
        m = calc_metrics(nav, trades, initial=cfg["initial"])
        sharpe_s = f"{m['夏普比率']:.2f}" if pd.notna(m["夏普比率"]) else "N/A"
        lines.append(f"累计收益 {m['总收益率']:+.2%}  年化 {m['年化收益率']:+.2%}"
                     f"  最大回撤 {m['最大回撤']:.2%}  夏普 {sharpe_s}"
                     f"  交易 {m['交易次数']} 笔")
        lines.append("执行偏差（信号日收盘 → 实际开盘成交价，bps）")
        if trades_df.empty or trades_df["dev_bps"].dropna().empty:
            lines.append("  暂无成交记录")
        else:
            for side, label in (("buy", "买入"), ("sell", "卖出")):
                g = trades_df[trades_df["side"] == side]["dev_bps"].dropna()
                if not g.empty:
                    lines.append(f"  {label}: n={len(g)}  均值 {g.mean():+.1f}  "
                                 f"中位 {g.median():+.1f}")
        if not trades_df.empty:
            lines.append("最近 5 笔成交：")
            for _, t in trades_df.tail(5).iterrows():
                lines.append(f"  {t['date']:%Y-%m-%d} {t['code']} "
                             f"{'买入' if t['side'] == 'buy' else '卖出'} {t['shares']}"
                             f" @ {t['price']:.3f}（偏差 {t['dev_bps']:+.1f}bps）")
    return "\n".join(lines)
