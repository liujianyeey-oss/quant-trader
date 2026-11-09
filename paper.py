"""模拟盘台账（paper trading）：跟踪策略信号与实际可成交价格的偏差。

两个独立虚拟账户（momentum / staggered），初始资金取 config backtest.initial_cash。

每日流程（paper-update，建议收盘后 cron 调用）：
1. 拉最新已完成日线（盘中快照不入账），计算各策略当前生效目标（与回测同口径）；
2. 上一运行日记下的"待成交目标"用今天真实开盘价成交（费率/滑点/100股整数倍
   与回测引擎同一套 _rebalance 逻辑；缺开盘价则顺延到下一交易日）；
3. 记录当日新信号为待成交目标（次日开盘成交），以今日收盘价估值追加净值。

幂等：同一天重复运行不产生重复记录；数据无新交易日则什么都不做。
存储：paper/ 目录纯文件（JSON 状态 + CSV 流水），文件缺失视为新账户。
"""
import json
import os
from datetime import datetime

import pandas as pd

from strategies import STRATEGIES
from backtest import Backtester
from metrics import calc_metrics

TRACKED_STRATEGIES = ["momentum", "staggered"]


# ---------- 状态与流水读写 ----------

def _state_path(paper_dir, name):
    return os.path.join(paper_dir, f"account_{name}.json")


def _csv_path(paper_dir, kind, name):
    return os.path.join(paper_dir, f"{kind}_{name}.csv")


def load_state(paper_dir, name, initial_cash):
    """读取账户状态；文件缺失或损坏视为新账户。"""
    try:
        with open(_state_path(paper_dir, name), encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return {"cash": initial_cash, "shares": {}, "cost_basis": {},
                "pending": None, "last_signal_date": None}


def save_state(paper_dir, name, state):
    with open(_state_path(paper_dir, name), "w", encoding="utf-8") as f:
        json.dump(state, f, ensure_ascii=False, indent=2)


def _append_csv(path, row: dict, columns):
    df = pd.DataFrame([row], columns=columns)
    df.to_csv(path, mode="a", header=not os.path.exists(path), index=False)


def _read_csv(path):
    try:
        return pd.read_csv(path, parse_dates=["date"])
    except (OSError, ValueError):
        return pd.DataFrame()


TRADE_COLUMNS = ["date", "code", "side", "shares", "price", "amount", "fee",
                 "pnl", "win", "signal_date", "signal_close", "dev_bps"]
NAV_COLUMNS = ["date", "nav"]
SIGNAL_COLUMNS = ["date", "target", "reason"]


# ---------- 每日更新 ----------

def update_account(cfg, name, data, paper_dir, stock_codes=(), strategy=None):
    """更新一个策略账户。data 只含已完成日线。返回动作描述列表。

    strategy：默认按 name 从 STRATEGIES 实例化；影子账户注入策略实例
    （如 D1 变体），其余流程（执行/估值/闸门/幂等）完全一致。

    逐日处理净值水印之后的所有新交易日（支持多日故障后的补记）：
    每个新交易日依次：① 用当日开盘价执行上一待成交目标（目标未变且策略非
    事件重校准型则跳过，与回测"目标不变不重调"一致；缺开盘价顺延）；
    ② 记录当日生效目标为新待成交；③ 以收盘价估值追加净值。
    """
    initial = cfg["backtest"]["initial_cash"]
    strategy = strategy or STRATEGIES[name](cfg)
    state = load_state(paper_dir, name, initial)
    state.setdefault("executed_target", {})
    actions = []

    signals = strategy.generate_signals(data)
    if signals.empty:
        return [f"{name}: 数据不足，无信号，跳过"]
    close_panel = pd.DataFrame(
        {c: df.set_index("date")["close"] for c, df in data.items()}).sort_index()
    open_panel = pd.DataFrame(
        {c: df.set_index("date")["open"] for c, df in data.items()}).sort_index()
    close_ffill = close_panel.ffill()  # 估值用（与回测引擎同口径，防单日缺行情跳水）
    calendar = close_panel.index
    aligned = (signals.reindex(calendar.union(signals.index)).sort_index().ffill()
               .reindex(calendar).fillna(0.0))

    # 净值水印 = 已处理到的交易日；新账户只处理最近一天（不回放历史）
    nav_df = _read_csv(_csv_path(paper_dir, "nav", name))
    watermark = nav_df["date"].max() if not nav_df.empty else None
    new_dates = [calendar[-1]] if watermark is None else calendar[calendar > watermark]
    if len(new_dates) == 0:
        return [f"{name}: 无新交易日，跳过"]

    bt = Backtester(cfg, stock_codes=stock_codes)
    cost_cfg = cfg["cost"]
    for date in new_dates:
        # 数据完整性闸门（2026-10-09 教训：部分标的当日数据未产出时，并集日历
        # 会提前推进水位线——持仓按 ffill 旧价记平、残缺截面信号永久留档、
        # 兜底 cron 被水位线架空）。任一标的缺当日行且其后仍无数据（=数据源
        # 尚未产出，非停牌）→ 当日不记账、不推进水位线，待补齐后自然补记；
        # 缺当日行但有更晚数据 = 停牌/单日缺行情，维持 ffill 旧逻辑。
        missing = [c for c in close_panel.columns
                   if pd.isna(close_panel.loc[date, c])
                   and not close_panel[c].loc[date:].iloc[1:].notna().any()]
        if missing:
            actions.append(f"{name}: {date:%Y-%m-%d} 数据未齐（缺 {missing} 当日数据，"
                           f"数据源可能尚未产出），当日不记账，待补齐后自动补记")
            break
        open_row = open_panel.loc[date]
        close_row = close_panel.loc[date]
        deferred = False

        # ---- ① 执行上一待成交目标（当日真实开盘价） ----
        pend = state["pending"]
        if pend and date > pd.Timestamp(pend["signal_date"]):
            tgt = pd.Series(0.0, index=close_panel.columns)
            for c, w in pend["target"].items():
                tgt[c] = w
            executed = state["executed_target"]
            need_trade = strategy.rebalance_on_event or any(
                abs(tgt[c] - executed.get(c, 0.0)) > 1e-9 for c in tgt.index)
            needed = [c for c in tgt.index if state["shares"].get(c, 0) > 0 or tgt[c] > 0]
            if not need_trade:
                state["executed_target"] = pend["target"]
                actions.append(f"{name}: {date:%m-%d} 目标未变，无需调仓")
            elif all(pd.notna(open_row[c]) and open_row[c] > 0 for c in needed):
                shares = {c: state["shares"].get(c, 0) for c in tgt.index}
                trades = []
                cash = bt._rebalance(date, tgt, open_row, state["cash"],
                                     shares, state["cost_basis"], trades)
                for t in trades:
                    sig_close = pend["signal_close"].get(t["code"])
                    dev = ((t["price"] / sig_close - 1) * 1e4
                           if sig_close and sig_close > 0 else None)
                    rate = (cost_cfg["stock_commission"] if t["code"] in stock_codes
                            else cost_cfg["etf_commission"])
                    amount = t["price"] * t["shares"]
                    fee = amount * rate
                    if cost_cfg.get("min_commission"):
                        fee = max(fee, cost_cfg["min_commission"])
                    if t["side"] == "sell" and t["code"] in stock_codes:
                        fee += amount * cost_cfg["stamp_tax"]
                    _append_csv(_csv_path(paper_dir, "trades", name), {
                        "date": t["date"], "code": t["code"], "side": t["side"],
                        "shares": t["shares"], "price": round(t["price"], 4),
                        "amount": round(amount, 2), "fee": round(fee, 2),
                        "pnl": round(t["pnl"], 2), "win": t["win"],
                        "signal_date": pend["signal_date"],
                        "signal_close": sig_close,
                        "dev_bps": round(dev, 1) if dev is not None else None},
                        TRADE_COLUMNS)
                state["cash"] = cash
                state["shares"] = {c: n for c, n in shares.items() if n > 0}
                state["executed_target"] = pend["target"]
                n_buy = sum(1 for t in trades if t["side"] == "buy")
                n_sell = sum(1 for t in trades if t["side"] == "sell")
                actions.append(f"{name}: {pend['signal_date']} 信号已按 {date:%Y-%m-%d} "
                               f"开盘价成交（买{n_buy}笔/卖{n_sell}笔）")
            else:
                deferred = True
                actions.append(f"{name}: {date:%Y-%m-%d} 开盘价缺失，待成交目标顺延")

        # ---- ② 记录当日生效目标为新待成交（延期时不覆盖旧目标） ----
        if not deferred:
            tgt_now = aligned.loc[date]
            tgt_dict = {c: round(float(w), 4) for c, w in tgt_now.items() if w > 0}
            if date == calendar[-1]:
                reason = strategy.describe_signal(data).replace("\n", " | ")
            else:
                reason = "（补记故障期间交易日信号）"
            state["pending"] = {
                "signal_date": f"{date:%Y-%m-%d}", "target": tgt_dict,
                "reason": reason,
                "signal_close": {c: round(float(close_row[c]), 4) for c in tgt_dict
                                 if pd.notna(close_row.get(c))},
            }
            state["last_signal_date"] = f"{date:%Y-%m-%d}"
            _append_csv(_csv_path(paper_dir, "signals", name),
                        {"date": f"{date:%Y-%m-%d}", "target": json.dumps(tgt_dict),
                         "reason": reason}, SIGNAL_COLUMNS)
            actions.append(f"{name}: {date:%m-%d} 信号 {tgt_dict or '空仓'}，"
                           f"待下一交易日开盘成交")

        # ---- ③ 以当日收盘价估值（前向填充价，与回测引擎同口径），追加净值 ----
        nav = state["cash"]
        for c, n in state["shares"].items():
            p = close_ffill.loc[date, c]
            if pd.notna(p):
                nav += n * p
        _append_csv(_csv_path(paper_dir, "nav", name),
                    {"date": f"{date:%Y-%m-%d}", "nav": round(nav, 2)}, NAV_COLUMNS)
        actions.append(f"{name}: 净值 {nav:,.0f}（{date:%Y-%m-%d}）")

    save_state(paper_dir, name, state)
    return actions


# ---------- 理想净值（信号当日收盘价成交、无成本、允许零碎股） ----------

def ideal_nav_series(signals, data, start, initial):
    """理论净值曲线：信号当日收盘即成交、无滑点无费用、允许零碎股。
    用于量化台账实际成交的执行损耗。"""
    panel = pd.DataFrame({c: df.set_index("date")["close"] for c, df in data.items()})
    panel = panel.sort_index().ffill()
    aligned = (signals.reindex(panel.index.union(signals.index)).sort_index().ffill()
               .reindex(panel.index).fillna(0.0))
    cal = panel.index[panel.index >= pd.Timestamp(start)]
    shares, cash, rows, prev = {}, initial, [], None
    for date in cal:
        w = aligned.loc[date]
        if prev is None or not w.equals(prev):
            total = cash + sum(n * panel.loc[date, c] for c, n in shares.items())
            new_shares = {}
            for c, wt in w.items():
                p = panel.loc[date, c]
                if wt > 0 and pd.notna(p) and p > 0:
                    new_shares[c] = wt * total / p
            shares, cash = new_shares, total * (1 - w.sum())
            prev = w
        rows.append((date, cash + sum(n * panel.loc[date, c] for c, n in shares.items())))
    return pd.Series(dict(rows)).sort_index()


# ---------- 报表 ----------

def _dev_stats_text(name, trades: pd.DataFrame) -> list:
    lines = [f"\n[{name}] 执行偏差（信号日收盘 → 实际开盘成交价，含滑点与隔夜跳空，bps）"]
    if trades.empty or trades["dev_bps"].dropna().empty:
        return lines + ["  暂无成交记录"]
    for side, label in (("buy", "买入"), ("sell", "卖出")):
        g = trades[trades["side"] == side]["dev_bps"].dropna()
        if g.empty:
            continue
        lines.append(f"  {label}: n={len(g)}  均值 {g.mean():+.1f}  中位 {g.median():+.1f}"
                     f"  P25 {g.quantile(0.25):+.1f}  P75 {g.quantile(0.75):+.1f}")
    return lines


def _monthly_text(nav: pd.Series) -> list:
    monthly = nav.resample("ME").last().pct_change().dropna()
    if monthly.empty:
        return ["  （净值数据不足一个月）"]
    return [f"  {d:%Y-%m}  {r:+.2%}" for d, r in monthly.items()]


def build_report(cfg, data_map, bench, paper_dir):
    """生成中文分析报表文本。data_map: {strategy_name: data}。"""
    initial = cfg["backtest"]["initial_cash"]
    cost = cfg["cost"]
    lines = [f"===== 模拟盘台账报表（{datetime.now():%Y-%m-%d %H:%M}）=====",
             f"成本口径：ETF 佣金 {cost['etf_commission']:.2%}、"
             f"最低佣金 {cost.get('min_commission') or 0:.0f} 元/笔"
             f"（2026-11-08 起生效，历史成交不改写）、"
             f"滑点 {cost['slippage']:.1%}、100 股整数倍"]
    nav_curves, ideal_curves = {}, {}

    for name in TRACKED_STRATEGIES:
        state = load_state(paper_dir, name, initial)
        nav_df = _read_csv(_csv_path(paper_dir, "nav", name))
        trades_df = _read_csv(_csv_path(paper_dir, "trades", name))
        signals_df = _read_csv(_csv_path(paper_dir, "signals", name))

        lines.append(f"\n----- 策略：{name} -----")
        if nav_df.empty:
            lines.append("台账为空，请先运行 paper-update 开户")
            continue
        nav = nav_df.set_index("date")["nav"].sort_index()
        nav_curves[name] = nav

        # 当前持仓与待成交
        if state["shares"]:
            last_close = {c: df.set_index("date")["close"].iloc[-1]
                          for c, df in data_map[name].items()}
            pos = "、".join(f"{c} {n}股" for c, n in state["shares"].items())
            mv = sum(n * last_close.get(c, 0) for c, n in state["shares"].items())
            lines.append(f"当前持仓：{pos}（市值约 {mv:,.0f}）；现金 {state['cash']:,.0f}")
        else:
            lines.append(f"当前持仓：空仓；现金 {state['cash']:,.0f}")
        pend = state["pending"]
        if pend:
            lines.append(f"待成交目标（{pend['signal_date']} 信号）：{pend['target'] or '空仓'}")

        # 绩效
        trades = trades_df.to_dict("records") if not trades_df.empty else []
        m = calc_metrics(nav, trades, initial=initial)
        sharpe_s = f"{m['夏普比率']:.2f}" if pd.notna(m["夏普比率"]) else "N/A"
        lines.append(f"累计收益 {m['总收益率']:+.2%}  年化 {m['年化收益率']:+.2%}"
                     f"  最大回撤 {m['最大回撤']:.2%}  夏普 {sharpe_s}"
                     f"  交易 {m['交易次数']} 笔")
        if not trades_df.empty:
            lines.append(f"成交记录 {len(trades_df)} 笔（首笔 {trades_df['date'].min():%Y-%m-%d}）")

        # 理想净值对比（执行损耗）
        data = data_map[name]
        strategy = STRATEGIES[name](cfg)
        signals = strategy.generate_signals(data)
        ideal = ideal_nav_series(signals, data, nav.index[0], initial)
        ideal_curves[name] = ideal
        drag = nav.iloc[-1] / initial - 1 - (ideal.iloc[-1] / initial - 1)
        lines.append(f"理想净值（收盘成交/无成本）累计 {ideal.iloc[-1] / initial - 1:+.2%}"
                     f"  vs  台账 {nav.iloc[-1] / initial - 1:+.2%}  → 执行损耗 {drag:+.2%}")

        lines += _dev_stats_text(name, trades_df)
        lines.append("\n月度收益：")
        lines += _monthly_text(nav)
        if not trades_df.empty:
            lines.append("\n最近 10 笔成交：")
            for _, t in trades_df.tail(10).iterrows():
                lines.append(f"  {t['date']:%Y-%m-%d} {t['code']} "
                             f"{'买入' if t['side'] == 'buy' else '卖出'} {int(t['shares'])}股"
                             f" @ {t['price']:.3f}（偏差 {t['dev_bps']:+.1f}bps）")

    # 基准对比
    if bench is not None and nav_curves:
        start = min(n.index[0] for n in nav_curves.values())
        b = bench[bench.index >= start]
        if not b.empty:
            lines.append(f"\n同期基准 {cfg.get('benchmark')}：{b.iloc[-1] / b.iloc[0] - 1:+.2%}")

    # 影子对照（D1 防守池轮动影子台账，存在时追加；不影响主报表结构）
    from paper_shadow import shadow_compare_lines
    lines += shadow_compare_lines(paper_dir,
                                  os.path.join(os.path.dirname(paper_dir), "paper_shadow"),
                                  nav_curves)
    return "\n".join(lines), nav_curves, ideal_curves


def plot_paper_nav(nav_curves, ideal_curves, bench, path):
    """两策略实际净值（实线）+ 理想净值（虚线）+ 基准。"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams["font.sans-serif"] = ["PingFang SC", "Arial Unicode MS", "Heiti TC", "SimHei"]
    plt.rcParams["axes.unicode_minus"] = False

    fig, ax = plt.subplots(figsize=(11, 6))
    colors = {"momentum": "tab:blue", "staggered": "tab:orange"}
    for name, nav in nav_curves.items():
        ax.plot(nav.index, nav / nav.iloc[0], label=f"{name} 台账", color=colors.get(name))
        ideal = ideal_curves.get(name)
        if ideal is not None and not ideal.empty:
            ax.plot(ideal.index, ideal / ideal.iloc[0], linestyle="--", alpha=0.7,
                    label=f"{name} 理想", color=colors.get(name))
    if bench is not None and nav_curves:
        start = min(n.index[0] for n in nav_curves.values())
        b = bench[bench.index >= start]
        if not b.empty:
            ax.plot(b.index, b / b.iloc[0], label="基准 510300",
                    color="gray", alpha=0.8)
    ax.set_title("模拟盘台账净值")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
