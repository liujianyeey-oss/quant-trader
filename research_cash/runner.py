"""现金管理小研究：511880 银华日利ETF 作为 staggered 现金腿停泊。

对照：生产 staggered 原样（现金收益 0）vs 现金腿停泊 511880
（现金部分按日享 511880 收益；摩擦：调仓日交易额按万1 计货币ETF进出费用，声明近似；
忽略货币ETF 自身微小波动直接按净值计息，声明近似）。
"""
import copy
import os
import sys

import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data import DataLoader  # noqa: E402
from backtest import Backtester  # noqa: E402
from strategies import StaggeredMomentum  # noqa: E402
from metrics import calc_metrics  # noqa: E402

PROJECT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
os.makedirs(RESULTS, exist_ok=True)


def staggered_equity_with_holdings(cfg, data, start, end):
    """复刻 staggered 回测主循环，额外输出每日现金权重与调仓日交易额。"""
    bt = Backtester(cfg, rebalance_on_event=True)
    signals = StaggeredMomentum(cfg).generate_signals(data)
    closes = {c: df.set_index("date")["close"] for c, df in data.items()}
    opens = {c: df.set_index("date")["open"] for c, df in data.items()}
    close_panel = pd.DataFrame(closes).sort_index()
    open_panel = pd.DataFrame(opens).sort_index().reindex(close_panel.index)
    calendar = close_panel.index
    marks = close_panel.ffill()
    s = signals.reindex(columns=close_panel.columns, fill_value=0.0)
    aligned = (s.reindex(calendar.union(s.index)).sort_index().ffill()
               .reindex(calendar).fillna(0.0))
    targets = aligned.shift(1).fillna(0.0)
    events = pd.Series(calendar.isin(s.index), index=calendar).shift(1, fill_value=False)

    cash = cfg["backtest"]["initial_cash"]
    shares = {c: 0 for c in close_panel.columns}
    cost_basis = {}
    completed = pd.Series(0.0, index=close_panel.columns)
    pending = False
    rows = []
    start, end = pd.Timestamp(start), pd.Timestamp(end)
    for i, date in enumerate(calendar):
        if date < start or date > end:
            continue
        w = targets.iloc[i]
        traded_amount = 0.0
        changed = not np.allclose(w.to_numpy(), completed.to_numpy(), atol=1e-10)
        if changed or pending or events.iloc[i]:
            o = open_panel.iloc[i]
            needed = [c for c in shares if shares[c] > 0 or w.get(c, 0) > 0]
            if all(pd.notna(o[c]) and o[c] > 0 for c in needed):
                before = dict(shares)
                cash = bt._rebalance(date, w, o, cash, shares, cost_basis, [])
                traded_amount = sum(
                    abs((shares[c] - before[c]) * o[c]) for c in shares)
                completed = w.copy()
                pending = False
            else:
                pending = True
        gross = sum(n * marks.iloc[i][c] for c, n in shares.items()
                    if n > 0 and pd.notna(marks.iloc[i][c]))
        nav = cash + gross
        rows.append((date, nav, cash / nav if nav > 0 else 0.0, traded_amount))
    df = pd.DataFrame(rows, columns=["date", "nav", "cash_w", "traded"]).set_index("date")
    return df


def cash_uplift(holdings: pd.DataFrame, money_close: pd.Series, fee_rate=0.0001):
    """现金腿停泊收益序列：现金权重(前一日) × 货币ETF日收益 - 调仓摩擦。
    摩擦 = 调仓日交易额 × fee_rate（货币ETF进出一次，声明近似）。"""
    money_ret = money_close.pct_change()
    common = holdings.index.intersection(money_ret.dropna().index)
    h = holdings.loc[common]
    r = money_ret.loc[common]
    cash_w_prev = h["cash_w"].shift(1).fillna(h["cash_w"].iloc[0])
    daily_gain = cash_w_prev * r.fillna(0.0)
    friction = h["traded"] * fee_rate
    nav_gain = (h["nav"].shift(1).fillna(h["nav"].iloc[0]) * daily_gain - friction)
    return nav_gain, daily_gain, friction


def main():
    cfg = yaml.safe_load(open(os.path.join(PROJECT, "config.yaml"), encoding="utf-8"))
    initial = cfg["backtest"]["initial_cash"]
    loader = DataLoader(cache_dir=os.path.join(PROJECT, "data_cache"))

    print("加载 511880...", flush=True)
    money = loader.load("511880", is_etf=True, start="2017-01-01")
    mc = money.set_index("date")["close"]
    m_ret = mc.pct_change().dropna()
    m_vol = m_ret.std() * np.sqrt(252)
    m_mdd = (mc / mc.cummax() - 1).min()
    m_years = ((mc.index[-1] - mc.index[0]).days + 1) / 365.25
    m_cagr = (mc.iloc[-1] / mc.iloc[0]) ** (1 / m_years) - 1
    m_amount20 = (money["close"] * money["volume"]).tail(20).mean() / 1e4
    print(f"511880: {mc.index.min().date()}~{mc.index.max().date()} "
          f"年化 {m_cagr:.2%} 波动 {m_vol:.2%} 回撤 {m_mdd:.2%} "
          f"近20日日均成交额(近似) {m_amount20:,.0f} 万元")

    print("复算 staggered（含现金权重）...", flush=True)
    pool = ["510300", "510500", "159915", "588000", "518880"]
    data = {c: loader.load(c, is_etf=True, start="2017-01-01") for c in pool}
    end = max(df["date"].max() for df in data.values())
    hold = staggered_equity_with_holdings(cfg, data, "2019-01-01", end)
    nav0 = hold["nav"]
    base = calc_metrics(nav0, [], initial=initial)

    gain, daily_gain, friction = cash_uplift(hold, mc)
    nav_cash = nav0.loc[gain.index] + gain.cumsum()
    parked = calc_metrics(nav_cash, [], initial=initial)

    mean_cash_w = hold["cash_w"].mean()
    uplift = parked["年化收益率"] - base["年化收益率"]
    rows = [
        {"variant": "生产staggered原样(现金0息)", "年化": base["年化收益率"],
         "回撤": base["最大回撤"], "夏普": base["夏普比率"], "卡玛": base["卡玛比率"]},
        {"variant": "现金腿停泊511880(万1摩擦)", "年化": parked["年化收益率"],
         "回撤": parked["最大回撤"], "夏普": parked["夏普比率"], "卡玛": parked["卡玛比率"]},
        {"variant": "提升幅度", "年化": uplift, "回撤": np.nan,
         "夏普": parked["夏普比率"] - base["夏普比率"],
         "卡玛": parked["卡玛比率"] - base["卡玛比率"]},
        {"variant": "511880本体", "年化": m_cagr, "回撤": m_mdd,
         "夏普": m_ret.mean() / m_ret.std() * np.sqrt(252),
         "卡玛": m_cagr / abs(m_mdd) if m_mdd < 0 else np.nan},
    ]
    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(RESULTS, "cash_management.csv"),
               index=False, encoding="utf-8-sig")

    print(f"\nstaggered 平均现金权重: {mean_cash_w:.1%}")
    print(f"摩擦合计: {friction.sum():,.0f} 元（{friction.sum() / initial:.2%} 初始资金）")
    show = out.assign(**{c: out[c].map(lambda x: f"{x:.2%}" if pd.notna(x) else "-")
                         for c in ("年化", "回撤")})
    show["夏普"] = out["夏普"].map(lambda x: f"{x:.2f}" if pd.notna(x) else "-")
    show["卡玛"] = out["卡玛"].map(lambda x: f"{x:.2f}" if pd.notna(x) else "-")
    print(show.to_string(index=False))
    print(f"\n结果 CSV：{RESULTS}")


if __name__ == "__main__":
    main()
