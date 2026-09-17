"""R7a：组合层大回测（四袖珍，工程性不挖新假设）。

袖珍（各自生产口径，全区间 2019-01-01~最新）：
S1 A股 momentum、S2 A股 staggered、S3 美股 donchian_QQQ、S4 BTC 唐奇安×F1。
对齐：A股交易日历，缺口日前向填充（声明）；本币计、按归一化净值指数合成，忽略汇率（声明）。
组合层再平衡成本忽略（袖珍内部成本已含，声明）。

预声明配置（无寻优）：
P1 四袖珍等权 25%、每月末再平衡；P2 风险平价（60 日波动倒数加权，clip[10%,40%]，
月末再平衡，clip 后归一）；P3 = P1 + 组合层波动目标（组合 60 日实现波动 >12% 时
按 12%/σ 降总仓，下限 30%，月末重估）。
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
from strategies import MomentumRotation, StaggeredMomentum  # noqa: E402
from macro_research.strategies import run_single_asset, perf  # noqa: E402
from research_r4.runner import btc_donchian_filtered, f1_volume_mask  # noqa: E402
from macro_research.data_layer import load_btc  # noqa: E402
from us_equities.research.etf_data import load_universe  # noqa: E402
from us_equities.research import etf_strategies as US_S  # noqa: E402
from us_equities.research.etf_engine import EtfBacktester  # noqa: E402

PROJECT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
os.makedirs(RESULTS, exist_ok=True)

SEGMENTS = [("full", "2019-01-01", None), ("train", "2019-01-01", "2023-12-31"),
            ("valid", "2024-01-01", None)]
BTC_COST = {"commission": 0.001, "slippage": 0.001, "lot": 1e-4}


def sleeve_navs(cfg, loader, end):
    """四袖珍归一化净值（各自生产口径），对齐 A股交易日历（ffill 缺口）。"""
    pool = ["510300", "510500", "159915", "588000", "518880"]
    a_data = {c: loader.load(c, is_etf=True, start="2017-01-01") for c in pool}
    calendar = pd.DatetimeIndex(sorted(set().union(*[set(df["date"]) for df in a_data.values()])))
    calendar = calendar[calendar >= "2019-01-01"]

    bt = Backtester(cfg)
    navs = {}
    sig1 = MomentumRotation(cfg).generate_signals(a_data)
    eq1, _ = bt.run(a_data, sig1, start="2019-01-01", end=end)
    navs["S1_A股momentum"] = eq1["total_value"]
    sig2 = StaggeredMomentum(cfg).generate_signals(a_data)
    bt2 = Backtester(cfg, rebalance_on_event=True)
    eq2, _ = bt2.run(a_data, sig2, start="2019-01-01", end=end)
    navs["S2_A股staggered"] = eq2["total_value"]

    us_data, _ = load_universe(symbols=["QQQ"], cache=os.path.join(
        PROJECT, "us_equities", "research", "cache"), offline=True)
    sig3 = US_S.donchian(us_data, "QQQ", 20, 10)
    bt3 = EtfBacktester()
    eq3, _ = bt3.run(us_data, sig3, "2019-01-01", end)
    navs["S3_美股donchian"] = eq3

    btc, _ = load_btc()
    sig4 = btc_donchian_filtered(btc, allow_entry=f1_volume_mask(btc))
    eq4, _ = run_single_asset(btc, sig4, "2019-01-02", end, BTC_COST, 50_000)
    navs["S4_BTC×F1"] = eq4

    aligned = {}
    for name, nav in navs.items():
        a = nav.reindex(calendar.union(nav.index)).sort_index().ffill().reindex(calendar)
        aligned[name] = a / a.iloc[0]  # 归一化净值指数，起点 1
    return pd.DataFrame(aligned), calendar


def perf_nav(eq, initial=1.0):
    return perf(eq, [], initial)


def simulate_portfolio(rets: pd.DataFrame, weight_fn, vol_target=None):
    """月末再平衡组合模拟：现金收益 0；每月末按 weight_fn 权重×波动系数重设袖珍市值。"""
    dates = rets.index
    month_ends = set(dates.to_series().groupby([dates.year, dates.month]).last())
    sleeve_vals, cash, rows = None, 0.0, []
    for i, date in enumerate(dates):
        if sleeve_vals is None or date in month_ends:
            total = 1.0 if sleeve_vals is None else sum(sleeve_vals.values()) + cash
            w = weight_fn(rets.loc[:date])
            scalar = 1.0
            if vol_target is not None:
                hist = rets.loc[:date].tail(60)
                if len(hist) >= 60:
                    sigma = (hist * w).sum(axis=1).std() * np.sqrt(252)
                    if sigma > vol_target[0]:
                        scalar = max(vol_target[0] / sigma, vol_target[1])
            sleeve_vals = {c: total * w[c] * scalar for c in rets.columns}
            cash = total * (1 - scalar)
        r = rets.loc[date].fillna(0.0)
        for c in sleeve_vals:
            sleeve_vals[c] *= (1 + r[c])
        rows.append((date, sum(sleeve_vals.values()) + cash))
    return pd.Series(dict(rows)).sort_index()


def p1_weights(hist):
    return pd.Series(0.25, index=hist.columns)


def p2_weights(hist):
    h = hist.tail(60)
    if len(h) < 60:
        return pd.Series(0.25, index=hist.columns)
    vol = h.std() * np.sqrt(252)
    inv = 1.0 / vol.replace(0, np.nan)
    inv = inv.replace([np.inf, -np.inf], np.nan).dropna()
    if inv.empty or inv.sum() <= 0:
        return pd.Series(0.25, index=hist.columns)
    w = inv / inv.sum()
    w = w.reindex(hist.columns).fillna(0.10)  # 缺波动数据的袖珍按下限 10%
    w = w.clip(0.10, 0.40)
    return w / w.sum()


def yearly(eq):
    out, prev = {}, 1.0
    for y, g in eq.groupby(eq.index.year):
        out[int(y)] = g.iloc[-1] / prev - 1
        prev = g.iloc[-1]
    return out


def main_a():
    cfg = yaml.safe_load(open(os.path.join(PROJECT, "config.yaml"), encoding="utf-8"))
    loader = DataLoader(cache_dir=os.path.join(PROJECT, "data_cache"))
    end = None
    print("生成四袖珍净值...", flush=True)
    navs, calendar = sleeve_navs(cfg, loader, end)
    rets = navs.pct_change().fillna(0.0)

    # 袖珍单体检视
    rows = []
    for name in navs.columns:
        for label, a, b in SEGMENTS:
            seg = navs[name][(navs.index >= a) & (navs.index <= b)] if b \
                else navs[name][navs.index >= a]
            m = perf_nav(seg / seg.iloc[0])
            rows.append({"item": name, "segment": label, **m})
        for y, r in yearly(navs[name]).items():
            rows.append({"item": f"{name}_yearly", "segment": y, "cagr": r})
    print("袖珍完成", flush=True)

    # 组合 P1/P2/P3
    ports = {
        "P1_等权月调": simulate_portfolio(rets, p1_weights),
        "P2_风险平价": simulate_portfolio(rets, p2_weights),
        "P3_P1+波动目标12%": simulate_portfolio(rets, p1_weights, vol_target=(0.12, 0.30)),
    }
    for name, eq in ports.items():
        for label, a, b in SEGMENTS:
            seg = eq[(eq.index >= a) & (eq.index <= b)] if b else eq[eq.index >= a]
            m = perf_nav(seg / seg.iloc[0])
            rows.append({"item": name, "segment": label, **m})
        for y, r in yearly(eq).items():
            rows.append({"item": f"{name}_yearly", "segment": y, "cagr": r})
        print(f"组合 {name} 完成", flush=True)

    # 基准：QQQ/510300 持有、四标的持有等权（股票 beta）
    loader2 = loader
    d300 = loader2.load("510300", is_etf=True, start="2019-01-01").set_index("date")["close"]
    qqq = navs["S3_美股donchian"]  # 已是策略，标的持有另算
    bench300 = d300 / d300.iloc[0]
    for label, a, b in SEGMENTS:
        seg = bench300[(bench300.index >= a) & (bench300.index <= b)] if b \
            else bench300[bench300.index >= a]
        m = perf_nav(seg / seg.iloc[0])
        rows.append({"item": "基准_510300持有", "segment": label, **m})

    result = pd.DataFrame(rows)
    result.to_csv(os.path.join(RESULTS, "r7a_portfolio.csv"),
                  index=False, encoding="utf-8-sig")

    # 相关矩阵
    corr = rets.corr()
    corr.to_csv(os.path.join(RESULTS, "r7a_corr.csv"), encoding="utf-8-sig")

    # P1 回撤归因：找 P1 最大回撤区间
    p1 = ports["P1_等权月调"]
    peak = p1.cummax()
    dd = p1 / peak - 1
    trough = dd.idxmin()
    peak_date = p1.loc[:trough].idxmax()
    contrib = {}
    for name in navs.columns:
        r = navs[name].loc[trough] / navs[name].loc[peak_date] - 1
        contrib[name] = {"sleeve_return": r, "contrib_to_dd": r * 0.25}
    attr = pd.DataFrame(contrib).T
    attr.to_csv(os.path.join(RESULTS, "r7a_dd_attribution.csv"), encoding="utf-8-sig")

    # 2022 剖析
    y2022 = pd.DataFrame({"P1": ports["P1_等权月调"], **{n: navs[n] for n in navs.columns}})
    y2022 = y2022[(y2022.index >= "2022-01-01") & (y2022.index <= "2022-12-31")]
    m2022 = (y2022.iloc[-1] / y2022.iloc[0] - 1).rename("return_2022")
    m2022.to_csv(os.path.join(RESULTS, "r7a_year2022.csv"), encoding="utf-8-sig")

    # 控制台摘要
    print("\n===== 袖珍与组合（full 全期） =====")
    full = result[(result["segment"] == "full") &
                  (~result["item"].str.contains("yearly"))]
    print(full.assign(cagr=full["cagr"].map(lambda x: f"{x:.1%}"),
                      mdd=full["mdd"].map(lambda x: f"{x:.1%}"),
                      sharpe=full["sharpe"].map(lambda x: f"{x:.2f}"),
                      calmar=full["calmar"].map(lambda x: f"{x:.2f}"),
                      turnover=full["turnover"].map(lambda x: "-"))
          [["item", "cagr", "mdd", "sharpe", "calmar"]].to_string(index=False))
    print("\n===== 分段卡玛 =====")
    piv = result[(~result["item"].str.contains("yearly"))].pivot_table(
        index="item", columns="segment", values="calmar")
    print(piv.map(lambda x: f"{x:.2f}" if pd.notna(x) else "-").to_string())
    print("\n===== 相关矩阵 =====")
    print(corr.round(2).to_string())
    print("\n===== P1 最大回撤归因（%s → %s） =====" % (peak_date.date(), trough.date()))
    print(attr.assign(sleeve_return=attr["sleeve_return"].map(lambda x: f"{x:.1%}"),
                      contrib_to_dd=attr["contrib_to_dd"].map(lambda x: f"{x:.1%}")).to_string())
    print("\n===== 2022 年收益 =====")
    print(m2022.map(lambda x: f"{x:.1%}").to_string())
    print(f"\n结果 CSV：{RESULTS}")


if __name__ == "__main__":
    main_a()
