"""全品类 ETF 策略研究编排：数据覆盖 → 策略电池 → 分段绩效 → 相位扰动 →
相关矩阵 → 组合层，结果 CSV 存 research/results/。

纪律：
- 训练 ≤2022-12-30 选优；验证 2023-2024 体检；2025+ 只报告（已被 us_swing 开发看过，标注污染）
- 统一引擎与成本（0.005/股 min$1 + 卖出0.5bps + 滑点10bps），$50,000，t收盘→t+1开盘
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from us_equities.research.etf_data import load_universe, ALL_ETFS
from us_equities.research.etf_engine import EtfBacktester, perf, yearly_returns, INITIAL_USD
from us_equities.research import etf_strategies as S

RESULTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")

SEGMENTS = [("full", "2019-01-02", None),
            ("train", "2019-01-02", "2022-12-30"),
            ("valid", "2023-01-01", "2024-12-31"),
            ("report", "2025-01-01", None)]  # 污染段：只报告


def build_variants(closes, data):
    """返回 {name: dict(family, signals, weekly, note)}。"""
    V = {}

    def add(name, family, signals, weekly, note=""):
        V[name] = dict(family=family, signals=signals, weekly=weekly, note=note)

    # A. 动量轮动族
    for w in (10, 20, 60, 120):
        for top in (1, 2):
            add(f"mom{w}_top{top}", "A动量轮动", S.mom_rotation(closes, w, top), True,
                f"{w}日动量top{top}周调")
    for top in (1, 2):
        add(f"mom126_21_top{top}", "A动量轮动", S.mom_126_21(closes, top), True,
            f"经典12-2动量top{top}周调")
    add("comp20_60_120_top2", "A动量轮动", S.comp_momentum(closes, 2), True, "复合动量top2")
    add("dm60_shy_top1", "A动量轮动", S.dual_momentum(closes, absolute="shy"), True,
        "双动量(胜SHY过滤,否则SHY)")
    add("dm60_zero_top1", "A动量轮动", S.dual_momentum(closes, absolute="zero"), True,
        "双动量(>0过滤,否则SHY)")
    add("invvol_top2", "A动量轮动", S.invvol_top2(closes), True, "top2逆波动加权")

    # B. 趋势族
    add("gtaa_ief", "B趋势", S.gtaa(closes, off=S.IEF), True, "SPY>MA200持top2否则IEF")
    add("gtaa_shy", "B趋势", S.gtaa(closes, off=S.SHY), True, "SPY>MA200持top2否则SHY")
    add("macross_SPY", "B趋势", S.ma_cross(closes, "SPY"), False, "SPY MA50/200")
    add("macross_QQQ", "B趋势", S.ma_cross(closes, "QQQ"), False, "QQQ MA50/200")
    add("donchian_SPY", "B趋势", S.donchian(data, "SPY"), False, "SPY唐奇安20/10")
    add("donchian_QQQ", "B趋势", S.donchian(data, "QQQ"), False, "QQQ唐奇安20/10")

    # C. 均值回归族
    add("rsi2_SPY", "C均值回归", S.rsi2_connors(closes, "SPY"), False, "SPY RSI2<10/MA200")
    add("rsi2_QQQ", "C均值回归", S.rsi2_connors(closes, "QQQ"), False, "QQQ RSI2<10/MA200")
    add("boll_SPY", "C均值回归", S.bollinger_revert(closes, "SPY"), False, "SPY布林z<-2")
    add("boll_QQQ", "C均值回归", S.bollinger_revert(closes, "QQQ"), False, "QQQ布林z<-2")

    # D. 多因子族
    add("mf_mom_invvol", "D多因子", S.mf_mom_invvol(closes), True, "复合动量top2逆波动")
    add("us_staggered", "D多因子", S.us_staggered(closes), False,
        "5组分批+波动控仓(A股移植)")

    # 基准：首日建立并持有
    first = closes.index[0]
    for sym in ("SPY", "QQQ"):
        w = pd.DataFrame(0.0, index=[first], columns=closes.columns)
        w.iloc[0, w.columns.get_loc(sym)] = 1.0
        add(f"bench_{sym}", "基准", w, False, f"{sym}买入持有(含成本)")
    return V


def run_segments(bt, data, signals):
    out = {}
    for label, a, b in SEGMENTS:
        end = b if b else max(df["date"].max() for df in data.values())
        eq, trades = bt.run(data, signals, a, end)
        out[label] = (eq, trades, perf(eq, trades))
    return out


def main(offline=False):
    os.makedirs(RESULTS, exist_ok=True)
    print("加载数据...", flush=True)
    data, coverage = load_universe(offline=offline)
    coverage.to_csv(os.path.join(RESULTS, "coverage.csv"), index=False, encoding="utf-8-sig")
    dropped = coverage[coverage["备注"] != ""]
    print(f"数据覆盖：{len(data)} 只可用" + (f"，剔除 {len(dropped)} 只" if len(dropped) else ""))
    print(coverage.to_string(index=False))

    closes = S.close_panel(data)
    variants = build_variants(closes, data)
    bt = EtfBacktester()

    summary_rows, yearly_rows, equities = [], [], {}
    for name, v in variants.items():
        segs = run_segments(bt, data, v["signals"])
        equities[name] = segs["full"][0]
        row = {"strategy": name, "family": v["family"], "note": v["note"]}
        for label, (eq, trades, m) in segs.items():
            for k, val in m.items():
                row[f"{label}_{k}"] = val
        summary_rows.append(row)
        yr = yearly_returns(segs["full"][0])
        for y, r in yr.items():
            yearly_rows.append({"strategy": name, "year": y, "return": r})
        print(f"完成 {name}", flush=True)

    summary = pd.DataFrame(summary_rows)
    summary.to_csv(os.path.join(RESULTS, "summary.csv"), index=False, encoding="utf-8-sig")
    pd.DataFrame(yearly_rows).to_csv(os.path.join(RESULTS, "yearly.csv"),
                                     index=False, encoding="utf-8-sig")

    # ---- 相位扰动（调仓周期>1日的策略） ----
    print("\n相位扰动测试...", flush=True)
    phase_rows = []
    full_end = max(df["date"].max() for df in data.values())
    for name, v in variants.items():
        if not v["weekly"]:
            continue
        cagrs, mdds = [], []
        for phase in range(5):
            sig = _regen_with_phase(name, closes, phase)
            if sig is None:
                break
            eq, trades = bt.run(data, sig, "2019-01-02", full_end)
            m = perf(eq, trades)
            cagrs.append(m["cagr"])
            mdds.append(m["mdd"])
        if cagrs:
            phase_rows.append({"strategy": name,
                               "cagr_min": min(cagrs), "cagr_median": np.median(cagrs),
                               "cagr_max": max(cagrs),
                               "mdd_worst": min(mdds), "mdd_best": max(mdds),
                               "n_phases": len(cagrs)})
    phase_df = pd.DataFrame(phase_rows)
    phase_df.to_csv(os.path.join(RESULTS, "phase.csv"), index=False, encoding="utf-8-sig")

    # ---- 相关矩阵（全期日收益） ----
    rets = pd.DataFrame({n: eq.pct_change() for n, eq in equities.items()}).dropna(how="all")
    corr = rets.corr()
    corr.to_csv(os.path.join(RESULTS, "correlation.csv"), encoding="utf-8-sig")

    # ---- 组合层：按训练段卡玛选 3 个不同族的优胜者等权 ----
    cand = summary[~summary["family"].isin(["基准"])].copy()
    cand = cand.sort_values("train_calmar", ascending=False)
    picked, families = [], set()
    for _, r in cand.iterrows():
        if r["family"] in families:
            continue
        picked.append(r["strategy"])
        families.add(r["family"])
        if len(picked) == 3:
            break
    port_ret = rets[picked].dropna().mean(axis=1)
    port_eq = INITIAL_USD * (1 + port_ret).cumprod()
    port_eq = pd.concat([pd.Series([INITIAL_USD],
                                   index=[port_ret.index[0] - pd.Timedelta(days=1)]),
                         port_eq]).sort_index()
    m_port = perf(port_eq, [])
    port_rows = [{"strategy": "portfolio_top3", "constituents": "+".join(picked), **m_port}]
    for name in picked:
        m = perf(equities[name], [])
        port_rows.append({"strategy": name, "constituents": "", **m})
    pd.DataFrame(port_rows).to_csv(os.path.join(RESULTS, "portfolio.csv"),
                                   index=False, encoding="utf-8-sig")

    # ---- 控制台摘要 ----
    show = summary[["strategy", "family", "train_cagr", "train_mdd", "train_calmar",
                    "valid_cagr", "valid_mdd", "valid_calmar",
                    "report_cagr", "report_mdd", "full_cagr", "full_mdd",
                    "full_sharpe", "full_calmar", "full_turnover", "full_fees"]].copy()
    show = show.sort_values("train_calmar", ascending=False)
    pct_cols = [c for c in show.columns if any(k in c for k in ("cagr", "mdd"))]
    show[pct_cols] = show[pct_cols].map(lambda x: f"{x:.1%}" if pd.notna(x) else "N/A")
    for c in ("train_calmar", "valid_calmar", "full_sharpe", "full_calmar"):
        show[c] = show[c].map(lambda x: f"{x:.2f}" if pd.notna(x) else "N/A")
    show["full_turnover"] = show["full_turnover"].map(lambda x: f"{x:.1f}x")
    show["full_fees"] = show["full_fees"].map(lambda x: f"${x:,.0f}")
    print("\n===== 策略电池总表（按训练段卡玛排序） =====")
    print(show.to_string(index=False))
    print("\n===== 相位扰动（全期 CAGR 范围） =====")
    print(phase_df.assign(**{c: phase_df[c].map(lambda x: f"{x:.1%}")
                             for c in ("cagr_min", "cagr_median", "cagr_max",
                                       "mdd_worst", "mdd_best")}).to_string(index=False))
    print("\n===== 组合层 =====")
    print(pd.DataFrame(port_rows)[["strategy", "constituents", "cagr", "mdd",
                                   "sharpe", "calmar"]].assign(
        cagr=lambda d: d["cagr"].map(lambda x: f"{x:.1%}"),
        mdd=lambda d: d["mdd"].map(lambda x: f"{x:.1%}"),
        sharpe=lambda d: d["sharpe"].map(lambda x: f"{x:.2f}"),
        calmar=lambda d: d["calmar"].map(lambda x: f"{x:.2f}")).to_string(index=False))
    print(f"\n结果 CSV 已保存：{RESULTS}")
    print(f"组合层入选（训练段卡玛，各族取优）：{picked}")


def _regen_with_phase(name, closes, phase):
    """按相位重新生成周调类策略信号。"""
    if name.startswith("mom126_21"):
        top = int(name[-1])
        return S.mom_126_21(closes, top, phase=phase)
    if name.startswith("mom"):
        w = int(name[3:name.index("_")])
        top = int(name[-1])
        return S.mom_rotation(closes, w, top, phase=phase)
    if name == "comp20_60_120_top2":
        return S.comp_momentum(closes, 2, phase=phase)
    if name == "dm60_shy_top1":
        return S.dual_momentum(closes, absolute="shy", phase=phase)
    if name == "dm60_zero_top1":
        return S.dual_momentum(closes, absolute="zero", phase=phase)
    if name == "invvol_top2":
        return S.invvol_top2(closes, phase=phase)
    if name == "gtaa_ief":
        return S.gtaa(closes, off=S.IEF, phase=phase)
    if name == "gtaa_shy":
        return S.gtaa(closes, off=S.SHY, phase=phase)
    if name == "mf_mom_invvol":
        return S.mf_mom_invvol(closes, phase=phase)
    return None


if __name__ == "__main__":
    main(offline="--offline" in sys.argv)
