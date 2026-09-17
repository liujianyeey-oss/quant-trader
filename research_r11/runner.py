"""R11 编排：数据覆盖汇总 → M1/M2/M3 回测 → 相位 → 成本压力 → 容量 → 基准对比 → 相关性。"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from research_r11.signals import build_panels, make_signals, perf_stats  # noqa: E402
from research_r11.engine import run_portfolio, COST_STOCK  # noqa: E402

PROJECT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
os.makedirs(RESULTS, exist_ok=True)

SEG = [("train", "2019-01-01", "2023-12-31"), ("valid", "2024-01-01", None),
       ("full", "2019-01-01", None)]
VARIANTS = ["mom60", "lowvol60", "combo"]
V_CN = {"mom60": "M1动量top30", "lowvol60": "M2低波top30", "combo": "M3复合top30"}


def load_universe_data():
    uni = pd.read_csv(os.path.join(RESULTS, "universe.csv"), dtype={"code": str})
    uni["code"] = uni["code"].str.zfill(6)
    data, missing = {}, []
    for _, r in uni.iterrows():
        path = os.path.join(PROJECT, "data_cache", f"{r['code']}_qfq.parquet")
        if os.path.exists(path):
            df = pd.read_parquet(path)
            if len(df) >= 60:
                data[r["code"]] = df
        else:
            missing.append(r["code"])
    return data, missing, uni


def bench_series(name):
    import akshare as ak
    df = ak.stock_zh_index_daily(symbol=name)
    df["date"] = pd.to_datetime(df["date"])
    return df.set_index("date")["close"]


def eval_variant(panel, data, kind, initial, slippage=None):
    out = {}
    phase_cagrs, phase_mdds, phase_calmars = [], [], []
    for phase in range(1, 6):
        signals = make_signals(panel, kind, phase=phase)
        eq, trades, stats = run_portfolio(data, signals, "2019-01-01",
                                          panel.index[-1], initial, slippage)
        m = perf_stats(eq, trades, initial)
        phase_cagrs.append(m["cagr"])
        phase_mdds.append(m["mdd"])
        phase_calmars.append(m["calmar"])
        if phase == 1:
            out["_eq_phase1"] = (eq, trades, stats, m)
    for label, a, b in SEG:
        signals = make_signals(panel, kind, phase=1)
        e = pd.Timestamp(b) if b else panel.index[-1]
        eq, trades, stats = run_portfolio(data, signals, a, e, initial, slippage)
        m = perf_stats(eq, trades, initial)
        out[label] = {**m, **{f"skip_{k}": v for k, v in stats.items()}}
    out["phase"] = dict(cagr_median=np.median(phase_cagrs),
                        cagr_spread_pp=(max(phase_cagrs) - min(phase_cagrs)) * 100,
                        mdd_worst=min(phase_mdds),
                        calmar_median=np.median(phase_calmars))
    return out


def main():
    print("加载 universe 数据...", flush=True)
    data, missing, uni = load_universe_data()
    print(f"可用 {len(data)} 只，缺失 {len(missing)} 只")
    panel = build_panels(data)
    print(f"面板：{panel.shape[1]} 只 × {panel.shape[0]} 交易日")

    rows = []
    equities = {}
    for kind in VARIANTS:
        res = eval_variant(panel, data, kind, 1_000_000)
        for label in ("train", "valid", "full"):
            m = res[label]
            rows.append({"variant": V_CN[kind], "segment": label, "phase": 1, **m})
        p = res["phase"]
        rows.append({"variant": V_CN[kind], "segment": "phase_5档", "cagr": p["cagr_median"],
                     "mdd": p["mdd_worst"], "calmar": p["calmar_median"],
                     "sharpe": np.nan, "total": np.nan,
                     "fees": np.nan, "turnover": p["cagr_spread_pp"]})
        equities[V_CN[kind]] = res["_eq_phase1"][0]
        print(f"完成 {V_CN[kind]}", flush=True)

    # 50bps 滑点压力（M1，全期）
    print("50bps 压力...", flush=True)
    res50 = eval_variant(panel, data, "mom60", 1_000_000, slippage=0.005)
    rows.append({"variant": "M1动量top30(滑点50bps)", "segment": "full", "phase": 1,
                 **res50["full"]})
    # 容量 1000 万（M1，全期）
    print("容量 1000 万...", flush=True)
    res10m = eval_variant(panel, data, "mom60", 10_000_000)
    rows.append({"variant": "M1动量top30(资金1000万)", "segment": "full", "phase": 1,
                 **res10m["full"]})

    result = pd.DataFrame(rows)
    result.to_csv(os.path.join(RESULTS, "variants.csv"), index=False, encoding="utf-8-sig")

    # 基准对比
    print("基准...", flush=True)
    bench_rows = []
    try:
        csi800 = bench_series("sh000906")
        b300 = bench_series("sh000300")
        b500 = bench_series("sh000905")
        for bname, bs in (("中证800指数", csi800), ("沪深300指数", b300), ("中证500指数", b500)):
            bs = bs[bs.index >= "2019-01-01"]
            for label, a, b in SEG:
                seg = bs[(bs.index >= a) & (bs.index <= b)] if b else bs[bs.index >= a]
                y = ((seg.index[-1] - seg.index[0]).days + 1) / 365.25
                cagr = (seg.iloc[-1] / seg.iloc[0]) ** (1 / y) - 1
                mdd = (seg / seg.cummax() - 1).min()
                bench_rows.append({"benchmark": bname, "segment": label,
                                   "cagr": cagr, "mdd": mdd,
                                   "calmar": cagr / abs(mdd) if mdd < 0 else np.nan})
    except Exception as e:  # noqa: BLE001
        print(f"基准拉取失败：{e}")
    bench = pd.DataFrame(bench_rows)
    bench.to_csv(os.path.join(RESULTS, "benchmarks.csv"),
                 index=False, encoding="utf-8-sig")

    # 与生产 momentum 相关性
    print("相关性...", flush=True)
    import yaml
    from data import DataLoader
    from backtest import Backtester
    from strategies import MomentumRotation
    cfg = yaml.safe_load(open(os.path.join(PROJECT, "config.yaml"), encoding="utf-8"))
    loader = DataLoader(cache_dir=os.path.join(PROJECT, "data_cache"))
    pool = ["510300", "510500", "159915", "588000", "518880"]
    pdata = {c: loader.load(c, is_etf=True, start="2017-01-01") for c in pool}
    eq_mom, _ = Backtester(cfg).run(pdata, MomentumRotation(cfg).generate_signals(pdata),
                                    start="2019-01-01")
    mom_ret = eq_mom["total_value"].pct_change()
    corr_rows = []
    for name, eq in equities.items():
        j = pd.concat([eq.pct_change().rename("p"), mom_ret.rename("m")], axis=1).dropna()
        corr_rows.append({"variant": name, "corr_vs_prod_momentum": j["p"].corr(j["m"])})
    pd.DataFrame(corr_rows).to_csv(os.path.join(RESULTS, "corr.csv"),
                                   index=False, encoding="utf-8-sig")

    # 控制台摘要
    show = result.assign(
        cagr=result["cagr"].map(lambda x: f"{x:.1%}" if pd.notna(x) else "-"),
        mdd=result["mdd"].map(lambda x: f"{x:.1%}" if pd.notna(x) else "-"),
        sharpe=result["sharpe"].map(lambda x: f"{x:.2f}" if pd.notna(x) else "-"),
        calmar=result["calmar"].map(lambda x: f"{x:.2f}" if pd.notna(x) else "-"),
        fees=result["fees"].map(lambda x: f"{x:,.0f}" if pd.notna(x) else "-"),
        turnover=result["turnover"].map(lambda x: f"{x:.1f}" if pd.notna(x) else "-"),
        total=result["total"].map(lambda x: f"{x:.1%}" if pd.notna(x) else "-"))
    print("\n===== R11 变体结果 =====")
    cols = ["variant", "segment", "total", "cagr", "mdd", "sharpe", "calmar",
            "turnover", "fees"]
    skip_cols = [c for c in show.columns if c.startswith("skip_")]
    print(show[cols + skip_cols].to_string(index=False))
    print("\n===== 基准 =====")
    print(bench.assign(cagr=bench["cagr"].map(lambda x: f"{x:.1%}"),
                       mdd=bench["mdd"].map(lambda x: f"{x:.1%}"),
                       calmar=bench["calmar"].map(lambda x: f"{x:.2f}")).to_string(index=False))
    print("\n===== 与生产 momentum 相关 =====")
    print(pd.DataFrame(corr_rows).assign(
        corr_vs_prod_momentum=lambda d: d["corr_vs_prod_momentum"].map(lambda x: f"{x:.2f}"))
        .to_string(index=False))
    print(f"\n结果 CSV：{RESULTS}")


if __name__ == "__main__":
    main()
