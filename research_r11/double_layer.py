"""R12 双层策略「定档×选股」：宽基 MA60 定档 + 横截面动量 top10 选股 + 518880 防守。

预声明规则（写死）：
- 每个月首个交易日（phase 可选 1~5）收盘定档：510300/510500/159915/588000 中
  收盘价 > 各自 MA60 的数量 >= 2 → 持股档；否则防守档
- 持股档：当期合格股票池（PIT 成员、>=250 根K线、当日有价）按 60 日动量取 top10 等权
- 防守档：100% 518880
- 信号 T 收盘 → T+1 开盘执行；引擎成本/涨跌停/顺延沿用（防御 ETF 也按个股费率，偏保守声明）
- 输出：幸存者池与 PIT 池（2019-2020 窗口）对照、训练/验证/全期、5 相位分布
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from research_r11.signals import build_panels, perf_stats, monthly_signal_dates  # noqa: E402
from research_r11.engine import run_portfolio  # noqa: E402
from research_r11.pit_interim import load_membership, load_all_data  # noqa: E402

PROJECT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
ETF_TIMING = ["510300", "510500", "159915", "588000"]
DEFENSIVE = "518880"
MOM_W, TOP_N, MIN_BARS = 60, 10, 250
WINDOWS = [("train", "2019-01-01", "2023-12-31"), ("valid", "2024-01-01", None),
           ("full", "2019-01-01", None), ("pit", "2019-01-01", "2020-12-31")]


def load_etf(df_loader_codes):
    d = {}
    for c in ETF_TIMING + [DEFENSIVE]:
        path = os.path.join(PROJECT, "data_cache", f"{c}_qfq.parquet")
        if os.path.exists(path):
            d[c] = pd.read_parquet(path)
    return d


def make_double_layer_signals(panel, timing_df, membership, phase, top_n=TOP_N,
                              pit_only=False):
    calendar = panel.index
    dates = monthly_signal_dates(calendar, phase)
    t = timing_df.set_index("date").sort_index()
    ma60 = {c: t[c].rolling(60).mean() for c in ETF_TIMING}
    mom = panel / panel.shift(MOM_W) - 1
    bars = panel.notna().cumsum()
    signals = []
    for d in dates:
        if d not in t.index:
            continue
        i = calendar.get_loc(d)
        if i < MIN_BARS:
            continue
        timing_row = t.loc[d]
        above = sum(1 for c in ETF_TIMING
                    if pd.notna(ma60[c].get(d)) and timing_row[c] > ma60[c][d])
        if above >= 2:
            # 持股档：横截面动量 top10
            elig = panel.columns[(bars.iloc[i] >= MIN_BARS) & panel.iloc[i].notna()]
            if pit_only:
                active = None
                for snap_date, codes in membership:
                    if snap_date <= d:
                        active = codes
                if active is None:
                    continue
                elig = [c for c in elig if c in active]
            if len(elig) < top_n:
                continue
            winners = mom.iloc[i][elig].dropna().nlargest(top_n).index
            w = {c: 1.0 / len(winners) for c in winners}
        else:
            w = {DEFENSIVE: 1.0}
        signals.append((d, w))
    return signals


def eval_run(label, data, panel, timing_df, membership, pit_only):
    out = {}
    ph_cagr, ph_calmar = [], []
    for p in range(1, 6):
        sig = make_double_layer_signals(panel, timing_df, membership, p, pit_only=pit_only)
        if not sig:
            continue
        end = "2020-12-31" if label == "pit" else panel.index[-1]
        start = "2019-01-01"
        eq, tr, st = run_portfolio(data, sig, start, end, 1_000_000)
        m = perf_stats(eq, tr, 1_000_000)
        ph_cagr.append(m["cagr"]); ph_calmar.append(m["calmar"])
        if p == 1:
            out["phase1"] = m
    out["phase_med_cagr"] = float(np.median(ph_cagr)) if ph_cagr else np.nan
    out["phase_med_calmar"] = float(np.median(ph_calmar)) if ph_calmar else np.nan
    out["phase_spread_pp"] = (max(ph_cagr) - min(ph_cagr)) * 100 if ph_cagr else np.nan
    # 分段
    for seg, a, b in WINDOWS:
        if seg == "pit" and not pit_only:
            continue
        sig = make_double_layer_signals(panel, timing_df, membership, 1, pit_only=pit_only)
        if not sig:
            continue
        e = pd.Timestamp(b) if b else panel.index[-1]
        if seg == "valid" and e < pd.Timestamp("2024-01-02"):
            continue
        eq, tr, st = run_portfolio(data, sig, a, e, 1_000_000)
        out[seg] = perf_stats(eq, tr, 1_000_000)
    return out


def main():
    membership = load_membership()
    current = set(pd.read_csv(os.path.join(RESULTS, "universe.csv"),
                              dtype={"code": str})["code"].str.zfill(6))
    pit_union = set().union(*[c for _, c in membership])
    data = load_all_data(pit_union | current)
    panel_all = build_panels(data)
    panel_srv = build_panels({c: df for c, df in data.items() if c in current})
    etf_data = load_etf(None)
    timing_rows = []
    for c in ETF_TIMING:
        df = etf_data[c][["date", "close"]].rename(columns={"close": c})
        timing_rows.append(df.set_index("date"))
    timing_df = pd.concat(timing_rows, axis=1).reset_index().rename(columns={"index": "date"})
    data.update(etf_data)

    rows = []
    # 幸存者池：全区间
    r = eval_run("full", data, panel_srv, timing_df, membership, pit_only=False)
    for seg in ("train", "valid", "full"):
        m = r.get(seg)
        if m:
            rows.append({"pool": "survivor", "segment": seg,
                         **{k: m[k] for k in ("cagr", "mdd", "sharpe", "calmar")}})
    rows.append({"pool": "survivor", "segment": "phase_med",
                 "cagr": r["phase_med_cagr"], "calmar": r["phase_med_calmar"],
                 "phase_spread_pp": r["phase_spread_pp"]})
    # PIT 池：2019-2020
    r2 = eval_run("pit", data, panel_all, timing_df, membership, pit_only=True)
    m = r2.get("pit")
    if m:
        rows.append({"pool": "PIT", "segment": "2019_2020",
                     **{k: m[k] for k in ("cagr", "mdd", "sharpe", "calmar")}})
    # PIT 窗口下幸存者池对照（同 pit_only=False 但截到 2020-12-31）
    sig = make_double_layer_signals(panel_srv, timing_df, membership, 1, pit_only=False)
    eq, tr, st = run_portfolio(data, sig, "2019-01-01", "2020-12-31", 1_000_000)
    m3 = perf_stats(eq, tr, 1_000_000)
    rows.append({"pool": "survivor", "segment": "2019_2020",
                 **{k: m3[k] for k in ("cagr", "mdd", "sharpe", "calmar")}})

    res = pd.DataFrame(rows)
    res.to_csv(os.path.join(RESULTS, "double_layer.csv"), index=False, encoding="utf-8-sig")
    show = res.assign(
        cagr=res["cagr"].map(lambda x: f"{x:.1%}" if pd.notna(x) else "-"),
        mdd=res["mdd"].map(lambda x: f"{x:.1%}" if pd.notna(x) else "-"),
        sharpe=res["sharpe"].map(lambda x: f"{x:.2f}" if pd.notna(x) else "-"),
        calmar=res["calmar"].map(lambda x: f"{x:.2f}" if pd.notna(x) else "-"))
    pd.set_option("display.width", 200)
    print(show.to_string(index=False))


if __name__ == "__main__":
    main()
