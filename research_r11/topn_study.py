"""R11B: 持仓数敏感性（N=10/20/30）× M1动量/M2低波，幸存者池与 PIT 池对比。

复用 research_r11 引擎与 signals；PIT 逻辑复用 pit_interim。
输出：训练/验证/全期指标 + 相位分布 + PIT窗口(2019-2020)对比 + 整手约束 + 持仓集中度。
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from research_r11.signals import build_panels, make_signals, perf_stats  # noqa: E402
from research_r11.engine import run_portfolio  # noqa: E402
from research_r11.pit_interim import load_membership, load_all_data, make_pit_signals  # noqa: E402

PROJECT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
SEG = [("train", "2019-01-01", "2023-12-31"), ("valid", "2024-01-01", None),
       ("full", "2019-01-01", None)]
NS = [10, 20, 30]
KINDS = ["mom60", "lowvol60"]


def main():
    membership = load_membership()
    current = set(pd.read_csv(os.path.join(RESULTS, "universe.csv"),
                              dtype={"code": str})["code"].str.zfill(6))
    pit_union = set().union(*[c for _, c in membership])
    data = load_all_data(pit_union | current)
    panel_all = build_panels(data)
    panel_srv = build_panels({c: df for c, df in data.items() if c in current})
    print(f"数据 {len(data)} 只（幸存者 {panel_srv.shape[1]}）", flush=True)

    rows = []
    for kind in KINDS:
        for n in NS:
            # 幸存者池：训练/验证/全期（phase=1）+ 相位分布
            for label, a, b in SEG:
                sig = make_signals(panel_srv, kind, phase=1, top_n=n)
                e = pd.Timestamp(b) if b else panel_srv.index[-1]
                eq, tr, st = run_portfolio(data, sig, a, e, 1_000_000)
                m = perf_stats(eq, tr, 1_000_000)
                rows.append({"kind": kind, "N": n, "pool": "survivor",
                             "segment": label, **{k: m[k] for k in
                             ("cagr", "mdd", "sharpe", "calmar", "turnover")}})
            ph = []
            for p in range(1, 6):
                sig = make_signals(panel_srv, kind, phase=p, top_n=n)
                eq, tr, st = run_portfolio(data, sig, "2019-01-01",
                                           panel_srv.index[-1], 1_000_000)
                mm = perf_stats(eq, tr, 1_000_000)
                ph.append((mm["cagr"], mm["calmar"]))
            rows.append({"kind": kind, "N": n, "pool": "survivor",
                         "segment": "phase_med",
                         "cagr": np.median([x[0] for x in ph]),
                         "calmar": np.median([x[1] for x in ph]),
                         "phase_spread_pp": (max(x[0] for x in ph) -
                                             min(x[0] for x in ph)) * 100})
            # PIT 池 2019-2020
            sigp = make_pit_signals(panel_all, membership, kind, phase=1, top_n=n)
            eqp, trp, stp = run_portfolio(data, sigp, "2019-01-01", "2020-12-31", 1_000_000)
            mp = perf_stats(eqp, trp, 1_000_000)
            rows.append({"kind": kind, "N": n, "pool": "PIT",
                         "segment": "2019_2020", **{k: mp[k] for k in
                         ("cagr", "mdd", "sharpe", "calmar", "turnover")}})
            # 幸存者池同窗口（对照）
            sigs = make_signals(panel_srv, kind, phase=1, top_n=n)
            eqs, trs, sts = run_portfolio(data, sigs, "2019-01-01", "2020-12-31", 1_000_000)
            ms = perf_stats(eqs, trs, 1_000_000)
            rows.append({"kind": kind, "N": n, "pool": "survivor",
                         "segment": "2019_2020", **{k: ms[k] for k in
                         ("cagr", "mdd", "sharpe", "calmar", "turnover")}})
            print(f"完成 {kind} N={n}", flush=True)

    res = pd.DataFrame(rows)
    res.to_csv(os.path.join(RESULTS, "topn_study.csv"), index=False, encoding="utf-8-sig")

    # 整手约束：top10 候选里，目标权重 10% 在 ¥100万/¥500万 下能否买 100 股
    lot_rows = []
    for kind in KINDS:
        for nav in (1_000_000, 5_000_000):
            sig = make_signals(panel_srv, kind, phase=1, top_n=10)
            picks = [c for _, w in sig for c in w]
            prices = []
            for c in set(picks):
                df = data.get(c)
                if df is not None:
                    prices.append({"code": c, "last_price": df["close"].iloc[-1]})
            pr = pd.DataFrame(prices)
            budget = 0.10 * nav
            blocked = pr[pr["last_price"] * 100 > budget]
            lot_rows.append({"kind": kind, "nav": nav, "n_picks": len(pr),
                             "blocked_by_lot": len(blocked),
                             "min_budget_needed": (pr["last_price"].quantile(0.95) * 100)
                             if len(pr) else np.nan})
    pd.DataFrame(lot_rows).to_csv(os.path.join(RESULTS, "topn_lot.csv"),
                                  index=False, encoding="utf-8-sig")

    # 持仓集中度：M1/M2 top10 全期出现频次最高的 10 只
    freq_rows = []
    for kind in KINDS:
        sig = make_signals(panel_srv, kind, phase=1, top_n=10)
        cnt = {}
        for _, w in sig:
            for c in w:
                cnt[c] = cnt.get(c, 0) + 1
        top = pd.Series(cnt).sort_values(ascending=False).head(10)
        names = pd.read_csv(os.path.join(RESULTS, "universe.csv"),
                            dtype={"code": str}).set_index(
            pd.read_csv(os.path.join(RESULTS, "universe.csv"),
                        dtype={"code": str})["code"].str.zfill(6))["name"]
        for c, v in top.items():
            freq_rows.append({"kind": kind, "code": c,
                              "name": names.get(c, ""), "count": v,
                              "total_rebalances": len(sig)})
    pd.DataFrame(freq_rows).to_csv(os.path.join(RESULTS, "topn_concentration.csv"),
                                   index=False, encoding="utf-8-sig")

    pd.set_option("display.width", 220)
    show = res.assign(
        cagr=res["cagr"].map(lambda x: f"{x:.1%}" if pd.notna(x) else "-"),
        mdd=res["mdd"].map(lambda x: f"{x:.1%}" if pd.notna(x) else "-"),
        sharpe=res["sharpe"].map(lambda x: f"{x:.2f}" if pd.notna(x) else "-"),
        calmar=res["calmar"].map(lambda x: f"{x:.2f}" if pd.notna(x) else "-"))
    print(show[["kind", "N", "pool", "segment", "cagr", "mdd", "sharpe",
                "calmar"]].to_string(index=False))
    print("\n相位极差（pp）:")
    print(res[res["segment"] == "phase_med"][["kind", "N", "cagr", "calmar",
                                              "phase_spread_pp"]].to_string(index=False))
    print("\n整手约束:")
    print(pd.DataFrame(lot_rows).to_string(index=False))
    print("\n集中度 top10 频次:")
    print(pd.DataFrame(freq_rows).to_string(index=False))


if __name__ == "__main__":
    main()
