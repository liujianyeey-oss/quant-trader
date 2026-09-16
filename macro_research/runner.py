"""宏观 regime 过滤器研究编排：数据覆盖 → 基准策略 → H1/H2/H3 过滤 →
分段绩效（训练判效、验证体检、2025+ 只报告）。结果 CSV 存 results/。
"""
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from macro_research import data_layer as DL
from macro_research import strategies as MS

RESULTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
SEGMENTS = [("full", "2019-01-02", None), ("train", "2019-01-02", "2022-12-30"),
            ("valid", "2023-01-01", "2024-12-31"), ("report", "2025-01-01", None)]

BTC_COST = {"commission": 0.001, "slippage": 0.001, "lot": 1e-4}   # taker 10bps + 滑点10bps
ETF513130_COST = {"commission": 0.0001, "slippage": 0.001, "lot": 100}  # 万1 + 0.1%，100份


def coverage_table():
    rows = []
    btc, btc_note = DL.load_btc()
    rows.append({"标的/输入": "BTC(BTCUSDT 币安)", "起始日": str(btc["date"].min().date()),
                 "截止日": str(btc["date"].max().date()), "行数": len(btc),
                 "备注": btc_note or "365天日历，不含当日未完成K线"})
    dxy, dxy_note = DL.load_dxy()
    rows.append({"标的/输入": "DXY代理(FRED DTWEXBGS)", "起始日": str(dxy["date"].min().date()),
                 "截止日": str(dxy["date"].max().date()), "行数": len(dxy),
                 "备注": dxy_note or "贸易加权美元指数（广义），FRED 发布有 1-2 天滞后"})
    uy, uy_note = DL.load_us10y()
    rows.append({"标的/输入": "美债10Y(bond_zh_us_rate)", "起始日": str(uy["date"].min().date()),
                 "截止日": str(uy["date"].max().date()), "行数": len(uy),
                 "备注": uy_note or "akshare 中国债券信息网"})
    hs, hs_note = DL.load_513130()
    rows.append({"标的/输入": "513130 恒生科技ETF", "起始日": str(hs["date"].min().date()),
                 "截止日": str(hs["date"].max().date()), "行数": len(hs),
                 "备注": (hs_note or "") + "；2021-05 上市，历史短、统计功效低（已声明）"})
    for name, a, b, n, note in DL.load_oil_gold_notes():
        rows.append({"标的/输入": name, "起始日": a, "截止日": b, "行数": n,
                     "备注": note + "；预先声明的 H1-H3 未使用"})
    return pd.DataFrame(rows), btc, dxy, uy, hs


def eval_segments(df, w, cost, end):
    out = {}
    for label, a, b in SEGMENTS:
        e = pd.Timestamp(b) if b else end
        eq, trades = MS.run_single_asset(df, w, a, e, cost)
        out[label] = MS.perf(eq, trades)
    return out


def main():
    os.makedirs(RESULTS, exist_ok=True)
    print("加载数据...", flush=True)
    cov, btc, dxy, uy, hs = coverage_table()
    cov.to_csv(os.path.join(RESULTS, "coverage.csv"), index=False, encoding="utf-8-sig")
    print(cov.to_string(index=False))

    end_btc = btc["date"].max()
    end_hs = hs["date"].max()
    rows = []

    # ---- BTC：唐奇安 20/10 × {无过滤, H1, H2, H3} ----
    base = MS.btc_donchian(btc)
    flags = MS.macro_flags(pd.DatetimeIndex(btc["date"]), dxy, uy)
    for fname in ("none", "H1", "H2", "H3"):
        w = base if fname == "none" else MS.apply_filter(base, flags[fname])
        res = eval_segments(btc, w, BTC_COST, end_btc)
        for label, m in res.items():
            rows.append({"asset": "BTC", "strategy": "donchian20/10", "filter": fname,
                         "segment": label, **m})
        print(f"BTC {fname} done", flush=True)

    # ---- 513130：动量 × {无过滤, H1, H2, H3} ----
    base_hs = MS.hstech_momentum(hs)
    flags_hs = MS.macro_flags(pd.DatetimeIndex(hs["date"]), dxy, uy)
    for fname in ("none", "H1", "H2", "H3"):
        w = base_hs if fname == "none" else MS.apply_filter(base_hs, flags_hs[fname])
        res = eval_segments(hs, w, ETF513130_COST, end_hs)
        for label, m in res.items():
            rows.append({"asset": "513130", "strategy": "mom20+MA60", "filter": fname,
                         "segment": label, **m})
        print(f"513130 {fname} done", flush=True)

    # ---- QQQ：donchian 20/10 × {无过滤, H2}（H1/H3 与 H2 同族不重复测? —— 按声明 H1-H3 都测） ----
    from us_equities.research.etf_data import load_universe
    from us_equities.research import etf_strategies as US_S
    from us_equities.research.etf_engine import EtfBacktester, perf as us_perf
    us_data, _ = load_universe(symbols=["QQQ"], cache=os.path.join(
        DL.PROJECT, "us_equities", "research", "cache"), offline=True)
    don_q = US_S.donchian(us_data, "QQQ", 20, 10)
    closes = US_S.close_panel(us_data)
    qqq_dates = closes.index
    flags_q = MS.macro_flags(qqq_dates, dxy, uy)
    for fname in ("none", "H1", "H2", "H3"):
        if fname == "none":
            sig = don_q
        else:
            w = MS.apply_filter(don_q["QQQ"], flags_q[fname])
            sig = pd.DataFrame({"QQQ": w})
        bt = EtfBacktester()
        end_q = us_data["QQQ"]["date"].max()
        for label, a, b in SEGMENTS:
            e = pd.Timestamp(b) if b else end_q
            eq, trades = bt.run(us_data, sig, a, e)
            m = us_perf(eq, trades)
            rows.append({"asset": "QQQ", "strategy": "donchian20/10", "filter": fname,
                         "segment": label, **m})
        print(f"QQQ {fname} done", flush=True)

    result = pd.DataFrame(rows)
    result.to_csv(os.path.join(RESULTS, "hypothesis_tests.csv"),
                  index=False, encoding="utf-8-sig")

    # ---- 判定汇总：训练段卡玛改善 ----
    print("\n===== 假设检验汇总（训练段判效：卡玛改善≥0.1 算有效） =====")
    summary_rows = []
    for asset in ("BTC", "513130", "QQQ"):
        sub = result[(result["asset"] == asset) & (result["segment"] == "train")]
        base_calmar = sub[sub["filter"] == "none"]["calmar"].iloc[0]
        for fname in ("H1", "H2", "H3"):
            f_calmar = sub[sub["filter"] == fname]["calmar"].iloc[0]
            delta = f_calmar - base_calmar
            verdict = "有效" if delta >= 0.1 else "无效"
            # 验证段体检
            subv = result[(result["asset"] == asset) & (result["segment"] == "valid")]
            v_base = subv[subv["filter"] == "none"]["calmar"].iloc[0]
            v_filt = subv[subv["filter"] == fname]["calmar"].iloc[0]
            summary_rows.append({"asset": asset, "filter": fname,
                                 "train_base": base_calmar, "train_filtered": f_calmar,
                                 "train_delta": delta, "verdict_train": verdict,
                                 "valid_base": v_base, "valid_filtered": v_filt,
                                 "valid_delta": v_filt - v_base})
    summ = pd.DataFrame(summary_rows)
    summ.to_csv(os.path.join(RESULTS, "verdicts.csv"), index=False, encoding="utf-8-sig")
    print(summ.assign(**{c: summ[c].map(lambda x: f"{x:+.2f}") for c in
                         ("train_base", "train_filtered", "train_delta",
                          "valid_base", "valid_filtered", "valid_delta")}).to_string(index=False))

    # 触发频率（训练段空仓占比）
    trig_rows = []
    for asset, cal in (("BTC", pd.DatetimeIndex(btc["date"])),
                       ("513130", pd.DatetimeIndex(hs["date"])),
                       ("QQQ", qqq_dates)):
        fl = MS.macro_flags(cal, dxy, uy)
        train_mask = (cal >= "2019-01-02") & (cal <= "2022-12-30")
        for fname in ("H1", "H2", "H3"):
            trig_rows.append({"asset": asset, "filter": fname,
                              "train_flat_share": fl[fname][train_mask].mean()})
    pd.DataFrame(trig_rows).to_csv(os.path.join(RESULTS, "trigger_frequency.csv"),
                                   index=False, encoding="utf-8-sig")
    print("\n===== 训练段触发频率（空仓时间占比） =====")
    print(pd.DataFrame(trig_rows).assign(
        train_flat_share=lambda d: d["train_flat_share"].map(lambda x: f"{x:.0%}")).to_string(index=False))

    # 关键数字表（full/train/valid/report 每策略）
    print("\n===== 全量结果（节选：train/valid 卡玛） =====")
    piv = result.pivot_table(index=["asset", "filter"], columns="segment",
                             values="calmar")[["train", "valid", "report", "full"]]
    print(piv.map(lambda x: f"{x:.2f}" if pd.notna(x) else "N/A").to_string())
    print(f"\n结果 CSV 已保存：{RESULTS}")


if __name__ == "__main__":
    main()
