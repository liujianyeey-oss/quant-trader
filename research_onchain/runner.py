"""OC1 评估：用生产引擎（research_r4）判定链上否决开关 O1/O2 + 哈希带宽 O3 输出。

两段式纪律：判定只以生产引擎（T+1 开盘 + taker/滑点）为准。
基线 = BTC donchian 20/10 + F1。
"""
import os
import sys

import numpy as np
import pandas as pd

PROJECT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT)
CACHE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cache")
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
os.makedirs(OUT, exist_ok=True)

from research_r4.runner import (  # noqa: E402
    btc_donchian_filtered, f1_volume_mask, eval_btc)
from macro_research.data_layer import load_btc  # noqa: E402

SEG = [("train", "2019-01-02", "2022-12-30"),
       ("valid", "2023-01-01", "2024-12-31"),
       ("report", "2025-01-01", None), ("full", "2019-01-02", None)]


def s(name):
    df = pd.read_csv(os.path.join(CACHE, f"{name}.csv"), parse_dates=["date"])
    return df.set_index("date").sort_index().iloc[:, 0]


def block_mask_o1(btc_dates):
    """O1：净流入 7 日累计 > 250 日 90 分位 → 该日禁新开仓。数据 T 日 T 日可用（收盘后）。"""
    f = s("FlowNetExch")
    roll = f.rolling(7).sum()
    th = roll.rolling(250, min_periods=60).quantile(0.90)
    blocked = roll > th
    return blocked.reindex(btc_dates).fillna(False)


def block_mask_o2(btc_dates, price):
    """O2：价格 20 日 +5% 以上且活跃地址 20 日变化 <0 → 暂停新开仓。"""
    a = s("AdrActCnt")
    p_up = price.pct_change(20) > 0.05
    a_dn = a.pct_change(20) < 0
    blocked = (p_up & a_dn.reindex(price.index).ffill()).fillna(False)
    return blocked


def main():
    btc, note = load_btc()
    d = btc.set_index("date").sort_index()
    f1 = f1_volume_mask(btc)

    coverage = []
    for name in ["hash_rate", "n_transactions", "n_unique_addresses",
                 "AdrActCnt", "TxCnt", "FlowNetExch", "SplyExchBal", "NVTAdj"]:
        p = os.path.join(CACHE, f"{name}.csv")
        if os.path.exists(p):
            df = pd.read_csv(p, parse_dates=["date"])
            coverage.append({"metric": name, "rows": len(df),
                             "start": str(df["date"].min().date()),
                             "end": str(df["date"].max().date())})
        else:
            coverage.append({"metric": name, "rows": 0, "start": "-", "end": "-"})
    cov = pd.DataFrame(coverage)
    cov.to_csv(os.path.join(OUT, "coverage.csv"), index=False)
    print("== 数据覆盖 ==")
    print(cov.to_string(index=False))

    rows = []
    # 基线
    w_base = btc_donchian_filtered(btc, allow_entry=f1)
    res = eval_btc(btc, w_base, d.index.max(), SEG)
    for label, m in res.items():
        rows.append({"variant": "baseline_20_10_F1", "segment": label,
                     **{k: m[k] for k in ("cagr", "mdd", "sharpe", "calmar", "trades")}})

    # O1
    if os.path.exists(os.path.join(CACHE, "FlowNetExch.csv")):
        o1 = block_mask_o1(d.index)
        allow = f1 & (~o1)
        w = btc_donchian_filtered(btc, allow_entry=allow)
        res = eval_btc(btc, w, d.index.max(), SEG)
        for label, m in res.items():
            rows.append({"variant": "O1_抛压阻塞", "segment": label,
                         "blocked_days": int(o1.sum()),
                         **{k: m[k] for k in ("cagr", "mdd", "sharpe", "calmar", "trades")}})
    else:
        print("O1 跳过：无 FlowNetExch 数据")

    # O2
    if os.path.exists(os.path.join(CACHE, "AdrActCnt.csv")):
        o2 = block_mask_o2(d.index, d["close"])
        allow = f1 & (~o2)
        w = btc_donchian_filtered(btc, allow_entry=allow)
        res = eval_btc(btc, w, d.index.max(), SEG)
        for label, m in res.items():
            rows.append({"variant": "O2_价活背离", "segment": label,
                         "blocked_days": int(o2.sum()),
                         **{k: m[k] for k in ("cagr", "mdd", "sharpe", "calmar", "trades")}})
    else:
        print("O2 跳过：无 AdrActCnt 数据")

    res_df = pd.DataFrame(rows)
    res_df.to_csv(os.path.join(OUT, "o1_o2_verdicts.csv"), index=False, encoding="utf-8-sig")

    # O3 哈希带宽（只输出状态）
    if os.path.exists(os.path.join(CACHE, "hash_rate.csv")):
        h = s("hash_rate")
        dd = h / h.rolling(60).max() - 1
        cur = dd.iloc[-1]
        in_capitulation = dd < -0.30
        print(f"\nO3 哈希带宽：当前回撤 {cur:.1%}（阈值 -30%）"
              f"{'⚠️ 处于恐慌区间' if in_capitulation.iloc[-1] else '正常区间'}")

    pd.set_option("display.width", 200)
    show = res_df.assign(
        cagr=res_df["cagr"].map(lambda x: f"{x:.1%}" if pd.notna(x) else "-"),
        mdd=res_df["mdd"].map(lambda x: f"{x:.1%}" if pd.notna(x) else "-"),
        sharpe=res_df["sharpe"].map(lambda x: f"{x:.2f}" if pd.notna(x) else "-"),
        calmar=res_df["calmar"].map(lambda x: f"{x:.2f}" if pd.notna(x) else "-"))
    print("\n== O1/O2 判定 ==")
    print(show.to_string(index=False))


if __name__ == "__main__":
    main()
