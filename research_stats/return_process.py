"""T1 收益率过程统计刻画：ACF / 方差比 / Hurst——推导"趋势 vs 回归"的客观尺度。

产出：docs/RETURN_PROCESS_STATISTICS.md 数据 + research_stats/results/*.csv
"""
import os
import sys

import numpy as np
import pandas as pd

PROJECT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
os.makedirs(OUT, exist_ok=True)

INSTRUMENTS = {
    "510300": "data_cache/510300_qfq.parquet",
    "510500": "data_cache/510500_qfq.parquet",
    "159915": "data_cache/159915_qfq.parquet",
    "588000": "data_cache/588000_qfq.parquet",
    "518880": "data_cache/518880_qfq.parquet",
}
SEGMENTS = [("全期", "2017-01-01", "2099-01-01"),
            ("2019-2023", "2019-01-01", "2023-12-31"),
            ("2024+", "2024-01-01", "2099-01-01")]


def load_series(rel):
    df = pd.read_parquet(os.path.join(PROJECT, rel))
    df["date"] = pd.to_datetime(df["date"])
    return df.set_index("date").sort_index()["close"]


def load_us(sym):
    sys.path.insert(0, PROJECT)
    import us_equities.data as ud
    d = ud.load(sym, os.path.join(PROJECT, "paper_us", "cache"), offline=True)
    d["date"] = pd.to_datetime(d["date"])
    return d.set_index("date").sort_index()["close"]


def load_btc():
    p = os.path.join(PROJECT, "macro_research", "cache", "btc.csv")
    df = pd.read_csv(p, parse_dates=["date"])
    return df.set_index("date").sort_index()["close"]


def variance_ratio(r, q):
    """Lo-MacKinlay 方差比（不重叠版本），返回 (VR, z统计量)。"""
    n = len(r)
    if n < q * 2:
        return np.nan, np.nan
    mu = r.mean()
    var1 = ((r - mu) ** 2).sum() / (n - 1)
    rq = r.rolling(q).sum().dropna()
    varq = ((rq - q * mu) ** 2).sum() / (len(rq) * q)
    vr = varq / var1 if var1 > 0 else np.nan
    # 同方差假设下的 z（近似）
    z = np.sqrt(n) * (vr - 1) / np.sqrt(2 * (2 * q - 1) * (q - 1) / (3 * q)) if vr == vr else np.nan
    return vr, z


def hurst_dfa(x, scale_min=8, scale_max=64):
    """DFA Hurst 指数：H>0.5 趋势持续，<0.5 均值回归。"""
    x = np.asarray(x)
    y = np.cumsum(x - x.mean())
    scales = np.unique(np.floor(np.logspace(np.log10(scale_min),
                                            np.log10(scale_max), 10)).astype(int))
    F = []
    for s in scales:
        n = len(y) // s
        if n < 2:
            continue
        rms = []
        for k in range(n):
            seg = y[k * s:(k + 1) * s]
            t = np.arange(s)
            c = np.polyfit(t, seg, 1)
            fit = np.polyval(c, t)
            rms.append(np.sqrt(np.mean((seg - fit) ** 2)))
        F.append(np.mean(rms))
    if len(F) < 3:
        return np.nan
    h = np.polyfit(np.log(scales[:len(F)]), np.log(F), 1)[0]
    return h


def analyze(name, close):
    rows = []
    r = close.pct_change().dropna()
    for seg, a, b in SEGMENTS:
        rs = r[(r.index >= a) & (r.index <= b)]
        if len(rs) < 100:
            continue
        row = {"instrument": name, "segment": seg, "n": len(rs)}
        acf = [rs.autocorr(lag=k) for k in (1, 2, 5, 10, 20)]
        row.update({f"acf{k}": acf[i] for i, k in enumerate((1, 2, 5, 10, 20))})
        for q in (5, 10, 21, 63):
            vr, z = variance_ratio(rs, q)
            row[f"VR{q}"] = vr
            row[f"VRz{q}"] = z
        row["hurst"] = hurst_dfa(rs.values)
        rows.append(row)
    return rows


def main():
    rows = []
    for code, rel in INSTRUMENTS.items():
        rows += analyze(code, load_series(rel))
    for sym in ("QQQ", "SPY"):
        try:
            rows += analyze(sym, load_us(sym))
        except Exception as e:  # noqa: BLE001
            print(f"{sym} 跳过: {e}")
    try:
        rows += analyze("BTC", load_btc())
    except Exception as e:  # noqa: BLE001
        print(f"BTC 跳过: {e}")

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT, "return_process.csv"), index=False, encoding="utf-8-sig")
    pd.set_option("display.width", 260)
    show = df.copy()
    for c in show.columns:
        if c not in ("instrument", "segment", "n"):
            show[c] = show[c].map(lambda x: f"{x:.3f}" if pd.notna(x) else "-")
    print("===== 收益率过程统计 =====")
    print(show.to_string(index=False))
    print("\n解读：ACF>0 且 VR>1 且 Hurst>0.5 → 趋势性；ACF<0 且 VR<1 且 Hurst<0.5 → 均值回归")
    print("VR z 统计量 |z|>1.96 为 5% 显著")


if __name__ == "__main__":
    main()
