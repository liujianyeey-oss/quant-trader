"""条件胜率评估器：基于"今日状态"的历史同形态统计，输出后续收益的胜率分布。

口径固定（违者改算新机制）：
- 状态指纹：MA200 上/下、20日/60日动量、距20日高点、距10日低点、量比
- 严格档阈值：c>MA200 且 20日动量∈[-8%,-3%] 且 60日动量>+10% 且 距20日高<-5% 且 距10日低<+3% 且 量比<1.3x
- 样本 <30 自动放宽：先去量比条件，再去距10日低条件，最后放宽动量带——并明确报告实际用档
- 独立事件数：用 20 日间隔去聚簇，报"净独立样本"
- 输出 T+5/10/20/60 的胜率/中位/均值/P25/P75 + 无条件基准（MA200 上方全样本）
"""
import os

import numpy as np
import pandas as pd

PROJECT = os.path.dirname(os.path.abspath(__file__))
HORIZONS = (5, 10, 20, 60)


def load_asset(symbol: str) -> pd.DataFrame:
    """返回带 date/open/high/low/close/volume 的 DataFrame。"""
    if symbol.upper() == "BTC":
        from macro_research.data_layer import load_btc
        df, _ = load_btc()
        return df.rename(columns=str.lower)
    if symbol.upper() in ("QQQ", "SPY"):
        p = os.path.join(PROJECT, "paper_us", "cache", f"{symbol.upper()}.csv")
        df = pd.read_csv(p, parse_dates=["date"])
        return df
    p = os.path.join(PROJECT, "data_cache", f"{symbol}_qfq.parquet")
    if os.path.exists(p):
        df = pd.read_parquet(p)
        df["date"] = pd.to_datetime(df["date"])
        if "volume" not in df.columns:
            df["volume"] = np.nan
        return df
    raise FileNotFoundError(f"未找到标的 {symbol} 的数据（支持 BTC/QQQ/SPY/或本地缓存代码）")


def fingerprint(d: pd.DataFrame) -> dict:
    c = d["close"]; h = d["high"]; l = d["low"]
    ma200 = c.rolling(200).mean()
    hi20 = h.rolling(20).max().shift(1)
    lo10 = l.rolling(10).min().shift(1)
    mom20 = c / c.shift(20) - 1
    mom60 = c / c.shift(60) - 1
    vr = d["volume"] / d["volume"].rolling(20).mean().shift(1)
    feats = {
        "ma200": ma200, "hi20": hi20, "lo10": lo10,
        "mom20": mom20, "mom60": mom60, "vratio": vr,
    }
    i = -1
    state = {
        "close": float(c.iloc[i]),
        "above_ma200": bool(c.iloc[i] > ma200.iloc[i]),
        "mom20": float(mom20.iloc[i]),
        "mom60": float(mom60.iloc[i]),
        "dist_hi20": float(c.iloc[i] / hi20.iloc[i] - 1),
        "dist_lo10": float(c.iloc[i] / lo10.iloc[i] - 1),
        "vratio": float(vr.iloc[i]) if pd.notna(vr.iloc[i]) else np.nan,
    }
    return {"series": feats, "state": state}


STRICT = dict(mom20=(-0.08, -0.03), mom60_min=0.10, dist_hi_max=-0.05,
              dist_lo_max=0.03, vr_max=1.3)
RELAXED = dict(mom20=(None, 0.0), mom60_min=0.0, dist_hi_max=-0.05,
               dist_lo_max=None, vr_max=None)


def build_mask(f, tier):
    mask = f["mom20"].notna() & f["ma200"].notna() & (f["mom60"] > tier["mom60_min"])
    mask &= f["ma200"] < f["close_px"]
    if tier["mom20"][0] is not None:
        mask &= f["mom20"].between(*tier["mom20"])
    else:
        mask &= f["mom20"] < 0
    mask &= (f["close_px"] / f["hi20"] - 1) < tier["dist_hi_max"]
    if tier["dist_lo_max"] is not None:
        mask &= (f["close_px"] / f["lo10"] - 1) < tier["dist_lo_max"]
    if tier["vr_max"] is not None:
        mask &= (f["vratio"] < tier["vr_max"]) | f["vratio"].isna()
    return mask


def independent_clusters(idx: pd.DatetimeIndex, gap_days=20) -> int:
    """按日期间隔聚簇，返回独立事件数（相邻匹配 <gap 天视为同一事件）。"""
    if len(idx) == 0:
        return 0
    days = pd.Series(idx).sort_values().diff().dt.days.fillna(gap_days)
    return int((days >= gap_days).sum())


def eval_winrate(symbol: str, asof: str | None = None) -> dict:
    d = load_asset(symbol)
    d = d.set_index(pd.to_datetime(d["date"])).sort_index()
    if "volume" not in d.columns:
        d["volume"] = np.nan
    if asof:
        d = d.loc[:asof]
    fp = fingerprint(d)
    f = {
        "close_px": d["close"], "ma200": fp["series"]["ma200"],
        "hi20": fp["series"]["hi20"], "lo10": fp["series"]["lo10"],
        "mom20": fp["series"]["mom20"], "mom60": fp["series"]["mom60"],
        "vratio": fp["series"]["vratio"],
    }
    tiers = [("严格档", STRICT), ("放宽档", RELAXED)]
    used_tier, mask = None, None
    for name, tier in tiers:
        m = build_mask(f, tier)
        if m.sum() >= 30 or name == "放宽档":
            used_tier, mask = name, m
            break
    c = d["close"]
    tiers_out = []
    for name, tier in tiers:
        m = build_mask(f, tier)
        idx = m[m].index
        rows = []
        for hz in HORIZONS:
            sel = (c.shift(-hz) / c - 1).loc[idx].dropna()
            if len(sel) < 1:
                continue
            rows.append({
                "horizon": hz, "n": len(sel), "win": (sel > 0).mean(),
                "median": sel.median(), "mean": sel.mean(),
                "p25": sel.quantile(.25), "p75": sel.quantile(.75),
            })
        tiers_out.append({"name": name, "n": int(m.sum()),
                          "n_clusters": independent_clusters(idx),
                          "show": len(rows) > 0, "rows": rows})
    # 无条件基准
    base_idx = (d["close"] > f["ma200"])[(d["close"] > f["ma200"])].index
    base = []
    for hz in (5, 20, 60):
        sel = (c.shift(-hz) / c - 1).loc[base_idx].dropna()
        base.append({"horizon": hz, "n": len(sel), "win": (sel > 0).mean(),
                     "median": sel.median()})
    return {"symbol": symbol, "asof": d.index[-1], "state": fp["state"],
            "tiers": tiers_out, "base": base}


def fmt_report(r: dict) -> str:
    s = r["state"]
    lines = [
        f"===== 条件胜率评估：{r['symbol']}（截至 {r['asof']:%Y-%m-%d}）=====",
        f"今日状态: close {s['close']:,.0f} | {'>' if s['above_ma200'] else '<'}MA200 | "
        f"20日动量 {s['mom20']:+.1%} | 60日动量 {s['mom60']:+.1%} | "
        f"距20日高 {s['dist_hi20']:+.1%} | 距10日低 {s['dist_lo10']:+.1%} | 量比 {s['vratio']:.2f}x",
        "",
    ]
    for t in r["tiers"]:
        if not t["show"]:
            continue
        warn = "（样本薄，警惕！）" if t["n_clusters"] < 15 else ""
        lines.append(f"【{t['name']}】匹配天数 {t['n']}，独立事件 {t['n_clusters']}{warn}")
        for row in t["rows"]:
            lines.append(
                f"  T+{row['horizon']:>2}日 | 胜率 {row['win']:.0%} | 中位 {row['median']:+.1%} | "
                f"均值 {row['mean']:+.1%} | P25 {row['p25']:+.1%} | P75 {row['p75']:+.1%}")
        lines.append("")
    lines.append("无条件基准（MA200 上方全样本）: " +
                 " | ".join(f"T+{b['horizon']} 胜率 {b['win']:.0%}(中位 {b['median']:+.1%})"
                            for b in r["base"]))
    lines.append("提示：独立事件数 << 匹配天数时按独立事件计置信度；历史频率，非投资建议。")
    return "\n".join(lines)


if __name__ == "__main__":
    import sys
    sym = sys.argv[1] if len(sys.argv) > 1 else "BTC"
    asof = sys.argv[2] if len(sys.argv) > 2 else None
    print(fmt_report(eval_winrate(sym, asof)))
