"""事件片段行为研究：四个策略体系在 5 个国际大事件窗口中的实际行为。

声明：片段级描述性研究，仓位按信号 T+1 生效、不计费用滑点（行为方向研究非绩效报告）。
产出：docs/EVENT_EPISODES.md 数据 + event_episodes/results/episodes.csv
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import yaml  # noqa: E402

PROJECT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
os.makedirs(OUT, exist_ok=True)

EPISODES = [
    ("E1_疫情冲击", "2020-01-20", "2020-04-30"),
    ("E2_白马崩盘", "2021-02-10", "2021-05-31"),
    ("E3_俄乌加息", "2022-02-24", "2022-10-31"),
    ("E4_924政策急转", "2024-09-13", "2024-11-08"),
    ("E5_关税冲击", "2025-03-27", "2025-05-31"),
]


def load_cache(code, is_etf=True):
    return pd.read_parquet(os.path.join(PROJECT, "data_cache", f"{code}_qfq.parquet"))


def series_positions_A():
    """A股 momentum / staggered 的日仓位（标的代码序列）。"""
    from strategies import MomentumRotation, StaggeredMomentum
    cfg = yaml.safe_load(open(os.path.join(PROJECT, "config.yaml"), encoding="utf-8"))
    pool = cfg["universe"]["etfs"] + [cfg["universe"]["defensive"]]
    data = {c: load_cache(c) for c in pool}
    mom = MomentumRotation(cfg).generate_signals(data)
    try:
        stag = StaggeredMomentum(cfg).generate_signals(data)
    except AttributeError:
        stag = StaggeredMomentum(cfg, **{}).generate_signals(data)
    return data, mom, stag


def position_share(signals, calendar):
    """把调仓日目标权重 ffill 成日仓位，返回每标的日权重 panel。"""
    w = signals.reindex(calendar).ffill().fillna(0.0)
    return w


def daily_returns(data, calendar):
    closes = {c: df.set_index("date")["close"] for c, df in data.items()}
    panel = pd.DataFrame(closes).sort_index().reindex(calendar)
    return panel.pct_change()


def donchian_positions(df, entry=20, exit_=10, vol_confirm=False):
    """唐奇安 20/10 仓位序列（T 收盘信号，T+1 生效由调用方 shift）。"""
    d = df.set_index("date").sort_index()
    c, h, l = d["close"], d["high"], d["low"]
    hi = h.shift(1).rolling(entry).max()
    lo = l.shift(1).rolling(exit_).min()
    pos = pd.Series(np.nan, index=d.index)
    pos[c > hi] = 1.0
    pos[c < lo] = 0.0
    if vol_confirm:
        v = d["volume"]
        vok = v > v.shift(1).rolling(20).mean() * 1.5
        pos[(c > hi) & (~vok)] = np.nan  # 无确认不入场，保持原仓
    return pos.ffill().fillna(0.0)


def episode_stats(name, a, b, strat_ret, bench_ret, pos=None):
    wr = strat_ret.loc[a:b].dropna()
    br = bench_ret.loc[a:b].dropna()
    eq_s = (1 + wr).cumprod()
    eq_b = (1 + br).cumprod()
    mdd_s = (eq_s / eq_s.cummax() - 1).min() if len(eq_s) else np.nan
    mdd_b = (eq_b / eq_b.cummax() - 1).min() if len(eq_b) else np.nan
    out = {"episode": name,
           "strat_ret": eq_s.iloc[-1] - 1 if len(eq_s) else np.nan,
           "bench_ret": eq_b.iloc[-1] - 1 if len(eq_b) else np.nan,
           "strat_mdd": mdd_s, "bench_mdd": mdd_b}
    if pos is not None:
        p = pos.loc[a:b]
        out["defensive_share"] = float((p == 0).mean())
        out["switches"] = int((p != p.shift()).sum() - 1)
    return out


def main():
    rows = []
    # ---- A股 ----
    data, mom_sig, stag_sig = series_positions_A()
    cal = mom_sig.index.union(stag_sig.index)
    rets = daily_returns(data, cal)
    for sname, sig in (("A股momentum", mom_sig), ("A股staggered", stag_sig)):
        w = position_share(sig, cal)
        strat_ret = (w.shift(1) * rets).sum(axis=1)
        defensive_pos = (w.get("518880", pd.Series(0, index=cal)) == 1).astype(float) if "518880" in w else None
        bench = rets["510300"]
        for name, a, b in EPISODES:
            rows.append({"strategy": sname, "bench": "510300",
                         **episode_stats(name, a, b, strat_ret, bench, defensive_pos)})
    # ---- 美股 donchian ----
    for sym in ("QQQ", "SPY"):
        import us_equities.data as ud
        d = ud.load(sym, os.path.join(PROJECT, "paper_us", "cache"), offline=True)
        pos = donchian_positions(d)
        r = d.set_index("date").sort_index()["close"].pct_change()
        strat_ret = pos.shift(1).reindex(r.index).fillna(0.0) * r
        for name, a, b in EPISODES:
            rows.append({"strategy": f"US_donchian_{sym}", "bench": sym,
                         **episode_stats(name, a, b, strat_ret, r, pos)})
    # ---- BTC ----
    btc_path = os.path.join(PROJECT, "macro_research", "cache", "btc.csv")
    if os.path.exists(btc_path):
        bd = pd.read_csv(btc_path, parse_dates=["date"])
        pos = donchian_positions(bd, vol_confirm=True)
        r = bd.set_index("date").sort_index()["close"].pct_change()
        strat_ret = pos.shift(1).reindex(r.index).fillna(0.0) * r
        for name, a, b in EPISODES:
            rows.append({"strategy": "BTC_donchian_F1", "bench": "BTC",
                         **episode_stats(name, a, b, strat_ret, r, pos)})

    res = pd.DataFrame(rows)
    res.to_csv(os.path.join(OUT, "episodes.csv"), index=False, encoding="utf-8-sig")
    pd.set_option("display.width", 250)
    show = res.assign(
        strat_ret=res["strat_ret"].map(lambda x: f"{x:.1%}" if pd.notna(x) else "-"),
        bench_ret=res["bench_ret"].map(lambda x: f"{x:.1%}" if pd.notna(x) else "-"),
        strat_mdd=res["strat_mdd"].map(lambda x: f"{x:.1%}" if pd.notna(x) else "-"),
        bench_mdd=res["bench_mdd"].map(lambda x: f"{x:.1%}" if pd.notna(x) else "-"),
        defensive_share=res["defensive_share"].map(lambda x: f"{x:.0%}" if pd.notna(x) else "-"))
    print(show.to_string(index=False))


if __name__ == "__main__":
    main()
