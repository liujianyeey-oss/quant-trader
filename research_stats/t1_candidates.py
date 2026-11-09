"""T1 派生候选检验（T2 sprint）：统计结构推导的三个假设。

C1 gold-MR：518880 价格>MA60 且 RSI(2)<10 → 次日开盘买入；收盘>MA5 卖出（单标的均值回归）
C2 BTC 长窗：Donchian 40/20、63/21、63 日纯动量 vs 基线 20/10（都带 F1 量能确认 1.5×）
C3 Hurst 动态池门控：生产 momentum 调仓日，剔除过去 250 日 DFA Hurst < 0.48 的动量候选
声明：单标的切换类用"T+1 生效 + 切换时双侧成本"的收盘近似回测（行为判定级，与事件研究同口径）。
"""
import os
import sys

import numpy as np
import pandas as pd

PROJECT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT)
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
WINDOWS = [("train", "2019-01-01", "2023-12-31"), ("valid", "2024-01-01", None),
           ("full", "2019-01-01", None)]


def load(code):
    df = pd.read_parquet(os.path.join(PROJECT, "data_cache", f"{code}_qfq.parquet"))
    df["date"] = pd.to_datetime(df["date"])
    return df.set_index("date").sort_index()


def load_btc():
    p = os.path.join(PROJECT, "macro_research", "cache", "btc.csv")
    df = pd.read_csv(p, parse_dates=["date"])
    return df.set_index("date").sort_index()


def rsi(close, n=2):
    delta = close.diff()
    up = delta.clip(lower=0)
    down = -delta.clip(upper=0)
    au = up.ewm(alpha=1 / n, adjust=False).mean()
    ad = down.ewm(alpha=1 / n, adjust=False).mean()
    rs = au / ad.replace(0, np.nan)
    return 100 - 100 / (1 + rs)


def switch_bt(df, entry_sig, exit_sig, fee=0.0001 + 0.001):
    """entry_sig/exit_sig: 收盘产生的布尔序列；T+1 生效；返回净值与统计。"""
    pos = pd.Series(index=df.index, dtype=float)
    cur = 0.0
    for i, d in enumerate(df.index):
        if i > 0:
            ps = entry_sig.iloc[i - 1]
            xs = exit_sig.iloc[i - 1]
            if xs and cur > 0:
                cur = 0.0
            elif ps and cur == 0:
                cur = 1.0
        pos.iloc[i] = cur
    r = df["close"].pct_change().fillna(0.0)
    ret = pos * r
    switches = int((pos != pos.shift()).sum())
    cost = switches * fee * 2
    eq = (1 + ret).cumprod() - cost / len(ret) * np.arange(1, len(ret) + 1)
    return eq, pos, switches, cost


def perf(eq):
    if len(eq) < 50:
        return {}
    years = max(((eq.index[-1] - eq.index[0]).days + 1) / 365.25, 1e-9)
    total = eq.iloc[-1] - 1
    cagr = (1 + total) ** (1 / years) - 1 if total > -1 else -1
    mdd = (eq / eq.cummax() - 1).min()
    r = eq.pct_change().dropna()
    sharpe = r.mean() / r.std() * np.sqrt(252) if r.std() > 0 else np.nan
    return dict(cagr=cagr, mdd=mdd, sharpe=sharpe,
                calmar=cagr / abs(mdd) if mdd < 0 else np.nan)


def donchian_signal(df, entry, exit_):
    c, h, l = df["close"], df["high"], df["low"]
    hi = h.shift(1).rolling(entry).max()
    lo = l.shift(1).rolling(exit_).min()
    pos = pd.Series(np.nan, index=df.index)
    pos[c > hi] = 1.0
    pos[c < lo] = 0.0
    return pos.ffill().fillna(0.0)


def donchian_bt(df, entry, exit_, use_f1=True):
    pos = donchian_signal(df, entry, exit_)
    if use_f1 and "volume" in df:
        vok = df["volume"] > df["volume"].shift(1).rolling(20).mean() * 1.5
        # F1：突破需放量确认；无确认保持原仓
        raw = donchian_signal(df, entry, exit_)
        c, h = df["close"], df["high"].shift(1).rolling(entry).max()
        is_entry = (c > h) & (raw == 1) & (raw.shift(1) != 1)
        blocked = is_entry & (~vok)
        pos2 = raw.copy()
        pos2[blocked] = np.nan
        pos = pos2.ffill().fillna(0.0)
    r = df["close"].pct_change().fillna(0.0)
    ret = pos.shift(1).fillna(0.0) * r
    switches = int((pos != pos.shift()).sum())
    cost = switches * 0.002 * 2   # taker 10bps + slippage 10bps, 双侧
    eq = (1 + ret).cumprod() - cost / len(ret) * np.arange(1, len(ret) + 1)
    return eq, pos, switches


def hurst_dfa(x, scale_min=8, scale_max=64):
    x = np.asarray(x)
    y = np.cumsum(x - x.mean())
    scales = np.unique(np.floor(np.logspace(np.log10(scale_min), np.log10(scale_max), 8)).astype(int))
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
            rms.append(np.sqrt(np.mean((seg - np.polyval(c, t)) ** 2)))
        F.append(np.mean(rms))
    if len(F) < 3:
        return np.nan
    return np.polyfit(np.log(scales[:len(F)]), np.log(F), 1)[0]


def main():
    rows = []

    # ===== C1 gold-MR =====
    g = load("518880")
    ma60 = g["close"].rolling(60).mean()
    r2 = rsi(g["close"], 2)
    ma5 = g["close"].rolling(5).mean()
    entry = (g["close"] > ma60) & (r2 < 10)
    exit_ = g["close"] > ma5
    eq, pos, sw, cost = switch_bt(g, entry, exit_)
    hold = g["close"] / g["close"].iloc[0]
    for label, a, b in WINDOWS:
        seg = eq.loc[a:b] if b else eq.loc[a:]
        segh = hold.loc[a:b] if b else hold.loc[a:]
        m, mh = perf(seg / seg.iloc[0]), perf(segh / segh.iloc[0])
        rows.append({"cand": "C1_goldMR", "segment": label, "switches": sw,
                     **{k: m.get(k) for k in ("cagr", "mdd", "sharpe", "calmar")},
                     "bench_cagr": mh.get("cagr"), "bench_mdd": mh.get("mdd")})

    # ===== C2 BTC 长窗 =====
    b = load_btc()
    for name, en, ex, f1 in [("C2_base_20_10", 20, 10, True),
                             ("C2_40_20", 40, 20, True),
                             ("C2_63_21", 63, 21, True),
                             ("C2_63omentum", None, None, None)]:
        if name == "C2_63omentum":
            pos = (b["close"] > b["close"].shift(63)).astype(float)
            r = b["close"].pct_change().fillna(0.0)
            switches = int((pos != pos.shift()).sum())
            cost = switches * 0.002 * 2
            eq = (1 + pos.shift(1).fillna(0.0) * r).cumprod() - cost / len(r) * np.arange(1, len(r) + 1)
        else:
            eq, pos, switches = donchian_bt(b, en, ex, use_f1=f1)
        bh = b["close"] / b["close"].iloc[0]
        for label, a, bb_ in [("train", "2019-01-01", "2022-12-30"),
                              ("valid", "2023-01-01", "2024-12-31"),
                              ("2025+", "2025-01-01", None),
                              ("full", "2019-01-01", None)]:
            seg = eq.loc[a:bb_] if bb_ else eq.loc[a:]
            if len(seg) < 30:
                continue
            segb = bh.loc[a:bb_] if bb_ else bh.loc[a:]
            m, mh = perf(seg / seg.iloc[0]), perf(segb / segb.iloc[0])
            rows.append({"cand": name, "segment": label, "switches": switches,
                         **{k: m.get(k) for k in ("cagr", "mdd", "sharpe", "calmar")},
                         "bench_cagr": mh.get("cagr"), "bench_mdd": mh.get("mdd")})

    # ===== C3 Hurst 动态池门控 =====
    import yaml
    from strategies import MomentumRotation
    cfg = yaml.safe_load(open(os.path.join(PROJECT, "config.yaml"), encoding="utf-8"))
    pool = cfg["universe"]["etfs"]
    defen = cfg["universe"]["defensive"]
    pdata = {c: pd.read_parquet(os.path.join(PROJECT, "data_cache", f"{c}_qfq.parquet"))
             for c in pool + [defen]}
    panel = pd.DataFrame({c: df["close"] for c, df in pdata.items()}).sort_index()
    hursts = {}
    for c in pool:
        r = panel[c].pct_change()
        h = r.rolling(250).apply(lambda x: hurst_dfa(x) if len(x) == 250 else np.nan,
                                 raw=False)
        hursts[c] = h
    hurst_df = pd.DataFrame(hursts)

    # 生产信号（基线）
    strat = MomentumRotation(cfg)
    sig_base = strat.generate_signals(pdata)
    # 门控信号：调仓日若动量冠军 Hurst<0.48 则视为不合格 → 切防守
    sig_gated = sig_base.copy()
    for d in sig_base.index:
        champ = sig_base.loc[d]
        champ = champ[champ > 0]
        if len(champ) == 0:
            continue
        c = champ.index[0]
        if c != defen and d in hurst_df.index and pd.notna(hurst_df.loc[d, c]) \
                and hurst_df.loc[d, c] < 0.48:
            sig_gated.loc[d] = 0.0
            sig_gated.loc[d, defen] = 1.0
    from backtest import Backtester
    bt = Backtester(cfg)
    for tag, sig in (("C3_base", sig_base), ("C3_hurst_gate", sig_gated)):
        eqt, _ = bt.run(pdata, sig)
        eqt = eqt["total_value"]
        for label, a, bb_ in WINDOWS:
            seg = eqt.loc[a:bb_] if bb_ else eqt.loc[a:]
            m = perf(seg / seg.iloc[0])
            rows.append({"cand": f"C3_{tag}", "segment": label,
                         **{k: m.get(k) for k in ("cagr", "mdd", "sharpe", "calmar")}})
    # 门控触发次数
    diffs = (sig_gated != sig_base).any(axis=1).sum()
    rows.append({"cand": "C3_gate_triggers", "segment": "info", "switches": int(diffs)})

    res = pd.DataFrame(rows)
    res.to_csv(os.path.join(OUT, "t1_candidates.csv"), index=False, encoding="utf-8-sig")
    pd.set_option("display.width", 260)
    show = res.assign(
        cagr=res["cagr"].map(lambda x: f"{x:.1%}" if pd.notna(x) else "-"),
        mdd=res["mdd"].map(lambda x: f"{x:.1%}" if pd.notna(x) else "-"),
        sharpe=res["sharpe"].map(lambda x: f"{x:.2f}" if pd.notna(x) else "-"),
        calmar=res["calmar"].map(lambda x: f"{x:.2f}" if pd.notna(x) else "-"),
        bench_cagr=lambda d: d["bench_cagr"].map(lambda x: f"{x:.1%}" if pd.notna(x) else "-"),
        bench_mdd=lambda d: d["bench_mdd"].map(lambda x: f"{x:.1%}" if pd.notna(x) else "-"))
    print(show.to_string(index=False))


if __name__ == "__main__":
    main()
