"""R8：动态/自适应指标四假设。

H1 KAMA(10,2,30) 替代固定 MA60：a) 生产 momentum 趋势过滤；b) donchian_QQQ 退出线。
H2 ATR 压缩突破：ATR14/价 < 120 日 20% 分位（压缩态）+ 收盘破 20 日高入场，10 日低退出。
H3 相关性 regime（二元开关，判效门槛提高到卡玛 +0.15）：4 宽基 20 日平均两两相关
  > 250 日均值+1σ → 该调仓周期进防守。
H4 多周期确认：周线收盘 > 30 周均线才允许持股（调仓日取最近完成周）。
训练 2019-2023 判效（卡玛改善≥0.1 且回撤不劣化），验证 2024+ 体检，2025+ 只报告。
"""
import copy
import os
import sys

import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data import DataLoader  # noqa: E402
from backtest import Backtester  # noqa: E402
from metrics import calc_metrics  # noqa: E402
from research_r1.runner import momentum_signals, POOL_V0  # noqa: E402
from us_equities.research.etf_data import load_universe  # noqa: E402
from us_equities.research import etf_strategies as US_S  # noqa: E402
from us_equities.research.etf_engine import EtfBacktester, perf as us_perf  # noqa: E402

PROJECT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
os.makedirs(RESULTS, exist_ok=True)

DEFENSIVE = "518880"
MOM, REB = 20, 5
SEGMENTS = [("train", "2019-01-01", "2023-12-31"), ("valid", "2024-01-01", None),
            ("full", "2019-01-01", None)]


# ---------- KAMA ----------

def kama(close: pd.Series, er_n=10, fast=2, slow=30):
    """Kaufman 自适应均线：ER=|Δclose(er_n)|/Σ|Δclose|(er_n)，SC 在 fast/slow 间自适应。"""
    er = (close - close.shift(er_n)).abs() / close.diff().abs().rolling(er_n).sum()
    er = er.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    fast_sc, slow_sc = 2 / (fast + 1), 2 / (slow + 1)
    sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
    out = pd.Series(np.nan, index=close.index, dtype=float)
    prev = close.iloc[0]
    for i in range(len(close)):
        prev = prev + sc.iloc[i] * (close.iloc[i] - prev)
        out.iloc[i] = prev
    return out


# ---------- 参数化动量信号（生产框架 + 可选趋势线/周线确认/相关性 regime） ----------

def momentum_signals_v2(data, pool, phase=0, trend=None, weekly_confirm=False,
                        corr_regime=False):
    """trend: None=MA60(生产原样) / pd.Series(如 KAMA，用于所有标的) 或 dict{code: Series}。"""
    panel = pd.DataFrame({c: df.set_index("date")["close"] for c, df in data.items()}).sort_index()
    mom = panel / panel.shift(MOM) - 1
    if trend is None:
        trend_line = {c: panel[c].rolling(60).mean() for c in panel.columns}
    elif isinstance(trend, pd.Series):
        trend_line = {c: trend.reindex(panel.index) for c in panel.columns}
    else:
        trend_line = trend
    has_def = DEFENSIVE in panel.columns

    # H4：周线收盘 > 30 周均线（最近完成周）
    weekly_ok = None
    if weekly_confirm:
        weekly_ok = {}
        for c in panel.columns:
            wk = panel[c].resample("W-FRI").last().dropna()
            wk_ma = wk.rolling(30).mean()
            ok = (wk > wk_ma).astype(float)
            weekly_ok[c] = ok.reindex(panel.index.union(ok.index)).sort_index().ffill() \
                .reindex(panel.index)

    # H3：相关性 regime（T-1）：4 宽基 6 个配对的 20 日滚动相关均值
    regime_def = None
    if corr_regime:
        codes = [c for c in pool if c in panel.columns]
        rets = panel[codes].pct_change()
        pairs = [(a, b) for i, a in enumerate(codes) for b in codes[i + 1:]]
        pair_c = pd.DataFrame({
            f"{a}_{b}": rets[a].rolling(20).corr(rets[b]) for a, b in pairs})
        avg_corr = pair_c.mean(axis=1)
        m = avg_corr.rolling(250, min_periods=60).mean()
        sd = avg_corr.rolling(250, min_periods=60).std()
        regime_def = (avg_corr > m + sd)

    rows = []
    for i in range(MOM + 1 + phase, len(panel), REB):
        date = panel.index[i]
        m = mom.iloc[i][[c for c in pool if c in panel.columns]].dropna()
        if m.empty:
            continue
        best = m.idxmax()
        trend_ok = pd.notna(trend_line[best].iloc[i]) \
            and panel.iloc[i][best] >= trend_line[best].iloc[i]
        if trend_ok and weekly_confirm:
            ok_w = weekly_ok[best].iloc[i]
            trend_ok = pd.notna(ok_w) and ok_w > 0
        if regime_def is not None and bool(regime_def.iloc[i]):
            trend_ok = False
        if trend_ok:
            rows.append((date, best))
        elif has_def:
            rows.append((date, DEFENSIVE))
        else:
            rows.append((date, None))

    signals = pd.DataFrame(0.0, index=[d for d, _ in rows], columns=list(panel.columns))
    for date, code in rows:
        if code is not None:
            signals.loc[date, code] = 1.0
    return signals


def eval_phases(bt, data, pool, end, initial, **kw):
    out = {}
    for label, a, b in SEGMENTS:
        calmars, mdds, cagrs, sharpes = [], [], [], []
        for phase in range(5):
            sig = momentum_signals_v2(data, pool, phase=phase, **kw)
            e = pd.Timestamp(b) if b else end
            eq, trades = bt.run(data, sig, start=a, end=e)
            m = calc_metrics(eq["total_value"], trades, initial=initial)
            calmars.append(m["卡玛比率"])
            mdds.append(m["最大回撤"])
            cagrs.append(m["年化收益率"])
            sharpes.append(m["夏普比率"])
        out[label] = dict(calmar_median=np.median(calmars), mdd_worst=min(mdds),
                          cagr_median=np.median(cagrs), sharpe_median=np.median(sharpes),
                          calmar_min=min(calmars), calmar_max=max(calmars))
    return out


# ---------- 唐奇安变体（KAMA 退出 / ATR 压缩突破） ----------

def donchian_kama_exit(data, symbol, entry=20):
    d = data[symbol].set_index("date")
    adj_c = d["close"] * d["factor"]
    adj_h = d["high"] * d["factor"]
    upper = adj_h.rolling(entry).max().shift(1)
    ka = kama(adj_c)
    holding, rows = False, []
    for i in range(entry + 1, len(d)):
        date = d.index[i]
        w = pd.Series(0.0, index=list(data.keys()))
        if not holding and adj_c.iloc[i] > upper.iloc[i]:
            holding = True
        elif holding and adj_c.iloc[i] < ka.iloc[i]:
            holding = False
        else:
            continue
        if holding:
            w[symbol] = 1.0
        rows.append(w.rename(date))
    return pd.DataFrame(rows)


def squeeze_breakout(data, symbol, entry=20, exit=10, atr_n=14,
                     q_window=120, q=0.2):
    """ATR 压缩突破：压缩态（ATR14/价 < 120 日 q 分位）当日收盘破 20 日高入场；10 日低退出。"""
    d = data[symbol].set_index("date")
    adj_c = d["close"] * d["factor"]
    adj_h = d["high"] * d["factor"]
    adj_l = d["low"] * d["factor"]
    tr = pd.concat([adj_h - adj_l, (adj_h - adj_c.shift(1)).abs(),
                    (adj_l - adj_c.shift(1)).abs()], axis=1).max(axis=1)
    atr = tr.rolling(atr_n).mean()
    atr_pct = atr / adj_c
    q_th = atr_pct.rolling(q_window, min_periods=q_window).quantile(q)
    in_squeeze = (atr_pct < q_th).fillna(False)
    upper = adj_h.rolling(entry).max().shift(1)
    lower = adj_l.rolling(exit).min().shift(1)
    holding, rows = False, []
    for i in range(max(entry, q_window) + 1, len(d)):
        date = d.index[i]
        w = pd.Series(0.0, index=list(data.keys()))
        if not holding and adj_c.iloc[i] > upper.iloc[i] and in_squeeze.iloc[i]:
            holding = True
        elif holding and adj_c.iloc[i] < lower.iloc[i]:
            holding = False
        else:
            continue
        if holding:
            w[symbol] = 1.0
        rows.append(w.rename(date))
    return pd.DataFrame(rows)


def main():
    cfg = yaml.safe_load(open(os.path.join(PROJECT, "config.yaml"), encoding="utf-8"))
    initial = cfg["backtest"]["initial_cash"]
    loader = DataLoader(cache_dir=os.path.join(PROJECT, "data_cache"))
    pool_codes = sorted(set(POOL_V0 + [DEFENSIVE, "510300", "510500"]))
    a_data = {c: loader.load(c, is_etf=True, start="2017-01-01") for c in pool_codes}
    end = max(df["date"].max() for df in a_data.values())
    print(f"A股数据截至 {end.date()}")
    us_data, _ = load_universe(symbols=["QQQ"], cache=os.path.join(
        PROJECT, "us_equities", "research", "cache"), offline=True)
    end_q = us_data["QQQ"]["date"].max()

    bt = Backtester(cfg)
    bt_us = EtfBacktester()
    rows = []

    def add(name, label, m):
        rows.append({"hypothesis": name, "segment": label, **m})

    # ---- 基准：生产 momentum（相位口径） ----
    print("基准 momentum...", flush=True)
    res = eval_phases(bt, a_data, POOL_V0, end, initial)
    for label, m in res.items():
        add("基准_momentum", label, m)

    # ---- H1a KAMA 趋势过滤 ----
    print("H1a KAMA...", flush=True)
    close_300 = a_data["510300"].set_index("date")["close"]
    ka = kama(close_300)
    res = eval_phases(bt, a_data, POOL_V0, end, initial,
                      trend={c: ka for c in POOL_V0})
    for label, m in res.items():
        add("H1a_KAMA趋势过滤", label, m)

    # ---- H3 相关性 regime（门槛 +0.15） ----
    print("H3 相关性regime...", flush=True)
    res = eval_phases(bt, a_data, POOL_V0, end, initial, corr_regime=True)
    for label, m in res.items():
        add("H3_相关性regime", label, m)

    # ---- H4 周线确认 ----
    print("H4 周线确认...", flush=True)
    res = eval_phases(bt, a_data, POOL_V0, end, initial, weekly_confirm=True)
    for label, m in res.items():
        add("H4_周线确认", label, m)

    # ---- H1b donchian_QQQ KAMA 退出 ----
    print("H1b donchian KAMA 退出...", flush=True)
    for label, a, b in [("train", "2019-01-02", "2022-12-30"),
                        ("valid", "2023-01-01", "2024-12-31"),
                        ("report", "2025-01-01", None),
                        ("full", "2019-01-02", None)]:
        e = pd.Timestamp(b) if b else end_q
        for vname, sig in (("基准_donchian_QQQ", US_S.donchian(us_data, "QQQ", 20, 10)),
                           ("H1b_KAMA退出", donchian_kama_exit(us_data, "QQQ"))):
            eq, trades = bt_us.run(us_data, sig, a, e)
            m = us_perf(eq, trades)
            add(vname, label, dict(calmar_median=m["calmar"], mdd_worst=m["mdd"],
                                   cagr_median=m["cagr"], sharpe_median=m["sharpe"],
                                   calmar_min=np.nan, calmar_max=np.nan))

    # ---- H2 ATR 压缩突破（510300/510500/QQQ） ----
    print("H2 squeeze...", flush=True)
    a_data_us = {c: a_data[c].assign(factor=1.0) for c in ("510300", "510500")}
    for sym in ("510300", "510500"):
        for label, a, b in SEGMENTS:
            e = pd.Timestamp(b) if b else end
            for vname, sig in (
                    (f"基准_donchian_{sym}", US_S.donchian(a_data_us, sym, 20, 10)),
                    (f"H2_squeeze_{sym}", squeeze_breakout(a_data_us, sym))):
                eq, trades = bt_us.run(a_data_us, sig, a, e)
                m = us_perf(eq, trades)
                add(vname, label, dict(calmar_median=m["calmar"], mdd_worst=m["mdd"],
                                       cagr_median=m["cagr"], sharpe_median=m["sharpe"],
                                       calmar_min=np.nan, calmar_max=np.nan))
    for label, a, b in [("train", "2019-01-02", "2022-12-30"),
                        ("valid", "2023-01-01", "2024-12-31"),
                        ("report", "2025-01-01", None),
                        ("full", "2019-01-02", None)]:
        e = pd.Timestamp(b) if b else end_q
        for vname, sig in (("基准_donchian_QQQ_H2", US_S.donchian(us_data, "QQQ", 20, 10)),
                           ("H2_squeeze_QQQ", squeeze_breakout(us_data, "QQQ"))):
            eq, trades = bt_us.run(us_data, sig, a, e)
            m = us_perf(eq, trades)
            add(vname, label, dict(calmar_median=m["calmar"], mdd_worst=m["mdd"],
                                   cagr_median=m["cagr"], sharpe_median=m["sharpe"],
                                   calmar_min=np.nan, calmar_max=np.nan))

    result = pd.DataFrame(rows)
    result.to_csv(os.path.join(RESULTS, "r8_hypotheses.csv"),
                  index=False, encoding="utf-8-sig")

    show = result.assign(
        cagr_median=result["cagr_median"].map(lambda x: f"{x:.1%}"),
        mdd_worst=result["mdd_worst"].map(lambda x: f"{x:.1%}"),
        sharpe_median=result["sharpe_median"].map(lambda x: f"{x:.2f}"),
        calmar_median=result["calmar_median"].map(lambda x: f"{x:.2f}"),
        calmar_min=result["calmar_min"].map(lambda x: f"{x:.2f}" if pd.notna(x) else "-"),
        calmar_max=result["calmar_max"].map(lambda x: f"{x:.2f}" if pd.notna(x) else "-"))
    print("\n===== R8 假设结果 =====")
    print(show.to_string(index=False))
    print(f"\n结果 CSV：{RESULTS}")


if __name__ == "__main__":
    main()
