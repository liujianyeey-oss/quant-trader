"""R9 常态化流水线第一批：① T+1 税方向测量 ② 日历效应 ③ 低波横截面 ④ VIX 只减不加。

预声明模板遵循 RESEARCH_PLAN 第六节；训练 2019-2023 判效（卡玛改善≥0.1），
验证 2024+ 体检，2025+ 只报告；④ 判效基准 donchian_QQQ（0.73/2.32）。
"""
import os
import sys

import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data import DataLoader  # noqa: E402
from backtest import Backtester  # noqa: E402
from metrics import calc_metrics  # noqa: E402
from research_r1.runner import momentum_signals, POOL_V0, DEFENSIVE  # noqa: E402
from macro_research.strategies import run_single_asset, perf  # noqa: E402
from us_equities.research.etf_data import load_universe  # noqa: E402
from us_equities.research import etf_strategies as US_S  # noqa: E402
from us_equities.research.etf_engine import EtfBacktester, perf as us_perf  # noqa: E402

PROJECT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
CACHE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cache")
os.makedirs(RESULTS, exist_ok=True)
os.makedirs(CACHE, exist_ok=True)

SEG = [("train", "2019-01-01", "2023-12-31"), ("valid", "2024-01-01", None),
       ("full", "2019-01-01", None)]
ETF_COST = {"commission": 0.0001, "slippage": 0.001, "lot": 100}


# ================= ① T+1 税方向测量 =================

def t1_tax_measurement(data):
    """生产 momentum（5 相位全样本）：冠军切换日的买入侧隔夜漂移
    = 次日开盘价 / 信号日收盘 - 1（正 = 多付钱，即正的 T+1 税）。
    卖出侧漂移符号相反（正漂移对卖出有利）。分标的/分年份统计。"""
    panel = pd.DataFrame({c: df.set_index("date")["close"] for c, df in data.items()}).sort_index()
    opens = pd.DataFrame({c: df.set_index("date")["open"] for c, df in data.items()}).sort_index()
    rows = []
    for phase in range(5):
        sig = momentum_signals(data, POOL_V0, phase=phase)
        prev = None
        for date, w in sig.iterrows():
            cur = w[w > 0].index[0] if (w > 0).any() else None
            if cur is not None and cur != prev:
                i = panel.index.get_loc(date)
                if i + 1 < len(panel):
                    nd = panel.index[i + 1]
                    o, c = opens.loc[nd, cur], panel.loc[date, cur]
                    if pd.notna(o) and pd.notna(c) and c > 0:
                        rows.append({"symbol": cur, "year": date.year,
                                     "phase": phase, "drift_bps": (o / c - 1) * 1e4,
                                     "is_defensive": cur == DEFENSIVE})
                prev = cur
    df = pd.DataFrame(rows)
    out = []
    for key, g in df.groupby(["symbol"]):
        out.append({"bucket": key, "n": len(g), "mean_bps": g["drift_bps"].mean(),
                    "median_bps": g["drift_bps"].median(),
                    "p25_bps": g["drift_bps"].quantile(0.25),
                    "p75_bps": g["drift_bps"].quantile(0.75)})
    out.append({"bucket": "ALL_进攻池", "n": len(df[~df["is_defensive"]]),
                "mean_bps": df.loc[~df["is_defensive"], "drift_bps"].mean(),
                "median_bps": df.loc[~df["is_defensive"], "drift_bps"].median(),
                "p25_bps": df.loc[~df["is_defensive"], "drift_bps"].quantile(0.25),
                "p75_bps": df.loc[~df["is_defensive"], "drift_bps"].quantile(0.75)})
    by_year = df[~df["is_defensive"]].groupby("year")["drift_bps"] \
        .agg(["count", "mean", "median"]).reset_index()
    return pd.DataFrame(out), by_year


# ================= ② 日历效应 =================

def c1_signal(calendar):
    """每月最后 2 个交易日至次月前 2 个交易日持有。"""
    hold = pd.Series(0.0, index=calendar)
    months = pd.Series(calendar).groupby([calendar.year, calendar.month])
    for _, g in months:
        days = list(g)
        window = set(days[-2:])  # 本月末 2 天
        hold.loc[list(window)] = 1.0
        # 次月前 2 天（下组的前两天，在下一次迭代置 1）
    # 补：次月前 2 天
    for i, d in enumerate(calendar):
        m = pd.Timestamp(d).to_period("M")
        first_days = [x for x in calendar if pd.Timestamp(x).to_period("M") == m][:2]
        if d in first_days:
            hold.loc[d] = 1.0
    return hold[hold != hold.shift(1)].dropna()


def c2_signal(calendar):
    """11月~4月持有，5~10月空仓。"""
    hold = pd.Series([1.0 if pd.Timestamp(d).month in (11, 12, 1, 2, 3, 4) else 0.0
                      for d in calendar], index=calendar)
    return hold[hold != hold.shift(1)].dropna()


def eval_cal(df, w, end, engine="etf"):
    out = {}
    for label, a, b in SEG:
        e = pd.Timestamp(b) if b else end
        eq, trades = run_single_asset(df, w, a, e, ETF_COST, 1_000_000)
        out[label] = perf(eq, trades, 1_000_000)
    return out


# ================= ③ 低波横截面 =================

def monthly_top_signals(closes, kind="lowvol", top=5, vol_w=60, mom_w=60):
    """每月末：60 日波动升序（lowvol）或 60 日动量降序（mom）取 top 等权。"""
    vol = closes.pct_change().rolling(vol_w).std() * np.sqrt(252)
    mom = closes / closes.shift(mom_w) - 1
    dates = closes.index
    month_ends = dates.to_series().groupby([dates.year, dates.month]).last()
    rows = []
    for d in month_ends:
        i = dates.get_loc(d)
        metric = vol.iloc[i] if kind == "lowvol" else mom.iloc[i]
        m = metric.dropna()
        if len(m) < top:
            continue
        winners = m.nsmallest(top).index if kind == "lowvol" else m.nlargest(top).index
        w = pd.Series(0.0, index=closes.columns)
        w[winners] = 1.0 / top
        rows.append(w.rename(d))
    return pd.DataFrame(rows)


# ================= ④ VIX 只减不加 =================

def load_vix():
    path = os.path.join(CACHE, "vixcls.csv")
    try:
        import requests
        from io import StringIO
        r = requests.get("https://fred.stlouisfed.org/graph/fredgraph.csv",
                         params={"id": "VIXCLS", "cosd": "2014-01-01"}, timeout=30)
        r.raise_for_status()
        df = pd.read_csv(StringIO(r.text))
        df.columns = ["date", "vix"]
        df["date"] = pd.to_datetime(df["date"])
        df["vix"] = pd.to_numeric(df["vix"], errors="coerce")
        df = df.dropna()
        df.to_csv(path, index=False)
        return df, ""
    except Exception as e:  # noqa: BLE001
        if os.path.exists(path):
            return pd.read_csv(path, parse_dates=["date"]), "降级缓存"
        return None, f"VIX 不可测：{type(e).__name__}"


def vix_mult(vix, dates):
    """min(1, VIX过去5年均值 / 当前VIX20日均值)，T-1。"""
    ref = pd.DataFrame({"t1": pd.DatetimeIndex(dates).astype("datetime64[ns]")
                        - pd.Timedelta(days=1)}).sort_values("t1")
    right = vix.set_index("date")["vix"].rename("v").reset_index().sort_values("date")
    right["date"] = right["date"].astype("datetime64[ns]")
    j = pd.merge_asof(ref, right, left_on="t1", right_on="date", direction="backward")
    v = pd.Series(j["v"].to_numpy(), index=dates)
    mean5y = v.rolling(1250, min_periods=250).mean()
    cur20 = v.rolling(20).mean()
    return (mean5y / cur20).clip(upper=1.0)


def main():
    cfg = yaml.safe_load(open(os.path.join(PROJECT, "config.yaml"), encoding="utf-8"))
    initial = cfg["backtest"]["initial_cash"]
    loader = DataLoader(cache_dir=os.path.join(PROJECT, "data_cache"))
    codes = sorted(set(POOL_V0 + [DEFENSIVE, "510500"]))
    a_data = {c: loader.load(c, is_etf=True, start="2017-01-01") for c in codes}
    end = max(df["date"].max() for df in a_data.values())
    print(f"A股数据截至 {end.date()}")
    us_data, _ = load_universe(offline=True, cache=os.path.join(
        PROJECT, "us_equities", "research", "cache"))
    us_closes = US_S.close_panel(us_data)
    end_q = us_closes.index.max()

    # ================= ① =================
    print("\n== ① T+1 税方向测量 ==", flush=True)
    t1, t1_year = t1_tax_measurement(a_data)
    t1.to_csv(os.path.join(RESULTS, "t1_tax_direction.csv"),
              index=False, encoding="utf-8-sig")
    t1_year.to_csv(os.path.join(RESULTS, "t1_tax_by_year.csv"),
                   index=False, encoding="utf-8-sig")
    print(t1.assign(**{c: t1[c].map(lambda x: f"{x:+.1f}") for c in
                       ("mean_bps", "median_bps", "p25_bps", "p75_bps")}).to_string(index=False))
    print(t1_year.assign(mean=t1_year["mean"].map(lambda x: f"{x:+.1f}"),
                         median=t1_year["median"].map(lambda x: f"{x:+.1f}")).to_string(index=False))

    # ================= ② =================
    print("\n== ② 日历效应 ==", flush=True)
    rows2 = []
    for sym, df in (("510300", a_data["510300"]), ("QQQ", None)):
        if sym == "QQQ":
            df = us_data["QQQ"].copy()
            df["factor"] = 1.0
        calendar = pd.DatetimeIndex(df["date"])
        for cname, sig in (("C1_月末效应", c1_signal(calendar)),
                           ("C2_sellinmay", c2_signal(calendar)),
                           ("买入持有", pd.Series(1.0, index=[calendar[0]]))):
            res = eval_cal(df, sig, df["date"].max())
            for label, m in res.items():
                rows2.append({"symbol": sym, "variant": cname, "segment": label, **m})
    r2df = pd.DataFrame(rows2)
    r2df.to_csv(os.path.join(RESULTS, "calendar.csv"), index=False, encoding="utf-8-sig")
    print(r2df.pivot_table(index=["symbol", "variant"], columns="segment",
                           values="calmar").map(lambda x: f"{x:.2f}").to_string())

    # ================= ③ =================
    print("\n== ③ 低波横截面 ==", flush=True)
    bt_us = EtfBacktester()
    rows3 = []
    us_core = [c for c in us_closes.columns]
    for vname, sig in (
            ("低波top5月调", monthly_top_signals(us_closes, "lowvol", 5)),
            ("动量top5月调", monthly_top_signals(us_closes, "mom", 5))):
        for label, a, b in [("train", "2019-01-02", "2022-12-30"),
                            ("valid", "2023-01-01", "2024-12-31"),
                            ("report", "2025-01-01", None), ("full", "2019-01-02", None)]:
            e = pd.Timestamp(b) if b else end_q
            eq, trades = bt_us.run(us_data, sig, a, e)
            m = us_perf(eq, trades)
            rows3.append({"universe": "US核心25", "variant": vname, "segment": label, **m})
    # SPY 持有基准
    w_spy = pd.DataFrame(1.0, index=[us_closes.index[0]], columns=us_closes.columns)
    w_spy[:] = 0.0
    w_spy["SPY"] = 1.0
    for label, a, b in [("train", "2019-01-02", "2022-12-30"),
                        ("valid", "2023-01-01", "2024-12-31"), ("full", "2019-01-02", None)]:
        e = pd.Timestamp(b) if b else end_q
        eq, trades = bt_us.run(us_data, w_spy, a, e)
        m = us_perf(eq, trades)
        rows3.append({"universe": "US核心25", "variant": "SPY持有", "segment": label, **m})
    # A股弱检验（6 只池，top2，功效不足标注）
    a6_codes = ["510300", "510500", "159915", "588000", "510880", "511010"]
    a6 = {c: loader.load(c, is_etf=True, start="2017-01-01") for c in a6_codes}
    a6_closes = pd.DataFrame({c: df.set_index("date")["close"] for c, df in a6.items()}).sort_index()
    sig_a6 = monthly_top_signals(a6_closes, "lowvol", 2)
    a6_us = {c: df.assign(factor=1.0) for c, df in a6.items()}
    bt_a = EtfBacktester()
    sig_hold300 = pd.DataFrame(0.0, index=[a6_closes.index[0]], columns=a6_closes.columns)
    sig_hold300["510300"] = 1.0
    for vname, sig in (("A股低波top2月调(功效不足)", sig_a6),
                       ("A股510300持有", sig_hold300)):
        for label, a, b in SEG:
            e = pd.Timestamp(b) if b else end
            eq, trades = bt_a.run(a6_us, sig, a, e)
            m = us_perf(eq, trades)
            rows3.append({"universe": "A股6只", "variant": vname, "segment": label, **m})
    r3df = pd.DataFrame(rows3)
    r3df.to_csv(os.path.join(RESULTS, "lowvol.csv"), index=False, encoding="utf-8-sig")
    print(r3df.pivot_table(index=["universe", "variant"], columns="segment",
                           values="calmar").map(lambda x: f"{x:.2f}"
                                                if pd.notna(x) else "-").to_string())

    # ================= ④ =================
    print("\n== ④ VIX 只减不加 ==", flush=True)
    vix, note = load_vix()
    rows4 = []
    if vix is None:
        print(note)
    else:
        don_q = US_S.donchian(us_data, "QQQ", 20, 10)
        mult = vix_mult(vix, us_closes.index)
        sig_vix = don_q.copy()
        sig_vix["QQQ"] = sig_vix["QQQ"] * mult.reindex(sig_vix.index).ffill().fillna(1.0)
        for label, a, b in [("train", "2019-01-02", "2022-12-30"),
                            ("valid", "2023-01-01", "2024-12-31"),
                            ("report", "2025-01-01", None), ("full", "2019-01-02", None)]:
            e = pd.Timestamp(b) if b else end_q
            for vname, sig in (("基准_donchian_QQQ", don_q), ("VIX只减不加", sig_vix)):
                eq, trades = bt_us.run(us_data, sig, a, e)
                m = us_perf(eq, trades)
                rows4.append({"variant": vname, "segment": label, **m})
    r4df = pd.DataFrame(rows4)
    r4df.to_csv(os.path.join(RESULTS, "vix_reduce_only.csv"),
                index=False, encoding="utf-8-sig")
    if len(r4df):
        print(r4df.pivot_table(index="variant", columns="segment", values="calmar")
              .map(lambda x: f"{x:.2f}").to_string())

    print(f"\n结果 CSV：{RESULTS}")


if __name__ == "__main__":
    main()
