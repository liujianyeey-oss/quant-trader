"""R1：A股账户跨品种动量轮动 + 量/融资过滤器测试。

框架与生产 momentum 完全一致：20 日动量最强、MA60 趋势过滤否则切 518880、
5 日调仓、T+1 开盘成交、ETF 万1 + 滑点 0.1%（复用生产 Backtester 与 config 成本）。
训练 2019-01-01~2023-12-31 选优，验证 2024-01-01~最新体检；V0/V1/V2 做 5 相位扰动。

预声明过滤器：
- 量：调仓日动量冠军当日成交量 < 其前 20 日均量 80%（缩量上涨）→ 跳过取次强；
  次强也缩量则进防守。
- 融资：全市场融资余额（SH+SZ）20 日增幅 > +5%（另跑 +10% 档）视为杠杆拥挤，
  该调仓周期进防守。宏观类数据用 T-1 观测。
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

PROJECT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
CACHE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cache")
os.makedirs(RESULTS, exist_ok=True)
os.makedirs(CACHE, exist_ok=True)

DEFENSIVE = "518880"
POOL_V0 = ["510300", "510500", "159915", "588000"]
POOL_V1 = POOL_V0 + ["513520", "513030"]
POOL_V2 = POOL_V1 + ["159920", "159980"]
ALL_CODES = sorted(set(POOL_V2 + [DEFENSIVE]))

SEGMENTS = [("train", "2019-01-01", "2023-12-31"),
            ("valid", "2024-01-01", None),
            ("full", "2019-01-01", None)]

MOM, TMA, REB = 20, 60, 5  # 与生产一致


# ---------- 数据 ----------

def load_all():
    loader = DataLoader(cache_dir=os.path.join(PROJECT, "data_cache"))
    return {c: loader.load(c, is_etf=True, start="2017-01-01") for c in ALL_CODES}


def load_margin():
    """全市场融资余额（SH+SZ 融资余额合计），带缓存。"""
    path = os.path.join(CACHE, "margin.csv")
    try:
        import akshare as ak
        sh = ak.macro_china_market_margin_sh()[["日期", "融资余额"]]
        sz = ak.macro_china_market_margin_sz()[["日期", "融资余额"]]
        sh.columns = sz.columns = ["date", "v"]
        sh["date"] = pd.to_datetime(sh["date"])
        sz["date"] = pd.to_datetime(sz["date"])
        m = sh.set_index("date").add(sz.set_index("date"), fill_value=0)
        m = m[m.index >= "2017-01-01"].reset_index().rename(columns={"v": "margin"})
        m.to_csv(path, index=False)
        return m, ""
    except Exception as e:  # noqa: BLE001
        if os.path.exists(path):
            return pd.read_csv(path, parse_dates=["date"]), \
                f"拉取失败降级缓存（{type(e).__name__}）"
        return None, f"融资融券数据不可测：{type(e).__name__}: {str(e)[:80]}"


# ---------- 信号生成（生产 momentum 框架 + 过滤器） ----------

def momentum_signals(data, pool, phase=0, volume_filter=False,
                     margin_chg=None, margin_th=None):
    panel = pd.DataFrame({c: df.set_index("date")["close"] for c, df in data.items()}).sort_index()
    vol_panel = pd.DataFrame({c: df.set_index("date")["volume"] for c, df in data.items()}).sort_index()
    mom = panel / panel.shift(MOM) - 1
    ma = panel.rolling(TMA).mean()
    avg_vol = vol_panel.rolling(MOM).mean().shift(1)  # 前 20 日均量（不含当日）
    has_def = DEFENSIVE in panel.columns

    rows = []
    for i in range(MOM + 1 + phase, len(panel), REB):
        date = panel.index[i]
        m = mom.iloc[i][[c for c in pool if c in panel.columns]].dropna()
        if m.empty:
            continue
        ordered = m.sort_values(ascending=False)
        if volume_filter:
            liquid = [c for c in ordered.index
                      if pd.notna(avg_vol.iloc[i][c]) and avg_vol.iloc[i][c] > 0
                      and vol_panel.iloc[i][c] >= 0.8 * avg_vol.iloc[i][c]]
            best = liquid[0] if liquid else None
        else:
            best = ordered.index[0]
        # 趋势过滤
        if best is not None and pd.notna(ma.iloc[i][best]) and panel.iloc[i][best] >= ma.iloc[i][best]:
            rows.append((date, best))
        elif best is not None and has_def:
            rows.append((date, DEFENSIVE))
        elif has_def:
            rows.append((date, DEFENSIVE))
        else:
            rows.append((date, None))

    signals = pd.DataFrame(0.0, index=[d for d, _ in rows], columns=list(panel.columns))
    for date, code in rows:
        if code is not None:
            signals.loc[date, code] = 1.0

    # 融资拥挤过滤：触发周期强制防守（在信号生成后统一覆盖）
    if margin_chg is not None and margin_th is not None and not signals.empty:
        ref = pd.DataFrame({"t1": signals.index - pd.Timedelta(days=1)})
        right = margin_chg.dropna().rename("chg").reset_index().sort_values("date")
        right["date"] = right["date"].astype("datetime64[ns]")
        ref["t1"] = ref["t1"].astype("datetime64[ns]")
        j = pd.merge_asof(ref.sort_values("t1"), right,
                          left_on="t1", right_on="date", direction="backward")
        crowded = j["chg"].to_numpy() > margin_th
        pool_cols = [c for c in pool if c in signals.columns]
        if has_def:
            signals.loc[crowded, pool_cols] = 0.0
            signals.loc[crowded, DEFENSIVE] = 1.0
    return signals


# ---------- 绩效 ----------

def run_segments(bt, data, signals, end, initial):
    out = {}
    for label, a, b in SEGMENTS:
        e = pd.Timestamp(b) if b else end
        eq, trades = bt.run(data, signals, start=a, end=e)
        m = calc_metrics(eq["total_value"], trades, initial=initial)
        amount = sum(abs(t["price"] * t["shares"]) for t in trades)
        years = max(((eq.index[-1] - eq.index[0]).days + 1) / 365.25, 1e-9)
        m["turnover"] = amount / eq["total_value"].mean() / years
        m["fees"] = 0.0  # Backtester trades 不拆费用字段，用换手近似
        out[label] = (eq, m)
    return out


def yearly(eq, initial):
    out, prev = {}, initial
    for y, g in eq.groupby(eq.index.year):
        out[int(y)] = g.iloc[-1] / prev - 1
        prev = g.iloc[-1]
    return out


def main():
    cfg = yaml.safe_load(open(os.path.join(PROJECT, "config.yaml"), encoding="utf-8"))
    initial = cfg["backtest"]["initial_cash"]
    print("加载数据...", flush=True)
    data = load_all()
    end = max(df["date"].max() for df in data.values())
    print(f"数据截至 {end.date()}")
    margin, margin_note = load_margin()
    print("融资余额数据:", "OK" if margin is not None else margin_note)
    if margin is not None:
        ms = margin.set_index("date")["margin"].sort_index()
        margin_chg = ms / ms.shift(20) - 1
        print(f"  覆盖 {margin['date'].min().date()} ~ {margin['date'].max().date()}")
    else:
        margin_chg = None

    bt = Backtester(cfg)
    rows, yearly_rows = [], []

    variants = [
        ("V0_基线池", POOL_V0, dict()),
        ("V1_扩跨境", POOL_V1, dict()),
        ("V2_对照池", POOL_V2, dict()),
        ("V1+量过滤", POOL_V1, dict(volume_filter=True)),
        ("V1+融资5%", POOL_V1, dict(margin_chg=margin_chg, margin_th=0.05)),
        ("V1+融资10%", POOL_V1, dict(margin_chg=margin_chg, margin_th=0.10)),
    ]

    signals_map = {}
    for name, pool, kw in variants:
        if ("margin" in name) and margin_chg is None:
            print(f"跳过 {name}（融资数据不可测）")
            continue
        signals_map[name] = momentum_signals(data, pool, **kw)
        res = run_segments(bt, data, signals_map[name], end, initial)
        row = {"variant": name}
        for label, (eq, m) in res.items():
            row[f"{label}_年化"] = m["年化收益率"]
            row[f"{label}_回撤"] = m["最大回撤"]
            row[f"{label}_夏普"] = m["夏普比率"]
            row[f"{label}_卡玛"] = m["卡玛比率"]
            row[f"{label}_换手"] = m["turnover"]
            row[f"{label}_交易"] = m["交易次数"]
        rows.append(row)
        for y, r in yearly(res["full"][0]["total_value"], initial).items():
            yearly_rows.append({"variant": name, "year": y, "return": r})
        print(f"完成 {name}", flush=True)

    result = pd.DataFrame(rows)
    result.to_csv(os.path.join(RESULTS, "variants.csv"), index=False, encoding="utf-8-sig")
    pd.DataFrame(yearly_rows).to_csv(os.path.join(RESULTS, "yearly.csv"),
                                     index=False, encoding="utf-8-sig")

    # ---- 相位扰动（V0/V1/V2） ----
    print("\n相位扰动...", flush=True)
    phase_rows = []
    for name, pool in (("V0_基线池", POOL_V0), ("V1_扩跨境", POOL_V1), ("V2_对照池", POOL_V2)):
        cagrs, mdds, calmars = [], [], []
        for phase in range(5):
            sig = momentum_signals(data, pool, phase=phase)
            eq, trades = bt.run(data, sig, start="2019-01-01", end=end)
            m = calc_metrics(eq["total_value"], trades, initial=initial)
            cagrs.append(m["年化收益率"])
            mdds.append(m["最大回撤"])
            calmars.append(m["卡玛比率"])
        phase_rows.append({"variant": name,
                           "cagr_min": min(cagrs), "cagr_median": np.median(cagrs),
                           "cagr_max": max(cagrs),
                           "极差pp": (max(cagrs) - min(cagrs)) * 100,
                           "mdd_worst": min(mdds), "mdd_best": max(mdds),
                           "calmar_median": np.median(calmars)})
    phase_df = pd.DataFrame(phase_rows)
    phase_df.to_csv(os.path.join(RESULTS, "phase.csv"), index=False, encoding="utf-8-sig")

    # ---- 控制台摘要 ----
    show = result.copy()
    for c in show.columns:
        if any(k in c for k in ("年化", "回撤")):
            show[c] = show[c].map(lambda x: f"{x:.1%}" if pd.notna(x) else "-")
        elif any(k in c for k in ("夏普", "卡玛")):
            show[c] = show[c].map(lambda x: f"{x:.2f}" if pd.notna(x) else "-")
        elif "换手" in c:
            show[c] = show[c].map(lambda x: f"{x:.0f}x" if pd.notna(x) else "-")
    print("\n===== 变体总表（训练 2019-2023 / 验证 2024+ / 全期） =====")
    cols = ["variant", "train_年化", "train_回撤", "train_卡玛", "train_夏普",
            "valid_年化", "valid_回撤", "valid_卡玛", "full_年化", "full_回撤",
            "full_卡玛", "full_夏普", "full_换手", "full_交易"]
    print(show[cols].to_string(index=False))
    print("\n===== 相位扰动（全期年化范围） =====")
    print(phase_df.assign(
        cagr_min=phase_df["cagr_min"].map(lambda x: f"{x:.1%}"),
        cagr_median=phase_df["cagr_median"].map(lambda x: f"{x:.1%}"),
        cagr_max=phase_df["cagr_max"].map(lambda x: f"{x:.1%}"),
        极差pp=phase_df["极差pp"].map(lambda x: f"{x:.1f}"),
        mdd_worst=phase_df["mdd_worst"].map(lambda x: f"{x:.1%}"),
        mdd_best=phase_df["mdd_best"].map(lambda x: f"{x:.1%}"),
        calmar_median=phase_df["calmar_median"].map(lambda x: f"{x:.2f}"),
    ).to_string(index=False))
    print(f"\n结果 CSV：{RESULTS}")


if __name__ == "__main__":
    main()
