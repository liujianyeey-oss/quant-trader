"""R6 仓位治理层研究：连续仓位乘数作用于生产 momentum 信号。

立项逻辑：二元过滤器反复证伪，R6 不动买卖信号，只研究 [0.5, 1.5] 连续乘数。
预声明规则：
- G1 股债性价比 ERP：沪深300 PE 日频 ≥5 年历史免费源不可得 → 不可测（不伪造）
- G3 隐含波动：乘数 = clip(iVIX 过去 5 年均值 / 当前 iVIX 20 日均值, 0.5, 1.5)
  （50ETF qvix，akshare index_option_50etf_qvix，2015-02 起）
- G4 股指期货基差：IF 主力年化贴水率 >8%（恐慌）×1.25；升水 >2%（过热）×0.75；
  其间 ×1.0（分段但预声明；IF0 新浪主力连续，现货用沪深300 指数日线）
- G13 = G1×G3：G1 不可测 → 组合不成立
作用细节（预声明）：调仓日股票敞口权重 × 当日乘数（T-1 数据），防守资产(518880)不乘；
乘数可 >1（允许借钱，利息按 0 计并声明）。
判定：训练 2019-2023 相位中位卡玛改善 ≥0.1 且最大回撤不劣化；验证 2024+ 体检。
"""
import os
import sys

import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data import DataLoader  # noqa: E402
from strategies import MomentumRotation  # noqa: E402
from metrics import calc_metrics  # noqa: E402

PROJECT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
CACHE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cache")
os.makedirs(RESULTS, exist_ok=True)
os.makedirs(CACHE, exist_ok=True)

STOCK_POOL = ["510300", "510500", "159915", "588000"]
DEFENSIVE = "518880"
COST = {"commission": 0.0001, "slippage": 0.001, "lot": 100}
SEGMENTS = [("train", "2019-01-01", "2023-12-31"), ("valid", "2024-01-01", None),
            ("full", "2019-01-01", None)]


# ---------- 数据 ----------

def load_ivix():
    path = os.path.join(CACHE, "ivix.csv")
    try:
        import akshare as ak
        df = ak.index_option_50etf_qvix()
        df["date"] = pd.to_datetime(df["date"])
        df = df[["date", "close"]].rename(columns={"close": "ivix"})
        df.to_csv(path, index=False)
        return df, ""
    except Exception as e:  # noqa: BLE001
        if os.path.exists(path):
            return pd.read_csv(path, parse_dates=["date"]), "降级缓存"
        return None, f"iVIX 不可测：{type(e).__name__}: {str(e)[:80]}"


def load_if_basis():
    """IF 主力连续(IF0) + 沪深300 现货 → 年化贴水率（贴水为正）。
    年化 = (现货-期货)/现货 ÷ 到期剩余天数 × 365；到期按当月第三个周五
    （过了则下月第三个周五）近似。"""
    path = os.path.join(CACHE, "if_basis.csv")
    try:
        import akshare as ak
        fut = ak.futures_main_sina(symbol="IF0", start_date="20190101")
        fut = fut[["日期", "收盘价"]].rename(columns={"日期": "date", "收盘价": "fut"})
        fut["date"] = pd.to_datetime(fut["date"])
        spot = ak.stock_zh_index_daily(symbol="sh000300")
        spot = spot[["date", "close"]].rename(columns={"close": "spot"})
        spot["date"] = pd.to_datetime(spot["date"])
        df = pd.merge_asof(fut.sort_values("date"), spot.sort_values("date"),
                           on="date", direction="backward").dropna()

        def third_friday(y, m):
            d = pd.Timestamp(y, m, 1)
            return d + pd.Timedelta(days=(4 - d.weekday()) % 7 + 14)

        def days_to_expiry(date):
            tf = third_friday(date.year, date.month)
            if date > tf:
                nm = date + pd.offsets.MonthBegin(1)
                tf = third_friday(nm.year, nm.month)
            return max((tf - date).days, 1)

        df["dte"] = df["date"].map(days_to_expiry)
        df["basis_ann"] = (df["spot"] - df["fut"]) / df["spot"] / df["dte"] * 365
        out = df[["date", "basis_ann"]].dropna()
        out.to_csv(path, index=False)
        return out, ""
    except Exception as e:  # noqa: BLE001
        if os.path.exists(path):
            return pd.read_csv(path, parse_dates=["date"]), "降级缓存"
        return None, f"IF 基差不可测：{type(e).__name__}: {str(e)[:80]}"


def _t1_align(series, dates):
    """宏观/参考数据 T-1 规则：date t 用 ≤ t-1 最近观测。"""
    ref = pd.DataFrame({"t1": pd.DatetimeIndex(dates).astype("datetime64[ns]")
                        - pd.Timedelta(days=1)}).sort_values("t1")
    right = series.rename("v").reset_index().sort_values("date")
    right["date"] = right["date"].astype("datetime64[ns]")
    j = pd.merge_asof(ref, right, left_on="t1", right_on="date", direction="backward")
    return pd.Series(j["v"].to_numpy(), index=dates)


def g3_multiplier(ivix, dates):
    v = _t1_align(ivix.set_index("date")["ivix"].sort_index(), dates)
    mean5y = v.rolling(1250, min_periods=250).mean()   # 5 年均值（数据不足 5 年用可得期）
    cur20 = v.rolling(20).mean()
    return (mean5y / cur20).clip(0.5, 1.5)


def g4_multiplier(basis, dates):
    b = _t1_align(basis.set_index("date")["basis_ann"].sort_index(), dates)
    mult = pd.Series(1.0, index=dates)
    mult[b > 0.08] = 1.25      # 年化贴水 >8%：恐慌区加仓
    mult[b < -0.02] = 0.75     # 升水 >2%（basis 为负）：过热区减仓
    return mult


# ---------- 治理引擎（允许权重>1，借款利息 0，与生产同成本口径） ----------

def run_governor(data, base_signals: pd.DataFrame, multiplier: pd.Series | None,
                 start, end, initial=1_000_000.0):
    """生产 momentum 信号 × 连续乘数（防守资产不乘），T+1 开盘成交。
    返回 (equity Series, trades)。"""
    panel = pd.DataFrame({c: df.set_index("date")["close"] for c, df in data.items()}).sort_index()
    opens = pd.DataFrame({c: df.set_index("date")["open"] for c, df in data.items()}).sort_index()
    calendar = panel.index
    marks = panel.ffill()
    s = base_signals.reindex(columns=panel.columns, fill_value=0.0)
    aligned = (s.reindex(calendar.union(s.index)).sort_index().ffill()
               .reindex(calendar).fillna(0.0))
    targets = aligned.shift(1).fillna(0.0)
    if multiplier is not None:
        mult = multiplier.reindex(calendar).ffill().fillna(1.0)
        stock_cols = [c for c in targets.columns if c != DEFENSIVE]
        targets[stock_cols] = targets[stock_cols].mul(mult, axis=0).clip(0.0, 1.5)
    start, end = pd.Timestamp(start), pd.Timestamp(end)
    comm, slip, lot = COST["commission"], COST["slippage"], COST["lot"]

    cash, shares = initial, {c: 0 for c in panel.columns}
    rows, trades = [], []
    pending = False
    done = pd.Series(0.0, index=panel.columns)
    for date in calendar:
        if date < start or date > end:
            continue
        w = targets.loc[date]
        if not np.allclose(w.to_numpy(), done.to_numpy(), atol=1e-10) or pending:
            o = opens.loc[date]
            needed = [c for c in shares if shares[c] > 0 or w.get(c, 0) > 0]
            if all(pd.notna(o[c]) and o[c] > 0 for c in needed):
                nav_open = cash + sum(n * o[c] for c, n in shares.items() if n > 0)
                desired = {c: (int(w.get(c, 0) * nav_open / (o[c] * (1 + slip)) / lot) * lot
                               if w.get(c, 0) > 0 else 0) for c in shares}
                for c in shares:  # 先卖
                    delta = desired[c] - shares[c]
                    if delta < 0:
                        price = o[c] * (1 - slip)
                        amount = -delta * price
                        cash += amount * (1 - comm)
                        trades.append(dict(date=date, code=c, side="sell", shares=-delta,
                                           price=price, amount=amount,
                                           pnl=0.0, win=None))
                        shares[c] = desired[c]
                for c in shares:  # 后买（允许现金为负=借款，利息 0）
                    delta = desired[c] - shares[c]
                    if delta > 0:
                        price = o[c] * (1 + slip)
                        amount = delta * price
                        cash -= amount * (1 + comm)
                        trades.append(dict(date=date, code=c, side="buy", shares=delta,
                                           price=price, amount=amount,
                                           pnl=0.0, win=None))
                        shares[c] = desired[c]
                done = w.copy()
                pending = False
            else:
                pending = True
        nav = cash + sum(n * marks.loc[date, c] for c, n in shares.items()
                         if n > 0 and pd.notna(marks.loc[date, c]))
        rows.append((date, nav))
    return pd.Series(dict(rows)).sort_index(), trades


def yearly_sharpe_calmar(eq, initial=1_000_000):
    ret = eq.pct_change().dropna()
    years = max(((eq.index[-1] - eq.index[0]).days + 1) / 365.25, 1e-9)
    cagr = (eq.iloc[-1] / initial) ** (1 / years) - 1
    mdd = (eq / eq.cummax() - 1).min()
    sharpe = ret.mean() / ret.std() * np.sqrt(252) if ret.std() > 0 else np.nan
    return dict(cagr=cagr, mdd=mdd, sharpe=sharpe,
                calmar=cagr / abs(mdd) if mdd < 0 else np.nan)


def main():
    cfg = yaml.safe_load(open(os.path.join(PROJECT, "config.yaml"), encoding="utf-8"))
    initial = cfg["backtest"]["initial_cash"]
    loader = DataLoader(cache_dir=os.path.join(PROJECT, "data_cache"))
    data = {c: loader.load(c, is_etf=True, start="2017-01-01")
            for c in STOCK_POOL + [DEFENSIVE]}
    end = max(df["date"].max() for df in data.values())
    print(f"行情截至 {end.date()}")

    ivix, note3 = load_ivix()
    basis, note4 = load_if_basis()
    print("iVIX:", note3 or f"{ivix['date'].min().date()}~{ivix['date'].max().date()}",
          "| IF 基差:", note4 or f"{basis['date'].min().date()}~{basis['date'].max().date()}")
    print("G1（沪深300 PE 日频≥5年）：免费源不可得 → 不可测；G13 随之不成立")

    calendar = pd.DatetimeIndex(sorted(set().union(
        *[set(df["date"]) for df in data.values()])))
    mults = {"基准(乘数=1)": None}
    if ivix is not None:
        mults["G3_iVIX"] = g3_multiplier(ivix, calendar)
    if basis is not None:
        mults["G4_IF基差"] = g4_multiplier(basis, calendar)

    rows = []
    for name, mult in mults.items():
        for label, a, b in SEGMENTS:
            seg_calmars, seg_mdds, seg_cagrs, seg_sharpes = [], [], [], []
            for phase in range(5):
                sig = MomentumRotation(cfg).generate_signals(data) \
                    if phase == 0 else _phase_signals(data, phase)
                e = pd.Timestamp(b) if b else end
                eq, trades = run_governor(data, sig, mult, a, e, initial)
                m = yearly_sharpe_calmar(eq, initial)
                seg_calmars.append(m["calmar"])
                seg_mdds.append(m["mdd"])
                seg_cagrs.append(m["cagr"])
                seg_sharpes.append(m["sharpe"])
            rows.append({"variant": name, "segment": label,
                         "cagr_median": np.median(seg_cagrs),
                         "mdd_worst": min(seg_mdds),
                         "sharpe_median": np.median(seg_sharpes),
                         "calmar_median": np.median(seg_calmars),
                         "calmar_min": min(seg_calmars), "calmar_max": max(seg_calmars)})
        print(f"完成 {name}", flush=True)

    result = pd.DataFrame(rows)
    result.to_csv(os.path.join(RESULTS, "governor.csv"), index=False, encoding="utf-8-sig")

    # 乘数行为统计
    stat_rows = []
    for name in ("G3_iVIX", "G4_IF基差"):
        if name in mults:
            m = mults[name]
            stat_rows.append({"variant": name, "mean": m.mean(), "p10": m.quantile(0.1),
                              "p90": m.quantile(0.9), "share_gt1": (m > 1.001).mean(),
                              "share_lt1": (m < 0.999).mean()})
    pd.DataFrame(stat_rows).to_csv(os.path.join(RESULTS, "multiplier_stats.csv"),
                                   index=False, encoding="utf-8-sig")

    print("\n===== 治理层效果（5 相位中位数口径） =====")
    show = result.assign(
        cagr_median=result["cagr_median"].map(lambda x: f"{x:.1%}"),
        mdd_worst=result["mdd_worst"].map(lambda x: f"{x:.1%}"),
        sharpe_median=result["sharpe_median"].map(lambda x: f"{x:.2f}"),
        calmar_median=result["calmar_median"].map(lambda x: f"{x:.2f}"),
        calmar_min=result["calmar_min"].map(lambda x: f"{x:.2f}"),
        calmar_max=result["calmar_max"].map(lambda x: f"{x:.2f}"))
    print(show.to_string(index=False))
    print("\n乘数行为：", stat_rows)
    print(f"\n结果 CSV：{RESULTS}")


def _phase_signals(data, phase):
    """生产 momentum 信号的相位变体（复用 research_r1 的参数化生成器）。"""
    from research_r1.runner import momentum_signals
    return momentum_signals(data, STOCK_POOL, phase=phase)


if __name__ == "__main__":
    main()
