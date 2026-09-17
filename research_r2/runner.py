"""R2：A股宽基均值回归系统测试。

预声明变体（标的 510300/510500/159915，单标的全进全出）：
- MR1 RSI(2) Connors：收盘>MA200 门控；RSI(2)<10（变体 <5）次日开盘买；收盘>MA5 次日开盘卖
- MR2 布林 z-score：收盘>MA200；z=(收盘-MA20)/20σ < -2 次日开盘买；z>0 次日开盘卖
- MR3 地量确认：MR1 信号 + 当日成交量为 60 日最低
执行：T+1 开盘成交（复用 macro_research 单资产引擎，ETF 万1 + 滑点 0.1%）。
T+1 税对照：同一信号按信号日收盘成交（理想化，仅作量化展示，不作为策略）。
训练 2019-01-01~2023-12-31 判定（卡玛>0.5 且优于持有基准卡玛），验证 2024+ 体检。
"""
import os
import sys

import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data import DataLoader  # noqa: E402
from macro_research.strategies import run_single_asset, perf  # noqa: E402
from us_equities.research.etf_strategies import _rsi  # noqa: E402
from backtest import Backtester  # noqa: E402
from strategies import MomentumRotation  # noqa: E402
from metrics import calc_metrics  # noqa: E402

PROJECT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
os.makedirs(RESULTS, exist_ok=True)

SYMBOLS = ["510300", "510500", "159915"]
COST = {"commission": 0.0001, "slippage": 0.001, "lot": 100}
INITIAL = 1_000_000.0
SEGMENTS = [("train", "2019-01-01", "2023-12-31"),
            ("valid", "2024-01-01", None), ("full", "2019-01-01", None)]


# ---------- 信号生成（weight Series，状态变化日） ----------

def _state_machine(dates, entries, exits):
    holding, rows = False, []
    for i in range(len(dates)):
        if not holding and entries.iloc[i]:
            holding = True
        elif holding and exits.iloc[i]:
            holding = False
        else:
            continue
        rows.append((dates[i], 1.0 if holding else 0.0))
    return pd.Series(dict(rows)).sort_index()


def sig_mr1(df, threshold=10):
    d = df.set_index("date")
    c = d["close"]
    rsi = _rsi(c, 2)
    ma200 = c.rolling(200).mean()
    ma5 = c.rolling(5).mean()
    entries = (c > ma200) & (rsi < threshold)
    exits = c > ma5
    return _state_machine(d.index, entries.fillna(False), exits.fillna(False))


def sig_mr2(df, z_in=-2.0):
    d = df.set_index("date")
    c = d["close"]
    ma200 = c.rolling(200).mean()
    ma, sd = c.rolling(20).mean(), c.rolling(20).std()
    z = (c - ma) / sd
    entries = (c > ma200) & (z < z_in)
    exits = z > 0
    return _state_machine(d.index, entries.fillna(False), exits.fillna(False))


def sig_mr3(df, threshold=10, vol_window=60):
    d = df.set_index("date")
    c = d["close"]
    rsi = _rsi(c, 2)
    ma200 = c.rolling(200).mean()
    ma5 = c.rolling(5).mean()
    vol_min = d["volume"].rolling(vol_window, min_periods=vol_window).min()
    entries = (c > ma200) & (rsi < threshold) & (d["volume"] <= vol_min)
    exits = c > ma5
    return _state_machine(d.index, entries.fillna(False), exits.fillna(False))


# ---------- 理想化收盘成交引擎（仅 T+1 税对照，不作为策略） ----------

def run_close_fill(df, target_w, start, end, cost, initial=INITIAL):
    """同一信号按信号日收盘成交（无 T+1 时滞），成本率相同。"""
    d = df.set_index("date").sort_index()
    calendar = d.index
    w = target_w.reindex(calendar.union(target_w.index)).sort_index().ffill() \
        .reindex(calendar).fillna(0.0)
    start, end = pd.Timestamp(start), pd.Timestamp(end)
    comm, slip, lot = cost["commission"], cost["slippage"], cost.get("lot", 1)
    cash, shares, rows = initial, 0.0, []
    done_w = 0.0
    for date in calendar:
        if date < start or date > end:
            continue
        tw = float(w.loc[date])
        c = d["close"].loc[date]
        if abs(tw - done_w) > 1e-12 and pd.notna(c) and c > 0:
            nav = cash + shares * c
            desired = int(tw * nav / (c * (1 + slip)) / lot) * lot
            if desired < shares:
                n = shares - desired
                price = c * (1 - slip)
                cash += n * price * (1 - comm)
                shares = desired
            elif desired > shares:
                n = desired - shares
                price = c * (1 + slip)
                while n > 0 and n * price * (1 + comm) > cash + 1e-8:
                    n -= lot
                amount = n * price
                cash -= amount * (1 + comm)
                shares += n
            done_w = tw
        rows.append((date, cash + shares * d["close"].loc[date]))
    return pd.Series(dict(rows)).sort_index()


def flat_share(w, start, end):
    seg = w.reindex(pd.DatetimeIndex(sorted(w.index.union([pd.Timestamp(start),
                                                           pd.Timestamp(end)])))).ffill()
    seg = seg[(seg.index >= pd.Timestamp(start)) & (seg.index <= pd.Timestamp(end))]
    return float((seg == 0).mean())


def main():
    cfg = yaml.safe_load(open(os.path.join(PROJECT, "config.yaml"), encoding="utf-8"))
    loader = DataLoader(cache_dir=os.path.join(PROJECT, "data_cache"))
    data = {s: loader.load(s, is_etf=True, start="2017-01-01") for s in SYMBOLS}
    end = max(df["date"].max() for df in data.values())
    print(f"数据截至 {end.date()}")

    variants = {
        "MR1_rsi2<10": lambda df: sig_mr1(df, 10),
        "MR1_rsi2<5": lambda df: sig_mr1(df, 5),
        "MR2_boll<-2": lambda df: sig_mr2(df, -2.0),
        "MR3_rsi2<10+地量60": lambda df: sig_mr3(df, 10, 60),
    }

    rows, equities = [], {}
    for sym in SYMBOLS:
        df = data[sym]
        for vname, gen in variants.items():
            w = gen(df)
            eqs = {}
            for label, a, b in SEGMENTS:
                e = pd.Timestamp(b) if b else end
                eq, trades = run_single_asset(df, w, a, e, COST, INITIAL)
                m = perf(eq, trades, INITIAL)
                n_trades = len([t for t in trades if t["side"] == "sell"])
                eqs[label] = m
                rows.append({"symbol": sym, "variant": vname, "segment": label,
                             "trades": n_trades,
                             "flat_share": flat_share(w, a, e), **m})
            equities[(sym, vname)] = run_single_asset(df, w, "2019-01-01", end,
                                                      COST, INITIAL)[0]
            print(f"{sym} {vname} done", flush=True)

        # 买入持有基准
        w_hold = pd.Series(1.0, index=[df.set_index("date").index[0]])
        for label, a, b in SEGMENTS:
            e = pd.Timestamp(b) if b else end
            eq, trades = run_single_asset(df, w_hold, a, e, COST, INITIAL)
            m = perf(eq, trades, INITIAL)
            rows.append({"symbol": sym, "variant": "买入持有", "segment": label,
                         "trades": 0, "flat_share": 0.0, **m})
        equities[(sym, "买入持有")] = run_single_asset(df, w_hold, "2019-01-01", end,
                                                     COST, INITIAL)[0]

    result = pd.DataFrame(rows)
    result.to_csv(os.path.join(RESULTS, "mr_matrix.csv"), index=False, encoding="utf-8-sig")

    # ---- T+1 税：同信号 T+1 开盘 vs 信号日收盘（理想化） ----
    print("\nT+1 税对照...", flush=True)
    tax_rows = []
    for sym in SYMBOLS:
        df = data[sym]
        for vname, gen in variants.items():
            w = gen(df)
            eq_real, _ = run_single_asset(df, w, "2019-01-01", end, COST, INITIAL)
            eq_ideal = run_close_fill(df, w, "2019-01-01", end, COST, INITIAL)
            m_r, m_i = perf(eq_real, [], INITIAL), perf(eq_ideal, [], INITIAL)
            tax_rows.append({"symbol": sym, "variant": vname,
                             "real_cagr": m_r["cagr"], "ideal_cagr": m_i["cagr"],
                             "t1_tax_pp": (m_i["cagr"] - m_r["cagr"]) * 100,
                             "real_mdd": m_r["mdd"], "ideal_mdd": m_i["mdd"]})
    tax = pd.DataFrame(tax_rows)
    tax.to_csv(os.path.join(RESULTS, "t1_tax.csv"), index=False, encoding="utf-8-sig")
    print(tax.assign(real_cagr=tax["real_cagr"].map(lambda x: f"{x:.1%}"),
                     ideal_cagr=tax["ideal_cagr"].map(lambda x: f"{x:.1%}"),
                     t1_tax_pp=tax["t1_tax_pp"].map(lambda x: f"{x:.1f}"),
                     real_mdd=tax["real_mdd"].map(lambda x: f"{x:.1%}"),
                     ideal_mdd=tax["ideal_mdd"].map(lambda x: f"{x:.1%}")).to_string(index=False))

    # ---- 与生产 momentum 的日收益相关 ----
    pool_data = {c: loader.load(c, is_etf=True, start="2017-01-01")
                 for c in ["510300", "510500", "159915", "588000", "518880"]}
    mom_sig = MomentumRotation(cfg).generate_signals(pool_data)
    bt = Backtester(cfg)
    eq_mom, _ = bt.run(pool_data, mom_sig, start="2019-01-01", end=end)
    mom_ret = eq_mom["total_value"].pct_change()
    corr_rows = []
    for (sym, vname), eq in equities.items():
        r = eq.pct_change()
        j = pd.concat([r.rename("mr"), mom_ret.rename("mom")], axis=1).dropna()
        corr_rows.append({"symbol": sym, "variant": vname,
                          "corr_vs_momentum": j["mr"].corr(j["mom"])})
    corr = pd.DataFrame(corr_rows)
    corr.to_csv(os.path.join(RESULTS, "corr_vs_momentum.csv"),
                index=False, encoding="utf-8-sig")

    # ---- 控制台摘要 ----
    print("\n===== MR 矩阵（训练段） =====")
    tr = result[result["segment"] == "train"]
    print(tr.assign(cagr=tr["cagr"].map(lambda x: f"{x:.1%}"),
                    mdd=tr["mdd"].map(lambda x: f"{x:.1%}"),
                    sharpe=tr["sharpe"].map(lambda x: f"{x:.2f}"),
                    calmar=tr["calmar"].map(lambda x: f"{x:.2f}"),
                    flat_share=tr["flat_share"].map(lambda x: f"{x:.0%}"),
                    turnover=tr["turnover"].map(lambda x: f"{x:.1f}x"))
          [["symbol", "variant", "trades", "flat_share", "cagr", "mdd",
            "sharpe", "calmar", "turnover"]].to_string(index=False))
    print("\n===== 验证段 =====")
    va = result[result["segment"] == "valid"]
    print(va.assign(cagr=va["cagr"].map(lambda x: f"{x:.1%}"),
                    mdd=va["mdd"].map(lambda x: f"{x:.1%}"),
                    calmar=va["calmar"].map(lambda x: f"{x:.2f}"))
          [["symbol", "variant", "trades", "cagr", "mdd", "calmar"]].to_string(index=False))
    print("\n===== 与 momentum 日收益相关 =====")
    print(corr.assign(corr_vs_momentum=corr["corr_vs_momentum"].map(lambda x: f"{x:.2f}"))
          .to_string(index=False))
    print(f"\n结果 CSV：{RESULTS}")


if __name__ == "__main__":
    main()
