"""第二轮研究编排：赢家邻域精耕 + 新假设探索。

纪律声明：支线一是同一假设族（唐奇安突破）在同一批数据上的敏感性分析，
不构成新的独立证据；任何"更优参数"若不在高原上，宁可放弃。
训练 ≤2022-12-30 排名、验证 2023-2024 体检、2025+ 只报告（污染段）。
结果 CSV 存 results/（_r2 后缀）。
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from us_equities.research.etf_data import load_universe
from us_equities.research.etf_engine import EtfBacktester, perf, yearly_returns, INITIAL_USD, DEFAULT_COST
from us_equities.research import etf_strategies as S
from us_equities.research import etf_strategies2 as S2

RESULTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
SEGMENTS = [("full", "2019-01-02", None), ("train", "2019-01-02", "2022-12-30"),
            ("valid", "2023-01-01", "2024-12-31"), ("report", "2025-01-01", None)]


def run_all_segments(bt, data, signals, end):
    row = {}
    eq_full = None
    for label, a, b in SEGMENTS:
        e = b if b else end
        eq, trades = bt.run(data, signals, a, e)
        m = perf(eq, trades)
        for k, v in m.items():
            row[f"{label}_{k}"] = v
        if label == "full":
            eq_full = eq
    return row, eq_full


def save(df, name):
    df.to_csv(os.path.join(RESULTS, f"{name}_r2.csv"), index=False, encoding="utf-8-sig")


def fmt_table(df, cols):
    out = df[cols].copy()
    for c in out.columns:
        if any(k in c for k in ("cagr", "mdd")):
            out[c] = out[c].map(lambda x: f"{x:.1%}" if pd.notna(x) else "N/A")
        elif any(k in c for k in ("calmar", "sharpe")):
            out[c] = out[c].map(lambda x: f"{x:.2f}" if pd.notna(x) else "N/A")
    return out.to_string(index=False)


def main(offline=True):
    data, coverage = load_universe(offline=offline)
    closes = S.close_panel(data)
    end = max(df["date"].max() for df in data.values())
    bt = EtfBacktester()

    # ========== 支线一 1：唐奇安参数高原（entry×exit 网格，QQQ/SPY） ==========
    print("== 1. 唐奇安参数网格 ==", flush=True)
    rows = []
    for sym in ("QQQ", "SPY"):
        for entry in (15, 20, 40, 55):
            for exit_ in (5, 10, 20):
                sig = S.donchian(data, sym, entry, exit_)
                row, _ = run_all_segments(bt, data, sig, end)
                row.update(symbol=sym, entry=entry, exit=exit_)
                rows.append(row)
                print(f"  {sym} {entry}/{exit_} done", flush=True)
    grid = pd.DataFrame(rows)
    cols = ["symbol", "entry", "exit", "train_cagr", "train_mdd", "train_calmar",
            "valid_cagr", "valid_mdd", "valid_calmar", "report_cagr", "report_mdd",
            "full_cagr", "full_mdd", "full_sharpe", "full_calmar", "full_turnover", "full_fees"]
    grid = grid[["symbol", "entry", "exit"] +
                [c for c in grid.columns if c not in ("symbol", "entry", "exit")]]
    save(grid, "grid_donchian")
    print(fmt_table(grid.sort_values(["symbol", "train_calmar"], ascending=[True, False]), cols))

    # ========== 支线一 2：退出变体（entry=20） ==========
    print("\n== 2. 退出变体 ==", flush=True)
    rows = []
    for sym in ("QQQ", "SPY"):
        variants = {
            "low10(基准)": S.donchian(data, sym, 20, 10),
            "chandelier3x22": S2.donchian_chandelier(data, sym),
            "ma20": S2.donchian_ma_exit(data, sym),
            "time20": S2.donchian_time_exit(data, sym),
        }
        for vname, sig in variants.items():
            row, _ = run_all_segments(bt, data, sig, end)
            row.update(symbol=sym, exit_variant=vname)
            rows.append(row)
    exits = pd.DataFrame(rows)
    save(exits, "exit_variants")
    print(fmt_table(exits, ["symbol", "exit_variant", "train_cagr", "train_mdd",
                            "train_calmar", "valid_cagr", "valid_mdd", "valid_calmar",
                            "full_cagr", "full_mdd", "full_calmar", "full_turnover"]))

    # ========== 支线一 3：标的泛化（20/10） ==========
    print("\n== 3. 标的泛化 ==", flush=True)
    rows = []
    gen_eqs = {}
    for sym in ("QQQ", "SPY", "XLK", "MDY", "IWM", "GLD", "TLT", "EFA"):
        sig = S.donchian(data, sym, 20, 10)
        row, eq = run_all_segments(bt, data, sig, end)
        row.update(symbol=sym)
        rows.append(row)
        gen_eqs[sym] = eq
    gen = pd.DataFrame(rows)
    save(gen, "generalization")
    print(fmt_table(gen, ["symbol", "train_cagr", "train_mdd", "train_calmar",
                          "valid_cagr", "valid_mdd", "valid_calmar",
                          "full_cagr", "full_mdd", "full_sharpe", "full_calmar"]))

    # ========== 支线一 4：仓位与加仓（donchian_QQQ） ==========
    print("\n== 4. 仓位与加仓 ==", flush=True)
    rows = []
    for vname, sig in {
        "full_100%(基准)": S.donchian(data, "QQQ", 20, 10),
        "atr_risk1%": S2.donchian_atr_sized(data, "QQQ", risk_pct=0.01),
        "atr_risk2%": S2.donchian_atr_sized(data, "QQQ", risk_pct=0.02),
        "pyramid_1%x(1+3)": S2.donchian_pyramid(data, "QQQ", risk_pct=0.01),
    }.items():
        row, _ = run_all_segments(bt, data, sig, end)
        row.update(sizing=vname)
        rows.append(row)
    sizing = pd.DataFrame(rows)
    save(sizing, "sizing")
    print(fmt_table(sizing, ["sizing", "train_cagr", "train_mdd", "train_calmar",
                             "valid_cagr", "valid_calmar", "full_cagr", "full_mdd",
                             "full_calmar", "full_turnover", "full_fees"]))

    # ========== 支线一 5：成本压力 ==========
    print("\n== 5. 成本压力 ==", flush=True)
    rows = []
    for sym in ("QQQ", "SPY"):
        sig = S.donchian(data, sym, 20, 10)
        for slip in (10, 20, 30, 50):
            cost = dict(DEFAULT_COST, slippage_bps=slip)
            bt2 = EtfBacktester(cost=cost)
            eq, trades = bt2.run(data, sig, "2019-01-02", end)
            m = perf(eq, trades)
            rows.append({"symbol": sym, "slippage_bps": slip, **m})
    stress = pd.DataFrame(rows)
    save(stress, "cost_stress")
    print(fmt_table(stress, ["symbol", "slippage_bps", "total", "cagr", "mdd",
                             "sharpe", "calmar", "fees"]))

    # ========== 支线二 6：regime 自适应 ==========
    print("\n== 6. regime 自适应 ==", flush=True)
    rows = []
    for sym in ("QQQ", "SPY"):
        for vname, sig in {
            "donchian20/10(基准)": S.donchian(data, sym, 20, 10),
            "regime_adx25": S2.regime_adaptive(data, sym),
        }.items():
            row, _ = run_all_segments(bt, data, sig, end)
            row.update(symbol=sym, variant=vname)
            rows.append(row)
    regime = pd.DataFrame(rows)
    save(regime, "regime_adaptive")
    print(fmt_table(regime, ["symbol", "variant", "train_cagr", "train_mdd", "train_calmar",
                             "valid_cagr", "valid_calmar", "full_cagr", "full_mdd",
                             "full_calmar", "full_turnover", "full_fees"]))

    # ========== 支线二 7：多均线投票 ==========
    print("\n== 7. 多均线投票 ==", flush=True)
    rows = []
    for sym in ("QQQ", "SPY"):
        for vname, sig in {
            "donchian20/10(基准)": S.donchian(data, sym, 20, 10),
            "ma_vote_4": S2.ma_vote(closes, sym),
        }.items():
            row, _ = run_all_segments(bt, data, sig, end)
            row.update(symbol=sym, variant=vname)
            rows.append(row)
    mavote = pd.DataFrame(rows)
    save(mavote, "ma_vote")
    print(fmt_table(mavote, ["symbol", "variant", "train_cagr", "train_mdd", "train_calmar",
                             "valid_cagr", "valid_calmar", "full_cagr", "full_mdd",
                             "full_calmar", "full_turnover"]))

    # ========== 支线二 8：防守 sleeve ==========
    print("\n== 8. 防守 sleeve ==", flush=True)
    don_q = S.donchian(data, "QQQ", 20, 10)
    don_g = S.donchian(data, "GLD", 20, 10)
    don_t = S.donchian(data, "TLT", 20, 10)
    combos = {
        "don_QQQ(基准)": don_q,
        "don_GLD_sleeve": don_g,
        "don_TLT_sleeve": don_t,
        "50/50_QQQ+GLD": S2.combine_weighted({"q": (don_q, 0.5), "g": (don_g, 0.5)}),
        "50/50_QQQ+TLT": S2.combine_weighted({"q": (don_q, 0.5), "t": (don_t, 0.5)}),
    }
    rows = []
    combo_eqs = {}
    for vname, sig in combos.items():
        row, eq = run_all_segments(bt, data, sig, end)
        row.update(variant=vname)
        rows.append(row)
        combo_eqs[vname] = eq
    sleeves = pd.DataFrame(rows)
    save(sleeves, "sleeves")
    print(fmt_table(sleeves, ["variant", "train_cagr", "train_mdd", "train_calmar",
                              "valid_cagr", "valid_calmar", "full_cagr", "full_mdd",
                              "full_sharpe", "full_calmar"]))
    corr = pd.DataFrame({k: v.pct_change() for k, v in combo_eqs.items()}).dropna().corr()
    corr.to_csv(os.path.join(RESULTS, "sleeves_corr_r2.csv"), encoding="utf-8-sig")
    print("\nsleeve 日收益相关矩阵:")
    print(corr.round(2).to_string())

    print(f"\n结果 CSV 已保存（_r2 后缀）：{RESULTS}")


if __name__ == "__main__":
    main(offline="--offline" in sys.argv or True)
