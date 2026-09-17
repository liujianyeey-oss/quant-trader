"""R10：市场宽度指标 + 印度市场纳入复核。

① 宽度 = 9 只行业 ETF（R7b 池）收盘价 > 各自 MA60 的比例（未上市剔除归一化）。
   预声明：调仓日宽度 < 30% → 该周期进防守；基准=生产 momentum（5 相位中位）。
   敏感性 20%/40% 只作展示。冗余自查：与 MA60 趋势过滤的触发重合度，>80% 判"内生已覆盖"。
② 印度 164824（QDII LOF）加入 R1 V1 池复核：与 510300 相关性 + 相位中位卡玛变化。
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
from research_r1.runner import momentum_signals, POOL_V0, POOL_V1, DEFENSIVE  # noqa: E402

PROJECT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
os.makedirs(RESULTS, exist_ok=True)

SECTOR_POOL = ["512880", "512480", "515030", "512010", "512660",
               "512800", "159928", "512690", "515790"]
SEGMENTS = [("train", "2019-01-01", "2023-12-31"), ("valid", "2024-01-01", None),
            ("full", "2019-01-01", None)]


# ---------- ① 宽度 ----------

def breadth_series(data):
    """每日宽度：已上市行业 ETF 中 收盘>MA60 的比例。"""
    frames = {}
    for c in SECTOR_POOL:
        d = data[c].set_index("date")["close"]
        above = (d > d.rolling(60).mean()).astype(float)
        above[d.isna()] = np.nan  # 未上市期剔除
        frames[c] = above
    panel = pd.DataFrame(frames).sort_index()
    return panel.mean(axis=1, skipna=True)


def apply_breadth_filter(signals, breadth, threshold):
    """宽度 < threshold 的信号日 → 强制防守（不持股）。"""
    if signals.empty:
        return signals
    out = signals.copy()
    b = breadth.reindex(out.index)
    pool_cols = [c for c in out.columns if c != DEFENSIVE]
    mask = (b < threshold).fillna(False).to_numpy()
    out.loc[mask, pool_cols] = 0.0
    out.loc[mask, DEFENSIVE] = 1.0
    return out


def eval_phases(bt, data, pool, end, initial, breadth=None, threshold=0.30):
    out = {}
    for label, a, b in SEGMENTS:
        calmars, mdds, cagrs, sharpes = [], [], [], []
        for phase in range(5):
            sig = momentum_signals(data, pool, phase=phase)
            if breadth is not None:
                sig = apply_breadth_filter(sig, breadth, threshold)
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


def redundancy_check(data, breadth, threshold=0.30):
    """宽度触发（<threshold）与 MA60 趋势过滤防守的重合度（按调仓信号日）。"""
    sig = momentum_signals(data, POOL_V0, phase=0)
    if sig.empty:
        return {}
    pool_cols = [c for c in sig.columns if c != DEFENSIVE]
    trend_def = (sig[pool_cols].sum(axis=1) == 0)  # 基准趋势过滤已防守
    b = breadth.reindex(sig.index)
    breadth_def = (b < threshold).fillna(False)
    n_b = int(breadth_def.sum())
    n_t = int(trend_def.sum())
    both = int((breadth_def & trend_def).sum())
    return {"信号日数": len(sig), "宽度触发日数": n_b, "趋势防守日数": n_t,
            "重合日数": both,
            "宽度触发中趋势也防守占比": both / n_b if n_b else np.nan,
            "趋势防守中宽度也触发占比": both / n_t if n_t else np.nan}


# ---------- 主流程 ----------

def main():
    cfg = yaml.safe_load(open(os.path.join(PROJECT, "config.yaml"), encoding="utf-8"))
    initial = cfg["backtest"]["initial_cash"]
    loader = DataLoader(cache_dir=os.path.join(PROJECT, "data_cache"))
    codes = sorted(set(POOL_V1 + [DEFENSIVE, "164824"] + SECTOR_POOL))
    print(f"加载 {len(codes)} 只...", flush=True)
    data = {c: loader.load(c, is_etf=True, start="2017-01-01") for c in codes}
    end = max(df["date"].max() for df in data.values())
    print(f"数据截至 {end.date()}")

    bt = Backtester(cfg)
    rows = []

    # ---- ① 宽度 ----
    print("\n== ① 市场宽度 ==", flush=True)
    b = breadth_series(data)
    b.to_csv(os.path.join(RESULTS, "breadth_series.csv"), encoding="utf-8-sig")
    red = redundancy_check(data, b, 0.30)
    print("冗余自查:", red)
    pd.DataFrame([red]).to_csv(os.path.join(RESULTS, "breadth_redundancy.csv"),
                               index=False, encoding="utf-8-sig")
    res = eval_phases(bt, data, POOL_V0, end, initial)  # 基准
    for label, m in res.items():
        rows.append({"part": "①宽度", "variant": "基准_momentum", "segment": label, **m})
    for th, name in ((0.30, "宽度30%(预声明档)"), (0.20, "宽度20%(敏感性)"),
                     (0.40, "宽度40%(敏感性)")):
        res = eval_phases(bt, data, POOL_V0, end, initial, breadth=b, threshold=th)
        for label, m in res.items():
            rows.append({"part": "①宽度", "variant": name, "segment": label, **m})
        print(f"完成 {name}", flush=True)

    # ---- ② 印度 ----
    print("\n== ② 印度纳入复核 ==", flush=True)
    ind = data["164824"].set_index("date")["close"]
    b300 = data["510300"].set_index("date")["close"]
    rr = pd.concat([ind.pct_change().rename("ind"), b300.pct_change().rename("b300")],
                   axis=1).dropna()
    rr = rr[rr.index >= "2019-01-01"]
    corr_ind = rr["ind"].corr(rr["b300"])
    ind_amount20 = (data["164824"]["close"] * data["164824"]["volume"]).tail(20).mean() / 1e4
    print(f"164824 vs 510300 相关系数: {corr_ind:.2f}，"
          f"近20日日均成交额(近似): {ind_amount20:,.0f} 万元")

    res = eval_phases(bt, data, POOL_V1, end, initial)  # R1 V1 复算（同引擎）
    for label, m in res.items():
        rows.append({"part": "②印度", "variant": "V1复算(无印度)", "segment": label, **m})
    res = eval_phases(bt, data, POOL_V1 + ["164824"], end, initial)
    for label, m in res.items():
        rows.append({"part": "②印度", "variant": "V1+印度164824", "segment": label, **m})

    result = pd.DataFrame(rows)
    result.to_csv(os.path.join(RESULTS, "r10.csv"), index=False, encoding="utf-8-sig")

    show = result.assign(
        cagr_median=result["cagr_median"].map(lambda x: f"{x:.1%}"),
        mdd_worst=result["mdd_worst"].map(lambda x: f"{x:.1%}"),
        sharpe_median=result["sharpe_median"].map(lambda x: f"{x:.2f}"),
        calmar_median=result["calmar_median"].map(lambda x: f"{x:.2f}"),
        calmar_min=result["calmar_min"].map(lambda x: f"{x:.2f}"),
        calmar_max=result["calmar_max"].map(lambda x: f"{x:.2f}"))
    print("\n===== R10 结果 =====")
    print(show.to_string(index=False))
    print(f"\n结果 CSV：{RESULTS}")


if __name__ == "__main__":
    main()
