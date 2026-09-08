"""调参寻优：MomentumRotation 网格搜索（样本内排名 + 样本外验证）。

- 样本内(IS): 2019-01-01 ~ 2023-12-31
- 样本外(OOS): 2024-01-01 ~ 最新
- 排名指标：样本内卡玛比率（年化收益 / |最大回撤|）
数据起点比回测起点早约 450 天，覆盖长周期参数的预热期。
"""
import copy
import itertools
import time
from datetime import datetime

import pandas as pd

from strategies import MomentumRotation
from backtest import Backtester
from metrics import calc_metrics

IS_START, IS_END = "2019-01-01", "2023-12-31"
OOS_START = "2024-01-01"
WARMUP_START = "2017-10-01"  # 比 IS 起点早 ~450 天，供动量/均线预热

GRID = {
    "momentum_days": [10, 20, 30, 40, 60],
    "rebalance_days": [5, 10, 20],
    "trend_ma": [0, 20, 60],
    "defensive": ["511010", "518880", None],   # None = 空仓
    "pool": ["default", "default+510880"],
}
POOLS = {
    "default": ["510300", "510500", "159915", "588000"],
    "default+510880": ["510300", "510500", "159915", "588000", "510880"],
}
ALL_CODES = sorted(set(POOLS["default+510880"]) | {"511010", "518880"})


def _sub_cfg(base_cfg, momentum_days, rebalance_days, trend_ma, defensive, pool):
    cfg = copy.deepcopy(base_cfg)
    cfg["strategy"]["momentum_days"] = momentum_days
    cfg["strategy"]["rebalance_days"] = rebalance_days
    cfg["strategy"]["trend_ma"] = trend_ma
    cfg["universe"]["etfs"] = POOLS[pool]
    cfg["universe"]["stocks"] = []
    cfg["universe"]["defensive"] = defensive
    return cfg


def _run_segment(cfg, data, signals, start, end):
    """在 [start, end] 窗口内跑回测（signals 已用预热数据生成，
    引擎会承接区间开始前最后的已知目标）。"""
    equity, trades = Backtester(cfg).run(data, signals, start=start, end=end)
    if equity.empty:
        return None
    return calc_metrics(equity["total_value"], trades,
                        initial=cfg["backtest"]["initial_cash"])


def run_optimize(base_cfg, loader, top_n=15, progress=True):
    """执行网格搜索。返回完整结果 DataFrame（按样本内卡玛降序）。"""
    end = datetime.now().strftime("%Y-%m-%d")
    # 一次性加载全部标的（含预热期），后续组合只切片
    data_all = {}
    for code in ALL_CODES:
        df = loader.load(code, is_etf=True, start=WARMUP_START, end=end)
        if df.empty:
            print(f"警告：{code} 无数据，相关组合将跳过")
            continue
        data_all[code] = df

    combos = list(itertools.product(GRID["momentum_days"], GRID["rebalance_days"],
                                    GRID["trend_ma"], GRID["defensive"], GRID["pool"]))
    total = len(combos)
    rows = []
    t0 = time.time()
    for idx, (mom, reb, tma, dfn, pool) in enumerate(combos, 1):
        cfg = _sub_cfg(base_cfg, mom, reb, tma, dfn, pool)
        codes = POOLS[pool] + ([dfn] if dfn else [])
        if not all(c in data_all for c in codes):
            continue
        data = {c: data_all[c] for c in codes}
        signals = MomentumRotation(cfg).generate_signals(data)

        is_m = _run_segment(cfg, data, signals, IS_START, IS_END)
        oos_m = _run_segment(cfg, data, signals, OOS_START, end)
        if is_m is None:
            continue
        row = {"momentum_days": mom, "rebalance_days": reb, "trend_ma": tma,
               "defensive": dfn or "空仓", "pool": pool}
        for prefix, m in (("IS", is_m), ("OOS", oos_m)):
            if m is None:
                continue
            row[f"{prefix}_年化"] = m["年化收益率"]
            row[f"{prefix}_回撤"] = m["最大回撤"]
            row[f"{prefix}_夏普"] = m["夏普比率"]
            row[f"{prefix}_卡玛"] = m["卡玛比率"]
            row[f"{prefix}_交易"] = m["交易次数"]
        rows.append(row)
        if progress and (idx % 10 == 0 or idx == total):
            print(f"进度 {idx}/{total} 组，已用时 {time.time() - t0:.0f}s", flush=True)

    result = pd.DataFrame(rows)
    result = result.sort_values("IS_卡玛", ascending=False, na_position="last").reset_index(drop=True)
    return result


def format_top_table(result: pd.DataFrame, top_n=15) -> str:
    """格式化 top N 表格（样本内/外关键指标）。"""
    cols = ["momentum_days", "rebalance_days", "trend_ma", "defensive", "pool",
            "IS_年化", "IS_回撤", "IS_夏普", "IS_卡玛", "IS_交易",
            "OOS_年化", "OOS_回撤", "OOS_夏普", "OOS_卡玛", "OOS_交易"]
    df = result.head(top_n)[cols].copy()
    for c in ["IS_年化", "IS_回撤", "OOS_年化", "OOS_回撤"]:
        df[c] = df[c].map(lambda x: f"{x:.1%}" if pd.notna(x) else "N/A")
    for c in ["IS_夏普", "IS_卡玛", "OOS_夏普", "OOS_卡玛"]:
        df[c] = df[c].map(lambda x: f"{x:.2f}" if pd.notna(x) else "N/A")
    df = df.rename(columns={"momentum_days": "动量", "rebalance_days": "调仓",
                            "trend_ma": "趋势MA", "defensive": "防守", "pool": "标的池"})
    return df.to_string(index=False)
