"""R7b：A股行业轮动（行业 ETF 池，生产框架参数沿用不寻优）。

池（预声明 9 只）：512880 券商/512480 半导体/515030 新能源车/512010 医药/
512660 军工/512800 银行/159928 消费/512690 酒/515790 光伏；防守 518880。
框架：生产 momentum（20日动量+MA60+5日调仓+T+1开盘，万1+0.1%滑点）原样换池；
对照：staggered 同框架（5组+波动控仓）。
训练 2019-2023 判定（行业池 vs 宽基 V0 训练卡玛中位 0.94），验证 2024+ 体检；
相位扰动必测；行业池胜出才加测"宽基+行业合并池"（预声明）。
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
from strategies import StaggeredMomentum  # noqa: E402
from metrics import calc_metrics  # noqa: E402
from research_r1.runner import momentum_signals, POOL_V0  # noqa: E402

PROJECT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
os.makedirs(RESULTS, exist_ok=True)

SECTOR_POOL = ["512880", "512480", "515030", "512010", "512660",
               "512800", "159928", "512690", "515790"]
SECTOR_NAMES = {"512880": "券商", "512480": "半导体", "515030": "新能源车",
                "512010": "医药", "512660": "军工", "512800": "银行",
                "159928": "消费", "512690": "酒", "515790": "光伏"}
DEFENSIVE = "518880"
SEGMENTS = [("train", "2019-01-01", "2023-12-31"), ("valid", "2024-01-01", None),
            ("full", "2019-01-01", None)]


def eval_momentum(bt, data, pool, end, initial, phases=range(5)):
    out = {}
    for label, a, b in SEGMENTS:
        calmars, mdds, cagrs, sharpes = [], [], [], []
        for phase in phases:
            sig = momentum_signals(data, pool, phase=phase)
            e = pd.Timestamp(b) if b else end
            eq, trades = bt.run(data, sig, start=a, end=e)
            m = calc_metrics(eq["total_value"], trades, initial=initial)
            calmars.append(m["卡玛比率"])
            mdds.append(m["最大回撤"])
            cagrs.append(m["年化收益率"])
            sharpes.append(m["夏普比率"])
        out[label] = dict(calmar_median=np.median(calmars), mdd_worst=min(mdds),
                          cagr_median=np.median(cagrs), sharpe_median=np.median(sharpes),
                          calmar_min=min(calmars), calmar_max=max(calmars),
                          cagr_spread_pp=(max(cagrs) - min(cagrs)) * 100)
    return out


def eval_staggered(bt, cfg, data, end, initial):
    sig = StaggeredMomentum(cfg).generate_signals(data)
    out = {}
    for label, a, b in SEGMENTS:
        e = pd.Timestamp(b) if b else end
        eq, trades = bt.run(data, sig, start=a, end=e)
        m = calc_metrics(eq["total_value"], trades, initial=initial)
        out[label] = dict(calmar_median=m["卡玛比率"], mdd_worst=m["最大回撤"],
                          cagr_median=m["年化收益率"], sharpe_median=m["夏普比率"],
                          calmar_min=np.nan, calmar_max=np.nan, cagr_spread_pp=np.nan)
    return out


def main():
    cfg = yaml.safe_load(open(os.path.join(PROJECT, "config.yaml"), encoding="utf-8"))
    initial = cfg["backtest"]["initial_cash"]
    loader = DataLoader(cache_dir=os.path.join(PROJECT, "data_cache"))
    codes = sorted(set(SECTOR_POOL + POOL_V0 + [DEFENSIVE]))
    print(f"加载 {len(codes)} 只...", flush=True)
    data = {c: loader.load(c, is_etf=True, start="2017-01-01") for c in codes}
    end = max(df["date"].max() for df in data.values())
    print(f"数据截至 {end.date()}")
    for c in SECTOR_POOL:
        print(f"  {c} {SECTOR_NAMES[c]}: 自 {data[c]['date'].min().date()} 起")

    bt = Backtester(cfg)
    bt_st = Backtester(cfg, rebalance_on_event=True)
    rows = []

    # 宽基 V0 基线（相位口径，复算同引擎）
    print("V0 宽基...", flush=True)
    res = eval_momentum(bt, data, POOL_V0, end, initial)
    for label, m in res.items():
        rows.append({"variant": "V0_宽基池(momentum)", "segment": label, **m})

    # 行业池 momentum
    print("行业池 momentum...", flush=True)
    res = eval_momentum(bt, data, SECTOR_POOL, end, initial)
    for label, m in res.items():
        rows.append({"variant": "行业池(momentum)", "segment": label, **m})

    # 行业池 staggered
    print("行业池 staggered...", flush=True)
    cfg_s = copy.deepcopy(cfg)
    cfg_s["universe"]["etfs"] = SECTOR_POOL
    res = eval_staggered(bt_st, cfg_s, data, end, initial)
    for label, m in res.items():
        rows.append({"variant": "行业池(staggered)", "segment": label, **m})

    result = pd.DataFrame(rows)
    # 合并池（预声明：行业池训练卡玛中位 > V0 才加测）
    sec_train = result[(result["variant"] == "行业池(momentum)")
                       & (result["segment"] == "train")]["calmar_median"].iloc[0]
    v0_train = result[(result["variant"] == "V0_宽基池(momentum)")
                      & (result["segment"] == "train")]["calmar_median"].iloc[0]
    print(f"\n训练卡玛中位：行业池 {sec_train:.2f} vs V0 {v0_train:.2f}")
    if sec_train > v0_train:
        print("行业池胜出 → 加测合并池（预声明触发）", flush=True)
        res = eval_momentum(bt, data, POOL_V0 + SECTOR_POOL, end, initial)
        for label, m in res.items():
            rows.append({"variant": "宽基+行业合并池(momentum)", "segment": label, **m})
    else:
        print("行业池未胜出 → 合并池不加测（预声明）")

    result = pd.DataFrame(rows)
    result.to_csv(os.path.join(RESULTS, "r7b_sector.csv"),
                  index=False, encoding="utf-8-sig")

    # 行业池 staggered 对照生产 staggered（宽基）数字引用（生产基线：全期14.73%/-10.48%/1.41）
    show = result.assign(
        cagr_median=result["cagr_median"].map(lambda x: f"{x:.1%}"),
        mdd_worst=result["mdd_worst"].map(lambda x: f"{x:.1%}"),
        sharpe_median=result["sharpe_median"].map(lambda x: f"{x:.2f}"),
        calmar_median=result["calmar_median"].map(lambda x: f"{x:.2f}"),
        calmar_min=result["calmar_min"].map(lambda x: f"{x:.2f}" if pd.notna(x) else "-"),
        calmar_max=result["calmar_max"].map(lambda x: f"{x:.2f}" if pd.notna(x) else "-"),
        cagr_spread_pp=result["cagr_spread_pp"].map(
            lambda x: f"{x:.1f}" if pd.notna(x) else "-"))
    print("\n===== R7b 行业轮动（相位口径） =====")
    print(show.to_string(index=False))
    print(f"\n结果 CSV：{RESULTS}")


if __name__ == "__main__":
    main()
