"""策略衰减监控（monitor 子命令）：五个台账的滚动体检 + 预警。

预警规则（预声明）：
- 滚动 252 日夏普 < 0 标黄、< -0.5 标红（数据不足 60 日则标"数据不足"）
- 台账净值与同策略理论信号净值偏离 >2% 标黄
输出中文体检表，存 reports/。
"""
import os
import sys
from datetime import datetime

import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def load_nav(path):
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path, parse_dates=["date"])
    if df.empty:
        return None
    return df.set_index("date")["nav"].sort_index()


def rolling_stats(nav: pd.Series):
    ret = nav.pct_change()
    r63 = nav.iloc[-1] / nav.iloc[-64] - 1 if len(nav) > 63 else None
    r252 = nav.iloc[-1] / nav.iloc[-253] - 1 if len(nav) > 252 else None
    sharpe252 = None
    if len(ret.dropna()) >= 60:
        w = ret.dropna().tail(252)
        if w.std() > 0:
            sharpe252 = w.mean() / w.std() * np.sqrt(252)
    cur_dd = nav.iloc[-1] / nav.cummax().iloc[-1] - 1
    max_dd = (nav / nav.cummax() - 1).min()
    return dict(days=len(nav), r63=r63, r252=r252, sharpe252=sharpe252,
                cur_dd=cur_dd, max_dd=max_dd)


def verdicts(stats, dev_pct):
    flags = []
    s = stats["sharpe252"]
    if s is None:
        flags.append("数据不足")
    elif s < -0.5:
        flags.append("🔴 滚动夏普<-0.5")
    elif s < 0:
        flags.append("🟡 滚动夏普<0")
    if dev_pct is not None and abs(dev_pct) > 0.02:
        flags.append(f"🟡 台账vs理论偏离{dev_pct:+.1%}")
    return "；".join(flags) if flags else "OK"


def theoretical_navs(project_dir, end=None):
    """各台账同策略理论净值（回测引擎同口径，台账起点起算）。"""
    from data import DataLoader
    from backtest import Backtester
    from strategies import MomentumRotation, StaggeredMomentum

    cfg = yaml.safe_load(open(os.path.join(project_dir, "config.yaml"), encoding="utf-8"))
    loader = DataLoader(cache_dir=os.path.join(project_dir, "data_cache"))
    pool = ["510300", "510500", "159915", "588000", "518880"]
    out = {}
    try:
        data = {c: loader.load(c, is_etf=True, start="2017-01-01") for c in pool}
        bt = Backtester(cfg)
        eq1, _ = bt.run(data, MomentumRotation(cfg).generate_signals(data),
                        start="2026-09-01")
        out["A股momentum"] = eq1["total_value"]
        bt2 = Backtester(cfg, rebalance_on_event=True)
        eq2, _ = bt2.run(data, StaggeredMomentum(cfg).generate_signals(data),
                         start="2026-09-01")
        out["A股staggered"] = eq2["total_value"]
    except Exception as e:  # noqa: BLE001
        print(f"警告：A股理论净值计算失败（{e}）", file=sys.stderr)
    try:
        from us_equities.research.etf_data import load_universe
        from us_equities.research import etf_strategies as US_S
        from us_equities.research.etf_engine import EtfBacktester
        us_data, _ = load_universe(symbols=["QQQ", "SPY"], cache=os.path.join(
            project_dir, "us_equities", "research", "cache"), offline=True)
        for name, sym in (("美股donchian_QQQ", "QQQ"), ("美股donchian_SPY", "SPY")):
            sig = US_S.donchian(us_data, sym, 20, 10)
            eq, _ = EtfBacktester().run(us_data, sig, "2026-09-01", None)
            out[name] = eq
    except Exception as e:  # noqa: BLE001
        print(f"警告：美股理论净值计算失败（{e}）", file=sys.stderr)
    try:
        from macro_research.data_layer import load_btc
        from macro_research.strategies import run_single_asset
        from research_r4.runner import btc_donchian_filtered, f1_volume_mask
        btc, _ = load_btc()
        sig = btc_donchian_filtered(btc, allow_entry=f1_volume_mask(btc))
        eq, _ = run_single_asset(btc, sig, "2026-09-01", None,
                                 {"commission": 0.001, "slippage": 0.001, "lot": 1e-4},
                                 50_000)
        out["BTC×F1"] = eq
    except Exception as e:  # noqa: BLE001
        print(f"警告：BTC 理论净值计算失败（{e}）", file=sys.stderr)
    return out


ACCOUNTS = [
    ("A股momentum", "paper", "nav_momentum.csv"),
    ("A股staggered", "paper", "nav_staggered.csv"),
    ("美股donchian_QQQ", "paper_us", "nav_qqq.csv"),
    ("美股donchian_SPY", "paper_us", "nav_spy.csv"),
    ("BTC×F1", "paper_btc", "nav_btc_f1.csv"),
]


def build_monitor(project_dir):
    theo = theoretical_navs(project_dir)
    lines = [f"===== 策略衰减监控体检表（{datetime.now():%Y-%m-%d %H:%M}）====="]
    rows = []
    for name, sub, fname in ACCOUNTS:
        nav = load_nav(os.path.join(project_dir, sub, fname))
        if nav is None:
            lines.append(f"\n[{name}] 台账不存在或无数据")
            continue
        st = rolling_stats(nav)
        t = theo.get(name)
        dev = None
        if t is not None and len(t) and len(nav):
            t_norm = t / t.iloc[0]
            n_norm = nav / nav.iloc[0]
            common = t_norm.index.intersection(n_norm.index)
            if len(common):
                dev = float(n_norm.loc[common[-1]] / t_norm.loc[common[-1]] - 1)
        flag = verdicts(st, dev)
        rows.append({"account": name, **st, "dev_vs_theory": dev, "预警": flag})
        lines.append(f"\n[{name}]（{st['days']} 天净值）")
        lines.append(f"  63日收益: {_pct(st['r63'])}  252日收益: {_pct(st['r252'])}"
                     f"  滚动252日夏普: {_f2(st['sharpe252'])}")
        lines.append(f"  当前回撤: {st['cur_dd']:.2%}  台账期最大回撤: {st['max_dd']:.2%}"
                     f"  台账vs理论偏离: {_pct(dev, plus=True)}")
        lines.append(f"  预警: {flag}")
    text = "\n".join(lines)
    return text, pd.DataFrame(rows)


def _pct(x, plus=False):
    if x is None:
        return "数据不足"
    return f"{x:+.2%}" if plus else f"{x:.2%}"


def _f2(x):
    return "数据不足" if x is None else f"{x:.2f}"
