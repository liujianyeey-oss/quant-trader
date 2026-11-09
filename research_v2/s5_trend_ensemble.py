"""S5：trend_ma 40/50/60 + 多数投票 ensemble 验证（策略 v2 设计文档 S5 阶段）。

背景：审计（reports/strategy_audit_20261108.md）发现 trend_ma=60 全期年化 18.5%，
而 40/120 仅 10.7%/8.9%——疑为孤峰。本脚本对 momentum/staggered 回测
trend_ma ∈ {40,50,60} 单参数与 ensemble 变体，含成本、5 相位中位数口径。

ensemble 语义说明：三周期多数投票（≥2/3 判定趋势成立）。对二元投票，
"等权信号均值 >0.5" 与 "≥2/3 多数" 数学等价，故只实现多数投票；
不采用分数仓位的信号均值方案——momentum 设计为满仓/空仓切换，
零碎暴露属 staggered 层的职责（保持策略语义干净、换手可控）。
"""
import copy, sys
import numpy as np
import pandas as pd
import yaml
sys.path.insert(0, "/Users/liujianye10/quant-trader")

from backtest import Backtester
from strategies import MomentumRotation, StaggeredMomentum
from metrics import calc_metrics

BASE = "/Users/liujianye10/quant-trader"
CODES = ["510300", "510500", "159915", "588000", "518880"]
FETCH_START = pd.Timestamp("2019-01-01") - pd.Timedelta(days=450)
START = "2019-01-01"
ENSEMBLE = (40, 50, 60)

data = {c: pd.read_parquet(f"{BASE}/data_cache/{c}_qfq.parquet") for c in CODES}
for c in CODES:
    data[c] = data[c][data[c]["date"] >= FETCH_START].reset_index(drop=True)
base_cfg = yaml.safe_load(open(f"{BASE}/config.yaml"))
assert base_cfg["cost"]["min_commission"] == 5


class MomentumEnsemble(MomentumRotation):
    """多数投票趋势过滤：close ≥ MA40/MA50/MA60 中至少 2 条成立。"""

    def generate_signals(self, data):
        panel = self._close_panel(data)
        if panel.empty:
            return pd.DataFrame()
        momentum = panel / panel.shift(self.momentum_days) - 1.0
        mas = {w: panel.rolling(w).mean() for w in ENSEMBLE}
        codes = [c for c in self.attack_pool if c in panel.columns]
        rows = []
        for i in range(self.momentum_days + 1, len(panel)):
            if (i - self.momentum_days - 1) % self.rebalance_days != 0:
                continue
            date = panel.index[i]
            mom = momentum.iloc[i][codes].dropna()
            if mom.empty:
                continue
            best = mom.idxmax()
            votes = sum(1 for w in ENSEMBLE
                        if pd.notna(mas[w].iloc[i][best])
                        and panel.iloc[i][best] >= mas[w].iloc[i][best])
            rows.append((date, {best: 1.0} if votes >= 2 else {self.defensive: 1.0}))
        sig = pd.DataFrame(0.0, index=[d for d, _ in rows], columns=list(panel.columns))
        for date, w in rows:
            for c, wt in w.items():
                sig.loc[date, c] = wt
        return sig


class StaggeredEnsemble(StaggeredMomentum):
    """组内趋势过滤改多数投票（≥2/3）。"""

    def _cohort_signals(self, panel, weekday):
        rets = panel.pct_change()
        vol = rets.rolling(self.vol_window, min_periods=self.vol_window).std() * np.sqrt(252)
        mom = panel / panel.shift(self.momentum_days) - 1.0
        mas = {w: panel.rolling(w, min_periods=w).mean() for w in ENSEMBLE}
        codes = [c for c in self.attack_pool if c in panel.columns]
        rows = []
        for i in range(self.warmup, len(panel)):
            date = panel.index[i]
            if date.weekday() != weekday:
                continue
            m = mom.iloc[i][codes].dropna()
            if m.empty:
                continue
            winner = m.idxmax()
            votes = sum(1 for w in ENSEMBLE
                        if panel.iloc[i][winner] >= mas[w].iloc[i][winner])
            pick = winner if votes >= 2 else self.defensive
            row = pd.Series(0.0, index=panel.columns)
            v = vol.iloc[i][pick]
            if pd.notna(v):
                row[pick] = min(1.0, self.vol_budget / max(0.03, v))
            rows.append(row.rename(date))
        return pd.DataFrame(rows)


def run(signals, roe, phase=0):
    if phase:
        cal = pd.DataFrame({c: df.set_index("date")["close"]
                            for c, df in data.items()}).sort_index().index
        pos = {d: i for i, d in enumerate(cal)}
        idx = [cal[min(pos[d] + phase, len(cal) - 1)] for d in signals.index]
        signals = signals.copy()
        signals.index = pd.DatetimeIndex(idx)
        signals = signals[~signals.index.duplicated(keep="last")]
    eq, trades = Backtester(base_cfg, rebalance_on_event=roe).run(
        data, signals, start=START)
    return calc_metrics(eq["total_value"], trades, initial=1_000_000), trades


def build(name, ma):
    cfg = copy.deepcopy(base_cfg)
    if name == "momentum":
        if ma == "ens":
            return MomentumEnsemble(cfg), False
        cfg["strategy"]["trend_ma"] = ma
        return MomentumRotation(cfg), False
    if ma == "ens":
        return StaggeredEnsemble(cfg), True
    cfg["staggered"]["trend_ma"] = ma
    return StaggeredMomentum(cfg), True


print(f"{'策略':<10}{'trend_ma':<9}{'年化':>8}{'夏普':>6}{'回撤':>9}{'卡玛':>6}"
      f"{'卖出笔数':>8}{'相位中位年化':>12}{'相位极差':>9}")
rows = []
for name in ["momentum", "staggered"]:
    for ma in [40, 50, 60, "ens"]:
        strat, roe = build(name, ma)
        sig = strat.generate_signals(data)
        m, trades = run(sig, roe)
        ph = [run(sig, roe, phase=k)[0]["年化收益率"] for k in range(5)]
        label = str(ma) if ma != "ens" else "投票ens"
        med, spread = np.median(ph), max(ph) - min(ph)
        rows.append({"策略": name, "trend_ma": label, "年化": m["年化收益率"],
                     "夏普": m["夏普比率"], "回撤": m["最大回撤"], "卡玛": m["卡玛比率"],
                     "卖出": m["交易次数"], "相位中位": med, "相位极差": spread,
                     "相位年化": [round(p, 4) for p in ph]})
        print(f"{name:<10}{label:<9}{m['年化收益率']:>+8.2%}{m['夏普比率']:>6.2f}"
              f"{m['最大回撤']:>9.2%}{m['卡玛比率']:>6.2f}{m['交易次数']:>8}"
              f"{med:>+12.2%}{spread*100:>8.1f}pp", flush=True)

import json
json.dump(rows, open(f"{BASE}/research_v2/s5_results.json", "w"),
          ensure_ascii=False, indent=1, default=str)
print("saved research_v2/s5_results.json")
