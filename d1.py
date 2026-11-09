"""D1 防守池轮动（策略 v2，docs/DESIGN_strategy_v2_20261108.md，S2 影子运行阶段）。

规则（已评审口径）：
- 核心仓（防守仓位的 80%）：仅允许 60 日年化波动 <3% 的候选（黄金被规则自动排除）；
  池内按 20 日动量排名持有第 1 名；迟滞——排名差 <1pp 或持有 <5 个交易日不切换
  （即切换须同时满足：新第 1 名动量超出现任 ≥1pp 且 现任持有 ≥5 个交易日）；
  现任失去资格（波动超标/无数据）时强制切换；兜底 511880。
- 卫星仓（≤20%）：黄金收盘 > MA60 且 20 日动量 > 3% 时持有 518880，
  否则该 20% 并入核心第 1 名。
- 切换周频上限：选择器只在策略信号日推进状态（momentum 每 5 交易日 /
  staggered 各组每周），天然满足每周至多一次。

实现说明：MomentumD1 / StaggeredD1 以子类方式复用线上策略，防守分支替换为池选择，
其余逻辑（动量/均线/网格/分组/波动控仓/预热期）逐行复刻——等价性由
tests/test_d1.py 的 forced-static 对照测试保证（defensive_override 强制单资产时
输出必须与父类完全一致）。线上 strategies.py 不改动。
"""
import numpy as np
import pandas as pd

from strategies import MomentumRotation, StaggeredMomentum

#: 核心池（511880 兜底 / 511360 短融 / 511010 5年国债 / 511260 10年国债）
CORE_POOL = ["511880", "511360", "511010", "511260"]
GOLD = "518880"
FALLBACK = "511880"
#: 影子运行需要额外加载的标的（核心池；黄金本就在 universe.defensive 中）
D1_EXTRA_CODES = [c for c in CORE_POOL if c != FALLBACK] + [FALLBACK]

DEFAULTS = {"mom_win": 20, "vol_win": 60, "vol_cap": 0.03,
            "gap": 0.01, "min_hold": 5, "sat_mom": 0.03, "sat_weight": 0.2}


class DefensivePoolSelector:
    """D1 防守池选择器。在信号日按面板行号顺序推进（状态机，含迟滞）。

    每次 generate_signals 都新建实例并从头推进，输出是数据的纯函数（可复现）。
    """

    def __init__(self, panel: pd.DataFrame, **kw):
        p = {**DEFAULTS, **kw}
        self.mom = panel / panel.shift(p["mom_win"]) - 1.0
        self.vol = panel.pct_change().rolling(p["vol_win"], min_periods=p["vol_win"]).std() \
            * np.sqrt(252)
        self.gold_ma = panel[GOLD].rolling(60).mean() if GOLD in panel.columns else None
        self.close = panel
        self.vol_cap = p["vol_cap"]
        self.gap = p["gap"]
        self.min_hold = p["min_hold"]
        self.sat_mom = p["sat_mom"]
        self.sat_weight = p["sat_weight"]
        self.current = FALLBACK
        self.last_switch = -10 ** 9
        self.n_switch = 0

    def _eligible(self, i):
        """60 日年化波动 <3% 且动量可算的池内候选（无数据自然出局，
        如 511360 上市前）。"""
        moms, vols = self.mom.iloc[i], self.vol.iloc[i]
        return [c for c in CORE_POOL if c in self.close.columns
                and pd.notna(vols.get(c)) and pd.notna(moms.get(c))
                and vols[c] < self.vol_cap]

    def satellite_on(self, i) -> bool:
        """黄金卫星条件：收盘 > MA60 且 20 日动量 > 3%。"""
        if self.gold_ma is None:
            return False
        g_close, g_ma = self.close.iloc[i][GOLD], self.gold_ma.iloc[i]
        g_mom = self.mom.iloc[i].get(GOLD)
        return bool(pd.notna(g_ma) and pd.notna(g_mom)
                    and g_close > g_ma and g_mom > self.sat_mom)

    def step(self, i) -> dict:
        """在第 i 行（信号日收盘）评估，返回 {code: weight}，合计 1.0。"""
        moms = self.mom.iloc[i]
        elig = self._eligible(i)
        if elig:
            best = max(elig, key=lambda c: moms[c])
            cur = self.current
            if cur not in elig:
                self.current, self.last_switch = best, i
                self.n_switch += 1
            elif (best != cur and moms[best] - moms[cur] >= self.gap
                    and i - self.last_switch >= self.min_hold):
                self.current, self.last_switch = best, i
                self.n_switch += 1
        core = self.current if elig else FALLBACK
        if self.satellite_on(i):
            return {GOLD: self.sat_weight, core: 1.0 - self.sat_weight}
        return {core: 1.0}


class MomentumD1(MomentumRotation):
    """momentum + D1 防守池：趋势过滤失败时按池轮动分配防守仓位。

    defensive_override：测试/对照用——设定后防守分支退回单资产死守，
    输出与父类 MomentumRotation 完全一致（等价性锚点）。
    """

    name = "momentum_d1"

    def __init__(self, cfg, defensive_override=None, selector_kw=None):
        super().__init__(cfg)
        self.defensive_override = defensive_override
        self.selector_kw = selector_kw or {}

    def _defense_row(self, selector, i):
        if self.defensive_override is not None:
            return {self.defensive_override: 1.0}
        return selector.step(i)

    def generate_signals(self, data: dict) -> pd.DataFrame:
        panel = self._close_panel(data)
        if panel.empty:
            return pd.DataFrame()
        momentum = panel / panel.shift(self.momentum_days) - 1.0
        trend_line = panel.rolling(self.trend_ma).mean() if self.trend_ma > 0 else None
        codes = [c for c in self.attack_pool if c in panel.columns]
        selector = DefensivePoolSelector(panel, **self.selector_kw)

        rows = []
        for i in range(self.momentum_days + 1, len(panel)):
            if (i - self.momentum_days - 1) % self.rebalance_days != 0:
                continue
            date = panel.index[i]
            mom = momentum.iloc[i][codes].dropna()
            if mom.empty:
                continue
            best = mom.idxmax()
            trend_ok = (trend_line is None
                        or (pd.notna(trend_line.iloc[i][best])
                            and panel.iloc[i][best] >= trend_line.iloc[i][best]))
            if trend_ok:
                rows.append((date, {best: 1.0}))
            else:
                rows.append((date, self._defense_row(selector, i)))
        signals = pd.DataFrame(0.0, index=[d for d, _ in rows], columns=list(panel.columns))
        for date, w in rows:
            for c, wt in w.items():
                signals.loc[date, c] = wt
        return signals


class StaggeredD1(StaggeredMomentum):
    """staggered + D1 防守池：组内趋势失败时按池轮动分配，组内投资比例按
    防守组合的加权波动折算（与父类 min(1, vol_budget/max(σ,0.03)) 同口径）。"""

    name = "staggered_d1"

    def __init__(self, cfg, defensive_override=None, selector_kw=None):
        super().__init__(cfg)
        self.defensive_override = defensive_override
        self.selector_kw = selector_kw or {}

    def _cohort_signals(self, panel, weekday):
        rets = panel.pct_change()
        vol = rets.rolling(self.vol_window, min_periods=self.vol_window).std() * np.sqrt(252)
        mom = panel / panel.shift(self.momentum_days) - 1.0
        ma = panel.rolling(self.trend_ma, min_periods=self.trend_ma).mean()
        codes = [c for c in self.attack_pool if c in panel.columns]
        selector = DefensivePoolSelector(panel, **self.selector_kw)

        rows = []
        for i in range(self.warmup, len(panel)):
            date = panel.index[i]
            if date.weekday() != weekday:
                continue
            m = mom.iloc[i][codes].dropna()
            if m.empty:
                continue
            winner = m.idxmax()
            row = pd.Series(0.0, index=panel.columns)
            if panel.iloc[i][winner] >= ma.iloc[i][winner]:
                v = vol.iloc[i][winner]
                if pd.notna(v):
                    row[winner] = min(1.0, self.vol_budget / max(0.03, v))
            else:
                if self.defensive_override is not None:
                    v = vol.iloc[i][self.defensive_override]
                    if pd.notna(v):
                        row[self.defensive_override] = min(1.0, self.vol_budget / max(0.03, v))
                else:
                    wts = selector.step(i)
                    sig_blend = sum(w * vol.iloc[i].get(c, np.nan) for c, w in wts.items())
                    if pd.notna(sig_blend):
                        w_total = min(1.0, self.vol_budget / max(0.03, sig_blend))
                        for c, wt in wts.items():
                            row[c] = w_total * wt
            rows.append(row.rename(date))
        return pd.DataFrame(rows)
