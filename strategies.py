"""策略模块：策略基类 + 动量轮动 / 双均线 / 五组分批动量（staggered）。

策略只负责输出目标权重（target weights），执行（T+1 开盘价成交）由回测引擎负责。
约定：generate_signals 返回 DataFrame，index=信号日期，columns=标的代码，
值=目标权重(0~1)，每行和<=1；只在信号日有行，引擎前向填充并次日执行。
signal 子命令的"当前目标"必须与回测来自同一份信号状态（前向填充后的生效目标）。
"""
from abc import ABC, abstractmethod

import numpy as np
import pandas as pd

WEEKDAYS_CN = ["周一", "周二", "周三", "周四", "周五", "周六", "周日"]


class BaseStrategy(ABC):
    """策略基类。"""

    name = "base"
    #: True 时引擎在每个信号事件次日重新校准权重（目标未变也调仓）；
    #: False 时目标不变则不重调
    rebalance_on_event = False

    def __init__(self, cfg):
        self.cfg = cfg

    @abstractmethod
    def generate_signals(self, data: dict) -> pd.DataFrame:
        """生成目标权重信号。

        data: {code: DataFrame(date, open, close, ...)}，可含预热期数据。
        """

    @staticmethod
    def _close_panel(data):
        closes = {c: df.set_index("date")["close"] for c, df in data.items()}
        return pd.DataFrame(closes).sort_index()

    def current_target(self, data: dict):
        """当前生效目标（与回测同口径：信号前向填充至最近一个交易日）。

        返回 (target: pd.Series, signals: DataFrame)。
        """
        signals = self.generate_signals(data)
        if signals.empty:
            return pd.Series(dtype=float), signals
        last_date = max(df["date"].max() for df in data.values())
        idx = signals.index.union(pd.DatetimeIndex([last_date]))
        eff = signals.reindex(idx).ffill().loc[:last_date]
        if eff.empty:
            return pd.Series(dtype=float), signals
        return eff.iloc[-1], signals

    @abstractmethod
    def next_update_desc(self, data: dict) -> str:
        """下次信号更新的说明（预计日期等）。"""

    def describe_signal(self, data: dict) -> str:
        """当前应持仓（= 回测引擎当前生效目标）+ 触发原因 + 下次更新日期。"""
        target, _ = self.current_target(data)
        hold = {c: w for c, w in target.items() if w > 0}
        pos = "、".join(f"{c}(目标仓位{w:.1%})" for c, w in hold.items()) if hold else "空仓"
        return f"当前生效目标：{pos}\n{self.next_update_desc(data)}"


class MomentumRotation(BaseStrategy):
    """动量轮动：每 rebalance_days 个交易日计算 momentum_days 日动量，
    持有动量最强的进攻池标的；若其收盘价 < trend_ma 日均线（趋势过滤，
    trend_ma=0 关闭）则切到防守资产；防守资产自身不加趋势过滤，
    无防守资产数据时则空仓。
    """

    name = "momentum"

    def __init__(self, cfg):
        super().__init__(cfg)
        s = cfg["strategy"]
        self.momentum_days = s["momentum_days"]
        self.rebalance_days = s["rebalance_days"]
        self.trend_ma = s["trend_ma"]
        self.defensive = cfg["universe"]["defensive"]
        # 进攻池显式限定为配置项，不用"除防守外所有列"
        self.attack_pool = list(cfg["universe"]["etfs"]) + list(cfg["universe"]["stocks"])

    def generate_signals(self, data: dict) -> pd.DataFrame:
        panel = self._close_panel(data)
        if panel.empty:
            return pd.DataFrame()

        momentum = panel / panel.shift(self.momentum_days) - 1.0
        # trend_ma=0 表示关闭趋势过滤
        trend_line = panel.rolling(self.trend_ma).mean() if self.trend_ma > 0 else None

        codes = [c for c in self.attack_pool if c in panel.columns]  # 显式进攻池
        has_defensive = self.defensive in panel.columns

        rows = []
        # 只在可调仓日出信号；需要足够的历史数据
        for i in range(self.momentum_days + 1, len(panel)):
            if (i - self.momentum_days - 1) % self.rebalance_days != 0:
                continue
            date = panel.index[i]
            mom = momentum.iloc[i][codes].dropna()
            if mom.empty:
                continue
            best = mom.idxmax()
            # 趋势过滤：最强标的收盘价需在均线之上（trend_ma=0 时不过滤）
            trend_ok = (trend_line is None
                        or (pd.notna(trend_line.iloc[i][best])
                            and panel.iloc[i][best] >= trend_line.iloc[i][best]))
            if trend_ok:
                rows.append((date, best))
            elif has_defensive:
                rows.append((date, self.defensive))
            else:
                rows.append((date, None))  # 空仓

        all_cols = list(panel.columns)
        signals = pd.DataFrame(0.0, index=[d for d, _ in rows], columns=all_cols)
        for date, code in rows:
            if code is not None:
                signals.loc[date, code] = 1.0
        return signals

    def next_update_desc(self, data: dict) -> str:
        panel = self._close_panel(data)
        last_i = len(panel) - 1
        rem = (last_i - self.momentum_days - 1) % self.rebalance_days
        offset = (self.rebalance_days - rem) % self.rebalance_days or self.rebalance_days
        est = panel.index[-1] + pd.offsets.BDay(offset)
        return (f"下次调仓：{offset} 个交易日后（预计 {est:%Y-%m-%d} 前后，"
                f"遇节假日顺延）；调仓周期 {self.rebalance_days} 个交易日")

    def describe_signal(self, data: dict) -> str:
        panel = self._close_panel(data)
        if len(panel) < self.momentum_days + 2:
            return "数据不足，无法计算动量信号"
        momentum = panel / panel.shift(self.momentum_days) - 1.0
        trend_line = panel.rolling(self.trend_ma).mean() if self.trend_ma > 0 else None
        codes = [c for c in self.attack_pool if c in panel.columns]
        mom = momentum.iloc[-1][codes].dropna()
        lines = []
        if not mom.empty:
            mom_str = "；".join(f"{c} {mom[c]:+.2%}" for c in mom.sort_values(ascending=False).index)
            lines.append(f"{self.momentum_days}日动量排名：{mom_str}")
            best = mom.idxmax()
            if trend_line is not None:
                ma = trend_line.iloc[-1][best]
                close = panel.iloc[-1][best]
                lines.append(f"趋势过滤：{best} 收盘价 {close:.3f} "
                             f"{'≥' if pd.notna(ma) and close >= ma else '<'} "
                             f"MA{self.trend_ma} {ma:.3f}")
        # 当前目标必须与回测一致（信号前向填充，而非当日重选）
        lines.append(super().describe_signal(data))
        return "\n".join(lines)


class DualMA(BaseStrategy):
    """双均线（对照策略）：MA5 上穿 MA20（金叉）买入，下穿（死叉）卖出。单标的。"""

    name = "dual_ma"

    def __init__(self, cfg, code=None):
        super().__init__(cfg)
        s = cfg["strategy"]
        self.fast = s["ma_fast"]
        self.slow = s["ma_slow"]
        self.code = code or cfg["universe"]["etfs"][0]

    def generate_signals(self, data: dict) -> pd.DataFrame:
        if self.code not in data:
            raise ValueError(f"双均线策略标的 {self.code} 无数据")
        df = data[self.code].set_index("date").sort_index()
        ma_fast = df["close"].rolling(self.fast).mean()
        ma_slow = df["close"].rolling(self.slow).mean()
        above = ma_fast > ma_slow
        cross = above != above.shift(1)  # 状态发生变化即为交叉
        dates = df.index[cross.fillna(False) & above.notna() & ma_slow.notna()]
        signals = pd.DataFrame(0.0, index=dates, columns=[self.code])
        for d in dates:
            signals.loc[d, self.code] = 1.0 if above.loc[d] else 0.0
        return signals

    def next_update_desc(self, data: dict) -> str:
        return "下次更新：下一交易日收盘后（双均线每日检查）"

    def describe_signal(self, data: dict) -> str:
        df = data[self.code].set_index("date").sort_index()
        ma_fast = df["close"].rolling(self.fast).mean()
        ma_slow = df["close"].rolling(self.slow).mean()
        f, s = ma_fast.iloc[-1], ma_slow.iloc[-1]
        if pd.isna(f) or pd.isna(s):
            return "数据不足，无法计算均线"
        state = "金叉" if f > s else "死叉"
        return (f"{self.code}：MA{self.fast}={f:.3f}，MA{self.slow}={s:.3f}，"
                f"收盘价={df['close'].iloc[-1]:.3f}（{state}）\n"
                + super().describe_signal(data))


class StaggeredMomentum(BaseStrategy):
    """五组分批动量 + 波动率控仓（staggered）。

    规则（与审计报告"推荐候选的明确规则"一致）：
    1. 进攻池、momentum_days 日动量、trend_ma 趋势过滤、防守资产与 momentum 相同；
    2. n_groups=5 个等权信号组（各占组合 20% 预算），分别在周一~周五收盘更新；
       对应星期是非交易日则该组本周跳过；
    3. 每组更新时选动量最强的进攻池标的；收盘价 >= 趋势均线选它，否则选防守资产
       （防守资产自身不加趋势过滤——有意保留）；
    4. 组内投资比例 = min(1, vol_budget / max(sigma, 0.03))，
       sigma = 被选资产最近 vol_window 个日收益 std × √252，其余为现金；
    5. 每日把 5 组最后目标按 1/5 平均、按代码合并（同代码净额合并），
       T+1 开盘成交，不使用杠杆（每组权重<=1，平均后总和<=1）。
    引擎侧按 rebalance_on_event=True 在每个信号事件次日重新校准权重。
    """

    name = "staggered"
    rebalance_on_event = True

    def __init__(self, cfg):
        super().__init__(cfg)
        s = cfg.get("staggered", {})
        self.momentum_days = s.get("momentum_days", 20)
        self.trend_ma = s.get("trend_ma", 60)
        self.vol_window = s.get("vol_window", 60)
        self.vol_budget = s.get("vol_budget", 0.12)
        self.n_groups = s.get("n_groups", 5)
        self.defensive = cfg["universe"]["defensive"]
        self.attack_pool = list(cfg["universe"]["etfs"]) + list(cfg["universe"]["stocks"])
        # 信号所需预热 K 线条数（与审计参考实现一致，下限 120）
        self.warmup = max(120, self.momentum_days, self.trend_ma, self.vol_window)

    def _cohort_signals(self, panel, weekday):
        """单个 weekday 组的原始目标（未除以组数）。"""
        rets = panel.pct_change()
        vol = rets.rolling(self.vol_window, min_periods=self.vol_window).std() * np.sqrt(252)
        mom = panel / panel.shift(self.momentum_days) - 1.0
        ma = panel.rolling(self.trend_ma, min_periods=self.trend_ma).mean()
        codes = [c for c in self.attack_pool if c in panel.columns]
        has_defensive = bool(self.defensive) and self.defensive in panel.columns

        rows = []
        for i in range(self.warmup, len(panel)):
            date = panel.index[i]
            if date.weekday() != weekday:
                continue
            m = mom.iloc[i][codes].dropna()
            if m.empty:
                continue
            winner = m.idxmax()
            # 趋势过滤：NaN 视为不通过（与审计参考实现一致）
            if not (panel.iloc[i][winner] >= ma.iloc[i][winner]):
                winner = self.defensive if has_defensive else None
            row = pd.Series(0.0, index=panel.columns)
            if winner is not None:
                v = vol.iloc[i][winner]
                if pd.notna(v):
                    row[winner] = min(1.0, self.vol_budget / max(0.03, v))
            rows.append(row.rename(date))
        return pd.DataFrame(rows)

    def generate_signals(self, data: dict) -> pd.DataFrame:
        panel = self._close_panel(data)
        if panel.empty or len(panel) <= self.warmup:
            return pd.DataFrame(columns=panel.columns)
        cohorts = [self._cohort_signals(panel, d) for d in range(self.n_groups)]
        cohorts = [c for c in cohorts if not c.empty]
        if not cohorts:
            return pd.DataFrame(columns=panel.columns)
        # 合并：各组前向填充后按 1/n 平均（同代码自然加总=净额合并）
        dates = cohorts[0].index
        for c in cohorts[1:]:
            dates = dates.union(c.index)
        merged = sum(c.reindex(dates).ffill().fillna(0.0) for c in cohorts) / self.n_groups
        return merged

    def next_update_desc(self, data: dict) -> str:
        panel = self._close_panel(data)
        nxt = panel.index[-1] + pd.offsets.BDay(1)
        return (f"下次更新：下一交易日（预计 {nxt:%Y-%m-%d}，{WEEKDAYS_CN[nxt.weekday()]}组）；"
                f"5 个组分别在周一~周五收盘更新")

    def describe_signal(self, data: dict) -> str:
        panel = self._close_panel(data)
        if len(panel) <= self.warmup:
            return "数据不足，无法计算分组动量信号"
        lines = ["五组状态（组内目标权重，组合占比=组内权重×20%）："]
        rets = panel.pct_change()
        vol = rets.rolling(self.vol_window, min_periods=self.vol_window).std() * np.sqrt(252)
        for d in range(self.n_groups):
            cohort = self._cohort_signals(panel, d)
            if cohort.empty:
                continue
            last_date = cohort.index[-1]
            last = cohort.iloc[-1]
            hold = {c: w for c, w in last.items() if w > 0}
            if hold:
                code, w = next(iter(hold.items()))
                v = vol.loc[last_date, code]
                lines.append(f"  {WEEKDAYS_CN[d]}组（最近更新 {last_date:%m-%d}）："
                             f"{code} 组内权重 {w:.0%}（σ={v:.0%}）")
            else:
                lines.append(f"  {WEEKDAYS_CN[d]}组（最近更新 {last_date:%m-%d}）：空仓/现金")
        lines.append(super().describe_signal(data))
        return "\n".join(lines)


STRATEGIES = {"momentum": MomentumRotation, "dual_ma": DualMA, "staggered": StaggeredMomentum}
