"""固定规则的港股通中期趋势策略，不进行样本内参数寻优。"""
import numpy as np
import pandas as pd


class HKConnectMomentum:
    name = "hk_connect"

    def __init__(self, cfg, universe, research=False):
        self.cfg = cfg
        self.universe = universe
        self.research = research
        self.s = cfg["hk_connect"]["strategy"]
        for key in ["momentum_fast", "momentum_slow", "trend_window", "vol_window", "liquidity_window", "top_n"]:
            if not isinstance(self.s[key], int) or self.s[key] < 2:
                raise ValueError(f"{key} 必须为至少 2 的整数")
        if not 0 <= self.s["rebalance_weekday"] <= 4:
            raise ValueError("rebalance_weekday 必须为 0–4")
        if not 0 < self.s["max_weight"] <= self.s["max_gross"] <= 1:
            raise ValueError("仓位应满足 0 < 单股上限 <= 总仓位上限 <= 1")
        if self.s["vol_target"] <= 0 or self.s["min_turnover_hkd"] < 0 or self.s["min_price_hkd"] <= 0:
            raise ValueError("波动预算、最低价格、流动性阈值无效")
        if not 1 <= self.s["min_positions"] <= self.s["top_n"]:
            raise ValueError("min_positions 必须在 1 到 top_n 之间")
        self.capital = float(cfg["hk_connect"]["initial_cash_hkd"])
        if not np.isfinite(self.capital) or self.capital <= 0:
            raise ValueError("initial_cash_hkd 必须为正数")

    @property
    def warmup(self):
        return max(self.s[k] for k in ["momentum_fast", "momentum_slow", "trend_window", "vol_window", "liquidity_window"])

    def generate_signals(self, data):
        if not data:
            raise ValueError("港股行情为空")
        close = pd.DataFrame({c: d.set_index("date")["signal_close"] for c, d in data.items()}).sort_index()
        raw = pd.DataFrame({c: d.set_index("date")["close"] for c, d in data.items()}).reindex(close.index)
        turnover = pd.DataFrame({c: d.set_index("date")["turnover_hkd"] for c, d in data.items()}).reindex(close.index)
        s = self.s
        fast, slow = close / close.shift(s["momentum_fast"]) - 1, close / close.shift(s["momentum_slow"]) - 1
        trend = close.rolling(s["trend_window"]).mean()
        returns = close.pct_change(fill_method=None)
        vol = returns.rolling(s["vol_window"]).std() * np.sqrt(252)
        liquidity = turnover.rolling(s["liquidity_window"]).mean()
        targets = pd.Series(0.0, index=close.columns)
        rows, week = [], None
        for i, date in enumerate(close.index):
            # 日期锚定：本周指定星期或之后首个有行情的交易日；节假日无需猜下周一。
            this_week = date.isocalendar()[:2]
            update = date.weekday() >= s["rebalance_weekday"] and this_week != week
            if update:
                week = this_week
            if i < self.warmup:
                continue
            eligible = pd.Series({c: self.universe.eligible(c, date, self.research) for c in close.columns})
            # 资格退出、趋势破坏每天清零；重新买入仅在周度事件。
            healthy = eligible & close.iloc[i].notna() & (close.iloc[i] > trend.iloc[i])
            targets = targets.where(healthy, 0.0)
            if update:
                candidates = healthy & (fast.iloc[i] > 0) & (slow.iloc[i] > 0)
                candidates &= (raw.iloc[i] >= s["min_price_hkd"]) & (liquidity.iloc[i] >= s["min_turnover_hkd"])
                candidates &= vol.iloc[i].notna() & (vol.iloc[i] > 0)
                score = (0.5 * fast.iloc[i] + 0.5 * slow.iloc[i]).where(candidates).dropna()
                # 代码先排序使同分确定性，不依赖数据字典顺序。
                ranked = list(score.sort_index().sort_values(ascending=False, kind="stable").index)
                targets[:] = 0
                # 固定研究预算筛选可负担手数，执行仍按当时账户净值取整。
                # 先剔除超出单股上限的一手，再按最终波动仓位检查并补入下一名。
                lot_cost = {c: raw.iloc[i][c] * self.universe.lot(c, date, self.research)
                            * (1 + self.cfg["hk_connect"]["cost"]["slippage"]) for c in ranked}
                ranked = [c for c in ranked if lot_cost[c] <= self.capital * s["max_weight"]]
                while ranked:
                    winners = ranked[:s["top_n"]]
                    if len(winners) < s["min_positions"]:
                        break
                    inverse = 1 / vol.iloc[i][winners].clip(lower=0.10)
                    weights = (inverse / inverse.sum() * s["max_gross"]).clip(upper=s["max_weight"])
                    # 用加权单股波动之和预算，相当于不依赖分散化相关性优惠。
                    risk = float((weights * vol.iloc[i][winners]).sum())
                    weights *= min(1.0, s["vol_target"] / max(risk, 1e-12))
                    too_large = {c for c in winners if lot_cost[c] > self.capital * weights[c]}
                    if too_large:
                        ranked = [c for c in ranked if c not in too_large]
                        continue
                    targets.loc[winners] = weights
                    break
            # 每日目标方便停牌/资格退出管理；执行器只在目标变化时交易，避免每日再平衡。
            rows.append(targets.copy().rename(date))
        return pd.DataFrame(rows, columns=close.columns)
