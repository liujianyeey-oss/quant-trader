"""HKD 研究账户：未复权开盘成交、逐股手数、双边费用、资格及通道日历。

不接券商。结算可用资金、逐笔撮合与价格档位未建模，日线结果不能当成交保证。
"""
from decimal import Decimal, ROUND_CEILING, ROUND_HALF_UP

import numpy as np
import pandas as pd


class HKCosts:
    def __init__(self, config):
        self.config = config
        if any(not np.isfinite(float(v)) or float(v) < 0 for v in config.values()):
            raise ValueError("港股成本参数必须为非负有限数字")
        if not 0 <= config["slippage"] < 1:
            raise ValueError("slippage 必须小于 1")

    def fee(self, amount, stamp=True):
        if amount <= 0:
            return 0.0
        a = Decimal(str(amount))
        c = self.config
        def charge(rate):
            return (a * Decimal(str(rate))).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
        fee = max(charge(c["commission"]), Decimal(str(c["min_commission_hkd"])))
        for key in ["trading_fee", "sfc_levy", "afrc_levy", "settlement_fee", "extra_fee"]:
            fee += charge(c[key])
        if stamp:
            fee += (a * Decimal(str(c["stamp_duty"]))).quantize(Decimal("1"), rounding=ROUND_CEILING)
        return float(fee)


class HKBacktester:
    def __init__(self, cfg, universe, research=False, connect_calendar=None):
        h = cfg["hk_connect"]
        self.initial = float(h["initial_cash_hkd"])
        if not np.isfinite(self.initial) or self.initial <= 0:
            raise ValueError("initial_cash_hkd 必须为正数")
        self.costs = HKCosts(h["cost"])
        self.universe = universe
        self.research = research
        self.connect_calendar = connect_calendar

    def run(self, data, signals, start, end):
        if self.universe.mode == "snapshot" and not self.research:
            raise ValueError("当前名单不能证明历史资格；提供 intervals 名单或显式使用 --research")
        if self.connect_calendar is None and not self.research:
            raise ValueError("缺少港股通通道交易日历；提供 calendar_csv 或显式使用 --research")
        close = pd.DataFrame({c: d.set_index("date")["close"] for c, d in data.items()}).sort_index()
        opens = pd.DataFrame({c: d.set_index("date")["open"] for c, d in data.items()}).reindex(close.index)
        volume = pd.DataFrame({c: d.set_index("date")["volume"] for c, d in data.items()}).reindex(close.index)
        if close.empty or signals.empty:
            raise ValueError("行情或信号不足；至少需要完整预热窗口")
        if not set(signals.columns) <= set(close.columns) or not signals.index.isin(close.index).all() or signals.index.has_duplicates:
            raise ValueError("信号含未知股票、非行情日期或重复日期")
        if not np.isfinite(signals.to_numpy()).all() or (signals < 0).any().any() or (signals.sum(axis=1) > 1 + 1e-9).any():
            raise ValueError("权重必须有限、非负且合计不超过 1")
        targets = signals.reindex(columns=close.columns, fill_value=0).reindex(close.index).ffill().shift(1).fillna(0)
        marks = close.ffill()
        window = (close.index >= pd.Timestamp(start)) & (close.index <= pd.Timestamp(end))
        if self.connect_calendar is not None and not close.index[window].isin(self.connect_calendar.index).all():
            raise ValueError("港股通日历未覆盖回测中的全部港股行情日期；缺失不等于休市")
        # 只检测实际回测窗口，不能因未来公司行为影响更早的结果。
        for c, frame in data.items():
            d = frame.set_index("date").sort_index()
            factor_change = (d.signal_close / d.close).pct_change(fill_method=None).abs()
            if (factor_change.loc[start:end] > 0.10).any():
                raise ValueError(f"{c} 复权因子有大幅变化，可能拆并股/特别分派；尚未支持该公司行为，拒绝回测")
        cash, shares = self.initial, {c: 0 for c in close.columns}
        completed = pd.Series(0.0, index=close.columns)
        trades, rows, pending = [], [], False
        self.deferred_days = 0
        self.last_positions = {}
        for i, date in enumerate(close.index):
            if not window[i]:
                continue
            w = targets.iloc[i].copy()
            for c in w.index:
                if not self.universe.eligible(c, date, self.research):
                    w[c] = 0  # 执行日再次检查资格，不因昨日信号允许今日新买
            changed = not np.allclose(w, completed, rtol=0, atol=1e-10)
            is_open = self.connect_calendar is None or bool(self.connect_calendar.loc[date])
            if changed or pending:
                needed = [c for c in shares if shares[c] or w[c] > 0]
                available = all(pd.notna(opens.iloc[i][c]) and opens.iloc[i][c] > 0 and volume.iloc[i][c] > 0 for c in needed)
                if is_open and available:
                    cash = self._rebalance(date, w, opens.iloc[i], cash, shares, trades)
                    completed, pending = w.copy(), False
                else:
                    pending = True
                    self.deferred_days += 1
            if cash < -1e-7:
                raise AssertionError(f"现金透支 {date}: {cash}")
            holding_value = sum(n * marks.iloc[i][c] for c, n in shares.items() if n)
            rows.append({"date": date, "total_value": cash + holding_value,
                         "cash_hkd": cash, "gross_weight": holding_value / (cash + holding_value),
                         "pending": pending})
        if not rows:
            raise ValueError("回测区间没有行情")
        self.last_positions = shares.copy()
        self.pending = pending
        return pd.DataFrame(rows).set_index("date"), trades

    def _rebalance(self, date, weights, prices, cash, shares, trades):
        slip = self.costs.config["slippage"]
        nav = cash + sum(n * prices[c] for c, n in shares.items() if n)
        desired = {}
        for c in shares:
            lot = self.universe.lot(c, date, self.research) if shares[c] or weights[c] else 1
            if shares[c] % lot:
                raise ValueError(f"{c} 每手股数变化产生碎股，需公司行为/碎股执行模型")
            desired[c] = int(weights[c] * nav / (prices[c] * (1 + slip)) / lot) * lot if weights[c] else 0

        def fee(c, amount):
            info = self.universe.info(c, date, self.research)
            if info is None:
                past = [r for r in self.universe.records[c] if r["begin"] <= pd.Timestamp(date)]
                info = past[-1] if past else {}
            return self.costs.fee(amount, stamp=info.get("stamp_duty", True))

        for c in sorted(shares):
            n = shares[c] - desired[c]
            if n > 0:
                p = prices[c] * (1 - slip)
                amount, cost = n * p, fee(c, n * p)
                if amount + cash < cost:
                    raise ValueError(f"{c} 卖出款不足支付最低收费")
                cash += amount - cost
                shares[c] -= n
                trades.append(dict(date=date, code=c, side="sell", price=p, shares=n, fee_hkd=cost,
                                   amount_hkd=amount, pnl=None, win=None))
        # 权重越高越先买；现金不足只减整数手，不透支。80% 总仓位预留费用空间。
        for c in weights.sort_index().sort_values(ascending=False, kind="stable").index:
            n = desired[c] - shares[c]
            if n <= 0:
                continue
            lot = self.universe.lot(c, date, self.research)
            p = prices[c] * (1 + slip)
            n = min(n, int(cash / p / lot) * lot)
            while n > 0 and n * p + fee(c, n * p) > cash + 1e-9:
                n -= lot
            if n:
                amount, cost = n * p, fee(c, n * p)
                cash -= amount + cost
                shares[c] += n
                trades.append(dict(date=date, code=c, side="buy", price=p, shares=n, fee_hkd=cost,
                                   amount_hkd=amount, pnl=None, win=None))
        return cash
