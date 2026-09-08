"""回测引擎（审计修正版）。

执行口径（与独立审计 strategy_research.py 一致）：
- T+1：目标权重先合并到完整交易日历、前向填充、后移一日，再裁剪回测区间；
  区间开始前最后的已知目标会被承接（不会在窗口起点丢掉持仓意图）。
- 开盘调仓用**开盘价**估值组合并确定目标股数，不使用当天收盘价（无前视）；
  开盘价缺失时不再用收盘价替代。
- 参与调仓的已有/目标持仓任一缺开盘价 → 整次调仓延期到下一交易日重试
  （pending 状态，直到成交；目标未成交不更新已完成状态）。
- 先卖后买；现金不足按比例缩减买入；最低佣金不会导致现金透支。
- 估值用收盘价前向填充（个别标的单日缺行情不至于持仓归零）。
- 支持 ETF/个股差异化费率、滑点、100 股整数倍。
"""
import numpy as np
import pandas as pd

LOT = 100  # A股最小交易单位


class Backtester:
    def __init__(self, cfg, stock_codes=None, rebalance_on_event=False):
        c = cfg["cost"]
        self.etf_commission = c["etf_commission"]
        self.stock_commission = c["stock_commission"]
        self.stamp_tax = c["stamp_tax"]
        self.slippage = c["slippage"]
        self.min_commission = c.get("min_commission", 0) or 0
        self.initial_cash = cfg["backtest"]["initial_cash"]
        self.stock_codes = set(stock_codes or [])
        # True 时在每个信号事件的次日重新校准权重（部分仓位策略用）；
        # False 时目标不变则不重调（momentum 等满仓切换策略用）
        self.rebalance_on_event = rebalance_on_event

    def _fee_rate(self, code):
        return self.stock_commission if code in self.stock_codes else self.etf_commission

    def _commission(self, code, amount):
        fee = amount * self._fee_rate(code)
        return max(fee, self.min_commission) if self.min_commission > 0 else fee

    def run(self, data: dict, signals: pd.DataFrame, start=None, end=None):
        """执行回测。

        data: {code: DataFrame(date, open, close, ...)}，可含回测区间前的预热数据。
        signals: 策略输出的目标权重（只在信号日有行），index=date。
        start/end: 回测区间（None = 数据范围）；区间外的日期只用于信号承接。
        返回 (equity_df, trades)。
        """
        closes = {c: df.set_index("date")["close"] for c, df in data.items()}
        opens = {c: df.set_index("date")["open"] for c, df in data.items()}
        close_panel = pd.DataFrame(closes).sort_index()
        open_panel = pd.DataFrame(opens).sort_index().reindex(close_panel.index)
        calendar = close_panel.index
        marks = close_panel.ffill()  # 估值价

        # ---- 信号对齐：并入完整日历 → 前向填充 → 后移一日(T+1) → 裁剪 ----
        s = signals.reindex(columns=close_panel.columns, fill_value=0.0)
        aligned = (s.reindex(calendar.union(s.index)).sort_index().ffill()
                   .reindex(calendar).fillna(0.0))
        targets = aligned.shift(1).fillna(0.0)
        # 信号事件的次一交易日（rebalance_on_event 时用于重新校准）
        events = pd.Series(calendar.isin(s.index), index=calendar).shift(1, fill_value=False)

        start = pd.Timestamp(start) if start else calendar[0]
        end = pd.Timestamp(end) if end else calendar[-1]

        cash = self.initial_cash
        shares = {c: 0 for c in close_panel.columns}
        cost_basis = {}          # code -> 平均买入成本（净价，用于胜率统计）
        trades = []
        equity_rows = []
        completed = pd.Series(0.0, index=close_panel.columns)  # 已成交目标
        pending = False          # 有待执行目标（上次调仓被延期）

        for i, date in enumerate(calendar):
            if date < start or date > end:
                continue
            w = targets.iloc[i]
            changed = not np.allclose(w.to_numpy(), completed.to_numpy(), atol=1e-10, rtol=0)
            if changed or pending or (self.rebalance_on_event and events.iloc[i]):
                open_row = open_panel.iloc[i]
                # 参与调仓的已有/目标持仓都必须有开盘价，否则整次延期
                needed = [c for c in shares if shares[c] > 0 or w.get(c, 0) > 0]
                if all(pd.notna(open_row[c]) and open_row[c] > 0 for c in needed):
                    cash = self._rebalance(date, w, open_row, cash, shares, cost_basis, trades)
                    completed = w.copy()
                    pending = False
                else:
                    pending = True
            assert cash >= -1e-6, f"现金透支: {cash} @ {date}"

            # ---- 收盘估值（前向填充价） ----
            value = cash
            for c, n in shares.items():
                price = marks.iloc[i][c]
                if n > 0 and pd.notna(price):
                    value += n * price
            equity_rows.append((date, value))

        equity = pd.DataFrame(equity_rows, columns=["date", "total_value"]).set_index("date")
        return equity, trades

    def _rebalance(self, date, tgt, open_row, cash, shares, cost_basis, trades):
        """在 date 的开盘价上把组合调整到目标权重 tgt。返回剩余现金。"""
        slip = self.slippage

        # 用开盘价估值（非当天收盘价，无前视）
        nav_open = cash + sum(n * open_row[c] for c, n in shares.items() if n > 0)

        # 目标股数（100 股整数倍，按开盘买入价折算）
        desired = {}
        for c in shares:
            w = tgt.get(c, 0.0)
            if w <= 0:
                desired[c] = 0
            else:
                buy_price = open_row[c] * (1 + slip)
                desired[c] = int(w * nav_open / buy_price / LOT) * LOT

        # ---- 先卖 ----
        for c in shares:
            delta = desired[c] - shares[c]
            if delta < 0:
                p = open_row[c] * (1 - slip)
                amount = -delta * p
                fee = self._commission(c, amount)
                tax = amount * self.stamp_tax if c in self.stock_codes else 0.0
                cash += amount - fee - tax
                pnl = (p - cost_basis.get(c, p)) * (-delta) - fee - tax
                trades.append({"code": c, "date": date, "side": "sell",
                               "price": p, "shares": -delta, "pnl": pnl,
                               "win": pnl > 0})
                shares[c] += delta
                if shares[c] == 0:
                    cost_basis.pop(c, None)

        # ---- 后买：现金不足按比例缩减，且最低佣金不得透支 ----
        buys = {c: desired[c] - shares[c] for c in shares if desired[c] - shares[c] > 0}
        if buys:
            def total_cost(quantities):
                return sum(n * open_row[c] * (1 + slip) + self._commission(c, n * open_row[c] * (1 + slip))
                           for c, n in quantities.items() if n > 0)
            cost = total_cost(buys)
            if cost > cash:
                ratio = max(0.0, cash / cost)
                buys = {c: int(n * ratio / LOT) * LOT for c, n in buys.items()}
            for c, n in buys.items():
                price = open_row[c] * (1 + slip)
                while n > 0 and n * price + self._commission(c, n * price) > cash + 1e-8:
                    n -= LOT
                if n <= 0:
                    continue
                amount = n * price
                fee = self._commission(c, amount)
                old_n = shares[c]
                cost_basis[c] = (cost_basis.get(c, 0.0) * old_n + price * n) / (old_n + n)
                shares[c] += n
                cash -= amount + fee
                trades.append({"code": c, "date": date, "side": "buy",
                               "price": price, "shares": n, "pnl": 0.0, "win": None})
        return cash
