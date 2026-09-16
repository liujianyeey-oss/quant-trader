"""美股 ETF 研究引擎：t 收盘信号 → t+1 开盘成交，US 成本口径。

纪律（与审计修正后的 A股引擎 / us_equities 框架一致）：
- 信号并入完整日历 → 前向填充 → shift(1)（T+1）→ 裁剪区间，承接区间前目标
- 开盘调仓用开盘价估值定股数（无前视）；参与调仓的已有/目标持仓缺开盘价
  整次延期到下一交易日重试；先卖后买，现金不足按比例缩减
- 成本：佣金 0.005/股（每笔最低 $1）+ 卖出金额 0.5bps + 滑点 10bps（单边）
- 整股（美股无 100 股限制）、无杠杆、现金收益 0
"""
import numpy as np
import pandas as pd

DEFAULT_COST = {"commission_per_share": 0.005, "minimum_commission": 1.0,
                "sell_fee_bps": 0.5, "slippage_bps": 10}
INITIAL_USD = 50_000.0


def adjusted_panels(data):
    """复权（含拆分、不含分红）开/收盘面板。"""
    adj_o, adj_c = {}, {}
    for s, df in data.items():
        d = df.set_index("date")
        adj_o[s] = d["open"] * d["factor"]
        adj_c[s] = d["close"] * d["factor"]
    return pd.DataFrame(adj_o).sort_index(), pd.DataFrame(adj_c).sort_index()


class EtfBacktester:
    def __init__(self, cost=None, initial=INITIAL_USD):
        self.c = dict(DEFAULT_COST if cost is None else cost)
        self.initial = float(initial)
        self.slip = self.c["slippage_bps"] / 10000.0

    def fee(self, n, price, sell=False):
        f = max(self.c["minimum_commission"], n * self.c["commission_per_share"])
        if sell:
            f += n * price * self.c["sell_fee_bps"] / 10000.0
        return f

    def run(self, data, signals, start, end):
        opens, closes = adjusted_panels(data)
        calendar = closes.index
        marks = closes.ffill()  # 估值价前向填充（单日缺行情不归零）

        s = signals.reindex(columns=closes.columns, fill_value=0.0)
        aligned = (s.reindex(calendar.union(s.index)).sort_index().ffill()
                   .reindex(calendar).fillna(0.0))
        targets = aligned.shift(1).fillna(0.0)
        start, end = pd.Timestamp(start), pd.Timestamp(end)

        cash = self.initial
        shares = {c: 0 for c in closes.columns}
        trades, rows = [], []
        pending = False
        completed = pd.Series(0.0, index=closes.columns)
        for date in calendar:
            if date < start or date > end:
                continue
            w = targets.loc[date]
            changed = not np.allclose(w.to_numpy(), completed.to_numpy(), atol=1e-10)
            if changed or pending:
                o = opens.loc[date]
                needed = [c for c in shares if shares[c] > 0 or w.get(c, 0) > 0]
                if all(pd.notna(o[c]) and o[c] > 0 for c in needed):
                    cash = self._rebalance(date, w, o, cash, shares, trades)
                    completed = w.copy()
                    pending = False
                else:
                    pending = True
            assert cash >= -1e-6, f"现金透支 {cash} @ {date}"
            nav = cash + sum(n * marks.loc[date, c] for c, n in shares.items()
                             if n > 0 and pd.notna(marks.loc[date, c]))
            rows.append((date, nav))
        return pd.Series(dict(rows)).sort_index(), trades

    def _rebalance(self, date, tgt, o, cash, shares, trades):
        slip = self.slip
        nav = cash + sum(n * o[c] for c, n in shares.items() if n > 0)
        desired = {}
        for c in shares:
            w = tgt.get(c, 0.0)
            desired[c] = int(w * nav / (o[c] * (1 + slip))) if w > 0 else 0
        # 先卖
        for c in shares:
            delta = desired[c] - shares[c]
            if delta < 0:
                price = o[c] * (1 - slip)
                amount = -delta * price
                fee = self.fee(-delta, price, sell=True)
                cash += amount - fee
                shares[c] += delta
                trades.append(dict(date=date, code=c, side="sell", shares=-delta,
                                   price=price, amount=amount, fee=fee))
        # 后买：现金不足按比例缩减，最低佣金不透支
        buys = {c: desired[c] - shares[c] for c in shares if desired[c] - shares[c] > 0}
        if buys:
            def total_cost(q):
                return sum(n * o[c] * (1 + slip) + self.fee(n, o[c] * (1 + slip))
                           for c, n in q.items() if n > 0)
            cost = total_cost(buys)
            if cost > cash:
                ratio = max(0.0, cash / cost)
                buys = {c: int(n * ratio) for c, n in buys.items()}
            for c, n in buys.items():
                price = o[c] * (1 + slip)
                while n > 0 and n * price + self.fee(n, price) > cash + 1e-8:
                    n -= 1
                if n <= 0:
                    continue
                amount = n * price
                fee = self.fee(n, price)
                shares[c] += n
                cash -= amount + fee
                trades.append(dict(date=date, code=c, side="buy", shares=n,
                                   price=price, amount=amount, fee=fee))
        return cash


def perf(eq, trades, initial=INITIAL_USD):
    """绩效指标：自然日年化、回撤、夏普、卡玛、费用总额、年均换手。"""
    years = max(((eq.index[-1] - eq.index[0]).days + 1) / 365.25, 1 / 365.25)
    total = eq.iloc[-1] / initial - 1
    cagr = (1 + total) ** (1 / years) - 1
    dd = (eq / eq.cummax() - 1).min()
    ret = eq.pct_change().dropna()
    sharpe = ret.mean() / ret.std() * np.sqrt(252) if ret.std() > 0 else np.nan
    calmar = cagr / abs(dd) if dd < 0 else np.nan
    fees = sum(t["fee"] for t in trades)
    amount = sum(t["amount"] for t in trades)
    turnover = amount / eq.mean() / years if len(eq) else np.nan  # 买卖双边金额/平均净值/年
    return dict(total=total, cagr=cagr, mdd=dd, sharpe=sharpe, calmar=calmar,
                fees=fees, turnover=turnover)


def yearly_returns(eq, initial=INITIAL_USD):
    """连续净值下的逐年收益。"""
    out, prev = {}, initial
    for y, g in eq.groupby(eq.index.year):
        out[int(y)] = g.iloc[-1] / prev - 1
        prev = g.iloc[-1]
    return out
