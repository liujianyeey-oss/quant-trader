"""回归测试：执行时点、无前视、延期重试、成本与信号因果性。

移植自独立审计 test_strategy_research.py 的 6 项测试意图，
适配本项目的 Backtester / 策略接口。运行：
    .venv/bin/python -m unittest discover -s tests -v
"""
import unittest

import numpy as np
import pandas as pd

from backtest import Backtester
from strategies import MomentumRotation, StaggeredMomentum

ATTACK = ["A", "B", "C", "D"]
GOLD = "GOLD"


def make_cfg(**over):
    cfg = {
        "cost": {"etf_commission": 0.0001, "stock_commission": 0.00025,
                 "stamp_tax": 0.0005, "slippage": 0.001, "min_commission": 0},
        "backtest": {"initial_cash": 1_000_000},
        "universe": {"etfs": ATTACK, "stocks": [], "defensive": GOLD},
        "strategy": {"momentum_days": 20, "rebalance_days": 5,
                     "ma_fast": 5, "ma_slow": 20, "trend_ma": 60},
        "staggered": {"momentum_days": 20, "trend_ma": 60, "vol_window": 60,
                      "vol_budget": 0.12, "n_groups": 5},
    }
    cfg["cost"].update(over.get("cost", {}))
    cfg["backtest"].update(over.get("backtest", {}))
    return cfg


def make_data(prices, opens=None):
    """单标的平坦/指定价格数据，代码为 'A'。"""
    dates = pd.bdate_range("2020-01-01", periods=len(prices))
    return {"A": pd.DataFrame({"date": dates,
                               "open": prices if opens is None else opens,
                               "close": prices, "high": prices, "low": prices,
                               "volume": 1})}


class ExecutionTests(unittest.TestCase):
    """引擎执行语义（对应审计测试 1-5）。"""

    def run_bt(self, data, signals, start, end, **cfg_over):
        return Backtester(make_cfg(**cfg_over)).run(data, signals, start=start, end=end)

    def test_next_day_execution_and_pre_start_carry(self):
        """信号次日开盘成交；区间开始前的已知目标会被承接。"""
        d = make_data([10.0, 10.0, 10.0])
        dates = d["A"]["date"]
        s = pd.DataFrame({"A": [1.0]}, index=[dates[0]])
        _, t1 = self.run_bt(d, s, dates[0], dates[2],
                            cost={"slippage": 0, "etf_commission": 0})
        self.assertEqual(t1[0]["date"], dates[1])  # 次日执行
        # 回测从 dates[1] 开始：区间前的信号目标仍被承接，首日即建仓
        _, t2 = self.run_bt(d, s, dates[1], dates[2],
                            cost={"slippage": 0, "etf_commission": 0})
        self.assertEqual(t2[0]["date"], dates[1])

    def test_close_does_not_change_open_order_size(self):
        """开盘调仓只用开盘价定股数；改当天收盘价不影响已成交订单（无前视）。"""
        d = make_data([10.0, 10.0, 20.0, 10.0], opens=[10.0, 10.0, 10.0, 10.0])
        dates = d["A"]["date"]
        s = pd.DataFrame({"A": [1.0, 0.5]}, index=[dates[0], dates[1]])
        zero = {"slippage": 0, "etf_commission": 0}
        _, t1 = self.run_bt(d, s, dates[0], dates[3], cost=zero)
        d["A"].loc[2, "close"] = 100.0  # 改 day2 收盘价
        _, t2 = self.run_bt(d, s, dates[0], dates[3], cost=zero)
        key = ("date", "code", "side", "shares", "price")
        self.assertEqual([tuple(x[k] for k in key) for x in t1 if x["date"] <= dates[2]],
                         [tuple(x[k] for k in key) for x in t2 if x["date"] <= dates[2]])
        sells = [t for t in t1 if t["side"] == "sell"]
        self.assertEqual(sells[0]["shares"], 50000)  # 按开盘价估值 1e6 的半仓

    def test_missing_open_defers_without_close_fallback(self):
        """缺开盘价时整次调仓延期到下一交易日，不用收盘价替代成交。"""
        d = make_data([10.0, 10.0, 10.0], opens=[10.0, np.nan, 10.0])
        dates = d["A"]["date"]
        s = pd.DataFrame({"A": [1.0]}, index=[dates[0]])
        _, t = self.run_bt(d, s, dates[0], dates[2],
                           cost={"slippage": 0, "etf_commission": 0})
        self.assertEqual(t[0]["date"], dates[2])  # day1 无开盘价 → 延期到 day2

    def test_min_fee_cannot_overdraw(self):
        """最低佣金不得导致现金透支（买不起就整单放弃，现金不为负）。"""
        d = make_data([10.0, 10.0, 10.0])
        dates = d["A"]["date"]
        s = pd.DataFrame({"A": [1.0]}, index=[dates[0]])
        eq, t = self.run_bt(d, s, dates[0], dates[2],
                            cost={"slippage": 0, "etf_commission": 0.0001,
                                  "min_commission": 5},
                            backtest={"initial_cash": 1001.0})
        self.assertEqual(len(t), 0)  # 100 股需 1000+5 元佣金 > 1001，放弃
        self.assertTrue((eq["total_value"] >= 0).all())

    def test_flat_prices_charge_both_sides(self):
        """平价买入卖出，买卖两侧都收佣金+滑点，期末净值与现金流一致。"""
        d = make_data([10.0] * 4)
        dates = d["A"]["date"]
        s = pd.DataFrame({"A": [1.0, 0.0]}, index=[dates[0], dates[2]])
        eq, t = self.run_bt(d, s, dates[0], dates[3])  # 默认滑点0.1% 佣金万1
        self.assertEqual([x["side"] for x in t], ["buy", "sell"])
        buy, sell = t
        buy_amt, sell_amt = buy["price"] * buy["shares"], sell["price"] * sell["shares"]
        buy_fee, sell_fee = buy_amt * 0.0001, sell_amt * 0.0001
        self.assertLess(eq["total_value"].iloc[-1], 998000.0)
        self.assertAlmostEqual(eq["total_value"].iloc[-1],
                               1e6 - buy_amt - buy_fee + sell_amt - sell_fee,
                               places=2)


class SignalCausalityTests(unittest.TestCase):
    """信号因果性与权重约束（对应审计测试 6）。"""

    def make_random_data(self, n=240):
        dates = pd.bdate_range("2018-01-01", periods=n)
        rng = np.random.default_rng(9)
        data = {}
        for code in ATTACK + [GOLD]:
            prices = 10 * np.exp(np.cumsum(rng.normal(0.001, 0.01, n)))
            data[code] = pd.DataFrame({"date": dates, "open": prices,
                                       "close": prices, "high": prices,
                                       "low": prices, "volume": 1})
        return data

    def test_future_data_does_not_change_historical_signals(self):
        """未来数据不改变历史信号；目标权重和不超过 1（不用杠杆）。"""
        data = self.make_random_data()
        cut_data = {c: df.iloc[:180] for c, df in data.items()}
        cfg = make_cfg()
        for cls in (MomentumRotation, StaggeredMomentum):
            strategy = cls(cfg)
            full = strategy.generate_signals(data)
            cut = cls(cfg).generate_signals(cut_data)
            pd.testing.assert_frame_equal(full.loc[cut.index], cut)
            self.assertTrue((full.sum(axis=1) <= 1 + 1e-9).all())
            self.assertTrue((full >= 0).all().all())


if __name__ == "__main__":
    unittest.main(verbosity=2)
