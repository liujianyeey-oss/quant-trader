"""组合视图合成数学 + 现金停泊收益计算的单元测试。

运行：.venv/bin/python -m unittest discover -s tests -v
"""
import unittest

import numpy as np
import pandas as pd

from portfolio_view import p2_weights, compose
from research_cash.runner import cash_uplift


class PortfolioMathTests(unittest.TestCase):
    def test_p2_weights_inverse_vol(self):
        """60 日波动倒数加权：A 波动 1%、B 波动 2% → 原始 2:1，经 clip[0.1,0.4] 后 1.2:1。"""
        dates = pd.bdate_range("2024-01-01", periods=60)
        # 确定性交替收益，std 精确 1% / 2%
        a = np.tile([0.01, -0.01], 30)
        b = np.tile([0.02, -0.02], 30)
        rets = pd.DataFrame({"A": a, "B": b}, index=dates)
        w, notes = p2_weights(rets)
        self.assertEqual(notes, [])
        # 原始权重 0.667/0.333 → clip 后 0.4/0.333 → 归一后 0.5455/0.4545（R7a 口径）
        self.assertAlmostEqual(w["A"], 0.4 / (0.4 + 1 / 3), places=4)
        self.assertAlmostEqual(w["A"] / w["B"], 1.2, delta=0.05)
        self.assertAlmostEqual(w.sum(), 1.0, places=9)

    def test_p2_weights_insufficient_data(self):
        """不足 60 日 → 等权 + 标注。"""
        rets = pd.DataFrame({"A": [0.01] * 10, "B": [0.0] * 10},
                            index=pd.bdate_range("2024-01-01", periods=10))
        w, notes = p2_weights(rets)
        self.assertTrue((w == 0.5).all())
        self.assertTrue(any("等权" in n for n in notes))

    def test_compose_monthly_rebalance_math(self):
        """已知序列手算验证：单月初始 50/50 + 月末再平衡的合成净值。"""
        dates = pd.bdate_range("2024-03-01", periods=10)  # 同一月内
        nav = pd.DataFrame({
            "A": [100 * 1.01 ** i for i in range(10)],   # 每天 +1%
            "B": [100.0] * 10},                           # 持平
            index=dates)
        port, target_w, current_w, notes = compose(nav)
        # 手算：Mar 13 时 A=0.5×1.01^8、B=0.5（共 8 个计息日）；
        # Mar 14 月末再平衡为 50/50，随后当日 A 再 +1%
        pre = 0.5 * 1.01 ** 8 + 0.5
        expect = pre / 2 * 1.01 + pre / 2
        self.assertAlmostEqual(port.iloc[-1], expect, places=9)
        self.assertAlmostEqual(target_w["A"], 0.5, places=9)  # 不足60日等权
        self.assertAlmostEqual(current_w["A"] + current_w["B"], 1.0, places=9)


class CashUpliftTests(unittest.TestCase):
    def test_cash_uplift_exact(self):
        """全现金(cash_w=1) + 货币日涨 0.01% + 无交易 → 每日增益 = 前日净值×0.0001。"""
        dates = pd.bdate_range("2024-01-01", periods=5)
        holdings = pd.DataFrame({"nav": [100.0] * 5, "cash_w": [1.0] * 5,
                                 "traded": [0.0] * 5}, index=dates)
        money_close = pd.Series([100 * 1.0001 ** i for i in range(5)], index=dates)
        gain, daily_gain, friction = cash_uplift(holdings, money_close)
        # 首日 cash_w 用首日值（1.0）；第 2~5 日每日增益 = 100×0.0001
        self.assertAlmostEqual(daily_gain.iloc[1], 0.0001, places=10)
        self.assertAlmostEqual(gain.iloc[1], 100 * 0.0001, places=9)
        self.assertAlmostEqual(friction.sum(), 0.0, places=9)
        self.assertAlmostEqual(gain.sum(), 100 * 0.0001 * 4, places=9)

    def test_cash_uplift_friction(self):
        """调仓日交易额按万1 计摩擦。"""
        dates = pd.bdate_range("2024-01-01", periods=3)
        holdings = pd.DataFrame({"nav": [100.0] * 3, "cash_w": [0.5] * 3,
                                 "traded": [0.0, 10.0, 0.0]}, index=dates)
        money_close = pd.Series([100.0, 100.0, 100.0], index=dates)
        gain, daily_gain, friction = cash_uplift(holdings, money_close)
        # common 索引剔除了首日（货币收益首日为 NaN），结果第 0 行即调仓日
        self.assertAlmostEqual(friction.iloc[0], 10.0 * 0.0001, places=10)
        self.assertAlmostEqual(daily_gain.iloc[0], 0.5 * 0.0, places=10)


if __name__ == "__main__":
    unittest.main(verbosity=2)
