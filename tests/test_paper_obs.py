"""观察账户 6/7（paper_obs）回归测试：
RSI2 规则正确性、H2 宏观窗口禁开仓、幂等、补记。

运行：.venv/bin/python -m unittest discover -s tests -v
"""
import os
import tempfile
import unittest

import numpy as np
import pandas as pd

from paper_obs import update_account, load_state, read_csv, _csv_path, sig_qqq_h2


def make_df(closes, opens=None, highs=None, lows=None, vols=None):
    n = len(closes)
    dates = pd.bdate_range("2020-01-01", periods=n)
    c = np.asarray(closes, dtype=float)
    return pd.DataFrame({
        "date": dates, "open": c if opens is None else np.asarray(opens, float),
        "high": c if highs is None else np.asarray(highs, float),
        "low": c if lows is None else np.asarray(lows, float),
        "close": c,
        "volume": np.full(n, 100.0) if vols is None else np.asarray(vols, float)})


def rsi2_data():
    """260 天缓慢上行（站上 MA200）后浅挫 3 天（RSI2<10 且收盘仍在 MA200 上），随后回升。"""
    prices = list(np.linspace(100, 120, 260))           # 260 天慢牛
    prices += [116.0, 114.0, 113.0]                      # 浅挫（RSI2 深跌但仍 > MA200）
    prices += [114.5, 116.0, 117.0, 118.0, 119.0]        # 回升（收盘 > MA5 触发卖出）
    return make_df(prices)


def qqq_macro_data():
    """40 天平坦 10 → index 35 突破 11（H2 窗口开启，禁开仓）→ index 40 新高 12（窗口关闭，开仓）。
    us10y：前 30 天 3.00 → index 31 起 3.30（20 日变化 +30bp > +25bp）→ index 39 回落。"""
    n = 45
    closes = [10.0] * 35 + [11.0] * 5 + [12.0] * 5
    uy = [3.0] * 31 + [3.30] * 8 + [3.0] * 6
    dates = pd.bdate_range("2020-01-01", periods=n)
    us10y = pd.DataFrame({"date": dates, "us10y": uy})
    return make_df(closes), us10y


class Rsi2RuleTests(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.mkdtemp()

    def step(self, data, i):
        cut = {"159915": data.iloc[:i + 1]}
        return update_account(cut, "hs_rsi2", self.dir)

    def test_rsi2_buy_and_sell(self):
        data = rsi2_data()
        dates = data["date"]
        # 首个浅挫日 index 260（-3.3%）即触发 RSI2<10 买入信号，index 261 开盘买入
        for i in range(250, 262):
            self.step(data, i)
        trades = read_csv(_csv_path(self.dir, "trades", "hs_rsi2"))
        self.assertEqual(len(trades), 1)
        self.assertEqual(trades.iloc[0]["side"], "buy")
        self.assertEqual(trades.iloc[0]["date"], dates.iloc[261])
        self.assertAlmostEqual(trades.iloc[0]["price"], 114.0 * 1.001, places=4)
        # 收盘 > MA5（index 264，118 > ~115.7）→ index 265 开盘卖出清仓
        for i in range(262, 266):
            self.step(data, i)
        trades = read_csv(_csv_path(self.dir, "trades", "hs_rsi2"))
        sells = trades[trades["side"] == "sell"]
        self.assertEqual(len(sells), 1)
        self.assertEqual(load_state(self.dir, "hs_rsi2")["shares"], 0)

    def test_idempotent(self):
        data = rsi2_data()
        self.step(data, 261)
        before = self._counts()
        self.step(data, 261)
        self.assertEqual(self._counts(), before)

    def test_catchup_two_days(self):
        data = rsi2_data()
        dates = data["date"]
        self.step(data, 260)
        # 缺跑 261、262，一次补记：261 开盘成交 + 两日信号/净值补齐
        self.step(data, 262)
        nav = read_csv(_csv_path(self.dir, "nav", "hs_rsi2"))
        self.assertEqual(list(nav["date"]), [dates.iloc[260], dates.iloc[261],
                                             dates.iloc[262]])
        trades = read_csv(_csv_path(self.dir, "trades", "hs_rsi2"))
        self.assertEqual(len(trades), 1)
        self.assertEqual(trades.iloc[0]["date"], dates.iloc[261])

    def _counts(self):
        return tuple(len(read_csv(_csv_path(self.dir, k, "hs_rsi2")))
                     for k in ("nav", "signals", "trades"))


class H2FilterTests(unittest.TestCase):
    def test_h2_blocks_entry_when_crowded(self):
        data, us10y = qqq_macro_data()
        w, h2 = sig_qqq_h2({"Q": data}, us10y, "Q")
        dates = data["date"]
        # index 35 的突破在 H2 窗口内 → 无买入；index 40 窗口关闭后的新高 → 买入
        self.assertTrue(bool(h2.loc[dates.iloc[35]]))
        buys = w[w > 0]
        self.assertFalse(buys.empty)
        first_buy_date = buys.index[0]
        self.assertEqual(first_buy_date, dates.iloc[40])
        # 宏观窗口关闭证据：40 日变化回落
        self.assertFalse(bool(h2.loc[dates.iloc[40]]))


if __name__ == "__main__":
    unittest.main(verbosity=2)
