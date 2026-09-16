"""美股 donchian 台账（paper_us）回归测试：
规则正确性、次日开盘价成交、幂等、两日缺跑补记、偏差计算。

运行：.venv/bin/python -m unittest discover -s tests -v
"""
import os
import tempfile
import unittest

import numpy as np
import pandas as pd

from us_equities.paper_us import update_account, load_state, read_csv, _csv_path, TRACKED


def make_data(prices, opens=None, highs=None, lows=None):
    """单标的 'Q' 合成日线（factor 全 1）。"""
    n = len(prices)
    dates = pd.bdate_range("2020-01-01", periods=n)
    closes = np.asarray(prices, dtype=float)
    return {"Q": pd.DataFrame({
        "date": dates,
        "open": closes if opens is None else np.asarray(opens, dtype=float),
        "high": closes if highs is None else np.asarray(highs, dtype=float),
        "low": closes if lows is None else np.asarray(lows, dtype=float),
        "close": closes, "volume": 1, "factor": 1.0})}


def breakout_data():
    """前 35 天平坦 10；index 35 收盘 11（突破前 20 日高点 10）；
    随后 11 平台；index 43 最低价 9.0（跌破前 10 日低点 10）；index 44 为卖出执行日。"""
    closes = [10.0] * 35 + [11.0] * 8
    highs = list(closes)
    lows = list(closes)
    lows.append(9.0)   # index 43
    closes.append(9.5)
    highs.append(9.5)
    lows.append(9.5)   # index 44（卖出执行日）
    closes.append(9.5)
    highs.append(9.5)
    opens = list(closes)
    opens[36] = 11.2   # 突破次日开盘跳空高开（验证开盘价成交 + 偏差）
    return make_data(closes, opens=opens, highs=highs, lows=lows)


class PaperUsTests(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.mkdtemp()

    def counts(self, name="qqq"):
        nav = read_csv(_csv_path(self.dir, "nav", name))
        sig = read_csv(_csv_path(self.dir, "signals", name))
        trd = read_csv(_csv_path(self.dir, "trades", name))
        return len(nav), len(sig), len(trd)

    def step_to(self, data, upto_idx):
        """用 data 的前 upto_idx+1 天跑一次（upto_idx 可超出当前长度）。"""
        cut = {c: df.iloc[:upto_idx + 1] for c, df in data.items()}
        return update_account(cut, "qqq", "Q", self.dir)

    def test_donchian_rules_buy_on_breakout(self):
        """20 日突破买入：第 36 天（index 35 收盘 11 突破）→ 信号，index 36 开盘成交。"""
        data = breakout_data()
        dates = data["Q"]["date"]
        # 逐日推进到 index 35（突破日收盘）
        for i in range(30, 36):
            self.step_to(data, i)
        state = load_state(self.dir, "qqq")
        self.assertEqual(state["pending"]["target"], {"Q": 1.0})  # 突破→全仓目标
        self.assertEqual(state["pending"]["signal_close"]["Q"], 11.0)
        self.assertEqual(self.counts()[2], 0)  # 尚未成交
        # index 36：按开盘价 11.2（+10bps 滑点）成交
        self.step_to(data, 36)
        trades = read_csv(_csv_path(self.dir, "trades", "qqq"))
        self.assertEqual(len(trades), 1)
        t = trades.iloc[0]
        self.assertEqual(t["date"], dates.iloc[36])
        self.assertEqual(t["side"], "buy")
        self.assertAlmostEqual(t["price"], 11.2 * 1.001, places=4)
        expect_dev = (11.2 * 1.001 / 11.0 - 1) * 1e4
        self.assertAlmostEqual(t["dev_bps"], expect_dev, places=1)
        # 佣金：max($1, n×0.005)
        self.assertGreaterEqual(t["fee"], 1.0)

    def test_donchian_rules_sell_on_10d_low(self):
        """跌破前 10 日低点卖出：index 43 低点 9.0 → 信号，index 44 开盘卖出。"""
        data = breakout_data()
        dates = data["Q"]["date"]
        for i in range(30, 45):
            self.step_to(data, i)
        trades = read_csv(_csv_path(self.dir, "trades", "qqq"))
        self.assertEqual(len(trades), 2)
        sell = trades.iloc[1]
        self.assertEqual(sell["side"], "sell")
        self.assertEqual(sell["date"], dates.iloc[44])
        self.assertAlmostEqual(sell["price"], 9.5 * 0.999, places=4)  # 开盘-滑点
        state = load_state(self.dir, "qqq")
        self.assertEqual(state["shares"], {})  # 已空仓
        self.assertEqual(state["pending"]["target"], {})  # 空仓目标待成交

    def test_idempotent_same_day(self):
        data = breakout_data()
        self.step_to(data, 36)
        before = self.counts()
        update_account({c: df.iloc[:37] for c, df in data.items()}, "qqq", "Q", self.dir)
        self.assertEqual(self.counts(), before)
        # 再跑一次同数据：action 应为跳过
        actions = update_account({c: df.iloc[:37] for c, df in data.items()},
                                 "qqq", "Q", self.dir)
        self.assertTrue(any("跳过" in a for a in actions))

    def test_two_day_outage_catchup(self):
        """两天缺跑：第 36 天一次性补记 index 34/35/36 三天——
        34 无新信号、35 突破信号、36 开盘成交，净值逐日补齐且幂等。"""
        data = breakout_data()
        dates = data["Q"]["date"]
        for i in range(30, 34):
            self.step_to(data, i)
        # 缺跑 index 34、35，数据直接到 36
        actions = self.step_to(data, 36)
        nav = read_csv(_csv_path(self.dir, "nav", "qqq"))
        self.assertEqual(list(nav["date"]), list(dates.iloc[30:37]))  # 逐日补齐
        trades = read_csv(_csv_path(self.dir, "trades", "qqq"))
        self.assertEqual(len(trades), 1)
        self.assertEqual(trades.iloc[0]["date"], dates.iloc[36])
        self.assertTrue(any("补记" in a or "信号" in a for a in actions))
        # 幂等
        self.step_to(data, 36)
        self.assertEqual(self.counts(), (7, 7, 1))


if __name__ == "__main__":
    unittest.main(verbosity=2)
