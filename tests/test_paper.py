"""模拟盘台账（paper.py）回归测试：幂等、次日开盘价成交、非交易日跳过、偏差计算。

运行：.venv/bin/python -m unittest discover -s tests -v
"""
import os
import tempfile
import unittest

import numpy as np
import pandas as pd

from paper import update_account, load_state, _csv_path, _read_csv

CODES = ["A", "B", "C", "D", "GOLD"]


def make_cfg():
    return {
        "cost": {"etf_commission": 0.0001, "stock_commission": 0.00025,
                 "stamp_tax": 0.0005, "slippage": 0.001, "min_commission": 0},
        "backtest": {"initial_cash": 1_000_000},
        "universe": {"etfs": ["A", "B", "C", "D"], "stocks": [], "defensive": "GOLD"},
        "strategy": {"momentum_days": 20, "rebalance_days": 5,
                     "ma_fast": 5, "ma_slow": 20, "trend_ma": 20},
        "staggered": {"momentum_days": 20, "trend_ma": 60, "vol_window": 60,
                      "vol_budget": 0.12, "n_groups": 5},
    }


def make_data(n=35):
    """合成数据：A 持续上涨（动量最强且在均线上），其余平坦。"""
    dates = pd.bdate_range("2020-01-01", periods=n)
    data = {}
    for code in CODES:
        if code == "A":
            close = 10 * 1.01 ** np.arange(n)
        else:
            close = np.full(n, 100.0 if code == "GOLD" else 10.0)
        data[code] = pd.DataFrame({"date": dates, "open": close, "close": close,
                                   "high": close, "low": close, "volume": 1})
    return data


def append_day(data, open_price, close_price):
    """给所有标的追加一个交易日，返回新日期。"""
    new_date = data["A"]["date"].max() + pd.offsets.BDay(1)
    for code in CODES:
        p_o, p_c = (open_price, close_price) if code == "A" else (
            100.0 if code == "GOLD" else 10.0,) * 2
        row = pd.DataFrame({"date": [new_date], "open": [p_o], "close": [p_c],
                            "high": [p_c], "low": [p_c], "volume": [1]})
        data[code] = pd.concat([data[code], row], ignore_index=True)
    return new_date


class PaperTests(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.mkdtemp()
        self.cfg = make_cfg()

    def counts(self, name="momentum"):
        nav = _read_csv(_csv_path(self.dir, "nav", name))
        sig = _read_csv(_csv_path(self.dir, "signals", name))
        trd = _read_csv(_csv_path(self.dir, "trades", name))
        return len(nav), len(sig), len(trd)

    def test_first_run_creates_pending_no_trade(self):
        """开户：记录信号为待成交 + 首条净值，不产生成交。"""
        data = make_data()
        actions = update_account(self.cfg, "momentum", data, self.dir)
        state = load_state(self.dir, "momentum", 1e6)
        self.assertEqual(state["pending"]["target"], {"A": 1.0})
        self.assertEqual(state["pending"]["signal_date"],
                         f"{data['A']['date'].max():%Y-%m-%d}")
        self.assertEqual(self.counts(), (1, 1, 0))  # nav/signals/trades
        self.assertTrue(any("待下一交易日开盘成交" in a for a in actions))

    def test_idempotent_same_day(self):
        """同一天重复运行不产生重复记录。"""
        data = make_data()
        update_account(self.cfg, "momentum", data, self.dir)
        before = {}
        for k in ("nav", "signals"):
            p = _csv_path(self.dir, k, "momentum")
            if os.path.exists(p):
                with open(p, encoding="utf-8") as f:
                    before[k] = f.read()
        actions = update_account(self.cfg, "momentum", data, self.dir)
        after = {}
        for k in ("nav", "signals"):
            with open(_csv_path(self.dir, k, "momentum"), encoding="utf-8") as f:
                after[k] = f.read()
        self.assertEqual(before, after)
        self.assertEqual(self.counts(), (1, 1, 0))
        self.assertTrue(any("跳过" in a for a in actions))

    def test_execution_at_next_open_and_deviation(self):
        """待成交目标按次日真实开盘价成交；偏差=成交价/信号日收盘-1。"""
        data = make_data()
        update_account(self.cfg, "momentum", data, self.dir)
        signal_close = data["A"]["close"].iloc[-1]
        # 次日：开盘价 低于信号日收盘（隔夜下跌）
        next_open = signal_close * 0.98
        d2 = append_day(data, next_open, next_open * 1.01)
        update_account(self.cfg, "momentum", data, self.dir)

        trades = _read_csv(_csv_path(self.dir, "trades", "momentum"))
        self.assertEqual(len(trades), 1)
        t = trades.iloc[0]
        self.assertEqual(t["date"], d2)
        self.assertEqual(t["side"], "buy")
        self.assertAlmostEqual(t["price"], next_open * 1.001, places=6)  # 开盘价+滑点
        self.assertEqual(t["shares"] % 100, 0)
        expect_dev = (next_open * 1.001 / signal_close - 1) * 1e4
        self.assertAlmostEqual(t["dev_bps"], expect_dev, places=1)
        self.assertAlmostEqual(t["signal_close"], signal_close, places=3)
        # 当日又记录新信号为下一笔待成交；净值两条
        self.assertEqual(self.counts(), (2, 2, 1))
        state = load_state(self.dir, "momentum", 1e6)
        self.assertEqual(state["pending"]["signal_date"], f"{d2:%Y-%m-%d}")

    def test_missing_open_defers(self):
        """开盘价缺失当日不成交、顺延；下一有开盘价的交易日成交。"""
        data = make_data()
        update_account(self.cfg, "momentum", data, self.dir)
        d2 = append_day(data, np.nan, 10.0)   # 缺开盘价
        update_account(self.cfg, "momentum", data, self.dir)
        self.assertEqual(self.counts()[2], 0)  # 无成交
        self.assertIsNotNone(load_state(self.dir, "momentum", 1e6)["pending"])

        d3 = append_day(data, 13.0, 13.5)
        update_account(self.cfg, "momentum", data, self.dir)
        trades = _read_csv(_csv_path(self.dir, "trades", "momentum"))
        self.assertEqual(len(trades), 1)
        self.assertEqual(trades.iloc[0]["date"], d3)  # 顺延至下一可成交日
    def test_two_day_outage_catchup(self):
        """cron 连续两天故障（数据源不可达）：第三天恢复后一次运行应
        补记所有遗漏交易日——旧待成交在遗漏首日开盘成交、逐日信号与净值补齐。"""
        data = make_data()
        update_account(self.cfg, "momentum", data, self.dir)  # day X 开户
        last_close = float(data["A"]["close"].iloc[-1])
        first_date = data["A"]["date"].max()
        # 连续两个交易日数据源不可达（未运行），数据层面直接多出两天
        d1 = append_day(data, last_close, last_close)
        d2 = append_day(data, last_close, last_close)
        # 第三天恢复，一次运行
        actions = update_account(self.cfg, "momentum", data, self.dir)

        # day X 的待成交目标在遗漏首日 d1 开盘成交
        trades = _read_csv(_csv_path(self.dir, "trades", "momentum"))
        self.assertEqual(len(trades), 1)
        self.assertEqual(trades.iloc[0]["date"], d1)
        self.assertAlmostEqual(trades.iloc[0]["price"], last_close * 1.001, places=4)
        # d1 记录的新信号目标与持仓一致 → d2 不再重复交易（目标不变不重调）
        self.assertTrue(any("目标未变" in a for a in actions))
        # 净值/信号逐日补齐（X, d1, d2 共 3 行）
        nav = _read_csv(_csv_path(self.dir, "nav", "momentum"))
        sig = _read_csv(_csv_path(self.dir, "signals", "momentum"))
        self.assertEqual(list(nav["date"]), [first_date, d1, d2])
        self.assertEqual(len(sig), 3)
        self.assertEqual(self.counts(), (3, 3, 1))
        # 幂等：恢复当天重复运行无重复记录
        update_account(self.cfg, "momentum", data, self.dir)
        self.assertEqual(self.counts(), (3, 3, 1))


if __name__ == "__main__":
    unittest.main(verbosity=2)
