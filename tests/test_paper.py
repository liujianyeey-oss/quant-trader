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

    def append_day_partial(self, data, skip_codes=()):
        """给除 skip_codes 外的标的追加一个交易日（模拟数据源延迟/缺失）。"""
        new_date = max(df["date"].max() for df in data.values()) + pd.offsets.BDay(1)
        for code in CODES:
            if code in skip_codes:
                continue
            p = 100.0 if code == "GOLD" else float(
                data[code]["close"].iloc[-1] * (1.01 if code == "A" else 1.0))
            row = pd.DataFrame({"date": [new_date], "open": [p], "close": [p],
                                "high": [p], "low": [p], "volume": [1]})
            data[code] = pd.concat([data[code], row], ignore_index=True)
        return new_date

    def append_rows(self, data, code, dates, price):
        """给单个标的补指定日期的行（模拟延迟数据补齐/停牌后恢复）。"""
        rows = pd.DataFrame({"date": list(dates), "open": [price] * len(dates),
                             "close": [price] * len(dates), "high": [price] * len(dates),
                             "low": [price] * len(dates), "volume": [1] * len(dates)})
        data[code] = (pd.concat([data[code], rows], ignore_index=True)
                      .sort_values("date").reset_index(drop=True))

    def test_partial_data_day_blocked_then_recovered(self):
        """最新日部分标的缺数据（源未产出）：当日不推进水位线、不算信号；
        数据补齐后由后续运行自然补记；全程幂等。"""
        data = make_data()
        update_account(self.cfg, "momentum", data, self.dir)
        self.assertEqual(self.counts(), (1, 1, 0))

        # d2：GOLD 当日数据未产出（其余标的正常）
        d2 = self.append_day_partial(data, skip_codes=("GOLD",))
        actions = update_account(self.cfg, "momentum", data, self.dir)
        self.assertEqual(self.counts(), (1, 1, 0))  # 水位线不推进
        self.assertTrue(any("数据未齐" in a and "GOLD" in a for a in actions))
        # 重复运行仍然不动（幂等）
        update_account(self.cfg, "momentum", data, self.dir)
        self.assertEqual(self.counts(), (1, 1, 0))

        # GOLD 的 d2 数据补齐（如 18:23 兜底运行时源已产出）
        self.append_rows(data, "GOLD", [d2], 100.0)
        update_account(self.cfg, "momentum", data, self.dir)
        self.assertEqual(self.counts(), (2, 2, 1))  # d2 正常补记并成交
        nav = _read_csv(_csv_path(self.dir, "nav", "momentum"))
        self.assertEqual(nav["date"].iloc[-1], d2)
        trades = _read_csv(_csv_path(self.dir, "trades", "momentum"))
        self.assertEqual(trades.iloc[0]["date"], d2)
        # 再跑幂等
        update_account(self.cfg, "momentum", data, self.dir)
        self.assertEqual(self.counts(), (2, 2, 1))

    def test_suspension_gap_allowed_with_later_data(self):
        """标的缺中间某日但之后有数据（=停牌/单日缺行情）：不阻塞，走 ffill 旧逻辑。"""
        data = make_data()
        update_account(self.cfg, "momentum", data, self.dir)
        d2 = self.append_day_partial(data)  # 全日数据正常
        update_account(self.cfg, "momentum", data, self.dir)
        self.assertEqual(self.counts(), (2, 2, 1))  # 已买入 A

        # d3：GOLD 无当日行（实时看与源延迟不可区分 → 先阻塞）
        d3 = self.append_day_partial(data, skip_codes=("GOLD",))
        actions = update_account(self.cfg, "momentum", data, self.dir)
        self.assertEqual(self.counts(), (2, 2, 1))
        self.assertTrue(any("数据未齐" in a for a in actions))

        # d4：GOLD 恢复有行（d3 成为"缺当日但有更晚数据"的停牌缺口）→ d3/d4 一起补记
        d4 = self.append_day_partial(data)
        update_account(self.cfg, "momentum", data, self.dir)
        self.assertEqual(self.counts(), (4, 4, 1))
        nav = _read_csv(_csv_path(self.dir, "nav", "momentum"))
        self.assertEqual(list(nav["date"][-2:]), [d3, d4])

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
