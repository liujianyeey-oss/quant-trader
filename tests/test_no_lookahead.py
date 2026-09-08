"""未来函数（look-ahead bias）专项测试。

验证回测/信号/台账全链路因果性：任何一天的信号、成交、净值只依赖当日及之前的数据。
运行：.venv/bin/python -m unittest discover -s tests -v
"""
import os
import tempfile
import unittest

import numpy as np
import pandas as pd
import yaml

from backtest import Backtester
from strategies import MomentumRotation, StaggeredMomentum
from paper import update_account, _csv_path, _read_csv

PROJECT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE_DIR = os.path.join(PROJECT, "data_cache")
ATTACK = ["A", "B", "C", "D"]
GOLD = "GOLD"


def make_cfg():
    return {
        "cost": {"etf_commission": 0.0001, "stock_commission": 0.00025,
                 "stamp_tax": 0.0005, "slippage": 0.001, "min_commission": 0},
        "backtest": {"initial_cash": 1_000_000},
        "universe": {"etfs": ["A", "B", "C", "D"], "stocks": [], "defensive": GOLD},
        "strategy": {"momentum_days": 20, "rebalance_days": 5,
                     "ma_fast": 5, "ma_slow": 20, "trend_ma": 60},
        "staggered": {"momentum_days": 20, "trend_ma": 60, "vol_window": 60,
                      "vol_budget": 0.12, "n_groups": 5},
    }


def random_data(n=300, seed=7):
    """5 标的随机游走（ATTACK + GOLD），开收同价。"""
    dates = pd.bdate_range("2018-01-01", periods=n)
    rng = np.random.default_rng(seed)
    data = {}
    for code in ATTACK + [GOLD]:
        prices = 10 * np.exp(np.cumsum(rng.normal(0.001, 0.01, n)))
        data[code] = pd.DataFrame({"date": dates, "open": prices, "close": prices,
                                   "high": prices, "low": prices, "volume": 1})
    return data


def truncate(data, cutoff):
    """截断到 cutoff（含）之前的数据，模拟"当时只有这些数据"。"""
    return {c: df[df["date"] <= cutoff].reset_index(drop=True)
            for c, df in data.items()}


def production_targets(signals, calendar):
    """与引擎一致的对齐：并入日历 → 前向填充（不含 shift，T 日信号记 T 日）。"""
    return (signals.reindex(calendar.union(signals.index)).sort_index().ffill()
            .reindex(calendar).fillna(0.0))


class SignalCausalityTests(unittest.TestCase):
    """1. 截断一致性：t 及之前的信号不受之后数据影响。"""

    def test_truncation_consistency_multiple_cuts(self):
        data = random_data(300)
        n = len(data["A"])
        cuts = [200, 220, 260]  # 相当于 t、t+20、t+60
        for cls in (MomentumRotation, StaggeredMomentum):
            cfg = make_cfg()
            full = cls(cfg).generate_signals(data)
            for cut in cuts:
                part = cls(cfg).generate_signals(truncate(data, data["A"]["date"].iloc[cut - 1]))
                pd.testing.assert_frame_equal(full.loc[part.index], part)


class ExecutionCausalityTests(unittest.TestCase):
    """2. 当日收盘免疫 + 成交时序。"""

    def _single_code_data(self, opens, closes):
        dates = pd.bdate_range("2020-01-01", periods=len(opens))
        return {"A": pd.DataFrame({"date": dates, "open": opens, "close": closes,
                                   "high": closes, "low": closes, "volume": 1})}

    def _trades(self, data):
        dates = data["A"]["date"]
        signals = pd.DataFrame({"A": [1.0, 0.5]}, index=[dates[0], dates[1]])
        _, trades = Backtester(make_cfg()).run(data, signals, dates.iloc[0], dates.iloc[-1])
        return [(t["date"], t["side"], t["shares"], round(t["price"], 6))
                for t in trades]

    def test_close_tampering_does_not_change_trades(self):
        """篡改 T 日收盘价 ±10%：T 日（及全部）成交的股数、价格完全不变。"""
        data = self._single_code_data([10.] * 5, [10.] * 5)
        base = self._trades(data)
        tampered = self._single_code_data([10.] * 5, [10., 11., 10., 10., 10.])  # T=day1 收盘 +10%
        self.assertEqual(base, self._trades(tampered))
        tampered_down = self._single_code_data([10.] * 5, [10., 9., 10., 10., 10.])
        self.assertEqual(base, self._trades(tampered_down))

    def test_next_day_open_tampering_only_affects_later_trades(self):
        """篡改 T+1 开盘价：T 日成交不变，T+1 及之后才变化。"""
        data = self._single_code_data([10.] * 5, [10.] * 5)
        base = self._trades(data)
        day1, day2 = data["A"]["date"].iloc[1], data["A"]["date"].iloc[2]
        tampered = self._single_code_data([10., 10., 9., 10., 10.], [10.] * 5)
        new = self._trades(tampered)
        self.assertEqual([t for t in base if t[0] <= day1],
                         [t for t in new if t[0] <= day1])     # T 日成交不变
        base_d2 = [t for t in base if t[0] == day2]
        new_d2 = [t for t in new if t[0] == day2]
        self.assertTrue(base_d2 and new_d2)
        self.assertNotEqual(base_d2[0][3], new_d2[0][3])       # T+1 成交价变化


@unittest.skipUnless(os.path.exists(os.path.join(CACHE_DIR, "510300_qfq.parquet")),
                     "无本地缓存，跳过真实数据测试")
class RealDataCausalityTests(unittest.TestCase):
    """3/4. 真实缓存数据上的成交时序与逐日因果扫描。"""

    @classmethod
    def setUpClass(cls):
        with open(os.path.join(PROJECT, "config.yaml"), encoding="utf-8") as f:
            cls.cfg = yaml.safe_load(f)
        codes = cls.cfg["universe"]["etfs"] + [cls.cfg["universe"]["defensive"]]
        cls.data = {c: pd.read_parquet(os.path.join(CACHE_DIR, f"{c}_qfq.parquet"))
                    for c in codes}

    def test_trades_always_after_signal(self):
        """全部成交日期必须严格晚于触发信号日期（至少 1 个日历日）。"""
        for cls in (MomentumRotation, StaggeredMomentum):
            strategy = cls(self.cfg)
            signals = strategy.generate_signals(self.data)
            bt = Backtester(self.cfg, rebalance_on_event=strategy.rebalance_on_event)
            _, trades = bt.run(self.data, signals, start="2019-01-01")
            self.assertTrue(trades, f"{strategy.name} 无成交")
            sig_dates = signals.index
            for t in trades:
                d = t["date"]
                prev = sig_dates[sig_dates < d]
                self.assertGreater(len(prev), 0, f"{strategy.name} {d} 成交无前置信号")
                self.assertGreaterEqual((d - prev[-1]).days, 1,
                                        f"{strategy.name} {d} 成交与信号同日")

    def test_daily_causal_scan_momentum(self):
        """每 20 天抽样：只用截至当天的数据重算目标，与生产信号序列该日一致。"""
        strategy = MomentumRotation(self.cfg)
        signals = strategy.generate_signals(self.data)
        calendar = pd.DatetimeIndex(sorted(set().union(
            *[set(df["date"]) for df in self.data.values()])))
        aligned = production_targets(signals, calendar)
        days = calendar[(calendar >= "2019-01-01")][::20]
        for d in days:
            cut = truncate(self.data, d)
            target, _ = strategy.current_target(cut)
            expect = aligned.loc[d]
            for c in expect.index:
                self.assertAlmostEqual(float(target.get(c, 0.0)), float(expect[c]),
                                       places=9, msg=f"momentum {d.date()} {c} 不一致")

    def test_daily_causal_scan_staggered(self):
        """staggered 同上（每 120 天抽样控制耗时）。"""
        strategy = StaggeredMomentum(self.cfg)
        signals = strategy.generate_signals(self.data)
        calendar = pd.DatetimeIndex(sorted(set().union(
            *[set(df["date"]) for df in self.data.values()])))
        aligned = production_targets(signals, calendar)
        days = calendar[(calendar >= "2019-01-01")][::120]
        for d in days:
            cut = truncate(self.data, d)
            target, _ = strategy.current_target(cut)
            expect = aligned.loc[d]
            for c in expect.index:
                self.assertAlmostEqual(float(target.get(c, 0.0)), float(expect[c]),
                                       places=9, msg=f"staggered {d.date()} {c} 不一致")


class PaperCausalityTests(unittest.TestCase):
    """5. 台账同口径：只用截至 D 的数据生成的记录，不被 D 之后的数据改写。"""

    def test_paper_records_immutable_to_future_data(self):
        cfg = make_cfg()
        # A 持续上涨（动量最强），其余平坦
        n = 40
        dates = pd.bdate_range("2020-01-01", periods=n)
        data = {}
        for code in ATTACK + [GOLD]:
            close = (10 * 1.01 ** np.arange(n) if code == "A"
                     else np.full(n, 100.0 if code == GOLD else 10.0))
            data[code] = pd.DataFrame({"date": dates, "open": close, "close": close,
                                       "high": close, "low": close, "volume": 1})
        d_mark = dates[35]
        paper_dir = tempfile.mkdtemp()
        # 模拟每日 cron：逐天喂到 D
        for i in range(30, 36):
            cut = {c: df.iloc[:i + 1] for c, df in data.items()}
            update_account(cfg, "momentum", cut, paper_dir)
        nav_before = _read_csv(_csv_path(paper_dir, "nav", "momentum"))
        sig_before = _read_csv(_csv_path(paper_dir, "signals", "momentum"))
        self.assertGreater(len(nav_before), 0)
        # 加入 D 之后的数据再跑
        for i in range(36, 40):
            cut = {c: df.iloc[:i + 1] for c, df in data.items()}
            update_account(cfg, "momentum", cut, paper_dir)
        nav_after = _read_csv(_csv_path(paper_dir, "nav", "momentum"))
        sig_after = _read_csv(_csv_path(paper_dir, "signals", "momentum"))
        pd.testing.assert_frame_equal(nav_after[nav_after["date"] <= d_mark].reset_index(drop=True),
                                      nav_before.reset_index(drop=True))
        pd.testing.assert_frame_equal(sig_after[sig_after["date"] <= d_mark].reset_index(drop=True),
                                      sig_before.reset_index(drop=True))


if __name__ == "__main__":
    unittest.main(verbosity=2)
