"""港股模块的因果性、市场约束与现金记账回归；不依赖网络。"""
import copy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
import yaml

from hk_connect.backtest import HKBacktester, HKCosts
from hk_connect.cli import load_calendar
from hk_connect.data import HKDataLoader, Universe, align_prices, hk_code, normalize_bars
from hk_connect.strategy import HKConnectMomentum


def config():
    cfg = yaml.safe_load((Path(__file__).resolve().parents[1] / "config.yaml").read_text())
    cfg["hk_connect"]["strategy"]["min_positions"] = 2  # 测试池只包含两只
    return cfg


def universe(mode="snapshot", end=None):
    return Universe({"mode": mode, "observed_at": "2026-09-09", "securities": [
        dict(code=c, lot=lot, name=c, currency="HKD", category="Equity",
             stamp_duty=True, valid_from="2024-01-01", valid_to=end)
        for c, lot in [("00700", 100), ("00005", 400)]]})


def bars(n=200):
    dates = pd.bdate_range("2025-01-01", periods=n)
    x = np.arange(n)
    result = {}
    for c, scale in [("00700", 1), ("00005", .5)]:
        price = (30 + x * .12 + np.sin(x / 3) * .15) * scale
        result[c] = normalize_bars(pd.DataFrame({"date": dates, "open": price,
            "high": price * 1.02, "low": price * .98, "close": price,
            "signal_close": price, "volume": 10000000}))
    return result


class TestHKConnect(unittest.TestCase):
    def test_code_and_board_lot_validation(self):
        self.assertEqual(hk_code("700"), "00700")
        for value in ["600519", "00700.HK", "../x", "700.0"]:
            with self.assertRaises(ValueError):
                hk_code(value)
        payload = copy.deepcopy(universe().payload)
        payload["securities"][0]["lot"] = 1.5
        with self.assertRaises(ValueError):
            Universe(payload)

    def test_snapshot_not_historical_permission(self):
        u = universe()
        self.assertFalse(u.eligible("00700", "2025-01-01"))
        self.assertTrue(u.eligible("00700", "2025-01-01", research=True))
        self.assertFalse(u.eligible("09999", "2026-09-09", research=True))

    def test_intervals_validate_overlap_and_expiry(self):
        u = universe("intervals", "2025-01-02")
        self.assertTrue(u.eligible("00700", "2025-01-02"))
        self.assertFalse(u.eligible("00700", "2025-01-03"))
        self.assertEqual(u.lot("00700", "2025-01-03"), 100)
        p = copy.deepcopy(u.payload)
        p["securities"].append(copy.deepcopy(p["securities"][0]))
        with self.assertRaisesRegex(ValueError, "重叠"):
            Universe(p)

    def test_fees_both_sides_rounding(self):
        c = HKCosts(config()["hk_connect"]["cost"])
        # 10000 HKD: 最低佣金5 + 印花税10 + .57 + .27 + .02 + .42 =16.28。
        self.assertAlmostEqual(c.fee(10000), 16.28)
        self.assertAlmostEqual(c.fee(10000, stamp=False), 6.28)
        self.assertEqual(c.fee(0), 0)
        self.assertGreaterEqual(c.fee(10000.01) - c.fee(10000), 1)

    def test_signal_prefix_invariance_and_limits(self):
        data, cfg = bars(220), config()
        st = HKConnectMomentum(cfg, universe(), research=True)
        full = st.generate_signals(data)
        cut = {c: d.iloc[:180] for c, d in data.items()}
        short = st.generate_signals(cut)
        pd.testing.assert_frame_equal(full.loc[short.index], short)
        self.assertGreater(full.sum().sum(), 0)
        self.assertTrue((full.max(axis=1) <= .2 + 1e-9).all())
        self.assertTrue((full.sum(axis=1) <= .8 + 1e-9).all())

    def test_liquidity_and_absolute_trend_filter(self):
        data = bars()
        data["00700"]["turnover_hkd"] = 0
        st = HKConnectMomentum(config(), universe(), research=True)
        self.assertEqual(st.generate_signals(data)["00700"].sum(), 0)
        for d in data.values():
            d["signal_close"] = d.signal_close.to_numpy()[::-1]
        self.assertEqual(st.generate_signals(data).sum().sum(), 0)

    def test_weekday_parameter_is_calendar_anchored(self):
        data = bars()
        for day in range(5):
            cfg = config()
            cfg["hk_connect"]["strategy"]["rebalance_weekday"] = day
            s = HKConnectMomentum(cfg, universe(), research=True).generate_signals(data)
            buys = s.diff().fillna(s).gt(1e-9).any(axis=1)
            self.assertTrue(all(d.weekday() == day for d in s.index[buys]))

    def test_next_open_and_variable_lots_cash_conservation(self):
        data = bars(4)
        dates = data["00700"].date
        sig = pd.DataFrame({"00700": [.2], "00005": [.2]}, index=[dates.iloc[0]])
        bt = HKBacktester(config(), universe(), research=True)
        eq, trades = bt.run(data, sig, str(dates.iloc[0].date()), str(dates.iloc[-1].date()))
        self.assertEqual(eq.iloc[0].total_value, 100000)
        self.assertEqual({t["date"] for t in trades}, {dates.iloc[1]})
        self.assertEqual({t["code"] for t in trades}, {"00700", "00005"})
        for t in trades:
            self.assertEqual(t["shares"] % universe().lot(t["code"], dates.iloc[1], True), 0)
        self.assertAlmostEqual(eq.iloc[1].cash_hkd, 100000 - sum(t["amount_hkd"] + t["fee_hkd"] for t in trades))
        self.assertTrue((eq.cash_hkd >= 0).all())

    def test_future_close_does_not_change_open_order(self):
        data = bars(4)
        dates = data["00700"].date
        sig = pd.DataFrame({"00700": [.2]}, index=[dates.iloc[0]])
        bt = HKBacktester(config(), universe(), True)
        _, first = bt.run(data, sig, str(dates.iloc[0].date()), str(dates.iloc[-1].date()))
        changed = copy.deepcopy(data)
        for col in ["close", "signal_close", "high"]:
            changed["00700"].loc[1, col] *= 1.08
        _, second = bt.run(changed, sig, str(dates.iloc[0].date()), str(dates.iloc[-1].date()))
        self.assertEqual(first, second)

    def test_closed_connect_day_and_halt_retry(self):
        data = bars(5)
        dates = data["00700"].date
        sig = pd.DataFrame({"00700": [.2]}, index=[dates.iloc[0]])
        calendar = pd.Series([True, False, True, True, True], index=dates)
        data["00700"].loc[2, "volume"] = 0
        bt = HKBacktester(config(), universe(), True, calendar)
        eq, trades = bt.run(data, sig, str(dates.iloc[0].date()), str(dates.iloc[-1].date()))
        self.assertEqual(trades[0]["date"], dates.iloc[3])
        self.assertEqual(bt.deferred_days, 2)
        self.assertFalse(bt.pending)
        self.assertTrue(eq.iloc[2].pending)

    def test_sell_only_after_membership_ends(self):
        data = bars(5)
        dates = data["00700"].date
        u = universe("intervals", str(dates.iloc[1].date()))
        calendar = pd.Series(True, index=dates)
        bt = HKBacktester(config(), u, False, calendar)
        sig = pd.DataFrame({"00700": [.2]}, index=[dates.iloc[0]])
        _, trades = bt.run(data, sig, str(dates.iloc[0].date()), str(dates.iloc[-1].date()))
        self.assertEqual([t["side"] for t in trades], ["buy", "sell"])
        self.assertEqual(trades[-1]["date"], dates.iloc[2])

    def test_strict_mode_rejects_snapshot_and_missing_calendar(self):
        data = bars(4)
        sig = pd.DataFrame({"00700": [.2]}, index=[data["00700"].date.iloc[0]])
        for u in [universe(), universe("intervals")]:
            with self.assertRaises(ValueError):
                HKBacktester(config(), u).run(data, sig, "2025-01-01", "2025-02-01")

    def test_minimum_commission_cannot_overdraw(self):
        cfg = config()
        cfg["hk_connect"]["initial_cash_hkd"] = 3200
        cfg["hk_connect"]["cost"]["min_commission_hkd"] = 500
        data = bars(4)
        sig = pd.DataFrame({"00700": [1.]}, index=[data["00700"].date.iloc[0]])
        eq, trades = HKBacktester(cfg, universe(), True).run(data, sig, "2025-01-01", "2025-02-01")
        self.assertEqual(trades, [])
        self.assertTrue((eq.cash_hkd == 3200).all())

    def test_reject_split_and_invalid_data(self):
        data = bars(4)
        data["00700"].loc[2:, "signal_close"] *= 3
        sig = pd.DataFrame({"00700": [.2]}, index=[data["00700"].date.iloc[0]])
        with self.assertRaisesRegex(ValueError, "公司行为"):
            HKBacktester(config(), universe(), True).run(data, sig, "2025-01-01", "2025-02-01")
        d = bars(4)["00700"]
        d.loc[0, "open"] = -1
        with self.assertRaises(ValueError):
            normalize_bars(d)

    def test_offline_drops_today_even_if_cache_contaminated(self):
        with tempfile.TemporaryDirectory() as td:
            d = bars(4)["00700"]
            d.to_parquet(Path(td) / "00700.parquet", index=False)
            with patch("hk_connect.data.today_hk", return_value=pd.Timestamp("2025-01-03")):
                out = HKDataLoader(td).load("00700", "2025-01-01", "2025-01-03", offline=True)
            self.assertEqual(out.date.max(), pd.Timestamp("2025-01-02"))

    def test_calendar_no_silent_string_bool(self):
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "calendar.csv"
            p.write_text("date,is_open\n2025-01-01,0\n2025-01-02,1\n")
            cal = load_calendar(p)
            self.assertFalse(cal.iloc[0])
            self.assertTrue(cal.iloc[1])
            p.write_text("date,is_open\n2025-01-01,false\n")
            with self.assertRaises(ValueError):
                load_calendar(p)

    def test_pair_alignment_keeps_raw_prices_and_rejects_internal_gaps(self):
        d = bars(8)["00700"]
        raw = d.drop(columns=["signal_close", "turnover_hkd"]).iloc[2:]
        adj = d.copy()
        adj["close"] *= .8
        paired = align_prices(raw, adj)
        self.assertEqual(len(paired), 6)
        np.testing.assert_allclose(paired.close, raw.close)
        np.testing.assert_allclose(paired.signal_close, raw.close * .8)
        with self.assertRaises(ValueError):
            align_prices(raw, adj.drop(index=4))

    def test_unaffordable_lots_do_not_create_fake_diversification(self):
        cfg = config()
        cfg["hk_connect"]["initial_cash_hkd"] = 1000
        signals = HKConnectMomentum(cfg, universe(), True).generate_signals(bars())
        self.assertEqual(signals.sum().sum(), 0)

    def test_no_adjustment_response_must_match_independent_raw_ohlc(self):
        raw = bars(8)["00700"]
        adj = raw.copy()
        adj.attrs["adjustment_response"] = "day"
        self.assertEqual(len(align_prices(raw.drop(columns=["signal_close"]), adj)), 8)
        adj.loc[3, "close"] *= .99
        with self.assertRaisesRegex(ValueError, "复权状态"):
            align_prices(raw, adj)


if __name__ == "__main__":
    unittest.main()
