import copy
import unittest
import numpy as np
import pandas as pd

from hk_connect.swing import SwingEngine, parameters, prepare
from test_hk_connect import config, universe


def sample(n=30):
    dates = pd.bdate_range("2025-01-01", periods=n)
    d = pd.DataFrame(dict(date=dates, open=100., high=102., low=98., close=100.,
                          signal_close=100., volume=2000000.))
    return {"00700": d}


def signals(data):
    d = data["00700"]
    date = d.date.iloc[0]
    return ({date: [dict(code="00700", signal_date=date, close=100., atr=3.,
                         level=99., invalid=99., score=1., adv=2e8)]},
            {"00700": pd.Series(3., index=d.date)})


def engine(p=None, calendar=None, capital=100000):
    return SwingEngine(parameters(dict(stall_r=0, take_profit_r=100, **(p or {}))), universe(),
                       config()["hk_connect"]["cost"], {"00700": "tech"}, capital, calendar, True)


def run(e, data, prepared=None, end=None):
    return e.run(data, "2025-01-01", end or str(data["00700"].date.max().date()), prepared or signals(data))


class TestSwing(unittest.TestCase):
    def test_entry_uses_next_open_and_actual_board_lot(self):
        d = sample()
        e = engine()
        eq, trades = run(e, d)
        self.assertEqual(trades[0]["date"], d["00700"].date.iloc[1])
        self.assertAlmostEqual(trades[0]["price"], 100.1)
        self.assertEqual(trades[0]["shares"] % 100, 0)
        self.assertTrue((eq.cash_hkd >= 0).all())
        self.assertAlmostEqual(eq.total_value.iloc[-1]-100000, sum(t["pnl_hkd"] for t in e.roundtrips))

    def test_future_high_low_close_cannot_resize_open_order(self):
        d = sample()
        a = engine(); _, ta = run(a, d)
        altered = copy.deepcopy(d)
        altered["00700"].loc[1, ["close", "signal_close", "high"]] = 108
        b = engine(); _, tb = run(b, altered)
        self.assertEqual(ta[0], tb[0])

    def test_close_stop_exits_next_open_and_gap_is_not_capped(self):
        d = sample()
        for c in ["close", "signal_close", "low"]:
            d["00700"].loc[2, c] = 90
        d["00700"].loc[3, ["open", "low"]] = 75
        e = engine(); _, t = run(e, d)
        self.assertEqual(t[1]["date"], d["00700"].date.iloc[3])
        self.assertEqual(t[1]["reason"], "stop_close")
        self.assertAlmostEqual(t[1]["price"], 74.925)
        self.assertLess(e.roundtrips[0]["return_net"], -.2)

    def test_low_alone_is_not_an_intraday_stop_fill(self):
        d = sample(); d["00700"].loc[2, "low"] = 50
        e = engine(); _, t = run(e, d)
        self.assertEqual(t[1]["reason"], "time_exit")

    def test_every_holding_limit_and_no_repeated_signal_clock_reset(self):
        d = sample()
        prep = signals(d)
        sig = prep[0][d["00700"].date.iloc[0]][0]
        for date in d["00700"].date.iloc[1:10]:
            prep[0][date] = [dict(sig, signal_date=date)]
        for hold in [5, 8, 10]:
            e = engine({"max_hold": hold}); run(e, d, prep)
            first = e.roundtrips[0]
            self.assertEqual(first["sessions"], hold)
            self.assertEqual(first["entry_date"], d["00700"].date.iloc[1])
            self.assertLessEqual(first["calendar_days"], 14)

    def test_known_long_holiday_exits_before_calendar_deadline(self):
        d = sample()
        cal = pd.Series(True, index=pd.date_range("2025-01-01", "2025-03-15"))
        cal.loc[cal.index.weekday >= 5] = False
        cal.loc["2025-01-08":"2025-01-20"] = False
        e = engine({"max_hold": 10}, cal); run(e, d)
        self.assertEqual(e.roundtrips[0]["exit_date"], pd.Timestamp("2025-01-07"))

    def test_halt_does_not_fake_exit_and_reports_overdue(self):
        d = sample()
        # 需要另一只股票提供市场日历，停牌股票本身缺行。
        d["00005"] = d["00700"].copy()
        d["00700"] = d["00700"].drop(index=range(5, 15)).reset_index(drop=True)
        prep = signals(d)
        prep[1]["00005"] = pd.Series(3., index=d["00005"].date)
        e = engine({"max_hold": 5}); e.sectors["00005"] = "bank"
        e.run(d, "2025-01-01", "2025-02-11", prep)
        self.assertEqual(e.roundtrips[0]["exit_date"], pd.Timestamp("2025-01-22"))
        self.assertTrue(e.roundtrips[0]["overdue"])
        self.assertEqual(len(e.audit), 1)

    def test_prefix_replay_same_account_and_trades(self):
        d = sample()
        full = engine(); eq, t = run(full, d)
        short = {c: f.iloc[:5].copy() for c, f in d.items()}
        cut = engine(); eq_short, t_short = run(cut, short)
        pd.testing.assert_frame_equal(eq.loc[eq_short.index], eq_short)
        self.assertEqual([x for x in t if x["date"] <= eq_short.index[-1]], t_short)
        self.assertEqual(len(cut.positions), 1)  # 不在回测结束日假平仓

    def test_profit_trim_is_integer_lots_and_keeps_original_deadline(self):
        d = sample()
        d["00700"].loc[2:, ["close", "signal_close", "high"]] = 115
        d["00700"].loc[3:, "open"] = 115
        e = engine(); e.p["take_profit_r"] = 2
        _, t = run(e, d)
        sells = [x for x in t if x["side"] == "sell"]
        self.assertEqual(sells[0]["reason"], "take_profit")
        self.assertTrue(all(x["shares"] % 100 == 0 for x in sells))
        self.assertEqual(sum(x["shares"] for x in sells), t[0]["shares"])
        self.assertEqual(e.roundtrips[0]["sessions"], 8)

    def test_catalyst_timestamp_cannot_be_used_before_publication(self):
        d = sample(100)["00700"]
        d.loc[80, ["close", "signal_close", "high", "open"]] = [104, 104, 104, 100]
        d.loc[80, "volume"] = 5000000
        p = parameters(dict(atr_min=.01, require_catalyst=True))
        date = d.date.iloc[80]
        events = pd.DataFrame([dict(code="00700", published_at=str(date.date())+" 18:00", approved=True)])
        c, _ = prepare({"00700": d}, p, events)
        self.assertNotIn(date, c)
        events.loc[0, "published_at"] = str(date.date())+" 12:00"
        c, _ = prepare({"00700": d}, p, events)
        self.assertIn(date, c)

    def test_feature_prefix_invariance(self):
        d = sample(110)
        p = parameters(dict(atr_min=.01))
        a, atr = prepare(d, p)
        cut = {c: f.iloc[:90] for c, f in d.items()}
        b, short_atr = prepare(cut, p)
        self.assertEqual({k:v for k,v in a.items() if k <= cut["00700"].date.iloc[-1]}, b)
        pd.testing.assert_series_equal(atr["00700"].loc[short_atr["00700"].index], short_atr["00700"])

    def test_no_unaffordable_entry_and_invalid_configuration(self):
        e = engine(capital=1000); _, t = run(e, sample())
        self.assertEqual(t, [])
        for invalid in [{"max_hold": 11}, {"max_calendar_days": 15}, {"risk_per_trade": -.1}]:
            with self.assertRaises(ValueError):
                parameters(invalid)

    def test_drawdown_pause_blocks_new_entries_but_keeps_exits(self):
        d=sample()
        d["00700"].loc[2,["close","signal_close","low"]]=60
        d["00700"].loc[3,["open","low"]]=60
        prep=signals(d)
        prep[0][d["00700"].date.iloc[10]]=prep[0][d["00700"].date.iloc[0]]
        e=engine();eq,t=run(e,d,prep)
        self.assertTrue(eq.paused.any())
        self.assertEqual(sum(x["side"]=="buy" for x in t),1)
        self.assertEqual(len(e.positions),0)

    def test_entry_day_does_not_bypass_cash_sector_or_planned_risk_cap(self):
        d=sample();d["00005"]=d["00700"].copy()
        prep=signals(d)
        first=d["00700"].date.iloc[0]
        prep[0][first].append(dict(prep[0][first][0],code="00005"))
        prep[1]["00005"]=pd.Series(3.,index=d["00005"].date)
        e=engine();e.sectors["00005"]="tech"
        e.run(d,"2025-01-01","2025-01-02",prep)
        total=sum(p.shares*p.entry for p in e.positions.values())
        risk=sum(p.shares*p.planned_risk_per_share for p in e.positions.values())
        self.assertLessEqual(total,100000*.45)
        self.assertLessEqual(risk,100000*.06)

    def test_pullback_positive_fixture(self):
        d=sample(100)["00700"]
        for i in range(71,81):
            d.loc[i,["open","close","signal_close"]]=100+2*(i-70)
            d.loc[i,"high"]=102+2*(i-70)
            d.loc[i,"low"]=98+2*(i-70)
        for i,price in [(81,118),(82,117),(83,121)]:
            d.loc[i,["open","close","signal_close"]]=price
            d.loc[i,"high"]=price+2;d.loc[i,"low"]=price-2
        d.loc[80:82,"volume"]=500000
        d.loc[83,"volume"]=1500000
        p=parameters(dict(kind="pullback",atr_min=.01))
        c,_=prepare({"00700":d},p)
        self.assertIn(d.date.iloc[83],c)

    def test_factor_jump_after_halt_is_not_silently_marked_as_profit(self):
        d=sample();d["00005"]=d["00700"].copy()
        d["00700"]=d["00700"].drop(index=[2,3]).reset_index(drop=True)
        d["00700"].loc[2:,"signal_close"]*=3
        prep=signals(d);e=engine();e.sectors["00005"]="bank"
        with self.assertRaisesRegex(ValueError,"公司行为"):
            e.run(d,"2025-01-01","2025-01-20",prep)

    def test_calendar_deadline_never_uses_close_after_fourteen_full_days(self):
        cal=pd.Series(False,index=pd.date_range("2025-01-01","2025-02-01"))
        cal.loc[["2025-01-02","2025-01-16"]]=True
        e=engine({"max_hold":10},cal)
        self.assertEqual(e._deadline(pd.Timestamp("2025-01-02")),pd.Timestamp("2025-01-02"))


if __name__ == "__main__":
    unittest.main()
