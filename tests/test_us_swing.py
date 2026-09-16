import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
import numpy as np
import pandas as pd
from us_equities.data import normalize,parse,calendar
from us_equities.strategy import parameters,prepare
from us_equities.engine import Engine,compile_market
from us_equities.cli import choose

COST=dict(commission_per_share=.005,minimum_commission=1.,sell_fee_bps=.5,slippage_bps=10)

class USTest(unittest.TestCase):
    def setUp(self):
        self.dates=pd.bdate_range('2024-01-01',periods=220)
        self.start=self.dates[140];self.end=self.dates[155]
        self.d=pd.DataFrame(dict(date=self.dates,open=100.,high=103.,low=97.,close=100.,volume=1e7,factor=1.))
        self.p=parameters(dict(market_filter=False,stall_days=20))
    def sig(self,code='A',i=139):
        return dict(code=code,close=100.,atr=3.,level=99.,invalid=94.,adv=1e9,score=1.,factor=1.)
    def run_e(self,data=None,p=None,signals=None,end=None,cap=50000):
        data={'A':self.d.copy()} if data is None else data
        signals={self.dates[139]:[self.sig()]} if signals is None else signals
        e=Engine(p or self.p,{s:'sector' for s in data},self.dates,COST,cap)
        atr={s:pd.Series(3.,index=d.date) for s,d in data.items()}
        eq=e.run(data,self.d,self.start,end or self.end,(signals,atr))
        return e,eq
    def test_next_open_integer_shares_cash_and_conservation(self):
        e,eq=self.run_e();buy=e.trades[0]
        self.assertEqual(buy['date'],self.start);self.assertEqual(buy['signal_date'],self.dates[139])
        self.assertEqual(buy['shares'],int(buy['shares']));self.assertGreaterEqual(eq.settled_cash.min(),0)
        self.assertAlmostEqual(eq.equity.iloc[-1]-50000,sum(x['pnl'] for x in e.roundtrips),6)
    def test_future_close_does_not_resize_open(self):
        e,_=self.run_e();d=self.d.copy();d.loc[140,['close','high']]=[150,155]
        e2,_=self.run_e({'A':d});self.assertEqual(e.trades[0],e2.trades[0])
    def test_same_day_low_is_not_intraday_stop(self):
        d=self.d.copy();d.loc[140,'low']=50
        e,_=self.run_e({'A':d},end=self.dates[141]);self.assertEqual(len(e.trades),1)
    def test_stop_fills_next_open_with_gap(self):
        d=self.d.copy();d.loc[140,['close','low']]=[90,85];d.loc[141,['open','low']]=[75,70]
        e,_=self.run_e({'A':d});t=e.trades[1]
        self.assertEqual(t['date'],self.dates[141]);self.assertAlmostEqual(t['price'],75*.999)
    def test_unsettled_sale_proceeds_cannot_fund_new_entry(self):
        d=self.d.copy();d.loc[140,['close','low']]=[90,85]
        p=dict(self.p,max_weight=.95,sector_cap=1.,max_gross=1.,risk_per_trade=.5,portfolio_risk=.8)
        signals={self.dates[139]:[self.sig()],self.dates[140]:[self.sig('B')],self.dates[141]:[self.sig('B')]}
        e,eq=self.run_e({'A':d,'B':self.d.copy()},p,signals)
        buys=[x for x in e.trades if x['side']=='buy' and x['code']=='B']
        self.assertLessEqual(buys[0]['shares'],30) # only leftover settled cash, not A sale
        self.assertGreater(eq.loc[self.dates[141],'unsettled_cash'],0)
    def test_settlement_regime_and_us_holidays(self):
        sessions=calendar('2024-05-01','2024-07-31');e=Engine(self.p,{},sessions,COST)
        self.assertEqual(e.settlement(pd.Timestamp('2024-05-24')),pd.Timestamp('2024-05-29'))
        self.assertEqual(e.settlement(pd.Timestamp('2024-05-28')),pd.Timestamp('2024-05-29'))
        self.assertNotIn(pd.Timestamp('2024-07-04'),sessions)
    def test_split_preserves_nav_and_scales_shares(self):
        d=self.d.copy();d.loc[141:,['open','high','low','close']]/=2;d.loc[141:,'factor']=2;d.loc[141:,'volume']*=2
        e,eq=self.run_e({'A':d});e0,eq0=self.run_e()
        np.testing.assert_allclose(eq.equity.iloc[:3],eq0.equity.iloc[:3])
        self.assertEqual(e.trades[1]['shares'],2*e.trades[0]['shares']);self.assertEqual(e.audit[0]['reason'],'inferred_split')
    def test_skip_old_signal_on_split_date(self):
        d=self.d.copy();d.loc[140:,['open','high','low','close']]/=2;d.loc[140:,'factor']=2
        e,_=self.run_e({'A':d});self.assertEqual(e.trades,[])
    def test_deadline_and_repeat_signals_do_not_extend(self):
        for h in [5,8,10]:
            sigs={date:[self.sig()] for date in self.dates[139:160]}
            e,_=self.run_e(p=dict(self.p,max_hold=h),signals=sigs)
            t=e.roundtrips[0];self.assertLessEqual(t['calendar_days'],13);self.assertEqual(t['sessions'],h)
    def test_halt_keeps_value_and_reports_overdue(self):
        d=self.d.drop(index=list(range(143,153)))
        e,eq=self.run_e({'A':d},p=dict(self.p,max_hold=5))
        self.assertGreater(eq.loc[self.dates[148],'equity'],49000)
        self.assertTrue(e.roundtrips[0]['overdue']);self.assertEqual(e.roundtrips[0]['exit_date'],self.dates[153])
    def test_partial_profit_keeps_deadline(self):
        d=self.d.copy();d.loc[141:,['open','high','low','close']]=[120,124,119,122]
        e,_=self.run_e({'A':d});sells=[x for x in e.trades if x['side']=='sell']
        self.assertEqual(sells[0]['reason'],'take_profit');self.assertEqual(sells[-1]['date'],self.dates[147])
        self.assertEqual(sum(x['shares'] for x in sells),e.trades[0]['shares'])
    def test_prefix_replay_no_end_liquidation(self):
        e,eq=self.run_e();cut=self.dates[143]
        e2,eq2=self.run_e({'A':self.d[self.d.date<=cut]},end=cut)
        pd.testing.assert_frame_equal(eq.loc[:cut],eq2)
        self.assertEqual(e2.trades,[x for x in e.trades if x['date']<=cut]);self.assertEqual(len(e2.positions),1)
    def test_features_prefix_and_future_split_unit_invariance(self):
        d=self.d.copy();d.loc[135:,'close']=np.arange(100,185);d['open']=d.close-.5;d['high']=d.close+1;d['low']=d.close-2
        p=parameters(dict(kind='breakout',atr_min=.001,volume_ratio=.1))
        all_c,_=prepare({'A':d},p);cut=self.dates[170]
        part,_=prepare({'A':d[d.date<=cut]},p)
        self.assertEqual({k:v for k,v in all_c.items() if k<=cut},part)
        d.factor=.1;c2,_=prepare({'A':d},p)
        self.assertEqual(set(c2),set(all_c))
        for date in all_c:self.assertAlmostEqual(all_c[date][0]['atr'],c2[date][0]['atr'])
    def test_safe_factor_parser_allows_provider_trailer(self):
        raw=[dict(d='2024-01-01',o='100',h='101',l='99',c='100',v='20')]
        d=parse(raw,'var A_qfq={"data":[{"d":"1900-01-01","f":"1","c":"0"}]};\ncallback();')
        self.assertEqual(d.factor.iloc[0],1)
    def test_rotation_generates_candidates_without_volume_spike(self):
        d=self.d.copy();d['close']=np.arange(100,320);d['open']=d.close-.5;d['high']=d.close+5;d['low']=d.close-5
        p=parameters(dict(kind='rotation',atr_min=.01,atr_max=.15))
        c,_=prepare({'A':d},p)
        self.assertGreater(len(c),50)
        self.assertIn(self.dates[150],c)
        prefix,_=prepare({'A':d.iloc[:170]},p)
        self.assertEqual({k:v for k,v in c.items() if k<=self.dates[169]},prefix)
    def test_bad_provider_bar_is_quarantined_not_repaired(self):
        raw=[dict(d='2024-01-01',o='100',h='101',l='99',c='100',v='20'),
             dict(d='2024-01-02',o='90',h='101',l='99',c='100',v='20')]
        d=parse(raw,'var A={"data":[{"d":"1900-01-01","f":"1","c":"0"}]}')
        self.assertEqual(len(d),1);self.assertEqual(len(d.attrs['quarantined']),1)
        self.assertEqual(d.attrs['quarantined'][0]['open'],90)
    def test_quarantine_blocks_future_signals_without_changing_past(self):
        d=self.d.copy();d.loc[135:,'close']=np.arange(100,185);d['open']=d.close-.5;d['high']=d.close+1;d['low']=d.close-2
        p=parameters(dict(kind='breakout',atr_min=.001,volume_ratio=.1))
        c,_=prepare({'A':d},p);bad=self.dates[160]
        d.attrs['quarantined']=[{'date':str(bad.date())}];q,_=prepare({'A':d},p)
        self.assertEqual({k:v for k,v in c.items() if k<bad},{k:v for k,v in q.items() if k<bad})
        self.assertFalse(any(bad<=k<=bad+pd.Timedelta(days=60) for k in q))
    def test_aggregate_risk_and_sector_caps(self):
        data={s:self.d.copy() for s in ['A','B','C','D']}
        sigs={self.dates[139]:[self.sig(s) for s in data]}
        e,eq=self.run_e(data,signals=sigs,end=self.dates[140])
        self.assertLessEqual(sum(p.shares*p.entry for p in e.positions.values()),50000*.5+1e-6)
        self.assertLessEqual(sum(p.shares*p.planned_risk for p in e.positions.values()),50000*.06+1e-6)
    def test_invalid_parameters_and_capital(self):
        for p in [{'max_hold':11},{'kind':'earnings'},{'market_filter':'true'},{'max_gross':1.1}]:
            with self.assertRaises(ValueError):parameters(p)
        with self.assertRaises(ValueError):self.run_e(cap=0)
    def test_selection_uses_training_only_and_neighborhood(self):
        records=[dict(candidate=i,kind='gap',market_filter=False,pause_sessions=0,score=s,qualified=True) for i,s in enumerate([.1,.2,.3])]
        _,selected=choose(records);self.assertEqual(selected,2)
        records[2]['qualified']=False;_,selected=choose(records);self.assertEqual(selected,1)
    def test_cooldown_waits_and_keeps_historical_losses(self):
        d=self.d.copy();d.loc[140,['close','low']]=[50,40]
        b=self.d.copy();b['close']=np.arange(100,320);b['open']=b.close;b['high']=b.close+2;b['low']=b.close-2
        p=dict(self.p,pause_drawdown=.10,pause_sessions=20)
        signals={x:[self.sig()] for x in self.dates[139:190]}
        atr={'A':pd.Series(3.,index=self.dates)}
        e=Engine(p,{'A':'x'},self.dates,COST)
        eq=e.run({'A':d},b,self.start,self.dates[190],(signals,atr))
        resume=[x for x in e.audit if x['reason']=='resume_after_cooldown']
        self.assertEqual(resume[0]['date'],self.dates[161])
        self.assertLess(eq.equity.iloc[0],45000)
        self.assertLess(e.stats(eq)['max_drawdown'],-.10)
        self.assertTrue(eq.loc[self.dates[141]:self.dates[160],'paused'].all())
    def test_compiled_execution_matches_uncached(self):
        e,eq=self.run_e();data={'A':self.d.copy()};pr=({self.dates[139]:[self.sig()]},{'A':pd.Series(3.,index=self.dates)})
        other=Engine(self.p,{'A':'sector'},self.dates,COST)
        eq2=other.run(data,self.d,self.start,self.end,pr,compile_market(data,self.d))
        pd.testing.assert_frame_equal(eq,eq2);self.assertEqual(e.trades,other.trades)
    def test_drawdown_pause_preserves_exit(self):
        d=self.d.copy();d.loc[140:,['open','high','low','close']]=[100,101,40,50]
        e,eq=self.run_e({'A':d},p=dict(self.p,pause_drawdown=.10))
        self.assertTrue(eq.paused.iloc[-1]);self.assertEqual(len(e.positions),0)

if __name__=='__main__':unittest.main()
