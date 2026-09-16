"""USD cash, settlement queue, integer shares, split-aware positions, no intraday fills."""
from dataclasses import dataclass,asdict
import math
import numpy as np
import pandas as pd
from .data import normalize
from .strategy import parameters,prepare

@dataclass
class Position:
    code:str
    entered:pd.Timestamp
    deadline:pd.Timestamp
    shares:float
    entry:float
    stop:float
    risk:float
    planned_risk:float
    cost:float
    peak:float
    proceeds:float=0.
    days:int=1
    trimmed:bool=False
    pending:str=''
    overdue:bool=False

def compile_market(data,benchmark):
    frames={c:normalize(d).set_index('date') for c,d in data.items()}
    b=normalize(benchmark).set_index('date');price=b.close*b.factor
    return dict(frames=frames,bars={c:d.to_dict('index') for c,d in frames.items()},benchmark=b,
                regime=price>price.rolling(50).mean(),momentum=price/price.shift(20)-1)

class Engine:
    def __init__(self,p,sectors,sessions,cost,capital=50000):
        self.p=parameters(p);self.sectors=sectors;self.sessions=pd.DatetimeIndex(sessions)
        if self.sessions.has_duplicates or not self.sessions.is_monotonic_increasing:raise ValueError('Invalid calendar')
        self.cost=dict(cost);self.initial=float(capital)
        if not math.isfinite(self.initial) or self.initial<=0:raise ValueError('Capital must be positive USD')
        if any(not np.isfinite(v) or v<0 for v in self.cost.values()) or not 0<=self.cost['slippage_bps']<10000:raise ValueError('Invalid USD costs')
        self.slip=self.cost['slippage_bps']/10000

    def fee(self,n,price,sell=False):
        return round(max(self.cost['minimum_commission'],n*self.cost['commission_per_share'])+
                     (n*price*self.cost['sell_fee_bps']/10000 if sell else 0),2)

    def settlement(self,date):
        lag=1 if date>=pd.Timestamp('2024-05-28') else 2
        i=self.sessions.get_indexer([date])[0]
        if i<0 or i+lag>=len(self.sessions):raise ValueError('Calendar lacks settlement coverage')
        return self.sessions[i+lag]

    def deadline(self,date):
        d=self.sessions[(self.sessions>=date)&(self.sessions<=date+pd.Timedelta(days=13))]
        if len(d)==0 or self.sessions[-1]<date+pd.Timedelta(days=13):raise ValueError('Calendar lacks deadline coverage')
        return d[min(self.p['max_hold'],len(d))-1]

    def _sell(self,pos,reference,date,reason,partial=False):
        n=max(1,math.floor(pos.shares/3)) if partial else pos.shares
        n=min(n,pos.shares);price=reference*(1-self.slip);fee=self.fee(n,price,True)
        proceeds=n*price-fee;self.pending_cash.append((self.settlement(date),proceeds))
        pos.shares-=n;pos.proceeds+=proceeds
        self.trades.append(dict(date=date,code=pos.code,side='sell',shares=n,price=price,fee=fee,reason=reason,
                                settlement=self.settlement(date)))
        if partial:pos.trimmed=True
        if pos.shares<=1e-8:
            self.roundtrips.append(dict(code=pos.code,entry_date=pos.entered,exit_date=date,pnl=pos.proceeds-pos.cost,
                 return_net=pos.proceeds/pos.cost-1,calendar_days=(date-pos.entered).days,sessions=pos.days,
                 reason=reason,overdue=pos.overdue,deadline=pos.deadline))

    def run(self,data,benchmark,start,end,prepared=None,compiled=None):
        start,end=pd.Timestamp(start),pd.Timestamp(end)
        if start>end or start<pd.Timestamp('2017-09-05'):raise ValueError('Use ordered dates from T+2 era (2017-09-05+)')
        compiled=compile_market(data,benchmark) if compiled is None else compiled
        frames=compiled['frames']
        if not frames or set(frames)-set(self.sectors):raise ValueError('Missing data or sector')
        candidates,atrs=prepared if prepared is not None else prepare(data,self.p)
        bars=compiled['bars'];b=compiled['benchmark'];regime=compiled['regime'];market_momentum=compiled['momentum']
        dates=self.sessions[(self.sessions>=start)&(self.sessions<=end)]
        if dates.empty or not dates.isin(b.index).all():raise ValueError('Benchmark/calendar coverage missing')
        self.positions={};self.pending_cash=[];self.trades=[];self.roundtrips=[];self.audit=[]
        self.pause_date=None;cash=self.initial;peak=self.initial;paused=False;reduced=False
        resume_after=None;rehab_until=None
        marks={};factors={};last_seen={};closed={};rows=[]
        for c,d in frames.items():
            past=d[d.index<dates[0]]
            if len(past):
                marks[c]=past.close.iloc[-1];factors[c]=past.factor.iloc[-1];last_seen[c]=past.index[-1]
        prev=self.sessions[self.sessions<dates[0]][-1]
        for date in dates:
            # Re-entry only after a completed cooldown and yesterday's market recovery.
            # Reset control episode peak, never historical account equity or reported drawdown.
            if paused and self.p['pause_sessions'] and resume_after is not None and date>=resume_after and not self.positions:
                if bool(regime.get(prev,False)) and market_momentum.get(prev,-1)>0:
                    paused=False;reduced=False;peak=rows[-1]['equity']
                    ix=self.sessions.get_indexer([date])[0]
                    rehab_until=self.sessions[min(ix+10,len(self.sessions)-1)]
                    self.audit.append(dict(date=date,code='',reason='resume_after_cooldown'))
            settled=[x for x in self.pending_cash if x[0]<=date]
            cash+=sum(x[1] for x in settled);self.pending_cash=[x for x in self.pending_cash if x[0]>date]
            current={c:v[date] for c,v in bars.items() if date in v};exited=set()
            # Apply inferred split before any open trade. Dividend cash is explicitly excluded.
            for code,pos in list(self.positions.items()):
                pos.days+=1;r=current.get(code)
                if r is not None and code in factors:
                    ratio=r['factor']/factors[code]
                    if not np.isclose(ratio,1,rtol=1e-7):
                        if not .01<=ratio<=100:raise ValueError('Suspect split '+code)
                        pos.shares*=ratio;pos.entry/=ratio;pos.stop/=ratio;pos.risk/=ratio;pos.peak/=ratio;pos.planned_risk/=ratio
                        marks[code]/=ratio
                        # Fractional entitlements retained until sale; explicit research approximation.
                        self.audit.append(dict(date=date,code=code,reason='inferred_split',ratio=ratio))
                if date>pos.deadline:
                    pos.pending='deadline_retry'
                    if not pos.overdue:
                        self.audit.append(dict(date=date,code=code,reason='deadline_breach'));pos.overdue=True
                if r is not None and r['volume']>0 and pos.pending:
                    self._sell(pos,r['open'],date,pos.pending,pos.pending=='take_profit');exited.add(code)
                    if pos.shares<=1e-8:del self.positions[code];closed[code]=date
                    else:pos.pending=''
            opens={c:r['open'] for c,r in current.items() if r['volume']>0}
            receivable=sum((x[1] for x in self.pending_cash),0.)
            nav=cash+receivable+sum(pos.shares*opens.get(c,marks[c]) for c,pos in self.positions.items())
            bull=bool(regime.get(prev,False)); scale=.5 if reduced or (rehab_until is not None and date<rehab_until) else 1.
            gross_cap=self.p['max_gross']
            if self.p['market_filter'] and not bull:scale*=.5;gross_cap=min(gross_cap,.5)
            if not paused:
                for sig in candidates.get(prev,[]):
                    code=sig['code']
                    if len(self.positions)>=self.p['max_positions']:break
                    if code in self.positions or code in exited or code not in opens:continue
                    if (date-closed.get(code,pd.Timestamp('1900-01-01'))).days<=2:continue
                    # Skip a signal spanning a split; its USD levels refer to the preceding unit.
                    if not np.isclose(current[code]['factor'],sig['factor'],rtol=1e-7):continue
                    price=opens[code]
                    if price<sig['level'] or price>sig['close']+min(sig['close']*self.p['gap_cap'],.5*sig['atr']):continue
                    fill=price*(1+self.slip)
                    stop=min(sig['invalid'],fill-self.p['stop_atr']*sig['atr'],fill*(1-self.p['min_stop']))
                    if stop<=0 or (fill-stop)/fill>self.p['max_stop']:continue
                    gross=sum(p.shares*opens.get(c,marks[c]) for c,p in self.positions.items())
                    sector=sum(p.shares*opens.get(c,marks[c]) for c,p in self.positions.items() if self.sectors[c]==self.sectors[code])
                    used=sum(p.shares*p.planned_risk for p in self.positions.values())
                    budget=min(nav*self.p['risk_per_trade']*scale,max(0,nav*self.p['portfolio_risk']-used))
                    room=min(cash,nav*self.p['max_weight'],nav*gross_cap-gross,nav*self.p['sector_cap']-sector,sig['adv']*.001)
                    n=max(0,int(min(room/fill,budget/(fill-stop))))
                    while n:
                        fee=self.fee(n,fill);exit_price=stop*(1-self.slip)
                        risk=n*(fill-exit_price)+fee+self.fee(n,exit_price,True)
                        if n*fill+fee<=cash and risk<=budget:break
                        n-=1
                    if not n:continue
                    cost=n*fill+self.fee(n,fill);cash-=cost
                    self.positions[code]=Position(code,date,self.deadline(date),n,fill,stop,fill-stop,risk/n,cost,fill)
                    self.trades.append(dict(date=date,code=code,side='buy',shares=n,price=fill,fee=cost-n*fill,
                        reason='entry',signal_date=prev,settlement=self.settlement(date)))
            for c,r in current.items():marks[c]=r['close'];factors[c]=r['factor'];last_seen[c]=date
            for code,pos in list(self.positions.items()):
                r=current.get(code)
                if date>=pos.deadline:
                    if r is not None and r['volume']>0:
                        self._sell(pos,r['close'],date,'time_exit');del self.positions[code];closed[code]=date
                    else:pos.pending='deadline_retry'
                    continue
                if r is None:continue
                if pos.pending and pos.pending!='take_profit':continue
                c=r['close']
                if c<=pos.stop:pos.pending='stop_close'
                elif pos.days>=self.p['stall_days'] and max(pos.peak,c)-pos.entry<self.p['stall_r']*pos.risk:pos.pending='stall_exit'
                elif not pos.trimmed and c-pos.entry>=self.p['take_profit_r']*pos.risk:pos.pending='take_profit'
                pos.peak=max(pos.peak,c);a=atrs[code].get(date,np.nan)
                if pos.trimmed and pd.notna(a):pos.stop=max(pos.stop,pos.peak-self.p['trail_atr']*a)
            receivable=sum((x[1] for x in self.pending_cash),0.)
            gross=sum(p.shares*marks[c] for c,p in self.positions.items());nav=cash+receivable+gross
            peak=max(peak,nav);draw=nav/peak-1
            reduced=reduced or draw<=-self.p['reduce_drawdown']
            if not paused and draw<=-self.p['pause_drawdown']:
                paused=True;self.pause_date=date
                self.audit.append(dict(date=date,code='',reason='drawdown_pause'))
                ix=self.sessions.get_indexer([date])[0]+self.p['pause_sessions']+1
                resume_after=self.sessions[ix] if ix<len(self.sessions) else None
            if cash < -1e-7:raise AssertionError('Borrowing is forbidden')
            rows.append(dict(date=date,equity=nav,settled_cash=cash,unsettled_cash=receivable,gross_weight=gross/nav,
                             positions=len(self.positions),paused=paused,reduced=reduced))
            prev=date
        eq=pd.DataFrame(rows).set_index('date');self.last_candidates=candidates.get(dates[-1],[])
        self.stale_positions={c:str(last_seen[c].date()) for c in self.positions if last_seen[c]<dates[-1]}
        return eq

    def stats(self,eq):
        n=eq.equity;draw=n/n.cummax().clip(lower=self.initial)-1
        pnl=pd.Series([x['pnl'] for x in self.roundtrips],dtype=float);years=max((eq.index[-1]-eq.index[0]).days/365.25,1/365.25)
        cagr=(n.iloc[-1]/self.initial)**(1/years)-1
        return dict(total_return=float(n.iloc[-1]/self.initial-1),cagr=float(cagr),max_drawdown=float(draw.min()),
            mean_gross=float(eq.gross_weight.mean()),roundtrips=len(pnl),win_rate=float((pnl>0).mean()) if len(pnl) else None,
            profit_factor=float(pnl[pnl>0].sum()/-pnl[pnl<0].sum()) if (pnl<0).any() else None,
            fees_usd=float(sum(t['fee'] for t in self.trades)),max_calendar_days=max((t['calendar_days'] for t in self.roundtrips),default=0),
            max_sessions=max((t['sessions'] for t in self.roundtrips),default=0),
            deadline_breaches=sum(t['reason']=='deadline_breach' for t in self.audit),open_positions=len(self.positions),
            pause_date=str(self.pause_date.date()) if self.pause_date is not None else None,stale_positions=self.stale_positions,
            pause_events=sum(x['reason']=='drawdown_pause' for x in self.audit),resume_events=sum(x['reason']=='resume_after_cooldown' for x in self.audit))

    def position_records(self):return [asdict(p) for p in self.positions.values()]
