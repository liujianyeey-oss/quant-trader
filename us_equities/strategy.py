"""Causal close-confirmed technical candidates. Gap is NOT an earnings label."""
import numpy as np
import pandas as pd
from .data import normalize

DEFAULTS=dict(kind='breakout',volume_ratio=1.5,max_hold=8,market_filter=True,
    risk_per_trade=.02,max_weight=.30,max_gross=.95,max_positions=4,sector_cap=.50,
    portfolio_risk=.06,atr_min=.03,atr_max=.08,min_price=10.,min_turnover=5e7,
    min_stop=.04,max_stop=.10,stop_atr=1.5,trail_atr=2.,stall_days=3,stall_r=.5,
    take_profit_r=2.,gap_cap=.03,reduce_drawdown=.08,pause_drawdown=.15,pause_sessions=0,momentum_days=20)

def parameters(overrides=None):
    p=dict(DEFAULTS)
    if set(overrides or {})-set(p):raise ValueError('Unknown US parameters')
    p.update(overrides or {})
    if p['kind'] not in {'breakout','pullback','gap','rotation'}:raise ValueError('Unknown US strategy kind; no verified earnings data')
    if type(p['market_filter']) is not bool:raise ValueError('market_filter must be bool')
    for k in ['max_hold','max_positions','stall_days','momentum_days']:
        if type(p[k]) is not int or p[k]<1:raise ValueError(k+' must be positive integer')
    if type(p['pause_sessions']) is not int or not 0<=p['pause_sessions']<=60:raise ValueError('pause_sessions must be integer 0..60')
    if p['max_hold']>10:raise ValueError('Maximum hold is 10 sessions and 14 calendar dates')
    for k,v in p.items():
        if k not in {'kind','market_filter'} and (not np.isfinite(v) or v<0):raise ValueError('Invalid '+k)
    if not(0<p['min_stop']<=p['max_stop']<1 and 0<p['atr_min']<=p['atr_max']<1 and
           0<p['risk_per_trade']<=p['portfolio_risk']<1 and 0<p['max_weight']<=p['max_gross']<=1 and
           0<p['sector_cap']<=1 and 0<p['reduce_drawdown']<p['pause_drawdown']<1):raise ValueError('Invalid risk relationship')
    return p

def prepare(data,p):
    features={}; momentum={}; eligible={};atrs={}
    for code,frame in sorted(data.items()):
        d=normalize(frame).set_index('date'); f=d.factor
        # Split-only adjusted units. All indicators are homogeneous in this unit;
        # convert back to actual signal-date USD before exposing an order level.
        c,h,l,o=d.close*f,d.high*f,d.low*f,d.open*f
        vol=d.volume/f
        atr=pd.concat([h-l,(h-c.shift()).abs(),(l-c.shift()).abs()],axis=1).max(axis=1).rolling(14).mean()
        adv=(d.close*d.volume).rolling(20).mean()
        mom=c/c.shift(p['momentum_days'])-1; momentum[code]=mom
        e=(d.close>=p['min_price'])&(adv>=p['min_turnover'])&(atr/c).between(p['atr_min'],p['atr_max'])
        e &= pd.Series(np.arange(len(d))>=120,index=d.index)
        # Quarantined provider bars are not imputed. Suppress new signals for 60
        # calendar days afterwards, covering every short technical window.
        for bad in frame.attrs.get('quarantined',[]):
            bad_date=pd.Timestamp(bad['date'])
            e &= ~pd.Series((d.index>=bad_date)&(d.index<=bad_date+pd.Timedelta(days=60)),index=d.index)
        eligible[code]=e
        high20=h.shift().rolling(20).max()
        if p['kind']=='breakout':
            trigger=(c>high20)&(vol>=vol.shift().rolling(20).mean()*p['volume_ratio'])
            trigger &= ((c-l)/(h-l).replace(0,np.nan)>=.75)&(c-high20<=atr)
            level=high20; invalid=high20-.5*atr
        elif p['kind']=='pullback':
            # Three-day strong-stock pullback proxy, not a verified first pullback after B.
            impulse=c.shift(3)/c.shift(13)-1
            peak=h.shift(3).rolling(5).max()
            trigger=(impulse>=.10)&((peak-c.shift())/atr.shift()).between(.5,1.5)
            trigger &= (vol.shift().rolling(3).mean()<vol.shift(4).rolling(20).mean()*.8)
            trigger &= (c>h.shift())&(c>c.rolling(20).mean())&(vol>=vol.shift().rolling(3).mean()*p['volume_ratio'])
            level=h.shift();invalid=l.shift().rolling(3).min()-.1*atr
        elif p['kind']=='rotation':
            # Continuous relative-strength participation, no requirement for a rare volume spike.
            trigger=(mom>0)&(c>c.rolling(20).mean())&(c/c.shift(5)>1)
            level=c-.5*atr;invalid=c-1.5*atr
        else:
            trigger=(o/c.shift()-1).between(.04,.15)&(vol>=vol.shift().rolling(20).mean()*p['volume_ratio'])
            trigger &= (c>=o)&((c-l)/(h-l).replace(0,np.nan)>=.75)
            level=c-.5*atr;invalid=l-.1*atr
        values=pd.DataFrame(dict(close=d.close,atr=atr/f,level=level/f,invalid=invalid/f,
                                  adv=adv,score=mom/(atr/c),factor=f))
        features[code]=(values,trigger)
        atrs[code]=atr/f
    ranks=pd.DataFrame(momentum).where(pd.DataFrame(eligible)).rank(axis=1,pct=True)
    candidates={}
    for code,(values,trigger) in features.items():
        select=eligible[code]&trigger
        if p['kind'] in {'breakout','rotation'}:select &= ranks[code].reindex(values.index)>=.9
        for date,row in values.loc[select].iterrows():
            candidates.setdefault(date,[]).append(dict(code=code,**row.to_dict()))
    for rows in candidates.values():rows.sort(key=lambda x:(-x['score'],x['code']))
    return candidates,atrs
