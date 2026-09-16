"""Sina raw daily OHLCV plus vendor split factors. No eval on remote responses."""
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo
import hashlib
import json
import re
import time
import numpy as np
import pandas as pd
import requests
import exchange_calendars as xc

DAILY='https://stock.finance.sina.com.cn/usstock/api/json_v2.php/US_MinKService.getDailyK'
FACTOR='https://finance.sina.com.cn/us_stock/company/reinstatement/{}_qfq.js'
ALIASES={'META':('FB','2022-06-09'),'XYZ':('SQ','2025-01-21')}

def calendar(start,end):
    return xc.get_calendar('XNYS',start=start,end=end).sessions.tz_localize(None)

def normalize(d):
    d=d.copy(); d['date']=pd.to_datetime(d.date).dt.normalize()
    keys=['open','high','low','close','volume','factor']
    d[keys]=d[keys].apply(pd.to_numeric,errors='raise')
    if d.date.duplicated().any() or not np.isfinite(d[keys].to_numpy()).all():
        raise ValueError('Duplicate dates or invalid US bars')
    if (d[['open','high','low','close','factor']]<=0).any().any() or (d.volume<0).any():
        raise ValueError('Nonpositive US price/factor or negative volume')
    if ((d.high<d[['open','low','close']].max(axis=1)-1e-6)|(d.low>d[['open','high','close']].min(axis=1)+1e-6)).any():
        raise ValueError('Invalid US OHLC relationship')
    return d.sort_values('date').reset_index(drop=True)

def parse(raw,factor_text):
    d=pd.DataFrame(raw).rename(columns={'d':'date','o':'open','h':'high','l':'low','c':'close','v':'volume','a':'amount'})
    if d.empty: raise ValueError('Empty price response')
    f=json.JSONDecoder().raw_decode(factor_text.split('=',1)[1].lstrip())[0]['data']
    f=pd.DataFrame(f).rename(columns={'d':'date','f':'factor','c':'dividend_adjustment'})
    if f.empty: raise ValueError('Empty factor history; do not assume no split')
    d['date']=pd.to_datetime(d.date); f['date']=pd.to_datetime(f.date)
    if f.date.duplicated().any(): raise ValueError('Duplicate factor dates')
    # Forward as-of only: an IPO cannot inherit a future factor by backfill.
    d=pd.merge_asof(d.sort_values('date'),f.sort_values('date'),on='date',direction='backward')
    # Prices before the first available factor cannot be used; retain explicit coverage metadata.
    d=d.dropna(subset=['factor'])
    # Fixed research cache floor, declared before returns: 2017 provides full 2019 warm-up.
    d=d[d.date>=pd.Timestamp('2017-01-01')]
    d[['open','high','low','close']]=d[['open','high','low','close']].astype(float)
    bad=(d.high<d[['open','low','close']].max(axis=1)-1e-6)|(d.low>d[['open','high','close']].min(axis=1)+1e-6)
    quarantine=d.loc[bad,['date','open','high','low','close']].to_dict('records')
    if len(quarantine)>5:raise ValueError('Too many invalid OHLC bars; require a better source')
    d=normalize(d.loc[~bad]);d.attrs['quarantined']=quarantine
    return d

def load(symbol,cache,offline=False):
    if not re.fullmatch('[A-Z][A-Z0-9.-]{0,9}',symbol):raise ValueError('Invalid US symbol')
    cache=Path(cache);cache.mkdir(parents=True,exist_ok=True)
    path=cache/f'{symbol}.csv'; meta=cache/f'{symbol}.json'
    if path.exists():
        d=normalize(pd.read_csv(path))
        if meta.exists():d.attrs['quarantined']=json.loads(meta.read_text()).get('quarantined',[])
        return d
    if offline:raise FileNotFoundError(path)
    error=None
    for attempt in range(2):
        try:
            r=requests.get(DAILY,params={'symbol':symbol.lower(),'___qn':3},timeout=25);r.raise_for_status()
            f=requests.get(FACTOR.format(symbol),timeout=25);f.raise_for_status()
            raw=r.json();alias_hash=None
            if symbol in ALIASES:
                alias,boundary=ALIASES[symbol]
                a=requests.get(DAILY,params={'symbol':alias.lower(),'___qn':3},timeout=25);a.raise_for_status()
                # Restrict the old ticker to its verified issuer interval: FB was reused later.
                raw=[x for x in a.json() if x['d']<boundary]+[x for x in raw if x['d']>=boundary]
                alias_hash=hashlib.sha256(a.content).hexdigest()
            d=parse(raw,f.text)
            # Cache only completed prior New York dates, conservative even after today's close.
            today=pd.Timestamp(datetime.now(ZoneInfo('America/New_York')).date())
            d=d[d.date<today]
            d.to_csv(path.with_suffix('.tmp'),index=False);path.with_suffix('.tmp').replace(path)
            meta.write_text(json.dumps(dict(symbol=symbol,fetched_at=datetime.now(ZoneInfo('UTC')).isoformat(),
                source=DAILY,factor_source=FACTOR.format(symbol),quarantined=d.attrs.get('quarantined',[]),alias=ALIASES.get(symbol),alias_sha256=alias_hash,raw_sha256=hashlib.sha256(r.content).hexdigest(),
                factor_sha256=hashlib.sha256(f.content).hexdigest(),first=str(d.date.min().date()),last=str(d.date.max().date())),indent=2,default=str))
            (cache/f'{symbol}_factors.json').write_text(json.dumps(json.JSONDecoder().raw_decode(f.text.split('=',1)[1].lstrip())[0]))
            return d
        except (ValueError,requests.RequestException) as exc:
            error=exc
            if attempt==0:time.sleep(1)
    raise RuntimeError(f'{symbol} fetch/validation failed: {error}') from error

def sectors(config):
    out={}
    for sector,names in config['sectors'].items():
        for symbol in names.split():
            if symbol in out:raise ValueError('Duplicate symbol '+symbol)
            out[symbol]=sector
    return out
