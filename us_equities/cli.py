"""Frozen US research experiment, training-only selection and subsequent diagnostics."""
from datetime import datetime
from pathlib import Path
import hashlib
import itertools
import json
import numpy as np
import pandas as pd
import yaml
from .data import load,sectors,calendar
from .strategy import parameters,prepare
from .engine import Engine,compile_market

WARNINGS=[
 '118-name predeclared current survivor watchlist; selection/survivorship bias, not the historical US market.',
 'Gap is a price-volume signal, NOT a verified earnings strategy. Pullback is a three-day proxy, not verified first pullback.',
 'Limited vendor history for some stocks; invalid OHLC quarantined with 60-calendar-day entry suppression. Sina raw OHLCV and inferred split factors; no dividend cash, FX, cash interest or personal taxes. Vendor corporate actions are not independently audited.',
 'Daily close-confirmed signals trade next open; only planned expiry assumes close execution. Halt/slippage can exceed risk and time limits.',
 'Cash-only settled funds, T+2 before 2024-05-28 and T+1 after. NYSE sessions proxy settlement business days; broker differences unmodeled.',
 'Fixed illustrative commission/sell fee and slippage scenarios, not historical broker fee statements.',
 'Historical development study with exposed periods; no claim of independent out-of-sample performance or global optimum.',
]

def write(path,obj):Path(path).write_text(json.dumps(obj,ensure_ascii=False,indent=2,default=str,allow_nan=False))

def inputs(args,base):
    root=Path(base)
    config_path=Path(args.config) if getattr(args,'config',None) else root/'us_equities/config.json'
    cfg=yaml.safe_load(config_path.read_text())
    if not args.research:raise ValueError('US study requires --research due to explicit data limitations')
    if getattr(args,'code',None) or getattr(args,'capital_hkd',None) is not None:raise ValueError('Use US watchlist and --capital-usd')
    p=parameters(cfg['params'])
    if getattr(args,'params',None):
        obj=json.loads(Path(args.params).read_text());p=parameters(obj.get('params',obj))
    start=getattr(args,'start',None) or cfg['start'];end=getattr(args,'end',None) or cfg['end']
    if pd.Timestamp(start)<pd.Timestamp('2019-01-01'):raise ValueError('US cache research supports start >= 2019-01-01')
    cap=getattr(args,'capital_usd',None);cap=cfg['initial_cash_usd'] if cap is None else cap
    names=sectors(cfg);data={};cache=root/'data_cache/us_equities'
    for i,symbol in enumerate(list(names)+['QQQ','SPY']):
        d=load(symbol,cache,args.offline)
        d=d[d.date.between(pd.Timestamp(start)-pd.Timedelta(days=400),pd.Timestamp(end))].copy()
        if d.empty or d.date.max()<pd.Timestamp(end)-pd.Timedelta(days=7):raise ValueError(symbol+' missing/stale coverage; no silent exclusions')
        data[symbol]=d
        if (i+1)%20==0:print(f'US data {i+1}/{len(names)+2}',flush=True)
    sessions=calendar(str((pd.Timestamp(start)-pd.Timedelta(days=420)).date()),str((pd.Timestamp(end)+pd.Timedelta(days=40)).date()))
    # Snapshot metadata before simulations. This is not historical membership proof.
    return cfg,p,names,{s:data[s] for s in names},{s:data[s] for s in ['QQQ','SPY']},sessions,start,end,cap

def one(cfg,p,names,data,bench,sessions,start,end,cap,prepared=None,slip=None,compiled=None):
    cost=dict(cfg['cost'])
    if slip is not None:cost['slippage_bps']=slip
    e=Engine(p,names,sessions,cost,cap);eq=e.run(data,bench['QQQ'],start,end,prepared,compiled)
    return e,eq,e.stats(eq)

def save(path,e,eq,stats,bench,p):
    path=Path(path);path.mkdir(parents=True,exist_ok=True);eq=eq.copy()
    for symbol,d in bench.items():
        b=d.set_index('date');unit=b.close*b.factor;initial=b.loc[eq.index[0],'open']*b.loc[eq.index[0],'factor']
        eq[symbol+'_price_nav']=unit.reindex(eq.index)/initial
    eq.to_csv(path/'equity_usd.csv');pd.DataFrame(e.trades).to_csv(path/'trades.csv',index=False)
    pd.DataFrame(e.roundtrips).to_csv(path/'roundtrips.csv',index=False);pd.DataFrame(e.audit).to_csv(path/'execution_audit.csv',index=False)
    stats=dict(stats,capital_usd=e.initial,actual_start=str(eq.index[0].date()),actual_end=str(eq.index[-1].date()),
               qqq_price_return=float(eq.QQQ_price_nav.iloc[-1]-1),spy_price_return=float(eq.SPY_price_nav.iloc[-1]-1))
    write(path/'summary.json',dict(stats=stats,params=p,positions=e.position_records(),last_candidates=e.last_candidates,warnings=WARNINGS))
    return stats

def choose(records):
    rank=pd.DataFrame(records)
    neighborhood=rank.groupby(['kind','market_filter','pause_sessions']).score.transform('median')
    rank['robust_score']=.5*rank.score+.5*neighborhood
    rank=rank.sort_values(['qualified','robust_score','candidate'],ascending=[False,False,True])
    good=rank[rank.qualified]
    return rank,None if good.empty else int(good.iloc[0].candidate)

def optimize(cfg,baseline,names,data,bench,sessions,start,end,cap,out):
    train_end=pd.Timestamp(cfg['train_end']);valid_end=pd.Timestamp(cfg['validation_end'])
    if not pd.Timestamp(start)<pd.Timestamp('2021-01-01')<train_end<valid_end<pd.Timestamp(end):raise ValueError('Need training folds and two later periods')
    train={s:d[d.date<=train_end].copy() for s,d in data.items()}
    training_bench={s:d[d.date<=train_end].copy() for s,d in bench.items()}
    folds=[(start,'2020-12-31'),('2021-01-01',str(train_end.date()))]
    grid=[parameters(dict(baseline,kind=k,volume_ratio=v,max_hold=h,market_filter=f,pause_sessions=ps)) for k,v,h,f,ps in
          itertools.product(['breakout','pullback','gap'],[1.5,2.0],[5,8,10],[False,True],[0,20])]
    grid += [parameters(dict(baseline,kind='rotation',volume_ratio=1.5,momentum_days=m,max_hold=h,market_filter=f,pause_sessions=20))
             for m,h,f in itertools.product([10,20],[5,8,10],[False,True])]
    write(out/'experiment.json',dict(config=cfg,candidates=grid,folds=folds,selection='mean fold CAGR + 0.5*negative max drawdown; half neighborhood median',
          validation_end=str(valid_end.date()),note='Technical grid frozen first; cooldown added after baseline pause; rotation added after technical results. All are development research.'))
    compiled_train=compile_market(train,training_bench['QQQ']);compiled_full=compile_market(data,bench['QQQ'])
    prepared={};records=[]
    for i,p in enumerate(grid):
        key=(p['kind'],p['volume_ratio'],p['momentum_days'])
        if key not in prepared:prepared[key]=prepare(train,p)
        stats=[one(cfg,p,names,train,training_bench,sessions,a,b,cap,prepared[key],compiled=compiled_train)[2] for a,b in folds]
        qualified=min(x['roundtrips'] for x in stats)>=15
        records.append(dict(candidate=i,kind=p['kind'],volume_ratio=p['volume_ratio'],momentum_days=p['momentum_days'],max_hold=p['max_hold'],market_filter=p['market_filter'],pause_sessions=p['pause_sessions'],
            score=float(np.mean([s['cagr']+.5*s['max_drawdown'] for s in stats])),qualified=qualified,
            fold1_return=stats[0]['total_return'],fold2_return=stats[1]['total_return'],trades=sum(x['roundtrips'] for x in stats),
            fold1_pause=stats[0]['pause_date'],fold2_pause=stats[1]['pause_date']))
        print(f'US training {i+1}/{len(grid)}',flush=True)
    rank,idx=choose(records);rank.to_csv(out/'training_candidates.csv',index=False)
    chosen=baseline if idx is None else grid[idx]
    write(out/'selected_params.json',dict(params=chosen,qualified=idx is not None,train_end=str(train_end.date()),selection='training only',automatic_promotion=False))
    periods=[('train',start,str(train_end.date())),('validation',str((train_end+pd.Timedelta(days=1)).date()),str(valid_end.date())),
             ('recent',str((valid_end+pd.Timedelta(days=1)).date()),end),('full',start,end)]
    full_prepared={};results=[]
    # Each strategy family's training winner is retained, not selected from later returns.
    models={'baseline':baseline,'selected':chosen}
    for kind in ['breakout','pullback','gap','rotation']:
        sub=rank[(rank.kind==kind)&rank.qualified]
        if len(sub):models[kind+'_train_best']=grid[int(sub.iloc[0].candidate)]
    for name,p in models.items():
        key=(p['kind'],p['volume_ratio'],p['momentum_days'])
        if key not in full_prepared:full_prepared[key]=prepare(data,p)
        for segment,a,b in periods:
            e,eq,s=one(cfg,p,names,data,bench,sessions,a,b,cap,full_prepared[key],compiled=compiled_full)
            s=save(out/f'{name}_{segment}',e,eq,s,bench,p);results.append(dict(model=name,segment=segment,**s))
    pd.DataFrame(results).to_csv(out/'comparison.csv',index=False)
    stress=[];key=(chosen['kind'],chosen['volume_ratio'],chosen['momentum_days'])
    for cash,slip in itertools.product([15000,50000,150000],[10,25,50]):
        for segment,a,b in [('full',start,end),('recent',str((valid_end+pd.Timedelta(days=1)).date()),end)]:
            _,_,s=one(cfg,chosen,names,data,bench,sessions,a,b,cash,full_prepared[key],slip,compiled=compiled_full)
            stress.append(dict(capital_usd=cash,slippage_bps=slip,segment=segment,**s))
    pd.DataFrame(stress).to_csv(out/'capital_slippage.csv',index=False)
    sensitivity=[]
    candidates,atrs=full_prepared[key]
    for weekday in range(5):
        pr=({d:v for d,v in candidates.items() if d.weekday()==weekday},atrs)
        _,_,s=one(cfg,chosen,names,data,bench,sessions,start,end,cap,pr,compiled=compiled_full)
        sensitivity.append(dict(signal_weekday=weekday,**s))
    pd.DataFrame(sensitivity).to_csv(out/'weekday_audit.csv',index=False)
    e,eq,s=one(cfg,chosen,names,data,bench,sessions,start,end,cap,full_prepared[key],compiled=compiled_full)
    trips=pd.DataFrame(e.roundtrips)
    if len(trips):
        trips.groupby('code').agg(trades=('pnl','size'),pnl=('pnl','sum')).sort_values('pnl',ascending=False).to_csv(out/'contributions.csv')
        total=float(trips.pnl.sum());top=float(trips.pnl.nlargest(5).sum())
    else:total=top=0.
    table=pd.DataFrame(results);selected=table[table.model=='selected'];later=selected[selected.segment.isin(['validation','recent'])]
    write(out/'assessment.json',dict(selected_changed=chosen!=baseline,qualified=idx is not None,
         robust_train_score=None if idx is None else float(rank[rank.candidate==idx].robust_score.iloc[0]),
         later_returns_positive=bool((later.total_return>0).all()),full_pnl=total,pnl_without_top5_trades=total-top,
         limitation='Top-5 subtraction is contribution arithmetic, not a re-simulation.',research_only=True,automatic_promotion=False))
    plot(out,cap)
    print(table[['model','segment','total_return','cagr','max_drawdown','roundtrips']].to_string(index=False));print('US report:',out)

def plot(out,capital):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    a=pd.read_csv(out/'baseline_full/equity_usd.csv',parse_dates=['date']).set_index('date')
    b=pd.read_csv(out/'selected_full/equity_usd.csv',parse_dates=['date']).set_index('date')
    fig,(ax,dd)=plt.subplots(2,1,figsize=(12,7),sharex=True,gridspec_kw={'height_ratios':[2,1]})
    ax.plot(a.index,a.equity/capital,label='Baseline');ax.plot(b.index,b.equity/capital,label='Train-selected',linewidth=2)
    ax.plot(a.index,a.QQQ_price_nav,label='QQQ price (100% invested)',alpha=.55);ax.plot(a.index,a.SPY_price_nav,label='SPY price (100% invested)',alpha=.55)
    ax.set_title('US cash-only development study | biased survivor watchlist | no dividends');ax.legend();ax.grid(alpha=.2)
    draw=b.equity/b.equity.cummax().clip(lower=capital)-1;dd.fill_between(b.index,draw*100,0,alpha=.45,color='firebrick')
    dd.set_ylabel('Selected drawdown %');dd.grid(alpha=.2);fig.tight_layout();fig.savefig(out/'equity_comparison.png',dpi=160);plt.close(fig)

def run(args,cfg_unused,base):
    cfg,p,names,data,bench,sessions,start,end,cap=inputs(args,base)
    out=Path(base)/'reports'/f'us_swing_{datetime.now():%Y%m%d_%H%M%S_%f}';out.mkdir(parents=True)
    files=list((Path(base)/'us_equities').glob('*.py'))
    write(out/'provenance.json',dict(config=cfg,params=p,start=start,end=end,capital_usd=cap,warnings=WARNINGS,
        code_hashes={x.name:hashlib.sha256(x.read_bytes()).hexdigest() for x in files},
        data_hashes={s:hashlib.sha256(d.to_csv(index=False).encode()).hexdigest() for s,d in (data|bench).items()},
        quarantined={s:d.attrs.get('quarantined',[]) for s,d in data.items()},
        coverage={s:dict(first=str(d.date.min().date()),last=str(d.date.max().date()),bars=len(d)) for s,d in data.items()}))
    if args.command=='optimize':return optimize(cfg,p,names,data,bench,sessions,start,end,cap,out)
    e,eq,s=one(cfg,p,names,data,bench,sessions,start,end,cap);s=save(out,e,eq,s,bench,p)
    if args.command=='signal':
        now=pd.Timestamp.now(tz='America/New_York').normalize().tz_localize(None)
        if (now-eq.index[-1]).days>7:raise ValueError('Stale US daily data')
        print('US simulated account, not broker holdings; data through',eq.index[-1].date())
        print('Positions:',e.position_records());print('Close candidates, not executable orders:',e.last_candidates)
    print(json.dumps(s,ensure_ascii=False,indent=2));print('US report:',out)
