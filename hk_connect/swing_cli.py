"""短线回测、模型状态及训练段选参；结果落盘，不自动覆盖默认参数。"""
import copy
import hashlib
import itertools
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from .cli import get_universe, load_calendar
from .data import HKDataLoader, hk_code, today_hk
from .swing import SwingEngine, parameters, prepare

WARNINGS = [
    "纯量价技术策略；没有事件标签时不构成催化/业绩事件回测。",
    "当前 SH 股票名单及每手股数回填历史，观察池亦有人为选择偏差；不是全港股通或无偏样本。",
    "未配置真实通道日历时按周一至五计划到期、按港股实际有行情日执行；可能包含通道关闭日交易。",
    "HKD 价格收益：未计现金分红、红利税、人民币汇率、换汇价差及持有费用，采用固定当前费用情景。",
    "止损、盈利减仓和停滞退出均为收盘确认次日开盘；仅预定到期按当日收盘价减滑点假设执行。",
    "历史已被研究和多次查看；后段只是开发验证，不能声称独立样本外。",
]


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str, allow_nan=False), encoding="utf-8")


def read_parameters(path, defaults):
    if not path:
        return parameters(defaults)
    obj = json.loads(Path(path).read_text(encoding="utf-8"))
    return parameters(obj.get("params", obj))


def inputs(args, cfg, base):
    h = cfg["hk_swing"]
    if getattr(args, "code", None):
        raise ValueError("hk_swing 是组合策略，请修改 hk_swing.watchlist；--code 仅适用于 dual_ma")
    if not getattr(args, "research", False):
        raise ValueError("短线模块当前为开发研究版，请显式 --research；现有数据不能支持实盘复刻")
    start = getattr(args, "start", None) or h["start"]
    end = getattr(args, "end", None) or h["end"] or str(today_hk().date())
    p = read_parameters(getattr(args, "params", None), h["params"])
    u = get_universe(cfg, base, getattr(args, "offline", False))
    codes = [hk_code(c) for c in h["watchlist"]]
    if len(set(codes)) != len(codes) or set(codes)-set(u.records):
        raise ValueError("观察池重复或有股票不在资格文件中，需核对研究范围")
    if set(codes)-set(h["sectors"]):
        raise ValueError("观察池缺少题材分类")
    loader = HKDataLoader(Path(base)/cfg["hk_connect"]["cache_dir"])
    fetch_start = str((pd.Timestamp(start)-pd.Timedelta(days=max(210, p["lookback"]*3))).date())
    data = {}
    for i, c in enumerate(codes):
        print(f"短线行情 {i+1}/{len(codes)} {c}", flush=True)
        data[c] = loader.load(c, fetch_start, end, getattr(args, "offline", False))
        if data[c].empty or len(data[c][data[c].date < pd.Timestamp(start)]) < max(60, p["lookback"]):
            raise ValueError(f"{c} 起点前预热不足，不能静默缩小样本或推迟起点")
    bench = loader.load(h["benchmark"], fetch_start, end, getattr(args, "offline", False))
    if bench.empty:
        raise ValueError("基准无行情")
    calendar = load_calendar(Path(base)/cfg["hk_connect"]["calendar_csv"]) if cfg["hk_connect"].get("calendar_csv") else None
    events = None
    if h.get("events_csv"):
        events = pd.read_csv(Path(base)/h["events_csv"], dtype=str)
        if not {"code", "published_at", "approved", "source"} <= set(events) or not events.approved.isin(["0", "1"]).all():
            raise ValueError("事件 CSV 需 code,published_at,approved(0/1),source")
        events["code"] = events.code.map(hk_code)
        events["approved"] = events.approved.eq("1")
    return h, p, u, data, bench, calendar, events, start, end


def benchmark_returns(bench, dates):
    d = bench.set_index("date")
    if not dates.isin(d.index).all():
        raise ValueError("基准缺少回测日，拒绝错位比较")
    first_open = d.loc[dates[0], "open"]
    return d.loc[dates, "close"]/first_open


def engine(cfg, h, p, u, calendar, capital, slip=None):
    cost = dict(cfg["hk_connect"]["cost"])
    if slip is not None:
        cost["slippage"] = slip
    return SwingEngine(p, u, cost, h["sectors"], capital, calendar, research=True)


def run_one(cfg, h, p, u, data, calendar, events, start, end, capital, slip=None):
    e = engine(cfg, h, p, u, calendar, capital, slip)
    eq, trades = e.run(data, start, end, events=events)
    return e, eq, e.stats(eq)


def save_run(path, e, eq, stats, bench, p):
    path.mkdir(parents=True, exist_ok=True)
    norm = benchmark_returns(bench, eq.index)
    eq = eq.copy()
    eq["benchmark_price_nav"] = norm
    eq.to_csv(path/"equity_hkd.csv")
    pd.DataFrame(e.trades).to_csv(path/"trades.csv", index=False)
    pd.DataFrame(e.roundtrips).to_csv(path/"roundtrips.csv", index=False)
    pd.DataFrame(e.audit).to_csv(path/"execution_exceptions.csv", index=False)
    stats = dict(stats, benchmark_price_return=float(norm.iloc[-1]-1),
                 initial_cash_hkd=e.initial, actual_start=str(eq.index[0].date()), actual_end=str(eq.index[-1].date()))
    write_json(path/"summary.json", dict(stats=stats, params=p, open_positions=e.position_records(),
                                        last_candidates=e.last_candidates, warnings=WARNINGS))
    return stats


def run(args, cfg, base):
    h, p, u, data, bench, calendar, events, start, end = inputs(args, cfg, base)
    capital = getattr(args, "capital_hkd", None)
    capital = h["initial_cash_hkd"] if capital is None else capital
    output = Path(base)/"reports"/f"hk_swing_{datetime.now():%Y%m%d_%H%M%S_%f}"
    output.mkdir(parents=True, exist_ok=True)
    write_json(output/"provenance.json", dict(universe=u.payload, config=h, costs=cfg["hk_connect"]["cost"],
               currency="HKD", data_hashes={c:hashlib.sha256(d.to_csv(index=False).encode()).hexdigest() for c,d in data.items()},
               code_sha256=hashlib.sha256(Path(__file__).with_name("swing.py").read_bytes()).hexdigest()))
    if args.command == "optimize":
        return optimize(cfg, h, p, u, data, bench, calendar, events, start, end, capital, output)
    e, eq, stats = run_one(cfg, h, p, u, data, calendar, events, start, end, capital)
    result = save_run(output, e, eq, stats, bench, p)
    if args.command == "signal":
        if (today_hk()-eq.index[-1]).days > cfg["hk_connect"]["max_stale_days"]:
            raise ValueError("日线过期，不能输出当前模型状态")
        lines = [f"hk_swing {p['kind']}｜开发模拟状态，非实际账户｜HKD {capital:,.0f}",
                 f"数据截至 {eq.index[-1]:%Y-%m-%d}；截止日由 hk_swing.end/--end 决定",
                 f"暂停新增：{bool(eq.paused.iloc[-1])}；持仓 {len(e.positions)} 只"]
        for pos in e.positions.values():
            lines.append(f"{pos.code} {pos.shares} 股，首次成交 {pos.entered:%Y-%m-%d}，到期 {pos.deadline:%Y-%m-%d}，"
                         f"失效位 {pos.stop:.3f}，待执行 {pos.pending or '无'}")
        lines.append("最近收盘技术候选（尚未通过下一开盘的资格/跳空/现金检查）：" +
                     ", ".join(x["code"] for x in e.last_candidates))
        lines.extend(WARNINGS)
        target = Path(base)/"signals"/f"{today_hk():%Y-%m-%d}_hk_swing.txt"
        target.parent.mkdir(exist_ok=True)
        target.write_text("\n".join(lines)+"\n", encoding="utf-8")
        print("\n".join(lines))
    print(json.dumps(result, ensure_ascii=False, indent=2))
    print("报告：", output)
    for line in WARNINGS:
        print("说明：", line)


def optimize(cfg, h, baseline, u, data, bench, calendar, events, start, end, capital, output):
    train_end = pd.Timestamp(h["train_end"])
    split = pd.Timestamp(h["train_split"])
    if not pd.Timestamp(start) < split < train_end < pd.Timestamp(end):
        raise ValueError("需要 start < train_split < train_end < end，优化禁止使用后段选参")
    # 训练函数只收到 train_end 及更早行情，物理阻断后段数据参与选择。
    train = {c:d[d.date <= train_end].copy() for c,d in data.items()}
    folds = [(start, str((split-pd.Timedelta(days=1)).date())), (str(split.date()), str(train_end.date()))]
    records, configs = [], []
    for kind, lookbacks in [("breakout", [10,20]), ("pullback", [20])]:
        for lookback, vr, hold in itertools.product(lookbacks, [1.5,1.8], [5,8,10]):
            p = dict(baseline, kind=kind, lookback=lookback, volume_ratio=vr, max_hold=hold)
            configs.append(parameters(p))
    for i, p in enumerate(configs):
        scores, trips, returns = [], [], []
        for a,b in folds:
            _, _, s = run_one(cfg,h,p,u,train,calendar,events,a,b,capital)
            scores.append(s["total_return"] + 1.5*s["max_drawdown"])
            trips.append(s["roundtrips"])
            returns.append(s["total_return"])
        qualified = min(trips) >= 3 and sum(trips) >= 10
        records.append(dict(candidate=i, kind=p["kind"], lookback=p["lookback"], volume_ratio=p["volume_ratio"],
                            max_hold=p["max_hold"], train_score=float(np.mean(scores)),
                            fold1_return=returns[0], fold2_return=returns[1], train_roundtrips=sum(trips), qualified=qualified))
        print(f"训练候选 {i+1}/{len(configs)} 完成", flush=True)
    rank = pd.DataFrame(records)
    # 同类相邻持有期/量比共享一半评分，降低单个孤峰的排序优势。
    neighbors = rank.groupby(["kind","lookback"]).train_score.transform("median")
    rank["robust_score"] = .5*rank.train_score+.5*neighbors
    rank = rank.sort_values(["qualified","robust_score","candidate"], ascending=[False,False,True])
    rank.to_csv(output/"training_candidates.csv", index=False)
    qualified = rank[rank.qualified]
    chosen = configs[int(qualified.iloc[0].candidate)] if len(qualified) else baseline
    write_json(output/"selected_params.json", dict(params=chosen, selected=bool(len(qualified)),
               selection="仅训练段均值风险调整收益+同类邻域中位数；不自动启用", train_end=str(train_end.date())))
    results = []
    for name,p in [("baseline",baseline),("selected",chosen)]:
        for segment,a,b in [("train",start,str(train_end.date())),("validation",str((train_end+pd.Timedelta(days=1)).date()),end),("full",start,end)]:
            e, eq, s = run_one(cfg,h,p,u,data,calendar,events,a,b,capital)
            s = save_run(output/f"{name}_{segment}",e,eq,s,bench,p)
            results.append(dict(model=name,segment=segment,**s))
    pd.DataFrame(results).to_csv(output/"comparison.csv",index=False)
    # 稳健性只在参数已锁定后报告，禁止反馈到选择。
    stress = []
    for cash,slip in itertools.product([100000,500000,1000000],[.001,.002,.003]):
        _,eq,s = run_one(cfg,h,chosen,u,data,calendar,events,start,end,cash,slip)
        stress.append(dict(capital_hkd=cash,slippage_bps=int(slip*10000),**s))
    pd.DataFrame(stress).to_csv(output/"capital_slippage.csv",index=False)
    # 逐星期触发检查（不是挑一个收益最高的星期代替每日扫描）。
    all_candidates, atr = prepare(data,chosen,events)
    phase = []
    for weekday in range(5):
        e=engine(cfg,h,chosen,u,calendar,capital)
        eq,_=e.run(data,start,end,prepared=({d:v for d,v in all_candidates.items() if d.weekday()==weekday},atr))
        phase.append(dict(signal_weekday=weekday,**e.stats(eq)))
    pd.DataFrame(phase).to_csv(output/"signal_weekday_audit.csv",index=False)
    table=pd.DataFrame(results)
    full_engine,_,full_stats=run_one(cfg,h,chosen,u,data,calendar,events,start,end,capital)
    contributions=pd.DataFrame(full_engine.roundtrips)
    if not contributions.empty:
        grouped=contributions.groupby("code").agg(roundtrips=("pnl_hkd","size"),pnl_hkd=("pnl_hkd","sum"))
        grouped.sort_values("pnl_hkd",ascending=False).to_csv(output/"contributions.csv")
    v=table[(table.model=="selected") & (table.segment=="validation")].iloc[0]
    failures=[]
    if v.total_return <= 0:
        failures.append("后段开发验证累计收益非正")
    if qualified.empty or float(qualified.iloc[0].robust_score) <= 0:
        failures.append("训练邻域风险调整评分非正或无合格候选")
    if full_stats["roundtrips"] < 50:
        failures.append("完整回合少于 50，仅作样本稀疏提示，不是充分统计标准")
    write_json(output/"assessment.json",dict(research_only=True,automatic_promotion=False,
                                           findings=failures,selected_changed=chosen!=baseline))
    plot_comparison(output, capital)
    print(table[["model","segment","total_return","max_drawdown","roundtrips","benchmark_price_return"]].to_string(index=False))
    print("选出的开发参数：",chosen)
    print("报告：",output)
    for line in WARNINGS:
        print("说明：",line)
    return output


def plot_comparison(path, capital):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    a=pd.read_csv(path/"baseline_full/equity_hkd.csv",parse_dates=["date"]).set_index("date")
    b=pd.read_csv(path/"selected_full/equity_hkd.csv",parse_dates=["date"]).set_index("date")
    fig,(ax,dd)=plt.subplots(2,1,figsize=(11,7),sharex=True,gridspec_kw={"height_ratios":[2,1]})
    ax.plot(a.index,a.total_value/capital,label="Baseline",linewidth=2)
    ax.plot(b.index,b.total_value/capital,label="Train-selected",linestyle="--")
    ax.plot(a.index,a.benchmark_price_nav,label="2800 price benchmark (100% exposure)",alpha=.65)
    ax.set_title("HK swing development study | HKD price returns, biased current universe")
    ax.set_ylabel("NAV / initial capital");ax.legend();ax.grid(alpha=.2)
    draw=b.total_value/b.total_value.cummax().clip(lower=capital)-1
    dd.fill_between(draw.index,draw.values*100,0,color="#b34c54",alpha=.5)
    dd.set_ylabel("Strategy drawdown %");dd.grid(alpha=.2)
    fig.tight_layout();fig.savefig(path/"equity_comparison.png",dpi=160);plt.close(fig)
