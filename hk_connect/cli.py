"""独立 CLI 路由；现有 A 股账户和 paper-update 不经过此模块。"""
from datetime import datetime
import json
from pathlib import Path

import pandas as pd

from .backtest import HKBacktester
from .data import HKDataLoader, Universe, hk_code, refresh_universe, save_json, today_hk
from .strategy import HKConnectMomentum

LIMITATIONS = [
    "HKD 未复权价格收益研究，未计现金分红、红利税、汇兑价差、人民币汇率损益、持有服务费。",
    "全部历史统一使用配置中的当前费用情景；不是逐时期实际账单。",
    "日线开盘加滑点仅为成交假设；未模拟逐笔流动性、最小价位、结算可用资金、碎股及临时通道限制。",
    "不支持拆并股等公司行为；检测到大幅复权因子变化时拒绝回测。小额分红缺失会压低价格收益。",
]


def _path(base, value):
    return Path(base) / value


def get_universe(cfg, base, offline=False):
    h = cfg["hk_connect"]
    path = _path(base, h["universe_file"])
    u = Universe.load(path) if path.exists() else None
    if u is not None and u.mode == "intervals":
        return u
    if not offline and (u is None or u.observed != today_hk()):
        print("更新沪港通可买股票名单及 HKEX 每手股数…", flush=True)
        u = refresh_universe(path)
    if u is None:
        raise ValueError("缺少港股通名单；先运行 python main.py hk-universe")
    if u.observed > today_hk():
        raise ValueError("名单抓取日期在未来")
    return u


def load_calendar(path):
    df = pd.read_csv(path, dtype={"is_open": str})
    if not {"date", "is_open"} <= set(df.columns) or not df.is_open.isin(["0", "1"]).all():
        raise ValueError("日历 CSV 必须为 date,is_open；is_open 只能为 0 或 1")
    dates = pd.to_datetime(df.date)
    if dates.duplicated().any():
        raise ValueError("通道日历日期重复")
    return pd.Series(df.is_open.eq("1").to_numpy(), index=dates).sort_index()


def run_universe(args, cfg, base):
    u = refresh_universe(_path(base, cfg["hk_connect"]["universe_file"]))
    print(f"已保存 {len(u.records)} 只沪港通可买 HKD 股票（SH 通道），观察日 {u.observed:%Y-%m-%d}。")


def run(args, cfg, base):
    h = cfg["hk_connect"]
    is_backtest = args.command == "backtest"
    research = getattr(args, "research", False)
    offline = getattr(args, "offline", False)
    if getattr(args, "code", None):
        raise ValueError("港股通是组合策略，请修改 hk_connect.watchlist；--code 仅适用于 dual_ma")
    u = get_universe(cfg, base, offline)
    warnings = list(LIMITATIONS)
    if is_backtest and u.mode == "snapshot":
        if not research:
            raise ValueError("当前名单回看历史存在幸存者偏差。请提供历史 intervals 名单，或加 --research 明确进入探索回测。")
        warnings.insert(0, "探索回测：使用今天名单和今天每手股数回看历史，存在幸存者偏差；不能证明历史可交易性或未来盈利。")
    if not is_backtest and u.mode == "snapshot" and (today_hk() - u.observed).days > h["max_stale_days"]:
        raise ValueError("港股通资格快照过期，拒绝生成当前目标；联网更新名单")
    end = getattr(args, "end", None) or str(today_hk().date())
    start = (getattr(args, "start", None) or h["start"]) if is_backtest else str((today_hk() - pd.Timedelta(days=400)).date())
    if pd.Timestamp(start) > pd.Timestamp(end):
        raise ValueError("回测开始日期晚于结束日期")
    strategy = HKConnectMomentum(cfg, u, research=research or u.mode == "snapshot")
    fetch_start = str((pd.Timestamp(start) - pd.Timedelta(days=max(400, strategy.warmup * 2))).date())
    watchlist = [hk_code(c) for c in h.get("watchlist", [])]
    unknown = set(watchlist) - set(u.records)
    if unknown:
        # 名单调整后用户应复核研究范围，不能自动将无资格股票纳入。
        warnings.append(f"观察池中已无名单记录的股票已排除：{sorted(unknown)}")
    codes = sorted(set(watchlist) & set(u.records)) if watchlist else sorted(u.records)
    if not codes:
        raise ValueError("观察池与港股通名单没有交集")
    if watchlist:
        warnings.append(f"仅研究配置中的 {len(codes)} 只观察池股票，非全港股通市场选优；观察池也可能有事后选择偏差。")
    calendar = load_calendar(_path(base, h["calendar_csv"])) if h.get("calendar_csv") else None
    if is_backtest and calendar is None:
        if not research:
            raise ValueError("缺少港股通交易日历；提供 calendar_csv 或使用 --research")
        warnings.append("探索回测：用港股行情日期近似通道交易日，可能模拟了港股通关闭日的成交。")
    loader = HKDataLoader(_path(base, h["cache_dir"]))
    data = {}
    for i, c in enumerate(codes, 1):
        print(f"港股行情 {i}/{len(codes)} {c}", flush=True)
        data[c] = loader.load(c, fetch_start, end, offline)
        if data[c].empty:
            raise ValueError(f"{c} 无行情，停止而不是静默删除样本")
    latest = max(d.date.max() for d in data.values())
    if not is_backtest:
        if (today_hk() - latest).days > h["max_stale_days"]:
            raise ValueError(f"行情过期：最新 {latest:%Y-%m-%d}，拒绝输出当前信号")
        stale = [c for c, d in data.items() if d.date.max() != latest]
        if stale:
            raise ValueError(f"股票最新日不一致（可能停牌或数据缺失）：{stale}；核查后再生成当前目标")
    signals = strategy.generate_signals(data)
    if signals.empty:
        raise ValueError("不足以生成信号；需至少一个完整预热窗口")
    if is_backtest:
        # 防止接口只返回最后 640 根，悄悄把多年前的回测从较晚日期启动。
        if signals.index[0] >= pd.Timestamp(start):
            raise ValueError(f"预热历史不足：首个信号 {signals.index[0]:%Y-%m-%d} 晚于或等于起点；导入更长历史或调晚 --start")
        engine = HKBacktester(cfg, u, research, calendar)
        equity, trades = engine.run(data, signals, start, end)
        warnings.append(f"实际行情区间 {equity.index[0]:%Y-%m-%d} 至 {equity.index[-1]:%Y-%m-%d}；不会补造请求结束日的行情。")
        initial = h["initial_cash_hkd"]
        path = _path(base, "reports") / f"hk_connect_{datetime.now():%Y%m%d_%H%M%S}"
        path.mkdir(parents=True, exist_ok=True)
        equity.to_csv(path / "equity_hkd.csv")
        pd.DataFrame(trades, columns=["date", "code", "side", "price", "shares", "fee_hkd", "amount_hkd", "pnl", "win"]).to_csv(path / "trades_hkd.csv", index=False)
        signals.to_csv(path / "targets.csv")
        peak = equity.total_value.cummax().clip(lower=initial)
        summary = {"strategy": "hk_connect", "currency": "HKD", "research": research,
                   "initial_cash_hkd": initial, "final_value_hkd": float(equity.total_value.iloc[-1]),
                   "total_return": float(equity.total_value.iloc[-1] / initial - 1),
                   "max_drawdown": float((equity.total_value / peak - 1).min()),
                   "trades": len(trades), "fees_hkd": sum(t["fee_hkd"] for t in trades),
                   "deferred_days": engine.deferred_days, "pending": bool(engine.pending),
                   "positions": engine.last_positions, "warnings": warnings,
                   "config": h, "universe": u.payload,
                   "data_ranges": {c: [str(d.date.min().date()), str(d.date.max().date())] for c, d in data.items()}}
        save_json(path / "summary.json", summary)
        print(f"\n港股通 HKD 价格收益研究：累计 {summary['total_return']:.2%}，最大回撤 {summary['max_drawdown']:.2%}，成交 {len(trades)} 笔。")
        print(f"报告：{path}")
    else:
        target = signals.iloc[-1].copy()
        for c in target.index:
            if not u.eligible(c, today_hk(), research=False):
                target[c] = 0
        lines = [f"港股通股票策略｜行情截至 {latest:%Y-%m-%d}｜HKD 研究账户",
                 f"资格观察日 {u.observed:%Y-%m-%d}；默认 SH 通道观察池。",
                 "以下为模型目标，不是实际持仓；参考股数按空仓账户及最近收盘价计算。",
                 f"账户 {h['initial_cash_hkd']:,.0f} HKD；每周指定日更新，趋势破坏/资格退出每日检查。"]
        prices = pd.Series({c: d.iloc[-1].close for c, d in data.items()})
        shares = {c: 0 for c in target.index}
        simulated = []
        engine = HKBacktester(cfg, u, research=False)
        remaining = engine._rebalance(today_hk(), target, prices, h["initial_cash_hkd"], shares, simulated)
        for c in target[target > 0].sort_values(ascending=False).index:
            info = u.info(c, today_hk())
            lines.append(f"{c} {info['name']}：目标 {target[c]:.1%}，收盘 {prices[c]:.3f} HKD，"
                         f"每手 {info['lot']} 股，参考 {shares[c]} 股" + ("（不足一手，保留现金）" if shares[c] == 0 else ""))
        lines.append(f"模型现金权重 {1-target.sum():.1%}；整数手及估算费用后参考现金 {remaining:,.2f} HKD。")
        if target.sum() == 0:
            lines.append("当前没有同时满足趋势、流动性、风险预算和最少可负担持仓数的组合，保持现金。")
        lines.append("使用前一香港交易日及更早的完整日线；实际委托前需券商确认下一可交易通道日及资格。")
        if calendar is None:
            warnings.append("未配置通道日历，不生成确定的执行日期。")
        if (today_hk() - latest).days > 1:
            warnings.append(f"最新行情距离今天 {(today_hk() - latest).days} 个自然日，请核对假期或更新延迟。")
        lines.extend("说明：" + w for w in warnings)
        output = "\n".join(lines)
        path = _path(base, "signals") / f"{today_hk():%Y-%m-%d}_hk_connect.txt"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(output + "\n", encoding="utf-8")
        print("\n" + output + f"\n信号已保存：{path}")
    if is_backtest:
        for w in warnings:
            print("说明：" + w)
