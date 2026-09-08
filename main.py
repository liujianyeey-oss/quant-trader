"""A股量化交易程序 CLI：backtest / signal 两个子命令。"""
import argparse
import logging
import os
import sys
from datetime import datetime

import pandas as pd
import yaml

from data import DataLoader
from strategies import STRATEGIES, DualMA
from backtest import Backtester
from metrics import calc_metrics, format_report, plot_equity

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def load_config(path=None):
    path = path or os.path.join(BASE_DIR, "config.yaml")
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def build_loader(cfg):
    return DataLoader(cache_dir=os.path.join(BASE_DIR, cfg["data"]["cache_dir"]),
                      adjust=cfg["data"]["adjust"])


def load_universe_data(cfg, loader, start, end, codes=None, allow_intraday=False):
    """加载标的池数据。返回 ({code: df}, 个股代码集合)。"""
    etfs = cfg["universe"]["etfs"]
    stocks = cfg["universe"]["stocks"]
    defensive = cfg["universe"]["defensive"]
    if codes is None:
        codes = etfs + stocks + ([defensive] if defensive else [])
    data, stock_set = {}, set(stocks)
    # 显式指定 codes 时（如 --code 600519）：不在 ETF 池/防守资产中的按个股处理
    etf_like = set(etfs) | {defensive}
    for code in codes:
        is_etf = code not in stock_set if len(codes) > 1 else code in etf_like
        if not is_etf:
            stock_set.add(code)
        df = loader.load(code, is_etf=is_etf, start=start, end=end,
                         allow_intraday=allow_intraday)
        if df.empty:
            print(f"警告：{code} 在 {start}~{end} 区间无数据，已跳过", file=sys.stderr)
            continue
        data[code] = df
    return data, stock_set


def load_benchmark(cfg, loader, start, end):
    bench_code = cfg.get("benchmark")
    if not bench_code:
        return None
    try:
        df = loader.load(bench_code, is_etf=True, start=start, end=end)
        if df.empty:
            return None
        return df.set_index("date")["close"].sort_index()
    except Exception as e:
        print(f"警告：基准 {bench_code} 数据加载失败（{e}），跳过基准对比", file=sys.stderr)
        return None


def cmd_backtest(args):
    cfg = load_config(args.config)
    start = args.start or cfg["backtest"]["start"]
    end = args.end or cfg["backtest"]["end"]
    loader = build_loader(cfg)

    strategy_name = args.strategy
    # 信号需要预热期（动量/均线窗口），拉数起点比回测起点早 450 天
    fetch_start = (pd.Timestamp(start) - pd.Timedelta(days=450)).strftime("%Y-%m-%d")
    if strategy_name == "dual_ma":
        code = args.code or cfg["universe"]["etfs"][0]
        data, stock_set = load_universe_data(cfg, loader, fetch_start, end, codes=[code])
        strategy = DualMA(cfg, code=code)
    else:
        data, stock_set = load_universe_data(cfg, loader, fetch_start, end)
        strategy = STRATEGIES[strategy_name](cfg)

    if not data:
        print("错误：无可用数据，回测终止", file=sys.stderr)
        sys.exit(1)

    signals = strategy.generate_signals(data)
    # 引擎内部完成信号对齐/T+1/区间裁剪，直接传含预热期的全量数据
    bt = Backtester(cfg, stock_codes=stock_set,
                    rebalance_on_event=strategy.rebalance_on_event)
    equity, trades = bt.run(data, signals, start=start, end=end)

    bench = load_benchmark(cfg, loader, start, end)
    initial = cfg["backtest"]["initial_cash"]
    metrics = calc_metrics(equity["total_value"], trades, initial=initial)
    bench_metrics = None
    if bench is not None:
        bench = bench.reindex(equity.index).ffill().dropna()
        if not bench.empty:
            bench_metrics = calc_metrics(bench, [])

    print(format_report(f"{strategy_name}（{start} ~ {end or '最新'}）", metrics, bench_metrics))
    print(f"\n期初资金：{cfg['backtest']['initial_cash']:,.0f}  期末资产：{equity['total_value'].iloc[-1]:,.0f}")

    reports_dir = os.path.join(BASE_DIR, "reports")
    os.makedirs(reports_dir, exist_ok=True)
    png = os.path.join(reports_dir, f"equity_{strategy_name}_{datetime.now():%Y%m%d_%H%M%S}.png")
    plot_equity(equity["total_value"], bench if bench_metrics else None,
                f"{strategy_name} 净值曲线（{start} ~ {end or '最新'}）", png)
    print(f"净值曲线已保存：{png}")


def cmd_paper_update(args):
    """每日台账更新（建议收盘后运行）。只用已完成日线，盘中快照不入账。"""
    from paper import TRACKED_STRATEGIES, update_account

    cfg = load_config(args.config)
    loader = build_loader(cfg)
    end = datetime.now().strftime("%Y-%m-%d")
    # 与 backtest 相同的预热口径
    fetch_start = (pd.Timestamp(cfg["backtest"]["start"])
                   - pd.Timedelta(days=450)).strftime("%Y-%m-%d")
    paper_dir = os.path.join(BASE_DIR, "paper")
    os.makedirs(paper_dir, exist_ok=True)

    data, stock_set = load_universe_data(cfg, loader, fetch_start, end)
    if not data:
        print("错误：无可用数据，台账更新终止", file=sys.stderr)
        sys.exit(1)
    latest = max(df["date"].max() for df in data.values())
    note = ""
    if latest.date() != datetime.now().date():
        note = ("（当日数据未入库：盘中未收盘不入账；"
                "若已收盘则为数据源不可达，已降级用缓存，恢复后自动补记）")
    print(f"数据截至 {latest:%Y-%m-%d}{note}")
    for name in TRACKED_STRATEGIES:
        for action in update_account(cfg, name, data, paper_dir, stock_set):
            print(action)


def cmd_paper_report(args):
    from paper import TRACKED_STRATEGIES, build_report, plot_paper_nav

    cfg = load_config(args.config)
    loader = build_loader(cfg)
    end = datetime.now().strftime("%Y-%m-%d")
    fetch_start = (pd.Timestamp(cfg["backtest"]["start"])
                   - pd.Timedelta(days=450)).strftime("%Y-%m-%d")
    paper_dir = os.path.join(BASE_DIR, "paper")

    data, _ = load_universe_data(cfg, loader, fetch_start, end)
    if not data:
        print("错误：无可用数据", file=sys.stderr)
        sys.exit(1)
    data_map = {name: data for name in TRACKED_STRATEGIES}
    bench = load_benchmark(cfg, loader, fetch_start, end)

    text, nav_curves, ideal_curves = build_report(cfg, data_map, bench, paper_dir)
    print(text)

    reports_dir = os.path.join(BASE_DIR, "reports")
    os.makedirs(reports_dir, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    txt_path = os.path.join(reports_dir, f"paper_report_{ts}.txt")
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(text + "\n")
    print(f"\n报表已保存：{txt_path}")
    if nav_curves:
        png = os.path.join(reports_dir, f"paper_nav_{ts}.png")
        plot_paper_nav(nav_curves, ideal_curves, bench, png)
        print(f"净值曲线已保存：{png}")


def cmd_optimize(args):
    from optimize import run_optimize, format_top_table

    cfg = load_config(args.config)
    loader = build_loader(cfg)
    print("开始网格搜索：动量轮动策略，样本内 2019-01-01~2023-12-31，"
          "样本外 2024-01-01~最新，按样本内卡玛比率排名")
    result = run_optimize(cfg, loader, top_n=args.top)

    reports_dir = os.path.join(BASE_DIR, "reports")
    os.makedirs(reports_dir, exist_ok=True)
    csv_path = os.path.join(reports_dir, "optimize_results.csv")
    result.to_csv(csv_path, index=False, encoding="utf-8-sig")
    print(f"\n===== 样本内卡玛 Top {args.top}（含样本外对照） =====")
    print(format_top_table(result, args.top))
    print(f"\n完整结果（{len(result)} 组）已保存：{csv_path}")


def cmd_signal(args):
    cfg = load_config(args.config)
    end = datetime.now().strftime("%Y-%m-%d")
    # 信号只需近期数据，但动量/均线需要足够窗口，多取一些历史
    start = cfg["backtest"]["start"]
    loader = build_loader(cfg)
    # 盘中运行时现场附加当日未完成 K 线（不写缓存），并给出明确提示
    intraday = not DataLoader._is_market_settled()

    strategy_name = args.strategy
    if strategy_name == "dual_ma":
        code = args.code or cfg["universe"]["etfs"][0]
        data, _ = load_universe_data(cfg, loader, start, end, codes=[code],
                                     allow_intraday=intraday)
        strategy = DualMA(cfg, code=code)
    else:
        data, _ = load_universe_data(cfg, loader, start, end,
                                     allow_intraday=intraday)
        strategy = STRATEGIES[strategy_name](cfg)

    if not data:
        print("错误：无可用数据，信号生成终止", file=sys.stderr)
        sys.exit(1)

    latest_date = max(df["date"].max() for df in data.values())
    is_intraday_bar = (latest_date.date() == datetime.now().date()
                       and not DataLoader._is_market_settled())
    desc = strategy.describe_signal(data)
    header = f"===== 每日信号（{strategy_name}） 数据截至 {latest_date:%Y-%m-%d} ====="
    output = f"{header}\n{desc}"
    if is_intraday_bar:
        note = (f"注意：当前为盘中（{datetime.now():%H:%M}），最新 K 线为未完成快照，"
                f"收盘价可能变化；该快照不会写入缓存。正式信号请收盘后再查。")
        output = f"{output}\n{note}"
    print(output)

    signals_dir = os.path.join(BASE_DIR, "signals")
    os.makedirs(signals_dir, exist_ok=True)
    path = os.path.join(signals_dir, f"{datetime.now():%Y-%m-%d}_{strategy_name}.txt")
    with open(path, "w", encoding="utf-8") as f:
        f.write(output + "\n")
    print(f"\n信号已写入：{path}")
    print("提示：信号基于最近收盘价，若需执行请在下一交易日开盘操作。仅供研究，不构成投资建议。")


def main():
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(message)s")
    parser = argparse.ArgumentParser(
        description="A股量化交易程序（回测 + 每日信号提醒，仅供研究）")
    parser.add_argument("--config", help="配置文件路径，默认 ./config.yaml")
    sub = parser.add_subparsers(dest="command", required=True)

    p_bt = sub.add_parser("backtest", help="运行历史回测")
    p_bt.add_argument("--strategy", choices=["momentum", "dual_ma", "staggered"],
                      default="momentum")
    p_bt.add_argument("--start", help="回测开始日期，如 2019-01-01")
    p_bt.add_argument("--end", help="回测结束日期，默认最新")
    p_bt.add_argument("--code", help="dual_ma 策略的标的代码，默认标的池第一个")
    p_bt.set_defaults(func=cmd_backtest)

    p_sig = sub.add_parser("signal", help="基于最新数据输出当前持仓信号")
    p_sig.add_argument("--strategy", choices=["momentum", "dual_ma", "staggered"],
                       default="momentum")
    p_sig.add_argument("--code", help="dual_ma 策略的标的代码，默认标的池第一个")
    p_sig.set_defaults(func=cmd_signal)

    p_opt = sub.add_parser("optimize", help="动量轮动策略网格调参（样本内排名 + 样本外验证）")
    p_opt.add_argument("--top", type=int, default=15, help="打印 top N 结果，默认 15")
    p_opt.set_defaults(func=cmd_optimize)

    p_pu = sub.add_parser("paper-update", help="模拟盘台账每日更新（建议收盘后运行）")
    p_pu.set_defaults(func=cmd_paper_update)

    p_pr = sub.add_parser("paper-report", help="模拟盘台账分析报表")
    p_pr.set_defaults(func=cmd_paper_report)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
