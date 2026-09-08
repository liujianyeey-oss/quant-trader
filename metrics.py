"""绩效指标与报表输出。"""
import numpy as np
import pandas as pd


def calc_metrics(equity: pd.Series, trades: list, periods_per_year=252,
                 initial=None) -> dict:
    """根据净值序列计算常用绩效指标。

    年化按自然日年数口径：(末日-首日).days+1 / 365.25（与审计口径一致），
    避免交易日条数/252 与自然日两种年化口径混用。
    initial: 期初资金；None 时用首个净值。
    """
    base = initial if initial is not None else equity.iloc[0]
    ret = equity.pct_change().dropna()
    total_return = equity.iloc[-1] / base - 1
    n_years = ((equity.index[-1] - equity.index[0]).days + 1) / 365.25
    annual_return = (1 + total_return) ** (1 / n_years) - 1 if n_years > 0 else np.nan

    drawdown = equity / equity.cummax() - 1
    max_drawdown = drawdown.min()

    sharpe = (ret.mean() / ret.std() * np.sqrt(periods_per_year)
              if ret.std() > 0 else np.nan)
    calmar = annual_return / abs(max_drawdown) if max_drawdown < 0 else np.nan

    sells = [t for t in trades if t["side"] == "sell"]
    n_trades = len(sells)
    win_rate = (sum(1 for t in sells if t["win"]) / n_trades) if n_trades else np.nan

    return {
        "总收益率": total_return,
        "年化收益率": annual_return,
        "最大回撤": max_drawdown,
        "夏普比率": sharpe,
        "卡玛比率": calmar,
        "交易次数": n_trades,
        "胜率": win_rate,
    }


def format_report(name, metrics, bench_metrics=None) -> str:
    """生成中文文本报表。"""
    lines = [f"===== 回测绩效报表：{name} =====", ""]
    rows = [
        ("总收益率", f"{metrics['总收益率']:.2%}"),
        ("年化收益率", f"{metrics['年化收益率']:.2%}"),
        ("最大回撤", f"{metrics['最大回撤']:.2%}"),
        ("夏普比率", f"{metrics['夏普比率']:.2f}"),
        ("卡玛比率", f"{metrics['卡玛比率']:.2f}"),
        ("交易次数", f"{metrics['交易次数']}"),
        ("胜率", f"{metrics['胜率']:.2%}" if pd.notna(metrics["胜率"]) else "N/A"),
    ]
    if bench_metrics is not None:
        lines.append(f"{'指标':<10}{'策略':>12}{'基准':>12}")
        lines.append("-" * 36)
        bmap = {
            "总收益率": f"{bench_metrics['总收益率']:.2%}",
            "年化收益率": f"{bench_metrics['年化收益率']:.2%}",
            "最大回撤": f"{bench_metrics['最大回撤']:.2%}",
            "夏普比率": f"{bench_metrics['夏普比率']:.2f}",
            "卡玛比率": f"{bench_metrics['卡玛比率']:.2f}",
            "交易次数": "-",
            "胜率": "-",
        }
        for k, v in rows:
            lines.append(f"{k:<10}{v:>12}{bmap[k]:>12}")
    else:
        for k, v in rows:
            lines.append(f"{k}：{v}")
    return "\n".join(lines)


def plot_equity(equity: pd.Series, bench: pd.Series | None, title: str, path: str):
    """保存净值曲线 PNG（含回撤子图）。"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams["font.sans-serif"] = ["PingFang SC", "Arial Unicode MS", "Heiti TC", "SimHei"]
    plt.rcParams["axes.unicode_minus"] = False

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 7), sharex=True,
                                   gridspec_kw={"height_ratios": [3, 1]})
    norm = equity / equity.iloc[0]
    ax1.plot(norm.index, norm.values, label="策略净值", linewidth=1.5)
    if bench is not None:
        bnorm = bench / bench.iloc[0]
        ax1.plot(bnorm.index, bnorm.values, label="基准净值", linewidth=1.2, alpha=0.8)
    ax1.set_title(title)
    ax1.legend()
    ax1.grid(alpha=0.3)

    dd = equity / equity.cummax() - 1
    ax2.fill_between(dd.index, dd.values, 0, alpha=0.4, color="tomato")
    ax2.set_ylabel("回撤")
    ax2.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
