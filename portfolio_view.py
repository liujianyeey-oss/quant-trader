"""组合视图（paper-portfolio）：跨台账只读聚合，P2 风险平价权重合成组合净值指数。

不写任何台账文件。权重规则与 R7a 一致：60 日波动倒数加权、clip[10%,40%]、
月末再平衡；数据不足 60 日的袖珍按等权并标注；无数据的袖珍当月剔除后归一。
币种声明：各账户本币净值归一化合成，忽略汇率（R7a 声明口径）。
"""
import os

import numpy as np
import pandas as pd

TRACKED = [
    ("A股momentum", "paper", "nav_momentum.csv"),
    ("A股staggered", "paper", "nav_staggered.csv"),
    ("美股donchian_QQQ", "paper_us", "nav_qqq.csv"),
    ("美股donchian_SPY", "paper_us", "nav_spy.csv"),
    ("BTC×F1", "paper_btc", "nav_btc_f1.csv"),
]
VOL_WINDOW = 60
CLIP = (0.10, 0.40)


def load_navs(project_dir):
    """读取全部台账净值。返回 {name: pd.Series(nav)}，缺失的跳过。"""
    out = {}
    for name, sub, fname in TRACKED:
        path = os.path.join(project_dir, sub, fname)
        if os.path.exists(path):
            df = pd.read_csv(path, parse_dates=["date"])
            if not df.empty:
                out[name] = df.set_index("date")["nav"].sort_index()
    return out


def p2_weights(rets_window: pd.DataFrame):
    """60 日波动倒数权重，clip 后归一。返回 (weights: Series, notes: list)。"""
    notes = []
    cols = list(rets_window.columns)
    if len(rets_window) < VOL_WINDOW:
        notes.append(f"数据不足 {VOL_WINDOW} 日（{len(rets_window)} 日），暂用等权")
        return pd.Series(1.0 / len(cols), index=cols), notes
    vol = rets_window.std() * np.sqrt(252)
    inv = 1.0 / vol.replace(0, np.nan)
    inv = inv.replace([np.inf, -np.inf], np.nan)
    if inv.isna().any():
        notes.append(f"{','.join(inv[inv.isna()].index)} 波动缺失/为 0，按下限 10% 处理")
    inv = inv.fillna(0.0)
    if inv.sum() <= 0:
        notes.append("全部袖珍波动为 0，暂用等权")
        return pd.Series(1.0 / len(cols), index=cols), notes
    w = inv / inv.sum()
    w = w.clip(*CLIP)
    w = w / w.sum()
    return w, notes


def compose(nav: pd.DataFrame):
    """月末再平衡合成组合净值指数（R7a 同口径模拟）。

    返回 (port_nav: Series, target_w: Series, current_w: Series, notes: list)。
    target_w = 最近一次月末权重；current_w = 当前漂移后的实际权重。
    """
    rets = nav.pct_change()
    dates = nav.index
    month_ends = list(dates.to_series().groupby([dates.year, dates.month]).last())
    notes = []
    sleeve_vals, target_w = None, None
    rows = []
    for i, date in enumerate(dates):
        avail = [c for c in nav.columns if pd.notna(nav.loc[date, c])]
        if sleeve_vals is None or date in month_ends:
            total = 1.0 if sleeve_vals is None else sum(
                sleeve_vals.get(c, 0.0) for c in avail)
            w, ns = p2_weights(rets.loc[:date].tail(VOL_WINDOW)[avail])
            notes.extend(f"{date:%Y-%m-%d} {n}" for n in ns)
            target_w = w
            sleeve_vals = {c: total * w.get(c, 0.0) for c in avail}
        r = rets.loc[date]
        for c in list(sleeve_vals):
            sleeve_vals[c] *= (1 + (0.0 if pd.isna(r.get(c)) else r[c]))
        rows.append((date, sum(sleeve_vals.values())))
    total_now = sum(sleeve_vals.values())
    current_w = pd.Series({c: v / total_now for c, v in sleeve_vals.items()})
    return pd.Series(dict(rows)).sort_index(), target_w, current_w, notes


def perf_stats(eq, initial=1.0):
    years = max(((eq.index[-1] - eq.index[0]).days + 1) / 365.25, 1 / 365.25)
    total = eq.iloc[-1] / initial - 1
    cagr = (1 + total) ** (1 / years) - 1 if years > 0 else np.nan
    mdd = (eq / eq.cummax() - 1).min()
    ret = eq.pct_change().dropna()
    sharpe = ret.mean() / ret.std() * np.sqrt(252) if ret.std() > 0 else np.nan
    return dict(total=total, cagr=cagr, mdd=mdd, sharpe=sharpe)


def build_view(project_dir):
    """生成组合视图文本。返回 (text, port_nav)。"""
    navs = load_navs(project_dir)
    if not navs:
        return "无台账净值数据", None
    nav = pd.DataFrame(navs).sort_index()
    common_start = min(s.index[0] for s in navs.values())
    nav = nav[nav.index >= common_start]
    # 各袖珍最新值前向填充至组合末日（美股数据滞后一日、假日缺口等），不回填上市前
    nav = nav.ffill()
    nav = nav / nav.apply(lambda s: s.dropna().iloc[0])  # 各本币归一化

    port, target_w, current_w, notes = compose(nav)
    lines = [f"===== 组合视图（P2 风险平价，{pd.Timestamp.now():%Y-%m-%d %H:%M}）=====",
             "口径：各账户本币净值归一化合成，忽略汇率（R7a 声明）；月末再平衡",
             f"覆盖账户：{len(navs)} 个；组合起点 {port.index[0]:%Y-%m-%d}"]

    lines.append("\n[当前权重 vs 目标权重（偏离 >5pp 高亮）]")
    for c in target_w.index:
        t = target_w[c]
        cur = current_w.get(c, 0.0)
        drift = cur - t
        flag = "  ⚠️ 偏离>5pp，建议再平衡" if abs(drift) > 0.05 else ""
        lines.append(f"  {c:<18} 目标 {t:5.1%}  当前 {cur:5.1%}  偏离 {drift:+.1%}{flag}")

    m = perf_stats(port)
    sharpe_s = f"{m['sharpe']:.2f}" if pd.notna(m["sharpe"]) else "数据不足"
    lines.append(f"\n[组合表现] 累计 {m['total']:+.2%}  年化 {m['cagr']:+.2%}"
                 f"  最大回撤 {m['mdd']:.2%}  夏普 {sharpe_s}")
    if len(port) < 60:
        lines.append(f"  ⚠️ 组合历史仅 {len(port)} 个交易日，年化/夏普仅供参考")

    # 当日贡献（最近一个交易日）
    rets = nav.pct_change()
    last = rets.index[-1]
    lines.append(f"\n[最近交易日 {last:%Y-%m-%d} 贡献]")
    prev_w = current_w
    for c in rets.columns:
        r = rets.loc[last, c]
        if pd.notna(r):
            lines.append(f"  {c:<18} 日收益 {r:+.2%}  贡献 {r * prev_w.get(c, 0):+.3%}")

    # 等权对照
    ew = nav.pct_change().mean(axis=1)
    ew_nav = (1 + ew.fillna(0)).cumprod()
    m_ew = perf_stats(ew_nav)
    lines.append(f"\n[对照] 四/五账户等权：累计 {m_ew['total']:+.2%}（P2 合成 {m['total']:+.2%}）")
    if notes:
        lines.append("\n[标注]")
        lines += [f"  - {n}" for n in notes[-5:]]
    return "\n".join(lines), port
