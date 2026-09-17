"""R11 信号生成：M1 动量 / M2 低波 / M3 复合 的月末 top30 等权目标。

执行语义：月末第 phase 个交易日（信号日 T 收盘）产生目标，T+1 开盘执行（引擎负责）。
入选门槛（预声明）：信号日可用历史 ≥250 根 K 线且当日有收盘价。
"""
import numpy as np
import pandas as pd

MOM_W, VOL_W, TOP_N, MIN_BARS = 60, 60, 30, 250


def build_panels(data):
    closes = {c: df.set_index("date")["close"] for c, df in data.items()}
    return pd.DataFrame(closes).sort_index()


def monthly_signal_dates(calendar, phase):
    """每月第 phase 个交易日（1~5）。"""
    s = pd.Series(calendar, index=calendar)
    firsts = s.groupby([calendar.year, calendar.month]).nth(phase - 1)
    return pd.DatetimeIndex(firsts.values)


def make_signals(panel, kind, phase=1, top_n=TOP_N):
    """生成 [(date, {code: weight})]。kind: mom60/lowvol60/combo。"""
    calendar = panel.index
    dates = monthly_signal_dates(calendar, phase)
    mom = panel / panel.shift(MOM_W) - 1
    vol = panel.pct_change().rolling(VOL_W).std() * np.sqrt(252)
    bars = panel.notna().cumsum()
    signals = []
    for d in dates:
        i = calendar.get_loc(d)
        if i < MIN_BARS:
            continue
        elig = panel.columns[(bars.iloc[i] >= MIN_BARS) & panel.iloc[i].notna()]
        if len(elig) < top_n:
            continue
        if kind == "mom60":
            metric = mom.iloc[i][elig].dropna()
            winners = metric.nlargest(top_n).index
        elif kind == "lowvol60":
            metric = vol.iloc[i][elig].dropna()
            winners = metric.nsmallest(top_n).index
        else:  # combo：动量排名分 + 低波排名分等权合成
            m_mom = mom.iloc[i][elig].dropna()
            m_vol = vol.iloc[i][elig].dropna()
            common = m_mom.index.intersection(m_vol.index)
            if len(common) < top_n:
                continue
            score = m_mom[common].rank(pct=True) + (-m_vol[common]).rank(pct=True)
            winners = score.nlargest(top_n).index
        w = {c: 1.0 / len(winners) for c in winners}
        signals.append((d, w))
    return signals


def perf_stats(eq, trades, initial):
    years = max(((eq.index[-1] - eq.index[0]).days + 1) / 365.25, 1 / 365.25)
    total = eq.iloc[-1] / initial - 1
    cagr = (1 + total) ** (1 / years) - 1
    mdd = (eq / eq.cummax() - 1).min()
    ret = eq.pct_change().dropna()
    sharpe = ret.mean() / ret.std() * np.sqrt(252) if ret.std() > 0 else np.nan
    calmar = cagr / abs(mdd) if mdd < 0 else np.nan
    fees = sum(t["fee"] for t in trades)
    amount = sum(t["amount"] for t in trades)
    turnover = amount / eq.mean() / years if len(eq) else np.nan
    return dict(total=total, cagr=cagr, mdd=mdd, sharpe=sharpe, calmar=calmar,
                fees=fees, turnover=turnover)
