"""美股 ETF 策略电池：全部输出目标权重信号（index=信号日，columns=标的，权重和<=1）。

统一纪律：信号只用 t 日及之前数据生成；引擎在 t+1 开盘执行。
轮动池 = 全部 26 只（不含现金代理 SHY）；防守资产 SHY 或 IEF。
"""
import numpy as np
import pandas as pd

from .etf_data import ALL_ETFS

POOL = [s for s in ALL_ETFS if s != "SHY"]  # 26 只轮动池
SHY, IEF = "SHY", "IEF"


def close_panel(data):
    """复权收盘面板。"""
    return pd.DataFrame({s: df.set_index("date")["close"] * df.set_index("date")["factor"]
                         for s, df in data.items()}).sort_index()


def _weekly_rows(closes, phase, rebalance, warmup):
    """每周（rebalance 个交易日）调仓的行索引，phase 为相位偏移。"""
    return range(warmup + phase, len(closes), rebalance)


# ---------- A. 动量轮动族 ----------

def mom_rotation(closes, window, top=1, rebalance=5, phase=0, pool=POOL):
    """单窗口动量 top1/top2 等权。"""
    mom = closes / closes.shift(window) - 1
    rows = []
    for i in _weekly_rows(closes, phase, rebalance, window + 1):
        m = mom.iloc[i][[c for c in pool if c in closes.columns]].dropna()
        if m.empty:
            continue
        w = pd.Series(0.0, index=closes.columns)
        w[m.nlargest(top).index] = 1.0 / top
        rows.append(w.rename(closes.index[i]))
    return pd.DataFrame(rows)


def mom_126_21(closes, top=1, rebalance=5, phase=0, pool=POOL):
    """经典 12-2 动量：close[t-21]/close[t-126]-1（跳过最近一个月）。"""
    score = closes.shift(21) / closes.shift(126) - 1
    rows = []
    for i in _weekly_rows(closes, phase, rebalance, 127):
        m = score.iloc[i][[c for c in pool if c in closes.columns]].dropna()
        if m.empty:
            continue
        w = pd.Series(0.0, index=closes.columns)
        w[m.nlargest(top).index] = 1.0 / top
        rows.append(w.rename(closes.index[i]))
    return pd.DataFrame(rows)


def comp_momentum(closes, top=2, rebalance=5, phase=0, pool=POOL,
                  windows=(20, 60, 120)):
    """多窗口复合动量（等权打分）top2。"""
    score = sum(closes / closes.shift(w) - 1 for w in windows) / len(windows)
    rows = []
    for i in _weekly_rows(closes, phase, rebalance, max(windows) + 1):
        m = score.iloc[i][[c for c in pool if c in closes.columns]].dropna()
        if m.empty:
            continue
        w = pd.Series(0.0, index=closes.columns)
        w[m.nlargest(top).index] = 1.0 / top
        rows.append(w.rename(closes.index[i]))
    return pd.DataFrame(rows)


def dual_momentum(closes, top=1, rebalance=5, phase=0, pool=POOL,
                  window=60, absolute="shy", fallback=SHY):
    """双动量：相对动量最强者，绝对动量 > SHY（或 >0）则持有，否则持防守资产。"""
    mom = closes / closes.shift(window) - 1
    rows = []
    for i in _weekly_rows(closes, phase, rebalance, window + 1):
        m = mom.iloc[i][[c for c in pool if c in closes.columns]].dropna()
        if m.empty:
            continue
        winner = m.idxmax()
        shy_mom = mom.iloc[i].get(SHY, np.nan)
        if absolute == "shy":
            ok = pd.notna(shy_mom) and m[winner] > shy_mom
        else:
            ok = m[winner] > 0
        w = pd.Series(0.0, index=closes.columns)
        w[winner if ok else fallback] = 1.0
        rows.append(w.rename(closes.index[i]))
    return pd.DataFrame(rows)


def invvol_top2(closes, rebalance=5, phase=0, pool=POOL, mom_window=60, vol_window=60):
    """top2 动量 + 逆波动加权。"""
    mom = closes / closes.shift(mom_window) - 1
    vol = closes.pct_change().rolling(vol_window, min_periods=vol_window).std() * np.sqrt(252)
    rows = []
    for i in _weekly_rows(closes, phase, rebalance, max(mom_window, vol_window) + 1):
        m = mom.iloc[i][[c for c in pool if c in closes.columns]].dropna()
        if len(m) < 2:
            continue
        winners = m.nlargest(2).index
        v = vol.iloc[i][winners]
        if v.isna().any():
            continue
        inv = 1.0 / v.clip(lower=0.03)
        w = pd.Series(0.0, index=closes.columns)
        w[winners] = (inv / inv.sum()).to_numpy()
        rows.append(w.rename(closes.index[i]))
    return pd.DataFrame(rows)


# ---------- B. 趋势族 ----------

def gtaa(closes, top=2, rebalance=5, phase=0, pool=POOL, off=IEF,
                 windows=(20, 60, 120)):
    """GTAA 式风险开/关：SPY>MA200 → 持 top2 复合动量；否则持 IEF/SHY。"""
    spy_ma200 = closes["SPY"].rolling(200, min_periods=200).mean()
    score = sum(closes / closes.shift(w) - 1 for w in windows) / len(windows)
    rows = []
    for i in _weekly_rows(closes, phase, rebalance, max(200, max(windows) + 1)):
        date = closes.index[i]
        w = pd.Series(0.0, index=closes.columns)
        if closes.iloc[i]["SPY"] > spy_ma200.iloc[i]:
            m = score.iloc[i][[c for c in pool if c in closes.columns]].dropna()
            if m.empty:
                continue
            w[m.nlargest(top).index] = 1.0 / top
        else:
            w[off] = 1.0
        rows.append(w.rename(date))
    return pd.DataFrame(rows)


def ma_cross(closes, symbol, fast=50, slow=200):
    """MA 金叉死叉单标的：MA50>MA200 持有，否则空仓。每日检查。"""
    ma_f = closes[symbol].rolling(fast).mean()
    ma_s = closes[symbol].rolling(slow).mean()
    above = (ma_f > ma_s)
    cross = above != above.shift(1)
    dates = closes.index[cross.fillna(False) & ma_s.notna()]
    rows = []
    for d in dates:
        w = pd.Series(0.0, index=closes.columns)
        if above.loc[d]:
            w[symbol] = 1.0
        rows.append(w.rename(d))
    return pd.DataFrame(rows)


def donchian(data, symbol, entry=20, exit=10):
    """唐奇安突破：收盘 > 前 20 日最高价买入；收盘 < 前 10 日最低价卖出。"""
    d = data[symbol].set_index("date")
    adj_h = d["high"] * d["factor"]
    adj_l = d["low"] * d["factor"]
    adj_c = d["close"] * d["factor"]
    upper = adj_h.rolling(entry).max().shift(1)  # 前 entry 日（不含当日）
    lower = adj_l.rolling(exit).min().shift(1)
    holding = False
    rows = []
    for i in range(entry + 1, len(d)):
        date = d.index[i]
        w = pd.Series(0.0, index=list(data.keys()))
        if not holding and adj_c.iloc[i] > upper.iloc[i]:
            holding = True
        elif holding and adj_c.iloc[i] < lower.iloc[i]:
            holding = False
        else:
            continue
        if holding:
            w[symbol] = 1.0
        rows.append(w.rename(date))
    return pd.DataFrame(rows)


# ---------- C. 均值回归族（只做指数 ETF） ----------

def _rsi(close, n=2):
    """Wilder RSI。"""
    diff = close.diff()
    gain = diff.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    loss = (-diff.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - 100 / (1 + rs)


def rsi2_connors(closes, symbol, rsi_n=2, ma_trend=200, ma_exit=5, threshold=10):
    """Connors RSI(2)：收盘>MA200 且 RSI2<10 买；收盘>MA5 卖。"""
    c = closes[symbol]
    rsi = _rsi(c, rsi_n)
    ma200 = c.rolling(ma_trend).mean()
    ma5 = c.rolling(ma_exit).mean()
    holding = False
    rows = []
    for i in range(ma_trend + 1, len(closes)):
        date = closes.index[i]
        w = pd.Series(0.0, index=closes.columns)
        if not holding and c.iloc[i] > ma200.iloc[i] and rsi.iloc[i] < threshold:
            holding = True
        elif holding and c.iloc[i] > ma5.iloc[i]:
            holding = False
        else:
            continue
        if holding:
            w[symbol] = 1.0
        rows.append(w.rename(date))
    return pd.DataFrame(rows)


def bollinger_revert(closes, symbol, window=20, z_in=-2.0, z_out=0.0):
    """布林 z-score 反转：z<-2 买入，z>0 卖出。"""
    c = closes[symbol]
    ma = c.rolling(window).mean()
    sd = c.rolling(window).std()
    z = (c - ma) / sd
    holding = False
    rows = []
    for i in range(window + 1, len(closes)):
        date = closes.index[i]
        w = pd.Series(0.0, index=closes.columns)
        if not holding and z.iloc[i] < z_in:
            holding = True
        elif holding and z.iloc[i] > z_out:
            holding = False
        else:
            continue
        if holding:
            w[symbol] = 1.0
        rows.append(w.rename(date))
    return pd.DataFrame(rows)


# ---------- D. 多因子族 ----------

def mf_mom_invvol(closes, rebalance=5, phase=0, pool=POOL,
                  windows=(20, 60, 120), vol_window=60):
    """动量×逆波动：复合动量 top2，逆波动加权。"""
    score = sum(closes / closes.shift(w) - 1 for w in windows) / len(windows)
    vol = closes.pct_change().rolling(vol_window, min_periods=vol_window).std() * np.sqrt(252)
    rows = []
    for i in _weekly_rows(closes, phase, rebalance, max(max(windows), vol_window) + 1):
        m = score.iloc[i][[c for c in pool if c in closes.columns]].dropna()
        if len(m) < 2:
            continue
        winners = m.nlargest(2).index
        v = vol.iloc[i][winners]
        if v.isna().any():
            continue
        inv = 1.0 / v.clip(lower=0.03)
        w = pd.Series(0.0, index=closes.columns)
        w[winners] = (inv / inv.sum()).to_numpy()
        rows.append(w.rename(closes.index[i]))
    return pd.DataFrame(rows)


def us_staggered(closes, pool=POOL, mom_window=20, trend_ma=60, vol_window=60,
                 vol_budget=0.12, n_groups=5, defensive=SHY):
    """A股 staggered 组合层移植：5 组周一~周五分组更新，20 日动量选最强，
    收盘价>=MA60 选它否则选 SHY；组内权重 = min(1, vol_budget/max(σ60,0.03))；
    5 组按 1/5 平均合并。"""
    rets = closes.pct_change()
    vol = rets.rolling(vol_window, min_periods=vol_window).std() * np.sqrt(252)
    mom = closes / closes.shift(mom_window) - 1
    ma = closes.rolling(trend_ma, min_periods=trend_ma).mean()
    codes = [c for c in pool if c in closes.columns]
    warmup = max(120, mom_window, trend_ma, vol_window)

    cohorts = []
    for weekday in range(n_groups):
        rows = []
        for i in range(warmup, len(closes)):
            date = closes.index[i]
            if date.weekday() != weekday:
                continue
            m = mom.iloc[i][codes].dropna()
            if m.empty:
                continue
            winner = m.idxmax()
            if not (closes.iloc[i][winner] >= ma.iloc[i][winner]):
                winner = defensive
            w = pd.Series(0.0, index=closes.columns)
            v = vol.iloc[i][winner]
            if pd.notna(v):
                w[winner] = min(1.0, vol_budget / max(0.03, v))
            rows.append(w.rename(date))
        cohorts.append(pd.DataFrame(rows))
    cohorts = [c for c in cohorts if not c.empty]
    dates = cohorts[0].index
    for c in cohorts[1:]:
        dates = dates.union(c.index)
    return sum(c.reindex(dates).ffill().fillna(0.0) for c in cohorts) / n_groups
