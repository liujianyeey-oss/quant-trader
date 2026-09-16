"""第二轮研究信号生成器：唐奇安退出变体、仓位/加仓、regime 自适应、
多均线投票、防守 sleeve 组合。同样输出目标权重信号（因果，t 收盘→t+1 开盘）。"""
import numpy as np
import pandas as pd


def _adj(data, symbol, field):
    d = data[symbol].set_index("date")
    return d[field] * d["factor"]


def _atr(data, symbol, n=22):
    """Wilder ATR（复权价）。"""
    h, l, c = _adj(data, symbol, "high"), _adj(data, symbol, "low"), _adj(data, symbol, "close")
    tr = pd.concat([h - l, (h - c.shift(1)).abs(), (l - c.shift(1)).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / n, adjust=False).mean()


def _empty(data):
    return pd.Series(0.0, index=list(data.keys()))


# ---------- 支线一 2：退出变体（entry 固定 20） ----------

def donchian_chandelier(data, symbol, entry=20, hh=22, atr_n=22, mult=3.0):
    """入场：20 日突破；退出：收盘 < Chandelier（22日最高价 - 3×ATR22，只上移）。"""
    adj_c, adj_h = _adj(data, symbol, "close"), _adj(data, symbol, "high")
    atr = _atr(data, symbol, atr_n)
    holding, stop = False, None
    rows = []
    for i in range(entry + 1, len(adj_c)):
        w = _empty(data)
        hh_now = adj_h.iloc[max(0, i - hh + 1):i + 1].max()
        if not holding:
            if adj_c.iloc[i] > adj_h.rolling(entry).max().iloc[i - 1]:
                holding, stop = True, hh_now - mult * atr.iloc[i]
            else:
                continue
        else:
            stop = max(stop, hh_now - mult * atr.iloc[i])
            if adj_c.iloc[i] < stop:
                holding, stop = False, None
            else:
                continue
        if holding:
            w[symbol] = 1.0
        rows.append(w.rename(adj_c.index[i]))
    return pd.DataFrame(rows)


def donchian_ma_exit(data, symbol, entry=20, ma=20):
    """入场：20 日突破；退出：收盘 < MA20。"""
    adj_c, adj_h = _adj(data, symbol, "close"), _adj(data, symbol, "high")
    ma_line = adj_c.rolling(ma).mean()
    holding = False
    rows = []
    for i in range(entry + 1, len(adj_c)):
        w = _empty(data)
        if not holding:
            if adj_c.iloc[i] > adj_h.rolling(entry).max().iloc[i - 1]:
                holding = True
            else:
                continue
        elif adj_c.iloc[i] < ma_line.iloc[i]:
            holding = False
        else:
            continue
        if holding:
            w[symbol] = 1.0
        rows.append(w.rename(adj_c.index[i]))
    return pd.DataFrame(rows)


def donchian_time_exit(data, symbol, entry=20, hold_days=20):
    """入场：20 日突破；退出：持仓满 hold_days 个交易日。"""
    adj_c, adj_h = _adj(data, symbol, "close"), _adj(data, symbol, "high")
    holding, days = False, 0
    rows = []
    for i in range(entry + 1, len(adj_c)):
        w = _empty(data)
        if not holding:
            if adj_c.iloc[i] > adj_h.rolling(entry).max().iloc[i - 1]:
                holding, days = True, 0
            else:
                continue
        else:
            days += 1
            if days >= hold_days:
                holding = False
            else:
                continue
        if holding:
            w[symbol] = 1.0
        rows.append(w.rename(adj_c.index[i]))
    return pd.DataFrame(rows)


# ---------- 支线一 4：仓位与加仓 ----------

def donchian_atr_sized(data, symbol, entry=20, exit=10, atr_n=22,
                       stop_atr=2.0, risk_pct=0.01):
    """20/10 唐奇安 + ATR 风险定仓：目标权重 = min(1, risk_pct×price/(stop_atr×ATR))。
    止损距离 = stop_atr×ATR22，单笔计划风险 ≈ risk_pct×净值。"""
    adj_c, adj_h, adj_l = (_adj(data, symbol, f) for f in ("close", "high", "low"))
    atr = _atr(data, symbol, atr_n)
    upper = adj_h.rolling(entry).max().shift(1)
    lower = adj_l.rolling(exit).min().shift(1)
    holding = False
    rows = []
    for i in range(entry + 1, len(adj_c)):
        w = _empty(data)
        if not holding:
            if adj_c.iloc[i] > upper.iloc[i]:
                holding = True
            else:
                continue
        elif adj_c.iloc[i] < lower.iloc[i]:
            holding = False
        else:
            continue
        if holding:
            risk_dist = stop_atr * atr.iloc[i]
            w[symbol] = min(1.0, risk_pct * adj_c.iloc[i] / risk_dist) if risk_dist > 0 else 0.0
        rows.append(w.rename(adj_c.index[i]))
    return pd.DataFrame(rows)


def donchian_pyramid(data, symbol, entry=20, exit=10, atr_n=22,
                     step_atr=0.5, max_adds=3, risk_pct=0.01):
    """海龟式金字塔：突破入 1 单位，每上浮 step_atr×ATR 加 1 单位（最多 max_adds 次），
    每单位权重按 risk_pct ATR 风险定仓，10 日低点全部退出。"""
    adj_c, adj_h, adj_l = (_adj(data, symbol, f) for f in ("close", "high", "low"))
    atr = _atr(data, symbol, atr_n)
    upper = adj_h.rolling(entry).max().shift(1)
    lower = adj_l.rolling(exit).min().shift(1)
    units, next_level, unit_w = 0, None, 0.0
    rows = []
    for i in range(entry + 1, len(adj_c)):
        w = _empty(data)
        price = adj_c.iloc[i]
        if units == 0:
            if price > upper.iloc[i]:
                units = 1
                unit_w = min(1.0 / (1 + max_adds),
                             risk_pct * price / (2.0 * atr.iloc[i])) if atr.iloc[i] > 0 else 0.0
                next_level = price + step_atr * atr.iloc[i]
            else:
                continue
        elif price < lower.iloc[i]:
            units, next_level, unit_w = 0, None, 0.0
        elif units <= max_adds and price >= next_level:
            units += 1
            next_level = price + step_atr * atr.iloc[i]
        else:
            continue
        if units > 0:
            w[symbol] = min(1.0, unit_w * units)
        rows.append(w.rename(adj_c.index[i]))
    return pd.DataFrame(rows)


# ---------- 支线二 6：regime 自适应 ----------

def _adx(data, symbol, n=14):
    """Wilder ADX 与 ±DI（复权价）。"""
    h, l, c = (_adj(data, symbol, f) for f in ("high", "low", "close"))
    up = h.diff()
    down = -l.diff()
    plus_dm = np.where((up > down) & (up > 0), up, 0.0)
    minus_dm = np.where((down > up) & (down > 0), down, 0.0)
    tr = pd.concat([h - l, (h - c.shift(1)).abs(), (l - c.shift(1)).abs()], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1 / n, adjust=False).mean()
    plus_di = 100 * pd.Series(plus_dm, index=h.index).ewm(alpha=1 / n, adjust=False).mean() / atr
    minus_di = 100 * pd.Series(minus_dm, index=h.index).ewm(alpha=1 / n, adjust=False).mean() / atr
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    adx = dx.ewm(alpha=1 / n, adjust=False).mean()
    return adx, plus_di, minus_di


def regime_adaptive(data, symbol, entry=20, exit=10, adx_n=14, adx_th=25):
    """ADX(14)>25 且 +DI>-DI → 唐奇安 20/10；否则 RSI(2) 均值回归
    （MA200 上方 RSI2<10 买、收盘>MA5 卖）。模式切换时先平仓。"""
    from .etf_strategies import _rsi
    adj_c, adj_h, adj_l = (_adj(data, symbol, f) for f in ("close", "high", "low"))
    adx, pdi, mdi = _adx(data, symbol, adx_n)
    upper = adj_h.rolling(entry).max().shift(1)
    lower = adj_l.rolling(exit).min().shift(1)
    rsi = _rsi(adj_c, 2)
    ma200, ma5 = adj_c.rolling(200).mean(), adj_c.rolling(5).mean()
    mode, don_hold, rsi_hold = None, False, False
    rows = []
    for i in range(201, len(adj_c)):
        w = _empty(data)
        new_mode = "trend" if (adx.iloc[i] > adx_th and pdi.iloc[i] > mdi.iloc[i]) else "mr"
        if new_mode != mode:
            if don_hold or rsi_hold:
                rows.append(w.rename(adj_c.index[i]))  # 切换先平仓
            mode, don_hold, rsi_hold = new_mode, False, False
            continue
        if mode == "trend":
            if not don_hold and adj_c.iloc[i] > upper.iloc[i]:
                don_hold = True
            elif don_hold and adj_c.iloc[i] < lower.iloc[i]:
                don_hold = False
            else:
                continue
            if don_hold:
                w[symbol] = 1.0
        else:
            if not rsi_hold and adj_c.iloc[i] > ma200.iloc[i] and rsi.iloc[i] < 10:
                rsi_hold = True
            elif rsi_hold and adj_c.iloc[i] > ma5.iloc[i]:
                rsi_hold = False
            else:
                continue
            if rsi_hold:
                w[symbol] = 1.0
        rows.append(w.rename(adj_c.index[i]))
    return pd.DataFrame(rows)


# ---------- 支线二 7：多均线投票仓位 ----------

def ma_vote(closes, symbol, windows=(20, 50, 100, 200)):
    """价格站上各均线的比例即目标仓位（0/25/50/75/100%），每日更新。"""
    c = closes[symbol]
    vote = sum((c > c.rolling(w).mean()).astype(float) for w in windows) / len(windows)
    rows = []
    prev = None
    for i in range(max(windows) + 1, len(closes)):
        v = vote.iloc[i]
        if prev is None or v != prev:
            w = pd.Series(0.0, index=closes.columns)
            w[symbol] = v
            rows.append(w.rename(closes.index[i]))
            prev = v
    return pd.DataFrame(rows)


# ---------- 工具：信号加权合并 ----------

def combine_weighted(parts):
    """{name: (signals, weight)} → 按权重加权合并（信号层组合，各信号前向填充）。"""
    dates = None
    for sig, _ in parts.values():
        dates = sig.index if dates is None else dates.union(sig.index)
    total = None
    for sig, wt in parts.values():
        aligned = sig.reindex(dates).ffill().fillna(0.0) * wt
        total = aligned if total is None else total + aligned
    return total
