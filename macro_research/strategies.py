"""基准策略信号与宏观过滤器。

基准策略（被过滤对象，全部输出 {symbol: weight} 日频信号，t 收盘→t+1 开盘）：
- BTC：唐奇安 20/10（收盘 > 前20日最高价买，< 前10日最低价卖，365 天日历）
- 513130：20 日动量 > 0 且收盘 > MA60 持有，否则空仓
- QQQ：复用 us_equities.research 的 donchian 20/10

过滤器（H1/H2/H3）：触发时目标权重归零。宏观输入用 T-1 观测。
"""
import numpy as np
import pandas as pd


# ---------- 基准策略 ----------

def btc_donchian(df, entry=20, exit=10):
    """BTC 唐奇安 20/10（自然日日历）。返回 date → weight 序列（仅状态变化日）。"""
    d = df.set_index("date")
    upper = d["high"].rolling(entry).max().shift(1)
    lower = d["low"].rolling(exit).min().shift(1)
    holding, rows = False, []
    for i in range(entry + 1, len(d)):
        date = d.index[i]
        if not holding and d["close"].iloc[i] > upper.iloc[i]:
            holding = True
        elif holding and d["close"].iloc[i] < lower.iloc[i]:
            holding = False
        else:
            continue
        rows.append((date, 1.0 if holding else 0.0))
    return pd.Series(dict(rows)).sort_index()


def hstech_momentum(df, mom_window=20, trend_ma=60):
    """513130：20 日动量>0 且收盘>MA60 持有，否则空仓。"""
    d = df.set_index("date")
    mom = d["close"] / d["close"].shift(mom_window) - 1
    ma = d["close"].rolling(trend_ma, min_periods=trend_ma).mean()
    w = ((mom > 0) & (d["close"] > ma)).astype(float)
    w[w.isna()] = 0.0
    w = w[w != w.shift(1)].dropna()
    return w.sort_index()


# ---------- 宏观过滤器 ----------

def macro_flags(dates, dxy, us10y, window=20, dxy_th=0.01, us10y_th=0.25):
    """在策略日历 dates 上给出 H1/H2/H3 触发标记（True=应空仓）。

    统一用 T-1 日宏观观测：date t 的信号使用 ≤ t-1 的最后一笔宏观数据
    （防宏观数据发布延迟/修订造成的隐性前视）。BTC 周末取前一工作日观测。
    """
    dx = dxy.set_index("date")["dxy"].sort_index()
    dxy_chg = dx / dx.shift(window) - 1
    uy = us10y.set_index("date")["us10y"].sort_index()
    uy_chg = uy - uy.shift(window)  # 百分点；+25bp = +0.25

    # 用 merge_asof 取 ≤ t-1 的最近观测（两侧统一 datetime64[ns]）
    ref = pd.DataFrame({"date": pd.DatetimeIndex(dates).astype("datetime64[ns]")})
    ref["t_minus_1"] = ref["date"] - pd.Timedelta(days=1)
    ref = ref.sort_values("t_minus_1")

    def _asof_join(chg: pd.Series, col: str):
        right = chg.dropna().rename(col).reset_index().sort_values("date")
        right["date"] = right["date"].astype("datetime64[ns]")
        return pd.merge_asof(ref[["t_minus_1"]], right,
                             left_on="t_minus_1", right_on="date", direction="backward")

    m_dxy = _asof_join(dxy_chg, "dxy_chg")
    m_uy = _asof_join(uy_chg, "uy_chg")
    h1 = (m_dxy["dxy_chg"] > dxy_th).to_numpy()
    h2 = (m_uy["uy_chg"] > us10y_th).to_numpy()
    flags = pd.DataFrame({"H1": h1, "H2": h2, "H3": h1 & h2}, index=dates)
    return flags.fillna(False)


def apply_filter(base_w: pd.Series, flag: pd.Series) -> pd.Series:
    """基准权重 × (1 - 触发标记)，输出去重后的信号序列。"""
    aligned_flag = flag.reindex(base_w.index.union(flag.index)).ffill().reindex(base_w.index).fillna(False)
    w = base_w * (~aligned_flag).astype(float)
    w = w[w != w.shift(1)].dropna()
    return w.sort_index()


# ---------- 单资产执行引擎（百分比成本 + 手数规则） ----------

def run_single_asset(df, target_w: pd.Series, start, end, cost, initial=50_000.0):
    """t 收盘信号 → t+1 开盘成交，单标的。

    cost: {"commission": 费率(双边), "slippage": 滑点(单边), "lot": 最小交易单位}
    开盘估值定股数（无前视），先卖后买，无杠杆。
    返回 (equity: pd.Series, trades: list)。
    """
    d = df.set_index("date").sort_index()
    calendar = d.index
    w = target_w.reindex(calendar.union(target_w.index)).sort_index().ffill().reindex(calendar).fillna(0.0)
    targets = w.shift(1).fillna(0.0)
    start, end = pd.Timestamp(start), pd.Timestamp(end)
    comm, slip, lot = cost["commission"], cost["slippage"], cost.get("lot", 1)

    cash, shares, rows, trades = initial, 0.0, [], []
    pending = False
    done_w = 0.0
    for date in calendar:
        if date < start or date > end:
            continue
        tw = float(targets.loc[date])
        if abs(tw - done_w) > 1e-12 or pending:
            o = d["open"].loc[date]
            if pd.notna(o) and o > 0:
                nav_open = cash + shares * o
                desired = int(tw * nav_open / (o * (1 + slip)) / lot) * lot
                if desired < shares:  # 先卖
                    n = shares - desired
                    price = o * (1 - slip)
                    amount = n * price
                    fee = amount * comm
                    cash += amount - fee
                    trades.append(dict(date=date, side="sell", shares=n, price=price,
                                       amount=amount, fee=fee))
                    shares = desired
                if desired > shares:  # 后买
                    n = desired - shares
                    price = o * (1 + slip)
                    while n > 0 and n * price * (1 + comm) > cash + 1e-8:
                        n -= lot
                    if n > 0:
                        amount = n * price
                        fee = amount * comm
                        cash -= amount + fee
                        trades.append(dict(date=date, side="buy", shares=n, price=price,
                                           amount=amount, fee=fee))
                        shares += n
                done_w = tw
                pending = False
            else:
                pending = True  # 缺开盘价顺延
        c = d["close"].loc[date]
        nav = cash + shares * (c if pd.notna(c) else o)
        rows.append((date, nav))
    return pd.Series(dict(rows)).sort_index(), trades


def perf(eq, trades, initial=50_000.0):
    years = max(((eq.index[-1] - eq.index[0]).days + 1) / 365.25, 1 / 365.25)
    total = eq.iloc[-1] / initial - 1
    cagr = (1 + total) ** (1 / years) - 1
    dd = (eq / eq.cummax() - 1).min()
    ret = eq.pct_change().dropna()
    sharpe = ret.mean() / ret.std() * np.sqrt(252) if ret.std() > 0 else np.nan
    calmar = cagr / abs(dd) if dd < 0 else np.nan
    fees = sum(t["fee"] for t in trades)
    amount = sum(t["amount"] for t in trades)
    turnover = amount / eq.mean() / years
    return dict(total=total, cagr=cagr, mdd=dd, sharpe=sharpe, calmar=calmar,
                fees=fees, turnover=turnover)
