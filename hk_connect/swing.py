"""港股短线：收盘确认、次日开盘执行。日线不假设盘中止损成交。"""
from dataclasses import dataclass, asdict
import math

import numpy as np
import pandas as pd

from .backtest import HKCosts
from .data import normalize_bars


DEFAULTS = dict(kind="breakout", lookback=20, volume_ratio=1.8, max_hold=8,
                max_calendar_days=14, min_turnover=1e8, min_price=3., atr_min=.03,
                atr_max=.08, risk_per_trade=.02, max_weight=.35, max_gross=.9,
                max_positions=3, sector_cap=.45, portfolio_risk=.06,
                min_stop=.04, max_stop=.10, stop_atr=1.5, trail_atr=2.,
                take_profit_r=2., stall_days=3, stall_r=.5, gap_cap=.03,
                pause_drawdown=.10, require_catalyst=False, cooldown=2)


def parameters(overrides=None):
    p = dict(DEFAULTS)
    unknown = set(overrides or {}) - set(p)
    if unknown:
        raise ValueError(f"未知短线参数：{sorted(unknown)}")
    p.update(overrides or {})
    if p["kind"] not in {"breakout", "pullback"}:
        raise ValueError("kind 必须为 breakout 或 pullback")
    for k in ["lookback", "max_hold", "max_calendar_days", "max_positions", "stall_days", "cooldown"]:
        if type(p[k]) is not int or p[k] < (0 if k == "cooldown" else 1):
            raise ValueError(f"{k} 必须为有效整数")
    if p["max_hold"] > 10 or p["max_calendar_days"] > 14 or p["lookback"] < 5:
        raise ValueError("持有上限为 10 个通道日、14 个自然日；lookback 至少 5")
    if type(p["require_catalyst"]) is not bool:
        raise ValueError("require_catalyst 必须为布尔值")
    for k, v in p.items():
        if k not in {"kind", "require_catalyst"} and (not np.isfinite(v) or v < 0):
            raise ValueError(f"无效参数 {k}")
    if not (0 < p["min_stop"] <= p["max_stop"] < 1 and
            0 < p["atr_min"] <= p["atr_max"] < 1 and
            0 < p["max_weight"] <= p["max_gross"] <= 1 and
            0 < p["risk_per_trade"] <= p["portfolio_risk"] <= 1 and
            0 < p["sector_cap"] <= 1 and 0 < p["pause_drawdown"] < 1):
        raise ValueError("风险、仓位或 ATR 参数关系无效")
    return p


def prepare(data, p, events=None):
    """返回日期到候选列表，以及逐日 ATR；不读取未来行。

    信号使用未复权 OHLC；发现复权因子有 >3% 跳变后，整个形态窗口不产生入场。
    避免将推算的复权 high/low 冒称为真实复权 OHLC。
    """
    if p["require_catalyst"] and events is None:
        raise ValueError("事件模式必须提供含发布时间的 events_csv；缺省只测试纯量价")
    approved = {}
    if events is not None:
        for row in events.to_dict("records"):
            if row["approved"]:
                ts = pd.Timestamp(row["published_at"])
                ts = ts.tz_localize("Asia/Hong_Kong") if ts.tz is None else ts.tz_convert("Asia/Hong_Kong")
                approved.setdefault(row["code"], []).append(ts)
    candidates, atrs = {}, {}
    for code, frame in sorted(data.items()):
        d = normalize_bars(frame).set_index("date")
        c, h, l, v = d.close, d.high, d.low, d.volume
        tr = pd.concat([h-l, (h-c.shift()).abs(), (l-c.shift()).abs()], axis=1).max(axis=1)
        atr = tr.rolling(14).mean()
        atrs[code] = atr
        adv = (c*v).rolling(20).mean()
        level = h.shift().rolling(p["lookback"]).max()
        action = (d.signal_close / c).pct_change(fill_method=None).abs() > .03
        clean = ~action.rolling(max(p["lookback"], 30), min_periods=1).max().astype(bool)
        eligible = (adv >= p["min_turnover"]) & (c >= p["min_price"])
        eligible &= (atr/c).between(p["atr_min"], p["atr_max"]) & clean
        eligible &= pd.Series(np.arange(len(d)) >= max(60, p["lookback"]), index=d.index)
        if p["kind"] == "breakout":
            trigger = (c > level) & (v >= v.shift().rolling(20).mean()*p["volume_ratio"])
            trigger &= ((c-l)/(h-l).replace(0, np.nan) >= .75) & (c-level <= atr)
            invalid = level
        else:
            # 上行在回撤开始前已成立；前 3 日为回撤观察期。
            impulse = c.shift(3)/c.shift(13)-1
            old_peak = h.shift(3).rolling(5).max()
            dip = (old_peak-c.shift()) / atr.shift()
            quiet = v.shift().rolling(3).mean() < v.shift(4).rolling(10).mean()
            trigger = (impulse >= .10) & dip.between(.5, 1.5) & quiet
            trigger &= (c > h.shift()) & (c > c.rolling(20).mean())
            # 保留量能恢复要求，volume_ratio=1.8 表示至少回撤均量的 1.8 倍。
            trigger &= v >= v.shift().rolling(3).mean()*p["volume_ratio"]
            level, invalid = h.shift(), l.shift().rolling(3).min()
        values = pd.DataFrame(dict(close=c, atr=atr, level=level, invalid=invalid, adv=adv,
                                   score=(c/c.shift(20)-1)/(atr/c)))
        for date, row in values[eligible & trigger].iterrows():
            if p["require_catalyst"]:
                cutoff = date.tz_localize("Asia/Hong_Kong") + pd.Timedelta(hours=16, minutes=10)
                if not any(cutoff-pd.Timedelta(days=5) <= ts <= cutoff for ts in approved.get(code, [])):
                    continue
            candidates.setdefault(date, []).append(dict(code=code, signal_date=date, **row.to_dict()))
    for rows in candidates.values():
        rows.sort(key=lambda x: (-x["score"], x["code"]))
    return candidates, atrs


@dataclass
class Position:
    code: str
    sector: str
    entered: pd.Timestamp
    deadline: pd.Timestamp
    entry: float
    shares: int
    original_shares: int
    stop: float
    risk_distance: float
    entry_cost: float
    planned_risk_per_share: float = 0.
    proceeds: float = 0.
    days: int = 0
    high_close: float = 0.
    trimmed: bool = False
    pending: str = ""
    overdue: bool = False


class SwingEngine:
    def __init__(self, params, universe, cost, sectors, initial_cash=500000,
                 calendar=None, research=False):
        self.p = parameters(params)
        self.u, self.costs, self.sectors = universe, HKCosts(cost), sectors
        self.initial = float(initial_cash)
        if not math.isfinite(self.initial) or self.initial <= 0:
            raise ValueError("研究资金必须为正 HKD 数额")
        self.calendar, self.research = calendar, research
        if not research and (universe.mode == "snapshot" or calendar is None):
            raise ValueError("当前资格/缺真实通道日历：必须显式 --research；这是有偏开发回测")

    def _deadline(self, date):
        # 入场日为第 1 个自然日，确保收盘退出也早于成交后满 14×24 小时。
        end = date + pd.Timedelta(days=self.p["max_calendar_days"]-1)
        if self.calendar is None:
            future = pd.bdate_range(date, end)
        else:
            # 日历必须含关闭日及向后 14 天覆盖，不能把缺行当假期。
            needed = pd.date_range(date, end)
            if not needed.isin(self.calendar.index).all():
                raise ValueError("通道日历需逐自然日覆盖入场后 14 天")
            future = self.calendar.loc[date:end].loc[lambda x: x].index
        if len(future) == 0:
            raise ValueError("没有已知退出通道日")
        return future[min(self.p["max_hold"], len(future))-1]

    def run(self, data, start, end, prepared=None, events=None):
        start, end = pd.Timestamp(start), pd.Timestamp(end)
        if start > end:
            raise ValueError("日期起止倒置")
        frames = {c: normalize_bars(d).set_index("date") for c, d in data.items()}
        if not frames or set(frames)-set(self.sectors):
            raise ValueError("行情为空或缺少股票题材分类")
        dates = pd.DatetimeIndex(sorted(set().union(*(d.index for d in frames.values()))))
        if self.calendar is not None and not dates[(dates >= start) & (dates <= end)].isin(self.calendar.index).all():
            raise ValueError("通道日历缺少回测行情日期")
        # 对截断样本保持同一预热口径；空仓开始，但首日可承接昨日候选。
        candidates, atrs = prepared if prepared is not None else prepare(data, self.p, events)
        bars = {c: d.to_dict("index") for c, d in frames.items()}
        self.positions, self.trades, self.roundtrips, self.audit = {}, [], [], []
        cash, peak, paused = self.initial, self.initial, False
        marks, factors, closed_at, rows = {}, {}, {}, []
        self.pause_date = None
        prev = None
        for date in dates:
            if date > end:
                break
            current = {c: b[date] for c, b in bars.items() if date in b}
            if date < start:
                marks.update({c: r["close"] for c, r in current.items()})
                factors.update({c:r["signal_close"]/r["close"] for c,r in current.items()})
                prev = date
                continue
            is_open = date.weekday() < 5 if self.calendar is None else bool(self.calendar.loc[date])
            exited_today = set()
            # 卖出独立处理：一只停牌不会冻结另一只已可卖的风险退出。
            for code, pos in list(self.positions.items()):
                if is_open:
                    pos.days += 1
                if date > pos.deadline and not pos.overdue:
                    pos.overdue = True
                    self.audit.append(dict(date=date, code=code, reason="deadline_breach"))
                if not self.u.eligible(code, date, self.research):
                    pos.pending = "eligibility_exit"
                if date > pos.deadline:
                    pos.pending = "deadline_retry"
                row = current.get(code)
                if row is not None and code in factors:
                    factor_move = (row["signal_close"]/row["close"])/factors[code]-1
                    if abs(factor_move) > .10:
                        raise ValueError(f"{code} 持仓中疑似拆并股/大额分派，需公司行为模型，拒绝伪造净值")
                if pos.pending and is_open and row is not None:
                    half = pos.pending == "take_profit"
                    cash += self._sell(pos, row["open"], date, pos.pending, half)
                    exited_today.add(code)
                    if pos.shares == 0:
                        del self.positions[code]
                        closed_at[code] = date
                    else:
                        pos.pending = ""
            # 买入只依赖上一行情日信号和当前开盘价；不使用当天高低收及成交量定手数。
            open_prices = {c: r["open"] for c, r in current.items()}
            nav = cash + sum(pos.shares*open_prices.get(c, marks.get(c)) for c, pos in self.positions.items())
            if is_open and not paused:
                for signal in candidates.get(prev, []):
                    code = signal["code"]
                    if len(self.positions) >= self.p["max_positions"]:
                        break
                    if code in self.positions or code in exited_today or code not in open_prices:
                        continue
                    if (date-closed_at.get(code, pd.Timestamp("1900-01-01"))).days <= self.p["cooldown"]:
                        continue
                    if not self.u.eligible(code, date, self.research):
                        continue
                    price = open_prices[code]
                    if price < signal["level"] or price > signal["close"]*(1+min(self.p["gap_cap"], .5*signal["atr"]/signal["close"])):
                        continue
                    fill = price*(1+self.costs.config["slippage"])
                    stop = min(signal["invalid"]-.5*signal["atr"], fill-self.p["stop_atr"]*signal["atr"])
                    stop = min(stop, fill*(1-self.p["min_stop"]))
                    if stop <= 0 or (fill-stop)/fill > self.p["max_stop"]:
                        continue
                    sector = self.sectors[code]
                    gross = sum(pos.shares*open_prices.get(c, marks.get(c)) for c, pos in self.positions.items())
                    exposure = sum(pos.shares*open_prices.get(c, marks.get(c)) for c, pos in self.positions.items() if pos.sector == sector)
                    risk_used = sum(pos.shares*pos.planned_risk_per_share for pos in self.positions.values())
                    risk_budget = min(nav*self.p["risk_per_trade"], max(0, nav*self.p["portfolio_risk"]-risk_used))
                    room = min(cash, nav*self.p["max_weight"], nav*self.p["max_gross"]-gross,
                               nav*self.p["sector_cap"]-exposure, signal["adv"]*.002)
                    lot = self.u.lot(code, date, self.research)
                    n = max(0, int(min(room/fill, risk_budget/(fill-stop))/lot)*lot)
                    while n:
                        entry_fee = self._fee(code, date, n*fill)
                        exit_fill = stop*(1-self.costs.config["slippage"])
                        risk = n*(fill-exit_fill)+entry_fee+self._fee(code, date, n*exit_fill)
                        if n*fill+entry_fee <= cash and risk <= risk_budget:
                            break
                        n -= lot
                    if not n:
                        continue
                    cost = n*fill+self._fee(code, date, n*fill)
                    cash -= cost
                    self.positions[code] = Position(code, sector, date, self._deadline(date), fill, n, n,
                                                    stop, fill-stop, cost, planned_risk_per_share=risk/n,
                                                    days=1, high_close=fill)
                    self.trades.append(dict(date=date, code=code, side="buy", shares=n, price=fill,
                                            fee_hkd=cost-n*fill, reason="entry", signal_date=prev))
            marks.update({c: r["close"] for c, r in current.items()})
            factors.update({c:r["signal_close"]/r["close"] for c,r in current.items()})
            for code, pos in list(self.positions.items()):
                row = current.get(code)
                if date >= pos.deadline:
                    if is_open and row is not None:
                        cash += self._sell(pos, row["close"], date, "time_exit", False)
                        del self.positions[code]
                        closed_at[code] = date
                    else:
                        pos.pending = "deadline_retry"
                    continue
                if row is None:
                    continue
                close = row["close"]
                # 昨日失效指令未完成时不可被新的获利信号覆盖。
                if pos.pending and pos.pending != "take_profit":
                    continue
                if close <= pos.stop:
                    pos.pending = "stop_close"
                elif pos.days >= self.p["stall_days"] and max(pos.high_close, close)-pos.entry < self.p["stall_r"]*pos.risk_distance:
                    pos.pending = "stall_exit"
                elif not pos.trimmed and close-pos.entry >= self.p["take_profit_r"]*pos.risk_distance:
                    pos.pending = "take_profit"
                pos.high_close = max(pos.high_close, close)
                a = atrs[code].get(date, np.nan)
                if pos.trimmed and pd.notna(a):
                    pos.stop = max(pos.stop, pos.high_close-self.p["trail_atr"]*a)
            total = cash + sum(pos.shares*marks[c] for c, pos in self.positions.items())
            peak = max(peak, total)
            if not paused and total/peak-1 <= -self.p["pause_drawdown"]:
                paused, self.pause_date = True, date  # 固定样本内暂停新增，需人工复核后另启新账户。
            if cash < -1e-6:
                raise AssertionError("现金透支")
            rows.append(dict(date=date, total_value=total, cash_hkd=cash,
                             gross_weight=(total-cash)/total, positions=len(self.positions), paused=paused))
            prev = date
        if not rows:
            raise ValueError("区间无行情")
        equity = pd.DataFrame(rows).set_index("date")
        self.last_candidates = candidates.get(equity.index[-1], [])
        return equity, self.trades

    def _fee(self, code, date, amount):
        row = self.u.info(code, date, self.research)
        if row is None:
            past = [r for r in self.u.records.get(code, []) if r["begin"] <= date]
            row = past[-1] if past else {}
        return self.costs.fee(amount, row.get("stamp_duty", True))

    def _sell(self, pos, reference, date, reason, half):
        lot = self.u.lot(pos.code, date, self.research)
        if pos.shares % lot:
            raise ValueError("持仓存在碎股/手数变更，拒绝按新手数伪造处置")
        n = int(pos.shares/2/lot)*lot if half else pos.shares
        if n == 0:
            n = pos.shares
        price = reference*(1-self.costs.config["slippage"])
        fee = self._fee(pos.code, date, n*price)
        proceeds = n*price-fee
        pos.proceeds += proceeds
        pos.shares -= n
        if half:
            pos.trimmed = True
        self.trades.append(dict(date=date, code=pos.code, side="sell", shares=n, price=price,
                                fee_hkd=fee, reason=reason, signal_date=None))
        if not pos.shares:
            self.roundtrips.append(dict(code=pos.code, entry_date=pos.entered, exit_date=date,
                                       calendar_days=(date-pos.entered).days, sessions=pos.days,
                                       pnl_hkd=pos.proceeds-pos.entry_cost, return_net=pos.proceeds/pos.entry_cost-1,
                                       reason=reason, deadline=pos.deadline, overdue=pos.overdue))
        return proceeds

    def stats(self, equity):
        series = equity.total_value
        dd = series/series.cummax().clip(lower=self.initial)-1
        pnl = pd.Series([r["pnl_hkd"] for r in self.roundtrips], dtype=float)
        wins, losses = pnl[pnl>0].sum(), -pnl[pnl<0].sum()
        return dict(total_return=float(series.iloc[-1]/self.initial-1), max_drawdown=float(dd.min()),
                    mean_gross=float(equity.gross_weight.mean()), roundtrips=len(pnl),
                    win_rate=float((pnl>0).mean()) if len(pnl) else None,
                    profit_factor=float(wins/losses) if losses else None,
                    fees_hkd=sum(t["fee_hkd"] for t in self.trades),
                    max_calendar_days=max((r["calendar_days"] for r in self.roundtrips), default=0),
                    max_sessions=max((r["sessions"] for r in self.roundtrips), default=0),
                    deadline_breaches=len(self.audit), open_positions=len(self.positions),
                    pause_date=str(self.pause_date.date()) if self.pause_date is not None else None)

    def position_records(self):
        return [asdict(p) for p in self.positions.values()]
