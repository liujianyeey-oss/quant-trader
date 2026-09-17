"""R11 组合引擎：月度调仓 30 只等权组合，全执行约束建模。

约束（全部建模，不许省略）：
- 个股佣金万2.5（双边）+ 卖出印花税 0.05% + 滑点 0.1%（单边，压力测试 50bps）
- 100 股整手、单票权重 ≤5%（30 只等权天然满足）、现金不足按比例缩仓
- 涨跌停跳过：开盘价 ≥ 前收 ×(1+limit-0.1%) 视为涨停禁买；≤ 前收×(1-limit+0.1%)
  视为跌停禁卖（主板 ±10%，创业板/科创板 ±20%，按代码前缀近似，声明）
- 被跳过/无开盘价的订单顺延至下一交易日重试（pending 机制）
- 信号 T 收盘 → T+1 开盘执行
"""
import numpy as np
import pandas as pd

COST_STOCK = {"commission": 0.00025, "stamp": 0.0005, "slippage": 0.001}
LOT = 100


def _limit_pct(code):
    return 0.20 if code.startswith(("30", "68")) else 0.10


def run_portfolio(data: dict, signals: list, start, end, initial=1_000_000.0,
                  slippage=None, cost=COST_STOCK):
    """signals: [(signal_date, {code: weight})]，按信号日排序。
    返回 (equity: pd.Series, trades: list, stats: dict)。"""
    slip = cost["slippage"] if slippage is None else slippage
    comm, stamp = cost["commission"], cost["stamp"]
    closes = {c: df.set_index("date")["close"] for c, df in data.items()}
    opens = {c: df.set_index("date")["open"] for c, df in data.items()}
    close_panel = pd.DataFrame(closes).sort_index()
    open_panel = pd.DataFrame(opens).sort_index().reindex(close_panel.index)
    marks = close_panel.ffill()
    calendar = close_panel.index
    start, end = pd.Timestamp(start), pd.Timestamp(end)

    sig_map = {pd.Timestamp(d): w for d, w in signals}
    cash, shares = initial, {}
    trades = []
    rows = []
    pending = None        # (signal_date, {code: weight})
    blocked_orders = []   # 顺延订单 [{code, side, shares}]
    stats = {"limit_up_skips": 0, "limit_down_skips": 0, "no_open_skips": 0,
             "retry_days": 0}

    for date in calendar:
        if date < start or date > end:
            continue
        prev_close_row = close_panel.shift(1).loc[date] if date != calendar[0] else None

        # 新信号登记（信号日次日才执行）
        if pending is None and date in sig_map and not blocked_orders:
            pending = (date, sig_map[date])

        # 执行（pending 信号 + 顺延订单）
        if pending is not None and date > pending[0] or (pending is None and blocked_orders):
            o = open_panel.loc[date]
            pc = prev_close_row
            orders = list(blocked_orders)
            blocked_orders = []

            if pending is not None:
                tgt = pending[1]
                # 顺延中的订单不再参与本次重算（只在下方重试，防重复成交）
                retry_codes = {od["code"] for od in orders}
                # 目标股数（开盘价定股数，无前视）
                nav_open = cash + sum(n * o.get(c, np.nan) for c, n in shares.items()
                                      if n > 0 and pd.notna(o.get(c, np.nan)))
                desired = {}
                for c, w in tgt.items():
                    if c in retry_codes:
                        continue
                    p = o.get(c, np.nan)
                    buy_price = p * (1 + slip) if pd.notna(p) and p > 0 else np.nan
                    desired[c] = int(w * nav_open / buy_price / LOT) * LOT \
                        if pd.notna(buy_price) else None
                # 卖出：持仓中超出目标或不在目标的（顺延订单代码由下方重试管理，跳过）
                for c in list(shares):
                    if c in retry_codes:
                        continue
                    cur = shares[c]
                    want = desired.get(c, 0) or 0
                    if cur > want:
                        p = o.get(c, np.nan)
                        if pd.isna(p) or p <= 0:
                            blocked_orders.append({"code": c, "side": "sell",
                                                   "shares": cur - want})
                            stats["no_open_skips"] += 1
                            continue
                        lim = _limit_pct(c)
                        if pc is not None and pd.notna(pc.get(c, np.nan)) and \
                                p <= pc[c] * (1 - lim + 0.001):
                            blocked_orders.append({"code": c, "side": "sell",
                                                   "shares": cur - want})
                            stats["limit_down_skips"] += 1
                            continue
                        n = cur - want
                        price = p * (1 - slip)
                        amount = n * price
                        fee = amount * (comm + stamp)
                        cash += amount - fee
                        trades.append({"date": date, "code": c, "side": "sell",
                                       "shares": n, "price": price, "amount": amount,
                                       "fee": fee})
                        shares[c] = want
                        if want == 0:
                            shares.pop(c)
                # 买入：先算总需求，现金不足按比例缩
                buys = []
                for c, want in desired.items():
                    if want is None:
                        stats["no_open_skips"] += 1
                        continue
                    cur = shares.get(c, 0)
                    if want > cur:
                        p = o.get(c, np.nan)
                        lim = _limit_pct(c)
                        if pc is not None and pd.notna(pc.get(c, np.nan)) and \
                                p >= pc[c] * (1 + lim - 0.001):
                            blocked_orders.append({"code": c, "side": "buy",
                                                   "shares": want - cur})
                            stats["limit_up_skips"] += 1
                            continue
                        buys.append((c, want - cur, p))
                need = sum(n * p * (1 + slip) * (1 + comm) for c, n, p in buys)
                ratio = min(1.0, cash / need) if need > 0 else 1.0
                for c, n, p in buys:
                    n2 = int(n * ratio / LOT) * LOT
                    if n2 <= 0:
                        if ratio < 1.0:
                            blocked_orders.append({"code": c, "side": "buy",
                                                   "shares": n})
                        continue
                    price = p * (1 + slip)
                    amount = n2 * price
                    fee = amount * comm
                    cash -= amount + fee
                    trades.append({"date": date, "code": c, "side": "buy",
                                   "shares": n2, "price": price, "amount": amount,
                                   "fee": fee})
                    shares[c] = shares.get(c, 0) + n2
                    # 现金不足产生的未买足残额视为调仓完成（与生产"缩仓即完成"同口径），
                    # 不顺延；顺延只用于涨停/无开盘价
                pending = None if not blocked_orders else pending

            # 顺延订单（沿用当日开盘价，含涨跌停/无价格再跳过的逻辑）
            for od in orders:
                c, side, n = od["code"], od["side"], od["shares"]
                p = o.get(c, np.nan)
                if pd.isna(p) or p <= 0:
                    blocked_orders.append(od)
                    stats["no_open_skips"] += 1
                    continue
                lim = _limit_pct(c)
                if side == "buy" and pc is not None and pd.notna(pc.get(c, np.nan)) and \
                        p >= pc[c] * (1 + lim - 0.001):
                    blocked_orders.append(od)
                    stats["limit_up_skips"] += 1
                    continue
                if side == "sell" and pc is not None and pd.notna(pc.get(c, np.nan)) and \
                        p <= pc[c] * (1 - lim + 0.001):
                    blocked_orders.append(od)
                    stats["limit_down_skips"] += 1
                    continue
                if side == "sell":
                    price = p * (1 - slip)
                    amount = n * price
                    fee = amount * (comm + stamp)
                    cash += amount - fee
                    shares[c] = shares.get(c, 0) - n
                    if shares.get(c, 0) <= 0:
                        shares.pop(c, None)
                    trades.append({"date": date, "code": c, "side": "sell",
                                   "shares": n, "price": price, "amount": amount,
                                   "fee": fee})
                else:
                    price = p * (1 + slip)
                    n2 = min(n, int(cash / (price * (1 + comm)) / LOT) * LOT)
                    if n2 > 0:
                        amount = n2 * price
                        fee = amount * comm
                        cash -= amount + fee
                        shares[c] = shares.get(c, 0) + n2
                        trades.append({"date": date, "code": c, "side": "buy",
                                       "shares": n2, "price": price, "amount": amount,
                                       "fee": fee})
                    # 现金不足未买足的残额直接放弃（缩仓即完成），不顺延
            if blocked_orders:
                stats["retry_days"] += 1

        # 收盘估值
        nav = cash + sum(n * marks.loc[date, c] for c, n in shares.items()
                         if n > 0 and pd.notna(marks.loc[date, c]))
        rows.append((date, nav))

    return pd.Series(dict(rows)).sort_index(), trades, stats
