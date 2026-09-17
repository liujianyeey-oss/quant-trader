"""个人手动操作台账（与策略账户完全隔离）。

用途：记录用户自己的主观交易（不占用任何策略账户），每日跟踪盈亏、
与"系统同期的机械化立场"对比、关键价位状态、回撤与持仓时长。

用法：
  python -m my_trades.track update            # 每日跟踪（BTC 收盘后）
  python -m my_trades.track report            # 输出评估报表
  python -m my_trades.track set-entry BTC 76206 2026-09-17   # 修正成交价/日期
  python -m my_trades.track add --code BTC --side long --weight 1.0 --price 76206 --date 2026-09-17
"""
import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

PROJECT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT)
DIR = os.path.dirname(os.path.abspath(__file__))
POS_FILE = os.path.join(DIR, "positions.json")
NAV_FILE = os.path.join(DIR, "nav.csv")
LOG_FILE = os.path.join(DIR, "evaluation_log.md")


def load_positions():
    if not os.path.exists(POS_FILE):
        return {"positions": []}
    return json.loads(open(POS_FILE, encoding="utf-8").read())


def save_positions(p):
    with open(POS_FILE, "w", encoding="utf-8") as f:
        json.dump(p, f, ensure_ascii=False, indent=2)


def get_prices(codes):
    out = {}
    for c in codes:
        if c.upper() == "BTC":
            from macro_research.data_layer import load_btc
            df, _ = load_btc()
            df = df.rename(columns=str.lower)
            out["BTC"] = df.set_index(pd.to_datetime(df["date"]))["close"].sort_index()
        else:
            from winrate import load_asset
            d = load_asset(c)
            out[c] = d.set_index(pd.to_datetime(d["date"]))["close"].sort_index()
    return out


def add_position(code, side, weight, price, date):
    p = load_positions()
    p["positions"].append({"code": code.upper(), "side": side, "weight": weight,
                           "entry_price": float(price), "entry_date": date,
                           "note": "用户手动操作，非策略信号"})
    save_positions(p)
    print(f"已记录：{side} {code} 权重 {weight:.0%} @ {price}（{date}）")


def set_entry(code, price, date=None):
    p = load_positions()
    hit = False
    for pos in p["positions"]:
        if pos["code"] == code.upper() and pos.get("active", True):
            pos["entry_price"] = float(price)
            if date:
                pos["entry_date"] = date
            hit = True
    save_positions(p)
    print("已修正" if hit else "未找到该标的的活跃持仓")


def snapshot():
    """按各市场最新收盘价计算持仓盈亏，追加 nav 记录。"""
    p = load_positions()
    if not p["positions"]:
        print("无持仓记录")
        return
    prices = get_prices([pos["code"] for pos in p["positions"]])
    rows = []
    for pos in p["positions"]:
        s = prices[pos["code"]]
        cur = float(s.iloc[-1])
        ret = (cur / pos["entry_price"] - 1) * (1 if pos["side"] == "long" else -1)
        rows.append({"code": pos["code"], "date": str(s.index[-1].date()),
                     "price": cur, "entry": pos["entry_price"],
                     "ret": ret, "weight": pos["weight"]})
    df = pd.DataFrame(rows)
    df["date"] = df["date"].astype(str)
    port_ret = float((df["ret"] * df["weight"]).sum() / max(df["weight"].sum(), 1e-9))
    hdr = not os.path.exists(NAV_FILE)
    with open(NAV_FILE, "a", encoding="utf-8") as f:
        if hdr:
            f.write("date,price_btc,ret_btc,port_ret\n")
        f.write(f"{df['date'].iloc[0]},{df['price'].iloc[0]:.2f},"
                f"{df['ret'].iloc[0]:.6f},{port_ret:.6f}\n")
    print(f"已记录 {df['date'].iloc[0]}：BTC {df['price'].iloc[0]:,.0f}，"
          f"持仓收益 {df['ret'].iloc[0]:+.2%}")
    return df


def report():
    p = load_positions()
    if not p["positions"] or not os.path.exists(NAV_FILE):
        print("暂无记录，先用 add 记录持仓")
        return
    nav = pd.read_csv(NAV_FILE, parse_dates=["date"]).sort_values("date")
    pos = p["positions"][0]
    s = get_prices([pos["code"]])[pos["code"]].reindex(
        nav["date"]).ffill()
    # 策略参照：BTC×F1 账户 5 的实际仓位（若台账可用）
    strat_note = "系统 BTC×F1 账户同期为空仓（无买入信号）"
    try:
        sig = pd.read_csv(os.path.join(PROJECT, "paper_btc", "signals_btc.csv"))
        strat_note = f"系统 BTC×F1 账户最后信号：{sig.iloc[-1]['date']} " \
                     f"{sig.iloc[-1]['target']}"
    except Exception:  # noqa: BLE001
        pass
    entry, cur = pos["entry_price"], s.iloc[-1]
    ret = cur / entry - 1
    peak = s.cummax()
    dd_from_entry = (s / s.iloc[0] - 1)
    lines = [
        f"===== 个人操作评估（{pos['code']}，{pos['side']}）=====",
        f"记录：{pos['entry_date']} 入场 @ {entry:,.2f}，权重 {pos['weight']:.0%}",
        f"最新：{s.index[-1]:%Y-%m-%d} @ {cur:,.2f}，收益 {ret:+.2%}",
        f"期间最高 {s.max():,.0f} / 最低 {s.min():,.0f}，最大浮亏 {dd_from_entry.min():+.2%}",
        f"系统立场：{strat_note}",
        "",
        "关键参考位（按系统口径）：",
        "  向上：放量收盘站上 20 日高点 → 系统会跟进（该位动态更新）",
        "  向下：跌破 10 日低点 → 系统视为趋势走坏区间",
    ]
    # 动态关键位
    try:
        from macro_research.data_layer import load_btc
        b, _ = load_btc()
        b = b.rename(columns=str.lower)
        hi20 = b["high"].rolling(20).max().shift(1).iloc[-1]
        lo10 = b["low"].rolling(10).min().shift(1).iloc[-1]
        lines[-2] = f"  向上：放量收盘站上 ${hi20:,.0f}（20日高点）→ 系统会跟进"
        lines[-1] = f"  向下：跌破 ${lo10:,.0f}（10日低点）→ 系统视为趋势走坏区间"
    except Exception:  # noqa: BLE001
        pass
    text = "\n".join(lines)
    print(text)
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(f"\n## {pd.Timestamp.now():%Y-%m-%d}\n{text}\n")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd")
    a = sub.add_parser("add")
    a.add_argument("--code", required=True)
    a.add_argument("--side", default="long")
    a.add_argument("--weight", type=float, default=1.0)
    a.add_argument("--price", type=float, required=True)
    a.add_argument("--date", required=True)
    s1 = sub.add_parser("set-entry")
    s1.add_argument("code")
    s1.add_argument("price", type=float)
    s1.add_argument("date", nargs="?", default=None)
    sub.add_parser("update")
    sub.add_parser("report")
    args = ap.parse_args()
    if args.cmd == "add":
        add_position(args.code, args.side, args.weight, args.price, args.date)
    elif args.cmd == "set-entry":
        set_entry(args.code, args.price, args.date)
    elif args.cmd == "update":
        snapshot()
    elif args.cmd == "report":
        report()
    else:
        ap.print_help()


if __name__ == "__main__":
    main()
