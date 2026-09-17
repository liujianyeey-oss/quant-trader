#!/usr/bin/env python3
"""BTC×F1 台账信号变化提醒（规则同 A股/美股版，独立基线）。

触发：持仓状态变化（全进全出，任何切换）或当日成交 >= 净值 2%。
通知降级链复用 paper/notify_signal_change.py 的 notify()（只读 import）。
"""
import ast
import csv
import importlib.util
import json
import sys
from pathlib import Path

PAPER_DIR = Path(__file__).resolve().parent
PROJECT = PAPER_DIR.resolve().parent
STATE_FILE = PAPER_DIR / ".notify_state_btc.json"
NAME = "btc_f1"
TRADE_VALUE_RATIO = 0.02

_spec = importlib.util.spec_from_file_location(
    "notify_a", PROJECT / "paper" / "notify_signal_change.py")
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
notify = _mod.notify


def read_last(p):
    if not p.exists():
        return None
    with p.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    return rows[-1] if rows else None


def main():
    state = json.loads(STATE_FILE.read_text(encoding="utf-8")) if STATE_FILE.exists() else {}
    row = read_last(PAPER_DIR / f"signals_{NAME}.csv")
    if not row:
        print("无信号记录")
        return
    try:
        curr = {k: float(v) for k, v in (ast.literal_eval(row["target"]) or {}).items()
                if float(v) > 0}
    except (ValueError, SyntaxError):
        curr = {}
    date = row.get("date", "")
    baseline = state.get(NAME)
    if baseline is None:
        state[NAME] = curr
        STATE_FILE.write_text(json.dumps(state, ensure_ascii=False), encoding="utf-8")
        print("基线建档，不通知")
        return
    nav_row = read_last(PAPER_DIR / f"nav_{NAME}.csv")
    nav = float(nav_row["nav"]) if nav_row else 0.0
    trades = []
    tp = PAPER_DIR / f"trades_{NAME}.csv"
    if tp.exists():
        with tp.open(newline="", encoding="utf-8") as f:
            trades = [t for t in csv.DictReader(f)
                      if t.get("date") == date
                      and abs(float(t.get("amount", 0))) >= TRADE_VALUE_RATIO * nav]
    changed = set(baseline) != set(curr)
    if changed or trades:
        pos = "持有 BTC" if curr else "空仓"
        part = f"BTC×F1: {pos}"
        if trades:
            t = trades[-1]
            part += (f"（已成交：{'买' if t['side'] == 'buy' else '卖'} "
                     f"{t['shares']}个 @ ${float(t['price']):,.0f}）")
        notify("BTC 量化信号变化", part + "，信号按币安日线 UTC 收盘、次根 K 线开盘成交")
        print("已通知:", part)
        state[NAME] = curr
        STATE_FILE.write_text(json.dumps(state, ensure_ascii=False), encoding="utf-8")
    else:
        print("信号无实质变化，不通知")


if __name__ == "__main__":
    sys.exit(main())
