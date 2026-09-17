#!/usr/bin/env python3
"""观察账户 6/7 信号变化提醒（规则同其他台账，独立基线）。"""
import ast
import csv
import importlib.util
import json
import sys
from pathlib import Path

PAPER_DIR = Path(__file__).resolve().parent
PROJECT = PAPER_DIR.resolve().parent
STATE_FILE = PAPER_DIR / ".notify_state_obs.json"
ACCOUNTS = {
    "qqq_h2": "今晚 21:30（北京，夏令时）/22:30（冬令时）美股开盘执行",
    "hs_rsi2": "次日 09:30 A股开盘执行",
}
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
    messages = []
    for name, exec_note in ACCOUNTS.items():
        row = read_last(PAPER_DIR / f"signals_{name}.csv")
        if not row:
            continue
        try:
            curr = {k: float(v) for k, v in (ast.literal_eval(row["target"]) or {}).items()
                    if float(v) > 0}
        except (ValueError, SyntaxError):
            curr = {}
        baseline = state.get(name)
        if baseline is None:
            state[name] = curr
            continue
        nav_row = read_last(PAPER_DIR / f"nav_{name}.csv")
        nav = float(nav_row["nav"]) if nav_row else 0.0
        trades = []
        tp = PAPER_DIR / f"trades_{name}.csv"
        if tp.exists():
            with tp.open(newline="", encoding="utf-8") as f:
                trades = [t for t in csv.DictReader(f)
                          if t.get("date") == row.get("date", "")
                          and abs(float(t.get("amount", 0))) >= TRADE_VALUE_RATIO * nav]
        if set(baseline) != set(curr) or trades:
            pos = f"持有 {list(curr)[0]}" if curr else "空仓"
            part = f"{name}: {pos}"
            if trades:
                t = trades[-1]
                part += (f"（已成交：{'买' if t['side'] == 'buy' else '卖'}"
                         f"{t['shares']} @ {float(t['price']):.3f}）")
            messages.append(part + f"，{exec_note}")
            state[name] = curr

    STATE_FILE.write_text(json.dumps(state, ensure_ascii=False), encoding="utf-8")
    if messages:
        notify("观察账户信号变化", " | ".join(messages))
        print("已通知:", "; ".join(messages))
    else:
        print("信号无实质变化，不通知")


if __name__ == "__main__":
    sys.exit(main())
