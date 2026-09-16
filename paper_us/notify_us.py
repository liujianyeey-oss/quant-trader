#!/usr/bin/env python3
"""美股台账信号变化提醒（规则同 A股版，独立基线，不读不写 paper/）。

通知规则（满足其一即弹 macOS 通知）：
1. 持仓标的集合/权重变化（donchian 全进全出，任何切换都通知）；
2. 今日有成交且成交金额 >= 账户净值 2%。

通知降级链复用 paper/notify_signal_change.py 的 notify()（只读 import，不改原文件）。
基线状态文件：paper_us/.notify_state_us.json。
"""
import ast
import csv
import importlib.util
import json
import sys
from pathlib import Path

PAPER_DIR = Path(__file__).resolve().parent
PROJECT = PAPER_DIR.resolve().parent
STATE_FILE = PAPER_DIR / ".notify_state_us.json"
STRATEGIES = ("qqq", "spy")
WEIGHT_THRESHOLD = 0.05
TRADE_VALUE_RATIO = 0.02

# 从 A股通知脚本导入 notify()（terminal-notifier → 模态弹窗两级降级）
_spec = importlib.util.spec_from_file_location(
    "notify_a", PROJECT / "paper" / "notify_signal_change.py")
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
notify = _mod.notify


def read_last(csv_path: Path):
    if not csv_path.exists():
        return None
    with csv_path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    return rows[-1] if rows else None


def parse_target(row):
    try:
        d = ast.literal_eval(row["target"]) or {}
        return {k: float(v) for k, v in d.items() if float(v) > 0}
    except (ValueError, SyntaxError, KeyError, TypeError):
        return {}


def main():
    state = json.loads(STATE_FILE.read_text(encoding="utf-8")) if STATE_FILE.exists() else {}
    new_state = dict(state)
    messages = []
    for s in STRATEGIES:
        row = read_last(PAPER_DIR / f"signals_{s}.csv")
        if not row:
            continue
        curr, date = parse_target(row), row.get("date", "")
        baseline = state.get(s)
        if baseline is None:
            new_state[s] = curr
            continue
        codes_changed = set(baseline) != set(curr)
        drift = max((abs(curr.get(c, 0.0) - baseline.get(c, 0.0))
                     for c in set(baseline) | set(curr)), default=0.0)
        nav_row = read_last(PAPER_DIR / f"nav_{s}.csv")
        nav = float(nav_row["nav"]) if nav_row else 0.0
        trades_path = PAPER_DIR / f"trades_{s}.csv"
        trades = []
        if trades_path.exists():
            with trades_path.open(newline="", encoding="utf-8") as f:
                trades = [t for t in csv.DictReader(f)
                          if t.get("date") == date
                          and abs(float(t.get("amount", 0))) >= TRADE_VALUE_RATIO * nav]
        if codes_changed or drift >= WEIGHT_THRESHOLD or trades:
            fmt = lambda d: ", ".join(f"{c} {w:.0%}" for c, w in d.items()) or "空仓"
            part = f"donchian_{s}: {fmt(curr)}"
            if trades:
                t_desc = "、".join(f"{'买' if t['side'] == 'buy' else '卖'}{t['code']}"
                                  f"{t['shares']}股@${float(t['price']):.2f}" for t in trades)
                part += f"（今日已成交：{t_desc}）"
            messages.append(part)
            new_state[s] = curr

    STATE_FILE.write_text(json.dumps(new_state, ensure_ascii=False), encoding="utf-8")
    if messages:
        notify("美股量化信号变化",
               " | ".join(messages) + "，今晚 21:30（北京，夏令时）/22:30（冬令时）美股开盘执行")
        print("已通知:", "; ".join(messages))
    else:
        print("信号无实质变化，不通知")


if __name__ == "__main__":
    sys.exit(main())
