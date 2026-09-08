#!/usr/bin/env python3
"""信号变化提醒（独立脚本，只读 paper/signals_*.csv，不依赖也不影响项目其他代码）。

规则：对比每个策略最近两条信号的生效目标——
- 持仓标的集合发生变化，或
- 任一标的权重变化 >= 5 个百分点（staggered 的波动控仓微调不打扰，实质性调仓才提醒）
满足其一即弹 macOS 通知。由 paper/daily_update.sh 在 paper-update 之后调用；也可手动执行。
"""
import ast
import csv
import subprocess
import sys
from pathlib import Path

PAPER_DIR = Path(__file__).resolve().parent
STRATEGIES = ("momentum", "staggered")
WEIGHT_THRESHOLD = 0.05


def last_two_targets(csv_path: Path):
    if not csv_path.exists():
        return None
    with csv_path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if len(rows) < 2:
        return None
    try:
        prev = ast.literal_eval(rows[-2]["target"]) or {}
        curr = ast.literal_eval(rows[-1]["target"]) or {}
    except (ValueError, SyntaxError, KeyError):
        return None
    return {k: float(v) for k, v in prev.items() if float(v) > 0}, \
           {k: float(v) for k, v in curr.items() if float(v) > 0}, \
           rows[-1].get("date", "")


def notify(title: str, message: str):
    subprocess.run(
        ["osascript", "-e", f'display notification "{message}" with title "{title}"'],
        check=False,
    )


def main():
    changes = []
    for s in STRATEGIES:
        res = last_two_targets(PAPER_DIR / f"signals_{s}.csv")
        if not res:
            continue
        prev, curr, date = res
        codes_changed = set(prev) != set(curr)
        big_weight_change = any(
            abs(curr.get(c, 0.0) - prev.get(c, 0.0)) >= WEIGHT_THRESHOLD
            for c in set(prev) | set(curr)
        )
        if codes_changed or big_weight_change:
            fmt = lambda d: ", ".join(f"{c} {w:.0%}" for c, w in d.items()) or "空仓"
            changes.append(f"{s}: {fmt(curr)}")

    if changes:
        notify("量化信号变化", " | ".join(changes) + "（次日开盘执行）")
        print("已通知:", "; ".join(changes))
    else:
        print("信号无实质变化，不通知")


if __name__ == "__main__":
    sys.exit(main())
