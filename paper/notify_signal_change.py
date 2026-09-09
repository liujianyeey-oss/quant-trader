#!/usr/bin/env python3
"""信号变化提醒（独立脚本，只读 paper/ 下 CSV/JSON，不依赖也不影响项目其他代码）。

通知规则（满足其一即弹 macOS 通知）：
1. 持仓标的集合变化（与上次已通知的状态比）；
2. 任一标的权重较上次已通知状态漂移 >= 5 个百分点（staggered 波动控仓的每日微调不打扰，
   但连续小幅爬行累计到 5pp 会触发——与"上次通知时"比而不是与昨天比）；
3. 今日有实际成交且成交金额 >= 账户净值 2%（实盘跟单需要知道的钱级变动）。

通知后更新 paper/.notify_state.json 基线。由 daily_update.sh 在 paper-update 之后调用。
"""
import ast
import csv
import json
import subprocess
import sys
from pathlib import Path

PAPER_DIR = Path(__file__).resolve().parent
STATE_FILE = PAPER_DIR / ".notify_state.json"
STRATEGIES = ("momentum", "staggered")
WEIGHT_THRESHOLD = 0.05      # 权重漂移阈值（对上次已通知状态）
TRADE_VALUE_RATIO = 0.02     # 单笔成交金额占净值比达到此值才通知


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


def todays_trades(strategy: str, date: str):
    path = PAPER_DIR / f"trades_{strategy}.csv"
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as f:
        return [r for r in csv.DictReader(f) if r.get("date") == date]


def nav_of(strategy: str) -> float:
    row = read_last(PAPER_DIR / f"nav_{strategy}.csv")
    try:
        return float(row["nav"]) if row else 0.0
    except (ValueError, KeyError):
        return 0.0


def notify(title: str, message: str):
    """两级降级：terminal-notifier → 模态弹窗（10分钟自动消失）。
    本机 osascript display notification 命令返回成功但通知中心实际不显示（已实测三次），故不用。"""
    tn = "/opt/homebrew/bin/terminal-notifier"
    if Path(tn).exists() and subprocess.run(
            [tn, "-title", title, "-message", message, "-sound", "Glass"],
            check=False, capture_output=True).returncode == 0:
        return
    subprocess.run(
        ["osascript", "-e",
         f'display alert "{title}" message "{message}" giving up after 600'],
        check=False)


def main():
    state = json.loads(STATE_FILE.read_text(encoding="utf-8")) if STATE_FILE.exists() else {}
    new_state = dict(state)
    messages = []

    for s in STRATEGIES:
        row = read_last(PAPER_DIR / f"signals_{s}.csv")
        if not row:
            continue
        curr, date = parse_target(row), row.get("date", "")
        baseline = state.get(s)          # 上次已通知的目标；无记录则以当前为基线静默建档
        if baseline is None:
            new_state[s] = curr
            continue

        codes_changed = set(baseline) != set(curr)
        drift = max((abs(curr.get(c, 0.0) - baseline.get(c, 0.0))
                     for c in set(baseline) | set(curr)), default=0.0)
        trades = [t for t in todays_trades(s, date)
                  if abs(float(t.get("amount", 0))) >= TRADE_VALUE_RATIO * nav_of(s)]

        if codes_changed or drift >= WEIGHT_THRESHOLD or trades:
            fmt = lambda d: ", ".join(f"{c} {w:.0%}" for c, w in d.items()) or "空仓"
            parts = [f"{s}: {fmt(curr)}"]
            if trades:
                t_desc = "、".join(f"{'买' if t['side']=='buy' else '卖'}{t['code']}"
                                  f"{t['shares']}份@{float(t['price']):.3f}" for t in trades)
                parts.append(f"今日已成交：{t_desc}")
            messages.append("（".join([parts[0], "；".join(parts[1:]) + "）" if len(parts) > 1 else ""]).rstrip("（"))
            new_state[s] = curr

    STATE_FILE.write_text(json.dumps(new_state, ensure_ascii=False), encoding="utf-8")

    if messages:
        notify("量化信号变化", " | ".join(messages) + "，次日开盘执行")
        print("已通知:", "; ".join(messages))
    else:
        print("信号无实质变化，不通知")


if __name__ == "__main__":
    sys.exit(main())
