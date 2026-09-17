#!/usr/bin/env python3
"""观察账户 8 号（低波 top30 月调）信号变化提醒：调仓日有成交时通知。"""
import csv
import importlib.util
import json
import sys
from pathlib import Path

PAPER_DIR = Path(__file__).resolve().parent
PROJECT = PAPER_DIR.resolve().parent
STATE_FILE = PAPER_DIR / ".notify_state_ms.json"
NAME = "lowvol30"

_spec = importlib.util.spec_from_file_location(
    "notify_a", PROJECT / "paper" / "notify_signal_change.py")
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
notify = _mod.notify


def main():
    trades_path = PAPER_DIR / f"trades_{NAME}.csv"
    state = json.loads(STATE_FILE.read_text(encoding="utf-8")) if STATE_FILE.exists() else {}
    today_trades = []
    if trades_path.exists():
        with trades_path.open(newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        if rows:
            last_date = rows[-1]["date"]
            today_trades = [r for r in rows if r["date"] == last_date]
    last_notified = state.get("last_trade_date")
    if today_trades and today_trades[0]["date"] != last_notified:
        buys = [t for t in today_trades if t["side"] == "buy"]
        sells = [t for t in today_trades if t["side"] == "sell"]
        msg = (f"低波30调仓（{today_trades[0]['date']} 执行）：买 {len(buys)} 只 / "
               f"卖 {len(sells)} 只，买入 {','.join(t['code'] for t in buys[:5])}"
               f"{'...' if len(buys) > 5 else ''}，次日开盘已成交口径")
        notify("低波30组合调仓", msg)
        print("已通知:", msg)
        state["last_trade_date"] = today_trades[0]["date"]
    else:
        print("无新调仓成交，不通知")
    STATE_FILE.write_text(json.dumps(state, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    sys.exit(main())
