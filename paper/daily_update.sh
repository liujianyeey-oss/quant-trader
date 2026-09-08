#!/bin/bash
# 模拟盘台账每日更新（cron 用）。cron 环境没有用户 PATH，一律绝对路径。
# crontab 示例（工作日 16:47 收盘后运行）：
#   47 16 * * 1-5 /Users/liujianye10/quant-trader/paper/daily_update.sh
cd /Users/liujianye10/quant-trader || exit 1
mkdir -p logs
.venv/bin/python main.py paper-update >> logs/paper_cron.log 2>&1
# 信号变化时弹 macOS 通知（失败不影响主流程）
.venv/bin/python paper/notify_signal_change.py >> logs/paper_cron.log 2>&1 || true
