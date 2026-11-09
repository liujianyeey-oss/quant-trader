#!/bin/bash
# 观察账户 8 号（低波 top30 月调）台账每日更新（cron 用，绝对路径）。
# 错开 16:47 主台账与 16:50 观察账户：
#   55 16 * * 1-5 /Users/liujianye10/quant-trader/paper_multistock/daily_update_ms.sh
# 注意：每月首个交易日为全量刷新（717 只，约 10~25 分钟），其余日子只刷持仓。
cd /Users/liujianye10/quant-trader || exit 1
mkdir -p logs
.venv/bin/python main.py paper-ms-update >> logs/paper_ms_cron.log 2>&1
.venv/bin/python paper_multistock/notify_ms.py >> logs/paper_ms_cron.log 2>&1
