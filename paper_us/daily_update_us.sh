#!/bin/bash
# 美股 donchian 台账每日更新（cron 用，绝对路径；cron 环境没有用户 PATH）。
# 建议 crontab（周二至周六 08:47 北京时间，对应美股周一至周五收盘后；
# 夏令时收盘 04:00/冬令时 05:00，08:47 均安全）：
#   47 8 * * 2-6 /Users/liujianye10/quant-trader/paper_us/daily_update_us.sh
cd /Users/liujianye10/quant-trader || exit 1
mkdir -p logs
.venv/bin/python main.py paper-us-update >> logs/paper_us_cron.log 2>&1
.venv/bin/python paper_us/notify_us.py >> logs/paper_us_cron.log 2>&1
