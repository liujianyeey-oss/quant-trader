#!/bin/bash
# BTC×F1 台账每日更新（cron 用，绝对路径）。币安日线 UTC 收盘 = 北京 08:00。
# 建议 crontab（每天 08:17，365 日历无周末）：
#   17 8 * * * /Users/liujianye10/quant-trader/paper_btc/daily_update_btc.sh
cd /Users/liujianye10/quant-trader || exit 1
mkdir -p logs
.venv/bin/python main.py paper-btc-update >> logs/paper_btc_cron.log 2>&1
.venv/bin/python paper_btc/notify_btc.py >> logs/paper_btc_cron.log 2>&1
