#!/bin/bash
# 观察账户 6/7 台账每日更新（cron 用，绝对路径）。两个时段都安全：
# 哪个账户有新数据就处理哪个（美股周二至周六早晨；A股工作日下午）。
# 建议 crontab（与 paper_us / paper 主台账错开）：
#   47 8 * * 2-6 /Users/liujianye10/quant-trader/paper_obs/daily_update_obs.sh
#   50 16 * * 1-5 /Users/liujianye10/quant-trader/paper_obs/daily_update_obs.sh
cd /Users/liujianye10/quant-trader || exit 1
mkdir -p logs
.venv/bin/python main.py paper-obs-update >> logs/paper_obs_cron.log 2>&1
.venv/bin/python paper_obs/notify_obs.py >> logs/paper_obs_cron.log 2>&1
