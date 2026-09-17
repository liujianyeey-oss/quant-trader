#!/bin/bash
# 个人手动操作台账每日跟踪（独立于策略账户）
cd /Users/liujianye10/quant-trader || exit 1
.venv/bin/python -m my_trades.track update >> logs/my_trades_cron.log 2>&1
