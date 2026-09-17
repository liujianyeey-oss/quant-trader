"""R5 假设检验 runner：卖空占比信号（H-A 顺势 / H-B 反向 / H-C 溢出）。

当前状态：历史日频卖空数据免费源不可得（见 collector.py 头注），
market_short.csv 从采集日起逐步积累。本 runner 在样本期 < 约 1 年时
直接给出"证据不足"判定，不硬选；积累足够后按预声明口径自动检验。

预声明口径（数据齐备时执行，不得事后改规则）：
- 占比 = 当日大市卖空成交额 / 大市成交额代理（159920+513130 合计成交额，
  收盘×量近似；HKEX 文件不含分母，已声明代理口径）
- 基线：均值 20 日 +1σ（H-A）/ +2σ（H-B）；±0.5σ 敏感性各一档
- H-A：占比超阈值 → 次日对 159920/513130 空仓（基准：买入持有、收盘>MA60）
- H-B：占比 > 均值+2σ 视为恐慌底 → 反向买入持有 5 个交易日
- H-C：H-A 同规则作用于 510300（溢出检验）
- 时序：T 日卖空数据当晚公布，信号最早 T+1 生效、T+2 开盘成交（等同 T-1 规则）
- 判效：训练段卡玛改善 ≥0.1；训练 ≤2022-12-31（以实际覆盖为准）、
  验证 2023-2024、2025+ 只报告
"""
import os
import sys

import pandas as pd

BASE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(BASE, "results")
os.makedirs(RESULTS, exist_ok=True)

MIN_OBS = 240  # 约 1 年交易日，低于此直接判证据不足


def main():
    path = os.path.join(BASE, "cache", "market_short.csv")
    status = []
    if not os.path.exists(path):
        status.append({"item": "market_short.csv", "status": "不存在（采集器尚未成功落盘）"})
        n = 0
    else:
        m = pd.read_csv(path, parse_dates=["date"])
        n = len(m)
        status.append({"item": "market_short.csv",
                       "status": f"{n} 个交易日（{m['date'].min().date()} ~ {m['date'].max().date()}）"})
    verdict = ("证据不足（样本期不足约 1 年，预声明的统计功效底线未达）"
               if n < MIN_OBS else "样本足够，执行假设检验")
    status.append({"item": "verdict", "status": verdict})
    out = pd.DataFrame(status)
    out.to_csv(os.path.join(RESULTS, "status.csv"), index=False, encoding="utf-8-sig")
    print(out.to_string(index=False))
    if n < MIN_OBS:
        print("\n判定：H-A / H-B / H-C 均为『不可测/证据不足』——历史源不可得，"
              "前向采集刚起步。按纪律不硬选，详见 docs/R5_HK_SHORT_INTEREST.md。")
        return
    # 样本足够后的检验逻辑（预留：占比计算 → 信号 → macro_research 引擎复用）
    raise NotImplementedError("样本积累满 1 年后启用；规则见本文件头注，不得修改")


if __name__ == "__main__":
    main()
