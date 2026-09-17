# R5：港股卖空数据作为信号——数据可得性调查与采集器交付

2026-09-17。**结论先行：多年历史日频港股卖空数据在本机所有免费源均不可得（逐条证据见下），H-A/H-B/H-C 三个预声明假设无法做历史检验，判定全部为「不可测/证据不足」——按纪律不硬选。已交付可用的 HKEX 当日卖空采集器（前向积累），建议挂 cron 每晚落盘，约 1 年后按预声明口径自动检验。**

## 一、数据可得性调查（每条源与实测结果）

| 源 | 结果 |
|---|---|
| HKEX 旧 daily 文件 `…/eng/stat/smstat/dayquot/d{YYMMDD}e.htm`（2019~2025 多个历史日期） | 全部 404（官网改版，旧路径已死） |
| HKEX 新"当日"端点（Short Selling Turnover Today，主板/GEM/NASDAQ，早收盘/全日收盘两档） | **可用**——纯文本 `<pre>` 报告，但只服务"当日"，发布前显示占位页 "will be available after day close"（本机 HKT 17:26 实测为占位页，发布通常在收盘后晚间） |
| HKEX 统计存档页（Securities Statistics Archive） | 无日频卖空历史入口 |
| web.archive.org（dayquot 快照回填） | 本机连接超时，不可达 |
| akshare | 无任何港股卖空接口（全库扫描确认） |
| 东财/新浪/百度 | 无日频港股卖空历史免费接口 |
| GitHub 社区镜像 | 仅 2 个 2017/2021 年停止更新的采集脚本仓库，无数据仓库存活 |
| AAStocks 沽空比例页 | JS 空壳，无服务端渲染内容，历史深度亦有限 |

**未伪造任何数据**：没有拿月度数据冒充日频，没有用代理序列编造卖空占比。

## 二、已交付：HKEX 每日卖空采集器（research_hkshort/collector.py）

- 抓 HKEX 当日主板卖空成交文本报告（`TOTAL` 行：大市卖空股数与成交额 + 个股明细）；
- 落盘：原始文本 `cache/raw/<date>.txt`、个股明细 `short_detail_<date>.csv`、大市汇总 `market_short.csv`（含 sha256 与来源 URL，provenance 可审计）；
- 占位页自动识别并退出码 2（供 cron 稍后重试）；解析器按多年稳定的 dayquot/ashtmain 规范格式编写，`--demo` 内置规范格式样例自检已通过；
- 分母口径（预声明）：HKEX 文件只含卖空方，不含大市成交额——占比分母用 **159920+513130 合计成交额（收盘×量近似）作"港股活跃大盘代理"**，报告发布时已声明，未来检验时不得更换口径。

cron 建议行（用户自行配置；HKEX 晚间发布，20:10 与 21:23 两次重试）：

```cron
10 20 * * 1-5 cd /Users/liujianye10/quant-trader && .venv/bin/python -m research_hkshort.collector >> logs/hkshort.log 2>&1
23 21 * * 1-5 cd /Users/liujianye10/quant-trader && .venv/bin/python -m research_hkshort.collector >> logs/hkshort.log 2>&1
```

也可以今晚手动跑一次验证真实文件：`python -m research_hkshort.collector`（发布后再跑）。

## 三、假设判定（预声明，失败全报告）

| 假设 | 判定 | 原因 |
|---|---|---|
| H-A 顺势（卖空占比 > 均值+1σ → 次日空仓 159920/513130） | **不可测/证据不足** | 历史日频占比序列不存在，无法回测 |
| H-B 反向（占比 > 均值+2σ 恐慌底 → 买入持有 5 日） | **不可测/证据不足** | 极端事件样本本就稀少，更依赖长历史，现状下零检验可能 |
| H-C 溢出（H-A 同规则作用于 510300） | **不可测/证据不足** | 同上 |

阈值敏感性（±0.5σ 档）随主判定一并搁置。runner（`research_hkshort/runner.py`）已实现"样本 < 约 240 个交易日直接判证据不足"的硬闸门，积累满约 1 年后按预声明口径自动检验（规则写死在文件头注，防事后改规则）。

## 四、样本量与功效警告（即使将来数据齐了也要记住）

- 港股卖空占比长期 10%~20% 波动，>均值+2σ 的"恐慌"事件每年通常仅个位数——H-B 的统计功效天然极低，3 年训练段可能只有 10~20 个事件点，"有效"也很可能是噪声；
- 卖空占比是**水平值聚类**的序列（高波动期持续偏高），按均值+σ 触发的信号高度自相关，有效样本数远小于日历天数；
- 任何"卖空数据预测恒指"的回测都必须过相位/滞后稳健性：T 日晚发布 → T+1 生效 → T+2 成交，比常规信号多一天时滞，动量类 alpha 会再衰减一截；
- 分母是代理口径（159920+513130 成交额），占比绝对水平与"全市场真实占比"有系统性偏差，只有**相对时序变化**可用——均值/σ 基线必须在同一口径内计算，不能与文献中的全市场占比阈值直接比较。

## 五、建议

1. 今晚起挂 cron 前向采集（成本为零）；3 个月后可以先做一次描述性统计（占比分布、与 159920 走势的同期相关），那不算假设检验、可写进月报；
2. 满 1 年后 runner 自动按预声明规则判定 H-A/H-B/H-C，期间不许改阈值；
3. 若未来找到合规的多年历史源（如付费 HKEX Data Portal 或学术数据集），先用它做训练段判定，前向积累段作验证段，两段口径一致再谈进生产。

## 复现

```bash
cd /Users/liujianye10/quant-trader
.venv/bin/python -m research_hkshort.collector --demo   # 解析器格式自检
.venv/bin/python -m research_hkshort.collector          # 抓当日（收盘后晚间）
.venv/bin/python -m research_hkshort.runner             # 样本闸门 + 判定状态
```

代码：`research_hkshort/{collector,runner}.py`；结果：`research_hkshort/results/status.csv`。未改动任何生产文件。
