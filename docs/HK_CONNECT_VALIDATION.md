# 港股通模块交付与验证（2026-09-09）

已在 `/Users/liujianye10/quant-trader` 新增独立 `hk_connect/` 模块，并接入 `main.py` 的信号和回测命令。当前已从官方数据交叉获取 660 只沪港通南向可买 HKD 股票及各自每手股数，默认仅研究配置中的 16 只观察池股票；不覆盖仅深港通可买的股票。

## 使用

```bash
cd /Users/liujianye10/quant-trader
source .venv/bin/activate
python main.py signal --strategy hk_connect
python main.py backtest --strategy hk_connect --research --start 2025-07-01 --offline
```

完整说明在项目 `docs/HK_CONNECT.md`。新增配置位于 `config.yaml` 的 `hk_connect` 段；资金单位是 **HKD**，默认 100,000，不会将人民币账户自动换汇。现有 A 股参数、模拟盘及定时任务未改变。

## 规则

60/120 日正动量、120 日趋势过滤、20 日成交额近似值门槛 5,000 万 HKD；周三或之后本周首个交易日更新，最多 5 只，单只目标不超过 20%，总目标不超过 80%。按 60 日波动控仓，并在选股时检查整手成本；周度建仓至少 3 只可负担股票，不足保留现金。趋势或资格退出每日处理，不为凑数周中开新仓。

复权价用于信号，原始开盘价用于成交；按股票实际手数取整，计双边税费和最低佣金。停牌、开盘缺价、通道关闭时延期；现金不透支。当前资格回看历史必须显式选择 `--research`，不会被标为无偏历史验证。

## 真实数据联调与研究结果

截至 2026-09-08 行情、2026-09-09 资格快照，默认 10 万 HKD 账户当前模型目标为现金：没有同时满足筛选、风险预算与最少可负担持仓数的组合。

以下区间为 2025-07-02 至 2026-09-08，全部为 **HKD 累计价格收益**，非年化、非人民币总回报。默认周三，单边滑点 10 bps；调仓星期覆盖周一至周五，压力情景为周三单边 30 bps。

| 研究资金 HKD | 默认累计收益 | 最大回撤 | 平均股票仓位 | 不同星期累计收益区间 | 30 bps 滑点累计收益 |
|---:|---:|---:|---:|---:|---:|
| 100,000 | -0.99% | -2.24% | 2.0% | -4.16% ～ -0.99% | -1.51% |
| 500,000 | 0.18% | -6.72% | 35.2% | 0.18% ～ 6.04% | -2.54% |
| 1,000,000 | 3.37% | -6.83% | 36.2% | 3.37% ～ 9.23% | 0.18% |

10 万 HKD 的低回撤主要来自长期低仓位，不应解读为良好风险收益能力。50 万和 100 万 HKD 的结果对调仓星期和滑点敏感，目前不足以证明稳定优势。保持预设周三参数，没有根据这轮结果挑选最高收益星期。

结果采用当前 SH 名单和手数回填历史，并有观察池选择偏差。未提供真实通道日历，未计现金分红、红利税、人民币汇率损益、换汇价差和持有费用；历史统一使用当前费用情景，尚无完整公司行为/结算撮合模型。高分红股票尤其不能用价格收益替代完整收益。该窗口已经用于开发验证，不能再称为未接触的样本外数据。

## 工程验证

- 完整回归：`python -m unittest discover -s tests`，**42 项全部通过**，其中 19 项为港股专项。
- 真实在线联调：官方名单与每手股数更新成功，16 只观察池日线拉取成功，信号文件生成成功。
- 回测联调：默认 10 万 HKD 探索回测成功，16 笔成交，估算交易费用 381.27 HKD；输出净值、交易、目标和配置/数据范围快照。
- 完成 21 组资金/星期/滑点情景；原始结果附在本目录 `scenarios.csv` 和 `assumptions.json`。
- `git diff --check` 通过；没有下单、提交代码、变更定时任务或写入现有模拟盘。

## 账户条件与来源

研究分配资金与账户开通条件不同。上交所列示个人申请港股通前 20 个交易日日均资产门槛为 50 万人民币，不含融资融券融入资产；已有权限后分配 10 万作为子组合是另一回事。[上交所开户条件](https://one.sse.com.cn/onething/gptz/)

资格及每手股数来源：[上交所南向名单](https://www.sse.com.cn/services/hkexsc/disclo/eligible/)、[HKEX 证券表](https://www.hkex.com.hk/eng/services/trading/securities/securitieslists/ListOfSecurities.xlsx)。费用情景参考：[HKEX 交易费用](https://www.hkex.com.hk/Services/Rules-and-Forms-and-Fees/Fees/Securities-%28Hong-Kong%29/Trading/Transaction?sc_lang=en)、[HKEX 结算费用](https://www.hkex.com.hk/Services/Rules-and-Forms-and-Fees/Fees/Securities-%28Hong-Kong%29/Clearing-and-Settlement/Operational?sc_lang=en)。实际南向券商转嫁费用需按账单校准。
