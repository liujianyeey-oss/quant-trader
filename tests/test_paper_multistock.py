"""观察账户 8 号（paper_multistock）回归测试：
低波 top30 排序正确性、幂等、非调仓日不交易、涨跌停顺延、缺跑补记。

运行：.venv/bin/python -m unittest discover -s tests -v
"""
import os
import tempfile
import unittest

import numpy as np
import pandas as pd

from paper_multistock import (update_account, load_state, read_csv, _csv_path,
                              compute_target, is_rebalance_day)


def make_universe(n=40, days=400, start="2023-01-02"):
    """合成 n 只：编号越小波动越低（低波在前）。价格缓涨，含量。"""
    dates = pd.bdate_range(start, periods=days)
    data = {}
    for i in range(n):
        code = f"{i:06d}"
        vol = 0.005 + 0.001 * i  # 波动递增（确定性交替收益，日振幅<10% 不触涨跌停）
        rets = np.tile([vol, -vol], days // 2 + 1)[:days] + 0.0002
        prices = 10 * np.exp(np.cumsum(rets))
        data[code] = pd.DataFrame({
            "date": dates, "open": prices, "high": prices, "low": prices,
            "close": prices, "volume": 1e6})
    return data, dates


def cut_at_second_rebalance_day(data, dates):
    """截取数据到某月第二个交易日（该月首个交易日=调仓日在历史内）。"""
    cal = pd.DatetimeIndex(dates)
    firsts = list(pd.Series(cal).groupby([cal.year, cal.month]).nth(0))
    d1 = pd.Timestamp(firsts[13])         # 第 14 个月首个交易日（历史 ≥250 根 K 线）
    d2 = cal[cal > d1][0]                 # 次日（执行日）
    cut = {c: df[df["date"] <= d2] for c, df in data.items()}
    return cut, d1, d2, cal


class MultiStockTests(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.mkdtemp()

    def test_lowvol_ranking(self):
        """波动最低的前 30 只被选中，等权。"""
        data, dates = make_universe()
        tgt, n_elig = compute_target(data, dates[-1])
        self.assertEqual(n_elig, 40)
        self.assertEqual(len(tgt), 30)
        expect = {f"{i:06d}" for i in range(30)}
        self.assertEqual(set(tgt.keys()), expect)
        self.assertTrue(all(abs(w - 1 / 30) < 1e-9 for w in tgt.values()))

    def test_first_rebalance_buys_30(self):
        """调仓日开户记信号，次日开盘建仓 30 只；非调仓日不交易。"""
        data, dates = make_universe()
        cut, d1, d2, cal = cut_at_second_rebalance_day(data, dates)
        # 调仓日运行：只记信号，不成交
        cut_d1 = {c: df[df["date"] <= d1] for c, df in data.items()}
        update_account(self.dir, cut_d1)
        self.assertEqual(len(load_state(self.dir)["shares"]), 0)
        # 次日运行：开盘执行建仓
        update_account(self.dir, cut)
        state = load_state(self.dir)
        self.assertEqual(len(state["shares"]), 30)
        trades = read_csv(_csv_path(self.dir, "trades"))
        trades["code"] = trades["code"].astype(str).str.zfill(6)  # CSV 读回丢失前导零
        self.assertEqual((trades["side"] == "buy").sum(), 30)
        # 非调仓日推进 3 天：无新交易
        cut2 = {c: df[df["date"] <= cal[cal.get_loc(d2) + 3]] for c, df in data.items()}
        update_account(self.dir, cut2)
        trades2 = read_csv(_csv_path(self.dir, "trades"))
        self.assertEqual(len(trades2), len(trades))

    def test_idempotent(self):
        data, dates = make_universe()
        cut, d1, d2, cal = cut_at_second_rebalance_day(data, dates)
        update_account(self.dir, {c: df[df["date"] <= d1] for c, df in data.items()})
        update_account(self.dir, cut)
        before = tuple(len(read_csv(_csv_path(self.dir, k)))
                       for k in ("nav", "signals", "trades"))
        update_account(self.dir, cut)
        after = tuple(len(read_csv(_csv_path(self.dir, k)))
                      for k in ("nav", "signals", "trades"))
        self.assertEqual(before, after)

    def test_limit_up_defer(self):
        """调仓执行日涨停 → 该股买入顺延，次日成交。"""
        data, dates = make_universe()
        cut, d1, d2, cal = cut_at_second_rebalance_day(data, dates)
        d3 = cal[cal.get_loc(d2) + 1]
        # 先正常跑到 d2 完成建仓
        update_account(self.dir, {c: df[df["date"] <= d1] for c, df in data.items()})
        update_account(self.dir, cut)
        # 找到下个月调仓日，把未持仓的 000039 波动压到最低（必进新目标），执行日涨停
        firsts = list(pd.Series(cal).groupby([cal.year, cal.month]).nth(0))
        nd1 = pd.Timestamp(firsts[14])          # 下一调仓日
        nd2 = cal[cal > nd1][0]                 # 下一执行日
        nd3 = cal[cal > nd2][0]
        data["000039"].loc[:, "close"] = 10.0
        data["000039"].loc[:, "open"] = 10.0
        data["000039"].loc[:, "high"] = 10.0
        data["000039"].loc[:, "low"] = 10.0
        d2x = data["000039"]
        prev_close = d2x.loc[d2x["date"] < nd2, "close"].iloc[-1]
        d2x.loc[d2x["date"] == nd2, "open"] = prev_close * 1.105
        update_account(self.dir, {c: df[df["date"] <= nd1] for c, df in data.items()})
        update_account(self.dir, {c: df[df["date"] <= nd2] for c, df in data.items()})
        trades = read_csv(_csv_path(self.dir, "trades"))
        trades["code"] = trades["code"].astype(str).str.zfill(6)  # CSV 读回丢失前导零
        day_buys = trades[(trades["side"] == "buy") & (trades["date"] == nd2)]
        self.assertFalse((day_buys["code"] == "000039").any())  # 涨停日未成交
        update_account(self.dir, {c: df[df["date"] <= nd3] for c, df in data.items()})
        trades = read_csv(_csv_path(self.dir, "trades"))
        trades["code"] = trades["code"].astype(str).str.zfill(6)
        day_buys = trades[(trades["side"] == "buy") & (trades["date"] == nd3)]
        self.assertTrue((day_buys["code"] == "000039").any())   # 次日顺延成交

    def test_catchup_two_days(self):
        """缺跑两天后一次补齐净值行。"""
        data, dates = make_universe()
        cut, d1, d2, cal = cut_at_second_rebalance_day(data, dates)
        update_account(self.dir, {c: df[df["date"] <= d1] for c, df in data.items()})
        update_account(self.dir, cut)
        n1 = len(read_csv(_csv_path(self.dir, "nav")))
        # 缺跑 2 天，直接到 d2+3
        cut2 = {c: df[df["date"] <= cal[cal.get_loc(d2) + 3]] for c, df in data.items()}
        update_account(self.dir, cut2)
        nav = read_csv(_csv_path(self.dir, "nav"))
        self.assertEqual(len(nav), n1 + 3)  # 3 日补齐（d2+1, d2+2, d2+3）


if __name__ == "__main__":
    unittest.main(verbosity=2)
