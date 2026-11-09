"""D1 防守池轮动（d1.py / paper_shadow.py）单元测试。

覆盖：波动过滤、动量排名、迟滞防抖（排名差/持有天数双条件）、卫星开关、
兜底、周频上限、forced-static 与父类等价性、影子台账与实际台账的隔离性。

运行：.venv/bin/python -m unittest discover -s tests -v
"""
import os
import tempfile
import unittest

import numpy as np
import pandas as pd

from d1 import (DefensivePoolSelector, MomentumD1, StaggeredD1,
                CORE_POOL, GOLD, FALLBACK)
from paper_shadow import init_shadow, update
from paper import update_account, load_state, _csv_path, _read_csv
from strategies import MomentumRotation, StaggeredMomentum

POOL = CORE_POOL  # ["511880","511360","511010","511260"]


def make_panel(ret_map, n=260, start="2024-01-01", breaks=None):
    """合成收盘价面板。ret_map: {code: 日收益} 常量序列；breaks: {code: [(i0, r), ...]}
    从 i0 起改用新日收益。常量日收益 ⇒ 实现波动≈0（必然通过 <3% 过滤）。"""
    dates = pd.bdate_range(start, periods=n)
    cols = {}
    for code, r in ret_map.items():
        rets = np.full(n, r, dtype=float)
        for i0, r2 in (breaks or {}).get(code, []):
            rets[i0:] = r2
        cols[code] = 100.0 * np.cumprod(1 + rets)
    return pd.DataFrame(cols, index=dates)


def make_cfg():
    return {
        "cost": {"etf_commission": 0.0001, "stock_commission": 0.00025,
                 "stamp_tax": 0.0005, "slippage": 0.001, "min_commission": 5},
        "backtest": {"initial_cash": 1_000_000},
        "universe": {"etfs": ["A", "B", "C", "D"], "stocks": [], "defensive": "GOLD"},
        "strategy": {"momentum_days": 20, "rebalance_days": 5,
                     "ma_fast": 5, "ma_slow": 20, "trend_ma": 20},
        "staggered": {"momentum_days": 20, "trend_ma": 60, "vol_window": 60,
                      "vol_budget": 0.12, "n_groups": 5},
    }


def make_data(defense_ok=False):
    """合成行情：attack 池 A~D + GOLD + D1 池四只。
    defense_ok=False：A 持续上涨（进攻态）；True：全部下跌（防守态）。"""
    n = 150
    dates = pd.bdate_range("2020-01-01", periods=n)
    data = {}
    for code in ["A", "B", "C", "D", "GOLD"] + POOL:
        if code == "A" and not defense_ok:
            close = 10 * 1.01 ** np.arange(n)
        elif defense_ok and code in ["A", "B", "C", "D"]:
            close = 10 * 0.995 ** np.arange(n)   # 持续下跌，趋势过滤失败
        elif code == "511010":
            close = 100 * 1.0008 ** np.arange(n)  # 核心池动量最高
        else:
            close = np.full(n, 100.0 if code == "GOLD" else 10.0)
        data[code] = pd.DataFrame({"date": dates, "open": close, "close": close,
                                   "high": close, "low": close, "volume": 1})
    return data


class SelectorTests(unittest.TestCase):
    """DefensivePoolSelector 规则行为。"""

    def test_vol_filter_excludes_high_vol(self):
        """60 日年化波动 ≥3% 的候选被过滤，即使动量最高。"""
        n = 260
        dates = pd.bdate_range("2024-01-01", periods=n)
        alt = np.where(np.arange(n) % 2 == 0, 0.03, -0.01)  # 高波动+正漂移
        cols = {"511260": 100 * np.cumprod(1 + alt),        # 动量最高但波动超标
                "511010": 100 * 1.0008 ** np.arange(n),
                "511880": 100 * 1.0001 ** np.arange(n),
                "511360": 100 * 1.0002 ** np.arange(n),
                GOLD: 100 * np.ones(n) / 1.0}
        sel = DefensivePoolSelector(pd.DataFrame(cols, index=dates))
        w = sel.step(150)
        self.assertNotIn("511260", w)
        self.assertEqual(w, {"511010": 1.0})

    def test_momentum_ranking_picks_top_eligible(self):
        panel = make_panel({"511010": 0.001, "511260": 0.0005,
                            "511880": 0.0001, "511360": 0.0002, GOLD: 0.0})
        sel = DefensivePoolSelector(panel)
        self.assertEqual(sel.step(100), {"511010": 1.0})  # 卫星关闭（黄金动量 0）
        self.assertEqual(sel.current, "511010")

    def test_hysteresis_gap_blocks_small_lead(self):
        """排名差 <1pp 不切换；≥1pp 且持有 ≥5 日才切换。"""
        # mom_win=5 加速动量变化；511010 先强，511260 后期缓强
        panel = make_panel({"511010": 0.003, "511260": 0.001,
                            "511880": 0.0001, "511360": 0.0001, GOLD: 0.0},
                           breaks={"511010": [(100, 0.0)],      # 100 起 511010 熄火
                                   "511260": [(100, 0.0007)]},  # 511260 缓升
                           )
        sel = DefensivePoolSelector(panel, mom_win=5)
        for i in range(65, 100):   # vol_win=60 预热后起步：i=65 建立 511010
            sel.step(i)
        self.assertEqual(sel.current, "511010")
        # 105 起：mom(511010)≈0，mom(511260)≈0.35%，差 <1pp → 不换
        for i in range(100, 110):
            sel.step(i)
        self.assertEqual(sel.current, "511010")
        # 511260 加速到 0.3%/日：5 日动量 ≈1.5%，差 >1pp 且持有远超 5 日 → 切换
        sel2 = DefensivePoolSelector(
            make_panel({"511010": 0.003, "511260": 0.001,
                        "511880": 0.0001, "511360": 0.0001, GOLD: 0.0},
                       breaks={"511010": [(100, 0.0)], "511260": [(100, 0.003)]}),
            mom_win=5)
        for i in range(65, 110):
            sel2.step(i)
        self.assertEqual(sel2.current, "511260")

    def test_min_hold_blocks_immediate_reswitch(self):
        """周频上限（持有 <5 交易日）即使排名差 ≥1pp 也不切换。"""
        # mom_win=3 + vol_win=10 + 关闭波动过滤，隔离验证 min_hold；B 在第 23 日暴涨
        panel = make_panel({"511010": 0.004, "511260": 0.001,
                            "511880": 0.0, "511360": 0.0001, GOLD: 0.0},
                           breaks={"511260": [(23, 0.02)]})
        sel = DefensivePoolSelector(panel, mom_win=3, vol_win=10, min_hold=5, vol_cap=1.0)
        for i in range(20, 23):
            sel.step(i)  # i=20 建立 511010（动量第一），last_switch=20
        self.assertEqual(sel.current, "511010")
        sel.step(23)  # 511260 3日动量 3.2% > 511010 的 1.2%，但持有仅 3 日
        sel.step(24)
        self.assertEqual(sel.current, "511010")
        sel.step(25)  # 持有满 5 个交易日 → 允许切换
        self.assertEqual(sel.current, "511260")

    def test_satellite_on_off(self):
        """卫星：黄金 收盘>MA60 且 20日动量>3% → {黄金0.2, 核心0.8}；破坏即关闭。"""
        panel = make_panel({"511010": 0.0008, "511260": 0.0005, "511880": 0.0001,
                            "511360": 0.0002, GOLD: 0.005},   # 黄金强势上行
                           breaks={GOLD: [(200, -0.005)]})     # 200 起转跌
        sel = DefensivePoolSelector(panel)
        w = sel.step(150)
        self.assertAlmostEqual(w.get(GOLD, 0), 0.2)
        self.assertAlmostEqual(w.get("511010", 0), 0.8)
        w2 = sel.step(230)   # 黄金已连跌 30 日：动量 <3% 且跌破 MA60
        self.assertNotIn(GOLD, w2)
        self.assertAlmostEqual(w2.get("511010", 0), 1.0)

    def test_fallback_when_none_eligible(self):
        """池内全部失去资格（波动超标）→ 兜底 511880。"""
        n = 200
        dates = pd.bdate_range("2024-01-01", periods=n)
        alt = np.where(np.arange(n) % 2 == 0, 0.04, -0.04)  # 全部高波动
        cols = {c: 100 * np.cumprod(1 + alt) for c in POOL}
        cols[GOLD] = 100 * np.ones(n)
        sel = DefensivePoolSelector(pd.DataFrame(cols, index=dates))
        self.assertEqual(sel.step(150), {FALLBACK: 1.0})

    def test_gold_never_in_core(self):
        """黄金再强也不进核心池（核心 80% 只能是债/货）。"""
        panel = make_panel({"511010": 0.0008, "511260": 0.0005, "511880": 0.0001,
                            "511360": 0.0002, GOLD: 0.02})  # 黄金暴涨
        sel = DefensivePoolSelector(panel)
        w = sel.step(150)
        self.assertAlmostEqual(w[GOLD], 0.2)          # 只能出现在卫星仓
        self.assertAlmostEqual(w["511010"], 0.8)
        self.assertEqual(sel.current, "511010")


class EquivalenceTests(unittest.TestCase):
    """forced-static：D1 变体退化为死守单资产时必须与父类逐行一致。"""

    def test_momentum_d1_static_equals_base(self):
        cfg, data = make_cfg(), make_data()
        s1 = MomentumD1(cfg, defensive_override="GOLD").generate_signals(data)
        s0 = MomentumRotation(cfg).generate_signals(data)
        pd.testing.assert_frame_equal(s1, s0)

    def test_staggered_d1_static_equals_base(self):
        cfg, data = make_cfg(), make_data()
        s1 = StaggeredD1(cfg, defensive_override="GOLD").generate_signals(data)
        s0 = StaggeredMomentum(cfg).generate_signals(data)
        pd.testing.assert_frame_equal(s1, s0)


class ShadowIsolationTests(unittest.TestCase):
    """影子台账与实际台账完全隔离。"""

    def setUp(self):
        self.real_dir = tempfile.mkdtemp()
        self.shadow_dir = tempfile.mkdtemp()
        self.cfg = make_cfg()
        self.data = make_data(defense_ok=True)  # 防守态：实际账户持 GOLD

    def _snapshot(self, d):
        out = {}
        for f in sorted(os.listdir(d)):
            p = os.path.join(d, f)
            if os.path.isfile(p):
                with open(p, "rb") as fh:
                    out[f] = fh.read()
        return out

    def test_shadow_init_and_isolation(self):
        # 实际两账户运行两天（开户 + 次日开盘成交 GOLD）
        for name in ("momentum", "staggered"):
            update_account(self.cfg, name, self.data, self.real_dir)
        new_date = self.data["GOLD"]["date"].max() + pd.offsets.BDay(1)
        for code, df in list(self.data.items()):
            p = 100.0 if code == "GOLD" else float(df["close"].iloc[-1])
            row = pd.DataFrame({"date": [new_date], "open": [p], "close": [p],
                                "high": [p], "low": [p], "volume": [1]})
            self.data[code] = pd.concat([df, row], ignore_index=True)
        for name in ("momentum", "staggered"):
            update_account(self.cfg, name, self.data, self.real_dir)
        before = self._snapshot(self.real_dir)

        # 影子初始化 + 更新（数据无新交易日 → 跳过，但状态已 seed）
        actions = update(self.cfg, self.data, self.real_dir, self.shadow_dir)
        self.assertTrue(any("影子初始化" in a for a in actions))
        st = load_state(self.shadow_dir, "shadow_d1_momentum", 0)
        actual = load_state(self.real_dir, "momentum", 0)
        # 起点净值 = 实际台账最新净值；持仓镜像实际（GOLD）
        nav = _read_csv(_csv_path(self.shadow_dir, "nav", "shadow_d1_momentum"))
        real_nav = _read_csv(_csv_path(self.real_dir, "nav", "momentum"))
        self.assertEqual(nav["nav"].iloc[-1], real_nav["nav"].iloc[-1])
        self.assertEqual(nav["date"].iloc[-1], real_nav["date"].iloc[-1])
        self.assertEqual(st["shares"], actual["shares"])
        # 待成交目标是 D1 组合（511010 核心满仓，卫星关闭因 GOLD 动量=0）
        self.assertEqual(st["pending"]["target"], {"511010": 1.0})
        # 幂等：再跑不重复初始化
        actions2 = update(self.cfg, self.data, self.real_dir, self.shadow_dir)
        self.assertFalse(any("影子初始化" in a for a in actions2))
        # 隔离：实际台账文件字节不变
        self.assertEqual(before, self._snapshot(self.real_dir))

    def test_shadow_executes_d1_target_next_open(self):
        """影子首日持仓镜像实际，次一交易日按 D1 目标开盘成交切换。"""
        for name in ("momentum", "staggered"):
            update_account(self.cfg, name, self.data, self.real_dir)
        # 实际账户次日开盘成交 GOLD（持仓落地后再镜像）
        new_date = self.data["GOLD"]["date"].max() + pd.offsets.BDay(1)
        for code, df in list(self.data.items()):
            p = 100.0 if code == "GOLD" else float(df["close"].iloc[-1])
            row = pd.DataFrame({"date": [new_date], "open": [p], "close": [p],
                                "high": [p], "low": [p], "volume": [1]})
            self.data[code] = pd.concat([df, row], ignore_index=True)
        for name in ("momentum", "staggered"):
            update_account(self.cfg, name, self.data, self.real_dir)
        update(self.cfg, self.data, self.real_dir, self.shadow_dir)
        # 追加一个交易日：影子应在此日开盘卖出 GOLD、买入 511010
        new_date = self.data["GOLD"]["date"].max() + pd.offsets.BDay(1)
        for code, df in list(self.data.items()):
            p = 100.0 if code in ("GOLD",) else float(df["close"].iloc[-1])
            if code == "511010":
                p = float(df["close"].iloc[-1])
            row = pd.DataFrame({"date": [new_date], "open": [p], "close": [p],
                                "high": [p], "low": [p], "volume": [1]})
            self.data[code] = pd.concat([df, row], ignore_index=True)
        update(self.cfg, self.data, self.real_dir, self.shadow_dir)
        trades = _read_csv(_csv_path(self.shadow_dir, "trades", "shadow_d1_momentum"))
        self.assertFalse(trades.empty)
        sells = trades[trades["side"] == "sell"]
        buys = trades[trades["side"] == "buy"]
        self.assertEqual(list(sells["code"]), ["GOLD"])
        self.assertEqual(list(buys["code"]), ["511010"])
        self.assertEqual(trades["date"].max(), new_date)
        st = load_state(self.shadow_dir, "shadow_d1_momentum", 0)
        self.assertEqual(set(st["shares"]), {"511010"})


if __name__ == "__main__":
    unittest.main(verbosity=2)
