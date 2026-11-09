"""D2 波动目标仓位（d2.py）与 D2 影子账户（paper_shadow.py）单元测试。

覆盖：scalar 缩放正确性、10% 防抖、回撤联动档位（×0.5 / ×0.25+仅防守池 / 慢速恢复）、
关闭联动与缩放时与父类等价、影子账户隔离性。

运行：.venv/bin/python -m unittest discover -s tests -v
"""
import os
import tempfile
import unittest

import numpy as np
import pandas as pd

from d2 import MomentumD2, wrap_signals
from paper import update_account, load_state, _csv_path, _read_csv
from paper_shadow import update as shadow_update
from strategies import MomentumRotation
try:  # discover -s tests 与 tests.test_d2 两种运行方式兼容
    from test_d1 import make_cfg, make_data  # 复用 D1 测试的合成数据/配置
except ImportError:
    from tests.test_d1 import make_cfg, make_data


def make_price_panel(ret_fn, n=400, start="2024-01-01"):
    """单标的 A + 平稳 GOLD 的面板。ret_fn: i -> 日收益。"""
    dates = pd.bdate_range(start, periods=n)
    r = np.array([ret_fn(i) for i in range(n)], dtype=float)
    cols = {"A": 10.0 * np.cumprod(1 + r), "GOLD": np.full(n, 100.0)}
    return pd.DataFrame(cols, index=dates), r


def const_signals(panel, code="A", weight=1.0, start_row=30):
    """常数基准信号：自 start_row 起持有 code。"""
    return pd.DataFrame({code: [weight]}, index=[panel.index[start_row]])


class WrapTests(unittest.TestCase):
    def test_scalar_matches_known_vol(self):
        """恒定 |日收益|=1% → EWMA 年化波动 ≈15.9%；tv10% → scalar≈0.63。"""
        panel, _ = make_price_panel(lambda i: 0.01 if i % 2 == 0 else -0.01)
        sig = wrap_signals(const_signals(panel), panel, target_vol=0.10,
                           defensive="GOLD", attack_codes=["A"],
                           use_governor=False, hyst=0.0,
                           emit_start=str(panel.index[100].date()))
        w_last = sig["A"].iloc[-1]
        self.assertAlmostEqual(w_last, 0.10 / (0.01 * np.sqrt(252)), places=2)

    def test_hysteresis_band(self):
        """scalar 变化 <10% 不动；>10% 才调整（允许滞后，最终值在 raw ±10% 带内）。"""
        # 前 200 日 ±1.0%（scalar≈0.63），之后 ±1.04%（raw≈0.61，差 -4% <10% 不动），
        # 第 300 日起 ±1.30%（raw 收敛至 ≈0.486，相对 0.63 差 -23% >10% → 调整）
        panel, _ = make_price_panel(
            lambda i: (0.01 if i % 2 == 0 else -0.01) * (1.0 if i < 200 else (1.04 if i < 300 else 1.30)))
        sig = wrap_signals(const_signals(panel), panel, target_vol=0.10,
                           defensive="GOLD", attack_codes=["A"],
                           use_governor=False, hyst=0.10,
                           emit_start=str(panel.index[50].date()))
        self.assertAlmostEqual(sig["A"].iloc[260], sig["A"].iloc[250], places=6)  # 微漂移不动
        self.assertLess(sig["A"].iloc[320], sig["A"].iloc[250] * 0.9)              # 大漂移已调整
        # EWMA 收敛需时日 + 防抖滞后：最终值落在收敛 raw 的 ±10% 带内即正确
        raw_final = 0.10 / (0.013 * np.sqrt(252))
        self.assertAlmostEqual(sig["A"].iloc[-1], raw_final, delta=raw_final * 0.11)

    def test_governor_levels_and_defensive_only(self):
        """联动：回撤 >10% → ×0.5；>15% → ×0.25 且进攻权重并入防守资产。"""
        # 120 日平稳上行（建仓+峰值），随后 -4%/日 连跌（穿透两档阈值），最后回稳
        def path(i):
            if i < 120:
                return 0.0005
            if i < 130:
                return -0.04
            return 0.0005
        panel, _ = make_price_panel(path)
        sig = wrap_signals(const_signals(panel), panel, target_vol=9.99,  # tv 极大→raw=1
                           defensive="GOLD", attack_codes=["A"],
                           use_governor=True, hyst=0.0,
                           emit_start=str(panel.index[30].date()))
        # 净值路径：跌 3 日 dd≈-11.5%（第 123 日起 level1 ×0.5），
        # 跌第 4 日 dd≈-15.0%（第 124 日起 level2 → 仅防守池 ×0.25）
        self.assertAlmostEqual(sig.loc[panel.index[115], "A"], 1.0, places=6)   # 联动未触发
        self.assertAlmostEqual(sig.loc[panel.index[123], "A"], 0.5, places=6)   # 一档
        deep = sig.loc[panel.index[126]]
        self.assertNotIn("A", deep[deep > 0].index)                             # 二档仅防守池
        self.assertAlmostEqual(deep.get("GOLD", 0.0), 0.25, places=6)

    def test_governor_recovery_is_slow(self):
        """恢复慢于降档：降档当日即生效；升级须回撤先修复至 -5% 以内。"""
        def path(i):
            if i < 120:
                return 0.0005
            if i < 124:
                return -0.03      # 跌 4 日 ≈ -11.5%（触发一档，不进二档）
            return 0.03           # 反弹（level1 仓位 0.5 → 代理净值 +1.5%/日）
        panel, _ = make_price_panel(path)
        sig = wrap_signals(const_signals(panel), panel, target_vol=9.99,
                           defensive="GOLD", attack_codes=["A"],
                           use_governor=True, hyst=0.0,
                           emit_start=str(panel.index[30].date()))
        w = sig["A"]
        self.assertAlmostEqual(w.loc[panel.index[124]], 0.5, places=6)  # 降档当日生效
        # dd：-8.9%→-7.5%→-6.2%→-4.8%（第 128 日升级回 level0）
        for k in (125, 126, 127):
            self.assertAlmostEqual(w.loc[panel.index[k]], 0.5, places=6)  # 修复途中不提前升档
        self.assertAlmostEqual(w.loc[panel.index[128]], 1.0, places=6)


class EquivalenceTests(unittest.TestCase):
    def test_d2_disabled_equals_base(self):
        """关闭缩放与联动时，MomentumD2 的生效目标与 MomentumRotation 完全一致。"""
        cfg, data = make_cfg(), make_data()  # A 持续上涨 → 基准目标恒为 {A:1.0}
        base = MomentumRotation(cfg).generate_signals(data)
        wrapped = MomentumD2(cfg, target_vol=9.99, use_governor=False).generate_signals(data)
        cal = pd.DataFrame({c: df.set_index("date")["close"]
                            for c, df in data.items()}).sort_index().index
        al = (base.reindex(cal.union(base.index)).sort_index().ffill()
              .reindex(cal).fillna(0.0))
        wr = wrapped.reindex(cal).ffill().fillna(0.0)
        wr = wr.loc[wr.index >= wrapped.index[0]]
        wr = wr.reindex(columns=al.columns, fill_value=0.0)  # 未出现的标的列补 0
        pd.testing.assert_frame_equal(wr, al.loc[wr.index],
                                      check_dtype=False, atol=1e-9)


class ShadowD2Tests(unittest.TestCase):
    def setUp(self):
        self.real_dir = tempfile.mkdtemp()
        self.shadow_dir = tempfile.mkdtemp()
        self.cfg = make_cfg()
        self.data = make_data(defense_ok=True)

    def _snapshot(self, d):
        out = {}
        for f in sorted(os.listdir(d)):
            p = os.path.join(d, f)
            if os.path.isfile(p):
                with open(p, "rb") as fh:
                    out[f] = fh.read()
        return out

    def test_shadow_d2_init_and_isolation(self):
        for name in ("momentum", "staggered"):
            update_account(self.cfg, name, self.data, self.real_dir)
        before = self._snapshot(self.real_dir)
        actions = shadow_update(self.cfg, self.data, self.real_dir, self.shadow_dir)
        self.assertTrue(any("shadow_d2_momentum: 影子初始化" in a for a in actions))
        st = load_state(self.shadow_dir, "shadow_d2_momentum", 0)
        actual = load_state(self.real_dir, "momentum", 0)
        self.assertEqual(st["shares"], actual["shares"])          # 镜像持仓
        nav = _read_csv(_csv_path(self.shadow_dir, "nav", "shadow_d2_momentum"))
        real_nav = _read_csv(_csv_path(self.real_dir, "nav", "momentum"))
        self.assertEqual(nav["nav"].iloc[-1], real_nav["nav"].iloc[-1])  # 起点=实际净值
        self.assertIn("D2 波动目标 tv10%", st["pending"]["reason"])
        self.assertAlmostEqual(sum(st["pending"]["target"].values()), 1.0, places=6)
        # 幂等 + 隔离
        actions2 = shadow_update(self.cfg, self.data, self.real_dir, self.shadow_dir)
        self.assertFalse(any("影子初始化" in a for a in actions2))
        self.assertEqual(before, self._snapshot(self.real_dir))


if __name__ == "__main__":
    unittest.main(verbosity=2)
