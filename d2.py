"""D2 波动目标仓位层（策略 v2，docs/DESIGN_strategy_v2_20261108.md）。

治理裁决口径（2026-11-09 用户拍板）：验收门槛改写为风险调整口径
（夏普不降 + 回撤显著压缩即过），D2 tv10% 过门进入影子运行（至 2026-12-08 评审）；
staggered 不叠加 D2（S4 已证无增量，组内 0.12 预算与本层同构，叠加=双重压缩）。

规则（与 S4 回测实现逐行一致，research_v2/s4_d2_voltarget.py 为证据链）：
- 实现波动：持仓篮子的 EWMA 年化波动（λ=0.94，前 20 日简单波动起步；
  篮子波动 = 权重加权各标的 EWMA 波动之和，与 staggered/D1 混合波动口径一致）；
- scalar = min(1, 目标波动/实现波动)，无杠杆；每日计算；|变化| <10% 不动（防抖）；
- 回撤联动（尾部保险）：信号级代理净值（T+1 收盘近似、无成本、无前视）距峰
  回撤 >10% → scalar×0.5；>15% → ×0.25 且仅允许防守池（进攻权重并入防守资产）；
  修复至 <5% 后每 5 个交易日至多升一档。S4 实测 tv≤12% 时该层几乎不触发。
- 代理净值起点 = 回测起点前 15 个交易日（保证建仓时点与未包装策略一致；
  S4 数字即此口径，影子运行与证据链严格同源）。
"""
import numpy as np
import pandas as pd

from strategies import MomentumRotation

DEFAULT_TV = 0.10
LAM = 0.94
EWMA_INIT = 20
HYST = 0.10            # scalar 变化 <10% 不动
GOV_DD1, GOV_DD2, GOV_RECOVER = -0.10, -0.15, -0.05
GOV_MULT = (1.0, 0.5, 0.25)
GOV_UP_DAYS = 5        # 恢复每 5 个交易日至多一档
EMIT_PAD_DAYS = 15     # 代理净值起点 = 回测起点前 15 个交易日


def ewma_vol(r: pd.Series, lam=LAM, init=EWMA_INIT) -> pd.Series:
    """λ=0.94 EWMA 年化波动（有效窗 ≈17 日，对应规则名义"20 日"）。"""
    arr = r.to_numpy(dtype=float)
    out = np.full(len(arr), np.nan)
    if len(arr) <= init:
        return pd.Series(out, index=r.index)
    var = float(np.nanvar(arr[1:init + 1]))
    out[init] = np.sqrt(var * 252)
    for t in range(init + 1, len(arr)):
        var = lam * var + (1 - lam) * arr[t] ** 2
        out[t] = np.sqrt(var * 252)
    return pd.Series(out, index=r.index)


def wrap_signals(base_signals: pd.DataFrame, panel: pd.DataFrame, target_vol=DEFAULT_TV,
                 defensive="518880", attack_codes=None, use_governor=True,
                 hyst=HYST, emit_start="2019-01-01") -> pd.DataFrame:
    """把基准目标权重包装成 D2 每日缩放目标（信号日口径；T+1 由引擎负责）。

    base_signals: 稀疏基准目标（如 MomentumRotation.generate_signals 输出）。
    panel: 收盘价面板（含全部相关标的，可含预热期）。
    emit_start: 代理净值/信号发射起点 = 该日前 15 个交易日（与 S4 口径一致）。
    无前视：第 T 日的 scalar/联动档位只用 ≤T-1 收盘的代理净值与 ≤T 的已实现波动。
    """
    attack_codes = list(attack_codes or [])
    cal = panel.index
    aligned = (base_signals.reindex(cal.union(base_signals.index)).sort_index().ffill()
               .reindex(cal).fillna(0.0))
    rets = panel.pct_change().fillna(0.0)
    ew = {c: ewma_vol(rets[c]) for c in panel.columns}

    start_i = cal.get_loc(cal[cal >= pd.Timestamp(emit_start)][0])
    emit_from = max(0, start_i - EMIT_PAD_DAYS)

    nav, peak = 1.0, 1.0
    level, last_up = 0, -10 ** 9
    scalar_applied = None
    rows = {}
    prev_w = {}
    for i in range(emit_from, len(cal)):
        date = cal[i]
        w = aligned.iloc[i]
        # 篮子 EWMA 波动（权重加权和）
        sig = sum(float(w.get(c, 0.0)) * float(ew[c].iloc[i])
                  for c in w.index if w.get(c, 0) > 0 and pd.notna(ew[c].iloc[i]))
        raw = min(1.0, target_vol / sig) if sig > 1e-8 else 1.0
        # 回撤联动（状态来自截至前一日的代理净值）
        dd = nav / peak - 1.0
        if use_governor:
            if dd <= GOV_DD2:
                level = 2
            elif dd <= GOV_DD1:
                level = max(level, 1)
            if level > 0 and dd > GOV_RECOVER and i - last_up >= GOV_UP_DAYS:
                level -= 1
                last_up = i
        s = raw * GOV_MULT[level]
        if scalar_applied is not None and abs(s / scalar_applied - 1.0) < hyst:
            s = scalar_applied
        scalar_applied = s
        # 缩放；level2 仅防守池（进攻权重并入防守资产）
        sw = {}
        atk = sum(float(w.get(c, 0.0)) for c in attack_codes)
        for c in w.index:
            v = float(w.get(c, 0.0))
            if v <= 0 or (level == 2 and c in attack_codes):
                continue
            sw[c] = sw.get(c, 0.0) + v
        if level == 2 and atk > 0:
            sw[defensive] = sw.get(defensive, 0.0) + atk
        sw = {c: v * s for c, v in sw.items()}
        rows[date] = sw
        # 代理净值：昨日缩放目标 × 今日收益（T+1 收盘近似，无成本）
        nav *= 1.0 + sum(v * float(rets[c].iloc[i]) for c, v in prev_w.items())
        peak = max(peak, nav)
        prev_w = sw
    if not rows:
        return pd.DataFrame(columns=panel.columns)
    return pd.DataFrame.from_dict(rows, orient="index").fillna(0.0)


class MomentumD2(MomentumRotation):
    """momentum + D2 波动目标仓位层（治理裁决档位 tv=10%）。"""

    name = "momentum_d2"

    def __init__(self, cfg, target_vol=None, use_governor=True):
        super().__init__(cfg)
        d2 = cfg.get("d2", {})
        self.target_vol = (target_vol if target_vol is not None
                           else d2.get("target_vol", DEFAULT_TV))
        self.use_governor = use_governor
        self.emit_start = cfg.get("backtest", {}).get("start", "2019-01-01")

    def generate_signals(self, data: dict) -> pd.DataFrame:
        base = super().generate_signals(data)
        if base.empty:
            return base
        panel = self._close_panel(data)
        return wrap_signals(base, panel, target_vol=self.target_vol,
                            defensive=self.defensive, attack_codes=self.attack_pool,
                            use_governor=self.use_governor, emit_start=self.emit_start)
