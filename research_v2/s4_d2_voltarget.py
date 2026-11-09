"""S4：D2 波动目标仓位回测（策略 v2 设计文档 D2 节，S4 阶段）。

实现口径（严格按 D2 已确认规则）：
- 实现波动：持仓篮子的 20 日 EWMA 年化波动（λ=0.94，有效窗 ≈17 日；
  篮子波动=权重加权各标的 EWMA 波动之和，与 staggered/D1 的混合波动口径一致，偏保守）。
- scalar = min(1, 目标波动/实现波动)，无杠杆；每日计算；|变化| <10% 不动（防抖）。
- 回撤联动：在**信号级代理净值**（T+1 收盘口径、无成本）上判定——距峰回撤 >10%
  scalar×0.5；>15% ×0.25 且仅允许防守池（进攻权重强制并入 518880）；
  修复至 <5% 后每 5 个交易日至多升一档（恢复慢于降档）。
  （代理净值无前视：day T 状态用 ≤T-1 收盘信息；实际成交仍由引擎 T+1 开盘价完成。）
- staggered 的 0.12 组内波动预算由本层取代（消除双重标准）：组内满仓选股、
  组合层统一 scalar。

自检：目标波动=999% 且关闭回撤联动时，D2 包装必须与现策略逐项一致。
"""
import copy, sys, json
import numpy as np
import pandas as pd
import yaml
sys.path.insert(0, "/Users/liujianye10/quant-trader")

from backtest import Backtester
from strategies import MomentumRotation, StaggeredMomentum
from metrics import calc_metrics

BASE = "/Users/liujianye10/quant-trader"
ATTACK = ["510300", "510500", "159915", "588000"]
GOLD = "518880"
CODES = ATTACK + [GOLD]
FETCH_START = pd.Timestamp("2019-01-01") - pd.Timedelta(days=450)
START = "2019-01-01"
LAM = 0.94

data = {c: pd.read_parquet(f"{BASE}/data_cache/{c}_qfq.parquet") for c in CODES}
for c in CODES:
    data[c] = data[c][data[c]["date"] >= FETCH_START].reset_index(drop=True)
base_cfg = yaml.safe_load(open(f"{BASE}/config.yaml"))
assert base_cfg["cost"]["min_commission"] == 5

panel = pd.DataFrame({c: df.set_index("date")["close"] for c, df in data.items()}).sort_index()
rets = panel.pct_change().fillna(0.0)


def ewma_vol(r: pd.Series, lam=LAM, init=20):
    """λ=0.94 EWMA 年化波动；前 init 日用简单波动起步。"""
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


EWMA = {c: ewma_vol(rets[c]) for c in CODES}


class StaggeredNoVolSize(StaggeredMomentum):
    """组内不做波动控仓（满仓选股），组合层仓位交给 D2。"""

    def _cohort_signals(self, panel_, weekday):
        mom = panel_ / panel_.shift(self.momentum_days) - 1.0
        ma = panel_.rolling(self.trend_ma, min_periods=self.trend_ma).mean()
        codes = [c for c in self.attack_pool if c in panel_.columns]
        rows = []
        for i in range(self.warmup, len(panel_)):
            date = panel_.index[i]
            if date.weekday() != weekday:
                continue
            m = mom.iloc[i][codes].dropna()
            if m.empty:
                continue
            winner = m.idxmax()
            row = pd.Series(0.0, index=panel_.columns)
            pick = winner if panel_.iloc[i][winner] >= ma.iloc[i][winner] else self.defensive
            row[pick] = 1.0
            rows.append(row.rename(date))
        return pd.DataFrame(rows)


def d2_wrap(base_signals, target_vol, use_governor=True, hyst=0.10):
    """把稀疏基准目标包装成 D2 每日缩放目标（信号日口径，T+1 由引擎负责）。"""
    cal = panel.index
    aligned = (base_signals.reindex(cal.union(base_signals.index)).sort_index().ffill()
               .reindex(cal).fillna(0.0))
    # 发射起点：回测开始前 15 个交易日（让建仓时点与对照组一致）
    start_i = cal.get_loc(cal[cal >= pd.Timestamp(START)][0])
    emit_from = max(0, start_i - 15)

    nav, peak = 1.0, 1.0
    level, last_up = 0, -10 ** 9
    scalar_applied = None
    rows = {}
    prev_w = aligned.iloc[emit_from - 1] if emit_from > 0 else aligned.iloc[0] * 0
    for i in range(emit_from, len(cal)):
        date = cal[i]
        w = aligned.iloc[i]
        # (b) 篮子 EWMA 波动（权重加权）
        sig = sum(float(w.get(c, 0.0)) * EWMA[c].iloc[i] for c in w.index if w.get(c, 0) > 0)
        raw = min(1.0, target_vol / sig) if sig and np.isfinite(sig) and sig > 1e-8 else 1.0
        # (c) 回撤联动（用截至前一日的代理净值状态）
        dd = nav / peak - 1.0
        if use_governor:
            if dd <= -0.15:
                level = 2
            elif dd <= -0.10:
                level = max(level, 1)
            if level > 0 and dd > -0.05 and i - last_up >= 5:
                level -= 1
                last_up = i
        mult = (1.0, 0.5, 0.25)[level]
        # (d) scalar + 防抖
        s = raw * mult
        if scalar_applied is not None and abs(s / scalar_applied - 1.0) < hyst:
            s = scalar_applied
        scalar_applied = s
        # (e) 目标缩放；level2 仅防守池（进攻权重并入黄金）
        sw = {}
        atk = sum(float(w.get(c, 0.0)) for c in ATTACK)
        for c in w.index:
            v = float(w.get(c, 0.0))
            if v <= 0:
                continue
            if level == 2 and c in ATTACK:
                continue
            sw[c] = sw.get(c, 0.0) + v
        if level == 2 and atk > 0:
            sw[GOLD] = sw.get(GOLD, 0.0) + atk
        sw = {c: v * s for c, v in sw.items()}
        rows[date] = sw
        # (f) 代理净值：昨日缩放目标 × 今日收益（T+1 近似，无成本）
        nav *= 1.0 + sum(v * float(rets[c].iloc[i]) for c, v in prev_w.items())
        peak = max(peak, nav)
        prev_w = sw
    return pd.DataFrame.from_dict(rows, orient="index").fillna(0.0)


def run(signals, roe, phase=0):
    sig = signals
    if phase:
        pos = {d: i for i, d in enumerate(panel.index)}
        idx = [panel.index[min(pos[d] + phase, len(panel) - 1)] for d in sig.index]
        sig = sig.copy()
        sig.index = pd.DatetimeIndex(idx)
        sig = sig[~sig.index.duplicated(keep="last")]
    eq, trades = Backtester(base_cfg, rebalance_on_event=roe).run(data, sig, start=START)
    m = calc_metrics(eq["total_value"], trades, initial=1_000_000)
    e = eq["total_value"]
    years = (eq.index[-1] - eq.index[0]).days / 365.25
    turnover = sum(t["shares"] * t["price"] for t in trades if t["side"] == "sell") / e.mean() / years
    # 仓位时间分布 & 防守期占比（T+1 生效口径）
    al = (sig.reindex(panel.index.union(sig.index)).sort_index().ffill()
          .reindex(panel.index).fillna(0.0)).shift(1).fillna(0.0)
    al = al.loc[al.index >= pd.Timestamp(START)]
    tot = al.sum(axis=1)
    defw = al[GOLD]
    atkw = al[[c for c in ATTACK]].sum(axis=1)
    # 窗口指标
    trend = e.loc["2019-01-01":"2021-12-31"]
    trend_ann = (trend.iloc[-1] / trend.iloc[0]) ** (365.25 / (trend.index[-1] - trend.index[0]).days) - 1
    def win_dd(a, b):
        w = e.loc[a:b]
        return float((w / w.cummax() - 1).min())
    return {"m": m, "换手": turnover, "卖出": m["交易次数"],
            "仓位均值": float(tot.mean()), "仓位p10": float(tot.quantile(0.1)),
            "仓位p50": float(tot.quantile(0.5)), "仓位p90": float(tot.quantile(0.9)),
            "满仓天数占比": float((tot >= 0.95).mean()),
            "防守期占比": float((defw > atkw).mean()),
            "趋势段年化1921": float(trend_ann),
            "回撤2022窗口": win_dd("2022-01-01", "2022-12-31"),
            "回撤2024窗口": win_dd("2024-01-01", "2024-12-31")}


base_mom = MomentumRotation(base_cfg).generate_signals(data)
base_stg_full = StaggeredNoVolSize(base_cfg).generate_signals(data)   # 供 D2 包装
base_stg_cur = StaggeredMomentum(base_cfg).generate_signals(data)    # 现役对照

# 自检：关掉缩放与联动时必须等于现役策略
chk = d2_wrap(base_mom, target_vol=9.99, use_governor=False, hyst=0.0)
m_chk = run(chk, False)["m"]
m_ctl = run(base_mom, False)["m"]
print(f"自检 momentum：D2(关闭缩放) 年化{m_chk['年化收益率']:+.2%} vs 现役{m_ctl['年化收益率']:+.2%}"
      f" —— {'✓一致' if abs(m_chk['年化收益率']-m_ctl['年化收益率']) < 1e-4 else '✗不一致'}")

out = {}
print(f"\n{'变体':<26}{'年化':>8}{'夏普':>6}{'回撤':>9}{'卡玛':>6}{'卖出':>6}{'换手/年':>8}"
      f"{'仓位均':>7}{'p10':>6}{'p90':>6}{'满仓%':>6}{'防守期':>6}"
      f"{'1921年化':>9}{'22窗DD':>8}{'24窗DD':>8}", flush=True)
for label, sig, roe in [
    ("A momentum 现役对照", base_mom, False),
    ("A staggered 现役对照", base_stg_cur, True),
    ("momentum-D2 tv8%", d2_wrap(base_mom, 0.08), False),
    ("momentum-D2 tv10%", d2_wrap(base_mom, 0.10), False),
    ("momentum-D2 tv12%", d2_wrap(base_mom, 0.12), False),
    ("staggered-D2 tv8%", d2_wrap(base_stg_full, 0.08), False),
    ("staggered-D2 tv10%", d2_wrap(base_stg_full, 0.10), False),
    ("staggered-D2 tv12%", d2_wrap(base_stg_full, 0.12), False),
]:
    r = run(sig, roe)
    m = r["m"]
    out[label] = {k: (v if not isinstance(v, dict) else
                      {"年化": v["年化收益率"], "夏普": v["夏普比率"], "回撤": v["最大回撤"],
                       "卡玛": v["卡玛比率"]} if k == "m" else v)
                  for k, v in r.items()}
    print(f"{label:<26}{m['年化收益率']:>+8.2%}{m['夏普比率']:>6.2f}{m['最大回撤']:>9.2%}"
          f"{m['卡玛比率']:>6.2f}{r['卖出']:>6}{r['换手']:>8.2f}{r['仓位均值']:>7.0%}"
          f"{r['仓位p10']:>6.0%}{r['仓位p90']:>6.0%}{r['满仓天数占比']:>6.0%}"
          f"{r['防守期占比']:>6.0%}{r['趋势段年化1921']:>+9.2%}"
          f"{r['回撤2022窗口']:>8.2%}{r['回撤2024窗口']:>8.2%}", flush=True)

print("\n== 相位中位（5 相位）==", flush=True)
for label, sig, roe in [
    ("momentum-D2 tv10%", d2_wrap(base_mom, 0.10), False),
    ("staggered-D2 tv10%", d2_wrap(base_stg_full, 0.10), False),
    ("A momentum 现役对照", base_mom, False),
    ("A staggered 现役对照", base_stg_cur, True),
]:
    ph = [run(sig, roe, phase=k)["m"]["年化收益率"] for k in range(5)]
    out[label]["相位中位"] = float(np.median(ph))
    out[label]["相位年化"] = [round(float(p), 4) for p in ph]
    print(f"  {label:<26} 中位 {np.median(ph):+.2%}  各相位 {['%+.1f%%' % (p*100) for p in ph]}",
          flush=True)

json.dump(out, open(f"{BASE}/research_v2/s4_results.json", "w"),
          ensure_ascii=False, indent=1, default=str)
print("saved research_v2/s4_results.json")
