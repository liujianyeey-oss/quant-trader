"""S6：D3 进攻池板块轮动回测（策略 v2 设计文档 D3 节定稿名单）。

D3 规则（按设计文档）：
- 打分 = 20 日与 60 日动量等权合成；
- 持有排名前 2 名、各 50%；单标的须过趋势过滤（MA60 或 40/50/60 多数投票）；
- 每周调仓（5 交易日网格）；挑战者合成动量须超出现任 ≥2pp 才换（迟滞）；
- 相关性护栏：候选与已选标的近 60 日相关 >0.9 时跳过低分者；
- 空仓槽位资金进 D1 防守池轮动（d1.DefensivePoolSelector，S1/S2 同口径：
  选择器只在需要防守仓位的信号日推进）。

对照：A=现有 4 宽基池（现策略）；B=D3 机制 × 4 宽基池（隔离机制效应）；
C=D3 × 18 只扩展池（MA60）；D=D3 × 18 池（ensemble 投票，S5 前提要求）。
"""
import copy, sys, json
import numpy as np
import pandas as pd
import yaml
sys.path.insert(0, "/Users/liujianye10/quant-trader")

from backtest import Backtester
from strategies import MomentumRotation, StaggeredMomentum
from metrics import calc_metrics
from d1 import DefensivePoolSelector, CORE_POOL, GOLD

BASE = "/Users/liujianye10/quant-trader"
BROAD = ["510300", "510500", "159915", "588000"]
INDUSTRY = ["512880", "512800", "512170", "512690", "159995", "515880", "512980",
            "512660", "512400", "515220", "159611", "515790", "159870", "512200"]
D3_POOL = BROAD + INDUSTRY
DEF_CODES = CORE_POOL + [GOLD]
ALL = D3_POOL + DEF_CODES
FETCH_START = pd.Timestamp("2019-01-01") - pd.Timedelta(days=450)
START = "2019-01-01"
WARM = 61           # mom60 预热
SLOT = 0.5          # 前 2 名各 50%
HYST = 0.02         # 迟滞 2pp
CORR_CAP = 0.9

data = {c: pd.read_parquet(f"{BASE}/data_cache/{c}_qfq.parquet") for c in ALL}
for c in ALL:
    data[c] = data[c][data[c]["date"] >= FETCH_START].reset_index(drop=True)
    assert not data[c].empty, f"{c} 无数据"
first_dates = {c: f"{data[c]['date'].min():%Y-%m-%d}" for c in INDUSTRY}
base_cfg = yaml.safe_load(open(f"{BASE}/config.yaml"))

panel = pd.DataFrame({c: df.set_index("date")["close"] for c, df in data.items()}).sort_index()
mom20 = panel / panel.shift(20) - 1.0
mom60 = panel / panel.shift(60) - 1.0
score = (mom20 + mom60) / 2
mas = {w: panel.rolling(w, min_periods=w).mean() for w in (40, 50, 60)}
vol60 = panel.pct_change().rolling(60, min_periods=60).std() * np.sqrt(252)
rets = panel.pct_change()


def trend_ok(i, c, mode):
    if mode == "ma60":
        ma = mas[60].iloc[i][c]
        return pd.notna(ma) and panel.iloc[i][c] >= ma
    return sum(1 for w in (40, 50, 60)
               if pd.notna(mas[w].iloc[i][c]) and panel.iloc[i][c] >= mas[w].iloc[i][c]) >= 2


def corr60(i, a, b):
    """近 60 日日收益相关（含当日）。"""
    if i < 60:
        return np.nan
    w = rets[[a, b]].iloc[i - 59:i + 1].dropna()
    return w[a].corr(w[b]) if len(w) >= 40 else np.nan


def d3_pick(i, pool, holdings, mode):
    """D3 选仓：趋势过滤→打分排名→相关性护栏→迟滞。返回目标 {code: weight}（攻击侧）。"""
    s = score.iloc[i]
    elig = [c for c in pool
            if pd.notna(s.get(c)) and c in panel.columns and trend_ok(i, c, mode)]
    elig.sort(key=lambda c: s[c], reverse=True)
    picked = []
    for c in elig:  # 相关性护栏
        if any(abs(corr60(i, c, p)) > CORR_CAP for p in picked):
            continue
        picked.append(c)
        if len(picked) == 2:
            break
    # 迟滞：现任合格且与挑战者（目标槽最弱者）分差 <2pp → 保留
    for h in sorted(holdings, key=lambda h: s.get(h, -9), reverse=True):
        if h in picked or h not in pool:
            continue
        if not (pd.notna(s.get(h)) and trend_ok(i, h, mode)):
            continue
        if picked and (len(picked) == 2):
            weakest = min(picked, key=lambda c: s[c])
            if s[weakest] - s[h] < HYST:
                picked[picked.index(weakest)] = h
        elif not picked:
            picked.append(h)
    return {c: SLOT for c in picked[:2]}


def _rows_to_frame(rows):
    sig = pd.DataFrame(0.0, index=[d for d, _ in rows], columns=list(panel.columns))
    for d, w in rows:
        for c, wt in w.items():
            sig.loc[d, c] = wt
    return sig


def d3_signals(pool, mode="ma60", staggered=False):
    """生成 D3 目标权重信号。staggered=True 时按 weekday 分 5 组、组内波动控仓。"""
    days = range(WARM, len(panel))
    if not staggered:
        grid = [i for i in days if (i - WARM) % 5 == 0]
        return _rows_to_frame(_d3_rows(grid, pool, mode, DefensivePoolSelector(panel),
                                       scale=False))
    cohorts = [_rows_to_frame(_d3_rows([i for i in days if panel.index[i].weekday() == wd],
                                       pool, mode, DefensivePoolSelector(panel), scale=True))
               for wd in range(5)]
    cohorts = [c for c in cohorts if not c.empty]
    dates = cohorts[0].index
    for c in cohorts[1:]:
        dates = dates.union(c.index)
    return sum(c.reindex(dates).ffill().fillna(0.0) for c in cohorts) / 5


def _d3_rows(grid, pool, mode, selector, scale):
    """沿网格推进：持仓状态（迟滞）与 D1 选择器（仅防守日推进）。"""
    holdings = []
    rows = []
    for i in grid:
        atk = d3_pick(i, pool, holdings, mode)
        holdings = list(atk)
        w = dict(atk)
        if len(atk) < 2:  # 空槽 → D1 防守池
            defw = selector.step(i)
            slot_left = 1.0 - sum(atk.values())
            for c, wt in defw.items():
                w[c] = w.get(c, 0.0) + slot_left * wt
        if scale:  # staggered 组内波动控仓：混合波动折算
            sig_blend = sum(wt * vol60.iloc[i].get(c, np.nan) for c, wt in w.items())
            if pd.isna(sig_blend):
                sig_blend = np.nan
            if pd.notna(sig_blend):
                w_total = min(1.0, base_cfg["staggered"]["vol_budget"] / max(0.03, sig_blend))
                w = {c: wt * w_total for c, wt in w.items()}
            else:
                w = {}
        rows.append((panel.index[i], w))
    return rows


def run(signals, roe, phase=0):
    sig = signals
    if phase:
        cal = panel.index
        pos = {d: i for i, d in enumerate(cal)}
        idx = [cal[min(pos[d] + phase, len(cal) - 1)] for d in sig.index]
        sig = sig.copy()
        sig.index = pd.DatetimeIndex(idx)
        sig = sig[~sig.index.duplicated(keep="last")]
    eq, trades = Backtester(base_cfg, rebalance_on_event=roe).run(data, sig, start=START)
    m = calc_metrics(eq["total_value"], trades, initial=1_000_000)
    # 年化换手（卖出额/均值净值/年）与防守期占比（防守权重>进攻权重的交易日占比）
    sell_amt = sum(t["shares"] * t["price"] for t in trades if t["side"] == "sell")
    years = (eq.index[-1] - eq.index[0]).days / 365.25
    turnover = sell_amt / eq["total_value"].mean() / years
    aligned = (sig.reindex(panel.index.union(sig.index)).sort_index().ffill()
               .reindex(panel.index).fillna(0.0)).shift(1).fillna(0.0)
    aligned = aligned.loc[aligned.index >= pd.Timestamp(START)]
    defw = aligned[[c for c in DEF_CODES if c in aligned.columns]].sum(axis=1)
    atkw = aligned[[c for c in D3_POOL if c in aligned.columns]].sum(axis=1)
    def_share = float((defw > atkw).mean())
    return m, turnover, def_share


def current_signals(name):
    strat = MomentumRotation(base_cfg) if name == "momentum" else StaggeredMomentum(base_cfg)
    return strat.generate_signals({c: data[c] for c in BROAD + [GOLD]}), strat.rebalance_on_event


variants = {}
out = {}
print("== S6 对照（2019-01-01 ~ 2026-11-06，含成本）==", flush=True)
for label, gen in [
    ("A momentum 现4宽基", lambda: current_signals("momentum")),
    ("A staggered 现4宽基", lambda: current_signals("staggered")),
    ("B D3机制×4宽基 MA60", lambda: (d3_signals(BROAD, "ma60"), False)),
    ("C D3×18池 MA60", lambda: (d3_signals(D3_POOL, "ma60"), False)),
    ("D D3×18池 ensemble", lambda: (d3_signals(D3_POOL, "ens"), False)),
    ("B-s D3机制×4宽基 staggered化", lambda: (d3_signals(BROAD, "ma60", staggered=True), True)),
    ("C-s D3×18池 staggered MA60", lambda: (d3_signals(D3_POOL, "ma60", staggered=True), True)),
    ("D-s D3×18池 staggered ens", lambda: (d3_signals(D3_POOL, "ens", staggered=True), True)),
]:
    sig, roe = gen()
    variants[label] = (sig, roe)
    m, tov, dfs = run(sig, roe)
    out[label] = {"年化": m["年化收益率"], "夏普": m["夏普比率"], "回撤": m["最大回撤"],
                  "卡玛": m["卡玛比率"], "卖出": m["交易次数"], "年化换手": tov,
                  "防守期占比": dfs}
    print(f"{label:<24} 年化{m['年化收益率']:>+8.2%} 夏普{m['夏普比率']:>5.2f} "
          f"回撤{m['最大回撤']:>8.2%} 卡玛{m['卡玛比率']:>5.2f} 卖出{m['交易次数']:>5}"
          f" 换手{tov:>5.2f}/年 防守期{dfs:>5.0%}", flush=True)

print("\n== 相位扰动（中位口径）==", flush=True)
for label in ["C D3×18池 MA60", "D D3×18池 ensemble", "C-s D3×18池 staggered MA60",
              "A momentum 现4宽基", "A staggered 现4宽基"]:
    sig, roe = variants[label]
    ph = [run(sig, roe, phase=k)[0]["年化收益率"] for k in range(5)]
    out[label]["相位中位"] = float(np.median(ph))
    out[label]["相位极差"] = float(max(ph) - min(ph))
    print(f"  {label:<24} 中位 {np.median(ph):+.2%} 极差 {(max(ph)-min(ph))*100:.1f}pp", flush=True)

out["_meta"] = {"industry_first_data": first_dates,
                "note": "行业ETF上市日不一，扩展池在 2019 初仅 6 只可排名，2022 后 14 只全在线"}
json.dump(out, open(f"{BASE}/research_v2/s6_results.json", "w"),
          ensure_ascii=False, indent=1, default=str)
print("saved research_v2/s6_results.json")
