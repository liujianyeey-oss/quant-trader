"""D1/D2 影子台账（策略 v2 影子运行：2026-11-09 ~ 2026-12-08 评审）。

- shadow_d1_*：D1 防守池轮动（d1.py），momentum/staggered 两策略；
- shadow_d2_momentum：D2 波动目标 tv10%（d2.py，仅 momentum——staggered 叠加
  D2 无增量，S4 已证）。

与实际台账完全隔离：独立目录 paper_shadow/、独立账户名 shadow_*，
只读共享 data_cache 与 paper/（初始化时读取一次实际账户状态）。
信号规则与实际账户相同，唯一差异 = 防守资产选择（D1）/仓位缩放（D2）。

初始化口径：起点净值 = 初始化日实际台账净值；持仓/成本镜像实际账户；
变体目标作为待成交信号（signal_date=初始化日），下一交易日按 T+1 开盘价
成交（费率/滑点/最低佣金与台账同一套 _rebalance）。幂等：状态已存在则不再初始化。
"""
import json
import os

import pandas as pd

from d1 import MomentumD1, StaggeredD1
from d2 import MomentumD2
from paper import (load_state, save_state, update_account, _csv_path, _read_csv,
                   _append_csv, NAV_COLUMNS, SIGNAL_COLUMNS)

TRACKED = {"shadow_d1_momentum": MomentumD1, "shadow_d1_staggered": StaggeredD1,
           "shadow_d2_momentum": MomentumD2}
ACTUAL_NAME = {"shadow_d1_momentum": "momentum", "shadow_d1_staggered": "staggered",
               "shadow_d2_momentum": "momentum"}
INIT_REASON = {"shadow_d1_momentum": "影子账户初始化（D1 防守池轮动，起点=实际台账净值）",
               "shadow_d1_staggered": "影子账户初始化（D1 防守池轮动，起点=实际台账净值）",
               "shadow_d2_momentum": "影子账户初始化（D2 波动目标 tv10%，起点=实际台账净值）"}


def _strategy(cfg, name):
    return TRACKED[name](cfg)


def init_shadow(cfg, name, data, paper_dir, shadow_dir):
    """从实际账户 seed 影子账户。返回描述；已初始化返回 None。"""
    state_path = os.path.join(shadow_dir, f"account_{name}.json")
    if os.path.exists(state_path):
        return None
    initial = cfg["backtest"]["initial_cash"]
    actual_name = ACTUAL_NAME[name]
    nav_df = _read_csv(_csv_path(paper_dir, "nav", actual_name))
    if nav_df.empty:
        raise RuntimeError(f"实际台账 {actual_name} 为空，无法初始化影子账户")
    actual = load_state(paper_dir, actual_name, initial)
    start_date = nav_df["date"].max()
    start_nav = round(float(nav_df["nav"].iloc[-1]), 2)

    strategy = _strategy(cfg, name)
    target, _ = strategy.current_target(data)
    tgt = {c: round(float(w), 4) for c, w in target.items() if w > 0}
    close_panel = pd.DataFrame(
        {c: df.set_index("date")["close"] for c, df in data.items()}).sort_index()
    close_row = close_panel.loc[start_date]

    state = {"cash": actual["cash"], "shares": actual["shares"],
             "cost_basis": actual["cost_basis"],
             "pending": {"signal_date": f"{start_date:%Y-%m-%d}", "target": tgt,
                         "reason": INIT_REASON[name],
                         "signal_close": {c: round(float(close_row[c]), 4)
                                          for c in tgt if pd.notna(close_row.get(c))}},
             "last_signal_date": f"{start_date:%Y-%m-%d}",
             "executed_target": actual.get("executed_target", {})}
    save_state(shadow_dir, name, state)
    _append_csv(_csv_path(shadow_dir, "nav", name),
                {"date": f"{start_date:%Y-%m-%d}", "nav": start_nav}, NAV_COLUMNS)
    _append_csv(_csv_path(shadow_dir, "signals", name),
                {"date": f"{start_date:%Y-%m-%d}",
                 "target": json.dumps(tgt),
                 "reason": state["pending"]["reason"]}, SIGNAL_COLUMNS)
    return (f"{name}: 影子初始化 @ {start_date:%Y-%m-%d}，起点净值 {start_nav:,.0f}"
            f"（=实际 {actual_name}），镜像持仓 {actual['shares']}，"
            f"待成交目标 {tgt}")


def update(cfg, data, paper_dir, shadow_dir, stock_codes=()):
    """每日影子更新：先确保已初始化，再逐账户走 update_account 标准流程
    （含数据完整性闸门、T+1 成交、幂等）。"""
    os.makedirs(shadow_dir, exist_ok=True)
    actions = []
    for name in TRACKED:
        strategy = _strategy(cfg, name)
        init_msg = init_shadow(cfg, name, data, paper_dir, shadow_dir)
        if init_msg:
            actions.append(init_msg)
        actions += update_account(cfg, name, data, shadow_dir, stock_codes,
                                  strategy=strategy)
    return actions


def shadow_compare_lines(paper_dir, shadow_dir, nav_curves):
    """paper-report 影子对照节文本。影子不存在时返回空列表。"""
    lines = []
    if not os.path.isdir(shadow_dir):
        return lines
    for name in TRACKED:
        nav_df = _read_csv(_csv_path(shadow_dir, "nav", name))
        if nav_df.empty:
            continue
        if not lines:
            lines.append("\n----- 影子对照（D1/D2 影子运行，评审日 2026-12-08） -----")
        actual_name = ACTUAL_NAME[name]
        shadow_nav = nav_df.set_index("date")["nav"].sort_index()
        start_nav = shadow_nav.iloc[0]
        last_nav = shadow_nav.iloc[-1]
        line = (f"[{name}] 影子 {last_nav:,.0f}（{shadow_nav.index[-1]:%m-%d}，"
                f"起点 {shadow_nav.index[0]:%m-%d} 以来 {last_nav / start_nav - 1:+.2%}）")
        actual = nav_curves.get(actual_name)
        if actual is not None and not actual.empty:
            a_seg = actual[actual.index >= shadow_nav.index[0]]
            if not a_seg.empty:
                line += (f" vs 实际({actual_name}) {a_seg.iloc[-1]:,.0f}"
                         f"（同期 {a_seg.iloc[-1] / a_seg.iloc[0] - 1:+.2%}）"
                         f" → 影子差 {last_nav / start_nav - (a_seg.iloc[-1] / a_seg.iloc[0]):+.2%}")
        st = load_state(shadow_dir, name, 0)
        pend = st.get("pending") or {}
        line += (f"\n  持仓差异：实际({_holdings_text(paper_dir, actual_name)})"
                 f" vs 影子生效 {st.get('executed_target') or '空仓'}"
                 f"，待成交 {pend.get('target') or '无'}")
        lines.append(line)
    return lines


def _holdings_text(paper_dir, actual_name):
    st = load_state(paper_dir, actual_name, 0)
    return st.get("executed_target") or "空仓"
