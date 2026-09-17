"""R4 编排：BTC/ETH 趋势过滤器 + 红利低波三变体。

过滤器（预声明，单独测试不叠加）：
- F1 量能确认：突破日成交量 > 前 20 日均量×1.5 才入场，否则放弃本次突破
- F2 大户多空比：数据不可得 → 不可测（见 data_layer.UNAVAILABLE）
- F3 资金费率拥挤：费率 20 日均值 > max(其 120 日均值+1σ, 0.05%/8h) → 禁止新开多仓
  （已持有不变）；T 日数据 T+1 可用（T-1 规则）
- F4 OI：数据不可得 → 不可测
过滤段与基准用**同一缩短窗口**对比（费率 2020-01 起 → 训练 2020-01~2022-12）。

红利低波（510880）：D0 买入持有；D1 MA60 趋势过滤；D2a 510880 进生产动量池（攻击池成员）、
D2b 防守资产换 510880（不重跑网格，独立检验）。
"""
import os
import sys

import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from macro_research.strategies import btc_donchian, run_single_asset, perf  # noqa: E402
from macro_research.data_layer import load_btc  # noqa: E402
from research_universe.runner import load_crypto  # noqa: E402
from research_r4.data_layer import load_funding, UNAVAILABLE  # noqa: E402
from data import DataLoader  # noqa: E402
from backtest import Backtester  # noqa: E402
from strategies import MomentumRotation  # noqa: E402
from metrics import calc_metrics  # noqa: E402

PROJECT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
os.makedirs(RESULTS, exist_ok=True)

BTC_COST = {"commission": 0.001, "slippage": 0.001, "lot": 1e-4}
ETF_COST = {"commission": 0.0001, "slippage": 0.001, "lot": 100}
INITIAL = 50_000.0


# ---------- BTC 唐奇安 + 入场过滤 ----------

def btc_donchian_filtered(df, entry=20, exit=10, allow_entry=None):
    """唐奇安 20/10，入场日需满足 allow_entry（Series, bool, T-1 对齐后传入）。"""
    d = df.set_index("date")
    upper = d["high"].rolling(entry).max().shift(1)
    lower = d["low"].rolling(exit).min().shift(1)
    holding, rows = False, []
    for i in range(entry + 1, len(d)):
        date = d.index[i]
        if not holding and d["close"].iloc[i] > upper.iloc[i]:
            if allow_entry is None or bool(allow_entry.get(date, False)):
                holding = True
            else:
                continue
        elif holding and d["close"].iloc[i] < lower.iloc[i]:
            holding = False
        else:
            continue
        rows.append((date, 1.0 if holding else 0.0))
    return pd.Series(dict(rows)).sort_index()


def f1_volume_mask(df, mult=1.5, window=20):
    """F1：突破日成交量 > 前 20 日均量 ×mult（不含当日）。"""
    d = df.set_index("date")
    avg = d["volume"].rolling(window).mean().shift(1)
    return (d["volume"] > mult * avg).fillna(False)


def f3_crowd_mask(funding, dates):
    """F3：拥挤=True 时禁止新开仓。费率日序列 → 20 日均值 vs max(120日均值+1σ, 0.05%)。
    T 日数据 T+1 可用：date t 用 ≤ t-1 的最近观测。"""
    f = funding.set_index("date")["rate"].sort_index()
    m20 = f.rolling(20).mean()
    base = m20.rolling(120).mean()
    sd = m20.rolling(120).std()
    th = pd.concat([base + sd, pd.Series(0.0005, index=base.index)], axis=1).max(axis=1)
    crowded = (m20 > th)
    ref = pd.DataFrame({"t1": pd.DatetimeIndex(dates).astype("datetime64[ns]") - pd.Timedelta(days=1)})
    right = crowded.rename("c").reset_index().sort_values("date")
    right["date"] = right["date"].astype("datetime64[ns]")
    j = pd.merge_asof(ref.sort_values("t1"), right, left_on="t1", right_on="date",
                      direction="backward")
    return pd.Series((~j["c"].fillna(False)).to_numpy(), index=dates)


def eval_btc(df, w, end, segments, initial=INITIAL):
    out = {}
    for label, a, b in segments:
        eq, trades = run_single_asset(df, w, a, pd.Timestamp(b) if b else end,
                                      BTC_COST, initial)
        m = perf(eq, trades, initial)
        out[label] = {**m, "trades": len([t for t in trades if t["side"] == "sell"])}
    return out


def main():
    rows = []

    # ================= 第一部分：BTC/ETH 趋势过滤 =================
    print("加载 BTC/ETH klines...", flush=True)
    btc, note_b = DL_note = load_btc()
    eth = load_crypto("ETH")
    print(f"BTC {btc['date'].min().date()}~{btc['date'].max().date()}, "
          f"ETH {eth['date'].min().date()}~{eth['date'].max().date()}")

    SEG = [("train", "2019-01-02", "2022-12-30"),
           ("valid", "2023-01-01", "2024-12-31"),
           ("report", "2025-01-01", None), ("full", "2019-01-02", None)]
    end_b, end_e = btc["date"].max(), eth["date"].max()

    # ---- BTC 基准 ----
    base_btc = btc_donchian(btc)
    res = eval_btc(btc, base_btc, end_b, SEG)
    for label, m in res.items():
        rows.append({"asset": "BTC", "variant": "基准donchian", "segment": label, **m})

    # ---- F1 量能确认 ----
    w_f1 = btc_donchian_filtered(btc, allow_entry=f1_volume_mask(btc))
    res = eval_btc(btc, w_f1, end_b, SEG)
    for label, m in res.items():
        rows.append({"asset": "BTC", "variant": "F1_量能确认", "segment": label, **m})
    print("BTC 基准/F1 done", flush=True)

    # ---- F3 资金费率拥挤（训练窗口随数据起点缩短） ----
    funding, f_note = load_funding("BTCUSDT")
    print(f"fundingRate {funding['date'].min().date()}~{funding['date'].max().date()} {f_note}")
    SEG_F3 = [("train", "2020-01-01", "2022-12-30"),
              ("valid", "2023-01-01", "2024-12-31"),
              ("report", "2025-01-01", None), ("full", "2020-01-01", None)]
    allow3 = f3_crowd_mask(funding, pd.DatetimeIndex(btc["date"]))
    w_f3 = btc_donchian_filtered(btc, allow_entry=allow3)
    res = eval_btc(btc, w_f3, end_b, SEG_F3)
    for label, m in res.items():
        rows.append({"asset": "BTC", "variant": "F3_费率拥挤过滤", "segment": label, **m})
    res = eval_btc(btc, base_btc, end_b, SEG_F3)
    for label, m in res.items():
        rows.append({"asset": "BTC", "variant": "基准donchian(F3同窗口)", "segment": label, **m})
    print("BTC F3 done", flush=True)

    # ---- F2/F4：不可测 ----
    for k, v in UNAVAILABLE.items():
        rows.append({"asset": "BTC", "variant": k, "segment": "all", "cagr": np.nan,
                     "mdd": np.nan, "sharpe": np.nan, "calmar": np.nan,
                     "trades": 0, "note": v})

    # ---- ETH：基准 + F1/F3（轻量复测，不重新寻优） ----
    base_eth = btc_donchian(eth)
    res = eval_btc(eth, base_eth, end_e, SEG)
    for label, m in res.items():
        rows.append({"asset": "ETH", "variant": "基准donchian", "segment": label, **m})
    w_e1 = btc_donchian_filtered(eth, allow_entry=f1_volume_mask(eth))
    res = eval_btc(eth, w_e1, end_e, SEG)
    for label, m in res.items():
        rows.append({"asset": "ETH", "variant": "F1_量能确认", "segment": label, **m})
    funding_e, _ = load_funding("ETHUSDT")
    allow3e = f3_crowd_mask(funding_e, pd.DatetimeIndex(eth["date"]))
    w_e3 = btc_donchian_filtered(eth, allow_entry=allow3e)
    res = eval_btc(eth, w_e3, end_e, SEG_F3)
    for label, m in res.items():
        rows.append({"asset": "ETH", "variant": "F3_费率拥挤过滤", "segment": label, **m})
    res = eval_btc(eth, base_eth, end_e, SEG_F3)
    for label, m in res.items():
        rows.append({"asset": "ETH", "variant": "基准donchian(F3同窗口)", "segment": label, **m})
    print("ETH done", flush=True)

    crypto = pd.DataFrame(rows)
    crypto.to_csv(os.path.join(RESULTS, "crypto_filters.csv"),
                  index=False, encoding="utf-8-sig")

    # ================= 第二部分：红利低波 =================
    print("\n红利低波...", flush=True)
    loader = DataLoader(cache_dir=os.path.join(PROJECT, "data_cache"))
    d880 = loader.load("510880", is_etf=True, start="2017-01-01")
    end_d = d880["date"].max()
    d_rows = []
    SEG_D = [("train", "2019-01-01", "2022-12-30"),
             ("valid", "2023-01-01", "2024-12-31"),
             ("full", "2019-01-01", None)]

    def eval_etf(df, w, label_name):
        for label, a, b in SEG_D:
            e = pd.Timestamp(b) if b else end_d
            eq, trades = run_single_asset(df, w, a, e, ETF_COST, 1_000_000)
            m = perf(eq, trades, 1_000_000)
            d_rows.append({"variant": label_name, "segment": label,
                           "trades": len([t for t in trades if t["side"] == "sell"]), **m})

    # D0 买入持有
    w0 = pd.Series(1.0, index=[d880.set_index("date").index[0]])
    eval_etf(d880, w0, "D0_买入持有")
    # D1 MA60 趋势过滤
    c = d880.set_index("date")["close"]
    ma60 = c.rolling(60).mean()
    w1 = (c > ma60).astype(float)
    w1 = w1[w1 != w1.shift(1)].dropna()
    eval_etf(d880, w1, "D1_MA60过滤")
    # D2a/D2b：生产动量框架
    cfg = yaml.safe_load(open(os.path.join(PROJECT, "config.yaml"), encoding="utf-8"))
    pool_codes = ["510300", "510500", "159915", "588000", "518880", "510880"]
    pdata = {code: loader.load(code, is_etf=True, start="2017-01-01") for code in pool_codes}
    bt = Backtester(cfg)

    def run_pool(name, cfg_mod):
        sig = MomentumRotation(cfg_mod).generate_signals(
            {k: v for k, v in pdata.items()
             if k in cfg_mod["universe"]["etfs"] + [cfg_mod["universe"]["defensive"]]})
        for label, a, b in SEG_D:
            e = pd.Timestamp(b) if b else end_d
            eq, trades = bt.run(pdata, sig, start=a, end=e)
            m = calc_metrics(eq["total_value"], trades,
                             initial=cfg["backtest"]["initial_cash"])
            d_rows.append({"variant": name, "segment": label,
                           "trades": m["交易次数"], "total": eq["total_value"].iloc[-1]
                           / cfg["backtest"]["initial_cash"] - 1,
                           "cagr": m["年化收益率"], "mdd": m["最大回撤"],
                           "sharpe": m["夏普比率"], "calmar": m["卡玛比率"]})

    import copy
    cfg_a = copy.deepcopy(cfg)
    cfg_a["universe"]["etfs"] = ["510300", "510500", "159915", "588000", "510880"]
    run_pool("D2a_510880进攻击池", cfg_a)
    cfg_b = copy.deepcopy(cfg)
    cfg_b["universe"]["defensive"] = "510880"
    run_pool("D2b_防守换510880", cfg_b)

    div = pd.DataFrame(d_rows)
    div.to_csv(os.path.join(RESULTS, "dividend.csv"), index=False, encoding="utf-8-sig")

    # ================= 控制台摘要 =================
    print("\n===== 加密过滤器矩阵（train/valid 卡玛） =====")
    piv = crypto.pivot_table(index=["asset", "variant"], columns="segment",
                             values="calmar", aggfunc="first")
    print(piv.map(lambda x: f"{x:.2f}" if pd.notna(x) else "-").to_string())
    print("\n===== 加密：年化/回撤（train） =====")
    tr = crypto[crypto["segment"] == "train"]
    print(tr.assign(cagr=tr["cagr"].map(lambda x: f"{x:.1%}"),
                    mdd=tr["mdd"].map(lambda x: f"{x:.1%}"),
                    calmar=tr["calmar"].map(lambda x: f"{x:.2f}"))
          [["asset", "variant", "trades", "cagr", "mdd", "calmar"]].to_string(index=False))
    print("\n===== 红利低波 =====")
    piv2 = div.pivot_table(index="variant", columns="segment",
                           values=["cagr", "mdd", "calmar"], aggfunc="first")
    print(piv2.map(lambda x: f"{x:.2f}" if pd.notna(x) and abs(x) < 10 else
                             (f"{x:.1%}" if pd.notna(x) else "-")).to_string())
    print(f"\n结果 CSV：{RESULTS}")


if __name__ == "__main__":
    main()
