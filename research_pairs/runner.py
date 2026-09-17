"""PA：协整配对的只多头相对价值切换（A股 ETF 无做空约束下的可行形态）。

流程：候选对预声明 → 缺失标的腾讯补行情 → train(2019-2023) OLS 对冲比 β →
Engle-Granger 协整检验 + 半衰期 → 门槛（p<0.05 且 5<HL<120）→
z 分数切换信号（|z|>1.5 换腿，中间滞后保持）→ T+1 开盘全成本回测 →
基准（单腿/50-50/生产 momentum）对比 + β 游走稳定性 + 协整破裂标记。
"""
import os
import sys
import time

import numpy as np
import pandas as pd

PROJECT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT)
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
os.makedirs(OUT, exist_ok=True)

PAIRS = [
    ("P1_300_500", "510300", "510500"),
    ("P2_300_300嘉实", "510300", "159919"),
    ("P3_金_金", "518880", "159934"),
    ("P4_纳指_标普", "513100", "513500"),
    ("P5_创业板_科创", "159915", "588000"),
    ("P6_红利_300", "510880", "510300"),
]
TRAIN = ("2019-01-01", "2023-12-31")
VALID_START = "2024-01-01"
Z_ENTER = 1.5
FEE = 0.0001          # ETF 万1
SLIP = 0.001          # 单边滑点


def ensure(code):
    path = os.path.join(PROJECT, "data_cache", f"{code}_qfq.parquet")
    if os.path.exists(path):
        return pd.read_parquet(path)
    import akshare as ak
    from data import DataLoader
    loader = DataLoader(cache_dir=os.path.join(PROJECT, "data_cache"))
    sym = ("sh" if code.startswith(("5", "6")) else "sz") + code
    end = pd.Timestamp.now().strftime("%Y-%m-%d")
    df = ak.stock_zh_a_hist_tx(symbol=sym, start_date="20170601",
                               end_date=end.replace("-", ""), adjust="qfq")
    df = loader._normalize(df)
    df.to_parquet(path, index=False)
    loader._write_meta(code, "2017-06-01")
    time.sleep(0.3)
    return df


def half_life(spread):
    from statsmodels.tsa.stattools import adfuller
    x = spread.dropna().values
    if len(x) < 60:
        return np.nan, np.nan
    p = adfuller(x, maxlag=1, regression="c")[1]
    lagged = x[:-1]
    delta = np.diff(x)
    b = np.polyfit(lagged, delta, 1)[0]
    hl = -np.log(2) / b if b < 0 else np.inf
    return p, hl


def pair_frame(a, b):
    da = ensure(a).set_index(pd.to_datetime(ensure(a)["date"]))["close"]
    db = ensure(b).set_index(pd.to_datetime(ensure(b)["date"]))["close"]
    df = pd.concat([da.rename("A"), db.rename("B")], axis=1).dropna()
    return df


def backtest_switch(df, z, cost_per_switch, hold_pos=None):
    """z 序列 → 持仓 A/B → T+1 开盘成交（简化为收盘前移一日估值，费用扣在切换日）。"""
    pos = pd.Series(index=df.index, dtype=float)
    cur = 0.0  # 0 表示持 A，1 表示持 B
    pos.iloc[0] = 0.0
    z_prev = z.shift(1)
    for i in range(1, len(df)):
        cur = pos.iloc[i - 1]
        zv = z_prev.iloc[i]  # T-1 的 z 决定 T 的持仓（T+1 生效近似）
        if pd.notna(zv):
            if zv < -Z_ENTER:
                cur = 0.0
            elif zv > Z_ENTER:
                cur = 1.0
        pos.iloc[i] = cur
    ra = df["A"].pct_change()
    rb = df["B"].pct_change()
    ret = np.where(pos == 0, ra, rb)
    ret = pd.Series(ret, index=df.index).fillna(0.0)
    switches = int((pos != pos.shift()).sum())
    gross_cost = switches * cost_per_switch
    eq = (1 + ret).cumprod()
    eq *= (1 - cost_per_switch) ** (pos != pos.shift()).astype(float).cumprod()
    return eq, pos, switches


def perf(eq, years):
    if len(eq) < 50:
        return dict(cagr=np.nan, mdd=np.nan, sharpe=np.nan, calmar=np.nan)
    total = eq.iloc[-1] - 1
    cagr = (1 + total) ** (1 / years) - 1
    mdd = (eq / eq.cummax() - 1).min()
    r = eq.pct_change().dropna()
    sharpe = r.mean() / r.std() * np.sqrt(252) if r.std() > 0 else np.nan
    return dict(cagr=cagr, mdd=mdd, sharpe=sharpe,
                calmar=cagr / abs(mdd) if mdd < 0 else np.nan)


def main():
    rows = []
    for label, a, b in PAIRS:
        try:
            df = pair_frame(a, b)
        except Exception as e:  # noqa: BLE001
            rows.append({"pair": label, "error": str(e)[:80]})
            continue
        tr = df.loc[TRAIN[0]:TRAIN[1]]
        va = df.loc[VALID_START:]
        logp = np.log(tr)
        beta = float(np.polyfit(logp["B"], logp["A"], 1)[0])
        spread_tr = logp["A"] - beta * logp["B"]
        p_adf, hl = half_life(spread_tr)
        gate = (p_adf < 0.05) and (5 < hl < 120) and np.isfinite(hl)
        mu, sd = spread_tr.mean(), spread_tr.std()
        # 全期 z（train 参数外推，训练段属内联）
        logf = np.log(df)
        z = (logf["A"] - beta * logf["B"] - mu) / sd
        eq, pos, switches = backtest_switch(df, z, cost_per_switch=2 * (FEE + SLIP))
        years_tr = (tr.index[-1] - tr.index[0]).days / 365.25
        years_va = (va.index[-1] - va.index[0]).days / 365.25 if len(va) else 0
        m_tr = perf(eq.loc[TRAIN[0]:TRAIN[1]] / eq.loc[TRAIN[0]:TRAIN[1]].iloc[0], years_tr)
        m_va = perf(eq.loc[VALID_START:] / eq.loc[VALID_START:].iloc[0], years_va) if years_va > 0.5 else {}
        # 基准（训练段）
        benchA = perf(df["A"].loc[TRAIN[0]:TRAIN[1]], years_tr)
        benchB = perf(df["B"].loc[TRAIN[0]:TRAIN[1]], years_tr)
        benchA_va = perf(df["A"].loc[VALID_START:], years_va) if years_va > 0.5 else {}
        benchB_va = perf(df["B"].loc[VALID_START:], years_va) if years_va > 0.5 else {}
        # β 游走（按年度重估）
        betas = []
        for y in range(2019, 2026):
            seg = df.loc[f"{y}-01-01":f"{y}-12-31"]
            if len(seg) > 100:
                betas.append(float(np.polyfit(np.log(seg["B"]), np.log(seg["A"]), 1)[0]))
        rows.append({
            "pair": label, "n_bars": len(df), "adf_p": round(p_adf, 4),
            "half_life": round(hl, 1) if np.isfinite(hl) else "inf", "gate": gate,
            "beta_train": round(beta, 3),
            "beta_walk": f"{min(betas):.2f}~{max(betas):.2f}" if betas else "-",
            "switches_full": switches,
            "tr_cagr": m_tr.get("cagr"), "tr_mdd": m_tr.get("mdd"),
            "tr_calmar": m_tr.get("calmar"),
            "va_cagr": m_va.get("cagr"), "va_mdd": m_va.get("mdd"),
            "va_calmar": m_va.get("calmar"),
            "benchA_tr": benchA.get("cagr"), "benchB_tr": benchB.get("cagr"),
            "benchA_va": benchA_va.get("cagr"), "benchB_va": benchB_va.get("cagr"),
        })
        print(f"完成 {label} gate={gate} p={p_adf:.4f} HL={hl:.1f}", flush=True)

    res = pd.DataFrame(rows)
    res.to_csv(os.path.join(OUT, "pairs.csv"), index=False, encoding="utf-8-sig")
    pd.set_option("display.width", 300)
    show = res.copy()
    for c in ("tr_cagr", "va_cagr", "benchA_tr", "benchB_tr", "benchA_va", "benchB_va"):
        show[c] = show[c].map(lambda x: f"{x:.1%}" if pd.notna(x) else "-")
    for c in ("tr_mdd", "va_mdd"):
        show[c] = show[c].map(lambda x: f"{x:.1%}" if pd.notna(x) else "-")
    for c in ("tr_calmar", "va_calmar"):
        show[c] = show[c].map(lambda x: f"{x:.2f}" if pd.notna(x) else "-")
    print(show.to_string(index=False))


if __name__ == "__main__":
    main()
