"""R11 PIT interim：基于 6 期真实成分快照（2018-06~2020-12）的无幸存者偏差复核。

对比口径：2019-01-01 ~ 2020-12-31 窗口，
  PIT 池（信号日只用最近一期真实成分）vs 幸存者池（当前 717 只回填，R11 原版）。
两版数字之差 = 幸存者偏差的直接测量（interim：仅 2 年窗口，2021+ 待完整 17 期）。
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from research_r11.signals import build_panels, make_signals, perf_stats  # noqa: E402
from research_r11.engine import run_portfolio  # noqa: E402

PROJECT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PIT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "pit")
RESULTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
SNAPSHOTS = ["20180629", "20181231", "20190628", "20191231", "20200630", "20201231"]
WINDOW = ("2019-01-01", "2020-12-31")
MOM_W, VOL_W, TOP_N, MIN_BARS = 60, 60, 30, 250


def load_membership():
    tl = []
    for s in SNAPSHOTS:
        df = pd.read_csv(os.path.join(PIT, f"membership_{s}.csv"), dtype=str)
        tl.append((pd.Timestamp(f"{s[:4]}-{s[4:6]}-{s[6:]}"),
                   set(df["code6"].str.zfill(6))))
    return sorted(tl)


def members_at(membership, date):
    active = None
    for snap_date, codes in membership:
        if snap_date <= date:
            active = codes
        else:
            break
    return active


def load_all_data(codes):
    data = {}
    for c in sorted(codes):
        path = os.path.join(PROJECT, "data_cache", f"{c}_qfq.parquet")
        if os.path.exists(path):
            df = pd.read_parquet(path)
            if len(df) >= 60:
                data[c] = df
    return data


def make_pit_signals(panel, membership, kind, phase=1, top_n=TOP_N):
    calendar = panel.index
    from research_r11.signals import monthly_signal_dates
    dates = monthly_signal_dates(calendar, phase)
    mom = panel / panel.shift(MOM_W) - 1
    vol = panel.pct_change().rolling(VOL_W).std() * np.sqrt(252)
    bars = panel.notna().cumsum()
    signals = []
    for d in dates:
        i = calendar.get_loc(d)
        if i < MIN_BARS:
            continue
        active = members_at(membership, d)
        if active is None:
            continue
        elig = panel.columns[(bars.iloc[i] >= MIN_BARS) & panel.iloc[i].notna()]
        elig = [c for c in elig if c in active]
        if len(elig) < top_n:
            continue
        if kind == "mom60":
            winners = mom.iloc[i][elig].dropna().nlargest(top_n).index
        elif kind == "lowvol60":
            winners = vol.iloc[i][elig].dropna().nsmallest(top_n).index
        else:
            raise ValueError(kind)
        w = {c: 1.0 / len(winners) for c in winners}
        signals.append((d, w))
    return signals


def coverage_report(panel, membership):
    rows = []
    for d in [pd.Timestamp("2019-06-03"), pd.Timestamp("2020-01-02"),
              pd.Timestamp("2020-07-01")]:
        active = members_at(membership, d)
        have = sum(1 for c in active if c in panel.columns and panel[c].loc[:d].notna().sum() >= MIN_BARS)
        rows.append({"date": d.date(), "members": len(active), "with_data": have})
    return rows


def main():
    membership = load_membership()
    current = set(pd.read_csv(os.path.join(RESULTS, "universe.csv"), dtype={"code": str})["code"].str.zfill(6))
    pit_union = set().union(*[c for _, c in membership])
    all_codes = pit_union | current
    print(f"PIT并集 {len(pit_union)}，当前 {len(current)}，总加载 {len(all_codes)}", flush=True)

    data = load_all_data(all_codes)
    have_pit = len([c for c in pit_union if c in data])
    print(f"加载成功 {len(data)}；PIT成员覆盖率 {have_pit}/{len(pit_union)}", flush=True)

    panel = build_panels(data)
    print("数据覆盖点检:", coverage_report(panel, membership), flush=True)

    data_survivor = {c: df for c, df in data.items() if c in current}
    panel_survivor = build_panels(data_survivor)

    out = []
    for kind in ("mom60", "lowvol60"):
        # PIT 版
        sig_pit = make_pit_signals(panel, membership, kind, phase=1)
        eq, trades, stats = run_portfolio(data, sig_pit, *WINDOW, 1_000_000)
        m = perf_stats(eq, trades, 1_000_000)
        out.append({"variant": kind, "pool": "PIT", "n_signals": len(sig_pit), **m,
                    **{f"skip_{k}": v for k, v in stats.items()}})
        # 幸存者版
        sig_srv = make_signals(panel_survivor, kind, phase=1)
        eq2, trades2, stats2 = run_portfolio(data_survivor, sig_srv, *WINDOW, 1_000_000)
        m2 = perf_stats(eq2, trades2, 1_000_000)
        out.append({"variant": kind, "pool": "survivor", "n_signals": len(sig_srv), **m2,
                    **{f"skip_{k}": v for k, v in stats2.items()}})
        print(f"{kind} 完成", flush=True)

    res = pd.DataFrame(out)
    os.makedirs(os.path.join(PIT, "results"), exist_ok=True)
    res.to_csv(os.path.join(PIT, "results", "pit_interim.csv"), index=False, encoding="utf-8-sig")
    pd.set_option("display.width", 200)
    show = res.assign(cagr=res["cagr"].map("{:.1%}".format),
                      mdd=res["mdd"].map("{:.1%}".format),
                      calmar=res["calmar"].map("{:.2f}".format),
                      sharpe=res["sharpe"].map("{:.2f}".format),
                      total=res["total"].map("{:.1%}".format),
                      turnover=res["turnover"].map("{:.1f}".format))
    print(show[["variant", "pool", "n_signals", "total", "cagr", "mdd", "sharpe",
                "calmar", "turnover"]].to_string(index=False))

    # 基准（窗口内）
    import akshare as ak
    for name, label in (("sh000300", "沪深300"), ("sh000905", "中证500")):
        try:
            b = ak.stock_zh_index_daily(symbol=name)
            b["date"] = pd.to_datetime(b["date"])
            bs = b.set_index("date")["close"]
            seg = bs[(bs.index >= WINDOW[0]) & (bs.index <= WINDOW[1])]
            y = ((seg.index[-1] - seg.index[0]).days + 1) / 365.25
            cagr = (seg.iloc[-1] / seg.iloc[0]) ** (1 / y) - 1
            mdd = (seg / seg.cummax() - 1).min()
            print(f"基准 {label}: cagr {cagr:.1%} mdd {mdd:.1%} calmar {cagr/abs(mdd):.2f}")
        except Exception as e:  # noqa: BLE001
            print(f"基准 {label} 拉取失败: {e}")


if __name__ == "__main__":
    main()
