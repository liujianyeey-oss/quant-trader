"""研究用数据加载：复用 us_equities.data.load（新浪日线+复权因子，CSV缓存），
缓存目录独立（research/cache/），并产出覆盖表。

口径说明：adjusted price = raw close × factor，factor 含拆分调整；
新浪 factor 的股息调整在 c 列（dividend_adjustment），us_equities 框架不使用，
因此收益不含分红（与 US_SWING 口径一致）。
"""
import os

import pandas as pd

from us_equities.data import load

CACHE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cache")

# 全品类 ETF 池（27 只，2017 起）
BROAD = ["SPY", "QQQ", "IWM", "DIA", "MDY", "RSP"]
SECTOR = ["XLK", "XLF", "XLE", "XLV", "XLY", "XLI", "XLP", "XLU", "XLB", "XLRE"]
BOND = ["TLT", "IEF", "LQD", "HYG", "SHY"]
COMMODITY = ["GLD", "SLV", "USO", "DBC"]
OVERSEAS = ["EFA", "EEM"]
ALL_ETFS = BROAD + SECTOR + BOND + COMMODITY + OVERSEAS

CATEGORY = {**{s: "宽基" for s in BROAD}, **{s: "行业" for s in SECTOR},
            **{s: "债券" for s in BOND}, **{s: "商品" for s in COMMODITY},
            **{s: "海外" for s in OVERSEAS}}


def load_universe(symbols=ALL_ETFS, cache=CACHE, offline=False):
    """加载全部标的。返回 (data: {symbol: df}, coverage: DataFrame)。
    拉取失败的标的剔除并在覆盖表注明原因，不伪造数据。"""
    data, rows = {}, []
    for sym in symbols:
        try:
            df = load(sym, cache, offline=offline)
            data[sym] = df
            cal = pd.bdate_range(df["date"].min(), df["date"].max())
            rows.append({"symbol": sym, "类别": CATEGORY.get(sym, ""),
                         "起始日": str(df["date"].min().date()),
                         "截止日": str(df["date"].max().date()),
                         "行数": len(df),
                         "缺口(相对工作日)": len(cal) - len(df),
                         "隔离坏行": len(df.attrs.get("quarantined", [])),
                         "备注": ""})
        except Exception as e:
            rows.append({"symbol": sym, "类别": CATEGORY.get(sym, ""),
                         "起始日": "", "截止日": "", "行数": 0,
                         "缺口(相对工作日)": "", "隔离坏行": "",
                         "备注": f"剔除：{type(e).__name__}: {e}"})
    return data, pd.DataFrame(rows)
