"""宏观与标的数据层：BTC（币安）、DXY 代理（FRED DTWEXBGS）、美债10Y（akshare
bond_zh_us_rate）、513130（项目 data.py 东财/腾讯双源）、USO/GLD（已有缓存）。

全部带 CSV 缓存（macro_research/cache/），失败降级缓存，不许伪造数据。
"""
import os
import sys
import time

import pandas as pd
import requests

CACHE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cache")
os.makedirs(CACHE, exist_ok=True)
PROJECT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT)


def _cache_or_fetch(name, fetch_fn):
    """先拉新，失败降级缓存。返回 (df, note)。"""
    path = os.path.join(CACHE, f"{name}.csv")
    try:
        df = fetch_fn()
        df.to_csv(path, index=False)
        return df, ""
    except Exception as e:  # noqa: BLE001
        if os.path.exists(path):
            return pd.read_csv(path, parse_dates=["date"]), \
                f"拉取失败降级缓存（{type(e).__name__}: {str(e)[:80]}）"
        raise


# ---------- BTC：币安日线（365 天日历） ----------

def _fetch_btc():
    """api.binance.com 对本机不可达，用官方镜像 data-api.binance.vision。"""
    rows = []
    end_ms = int(pd.Timestamp.now().timestamp() * 1000)
    start_ms = int(pd.Timestamp("2017-08-17").timestamp() * 1000)  # BTCUSDT 上市日
    while True:
        r = requests.get("https://data-api.binance.vision/api/v3/klines",
                         params={"symbol": "BTCUSDT", "interval": "1d",
                                 "startTime": start_ms, "endTime": end_ms,
                                 "limit": 1000}, timeout=30)
        r.raise_for_status()
        batch = r.json()
        if not batch:
            break
        rows += batch
        start_ms = batch[-1][0] + 86_400_000
        if len(batch) < 1000:
            break
        time.sleep(0.5)
    df = pd.DataFrame(rows, columns=["open_time", "open", "high", "low", "close",
                                     "volume", "_ct", "_qv", "_n", "_tb", "_tq", "_i"])
    df["date"] = pd.to_datetime(df["open_time"], unit="ms").dt.normalize()
    for c in ("open", "high", "low", "close", "volume"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df[["date", "open", "high", "low", "close", "volume"]]
    df = df[df["date"] < pd.Timestamp.now().normalize()]  # 不含当日未完成 K 线
    return df.drop_duplicates("date").sort_values("date").reset_index(drop=True)


def load_btc():
    return _cache_or_fetch("btc", _fetch_btc)


# ---------- DXY 代理：FRED DTWEXBGS（贸易加权美元指数：广义） ----------

def _fetch_dxy():
    r = requests.get("https://fred.stlouisfed.org/graph/fredgraph.csv",
                     params={"id": "DTWEXBGS", "cosd": "2017-01-01"}, timeout=30)
    r.raise_for_status()
    from io import StringIO
    df = pd.read_csv(StringIO(r.text))
    df.columns = ["date", "dxy"]
    df["date"] = pd.to_datetime(df["date"])
    df["dxy"] = pd.to_numeric(df["dxy"], errors="coerce")
    return df.dropna().reset_index(drop=True)


def load_dxy():
    return _cache_or_fetch("dxy", _fetch_dxy)


# ---------- 美债 10 年收益率：akshare bond_zh_us_rate ----------

def _fetch_us10y():
    import akshare as ak
    df = ak.bond_zh_us_rate(start_date="20170101")
    out = df[["日期", "美国国债收益率10年"]].copy()
    out.columns = ["date", "us10y"]
    out["date"] = pd.to_datetime(out["date"])
    out["us10y"] = pd.to_numeric(out["us10y"], errors="coerce")
    return out.dropna().reset_index(drop=True)


def load_us10y():
    return _cache_or_fetch("us10y", _fetch_us10y)


# ---------- 恒生科技 ETF 513130（项目双源数据层） ----------

def load_513130():
    from data import DataLoader
    loader = DataLoader(cache_dir=os.path.join(PROJECT, "data_cache"))
    df = loader.load("513130", is_etf=True, start="2021-01-01")
    return df, ""


# ---------- USO / GLD（已有缓存，仅作覆盖表展示；预先声明的假设未使用） ----------

def load_oil_gold_notes():
    notes = []
    uso_path = os.path.join(PROJECT, "us_equities", "research", "cache", "USO.csv")
    if os.path.exists(uso_path):
        d = pd.read_csv(uso_path, parse_dates=["date"])
        notes.append(("USO(油)", str(d["date"].min().date()), str(d["date"].max().date()),
                      len(d), "us_equities 缓存（未刷新）"))
    gld_path = os.path.join(PROJECT, "data_cache", "518880_qfq.parquet")
    if os.path.exists(gld_path):
        d = pd.read_parquet(gld_path)
        notes.append(("518880(金)", str(d["date"].min().date()), str(d["date"].max().date()),
                      len(d), "A股 data_cache 缓存（未刷新）"))
    return notes
