"""OC1 数据采集：全免费 BTC 链上指标。

- blockchain.info charts：hash-rate / n-transactions / n-unique-addresses（2019 起）
- Coin Metrics 社区 API：AdrActCnt / TxCnt / FlowNetExch / SplyExchBal / NVTAdj
缺口记录到 coverage 表，不许伪造。
"""
import json
import os
import time

import pandas as pd
import requests

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cache")
os.makedirs(OUT, exist_ok=True)


def fetch_blockchaininfo(chart):
    url = f"https://api.blockchain.info/charts/{chart}?timespan=all&format=json&cors=true"
    r = requests.get(url, timeout=30)
    r.raise_for_status()
    data = r.json()["values"]
    df = pd.DataFrame(data)
    df["date"] = pd.to_datetime(df["x"], unit="s")
    df = df[["date", "y"]].rename(columns={"y": chart.replace("-", "_")})
    return df


def fetch_coinmetrics(metric, page_limit=400):
    url = ("https://community-api.coinmetrics.io/v4/timeseries/asset-metrics"
           f"?assets=btc&metrics={metric}&start_time=2019-01-01&page_size=10000")
    rows, token = [], None
    for _ in range(page_limit):
        u = url + (f"&next_page_token={token}" if token else "")
        r = requests.get(u, timeout=30)
        r.raise_for_status()
        j = r.json()
        rows += j.get("data", [])
        token = j.get("next_page_token")
        if not token:
            break
        time.sleep(0.2)
    if not rows:
        return None
    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["time"]).dt.tz_localize(None).dt.normalize()
    df[metric] = pd.to_numeric(df[metric], errors="coerce")
    return df[["date", metric]]


def save(df, name):
    p = os.path.join(OUT, f"{name}.csv")
    df.to_csv(p, index=False)
    print(f"{name}: {len(df)} 行，{df['date'].min().date()} ~ {df['date'].max().date()}")


def main():
    for chart in ["hash-rate", "n-transactions", "n-unique-addresses"]:
        try:
            save(fetch_blockchaininfo(chart), chart.replace("-", "_"))
            time.sleep(1)
        except Exception as e:  # noqa: BLE001
            print(f"{chart} 失败: {e}")
    for metric in ["AdrActCnt", "TxCnt", "FlowNetExch", "SplyExchBal", "NVTAdj"]:
        try:
            df = fetch_coinmetrics(metric)
            if df is None:
                print(f"{metric}: 无数据")
            else:
                save(df, metric)
            time.sleep(1)
        except Exception as e:  # noqa: BLE001
            print(f"{metric} 失败: {type(e).__name__} {str(e)[:100]}")


if __name__ == "__main__":
    main()
