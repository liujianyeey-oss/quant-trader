"""R4 数据层：币安 klines（复用镜像）、资金费率月度 dump、大户多空比/OI 可得性登记。"""
import io
import os
import time
import zipfile

import pandas as pd
import requests

CACHE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cache")
os.makedirs(CACHE, exist_ok=True)
S3 = "https://data.binance.vision"


def load_funding(symbol="BTCUSDT", start="2020-01", end="2026-08"):
    """币安 U 本位合约资金费率（月度 zip dump）。日频：当日 3 次费率均值。
    返回 (df[date, rate], note)。缓存于 cache/funding_{symbol}.csv。"""
    path = os.path.join(CACHE, f"funding_{symbol}.csv")
    try:
        months = pd.period_range(start, end, freq="M")
        frames = []
        for m in months:
            ym = str(m)
            url = f"{S3}/data/futures/um/monthly/fundingRate/{symbol}/{symbol}-fundingRate-{ym}.zip"
            r = requests.get(url, timeout=30)
            if r.status_code != 200:
                continue
            z = zipfile.ZipFile(io.BytesIO(r.content))
            name = z.namelist()[0]
            frames.append(pd.read_csv(z.open(name)))
            time.sleep(0.2)
        if not frames:
            raise RuntimeError("月度 fundingRate 全部下载失败")
        df = pd.concat(frames, ignore_index=True)
        df["date"] = pd.to_datetime(df["calc_time"], unit="ms").dt.normalize()
        df = df.groupby("date")["last_funding_rate"].mean().rename("rate").reset_index()
        df.to_csv(path, index=False)
        return df, ""
    except Exception as e:  # noqa: BLE001
        if os.path.exists(path):
            return pd.read_csv(path, parse_dates=["date"]), f"降级缓存（{type(e).__name__}）"
        raise


# 大户持仓多空比 / 持仓量 OI 数据可得性（2026-09-17 实测）：
# - fapi.binance.com / fapi1 对本机连接超时（与 api.binance.com 同被封禁段）
# - data.binance.vision 公共 dump 的 um/daily 下无 topLongShortPositionRatio /
#   topLongShortAccountRatio / openInterestHist 数据集（prefix 列举为空）
# - 无其他免费镜像 → 过滤器 F2（大户趋势）、F4（OI）标记「不可测」，不伪造
UNAVAILABLE = {
    "F2_大户多空比": "fapi 被封禁 + 公共 dump 无此数据集（topLongShort*），无免费镜像 → 不可测",
    "F4_OI持仓量": "fapi 被封禁 + 公共 dump 无此数据集（openInterestHist），无免费镜像 → 不可测",
}
