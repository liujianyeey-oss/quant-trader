"""港股通资格快照、每手股数与港币日线。网络失败不静默缩小股票池。"""
from datetime import datetime
from io import BytesIO
import json
import logging
from pathlib import Path
import re
import time
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import requests

SSE_URL = "https://query.sse.com.cn/commonQuery.do"
SSE_PAGE = "https://www.sse.com.cn/services/hkexsc/disclo/eligible/"
HKEX_LIST = "https://www.hkex.com.hk/eng/services/trading/securities/securitieslists/ListOfSecurities.xlsx"


def hk_code(value):
    value = str(value).strip()
    if not re.fullmatch(r"\d{1,5}", value):
        raise ValueError(f"港股代码必须为 1–5 位数字：{value!r}")
    return value.zfill(5)


def today_hk():
    return pd.Timestamp(datetime.now(ZoneInfo("Asia/Hong_Kong")).date())


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)


class Universe:
    """snapshot 不代表历史资格；intervals 为用户提供的逐时点完整资格记录。

    intervals 每行有 valid_from / valid_to（含首尾）、buy_allowed、lot。
    同一股票不允许区间重叠；退出区间后禁止买入，卖出沿用最近已知手数。
    """

    def __init__(self, payload):
        self.payload = payload
        self.mode = payload.get("mode")
        if self.mode not in {"snapshot", "intervals"}:
            raise ValueError("universe.mode 必须为 snapshot 或 intervals")
        self.observed = pd.Timestamp(payload["observed_at"])
        self.records = {}
        for raw in payload["securities"]:
            row = dict(raw)
            c = hk_code(row["code"])
            lot = float(str(row["lot"]).replace(",", ""))
            if not np.isfinite(lot) or lot <= 0 or lot != int(lot):
                raise ValueError(f"{c} 每手股数无效")
            if row.get("currency") != "HKD" or row.get("category") != "Equity":
                raise ValueError(f"{c} 仅支持 HKD 普通股票")
            if not isinstance(row.get("buy_allowed", True), bool):
                raise ValueError(f"{c} buy_allowed 必须为布尔值")
            row.update(code=c, lot=int(lot))
            row["begin"] = pd.Timestamp(row["valid_from"]) if self.mode == "intervals" else self.observed
            row["finish"] = pd.Timestamp(row.get("valid_to") or "2099-12-31")
            if row["finish"] < row["begin"]:
                raise ValueError(f"{c} 资格区间起止错误")
            self.records.setdefault(c, []).append(row)
        if not self.records:
            raise ValueError("港股通名单为空")
        for c, rows in self.records.items():
            rows.sort(key=lambda r: r["begin"])
            if any(a["finish"] >= b["begin"] for a, b in zip(rows, rows[1:])):
                raise ValueError(f"{c} 资格区间重叠")

    @classmethod
    def load(cls, path):
        return cls(json.loads(Path(path).read_text(encoding="utf-8")))

    def info(self, code, date, research=False):
        rows = self.records.get(code, [])
        date = pd.Timestamp(date)
        if self.mode == "snapshot" and research:
            return rows[0] if rows else None
        return next((r for r in rows if r["begin"] <= date <= r["finish"]), None)

    def eligible(self, code, date, research=False):
        row = self.info(code, date, research)
        return row is not None and row.get("buy_allowed", True)

    def lot(self, code, date, research=False):
        row = self.info(code, date, research)
        if row is None:
            past = [r for r in self.records.get(code, []) if r["begin"] <= pd.Timestamp(date)]
            row = past[-1] if past else None
        if row is None:
            raise ValueError(f"{code} 在 {date} 没有已知每手股数")
        return row["lot"]


def refresh_universe(path):
    """默认采用沪港通南向可买股票，与 HKEX 股票/币种/手数交叉校验。

    沪、深名单并非完全相同：这里明确是 SH 通道子集，不冒称全港股通全集。
    保存抓取日期，绝不把 UPDATE_DATE 当成股票的历史纳入日期。
    """
    r = requests.get(SSE_URL, params={"sqlId": "COMMON_SSE_JYFW_HGT_XXPL_BDZQQD_L",
                                    "isPagination": "false"},
                     headers={"Referer": SSE_PAGE}, timeout=30)
    r.raise_for_status()
    all_rows = r.json()["result"]
    if len(all_rows) < 100:
        raise ValueError("上交所名单异常或返回了分页，拒绝覆盖缓存")
    if not set(str(x["TRADE_FLAG"]) for x in all_rows) <= {"1", "2"}:
        raise ValueError("上交所可交易状态字段变化，需检查解析器")
    r = requests.get(HKEX_LIST, timeout=30)
    r.raise_for_status()
    sheet = pd.read_excel(BytesIO(r.content), header=None, dtype=str)
    header = sheet.index[sheet[0].eq("Stock Code")]
    if len(header) != 1:
        raise ValueError("HKEX 证券列表格式变化")
    table = sheet.iloc[header[0] + 1:].copy()
    table.columns = sheet.iloc[header[0]].str.strip()
    lookup = {hk_code(x["Stock Code"]): x for _, x in table.iterrows()
              if re.fullmatch(r"\d{1,5}", str(x["Stock Code"]))}
    rows = []
    for stock in all_rows:
        if stock["SECURITY_TYPE"] != "股票" or str(stock["TRADE_FLAG"]) != "1":
            continue
        c = hk_code(stock["SECURITY_CODE"])
        if c not in lookup:
            raise ValueError(f"{c} 在 HKEX 手数表缺失，拒绝缩小名单")
        x = lookup[c]
        if x["Category"] != "Equity" or x["Trading Currency"] != "HKD":
            continue
        rows.append({"code": c, "name": stock["ABBR_CN"].strip(), "category": "Equity",
                     "currency": "HKD", "lot": int(x["Board Lot"].replace(",", "")),
                     "stamp_duty": x["Subject to Stamp Duty"] == "Y", "buy_allowed": True})
    payload = {"mode": "snapshot", "route": "SH", "observed_at": str(today_hk().date()),
               "source_updates": sorted(set(x["UPDATE_DATE"] for x in all_rows)),
               "sources": [SSE_PAGE, HKEX_LIST], "securities": rows}
    Universe(payload)  # 全部验证通过才原子替换
    save_json(Path(path).with_name(f"universe_{today_hk():%Y%m%d}.json"), payload)
    save_json(path, payload)
    return Universe(payload)


def normalize_bars(df):
    required = ["date", "open", "high", "low", "close", "volume", "signal_close"]
    if not set(required) <= set(df.columns):
        raise ValueError(f"HK 日线缺少列：{set(required) - set(df.columns)}")
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"], errors="raise").dt.normalize()
    if df["date"].duplicated().any():
        raise ValueError("HK 日线日期重复")
    for c in required[1:]:
        df[c] = pd.to_numeric(df[c], errors="raise")
    values = df[required[1:]].to_numpy(dtype=float)
    if not np.isfinite(values).all() or (df[["open", "high", "low", "close", "signal_close"]] <= 0).any().any() or (df.volume < 0).any():
        raise ValueError("HK 日线含非正价格、负成交量或无效数字；停牌请缺行表示")
    if ((df.high < df[["open", "close", "low"]].max(axis=1)) |
        (df.low > df[["open", "close", "high"]].min(axis=1))).any():
        raise ValueError("HK OHLC 价格关系错误")
    # 成交额近似值用于流动性筛选；volume 明确是股，非 A 股的手。
    df["turnover_hkd"] = df["close"] * df["volume"]
    return df.sort_values("date").reset_index(drop=True)


def align_prices(raw, adjusted):
    """供应商 qfq 可能忽略 start；截取共同边界，区间内部缺行仍报错。"""
    raw, adjusted = raw.copy(), adjusted.copy()
    raw["date"] = pd.to_datetime(raw.date)
    adjusted["date"] = pd.to_datetime(adjusted.date)
    begin = max(raw.date.min(), adjusted.date.min())
    finish = min(raw.date.max(), adjusted.date.max())
    if pd.isna(begin) or pd.isna(finish) or begin > finish:
        raise ValueError("复权/原始行情没有共同区间")
    raw = raw[raw.date.between(begin, finish)]
    adjusted = adjusted[adjusted.date.between(begin, finish)]
    if adjusted.attrs.get("adjustment_response") == "day":
        a = raw.set_index("date").sort_index()
        b = adjusted.set_index("date").sort_index()
        cols = ["open", "high", "low", "close"]
        if not a.index.equals(b.index) or not np.allclose(a[cols].astype(float), b[cols].astype(float), rtol=0, atol=.001):
            raise ValueError("qfq 接口返回 day，但与原始 OHLC 不一致，拒绝推断复权状态")
    joined = raw.merge(adjusted[["date", "close"]].rename(columns={"close": "signal_close"}),
                       on="date", how="outer", validate="one_to_one")
    return normalize_bars(joined)


class HKDataLoader:
    def __init__(self, cache_dir):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _fetch(code, start, end, adjust):
        endpoint = "hkfqkline" if adjust else "fqkline"
        param = f"hk{code},day,{start},{end},640,{adjust}"
        r = requests.get(f"https://web.ifzq.gtimg.cn/appstock/app/{endpoint}/get",
                         params={"param": param}, timeout=25)
        r.raise_for_status()
        payload = r.json()
        if payload.get("code") != 0:
            raise ValueError(f"腾讯 {code} {adjust or 'raw'}: {payload.get('msg')}")
        block = payload["data"]["hk" + code]
        key = adjust + "day" if adjust else "day"
        if adjust and not block.get(key) and block.get("day"):
            # 无调整记录的股票，成功的 qfq 请求也会使用 day 键。
            # 后续必须与独立原始行情逐日 OHLC 对照，而非失败后自行回退。
            key = "day"
            logging.warning("%s 的 qfq 响应使用 day 键，将逐日核对原始 OHLC", code)
        rows = block.get(key)
        if not rows:
            raise ValueError(f"{code} 没有明确的 {adjust or 'raw'} 数据，拒绝混用复权价格")
        frame = pd.DataFrame([x[:6] for x in rows],
                             columns=["date", "open", "close", "high", "low", "volume"])
        frame.attrs["adjustment_response"] = key
        return frame

    def load(self, code, start, end, offline=False):
        code = hk_code(code)
        path = self.cache_dir / f"{code}.parquet"
        # 为保证全日线，在线和离线都不使用香港当天的数据（次日可用）。
        cutoff = min(pd.Timestamp(end), today_hk() - pd.Timedelta(days=1))
        fresh = path.exists() and pd.Timestamp(path.stat().st_mtime, unit="s", tz="UTC").tz_convert("Asia/Hong_Kong").date() == today_hk().date()
        if offline or fresh:
            if not path.exists():
                raise ValueError(f"缺少缓存 {path}；先联网运行 signal 或导入文档格式的 parquet")
            df = normalize_bars(pd.read_parquet(path))
        else:
            # 全量成对刷新，禁止 qfq 尾部拼接；腾讯单次最多约 640 根，显式检查覆盖。
            raw = self._fetch(code, start, str(cutoff.date()), "")
            time.sleep(0.2)
            adj = self._fetch(code, start, str(cutoff.date()), "qfq")
            df = align_prices(raw, adj)
            df = df[df.date <= cutoff]
            if df.empty:
                raise ValueError(f"{code} 无已完成日线")
            tmp = path.with_suffix(".tmp.parquet")
            df.to_parquet(tmp, index=False)
            tmp.replace(path)
            save_json(path.with_suffix(".meta.json"), {"source": "Tencent", "code": code,
                      "updated_at": str(today_hk().date()), "qfq_response": adj.attrs.get("adjustment_response"),
                      "currency": "HKD", "volume_unit": "shares", "execution_prices": "unadjusted"})
        return df[(df.date >= pd.Timestamp(start)) & (df.date <= cutoff)].copy()
