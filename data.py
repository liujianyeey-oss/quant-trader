"""数据层：akshare 拉取日线 + 本地 parquet 缓存。

主数据源东财（ETF 走 fund_etf_hist_em，个股走 stock_zh_a_hist），
备用数据源腾讯（stock_zh_a_hist_tx，ETF/个股通用），均为前复权(qfq)日线。
东财全部镜像重试失败 → 自动切腾讯；腾讯也失败且有本地缓存 → 降级用缓存
（WARN 提示），缓存也没有才 raise。

缓存一致性规则（审计修复）：
- 前复权历史会在分红等事件后整体重算，只做尾部拼接可能混用不同复权基准。
  因此缓存一旦过期（非当天新鲜），就按 meta 记录的起始日**全量重拉**，保证
  全历史使用同一复权基准，不存在接缝。
- 当日未完成的 K 线（盘中快照）**不写入缓存**；盘中运行时通过
  allow_intraday=True 现场拉取尾部、仅在内存中附加当日行。
- 同一天内重复运行不重复拉取（盘中拉过的缓存当天有效；收盘后若缓存是
  收盘前写的，会再全量重拉一次以纳入当日完成的 K 线）。
"""
import os
import time
import json
import logging
from datetime import datetime, time as dtime

import pandas as pd
import requests
import akshare as ak

logger = logging.getLogger(__name__)

MARKET_CLOSE_SETTLE = dtime(15, 30)  # 当日收盘数据大致可用的时刻

# 东财 push2his.eastmoney.com 接口在本网络下间歇性断连（curl 52），
# 其镜像子域（33/63/83.push2his...）提供相同 API。失败后启用镜像轮询重试。
_EM_HOST = "push2his.eastmoney.com"
_EM_MIRRORS = ["push2his.eastmoney.com", "33.push2his.eastmoney.com",
               "63.push2his.eastmoney.com", "83.push2his.eastmoney.com"]
_mirror_state = {"patched": False, "idx": 0}


def _patch_eastmoney_mirrors():
    if _mirror_state["patched"]:
        return
    _orig_get = requests.get

    def _get_with_mirror(url, *args, **kwargs):
        if isinstance(url, str) and _EM_HOST in url:
            # 每次请求轮换镜像域名
            _mirror_state["idx"] += 1
            mirror = _EM_MIRRORS[_mirror_state["idx"] % len(_EM_MIRRORS)]
            url = url.replace(_EM_HOST, mirror, 1)
        return _orig_get(url, *args, **kwargs)

    requests.get = _get_with_mirror
    _mirror_state["patched"] = True
    logger.info("已启用东财接口镜像轮询（默认域名不稳定）")


class DataLoader:
    def __init__(self, cache_dir="data_cache", adjust="qfq"):
        self.cache_dir = cache_dir
        self.adjust = adjust
        os.makedirs(cache_dir, exist_ok=True)

    def _cache_path(self, code):
        return os.path.join(self.cache_dir, f"{code}_{self.adjust}.parquet")

    def _meta_path(self, code):
        return os.path.join(self.cache_dir, f"{code}_{self.adjust}.meta.json")

    def _read_meta(self, code):
        try:
            with open(self._meta_path(code), encoding="utf-8") as f:
                return json.load(f)
        except (OSError, ValueError):
            return {}

    def _write_meta(self, code, start):
        with open(self._meta_path(code), "w", encoding="utf-8") as f:
            json.dump({"start": start}, f)

    def _fetch_em(self, code, is_etf, start, end):
        """东财源（主）：akshare fund_etf_hist_em / stock_zh_a_hist，失败重试。"""
        start_s = start.replace("-", "")
        end_s = end.replace("-", "")
        last_err = None
        for attempt in range(8):  # 东财接口偶发断连，多重试几次
            try:
                if is_etf:
                    df = ak.fund_etf_hist_em(symbol=code, period="daily",
                                             start_date=start_s, end_date=end_s,
                                             adjust=self.adjust)
                else:
                    df = ak.stock_zh_a_hist(symbol=code, period="daily",
                                            start_date=start_s, end_date=end_s,
                                            adjust=self.adjust)
                if df is None or df.empty:
                    raise ValueError(f"{code} 返回空数据")
                return df
            except Exception as e:
                last_err = e
                logger.warning("拉取 %s 失败(第%d次): %s", code, attempt + 1, e)
                if isinstance(e, requests.exceptions.ConnectionError):
                    _patch_eastmoney_mirrors()  # 默认域名不通时启用镜像轮询
                time.sleep(min(2 + attempt * 2, 10))  # 退避，避开限流窗口
        raise RuntimeError(f"东财源拉取 {code} 数据失败: {last_err}")

    @staticmethod
    def _tx_symbol(code):
        """东财 6 位代码 → 腾讯带市场前缀代码（5/6 开头=sh，0/3/15 开头=sz）。"""
        return ("sh" if code.startswith(("5", "6")) else "sz") + code

    def _fetch_tx(self, code, start, end):
        """腾讯源（备用）：akshare stock_zh_a_hist_tx，ETF/个股通用，支持 qfq。"""
        symbol = self._tx_symbol(code)
        start_s = start.replace("-", "")
        end_s = end.replace("-", "")
        last_err = None
        for attempt in range(3):
            try:
                df = ak.stock_zh_a_hist_tx(symbol=symbol, start_date=start_s,
                                           end_date=end_s, adjust=self.adjust)
                if df is None or df.empty:
                    raise ValueError(f"{symbol} 返回空数据")
                return df
            except Exception as e:
                last_err = e
                logger.warning("腾讯源拉取 %s 失败(第%d次): %s", symbol, attempt + 1, e)
                time.sleep(min(2 + attempt * 2, 6))
        raise RuntimeError(f"腾讯源拉取 {code} 数据失败: {last_err}")

    def _fetch(self, code, is_etf, start, end):
        """主源东财，全部镜像重试失败后自动切腾讯备用源；都失败则 raise。"""
        try:
            return self._fetch_em(code, is_etf, start, end)
        except RuntimeError as e:
            logger.warning("%s，切换腾讯备用源拉取 %s", e, code)
            return self._fetch_tx(code, start, end)

    @staticmethod
    def _normalize(df):
        """统一列名为 date/open/high/low/close/volume，按日期排序。
        兼容东财中文列名与腾讯英文列名。"""
        if "date" not in df.columns:  # 东财
            rename = {"日期": "date", "开盘": "open", "最高": "high",
                      "最低": "low", "收盘": "close", "成交量": "volume"}
            df = df.rename(columns=rename)
        df = df[["date", "open", "high", "low", "close", "volume"]].copy()
        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values("date").drop_duplicates("date").reset_index(drop=True)
        for c in ["open", "high", "low", "close", "volume"]:
            df[c] = pd.to_numeric(df[c], errors="coerce")
        return df

    @staticmethod
    def _is_market_settled(now=None):
        """当前是否已过当日收盘数据可用时刻。"""
        now = now or datetime.now()
        return now.time() >= MARKET_CLOSE_SETTLE

    def _is_fresh(self, path, now):
        """缓存是否当天新鲜。收盘后要求缓存是收盘后更新的（含当日完成K线）。"""
        mtime = datetime.fromtimestamp(os.path.getmtime(path))
        if mtime.date() != now.date():
            return False
        if self._is_market_settled(now):
            return mtime.time() >= MARKET_CLOSE_SETTLE
        return True

    def load(self, code, is_etf=True, start="2019-01-01", end=None,
             allow_intraday=False):
        """加载日线数据。

        - 缓存新鲜（当天已拉取）则直接复用；否则按一致复权基准全量重拉。
        - 缓存只保存已完成的日线（不含当日盘中快照）。
        - allow_intraday=True 且盘中运行时，现场拉取尾部并在返回值里附加
          当日未完成 K 线（不落盘），供 signal 子命令使用。
        """
        now = datetime.now()
        today = now.date()
        if end is None:
            end = now.strftime("%Y-%m-%d")
        start = pd.to_datetime(start)
        path = self._cache_path(code)

        cached = None
        meta_start = None
        if os.path.exists(path):
            cached = pd.read_parquet(path)
            meta_start = self._read_meta(code).get("start")
            # 盘中读取时剔除可能混入的当日未完成 K 线并回写缓存（运行时自愈）
            if not self._is_market_settled(now) and not cached.empty:
                clean = cached[cached["date"].dt.date < today]
                if len(clean) < len(cached):
                    logger.info("%s 缓存含当日未完成K线，已剔除并回写", code)
                    if not clean.empty:
                        clean.to_parquet(path, index=False)
                    cached = clean

        covers_start = meta_start is not None and pd.to_datetime(meta_start) <= start

        # 当日 K 线：收盘结算后视为已完成可入缓存；盘中视为未完成快照不入缓存
        keep_today = self._is_market_settled(now)

        if cached is None or cached.empty or not covers_start:
            fetch_start = min(start, pd.to_datetime(meta_start)) if meta_start else start
            df = self._normalize(self._fetch(code, is_etf,
                                             fetch_start.strftime("%Y-%m-%d"), end))
            completed = df if keep_today else df[df["date"].dt.date < today]
            if not completed.empty:
                completed.to_parquet(path, index=False)
                self._write_meta(code, fetch_start.strftime("%Y-%m-%d"))
            cached = completed if not completed.empty else df
        elif not self._is_fresh(path, now):
            # 过期：按一致基准全量重拉（修复前复权接缝 + 自愈盘中污染）
            try:
                df = self._normalize(self._fetch(code, is_etf, meta_start, end))
                completed = df if keep_today else df[df["date"].dt.date < today]
                if not completed.empty:
                    completed.to_parquet(path, index=False)
                cached = completed if not completed.empty else df
            except RuntimeError as e:
                # 数据源整体不可达：降级使用本地缓存，不抛异常（次日自愈）
                logger.warning("数据源不可达，%s 本次使用截至 %s 的缓存（%s）",
                               code, cached["date"].max().date(), e)

        # 盘中现场附加当日未完成 K 线（不写入缓存）
        if allow_intraday and not self._is_market_settled(now) \
                and pd.Timestamp(end) >= pd.Timestamp(today) \
                and (cached.empty or cached["date"].max().date() < today):
            try:
                tail_start = (cached["date"].max() - pd.Timedelta(days=20)).strftime("%Y-%m-%d") \
                    if not cached.empty else start.strftime("%Y-%m-%d")
                tail = self._normalize(self._fetch(code, is_etf, tail_start, end))
                new_rows = tail[tail["date"] > cached["date"].max()] if not cached.empty else tail
                if not new_rows.empty:
                    cached = pd.concat([cached, new_rows], ignore_index=True)
            except Exception as e:
                logger.warning("拉取 %s 盘中数据失败，使用缓存: %s", code, e)

        df = cached[(cached["date"] >= start) & (cached["date"] <= pd.Timestamp(end))]
        return df.reset_index(drop=True)
