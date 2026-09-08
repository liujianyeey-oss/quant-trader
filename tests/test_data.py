"""数据层缓存兜底测试：数据源不可达时降级使用缓存，不抛异常。

运行：.venv/bin/python -m unittest discover -s tests -v
"""
import os
import tempfile
import unittest
from datetime import datetime, timedelta
from unittest import mock

import pandas as pd

from data import DataLoader


def fake_history():
    dates = pd.bdate_range("2020-01-01", periods=30)
    return pd.DataFrame({"date": dates, "open": 10.0, "close": 10.0,
                         "high": 10.0, "low": 10.0, "volume": 1})


class DataFallbackTests(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.mkdtemp()

    def _seed_cache(self, loader, code="510300"):
        """写入一份昨日修改时间的缓存 + meta（模拟此前正常抓取的存量）。"""
        df = fake_history()
        path = loader._cache_path(code)
        df.to_parquet(path, index=False)
        loader._write_meta(code, "2019-01-01")
        yesterday = (datetime.now() - timedelta(days=1)).timestamp()
        os.utime(path, (yesterday, yesterday))  # 让缓存"过期"，触发全量重拉
        return df

    def test_unreachable_source_falls_back_to_cache(self):
        """全量重拉彻底失败且本地有缓存：降级用缓存并 WARN，不抛异常。"""
        loader = DataLoader(cache_dir=self.dir)
        cached = self._seed_cache(loader)
        with mock.patch.object(DataLoader, "_fetch",
                               side_effect=RuntimeError("模拟数据源不可达")):
            with self.assertLogs("data", level="WARNING") as logs:
                df = loader.load("510300", is_etf=True, start="2019-01-01")
        self.assertEqual(len(df), len(cached))
        self.assertEqual(df["date"].max(), cached["date"].max())
        self.assertTrue(any("数据源不可达" in m and "缓存" in m
                            for m in logs.output))

    def test_unreachable_source_without_cache_raises(self):
        """缓存也不存在时仍然 raise（无兜底可用）。"""
        loader = DataLoader(cache_dir=self.dir)
        with mock.patch.object(DataLoader, "_fetch",
                               side_effect=RuntimeError("模拟数据源不可达")):
            with self.assertRaises(RuntimeError):
                loader.load("510300", is_etf=True, start="2019-01-01")


class TxFallbackTests(unittest.TestCase):
    """双源容错：东财全挂自动切腾讯备用源。"""

    def tx_fake(self):
        dates = pd.bdate_range("2020-01-01", periods=5)
        return pd.DataFrame({"date": dates, "open": 10.0, "close": 10.0,
                             "high": 10.0, "low": 10.0, "volume": 100,
                             "turnover": 1.0, "amount": 1000.0})

    def test_em_down_falls_back_to_tx(self):
        """东财失败 → 腾讯源返回英文列名数据并被正确归一化；sh/sz 前缀正确。"""
        loader = DataLoader(cache_dir=tempfile.mkdtemp())
        calls = []

        def fake_tx(symbol, start_date, end_date, adjust, timeout=None):
            calls.append((symbol, adjust))
            return self.tx_fake()

        with mock.patch.object(DataLoader, "_fetch_em",
                               side_effect=RuntimeError("东财全挂")), \
                mock.patch("data.ak.stock_zh_a_hist_tx", side_effect=fake_tx):
            df = loader.load("510300", is_etf=True, start="2019-01-01")
        self.assertEqual(calls, [("sh510300", "qfq")])  # ETF 5 开头 → sh
        self.assertEqual(list(df.columns),
                         ["date", "open", "high", "low", "close", "volume"])
        self.assertEqual(len(df), 5)
        # 缓存已落盘（腾讯数据同样进缓存）
        self.assertTrue(os.path.exists(loader._cache_path("510300")))

    def test_sz_prefix(self):
        """15/0/3 开头 → sz 前缀。"""
        loader = DataLoader(cache_dir=tempfile.mkdtemp())
        self.assertEqual(loader._tx_symbol("159915"), "sz159915")
        self.assertEqual(loader._tx_symbol("510300"), "sh510300")
        self.assertEqual(loader._tx_symbol("600519"), "sh600519")
        self.assertEqual(loader._tx_symbol("000001"), "sz000001")

    def test_both_sources_down_raises(self):
        """两个源都失败：无缓存时 raise。"""
        loader = DataLoader(cache_dir=tempfile.mkdtemp())
        with mock.patch.object(DataLoader, "_fetch_em",
                               side_effect=RuntimeError("东财全挂")), \
                mock.patch("data.ak.stock_zh_a_hist_tx",
                           side_effect=RuntimeError("腾讯也挂")):
            with self.assertRaises(RuntimeError):
                loader.load("510300", is_etf=True, start="2019-01-01")


if __name__ == "__main__":
    unittest.main(verbosity=2)
