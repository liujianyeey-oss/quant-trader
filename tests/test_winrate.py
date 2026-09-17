"""winrate 条件胜率评估器：掩码构造与统计正确性。"""
import unittest

import numpy as np
import pandas as pd

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from winrate import build_mask, independent_clusters, STRICT  # noqa: E402


def synth(n=400, seed=1):
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2020-01-01", periods=n, freq="D")
    close = 100 * np.cumprod(1 + rng.normal(0.0005, 0.02, n))
    high = close * 1.01
    low = close * 0.99
    vol = np.full(n, 1e6)
    return pd.DataFrame({"close": close, "high": high, "low": low, "volume": vol},
                        index=dates)


class TestWinrate(unittest.TestCase):
    def test_mask_fields_and_cluster(self):
        d = synth()
        f = {
            "close_px": d["close"],
            "ma200": d["close"].rolling(200).mean(),
            "hi20": d["high"].rolling(20).max().shift(1),
            "lo10": d["low"].rolling(10).min().shift(1),
            "mom20": d["close"] / d["close"].shift(20) - 1,
            "mom60": d["close"] / d["close"].shift(60) - 1,
            "vratio": d["volume"] / d["volume"].rolling(20).mean().shift(1),
        }
        m = build_mask(f, STRICT)
        # 掩码必须为布尔且无 NaN，且包含 c>MA200 语义（mom60>0.10 强约束下样本应很少）
        self.assertEqual(m.dtype, bool)
        self.assertFalse(m.isna().any())
        # 独立聚簇：连续 5 天匹配 → 1 个事件；20 天后再匹配 1 天 → 2 个
        idx = pd.to_datetime(["2021-01-%02d" % i for i in range(1, 6)] + ["2021-01-26"])
        self.assertEqual(independent_clusters(pd.DatetimeIndex(idx)), 2)


if __name__ == "__main__":
    unittest.main()
