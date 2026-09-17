"""interim: 为 R11 PIT 差集（调出/退市股票）补腾讯行情，写 data_cache 兼容缓存。

只走腾讯免费源，绝不调用 Wind。失败（多半是退市股）记入 gaps 清单。
"""
import os
import sys
import time

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data import DataLoader  # noqa: E402

PROJECT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PIT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "pit")
FETCH_START = "2017-06-01"


def tx_symbol(code):
    return ("sh" if code.startswith(("5", "6")) else "sz") + code


def main():
    import akshare as ak
    dropped = pd.read_csv(os.path.join(PIT, "dropped.csv"), dtype={"code": str})
    dropped["code"] = dropped["code"].str.zfill(6)
    loader = DataLoader(cache_dir=os.path.join(PROJECT, "data_cache"))
    end = pd.Timestamp.now().strftime("%Y-%m-%d")
    reports = []
    for i, row in enumerate(dropped.to_dict("records"), 1):
        code, name = row["code"], row["name"]
        path = loader._cache_path(code)
        if os.path.exists(path):
            reports.append({"code": code, "name": name, "status": "cached",
                            "rows": len(pd.read_parquet(path)), "error": ""})
            continue
        err = ""
        for attempt in range(2):
            try:
                df = ak.stock_zh_a_hist_tx(symbol=tx_symbol(code),
                                           start_date=FETCH_START.replace("-", ""),
                                           end_date=end.replace("-", ""), adjust="qfq")
                if df is None or df.empty:
                    raise ValueError("空数据")
                df = loader._normalize(df)
                df.to_parquet(path, index=False)
                loader._write_meta(code, FETCH_START)
                reports.append({"code": code, "name": name, "status": "fetched",
                                "rows": len(df), "error": ""})
                err = ""
                break
            except Exception as e:  # noqa: BLE001
                err = f"{type(e).__name__}: {str(e)[:60]}"
                time.sleep(1.5)
        if err:
            reports.append({"code": code, "name": name, "status": "failed",
                            "rows": 0, "error": err})
        if i % 50 == 0 or i == len(dropped):
            ok = sum(1 for r in reports if r["status"] != "failed")
            print(f"进度 {i}/{len(dropped)}，成功 {ok}，失败 {i - ok}", flush=True)
    rep = pd.DataFrame(reports)
    rep.to_csv(os.path.join(PIT, "fetch_dropped.csv"), index=False, encoding="utf-8-sig")
    gaps = rep[rep["status"] == "failed"]
    gaps.to_csv(os.path.join(PIT, "gaps.csv"), index=False, encoding="utf-8-sig")
    print(f"完成：成功 {len(rep) - len(gaps)}，缺口 {len(gaps)}", flush=True)


if __name__ == "__main__":
    main()
