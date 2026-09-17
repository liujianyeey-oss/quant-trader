"""R11 数据采集器：中证800 当前成分股日线（腾讯源直通，写 data.py 兼容缓存）。

说明：东财对本机持续封禁（8 次镜像重试必败，每股浪费 ~60s 退避），
本采集器直接用腾讯备用源（stock_zh_a_hist_tx，与 data.py 备用源同源），
缓存文件格式与 data.py 完全一致（parquet + meta.json），后续 load() 直接命中。
成分股为**当前名单回填历史**，含幸存者偏差（报告预声明）；ST 股按名称剔除。
"""
import os
import sys
import time

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data import DataLoader  # noqa: E402

PROJECT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
os.makedirs(RESULTS, exist_ok=True)
FETCH_START = "2017-06-01"


def get_universe():
    """中证800 = 沪深300 + 中证500 当前成分股；ST 按名称剔除。"""
    import akshare as ak
    rows = []
    for idx in ("000300", "000905"):
        df = ak.index_stock_cons(symbol=idx)
        for _, r in df.iterrows():
            rows.append({"code": str(r["品种代码"]).zfill(6),
                         "name": str(r["品种名称"]), "index": idx})
    uni = pd.DataFrame(rows).drop_duplicates("code")
    st = uni[uni["name"].str.contains("ST", case=False, na=False)]
    uni = uni[~uni["name"].str.contains("ST", case=False, na=False)].reset_index(drop=True)
    return uni, st


def tx_symbol(code):
    return ("sh" if code.startswith(("5", "6")) else "sz") + code


def collect(uni, start=FETCH_START, sleep_s=0.25, limit=None):
    """逐只拉取（缓存命中跳过）。返回报告 DataFrame。"""
    import akshare as ak
    loader = DataLoader(cache_dir=os.path.join(PROJECT, "data_cache"))
    end = pd.Timestamp.now().strftime("%Y-%m-%d")
    reports = []
    items = uni.to_dict("records")
    if limit:
        items = items[:limit]
    for i, row in enumerate(items, 1):
        code, name = row["code"], row["name"]
        path = loader._cache_path(code)
        meta = loader._read_meta(code)
        if os.path.exists(path) and meta.get("start") and \
                pd.to_datetime(meta["start"]) <= pd.Timestamp(start):
            reports.append({"code": code, "name": name, "status": "cached",
                            "rows": len(pd.read_parquet(path)), "error": ""})
            continue
        err = ""
        for attempt in range(3):
            try:
                df = ak.stock_zh_a_hist_tx(symbol=tx_symbol(code),
                                           start_date=start.replace("-", ""),
                                           end_date=end.replace("-", ""),
                                           adjust="qfq")
                if df is None or df.empty:
                    raise ValueError("空数据")
                df = loader._normalize(df)
                df.to_parquet(path, index=False)
                loader._write_meta(code, start)
                reports.append({"code": code, "name": name, "status": "fetched",
                                "rows": len(df), "error": ""})
                err = ""
                break
            except Exception as e:  # noqa: BLE001
                err = f"{type(e).__name__}: {str(e)[:80]}"
                time.sleep(2)
        if err:
            reports.append({"code": code, "name": name, "status": "failed",
                            "rows": 0, "error": err})
        if i % 50 == 0 or i == len(items):
            ok = sum(1 for r in reports if r["status"] != "failed")
            print(f"进度 {i}/{len(items)}，成功 {ok}，失败 {i - ok}", flush=True)
        time.sleep(sleep_s)
    rep = pd.DataFrame(reports)
    rep.to_csv(os.path.join(RESULTS, "fetch_report.csv"),
               index=False, encoding="utf-8-sig")
    return rep


if __name__ == "__main__":
    uni, st = get_universe()
    print(f"成分股合计 {len(uni) + len(st)}，剔除 ST {len(st)} 只，待拉取 {len(uni)} 只")
    uni.to_csv(os.path.join(RESULTS, "universe.csv"), index=False, encoding="utf-8-sig")
    st.to_csv(os.path.join(RESULTS, "st_excluded.csv"), index=False, encoding="utf-8-sig")
    collect(uni)
    print("采集完成")
