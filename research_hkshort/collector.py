"""HKEX 每日卖空成交采集器（前向收集用）。

数据可得性调查结论（2026-09-17，逐条记录见 docs/R5_HK_SHORT_INTEREST.md）：
- HKEX 官网仅提供"当日"卖空成交文本报告（收盘后晚间发布，占位页会先显示
  "will be available after day close"）；历史 dayquot URL 已全部 404。
- web.archive.org 本机不可达；akshare 无港股卖空接口；GitHub 无存活镜像；
  AAStocks 沽空页为 JS 空壳。多年历史免费源：无。
因此本采集器只解决"从今晚起每天落盘"，历史回测标记为不可测。

用法：
  python -m research_hkshort.collector            # 抓当日（主板，收盘后）
  python -m research_hkshort.collector --file X   # 解析已落盘文件（自检）
  python -m research_hkshort.collector --demo     # 写入规范格式样例并解析（格式自检）
"""
import hashlib
import json
import os
import re
import sys
from datetime import datetime
from zoneinfo import ZoneInfo

import pandas as pd
import requests

BASE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(BASE, "cache")
RAW = os.path.join(CACHE, "raw")
os.makedirs(RAW, exist_ok=True)

URL_MAIN = ("https://www.hkex.com.hk/Market-Data/Statistics/Securities-Market/"
            "Short-Selling-Turnover-Today/Short-Selling-Turnover-(Main-Board)"
            "-up-to-day-close-today?sc_lang=en")
UA = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/128.0 Safari/537.36"}
PLACEHOLDER_PAT = re.compile(r"will be available after", re.I)

# 规范格式（多年稳定的 dayquot/ashtmain 文本）：
#   <code> <name> <short shares> <short amount>
#   TOTAL <shares> <amount>
ROW_PAT = re.compile(r"^\s*(\d{1,5})\s+(\S.*?)\s{2,}([\d,]+)\s+([\d,]+(?:\.\d+)?)\s*$")
TOTAL_PAT = re.compile(r"TOTAL\s+([\d,]+)\s+([\d,]+(?:\.\d+)?)", re.I)
DATE_PAT = re.compile(r"(\d{1,2}/\d{1,2}/\d{4})")


def is_placeholder(text):
    return bool(PLACEHOLDER_PAT.search(text))


def parse_report(text, report_date=None):
    """解析 HKEX 卖空成交文本报告。返回 (detail_df, total_dict)。"""
    if is_placeholder(text):
        raise ValueError("占位页：当日报告尚未发布（will be available after day close）")
    rows, total = [], None
    for line in text.splitlines():
        m = ROW_PAT.match(line)
        if m and not line.strip().upper().startswith("TOTAL"):
            code, name, shares, amount = m.groups()
            rows.append({"code": code.zfill(5), "name": name.strip(),
                         "short_shares": int(shares.replace(",", "")),
                         "short_amount": float(amount.replace(",", ""))})
            continue
        t = TOTAL_PAT.search(line)
        if t:
            total = {"short_shares": int(t.group(1).replace(",", "")),
                     "short_amount_hkd": float(t.group(2).replace(",", ""))}
    if total is None:
        raise ValueError("未找到 TOTAL 行：报告格式变化或内容不完整")
    if report_date is None:
        dm = DATE_PAT.search(text)
        report_date = (datetime.strptime(dm.group(1), "%d/%m/%Y").strftime("%Y-%m-%d")
                       if dm else None)
    detail = pd.DataFrame(rows)
    return detail, {**total, "date": report_date}


def fetch_today(out_dir=RAW):
    """抓当日主板卖空报告。发布前返回占位页 → 抛 ValueError（cron 稍后重试）。"""
    r = requests.get(URL_MAIN, headers=UA, timeout=25)
    r.raise_for_status()
    text = r.text
    if is_placeholder(text):
        raise ValueError("当日报告尚未发布（HKEX 通常收盘后晚间发布，稍后重试）")
    sha = hashlib.sha256(text.encode()).hexdigest()
    detail, total = parse_report(text)
    date = total["date"] or datetime.now(ZoneInfo("Asia/Hong_Kong")).strftime("%Y-%m-%d")
    raw_path = os.path.join(out_dir, f"{date}.txt")
    with open(raw_path, "w", encoding="utf-8") as f:
        f.write(text)
    detail.to_csv(os.path.join(CACHE, f"short_detail_{date}.csv"), index=False)
    meta_path = os.path.join(CACHE, "market_short.csv")
    row = pd.DataFrame([{**total, "date": date, "sha256": sha[:16],
                         "source": URL_MAIN}])
    row.to_csv(meta_path, mode="a", header=not os.path.exists(meta_path), index=False)
    return date, total, len(detail)


DEMO_SAMPLE = """Short Selling Turnover (Main Board) up to day close today

DATE: 16/09/2025 (TUESDAY)

STOCK CODE  STOCK NAME              SHORT SELLING TURNOVER
                                      (SHARES)          (HK$)
----------------------------------------------------------------------
00001       CKH HOLDINGS            2,345,000         123,456,789.00
00700       TENCENT                12,345,678       4,567,890,123.00
09988       BABA-SW                 8,765,432       1,234,567,890.00
----------------------------------------------------------------------
TOTAL                              987,654,321       12,345,678,900.00
"""


def _demo():
    detail, total = parse_report(DEMO_SAMPLE)
    assert total["short_shares"] == 987654321 and total["short_amount_hkd"] == 12345678900.0
    assert len(detail) == 3 and detail.iloc[1]["code"] == "00700"
    print("格式自检通过:", total, f"明细 {len(detail)} 行")


if __name__ == "__main__":
    if "--demo" in sys.argv:
        _demo()
    elif "--file" in sys.argv:
        p = sys.argv[sys.argv.index("--file") + 1]
        d, t = parse_report(open(p, encoding="utf-8").read())
        print(t, f"明细 {len(d)} 行")
    else:
        try:
            date, total, n = fetch_today()
            print(f"已落盘 {date}: 大市卖空成交额 HK${total['short_amount_hkd']:,.0f}，明细 {n} 只")
        except ValueError as e:
            print(f"未采集：{e}")
            sys.exit(2)
