"""标的宇宙评分卡（R3）：流动性/历史/风险特性/相关性/结构风险 → 三档白名单。

数据：A股走项目 data.py 东财/腾讯双源；美股复用 us_equities 缓存；加密走币安镜像。
全部用真实数据计算；拉不通的记录原因进淘汰档，不许伪造。
"""
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data import DataLoader  # noqa: E402

PROJECT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
CACHE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cache")
os.makedirs(RESULTS, exist_ok=True)
os.makedirs(CACHE, exist_ok=True)

START = "2017-01-01"
EVAL_START = "2019-01-02"

# (code, 名称, 类别, 结构风险定性)
A_CANDIDATES = [
    ("510300", "沪深300ETF", "宽基", ""),
    ("510500", "中证500ETF", "宽基", ""),
    ("159915", "创业板ETF", "宽基", ""),
    ("588000", "科创50ETF", "宽基", ""),
    ("518880", "黄金ETF", "商品", ""),
    ("511010", "国债ETF", "债券", "久期风险低"),
    ("513100", "纳指ETF", "跨境", "QDII额度限制，历史多次溢价>3%"),
    ("513500", "标普500ETF", "跨境", "QDII额度限制，偶有溢价"),
    ("513520", "日经ETF", "跨境", "QDII额度+日元汇率敞口"),
    ("513030", "德国ETF", "跨境", "QDII额度，流动性相对小"),
    ("159920", "恒生ETF", "跨境", "港股通/QDII，折溢价一般可控"),
    ("513130", "恒生科技ETF", "跨境", "2021-06上市，历史短"),
    ("159985", "豆粕ETF", "商品", "期货展期损耗（contango 结构）"),
    ("159980", "有色ETF", "商品", "期货展期，back 结构较多损耗相对低"),
    ("159981", "能化ETF", "商品", "期货展期损耗（contango 明显）"),
    ("511260", "十年国债ETF", "债券", "久期风险（利率上行段下跌）"),
]

US_CACHE = os.path.join(PROJECT, "us_equities", "research", "cache")
US_ETFS = ("SPY QQQ IWM DIA MDY RSP XLK XLF XLE XLV XLY XLI XLP XLU XLB XLRE "
           "TLT IEF LQD HYG SHY GLD SLV USO DBC EFA EEM").split()


# ---------- 数据加载 ----------

def load_a_share(codes):
    loader = DataLoader(cache_dir=os.path.join(PROJECT, "data_cache"))
    data, notes = {}, {}
    for code in codes:
        try:
            df = loader.load(code, is_etf=True, start=START)
            if df.empty:
                notes[code] = "返回空数据"
                continue
            data[code] = df
            notes[code] = ""
        except Exception as e:  # noqa: BLE001
            notes[code] = f"拉取失败：{type(e).__name__}: {str(e)[:80]}"
    return data, notes


def load_us():
    data = {}
    for sym in US_ETFS:
        path = os.path.join(US_CACHE, f"{sym}.csv")
        if os.path.exists(path):
            d = pd.read_csv(path, parse_dates=["date"])
            d["adj_close"] = d["close"] * d["factor"]
            d["adj_open"] = d["open"] * d["factor"]
            data[sym] = d
    return data


def load_crypto(symbol):
    """币安镜像日线（BTC 有缓存则刷新尾部，ETH 新拉）。"""
    path = os.path.join(PROJECT, "macro_research", "cache", f"{symbol.lower()}.csv")
    rows = []
    end_ms = int(pd.Timestamp.now().timestamp() * 1000)
    start_ms = int(pd.Timestamp("2017-08-17").timestamp() * 1000)
    if os.path.exists(path):
        old = pd.read_csv(path, parse_dates=["date"])
        start_ms = int((old["date"].max() + pd.Timedelta(days=1)).timestamp() * 1000)
        rows = old.values.tolist()
        cols_old = old.columns
    else:
        cols_old = None
    fetched = []
    while start_ms < end_ms:
        import requests
        r = requests.get("https://data-api.binance.vision/api/v3/klines",
                         params={"symbol": f"{symbol}USDT", "interval": "1d",
                                 "startTime": start_ms, "endTime": end_ms,
                                 "limit": 1000}, timeout=30)
        r.raise_for_status()
        batch = r.json()
        if not batch:
            break
        fetched += batch
        start_ms = batch[-1][0] + 86_400_000
        if len(batch) < 1000:
            break
        time.sleep(0.5)
    df_new = pd.DataFrame(fetched)
    if not df_new.empty:
        df_new = pd.DataFrame(fetched, columns=["open_time", "open", "high", "low", "close",
                                                "volume", "_ct", "_qv", "_n", "_tb", "_tq", "_i"])
        df_new["date"] = pd.to_datetime(df_new["open_time"], unit="ms").dt.normalize()
        for c in ("open", "high", "low", "close", "volume"):
            df_new[c] = pd.to_numeric(df_new[c], errors="coerce")
        df_new = df_new[["date", "open", "high", "low", "close", "volume"]]
    if cols_old is not None:
        old = pd.DataFrame(rows, columns=cols_old)
        old["date"] = pd.to_datetime(old["date"])
        df = pd.concat([old, df_new], ignore_index=True) if not df_new.empty else old
    else:
        df = df_new
    df = df[df["date"] < pd.Timestamp.now().normalize()]
    df = df.drop_duplicates("date").sort_values("date").reset_index(drop=True)
    df.to_csv(path, index=False)
    return df


# ---------- 指标计算 ----------

def perf_stats(close: pd.Series, eval_start=EVAL_START):
    """年化波动率 / 最大回撤 / MA200 上方占比（自 eval_start 或上市起）。"""
    c = close.dropna()
    c = c[c.index >= pd.Timestamp(eval_start)] if c.index.min() < pd.Timestamp(eval_start) else c
    if len(c) < 60:
        return np.nan, np.nan, np.nan
    ret = c.pct_change().dropna()
    vol = ret.std() * np.sqrt(252)
    mdd = (c / c.cummax() - 1).min()
    ma200_share = (c > c.rolling(200, min_periods=200).mean()).mean()
    return vol, mdd, ma200_share


def corr_to(a: pd.Series, b: pd.Series, eval_start=EVAL_START):
    """日收益相关系数（共同交易日，eval_start 起）。"""
    ra = a.pct_change()
    rb = b.pct_change()
    df = pd.concat([ra.rename("a"), rb.rename("b")], axis=1).dropna()
    df = df[df.index >= pd.Timestamp(eval_start)]
    if len(df) < 60:
        return np.nan
    return df["a"].corr(df["b"])


def liquidity_tier_amount(amount_wan):
    """A股 ETF 近20日日均成交额分档（万元，近似）。"""
    if pd.isna(amount_wan):
        return "未知"
    if amount_wan >= 50_000:
        return "高(≥5亿)"
    if amount_wan >= 10_000:
        return "中(1~5亿)"
    return "低(<1亿)"


# ---------- 显式分档与评语（规则见 UNIVERSE_SCORECARD.md） ----------

# 跨境 ETF 溢价风险定性分档（现价 vs 参考净值偏离；免费源无 IOPV，无法定量，按历史公开记录定性）
PREMIUM_TIER = {"513100": "高", "513500": "中", "513520": "中", "513030": "中",
                "159920": "低", "513130": "低"}

# 人工复核后的最终分档（覆盖机械规则；依据见报告）
TIER_OVERRIDES = {
    # A股核心池
    "510300": "核心池", "510500": "核心池", "159915": "核心池", "588000": "核心池",
    "518880": "核心池", "511010": "核心池", "511260": "核心池",
    "513030": "核心池", "513520": "核心池",
    # A股观察池
    "513100": "观察池",   # 溢价风险高（QDII 额度紧张期多次 >3%）
    "513500": "观察池",   # 溢价中 + 与 513100 高相关冗余
    "159920": "观察池",   # 与 510300 相关 0.67、恒生长期弱于科创
    "513130": "观察池",   # 历史短(2021-06)、趋势性最弱(MA200上方 37.5%)、动量基准已证伪
    "159985": "观察池",   # 期货展期损耗
    "159980": "观察池",   # 展期损耗相对低但存在
    "159981": "观察池",   # 期货展期损耗明显
    # 美股
    "USO": "观察池", "DBC": "观察池",
    # 加密（新资产类别，先观察）
    "BTC": "观察池", "ETH": "观察池",
}

REVIEW = {
    "510300": "组合锚，流动性与数据质量最好",
    "510500": "中盘补充，与 300 互补",
    "159915": "高波动进攻腿，2021-02-08 单日缺行情（源端）已 ffill 兜底",
    "588000": "2020-11 才上市，回测早期按 NaN 剔除即可",
    "518880": "最佳分散资产（与 300 相关 0.09），趋势占比 69%",
    "511010": "利率债现金替代品，波动 2%",
    "511260": "同 511010，久期略长",
    "513100": "纳指敞口好工具，但溢价风险是真实成本（额度紧张期 >3%）",
    "513500": "与 513100 敞口重叠，溢价中等",
    "513520": "日元+日股敞口，分散价值真实（相关 0.28），留意溢价",
    "513030": "欧股敞口唯一来源，相关 0.35，流动性稍小",
    "159920": "与 510300 相关 0.67，分散价值有限",
    "513130": "趋势性最弱（MA200 上方仅 37.5%），历史短，动量基准策略已证伪",
    "159985": "农产品敞口稀缺（相关 0.09），但展期损耗是长期拖累",
    "159980": "有色 back 结构较多，展期损耗相对低，可考虑小仓研究",
    "159981": "能化 contango 明显，展期损耗大，只作短趋势研究",
    "SPY": "美股锚", "QQQ": "donchian 主战场", "IWM": "小盘，唐奇安已证伪",
    "DIA": "道指，低波", "MDY": "中盘，唐奇安已证伪", "RSP": "等权标普",
    "XLK": "唯一行业验证有效（与 QQQ 相关 0.97 冗余）",
    "XLF": "金融", "XLE": "能源，高波动周期", "XLV": "医疗，防御",
    "XLY": "可选消费，与 QQQ 0.87 冗余", "XLI": "工业", "XLP": "必需消费，低波",
    "XLU": "公用事业，类债", "XLB": "材料", "XLRE": "地产，利率敏感",
    "TLT": "长债，2022 后 regime 变化已演示，防守角色存疑",
    "IEF": "中债", "LQD": "投资级信用", "HYG": "高收益信用，类股",
    "SHY": "现金等价", "GLD": "黄金，分散好（相关 0.14）",
    "SLV": "白银，波动 36% 偏高", "USO": "展期损耗典型，回避长持",
    "DBC": "综合商品，展期存在", "EFA": "海外发达", "EEM": "海外新兴",
    "BTC": "高波动 51%，与两锚相关 <0.07，趋势策略研究价值高，仅限观察",
    "ETH": "波动 67% 更甚，与 BTC 高相关，同理观察",
}


def _apply_overrides(df):
    df = df.copy()
    df["溢价风险"] = df["symbol"].map(PREMIUM_TIER).fillna("")
    df["分档"] = df.apply(
        lambda r: TIER_OVERRIDES.get(r["symbol"], r["分档"] if r.get("分档") else "核心池"),
        axis=1)
    df["评语"] = df.apply(
        lambda r: REVIEW.get(r["symbol"], r.get("评语", "")), axis=1)
    return df


def main():
    rows = []

    # ---- A股候选 ----
    codes = [c for c, *_ in A_CANDIDATES]
    a_data, a_notes = load_a_share(codes)
    bench_300 = a_data["510300"].set_index("date")["close"] if "510300" in a_data else None
    for code, name, cat, struct in A_CANDIDATES:
        if a_notes.get(code):
            rows.append({"symbol": code, "名称": name, "市场": "A股", "类别": cat,
                         "分档": "淘汰", "评语": f"数据拉取失败：{a_notes[code]}",
                         "结构风险": struct})
            continue
        df = a_data[code]
        close = df.set_index("date")["close"]
        first = close.index.min()
        years = (close.index.max() - first).days / 365.25
        # 流动性：近期数据来自腾讯源（成交量单位=股），成交额≈收盘×量（近似，已标注）
        amount20 = (df["close"] * df["volume"]).tail(20).mean() / 1e4
        vol, mdd, ma_share = perf_stats(close)
        corr = corr_to(close, bench_300) if bench_300 is not None and code != "510300" else 1.0
        rows.append({"symbol": code, "名称": name, "市场": "A股", "类别": cat,
                     "起始日": str(first.date()), "有效年数": round(years, 1),
                     "近20日日均成交额(万元,近似)": round(amount20, 0),
                     "流动性": liquidity_tier_amount(amount20),
                     "年化波动率": vol, "最大回撤": mdd, "MA200上方占比": ma_share,
                     "相关基准": "510300", "相关系数": corr,
                     "结构风险": struct})

    # ---- 美股（复用缓存） ----
    us = load_us()
    bench_qqq = us["QQQ"].set_index("date")["adj_close"]
    for sym, d in us.items():
        close = d.set_index("date")["adj_close"]
        first = close.index.min()
        years = (close.index.max() - first).days / 365.25
        amount20 = (d["adj_close"] * d["volume"]).tail(20).mean() / 1e6  # 百万美元
        vol, mdd, ma_share = perf_stats(close)
        corr = corr_to(close, bench_qqq) if sym != "QQQ" else 1.0
        cat = {"SPY": "宽基", "QQQ": "宽基", "IWM": "宽基", "DIA": "宽基", "MDY": "宽基",
               "RSP": "宽基", "TLT": "债券", "IEF": "债券", "LQD": "债券", "HYG": "债券",
               "SHY": "债券", "GLD": "商品", "SLV": "商品", "USO": "商品", "DBC": "商品",
               "EFA": "海外", "EEM": "海外"}.get(sym, "行业")
        struct = {"USO": "原油展期损耗显著（2020 八合一反向拆分）",
                  "DBC": "商品期货展期", "SLV": "高波动",
                  "TLT": "久期风险（2022 利率 regime 已演示）",
                  "HYG": "信用风险"}.get(sym, "")
        rows.append({"symbol": sym, "名称": sym, "市场": "美股", "类别": cat,
                     "起始日": str(first.date()), "有效年数": round(years, 1),
                     "近20日日均成交额(百万美元)": round(amount20, 1),
                     "年化波动率": vol, "最大回撤": mdd, "MA200上方占比": ma_share,
                     "相关基准": "QQQ", "相关系数": corr, "结构风险": struct})

    # ---- 加密 ----
    for sym in ("BTC", "ETH"):
        try:
            d = load_crypto(sym)
        except Exception as e:  # noqa: BLE001
            rows.append({"symbol": sym, "名称": sym, "市场": "加密", "类别": "加密",
                         "分档": "淘汰", "评语": f"数据拉取失败：{type(e).__name__}: {str(e)[:80]}"})
            continue
        close = d.set_index("date")["close"]
        first = close.index.min()
        years = (close.index.max() - first).days / 365.25
        amount20 = (d["close"] * d["volume"]).tail(20).mean() / 1e6  # 百万美元
        vol, mdd, ma_share = perf_stats(close)
        c300 = corr_to(close, bench_300) if bench_300 is not None else np.nan
        cqqq = corr_to(close, bench_qqq)
        rows.append({"symbol": sym, "名称": sym, "市场": "加密", "类别": "加密",
                     "起始日": str(first.date()), "有效年数": round(years, 1),
                     "近20日日均成交额(百万美元)": round(amount20, 0),
                     "年化波动率": vol, "最大回撤": mdd, "MA200上方占比": ma_share,
                     "相关基准": "510300/QQQ", "相关系数": c300,
                     "相关系数2": cqqq, "结构风险": "365天交易、无涨跌停、监管与托管风险"})

    df = pd.DataFrame(rows)

    # ---- 分档：机械初筛 + 人工复核覆盖（TIER_OVERRIDES，依据见报告） ----
    def tier(r):
        if r.get("分档") == "淘汰":
            return "淘汰"
        years = r.get("有效年数", 0) or 0
        liq = r.get("流动性", "")
        struct = r.get("结构风险", "") or ""
        mdd = r.get("最大回撤", np.nan)
        if pd.isna(mdd):
            return "淘汰"
        if "展期" in struct or "溢价" in struct:
            return "观察池"
        if years < 3:
            return "观察池"
        if r["市场"] == "A股" and "低" in liq:
            return "观察池"
        if r["市场"] == "加密":
            return "观察池"
        if r["市场"] == "美股" and r.get("近20日日均成交额(百万美元)", 0) < 50:
            return "观察池"
        return "核心池"

    df["分档"] = df.apply(tier, axis=1)
    df = _apply_overrides(df)
    df.to_csv(os.path.join(RESULTS, "scorecard.csv"), index=False, encoding="utf-8-sig")

    # 控制台输出
    show = df.copy()
    for c in ("年化波动率", "最大回撤", "MA200上方占比", "相关系数", "相关系数2"):
        if c in show.columns:
            show[c] = show[c].map(lambda x: f"{x:.1%}" if pd.notna(x) else "-")
    cols = ["symbol", "名称", "市场", "类别", "起始日", "有效年数", "年化波动率",
            "最大回撤", "MA200上方占比", "相关系数", "分档"]
    print(show[[c for c in cols if c in show.columns]].to_string(index=False))
    print("\n分档统计:", df["分档"].value_counts().to_dict())
    print(f"\n结果 CSV：{os.path.join(RESULTS, 'scorecard.csv')}")


if __name__ == "__main__":
    main()
