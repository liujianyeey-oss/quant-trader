"""PIT 成分装配与校验：把分页 membership_YYYYMMDD_{300,500}_pN.csv 合成每期名单。

用法：python -m research_r11.pit_assemble
输出：pit/membership_YYYYMMDD.csv（code,name,delisted）+ 完整度校验报告。
"""
import glob
import os
import re

import pandas as pd

PIT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "pit")


def load_pages(date_s):
    """读取某期两个指数的全部分页，返回 {code: name}（合并 300+500）。"""
    out = {}
    pages = sorted(glob.glob(os.path.join(PIT, f"membership_{date_s}_*_p*.csv")))
    for p in pages:
        df = pd.read_csv(p)
        code_col = "指数成份代码" if "指数成份代码" in df.columns else df.columns[0]
        name_col = "指数成份简称" if "指数成份简称" in df.columns else df.columns[1]
        for _, r in df.iterrows():
            code = str(r[code_col]).strip()
            if re.fullmatch(r"\d{6}\.(SZ|SH)", code):
                out[code] = str(r[name_col]).strip()
    return out, len(pages)


def assemble_all():
    dates = sorted({os.path.basename(p).split("_")[1]
                    for p in glob.glob(os.path.join(PIT, "membership_*_p*.csv"))})
    rows = []
    for d in dates:
        members, npages = load_pages(d)
        n_300 = sum(1 for _ in glob.glob(os.path.join(PIT, f"membership_{d}_300_p*.csv")))
        n_500 = sum(1 for _ in glob.glob(os.path.join(PIT, f"membership_{d}_500_p*.csv")))
        expect_pages = 3 + 5  # 300 只 3 页 + 500 只 5 页
        n_delist = sum(1 for n in members.values() if "退市" in n)
        complete = (npages == expect_pages) and (len(members) >= 790)
        df = pd.DataFrame({"code": list(members.keys()),
                           "name": [members[c] for c in members],
                           "delisted": [1 if "退市" in members[c] else 0
                                        for c in members]})
        df["code6"] = df["code"].str[:6]
        df.to_csv(os.path.join(PIT, f"membership_{d}.csv"), index=False,
                  encoding="utf-8-sig")
        rows.append({"date": d, "pages": f"{npages}/{expect_pages}",
                     "members": len(members), "delisted": n_delist,
                     "complete": complete})
    rep = pd.DataFrame(rows)
    rep.to_csv(os.path.join(PIT, "assemble_report.csv"), index=False,
               encoding="utf-8-sig")
    print(rep.to_string(index=False))
    return rep


if __name__ == "__main__":
    assemble_all()
