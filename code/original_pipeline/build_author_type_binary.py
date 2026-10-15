#!/usr/bin/env python3
"""
build_author_type_binary.py —— 合成发帖者类型二分变量（官方 vs 其他）

依据日志 7.5b 的测量决策：六类细分信度不足（κ=0.473），只用二分
（官方媒体 + 政府与文博机构 = 官方；其余 = 其他），两人重叠 100 帖上 κ=0.965。
两人各编 226 帖、重叠 100 帖，合起来覆盖全部 352 帖；非重叠帖只有一人编码。
重叠帖上二分不一致且负责人尚未裁决的，official 留空并标 pending=1，
由 recheck_with_author_type.py 按两种取值各跑一遍。

输出 db/post_author_type_binary.csv
用法：python3 build_author_type_binary.py
"""
import os

import pandas as pd

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PACK = os.path.join(BASE, "人工编码任务包")
OFFICIAL = {"官方媒体", "政府与文博机构"}
KEY = ["platform", "pid"]


def read(who):
    d = pd.read_csv(os.path.join(PACK, f"B_发帖者类型_{who}.csv"), encoding="utf-8-sig", dtype=str)
    d["author_type"] = d.author_type.str.strip()
    return d[KEY + ["author_type"]]


def main():
    m = read("复核员A").merge(read("复核员B"), on=KEY, how="outer", suffixes=("_A", "_B"))
    # 补编：只在事件前有相关评论的 39 个帖子（填好后放在该路径，记在 A 列，n_coders=1）
    extra = os.path.join(PACK, "补编_事件前帖子39个", "B2_发帖者类型_事件前39帖.csv")
    if os.path.exists(extra):
        e = pd.read_csv(extra, encoding="utf-8-sig", dtype=str)
        e = e[e.author_type.fillna("").str.strip() != ""]
        m = pd.concat([m, e.assign(author_type_A=e.author_type.str.strip())[KEY + ["author_type_A"]]], ignore_index=True)
    off_a = m.author_type_A.map(lambda v: v in OFFICIAL if pd.notna(v) else None)
    off_b = m.author_type_B.map(lambda v: v in OFFICIAL if pd.notna(v) else None)
    m["n_coders"] = m.author_type_A.notna().astype(int) + m.author_type_B.notna().astype(int)
    m["official"] = off_a.where(off_a.notna(), off_b)
    conflict = off_a.notna() & off_b.notna() & (off_a != off_b)

    adj_path = os.path.join(PACK, "裁决表_发帖者类型_仅需裁决1条.csv")
    adj = pd.read_csv(adj_path, encoding="utf-8-sig", dtype=str).set_index(KEY)["裁决_author_type"].dropna()
    adj = adj[adj.str.strip() != ""].str.strip()
    m["pending"] = 0
    for i in m.index[conflict]:
        k = (m.at[i, "platform"], m.at[i, "pid"])
        if k in adj.index:
            m.at[i, "official"] = adj[k] in OFFICIAL
        else:
            m.at[i, "official"] = None
            m.at[i, "pending"] = 1
    m["official"] = m.official.map({True: 1, False: 0})
    out = os.path.join(BASE, "db", "post_author_type_binary.csv")
    m.to_csv(out, index=False, encoding="utf-8-sig")
    print(f"{len(m)} 帖：官方 {int((m.official == 1).sum())}，其他 {int((m.official == 0).sum())}，"
          f"待裁决 {int(m.pending.sum())}；两人都编 {int((m.n_coders == 2).sum())}，只一人编 {int((m.n_coders == 1).sum())}")
    print(m.groupby("platform").official.agg(["sum", "count"]).rename(columns={"sum": "官方", "count": "已定"}).to_string())
    print("→", out)


if __name__ == "__main__":
    main()
