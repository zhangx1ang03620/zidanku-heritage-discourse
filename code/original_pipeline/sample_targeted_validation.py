#!/usr/bin/env python3
"""
sample_targeted_validation.py —— 定向验证："质疑异见"在各平台的测量准确度

背景（日志 7.10）：模型标为质疑异见的评论，人工只认可约四成；"平台 × 质疑异见"是主要发现之一，
但金标准里各平台只有几条，无法判断误判是否集中在某个平台。

抽样框：事件后、模型判相关的评论（主分析用的核心语料），排除此前任何一轮用过的评论。
每个平台两层：
  S   模型标为质疑异见        → 估计该平台的精确率
  N   模型标为其他三种语气    → 估计该平台的漏标率
任务表与前几轮完全相同（是否相关 / 语气 / 功能三列），两层混在一起打乱，不含任何模型标签，
编码者不知道哪些是 S。抽样框（含层标记）只存 db/targeted_validation_frame.csv，不外发。

用法：python3 sample_targeted_validation.py
"""
import csv
import glob
import os
import random
import sqlite3

import pandas as pd

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PACK = os.path.join(BASE, "人工编码任务包")
OUT = os.path.join(PACK, "第4轮_定向验证")
SEED = 20261010
N_S = {"zhihu": 40, "bilibili": 30, "douyin": 20, "xhs": 20, "weibo": 20}   # 不足则全取
N_N = 15
HEADER = ["platform", "comment_id", "帖子主题", "评论内容", "human_on_topic", "human_tone", "human_function", "备注"]


def main():
    used = set()
    for f in glob.glob(os.path.join(PACK, "第3轮_v6", "发给复核员A", "*.csv")) + [os.path.join(PACK, "培训练习_20条.csv")]:
        d = pd.read_csv(f, dtype=str, encoding="utf-8-sig")
        used |= set(zip(d.platform, d.comment_id))
    ds = pd.read_csv(os.path.join(os.path.dirname(BASE), "8.分析与图表", "data", "analysis_dataset.csv"),
                     dtype={"comment_id": str, "post_id": str})
    core = ds[(ds.on_topic == 1) & (ds.is_post_event == 1)].copy()
    core["stratum"] = core.tone.eq("质疑异见").map({True: "S", False: "N"})
    pool = core[[k not in used for k in zip(core.platform, core.comment_id)]]
    rng = random.Random(SEED)
    picked, rows = [], []
    for plat in sorted(N_S):
        for st, n in (("S", N_S[plat]), ("N", N_N)):
            ids = sorted(pool[(pool.platform == plat) & (pool.stratum == st)].comment_id)
            take = rng.sample(ids, min(n, len(ids)))
            pop = int(((core.platform == plat) & (core.stratum == st)).sum())
            rows.append({"platform": plat, "stratum": st, "population": pop, "available": len(ids), "sampled": len(take)})
            picked += [(plat, c, st) for c in take]
    print(pd.DataFrame(rows).to_string(index=False))
    rng.shuffle(picked)

    conn = sqlite3.connect(os.path.join(BASE, "db", "heritage.sqlite"))
    out = []
    for plat, cid, st in picked:
        text, post = conn.execute(
            "SELECT c.text, p.text FROM comments c LEFT JOIN posts p ON p.platform=c.platform AND p.post_id=c.post_id "
            "WHERE c.platform=? AND c.comment_id=?", (plat, cid)).fetchone()
        out.append({"platform": plat, "comment_id": cid, "stratum": st,
                    "帖子主题": (post or "")[:80].replace("\n", " "), "评论内容": text})
    pd.DataFrame(out)[["platform", "comment_id", "stratum"]].to_csv(
        os.path.join(BASE, "db", "targeted_validation_frame.csv"), index=False, encoding="utf-8-sig")
    pd.DataFrame(rows).to_csv(os.path.join(BASE, "db", "targeted_validation_strata.csv"), index=False, encoding="utf-8-sig")
    name = f"4_补充{len(out)}条"
    for who in ("复核员A", "复核员B"):
        d = os.path.join(OUT, f"发给{who}")
        os.makedirs(d, exist_ok=True)
        with open(os.path.join(d, f"{name}_{who}.csv"), "w", newline="", encoding="utf-8-sig") as f:
            w = csv.writer(f)
            w.writerow(HEADER)
            for r in out:
                w.writerow([r["platform"], r["comment_id"], r["帖子主题"], r["评论内容"], "", "", "", ""])
    print(f"共 {len(out)} 条 → {OUT}")


if __name__ == "__main__":
    main()
