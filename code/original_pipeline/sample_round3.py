#!/usr/bin/env python3
"""
sample_round3.py —— 第 3 轮（教材 v6）人工编码任务文件

生成三类空白表，两名复核员各一份（内容相同、文件名不同）：
  1_新题150条      全新 150 条，排除此前任何一轮用过的评论（507 + 复测 150 + 练习 20）。
                   分层方式与 sample_for_review.py 相同：平台 × 是否相关 × 话语功能（luna full_v2），每层至少 2 条。
                   这一轮两人之间的 κ 是论文报告的信度。
  2_评论507条重做  原 507 条，清空答案并重新打乱顺序。只用于生成人工金标准，不再计算两人之间的 κ。
  3_复测150条重做  原复测 150 条，清空答案并重新打乱顺序。只作过程记录。

脚本只把模型标签用于分层，不输出任何模型标签。
用法：python3 sample_round3.py
"""
import csv
import os
import random
import sqlite3
from collections import defaultdict

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PACK = os.path.join(BASE, "人工编码任务包")
OUT = os.path.join(PACK, "第3轮_v6")
ARCH = os.path.join(PACK, "历史轮次_评论编码_仅存档")   # 前几轮的评论编码文件 2026-10-09 起移到这里
SEED = 20261009
N_NEW = 150
HEADER = ["platform", "comment_id", "帖子主题", "评论内容", "human_on_topic", "human_tone", "human_function", "备注"]


def read(path):
    with open(path, encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def write(name, rows):
    for who in ("复核员A", "复核员B"):
        d = os.path.join(OUT, f"发给{who}")
        os.makedirs(d, exist_ok=True)
        with open(os.path.join(d, f"{name}_{who}.csv"), "w", newline="", encoding="utf-8-sig") as f:
            w = csv.writer(f)
            w.writerow(HEADER)
            for r in rows:
                w.writerow([r["platform"], r["comment_id"], r["帖子主题"], r["评论内容"], "", "", "", ""])


def main():
    old507 = read(os.path.join(ARCH, "A1_评论编码_复核员A.csv"))
    old150 = read(os.path.join(ARCH, "复测150条_复核员A.csv"))
    practice = read(os.path.join(PACK, "培训练习_20条.csv"))
    blind120 = read(os.path.join(ARCH, "盲测120条_复核员A.csv"))
    used = {(r["platform"], r["comment_id"]) for r in old507 + old150 + practice + blind120}
    print(f"已用过的评论 {len(used)} 条（507 + 复测 150 + 练习 20；盲测 120 是 507 的子集）")

    conn = sqlite3.connect(os.path.join(BASE, "db", "heritage.sqlite"))
    conn.execute("ATTACH DATABASE ? AS ann", (os.path.join(BASE, "db", "annotations.sqlite"),))
    rows = conn.execute("""
        SELECT l.platform, l.comment_id, l.on_topic, l.function, c.text, p.text
        FROM ann.labels l
        JOIN comments c ON c.platform = l.platform AND c.comment_id = l.comment_id
        LEFT JOIN posts p ON p.platform = c.platform AND p.post_id = c.post_id
        WHERE l.codebook = 'v2' AND l.model = 'gpt-5.6-luna' AND l.run_tag = 'full_v2'
        ORDER BY l.platform, l.comment_id
    """).fetchall()
    pool = [r for r in rows if (r[0], str(r[1])) not in used and (r[4] or "").strip()]

    rng = random.Random(SEED)
    strata = defaultdict(list)
    for r in pool:
        strata[(r[0], r[2], r[3])].append(r)
    picked, rest = [], []
    for key in sorted(strata, key=str):
        items = strata[key]
        k = min(len(items), max(2, round(N_NEW * len(items) / len(pool))))
        rng.shuffle(items)
        picked += items[:k]
        rest += items[k:]
    # 每层至少 2 条会使总数偏离 150：多了随机去掉，少了从剩余里随机补
    rng.shuffle(picked)
    rng.shuffle(rest)
    picked = (picked + rest)[:N_NEW]
    rng.shuffle(picked)
    new = [{"platform": p, "comment_id": c, "帖子主题": (post or "")[:80].replace("\n", " "), "评论内容": text}
           for p, c, _, _, text, post in picked]
    assert len(new) == N_NEW and not {(r["platform"], str(r["comment_id"])) for r in new} & used
    write("1_新题150条", new)

    for name, src, seed in (("2_评论507条重做", old507, SEED + 1), ("3_复测150条重做", old150, SEED + 2)):
        src = list(src)
        random.Random(seed).shuffle(src)
        write(name, src)

    by_plat = defaultdict(int)
    for r in new:
        by_plat[r["platform"]] += 1
    print(f"新题 {len(new)} 条，各平台：{dict(by_plat)}（{len(strata)} 层）")
    print(f"输出目录：{OUT}")


if __name__ == "__main__":
    main()
