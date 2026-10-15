#!/usr/bin/env python3
"""
sample_for_review.py —— 导出人工盲审表（编码本 v2）

两份表，都是盲审（不显示模型或规则的判断，避免复核者跟着答案走的锚定效应）：

  1. human_review_comments.csv  评论编码复核：是否相关 / 情感基调 / 话语功能
     按 (平台 × 是否相关 × 话语功能) 分层抽样，默认 500 条。
     方案原定抽 10–15%，全量约 1.9 万条会有近 2,000 条，工作量过大；
     估计 Kappa 约 500 条即可使 95% 置信区间在 ±0.06 左右，故默认 500，可用 --n 调整。

  2. human_review_posts.csv     帖子相关性复核：验证关键词规则本身（规则判相关 / 不相关各一半）
     规则的"精确度"用规则自己算会有循环论证，必须靠人工判断得出真实的精确率与漏判率。

填完后运行 kappa.py 计算一致性。
"""
import argparse
import csv
import os
import random
import sqlite3
from collections import defaultdict

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_PATH = os.path.join(BASE_DIR, "db", "heritage.sqlite")
ANN_DB_PATH = os.path.join(BASE_DIR, "db", "annotations.sqlite")

GUIDE = """
【评论表】每行填三列（与 8.分析与图表/prompts/codebook_v2_prompt.txt 的定义一致，请先读一遍）：
  human_on_topic : 是 / 否   （是否实质性讨论子弹库帛书、它的流失与归还、文物追索）
  human_tone     : 喜悦自豪 | 愤慨痛惜 | 中性理性 | 质疑异见        （判"否"时留空）
  human_function : 情感表态 | 追索诉求与国家能力 | 文物知识与归还细节 | 历史与文明论战 | 处置过程的信任与怀疑（判"否"时留空）
【帖子表】每行填一列：
  human_relevant : 是 / 否   （这个帖子是否在讲长沙子弹库帛书，包括它的流失、归还、内容）
建议两位组员独立各填一份，第三人裁决分歧——这样还能报告人工之间的一致性。"""


def stratified(rows, key, n, rng):
    strata = defaultdict(list)
    for r in rows:
        strata[key(r)].append(r)
    total = len(rows)
    out = []
    for items in strata.values():
        k = min(len(items), max(2, round(n * len(items) / total)))  # 每层至少 2 条，保证稀有类别有样本
        out.extend(rng.sample(items, k))
    rng.shuffle(out)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=500, help="评论复核样本量")
    ap.add_argument("--n-posts", type=int, default=200, help="帖子复核样本量（规则判相关/不相关各一半）")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    rng = random.Random(args.seed)
    out_dir = os.path.join(BASE_DIR, "db")

    conn = sqlite3.connect(DB_PATH)
    conn.execute("ATTACH DATABASE ? AS ann", (ANN_DB_PATH,))
    rows = conn.execute("""
        SELECT l.platform, l.comment_id, l.on_topic, l.function, c.text, p.text
        FROM ann.labels l
        JOIN comments c ON c.platform = l.platform AND c.comment_id = l.comment_id
        LEFT JOIN posts p ON p.platform = c.platform AND p.post_id = c.post_id
        WHERE l.codebook = 'v2' AND l.model = 'gpt-5.6-luna' AND l.run_tag = 'full_v2'
    """).fetchall()
    sample = stratified(rows, lambda r: (r[0], r[2], r[3]), args.n, rng)
    path = os.path.join(out_dir, "human_review_comments.csv")
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["platform", "comment_id", "帖子主题", "评论内容", "human_on_topic", "human_tone", "human_function", "备注"])
        for plat, cid, _, _, text, post in sample:
            w.writerow([plat, cid, (post or "")[:80].replace("\n", " "), text, "", "", "", ""])
    print(f"评论盲审表：{len(sample)} 条 → {path}")

    posts = conn.execute("SELECT platform, post_id, is_relevant, text FROM posts WHERE text IS NOT NULL AND TRIM(text) != ''").fetchall()
    half = args.n_posts // 2
    pick = []
    for flag in (1, 0):
        pool = [p for p in posts if p[2] == flag]
        pick += stratified(pool, lambda r: r[0], half, rng)[:half]
    rng.shuffle(pick)
    path = os.path.join(out_dir, "human_review_posts.csv")
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["platform", "post_id", "帖子内容（前300字）", "human_relevant", "备注"])
        for plat, pid, _, text in pick:
            w.writerow([plat, pid, (text or "")[:300].replace("\n", " "), "", ""])
    print(f"帖子盲审表：{len(pick)} 条 → {path}")
    print(GUIDE)


if __name__ == "__main__":
    main()
