#!/usr/bin/env python3
"""
kappa.py —— 人工盲审 vs 模型 / 规则 的一致性（编码本 v2）

评论表：on_topic 在全部样本上比较；tone / function 只在人和模型都判"相关"的评论上比较
       （否则双方都是"不适用"会被算作一致，抬高 Kappa）。
       同时报告以人工为基准时模型"是否相关"的精确率与召回率。
帖子表：以人工为基准，报告关键词规则的精确率（规则判相关里真相关的比例）和漏判率（规则判不相关里其实相关的比例）。

Kappa 解读（Landis & Koch）：0.41–0.60 中等，0.61–0.80 高度，>0.80 几乎完全一致；社科文本标注一般要求 ≥ 0.60。
用法：python3 kappa.py [--comments db/human_review_comments.csv] [--posts db/human_review_posts.csv]
"""
import argparse
import csv
import os
import sqlite3

from sklearn.metrics import cohen_kappa_score, confusion_matrix

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
YES, NO = ("是", "y", "yes", "1", "true", "相关"), ("否", "n", "no", "0", "false", "不相关")


def to_bool(v):
    v = (v or "").strip().lower()
    return 1 if v in YES else 0 if v in NO else None


def interpret(k):
    return "几乎完全一致" if k > 0.8 else "高度一致" if k > 0.6 else "中等" if k > 0.4 else "一般" if k > 0.2 else "差"


def report(title, a, b, labels=None):
    if len(a) < 2:
        print(f"\n{title}：样本不足")
        return
    k = cohen_kappa_score(a, b)
    agree = sum(x == y for x, y in zip(a, b)) / len(a)
    print(f"\n{title}：n={len(a)}  一致率={agree:.1%}  Kappa={k:.3f}（{interpret(k)}）")
    labels = labels or sorted(set(a) | set(b))
    cm = confusion_matrix(a, b, labels=labels)
    print("  混淆矩阵（行=模型/规则，列=人工）：", labels)
    for l, row in zip(labels, cm):
        print(f"    {str(l):<14}", " ".join(f"{v:>5}" for v in row))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--comments", default=os.path.join(BASE_DIR, "db", "human_review_comments.csv"))
    ap.add_argument("--posts", default=os.path.join(BASE_DIR, "db", "human_review_posts.csv"))
    args = ap.parse_args()
    conn = sqlite3.connect(os.path.join(BASE_DIR, "db", "heritage.sqlite"))
    conn.execute("ATTACH DATABASE ? AS ann", (os.path.join(BASE_DIR, "db", "annotations.sqlite"),))

    if os.path.exists(args.comments):
        model = {(p, c): (o, t, f) for p, c, o, t, f in conn.execute(
            "SELECT platform, comment_id, on_topic, tone, function FROM ann.labels "
            "WHERE codebook='v2' AND model='gpt-5.6-luna' AND run_tag='full_v2'")}
        on_m, on_h, tone_pairs, fn_pairs = [], [], [], []
        for r in csv.DictReader(open(args.comments, encoding="utf-8-sig")):
            h = to_bool(r.get("human_on_topic"))
            m = model.get((r["platform"], r["comment_id"]))
            if h is None or m is None:
                continue
            on_m.append(m[0]); on_h.append(h)
            if m[0] == 1 and h == 1:
                if (r.get("human_tone") or "").strip():
                    tone_pairs.append((m[1], r["human_tone"].strip()))
                if (r.get("human_function") or "").strip():
                    fn_pairs.append((m[2], r["human_function"].strip()))
        print("=" * 60 + "\n评论编码：模型 vs 人工")
        report("是否相关（on_topic）", on_m, on_h, [1, 0])
        if on_m:
            tp = sum(m == 1 and h == 1 for m, h in zip(on_m, on_h))
            print(f"  以人工为基准：精确率 {tp / max(sum(on_m), 1):.1%}，召回率 {tp / max(sum(on_h), 1):.1%}")
        report("情感基调（双方都判相关）", *zip(*tone_pairs)) if tone_pairs else print("\n情感基调：未填写")
        report("话语功能（双方都判相关）", *zip(*fn_pairs)) if fn_pairs else print("\n话语功能：未填写")

    if os.path.exists(args.posts):
        rule = dict(((p, str(i)), r) for p, i, r in conn.execute("SELECT platform, post_id, is_relevant FROM posts"))
        pairs = []
        for r in csv.DictReader(open(args.posts, encoding="utf-8-sig")):
            h = to_bool(r.get("human_relevant"))
            if h is not None and (r["platform"], r["post_id"]) in rule:
                pairs.append((rule[(r["platform"], r["post_id"])], h))
        if pairs:
            print("\n" + "=" * 60 + "\n帖子相关性：关键词规则 vs 人工")
            report("帖子是否相关", [a for a, _ in pairs], [b for _, b in pairs], [1, 0])
            pos = [h for r, h in pairs if r == 1]
            neg = [h for r, h in pairs if r == 0]
            print(f"  规则精确率（规则判相关中人工也判相关）：{sum(pos) / max(len(pos), 1):.1%}（n={len(pos)}）")
            print(f"  规则漏判率（规则判不相关中人工判相关）：{sum(neg) / max(len(neg), 1):.1%}（n={len(neg)}）")


if __name__ == "__main__":
    main()
