#!/usr/bin/env python3
"""
merge_human_coding.py —— 汇总两名复核员的人工编码

第 1 步（收齐两人的文件后运行）：
    python3 merge_human_coding.py --step 1
    → 计算两人之间的一致性（Cohen's κ），并把有分歧的行导出成裁决表：
      人工编码任务包/裁决表_*.csv（负责人只需填「裁决结果」一列）

第 2 步（裁决表填完后运行）：
    python3 merge_human_coding.py --step 2
    → 合成最终人工编码：db/human_review_comments.csv、db/human_review_posts.csv、db/post_author_type_final.csv
      （两人一致的直接采用，有分歧的用裁决结果）
    → 之后运行 kappa.py 得到"人工 vs 模型"的一致性

注意：人工之间的 κ（本脚本）衡量编码本是否清晰；人工 vs 模型的 κ（kappa.py）才是模型的有效性证据。
"""
import argparse
import os

import pandas as pd
from sklearn.metrics import cohen_kappa_score

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PACK = os.path.join(BASE, "人工编码任务包")
DB = os.path.join(BASE, "db")
TASKS = [
    # (任务名, 文件前缀, 主键, 需要比较的列)
    ("评论编码", "A1_评论编码", ["platform", "comment_id"], ["human_on_topic", "human_tone", "human_function"]),
    ("帖子相关性", "A2_帖子相关性", ["platform", "post_id"], ["human_relevant"]),
    ("发帖者类型", "B_发帖者类型", ["platform", "pid"], ["author_type", "帖子体裁"]),
]


def read(prefix, who):
    path = os.path.join(PACK, f"{prefix}_{who}.csv")
    if not os.path.exists(path):
        return None
    d = pd.read_csv(path, encoding="utf-8-sig", dtype=str)
    return d.apply(lambda s: s.str.strip() if s.dtype == object else s)


def interpret(k):
    return "几乎完全一致" if k > 0.8 else "高度一致" if k > 0.6 else "中等（需讨论编码本）" if k > 0.4 else "偏低（编码本或培训有问题）"


def step1():
    for name, prefix, key, cols in TASKS:
        a, b = read(prefix, "复核员A"), read(prefix, "复核员B")
        if a is None or b is None:
            print(f"跳过「{name}」：缺少文件")
            continue
        m = a.merge(b, on=key, suffixes=("_A", "_B"))
        print(f"\n=== {name} ===（两人共同编码 {len(m)} 条）")
        dis = pd.Series(False, index=m.index)
        for c in cols:
            pair = m[[f"{c}_A", f"{c}_B"]].fillna("")
            both = pair[(pair[f"{c}_A"] != "") & (pair[f"{c}_B"] != "")]
            if len(both) > 1:
                k = cohen_kappa_score(both[f"{c}_A"], both[f"{c}_B"])
                agree = (both[f"{c}_A"] == both[f"{c}_B"]).mean()
                print(f"  {c:<16} n={len(both):<5} 一致率={agree:6.1%}  κ={k:.3f}（{interpret(k)}）")
            dis |= pair[f"{c}_A"] != pair[f"{c}_B"]
        out = m[dis].copy()
        for c in cols:
            out[f"裁决_{c}"] = ""
        keep = [k for k in key] + [c for c in m.columns if c.endswith(("_A", "_B")) and not c.startswith("备注")] + \
               [f"裁决_{c}" for c in cols]
        ctx = [c for c in ("帖子主题_A", "评论内容_A", "帖子内容（前300字）_A", "title_or_text_A", "url_A", "author_A") if c in m.columns]
        path = os.path.join(PACK, f"裁决表_{name}.csv")
        out[key + ctx + keep[len(key):]].drop_duplicates().to_csv(path, index=False, encoding="utf-8-sig")
        print(f"  分歧 {len(out)} 条（{len(out) / max(len(m), 1):.1%}）→ {path}")
        print("  请在裁决表里填「裁决_xxx」列：两人谁对就填谁的答案，都不对就填你认为正确的。")


def step2():
    outputs = {}
    for name, prefix, key, cols in TASKS:
        a, b = read(prefix, "复核员A"), read(prefix, "复核员B")
        if a is None or b is None:
            continue
        adj_path = os.path.join(PACK, f"裁决表_{name}.csv")
        adj = pd.read_csv(adj_path, encoding="utf-8-sig", dtype=str) if os.path.exists(adj_path) else pd.DataFrame(columns=key)
        merged = pd.concat([a, b], ignore_index=True).drop_duplicates(subset=key, keep="first").set_index(key)
        m = a.merge(b, on=key, suffixes=("_A", "_B")).set_index(key)
        for c in cols:
            same = m[f"{c}_A"].fillna("") == m[f"{c}_B"].fillna("")
            merged.loc[m.index[same], c] = m.loc[same, f"{c}_A"]          # 两人一致 → 直接用
            if f"裁决_{c}" in adj.columns and len(adj):
                v = adj.set_index(key)[f"裁决_{c}"].dropna()
                v = v[v.astype(str).str.strip() != ""]
                merged.loc[merged.index.intersection(v.index), c] = v     # 有分歧 → 用裁决结果
        merged = merged.reset_index()
        outputs[name] = merged
        # 统计真正缺的格：判「否」的评论本来就不填语气/功能，不算缺
        applicable = merged.human_on_topic.astype(str).str.strip().eq("是") if "human_on_topic" in merged else None
        left = 0
        for c in cols:
            miss = merged[c].isna() | merged[c].astype(str).str.strip().isin(["", "nan"])
            if applicable is not None and c in ("human_tone", "human_function"):
                miss &= applicable
            left += int(miss.sum())
        print(f"{name}：合成 {len(merged)} 条，仍缺 {left} 格（分歧未裁决的会留空，需回到裁决表补齐）")
    if "评论编码" in outputs:
        outputs["评论编码"].to_csv(os.path.join(DB, "human_review_comments.csv"), index=False, encoding="utf-8-sig")
    if "帖子相关性" in outputs:
        outputs["帖子相关性"].to_csv(os.path.join(DB, "human_review_posts.csv"), index=False, encoding="utf-8-sig")
    if "发帖者类型" in outputs:
        outputs["发帖者类型"].to_csv(os.path.join(DB, "post_author_type_final.csv"), index=False, encoding="utf-8-sig")
    print("\n下一步：\n  python3 kappa.py                                   # 人工 vs 模型 的一致性（论文的有效性证据）"
          "\n  python3 ../../8.分析与图表/scripts/recheck_with_author_type.py   # 控制发帖者类型后复核平台/阶段结论")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--step", type=int, choices=(1, 2), required=True)
    args = ap.parse_args()
    step1() if args.step == 1 else step2()
