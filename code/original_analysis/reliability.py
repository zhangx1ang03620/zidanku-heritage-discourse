#!/usr/bin/env python3
"""
reliability.py —— 编码本的模型间 / 重测信度

注意：这是 LLM 之间的一致性，只能说明编码本定义是否清晰、标注是否稳定，
不能替代人工 Kappa（论文里标注有效性的证据必须来自人工盲审，见 kappa.py）。

比较方式：
  on_topic 在全部样本上比较；
  tone / function 只在两方都判为相关的评论上比较——否则双方都填"不适用"会被算作一致，抬高 Kappa。

用法：python3 reliability.py --run-a pilot_v2:gpt-5.6-luna --run-b pilot_v2:gpt-5.6-terra ...
"""
import argparse
import os
import sqlite3

import pandas as pd
from sklearn.metrics import cohen_kappa_score, confusion_matrix

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ANN_DB = os.path.join(ROOT, "7.数据清洗与标注", "db", "annotations.sqlite")
OUT = os.path.join(ROOT, "8.分析与图表")


def load(run):
    tag, model = run.split(":")
    df = pd.read_sql("SELECT platform, comment_id, on_topic, tone, function FROM labels "
                     "WHERE codebook='v2' AND run_tag=? AND model=?",
                     sqlite3.connect(ANN_DB), params=(tag, model))
    return df.set_index(["platform", "comment_id"])


def compare(a, b, name_a, name_b):
    m = a.join(b, lsuffix="_a", rsuffix="_b", how="inner")
    rows = [{"比较": f"{name_a} vs {name_b}", "字段": "on_topic", "样本数": len(m),
             "一致率": (m.on_topic_a == m.on_topic_b).mean(),
             "Kappa": cohen_kappa_score(m.on_topic_a, m.on_topic_b)}]
    both = m[(m.on_topic_a == 1) & (m.on_topic_b == 1)]
    for f in ("tone", "function"):
        rows.append({"比较": f"{name_a} vs {name_b}", "字段": f, "样本数": len(both),
                     "一致率": (both[f + "_a"] == both[f + "_b"]).mean(),
                     "Kappa": cohen_kappa_score(both[f + "_a"], both[f + "_b"])})
    return pd.DataFrame(rows), both


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", nargs="+", required=True, help="形如 A=tag:model,B=tag:model")
    ap.add_argument("--out-name", default="reliability")
    args = ap.parse_args()

    tables, confusions = [], {}
    for pair in args.pairs:
        (na, ra), (nb, rb) = [p.split("=", 1) for p in pair.split(",")]
        t, both = compare(load(ra), load(rb), na, nb)
        tables.append(t)
        for f in ("tone", "function"):
            labels = sorted(set(both[f + "_a"]) | set(both[f + "_b"]))
            confusions[f"{na}_vs_{nb}_{f}"] = pd.DataFrame(
                confusion_matrix(both[f + "_a"], both[f + "_b"], labels=labels),
                index=[f"{na}:{l}" for l in labels], columns=[f"{nb}:{l}" for l in labels])

    table = pd.concat(tables, ignore_index=True)
    table.to_csv(os.path.join(OUT, "tables", f"{args.out_name}.csv"), index=False, encoding="utf-8-sig")
    with pd.ExcelWriter(os.path.join(OUT, "data", f"{args.out_name}_confusion.xlsx")) as w:
        for k, df in confusions.items():
            df.to_excel(w, sheet_name=k[:31])
    pd.set_option("display.width", 200)
    print(table.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    for k, df in confusions.items():
        if k.endswith("function"):
            print(f"\n{k}\n{df}")


if __name__ == "__main__":
    main()
