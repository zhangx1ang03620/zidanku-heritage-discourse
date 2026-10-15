#!/usr/bin/env python3
"""
validity_weighted.py —— 模型标注的有效性：以人工金标准为准，按抽样层反加权

507 条盲审样本是按（平台 × 模型是否判相关 × 模型功能类别）分层抽的，每层至少 2 条，稀有类被过度代表。
κ 可以直接在样本上算（kappa.py）；精确率、召回率和各类占比要推回全量，必须按层加权：
权重 = 该层在全量中的条数 / 该层在样本中的条数。置信区间用层内 bootstrap。

输出 8.分析与图表/tables/T15_model_validity_weighted.csv
用法：python3 validity_weighted.py [--comments db/human_review_comments.csv]
前置：round3_merge.py --finalize 已生成金标准
"""
import argparse
import os
import sqlite3

import numpy as np
import pandas as pd

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(os.path.dirname(BASE), "8.分析与图表", "tables", "T15_model_validity_weighted.csv")
B = 2000


def load(path):
    conn = sqlite3.connect(os.path.join(BASE, "db", "heritage.sqlite"))
    conn.execute("ATTACH DATABASE ? AS ann", (os.path.join(BASE, "db", "annotations.sqlite"),))
    m = pd.read_sql("SELECT platform, comment_id, on_topic AS m_on, tone AS m_tone, function AS m_fn FROM ann.labels "
                    "WHERE codebook='v2' AND model='gpt-5.6-luna' AND run_tag='full_v2'", conn)
    m["comment_id"] = m.comment_id.astype(str)
    m["stratum"] = m.platform + "|" + m.m_on.astype(str) + "|" + m.m_fn.fillna("")
    h = pd.read_csv(path, dtype=str, encoding="utf-8-sig")
    d = h.merge(m, on=["platform", "comment_id"], how="left")
    if d.m_on.isna().any():
        raise SystemExit(f"{int(d.m_on.isna().sum())} 条金标准在模型标注里找不到")
    d["h_on"] = (d.human_on_topic.str.strip() == "是").astype(int)
    d["w"] = d.stratum.map(m.stratum.value_counts()) / d.stratum.map(d.stratum.value_counts())
    print(f"金标准 {len(d)} 条，覆盖 {d.stratum.nunique()} / {m.stratum.nunique()} 层，"
          f"权重范围 {d.w.min():.1f}–{d.w.max():.1f}，加权后代表 {d.w.sum():.0f} / {len(m)} 条")
    return d


def metrics(d):
    w = d.w.to_numpy()
    r = {}
    tp = w[(d.m_on == 1) & (d.h_on == 1)].sum()
    r["on_topic|precision"] = tp / w[d.m_on == 1].sum()
    r["on_topic|recall"] = tp / w[d.h_on == 1].sum()
    r["on_topic|share_model"] = w[d.m_on == 1].sum() / w.sum()
    r["on_topic|share_human"] = w[d.h_on == 1].sum() / w.sum()
    both = d[(d.m_on == 1) & (d.h_on == 1)]
    for layer, mc, hc in (("tone", "m_tone", "human_tone"), ("function", "m_fn", "human_function")):
        r[f"{layer}|accuracy(both on-topic)"] = np.average(both[mc] == both[hc], weights=both.w)
        for c in sorted(set(d[hc].dropna()) | set(both[mc])):
            mm, hh = (d.m_on == 1) & (d[mc] == c), (d.h_on == 1) & (d[hc] == c)
            hit = w[mm & hh].sum()
            r[f"{layer}|{c}|precision"] = hit / w[mm].sum() if w[mm].sum() else np.nan
            r[f"{layer}|{c}|recall"] = hit / w[hh].sum() if w[hh].sum() else np.nan
            # 占比：各自判相关的评论里该类所占比例（论文报告的量）
            r[f"{layer}|{c}|share_model"] = w[mm].sum() / w[d.m_on == 1].sum()
            r[f"{layer}|{c}|share_human"] = w[hh].sum() / w[d.h_on == 1].sum()
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--comments", default=os.path.join(BASE, "db", "human_review_comments.csv"))
    ap.add_argument("--out", default=OUT)
    args = ap.parse_args()
    d = load(args.comments)
    point = metrics(d)
    rng = np.random.default_rng(20261009)
    groups = [g for _, g in d.groupby("stratum")]
    boots = []
    for _ in range(B):
        boots.append(metrics(pd.concat([g.iloc[rng.integers(0, len(g), len(g))] for g in groups])))
    bt = pd.DataFrame(boots)
    t = pd.DataFrame({"metric": list(point), "estimate": list(point.values())})
    t["ci_low"] = [np.nanpercentile(bt[k], 2.5) if k in bt else np.nan for k in point]
    t["ci_high"] = [np.nanpercentile(bt[k], 97.5) if k in bt else np.nan for k in point]
    t.to_csv(args.out, index=False, encoding="utf-8-sig")
    print(t.round(3).to_string(index=False))
    print("→", args.out)


if __name__ == "__main__":
    main()
