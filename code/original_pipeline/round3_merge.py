#!/usr/bin/env python3
"""
round3_merge.py —— 第 3 轮（教材 v6）回收文件的检查、信度与金标准合成

把两人交回的文件放进 人工编码任务包/第3轮_v6/回收/，文件名保持发出时的样子
（1_新题150条_复核员A.csv、2_评论507条重做_复核员B.csv ……），然后：

  python3 round3_merge.py --task new150     检查格式 + 两人之间的 κ（论文报告的信度，含 bootstrap 95% CI）
  python3 round3_merge.py --task redo507    检查格式 + 导出分歧表 裁决表_评论编码_第3轮.csv（不报 κ）
  python3 round3_merge.py --task retest150  检查格式 + κ（仅过程记录）
  python3 round3_merge.py --finalize        裁决表填完后：合成 db/human_review_comments.csv，之后跑 kappa.py

本脚本不读取任何模型标签。
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
R3 = os.path.join(BASE, "人工编码任务包", "第3轮_v6")
BACK = os.path.join(R3, "回收")
KEY = ["platform", "comment_id"]
COLS = ["human_on_topic", "human_tone", "human_function"]
TONES = ["喜悦自豪", "愤慨痛惜", "中性理性", "质疑异见"]
FUNCTIONS = ["情感表态", "追索诉求与国家能力", "文物知识与归还细节", "历史与文明论战", "处置过程的信任与怀疑"]
LEGAL = {"human_on_topic": ["是", "否"], "human_tone": TONES, "human_function": FUNCTIONS}
FILES = {"new150": "1_新题150条", "redo507": "2_评论507条重做", "retest150": "3_复测150条重做"}
ADJ = os.path.join(R3, "裁决表_评论编码_第3轮.csv")
WHO = ("复核员A", "复核员B")


def read(path):
    d = pd.read_csv(path, encoding="utf-8-sig", dtype=str)
    for c in d.columns:
        d[c] = d[c].str.strip()
    return d.replace({"": np.nan})


def load(task, who):
    """读回收文件并对照发出的模板检查；返回 (数据, 问题列表)"""
    name = f"{FILES[task]}_{who}.csv"
    path = os.path.join(BACK, name)
    if not os.path.exists(path):
        return None, [f"缺少 {path}"]
    d, tpl = read(path), read(os.path.join(R3, f"发给{who}", name))
    probs = []
    got, want = set(map(tuple, d[KEY].values)), set(map(tuple, tpl[KEY].values))
    if got != want or len(d) != len(tpl):
        probs.append(f"题目与模板不一致：共 {len(d)} 行，缺 {len(want - got)} 条，多 {len(got - want)} 条"
                     "（comment_id 常被 Excel 改成科学计数法，需让组员用原文件重填）")
    for c in COLS:
        bad = d[d[c].notna() & ~d[c].isin(LEGAL[c])]
        if len(bad):
            probs.append(f"{c} 有 {len(bad)} 个不合法取值：{bad[c].value_counts().to_dict()}")
    yes, no = d.human_on_topic == "是", d.human_on_topic == "否"
    n = int(d.human_on_topic.isna().sum())
    if n:
        probs.append(f"human_on_topic 未填 {n} 条")
    n = int((yes & (d.human_tone.isna() | d.human_function.isna())).sum())
    if n:
        probs.append(f"判「是」但语气或功能未填 {n} 条")
    n = int((no & (d.human_tone.notna() | d.human_function.notna())).sum())
    if n:
        probs.append(f"判「否」却填了语气或功能 {n} 条")
    return d, probs


def kappa_ci(a, b, n_boot=2000, seed=20261009):
    a, b = np.asarray(a), np.asarray(b)
    k = cohen_kappa_score(a, b)
    rng = np.random.default_rng(seed)
    ks = []
    for _ in range(n_boot):
        i = rng.integers(0, len(a), len(a))
        if len(set(a[i]) | set(b[i])) > 1:
            ks.append(cohen_kappa_score(a[i], b[i]))
    return k, np.percentile(ks, 2.5), np.percentile(ks, 97.5)


def pair(task):
    out = []
    for who in WHO:
        d, probs = load(task, who)
        print(f"\n[{who}] {FILES[task]}：" + ("格式检查通过" if not probs else ""))
        for p in probs:
            print("  ✗", p)
        if d is not None:
            print(f"  相关率 {(d.human_on_topic == '是').mean():.1%}；语气 {d.human_tone.value_counts().to_dict()}；"
                  f"功能 {d.human_function.value_counts().to_dict()}")
        out.append((d, probs))
    if any(p for _, p in out):
        sys.exit("\n先解决上面的格式问题再往下算。")
    return out[0][0].merge(out[1][0], on=KEY, suffixes=("_A", "_B"))


def report_kappa(m):
    both_yes = m[(m.human_on_topic_A == "是") & (m.human_on_topic_B == "是")]
    print("\n两人之间的一致性（语气 / 功能只在两人都判「是」的评论上算）")
    res = {}
    for c, d in (("human_on_topic", m), ("human_tone", both_yes), ("human_function", both_yes)):
        a, b = d[f"{c}_A"], d[f"{c}_B"]
        k, lo, hi = kappa_ci(a, b)
        res[c] = k
        print(f"  {c:<16} n={len(d):<4} 一致率={(a == b).mean():6.1%}  κ={k:.3f}  95% CI [{lo:.3f}, {hi:.3f}]")
        if c != "human_on_topic":
            print(pd.crosstab(a, b, rownames=["A"], colnames=["B"]).to_string().replace("\n", "\n      ").join(["      ", ""]))
    na, nb = (m.human_on_topic_A == "是").sum(), (m.human_on_topic_B == "是").sum()
    print(f"  相关判定方向：A是/B否 {((m.human_on_topic_A == '是') & (m.human_on_topic_B == '否')).sum()} 条，"
          f"A否/B是 {((m.human_on_topic_A == '否') & (m.human_on_topic_B == '是')).sum()} 条（A 判是 {na}，B 判是 {nb}）")
    return res


def export_adjudication(m):
    dis = np.zeros(len(m), bool)
    for c in COLS:
        dis |= (m[f"{c}_A"].fillna("") != m[f"{c}_B"].fillna("")).to_numpy()
    out = m[dis][KEY + ["帖子主题_A", "评论内容_A"] + [f"{c}_{w}" for c in COLS for w in "AB"] + ["备注_A", "备注_B"]].copy()
    out = out.rename(columns={"帖子主题_A": "帖子主题", "评论内容_A": "评论内容"})
    for c in COLS:
        out[f"裁决_{c}"] = ""
    out.to_csv(ADJ, index=False, encoding="utf-8-sig")
    print(f"\n两人一致 {len(m) - len(out)} 条，分歧 {len(out)} 条（{len(out) / len(m):.1%}）→ {ADJ}")
    print("负责人填三列「裁决_xxx」：判否时语气、功能两列留空。填完运行 --finalize。")


def finalize():
    m = pair("redo507")
    xlsx = ADJ[:-4] + ".xlsx"                      # 负责人在带下拉框的 xlsx 里填；没有 xlsx 时读 csv
    if os.path.exists(xlsx):
        adj = pd.read_excel(xlsx, sheet_name=0, dtype=str)
        adj = adj.apply(lambda c: c.str.strip()).replace({"": np.nan}).set_index(KEY)
    else:
        adj = read(ADJ).set_index(KEY)
    if set(adj.index) - set(map(tuple, m[KEY].values)):
        sys.exit("裁决表里有对不上的 comment_id（可能被 Excel 改写），请不要改动前两列。")
    gold = m[KEY + ["帖子主题_A", "评论内容_A"]].rename(columns={"帖子主题_A": "帖子主题", "评论内容_A": "评论内容"})
    mi = m.set_index(KEY)
    for c in COLS:
        same = (mi[f"{c}_A"].fillna("") == mi[f"{c}_B"].fillna(""))
        v = mi[f"{c}_A"].where(same)
        v.update(adj[f"裁决_{c}"].dropna())
        gold[c] = v.to_numpy()
    gold["来源"] = np.where(mi.index.isin(adj.index), "裁决", "两人一致")
    missing = int(gold.human_on_topic.isna().sum()) + int(((gold.human_on_topic == "是") &
                                                           (gold.human_tone.isna() | gold.human_function.isna())).sum())
    bad = sum(int((gold[c].notna() & ~gold[c].isin(LEGAL[c])).sum()) for c in COLS)
    if missing or bad:
        sys.exit(f"裁决表未填完或有不合法取值：缺 {missing} 格，不合法 {bad} 格。补齐后重跑。")
    gold.loc[gold.human_on_topic == "否", ["human_tone", "human_function"]] = np.nan
    out = os.path.join(BASE, "db", "human_review_comments.csv")
    gold.to_csv(out, index=False, encoding="utf-8-sig")
    print(f"\n金标准 {len(gold)} 条（两人一致 {(gold.来源 == '两人一致').sum()}，裁决 {(gold.来源 == '裁决').sum()}），"
          f"相关 {(gold.human_on_topic == '是').sum()} 条 → {out}\n下一步：python3 kappa.py")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", choices=list(FILES))
    ap.add_argument("--finalize", action="store_true")
    args = ap.parse_args()
    if args.finalize:
        finalize()
    elif args.task:
        m = pair(args.task)
        if args.task == "redo507":
            export_adjudication(m)
        else:
            res = report_kappa(m)
            if args.task == "new150":
                print("\n判定（日志 7.5c）：" + ("语气 κ ≥ 0.6，情感基调保留为主分析层。" if res["human_tone"] >= 0.6
                                          else "语气 κ < 0.6 → 情感基调降为探索性，主结论只用话语功能层。"))
    else:
        ap.print_help()
