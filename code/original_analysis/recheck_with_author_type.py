#!/usr/bin/env python3
"""
recheck_with_author_type.py —— 把"平台差异"和"发帖者差异"分开

背景（日志 7.5 R1）：微博约 83% 的相关评论出自官方媒体帖子，所以原来的"平台差异"里混着"谁在发帖"的差异。
测量决策（日志 7.5b）：发帖者类型只用二分（官方媒体 + 政府与文博机构 = 官方；其余 = 其他），帖子体裁弃用。

  T14a 各平台的帖子 / 相关评论中官方帖所占比例
  T14b 控制发帖者类型后的平台差异：logit ~ 平台 + 官方，按帖子聚类；
       置换检验为分层置换（只在同一发帖者类型内部打乱帖子的平台标签）
  T14c 分别只在官方帖、只在其他帖内部比较平台（检验 + 占比）
  T14d 阶段效应：logit ~ 阶段 + 平台 + 官方
  T14e 官方帖效应：同一模型里"官方"的 OR；置换检验只在同一平台内部打乱帖子的官方标签

重叠帖上两人二分不一致、负责人尚未裁决的帖子（pending=1），按"记其他""记官方"各跑一遍，
表里用 scenario 列区分；裁决后重跑 build_author_type_binary.py 和本脚本即只剩一种。

用法：python3 recheck_with_author_type.py
前置：7.数据清洗与标注/db/post_author_type_binary.csv（pipeline/build_author_type_binary.py 生成）
"""
import os
import sys

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats.multitest import multipletests

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import analyze as A  # noqa: E402

OUT = os.path.dirname(HERE)
ROOT = os.path.dirname(OUT)
AT_PATH = os.path.join(ROOT, "7.数据清洗与标注", "db", "post_author_type_binary.csv")
LAYERS = (("tone_en", A.TONES), ("function_en", A.FUNCTIONS))


def strat_perm(data, layer, cat, term, strata_col, n_perm=2000):
    """分层置换：帖子整体保留，只在 strata_col 的同一层内部打乱帖子的 term 标签"""
    d = data.assign(y=(data[layer] == cat).astype(int))
    d, _ = A.usable(d, term)
    post = d.groupby("post_key").agg(g=(term, "first"), s=(strata_col, "first"), y=("y", "sum"), n=("y", "size"))
    codes = pd.factorize(post.g)[0]
    strata = pd.factorize(post.s)[0]
    y, n = post.y.to_numpy(), post.n.to_numpy()
    k, p0 = codes.max() + 1, y.sum() / n.sum()
    if p0 in (0, 1) or k < 2:
        return np.nan

    def stat(c):
        ys, ns = np.bincount(c, y, k), np.bincount(c, n, k)
        e = ns * p0
        ok = (e > 0) & (ns - e > 0)
        return (((ys - e) ** 2 / e)[ok] + ((e - ys) ** 2 / (ns - e))[ok]).sum()

    obs = stat(codes)
    rng = np.random.default_rng(20260921)
    null = []
    for _ in range(n_perm):
        c = codes.copy()
        for s in np.unique(strata):
            m = strata == s
            c[m] = rng.permutation(codes[m])
        null.append(stat(c))
    return float((1 + (np.array(null) >= obs).sum()) / (1 + n_perm))


def fit(d, layer, cat, term, controls):
    """返回 (拟合结果, 数据)；类别过少、完全分离或拟合失败时结果为 None"""
    dd = d.assign(y=(d[layer] == cat).astype(int))
    dd, _ = A.usable(dd, term)
    ctrl = [c for c in ("author_type", "platform_en") if c in controls and c != term]
    if dd.y.sum() < 5 or A.separated(dd, [term] + ctrl):
        return None, dd
    try:
        res = smf.glm(f"y ~ C({term})" + (f" + {controls}" if controls else ""), dd, family=sm.families.Binomial()).fit(
            cov_type="cluster", cov_kwds={"groups": pd.factorize(dd.post_key)[0]})
    except Exception:
        return None, dd
    return res, dd


def wald(d, layer, cat, term, controls, diag=None):
    """退化保护：帖子层控制变量可能几乎吸收掉阶段 / 平台，此时聚类稳健协方差接近奇异，
    Wald 会给出 p≈1 这种看似正常却无意义的结果 → 返回 NaN 并记下原因"""
    res, _ = fit(d, layer, cat, term, controls)
    if res is None:
        return np.nan
    names = [n for n in res.params.index if n.startswith(f"C({term})")]
    R = np.zeros((len(names), len(res.params)))
    for i, n in enumerate(names):
        R[i, list(res.params.index).index(n)] = 1
    cov = R @ np.asarray(res.cov_params()) @ R.T
    if np.linalg.matrix_rank(cov, tol=1e-10) < len(names) or not np.all(np.isfinite(cov)):
        if diag is not None:
            diag.append(f"{cat}：控制变量吸收了 {term} 的变异，协方差奇异，无法检验")
        return np.nan
    p = float(res.wald_test(R, scalar=True).pvalue)
    if p > 0.999 and diag is not None:
        diag.append(f"{cat}：p≈1，可能是控制变量与 {term} 高度共线，结果不可靠")
    return p


def holm(df, col, newcol):
    df[newcol] = np.nan
    ok = df[col].notna()
    if ok.any():
        df.loc[ok, newcol] = multipletests(df.loc[ok, col], method="holm")[1]
    return df


def run(ds, scenario):
    core = ds[(ds.on_topic == 1) & (ds.is_post_event == 1)].copy()
    on = ds[ds.on_topic == 1].copy()
    print(f"\n{'=' * 70}\n情形：{scenario}　事件后相关评论 {len(core)} 条，{core.post_key.nunique()} 帖")

    # T14a 构成
    posts = core.drop_duplicates("post_key")
    t14a = pd.DataFrame({
        "n_posts": posts.groupby("platform_en").size(),
        "n_official_posts": posts.groupby("platform_en").author_type.apply(lambda s: (s == "官方").sum()),
        "n_comments": core.groupby("platform_en").size(),
        "share_comments_under_official": core.groupby("platform_en").author_type.apply(lambda s: (s == "官方").mean()).round(3),
    }).reset_index()
    print("\nT14a 各平台官方帖占比\n", t14a.to_string(index=False))

    # T14b 控制发帖者类型后的平台差异
    rows = []
    for layer, cats in LAYERS:
        for c in cats:
            rows.append({"layer": layer.replace("_en", ""), "category": c,
                         "p_wald_ctrl_author": wald(core, layer, c, "platform_en", "C(author_type)"),
                         "p_perm_stratified": strat_perm(core, layer, c, "platform_en", "author_type")})
    t14b = holm(holm(pd.DataFrame(rows), "p_wald_ctrl_author", "p_wald_holm"), "p_perm_stratified", "p_perm_holm")
    t14b["still_significant"] = (t14b.p_wald_holm < .05) & (t14b.p_perm_holm < .05)
    print("\nT14b 控制发帖者类型后的平台差异\n", t14b.round(4).to_string(index=False))

    # T14c 官方帖内部 / 其他帖内部
    tests, shares = [], []
    for sub_name in ("官方", "其他"):
        sub = core[core.author_type == sub_name]
        print(f"\n{sub_name}帖下的相关评论 {len(sub)} 条，{sub.post_key.nunique()} 帖，各平台帖子数 "
              f"{sub.groupby('platform_en').post_key.nunique().to_dict()}")
        for layer, cats in LAYERS:
            t = A.platform_tests(sub, layer, cats)
            t.insert(0, "layer", layer.replace("_en", ""))
            t.insert(0, "subset", sub_name)
            tests.append(t)
            s = A.cluster_boot_shares(sub, "platform_en", layer, cats)
            s.insert(0, "layer", layer.replace("_en", ""))
            s.insert(0, "subset", sub_name)
            shares.append(s)
    t14c, t14c2 = pd.concat(tests, ignore_index=True), pd.concat(shares, ignore_index=True)
    print("\nT14c 子集内部的平台差异\n",
          t14c[["subset", "layer", "category", "p_holm", "p_perm_holm", "test_used"]].round(4).to_string(index=False))

    # T14d 阶段效应
    rows, diag = [], []
    on_trim = on[~on.post_key.isin(on.post_key.value_counts().head(5).index)]   # 与 analyze.py 的 T8 相同：去掉评论最多的 5 帖
    for layer, cats in LAYERS:
        for c in cats:
            rows.append({"layer": layer.replace("_en", ""), "category": c,
                         "p_phase_ctrl_platform_author": wald(on, layer, c, "phase_en",
                                                              "C(platform_en) + C(author_type)", diag),
                         "p_drop_top5_posts": wald(on_trim, layer, c, "phase_en",
                                                   "C(platform_en) + C(author_type)", diag)})
    t14d = holm(holm(pd.DataFrame(rows), "p_phase_ctrl_platform_author", "p_holm"), "p_drop_top5_posts", "p_drop_top5_holm")
    t14d["significant_both"] = (t14d.p_holm < .05) & (t14d.p_drop_top5_holm < .05)
    print("\nT14d 阶段效应（控制平台 + 发帖者类型）\n", t14d.round(4).to_string(index=False))
    for x in dict.fromkeys(diag):
        print("   退化诊断（不能当作'不显著'解读）：", x)

    # T14e 官方帖效应
    rows = []
    for layer, cats in LAYERS:
        for c in cats:
            res, dd = fit(core, layer, c, "author_type", "C(platform_en)")
            r = {"layer": layer.replace("_en", ""), "category": c,
                 "share_official": (core[core.author_type == "官方"][layer] == c).mean(),
                 "share_other": (core[core.author_type == "其他"][layer] == c).mean(),
                 "OR_official": np.nan, "ci_low": np.nan, "ci_high": np.nan, "p_wald": np.nan,
                 "p_perm_within_platform": strat_perm(core, layer, c, "author_type", "platform_en")}
            if res is not None:
                name = next(n for n in res.params.index if n.startswith("C(author_type)"))
                sign = 1 if "官方" in name else -1          # 参照水平取决于排序，统一成"官方相对其他"
                b, ci = sign * res.params[name], sign * res.conf_int().loc[name].to_numpy()
                r.update(OR_official=np.exp(b), ci_low=np.exp(min(ci)), ci_high=np.exp(max(ci)), p_wald=res.pvalues[name])
            rows.append(r)
    t14e = holm(holm(pd.DataFrame(rows), "p_wald", "p_wald_holm"), "p_perm_within_platform", "p_perm_holm")
    t14e["significant_both"] = (t14e.p_wald_holm < .05) & (t14e.p_perm_holm < .05)
    print("\nT14e 官方帖效应（控制平台）\n", t14e.round(4).to_string(index=False))

    out = {"T14a_author_type_by_platform": t14a, "T14b_platform_tests_controlling_author_type": t14b,
           "T14c_platform_tests_within_author_type": t14c, "T14c2_shares_within_author_type": t14c2,
           "T14d_phase_tests_controlling_author_type": t14d, "T14e_official_post_effect": t14e}
    for t in out.values():
        t.insert(0, "scenario", scenario)
    return out


def main():
    if not os.path.exists(AT_PATH):
        sys.exit(f"缺少 {AT_PATH}\n请先运行 7.数据清洗与标注/pipeline/build_author_type_binary.py")
    at = pd.read_csv(AT_PATH, dtype=str)
    at["post_key_tail"] = at.pid.astype(str)
    at["official"] = pd.to_numeric(at.official).map({1.0: "1", 0.0: "0"})
    ds = pd.read_csv(os.path.join(OUT, "data", "analysis_dataset.csv"))
    ds["post_key_tail"] = ds.post_key.astype(str).str.split(":").str[-1]
    ds = ds.merge(at[["platform", "post_key_tail", "official", "pending"]], on=["platform", "post_key_tail"], how="left")
    rel = ds[ds.on_topic == 1]
    uncoded = rel[rel.official.isna() & (rel.pending != "1")]
    print(f"相关评论中发帖者类型未编码：{len(uncoded)} 条，{uncoded.post_key.nunique()} 帖"
          f"（事件后 {int((uncoded.is_post_event == 1).sum())} 条）→ 不进入本脚本的模型")
    ds = ds[ds.official.notna() | (ds.pending == "1")]
    n_pending = ds[ds.pending == "1"].post_key.nunique()
    scenarios = [("pending=其他", "0"), ("pending=官方", "1")] if n_pending else [("已裁决", None)]
    if n_pending:
        p = ds[(ds.pending == "1") & (ds.on_topic == 1) & (ds.is_post_event == 1)]
        print(f"待裁决帖 {n_pending} 个，其下事件后相关评论 {len(p)} 条 → 两种取值各跑一遍")
    results = {}
    for label, val in scenarios:
        d = ds.copy()
        if val is not None:
            d.loc[d.pending == "1", "official"] = val
        d["author_type"] = d.official.map({"1": "官方", "0": "其他"})
        for name, t in run(d, label).items():
            results.setdefault(name, []).append(t)
    for name, ts in results.items():
        pd.concat(ts, ignore_index=True).to_csv(os.path.join(OUT, "tables", name + ".csv"), index=False, encoding="utf-8-sig")
    print("\n已写出 tables/T14a–T14e。")


if __name__ == "__main__":
    main()
