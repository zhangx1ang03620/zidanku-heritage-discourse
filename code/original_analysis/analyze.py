#!/usr/bin/env python3
"""
analyze.py —— 全部统计分析，结果写入 tables/（论文用）和 data/（作图用）

防线（见实验日志第 4 节）：
  - 评论按帖子聚类：置信区间用"按帖子重抽样"的聚类 bootstrap；回归用按帖子聚类的稳健标准误
  - 逐类别检验做 Holm 校正
  - 阶段比较控制平台固定效应（事件前语料以知乎为主，不控制会把平台差异误当成时间变化）
  - 互动量：帖子内比较（去掉帖子热度、曝光时长等帖子层混杂），并在"评论采集完整"的帖子上复核
"""
import glob
import json
import os
import sqlite3
import warnings

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy.stats import chi2_contingency, spearmanr
from sklearn.metrics import cohen_kappa_score
from statsmodels.stats.multitest import multipletests

warnings.filterwarnings("ignore")
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DB = os.path.join(ROOT, "7.数据清洗与标注", "db", "heritage.sqlite")
ANN_DB = os.path.join(ROOT, "7.数据清洗与标注", "db", "annotations.sqlite")
RAW = os.path.join(ROOT, "6.爬取的评论数据")
OUT = os.path.join(ROOT, "8.分析与图表")
T = lambda name: os.path.join(OUT, "tables", name)
D = lambda name: os.path.join(OUT, "data", name)

PLATFORMS = ["Bilibili", "Douyin", "Weibo", "Xiaohongshu", "Zhihu"]
TONES = ["Joy & pride", "Indignation & grief", "Neutral/informational", "Skepticism & dissent"]
FUNCTIONS = ["Affective expression", "Restitution claims & state capacity", "Artifact knowledge & return details",
             "Historical & civilizational debate", "Institutional trust & distrust"]
B = 1000
RNG = np.random.default_rng(20260921)


def save(df, name, table=True):
    df.to_csv(T(name) if table else D(name), index=False, encoding="utf-8-sig")


# ---------- 通用工具 ----------

def cluster_boot_shares(data, group_col, cat_col, cats, b=B):
    """各组内类别占比 + 按帖子重抽样的 95% 置信区间"""
    rows = []
    groups = [None] if group_col is None else sorted(data[group_col].dropna().unique())
    for g in groups:
        sub = data if g is None else data[data[group_col] == g]
        m = pd.crosstab(sub.post_key, sub[cat_col]).reindex(columns=cats, fill_value=0).to_numpy()
        tot = m.sum(1)
        idx = RNG.integers(0, len(m), size=(b, len(m)))
        boot = m[idx].sum(1) / tot[idx].sum(1, keepdims=True)
        share = m.sum(0) / tot.sum()
        lo, hi = np.percentile(boot, [2.5, 97.5], axis=0)
        for i, c in enumerate(cats):
            rows.append({"group": "All" if g is None else g, "category": c, "share": share[i], "ci_low": lo[i],
                         "ci_high": hi[i], "n_comments": int(m[:, i].sum()), "group_total": int(tot.sum()),
                         "n_posts": len(m)})
    return pd.DataFrame(rows)


def cramers_v(table):
    chi2 = chi2_contingency(table, correction=False)[0]
    n = table.to_numpy().sum()
    return float(np.sqrt(chi2 / (n * (min(table.shape) - 1))))


MIN_POSTS = 10     # 帖子数少于此值的组不参与推断（聚类稳健标准误在极少聚类下不可靠），只做描述
N_PERM = 2000


def usable(data, term, controls_cols=()):
    """去掉帖子数 < MIN_POSTS 的组；返回 (数据, 被剔除的组)"""
    n_posts = data.groupby(term).post_key.nunique()
    small = sorted(n_posts[n_posts < MIN_POSTS].index)
    return data[~data[term].isin(small)], small


def separated(d, cols):
    """完全分离：某个水平下类别计数为 0 或全部属于该类 → logit 系数发散，Wald p 值无意义"""
    for col in cols:
        g = d.groupby(col).y.agg(["sum", "count"])
        if ((g["sum"] == 0) | (g["sum"] == g["count"])).any():
            return True
    return False


def cluster_wald(data, cat_col, cat, term, controls=""):
    """logit(是否属于某类别) ~ term + controls，按帖子聚类；返回 term 整体的 Wald 检验 p 值。
    修正（2026-09-21，全量结果后发现的 bug，见日志第 7 节）：先剔除帖子数 < MIN_POSTS 的组；
    若出现完全分离则返回 NaN（原实现会给出 1e-194 这类发散的 p 值）。"""
    d = data.assign(y=(data[cat_col] == cat).astype(int))
    d, _ = usable(d, term)
    if d.y.sum() < 5:
        return np.nan, None
    ctrl_cols = ["platform_en"] if "platform_en" in controls else []
    if separated(d, [term] + ctrl_cols):
        return np.nan, None
    f = f"y ~ C({term})" + (f" + {controls}" if controls else "")
    try:
        res = smf.glm(f, d, family=sm.families.Binomial()).fit(
            cov_type="cluster", cov_kwds={"groups": pd.factorize(d.post_key)[0]})
    except Exception:
        return np.nan, None
    names = [n for n in res.params.index if n.startswith(f"C({term})")]
    if not names:
        return np.nan, res
    R = np.zeros((len(names), len(res.params)))
    for i, n in enumerate(names):
        R[i, list(res.params.index).index(n)] = 1
    return float(res.wald_test(R, scalar=True).pvalue), res


def perm_test(data, cat_col, cat, term, n_perm=N_PERM):
    """按帖子置换检验（term 必须是帖子层变量，如平台）：帖子整体保留，只打乱帖子的组标签，
    统计量 = 组 × 是否属于该类 的 Pearson 卡方。对聚类和完全分离都稳健。"""
    d = data.assign(y=(data[cat_col] == cat).astype(int))
    d, _ = usable(d, term)
    post = d.groupby("post_key").agg(g=(term, "first"), y=("y", "sum"), n=("y", "size"))
    codes = pd.factorize(post.g)[0]
    y, n = post.y.to_numpy(), post.n.to_numpy()
    k, p0 = codes.max() + 1, y.sum() / n.sum()
    if p0 in (0, 1):
        return np.nan

    def stat(c):
        ys, ns = np.bincount(c, y, k), np.bincount(c, n, k)
        e1 = ns * p0
        return (((ys - e1) ** 2) / e1 + ((ns - ys - (ns - e1)) ** 2) / (ns - e1)).sum()

    obs = stat(codes)
    rng = np.random.default_rng(20260921)
    null = np.array([stat(rng.permutation(codes)) for _ in range(n_perm)])
    return float((1 + (null >= obs).sum()) / (1 + n_perm))


def platform_tests(data, layer, cats):
    """平台差异逐类别检验：主检验 = 聚类稳健 Wald；分离时改用置换 p；两者都报告"""
    _, dropped = usable(data, "platform_en")
    rows = []
    for c in cats:
        pw, _ = cluster_wald(data, layer, c, "platform_en")
        pp = perm_test(data, layer, c, "platform_en")
        rows.append({"category": c, "p_cluster_wald": pw, "p_perm": pp,
                     "p_used": pw if pd.notna(pw) else pp, "test_used": "wald" if pd.notna(pw) else "permutation",
                     "excluded_groups(<%d posts)" % MIN_POSTS: ";".join(dropped)})
    t = pd.DataFrame(rows)
    t["p_holm"] = np.nan
    ok = t.p_used.notna()
    if ok.any():
        t.loc[ok, "p_holm"] = multipletests(t.loc[ok, "p_used"], method="holm")[1]
    okp = t.p_perm.notna()
    t["p_perm_holm"] = np.nan
    if okp.any():
        t.loc[okp, "p_perm_holm"] = multipletests(t.loc[okp, "p_perm"], method="holm")[1]
    return t


# ---------- 各项分析 ----------

def funnel():
    raw = {}
    dirs = {"douyin": "1.douyin/json", "bilibili": "2.bilibili", "xhs": "3.xhs/json", "weibo": "4.weibo/json",
            "zhihu": "5.zhihu/json"}
    for p, d in dirs.items():
        raw[p] = sum(len(json.load(open(f, encoding="utf-8"))) for f in glob.glob(os.path.join(RAW, d, "*_comments_*.json")))
    c = sqlite3.connect(DB)
    q = lambda s: dict(c.execute(f"SELECT platform, COUNT(*) FROM comments WHERE {s} GROUP BY platform").fetchall())
    stages = [("Raw comments collected", raw),
              ("After de-duplication", q("1=1")),
              ("Under event-relevant posts", q("post_relevant=1")),
              ("With text, post-event (core)", q("post_relevant=1 AND is_post_event=1 AND TRIM(COALESCE(text,''))!=''")),
              ("With text, pre-event (baseline)", q("post_relevant=1 AND is_post_event=0 AND TRIM(COALESCE(text,''))!=''"))]
    ds = pd.read_csv(D("analysis_dataset.csv"))
    stages.append(("LLM-coded on-topic (core)", ds[(ds.on_topic == 1) & (ds.is_post_event == 1)].platform.value_counts().to_dict()))
    stages.append(("LLM-coded on-topic (baseline)", ds[(ds.on_topic == 1) & (ds.is_post_event == 0)].platform.value_counts().to_dict()))
    rows = []
    for name, d in stages:
        r = {"stage": name, **{k: int(d.get(k, 0)) for k in dirs}}
        r["total"] = sum(r[k] for k in dirs)
        rows.append(r)
    df = pd.DataFrame(rows)
    save(df, "T1_data_funnel.csv")
    return df


def keyword_precision():
    posts = pd.read_sql("SELECT platform, keyword, is_relevant FROM posts WHERE keyword IS NOT NULL", sqlite3.connect(DB))
    g = posts.groupby(["platform", "keyword"]).is_relevant.agg(["mean", "count"]).reset_index()
    g = g[g["count"] >= 20].rename(columns={"mean": "precision", "count": "n_posts"})
    g["platform"] = g.platform.map({"bilibili": "Bilibili", "douyin": "Douyin", "xhs": "Xiaohongshu",
                                    "weibo": "Weibo", "zhihu": "Zhihu"})
    save(g, "T2_keyword_precision.csv")
    return g


def on_topic_rates(ds):
    rel = ds.assign(on=ds.on_topic.map({1: "on", 0: "off"}))
    r = cluster_boot_shares(rel[rel.is_post_event == 1], "platform_en", "on", ["on", "off"])
    r = r[r.category == "on"].drop(columns="category").rename(columns={"group": "platform"})
    save(r, "T3_on_topic_rate_by_platform.csv")
    return r


def composition(core):
    out = {}
    for layer, cats in (("tone_en", TONES), ("function_en", FUNCTIONS)):
        overall = cluster_boot_shares(core, None, layer, cats)
        by_plat = cluster_boot_shares(core, "platform_en", layer, cats)
        tab = pd.crosstab(core.platform_en, core[layer]).reindex(columns=cats, fill_value=0)
        tab = tab.loc[:, tab.sum() > 0]
        tests = platform_tests(core, layer, cats)
        tests["cramers_v_layer"] = cramers_v(tab)
        name = "tone" if layer == "tone_en" else "function"
        save(pd.concat([overall, by_plat]), f"T4_{name}_by_platform.csv")
        save(tests, f"T5_{name}_platform_tests.csv")
        out[name] = (overall, by_plat, tests)
    joint = pd.crosstab(core.tone_en, core.function_en).reindex(index=TONES, columns=FUNCTIONS, fill_value=0)
    joint = joint.loc[:, joint.sum() > 0]
    chi2, p, dof, exp = chi2_contingency(joint, correction=False)
    resid = (joint - exp) / np.sqrt(exp * (1 - joint.sum(1).to_numpy()[:, None] / joint.to_numpy().sum())
                                    * (1 - joint.sum(0).to_numpy()[None, :] / joint.to_numpy().sum()))
    joint_long = joint.stack().rename("n").reset_index().merge(
        resid.stack().rename("adj_std_residual").reset_index(), on=["tone_en", "function_en"])
    joint_long["share_of_all"] = joint_long.n / joint_long.n.sum()
    save(joint_long, "T6_tone_x_function.csv")
    out["joint"] = joint_long
    return out


def robustness_composition(core):
    """两项稳健性检验（结果出来前预设）：
    (a) 帖子等权：按评论计的占比会被少数爆款帖子主导，改为每个帖子内先算占比、再对帖子取平均
        （只用 on-topic 评论 ≥ 3 条的帖子，避免 1–2 条评论的帖子占比非 0 即 1）
    (b) 同一检索词：各平台用的检索词不同，平台差异可能来自检索词差异。只保留五个平台都检索过的
        "子弹库帛书"下的评论，重做平台差异检验"""
    rows = []
    for layer, cats in (("tone_en", TONES), ("function_en", FUNCTIONS)):
        n_post = core.groupby("post_key").size()
        big = core[core.post_key.isin(n_post[n_post >= 3].index)]
        per_post = pd.crosstab(big.post_key, big[layer], normalize="index").reindex(columns=cats, fill_value=0)
        per_post["platform_en"] = big.groupby("post_key").platform_en.first()
        for g, sub in [("All", per_post)] + list(per_post.groupby("platform_en")):
            for c in cats:
                rows.append({"layer": layer.replace("_en", ""), "group": g, "category": c,
                             "share_post_weighted": sub[c].mean(), "n_posts": len(sub)})
    pw = pd.DataFrame(rows)
    cw = pd.concat([cluster_boot_shares(core, None, "function_en", FUNCTIONS).assign(layer="function"),
                    cluster_boot_shares(core, "platform_en", "function_en", FUNCTIONS).assign(layer="function"),
                    cluster_boot_shares(core, None, "tone_en", TONES).assign(layer="tone"),
                    cluster_boot_shares(core, "platform_en", "tone_en", TONES).assign(layer="tone")])
    pw = pw.merge(cw[["layer", "group", "category", "share"]].rename(columns={"share": "share_comment_weighted"}),
                  on=["layer", "group", "category"], how="left")
    save(pw, "T4b_robust_post_weighted.csv")

    same_kw = core[core.keyword == "子弹库帛书"]
    tests = []
    for layer, cats in (("tone_en", TONES), ("function_en", FUNCTIONS)):
        t = platform_tests(same_kw, layer, cats).drop(columns="p_holm")
        t.insert(0, "layer", layer.replace("_en", ""))
        tests.append(t)
    tests = pd.concat(tests, ignore_index=True)
    tests["n"], tests["platforms_tested"] = len(same_kw), same_kw.platform_en.nunique() - len(usable(same_kw, "platform_en")[1])
    tests["p_holm"] = multipletests(tests.p_used, method="holm")[1]  # 9 项检验一起校正，与冻结时一致
    tests["p_perm_holm"] = multipletests(tests.p_perm, method="holm")[1]
    save(tests, "T5b_robust_same_keyword_tests.csv")
    save(cluster_boot_shares(same_kw, "platform_en", "function_en", FUNCTIONS), "T5c_robust_same_keyword_function_shares.csv")
    return pw, tests


def temporal(on):
    comp = []
    for layer, cats in (("tone_en", TONES), ("function_en", FUNCTIONS)):
        r = cluster_boot_shares(on, "phase_en", layer, cats)
        r["layer"] = layer.replace("_en", "")
        comp.append(r)
    comp = pd.concat(comp)
    save(comp, "T7_composition_by_phase.csv")
    cells = pd.crosstab(on.platform_en, on.phase_en)
    cells.reset_index().to_csv(T("T7b_phase_by_platform_counts.csv"), index=False, encoding="utf-8-sig")

    # 冻结后收紧（日志第 7 节）：评论高度集中在少数爆款帖子，去掉评论数最多的 5 个帖子复核
    top5 = on.post_key.value_counts().head(5).index
    on_trim = on[~on.post_key.isin(top5)]
    tests = []
    for layer, cats in (("tone_en", TONES), ("function_en", FUNCTIONS)):
        for c in cats:
            p, res = cluster_wald(on, layer, c, "phase_en", controls="C(platform_en)")
            row = {"layer": layer.replace("_en", ""), "category": c, "p_phase_given_platform": p,
                   "p_drop_top5_posts": cluster_wald(on_trim, layer, c, "phase_en", controls="C(platform_en)")[0]}
            if res is not None:
                for n in res.params.index:
                    if n.startswith("C(phase_en)"):
                        lvl = n.split("[T.")[1].rstrip("]")
                        ci = res.conf_int().loc[n]
                        row[f"OR {lvl}"] = f"{np.exp(res.params[n]):.2f} [{np.exp(ci[0]):.2f}, {np.exp(ci[1]):.2f}]"
            tests.append(row)
    tests = pd.DataFrame(tests)
    tests["p_holm"] = np.nan
    ok = tests.p_phase_given_platform.notna()
    if ok.any():
        tests.loc[ok, "p_holm"] = multipletests(tests.loc[ok, "p_phase_given_platform"], method="holm")[1]
    tests["p_drop_top5_holm"] = np.nan
    ok2 = tests.p_drop_top5_posts.notna()
    if ok2.any():
        tests.loc[ok2, "p_drop_top5_holm"] = multipletests(tests.loc[ok2, "p_drop_top5_posts"], method="holm")[1]
    save(tests, "T8_phase_effects_platform_adjusted.csv")

    # 稳健性：月度序列（不依赖阶段划分）
    vol = pd.read_sql("SELECT substr(timestamp_iso,1,7) AS month, platform, COUNT(*) AS n FROM comments "
                      "WHERE post_relevant=1 GROUP BY 1,2", sqlite3.connect(DB))
    save(vol, "monthly_volume_all_relevant.csv", table=False)
    monthly = pd.crosstab(on.month, on.function_en, normalize="index").reindex(columns=FUNCTIONS, fill_value=0)
    monthly["n"] = on.month.value_counts()
    save(monthly.reset_index(), "monthly_function_share.csv", table=False)
    daily = pd.read_sql("SELECT substr(timestamp_iso,1,10) AS date, COUNT(*) AS n FROM comments "
                        "WHERE post_relevant=1 GROUP BY 1", sqlite3.connect(DB))
    save(daily, "daily_volume_all_relevant.csv", table=False)
    adjusted_phase_shares(on)
    return comp, tests


# ---------- 控制平台后的阶段占比（Fig6b 用；原始占比受阶段内平台构成混杂） ----------

B_ADJ = 400


def _adj_point(d, layer, cats, phases):
    """每个类别拟合 logit(y) ~ C(phase) + C(platform)；把所有评论的阶段依次设为 P，取预测概率均值
    （边际标准化：各阶段都按全样本的平台构成来算）"""
    out = {}
    for c in cats:
        dd = d.assign(y=(d[layer] == c).astype(int))
        try:
            res = smf.glm("y ~ C(phase_en) + C(platform_en)", dd, family=sm.families.Binomial()).fit()
        except Exception:
            continue
        for p in phases:
            out[(c, p)] = float(res.predict(dd.assign(phase_en=p)).mean())
    return out


def _adj_boot(args):
    d, layer, cats, phases, seed = args
    rng = np.random.default_rng(seed)
    keys = d.post_key.unique()
    pick = rng.choice(keys, len(keys), replace=True)
    groups = {k: g for k, g in d.groupby("post_key")}
    bd = pd.concat([groups[k] for k in pick], ignore_index=True)
    if bd.phase_en.nunique() < len(phases):
        return {}
    return _adj_point(bd, layer, cats, phases)


def adjusted_phase_shares(on):
    from concurrent.futures import ProcessPoolExecutor
    phases = sorted(on.phase_en.unique())
    rows = []
    for layer, cats in (("tone_en", TONES), ("function_en", FUNCTIONS)):
        d = on[["post_key", "phase_en", "platform_en", layer]]
        point = _adj_point(d, layer, cats, phases)
        with ProcessPoolExecutor(max_workers=max(1, (os.cpu_count() or 2) - 1)) as ex:
            boots = list(ex.map(_adj_boot, [(d, layer, cats, phases, 20260921 + i) for i in range(B_ADJ)], chunksize=10))
        for (c, p), v in point.items():
            bs = np.array([b[(c, p)] for b in boots if (c, p) in b])
            rows.append({"layer": layer.replace("_en", ""), "category": c, "phase": p, "adj_share": v,
                         "ci_low": np.percentile(bs, 2.5), "ci_high": np.percentile(bs, 97.5), "n_boot": len(bs),
                         "raw_share": (on[on.phase_en == p][layer] == c).mean()})
    df = pd.DataFrame(rows)
    save(df, "T8b_phase_shares_platform_adjusted.csv")
    return df


def engagement(on):
    d = on.copy()
    d["log_likes"] = np.log1p(d.like_count.clip(lower=0))
    d["log_len"] = np.log1p(d.text_len)
    results = []
    for label, sub in (("All on-topic comments", d), ("Posts with >=80% comment coverage", d[d.coverage >= 0.8])):
        # 帖子内比较：减去帖子均值（等价于帖子固定效应）
        X = pd.get_dummies(sub[["function_en", "tone_en"]], drop_first=False).astype(float)
        fx = [c for c in X.columns if c != "function_en_Affective expression" and c != "tone_en_Neutral/informational"]
        X = X[fx].assign(is_reply=sub.is_reply.astype(float), log_len=sub.log_len)
        y = sub.log_likes
        grp = sub.post_key
        Xd = X - X.groupby(grp).transform("mean")
        yd = y - y.groupby(grp).transform("mean")
        keep = Xd.std() > 0
        res = sm.OLS(yd, Xd.loc[:, keep]).fit(cov_type="cluster", cov_kwds={"groups": pd.factorize(grp)[0]})
        ci = res.conf_int()
        for n in res.params.index:
            if n.startswith(("function_en_", "tone_en_")):
                results.append({"sample": label, "term": n.split("_en_")[1], "layer": n.split("_en_")[0],
                                "coef_log_likes": res.params[n], "ci_low": ci.loc[n, 0], "ci_high": ci.loc[n, 1],
                                "pct_change": np.expm1(res.params[n]) * 100, "p": res.pvalues[n],
                                "n_comments": int(res.nobs), "n_posts": sub.post_key.nunique()})
    r = pd.DataFrame(results)
    save(r, "T9_engagement_within_post.csv")
    return r


def sensitivity():
    c = sqlite3.connect(ANN_DB)
    luna = pd.read_sql("SELECT platform, comment_id, on_topic, tone, function FROM labels WHERE codebook='v2' "
                       "AND model='gpt-5.6-luna' AND run_tag='full_v2'", c)
    terra = pd.read_sql("SELECT platform, comment_id, on_topic, tone, function FROM labels WHERE codebook='v2' "
                        "AND model='gpt-5.6-terra' AND run_tag='sens_v2'", c)
    m = luna.merge(terra, on=["platform", "comment_id"], suffixes=("_luna", "_terra"))
    rel = [{"field": "on_topic", "n": len(m), "agreement": (m.on_topic_luna == m.on_topic_terra).mean(),
            "kappa": cohen_kappa_score(m.on_topic_luna, m.on_topic_terra)}]
    both = m[(m.on_topic_luna == 1) & (m.on_topic_terra == 1)]
    for f in ("tone", "function"):
        rel.append({"field": f, "n": len(both), "agreement": (both[f + "_luna"] == both[f + "_terra"]).mean(),
                    "kappa": cohen_kappa_score(both[f + "_luna"], both[f + "_terra"])})
    save(pd.DataFrame(rel), "T10_luna_vs_terra_sens3000.csv")

    rows = []
    for model in ("luna", "terra"):
        sub = m[m[f"on_topic_{model}"] == 1]
        for f in ("tone", "function"):
            share = pd.crosstab(sub.platform, sub[f"{f}_{model}"], normalize="index")
            for p in share.index:
                for cat in share.columns:
                    rows.append({"model": model, "layer": f, "platform": p, "category": cat, "share": share.loc[p, cat],
                                 "on_topic_rate": m[m.platform == p][f"on_topic_{model}"].mean()})
    s = pd.DataFrame(rows)
    save(s, "sens_shares_luna_vs_terra.csv", table=False)
    w = s.pivot_table(index=["layer", "platform", "category"], columns="model", values="share").fillna(0).reset_index()
    rank = []
    for (layer, cat), g in w.groupby(["layer", "category"]):
        if len(g) >= 3 and g.luna.std() > 0 and g.terra.std() > 0:
            rank.append({"layer": layer, "category": cat, "spearman_platform_rank": spearmanr(g.luna, g.terra)[0],
                         "max_abs_diff": (g.luna - g.terra).abs().max()})
    save(pd.DataFrame(rank), "T11_sensitivity_platform_ranking.csv")
    return pd.DataFrame(rel), pd.DataFrame(rank)


def main():
    ds = pd.read_csv(D("analysis_dataset.csv"))
    on = ds[ds.on_topic == 1]
    core = on[on.is_post_event == 1]
    print(f"数据：{len(ds)} 条已标注，on_topic {len(on)}（事件后 {len(core)}，事件前 {len(on) - len(core)}）\n")

    f = funnel()
    print("T1 数据漏斗\n", f.to_string(index=False), "\n")
    keyword_precision()
    ot = on_topic_rates(ds)
    print("T3 事件后 on_topic 率\n", ot[["platform", "share", "ci_low", "ci_high", "group_total"]].to_string(index=False), "\n")

    comp = composition(core)
    for name in ("tone", "function"):
        overall, _, tests = comp[name]
        print(f"T4/T5 {name} 总体占比 + 平台差异检验（Cramér's V={tests.cramers_v_layer.iloc[0]:.3f}）")
        print(overall[["category", "share", "ci_low", "ci_high", "n_comments"]].merge(
            tests[["category", "p_holm", "p_perm_holm"]], on="category").to_string(index=False), "\n")

    pw, same_kw = robustness_composition(core)
    print("T4b 稳健性：评论等权 vs 帖子等权（总体）\n",
          pw[pw.group == "All"][["layer", "category", "share_comment_weighted", "share_post_weighted"]]
          .to_string(index=False), "\n")
    print("T5b 稳健性：仅'子弹库帛书'检索结果下的平台差异（Holm）\n",
          same_kw[["layer", "category", "test_used", "p_holm", "p_perm_holm", "platforms_tested"]].to_string(index=False), "\n")

    comp_t, tests_t = temporal(on)
    print("T8 阶段效应（控制平台，Holm 校正后 p）\n", tests_t[["layer", "category", "p_holm", "p_drop_top5_holm"]].to_string(index=False), "\n")

    e = engagement(on)
    print("T9 互动量（帖子内，相对'情感表态'/'中性理性'的点赞变化%）\n",
          e[["sample", "term", "pct_change", "p"]].to_string(index=False), "\n")

    try:
        rel, rank = sensitivity()
        print("T10 luna vs terra（3000 分层样本）\n", rel.to_string(index=False))
        print("\nT11 平台排序一致性\n", rank.to_string(index=False))
    except Exception as ex:
        print("敏感性分析未完成：", ex)


if __name__ == "__main__":
    main()
