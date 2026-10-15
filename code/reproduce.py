#!/usr/bin/env python3
"""reproduce.py: recomputes the headline results of the article from the de-identified data in ../data
and compares each with the value reported in the article. Run:  python code/reproduce.py
No raw text, credentials or network access are needed. Runtime is about one minute."""
import os, sys
import numpy as np, pandas as pd
import statsmodels.api as sm, statsmodels.formula.api as smf
from sklearn.metrics import cohen_kappa_score
from statsmodels.stats.multitest import multipletests

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
D = lambda f: pd.read_csv(os.path.join(ROOT, "data", f))
T = lambda f: pd.read_csv(os.path.join(ROOT, "results", "tables", f), encoding="utf-8-sig")
TONES = ["Joy & pride", "Indignation & grief", "Neutral/informational", "Skepticism & dissent"]
FUNCTIONS = ["Affective expression", "Restitution claims & state capacity", "Artifact knowledge & return details", "Historical & civilizational debate", "Institutional trust & distrust"]
MIN_POSTS, N_PERM, SEED = 10, 2000, 20260921
fails = []
def check(name, got, want, tol=0.005):
    ok = (got == want) if isinstance(want, (int, str, list, dict)) else abs(got - want) <= tol
    print(f"  [{'ok' if ok else 'MISMATCH'}] {name}: {got if isinstance(got, (int, str, list, dict)) else round(float(got), 4)}  (article: {want})")
    if not ok: fails.append(name)

# ---------------- corpus ----------------
c = D("comments.csv"); core = c[c.is_post_event == 1]; base = c[c.is_post_event == 0]; on = core[core.on_topic == 1]
print("Corpus")
check("comments coded", len(c), 19128); check("core corpus", len(core), 15660); check("baseline corpus", len(base), 3468)
check("on-topic, core", len(on), 6771); check("on-topic, baseline", int(base.on_topic.sum()), 982)
check("on-topic core by platform", on.platform.value_counts().sort_index().to_dict(), {"Bilibili": 2871, "Douyin": 1128, "Weibo": 699, "Xiaohongshu": 1383, "Zhihu": 690})
sizes = on.post_key.value_counts()
check("posts with on-topic comments after the return", len(sizes), 352)
check("share of five largest posts (%)", 100 * sizes.head(5).sum() / len(on), 41.34); check("share of twenty largest posts (%)", 100 * sizes.head(20).sum() / len(on), 68.56)
check("on-topic comments by phase", c[c.on_topic == 1].phase.value_counts().sort_index().tolist(), [355, 627, 4888, 609, 643, 631])

# ---------------- composition ----------------
print("Composition")
wb = on[on.platform == "Weibo"]
check("joy and pride, pooled (%)", 100 * (on.tone == "Joy & pride").mean(), 26.76); check("joy and pride, Weibo (%)", 100 * (wb.tone == "Joy & pride").mean(), 75.54)
check("affective expression, pooled (%)", 100 * (on.function == "Affective expression").mean(), 26.79); check("affective expression, Weibo (%)", 100 * (wb.function == "Affective expression").mean(), 68.96)
sh = (100 * on.groupby("platform").official_account.mean()).round(2).to_dict()
check("comments under official posts by platform (%)", sh, {"Bilibili": 79.59, "Douyin": 30.76, "Weibo": 85.98, "Xiaohongshu": 48.73, "Zhihu": 0.0})

# ---------------- platform tests (cluster-robust Wald and post-level permutation) ----------------
def usable(d, term):
    n = d.groupby(term).post_key.nunique(); return d[~d[term].isin(n[n < MIN_POSTS].index)]
def separated(d, col):
    g = d.groupby(col).y.agg(["sum", "count"]); return bool(((g["sum"] == 0) | (g["sum"] == g["count"])).any())
def cluster_wald(data, col, cat, term="platform"):
    d = usable(data.assign(y=(data[col] == cat).astype(int)), term)
    if d.y.sum() < 5 or separated(d, term): return np.nan
    res = smf.glm(f"y ~ C({term})", d, family=sm.families.Binomial()).fit(cov_type="cluster", cov_kwds={"groups": pd.factorize(d.post_key)[0]})
    names = [n for n in res.params.index if n.startswith(f"C({term})")]
    R = np.zeros((len(names), len(res.params)))
    for i, n in enumerate(names): R[i, list(res.params.index).index(n)] = 1
    return float(res.wald_test(R, scalar=True).pvalue)
def perm_test(data, col, cat, term="platform"):
    d = usable(data.assign(y=(data[col] == cat).astype(int)), term)
    post = d.groupby("post_order").agg(g=(term, "first"), y=("y", "sum"), n=("y", "size"))  # original post order, so the seeded draw is identical
    codes = pd.factorize(post.g)[0]; y, n = post.y.to_numpy(), post.n.to_numpy(); k, p0 = codes.max() + 1, y.sum() / n.sum()
    def stat(cc):
        ys, ns = np.bincount(cc, y, k), np.bincount(cc, n, k); e1 = ns * p0
        return (((ys - e1) ** 2) / e1 + ((ns - ys - (ns - e1)) ** 2) / (ns - e1)).sum()
    obs = stat(codes); rng = np.random.default_rng(SEED)
    return float((1 + (np.array([stat(rng.permutation(codes)) for _ in range(N_PERM)]) >= obs).sum()) / (1 + N_PERM))
print("Platform differences (compared with results/tables/T5_*.csv)")
for layer, cats, tab in (("tone", TONES, "T5_tone_platform_tests.csv"), ("function", FUNCTIONS, "T5_function_platform_tests.csv")):
    ref = T(tab).set_index("category")
    pw = [cluster_wald(on, layer, k) for k in cats]; pp = [perm_test(on, layer, k) for k in cats]
    hw, hp = multipletests(pw, method="holm")[1], multipletests(pp, method="holm")[1]
    for k, a, b in zip(cats, hw, hp):
        check(f"{layer} | {k} | permutation P (Holm)", b, float(ref.loc[k, "p_perm_holm"]), 1e-9)
        check(f"{layer} | {k} | Wald P (Holm), log10", np.log10(a), float(np.log10(ref.loc[k, "p_holm"])), 0.01)

# ---------------- reliability and validity ----------------
def kap(a, b): return cohen_kappa_score(a.astype(str), b.astype(str))
print("Human reliability, 150 unseen comments")
r = D("human_reliability_150.csv"); both = r[(r.on_topic_A == 1) & (r.on_topic_B == 1)]
check("kappa relevance", kap(r.on_topic_A, r.on_topic_B), 0.865, 0.0005); check("kappa tone", kap(both.tone_A, both.tone_B), 0.846, 0.0005); check("kappa function", kap(both.function_A, both.function_B), 0.917, 0.0005)
print("Model against the gold standard, 507 comments")
ml = D("model_labels.csv"); full = ml[(ml.model == "primary") & (ml.run == "full_v2")]
g = D("human_gold_507.csv").merge(full[["comment_key", "on_topic", "tone", "function"]], on="comment_key", how="left")
check("gold-standard comments", len(g), 507); check("coded identically / adjudicated", g.source.value_counts().to_dict(), {"coders agreed": 451, "adjudicated": 56})
bo = g[(g.on_topic == 1) & (g.on_topic_gold == 1)]
check("kappa relevance", kap(g.on_topic, g.on_topic_gold), 0.745, 0.0005); check("kappa tone", kap(bo.tone, bo.tone_gold), 0.659, 0.0005); check("kappa function", kap(bo.function, bo.function_gold), 0.757, 0.0005)
strat = lambda d, f: d.platform + "|" + d.on_topic.astype(str) + "|" + d[f].fillna("")
g["stratum"] = strat(g, "function"); pop = strat(full, "function").value_counts()
g["w"] = g.stratum.map(pop) / g.stratum.map(g.stratum.value_counts()); w = g.w.to_numpy()
tp = w[(g.on_topic == 1) & (g.on_topic_gold == 1)].sum()
check("relevance precision, reweighted (%)", 100 * tp / w[g.on_topic == 1].sum(), 87.23); check("relevance recall, reweighted (%)", 100 * tp / w[g.on_topic_gold == 1].sum(), 83.44)
mm, hh = (g.on_topic == 1) & (g.tone == "Skepticism & dissent"), (g.on_topic_gold == 1) & (g.tone_gold == "Skepticism & dissent")
check("skepticism precision, reweighted (%)", 100 * w[mm & hh].sum() / w[mm].sum(), 42.43)
check("skepticism share, model (%)", 100 * w[mm].sum() / w[g.on_topic == 1].sum(), 16.25); check("skepticism share, human (%)", 100 * w[hh].sum() / w[g.on_topic_gold == 1].sum(), 8.99)
print("Model stability")
def pair(a, b):
    m = a.merge(b, on="comment_key", suffixes=("_1", "_2")); bb = m[(m.on_topic_1 == 1) & (m.on_topic_2 == 1)]
    return kap(m.on_topic_1, m.on_topic_2), kap(bb.tone_1, bb.tone_2), kap(bb.function_1, bb.function_2), len(m)
run = lambda mo, ru: ml[(ml.model == mo) & (ml.run == ru)][["comment_key", "on_topic", "tone", "function"]]
k1 = pair(run("primary", "pilot_v2"), run("primary", "pilot_v2_retest")); k2 = pair(full[["comment_key", "on_topic", "tone", "function"]], run("second", "sens_v2"))
check("test-retest comments", k1[3], 499); check("test-retest kappa relevance", k1[0], 0.928, 0.0005); check("test-retest kappa tone", k1[1], 0.881, 0.0005); check("test-retest kappa function", k1[2], 0.858, 0.0005)
check("second-model comments", k2[3], 3000); check("second-model kappa relevance", k2[0], 0.794, 0.0005); check("second-model kappa tone", k2[1], 0.791, 0.0005); check("second-model kappa function", k2[2], 0.773, 0.0005)
print("Post-level relevance rule, 200 audited posts")
a = D("human_post_audit_200.csv")
check("rule precision (%)", 100 * a[a.rule_relevant == 1].human_relevant.mean(), 76.00); check("rule miss rate (%)", 100 * a[a.rule_relevant == 0].human_relevant.mean(), 1.00)

# ---------------- targeted validation ----------------
print("Targeted validation of skepticism and dissent, 193 comments")
tv = D("human_targeted_193.csv"); st = D("targeted_strata.csv")
tv["hit"] = ((tv.on_topic_final == 1) & (tv.tone_final == "Skepticism & dissent")).astype(int)
def corrected(pl):
    s = st[st.platform == pl].set_index("stratum").population; pi = s["model_skepticism"] / s.sum()
    d = tv[tv.platform == pl]; ppv = d[d.stratum == "model_skepticism"].hit.mean(); fo = d[d.stratum == "model_other_tone"].hit.mean()
    return 100 * pi, 100 * ppv, 100 * (pi * ppv + (1 - pi) * fo)
z = corrected("Zhihu")
check("comments", len(tv), 193); check("Zhihu confirmed of 40 flagged", int(tv[(tv.platform == "Zhihu") & (tv.stratum == "model_skepticism")].hit.sum()), 5)
check("Zhihu share by the model (%)", z[0], 35.65); check("Zhihu positive predictive value (%)", z[1], 12.50); check("Zhihu corrected share (%)", z[2], 8.75); check("Xiaohongshu corrected share (%)", corrected("Xiaohongshu")[2], 11.49)

print("\n" + ("All checks passed." if not fails else f"{len(fails)} MISMATCH(ES): " + "; ".join(fails)))
sys.exit(1 if fails else 0)
