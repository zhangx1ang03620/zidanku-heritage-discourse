#!/usr/bin/env python3
"""permutation_high_precision.py: repeats the three families of post-level permutation tests with 200,000 draws.
The analysis plan fixed 2,000 draws; at that number a Holm-corrected value near 0.05 carries visible Monte Carlo error.
This script quantifies that error. It does not replace the pre-specified tests.
Run:  python code/permutation_high_precision.py     (about 10 minutes; writes results/tables/T17_permutation_high_precision.csv)"""
import os
import numpy as np, pandas as pd
from statsmodels.stats.multitest import multipletests
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
N, SEED = 200_000, 20261012
TONES = ["Joy & pride", "Indignation & grief", "Neutral/informational", "Skepticism & dissent"]
FUNCTIONS = ["Affective expression", "Restitution claims & state capacity", "Artifact knowledge & return details", "Historical & civilizational debate", "Institutional trust & distrust"]
c = pd.read_csv(os.path.join(ROOT, "data", "comments.csv")); on = c[(c.is_post_event == 1) & (c.on_topic == 1)]

def perm(col, cat, term, strata=None):
    """Posts are kept intact; the labels of `term` are shuffled among posts (within levels of `strata` if given).
    Statistic: Pearson chi-square of group by category membership. Returns the P value and its Monte Carlo standard error."""
    d = on.assign(y=(on[col] == cat).astype(int))
    post = d.groupby("post_order").agg(g=(term, "first"), y=("y", "sum"), n=("y", "size"), **({"s": (strata, "first")} if strata else {}))
    codes = pd.factorize(post.g)[0]; y, n = post.y.to_numpy(), post.n.to_numpy(); k, p0 = codes.max() + 1, y.sum() / n.sum()
    idx = [np.where(post.s.to_numpy() == s)[0] for s in np.unique(post.s)] if strata else [np.arange(len(codes))]
    def stat(cc):
        ys, ns = np.bincount(cc, y, k), np.bincount(cc, n, k); e = ns * p0; ok = (e > 0) & (ns - e > 0)
        return (((ys - e) ** 2 / e)[ok] + ((e - ys) ** 2 / (ns - e))[ok]).sum()
    obs, rng, ge = stat(codes), np.random.default_rng(SEED), 0
    for _ in range(N):
        cc = codes.copy()
        for m in idx: cc[m] = rng.permutation(codes[m])
        ge += stat(cc) >= obs
    p = (1 + ge) / (1 + N)
    return p, float(np.sqrt(p * (1 - p) / N))

FAM = [("Platform difference", "platform", None, "within layer"), ("Platform difference, account type held fixed", "platform", "official_account", "across the nine categories"),
       ("Official versus other posts, platform held fixed", "official_account", "platform", "across the nine categories")]
rows = []
for name, term, strata, scope in FAM:
    res = [(layer, cat) + perm(layer, cat, term, strata) for layer, cats in (("tone", TONES), ("function", FUNCTIONS)) for cat in cats]
    df = pd.DataFrame(res, columns=["layer", "category", "p_raw", "mc_se_raw"])
    if scope == "within layer": df["p_holm"] = np.concatenate([multipletests(df[df.layer == l].p_raw, method="holm")[1] for l in ("tone", "function")])
    else: df["p_holm"] = multipletests(df.p_raw, method="holm")[1]
    df.insert(0, "test", name); df["holm_scope"] = scope; df["draws"] = N; rows.append(df); print(df.round(5).to_string(index=False))
pd.concat(rows).to_csv(os.path.join(ROOT, "results", "tables", "T17_permutation_high_precision.csv"), index=False)
