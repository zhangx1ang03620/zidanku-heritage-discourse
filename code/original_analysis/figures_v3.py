#!/usr/bin/env python3
"""
figures_v3.py —— 论文数据图 Figure 3–9（Figure 1、2 为流程示意图，另行制作）

每张图由多个子图组成，全部取自 tables/、data/ 和 db/ 中已有的结果；
少数面板是对已有数据的纯描述（帖子内评论数分布、评论长度、帖子层占比），不含新的检验。
输出 figures/v3/Figure{3..9}.pdf/.png。用法：python3 figures_v3.py [图号 ...]
"""
import os
import re
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import figures_v2 as V  # noqa: E402  配色、字体、面板标注等沿用 v2
from figures_v2 import (C, INK, INK2, MUTED, GRID, AXIS, SEQ, DIV, PLAT, PKEY, TONES, FUNCS, SHORT, ZH2EN, T, W, OUT, DBDIR,  # noqa: E402
                        panel, grid, dataset, gold_vs_model)

FIG = os.path.join(OUT, "figures", "v3")
os.makedirs(FIG, exist_ok=True)
CATS = TONES + FUNCS
EN2ZH = {v: k for k, v in ZH2EN.items()}
PCOL = dict(zip(PLAT, C))                                             # 平台固定配色
KW = {"五行令": "Wuxing ling (volume title)", "史密森尼": "Smithsonian", "子弹库帛书": "Zidanku silk manuscript",
      "战国帛书": "Warring States silk manuscript", "文物回归": "relic return", "文物返还": "relic restitution",
      "国宝回家": "national treasure comes home", "子弹库楚帛书": "Zidanku Chu silk manuscript", "文物追索": "relic recovery",
      "楚帛书": "Chu silk manuscript", "流失文物": "lost relics", "长沙帛书": "Changsha silk manuscript",
      "帛书归湘": "silk manuscript returns to Hunan"}


def save(fig, n):
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(FIG, f"Figure{n}.{ext}"), bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("Figure", n, "saved")


def note(ax, text, y=-0.3, x=0.0):
    ax.text(x, y, text, transform=ax.transAxes, fontsize=5.4, color=MUTED, va="top", ha="left", linespacing=1.25)


def heat(ax, M, cmap, vmin, vmax, fmt="{:.0f}", thresh=None, fs=6):
    im = ax.imshow(M, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")
    thresh = thresh if thresh is not None else vmin + 0.6 * (vmax - vmin)
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            v = M[i, j]
            if np.isfinite(v):
                dark = (abs(v) > thresh) if cmap is DIV else (v > thresh)
                ax.text(j, i, fmt.format(v), ha="center", va="center", fontsize=fs, color="white" if dark else INK)
    ax.set_xticks(np.arange(-0.5, M.shape[1]), minor=True)
    ax.set_yticks(np.arange(-0.5, M.shape[0]), minor=True)
    ax.grid(which="minor", color="white", lw=1.2)
    ax.tick_params(which="both", length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    return im


def core_with_author():
    ds = dataset()
    at = pd.read_csv(os.path.join(DBDIR, "post_author_type_binary.csv"), dtype=str)
    ds["pid"] = ds.post_key.str.split(":").str[-1]
    m = ds.merge(at[["platform", "pid", "official"]], on=["platform", "pid"], how="left")
    m["official"] = pd.to_numeric(m.official)
    return m


# ================================================================ Figure 3  corpus
def fig3():
    ds = dataset()
    core = ds[(ds.on_topic == 1) & (ds.is_post_event == 1)]
    fig = plt.figure(figsize=(W, 8.3))
    gs = fig.add_gridspec(3, 12, hspace=0.72, wspace=2.6, height_ratios=[1, 1.15, 0.95])
    r0 = gs[0, :].subgridspec(1, 2, width_ratios=[2, 1], wspace=0.5)
    r1 = gs[1, :].subgridspec(1, 3, width_ratios=[1.3, 1, 1], wspace=0.62)
    r2 = gs[2, :].subgridspec(1, 2, width_ratios=[2.6, 1], wspace=0.3)

    ax = fig.add_subplot(r0[0])
    f = T("T1_data_funnel.csv")
    stages = ["Retrieved", "After de-duplication", "Under event-relevant posts", "Core (post-event, with text)",
              "Baseline (pre-event, with text)", "On topic, core", "On topic, baseline"]
    y = np.arange(len(stages))[::-1]
    left = np.zeros(len(stages))
    for k, name in PKEY.items():
        v = f[k].to_numpy(float)
        ax.barh(y, v, left=left, color=PCOL[name], height=0.62, edgecolor="white", lw=0.8, label=name, zorder=3)
        left += v
    for yy, tot in zip(y, f.total):
        ax.text(tot + 1500, yy, f"{tot:,}", va="center", fontsize=6.3, color=INK2)
    ax.set_yticks(y, stages)
    ax.set_xlim(0, 158000)
    ax.set_xlabel("Comments")
    ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{int(v/1000)}k" if v else "0"))
    ax.legend(ncol=5, loc="lower right", bbox_to_anchor=(1.0, 1.0), handlelength=1, columnspacing=0.9, borderaxespad=0.2)
    grid(ax)
    panel(ax, "a", x=-0.46)

    ax = fig.add_subplot(r0[1])
    r = T("T3_on_topic_rate_by_platform.csv").set_index("platform").loc[PLAT]
    yy = np.arange(5)[::-1]
    for y0, p in zip(yy, PLAT):
        ax.plot([r.ci_low[p] * 100, r.ci_high[p] * 100], [y0, y0], color=PCOL[p], lw=1.2, zorder=2)
        ax.plot([r.share[p] * 100], [y0], "o", color=PCOL[p], ms=5, mec="white", mew=0.8, zorder=3)
        ax.text(r.share[p] * 100, y0 + 0.3, f"{r.share[p]*100:.1f}", ha="center", fontsize=6.3, color=INK2)
    ax.set_yticks(yy, PLAT)
    ax.set_xlim(0, 80)
    ax.set_ylim(-0.6, 4.75)
    ax.set_xlabel("Core comments coded on topic (%)")
    grid(ax)
    panel(ax, "b", x=-0.42)
    note(ax, "Whiskers, 95% cluster bootstrap interval over posts.", y=-0.3)

    ax = fig.add_subplot(r1[0])
    t2 = T("T2_keyword_precision.csv")
    order = t2.groupby("keyword").precision.mean().sort_values(ascending=False).index
    M = t2.pivot(index="keyword", columns="platform", values="precision").reindex(index=order, columns=PLAT) * 100
    N = t2.pivot(index="keyword", columns="platform", values="n_posts").reindex(index=order, columns=PLAT)
    ax.set_facecolor("#f0efec")
    heat(ax, M.to_numpy(), SEQ, 0, 100, fs=5.8)
    ax.set_xticks(range(5), PLAT, rotation=30, ha="right", rotation_mode="anchor")
    ax.set_yticks(range(len(order)), [KW[k] for k in order], fontsize=6)
    panel(ax, "c", x=-0.95, y=1.03)
    note(ax, "Share of retrieved posts judged relevant by the keyword rule (%), by search term.\nGrey, term not searched on that platform.", y=-0.3, x=-0.9)

    ax = fig.add_subplot(r1[1])
    vc_all = core.post_key.value_counts().to_numpy()
    for p in PLAT:
        vc = core[core.platform_en == p].post_key.value_counts().to_numpy()
        ax.plot(np.arange(1, len(vc) + 1) / len(vc) * 100, np.cumsum(vc) / vc.sum() * 100, color=PCOL[p], lw=1.1, label=p, zorder=3)
    ax.plot(np.arange(1, len(vc_all) + 1) / len(vc_all) * 100, np.cumsum(vc_all) / vc_all.sum() * 100, color=INK, lw=1.6, ls=(0, (4, 2)),
            label="All", zorder=4)
    ax.plot([0, 100], [0, 100], color=AXIS, lw=0.7)
    ax.set_xlabel("Posts, ranked by size (% of posts)")
    ax.set_ylabel("Cumulative share of\non-topic comments (%)")
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 101)
    ax.legend(loc="lower right", fontsize=5.6, handlelength=1.4, labelspacing=0.25)
    ax.text(40, 70, f"top 5 posts: {vc_all[:5].sum()/vc_all.sum()*100:.1f}%\ntop 20 posts: {vc_all[:20].sum()/vc_all.sum()*100:.1f}%", fontsize=6, color=INK2,
            va="top")
    grid(ax, "y")
    panel(ax, "d", x=-0.4, y=1.03)

    ax = fig.add_subplot(r1[2])
    rng = np.random.default_rng(1)
    for i, p in enumerate(PLAT):
        vc = core[core.platform_en == p].post_key.value_counts().to_numpy()
        ax.scatter(i + rng.uniform(-0.24, 0.24, len(vc)), vc, s=5, color=PCOL[p], alpha=0.55, lw=0, zorder=2)
        q1, md, q3 = np.percentile(vc, [25, 50, 75])
        ax.plot([i - 0.3, i + 0.3], [md, md], color=INK, lw=1.3, zorder=4)
        ax.plot([i, i], [q1, q3], color=INK, lw=0.8, zorder=3)
        ax.text(i, 2100, f"{len(vc)}", ha="center", fontsize=5.8, color=INK2)
    ax.set_yscale("log")
    ax.set_ylim(0.8, 5500)
    ax.set_xticks(range(5), [p[:5] + "." if len(p) > 7 else p for p in PLAT], rotation=30, ha="right", rotation_mode="anchor")
    ax.set_ylabel("On-topic comments per post")
    grid(ax, "y")
    panel(ax, "e", x=-0.42, y=1.03)
    note(ax, "Bar, median; line, interquartile range;\nnumbers, posts per platform.", y=-0.3)

    ax = fig.add_subplot(r2[0])
    d = pd.read_csv(os.path.join(OUT, "data", "daily_volume_all_relevant.csv"))
    d["date"] = pd.to_datetime(d["date"])
    wk = d[(d.date >= "2024-06-01") & (d.date <= "2026-03-31")].set_index("date").n.resample("W").sum()
    bounds = pd.to_datetime(["2024-06-01", "2024-09-01", "2025-05-16", "2025-06-01", "2025-10-01", "2025-12-01", "2026-03-31"])
    names = ["P1", "P2 documentary\n& advocacy", "", "P4 consolidation", "P5", "P6 long tail"]
    for i in range(6):
        if i % 2 == 1:
            ax.axvspan(bounds[i], bounds[i + 1], color="#f0efec", zorder=0, lw=0)
        ax.text(bounds[i] + (bounds[i + 1] - bounds[i]) / 2, 1.02, names[i], transform=ax.get_xaxis_transform(), ha="center", va="bottom",
                fontsize=5.8, color=MUTED)
    ax.fill_between(wk.index, wk.values, 0.5, color=C[0], alpha=0.22, lw=0, zorder=2)
    ax.plot(wk.index, wk.values, color=C[0], lw=1.1, zorder=3)
    ax.set_yscale("log")
    ax.set_ylim(0.8, wk.max() * 7)
    for dt, lab in (("2025-05-16", "16 May 2025\nreturn (P3)"), ("2025-10-13", "13 Oct 2025\nenters Hunan Museum")):
        ax.axvline(pd.Timestamp(dt), color=INK2, lw=0.6, ls=(0, (3, 2)), zorder=1)
        ax.text(pd.Timestamp(dt) + pd.Timedelta(days=6), wk.max() * 5, lab, fontsize=5.8, color=INK2, va="top")
    ax.set_ylabel("Comments per week\n(log scale)")
    ax.xaxis.set_major_locator(matplotlib.dates.MonthLocator(bymonth=(1, 4, 7, 10)))
    ax.xaxis.set_major_formatter(matplotlib.dates.DateFormatter("%b\n%Y"))
    grid(ax, "y")
    panel(ax, "f", x=-0.1, y=1.12)

    ax = fig.add_subplot(r2[1])
    t7 = T("T7b_phase_by_platform_counts.csv").set_index("platform_en").loc[PLAT]
    sh = t7.div(t7.sum(0), axis=1) * 100
    bottom = np.zeros(6)
    for p in PLAT:
        ax.bar(range(6), sh.loc[p], bottom=bottom, color=PCOL[p], width=0.7, edgecolor="white", lw=0.7, zorder=3)
        bottom += sh.loc[p].to_numpy()
    for i, n in enumerate(t7.sum(0)):
        ax.text(i, 102, f"{int(n):,}", ha="center", fontsize=5.4, color=INK2)
    ax.set_xticks(range(6), ["P1", "P2", "P3", "P4", "P5", "P6"])
    ax.set_ylim(0, 110)
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.set_ylabel("Platform mix of on-topic\ncomments (%)")
    panel(ax, "g", x=-0.42, y=1.12)
    note(ax, "Colours as in a; numbers, comments per phase.", y=-0.22)
    save(fig, 3)


# ================================================================ Figure 4  reliability & validity
def fig4():
    d = gold_vs_model()
    d["h_on"] = (d.human_on_topic == "是").astype(int)
    fig = plt.figure(figsize=(W, 8.6))
    gs = fig.add_gridspec(3, 12, hspace=0.75, wspace=3.2, height_ratios=[1, 1, 0.9])
    r0 = gs[0, :].subgridspec(1, 3, width_ratios=[1.15, 1, 1], wspace=1.05)
    r1 = gs[1, :].subgridspec(1, 3, width_ratios=[1.3, 1, 0.9], wspace=0.62)
    r2 = gs[2, :].subgridspec(1, 3, width_ratios=[0.8, 1, 0.9], wspace=0.7)

    ax = fig.add_subplot(r0[0])
    rel = T("reliability_pilot_v2.csv")
    s3 = T("T10_luna_vs_terra_sens3000.csv").set_index("field").kappa
    rows = [("Human coder A vs B\n(150 unseen comments)", [(0.865, 0.780, 0.946), (0.846, 0.734, 0.935), (0.917, 0.828, 0.980)]),
            ("Primary model,\ntest–retest (n = 499)", [(float(rel.Kappa[i]), None, None) for i in (0, 1, 2)]),
            ("Primary vs second\nmodel (n = 3,000)", [(s3["on_topic"], None, None), (s3["tone"], None, None), (s3["function"], None, None)]),
            ("Primary model vs\ngold standard (n = 507)", [(0.745, None, None), (0.659, None, None), (0.757, None, None)])]
    labs = ["Relevance", "Tone", "Function"]
    for i, (name, vals) in enumerate(rows):
        y0 = len(rows) - 1 - i
        for j, (k, lo, hi) in enumerate(vals):
            yy = y0 + (1 - j) * 0.22
            if lo is not None:
                ax.plot([lo, hi], [yy, yy], color=C[j], lw=1, zorder=2)
            ax.plot([k], [yy], "o", color=C[j], ms=4.3, mec="white", mew=0.8, zorder=3, label=labs[j] if i == 0 else None)
            ax.text(1.03, yy, f"{k:.3f}", fontsize=5.8, color=INK2, va="center", ha="left", transform=ax.get_yaxis_transform())
    ax.axvline(0.6, color=INK2, lw=0.6, ls=(0, (3, 2)), zorder=1)
    ax.set_yticks(range(len(rows)), [r[0] for r in rows][::-1])
    ax.set_xlim(0.3, 1.0)
    ax.set_xticks([0.4, 0.6, 0.8, 1.0])
    ax.set_ylim(-0.5, len(rows) - 0.3)
    ax.set_xlabel("Cohen's κ")
    ax.legend(ncol=3, loc="lower left", bbox_to_anchor=(-0.35, 1.0), handletextpad=0.2, columnspacing=0.9)
    grid(ax)
    panel(ax, "a", x=-0.78, y=1.12)

    both = d[(d.m_on == 1) & (d.h_on == 1)]
    for k, (mc, hc, cats, ttl, sl, letter) in enumerate((("m_tone", "human_tone", TONES, "Tone", r0[1], "b"),
                                                          ("m_fn", "human_function", FUNCS, "Function", r0[2], "c"))):
        ax = fig.add_subplot(sl)
        zh = [EN2ZH[c] for c in cats]
        cm = pd.crosstab(both[mc], both[hc]).reindex(index=zh, columns=zh, fill_value=0).to_numpy().astype(float)
        rn = cm / cm.sum(1, keepdims=True)
        ax.imshow(rn, cmap=SEQ, vmin=0, vmax=1)
        for i in range(len(cats)):
            for j in range(len(cats)):
                ax.text(j, i, int(cm[i, j]), ha="center", va="center", fontsize=6, color="white" if rn[i, j] > 0.55 else INK)
        ab = [SHORT[c].split(" ")[0][:6] + "." for c in cats]
        ax.set_xticks(range(len(cats)), ab, rotation=35, ha="right", rotation_mode="anchor", fontsize=5.6)
        ax.set_yticks(range(len(cats)), [SHORT[c] for c in cats], fontsize=5.6)
        ax.tick_params(length=0)
        for s in ax.spines.values():
            s.set_visible(False)
        ax.set_xlabel("Gold standard")
        ax.set_ylabel("Model")
        ax.set_title(f"{ttl} (n = {int(cm.sum())})", loc="left", fontsize=6.5, color=INK2, pad=3)
        panel(ax, letter, x=-0.95, y=1.12)

    ax = fig.add_subplot(r1[0])
    v = T("T15_model_validity_weighted.csv").set_index("metric")
    ypos = {c: len(CATS) - 1 - i + (0.5 if c in TONES else 0) for i, c in enumerate(CATS)}
    for i, c in enumerate(CATS):
        layer = "tone" if c in TONES else "function"
        m, h = v.loc[f"{layer}|{EN2ZH[c]}|share_model"], v.loc[f"{layer}|{EN2ZH[c]}|share_human"]
        y0 = ypos[c]
        ax.plot([m.estimate * 100, h.estimate * 100], [y0, y0], color=AXIS, lw=1.2, zorder=1)
        ax.plot([h.ci_low * 100, h.ci_high * 100], [y0, y0], color=C[1], lw=2.6, alpha=0.3, solid_capstyle="round", zorder=2)
        ax.plot([m.estimate * 100], [y0], "o", color=C[0], ms=4.6, mec="white", mew=0.8, zorder=3, label="Model-assigned" if i == 0 else None)
        ax.plot([h.estimate * 100], [y0], "D", color=C[1], ms=4, mec="white", mew=0.8, zorder=3, label="Human-based (95% CI)" if i == 0 else None)
    ax.axhline(4.75, color=GRID, lw=0.6)
    ax.set_yticks(list(ypos.values()), [SHORT[c] for c in CATS])
    ax.set_xlim(0, 55)
    ax.set_xlabel("Share of on-topic comments (%)")
    ax.legend(loc="lower left", bbox_to_anchor=(-0.35, 1.0), ncol=2, handletextpad=0.3, columnspacing=0.9)
    grid(ax)
    panel(ax, "d", x=-0.78, y=1.1)

    ax = fig.add_subplot(r1[1])
    for i, c in enumerate(CATS):
        layer = "tone" if c in TONES else "function"
        p, r = v.loc[f"{layer}|{EN2ZH[c]}|precision"], v.loc[f"{layer}|{EN2ZH[c]}|recall"]
        col = C[1] if c in TONES else C[2]
        ax.plot([p.ci_low * 100, p.ci_high * 100], [r.estimate * 100] * 2, color=col, lw=0.6, alpha=0.6)
        ax.plot([p.estimate * 100] * 2, [r.ci_low * 100, min(r.ci_high, 1) * 100], color=col, lw=0.6, alpha=0.6)
        ax.plot([p.estimate * 100], [r.estimate * 100], "o" if c in TONES else "s", color=col, ms=4.3, mec="white", mew=0.7, zorder=3)
        ax.annotate(str(i + 1), (p.estimate * 100, r.estimate * 100), xytext=(3, 3), textcoords="offset points", fontsize=5.6, color=INK2)
    p, r = v.loc["on_topic|precision"], v.loc["on_topic|recall"]
    ax.plot([p.estimate * 100], [r.estimate * 100], "*", color=C[0], ms=8, mec="white", mew=0.6, zorder=4)
    ax.annotate("relevance", (p.estimate * 100, r.estimate * 100), xytext=(-4, 5), textcoords="offset points", fontsize=5.6, color=INK2, ha="right")
    ax.plot([20, 100], [20, 100], color=AXIS, lw=0.6)
    ax.set_xlim(20, 102)
    ax.set_ylim(20, 102)
    ax.set_xlabel("Precision (%)")
    ax.set_ylabel("Recall (%)")
    grid(ax, "both")
    panel(ax, "e", x=-0.36, y=1.1)
    note(ax, "Circles, tone; squares, function. " + "; ".join(f"{i+1} {SHORT[c]}" for i, c in enumerate(CATS[:5])) + ";\n" +
         "; ".join(f"{i+6} {SHORT[c]}" for i, c in enumerate(CATS[5:])) + ". Stratum-weighted, 95% CI.", y=-0.3, x=-0.3)

    ax = fig.add_subplot(r1[2])
    rng = np.random.default_rng(20261009)
    yy = np.arange(5)[::-1]
    for y0, (pk, pn) in zip(yy, [(k, PKEY[k]) for k in ("bilibili", "douyin", "xhs", "weibo", "zhihu")]):
        g = d[d.platform == pk]
        k0 = cohen_kappa_score(g.m_on, g.h_on)
        bs = []
        for _ in range(1000):
            s = g.iloc[rng.integers(0, len(g), len(g))]
            if s.m_on.nunique() > 1 or s.h_on.nunique() > 1:
                bs.append(cohen_kappa_score(s.m_on, s.h_on))
        lo, hi = np.percentile(bs, [2.5, 97.5])
        ax.plot([lo, hi], [y0, y0], color=PCOL[pn], lw=1.2)
        ax.plot([k0], [y0], "o", color=PCOL[pn], ms=5, mec="white", mew=0.8, zorder=3)
        ax.text(1.02, y0, f"{k0:.2f}  (n={len(g)})", fontsize=5.8, color=INK2, va="center", transform=ax.get_yaxis_transform())
    ax.axvline(0.6, color=INK2, lw=0.6, ls=(0, (3, 2)))
    ax.set_yticks(yy, [PKEY[k] for k in ("bilibili", "douyin", "xhs", "weibo", "zhihu")])
    ax.set_xlim(0.2, 1.0)
    ax.set_ylim(-0.6, 4.6)
    ax.set_xlabel("κ, relevance,\nmodel vs gold standard")
    grid(ax)
    panel(ax, "f", x=-0.55, y=1.1)

    ax = fig.add_subplot(r2[0])
    M = np.array([[76, 24], [1, 99]], float)
    ax.imshow(M / M.sum(1, keepdims=True), cmap=SEQ, vmin=0, vmax=1)
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{int(M[i, j])}", ha="center", va="center", fontsize=8, color="white" if M[i, j] > 55 else INK)
    ax.set_xticks([0, 1], ["Relevant", "Not relevant"])
    ax.set_yticks([0, 1], ["Rule:\nrelevant", "Rule:\nnot relevant"])
    ax.set_xlabel("Human audit of 200 posts")
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    panel(ax, "g", x=-0.55, y=1.08)
    note(ax, "Precision 76.00%, miss rate 1.00%; κ = 0.750.", y=-0.42, x=-0.5)

    ax = fig.add_subplot(r2[1])
    s = pd.read_csv(os.path.join(OUT, "data", "sens_shares_luna_vs_terra.csv"))
    pv = s.pivot_table(index=["layer", "platform", "category"], columns="model", values="share").reset_index()
    for layer, col, mk in (("tone", C[1], "o"), ("function", C[2], "s")):
        g = pv[pv.layer == layer]
        ax.scatter(g.luna * 100, g.terra * 100, s=14, color=col, marker=mk, edgecolor="white", lw=0.5, zorder=3, label=layer.capitalize())
    ax.plot([0, 80], [0, 80], color=AXIS, lw=0.7)
    wst = pv.assign(dv=(pv.luna - pv.terra).abs()).sort_values("dv").iloc[-1]
    ax.annotate(f"{PKEY[wst.platform]}, {SHORT[ZH2EN[wst.category]]}", (wst.luna * 100, wst.terra * 100), xytext=(4, -16), textcoords="offset points",
                fontsize=5.6, color=INK2, ha="center", arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.5))
    rho = np.corrcoef(pv.luna, pv.terra)[0, 1]
    ax.text(3, 74, f"r = {rho:.3f}\n45 platform × category shares", fontsize=5.8, color=INK2, va="top")
    ax.set_xlim(0, 80)
    ax.set_ylim(0, 80)
    ax.set_xlabel("Primary model, share (%)")
    ax.set_ylabel("Second model, share (%)")
    ax.legend(loc="lower right", handletextpad=0.2)
    grid(ax, "both")
    panel(ax, "h", x=-0.3, y=1.08)

    ax = fig.add_subplot(r2[2])
    src = d.groupby("来源").apply(lambda g: pd.Series({"agree": (g.m_on == g.h_on).mean() * 100, "n": len(g)}))
    order = ["两人一致", "裁决"]
    ax.bar([0, 1], src.loc[order, "agree"], color=[C[0], MUTED], width=0.55, zorder=3)
    for i, k in enumerate(order):
        ax.text(i, src.loc[k, "agree"] + 1.5, f"{src.loc[k, 'agree']:.1f}%", ha="center", fontsize=6.3, color=INK2)
    ax.set_xticks([0, 1], [f"Coders identical\n(n = {int(src.loc['两人一致', 'n'])})", f"Adjudicated\n(n = {int(src.loc['裁决', 'n'])})"])
    ax.set_ylim(0, 105)
    ax.set_ylabel("Model agrees with gold\nstandard on relevance (%)")
    grid(ax, "y")
    panel(ax, "i", x=-0.5, y=1.08)
    save(fig, 4)


# ================================================================ Figure 5  composition
def fig5():
    ds = dataset()
    core = ds[(ds.on_topic == 1) & (ds.is_post_event == 1)]
    fig = plt.figure(figsize=(W, 8.8))
    gs = fig.add_gridspec(3, 12, hspace=0.7, wspace=3.0, height_ratios=[1, 1.1, 0.85])
    r0 = gs[0, :].subgridspec(1, 2, wspace=0.3)
    r1 = gs[1, :].subgridspec(1, 2, width_ratios=[1, 1.2], wspace=0.62)
    r2 = gs[2, :].subgridspec(1, 3, width_ratios=[1, 1, 1.15], wspace=0.55)
    v = T("T15_model_validity_weighted.csv").set_index("metric").estimate
    groups = ["All"] + PLAT
    shares = {}
    for k, (fname, cats, letter, layer, sl) in enumerate((("T4_tone_by_platform.csv", TONES, "a", "tone", r0[0]),
                                                           ("T4_function_by_platform.csv", FUNCS, "b", "function", r0[1]))):
        ax = fig.add_subplot(sl)
        t = T(fname)
        shares[layer] = t
        y = np.arange(len(groups))[::-1].astype(float)
        y[0] += 0.35
        left = np.zeros(len(groups))
        for i, c in enumerate(cats):
            s = t[t.category == c].set_index("group").share.reindex(groups).to_numpy() * 100
            ax.barh(y, s, left=left, color=C[i], height=0.62, edgecolor="white", lw=1, zorder=3, label=SHORT[c])
            for yy, l0, w in zip(y, left, s):
                if w >= 8:
                    ax.text(l0 + w / 2, yy, f"{w:.0f}", ha="center", va="center", fontsize=5.8, color=INK)
            left += s
        cum = 0
        for c in cats[:-1]:
            cum += v[f"{layer}|{EN2ZH[c]}|share_human"] * 100
            ax.plot([cum], [y[0] - 0.47], marker="^", color=INK, ms=3.5, zorder=5, clip_on=False)
        ax.set_yticks(y, groups)
        ax.set_xlim(0, 100)
        ax.set_xlabel("Share of on-topic core comments (%)")
        ax.legend(ncol=2 if len(cats) == 4 else 3, loc="lower left", bbox_to_anchor=(-0.22, 1.0), handlelength=1, columnspacing=0.8, handletextpad=0.4,
                  fontsize=6)
        ax.spines["left"].set_visible(False)
        ax.tick_params(axis="y", length=0)
        panel(ax, letter, x=-0.24, y=1.22)
        if k == 0:
            note(ax, "▲ category boundaries implied by the human-based estimates (pooled corpus).", y=-0.24, x=-0.2)

    ax = fig.add_subplot(r1[0])
    t6 = T("T6_tone_x_function.csv")
    R = t6.pivot(index="function_en", columns="tone_en", values="adj_std_residual").reindex(index=FUNCS, columns=TONES)
    N = t6.pivot(index="function_en", columns="tone_en", values="n").reindex(index=FUNCS, columns=TONES)
    im = ax.imshow(R.to_numpy(), cmap=DIV, vmin=-46, vmax=46, aspect="auto")
    for i in range(5):
        for j in range(4):
            r = R.iloc[i, j]
            ax.text(j, i - 0.12, f"{r:+.1f}", ha="center", va="center", fontsize=6.3, color="white" if abs(r) > 28 else INK)
            ax.text(j, i + 0.22, f"{int(N.iloc[i, j]):,}", ha="center", va="center", fontsize=5.2, color="white" if abs(r) > 28 else INK2)
    ax.set_xticks(range(4), [SHORT[c].replace(" & ", " &\n").replace(" / ", " /\n") for c in TONES], fontsize=5.8)
    ax.set_yticks(range(5), [SHORT[c].replace(" ", "\n", 1) for c in FUNCS], fontsize=5.8)
    ax.xaxis.tick_top()
    ax.set_xticks(np.arange(-0.5, 4), minor=True)
    ax.set_yticks(np.arange(-0.5, 5), minor=True)
    ax.grid(which="minor", color="white", lw=1.5)
    ax.tick_params(which="both", length=0)
    for s_ in ax.spines.values():
        s_.set_visible(False)
    panel(ax, "c", x=-0.3, y=1.12)
    note(ax, "Adjusted standardized residual (upper) and count (lower).\nBlue, more than expected; orange, fewer.", y=-0.04, x=-0.28)

    ax = fig.add_subplot(r1[1])
    dev = []
    for layer, cats in (("tone", TONES), ("function", FUNCS)):
        t = shares[layer].pivot(index="category", columns="group", values="share").reindex(cats)
        dev.append((t[PLAT].sub(t["All"], axis=0) * 100))
    Dv = pd.concat(dev)
    heat(ax, Dv.to_numpy(), DIV, -50, 50, fmt="{:+.0f}", thresh=30, fs=6)
    ax.set_xticks(range(5), PLAT, fontsize=6)
    ax.xaxis.tick_top()
    ax.set_yticks(range(9), [SHORT[c] for c in CATS], fontsize=6)
    ax.axhline(3.5, color=INK2, lw=0.8)
    panel(ax, "d", x=-0.5, y=1.12)
    note(ax, "Platform share minus pooled share (percentage points). Blue, above the pooled share; orange, below.", y=-0.04, x=-0.4)

    ax = fig.add_subplot(r2[0])
    pw = T("T4b_robust_post_weighted.csv")
    pw = pw[pw.group != "All"]
    for layer, col, mk in (("tone", C[1], "o"), ("function", C[2], "s")):
        g = pw[pw.layer == layer]
        ax.scatter(g.share_comment_weighted * 100, g.share_post_weighted * 100, s=13, color=col, marker=mk, edgecolor="white", lw=0.5, zorder=3,
                   label=layer.capitalize())
    ax.plot([0, 80], [0, 80], color=AXIS, lw=0.7)
    ok = pw.dropna(subset=["share_post_weighted"])
    ax.text(3, 76, f"r = {np.corrcoef(ok.share_comment_weighted, ok.share_post_weighted)[0, 1]:.3f}", fontsize=6, color=INK2, va="top")
    ax.set_xlim(0, 80)
    ax.set_ylim(0, 80)
    ax.set_xlabel("Comment-weighted share (%)")
    ax.set_ylabel("Post-weighted share (%)")
    ax.legend(loc="lower right", handletextpad=0.2)
    grid(ax, "both")
    panel(ax, "e", x=-0.32, y=1.08)

    ax = fig.add_subplot(r2[1])
    k5 = T("T5c_robust_same_keyword_function_shares.csv")
    t4 = shares["function"]
    m = k5.merge(t4[["group", "category", "share"]], on=["group", "category"], suffixes=("_kw", "_all"))
    for p in PLAT:
        g = m[m.group == p]
        ax.scatter(g.share_all * 100, g.share_kw * 100, s=14, color=PCOL[p], edgecolor="white", lw=0.5, zorder=3, label=p)
    ax.plot([0, 80], [0, 80], color=AXIS, lw=0.7)
    ax.set_xlim(0, 80)
    ax.set_ylim(0, 80)
    ax.set_xlabel("All search terms, function share (%)")
    ax.set_ylabel("Single shared term only (%)")
    ax.legend(loc="upper left", fontsize=5.4, handletextpad=0.1, labelspacing=0.2)
    grid(ax, "both")
    panel(ax, "f", x=-0.32, y=1.08)

    ax = fig.add_subplot(r2[2])
    data = [np.log10(core[core.function_en == c].text_len.clip(lower=1)) for c in FUNCS]
    parts = ax.violinplot(data, positions=range(5), widths=0.8, showextrema=False)
    for b, col in zip(parts["bodies"], C):
        b.set_facecolor(col)
        b.set_alpha(0.45)
        b.set_edgecolor("none")
    for i, x in enumerate(data):
        q1, md, q3 = np.percentile(x, [25, 50, 75])
        ax.plot([i, i], [q1, q3], color=INK, lw=1.6, zorder=3)
        ax.plot([i], [md], "o", color="white", ms=3, mec=INK, mew=0.8, zorder=4)
        ax.text(i, 3.05, f"{10**md:.0f}", ha="center", fontsize=5.8, color=INK2)
    ax.set_xticks(range(5), ["Affective", "Restitution", "Artifact", "Historical", "Institutional"], fontsize=5.2, rotation=30, ha="right", rotation_mode="anchor")
    ax.set_yticks([0, 1, 2, 3], ["1", "10", "100", "1,000"])
    ax.set_ylim(0, 3.25)
    ax.set_ylabel("Comment length (characters)")
    grid(ax, "y")
    panel(ax, "g", x=-0.32, y=1.08)
    note(ax, "Numbers, median length.", y=-0.3)
    save(fig, 5)


# ================================================================ Figure 6  platform vs poster
def fig6():
    fig = plt.figure(figsize=(W, 8.6))
    gs = fig.add_gridspec(3, 12, hspace=0.7, wspace=3.0, height_ratios=[1, 1, 0.95])
    r0 = gs[0, :].subgridspec(1, 3, width_ratios=[1, 1, 0.8], wspace=0.75)
    r1 = gs[1, :].subgridspec(1, 3, width_ratios=[1, 1, 0.8], wspace=0.75)
    yy = np.arange(5)[::-1]

    ax = fig.add_subplot(r0[0])
    for k, (fname, cat) in enumerate((("T4_tone_by_platform.csv", "Joy & pride"), ("T4_function_by_platform.csv", "Affective expression"))):
        t = T(fname)
        t = t[t.category == cat].set_index("group")
        r = t.loc[PLAT]
        off = 0.16 if k == 0 else -0.16
        ax.errorbar(r.share * 100, yy + off, xerr=[(r.share - r.ci_low) * 100, (r.ci_high - r.share) * 100], fmt="o" if k == 0 else "s", color=C[k],
                    ms=4.3, elinewidth=1, mec="white", mew=0.8, zorder=3, label=SHORT[cat])
        ax.axvline(t.loc["All", "share"] * 100, color=C[k], lw=0.8, ls=(0, (3, 2)), zorder=1)
    ax.text(28.5, 4.55, "pooled", fontsize=5.8, color=MUTED)
    ax.set_yticks(yy, PLAT)
    ax.set_xlim(0, 100)
    ax.set_ylim(-0.6, 4.8)
    ax.set_xlabel("Share of on-topic comments (%)")
    ax.legend(loc="lower left", bbox_to_anchor=(-0.05, 1.0), ncol=2, fontsize=5.8, columnspacing=0.8, handletextpad=0.2)
    grid(ax)
    panel(ax, "a", x=-0.45, y=1.12)

    ax = fig.add_subplot(r0[1])
    m = core_with_author()
    core = m[(m.on_topic == 1) & (m.is_post_event == 1)]
    a = T("T14a_author_type_by_platform.csv").set_index("platform_en").loc[PLAT]
    sh = core.groupby("platform_en").official.mean().reindex(PLAT) * 100
    ax.barh(yy, sh, color=[PCOL[p] for p in PLAT], height=0.58, zorder=3)
    for y0, v, p in zip(yy, sh, PLAT):
        ax.text(v + 2, y0, f"{v:.1f}%  ({int(a.loc[p, 'n_official_posts'])}/{int(a.loc[p, 'n_posts'])} posts)", va="center", fontsize=5.8, color=INK2)
    ax.set_yticks(yy, PLAT)
    ax.set_xlim(0, 160)
    ax.set_xticks([0, 50, 100])
    ax.set_xlabel("On-topic comments under\nofficial-account posts (%)")
    grid(ax)
    panel(ax, "b", x=-0.45)

    ax = fig.add_subplot(r0[2])
    t5 = pd.concat([T("T5_tone_platform_tests.csv"), T("T5_function_platform_tests.csv")]).set_index("category")
    b = T("T14b_platform_tests_controlling_author_type.csv").set_index("category")
    P = np.column_stack([t5.loc[CATS, "p_holm"], t5.loc[CATS, "p_perm_holm"], b.loc[CATS, "p_wald_holm"], b.loc[CATS, "p_perm_holm"]])
    L = -np.log10(np.clip(P, 1e-4, 1))
    ax.imshow(L, cmap=SEQ, vmin=0, vmax=4, aspect="auto")
    for i in range(9):
        for j in range(4):
            ax.text(j, i, "<.001" if P[i, j] < .001 else f"{P[i, j]:.3f}".lstrip("0"), ha="center", va="center", fontsize=5.2,
                    color="white" if L[i, j] > 2.4 else INK)
    ax.set_xticks(range(4), ["Wald", "Perm.", "Wald", "Perm."], fontsize=5.8)
    ax.xaxis.tick_top()
    ax.text(0.5, -1.75, "unadjusted", ha="center", fontsize=5.6, color=MUTED)
    ax.text(2.5, -1.75, "account fixed", ha="center", fontsize=5.6, color=MUTED)
    ax.set_yticks(range(9), [SHORT[c] for c in CATS], fontsize=5.8)
    ax.axvline(1.5, color="white", lw=2.5)
    ax.axhline(3.5, color="white", lw=2.5)
    ax.tick_params(length=0)
    for s_ in ax.spines.values():
        s_.set_visible(False)
    panel(ax, "c", x=-1.0, y=1.2)
    note(ax, "Holm-corrected P, platform term.", y=-0.04, x=-1.0)

    s = T("T14c2_shares_within_author_type.csv")
    for k, (cat, sl, letter) in enumerate((("Joy & pride", r1[0], "d"), ("Affective expression", r1[1], "e"))):
        ax = fig.add_subplot(sl)
        ss = s[s.category == cat]
        for j, (sub, lab) in enumerate((("官方", "Official accounts"), ("其他", "Other accounts"))):
            r = ss[ss.subset == sub].set_index("group").reindex(PLAT)
            off = 0.16 if j == 0 else -0.16
            ok = r.share.notna()
            ax.errorbar(r.share[ok] * 100, yy[ok.to_numpy()] + off, xerr=[(r.share - r.ci_low)[ok] * 100, (r.ci_high - r.share)[ok] * 100],
                        fmt="o" if j == 0 else "s", color=C[j], ms=4.3, elinewidth=1, mec="white", mew=0.8, zorder=3, label=lab)
        ax.text(35, yy[4] + 0.16, "no official posts", fontsize=5.6, color=MUTED, va="center")
        ax.set_yticks(yy, PLAT)
        ax.set_xlim(0, 100)
        ax.set_ylim(-0.6, 4.8)
        ax.set_xlabel(f"{SHORT[cat]} (% of on-topic comments)")
        if k == 0:
            ax.legend(loc="lower left", bbox_to_anchor=(-0.05, 1.0), ncol=2, fontsize=5.8, columnspacing=0.8, handletextpad=0.2)
        grid(ax)
        panel(ax, letter, x=-0.45, y=1.12)

    ax = fig.add_subplot(r1[2])
    e = T("T14e_official_post_effect.csv").set_index("category")
    y9 = np.arange(9)[::-1]
    for y0, c in zip(y9, CATS):
        r = e.loc[c]
        ax.plot([r.ci_low, r.ci_high], [y0, y0], color=C[0], lw=1, zorder=2)
        ax.plot([r.OR_official], [y0], "o", ms=4.5, mfc="white", mec=C[0], mew=1.1, zorder=3)
    ax.axvline(1, color=INK2, lw=0.6)
    ax.set_xscale("log")
    ax.set_xlim(0.04, 8)
    ax.set_xticks([0.1, 0.5, 1, 2, 4], ["0.1", "0.5", "1", "2", "4"])
    ax.minorticks_off()
    ax.set_yticks(y9, [SHORT[c] for c in CATS], fontsize=5.8)
    ax.set_ylim(-0.6, 8.6)
    ax.set_xlabel("Odds ratio, official vs other\nposts (platform-adjusted)")
    grid(ax)
    panel(ax, "f", x=-1.0)

    ax = fig.add_subplot(gs[2, :])
    post = core.groupby(["platform_en", "post_key"]).agg(n=("on_topic", "size"), joy=("tone_en", lambda x: (x == "Joy & pride").mean() * 100),
                                                          official=("official", "first")).reset_index()
    post = post[post.n >= 5]
    rng = np.random.default_rng(3)
    for i, p in enumerate(PLAT):
        for j, (flag, col) in enumerate(((1, C[0]), (0, C[1]))):
            g = post[(post.platform_en == p) & (post.official == flag)]
            x = i + (-0.2 if j == 0 else 0.2) + rng.uniform(-0.13, 0.13, len(g))
            ax.scatter(x, g.joy, s=4 + np.sqrt(g.n) * 4.5, color=col, alpha=0.55, edgecolor="white", lw=0.4, zorder=3,
                       label=("Official accounts" if flag else "Other accounts") if i == 0 else None)
            if len(g):
                wm = np.average(g.joy, weights=g.n)
                ax.plot([i + (-0.2 if j == 0 else 0.2) - 0.16, i + (-0.2 if j == 0 else 0.2) + 0.16], [wm, wm], color=col, lw=1.6, zorder=4)
        ax.text(i, 106, f"{int((post.platform_en == p).sum())} posts", ha="center", fontsize=5.8, color=INK2)
    ax.set_xticks(range(5), PLAT)
    ax.set_xlim(-0.6, 4.6)
    ax.set_ylim(-4, 112)
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.set_ylabel("Joy & pride within post (%)")
    ax.legend(loc="upper left", bbox_to_anchor=(0.0, 0.93), fontsize=5.8, handletextpad=0.1)
    grid(ax, "y")
    panel(ax, "g", x=-0.06, y=1.02)
    note(ax, "Each point is one post with at least five on-topic comments; area scales with the number of comments; bars, comment-weighted means.", y=-0.14)
    save(fig, 6)


# ================================================================ Figure 7  temporal
def fig7():
    fig = plt.figure(figsize=(W, 8.8))
    gs = fig.add_gridspec(4, 12, hspace=0.8, wspace=3.0, height_ratios=[0.95, 1.05, 0.8, 0.8])
    r0 = gs[0, :].subgridspec(1, 2, width_ratios=[2.7, 1], wspace=0.32)
    r1 = gs[1, :].subgridspec(1, 2, width_ratios=[1, 1.3], wspace=0.8)
    r23 = gs[2:, :].subgridspec(2, 3, wspace=0.28, hspace=0.6)

    ax = fig.add_subplot(r0[0])
    mf = pd.read_csv(os.path.join(OUT, "data", "monthly_function_share.csv"))
    mf = mf[(mf.month >= "2024-06") & (mf.month <= "2026-03")].set_index("month")
    idx = pd.period_range("2024-06", "2026-03", freq="M").astype(str)
    mf = mf.reindex(idx)
    ok = mf.n.fillna(0) >= 10
    x = np.arange(len(idx))
    bottom = np.zeros(len(idx))
    for i, c in enumerate(FUNCS):
        v = np.where(ok, mf[c].fillna(0) * 100, 0)
        ax.bar(x, v, bottom=bottom, color=C[i], width=0.86, edgecolor="white", lw=0.4, zorder=3, label=SHORT[c])
        bottom += v
    for xi, flag in zip(x, ok):
        if not flag:
            ax.text(xi, 3, "·", ha="center", fontsize=8, color=MUTED)
    ax.set_xticks(x[::3], [m[2:] for m in idx[::3]], fontsize=5.8)
    ax.set_ylim(0, 100)
    ax.set_xlim(-0.7, len(idx) - 0.3)
    ax.set_ylabel("Function share of on-topic\ncomments (%)")
    ax.axvline(list(idx).index("2025-05") - 0.5, color=INK, lw=0.7, ls=(0, (3, 2)), zorder=5)
    ax.legend(ncol=5, loc="lower left", bbox_to_anchor=(-0.08, 1.0), handlelength=1, columnspacing=0.7, handletextpad=0.3, fontsize=5.8)
    panel(ax, "a", x=-0.1, y=1.14)
    note(ax, "Monthly composition (unadjusted); months with fewer than 10 on-topic comments are left blank (·). Dashed line, May 2025.", y=-0.18)

    ax = fig.add_subplot(r0[1])
    t7 = T("T7b_phase_by_platform_counts.csv").set_index("platform_en").loc[PLAT]
    heat(ax, np.log10(t7.to_numpy().astype(float).clip(min=1)), SEQ, 0, 3.6, fmt="{:.0f}", thresh=2.4, fs=5.2)
    for txt, val in zip(ax.texts, t7.to_numpy().ravel()):
        txt.set_text(f"{int(val):,}")
    ax.set_xticks(range(6), ["P1", "P2", "P3", "P4", "P5", "P6"], fontsize=5.8)
    ax.xaxis.tick_top()
    ax.set_yticks(range(5), PLAT, fontsize=5.8)
    panel(ax, "b", x=-0.6, y=1.14)
    note(ax, "On-topic comments per platform and phase.", y=-0.06, x=-0.5)

    ax = fig.add_subplot(r1[0])
    d = T("T14d_phase_tests_controlling_author_type.csv").set_index("category")
    y9 = np.arange(9)[::-1]
    for k, (col, lab, mk) in enumerate((("p_holm", "All posts", "o"), ("p_drop_top5_holm", "Five largest posts removed", "s"))):
        xx = -np.log10(d.loc[CATS, col].clip(lower=1e-6))
        ax.plot(xx, y9 + (0.14 if k == 0 else -0.14), mk, color=C[k], ms=4.6, mec="white", mew=0.8, zorder=3, label=lab, ls="none")
    for y0, c in zip(y9, CATS):
        ax.plot([-np.log10(max(d.loc[c, "p_holm"], 1e-6)), -np.log10(max(d.loc[c, "p_drop_top5_holm"], 1e-6))], [y0 + 0.14, y0 - 0.14], color=AXIS,
                lw=0.6, zorder=1)
        if d.loc[c, "significant_both"]:
            ax.text(6.55, y0, "primary", fontsize=5.4, color=INK2, va="center")
    ax.axvline(-np.log10(0.05), color=INK2, lw=0.6, ls=(0, (3, 2)))
    ax.set_yticks(y9, [SHORT[c] for c in CATS])
    ax.set_xlim(0, 7.6)
    ax.set_xticks([0, 1, 2, 4, 6], ["1", "0.1", "0.01", r"$10^{-4}$", r"$\leq10^{-6}$"])
    ax.set_ylim(-0.6, 8.8)
    ax.set_xlabel("Holm-corrected P, overall phase effect")
    ax.legend(loc="lower left", bbox_to_anchor=(-0.5, 1.0), ncol=2, handletextpad=0.2, columnspacing=0.8, fontsize=5.8)
    grid(ax)
    panel(ax, "c", x=-0.62, y=1.12)

    ax = fig.add_subplot(r1[1])
    t8 = T("T8_phase_effects_platform_adjusted.csv").set_index("category")
    cols = [c for c in t8.columns if c.startswith("OR ")]
    M = np.array([[np.log2(float(re.match(r"\s*([\d.]+)", str(t8.loc[c, col])).group(1))) for col in cols] for c in CATS])
    heat(ax, M, DIV, -3.4, 3.4, fmt="{:.1f}", thresh=2.2, fs=5.8)
    for txt, val in zip(ax.texts, M.ravel()):
        txt.set_text(f"{2**val:.2f}")
    ax.set_xticks(range(5), ["P2", "P3", "P4", "P5", "P6"], fontsize=5.8)
    ax.xaxis.tick_top()
    ax.set_yticks(range(9), [SHORT[c] for c in CATS], fontsize=5.8)
    ax.axhline(3.5, color=INK2, lw=0.8)
    panel(ax, "d", x=-0.55, y=1.12)
    note(ax, "Platform-adjusted odds ratio relative to P1 (background). Blue, above 1; orange, below 1.", y=-0.04, x=-0.45)

    t = T("T8b_phase_shares_platform_adjusted.csv")
    t["ph"] = t.phase.str[:2]
    six = ["Joy & pride", "Indignation & grief", "Neutral/informational", "Restitution claims & state capacity",
           "Historical & civilizational debate", "Institutional trust & distrust"]
    for i, c in enumerate(six):
        ax = fig.add_subplot(r23[i // 3, i % 3])
        r = t[t.category == c].sort_values("ph")
        x = np.arange(6)
        ax.axvspan(1.5, 2.5, color="#f0efec", lw=0, zorder=0)
        ax.fill_between(x, r.ci_low * 100, r.ci_high * 100, color=C[0], alpha=0.18, lw=0, zorder=1)
        ax.plot(x, r.adj_share * 100, "-o", color=C[0], lw=1.3, ms=3.4, mec="white", mew=0.7, zorder=3)
        ax.plot(x, r.raw_share * 100, "o", mfc="none", mec=MUTED, ms=3.4, mew=0.8, zorder=2)
        ax.set_xticks(x, ["P1", "P2", "P3", "P4", "P5", "P6"], fontsize=5.8)
        top = max(r.ci_high.max(), r.raw_share.max()) * 100
        ax.set_ylim(0, 15 if top < 15 else 60 if top < 60 else 85)
        ax.set_title(SHORT[c], loc="left", fontsize=6.5, pad=3)
        grid(ax, "y")
        if i % 3 == 0:
            ax.set_ylabel("Share (%)")
        if i == 0:
            panel(ax, "e", x=-0.3, y=1.1)
            ax.text(2, 0.95, "return\npeak", transform=ax.get_xaxis_transform(), fontsize=5.4, color=MUTED, ha="center", va="top")
        if i == 3:
            note(ax, "Filled, platform-standardized share with 95% cluster bootstrap interval; open, unadjusted share.", y=-0.3, x=-0.28)
    save(fig, 7)


# ================================================================ Figure 8  targeted validation
def fig8():
    fig = plt.figure(figsize=(W, 8.4))
    gs = fig.add_gridspec(3, 12, hspace=0.75, wspace=3.0)
    r0 = gs[0, :].subgridspec(1, 3, wspace=0.22)
    r1 = gs[1, :].subgridspec(1, 3, width_ratios=[1, 0.95, 1.1], wspace=0.6)
    r2 = gs[2, :].subgridspec(1, 2, wspace=0.1)
    t = T("T16_targeted_validation_skepticism.csv").set_index("metric")
    st = pd.read_csv(os.path.join(DBDIR, "targeted_validation_strata.csv"))
    fr = pd.read_csv(os.path.join(DBDIR, "targeted_validation_frame.csv"), dtype=str)
    g = pd.read_csv(os.path.join(DBDIR, "targeted_validation_human.csv"), dtype=str).merge(fr, on=["platform", "comment_id"])
    R4 = os.path.join(os.path.dirname(DBDIR), "人工编码任务包", "第4轮_定向验证", "回收")
    A = pd.read_csv(os.path.join(R4, "4_补充193条_复核员A.csv"), dtype=str)
    B = pd.read_csv(os.path.join(R4, "4_补充193条_复核员B.csv"), dtype=str)
    ab = A.merge(B, on=["platform", "comment_id"], suffixes=("_A", "_B")).merge(fr, on=["platform", "comment_id"])
    order = ["zhihu", "bilibili", "douyin", "xhs", "weibo"]
    names = [PKEY[p] for p in order]
    yy = np.arange(5)[::-1]

    ax = fig.add_subplot(r0[0])
    for y0, p in zip(yy, order):
        s = st[st.platform == p].set_index("stratum")
        pi = s.population["S"] / s.population.sum() * 100
        ax.barh(y0, pi, color=C[1], height=0.56, zorder=3)
        ax.barh(y0, 100 - pi, left=pi, color="#dfe6ef", height=0.56, zorder=3)
        ax.text(pi / 2 if pi > 14 else pi + 1.5, y0, f"{int(s.sampled['S'])}/{int(s.population['S'])}", va="center", ha="center" if pi > 14 else "left",
                fontsize=5.8, color=INK)
        ax.text(99, y0, f"{int(s.sampled['N'])}/{int(s.population['N']):,}", va="center", ha="right", fontsize=5.8, color=INK2)
    ax.set_yticks(yy, names)
    ax.set_xlim(0, 100)
    ax.set_xlabel("On-topic comments by model label (%)")
    ax.text(0, 4.62, "labeled skepticism", fontsize=5.8, color=C[1], va="bottom", fontweight="bold")
    ax.text(100, 4.62, "other tones", fontsize=5.8, color=INK2, va="bottom", ha="right")
    ax.set_ylim(-0.5, 4.95)
    ax.tick_params(axis="y", length=0)
    panel(ax, "a", x=-0.45)
    note(ax, "Numbers, sampled / available in stratum (193 sampled in total).", y=-0.3, x=-0.4)

    ax = fig.add_subplot(r0[1])
    for y0, p in zip(yy, order):
        r = t.loc[f"{p}|PPV"]
        n = int(st[(st.platform == p) & (st.stratum == "S")].sampled.iloc[0])
        ax.plot([r.ci_low * 100, r.ci_high * 100], [y0, y0], color=PCOL[PKEY[p]], lw=1.2, zorder=2)
        ax.plot([r.estimate * 100], [y0], "o", color=PCOL[PKEY[p]], ms=5, mec="white", mew=0.8, zorder=3)
        ax.text(101, y0, f"{round(r.estimate * n)}/{n}", fontsize=6, color=INK2, va="center")
    ax.set_yticks(yy, [""] * 5)
    ax.tick_params(axis="y", length=0)
    ax.set_xlim(0, 118)
    ax.set_xticks([0, 25, 50, 75, 100])
    ax.set_ylim(-0.6, 4.8)
    ax.set_xlabel("Skepticism label confirmed\nby humans (PPV, %)")
    grid(ax)
    panel(ax, "b", x=-0.02)

    ax = fig.add_subplot(r0[2])
    for y0, p in zip(yy, order):
        m, c = t.loc[f"{p}|share_model", "estimate"] * 100, t.loc[f"{p}|share_corrected"]
        ax.plot([m, c.estimate * 100], [y0, y0], color=AXIS, lw=1.2, zorder=1)
        ax.plot([c.ci_low * 100, c.ci_high * 100], [y0, y0], color=C[1], lw=2.6, alpha=0.3, solid_capstyle="round", zorder=2)
        ax.plot([m], [y0], "o", color=C[0], ms=4.8, mec="white", mew=0.8, zorder=3, label="Model-assigned" if p == "zhihu" else None)
        ax.plot([c.estimate * 100], [y0], "D", color=C[1], ms=4.2, mec="white", mew=0.8, zorder=3, label="Corrected (95% CI)" if p == "zhihu" else None)
    ax.text(35.65, 4.33, "35.65%", fontsize=6, color=INK2, ha="center")
    ax.text(8.75, 4.33, "8.75%", fontsize=6, color=INK2, ha="center")
    ax.set_yticks(yy, [""] * 5)
    ax.tick_params(axis="y", length=0)
    ax.set_xlim(0, 42)
    ax.set_ylim(-0.6, 4.8)
    ax.set_xlabel("Skepticism & dissent, share of\non-topic comments (%)")
    ax.legend(loc="lower right", handletextpad=0.3, fontsize=5.8)
    grid(ax)
    panel(ax, "c", x=-0.02)

    ax = fig.add_subplot(r1[0])
    s = T("T16b_targeted_validation_sensitivity.csv").set_index("answers")
    dd = t.loc["zhihu_minus_others"]
    rows = [("Adjudicated\n(pre-specified)", dd.estimate * 100, dd.ci_low * 100, dd.ci_high * 100, C[0]),
            ("Coder A only", s.loc["only A", "zhihu_minus_others"] * 100, s.loc["only A", "diff_ci_low"] * 100, s.loc["only A", "diff_ci_high"] * 100, MUTED),
            ("Coder B only", s.loc["only B", "zhihu_minus_others"] * 100, s.loc["only B", "diff_ci_low"] * 100, s.loc["only B", "diff_ci_high"] * 100, MUTED)]
    for i, (lab, est, lo, hi, col) in enumerate(rows):
        y0 = 2 - i
        ax.plot([lo, hi], [y0, y0], color=col, lw=1.2, zorder=2)
        ax.plot([est], [y0], "o", color=col, ms=5, mec="white", mew=0.8, zorder=3)
        ax.text(hi + 1.5, y0, f"{est:+.1f}".replace("-", "−"), fontsize=6, color=INK2, va="center")
    ax.axvline(0, color=INK2, lw=0.6)
    ax.set_yticks([2, 1, 0], [r[0] for r in rows])
    ax.set_xlim(-12, 36)
    ax.set_ylim(-0.6, 2.6)
    ax.set_xlabel("Zhihu minus other platforms,\ncorrected share (pp, 95% CI)")
    grid(ax)
    panel(ax, "d", x=-0.62)

    ax = fig.add_subplot(r1[1])
    lv = ["喜悦自豪", "愤慨痛惜", "中性理性", "质疑异见", "NR"]
    ta = ab.human_tone_A.where(ab.human_on_topic_A == "是", "NR").fillna("NR")
    tb = ab.human_tone_B.where(ab.human_on_topic_B == "是", "NR").fillna("NR")
    cm = pd.crosstab(ta, tb).reindex(index=lv, columns=lv, fill_value=0).to_numpy().astype(float)
    ax.imshow(cm / cm.sum(1, keepdims=True), cmap=SEQ, vmin=0, vmax=1)
    for i in range(5):
        for j in range(5):
            ax.text(j, i, int(cm[i, j]), ha="center", va="center", fontsize=6, color="white" if cm[i, j] / cm[i].sum() > 0.55 else INK)
    tl = ["Joy", "Indign.", "Neutral", "Skept.", "Not rel."]
    ax.set_xticks(range(5), tl, fontsize=5.6, rotation=30, ha="right", rotation_mode="anchor")
    ax.set_yticks(range(5), tl, fontsize=5.6)
    ax.set_xlabel("Coder B")
    ax.set_ylabel("Coder A")
    ax.tick_params(length=0)
    for s_ in ax.spines.values():
        s_.set_visible(False)
    panel(ax, "e", x=-0.4, y=1.04)

    ax = fig.add_subplot(r1[2])
    S = ab[ab.stratum == "S"].merge(g[["platform", "comment_id", "human_tone"]], on=["platform", "comment_id"])
    x = np.arange(5)
    wbar = 0.26
    for j, (col, lab, colr) in enumerate((("human_tone_A", "Coder A", C[0]), ("human_tone_B", "Coder B", C[2]), ("human_tone", "Adjudicated", C[1]))):
        vals = [int((S[S.platform == p][col] == "质疑异见").sum()) for p in order]
        ax.bar(x + (j - 1) * wbar, vals, width=wbar - 0.02, color=colr, zorder=3, label=lab)
        for xi, vv in zip(x, vals):
            ax.text(xi + (j - 1) * wbar, vv + 0.4, str(vv), ha="center", fontsize=4.8, color=INK2)
    ax.set_xticks(x, [f"{n[:5]}.\n(n={int((S.platform == p).sum())})" if len(n) > 6 else f"{n}\n(n={int((S.platform == p).sum())})" for n, p in zip(names, order)],
                  fontsize=5.6)
    ax.set_ylabel("Coded as skepticism (n) among\ncomments the model flagged")
    ax.set_ylim(0, 25)
    ax.legend(loc="upper right", fontsize=5.8, handlelength=1)
    grid(ax, "y")
    panel(ax, "f", x=-0.4, y=1.04)

    Sg = g[g.stratum == "S"]
    for k, (col, cats, letter, sl, ttl) in enumerate((("human_tone", TONES, "g", r2[0], "Human-coded tone"), ("human_function", FUNCS, "h", r2[1], "Human-coded function"))):
        ax = fig.add_subplot(sl)
        left = np.zeros(5)
        tot = np.array([(Sg.platform == p).sum() for p in order], float)
        for i, c in enumerate(cats + ["NR"]):
            if c == "NR":
                vals = np.array([Sg[Sg.platform == p][col].isna().sum() for p in order]) / tot * 100
                colr, lab = "#d9d8d2", "Not relevant"
            else:
                vals = np.array([(Sg[Sg.platform == p][col] == EN2ZH[c]).sum() for p in order]) / tot * 100
                colr, lab = C[i], SHORT[c]
            ax.barh(yy, vals, left=left, color=colr, height=0.6, edgecolor="white", lw=0.8, zorder=3, label=lab)
            for y0, l0, w_ in zip(yy, left, vals):
                if w_ >= 9:
                    ax.text(l0 + w_ / 2, y0, f"{w_:.0f}", ha="center", va="center", fontsize=5.6, color=INK)
            left += vals
        ax.set_yticks(yy, [f"{n} ({int(t_)})" for n, t_ in zip(names, tot)] if k == 0 else [""] * 5)
        ax.set_xlim(0, 100)
        ax.set_xlabel(f"{ttl} (%), comments the model labeled as skepticism" if k == 0 else f"{ttl} (%), same comments")
        ax.legend(ncol=3, loc="lower left", bbox_to_anchor=(-0.02, 1.0), handlelength=1, columnspacing=0.7, handletextpad=0.3, fontsize=5.6)
        ax.spines["left"].set_visible(False)
        ax.tick_params(axis="y", length=0)
        panel(ax, letter, x=-0.3 if k == 0 else -0.02, y=1.2)
    save(fig, 8)


# ================================================================ Figure 9  exploratory + evidence map
def fig9():
    fig = plt.figure(figsize=(W, 8.6))
    gs = fig.add_gridspec(3, 12, hspace=0.62, wspace=3.0, height_ratios=[1, 0.85, 1.3])
    r0 = gs[0, :].subgridspec(1, 2, width_ratios=[1.2, 1], wspace=0.5)
    r1 = gs[1, :].subgridspec(1, 2, wspace=0.08)

    ax = fig.add_subplot(r0[0])
    t9 = T("T9_engagement_within_post.csv")
    terms = ["Restitution claims & state capacity", "Institutional trust & distrust", "Historical & civilizational debate",
             "Artifact knowledge & return details", "Joy & pride", "Indignation & grief", "Skepticism & dissent"]
    y7 = np.arange(len(terms))[::-1]
    for k, (smp, lab, col, mk) in enumerate((("All on-topic comments", "All posts (7,753 comments)", C[0], "o"),
                                             ("Posts with >=80% comment coverage", "Posts with ≥80% of comments collected (5,643)", C[1], "s"))):
        r = t9[t9["sample"] == smp].set_index("term").loc[terms]
        est, lo, hi = (np.exp(r.coef_log_likes) - 1) * 100, (np.exp(r.ci_low) - 1) * 100, (np.exp(r.ci_high) - 1) * 100
        off = 0.15 if k == 0 else -0.15
        ax.hlines(y7 + off, lo, hi, color=col, lw=1, zorder=2)
        ax.plot(est, y7 + off, mk, color=col, ms=4.3, mec="white", mew=0.8, zorder=3, ls="none", label=lab)
    ax.axvline(0, color=INK2, lw=0.6)
    ax.axhline(2.5, color=GRID, lw=0.6)
    ax.set_yticks(y7, [SHORT[c] for c in terms])
    ax.set_xlabel("Difference in likes within post (%, 95% CI)")
    ax.legend(loc="lower left", bbox_to_anchor=(-0.42, 1.0), fontsize=5.8, handletextpad=0.2)
    grid(ax)
    panel(ax, "a", x=-0.45, y=1.14)
    note(ax, "Reference categories: affective expression (function, upper block) and neutral tone (lower block).\nNot corrected for multiplicity.", y=-0.24, x=-0.42)

    ax = fig.add_subplot(r0[1])
    t4 = T("T4_function_by_platform.csv")
    r = t4[t4.category == "Historical & civilizational debate"].set_index("group").loc[PLAT]
    s = pd.read_csv(os.path.join(OUT, "data", "sens_shares_luna_vs_terra.csv"))
    te = s[(s.model == "terra") & (s.category == "历史与文明论战")].set_index("platform").share.rename(index=PKEY).reindex(PLAT)
    yy = np.arange(5)[::-1]
    ax.hlines(yy, r.ci_low * 100, r.ci_high * 100, color=C[0], lw=1, zorder=2)
    ax.plot(r.share * 100, yy, "o", color=C[0], ms=4.8, mec="white", mew=0.8, zorder=3, ls="none", label="Primary model, full corpus (95% CI)")
    ax.plot(te * 100, yy, "D", color=C[2], ms=4.2, mec="white", mew=0.8, zorder=3, ls="none", label="Second model, 3,000-comment sample")
    ax.set_yticks(yy, PLAT)
    ax.set_xlim(0, 90)
    ax.set_ylim(-0.6, 4.6)
    ax.set_xlabel("Historical & civilizational debate,\nshare of on-topic comments (%)")
    ax.legend(loc="lower left", bbox_to_anchor=(-0.3, 1.0), fontsize=5.8, handletextpad=0.2)
    grid(ax)
    panel(ax, "b", x=-0.4, y=1.14)
    ax.text(0.98, 0.5, "Platform difference:\nWald P < 0.001,\npermutation P = 0.062", transform=ax.transAxes, fontsize=5.8, color=INK2, ha="right", va="center")

    TG = {"国内保管与归属": "Domestic custody & allocation", "真伪与鉴定": "Authenticity", "过程透明与交易代价": "Transparency & cost",
          "外方动机": "Returning side's motives", "官方宣传与叙事": "Official narrative", "其他": "Other"}
    STN = {"怀疑或不信任": "Distrust", "信任或为机构辩护": "Trust /\ndefense", "调侃无明确立场": "Banter,\nno stance"}
    for k, (fname, ttl, sl, letter) in enumerate((("T13b_distrust_targets_luna.csv", "Primary model", r1[0], "c"),
                                                   ("T13_distrust_targets_terra.csv", "Second model", r1[1], "d"))):
        ax = fig.add_subplot(sl)
        t = T(fname)
        t = t.set_index(t.columns[0]).drop(index="All").drop(columns="All").reindex(index=list(TG), columns=list(STN))
        heat(ax, t.to_numpy().astype(float), SEQ, 0, 100, thresh=55, fs=6.3)
        ax.set_xticks(range(3), list(STN.values()), fontsize=5.8)
        ax.xaxis.tick_top()
        ax.set_yticks(range(6), list(TG.values()) if k == 0 else [""] * 6, fontsize=5.8)
        ax.set_title(f"{ttl} (n = 251)", loc="left", fontsize=6.5, color=INK2, pad=22)
        panel(ax, letter, x=-0.62 if k == 0 else -0.04, y=1.3)
        if k == 0:
            note(ax, "Comments the primary model assigned to institutional trust, re-coded by target and stance (counts).", y=-0.06, x=-0.6)

    ax = fig.add_subplot(gs[2, :])
    t12, tb = T("T12_decisions.csv"), T("T12b_decisions_after_author_control.csv")
    crit = ["Wald test", "Permutation /\ntop-5 removed", "Post-\nweighted", "Single\nkeyword", "Second\nmodel", "Account type\nheld fixed", "Targeted\nvalidation",
            "Final"]
    M = np.full((len(crit), 18), np.nan)                               # 1 通过，0 未通过，nan 不适用
    labels = []
    for j, (_, r) in enumerate(t12.iterrows()):
        b = tb.iloc[j]
        plat = r.claim.startswith("platform")
        labels.append(SHORT[r.category])
        M[0, j] = float(r.p_wald_holm < .05)
        M[1, j] = float(r.p_perm_holm < .05) if plat else float(r.p_drop_top5_holm < .05)
        if plat:
            for i, col in ((2, "R1_post_weighted"), (3, "R2_same_keyword"), (4, "R3_terra")):
                M[i, j] = np.nan if pd.isna(r[col]) else float(str(r[col]) == "True")
        M[5, j] = float(bool(b.passes_ctrl_author))
        if plat and r.category == "Skepticism & dissent":
            M[6, j] = 0.0
        M[7, j] = float(str(b.verdict_final).startswith("PRIMARY"))
    cm3 = matplotlib.colors.ListedColormap(["#f2c4b0", "#8fb9ee"])
    ax.set_facecolor("#efeeea")
    ax.imshow(np.ma.masked_invalid(M[:7]), cmap=cm3, vmin=0, vmax=1, aspect="auto", extent=(-0.5, 17.5, 6.5, -0.5))
    for i in range(7):
        for j in range(18):
            if np.isfinite(M[i, j]):
                ax.text(j, i, "✓" if M[i, j] else "✗", ha="center", va="center", fontsize=7, color=INK, fontfamily="DejaVu Sans")
    for j in range(18):
        prim = M[7, j] == 1
        ax.add_patch(matplotlib.patches.FancyBboxPatch((j - 0.42, 7.08), 0.84, 0.74, boxstyle="round,pad=0.02,rounding_size=0.12", fc=C[0] if prim else "white",
                                                       ec=C[0] if prim else AXIS, lw=0.8, clip_on=False))
        ax.text(j, 7.45, "primary" if prim else "explor.", ha="center", va="center", fontsize=5.2, color="white" if prim else INK2)
    ax.set_xlim(-0.5, 17.5)
    ax.set_ylim(8.0, -0.5)
    ax.set_xticks(range(18), labels, rotation=40, ha="right", rotation_mode="anchor", fontsize=5.8)
    ax.set_yticks(range(8), crit, fontsize=5.8)
    ax.set_xticks(np.arange(-0.5, 18), minor=True)
    ax.set_yticks(np.arange(-0.5, 7), minor=True)
    ax.grid(which="minor", color="white", lw=1.4)
    ax.tick_params(which="both", length=0)
    for s_ in ax.spines.values():
        s_.set_visible(False)
    ax.axvline(8.5, color=INK2, lw=1)
    ax.text(4, -0.75, "Platform differences", ha="center", fontsize=6.5, color=INK2)
    ax.text(13, -0.75, "Phase effects (platform-adjusted)", ha="center", fontsize=6.5, color=INK2)
    panel(ax, "e", x=-0.085, y=1.04)
    note(ax, "Tick, criterion met; cross, not met; grey, not applicable or not evaluable. Robustness checks (rows 3–5) apply to platform differences only.", y=-0.42, x=-0.08)
    save(fig, 9)


if __name__ == "__main__":
    for n in [int(a) for a in sys.argv[1:]] or [3, 4, 5, 6, 7, 8, 9]:
        globals()[f"fig{n}"]()
