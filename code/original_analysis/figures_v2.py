#!/usr/bin/env python3
"""
figures_v2.py —— 论文主图 Figure 1–6（对应 9.论文稿/manuscript_v2.md 的图注）

全部由 tables/ 与 data/ 里已有的结果重画，不做新的统计。输出 figures/v2/Figure{1..6}.pdf/.png。
配色：分类色取自已验证的 5 色序列（蓝、橙、青绿、黄、品红），固定顺序不循环；
顺序量用单一蓝色由浅到深；有正负的量用蓝–灰–橙。
用法：python3 figures_v2.py [图号 ...]
"""
import os
import sqlite3
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.dirname(HERE)
ROOT = os.path.dirname(OUT)
DBDIR = os.path.join(ROOT, "7.数据清洗与标注", "db")
T = lambda n: pd.read_csv(os.path.join(OUT, "tables", n))
FIG = os.path.join(OUT, "figures", "v2")
os.makedirs(FIG, exist_ok=True)

C = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]          # 分类色，固定顺序
INK, INK2, MUTED, GRID, AXIS = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
SEQ = LinearSegmentedColormap.from_list("seq", ["#ffffff", "#cde2fb", "#6da7ec", "#256abf", "#0d366b"])
DIV = LinearSegmentedColormap.from_list("div", ["#eb6834", "#f0efec", "#2a78d6"])
PLAT = ["Bilibili", "Douyin", "Xiaohongshu", "Weibo", "Zhihu"]
PKEY = {"bilibili": "Bilibili", "douyin": "Douyin", "xhs": "Xiaohongshu", "weibo": "Weibo", "zhihu": "Zhihu"}
TONES = ["Joy & pride", "Indignation & grief", "Neutral/informational", "Skepticism & dissent"]
FUNCS = ["Affective expression", "Restitution claims & state capacity", "Artifact knowledge & return details",
         "Historical & civilizational debate", "Institutional trust & distrust"]
SHORT = {"Joy & pride": "Joy & pride", "Indignation & grief": "Indignation & grief", "Neutral/informational": "Neutral / informational",
         "Skepticism & dissent": "Skepticism & dissent", "Affective expression": "Affective expression",
         "Restitution claims & state capacity": "Restitution claims", "Artifact knowledge & return details": "Artifact knowledge",
         "Historical & civilizational debate": "Historical debate", "Institutional trust & distrust": "Institutional trust"}
ZH2EN = {"喜悦自豪": TONES[0], "愤慨痛惜": TONES[1], "中性理性": TONES[2], "质疑异见": TONES[3], "情感表态": FUNCS[0],
         "追索诉求与国家能力": FUNCS[1], "文物知识与归还细节": FUNCS[2], "历史与文明论战": FUNCS[3], "处置过程的信任与怀疑": FUNCS[4]}

plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"], "font.size": 7,
                     "axes.labelsize": 7, "axes.titlesize": 7.5, "xtick.labelsize": 6.5, "ytick.labelsize": 6.5,
                     "legend.fontsize": 6.5, "axes.edgecolor": AXIS, "axes.linewidth": 0.6, "xtick.color": INK2, "ytick.color": INK2,
                     "axes.labelcolor": INK2, "text.color": INK, "xtick.major.width": 0.5, "ytick.major.width": 0.5,
                     "xtick.major.size": 2.5, "ytick.major.size": 2.5, "pdf.fonttype": 42, "legend.frameon": False,
                     "axes.spines.top": False, "axes.spines.right": False, "savefig.dpi": 400, "figure.dpi": 150})
W = 7.09                                                             # 180 mm


def panel(ax, letter, title=None, x=-0.02, y=1.06):
    ax.text(x, y, letter, transform=ax.transAxes, fontsize=9, fontweight="bold", ha="right", va="bottom")
    if title:
        ax.set_title(title, loc="left", pad=6, color=INK)


def grid(ax, axis="x"):
    ax.grid(axis=axis, color=GRID, lw=0.5, zorder=0)
    ax.set_axisbelow(True)


def save(fig, n):
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(FIG, f"Figure{n}.{ext}"), bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("Figure", n, "saved")


def dataset():
    return pd.read_csv(os.path.join(OUT, "data", "analysis_dataset.csv"), dtype={"comment_id": str})


# ---------------------------------------------------------------- Figure 1
def fig1():
    f = T("T1_data_funnel.csv")
    ds = dataset()
    fig = plt.figure(figsize=(W, 5.6))
    gs = fig.add_gridspec(2, 3, height_ratios=[1, 0.95], hspace=0.55, wspace=0.55)

    ax = fig.add_subplot(gs[0, :2])
    stages = ["Retrieved", "After de-duplication", "Under event-relevant posts", "Core (post-event, with text)",
              "Baseline (pre-event, with text)", "On topic, core", "On topic, baseline"]
    y = np.arange(len(stages))[::-1]
    left = np.zeros(len(stages))
    for i, (k, name) in enumerate(PKEY.items()):
        v = f[k].to_numpy(float)
        ax.barh(y, v, left=left, color=C[i], height=0.62, edgecolor="white", lw=0.8, label=name, zorder=3)
        left += v
    for yy, tot in zip(y, f.total):
        ax.text(tot + 1500, yy, f"{tot:,}", va="center", fontsize=6.5, color=INK2)
    ax.set_yticks(y, stages)
    ax.set_xlim(0, 155000)
    ax.set_xlabel("Comments")
    ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{int(v/1000)}k" if v else "0"))
    ax.legend(ncol=5, loc="lower right", bbox_to_anchor=(1.0, 1.0), handlelength=1, columnspacing=1, borderaxespad=0.2)
    grid(ax)
    panel(ax, "a", x=-0.36)

    ax = fig.add_subplot(gs[0, 2])
    r = T("T3_on_topic_rate_by_platform.csv").set_index("platform").loc[PLAT]
    yy = np.arange(5)[::-1]
    ax.errorbar(r.share * 100, yy, xerr=[(r.share - r.ci_low) * 100, (r.ci_high - r.share) * 100], fmt="o", color=C[0], ms=4.5,
                elinewidth=1, capsize=0, zorder=3)
    for v, y0 in zip(r.share * 100, yy):
        ax.text(v, y0 + 0.28, f"{v:.1f}", ha="center", fontsize=6.5, color=INK2)
    ax.set_yticks(yy, PLAT)
    ax.set_xlim(0, 80)
    ax.set_ylim(-0.6, 4.7)
    ax.set_xlabel("Core comments coded on topic (%)")
    grid(ax)
    panel(ax, "b", x=-0.42)

    ax = fig.add_subplot(gs[1, 0])
    core = ds[(ds.on_topic == 1) & (ds.is_post_event == 1)]
    vc = core.post_key.value_counts().to_numpy()
    cum = np.cumsum(vc) / vc.sum() * 100
    ax.plot(np.arange(1, len(cum) + 1), cum, color=C[0], lw=1.6, zorder=3)
    for k in (5, 20):
        ax.plot([k], [cum[k - 1]], "o", color=C[0], ms=4.5, mec="white", mew=1, zorder=4)
        ax.annotate(f"top {k}: {cum[k-1]:.1f}%", (k, cum[k - 1]), xytext=(6, -9), textcoords="offset points", fontsize=6.5, color=INK2)
    ax.set_xscale("log")
    ax.set_xlim(1, 400)
    ax.set_ylim(0, 102)
    ax.set_xlabel("Posts ranked by on-topic comments (log scale)")
    ax.set_ylabel("Cumulative share of\non-topic core comments (%)")
    grid(ax, "y")
    panel(ax, "c", x=-0.3)

    ax = fig.add_subplot(gs[1, 1:])
    d = pd.read_csv(os.path.join(OUT, "data", "daily_volume_all_relevant.csv"))
    dc = [c for c in d.columns if "date" in c.lower()][0]
    vcol = [c for c in d.columns if c != dc][0]
    d[dc] = pd.to_datetime(d[dc])
    d = d[(d[dc] >= "2024-06-01") & (d[dc] <= "2026-03-31")].groupby(dc)[vcol].sum()
    wk = d.resample("W").sum()
    bounds = pd.to_datetime(["2024-06-01", "2024-09-01", "2025-05-16", "2025-06-01", "2025-10-01", "2025-12-01", "2026-03-31"])
    names = ["P1", "P2 documentary\n& advocacy", "", "P4 consolidation", "P5", "P6 long tail"]
    for i in range(6):
        if i % 2 == 1:
            ax.axvspan(bounds[i], bounds[i + 1], color="#f0efec", zorder=0, lw=0)
        ax.text(bounds[i] + (bounds[i + 1] - bounds[i]) / 2, 1.02, names[i], transform=ax.get_xaxis_transform(), ha="center",
                va="bottom", fontsize=6, color=MUTED)
    ax.fill_between(wk.index, wk.values, 0.5, color=C[0], alpha=0.25, lw=0, zorder=2)
    ax.plot(wk.index, wk.values, color=C[0], lw=1.2, zorder=3)
    ax.set_yscale("log")
    ax.set_ylim(0.8, wk.max() * 6)
    for dt, lab, dy in (("2025-05-16", "16 May 2025\nreturn", 0), ("2025-10-13", "13 Oct 2025\nenters Hunan Museum", 0)):
        ax.axvline(pd.Timestamp(dt), color=INK2, lw=0.6, ls=(0, (3, 2)), zorder=1)
        ax.text(pd.Timestamp(dt) + pd.Timedelta(days=6), wk.max() * 4.5, lab, fontsize=6, color=INK2, va="top")
    ax.set_ylabel("Comments per week\n(relevant posts, log scale)")
    ax.xaxis.set_major_locator(matplotlib.dates.MonthLocator(bymonth=(1, 4, 7, 10)))
    ax.xaxis.set_major_formatter(matplotlib.dates.DateFormatter("%b\n%Y"))
    grid(ax, "y")
    panel(ax, "d", x=-0.11, y=1.12)
    save(fig, 1)


# ---------------------------------------------------------------- Figure 2
def gold_vs_model():
    conn = sqlite3.connect(os.path.join(DBDIR, "heritage.sqlite"))
    conn.execute("ATTACH DATABASE ? AS ann", (os.path.join(DBDIR, "annotations.sqlite"),))
    m = pd.read_sql("SELECT platform, comment_id, on_topic AS m_on, tone AS m_tone, function AS m_fn FROM ann.labels "
                    "WHERE codebook='v2' AND model='gpt-5.6-luna' AND run_tag='full_v2'", conn)
    m["comment_id"] = m.comment_id.astype(str)
    h = pd.read_csv(os.path.join(DBDIR, "human_review_comments.csv"), dtype=str)
    return h.merge(m, on=["platform", "comment_id"])


def fig2():
    fig = plt.figure(figsize=(W, 6.6))
    gs = fig.add_gridspec(2, 2, height_ratios=[0.8, 1], hspace=0.55, wspace=0.95, width_ratios=[1.25, 1])

    ax = fig.add_subplot(gs[0, 0])
    rel = T("reliability_pilot_v2.csv")
    s3 = T("T10_luna_vs_terra_sens3000.csv").set_index("field").kappa
    rows = [("Human coder A vs B\n(150 unseen comments)", [(0.865, 0.780, 0.946), (0.846, 0.734, 0.935), (0.917, 0.828, 0.980)]),
            ("Primary model, test–retest\n(n = 499)", [(float(rel.Kappa[i]), None, None) for i in (0, 1, 2)]),
            ("Primary vs second model\n(n = 3,000)", [(s3["on_topic"], None, None), (s3["tone"], None, None), (s3["function"], None, None)]),
            ("Primary model vs gold standard\n(n = 507)", [(0.745, None, None), (0.659, None, None), (0.757, None, None)])]
    labs = ["Relevance", "Tone", "Function"]
    for i, (name, vals) in enumerate(rows):
        y0 = len(rows) - 1 - i
        for j, (k, lo, hi) in enumerate(vals):
            yy = y0 + (1 - j) * 0.2
            if lo is not None:
                ax.plot([lo, hi], [yy, yy], color=C[j], lw=1, zorder=2)
            ax.plot([k], [yy], "o", color=C[j], ms=4.5, mec="white", mew=0.8, zorder=3, label=labs[j] if i == 0 else None)
            ax.text(1.065, yy, f"{k:.3f}", fontsize=6, color=INK2, va="center", ha="right")
    ax.axvline(0.6, color=INK2, lw=0.6, ls=(0, (3, 2)), zorder=1)
    ax.text(0.6, len(rows) - 0.42, "0.60", fontsize=6, color=MUTED, ha="center")
    ax.set_yticks(range(len(rows)), [r[0] for r in rows][::-1])
    ax.set_xlim(0.3, 1.07)
    ax.set_xticks([0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    ax.set_ylim(-0.5, len(rows) - 0.25)
    ax.set_xlabel("Cohen's κ (bars, 95% CI where estimated)")
    ax.legend(ncol=3, loc="lower left", bbox_to_anchor=(0, 1.0), handletextpad=0.2, columnspacing=1)
    grid(ax)
    panel(ax, "a", x=-0.6, y=1.1)

    d = gold_vs_model()
    both = d[(d.m_on == 1) & (d.human_on_topic == "是")]
    for k, (mc, hc, cats, ttl) in enumerate((("m_tone", "human_tone", TONES, "Tone"), ("m_fn", "human_function", FUNCS, "Function"))):
        ax = fig.add_subplot(gs[k, 1])
        zh = [z for c in cats for z, e in ZH2EN.items() if e == c]
        cm = pd.crosstab(both[mc], both[hc]).reindex(index=zh, columns=zh, fill_value=0).to_numpy()
        rn = cm / cm.sum(1, keepdims=True)
        ax.imshow(rn, cmap=SEQ, vmin=0, vmax=1)
        for i in range(len(cats)):
            for j in range(len(cats)):
                ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=6.5, color="white" if rn[i, j] > 0.55 else INK)
        ax.set_xticks(range(len(cats)), [SHORT[c] for c in cats], rotation=35, ha="right", rotation_mode="anchor", fontsize=6)
        ax.set_yticks(range(len(cats)), [SHORT[c] for c in cats], fontsize=6)
        ax.set_xlabel("Gold standard")
        ax.set_ylabel("Model")
        ax.tick_params(length=0)
        for s in ax.spines.values():
            s.set_visible(False)
        ax.set_title(f"{ttl} (n = {cm.sum()}; counts, shading = row share)", loc="left", fontsize=6.5, color=INK2, pad=4)
        if k == 0:
            panel(ax, "b", x=-0.75, y=1.1)

    ax = fig.add_subplot(gs[1, 0])
    v = T("T15_model_validity_weighted.csv").set_index("metric")
    cats = TONES + FUNCS
    for i, c in enumerate(cats):
        zh = [z for z, e in ZH2EN.items() if e == c][0]
        layer = "tone" if c in TONES else "function"
        y0 = len(cats) - 1 - i + (0.5 if c in TONES else 0)
        m = v.loc[f"{layer}|{zh}|share_model"]
        h = v.loc[f"{layer}|{zh}|share_human"]
        ax.plot([m.estimate * 100, h.estimate * 100], [y0, y0], color=AXIS, lw=1.2, zorder=1)
        ax.plot([h.ci_low * 100, h.ci_high * 100], [y0, y0], color=C[1], lw=2.6, alpha=0.3, solid_capstyle="round", zorder=2)
        ax.plot([m.estimate * 100], [y0], "o", color=C[0], ms=4.8, mec="white", mew=0.8, zorder=3, label="Model-assigned" if i == 0 else None)
        ax.plot([h.estimate * 100], [y0], "D", color=C[1], ms=4.2, mec="white", mew=0.8, zorder=3,
                label="Human-based estimate (95% CI)" if i == 0 else None)
    ax.set_yticks([len(cats) - 1 - i + (0.5 if c in TONES else 0) for i, c in enumerate(cats)], [SHORT[c] for c in cats])
    ax.text(54, 8.5, "Tone", fontsize=6.5, color=MUTED, va="center", ha="right")
    ax.text(54, 4, "Function", fontsize=6.5, color=MUTED, va="center", ha="right")
    ax.axhline(4.75, color=GRID, lw=0.6)
    ax.set_xlim(0, 55)
    ax.set_xlabel("Share of on-topic comments (%),\nreweighted to sampling strata")
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1.0), ncol=2, handletextpad=0.3, columnspacing=1)
    grid(ax)
    panel(ax, "c", x=-0.6, y=1.1)
    save(fig, 2)


# ---------------------------------------------------------------- Figure 3
def fig3():
    fig = plt.figure(figsize=(W, 6.0))
    gs = fig.add_gridspec(2, 2, hspace=0.5, wspace=0.75, height_ratios=[1, 1])
    v = T("T15_model_validity_weighted.csv").set_index("metric").estimate
    groups = ["All"] + PLAT
    for k, (fname, cats, letter, layer) in enumerate((("T4_tone_by_platform.csv", TONES, "a", "tone"), ("T4_function_by_platform.csv", FUNCS, "b", "function"))):
        ax = fig.add_subplot(gs[k, 0])
        t = T(fname)
        y = np.arange(len(groups))[::-1].astype(float)
        y[0] += 0.35
        left = np.zeros(len(groups))
        for i, c in enumerate(cats):
            s = t[t.category == c].set_index("group").share.reindex(groups).to_numpy() * 100
            ax.barh(y, s, left=left, color=C[i], height=0.62, edgecolor="white", lw=1, zorder=3, label=SHORT[c])
            for yy, l0, w in zip(y, left, s):
                if w >= 7:
                    ax.text(l0 + w / 2, yy, f"{w:.0f}", ha="center", va="center", fontsize=6, color=INK)
            left += s
        cum = 0
        for i, c in enumerate(cats[:-1]):                                   # 人工估计的累计分界
            zh = [z for z, e in ZH2EN.items() if e == c][0]
            cum += v[f"{layer}|{zh}|share_human"] * 100
            ax.plot([cum], [y[0] - 0.47], marker="^", color=INK, ms=4, zorder=5, clip_on=False)
        ax.text(101.5, y[0] - 0.47, "▲ human-based\n    boundaries", fontsize=5.6, color=INK2, va="center")
        n = t[t.category == cats[0]].set_index("group").group_total.reindex(groups)
        ax.set_yticks(y, [f"{g}  (n = {int(n[g]):,})" for g in groups])
        ax.set_xlim(0, 100)
        ax.set_xlabel("Share of on-topic core comments (%)")
        ax.legend(ncol=2 if len(cats) == 4 else 3, loc="lower left", bbox_to_anchor=(-0.42, 1.0), handlelength=1, columnspacing=0.9,
                  handletextpad=0.4)
        for s_ in ("left",):
            ax.spines[s_].set_visible(False)
        ax.tick_params(axis="y", length=0)
        panel(ax, letter, x=-0.5, y=1.2)

    ax = fig.add_subplot(gs[:, 1])
    t6 = T("T6_tone_x_function.csv")
    R = t6.pivot(index="function_en", columns="tone_en", values="adj_std_residual").reindex(index=FUNCS, columns=TONES)
    N = t6.pivot(index="function_en", columns="tone_en", values="n").reindex(index=FUNCS, columns=TONES)
    im = ax.imshow(R.to_numpy(), cmap=DIV, vmin=-46, vmax=46, aspect="auto")
    for i in range(5):
        for j in range(4):
            r = R.iloc[i, j]
            ax.text(j, i - 0.1, f"{r:+.1f}", ha="center", va="center", fontsize=7, color="white" if abs(r) > 28 else INK)
            ax.text(j, i + 0.2, f"n = {int(N.iloc[i, j]):,}", ha="center", va="center", fontsize=5.6, color="white" if abs(r) > 28 else INK2)
    ax.set_xticks(range(4), [SHORT[c].replace(" & ", " &\n").replace(" / ", " /\n") for c in TONES])
    ax.set_yticks(range(5), [SHORT[c].replace(" ", "\n", 1) for c in FUNCS])
    ax.xaxis.tick_top()
    ax.tick_params(length=0)
    for s_ in ax.spines.values():
        s_.set_visible(False)
    ax.set_xticks(np.arange(-0.5, 4), minor=True)
    ax.set_yticks(np.arange(-0.5, 5), minor=True)
    ax.grid(which="minor", color="white", lw=2)
    ax.tick_params(which="minor", length=0)
    cb = fig.colorbar(im, ax=ax, orientation="horizontal", fraction=0.035, pad=0.04)
    cb.set_label("Adjusted standardized residual\n(orange, fewer than expected; blue, more)", fontsize=6.5)
    cb.outline.set_visible(False)
    cb.ax.tick_params(length=2)
    panel(ax, "c", x=-0.16, y=1.07)
    save(fig, 3)


# ---------------------------------------------------------------- Figure 4
def fig4():
    fig = plt.figure(figsize=(W, 6.0))
    gs = fig.add_gridspec(2, 2, hspace=0.6, wspace=0.6)
    yy = np.arange(5)[::-1]

    ax = fig.add_subplot(gs[0, 0])
    for k, (fname, cat) in enumerate((("T4_tone_by_platform.csv", "Joy & pride"), ("T4_function_by_platform.csv", "Affective expression"))):
        t = T(fname)
        t = t[t.category == cat].set_index("group")
        r = t.loc[PLAT]
        off = 0.16 if k == 0 else -0.16
        ax.errorbar(r.share * 100, yy + off, xerr=[(r.share - r.ci_low) * 100, (r.ci_high - r.share) * 100], fmt="o" if k == 0 else "s",
                    color=C[k], ms=4.5, elinewidth=1, mec="white", mew=0.8, zorder=3, label=SHORT[cat])
        ax.axvline(t.loc["All", "share"] * 100, color=C[k], lw=0.8, ls=(0, (3, 2)), zorder=1)
    ax.text(27.5, 4.55, "pooled", fontsize=6, color=MUTED)
    ax.set_yticks(yy, PLAT)
    ax.set_xlim(0, 100)
    ax.set_ylim(-0.6, 4.8)
    ax.set_xlabel("Share of on-topic comments (%)")
    ax.legend(loc="lower right")
    grid(ax)
    panel(ax, "a", x=-0.3)

    ax = fig.add_subplot(gs[0, 1])
    a = T("T14a_author_type_by_platform.csv").set_index("platform_en").loc[PLAT]
    ds = dataset()
    at = pd.read_csv(os.path.join(DBDIR, "post_author_type_binary.csv"), dtype=str)
    ds["pid"] = ds.post_key.str.split(":").str[-1]
    m = ds.merge(at[["platform", "pid", "official"]], on=["platform", "pid"])
    core = m[(m.on_topic == 1) & (m.is_post_event == 1)]
    sh = core.groupby("platform_en").official.apply(lambda x: (pd.to_numeric(x) == 1).mean() * 100).reindex(PLAT)
    ax.barh(yy, sh, color=C[0], height=0.58, zorder=3)
    for y0, v, p in zip(yy, sh, PLAT):
        ax.text(v + 1.5, y0, f"{v:.1f}%   ({int(a.loc[p, 'n_official_posts'])} of {int(a.loc[p, 'n_posts'])} posts)", va="center", fontsize=6.5,
                color=INK2)
    ax.set_yticks(yy, PLAT)
    ax.set_xlim(0, 135)
    ax.set_xticks([0, 25, 50, 75, 100])
    ax.set_xlabel("On-topic comments under posts by official accounts (%)")
    grid(ax)
    panel(ax, "b", x=-0.3)

    ax = fig.add_subplot(gs[1, 0])
    s = T("T14c2_shares_within_author_type.csv")
    s = s[s.category == "Joy & pride"]
    for k, (sub, lab) in enumerate((("官方", "Posts by official accounts"), ("其他", "Posts by other accounts"))):
        r = s[s.subset == sub].set_index("group").reindex(PLAT)
        off = 0.16 if k == 0 else -0.16
        ok = r.share.notna()
        ax.errorbar(r.share[ok] * 100, yy[ok.to_numpy()] + off, xerr=[(r.share - r.ci_low)[ok] * 100, (r.ci_high - r.share)[ok] * 100],
                    fmt="o" if k == 0 else "s", color=C[k], ms=4.5, elinewidth=1, mec="white", mew=0.8, zorder=3, label=lab)
    ax.text(3, yy[4] + 0.16, "no official posts", fontsize=6, color=MUTED, va="center")
    ax.set_yticks(yy, PLAT)
    ax.set_xlim(0, 100)
    ax.set_ylim(-0.6, 4.8)
    ax.set_xlabel("Joy & pride, share of on-topic comments (%)")
    ax.legend(loc="lower right")
    grid(ax)
    panel(ax, "c", x=-0.3)

    ax = fig.add_subplot(gs[1, 1])
    e = T("T14e_official_post_effect.csv").set_index("category")
    cats = TONES + FUNCS
    y9 = np.arange(len(cats))[::-1]
    for y0, c in zip(y9, cats):
        r = e.loc[c]
        ax.plot([r.ci_low, r.ci_high], [y0, y0], color=C[0], lw=1, zorder=2)
        ax.plot([r.OR_official], [y0], "o", ms=4.8, mfc="white", mec=C[0], mew=1.1, zorder=3)
        ax.text(1.02, y0, f"{r.p_wald_holm:.3f} / {r.p_perm_holm:.3f}", transform=ax.get_yaxis_transform(), fontsize=5.8, color=INK2, va="center")
    ax.text(1.02, len(cats) - 0.35, "P (Wald / perm.)", transform=ax.get_yaxis_transform(), fontsize=5.8, color=MUTED, va="center")
    ax.axvline(1, color=INK2, lw=0.6)
    ax.set_xscale("log")
    ax.set_xlim(0.04, 8)
    ax.set_xticks([0.1, 0.25, 0.5, 1, 2, 4], ["0.1", "0.25", "0.5", "1", "2", "4"])
    ax.minorticks_off()
    ax.set_yticks(y9, [SHORT[c] for c in cats])
    ax.set_ylim(-0.6, len(cats) - 0.1)
    ax.set_xlabel("Odds ratio, official vs other posts,\nadjusted for platform (95% CI)")
    grid(ax)
    panel(ax, "d", x=-0.46)
    save(fig, 4)


# ---------------------------------------------------------------- Figure 5
def fig5():
    fig = plt.figure(figsize=(W, 5.4))
    gs = fig.add_gridspec(2, 5, hspace=0.6, wspace=0.42, width_ratios=[1.55, 0.25, 1, 1, 1])
    ax = fig.add_subplot(gs[:, 0])
    d = T("T14d_phase_tests_controlling_author_type.csv").set_index("category")
    cats = TONES + FUNCS
    y9 = np.arange(len(cats))[::-1]
    for k, (col, lab, mk) in enumerate((("p_holm", "All posts", "o"), ("p_drop_top5_holm", "Five largest posts removed", "s"))):
        x = -np.log10(d.loc[cats, col].clip(lower=1e-6))
        ax.plot(x, y9 + (0.14 if k == 0 else -0.14), mk, color=C[k], ms=4.8, mec="white", mew=0.8, zorder=3, label=lab, ls="none")
    for y0, c in zip(y9, cats):
        if d.loc[c, "significant_both"]:
            ax.text(-0.03, y0, "●", transform=ax.get_yaxis_transform(), fontsize=5, color=INK, ha="right", va="center")
    ax.axvline(-np.log10(0.05), color=INK2, lw=0.6, ls=(0, (3, 2)))
    ax.text(-np.log10(0.05) + 0.08, len(cats) - 0.55, "P = 0.05", fontsize=6, color=MUTED)
    ax.set_yticks(y9, [SHORT[c] for c in cats])
    ax.tick_params(axis="y", pad=9)
    ax.set_xlim(0, 6.4)
    ax.set_xticks([0, 1, 2, 4, 6], ["1", "0.1", "0.01", r"$10^{-4}$", r"$\leq10^{-6}$"])
    ax.set_ylim(-0.6, len(cats) - 0.2)
    ax.set_xlabel("Holm-corrected P, overall phase effect\n(adjusted for platform and account type)")
    ax.plot([], [], "o", color=INK, ms=2.2, ls="none", label="left of name: met both criteria")
    ax.legend(loc="lower left", bbox_to_anchor=(-0.6, 1.0), ncol=1, handletextpad=0.2)
    grid(ax)
    panel(ax, "a", x=-0.62, y=1.1)

    t = T("T8b_phase_shares_platform_adjusted.csv")
    t["ph"] = t.phase.str[:2]
    six = ["Joy & pride", "Indignation & grief", "Neutral/informational", "Restitution claims & state capacity",
           "Historical & civilizational debate", "Institutional trust & distrust"]
    for i, c in enumerate(six):
        ax = fig.add_subplot(gs[i // 3, 2 + i % 3])
        r = t[t.category == c].sort_values("ph")
        x = np.arange(6)
        ax.axvspan(1.5, 2.5, color="#f0efec", lw=0, zorder=0)
        ax.fill_between(x, r.ci_low * 100, r.ci_high * 100, color=C[0], alpha=0.18, lw=0, zorder=1)
        ax.plot(x, r.adj_share * 100, "-o", color=C[0], lw=1.4, ms=3.6, mec="white", mew=0.7, zorder=3)
        ax.plot(x, r.raw_share * 100, "o", mfc="none", mec=MUTED, ms=3.6, mew=0.8, zorder=2)
        ax.set_xticks(x, ["P1", "P2", "P3", "P4", "P5", "P6"])
        top = max(r.ci_high.max(), r.raw_share.max()) * 100
        ax.set_ylim(0, 15 if top < 15 else 60 if top < 60 else 85)
        ax.set_title(SHORT[c], loc="left", fontsize=7, pad=3)
        grid(ax, "y")
        if i % 3 == 0:
            ax.set_ylabel("Share of on-topic\ncomments (%)")
        if i == 0:
            panel(ax, "b", x=-0.32, y=1.1)
            ax.text(2, 0.93, "return\npeak", transform=ax.get_xaxis_transform(), fontsize=5.6, color=MUTED, ha="center", va="top")
    h = [matplotlib.lines.Line2D([], [], marker="o", color=C[0], lw=1.4, ms=3.6, mec="white", label="Platform-standardized share (95% CI)"),
         matplotlib.lines.Line2D([], [], marker="o", mfc="none", mec=MUTED, ls="none", ms=3.6, label="Unadjusted share")]
    fig.legend(handles=h, loc="upper right", bbox_to_anchor=(0.9, 0.99), ncol=2, handletextpad=0.4)
    save(fig, 5)


# ---------------------------------------------------------------- Figure 6
def fig6():
    fig = plt.figure(figsize=(W, 5.8))
    gs = fig.add_gridspec(2, 2, hspace=0.62, wspace=0.6)
    t = T("T16_targeted_validation_skepticism.csv").set_index("metric")
    st = pd.read_csv(os.path.join(DBDIR, "targeted_validation_strata.csv"))
    order = ["zhihu", "bilibili", "douyin", "xhs", "weibo"]
    yy = np.arange(5)[::-1]

    ax = fig.add_subplot(gs[0, 0])
    for y0, p in zip(yy, order):
        r = t.loc[f"{p}|PPV"]
        n = int(st[(st.platform == p) & (st.stratum == "S")].sampled.iloc[0])
        ax.plot([r.ci_low * 100, r.ci_high * 100], [y0, y0], color=C[0], lw=1, zorder=2)
        ax.plot([r.estimate * 100], [y0], "o", color=C[0], ms=5, mec="white", mew=0.8, zorder=3)
        ax.text(101, y0, f"{round(r.estimate * n)}/{n}", fontsize=6.5, color=INK2, va="center")
    ax.text(101, 4.6, "confirmed", fontsize=6, color=MUTED, va="center")
    ax.set_yticks(yy, [PKEY[p] for p in order])
    ax.set_xlim(0, 100)
    ax.set_ylim(-0.6, 4.8)
    ax.set_xlabel("Model's skepticism label confirmed by humans (%)\npositive predictive value, 95% CI")
    grid(ax)
    panel(ax, "a", x=-0.3)

    ax = fig.add_subplot(gs[0, 1])
    for y0, p in zip(yy, order):
        m, c = t.loc[f"{p}|share_model", "estimate"] * 100, t.loc[f"{p}|share_corrected"]
        ax.annotate("", xy=(c.estimate * 100, y0), xytext=(m, y0), arrowprops=dict(arrowstyle="-", color=AXIS, lw=1.2), zorder=1)
        ax.plot([c.ci_low * 100, c.ci_high * 100], [y0, y0], color=C[1], lw=2.6, alpha=0.3, solid_capstyle="round", zorder=2)
        ax.plot([m], [y0], "o", color=C[0], ms=5, mec="white", mew=0.8, zorder=3, label="Model-assigned" if p == "zhihu" else None)
        ax.plot([c.estimate * 100], [y0], "D", color=C[1], ms=4.4, mec="white", mew=0.8, zorder=3,
                label="Corrected (95% CI)" if p == "zhihu" else None)
    ax.text(t.loc["zhihu|share_model", "estimate"] * 100, 4.33, "35.65%", fontsize=6.5, color=INK2, ha="center")
    ax.text(t.loc["zhihu|share_corrected", "estimate"] * 100, 4.33, "8.75%", fontsize=6.5, color=INK2, ha="center")
    ax.set_yticks(yy, [PKEY[p] for p in order])
    ax.set_xlim(0, 42)
    ax.set_ylim(-0.6, 4.8)
    ax.set_xlabel("Skepticism & dissent, share of on-topic comments (%)")
    ax.legend(loc="lower right", handletextpad=0.3)
    grid(ax)
    panel(ax, "b", x=-0.3)

    ax = fig.add_subplot(gs[1, 0])
    s = T("T16b_targeted_validation_sensitivity.csv").set_index("answers")
    d = t.loc["zhihu_minus_others"]
    rows = [("Adjudicated labels\n(pre-specified analysis)", d.estimate * 100, d.ci_low * 100, d.ci_high * 100, C[0]),
            ("Coder A labels only", s.loc["only A", "zhihu_minus_others"] * 100, s.loc["only A", "diff_ci_low"] * 100, s.loc["only A", "diff_ci_high"] * 100, MUTED),
            ("Coder B labels only", s.loc["only B", "zhihu_minus_others"] * 100, s.loc["only B", "diff_ci_low"] * 100, s.loc["only B", "diff_ci_high"] * 100, MUTED)]
    for i, (lab, est, lo, hi, col) in enumerate(rows):
        y0 = 2 - i
        ax.plot([lo, hi], [y0, y0], color=col, lw=1.2, zorder=2)
        ax.plot([est], [y0], "o", color=col, ms=5, mec="white", mew=0.8, zorder=3)
        ax.text(hi + 1, y0, f"{est:+.1f} ({lo:.1f} to {hi:.1f})".replace("-", "−"), fontsize=6.5, color=INK2, va="center")
    ax.axvline(0, color=INK2, lw=0.6)
    ax.set_yticks([2, 1, 0], [r[0] for r in rows])
    ax.set_xlim(-12, 48)
    ax.set_ylim(-0.6, 2.6)
    ax.set_xlabel("Zhihu minus other platforms pooled,\ncorrected share (percentage points, 95% CI)")
    grid(ax)
    panel(ax, "c", x=-0.52)

    ax = fig.add_subplot(gs[1, 1])
    g = pd.read_csv(os.path.join(DBDIR, "targeted_validation_human.csv"), dtype=str).merge(
        pd.read_csv(os.path.join(DBDIR, "targeted_validation_frame.csv"), dtype=str), on=["platform", "comment_id"])
    z = g[(g.platform == "zhihu") & (g.stratum == "S")]
    tone = z.human_tone.fillna("Not relevant").map(lambda v: ZH2EN.get(v, v)).value_counts()
    order_t = [c for c in ["Neutral/informational", "Skepticism & dissent", "Indignation & grief", "Joy & pride", "Not relevant"] if c in tone.index]
    yb = np.arange(len(order_t))[::-1]
    ax.barh(yb, [tone[c] for c in order_t], color=[C[1] if c == "Skepticism & dissent" else C[0] for c in order_t], height=0.58, zorder=3)
    for y0, c in zip(yb, order_t):
        ax.text(tone[c] + 0.6, y0, str(int(tone[c])), va="center", fontsize=6.5, color=INK2)
    ax.set_yticks(yb, [SHORT.get(c, c) for c in order_t])
    ax.set_xlim(0, 40)
    ax.set_xlabel("Human-coded tone of the 40 Zhihu comments\nthat the model labeled as skepticism")
    n_hist = int((z.human_function == "历史与文明论战").sum())
    ax.text(0.98, 0.06, f"{n_hist} of 40 coded by humans as\nhistorical & civilizational debate", transform=ax.transAxes, ha="right", fontsize=6.5,
            color=INK2)
    grid(ax)
    panel(ax, "d", x=-0.5)
    save(fig, 6)


if __name__ == "__main__":
    todo = [int(a) for a in sys.argv[1:]] or [1, 2, 3, 4, 5, 6]
    for n in todo:
        globals()[f"fig{n}"]()
