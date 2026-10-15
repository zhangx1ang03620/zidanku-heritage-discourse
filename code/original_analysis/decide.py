#!/usr/bin/env python3
"""
decide.py —— 按实验日志第 5 节（及第 7 节的收紧）逐条判定：主要发现 / 探索性

平台差异（每个类别一条）：
  显著性：聚类稳健 Wald（Holm）< .05  且  按帖子置换检验（Holm）< .05          ← 第 7 节收紧
  方向一致（稳健性检验 1–3 各自判断；定义在看判定结果之前写定）：
    "驱动平台" = 主分析中占比偏离合并占比最大的平台；
    一致 = 该平台在稳健性数据里偏离方向相同，且 5 平台占比排序与主分析 Spearman ρ > 0
    R1 帖子等权；R2 同一检索词"子弹库帛书"（知乎帖子 < 10 个，只比 4 个平台）；R3 terra 3,000 条
阶段效应（每个类别一条）：
  显著性：Wald（控制平台，Holm）< .05  且  去掉最大 5 个帖子后（Holm）< .05       ← 第 7 节收紧
  稳健性：预设的替代检验是月度序列（描述性），另报控制平台后的 P3 vs 其余阶段对比
输出 tables/T12_decisions.csv
"""
import os

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.dirname(HERE)
T = lambda n: pd.read_csv(os.path.join(OUT, "tables", n))
PLAT = ["Bilibili", "Douyin", "Weibo", "Xiaohongshu", "Zhihu"]
ZH2EN = {"喜悦自豪": "Joy & pride", "愤慨痛惜": "Indignation & grief", "中性理性": "Neutral/informational",
         "质疑异见": "Skepticism & dissent", "情感表态": "Affective expression",
         "追索诉求与国家能力": "Restitution claims & state capacity", "文物知识与归还细节": "Artifact knowledge & return details",
         "历史与文明论战": "Historical & civilizational debate", "处置过程的信任与怀疑": "Institutional trust & distrust"}
PL = {"bilibili": "Bilibili", "douyin": "Douyin", "weibo": "Weibo", "xhs": "Xiaohongshu", "zhihu": "Zhihu"}


def direction(main, pooled, alt):
    """main/alt: Series platform→share；返回 (驱动平台, 一致?, ρ)"""
    common = [p for p in main.index if p in alt.index and pd.notna(alt[p])]
    dev = (main - pooled)
    driver = dev.abs().idxmax()
    if driver not in common:
        return driver, np.nan, np.nan
    alt_pooled = alt[common].mean()
    rho = spearmanr(main[common], alt[common]).correlation if len(common) >= 3 else np.nan
    same_sign = np.sign(dev[driver]) == np.sign(alt[driver] - alt_pooled)
    return driver, bool(same_sign and rho > 0), rho


def main():
    ds = pd.read_csv(os.path.join(OUT, "data", "analysis_dataset.csv"))
    core = ds[(ds.on_topic == 1) & (ds.is_post_event == 1)]
    kw = core[core.keyword == "子弹库帛书"]
    kw = kw[kw.platform_en.map(kw.groupby("platform_en").post_key.nunique()) >= 10]
    pw = T("T4b_robust_post_weighted.csv")
    sens = pd.read_csv(os.path.join(OUT, "data", "sens_shares_luna_vs_terra.csv"))
    sens = sens[sens.model == "terra"].assign(platform=lambda d: d.platform.map(PL), category=lambda d: d.category.map(ZH2EN))
    t5b = T("T5b_robust_same_keyword_tests.csv")
    t8 = T("T8_phase_effects_platform_adjusted.csv")
    rows = []
    for layer, fname in (("tone", "tone"), ("function", "function")):
        shares = T(f"T4_{fname}_by_platform.csv")
        tests = T(f"T5_{fname}_platform_tests.csv")
        col = f"{layer}_en"
        for _, t in tests.iterrows():
            c = t.category
            s = shares[shares.category == c].set_index("group").share
            m, pooled = s.reindex(PLAT), s["All"]
            r1 = pw[(pw.layer == layer) & (pw.category == c) & (pw.group != "All")].set_index("group").share_post_weighted
            r2 = kw.groupby("platform_en")[col].apply(lambda x: (x == c).mean())
            r3 = sens[(sens.layer == layer) & (sens.category == c)].set_index("platform").share
            d1, d2, d3 = direction(m, pooled, r1), direction(m, pooled, r2), direction(m, pooled, r3)
            sig = (t.p_holm < .05) and (t.p_perm_holm < .05)
            k = t5b[(t5b.layer == layer) & (t5b.category == c)].iloc[0]
            consistent = all(x[1] is True for x in (d1, d2, d3))
            # 驱动平台在 R2 中无法评估（知乎在该检索词下帖子 < 10 个）时，不算通过也不算失败，单列一档
            conditional = d1[1] is True and d3[1] is True and pd.isna(d2[1])
            rows.append({"claim": "platform difference", "layer": layer, "category": c,
                         "driver_platform": d1[0], "driver_share": round(m[d1[0]], 3), "pooled_share": round(pooled, 3),
                         "p_wald_holm": t.p_holm, "p_perm_holm": t.p_perm_holm, "significant_both": sig,
                         "R1_post_weighted": d1[1], "R1_rho": d1[2], "R2_same_keyword": d2[1], "R2_rho": d2[2],
                         "R2_sig_both": (k.p_holm < .05) and (k.p_perm_holm < .05),
                         "R3_terra": d3[1], "R3_rho": d3[2],
                         "verdict": "PRIMARY" if sig and consistent else
                         "PRIMARY* (R2 not evaluable)" if sig and conditional else "exploratory"})
    for _, t in t8.iterrows():
        sig = (t.p_holm < .05) and (t.p_drop_top5_holm < .05)
        rows.append({"claim": "phase effect (platform-adjusted)", "layer": t.layer, "category": t.category,
                     "p_wald_holm": t.p_holm, "p_drop_top5_holm": t.p_drop_top5_holm, "significant_both": sig,
                     "verdict": "PRIMARY" if sig else "exploratory"})
    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(OUT, "tables", "T12_decisions.csv"), index=False, encoding="utf-8-sig")
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 30)
    print(out.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
