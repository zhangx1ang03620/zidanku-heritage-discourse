#!/usr/bin/env python3
"""
decide_final.py —— 在 T12（冻结规则下的判定）之上叠加发帖者类型控制，输出 T12b

规则（日志 7.7、7.11）：一项结论要保持"主要发现"，须在 T12 中已是 PRIMARY，并且
  平台差异：控制发帖者类型后 Wald（Holm）与分层置换（Holm）都 < .05（T14b）
  阶段效应：控制平台 + 发帖者类型后，全量与去掉最大 5 帖两种情况下 Wald（Holm）都 < .05（T14d）
另附模型在该类别上的精确率 / 召回率和人工估计占比（T15），供判断测量是否可靠。
T12 本身不改。
"""
import os

import pandas as pd

OUT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
T = lambda n: pd.read_csv(os.path.join(OUT, "tables", n))
EN2ZH = {"Joy & pride": "喜悦自豪", "Indignation & grief": "愤慨痛惜", "Neutral/informational": "中性理性",
         "Skepticism & dissent": "质疑异见", "Affective expression": "情感表态",
         "Restitution claims & state capacity": "追索诉求与国家能力", "Artifact knowledge & return details": "文物知识与归还细节",
         "Historical & civilizational debate": "历史与文明论战", "Institutional trust & distrust": "处置过程的信任与怀疑"}


def main():
    t12, b, d = T("T12_decisions.csv"), T("T14b_platform_tests_controlling_author_type.csv"), T("T14d_phase_tests_controlling_author_type.csv")
    v = T("T15_model_validity_weighted.csv").set_index("metric").estimate
    rows = []
    for _, r in t12.iterrows():
        key = (r.layer, r.category)
        if r.claim.startswith("platform"):
            x = b[(b.layer == key[0]) & (b.category == key[1])].iloc[0]
            p1, p2, ok = x.p_wald_holm, x.p_perm_holm, bool(x.still_significant)
        else:
            x = d[(d.layer == key[0]) & (d.category == key[1])].iloc[0]
            p1, p2, ok = x.p_holm, x.p_drop_top5_holm, bool(x.significant_both)
        was = str(r.verdict).startswith("PRIMARY")
        zh = EN2ZH[r.category]
        rows.append({"claim": r.claim, "layer": r.layer, "category": r.category, "verdict_T12": r.verdict,
                     "p1_holm_ctrl_author": p1, "p2_holm_ctrl_author": p2, "passes_ctrl_author": ok,
                     "verdict_final": r.verdict if was and ok else "exploratory",
                     "changed": was and not ok,
                     "model_precision": v.get(f"{r.layer}|{zh}|precision"), "model_recall": v.get(f"{r.layer}|{zh}|recall"),
                     "share_model": v.get(f"{r.layer}|{zh}|share_model"), "share_human": v.get(f"{r.layer}|{zh}|share_human")})
    t = pd.DataFrame(rows)
    # 定向验证（日志 7.12、7.13）：平台 × 质疑异见须满足"知乎校正占比最高且与其余平台差值的 95% 区间不含 0"
    t["note"] = ""
    t16 = T("T16_targeted_validation_skepticism.csv").set_index("metric")
    keep = t16.loc["zhihu_is_highest", "estimate"] == 1 and t16.loc["zhihu_minus_others", "ci_low"] > 0
    i = t.index[(t.claim == "platform difference") & (t.category == "Skepticism & dissent")][0]
    t.loc[i, "note"] = (f"targeted validation: Zhihu corrected share {t16.loc['zhihu|share_corrected', 'estimate']:.3f}, "
                        f"Zhihu minus others {t16.loc['zhihu_minus_others', 'estimate']:.3f} "
                        f"[{t16.loc['zhihu_minus_others', 'ci_low']:.3f}, {t16.loc['zhihu_minus_others', 'ci_high']:.3f}]")
    if not keep:
        t.loc[i, ["verdict_final", "changed"]] = ["exploratory", True]
    t.to_csv(os.path.join(OUT, "tables", "T12b_decisions_after_author_control.csv"), index=False, encoding="utf-8-sig")
    print(t[["claim", "category", "verdict_T12", "verdict_final", "changed", "model_precision", "share_model", "share_human"]]
          .round(3).to_string(index=False))


if __name__ == "__main__":
    main()
