# Data dictionary

All files are UTF-8 CSV. `comment_key` and `post_key` are salted SHA-256 digests (first 16 hexadecimal characters, prefixed `c` or `p`). They are consistent across files and cannot be mapped back to platform identifiers. Platforms are Bilibili, Douyin, Xiaohongshu, Weibo and Zhihu.

Tone categories: `Joy & pride`, `Indignation & grief`, `Neutral/informational`, `Skepticism & dissent`.
Function categories: `Affective expression`, `Restitution claims & state capacity`, `Artifact knowledge & return details`, `Historical & civilizational debate`, `Institutional trust & distrust`.
Tone and function are empty when a comment was coded as not relevant.

## comments.csv (19,128 rows)
One row per comment with text under a post that met the post-level relevance rule. Labels are those of the primary model in the full run.

| Column | Meaning |
|---|---|
| comment_key, post_key | Pseudonymous identifiers of the comment and of its post |
| post_order | Rank of the post in the sort order of the original identifiers; used only to make the seeded permutation test reproducible |
| platform | Platform |
| date | Date of the comment (China Standard Time) |
| phase | P1 background (before 1 Sep 2024), P2 documentary and advocacy (to 15 May 2025), P3 return peak (16 to 31 May 2025), P4 consolidation (Jun to Sep 2025), P5 return to Hunan (Oct to Nov 2025), P6 long tail (from 1 Dec 2025) |
| is_post_event | 1 if posted on or after 16 May 2025 (core corpus), 0 otherwise (baseline) |
| search_term | Search term that retrieved the post (Chinese) |
| text_length | Number of characters in the comment |
| is_reply | 1 if the comment replies to another comment |
| like_count | Likes at the time of collection |
| post_comment_coverage | Collected comments of the post divided by the comment count shown by the platform |
| on_topic | 1 if the model coded the comment as relevant |
| tone, function | Model labels |
| official_account | 1 if the post came from official media or a government or heritage institution, 0 otherwise; empty for posts without on-topic comments, which were not coded |

## model_labels.csv (23,625 rows)
Every label produced by the two models. `model` is `primary` or `second`; `model_name` is the name used by the provider; `run` is `pilot_v2`, `pilot_v2_retest`, `full_v2` (primary model, whole corpus) or `sens_v2` (second model, 3,000-comment sample).

## posts.csv (5,512 rows)
Every retrieved post: `search_term`, `rule_relevant` (1 if the post met the keyword rule), `is_post_event`, `official_account`.

## corpus_funnel.csv
Numbers of comments at each stage of corpus construction, by platform.

## human_reliability_150.csv
Comments that neither coder had seen before. Columns ending in `_A` and `_B` are the two coders.

## human_gold_507.csv
Gold standard. Columns ending in `_gold` are the final labels, `source` is `coders agreed` or `adjudicated`, and columns ending in `_A` and `_B` are the two coders. The sample was stratified by platform, model relevance and model function, so population estimates require the weights computed in `code/reproduce.py`.

## human_targeted_193.csv and targeted_strata.csv
Targeted validation of skepticism and dissent. `stratum` is `model_skepticism` or `model_other_tone`. Columns ending in `_final` are the adjudicated labels. `targeted_strata.csv` gives the population of each stratum, the number available after exclusion of comments used earlier, and the number sampled.

## human_post_audit_200.csv
Audit of the post-level relevance rule: `rule_relevant` and `human_relevant` for 100 accepted and 100 rejected posts.

## institutional_trust_targets.csv
Target and stance of the 251 comments that the primary model assigned to institutional trust, as re-coded by each model.
