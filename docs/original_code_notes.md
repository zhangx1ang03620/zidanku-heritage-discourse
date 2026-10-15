# Notes on the original code

The scripts in `code/original_pipeline/` and `code/original_analysis/` are provided as they were run. Comments and messages in them are in Chinese. They expect the private project layout and two SQLite databases (`heritage.sqlite` with comment and post text, `annotations.sqlite` with model labels), which are not distributed. Their outputs are in `results/tables/`.

| Script | Purpose | Output |
|---|---|---|
| `original_pipeline/ingest.py` | Reads crawler output, removes duplicates, applies the post-level relevance rule | databases |
| `original_pipeline/llm_annotate.py` | Sends comments to the language model with the fixed prompt and stores the labels | `data/model_labels.csv` |
| `original_pipeline/sample_for_review.py`, `sample_round3.py` | Draw the stratified samples for human coding | coding sheets |
| `original_pipeline/merge_human_coding.py`, `round3_merge.py`, `kappa.py` | Check and merge the coding sheets, compute agreement | reliability and gold standard |
| `original_pipeline/validity_weighted.py` | Precision, recall and shares after reweighting to strata | `T15` |
| `original_pipeline/build_author_type_binary.py` | Builds the account-type variable | `official_account` |
| `original_pipeline/sample_targeted_validation.py`, `round4_targeted.py` | Targeted validation: sampling and analysis | `T16`, `T16b` |
| `original_analysis/build_dataset.py` | Builds the analysis dataset | `data/comments.csv` |
| `original_analysis/analyze.py` | Composition, platform and phase tests, robustness checks, likes | `T1` to `T11` |
| `original_analysis/decide.py` | Applies the decision rule fixed in the analysis plan | `T12` |
| `original_analysis/recheck_with_author_type.py` | Repeats the tests with account type held fixed | `T14a` to `T14e` |
| `original_analysis/decide_final.py` | Final classification of findings | `T12b` |
| `original_analysis/distrust_targets.py` | Re-codes institutional-trust comments by target and stance | `T13`, `T13b` |
| `original_analysis/figures_v2.py`, `figures_v3.py` | Figures 3 to 9 | `results/figures/` |

The analysis plan was fixed on 21 September 2026 before the full-corpus results were examined. The plan, the one software defect corrected afterwards, the criteria that were made stricter and the two analyses that were added are described in the Supplementary Methods of the article.
