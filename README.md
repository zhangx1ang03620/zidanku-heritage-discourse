# Replication package: public responses to the return of the Zidanku Silk Manuscripts

Code and de-identified data for the article

> Zhe Wang, Xiang Zhang, Tianchen Luo, ChaoFan Yue, Jinjiang Liu, RongQi Zhang, Alina Andreeva, Jiahao Yuan, Chunhong Yuan and Xiaoge Zhang. Public responses to the return of cultural heritage on Chinese social media: a validated large language model analysis of the Zidanku Silk Manuscripts. Manuscript submitted to *npj Heritage Science* (2026).

The study analyzes 19,128 public comments from Bilibili, Douyin, Xiaohongshu, Weibo and Zhihu on the 2025 return of two volumes of the Zidanku Silk Manuscripts to China. A large language model coded every comment for relevance, tone and discursive function, and each result was accepted only after validation against a human gold standard, inference clustered on posts and control for the type of posting account.

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python code/reproduce.py
```

`code/reproduce.py` needs only the files in `data/`. It recomputes 62 quantities reported in the article and prints each next to the published value; it exits with status 0 when all agree. It takes about one minute and needs no network access and no credentials.

## Contents

| Path | Content |
|---|---|
| `data/` | De-identified analysis data (see `docs/data_dictionary.md`) |
| `code/reproduce.py` | Recomputes the headline results from `data/` |
| `code/permutation_high_precision.py` | Repeats the permutation tests with 200,000 draws |
| `code/original_pipeline/` | Scripts used for ingestion, model annotation, sampling and validity estimation, as run |
| `code/original_analysis/` | Scripts used for the statistical analysis, the decision rule and the figures, as run |
| `prompts/` | Model prompt (Chinese original). An English translation is in the Supplementary Information of the article |
| `results/tables/` | Every result table produced by the analysis (CSV) |
| `results/figures/` | Figures 3 to 9 of the article (PDF) |
| `docs/` | Data dictionary, analysis plan and its later changes, notes on the original code |

## What can be reproduced from this package

`code/reproduce.py` reproduces, from the released data alone:

- corpus counts, the concentration of comments in posts and the counts by phase;
- the composition of discourse and the share of comments under official posts on each platform;
- the platform tests for all nine categories: cluster-robust Wald tests and post-level permutation tests with Holm correction (identical to `results/tables/T5_*.csv`);
- agreement between the two human coders, between the model and the gold standard, between two runs of the primary model and between the two models;
- precision, recall and corrected shares after reweighting to the sampling strata;
- the audit of the post-level relevance rule;
- the targeted validation of skepticism and dissent.

The scripts in `code/original_pipeline/` and `code/original_analysis/` are the ones that produced the article. They read two SQLite databases that contain the comment text and are therefore not part of this package (see below), so they are provided for inspection and do not run as shipped. `docs/original_code_notes.md` maps each script to the tables and figures it produced.

## What is not included, and why

- **Comment and post text, account names, URLs and platform identifiers.** The comments were posted publicly, but they were written by private individuals and are subject to the terms of service of the platforms. Comments and posts are identified here by salted SHA-256 digests that cannot be mapped back to the platforms. Exact timestamps are reduced to dates.
- **The human coding manual.** It quotes real comments as examples. The category definitions and the full model prompt are provided.
- **The crawler.** Collection used the open-source crawler MediaCrawler (https://github.com/NanmiCoder/MediaCrawler) between 18 August and 19 September 2026. The search terms are listed in `results/tables/T2_keyword_precision.csv` and in Supplementary Table S1 of the article.
- **Credentials.** No API key is included. `code/original_pipeline/llm_annotate.py` reads a key from an environment file that each user must supply.

Researchers who need the comment text for replication may contact the corresponding author, Xiaoge Zhang (xiaogezhang@graduate.utm.my). It can be shared for non-commercial research, subject to the terms of service of the platforms.

## Reproducibility notes

- All random operations use fixed seeds, which are set in the code (20260921 for the analysis, 20261009 and 20261010 for the validation samples).
- The language models are proprietary services. Their outputs may not be reproducible after the provider updates or retires the versions used on 21 September 2026, which is why every label they produced is stored in `data/model_labels.csv`.
- Permutation tests used 2,000 draws, as fixed in the analysis plan. Two Holm-corrected values lie close to 0.05 and carry visible Monte Carlo error at that number of draws. `code/permutation_high_precision.py` repeats all three families of permutation tests with 200,000 draws (about four minutes; output in `results/tables/T17_permutation_high_precision.csv`). No classification changes: for the platform difference in neutral tone with account type held fixed, *P* = 0.054 (reported 0.060); for historical debate, *P* = 0.046 without and 0.105 with account type held fixed (reported 0.062 and 0.112).
- Software: Python 3.13.5, pandas 2.2.3, NumPy 2.1.3, statsmodels 0.14.4, scikit-learn 1.6.1, SciPy 1.15.3, matplotlib 3.10.0.

## Ethics

The study is an observational analysis of publicly posted comments. It involved no human participants, no interaction with users, no intervention and no identifiable personal data, and it was therefore not submitted for ethics review. No comment text or account information is released.

## Use of AI tools

Two large language models (gpt-5.6-luna and gpt-5.6-terra, OpenAI) were the coding instrument of the study. GPT-5.6 (OpenAI) was used to polish the language of the article. An AI assistant (Claude, Anthropic) was used to write analysis code, prepare figures and draft the article and this documentation under the direction of the authors. All code and text were checked by the authors, who take responsibility for them.

## License

Code is released under the MIT License (`LICENSE`). Data, tables and documentation are released under the Creative Commons Attribution 4.0 International License (`LICENSE-DATA.md`).

## Citation

Please cite the article and this package. See `CITATION.cff`. Archived version: [Zenodo DOI to be added after the first release].
