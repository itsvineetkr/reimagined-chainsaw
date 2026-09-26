# Business Entity Resolution: Methods 1–3

For every Source 1 business, the task is to find all Source 2 and Source 3 records that describe
the same real-world entity. Scoring is the macro-averaged **F0.5** per S1 entity, which weighs
precision above recall.

This repository implements the first three of the five planned methods, one folder each. All three
share a single core, `er_common/`, for IO, normalisation, the metric, folds, the decision layer and
the CLI. Their scores are therefore directly comparable. Methods 4 (bi-encoder) and 5
(cross-encoder + graph) are deliberately left for later.

| Folder | Method | Candidate generation | Matcher |
|---|---|---|---|
| [`method1_fuzzy_rules/`](method1_fuzzy_rules/README.md) | 1. Multi-field fuzzy + rules | exact-key blocks | rules with tuned thresholds |
| [`method2_tfidf_retrieval/`](method2_tfidf_retrieval/README.md) | 2. TF-IDF/BM25 retrieval + fuzzy scoring | 4-channel retrieval + reciprocal-rank fusion | tuned weighted score |
| [`method3_gbdt/`](method3_gbdt/README.md) | 3. GBDT + hard negatives (**primary**) | Method 2 retrieval | LightGBM + F0.5 decision policy |
| `er_common/` | shared core | | |

## Quick start

```bash
pip install -r requirements.txt           # numpy, pandas, scipy, scikit-learn, rapidfuzz, lightgbm (all pinned)

# challenge data: dataset/train/train_source{1,2,3}.tsv, train_ground_truth.tsv, dataset/test/test_source{1,2,3}.tsv
python -m method3_gbdt all --data-dir dataset --out-dir outputs/method3_gbdt
#   -> outputs/method3_gbdt/validation/report.json          cross-validated F0.5 + diagnostics
#   -> outputs/method3_gbdt/validation/errors.tsv           every FP / FN / blocking miss, side by side
#   -> outputs/method3_gbdt/test/matching_results.tsv       submission file
#   -> outputs/method3_gbdt/test/candidate_pairs.tsv        submission file (exact matcher input)

python scripts/make_submission.py --method method3_gbdt --team <team> \
    --data-dir dataset --doc path/to/filled/Documentation_template.md
#   -> submissions/<team>_submission.zip in the required layout (output/, code/…/src, README, requirements)
```

The same commands work for `method1_fuzzy_rules` and `method2_tfidf_retrieval`.
`validate` and `predict` run the two halves separately.

**No data yet?** Generate a synthetic dataset with the same layout and noise patterns, including a
test-only France split with hidden labels:

```bash
python scripts/make_synthetic_dataset.py --out dataset_synth
python -m method3_gbdt all --data-dir dataset_synth \
    --test-gt dataset_synth/test/_hidden_test_ground_truth.tsv
```

## Results on synthetic data

> **Caveat:** the challenge data was not available while building this. The numbers below come
> from the synthetic generator (`er_common/synth.py`): 3 000 train S1 and 1 500 test S1, where the
> test set contains 30 % France. They show relative ordering and that the pipelines work end to
> end. They are **not** leaderboard estimates. Every threshold and weight is re-tuned on whatever
> training data is supplied.

| Method | CV F0.5 (nested, train) | Test F0.5 | France F0.5 (unseen in train) | Test false-merge entities | Candidate recall | Runtime |
|---|---|---|---|---|---|---|
| 1. Fuzzy + rules | 0.9682 | 0.9657 | 0.9564 | 31 | 0.9953 | 26 s |
| 2. TF-IDF retrieval + scoring | 0.9693 | 0.9685 | 0.9563 | 36 | 1.0000 | 31 s |
| 3. GBDT + hard negatives | **0.9889** | **0.9892** | **0.9877** | **16** | 1.0000 | 55 s |

- The false-merge column counts S1 entities with at least one wrong match.
- Candidate recall is pair recall on train.
- Runtime is validate + predict on 4 cores.

The synthetic data plants specific hard cases:

- chain branches (the same brand at several addresses, several of them in S1);
- same-name branches in the same postal area;
- different businesses at the same address;
- near-identical names in the same city;
- singletons;
- noisy variants of each record: transliterations (Shree/Sri, Mohammed/Mohd), abbreviations, typos, dba trade names, reordered and partial addresses, landmark phrases, PIN codes written as `226 001`, ZIP+4, renamed cities (Bombay/Mumbai) and country-label aliases.

## Additions beyond the brief

| Area | Addition | Why it matters for F0.5 |
|---|---|---|
| Evaluation | **Nested 5-fold CV over S1 entities** (stratified by country × #matches) instead of one holdout. Every tuned parameter and threshold is fitted inside the fold | Honest, lower-variance estimate; the per-fold spread is reported |
| Evaluation | The metric counts GT matches **missed by blocking** | Recall lost in candidate generation is charged to the score |
| Evaluation | **Leave-one-country-out** diagnostic in Method 3 | Proxy for the unseen-France shift |
| Normalisation | Canonical token maps (road/rd, private/pvt, centre/center/ctr, saint/st, shree/sri, …), state names → codes, renamed cities, initialism joining (S.B.I.→sbi), dba/trade-name split, landmark extraction capped at 2 words, postal/house-number extraction that survives reordered components, open-set country aliases | Both sides map to the same tokens; identity numbers are kept, not deleted |
| Normalisation | Transliteration **consonant skeleton** (shree/sri → `sr`, mohammed/muhammad → `mhmd`) as a feature and a blocking key | Indian name spelling variants |
| Candidates | **Per-source quotas**: retrieval runs against S2 and S3 separately | One source's near-duplicates cannot push the other out of the top-k |
| Decision | **Exclusivity**: an S2/S3 record goes to at most one S1. Hard and **soft** (categorical-with-null) variants, selected by CV | Removes false merges on chains; the GT is checked for the constraint at start-up |
| Decision | **Bayes-optimal expected-F0.5 top-k** per entity, from calibrated probabilities | Handles singletons and multi-matches without an ad-hoc threshold |
| Model | Reverse-rank and competition features; name-frequency (chain) features; **monotone constraints** | Chains and branches, which are the main source of false merges; robustness under domain shift |
| Engineering | Deterministic end to end (checked across hash seeds); run manifest with data SHA-256, package versions, git commit and config; output validator for every challenge rule; robust TSV reading (`"NA"` stays a name; stray quotes are detected); `mypy --strict` and ruff clean; 28 tests including brute-force checks of the expected-F solver | Reproducibility and submission safety |
| Packaging | `scripts/make_submission.py` builds the exact ZIP layout, vendors only the needed packages and re-validates the outputs | One command to a valid submission |

## Layout

```text
er_common/              shared core
  io.py                 TSV loading, GT parsing, output writer + constraint validator
  normalize.py          names / addresses / countries → canonical fields
  similarity.py         vectorised pairwise fuzzy + TF-IDF + token-set similarities, sparse top-k
  metrics.py            exact leaderboard F0.5, diagnostics, fast grouped evaluator
  decision.py           threshold / margin / exclusivity / expected-F0.5 / Platt calibration
  folds.py, tuning.py   stratified S1 folds, derivative-free parameter search
  pairs.py, runner.py   candidate pair table, CLI (validate / predict / all), manifests, error dump
  synth.py              synthetic data generator (tests / demos only; excluded from submissions)
method1_fuzzy_rules/    blocking.py, rules.py, method.py
method2_tfidf_retrieval/retrieval.py, scorer.py, method.py
method3_gbdt/           features.py, model.py, policy.py, method.py
scripts/                make_synthetic_dataset.py, make_submission.py
tests/                  unit tests + end-to-end pipeline tests
```

## Development

```bash
pip install -r requirements-dev.txt
python -m pytest -q                         # 28 tests, ~15 s
ruff check . && ruff format --check .
python -m mypy er_common method1_fuzzy_rules method2_tfidf_retrieval method3_gbdt scripts
```

## Constraints check

- **Runs offline.** No external lookups, APIs or business data. The abbreviation and alias tables in `normalize.py` are generic linguistic normalisation.
- **Licences and model size.** Every dependency is MIT, BSD or Apache-2.0. The model is a LightGBM ensemble of a few hundred trees, far below 8 B parameters.
- **Country is an open set.** Countries are only normalised aliases and never one-hot encoded. Every test S1 entity gets exactly one row in both output files.
- **Output rules are enforced.** Match lists contain only test S2/S3 ids, have no duplicates, and are always a subset of the candidate list. `candidate_pairs.tsv` is exactly the matcher's input.

## Next: Methods 4 and 5

Method 3 is built to absorb them:

- **Method 4.** Add a bi-encoder cosine as one more column in `method3_gbdt/features.py`, and optionally as a fifth retrieval channel in `method2_tfidf_retrieval/retrieval.py`.
- **Method 5.** Add a cross-encoder probability as a feature. Graph consistency (S2↔S3 agreement) can be built on the soft-exclusivity step in `method3_gbdt/policy.py`.

Keep each addition only if the nested CV and the leave-one-country-out score improve.
