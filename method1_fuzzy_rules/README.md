# Method 1: multi-field fuzzy matching + tuned rules

This is the baseline and diagnostic system. Names and addresses are compared field by field with
`rapidfuzz` string similarities, and a small set of rules decides MATCH / NO MATCH. Every
threshold and weight in the rules is tuned on entity-level F0.5, the leaderboard metric. None of
them are hand-fixed.

```text
S1 / S2 / S3 ──► normalise (er_common) ──► exact-key blocking ──► fuzzy features ──► rules ──► exclusivity + margin ──► matches
```

## Pipeline

| Stage | What happens | Code |
|---|---|---|
| Normalisation | Unicode folding, `&`→and, abbreviation canonicalisation (Pvt/Private→pvt, Rd/Road→rd, Shree/Sri→sri), legal-suffix stripping, initialism joining (S.B.I.→sbi), dba/trade-name split, landmark extraction ("Near SBI ATM"), postal/house-number extraction, open-set country labels | `er_common/normalize.py` |
| Blocking | Exact keys: postal code, postal + name initial, each name-token skeleton, name-token bigrams, acronym, house number + street token. Keys whose S1×S2/S3 product exceeds `max_block_pairs` are skipped as too generic; an S1 left with no candidate falls back to its smallest block. At most 25 candidates per S1 per source are kept, ranked by a token-set pre-score | `blocking.py` |
| Features | Name: token-sort, token-set, Jaro-Winkler, transliteration-skeleton ratio, acronym, trade-name best. Address: token-set and token-sort on the landmark-free core. Context: postal, house number and country, each as equal / conflict / missing | `rules.py::fuzzy_features` |
| Rules | veto on country conflict; **strong** (name ≥ t₁ ∧ addr ≥ t₂); **address rule** (addr ≥ t₃ ∧ name ≥ t₄); **score rule** (evidence ≥ τ) | `rules.py::RuleEngine` |
| Decision | Optional exclusivity (an S2/S3 record goes only to its best S1). Per S1, keep candidates within δ of the S1's best evidence | `er_common/decision.py` |

```text
evidence = w_name·name + (1 − w_name)·address
         + b_postal·[postal equal] − p_postal·[postal conflict]
         + b_house·[house equal]   − p_house·[house conflict]
address  = v_addr_missing   when either address is empty (tuned, not assumed to be a mismatch)
```

## Tuning

There are 13 parameters: the weights, bonuses and penalties, t₁…t₄, τ, δ, `v_addr_missing`, and an
exclusivity on/off switch. They are searched with 600 seeded random draws, started from sensible
defaults, followed by 3 rounds of coordinate refinement on a shrinking grid (`er_common/tuning.py`).
The objective is the exact per-entity macro F0.5, evaluated in a vectorised way
(`GroupedEvaluator`). The metric counts ground-truth matches that blocking missed, so a recall loss
in blocking is charged to the score. One tuning run takes about 3 s for 150k pairs.

## Run

```bash
python -m method1_fuzzy_rules validate --data-dir dataset --out-dir outputs/method1_fuzzy_rules  # 5-fold CV
python -m method1_fuzzy_rules predict  --data-dir dataset --out-dir outputs/method1_fuzzy_rules  # test files
python -m method1_fuzzy_rules all      --data-dir dataset                                         # both
```

Outputs:

- `validation/report.json`: CV F0.5 with its precision/recall split, singleton accuracy, per-fold and per-country scores, blocking recall and the oracle ceiling, the tuned parameters per fold, data hashes and package versions.
- `validation/errors.tsv`: every false positive, false negative and blocking miss, side by side.
- `test/matching_results.tsv` and `test/candidate_pairs.tsv`: the submission files. They are validated against every challenge rule before they are written.

`--config overrides.json` changes any field of `Method1Config`, for example
`{"n_random": 1500, "blocking": {"max_block_pairs": 8000, "max_cands_per_source": 40, "prescore_name_weight": 0.6}}`.

## Results on the synthetic dataset

These numbers are not leaderboard estimates. The synthetic data only mimics the documented noise
patterns (see the root README), and the method re-tunes itself on the real training data.

| | F0.5 | macro P | macro R | singleton acc. | false-merge entities |
|---|---|---|---|---|---|
| 5-fold nested CV (train, 3 000 S1) | **0.9682** | 0.9844 | 0.9537 | 0.9821 | 57 |
| Hidden test (1 500 S1, incl. unseen France) | **0.9657** | 0.9811 | 0.9552 | 0.9760 | 31 |

- Blocking pair recall is 0.9953 on train, with about 48 candidates per S1.
- Test F0.5 by country: FR 0.956, IN 0.977, US 0.961.
- One full run (validate + predict) takes about 26 s on 4 CPU cores.

## Strengths and limits

- **Strengths:** transparent and fast. Error analysis is easy, because every decision traces to a named rule and a number.
- **Limits:**
  - Rules combine evidence additively. They cannot learn interactions, such as "identical name but a different house number in the same postal code means another branch".
  - Generic tokens ("hospital", "traders", "pvt") weigh as much as distinctive ones, because there is no IDF weighting. Method 2 adds it.
  - Exact-key blocking loses matches whose name and address are both corrupted (pair recall 0.995 vs 1.000 for Method 2's retrieval).
