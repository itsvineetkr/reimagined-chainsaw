# Method 3: supervised pairwise GBDT + retrieval blocking + hard negatives

**Recommended primary solution.** Methods 1 and 2 compute similarities and combine them by hand.
Here LightGBM learns how about 80 similarity, retrieval and competition signals combine. A
separate, validated decision layer then turns the probabilities into F0.5-optimal match sets.

```text
S1/S2/S3 ─► normalise ─► Method 2 retrieval (4 channels, RRF, per-source quota)
         ─► ~80 pair features ─► LightGBM P(match) ─► Platt calibration
         ─► exclusivity (none / hard / soft) ─► strategy (expected-F0.5 top-k | threshold+margin) ─► matches
```

## Training pairs and hard negatives

Training pairs are exactly the candidates the retrieval layer produces for the training S1
records, labelled with the ground truth. Every negative was retrieved because it resembles the S1
record in at least one channel: a same-name branch, a same-building neighbour, a shared postal
code, or a near-identical name. These are the hard negatives described in the brief. The training
set contains no random, easy negatives. Train and inference also see the same pair distribution,
which keeps the probabilities calibrated.

## Features (`features.py`)

| Group | Examples |
|---|---|
| Retrieval | per-channel rank and score, RRF, number of channels, fused rank, S2/S3 flag |
| Name | char and word TF-IDF cosine; ratio, partial, token-sort, token-set, Jaro-Winkler, Levenshtein; transliteration-skeleton ratio; IDF-weighted Jaccard; containment; exact flags; acronym (SBI ↔ State Bank of India); best trade-name (dba) similarity; legal-form agreement; first/last-token agreement; length deltas |
| Address | char and word TF-IDF cosine; fuzzy scores on the core and full address; landmark similarity; IDF-weighted Jaccard; postal equal / conflict / missing; postal 3-digit prefix; house number; numeric-token overlap and conflicts |
| Frequency | how common the name is in S1 and in S2/S3 (chains: many branches share a name, so a name match is weaker evidence) |
| Context (per S1) | this candidate's rank and gap to the S1's best; number of strong-name and strong-address candidates |
| Context (per candidate) | number of S1 records competing for this S2/S3 record; the margin over the best competing S1 (a reverse rank). This separates branches of a chain that are all in S1 |

There is **no country one-hot**. The only country signal is label agreement, so France (absent
from training) is handled by the same features. The model is also trained with **monotone
constraints**: the core similarity features may only increase P(match). This guards against
spurious, non-monotone splits that would not transfer to an unseen country.

## Model (`model.py`)

- **Hyperparameters.** LightGBM binary logloss, learning rate 0.03, 31 leaves, feature and bagging fraction 0.8, L2 = 1.
- **Early stopping.** 15 % of the *training S1 groups* are held out for early stopping, never the fold being evaluated.
- **Final model.** Refit on all training pairs for median(best_iter) / 0.85 rounds.
- **Determinism.** `deterministic=True`, `force_row_wise`, fixed thread count and seeds. Two runs with different `PYTHONHASHSEED` give byte-identical outputs.

## Decision layer (`policy.py`)

The metric is averaged per S1 entity and weighs precision above recall, so the decision is part of
the model.

1. **Platt calibration** is fitted on out-of-fold probabilities.
2. **Exclusivity.** Each S2/S3 record belongs to at most one S1, because S1 is deduplicated. The training ground truth confirms this at start-up; the log prints the maximum number of S1 owners per record.
   - `hard`: only the best S1 keeps the record.
   - `soft`: ownership is one categorical choice among the competing S1 records plus "no owner", with qᵢ = oᵢ / (1 + Σⱼ oⱼ) and o = p / (1 − p). A record claimed by a single S1 keeps its probability. Two strong claims (0.9 / 0.9) both fall to 0.47, so ambiguous records are withheld.
3. **Strategy.**
   - `expected_f`: for each S1, predict the top-k set that maximises the expected F0.5 over the independent-Bernoulli label distribution. This is Bayes-optimal given calibrated probabilities, is computed exactly with Poisson-binomial dynamic programming, and is unit-tested against brute-force enumeration. It returns 0, 1 or many matches without a global threshold, so singletons are handled by construction.
   - `threshold`: q ≥ τ and q ≥ best_q_of_S1 − δ.
4. Every (exclusivity × strategy × τ × δ) combination is scored on out-of-fold probabilities, and the best one is kept.

## Validation protocol

There are 5 folds over S1 entities, stratified by country × {0, 1, 2+} true matches.

- **Out-of-fold probabilities.** Every training pair gets a probability from a model that never saw its S1 entity's labels.
- **Nested decision layer.** The decision layer is fitted on four folds' out-of-fold probabilities and scored on the fifth. The reported F0.5 is therefore out-of-sample for both the model and the thresholds.
- **Exclusivity.** Competition between S1 records uses all S1 records, as it does on test.
- **Leave-one-country-out.** The model is also trained without one country and scored on it, as a proxy for the France shift.

## Run

```bash
python -m method3_gbdt all --data-dir dataset --out-dir outputs/method3_gbdt
```

In addition to Method 1's outputs, `validation/report.json` contains:

- the decision-layer ablation table;
- the leave-one-country-out scores;
- each fold's selected policy.

`test/manifest.json` contains the final policy, the number of boosting rounds and the top feature
gains.

## Results on the synthetic dataset (not leaderboard estimates)

| | F0.5 | macro P | macro R | singleton acc. | false-merge entities |
|---|---|---|---|---|---|
| 5-fold nested CV (train) | **0.9889** | 0.9875 | 0.9925 | 0.9787 | 37 |
| Hidden test (incl. unseen France) | **0.9892** | 0.9870 | 0.9933 | 0.9739 | 16 |

- Test F0.5 by country: FR **0.988**, IN 0.988, US 0.992.
- One full run takes about 55 s: 5 out-of-fold models, the leave-one-country-out models and the final fit.

**Decision-layer ablation (nested-CV F0.5):**

| Variant | F0.5 |
|---|---|
| raw p ≥ 0.5 | 0.9876 |
| tuned threshold + margin | 0.9886 |
| tuned threshold + margin + exclusivity | 0.9893 |
| expected-F0.5 top-k | 0.9881 |
| expected-F0.5 + soft exclusivity | 0.9881 |
| auto (selected per fold) | 0.9889 |

On this data the model is confident enough that the decision variants differ by about 1–5
entities out of 3 000, which is within noise. The variants are expected to separate on real data,
where probabilities are less extreme. The selection is automatic either way.

**Leave-one-country-out:**

| Held-out country | trained without it | in-distribution CV |
|---|---|---|
| India | 0.974 | 0.989 |
| US | 0.987 | 0.989 |

An unseen country costs up to about 1.5 points. On the real data, this gap is the number to watch
for France.

**Top feature gain shares:**

| Feature | Gain share |
|---|---|
| margin over the best competing S1 | 0.55 |
| combined name/address score | 0.22 |
| name × address | 0.10 |
| numeric containment | 0.05 |

## Strengths and limits

- **Strengths:**
  - Learns the interactions the rules could not express, such as "same name, same postal code, different house number means another branch".
  - Uses the ground truth directly.
  - Every stage is validated by nested CV.
- **Limits:**
  - Recall is capped by retrieval: the 25 candidates per source per S1.
  - Semantic equivalences that string similarity cannot see (for example "Saint Mary's Medical Center" vs "St Marys Hospital") need Method 4's learned encoder. The feature and policy code is built to take such a score as one more column.
