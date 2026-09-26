# Method 2: TF-IDF / BM25 retrieval + weighted similarity scoring

This method separates **finding** plausible candidates from **scoring** them. A four-channel
retrieval layer replaces Method 1's exact-key blocks and gives higher recall. TF-IDF weighting
makes generic tokens ("hospital", "pvt", "traders") count for little, while distinctive tokens
("balaji", "dupont") carry the decision. This retrieval layer is reused unchanged as the candidate
generator of Method 3.

```text
S1 record ──► retrieve against S2 and against S3 separately
                ├─ name     char 2-4-gram TF-IDF cosine            top-15
                ├─ address  char 3-4-gram TF-IDF cosine            top-15
                ├─ BM25     name + address + postal tokens         top-15
                └─ postal   exact code block, ranked by name cos   top-10
          ──► reciprocal-rank fusion ──► top-25 per source ──► features ──► weighted score ──► decision
```

## Candidate generation (`retrieval.py`)

- **Channels.** Each channel is exact sparse top-k retrieval. Dense scoring is chunked, so memory stays bounded at any corpus size (`er_common.similarity.sparse_topk`). Retrieval runs separately against S2 and against S3. This per-source quota stops a source with many near-duplicate records from crowding the other source out of the top-k.
- **Fusion.** Reciprocal-rank fusion scores each pair as `Σ 1/(k + rank)` over the channels that retrieved it. The best 25 per (S1, source) survive.
- **Features for later methods.** Each pair keeps its per-channel ranks and scores, its RRF score, the number of channels that found it, and its fused rank. Method 3 uses these as features.
- **Recall.** Pair recall is 1.000 on synthetic train, against 0.995 for Method 1's blocking. The candidate cap and every k are in `RetrievalConfig` and can be traded against runtime.

## Scoring (`scorer.py`)

```text
name_score    = mean(name char-TF-IDF cos, name word-TF-IDF cos, token_sort, Jaro-Winkler)
address_score = mean(addr char-TF-IDF cos, addr word-TF-IDF cos, token_set)       (v_addr_missing if empty)
score         = w_name·name_score + (1 − w_name)·address_score
              + postal / house-number bonus − penalty − p_country·[country conflict]
match         = score ≥ τ  ∧  score ≥ best_score_of_S1 − δ  ∧  (optional) best S1 for that record
```

There are 10 parameters (weights, bonuses/penalties, τ, δ, `v_addr_missing`, exclusivity). They
are tuned exactly as in Method 1: seeded random search plus coordinate refinement on the exact
entity-level F0.5.

## Run

```bash
python -m method2_tfidf_retrieval all --data-dir dataset --out-dir outputs/method2_tfidf_retrieval
```

The outputs and the `--config` override mechanism are the same as in Method 1 (`Method2Config`,
with the `retrieval` sub-config).

## Results on the synthetic dataset (not leaderboard estimates)

| | F0.5 | macro P | macro R | singleton acc. | false-merge entities |
|---|---|---|---|---|---|
| 5-fold nested CV (train) | **0.9693** | 0.9857 | 0.9563 | 0.9832 | 50 |
| Hidden test (incl. unseen France) | **0.9685** | 0.9780 | 0.9652 | 0.9695 | 36 |

- Candidate pair recall is 1.000 on train and 0.9994 on test, with 50 candidates per S1.
- Test F0.5 by country: FR 0.956, IN 0.978, US 0.969.
- One full run takes about 31 s.

## Strengths and limits

- **Strengths:**
  - Retrieval is scalable and interpretable, and its recall is the ceiling for every later method.
  - Character n-grams tolerate typos and transliterations.
  - IDF weighting fixes Method 1's generic-token problem.
- **Limits:**
  - The matcher is still a linear score, so it cannot express interactions.
  - Most remaining errors come from hard negatives: branches of a chain, and different businesses in the same building.
  - These cases need a model that learns how name, address and number evidence combine, which is what Method 3 adds.
