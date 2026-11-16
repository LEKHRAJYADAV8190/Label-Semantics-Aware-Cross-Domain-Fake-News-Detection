# Verified-result analysis workspace

Run `scripts/analyze_results.py` with the project Python to regenerate tables,
the run log and examples. It does not train any model. The independent checks
are `scripts/verify_results.py` and `scripts/verify_model_reload.py`; inspect
their JSON status files before treating verification as complete.

## Files for the teacher

- Table_1_InDomain.md / .csv: six in-domain model/dataset comparisons.
- Table_2_LODO.md / .csv: six LODO comparisons and the in-domain-to-LODO drop.
- Table_3_Abstention.md / .csv: nine fixed-threshold selective prediction results.
- Table_4_LoRA.md / .csv: four K=16 adaptation results and support class counts.
- error_abstention_examples.md / .csv: ten actual diagnostic examples.
- R4/R5/R6_selective_predictions.csv: per-item compatibility and answer decisions.
- independent_verification.json: all test metrics and split/label verification.
- model_reload_verification.json: offline reload prediction checks on samples.
- ../../docs/run_log.csv: consolidated 16 locked training runs.
- Table_6_ImprovementExtras.md / .json: extra K=64/weighted runs; R1-R10 untouched.
- Table_7_TenDatasetExtras.md / .json: extra DistilBERT in-domain + 9-source LODO (Cidii included).
- Table_7b_TFIDF_TenDataset.md: extra TF-IDF on the same nine processed datasets.
- Table_5_EmbeddingDistance.md / .csv: frozen-model embedding distance vs uncertainty.
- R4/R5/R6_embedding_predictions.csv: per-item uncertainty, centroid, NN and rank scores.
- ../../docs/embedding_distance_protocol.md: distance formulas; no test-label tuning.

CSV values are full precision; Markdown values are rounded for readability.

## What the current results say

DistilBERT in-domain Macro-F1 is 0.988889 (ISOT), 0.623827 (LIAR-PLUS), and
0.731178 (FakeVsSatire). LODO scores are 0.586746, 0.377758 and 0.391667.
The drops are large in all three domains. The simpler TF-IDF baseline has a
higher LODO Macro-F1 on all three targets in this single-seed experiment.

The uncertainty-only compatibility rule does not consistently improve selective
prediction. On ISOT, threshold 0.3 answers only 11/900 cases, so the lower risk
there applies to a tiny subset; thresholds 0.4/0.5 answer 4/2 cases and have worse
risk. LIAR-PLUS keeps over 93% coverage at every threshold but error risk remains
about 54%. On FakeVsSatire, thresholds 0.3 and 0.4 abstain on nothing, and 0.5
abstains on two cases without reducing risk. This is evidence that uncertainty
alone is insufficient here, not evidence of a generally reliable detector.

Guided LoRA on LIAR improves Macro-F1 by 0.003601 (about 0.36 percentage points),
while Random improves it by 0.000270. Neither approach changes FakeVsSatire's
aggregate classification metrics. These are small descriptive differences, not
statistically established improvements. Only 16 target training labels and two
epochs were used: eight optimizer steps per adapter. Guided sets are class-skewed
(LIAR 13 Real/3 Fake; FVS 2 Satire/14 Fake), which further limits interpretation.

The project demonstrates the cross-domain failure problem. It does not establish
that the proposed uncertainty proxy or tiny adaptation reliably solves it.

The optional embedding-distance term was added on frozen R4/R5/R6 models
(scripts/embedding_distance.py). Raw cosine to the predicted source-class
centroid is high on almost every LODO test row (typically 0.6–0.96), so the
guide thresholds 0.3/0.4/0.5 do not abstain extra cases. Mapping that cosine
through the source-validation empirical CDF does change coverage: on ISOT at
t=0.4 it answers 765/900 with risk 0.4144 (slightly worse than forced 0.3811);
on LIAR-PLUS it answers 0–1 rows; on FakeVsSatire at t=0.4 it answers 11/73
with risk 0.3636 versus forced 0.5753 and answered Macro-F1 0.5417 versus
0.3917. The FakeVsSatire slice has 11 rows, so it is descriptive only. Distance
did not raise forced LODO accuracy. It completes the guide’s optional Gap-1
term and shows that CLS cosine is a weak OOD detector after these fine-tunes.

Extra improvement runs (7 October 2026, new IDs E1–E7, R1–R10 untouched):
K=64 class-balanced LoRA (10 epochs, rank 16, lr 2e-4) and class-weighted LODO.
See Table_6_ImprovementExtras.md. Largest honest gains: LIAR E2 accuracy 0.5328
(from R5 0.4538) and FakeVsSatire E4 accuracy 0.5890 (from R6 0.4247, n_test=73).
These extras are not replacements for the locked 30-mark tables.

Extra 9-source DistilBERT LODO Macro-F1 (7 October 2026, Cidii included):
ISOT 0.667705 (locked R4 0.586746), LIAR-PLUS 0.354376 (R5 0.377758),
FakeVsSatire 0.710317 (R6 0.391667). Extra TF-IDF LODO ISOT 0.634562.
Paper DeBERTa LODO (their Table 12) is 0.83 / 0.45 / 0.60 on those three
names; cite as theirs only. Rebuild writing with scripts/build_deliverables.py;
do not overwrite R1-R10.

## Limits that must appear in the report

Locked protocol: three datasets. Extra add-on: ten datasets, Cidii included.
One seed, a refined ISOT variant and balanced ISOT subsample,
heterogeneous label meanings, exact-deduplication only, max input length 128,
small FVS test set (73), no calibrated uncertainty. Embedding distance was added
after the original tables on frozen LODO models; it does not replace Table 3.
Thresholds were fixed by the guide and all are reported; none was tuned on test.
R3 finishes before the 250-step validation interval and uses its final two-epoch
model, while longer core runs choose source-validation checkpoints. Adapter
runs use final two-epoch weights and no target validation tuning.
