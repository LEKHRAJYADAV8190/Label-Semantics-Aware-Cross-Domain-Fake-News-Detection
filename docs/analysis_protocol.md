# Analysis protocol (7 October 2026)

This analysis uses existing saved predictions; it does not retrain or select a
new model. All three thresholds (0.3, 0.4, 0.5) come from the original project
guide. The guide did not specify a score formula. We now operationalize its
uncertainty proxy as `compatibility = 1 - H(p)/ln(2)`, with
`H(p) = -sum(p_i ln(p_i))` for the two predicted class probabilities.
At a threshold t, answer when compatibility >= t; otherwise abstain.
This formula was specified after training, not preregistered before the study.
It is not optimized against test outcomes. Report all thresholds, not a chosen
winner. It is an uncertainty-only proxy; it does not measure semantics directly
or establish that label mismatch caused a particular error.

Coverage = answered / all target test samples. Risk = wrong / answered.
Answered Macro-F1 always averages both binary classes with zero_division=0.
For zero answered samples, risk and F1 are undefined (N/A, never zero).
Include forced-prediction risk and class support so apparent improvements cannot
hide class imbalance or selective coverage. No calibration claim is made.

Tables 1 and 2 use identical target test samples per model/domain comparison.
Drop = in-domain Macro-F1 minus LODO Macro-F1. Table 4 reports all four K=16
adapter runs and changes from their corresponding source model. With one seed
and a 73-item FakeVsSatire test set, small differences are descriptive only.

Ten examples are deterministic diagnostics from LODO predictions at fixed t=0.4:
prioritize a confident error, an uncertain abstention, and a correct prediction
per target, then fill distinct examples. This is illustrative, not representative
sampling or a causal error taxonomy. Excerpts are source text; neural models
actually saw only the first 128 token positions (including special tokens).

Embedding distance was added later on frozen R4/R5/R6 models. See
docs/embedding_distance_protocol.md and results/analysis/Table_5_EmbeddingDistance.csv.
It does not replace Table 3.
