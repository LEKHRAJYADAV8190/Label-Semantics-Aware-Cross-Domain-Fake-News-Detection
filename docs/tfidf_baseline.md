# TF-IDF baseline experiments

Run locally from the project root:

```powershell
.\.venv\Scripts\python.exe scripts\train_tfidf.py
```

The script runs three in-domain experiments and three leave-one-dataset-out
(LODO) experiments. In-domain models use one dataset's training split and are
evaluated on its validation and test splits. LODO models use the other two
datasets' training splits, report validation on those same source datasets,
and evaluate on the held-out target's test split. LODO does not use target
training or validation data. Test splits are identical across the two protocols.

All parameters are fixed before evaluation: seed 42, unigram/bigram TF-IDF,
minimum document frequency 2, at most 50,000 features, sublinear term frequency,
and Logistic Regression with C=1, liblinear solver and at most 1,000 iterations.
TF-IDF vocabulary and IDF weights are fit only on training text. No validation
or target labels choose hyperparameters or thresholds. Predictions use the
classifier's default decision rule. Source rows are pooled without balancing
datasets; larger sources therefore contribute more training examples.

Inputs contain only the processed article/claim text. Metadata such as speaker
history, source ID and dataset name is not a feature. Binary class 0 means Real
for ISOT, half-true/mostly-true/true for LIAR-PLUS, and Satire for FakeVsSatire.
Consequently cross-dataset scores cannot establish identical label semantics.

`results/tfidf/summary.csv` provides the teacher-facing table. Each run folder
contains `metrics.json` (including accuracy, macro-F1, per-class report,
confusion matrix, input checksums, counts and timings), validation/test
prediction CSVs with original IDs, and `model.joblib`. Matrix rows are true
labels and columns predicted labels, in the order 0, 1.

Before training, the script rejects invalid rows and any duplicate ID or
normalized exact text across all datasets and splits. Near duplicates are not
ruled out. It reloads each saved model and checks that predictions are identical.
Convergence warnings stop the run rather than accepting an unconverged fit.

Training and inference require no internet after NumPy, scikit-learn and joblib
are installed locally. Completed runs survive a stopped chat or script. On
restart, matching input, code and parameter hashes plus verified artifact
checksums allow completed runs to be skipped. An interrupted individual run is
retrained. `--force` retrains everything. Do not load untrusted joblib files.
