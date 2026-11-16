# Project overview for teacher

## Problem

A text classifier can score well on the dataset it was trained on and perform
poorly on another dataset. Writing style, topic and label meaning change together.
ISOT contains articles, LIAR-PLUS contains claims, and FakeVsSatire compares fake
articles with satire. Their binary labels do not have identical meanings.

## Base paper

Pietro Dell'Oglio, Alessandro Bondielli, Francesco Marcelloni and Lucia C. Passaro,
An experimental comparison of the most popular approaches to fake news detection,
Information Sciences 745 (2026), 123407.
https://doi.org/10.1016/j.ins.2026.123407

The base paper compares 12 methods across 10 datasets. The locked student
protocol uses three datasets and DistilBERT. An extra 7 October add-on
processes the remaining paper datasets, including Cidii, for 9-source LODO.
It is still not an exact reproduction (DistilBERT, 6k ISOT, one seed).

## Proposed work

- Compare TF-IDF + Logistic Regression with DistilBERT.
- Compare in-domain evaluation with leave-one-dataset-out (LODO) evaluation.
- Investigate an uncertainty-based compatibility score and abstention.
- Compare guided and random LoRA adaptation using K=16 labels for two targets.

These are experiments, not guaranteed improvements. Uncertainty alone does not
establish why a label-semantic mismatch occurred.

## Evaluation rules

Primary metric: Macro-F1. Also report accuracy, per-class performance, confusion
matrix, train/test counts and model checkpoint path. For abstention, report
coverage, risk and Macro-F1 on answered samples. Use seed 42.

Keep target test labels out of model training, threshold selection and LoRA
example selection. Select adaptation examples from the target training pool.
In-domain and LODO comparisons should use the same held-out target test split.
Fit TF-IDF only on training data. Keep source validation and test data separate.

## Completed evidence

Processed CSVs, audits, TF-IDF, DistilBERT R1-R10, analysis tables, report,
slides, demo and paper draft now exist. Locked DistilBERT test Macro-F1:
in-domain 0.9889 / 0.6238 / 0.7312; LODO 0.5867 / 0.3778 / 0.3917.
Abstention and K=16 LoRA did not reliably repair LODO scores.
Extra 9-source DistilBERT LODO: 0.6677 / 0.3544 / 0.7103 on the same three
targets (Cidii included). See Table 7 and report/Project_Report_30marks.docx.
The remaining packaging step is the teacher submission archive.
Do not replace saved scores with estimates. Do not overwrite R1-R10.
