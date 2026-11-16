# Dataset semantics and preparation

Processed tables live in `data/processed`. Raw inputs are not stored in this
repository. Step 1 was completed on 6 October 2026.

## ISOT

The supplied files are the **Phoenyx83 refined 2022 variant**, not the original
full ISOT release. Source card:
https://huggingface.co/datasets/Phoenyx83/ISOT-Fake-News-Dataset-FineTuned-2022
Official dataset: https://onlineacademiccommunity.uvic.ca/isot/datasets/
The variant card describes prior filtering and downsampling. Local schema and
counts agree with the card: True.csv 17,058 and Fake.csv 17,050.
ISOT file hashes are recorded locally; byte equality to that remote variant
has not been checked. Do not describe these as untouched official raw files.

Label 0 = Real, 1 = Fake. Use article body only. Normalize Unicode, HTML entities
and whitespace; remove leading Reuters dateline and '21st Century Wire says'.
Exclude one missing type, one empty/URL-only record and seven duplicate texts.
34,099 valid unique records remain before sampling. With seed 42, select 3,000
per class; each class has 2,100 train, 450 val and 450 test records.

## LIAR-PLUS

https://github.com/Tariq60/LIAR-PLUS
On 6 October 2026, all three local TSV files matched SHA-256 hashes of their
public upstream master versions fetched from raw.githubusercontent.com.
The hashes of those verified local files are in step1_audit.json.
Headerless CSV-aware TSV parsing finds 16 columns (including the extra leading
index). Zero-based column 1 is the statement ID, 2 label, 3 claim. First rows
were inspected before these positions were selected and are saved in the audit.
Use **claim only**. Exclude explanations, speaker history and other metadata.

Mapping: pants-fire / false / barely-true -> 1;
half-true / mostly-true / true -> 0. Zero unmapped labels are permitted.
An unknown label is an error, never an automatic Real label.
'Real' here includes partially true claims; it is not a guarantee of truth.

Four malformed parser records exist in the upstream training file itself:
two have four fields and two have two fields. They are excluded without trying
to reconstruct claims or labels. Also exclude 22 repeated texts and all eight
records belonging to text groups with contradictory binary labels.
Final rows: 12,759 (train 10,208; val 1,284; test 1,267).
Surviving records keep their official split. For same-label text repeats,
retain test before val before train to avoid training/evaluation leakage.
This is a cleaned official split, not a claim of unchanged official counts.

## Validation and limitations

Every output has columns id,text,label,dataset,split. No empty texts, duplicate
IDs, normalized exact duplicates, unknown labels or cross-split exact overlap
remain. No exact text overlap occurs between the two processed datasets.
Every exclusion is in step1_exclusions.csv. Raw files are unchanged.
Near-duplicate and semantic overlap have not been ruled out.
Changing source files or cleaning changes the experiment and must be logged.

FakeVsSatire Step 2 is now processed: 478 records from 486 final source files.
See dataset_validation.md, data_summary.csv and fvs_audit.json. Map Fake -> 1, Satire -> 0;
satire is not factual reporting. Use only finalFake and finalSatire, never
ditched/rejected directories. UTF-8 and Windows-1252 inputs are decoded strictly
and recorded in the audit. One pre-existing replacement glyph is documented.

## Relation to the base paper

Dell'Oglio et al., Information Sciences 745 (2026), 123407,
https://doi.org/10.1016/j.ins.2026.123407 . Read the start guide first, then the
base paper; also reviewed Problem_and_What_We_Solve.pdf.
The base paper's label mapping agrees with this project. Its Table A.1 uses
different sample sizes/splits; this is a smaller student experiment, not an
exact reproduction. The paper's scores are not our results.
Extra 7 October files: Celebrity, Cidii, Fakes (FA-KES), Horne, Infodemic, NDF
(IC labels + URL fetch), Politifact (FakeNewsNet body/title). Cidii is the
Kaggle CIDII file (Disinformation=1, Real Information=0; 719 rows after
dropping 3 duplicate texts). Audits: docs/celebrity_audit.json and sibling *_audit.json.
These extras feed Table 7 only; they do not change locked R1-R10.
Uncertainty is a proposed compatibility proxy, not proof of label-semantic
compatibility. Any improvement from abstention or LoRA still needs measurement.

## Hardware used for the scored runs

The training machine was an Intel Core i7-13700HX with 16,886,128,640 bytes of
RAM and an NVIDIA GeForce RTX 4050 Laptop GPU with 6141 MiB of video memory.
See `docs/hardware_check.json`. DistilBERT settings were max length 128, two
epochs, learning rate 2e-5, batch size 4, and seed 42.
