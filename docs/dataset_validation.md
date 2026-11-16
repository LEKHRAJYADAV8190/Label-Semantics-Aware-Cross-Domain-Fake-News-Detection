# Dataset validation

Validation date: 6 October 2026 (Asia/Calcutta). Project: `C:\ml\fake-news-30`.
Raw inputs remain in `C:\ml\dataset\raw`. This report concerns data preparation only.

## Result

All three processed CSVs reproduce the current processing scripts exactly, including
every exclusion record. Their output SHA-256 hashes and all 491 audited raw file
hashes match the saved audit manifests. The data are usable for the planned student
experiment, with the provenance and semantic limitations below.

| Dataset | Raw records/files | Processed | Train | Validation | Test |
| --- | ---: | ---: | ---: | ---: | ---: |
| ISOT variant | 34,108 | 6,000 | 4,200 | 900 | 900 |
| LIAR-PLUS | 12,793 parsed records | 12,759 | 10,208 | 1,284 | 1,267 |
| FakeVsSatire | 486 final text files | 478 | 334 | 71 | 73 |

Per-class counts and label meanings are in [data_summary.csv](data_summary.csv).

## What was checked

- Independently reconstructed parsing, normalization, binary mapping, exact-text
  duplicate removal, and seed-42 sampling/split assignment from the raw inputs.
- Replayed both processing scripts in memory with all file writes intercepted;
  all three output tables and both exclusion ledgers matched the saved files.
- Verified columns `id,text,label,dataset,split`, nonempty text, binary labels,
  dataset identifiers, split names, unique IDs, and audit split counts.
- Verified zero normalized exact-text duplicates within each processed dataset,
  hence zero exact overlap between its train, validation and test subsets.
- Verified zero normalized exact-text overlap between any pair of processed datasets.
- Confirmed all source/output hashes against `step1_audit.json` and `fvs_audit.json`.
- Examined FakeVsSatire encoding selection and ensured only `finalFake` and
  `finalSatire` directories contributed records.

## ISOT provenance and exclusions

The supplied ISOT files are the Phoenyx83 refined 2022 variant, as documented in
[semantics_notes.md](semantics_notes.md), rather than untouched original ISOT files.
Local raw counts are 17,058 True and 17,050 Fake. The variant has a `type,text`
schema. Remote byte equality for this variant has not been established.

One missing/invalid label, one empty/URL-only text, and seven duplicate texts are
excluded. This leaves 34,099 valid unique records before sampling. A seed-42
balanced sample keeps 3,000 per class: 2,100 train, 450 validation and 450 test.
The 28,099 valid records outside the sample are intentionally unused; they are not
invalid records and do not belong in the exclusion ledger. Label 0 means Real and
label 1 means Fake. Leading Reuters datelines and the named source phrase are removed.

## LIAR-PLUS provenance and exclusions

Previous source verification, recorded in `semantics_notes.md`, found all three
local files byte-identical to upstream LIAR-PLUS master files on 6 October 2026.
This validation rechecked the local hashes against that manifest; it did not
perform a new remote fetch. The verification is a dated master snapshot, not a
claim that a moving branch will always retain the same bytes.

CSV-aware TSV parsing yields 10,242 train, 1,284 validation and 1,267 test records.
The extra leading index makes the statement ID column 1, label column 2, and claim
column 3 (zero-based). Training uses claim text only, without explanations or
speaker-history metadata. All six labels are recognized:

- `pants-fire`, `false`, `barely-true` map to 1.
- `half-true`, `mostly-true`, `true` map to 0.

Four malformed training parser records (two with four columns and two with two
columns), 22 repeated texts, and eight records with conflicting binary text labels
are excluded. All 34 removals are recorded. Surviving examples retain their
official split. Same-label repeats retain test before validation before train.
These are cleaned official splits, not unchanged official split counts.

## FakeVsSatire source selection, encodings and exclusions

Only local `StoryText 2/Fake/finalFake` (283 files) and
`StoryText 2/Satire/finalSatire` (203 files) are included. Rejected/ditched files
and other archive directories do not contribute. The archive itself has not been
byte-verified against the remote publisher.

Strict decoding selects UTF-8 with optional BOM for 421 files and Windows-1252
for 65 files. No decoder uses `errors=ignore` or replacement fallback. URL strings
are removed before whitespace/Unicode normalization. All texts satisfy the
20-word minimum. Six duplicate records and two conflicting-label records are
excluded, leaving 276 Fake and 202 Satire records. The per-class seed-42 split uses
floor(70%) train, floor(15%) validation and the remainder test.

One source contains an already encoded U+FFFD replacement glyph:
`Satire/finalSatire/517.txt`, in `Donald Trump\ufffds hair`. The raw UTF-8 bytes
contain this glyph once. It is preserved in the processed text, so strict decoding
does not remove it. This is a minor source-text defect, not a new decoding error;
the missing punctuation cannot be recovered reliably from local data alone.

## Interpretation and limitations

Label 0 differs across datasets: factual articles in ISOT, the upper three truth
categories in LIAR-PLUS, and satire in FakeVsSatire. Satire is not factual reporting,
and a half-true claim is not wholly true. This difference is central to the study;
do not present all label-0 examples as equivalent ground-truth truthfulness.

Normalized exact deduplication does not rule out near duplicates, paraphrases,
shared topics or source/style shortcuts. Source names or stylistic cues may remain
after URL and limited dateline removal. The small FakeVsSatire test set has only
73 examples, so its measured estimates can be unstable. These datasets and sample
sizes define a smaller student experiment, not an exact reproduction of the paper.
This report asserts no model performance or improvement.
