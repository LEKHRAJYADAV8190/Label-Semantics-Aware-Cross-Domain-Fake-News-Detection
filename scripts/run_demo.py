"""Local viva demonstration. No training. Optional live TF-IDF prediction."""
from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ANALYSIS = ROOT / 'results' / 'analysis'


def compatibility(p_real, p_fake):
    probs = [p_real, p_fake]
    entropy = -sum(p * math.log(p) for p in probs if p > 0)
    return min(1.0, max(0.0, 1.0 - entropy / math.log(2)))


def print_table(title, path):
    rows = list(csv.DictReader(path.open(encoding='utf-8')))
    print('\n==', title, '==')
    if not rows:
        print('(empty)')
        return
    cols = list(rows[0])
    def cell(value, width):
        text = str(value).replace('\n', ' ')
        return text if len(text) <= width else text[: max(0, width - 3)] + '...'
    widths = [min(28, max(len(c), *(len(str(r[c])) for r in rows))) for c in cols]
    fmt = '  '.join(f'{{:{w}}}' for w in widths)
    print(fmt.format(*[cell(c, w) for c, w in zip(cols, widths)]))
    for row in rows:
        print(fmt.format(*[cell(row[c], widths[i]) for i, c in enumerate(cols)]))


def show_examples(n=5):
    rows = list(csv.DictReader((ANALYSIS / 'error_abstention_examples.csv').open(encoding='utf-8')))
    print('\n== Diagnostic examples (threshold 0.4) ==')
    for row in rows[:n]:
        text = row['text'].replace('\n', ' ')[:220]
        score = compatibility(float(row['prob_real']), float(row['prob_fake']))
        print(f"\n{row['id']}")
        print(f"  gold={row['label']} pred={row['prediction']} saved_score={float(row['compatibility']):.4f} "
              f"recomputed={score:.4f} abstain={row['abstained']}")
        print(' ', text)


def live_tfidf():
    import joblib
    rows = list(csv.DictReader((ROOT / 'data' / 'processed' / 'ISOT.csv').open(encoding='utf-8')))
    test = [r for r in rows if r['split'] == 'test']
    sample = next(r for r in test if r['id'] == 'ISOT:True.csv:3308')
    model = joblib.load(ROOT / 'results' / 'tfidf' / 'in_domain__ISOT' / 'model.joblib')
    pred = int(model.predict([sample['text']])[0])
    proba = model.predict_proba([sample['text']])[0]
    print('\n== Live TF-IDF prediction (in-domain ISOT model, one test row) ==')
    print('id:', sample['id'])
    print('gold label:', sample['label'], '(0=Real, 1=Fake)')
    print('predicted:', pred, 'proba_real/fake:', [round(float(x), 4) for x in proba])
    print('text:', sample['text'][:240].replace('\n', ' '))


def main():
    parser = argparse.ArgumentParser(description='Offline project demonstration')
    parser.add_argument('--skip-tfidf', action='store_true')
    args = parser.parse_args()
    print('Label-Semantics-Aware Cross-Domain Fake News Detection')
    print('Project:', ROOT)
    print('This demo loads saved tables and one TF-IDF model. It does not train.')
    print('Neural checkpoints are not loaded here; use results/neural/*/predictions.csv.')
    print_table('Table 1 in-domain', ANALYSIS / 'Table_1_InDomain.csv')
    print_table('Table 2 LODO', ANALYSIS / 'Table_2_LODO.csv')
    print_table('Table 3 abstention', ANALYSIS / 'Table_3_Abstention.csv')
    print_table('Table 4 LoRA', ANALYSIS / 'Table_4_LoRA.csv')
    show_examples()
    if not args.skip_tfidf:
        live_tfidf()
    print('\nDemo finished. Teacher files: report/Project_Report_30marks.docx, '
          'slides/Project_Slides_8.pptx, paper/Research_Paper_Draft.docx')


if __name__ == '__main__':
    main()
