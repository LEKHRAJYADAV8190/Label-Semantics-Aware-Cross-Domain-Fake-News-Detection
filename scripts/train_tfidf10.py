"""TF-IDF extras for all processed datasets. Writes results/tfidf10, not locked tfidf."""
import csv
import json
import time
import warnings
from collections import Counter
from pathlib import Path

import joblib
import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.pipeline import Pipeline

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / 'results' / 'tfidf10'
CANDIDATES = [
    'ISOT', 'LIAR_PLUS', 'FakeVsSatire', 'Celebrity', 'Cidii', 'Fakes',
    'Horne', 'Infodemic', 'NDF', 'Politifact',
]
csv.field_size_limit(10_000_000)


def names():
    return [n for n in CANDIDATES if (ROOT / 'data' / 'processed' / f'{n}.csv').exists()]


def load_data(found):
    data = {}
    for name in found:
        with (ROOT / 'data' / 'processed' / f'{name}.csv').open(encoding='utf-8', newline='') as f:
            data[name] = list(csv.DictReader(f))
        for r in data[name]:
            r['label'] = int(r['label'])
    return data


def measures(labels, predictions):
    return dict(
        n=len(labels),
        accuracy=float(accuracy_score(labels, predictions)),
        macro_f1=float(f1_score(labels, predictions, average='macro', zero_division=0)),
        confusion_matrix=confusion_matrix(labels, predictions, labels=[0, 1]).tolist(),
        classification_report=classification_report(
            labels, predictions, labels=[0, 1], output_dict=True, zero_division=0),
    )


def main():
    found = names()
    datasets = load_data(found)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    summary = []
    jobs = [('in_domain', t, [t]) for t in found]
    jobs += [('lodo', t, [n for n in found if n != t]) for t in found]
    for protocol, target, sources in jobs:
        run_id = f'{protocol}__{target}'
        folder = OUTPUT / run_id
        folder.mkdir(parents=True, exist_ok=True)
        metrics_path = folder / 'metrics.json'
        if metrics_path.exists():
            summary.append(json.loads(metrics_path.read_text(encoding='utf-8')))
            print('SKIP', run_id, flush=True)
            continue
        train = [r for s in sources for r in datasets[s] if r['split'] == 'train']
        test = [r for r in datasets[target] if r['split'] == 'test']
        model = Pipeline([
            ('tfidf', TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=50000,
                                      sublinear_tf=True, dtype=np.float32)),
            ('classifier', LogisticRegression(C=1.0, solver='liblinear', max_iter=1000,
                                               random_state=42)),
        ])
        started = time.perf_counter()
        print(f'START {run_id} train={len(train)} test={len(test)}', flush=True)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', ConvergenceWarning)
            model.fit([r['text'] for r in train], [r['label'] for r in train])
        predicted = model.predict([r['text'] for r in test])
        labels = [r['label'] for r in test]
        joblib.dump(model, folder / 'model.joblib', compress=3)
        test_metrics = measures(labels, predicted)
        result = dict(
            run_id=run_id, protocol=protocol, target=target, sources=sources,
            n_train=len(train), n_test=len(test), family='ten_dataset_extra',
            fit_seconds=time.perf_counter() - started, metrics={'test': test_metrics},
            train_dataset_counts=dict(Counter(r['dataset'] for r in train)),
        )
        (metrics_path).write_text(json.dumps(result, indent=2), encoding='utf-8')
        summary.append(result)
        print(f'DONE {run_id} F1={test_metrics["macro_f1"]:.6f}', flush=True)
    (OUTPUT / 'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    rows = ['# Table 7b Extra TF-IDF 10-dataset\n\n',
            '|run_id|protocol|target|n_train|n_test|accuracy|macro_f1|\n',
            '|---|---|---|---|---|---|---|\n']
    for r in summary:
        m = r['metrics']['test']
        rows.append(f"|{r['run_id']}|{r['protocol']}|{r['target']}|{r['n_train']}|{r['n_test']}|"
                    f"{m['accuracy']:.6f}|{m['macro_f1']:.6f}|\n")
    (ROOT / 'results' / 'analysis' / 'Table_7b_TFIDF_TenDataset.md').write_text(
        ''.join(rows), encoding='utf-8')
    print('COMPLETE TF-IDF extras', len(summary), flush=True)


if __name__ == '__main__':
    main()
