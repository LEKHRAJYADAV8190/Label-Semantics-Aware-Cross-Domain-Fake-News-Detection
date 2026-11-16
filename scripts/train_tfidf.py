"""Six reproducible, offline TF-IDF/LogisticRegression baseline experiments.

Run from any directory: .venv/Scripts/python.exe scripts/train_tfidf.py
Each completed run is saved independently; restarting skips verified matching
artifacts. Use --force to retrain. No target validation is used in LODO.
"""
import os

# Limit CPU pressure while transformer setup/training runs alongside this job.
for variable in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ.setdefault(variable, '2')

import argparse
import csv
import hashlib
import json
import platform
import time
import warnings
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import sklearn
from sklearn.exceptions import ConvergenceWarning
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.pipeline import Pipeline

from process_step1 import key

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / 'results' / 'tfidf'
NAMES = ('ISOT', 'LIAR_PLUS', 'FakeVsSatire')
PARAMS = dict(seed=42, ngram_range=[1, 2], min_df=2, max_features=50000,
              sublinear_tf=True, lowercase=True, C=1.0, solver='liblinear',
              max_iter=1000, class_weight=None)
csv.field_size_limit(10_000_000)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path, value):
    temporary = path.with_suffix('.json.tmp')
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding='utf-8')
    temporary.replace(path)


def load_data():
    datasets, hashes, all_ids, all_texts = {}, {}, set(), set()
    for name in NAMES:
        path = ROOT / 'data' / 'processed' / f'{name}.csv'
        with path.open(encoding='utf-8', newline='') as stream:
            reader = csv.DictReader(stream)
            if reader.fieldnames != ['id', 'text', 'label', 'dataset', 'split']:
                raise ValueError(f'Unexpected columns in {path}')
            rows = list(reader)
        if not rows:
            raise ValueError(f'Empty dataset: {name}')
        for row in rows:
            row['label'] = int(row['label'])
            if (row['label'] not in (0, 1) or row['dataset'] != name or
                    row['split'] not in ('train', 'val', 'test') or not row['text'].strip()):
                raise ValueError(f'Invalid row: {row["id"]}')
            normalized = key(row['text'])
            if row['id'] in all_ids or normalized in all_texts:
                raise ValueError(f'Duplicate ID/text or cross-split/dataset leakage: {row["id"]}')
            all_ids.add(row['id'])
            all_texts.add(normalized)
        for split in ('train', 'val', 'test'):
            if {r['label'] for r in rows if r['split'] == split} != {0, 1}:
                raise ValueError(f'Missing label in {name}/{split}')
        datasets[name] = rows
        hashes[name] = digest(path)
    return datasets, hashes


def measures(labels, predictions):
    return dict(n=len(labels), accuracy=float(accuracy_score(labels, predictions)),
                macro_f1=float(f1_score(labels, predictions, average='macro', zero_division=0)),
                confusion_matrix=confusion_matrix(labels, predictions, labels=[0, 1]).tolist(),
                confusion_matrix_labels=[0, 1],
                classification_report=classification_report(
                    labels, predictions, labels=[0, 1], output_dict=True, zero_division=0))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--force', action='store_true')
    args = parser.parse_args()
    datasets, hashes = load_data()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    signature = dict(parameters=PARAMS, processed_sha256=hashes,
                     script_sha256=digest(Path(__file__)), sklearn_version=sklearn.__version__)
    summary = []
    for protocol in ('in_domain', 'lodo'):
        for target in NAMES:
            run_id = f'{protocol}__{target}'
            folder = OUTPUT / run_id
            folder.mkdir(parents=True, exist_ok=True)
            metrics_path = folder / 'metrics.json'
            if metrics_path.exists() and not args.force:
                previous = json.loads(metrics_path.read_text(encoding='utf-8'))
                if previous['signature'] == signature and all(
                        (folder / file).exists() and digest(folder / file) == checksum
                        for file, checksum in previous.get('artifact_sha256', {}).items()
                ) and previous.get('artifact_sha256'):
                    summary.append(previous)
                    print(f'SKIP verified completed run: {run_id}', flush=True)
                    continue
            sources = [target] if protocol == 'in_domain' else [n for n in NAMES if n != target]
            train = [r for source in sources for r in datasets[source] if r['split'] == 'train']
            val = [r for source in sources for r in datasets[source] if r['split'] == 'val']
            test = [r for r in datasets[target] if r['split'] == 'test']
            model = Pipeline([
                ('tfidf', TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=50000,
                                        sublinear_tf=True, dtype=np.float32)),
                ('classifier', LogisticRegression(C=1.0, solver='liblinear', max_iter=1000,
                                                   random_state=42)),
            ])
            started = time.perf_counter()
            print(f'START {run_id}: train={len(train)} val={len(val)} test={len(test)}', flush=True)
            with warnings.catch_warnings():
                warnings.simplefilter('error', ConvergenceWarning)
                model.fit([r['text'] for r in train], [r['label'] for r in train])
            fit_seconds = time.perf_counter() - started
            artifact = folder / 'model.joblib'
            temporary = folder / 'model.joblib.tmp'
            joblib.dump(model, temporary, compress=3)
            temporary.replace(artifact)
            restored = joblib.load(artifact)
            outputs = {}
            for split, rows in [('validation', val), ('test', test)]:
                texts = [r['text'] for r in rows]
                labels = [r['label'] for r in rows]
                predicted = model.predict(texts)
                probabilities = model.predict_proba(texts)
                if not np.array_equal(predicted, restored.predict(texts)):
                    raise ValueError('Saved model predictions differ after reload')
                outputs[split] = measures(labels, predicted)
                with (folder / f'{split}_predictions.csv').open('w', encoding='utf-8', newline='') as stream:
                    writer = csv.writer(stream)
                    writer.writerow(['id', 'dataset', 'split', 'label', 'prediction', 'probability_0', 'probability_1'])
                    for row, prediction, probability in zip(rows, predicted, probabilities):
                        writer.writerow([row['id'], row['dataset'], row['split'], row['label'],
                                         int(prediction), float(probability[0]), float(probability[1])])
            result = dict(run_id=run_id, model='TF-IDF + LogisticRegression', protocol=protocol,
                          target=target, sources=sources, signature=signature,
                          completed_utc=datetime.now(timezone.utc).isoformat(),
                          python_version=platform.python_version(), n_train=len(train),
                          train_label_counts=dict(Counter(str(r['label']) for r in train)),
                          train_dataset_counts=dict(Counter(r['dataset'] for r in train)),
                          vocabulary_size=len(model.named_steps['tfidf'].vocabulary_),
                          fit_seconds=fit_seconds, elapsed_seconds=time.perf_counter()-started,
                          model_path=str(artifact), metrics=outputs, persistence_prediction_check='passed',
                          validation_role='Report only; fixed hyperparameters, no selection or tuning.',
                          label_semantics={'ISOT': {'0': 'Real', '1': 'Fake'},
                                           'LIAR_PLUS': {'0': 'half-true/mostly-true/true', '1': 'pants-fire/false/barely-true'},
                                           'FakeVsSatire': {'0': 'Satire', '1': 'Fake'}},
                          leakage_checks='All processed IDs and normalized exact texts unique globally; vectorizer fit only on train. Near duplicates not ruled out.',
                          artifact_sha256={file: digest(folder/file) for file in (
                              'model.joblib', 'validation_predictions.csv', 'test_predictions.csv')})
            write_json(metrics_path, result)
            summary.append(result)
            print(f'DONE {run_id}: test macro-F1={outputs["test"]["macro_f1"]:.6f}, '
                  f'accuracy={outputs["test"]["accuracy"]:.6f}, fit={fit_seconds:.1f}s', flush=True)
    write_json(OUTPUT / 'summary.json', summary)
    with (OUTPUT / 'summary.csv').open('w', encoding='utf-8', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(['run_id', 'protocol', 'target', 'n_train', 'n_validation', 'n_test',
                         'validation_macro_f1', 'test_macro_f1', 'test_accuracy', 'fit_seconds', 'model_path'])
        for result in summary:
            writer.writerow([result['run_id'], result['protocol'], result['target'], result['n_train'],
                             result['metrics']['validation']['n'], result['metrics']['test']['n'],
                             result['metrics']['validation']['macro_f1'], result['metrics']['test']['macro_f1'],
                             result['metrics']['test']['accuracy'], result['fit_seconds'], result['model_path']])
    print(f'COMPLETE: {len(summary)} runs; saved {OUTPUT / "summary.csv"}', flush=True)


if __name__ == '__main__':
    main()
