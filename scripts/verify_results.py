"""Independently verify saved results without training or modifying original artifacts.

Uses direct confusion-matrix arithmetic rather than training-script metric helpers.
Run: .venv/Scripts/python.exe scripts/verify_results.py
"""
import csv
import hashlib
import html
import json
import math
import re
import unicodedata
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NAMES = ['ISOT', 'LIAR_PLUS', 'FakeVsSatire']
csv.field_size_limit(10_000_000)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_csv(path):
    with path.open(encoding='utf-8-sig', newline='') as stream:
        return list(csv.DictReader(stream))


def read_json(path):
    return json.loads(path.read_text(encoding='utf-8'))


def require(condition, message):
    if not condition:
        raise ValueError(message)


def close(actual, expected, context):
    require(math.isfinite(float(actual)) and math.isclose(float(actual), float(expected),
            rel_tol=1e-10, abs_tol=1e-12), f'{context}: {actual} != {expected}')


def compare_tree(actual, expected, context):
    if isinstance(expected, dict):
        require(set(actual) == set(expected), f'{context}: keys differ')
        for key, value in expected.items():
            compare_tree(actual[key], value, f'{context}/{key}')
    elif isinstance(expected, list):
        require(actual == expected, f'{context}: lists differ')
    else:
        close(actual, expected, context)


def recompute(rows, class_names):
    matrix = [[0, 0], [0, 0]]
    for row in rows:
        matrix[int(row['label'])][int(row['prediction'])] += 1
    n = len(rows)
    accuracy = (matrix[0][0] + matrix[1][1]) / n
    report = {}
    for label, name in enumerate(class_names):
        tp = matrix[label][label]
        support = sum(matrix[label])
        predicted = sum(line[label] for line in matrix)
        precision = tp / predicted if predicted else 0.0
        recall = tp / support if support else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        report[name] = dict(precision=precision, recall=recall, **{'f1-score': f1}, support=support)
    report['accuracy'] = accuracy
    for aggregate in ['macro avg', 'weighted avg']:
        report[aggregate] = {
            measure: sum(report[name][measure] * (0.5 if aggregate == 'macro avg' else
                         report[name]['support'] / n) for name in class_names)
            for measure in ['precision', 'recall', 'f1-score']}
        report[aggregate]['support'] = n
    return dict(accuracy=accuracy, macro_f1=report['macro avg']['f1-score'],
                confusion_matrix=matrix, classification_report=report)


def verify_predictions(path, expected, saved, neural):
    rows = read_csv(path)
    ids = [row['id'] for row in rows]
    require(len(ids) == len(set(ids)), f'{path}: repeated prediction IDs')
    require(set(ids) == set(expected), f'{path}: missing/extra evaluation IDs')
    for row in rows:
        truth = expected[row['id']]
        require(row['label'] == truth['label'], f'{path}: wrong label for {row["id"]}')
        require(row['prediction'] in ('0', '1'), f'{path}: invalid prediction')
        if not neural:
            require(row['dataset'] == truth['dataset'] and row['split'] == truth['split'],
                    f'{path}: wrong dataset/split')
        fields = ['prob_real', 'prob_fake'] if neural else ['probability_0', 'probability_1']
        p = [float(row[field]) for field in fields]
        require(all(math.isfinite(v) and 0 <= v <= 1 for v in p), f'{path}: invalid probabilities')
        require(abs(sum(p) - 1) <= 2e-6, f'{path}: probabilities do not sum to one')
        require(int(row['prediction']) == int(p[1] > p[0]), f'{path}: prediction/argmax mismatch')
    metrics = recompute(rows, ['Real', 'Fake'] if neural else ['0', '1'])
    for key, value in metrics.items():
        compare_tree(saved[key], value, f'{path.parent.name}/{key}')
    count_key = 'n_test' if neural else 'n'
    require(saved[count_key] == len(rows), f'{path}: wrong evaluation count')
    return dict(n=len(rows), accuracy=metrics['accuracy'], macro_f1=metrics['macro_f1'],
                confusion_matrix=metrics['confusion_matrix'], predictions_sha256=sha(path))


def main():
    output = ROOT / 'results/analysis/independent_verification.json'
    originals = [p for base in ['results/tfidf', 'results/neural', 'data/processed']
                 for p in (ROOT / base).rglob('*') if p.is_file()]
    before = {str(p.relative_to(ROOT)): sha(p) for p in originals}
    data = {name: read_csv(ROOT / 'data/processed' / f'{name}.csv') for name in NAMES}
    hashes = {name: sha(ROOT / 'data/processed' / f'{name}.csv') for name in NAMES}
    report = dict(checked_utc=datetime.now(timezone.utc).isoformat(), status='PASS',
                  processed_sha256=hashes, runs=[], failures=[],
                  limitations=['Verifies saved metrics and split membership, not fresh model inference.',
                               'Training counts are verified against manifests and source data; actual optimizer history is not replayed.',
                               'Guided selection IDs are checked for pool membership; entropy ranking is not recomputed.',
                               'Near-duplicate and semantic leakage are not ruled out.'])
    all_ids, all_texts = set(), set()
    for name, rows in data.items():
        for row in rows:
            key = re.sub(r'\s+', ' ', unicodedata.normalize('NFKC', html.unescape(row['text']))).strip().casefold()
            require(row['id'] not in all_ids and key not in all_texts, 'Global duplicate ID/text')
            all_ids.add(row['id']); all_texts.add(key)
    first = read_json(ROOT / 'docs/step1_audit.json')
    fvs = read_json(ROOT / 'docs/fvs_audit.json')
    require(all(first['datasets'][name]['output_sha256'] == hashes[name] for name in NAMES[:2]),
            'Step 1 audit hash mismatch')
    require(fvs['output_sha256'] == hashes['FakeVsSatire'], 'FVS audit hash mismatch')
    for filename, digest in {**first['sources'], **fvs['raw_sha256']}.items():
        require(sha(Path(filename)) == digest, f'Raw source hash mismatch: {filename}')
    report['raw_hashes_verified'] = len(first['sources']) + len(fvs['raw_sha256'])
    plans = []
    for protocol in ['in_domain', 'lodo']:
        for target in NAMES:
            sources = [target] if protocol == 'in_domain' else [n for n in NAMES if n != target]
            plans.append(('tfidf', f'{protocol}__{target}', target, sources, None))
    for i, target in enumerate(NAMES * 2, 1):
        plans.append(('neural', f'R{i}', target, [target] if i <= 3 else [n for n in NAMES if n != target], None))
    for i, target, method in [(7, 'LIAR_PLUS', 'Guided'), (8, 'FakeVsSatire', 'Guided'),
                               (9, 'LIAR_PLUS', 'Random'), (10, 'FakeVsSatire', 'Random')]:
        plans.append(('neural', f'R{i}', target, [n for n in NAMES if n != target], method))
    for family, run, target, sources, adaptation in plans:
        item = dict(family=family, run_id=run, status='PASS')
        try:
            folder = ROOT / 'results' / family / run
            saved = read_json(folder / 'metrics.json')
            require(saved['run_id'] == run and saved['target'] == target and saved['sources'] == sources,
                    f'{run}: unexpected run configuration')
            signature = saved['signature']
            require(signature['data_sha256' if family == 'neural' else 'processed_sha256'] == hashes,
                    f'{run}: processed hash mismatch')
            train = [r for n in sources for r in data[n] if r['split'] == 'train']
            if adaptation:
                selection = read_json(folder / 'selection.json')
                ids = selection['ids']
                pool = [r for r in data[target] if r['split'] == 'train']
                lookup = {r['id']: r for r in pool}
                require(selection['K'] == saved['K'] == 16 and len(ids) == len(set(ids)) == 16,
                        f'{run}: adaptation must have 16 unique IDs')
                require(set(ids) <= set(lookup), f'{run}: adaptation ID outside target train')
                require(selection['method'] == adaptation and selection['pool'] == 'target train only',
                        f'{run}: selection metadata mismatch')
                require(signature['adaptation'] == ['R5' if target == 'LIAR_PLUS' else 'R6', adaptation],
                        f'{run}: wrong adaptation parent')
                if adaptation == 'Random':
                    import numpy as np
                    expected_ids = [pool[int(i)]['id'] for i in np.random.default_rng(42).choice(len(pool), 16, replace=False)]
                    require(ids == expected_ids, f'{run}: random selection differs from seed-42 rule')
                train = [lookup[ident] for ident in ids]
                item['selection'] = dict(unique_target_train_ids=16, method=adaptation,
                                         label_counts=dict(Counter(r['label'] for r in train)))
            require(saved['n_train'] == len(train), f'{run}: training count mismatch')
            item['n_train_verified'] = len(train)
            if family == 'tfidf':
                require(saved['train_label_counts'] == dict(Counter(r['label'] for r in train)), f'{run}: train label counts')
                require(saved['train_dataset_counts'] == dict(Counter(r['dataset'] for r in train)), f'{run}: train dataset counts')
                require(signature['script_sha256'] == sha(ROOT / 'scripts/train_tfidf.py'), f'{run}: training script hash')
                for filename, digest in saved['artifact_sha256'].items():
                    require(sha(folder / filename) == digest, f'{run}: artifact hash: {filename}')
                val = {r['id']: r for n in sources for r in data[n] if r['split'] == 'val'}
                item['validation'] = verify_predictions(folder / 'validation_predictions.csv', val,
                                                         saved['metrics']['validation'], False)
            else:
                require(read_json(ROOT / 'checkpoints' / run / 'experiment.json') == signature,
                        f'{run}: checkpoint manifest mismatch')
            test = {r['id']: r for r in data[target] if r['split'] == 'test'}
            item['test'] = verify_predictions(folder / ('predictions.csv' if family == 'neural' else 'test_predictions.csv'),
                                             test, saved if family == 'neural' else saved['metrics']['test'], family == 'neural')
        except Exception as exc:
            item.update(status='FAIL', error=f'{type(exc).__name__}: {exc}')
            report['failures'].append(dict(run_id=run, error=item['error']))
        report['runs'].append(item)
    report['original_artifacts_unchanged'] = all(sha(ROOT / p) == digest for p, digest in before.items())
    if not report['original_artifacts_unchanged']:
        report['failures'].append(dict(error='An original artifact changed during verification'))
    report['test_runs_passed'] = sum(r['status'] == 'PASS' for r in report['runs'])
    report['test_runs_expected'] = 16
    if report['failures']:
        report['status'] = 'FAIL'
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({key: report[key] for key in ['status', 'test_runs_passed', 'test_runs_expected',
                                                 'original_artifacts_unchanged', 'failures']}, indent=2))
    return 0 if report['status'] == 'PASS' else 1


if __name__ == '__main__':
    raise SystemExit(main())
