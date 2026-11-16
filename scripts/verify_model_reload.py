"""Independently reload 16 local models and compare eight test predictions each.

No training, downloading, or original-artifact modification is performed.
"""
import os
for name in ('HF_HUB_OFFLINE', 'TRANSFORMERS_OFFLINE', 'HF_HUB_DISABLE_TELEMETRY'):
    os.environ[name] = '1'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
import csv
import gc
import hashlib
import json
import socket
import time
from datetime import datetime, timezone
from pathlib import Path
import joblib
import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from peft import PeftModel

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'results' / 'analysis' / 'model_reload_verification.json'
ADAPTER_BASE = {'R7': 'R5', 'R8': 'R6', 'R9': 'R5', 'R10': 'R6'}
NETWORK_ATTEMPTS = []


def denied(*args, **kwargs):
    NETWORK_ATTEMPTS.append('blocked socket operation')
    raise RuntimeError('Network disabled during independent model reload verification')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_csv(path):
    with path.open(encoding='utf-8', newline='') as stream:
        return list(csv.DictReader(stream))


def sample(target, saved_path):
    test = {r['id']: r for r in read_csv(ROOT / 'data' / 'processed' / f'{target}.csv')
            if r['split'] == 'test'}
    saved = {r['id']: r for r in read_csv(saved_path)}
    if set(test) != set(saved):
        raise ValueError('Prediction IDs do not exactly match target test split')
    # Sorting by ID and choosing evenly spaced positions avoids class-blocked
    # first-eight selections; sample selection never consults predictions/labels.
    ordered = sorted(test)
    indices = np.linspace(0, len(ordered)-1, min(8, len(ordered)), dtype=int)
    ids = [ordered[int(i)] for i in indices]
    for rid in ids:
        if int(test[rid]['label']) != int(saved[rid]['label']):
            raise ValueError(f'Saved/processed label mismatch: {rid}')
    return [test[rid] for rid in ids], [saved[rid] for rid in ids]


def compare(run, target, rows, saved, probabilities, columns, tolerance, artifacts):
    expected = np.array([[float(r[c]) for c in columns] for r in saved])
    predicted = probabilities.argmax(axis=1)
    saved_predicted = np.array([int(r['prediction']) for r in saved])
    errors = np.abs(probabilities-expected)
    labels_match = bool(np.array_equal(predicted, saved_predicted))
    probability_match = bool(np.isfinite(probabilities).all() and np.all(errors <= tolerance))
    return {'run_id': run, 'target': target, 'n_test_samples': len(rows),
            'sample_ids': [r['id'] for r in rows], 'prediction_labels_match': labels_match,
            'probabilities_within_tolerance': probability_match,
            'max_absolute_probability_difference': float(errors.max()),
            'absolute_probability_tolerance': tolerance,
            'reloaded_predictions': predicted.tolist(), 'saved_predictions': saved_predicted.tolist(),
            'reloaded_probabilities': probabilities.tolist(), 'saved_probabilities': expected.tolist(),
            'artifact_sha256': {str(path.relative_to(ROOT)): sha(path) for path in artifacts},
            'status': 'passed' if labels_match and probability_match else 'failed'}


def main():
    started = time.perf_counter()
    # Python socket connections/DNS fail even if a dependency ignores offline flags.
    # Imports precede the guard; no remote model/data operations occur at import.
    socket.socket.connect = denied
    socket.socket.connect_ex = denied
    socket.create_connection = denied
    socket.getaddrinfo = denied
    csv.field_size_limit(10_000_000)
    torch.set_num_threads(4)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    results = []
    for folder in sorted((ROOT/'results'/'tfidf').iterdir()):
        if not folder.is_dir():
            continue
        metadata = json.loads((folder/'metrics.json').read_text(encoding='utf-8'))
        predictions = folder/'test_predictions.csv'
        rows, saved = sample(metadata['target'], predictions)
        model_path = folder/'model.joblib'
        model = joblib.load(model_path)
        probability = model.predict_proba([r['text'] for r in rows])
        result = compare(folder.name, metadata['target'], rows, saved, probability,
                         ['probability_0', 'probability_1'], 1e-10, [model_path, predictions])
        result['inference_precision'] = 'saved scikit-learn model native precision'
        results.append(result)
        print(result['run_id'], result['status'], result['max_absolute_probability_difference'], flush=True)
        del model
    for number in range(1, 11):
        run = f'R{number}'
        folder = ROOT/'results'/'neural'/run
        final = ROOT/'checkpoints'/run/'final'
        metadata = json.loads((folder/'metrics.json').read_text(encoding='utf-8'))
        predictions = folder/'predictions.csv'
        rows, saved = sample(metadata['target'], predictions)
        tokenizer = AutoTokenizer.from_pretrained(final, local_files_only=True)
        artifacts = [predictions]
        if run in ADAPTER_BASE:
            base = ROOT/'checkpoints'/ADAPTER_BASE[run]/'final'
            model = AutoModelForSequenceClassification.from_pretrained(base, local_files_only=True)
            model = PeftModel.from_pretrained(model, final, local_files_only=True, is_trainable=False)
            artifacts.extend([base/'model.safetensors', final/'adapter_model.safetensors', final/'adapter_config.json'])
        else:
            model = AutoModelForSequenceClassification.from_pretrained(final, local_files_only=True)
            artifacts.extend([final/'model.safetensors', final/'config.json'])
        model = model.to(device=device, dtype=torch.float32).eval()
        inputs = tokenizer([r['text'] for r in rows], truncation=True, padding='max_length',
                           max_length=128, return_tensors='pt').to(device)
        with torch.inference_mode():
            probability = torch.softmax(model(**inputs).logits.float(), dim=-1).cpu().numpy()
        result = compare(run, metadata['target'], rows, saved, probability,
                         ['prob_real', 'prob_fake'], 0.005, artifacts)
        result['inference_precision'] = 'FP32; saved predictions used Trainer mixed FP16 evaluation'
        result['adapter_base_run'] = ADAPTER_BASE.get(run)
        results.append(result)
        print(result['run_id'], result['status'], result['max_absolute_probability_difference'], flush=True)
        del model, inputs, tokenizer
        gc.collect()
        if device.type == 'cuda':
            torch.cuda.empty_cache()
    passed = len(results) == 16 and all(r['status'] == 'passed' for r in results) and not NETWORK_ATTEMPTS
    report = {'status': 'passed' if passed else 'failed',
              'completed_utc': datetime.now(timezone.utc).isoformat(),
              'elapsed_seconds': time.perf_counter()-started, 'model_count': len(results),
              'device': str(device), 'torch_version': torch.__version__,
              'offline_environment': {k: os.environ[k] for k in ('HF_HUB_OFFLINE', 'TRANSFORMERS_OFFLINE')},
              'network_guard': 'Python socket connect/connect_ex/create_connection/getaddrinfo blocked after library imports',
              'blocked_network_attempts': len(NETWORK_ATTEMPTS),
              'sample_rule': 'Up to eight evenly spaced positions in lexicographically sorted target TEST IDs; no labels/predictions used for selection.',
              'caveats': ['Sampled reload check, not full-test score recomputation or a training-resume test.',
                          'Neural probability tolerance is absolute 0.005 for FP16 saved versus FP32 reload; labels must match exactly.',
                          'Socket guard covers Python networking in this verification process, not a system-wide firewall.'],
              'runs': results}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    temporary = OUT.with_suffix('.json.tmp')
    temporary.write_text(json.dumps(report, indent=2), encoding='utf-8')
    temporary.replace(OUT)
    print(f'{report["status"]}: {len(results)} models; {OUT}', flush=True)
    if not passed:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
