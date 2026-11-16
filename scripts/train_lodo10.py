"""Extra 10-dataset DistilBERT in-domain + 9-source LODO. Never writes R1-R10."""
import os
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'
os.environ['HF_HUB_DISABLE_TELEMETRY'] = '1'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
os.environ['WANDB_DISABLED'] = 'true'

import csv
import json
import time
from pathlib import Path

csv.field_size_limit(10_000_000)

import numpy as np
import torch
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from transformers import (AutoModelForSequenceClassification, AutoTokenizer, Trainer,
                          TrainingArguments, set_seed)
from transformers.trainer_utils import get_last_checkpoint

from train_neural import BASE, Progress, TextDataset, metrics, probabilities, save_json

ROOT = Path(__file__).resolve().parents[1]
OUT_ROOT = ROOT / 'results' / 'neural'
CKPT_ROOT = ROOT / 'checkpoints'
LOCKED = {'R1', 'R2', 'R3', 'R4', 'R5', 'R6', 'R7', 'R8', 'R9', 'R10'}
CANDIDATES = [
    'ISOT', 'LIAR_PLUS', 'FakeVsSatire', 'Celebrity', 'Cidii', 'Fakes',
    'Horne', 'Infodemic', 'NDF', 'Politifact',
]


def names():
    found = []
    for name in CANDIDATES:
        path = ROOT / 'data' / 'processed' / f'{name}.csv'
        if path.exists():
            found.append(name)
    if len(found) < 4:
        raise SystemExit(f'Need extra processed datasets, found {found}')
    return found


def load_data(found):
    data = {}
    for name in found:
        with (ROOT / 'data' / 'processed' / f'{name}.csv').open(encoding='utf-8', newline='') as f:
            data[name] = list(csv.DictReader(f))
        for r in data[name]:
            r['label'] = int(r['label'])
    return data


def train_args(path):
    return TrainingArguments(
        output_dir=str(path), num_train_epochs=2, learning_rate=2e-5,
        per_device_train_batch_size=4, per_device_eval_batch_size=16, fp16=True,
        seed=42, data_seed=42, dataloader_num_workers=0, report_to=[], disable_tqdm=True,
        logging_steps=50, save_strategy='epoch', save_total_limit=1, eval_strategy='no',
        optim='adamw_torch', lr_scheduler_type='linear', warmup_ratio=0.0,
        save_safetensors=True, load_best_model_at_end=False)


def evaluate(run, trainer, test, out, extra):
    predicted = trainer.predict(TextDataset(test, extra['tokenizer']))
    prob = probabilities(predicted.predictions)
    y = np.array([r['label'] for r in test])
    pred = prob.argmax(axis=1)
    out.mkdir(parents=True, exist_ok=True)
    with (out / 'predictions.csv').open('w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['id', 'label', 'prediction', 'prob_real', 'prob_fake'])
        writer.writerows((r['id'], int(y[i]), int(pred[i]), float(prob[i, 0]), float(prob[i, 1]))
                         for i, r in enumerate(test))
    result = {
        'run_id': run, 'family': 'ten_dataset_extra',
        'accuracy': float(accuracy_score(y, pred)),
        'macro_f1': float(f1_score(y, pred, average='macro', labels=[0, 1], zero_division=0)),
        'classification_report': classification_report(
            y, pred, labels=[0, 1], target_names=['Real', 'Fake'], output_dict=True, zero_division=0),
        'confusion_matrix': confusion_matrix(y, pred, labels=[0, 1]).tolist(),
        'n_test': len(test),
        **{k: v for k, v in extra.items() if k != 'tokenizer'},
    }
    save_json(out / 'metrics.json', result)
    print(run, 'ACC', result['accuracy'], 'MACRO_F1', result['macro_f1'], flush=True)
    return result


def run_job(run, data, tokenizer, target, sources, phase):
    if run in LOCKED:
        raise ValueError('refusing locked run id')
    out = OUT_ROOT / run
    ckpt = CKPT_ROOT / run
    final = ckpt / 'final'
    if (out / 'metrics.json').exists() and (final / 'config.json').exists():
        print(run, 'already complete', flush=True)
        return json.loads((out / 'metrics.json').read_text(encoding='utf-8'))
    set_seed(42)
    train = [r for name in sources for r in data[name] if r['split'] == 'train']
    val = [r for name in sources for r in data[name] if r['split'] == 'val']
    test = [r for r in data[target] if r['split'] == 'test']
    model = AutoModelForSequenceClassification.from_pretrained(
        BASE, local_files_only=True, num_labels=2,
        id2label={0: 'Real', 1: 'Fake'}, label2id={'Real': 0, 'Fake': 1})
    trainer = Trainer(
        model=model, args=train_args(ckpt), train_dataset=TextDataset(train, tokenizer),
        eval_dataset=TextDataset(val, tokenizer) if val else None, compute_metrics=metrics,
        callbacks=[Progress(run)])
    previous = get_last_checkpoint(str(ckpt)) if ckpt.exists() else None
    print(run, phase, 'train', len(train), 'val', len(val), 'test', len(test),
          'resume', previous, flush=True)
    started = time.perf_counter()
    trainer.train(resume_from_checkpoint=previous)
    seconds = time.perf_counter() - started
    trainer.save_model(str(final))
    tokenizer.save_pretrained(final)
    result = evaluate(run, trainer, test, out, {
        'tokenizer': tokenizer, 'target': target, 'sources': sources, 'phase': phase,
        'method': 'DistilBERT', 'n_train': len(train), 'n_val': len(val),
        'train_seconds': seconds, 'checkpoint': str(final),
    })
    del trainer, model
    torch.cuda.empty_cache()
    return result


def write_table(found, results):
    rows = [['run_id', 'phase', 'target', 'n_sources', 'n_train', 'n_test', 'accuracy', 'macro_f1']]
    for result in results:
        rows.append([
            result.get('run_id', ''), result.get('phase', ''), result.get('target', ''),
            str(len(result.get('sources') or [])), str(result.get('n_train', '')),
            str(result.get('n_test', '')), f"{result.get('accuracy', 0):.6f}",
            f"{result.get('macro_f1', 0):.6f}",
        ])
    text = ['# Table 7 Extra 10-dataset DistilBERT (does not replace Tables 1-4)\n',
            f'Datasets used: {", ".join(found)}\n',
            'Locked R1-R10 unchanged. Cidii included only if processed CSV exists.\n\n',
            '| ' + ' | '.join(rows[0]) + ' |\n',
            '|' + '|'.join(['---'] * len(rows[0])) + '|\n']
    text.extend('| ' + ' | '.join(r) + ' |\n' for r in rows[1:])
    path = ROOT / 'results' / 'analysis' / 'Table_7_TenDatasetExtras.md'
    path.write_text(''.join(text), encoding='utf-8')
    save_json(ROOT / 'results' / 'analysis' / 'Table_7_TenDatasetExtras.json', results)
    print('Wrote', path, flush=True)


def main():
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable')
    torch.set_num_threads(4)
    found = names()
    print('DATASETS', found, flush=True)
    data = load_data(found)
    tokenizer = AutoTokenizer.from_pretrained(BASE, local_files_only=True)
    results = []
    locked_in_domain = {'ISOT', 'LIAR_PLUS', 'FakeVsSatire'}
    jobs = []
    for target in found:
        if target not in locked_in_domain:
            jobs.append((f'X_in_{target}', target, [target], 'in_domain'))
    for target in found:
        jobs.append((f'X_lodo_{target}', target, [n for n in found if n != target], 'lodo_9source'))
    for run, target, sources, phase in jobs:
        results.append(run_job(run, data, tokenizer, target, sources, phase))
        write_table(found, results)


if __name__ == '__main__':
    main()
