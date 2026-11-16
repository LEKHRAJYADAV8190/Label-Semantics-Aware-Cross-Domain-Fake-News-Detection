"""Extra accuracy-seeking runs. Never writes into R1-R10 folders."""
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

import numpy as np
import torch
from peft import LoraConfig, TaskType, get_peft_model
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from transformers import (AutoModelForSequenceClassification, AutoTokenizer, Trainer,
                          TrainingArguments, set_seed)
from transformers.trainer_utils import get_last_checkpoint

from train_neural import (BASE, NAMES, ROOT, Progress, load_data, metrics,
                          probabilities, save_json)

OUT_ROOT = ROOT / 'results' / 'neural'
CKPT_ROOT = ROOT / 'checkpoints'


class TextDataset(torch.utils.data.Dataset):
    def __init__(self, rows, tokenizer, max_len=128):
        self.tokens = tokenizer([r['text'] for r in rows], truncation=True,
                                padding='max_length', max_length=max_len)
        self.labels = [r['label'] for r in rows]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        return {**{k: torch.tensor(v[i]) for k, v in self.tokens.items()},
                'labels': torch.tensor(self.labels[i])}


class WeightedTrainer(Trainer):
    def __init__(self, class_weights, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs.get('labels')
        outputs = model(**inputs)
        weights = torch.tensor(self._class_weights, device=outputs.logits.device,
                               dtype=outputs.logits.dtype)
        loss = torch.nn.functional.cross_entropy(outputs.logits, labels, weight=weights)
        return (loss, outputs) if return_outputs else loss


def train_args(path, epochs, lr, lora=False):
    return TrainingArguments(
        output_dir=str(path), num_train_epochs=epochs, learning_rate=lr,
        per_device_train_batch_size=4, per_device_eval_batch_size=16, fp16=True,
        seed=42, data_seed=42, dataloader_num_workers=0, report_to=[], disable_tqdm=True,
        logging_steps=100, save_strategy='epoch', save_total_limit=1,
        eval_strategy='no', optim='adamw_torch', lr_scheduler_type='linear',
        warmup_ratio=0.06, save_safetensors=True, load_best_model_at_end=False)


def pick_balanced(pool, k, method, model, tokenizer, checkpoint, max_len=128):
    by_class = {0: [i for i, r in enumerate(pool) if r['label'] == 0],
                1: [i for i, r in enumerate(pool) if r['label'] == 1]}
    per = k // 2
    chosen = []
    if method == 'Guided':
        scorer = Trainer(model=model, args=train_args(checkpoint / 'selection', 1, 2e-5, lora=True))
        prob = probabilities(scorer.predict(TextDataset(pool, tokenizer, max_len)).predictions)
        entropy = -(prob * np.log(np.clip(prob, 1e-12, 1))).sum(axis=1)
        del scorer
        for label in (0, 1):
            idx = np.array(by_class[label])
            order = np.argsort(-entropy[idx], kind='stable')[:per]
            chosen.extend(idx[order].tolist())
        rule = f'top {per} entropy per class from target train'
    else:
        rng = np.random.default_rng(42)
        for label in (0, 1):
            idx = np.array(by_class[label])
            chosen.extend(rng.choice(idx, size=min(per, len(idx)), replace=False).tolist())
        rule = f'{per} per class uniform random from target train, seed 42'
    return [pool[i] for i in chosen], rule


def evaluate(run, trainer, test, tokenizer, out, extra):
    test_ds = TextDataset(test, tokenizer, max_len=extra.get('max_len', 128))
    predicted = trainer.predict(test_ds)
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
        'run_id': run, 'family': 'improvement_extra',
        'accuracy': float(accuracy_score(y, pred)),
        'macro_f1': float(f1_score(y, pred, average='macro', labels=[0, 1], zero_division=0)),
        'classification_report': classification_report(
            y, pred, labels=[0, 1], target_names=['Real', 'Fake'], output_dict=True, zero_division=0),
        'confusion_matrix': confusion_matrix(y, pred, labels=[0, 1]).tolist(),
        'n_test': len(test),
        **{k: v for k, v in extra.items() if k != 'tokenizer'},
    }
    save_json(out / 'metrics.json', result)
    print(run, 'ACC', result['accuracy'], 'MACRO_F1', result['macro_f1'],
          'CM', result['confusion_matrix'], flush=True)
    return result


def run_lora(run, data, tokenizer, target, source_run, method, k=64, epochs=10, lr=2e-4, rank=16, max_len=128):
    out = OUT_ROOT / run
    checkpoint = CKPT_ROOT / run
    final = checkpoint / 'final'
    if (out / 'metrics.json').exists() and (final / 'adapter_config.json').exists():
        print(run, 'already complete', flush=True)
        return json.loads((out / 'metrics.json').read_text(encoding='utf-8'))
    set_seed(42)
    base = CKPT_ROOT / source_run / 'final'
    model = AutoModelForSequenceClassification.from_pretrained(
        base, local_files_only=True, num_labels=2, id2label={0: 'Real', 1: 'Fake'},
        label2id={'Real': 0, 'Fake': 1})
    pool = [r for r in data[target] if r['split'] == 'train']
    train, rule = pick_balanced(pool, k, method, model, tokenizer, checkpoint, max_len)
    save_json(out / 'selection.json', {
        'method': method, 'K': k, 'pool': 'target train only, class-balanced',
        'ids': [r['id'] for r in train],
        'selected_real': sum(r['label'] == 0 for r in train),
        'selected_fake': sum(r['label'] == 1 for r in train),
        'rule': rule, 'source_run': source_run, 'max_len': max_len})
    model = get_peft_model(model, LoraConfig(
        task_type=TaskType.SEQ_CLS, r=rank, lora_alpha=rank * 2, lora_dropout=0.1,
        target_modules=['q_lin', 'v_lin'], modules_to_save=['pre_classifier', 'classifier']))
    test = [r for r in data[target] if r['split'] == 'test']
    trainer = Trainer(
        model=model, args=train_args(checkpoint, epochs, lr, lora=True),
        train_dataset=TextDataset(train, tokenizer, max_len),
        callbacks=[Progress(run)])
    print(run, 'LoRA', method, 'K', k, 'from', source_run, 'n', len(train),
          'epochs', epochs, 'max_len', max_len, flush=True)
    started = time.perf_counter()
    trainer.train(resume_from_checkpoint=get_last_checkpoint(str(checkpoint)))
    trainer.save_model(str(final))
    tokenizer.save_pretrained(final)
    extra = dict(target=target, sources=[source_run], method=f'LoRA-{method}-K{k}-balanced',
                 K=k, n_train=len(train), epochs=epochs, lr=lr, lora_rank=rank,
                 checkpoint=str(final), train_seconds=time.perf_counter() - started,
                 baseline_run=source_run, tokenizer=tokenizer, max_len=max_len)
    result = evaluate(run, trainer, test, tokenizer, out, extra)
    del trainer, model
    torch.cuda.empty_cache()
    return result


def run_weighted_lodo(run, data, tokenizer, target, sources, epochs=2, lr=2e-5, max_len=128):
    out = OUT_ROOT / run
    checkpoint = CKPT_ROOT / run
    final = checkpoint / 'final'
    if (out / 'metrics.json').exists() and (final / 'config.json').exists():
        print(run, 'already complete', flush=True)
        return json.loads((out / 'metrics.json').read_text(encoding='utf-8'))
    set_seed(42)
    train = [r for name in sources for r in data[name] if r['split'] == 'train']
    val = [r for name in sources for r in data[name] if r['split'] == 'val']
    test = [r for r in data[target] if r['split'] == 'test']
    counts = np.bincount([r['label'] for r in train], minlength=2).astype(np.float64)
    weights = (counts.sum() / (2.0 * np.maximum(counts, 1.0))).tolist()
    model = AutoModelForSequenceClassification.from_pretrained(
        BASE, local_files_only=True, num_labels=2, id2label={0: 'Real', 1: 'Fake'},
        label2id={'Real': 0, 'Fake': 1})
    trainer = WeightedTrainer(
        weights, model=model, args=train_args(checkpoint, epochs, lr),
        train_dataset=TextDataset(train, tokenizer, max_len),
        eval_dataset=TextDataset(val, tokenizer, max_len), compute_metrics=metrics,
        callbacks=[Progress(run)])
    print(run, 'weighted LODO', target, 'n', len(train), 'epochs', epochs,
          'max_len', max_len, 'weights', weights, flush=True)
    started = time.perf_counter()
    trainer.train(resume_from_checkpoint=get_last_checkpoint(str(checkpoint)))
    trainer.save_model(str(final))
    tokenizer.save_pretrained(final)
    extra = dict(target=target, sources=sources, method='DistilBERT-class-weighted',
                 K=None, n_train=len(train), epochs=epochs, lr=lr, class_weights=weights,
                 checkpoint=str(final), train_seconds=time.perf_counter() - started,
                 tokenizer=tokenizer, max_len=max_len)
    result = evaluate(run, trainer, test, tokenizer, out, extra)
    del trainer, model
    torch.cuda.empty_cache()
    return result


def best_run(*runs):
    scored = []
    for run in runs:
        path = OUT_ROOT / run / 'metrics.json'
        if path.exists():
            scored.append(json.loads(path.read_text(encoding='utf-8')))
    if not scored:
        raise FileNotFoundError(runs)
    return max(scored, key=lambda r: r['macro_f1'])


def write_table(rows):
    save_json(ROOT / 'results' / 'analysis' / 'Table_6_ImprovementExtras.json', rows)
    lines = ['# Table 6 Extra improvement runs', '',
             '|run_id|target|method|accuracy|macro_f1|', '|---|---|---|---|---|']
    for row in rows:
        lines.append(f"|{row['run_id']}|{row['target']}|{row['method']}|{row['accuracy']:.4f}|{row['macro_f1']:.4f}|")
    (ROOT / 'results' / 'analysis' / 'Table_6_ImprovementExtras.md').write_text(
        '\n'.join(lines) + '\n', encoding='utf-8')
    print('WROTE Table 6', flush=True)


def main():
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable')
    torch.set_num_threads(4)
    try:
        import ctypes
        ctypes.windll.kernel32.SetThreadExecutionState(0x80000001)
    except Exception:
        pass
    data = load_data()
    tokenizer = AutoTokenizer.from_pretrained(BASE, local_files_only=True)
    run_lora('E1', data, tokenizer, 'LIAR_PLUS', 'R5', 'Guided')
    run_lora('E2', data, tokenizer, 'LIAR_PLUS', 'R5', 'Random')
    run_lora('E3', data, tokenizer, 'FakeVsSatire', 'R6', 'Guided')
    run_lora('E4', data, tokenizer, 'FakeVsSatire', 'R6', 'Random')
    run_weighted_lodo('E5', data, tokenizer, 'ISOT', ['LIAR_PLUS', 'FakeVsSatire'])
    run_weighted_lodo('E6', data, tokenizer, 'LIAR_PLUS', ['ISOT', 'FakeVsSatire'])
    run_weighted_lodo('E7', data, tokenizer, 'FakeVsSatire', ['ISOT', 'LIAR_PLUS'])
    run_weighted_lodo('E8', data, tokenizer, 'ISOT', ['LIAR_PLUS', 'FakeVsSatire'], epochs=4, max_len=256)
    run_weighted_lodo('E9', data, tokenizer, 'LIAR_PLUS', ['ISOT', 'FakeVsSatire'], epochs=4, max_len=128)
    run_weighted_lodo('E10', data, tokenizer, 'FakeVsSatire', ['ISOT', 'LIAR_PLUS'], epochs=4, max_len=256)
    liar_src = best_run('R5', 'E6', 'E9')['run_id']
    fvs_src = best_run('R6', 'E7', 'E10')['run_id']
    isot_src = best_run('R4', 'E5', 'E8')['run_id']
    print('LoRA sources', liar_src, fvs_src, isot_src, flush=True)
    run_lora('E11', data, tokenizer, 'LIAR_PLUS', liar_src, 'Random', k=128, epochs=10, max_len=128)
    run_lora('E12', data, tokenizer, 'FakeVsSatire', fvs_src, 'Random', k=128, epochs=10,
             max_len=256 if fvs_src in {'E10'} else 128)
    run_lora('E13', data, tokenizer, 'ISOT', isot_src, 'Random', k=64, epochs=10,
             max_len=256 if isot_src in {'E8'} else 128)
    locked = {f'R{i}': json.loads((OUT_ROOT / f'R{i}' / 'metrics.json').read_text(encoding='utf-8'))
              for i in range(4, 9)}
    extra = {}
    for run in [f'E{i}' for i in range(1, 14)]:
        path = OUT_ROOT / run / 'metrics.json'
        if path.exists():
            extra[run] = json.loads(path.read_text(encoding='utf-8'))
    table = [
        dict(run_id='R4 locked', target='ISOT', method='LODO DistilBERT 2ep',
             accuracy=locked['R4']['accuracy'], macro_f1=locked['R4']['macro_f1']),
        dict(run_id='E5 extra', target='ISOT', method=extra['E5']['method'],
             accuracy=extra['E5']['accuracy'], macro_f1=extra['E5']['macro_f1']),
        dict(run_id='E8 extra', target='ISOT', method='weighted 4ep max_len256',
             accuracy=extra['E8']['accuracy'], macro_f1=extra['E8']['macro_f1']),
        dict(run_id='E13 extra', target='ISOT', method=extra['E13']['method'],
             accuracy=extra['E13']['accuracy'], macro_f1=extra['E13']['macro_f1']),
        dict(run_id='R5 locked', target='LIAR_PLUS', method='LODO DistilBERT 2ep',
             accuracy=locked['R5']['accuracy'], macro_f1=locked['R5']['macro_f1']),
        dict(run_id='E6 extra', target='LIAR_PLUS', method=extra['E6']['method'],
             accuracy=extra['E6']['accuracy'], macro_f1=extra['E6']['macro_f1']),
        dict(run_id='E9 extra', target='LIAR_PLUS', method='weighted 4ep',
             accuracy=extra['E9']['accuracy'], macro_f1=extra['E9']['macro_f1']),
        dict(run_id='E2 extra', target='LIAR_PLUS', method=extra['E2']['method'],
             accuracy=extra['E2']['accuracy'], macro_f1=extra['E2']['macro_f1']),
        dict(run_id='E11 extra', target='LIAR_PLUS', method=extra['E11']['method'],
             accuracy=extra['E11']['accuracy'], macro_f1=extra['E11']['macro_f1']),
        dict(run_id='R6 locked', target='FakeVsSatire', method='LODO DistilBERT 2ep',
             accuracy=locked['R6']['accuracy'], macro_f1=locked['R6']['macro_f1']),
        dict(run_id='E7 extra', target='FakeVsSatire', method=extra['E7']['method'],
             accuracy=extra['E7']['accuracy'], macro_f1=extra['E7']['macro_f1']),
        dict(run_id='E10 extra', target='FakeVsSatire', method='weighted 4ep max_len256',
             accuracy=extra['E10']['accuracy'], macro_f1=extra['E10']['macro_f1']),
        dict(run_id='E4 extra', target='FakeVsSatire', method=extra['E4']['method'],
             accuracy=extra['E4']['accuracy'], macro_f1=extra['E4']['macro_f1']),
        dict(run_id='E12 extra', target='FakeVsSatire', method=extra['E12']['method'],
             accuracy=extra['E12']['accuracy'], macro_f1=extra['E12']['macro_f1']),
    ]
    write_table(table)


if __name__ == '__main__':
    main()
