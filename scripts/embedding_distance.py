"""Add embedding-distance compatibility on frozen LODO models. No training.

Defined before looking at answered F1:
  uncertainty = 1 - H(p)/ln(2)
  cls = L2-normalized DistilBERT [CLS] from the same LODO checkpoint
  centroid[y] = mean CLS of source TRAIN rows with label y
  label_compat = (1 + cosine(cls(x), centroid[predicted_y])) / 2
  nn_compat = (1 + max cosine to any source TRAIN CLS) / 2
  combined = min(uncertainty, label_compat)
Answer if score >= t for t in {0.3, 0.4, 0.5}. Thresholds are not fit on test.
Saved R4/R5/R6 predicted labels are reused; this script does not retrain.
"""
import os
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'
os.environ['HF_HUB_DISABLE_TELEMETRY'] = '1'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
os.environ['WANDB_DISABLED'] = 'true'

import csv
import json
import math
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import f1_score
from transformers import AutoModelForSequenceClassification, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'results' / 'analysis'
EMB = OUT / 'embeddings'
NAMES = ['ISOT', 'LIAR_PLUS', 'FakeVsSatire']
RUNS = {
    'R4': ('ISOT', ['LIAR_PLUS', 'FakeVsSatire']),
    'R5': ('LIAR_PLUS', ['ISOT', 'FakeVsSatire']),
    'R6': ('FakeVsSatire', ['ISOT', 'LIAR_PLUS']),
}
THRESHOLDS = (0.3, 0.4, 0.5)
BATCH = 32
MAX_LEN = 128
csv.field_size_limit(10_000_000)


def read_csv(path):
    with path.open(encoding='utf-8', newline='') as f:
        return list(csv.DictReader(f))


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def markdown(title, rows):
    cols = list(rows[0])

    def fmt(v):
        return 'N/A' if v is None else f'{v:.4f}' if isinstance(v, float) else str(v)

    return '# ' + title + '\n\n|' + '|'.join(cols) + '|\n|' + '|'.join(['---'] * len(cols)) + '|\n' + ''.join(
        '|' + '|'.join(fmt(r[c]) for c in cols) + '|\n' for r in rows)


def uncertainty(p_real, p_fake):
    score = 1 + sum(v * math.log(v) if v else 0 for v in (p_real, p_fake)) / math.log(2)
    return min(1.0, max(0.0, score))


def load_data():
    data = {}
    for name in NAMES:
        rows = read_csv(ROOT / 'data' / 'processed' / f'{name}.csv')
        for r in rows:
            r['label'] = int(r['label'])
        data[name] = rows
    return data


@torch.no_grad()
def encode(texts, tokenizer, model, device):
    model.eval()
    hidden_chunks, logit_chunks = [], []
    for i in range(0, len(texts), BATCH):
        batch = tokenizer(texts[i:i + BATCH], truncation=True, padding='max_length',
                          max_length=MAX_LEN, return_tensors='pt')
        batch = {k: v.to(device) for k, v in batch.items()}
        out = model(**batch, output_hidden_states=True)
        hidden = torch.nn.functional.normalize(out.hidden_states[-1][:, 0].float(), p=2, dim=1)
        hidden_chunks.append(hidden.cpu().numpy())
        logit_chunks.append(out.logits.float().cpu().numpy())
        del batch, out, hidden
    return np.concatenate(hidden_chunks, axis=0), np.concatenate(logit_chunks, axis=0)


def summarize(rows, score_key, threshold, forced_risk):
    answered = [r for r in rows if r[score_key] >= threshold]
    if not answered:
        return dict(n_answered=0, n_abstained=len(rows), answered_real=0, answered_fake=0,
                    coverage=0.0, risk=None, answered_macro_f1=None, answered_accuracy=None,
                    forced_risk=forced_risk)
    y = [r['label'] for r in answered]
    yh = [r['prediction'] for r in answered]
    wrong = sum(a != b for a, b in zip(y, yh))
    return dict(
        n_answered=len(answered),
        n_abstained=len(rows) - len(answered),
        answered_real=sum(r['label'] == 0 for r in answered),
        answered_fake=sum(r['label'] == 1 for r in answered),
        coverage=len(answered) / len(rows),
        risk=wrong / len(answered),
        answered_macro_f1=float(f1_score(y, yh, average='macro', labels=[0, 1], zero_division=0)),
        answered_accuracy=1 - wrong / len(answered),
        forced_risk=forced_risk,
    )


def main():
    EMB.mkdir(parents=True, exist_ok=True)
    data = load_data()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('device', device, flush=True)
    table = []
    for run, (target, sources) in RUNS.items():
        preds = read_csv(ROOT / 'results' / 'neural' / run / 'predictions.csv')
        metrics = json.loads((ROOT / 'results' / 'neural' / run / 'metrics.json').read_text(encoding='utf-8'))
        forced_risk = 1 - metrics['accuracy']
        train = [r for name in sources for r in data[name] if r['split'] == 'train']
        val = [r for name in sources for r in data[name] if r['split'] == 'val']
        test_map = {r['id']: r for r in data[target] if r['split'] == 'test'}
        assert {r['id'] for r in preds} == set(test_map)
        final = ROOT / 'checkpoints' / run / 'final'
        tokenizer = AutoTokenizer.from_pretrained(final, local_files_only=True)
        model = AutoModelForSequenceClassification.from_pretrained(final, local_files_only=True).to(device)
        source_path = EMB / f'{run}_source_cls.npy'
        source_label_path = EMB / f'{run}_source_labels.npy'
        test_path = EMB / f'{run}_test_cls.npy'
        val_path = EMB / f'{run}_val_cls.npy'
        val_logit_path = EMB / f'{run}_val_logits.npy'
        if all(p.exists() for p in (source_path, source_label_path, test_path, val_path, val_logit_path)):
            source_cls = np.load(source_path)
            source_y = np.load(source_label_path)
            test_cls = np.load(test_path)
            val_cls = np.load(val_path)
            val_logits = np.load(val_logit_path)
            print(run, 'loaded cached embeddings', flush=True)
        else:
            source_cls, _ = encode([r['text'] for r in train], tokenizer, model, device)
            source_y = np.array([r['label'] for r in train], dtype=np.int64)
            test_cls, _ = encode([test_map[r['id']]['text'] for r in preds], tokenizer, model, device)
            val_cls, val_logits = encode([r['text'] for r in val], tokenizer, model, device)
            np.save(source_path, source_cls)
            np.save(source_label_path, source_y)
            np.save(test_path, test_cls)
            np.save(val_path, val_cls)
            np.save(val_logit_path, val_logits)
            print(run, 'encoded', len(train), 'source', len(val), 'val', len(preds), 'test', flush=True)
        centroids = {}
        for label in (0, 1):
            subset = source_cls[source_y == label]
            vec = subset.mean(axis=0)
            vec = vec / max(np.linalg.norm(vec), 1e-12)
            centroids[label] = vec

        def label_score(cls, yhat):
            return min(1.0, max(0.0, (1.0 + float(np.dot(cls, centroids[int(yhat)]))) / 2.0))

        val_pred = val_logits.argmax(axis=1)
        val_label_c = np.array([label_score(val_cls[i], val_pred[i]) for i in range(len(val))])
        nn_sim = test_cls @ source_cls.T
        nn_max = nn_sim.max(axis=1)
        rows = []
        for i, pred in enumerate(preds):
            yhat = int(pred['prediction'])
            gold = int(pred['label'])
            u = uncertainty(float(pred['prob_real']), float(pred['prob_fake']))
            label_c = label_score(test_cls[i], yhat)
            nn_c = min(1.0, max(0.0, (1.0 + float(nn_max[i])) / 2.0))
            rank = float(np.mean(val_label_c <= label_c))
            rows.append(dict(
                id=pred['id'], dataset=target, label=gold, prediction=yhat,
                uncertainty=u, label_compat=label_c, nn_compat=nn_c,
                label_rank_vs_source_val=rank,
                combined=min(u, label_c),
                combined_rank=min(u, rank),
                correct=gold == yhat,
            ))
        write_csv(OUT / f'{run}_embedding_predictions.csv', rows)
        for rule, key in [
            ('uncertainty_only', 'uncertainty'),
            ('label_centroid_distance', 'label_compat'),
            ('source_nn_distance', 'nn_compat'),
            ('label_rank_vs_source_val', 'label_rank_vs_source_val'),
            ('min_uncertainty_and_label_distance', 'combined'),
            ('min_uncertainty_and_label_rank', 'combined_rank'),
        ]:
            for t in THRESHOLDS:
                stats = summarize(rows, key, t, forced_risk)
                table.append(dict(run_id=run, target=target, rule=rule, threshold=t,
                                  n_test=len(rows), **stats))
        del model
        torch.cuda.empty_cache()
    write_csv(OUT / 'Table_5_EmbeddingDistance.csv', table)
    (OUT / 'Table_5_EmbeddingDistance.md').write_text(
        markdown('Table 5 Embedding Distance', table), encoding='utf-8')
    protocol = ROOT / 'docs' / 'embedding_distance_protocol.md'
    protocol.write_text('''# Embedding-distance compatibility (added 7 October 2026)

This is the optional “embedding distance if easy” term from the start guide.
It uses frozen R4/R5/R6 checkpoints. It does not retrain, does not change
Tables 1–4, and does not use target test labels to fit a threshold.

For a target test item x with saved probabilities p and saved prediction ŷ:
uncertainty = 1 − H(p)/ln(2)
cls(x) = L2-normalized DistilBERT [CLS] (max_len 128)
centroid[y] = mean cls of source TRAIN items with label y, then L2-normalized
label_compat = (1 + cosine(cls(x), centroid[ŷ])) / 2
nn_compat = (1 + max cosine(cls(x), source TRAIN cls)) / 2
label_rank = share of source VAL items whose label_compat is ≤ this item
combined = min(uncertainty, label_compat)
combined_rank = min(uncertainty, label_rank)

Answer if the chosen score ≥ t, t ∈ {0.3, 0.4, 0.5}. Report every rule and
threshold. Raw cosine values on this DistilBERT sit near 0.8–0.95, so the
guide’s 0.3/0.4/0.5 cuts barely move. label_rank rescales distance using
source validation only, which is allowed; target test labels are not used.

Limitation: cosine-to-centroid is still a proxy, not a proof of label meaning.
''', encoding='utf-8')
    print('WROTE', OUT / 'Table_5_EmbeddingDistance.csv')
    for row in table:
        risk = 'N/A' if row['risk'] is None else f"{row['risk']:.4f}"
        f1 = 'N/A' if row['answered_macro_f1'] is None else f"{row['answered_macro_f1']:.4f}"
        print(row['run_id'], row['rule'], row['threshold'],
              'cov', f"{row['coverage']:.4f}", 'risk', risk, 'f1', f1,
              'n', row['n_answered'], flush=True)


if __name__ == '__main__':
    main()
