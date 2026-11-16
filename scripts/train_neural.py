"""Offline DistilBERT / LoRA experiments, sequential GPU use and resumable checkpoints."""
import os
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'
os.environ['HF_HUB_DISABLE_TELEMETRY'] = '1'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
os.environ['WANDB_DISABLED'] = 'true'
import argparse
import csv
import hashlib
import json
import time
from pathlib import Path
import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
from transformers import (AutoTokenizer, AutoModelForSequenceClassification, Trainer,
                          TrainingArguments, TrainerCallback, set_seed)
from transformers.trainer_utils import get_last_checkpoint

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT/'models'/'distilbert-base-uncased'
NAMES = ['ISOT', 'LIAR_PLUS', 'FakeVsSatire']
RUNS = {
    'R1': ('ISOT', ['ISOT'], None),
    'R2': ('LIAR_PLUS', ['LIAR_PLUS'], None),
    'R3': ('FakeVsSatire', ['FakeVsSatire'], None),
    'R4': ('ISOT', ['LIAR_PLUS', 'FakeVsSatire'], None),
    'R5': ('LIAR_PLUS', ['ISOT', 'FakeVsSatire'], None),
    'R6': ('FakeVsSatire', ['ISOT', 'LIAR_PLUS'], None),
    'R7': ('LIAR_PLUS', ['ISOT', 'FakeVsSatire'], ('R5', 'Guided')),
    'R8': ('FakeVsSatire', ['ISOT', 'LIAR_PLUS'], ('R6', 'Guided')),
    'R9': ('LIAR_PLUS', ['ISOT', 'FakeVsSatire'], ('R5', 'Random')),
    'R10': ('FakeVsSatire', ['ISOT', 'LIAR_PLUS'], ('R6', 'Random')),
}


def save_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value, indent=2), encoding='utf-8')
    tmp.replace(path)


def load_data():
    data = {}
    for name in NAMES:
        with (ROOT/'data'/'processed'/f'{name}.csv').open(encoding='utf-8', newline='') as f:
            data[name] = list(csv.DictReader(f))
        for r in data[name]:
            r['label'] = int(r['label'])
    return data


class TextDataset(torch.utils.data.Dataset):
    def __init__(self, rows, tokenizer):
        self.tokens = tokenizer([r['text'] for r in rows], truncation=True,
                                padding='max_length', max_length=128)
        self.labels = [r['label'] for r in rows]
    def __len__(self):
        return len(self.labels)
    def __getitem__(self, i):
        return {**{k: torch.tensor(v[i]) for k, v in self.tokens.items()},
                'labels': torch.tensor(self.labels[i])}


def metrics(pred):
    y = pred.label_ids
    yhat = pred.predictions.argmax(-1)
    return {'accuracy': accuracy_score(y, yhat), 'macro_f1': f1_score(y, yhat, average='macro', labels=[0,1], zero_division=0)}


class Progress(TrainerCallback):
    def __init__(self, run, stop_at=None):
        self.run, self.stop_at = run, stop_at
    def on_log(self, args, state, control, logs=None, **kwargs):
        save_json(ROOT/'logs'/f'{self.run}_progress.json', {'step': state.global_step,
                  'max_steps': state.max_steps, 'epoch': state.epoch, 'logs': logs or {}})
    def on_step_end(self, args, state, control, **kwargs):
        if self.stop_at and state.global_step >= self.stop_at:
            control.should_save = True
            control.should_training_stop = True


def arguments(path, smoke=False, lora=False):
    return TrainingArguments(
        output_dir=str(path), num_train_epochs=2, max_steps=20 if smoke else -1,
        learning_rate=2e-5, per_device_train_batch_size=4, per_device_eval_batch_size=16,
        fp16=True, seed=42, data_seed=42, dataloader_num_workers=0,
        report_to=[], disable_tqdm=True, logging_steps=5 if smoke else 50,
        save_strategy='steps', save_steps=10 if smoke else 250,
        save_total_limit=1, eval_strategy='no' if smoke or lora else 'steps',
        eval_steps=250, load_best_model_at_end=not (smoke or lora),
        metric_for_best_model='macro_f1', greater_is_better=True,
        optim='adamw_torch', lr_scheduler_type='linear', warmup_ratio=0.0,
        save_safetensors=True)


def probabilities(logits):
    a = logits - logits.max(axis=1, keepdims=True)
    a = np.exp(a)
    return a/a.sum(axis=1, keepdims=True)


def run_one(run, data, tokenizer):
    target, sources, adaptation = RUNS[run]
    out = ROOT/'results'/'neural'/run
    checkpoint = ROOT/'checkpoints'/run
    final = checkpoint/'final'
    fingerprints = {name: hashlib.sha256((ROOT/'data'/'processed'/f'{name}.csv').read_bytes()).hexdigest()
                    for name in NAMES}
    signature = {'data_sha256': fingerprints, 'batch':4, 'max_len':128, 'epochs':2,
                 'seed':42, 'lr':2e-5, 'adaptation':adaptation, 'sources':sources, 'target':target}
    # JSON canonical conversion changes tuple to list, so compare serialized form.
    signature = json.loads(json.dumps(signature))
    manifest = checkpoint/'experiment.json'
    if manifest.exists() and json.loads(manifest.read_text()) != signature:
        raise ValueError(f'{run}: settings/data changed; keep prior run and choose a new experiment folder')
    if (out/'metrics.json').exists() and (final/'config.json').exists():
        print(run, 'already complete', flush=True)
        return
    if (out/'metrics.json').exists() and (final/'adapter_config.json').exists():
        print(run, 'already complete', flush=True)
        return
    save_json(manifest, signature)
    set_seed(42)
    train = [r for name in sources for r in data[name] if r['split']=='train']
    val = [r for name in sources for r in data[name] if r['split']=='val']
    test = [r for r in data[target] if r['split']=='test']
    model_path = BASE
    if adaptation:
        model_path = ROOT/'checkpoints'/adaptation[0]/'final'
        if not (model_path/'config.json').exists():
            raise ValueError(f'{run} needs completed {adaptation[0]}')
    model = AutoModelForSequenceClassification.from_pretrained(
        model_path, local_files_only=True, num_labels=2, id2label={0:'Real',1:'Fake'}, label2id={'Real':0,'Fake':1})
    if adaptation:
        from peft import LoraConfig, TaskType, get_peft_model
        pool = [r for r in data[target] if r['split']=='train']
        if adaptation[1]=='Guided':
            scorer = Trainer(model=model, args=arguments(checkpoint/'selection', lora=True))
            prob = probabilities(scorer.predict(TextDataset(pool, tokenizer)).predictions)
            entropy = -(prob*np.log(np.clip(prob,1e-12,1))).sum(axis=1)
            indices = np.argsort(-entropy, kind='stable')[:16]
            del scorer
        else:
            indices = np.random.default_rng(42).choice(len(pool), 16, replace=False)
        train = [pool[int(i)] for i in indices]
        save_json(out/'selection.json', {'method':adaptation[1], 'K':16,
                  'pool':'target train only', 'ids':[r['id'] for r in train],
                  'rule':'highest source-model predictive entropy' if adaptation[1]=='Guided' else 'uniform random without replacement'})
        model = get_peft_model(model, LoraConfig(task_type=TaskType.SEQ_CLS, r=8,
                    lora_alpha=16, lora_dropout=0.1, target_modules=['q_lin','v_lin'],
                    modules_to_save=['pre_classifier','classifier']))
    train_ds = TextDataset(train, tokenizer)
    val_ds = TextDataset(val, tokenizer)
    test_ds = TextDataset(test, tokenizer)
    trainer = Trainer(model=model, args=arguments(checkpoint, lora=bool(adaptation)),
                      train_dataset=train_ds, eval_dataset=val_ds, compute_metrics=metrics,
                      callbacks=[Progress(run)])
    previous = get_last_checkpoint(str(checkpoint))
    print(run, 'train', len(train), 'test', len(test), 'resume', previous, flush=True)
    started=time.perf_counter()
    trainer.train(resume_from_checkpoint=previous)
    train_seconds=time.perf_counter()-started
    trainer.save_model(str(final))
    tokenizer.save_pretrained(final)
    predicted = trainer.predict(test_ds)
    prob = probabilities(predicted.predictions)
    y = np.array([r['label'] for r in test])
    pred = prob.argmax(axis=1)
    out.mkdir(parents=True, exist_ok=True)
    with (out/'predictions.csv').open('w',encoding='utf-8',newline='') as f:
        writer=csv.writer(f); writer.writerow(['id','label','prediction','prob_real','prob_fake'])
        writer.writerows((r['id'],int(y[i]),int(pred[i]),float(prob[i,0]),float(prob[i,1])) for i,r in enumerate(test))
    result={'run_id':run,'target':target,'sources':sources,'method':'LoRA-'+adaptation[1] if adaptation else 'DistilBERT',
            'K':16 if adaptation else None, 'n_train':len(train),'n_test':len(test),
            'accuracy':float(accuracy_score(y,pred)),
            'macro_f1':float(f1_score(y,pred,average='macro',labels=[0,1],zero_division=0)),
            'classification_report':classification_report(y,pred,labels=[0,1],target_names=['Real','Fake'],output_dict=True,zero_division=0),
            'confusion_matrix':confusion_matrix(y,pred,labels=[0,1]).tolist(),
            'checkpoint':str(final),'train_seconds_this_invocation':train_seconds,
            'best_checkpoint':trainer.state.best_model_checkpoint,'resumed_from':previous,'signature':signature}
    save_json(out/'metrics.json', result)
    print(json.dumps(result), flush=True)
    del trainer,model
    torch.cuda.empty_cache()


def smoke(data, tokenizer):
    """Actually train, save, reload optimizer/RNG state, and continue offline."""
    path=ROOT/'checkpoints'/'smoke'
    rows=[r for r in data['ISOT'] if r['split']=='train'][:64]
    ds=TextDataset(rows,tokenizer)
    set_seed(42)
    model=AutoModelForSequenceClassification.from_pretrained(BASE,local_files_only=True,num_labels=2)
    trainer=Trainer(model=model,args=arguments(path,smoke=True),train_dataset=ds,
                    callbacks=[Progress('smoke',stop_at=10)])
    trainer.train()
    previous=get_last_checkpoint(str(path))
    assert trainer.state.global_step==10 and previous
    for name in ['optimizer.pt','scheduler.pt','rng_state.pth','trainer_state.json']:
        assert (Path(previous)/name).exists(), name
    del trainer,model
    torch.cuda.empty_cache()
    model=AutoModelForSequenceClassification.from_pretrained(BASE,local_files_only=True,num_labels=2)
    trainer=Trainer(model=model,args=arguments(path,smoke=True),train_dataset=ds,callbacks=[Progress('smoke')])
    torch.cuda.reset_peak_memory_stats()
    start=time.perf_counter()
    trainer.train(resume_from_checkpoint=previous)
    elapsed=time.perf_counter()-start
    assert trainer.state.global_step==20
    result={'gpu':torch.cuda.get_device_name(0),'torch':torch.__version__,
            'offline':True,'resume_from_step':10,'finished_step':20,
            'resume_segment_seconds_including_load_and_save':elapsed,
            'peak_allocated_gib':torch.cuda.max_memory_allocated()/1024**3,
            'batch_size':4,'max_len':128,'fp16':True,
            'note':'Smoke data only, not a scored experiment. Segment time includes checkpoint I/O.'}
    save_json(ROOT/'logs'/'gpu_smoke.json',result)
    print(json.dumps(result,indent=2),flush=True)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--smoke',action='store_true')
    parser.add_argument('--runs',nargs='+',default=list(RUNS),choices=list(RUNS))
    args=parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable; fix environment instead of silently starting slow CPU training')
    torch.set_num_threads(4)
    data=load_data()
    tokenizer=AutoTokenizer.from_pretrained(BASE,local_files_only=True)
    if args.smoke:
        smoke(data,tokenizer)
    else:
        for run in args.runs:
            run_one(run,data,tokenizer)


if __name__=='__main__':
    main()
