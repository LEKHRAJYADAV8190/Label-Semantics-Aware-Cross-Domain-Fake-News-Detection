"""Reproducible tables, selective prediction and error examples; no training."""
import csv
import json
import math
from pathlib import Path
from sklearn.metrics import accuracy_score, f1_score

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results'/'analysis'
NAMES=['ISOT','LIAR_PLUS','FakeVsSatire']


def read_csv(path):
    with path.open(encoding='utf-8',newline='') as f:
        return list(csv.DictReader(f))


def write_csv(path,rows,fields=None):
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('w',encoding='utf-8',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=fields or list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def markdown(title,rows):
    cols=list(rows[0])
    def format_value(v):
        return 'N/A' if v is None else f'{v:.4f}' if isinstance(v,float) else str(v)
    return '# '+title+'\n\n|'+'|'.join(cols)+'|\n|'+'|'.join(['---']*len(cols))+'|\n'+''.join(
        '|'+'|'.join(format_value(r[c]) for c in cols)+'|\n' for r in rows)


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    data={name:read_csv(ROOT/'data'/'processed'/f'{name}.csv') for name in NAMES}
    tf={r['run_id']:r for r in json.loads((ROOT/'results'/'tfidf'/'summary.json').read_text())}
    nn={f'R{i}':json.loads((ROOT/'results'/'neural'/f'R{i}'/'metrics.json').read_text()) for i in range(1,11)}
    preds={run:read_csv(ROOT/'results'/'neural'/run/'predictions.csv') for run in nn}
    for run,rows in preds.items():
        truth={r['id']:int(r['label']) for r in data[nn[run]['target']] if r['split']=='test'}
        assert len(rows)==len(truth) and {r['id'] for r in rows}==set(truth)
        assert all(int(r['label'])==truth[r['id']] for r in rows)
        y=[int(r['label']) for r in rows]; yh=[int(r['prediction']) for r in rows]
        assert abs(f1_score(y,yh,average='macro',labels=[0,1],zero_division=0)-nn[run]['macro_f1'])<1e-12
        assert abs(accuracy_score(y,yh)-nn[run]['accuracy'])<1e-12
    t1=[]; t2=[]; t3=[]; t4=[]; log=[]
    for i,name in enumerate(NAMES,1):
        for model,ind,ood in [('TF-IDF+LR',tf['in_domain__'+name]['metrics']['test'],tf['lodo__'+name]['metrics']['test']),
                              ('DistilBERT',nn[f'R{i}'],nn[f'R{i+3}'])]:
            ntest=sum(r['split']=='test' for r in data[name])
            t1.append(dict(dataset=name,model=model,n_test=ntest,accuracy=ind['accuracy'],macro_f1=ind['macro_f1']))
            t2.append(dict(target=name,model=model,n_test=ntest,in_domain_f1=ind['macro_f1'],lodo_f1=ood['macro_f1'],
                           f1_drop=ind['macro_f1']-ood['macro_f1'],lodo_accuracy=ood['accuracy']))
    enriched={}
    for run in ['R4','R5','R6']:
        rows=preds[run]
        target=nn[run]['target']
        fulltext={r['id']:r['text'] for r in data[target]}
        decorated=[]
        for row in rows:
            p=[float(row['prob_real']),float(row['prob_fake'])]
            assert all(math.isfinite(v) and 0<=v<=1 for v in p) and abs(sum(p)-1)<1e-5
            score=1+sum(v*math.log(v) if v else 0 for v in p)/math.log(2)
            score=min(1.,max(0.,score))
            decorated.append(dict(id=row['id'],dataset=target,label=int(row['label']),prediction=int(row['prediction']),
                                  compatibility=score,prob_real=p[0],prob_fake=p[1],
                                  correct=int(row['label'])==int(row['prediction']),text=fulltext[row['id']]))
        enriched[run]=decorated
        write_csv(OUT/f'{run}_selective_predictions.csv',[
            {**{k:v for k,v in r.items() if k!='text'},**{f'answer_at_{t}':r['compatibility']>=t for t in [0.3,0.4,0.5]}}
            for r in decorated])
        for t in [0.3,0.4,0.5]:
            answered=[r for r in decorated if r['compatibility']>=t]
            wrong=sum(not r['correct'] for r in answered)
            af1=float(f1_score([r['label'] for r in answered],[r['prediction'] for r in answered],average='macro',labels=[0,1],zero_division=0)) if answered else None
            risk=wrong/len(answered) if answered else None
            t3.append(dict(run_id=run,target=target,threshold=t,n_test=len(rows),n_answered=len(answered),n_abstained=len(rows)-len(answered),
                answered_real=sum(r['label']==0 for r in answered),answered_fake=sum(r['label']==1 for r in answered),
                coverage=len(answered)/len(rows),risk=risk,answered_macro_f1=af1,forced_risk=1-nn[run]['accuracy']))
    for target,base,guided,random in [('LIAR_PLUS','R5','R7','R9'),('FakeVsSatire','R6','R8','R10')]:
        for run in [guided,random]:
            selection=json.loads((ROOT/'results'/'neural'/run/'selection.json').read_text())
            train={r['id']:int(r['label']) for r in data[target] if r['split']=='train'}
            ids=selection['ids']; assert len(ids)==len(set(ids))==16 and set(ids)<=set(train)
            t4.append(dict(run_id=run,target=target,method=selection['method'],K=16,
                          selected_real=sum(train[x]==0 for x in ids),selected_fake=sum(train[x]==1 for x in ids),
                          source_macro_f1=nn[base]['macro_f1'],adapted_macro_f1=nn[run]['macro_f1'],
                          f1_change=nn[run]['macro_f1']-nn[base]['macro_f1'],accuracy=nn[run]['accuracy']))
    for title,rows in [('Table_1_InDomain',t1),('Table_2_LODO',t2),('Table_3_Abstention',t3),('Table_4_LoRA',t4)]:
        write_csv(OUT/(title+'.csv'),rows)
        (OUT/(title+'.md')).write_text(markdown(title.replace('_',' '),rows),encoding='utf-8')
    for r in tf.values():
        m=r['metrics']['test']
        log.append(dict(run_id='TFIDF_'+r['run_id'],phase=r['protocol'],train_data='+'.join(r['sources']),test_data=r['target'],K='',method='TF-IDF+LR',
                        n_train=r['n_train'],n_test=m['n'],accuracy=m['accuracy'],macro_f1=m['macro_f1'],coverage=1,risk=1-m['accuracy'],checkpoint=r['model_path'],notes='Verified saved test predictions; fixed hyperparameters'))
    for run,r in nn.items():
        log.append(dict(run_id=run,phase='adaptation' if r['K'] else 'in_domain' if run in ['R1','R2','R3'] else 'lodo',
            train_data=r['target'] if r['K'] else '+'.join(r['sources']),test_data=r['target'],K=r['K'] or '',method=r['method'],n_train=r['n_train'],n_test=r['n_test'],
            accuracy=r['accuracy'],macro_f1=r['macro_f1'],coverage=1,risk=1-r['accuracy'],checkpoint=r['checkpoint'],
            notes='K counts target labels only; adapted from source LODO model' if r['K'] else 'Saved predictions verified'))
    write_csv(ROOT/'docs'/'run_log.csv',log)
    examples=[]; seen=set()
    def take(run,candidates,reason):
        for row in candidates:
            if row['id'] not in seen:
                seen.add(row['id']); examples.append({**row,'run_id':run,'selection_reason':reason,'threshold':0.4,'abstained':row['compatibility']<0.4})
                return
    for run,rows in enriched.items():
        take(run,sorted([r for r in rows if not r['correct']],key=lambda r:(-r['compatibility'],r['id'])),'highest-compatibility error')
        take(run,sorted([r for r in rows if r['compatibility']<0.4],key=lambda r:(r['compatibility'],r['id'])),'lowest-compatibility abstention')
        take(run,sorted([r for r in rows if r['correct']],key=lambda r:(r['compatibility'],r['id'])),'lowest-compatibility correct prediction')
    for run,rows in enriched.items():
        if len(examples)>=10: break
        take(run,sorted(rows,key=lambda r:(r['compatibility'],r['id'])),'additional uncertainty example')
    assert len(examples)==10
    write_csv(OUT/'error_abstention_examples.csv',examples)
    txt='# Ten illustrative LODO examples\n\nFixed threshold 0.4; deterministic diagnostic selection, not representative sampling. Label 0 is Satire for FakeVsSatire. Excerpts below are limited to 500 characters; model input was max 128 tokens.\n'
    for i,r in enumerate(examples,1):
        txt+=f"\n## {i}. {r['dataset']} — {r['id']}\n\nTrue binary label: {r['label']}; prediction: {r['prediction']}; compatibility: {r['compatibility']:.4f}; abstained: {r['abstained']}. Selection: {r['selection_reason']}.\n\n> "+r['text'][:500].replace('\n',' ')+'\n'
    (OUT/'error_abstention_examples.md').write_text(txt,encoding='utf-8')
    status={'analysis_complete':True,'tables':4,'run_log_rows':len(log),'examples':len(examples),
            'verification':'See independent_verification.json and model_reload_verification.json; separate checks must also pass.',
            'remaining':['teacher submission archive']}
    (OUT/'analysis_status.json').write_text(json.dumps(status,indent=2),encoding='utf-8')
    print(json.dumps(status,indent=2))


if __name__=='__main__':
    main()
