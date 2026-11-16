"""Process the extra paper datasets. Does not overwrite ISOT/LIAR_PLUS/FakeVsSatire."""
import csv
import html as html_lib
import json
import random
import re
import ssl
import urllib.error
import urllib.request
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from process_step1 import (
    RAW, ROOT, clean, deduplicate, key, sha, validate, write_csv,
)

csv.field_size_limit(10_000_000)
UA = {'User-Agent': 'Mozilla/5.0 (compatible; fake-news-30 research; +local)'}
CTX = ssl.create_default_context()
CTX.check_hostname = False
CTX.verify_mode = ssl.CERT_NONE
LOCKED = {'ISOT', 'LIAR_PLUS', 'FakeVsSatire'}


def read_txt(path):
    try:
        return path.read_text(encoding='utf-8-sig')
    except UnicodeDecodeError:
        return path.read_text(encoding='cp1252', errors='replace')


def html_to_text(blob):
    blob = re.sub(r'(?is)<script.*?>.*?</script>', ' ', blob)
    blob = re.sub(r'(?is)<style.*?>.*?</style>', ' ', blob)
    blob = re.sub(r'(?s)<[^>]+>', ' ', blob)
    return clean(html_lib.unescape(blob))


def fetch_url(url, timeout=8):
    if not url or not url.startswith('http'):
        return ''
    req = urllib.request.Request(url, headers=UA)
    try:
        with urllib.request.urlopen(req, timeout=timeout, context=CTX) as resp:
            raw = resp.read(400_000)
        return html_to_text(raw.decode('utf-8', 'replace'))
    except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, ValueError, ssl.SSLError):
        return ''


def fetch_many(urls, workers=8):
    out = {}
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futs = {pool.submit(fetch_url, u): u for u in urls}
        for i, fut in enumerate(as_completed(futs), 1):
            url = futs[fut]
            try:
                out[url] = fut.result()
            except Exception:
                out[url] = ''
            if i % 50 == 0:
                print(f'  fetched {i}/{len(urls)}', flush=True)
    return out


def split_60_20_20(rows, seed=42):
    rng = random.Random(seed)
    for label in (0, 1):
        group = sorted((r for r in rows if r['label'] == label), key=lambda r: r['id'])
        if len(group) < 8:
            raise ValueError(f'Too few class-{label} rows: {len(group)}')
        rng.shuffle(group)
        n = len(group)
        n_train = max(1, int(round(n * 0.60)))
        n_val = max(1, int(round(n * 0.20)))
        if n_train + n_val >= n:
            n_val = max(1, n - n_train - 1)
        for i, row in enumerate(group):
            if i < n_train:
                row['split'] = 'train'
            elif i < n_train + n_val:
                row['split'] = 'val'
            else:
                row['split'] = 'test'
    return rows


def finalize(name, rows, excluded, extra):
    rows = deduplicate(rows, excluded)
    if any(r.get('split') for r in rows):
        for r in rows:
            if r['split'] not in ('train', 'val', 'test'):
                r['split'] = ''
        if any(not r['split'] for r in rows):
            rows = split_60_20_20(rows)
    else:
        rows = split_60_20_20(rows)
    stats = validate(rows)
    path = ROOT / 'data' / 'processed' / f'{name}.csv'
    write_csv(path, sorted(rows, key=lambda r: (r['split'], r['id'])))
    write_csv(ROOT / 'docs' / f'{name.lower()}_exclusions.csv', excluded, ['id', 'reason', 'kept_id'])
    labels = Counter(r['label'] for r in rows)
    audit = {
        'dataset': name, 'rows': len(rows), 'splits': stats,
        'fake': labels.get(1, 0), 'real': labels.get(0, 0),
        'output_sha256': sha(path),
        'excluded': dict(Counter(r['reason'] for r in excluded)),
        **extra,
    }
    (ROOT / 'docs' / f'{name.lower()}_audit.json').write_text(
        json.dumps(audit, indent=2), encoding='utf-8')
    print(json.dumps({'dataset': name, 'rows': len(rows), 'splits': stats,
                      'fake': labels.get(1, 0), 'real': labels.get(0, 0)}, indent=2), flush=True)
    return audit


def celebrity():
    base = RAW / 'Celebrity' / 'unzipped' / 'fakeNewsDatasets' / 'celebrityDataset'
    rows, excluded = [], []
    for folder, label in [('fake', 1), ('legit', 0)]:
        for path in sorted((base / folder).glob('*.txt')):
            rid = f'Celebrity:{folder}:{path.name}'
            text = clean(read_txt(path))
            if len(text.split()) < 8:
                excluded.append({'id': rid, 'reason': 'too_short', 'kept_id': ''})
                continue
            rows.append(dict(id=rid, text=text, label=label, dataset='Celebrity', split=''))
    return finalize('Celebrity', rows, excluded, {
        'source': 'Perez-Rosas UMich fakeNewsDatasets.zip celebrityDataset',
        'mapping': {'fake': 1, 'legit': 0},
    })


def fakes():
    path = RAW / 'Fakes' / 'FA-KES-Dataset.csv'
    rows, excluded = [], []
    with path.open(encoding='latin-1', newline='') as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader, 2):
            rid = f'Fakes:{i}:{row.get("unit_id") or "na"}'
            text = clean((row.get('article_title') or '') + ' ' + (row.get('article_content') or ''))
            raw = (row.get('labels') or '').strip()
            if raw not in ('0', '1'):
                excluded.append({'id': rid, 'reason': 'invalid_label', 'kept_id': ''})
                continue
            # FA-KES: 0=fake, 1=credible. Ours: Fake=1, Real=0.
            label = 1 if raw == '0' else 0
            if len(text.split()) < 8:
                excluded.append({'id': rid, 'reason': 'too_short', 'kept_id': ''})
                continue
            rows.append(dict(id=rid, text=text, label=label, dataset='Fakes', split=''))
    return finalize('Fakes', rows, excluded, {
        'source': 'Zenodo 2607278 FA-KES-Dataset.csv',
        'mapping': {'FA-KES 0 fake': 1, 'FA-KES 1 credible': 0},
    })


def horne():
    base = RAW / 'Horne' / 'data' / 'Public Data'
    rows, excluded = [], []
    mapping = [
        (base / 'Buzzfeed Political News Dataset' / 'Fake', 1),
        (base / 'Buzzfeed Political News Dataset' / 'Real', 0),
        (base / 'Random Poltical News Dataset' / 'Fake', 1),
        (base / 'Random Poltical News Dataset' / 'Real', 0),
        (base / 'Random Poltical News Dataset' / 'Satire', 0),
    ]
    for folder, label in mapping:
        for path in sorted(folder.glob('*.txt')):
            rid = f'Horne:{folder.parent.name}:{folder.name}:{path.name}'
            text = clean(read_txt(path))
            if len(text.split()) < 8:
                excluded.append({'id': rid, 'reason': 'too_short', 'kept_id': ''})
                continue
            rows.append(dict(id=rid, text=text, label=label, dataset='Horne', split=''))
    return finalize('Horne', rows, excluded, {
        'source': 'rpitrust/fakenewsdata1 Horne2017 Public Data',
        'mapping': {'Fake': 1, 'Real or Satire': 0},
    })


def infodemic():
    rows, excluded = [], []
    files = [
        (RAW / 'Infodemic' / 'Constraint_Train.csv', 'train'),
        (RAW / 'Infodemic' / 'Constraint_Val.csv', 'val'),
        (RAW / 'Infodemic' / 'english_test_with_labels.csv', 'test'),
    ]
    for path, split in files:
        with path.open(encoding='utf-8', newline='') as f:
            for row in csv.DictReader(f):
                rid = f"Infodemic:{split}:{row.get('id')}"
                text = clean(row.get('tweet') or '')
                lab = (row.get('label') or '').strip().lower()
                if lab not in ('fake', 'real'):
                    excluded.append({'id': rid, 'reason': 'invalid_label', 'kept_id': ''})
                    continue
                if len(text.split()) < 3:
                    excluded.append({'id': rid, 'reason': 'too_short', 'kept_id': ''})
                    continue
                rows.append(dict(id=rid, text=text, label=1 if lab == 'fake' else 0,
                                 dataset='Infodemic', split=split))
    return finalize('Infodemic', rows, excluded, {
        'source': 'CONSTRAINT 2021 English (diptamath/covid_fake_news)',
        'mapping': {'fake': 1, 'real': 0},
        'splits': 'official train/val/labeled-test',
    })


def politifact():
    rows, excluded = [], []
    items = []
    for fn, label in [('politifact_fake.csv', 1), ('politifact_real.csv', 0)]:
        path = RAW / 'Politifact' / fn
        with path.open(encoding='utf-8', newline='') as f:
            for row in csv.DictReader(f):
                rid = f"Politifact:{row.get('id')}"
                title = clean(row.get('title') or '')
                url = (row.get('news_url') or '').strip()
                items.append((rid, title, url, label))
    print('Politifact fetching', len(items), 'urls', flush=True)
    bodies = fetch_many([u for _, _, u, _ in items if u], workers=10)
    for rid, title, url, label in items:
        body = clean(bodies.get(url) or '')
        text = body if len(body.split()) >= 20 else title
        if len(text.split()) < 3:
            excluded.append({'id': rid, 'reason': 'too_short', 'kept_id': ''})
            continue
        rows.append(dict(id=rid, text=text, label=label, dataset='Politifact', split=''))
    n_body = sum(1 for r in rows if r['id'] in {i[0] for i in items} and len(r['text'].split()) >= 20)
    return finalize('Politifact', rows, excluded, {
        'source': 'KaiDMML/FakeNewsNet politifact_fake/real + URL fetch fallback to title',
        'n_with_fetched_body_approx': n_body,
        'limitation': 'Dead URLs fall back to headline text, unlike the paper crawl date.',
    })


def ndf():
    path = RAW / 'NDF' / 'NDFDataset.tsv'
    with path.open(encoding='utf-8', newline='') as f:
        raw_rows = list(csv.DictReader(f, delimiter='\t'))
    urls = [r['url'].strip() for r in raw_rows if (r.get('url') or '').strip()]
    print('NDF fetching', len(urls), 'urls (IC labels)', flush=True)
    bodies = fetch_many(urls, workers=10)
    rows, excluded = [], []
    for r in raw_rows:
        rid = f"NDF:{r.get('id')}"
        lab = (r.get('in-context') or '').strip().lower()
        if lab not in ('fake', 'real'):
            excluded.append({'id': rid, 'reason': 'invalid_ic_label', 'kept_id': ''})
            continue
        text = clean(bodies.get((r.get('url') or '').strip()) or '')
        if len(text.split()) < 8:
            excluded.append({'id': rid, 'reason': 'fetch_failed_or_too_short', 'kept_id': ''})
            continue
        rows.append(dict(id=rid, text=text, label=1 if lab == 'fake' else 0,
                         dataset='NDF', split=''))
    if len(rows) < 30:
        print('NDF too small after fetch; keeping whatever remains', flush=True)
    return finalize('NDF', rows, excluded, {
        'source': 'Unipisa/NDFDataset IC variant + live URL fetch',
        'mapping': {'IC Fake': 1, 'IC Real': 0},
        'limitation': 'Official repo has URLs only; tweets/articles that 404 are dropped.',
    })


def cidii():
    candidates = list((RAW / 'Cidii').rglob('*.csv')) + list((RAW / 'Cidii').rglob('*.xlsx'))
    if not candidates:
        print('Cidii missing (Kaggle login required)', flush=True)
        return {'dataset': 'Cidii', 'status': 'missing_kaggle'}
    raise NotImplementedError(f'Cidii files present but parser not wired: {candidates}')


def main():
    report = {'locked_untouched': sorted(LOCKED), 'datasets': {}}
    for fn in (celebrity, fakes, horne, infodemic, politifact, ndf, cidii):
        print('====', fn.__name__, flush=True)
        try:
            report['datasets'][fn.__name__] = fn()
        except Exception as exc:
            print(f'FAILED {fn.__name__}: {type(exc).__name__}: {exc}', flush=True)
            report['datasets'][fn.__name__] = {'status': 'failed', 'error': str(exc)}
    out = ROOT / 'docs' / 'extra_datasets_audit.json'
    out.write_text(json.dumps(report, indent=2), encoding='utf-8')
    print('Wrote', out, flush=True)
    ok = [k for k, v in report['datasets'].items() if isinstance(v, dict) and v.get('rows')]
    print('OK_DATASETS', ok, flush=True)


if __name__ == '__main__':
    main()
