"""Offline, standard-library-only ISOT and LIAR-PLUS preparation. Python 3.10+."""
import csv
import hashlib
import html
import json
import random
import re
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT.parent / 'dataset' / 'raw'
FIELDS = ['id', 'text', 'label', 'dataset', 'split']
LABELS = {'pants-fire': 1, 'false': 1, 'barely-true': 1,
          'half-true': 0, 'mostly-true': 0, 'true': 0}
csv.field_size_limit(10_000_000)


def clean(text, isot=False):
    text = unicodedata.normalize('NFKC', html.unescape(text))
    if isot:
        text = re.sub(r'^.{0,180}?\(Reuters\)\s*[-–—:]?\s*', '', text, flags=re.I)
        text = re.sub(r'21st\s+Century\s+Wire\s+says\s*[:\-–—]?', '', text, flags=re.I)
    return re.sub(r'\s+', ' ', text).strip()


def key(text):
    return clean(text).casefold()


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def deduplicate(rows, exclusions):
    groups = defaultdict(list)
    for row in rows:
        groups[key(row['text'])].append(row)
    result = []
    # Preserve official evaluation allocation: test > val > train.
    priority = {'test': 0, 'val': 1, 'train': 2, '': 3}
    for group in groups.values():
        group.sort(key=lambda r: (priority[r['split']], r['id']))
        conflict = len({r['label'] for r in group}) > 1
        if not conflict:
            result.append(group[0])
        for row in group if conflict else group[1:]:
            exclusions.append({'id': row['id'], 'reason': 'conflicting_text_labels' if conflict else 'duplicate_text',
                               'kept_id': '' if conflict else group[0]['id']})
    return result


def split_sample(rows):
    result = []
    rng = random.Random(42)
    for label in (0, 1):
        group = sorted((r for r in rows if r['label'] == label), key=lambda r: r['id'])
        if len(group) < 3000:
            raise ValueError('Not enough valid ISOT articles for 3000 per class')
        rng.shuffle(group)
        for i, row in enumerate(group[:3000]):
            row['split'] = 'train' if i < 2100 else 'val' if i < 2550 else 'test'
            result.append(row)
    return sorted(result, key=lambda r: (r['split'], r['id']))


def validate(rows):
    assert rows and len({r['id'] for r in rows}) == len(rows), 'Duplicate IDs'
    assert len({key(r['text']) for r in rows}) == len(rows), 'Duplicate text / split leakage'
    for row in rows:
        assert row['label'] in (0, 1) and row['split'] in ('train', 'val', 'test')
        assert row['text'] and not re.fullmatch(r'https?://\S+', row['text'])
    return {split: dict(sorted(Counter(str(r['label']) for r in rows if r['split'] == split).items()))
            for split in ('train', 'val', 'test')}


def write_csv(path, rows, fields=FIELDS):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    with tmp.open('w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    tmp.replace(path)


def main():
    report = {'seed': 42, 'raw_root': str(RAW), 'sources': {}, 'datasets': {},
              'limitations': ['ISOT uses the Phoenyx83 refined 2022 variant, not the full original release.',
                              'Only normalized exact duplicates are removed; near duplicates are not detected.',
                              'No training or scores exist yet.']}
    excluded = []
    isot = []
    raw_counts = {}
    for filename, label in [('True.csv', 0), ('Fake.csv', 1)]:
        path = RAW / 'ISOT' / filename
        report['sources'][str(path)] = sha(path)
        with path.open(encoding='utf-8-sig', newline='') as f:
            reader = csv.DictReader(f)
            if reader.fieldnames != ['type', 'text']:
                raise ValueError(f'Unexpected ISOT schema: {reader.fieldnames}')
            count = 0
            for number, row in enumerate(reader, 2):
                count += 1
                rid = f'ISOT:{filename}:{number}'
                text = clean(row.get('text') or '', isot=True)
                reason = ('invalid_or_missing_label' if row.get('type') != str(label) else
                          'empty_or_url_only' if not text or re.fullmatch(r'https?://\S+', text) else
                          'malformed_columns' if None in row else '')
                if reason:
                    excluded.append({'id': rid, 'reason': reason, 'kept_id': ''})
                else:
                    isot.append(dict(id=rid, text=text, label=label, dataset='ISOT', split=''))
            raw_counts[filename] = count
    isot_clean = deduplicate(isot, excluded)
    isot = split_sample(isot_clean)
    report['datasets']['ISOT'] = {'raw_counts': raw_counts, 'valid_unique_before_sampling': len(isot_clean),
                                   'rows': len(isot), 'splits': validate(isot)}
    liar = []
    raw_counts = {}
    previews = []
    unmapped = Counter()
    for filename, split in [('train2.tsv', 'train'), ('val2.tsv', 'val'), ('test2.tsv', 'test')]:
        path = RAW / 'LIAR_PLUS' / filename
        report['sources'][str(path)] = sha(path)
        with path.open(encoding='utf-8-sig', newline='') as f:
            reader = csv.reader(f, delimiter='\t')
            count = 0
            for number, row in enumerate(reader, 1):
                count += 1
                rid = f'LIAR_PLUS:{filename}:record{number}'
                if number <= 3:
                    previews.append({'file': filename, 'record': number, 'columns': len(row),
                                     'label_col_2': row[2] if len(row)>2 else None,
                                     'text_col_3': row[3][:200] if len(row)>3 else None})
                if len(row) != 16:
                    excluded.append({'id': rid, 'reason': f'malformed_columns_{len(row)}', 'kept_id': ''})
                    continue
                if row[2] not in LABELS:
                    unmapped[row[2]] += 1
                    continue
                text = clean(row[3])
                if not text or not re.fullmatch(r'\d+\.json', row[1]):
                    excluded.append({'id': rid, 'reason': 'empty_text_or_invalid_id', 'kept_id': ''})
                    continue
                liar.append(dict(id='LIAR_PLUS:'+row[1], text=text, label=LABELS[row[2]],
                                 dataset='LIAR_PLUS', split=split))
            raw_counts[filename] = count
    # "Unmapped labels must be 0" means zero unmapped records, never map unknown labels to Real.
    if unmapped:
        raise ValueError(f'Unmapped LIAR labels: {dict(unmapped)}')
    liar = deduplicate(liar, excluded)
    report['datasets']['LIAR_PLUS'] = {'raw_counts': raw_counts, 'rows': len(liar),
                                      'splits': validate(liar), 'unmapped_labels': 0}
    assert not ({key(r['text']) for r in isot} & {key(r['text']) for r in liar}), 'Cross-dataset overlap'
    report['cross_dataset_exact_overlap'] = 0
    report['exclusions'] = dict(Counter(r['reason'] for r in excluded))
    report['liar_first_rows'] = previews
    for name, rows in [('ISOT', isot), ('LIAR_PLUS', liar)]:
        path = ROOT / 'data' / 'processed' / (name + '.csv')
        write_csv(path, rows)
        report['datasets'][name]['output_sha256'] = sha(path)
    write_csv(ROOT/'docs'/'step1_exclusions.csv', excluded, ['id', 'reason', 'kept_id'])
    (ROOT/'docs'/'step1_audit.json').write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding='utf-8')
    print(json.dumps(report['datasets'], indent=2))
    print('Excluded records:', dict(Counter(r['reason'] for r in excluded)))
    print('PASS: no empty texts, unknown labels, duplicate IDs, normalized exact text duplicates or split leakage.')


if __name__ == '__main__':
    main()
