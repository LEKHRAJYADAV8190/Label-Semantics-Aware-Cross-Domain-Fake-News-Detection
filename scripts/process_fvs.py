"""Next step: prepare local final FakeVsSatire texts. No network needed."""
import json
import random
import re
from collections import Counter
from process_step1 import ROOT, RAW, clean, key, sha, deduplicate, validate, write_csv


def main():
    base = RAW/'FakeVsSatire'/'unzipped'/'FakeNewsData'/'StoryText 2'
    rows, excluded, sources, encodings = [], [], {}, {}
    for relative, label in [('Fake/finalFake', 1), ('Satire/finalSatire', 0)]:
        paths = sorted((base/relative).glob('*.txt'))
        if not paths:
            raise ValueError(f'No final text files: {base/relative}')
        for path in paths:
            rid = f'FakeVsSatire:{relative}:{path.name}'
            sources[str(path)] = sha(path)
            # Strict decoding: do not silently discard corrupt characters.
            try:
                text = path.read_text(encoding='utf-8-sig')
                encodings[str(path)] = 'utf-8-sig'
            except UnicodeDecodeError:
                text = path.read_text(encoding='cp1252')
                encodings[str(path)] = 'cp1252'
            # Texts may contain a title and a source URL. Retain title/body,
            # remove URL strings so the source domain is not a classifier input.
            text = clean(re.sub(r'https?://\S+', ' ', text))
            if len(text.split()) < 20:
                excluded.append({'id': rid, 'reason': 'fewer_than_20_words', 'kept_id': ''})
                continue
            rows.append(dict(id=rid, text=text, label=label, dataset='FakeVsSatire', split=''))
    rows = deduplicate(rows, excluded)
    if len(rows) < 200:
        raise ValueError('Fewer than 200 usable texts: stop and document a text backup.')
    rng = random.Random(42)
    for label in (0, 1):
        group = sorted((r for r in rows if r['label'] == label), key=lambda r: r['id'])
        if len(group) < 10:
            raise ValueError('Too few examples for a class')
        rng.shuffle(group)
        train_end = int(len(group)*0.7)
        val_end = train_end + int(len(group)*0.15)
        for i, row in enumerate(group):
            row['split'] = 'train' if i < train_end else 'val' if i < val_end else 'test'
    stats = validate(rows)
    import csv
    for name in ['ISOT', 'LIAR_PLUS']:
        with (ROOT/'data'/'processed'/f'{name}.csv').open(encoding='utf-8', newline='') as f:
            other = {key(r['text']) for r in csv.DictReader(f)}
        if other & {key(r['text']) for r in rows}:
            raise ValueError(f'Cross-dataset overlap with {name}; resolve before training.')
    output = ROOT/'data'/'processed'/'FakeVsSatire.csv'
    write_csv(output, sorted(rows, key=lambda r: (r['split'], r['id'])))
    write_csv(ROOT/'docs'/'fvs_exclusions.csv', excluded, ['id', 'reason', 'kept_id'])
    audit = {'rows': len(rows), 'splits': stats, 'raw_sha256': sources, 'encodings': encodings,
             'output_sha256': sha(output), 'excluded': dict(Counter(r['reason'] for r in excluded)),
             'source': 'Local Golbeck archive finalFake/finalSatire only; no backup used',
             'mapping': {'Fake': 1, 'Satire': 0}, 'seed': 42,
             'limitations': ['Exact duplicates only; near duplicates not ruled out.',
                             'Local archive has not been byte-verified against remote publisher.']}
    (ROOT/'docs'/'fvs_audit.json').write_text(json.dumps(audit, indent=2), encoding='utf-8')
    print(json.dumps({'rows': len(rows), 'splits': stats}, indent=2))
    print('Saved', output)


if __name__ == '__main__':
    main()
