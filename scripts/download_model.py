"""One-time online setup. Fetch a pinned public model revision; never execute remote code."""
import hashlib
import json
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'models'/'distilbert-base-uncased'
OUT.mkdir(parents=True, exist_ok=True)
repo = 'distilbert/distilbert-base-uncased'
with urllib.request.urlopen('https://huggingface.co/api/models/'+repo, timeout=60) as response:
    info = json.load(response)
revision = info['sha']
manifest = {'repo': repo, 'revision': revision, 'files': {}}
for name in ['config.json', 'tokenizer_config.json', 'tokenizer.json', 'vocab.txt', 'model.safetensors']:
    path = OUT/name
    url = f'https://huggingface.co/{repo}/resolve/{revision}/{name}'
    print('Downloading', name, flush=True)
    partial = path.with_suffix(path.suffix+'.part')
    with urllib.request.urlopen(url, timeout=120) as response, partial.open('wb') as f:
        while chunk := response.read(1024*1024):
            f.write(chunk)
    partial.replace(path)
    manifest['files'][name] = {'bytes': path.stat().st_size,
                              'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
(OUT/'download_manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
print('Model files ready:', OUT, flush=True)
