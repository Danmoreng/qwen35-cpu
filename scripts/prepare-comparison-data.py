#!/usr/bin/env python3
"""Pin public GGUFs and WikiText test windows; share canonical HF token IDs."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys

p = argparse.ArgumentParser()
p.add_argument('--output', type=Path, default=Path('benchmarks/comparison-2026-09-06'))
p.add_argument('--model-dir', default='models/hf-download-test')
p.add_argument('--cli', default='build/qwen35_cpu.exe')
p.add_argument('--skip-downloads', action='store_true')
args = p.parse_args()
sys.path.insert(0, str(Path('.cache/python-deps').resolve()))
from huggingface_hub import hf_hub_download
from transformers import AutoTokenizer
import pyarrow.parquet as pq

repo = 'unsloth/Qwen3.5-0.8B-GGUF'
revision = '6ab461498e2023f6e3c1baea90a8f0fe38ab64d0'
dataset_revision = 'b08601e04326c79dfdd32d625aee71d232d685c3'
data_file = 'wikitext-2-raw-v1/test-00000-of-00001.parquet'
if not args.skip_downloads:
    for name in ['BF16', 'Q4_0', 'Q4_K_M', 'IQ4_XS']:
        hf_hub_download(repo, f'Qwen3.5-0.8B-{name}.gguf', revision=revision,
                        local_dir='models/llama-comparison')
    hf_hub_download('Salesforce/wikitext', data_file, repo_type='dataset',
                    revision=dataset_revision, local_dir='.cache/corpus')
corpus = Path('.cache/corpus') / data_file
rows = pq.read_table(corpus).column('text').to_pylist()
docs, current = [], []
for row in rows:
    if re.match(r'^\s*= [^=].*[^=] =\s*$', row) and current:
        docs.append(''.join(current)); current = []
    current.append(row)
if current:
    docs.append(''.join(current))
tokenizer = AutoTokenizer.from_pretrained(args.model_dir, local_files_only=True)
eligible = []
for index, text in enumerate(docs):
    ids = tokenizer.encode(text, add_special_tokens=False)
    if len(ids) >= 768:
        eligible.append((index, text, ids))
if len(eligible) < 16:
    raise ValueError('Not enough eligible articles')
args.output.mkdir(parents=True, exist_ok=True)
windows, mismatches = [], []
for n in range(16):
    index, text, ids = eligible[n * (len(eligible) - 1) // 15]
    folder = args.output / f'quality-{n:02d}'
    folder.mkdir(exist_ok=True)
    (folder / 'source.txt').write_text(text, encoding='utf-8', newline='\n')
    subprocess.run([args.cli, '--model-dir', args.model_dir, '--prompt-file',
                    str(folder / 'source.txt'), '--tokenize-out', str(folder / 'native-tokens.txt')], check=True)
    native = list(map(int, (folder / 'native-tokens.txt').read_text().split()))
    if native != ids:
        first = next((i for i, (a, b) in enumerate(zip(ids, native)) if a != b), min(len(ids), len(native)))
        mismatches.append(dict(window=n, article_index=index, first_mismatch=first,
                               hf_length=len(ids), native_length=len(native)))
    for name, values in [('prompt', ids[:256]), ('targets', ids[256:768])]:
        (folder / (name + '.csv')).write_text(','.join(map(str, values)), encoding='ascii')
    windows.append(dict(window=n, article_index=index, title=text.strip().splitlines()[0],
                        article_sha256=hashlib.sha256(text.encode()).hexdigest(),
                        prompt_tokens=256, scored_tokens=512, folder=str(folder)))
manifest = dict(dataset=dict(repo='Salesforce/wikitext', revision=dataset_revision,
                             file=data_file, sha256=hashlib.sha256(corpus.read_bytes()).hexdigest()),
                windows=windows, native_tokenizer_mismatches=mismatches,
                tokenization='HF AutoTokenizer, add_special_tokens=False, no chat template; identical IDs for every engine')
(args.output / 'quality-windows.json').write_text(json.dumps(manifest, indent=2) + '\n')
print(f'Prepared {len(windows)} windows / 8192 scored tokens; native tokenizer mismatches: {mismatches}')
