#!/usr/bin/env python3
"""Collect actual CPU BF16 projection inputs from disjoint local source documents."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys
from evaluation_common import sha, source_identity

parser = argparse.ArgumentParser()
parser.add_argument('--out', type=Path, required=True)
parser.add_argument('--collector', type=Path, default=Path('build-llama-capture/bin/llama-fixed-cpu-bench.exe'))
parser.add_argument('--teacher', type=Path, default=Path('models/llama-comparison/Qwen3.5-0.8B-BF16.gguf'))
parser.add_argument('--evaluation', type=Path, default=Path('benchmarks/comparison-2026-09-06/quality-windows.json'))
parser.add_argument('--corpus', type=Path, default=Path('.cache/corpus/wikitext-2-raw-v1/test-00000-of-00001.parquet'))
parser.add_argument('--documents', type=int, default=4)
args = parser.parse_args()
if args.documents <= 0:
    parser.error('documents must be positive')
sys.path.insert(0, str(Path('.cache/python-deps').resolve()))
import pyarrow.parquet as pq
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained('models/hf-download-test', local_files_only=True)
evaluation = json.loads(args.evaluation.read_text())
excluded = {w['article_sha256'] for w in evaluation['windows']}
excluded_titles = {w.get('title', '').strip() for w in evaluation['windows']}
# Compare normalized paragraphs in complete source articles, not only scored windows.
def paragraphs(text):
    return {hashlib.sha256(re.sub(r'\s+', ' ', p).strip().lower().encode()).hexdigest()
            for p in text.split('\n') if len(p.strip()) >= 80}
eval_paragraphs = set()
for window in evaluation['windows']:
    source = Path(window['folder'])/'source.txt'
    if not source.exists():
        raise ValueError('Complete evaluation sources required for deduplication')
    eval_paragraphs.update(paragraphs(source.read_text(encoding='utf-8')))
docs, current = [], []
for row in pq.read_table(args.corpus).column('text').to_pylist():
    if re.match(r'^\s*= [^=].*[^=] =\s*$', row) and current:
        docs.append(''.join(current)); current = []
    current.append(row)
if current:
    docs.append(''.join(current))
selected = []
for index, text in enumerate(docs):
    if not text.strip():
        continue
    digest = hashlib.sha256(text.encode()).hexdigest()
    if digest in excluded or text.strip().splitlines()[0].strip() in excluded_titles or paragraphs(text) & eval_paragraphs:
        continue
    ids = tokenizer.encode(text, add_special_tokens=False)
    if len(ids) < 640:
        continue
    selected.append((index, digest, ids[:640]))
    eval_paragraphs.update(paragraphs(text))
    if len(selected) == args.documents:
        break
if len(selected) != args.documents:
    raise ValueError('Insufficient disjoint calibration documents')
args.out.mkdir(parents=True, exist_ok=False)
manifest = dict(version=1, seed=1234, selection='First eligible documents in corpus order',
    limitation='English prose pilot; broader calibration is required before promotion',
    teacher_sha256=sha(args.teacher), collector_sha256=sha(args.collector), corpus_sha256=sha(args.corpus),
    evaluation_sha256=sha(args.evaluation), source=source_identity(Path('.')),
    deduplication='Distinct source hashes/titles and normalized complete-source paragraphs >=80 characters', documents=[])
for order, (index, digest, ids) in enumerate(selected):
    prompt = args.out/f'prompt-{order}.csv'
    prompt.write_text(','.join(map(str, ids[:512])))
    command = [str(args.collector.resolve()), '--cpu-gguf', str(args.teacher), '--prompt-tokens-file', str(prompt),
        '--forced-output-tokens', ','.join(map(str, ids[512:])), '--max-new-tokens', '128',
        '--max-context', '1024', '--cpu-threads', '8', '--capture-inputs', str(args.out/str(order)),
        '--profile-json', str(args.out/f'profile-{order}.json')]
    with (args.out/f'capture-{order}.log').open('wb') as log:
        subprocess.run(command, check=True, stdout=log, stderr=subprocess.STDOUT)
    manifest['documents'].append(dict(article_index=index, article_sha256=digest,
        token_sha256=hashlib.sha256(','.join(map(str, ids)).encode()).hexdigest(),
        command=command, directory=str(args.out/str(order)), prompt_tokens=512, continuation_tokens=128))
    (args.out/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    print(f'Captured document {order+1}/{args.documents}', flush=True)
