#!/usr/bin/env python3
"""Capture a frozen screening corpus with reproducible whole-context sampling."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import shutil
from evaluation_common import sha, source_identity

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--corpus', type=Path, required=True)
p.add_argument('--out', type=Path, required=True)
p.add_argument('--collector', type=Path, default=Path('build-llama-capture/bin/llama-fixed-cpu-bench.exe'))
p.add_argument('--keep-raw', action='store_true', help='Retain large raw activation files for covariance fitting')
p.add_argument('--teacher-backend', choices=['cpu', 'cuda'], default='cpu')
a = p.parse_args()
a.out.mkdir(parents=True, exist_ok=False)
teacher = Path('models/llama-comparison/Qwen3.5-0.8B-BF16.gguf')
selection = json.loads((a.corpus / 'selection.json').read_text())
metadata = dict(version=1, teacher_sha256=sha(teacher), collector_sha256=sha(a.collector),
    teacher_backend=a.teacher_backend,
    selection_sha256=sha(a.corpus / 'selection.json'), source=source_identity(Path('.')),
    sampling='Algorithm R reservoir, seed 1234 independently per projection, capacity 256 per document.',
    documents=[])
for document in selection['documents']:
    if document['split'] != 'calibration':
        continue
    if shutil.disk_usage(a.out).free < 8 * 1024**3:
        raise RuntimeError('Less than 8 GiB free; stopping before another capture')
    folder = a.out / document['id']
    tokens = list(map(int, (Path(document['folder']) / 'tokens.csv').read_text().split(',')))
    prompt = a.out / (document['id'] + '-prompt.csv')
    prompt.write_text(','.join(map(str, tokens[:-128])))
    command = [str(a.collector.resolve()), '--cpu-gguf', str(teacher), '--prompt-tokens-file', str(prompt),
        '--forced-output-tokens', ','.join(map(str, tokens[-128:])), '--max-new-tokens', '128',
        '--max-context', str(len(tokens)), '--cpu-threads', '8', '--capture-inputs', str(folder),
        '--capture-reservoir', '--profile-json', str(a.out / (document['id'] + '-profile.json'))]
    if sha(a.collector) != metadata['collector_sha256']:
        raise RuntimeError('Collector changed during capture')
    with (a.out / (document['id'] + '.log')).open('wb') as log:
        subprocess.run(command, check=True, stdout=log, stderr=subprocess.STDOUT)
    capture = json.loads((folder / 'capture.json').read_text())
    if len(capture['tensors']) != 187 or any(t['samples'] < 128 for t in capture['tensors']):
        raise RuntimeError('Incomplete projection capture')
    metadata['documents'].append(dict(**document, directory=str(folder), command=command))
    (a.out / 'manifest.json').write_text(json.dumps(metadata, indent=2) + '\n')
    if not a.keep_raw:
        cleanup = a.out / (document['id'] + '-cleanup.json')
        subprocess.run([sys.executable, str(Path(__file__).with_name('compact-calibration-captures.py')),
            str(a.out), '--document-id', document['id'], '--cleanup-manifest', str(cleanup)], check=True)
        subprocess.run(['pwsh', '-File', str(Path(__file__).with_name('remove-compacted-capture-raw.ps1')),
            '-Root', str(a.out), '-Manifest', str(cleanup)], check=True)
    print('Captured', document['id'], flush=True)
for name in ['prose', 'mixture']:
    chosen = json.loads((a.corpus / (name + '.json')).read_text())
    ids = {d['id'] for d in chosen['documents']}
    subset = a.out / name
    subset.mkdir()
    (subset / 'manifest.json').write_text(json.dumps(dict(metadata,
        documents=[d for d in metadata['documents'] if d['id'] in ids]), indent=2) + '\n')
