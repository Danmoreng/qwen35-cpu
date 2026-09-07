#!/usr/bin/env python3
"""Keep the CUDA BF16 teacher resident and compact captures with bounded disk use."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from evaluation_common import sha, source_identity

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--corpus', type=Path, required=True)
p.add_argument('--out', type=Path, required=True)
p.add_argument('--collector', type=Path, default=Path('build-llama-calibration-cuda/bin/llama-calibration-resident-cuda.exe'))
p.add_argument('--documents', type=int, help='Optional small validation subset')
p.add_argument('--sequential-teacher', action='store_true')
p.add_argument('--minimum-free-gib', type=float, default=16)
a = p.parse_args()
if a.documents is not None and a.documents <= 0:
    p.error('documents must be positive')
if a.minimum_free_gib <= 0:
    p.error('minimum-free-gib must be positive')
documents = [d for d in json.loads((a.corpus/'selection.json').read_text())['documents'] if d['split']=='calibration']
if a.documents:
    documents = documents[:a.documents]
if not documents:
    raise ValueError('No calibration documents')
a.out.mkdir(parents=True, exist_ok=False)
jobs = a.out/'jobs.tsv'
jobs.write_text(''.join(f"{d['id']}\t{Path(d['folder'])/'tokens.csv'}\n" for d in documents))
teacher = Path('models/llama-comparison/Qwen3.5-0.8B-BF16.gguf')
captures = a.out/'captures'
command = [str(a.collector.resolve()), '--cpu-gguf', str(teacher), '--jobs-file', str(jobs),
    '--output-root', str(captures), '--max-context', str(max(d['tokens'] for d in documents)),
    '--cpu-threads', '8', '--profile-json', str(a.out/'profile.json'), '--wait-for-ack']
if not a.sequential_teacher:
    command.append('--batched-teacher')
metadata = dict(version=1, teacher_backend='cuda', resident=True,
    batched_teacher=not a.sequential_teacher, teacher_sha256=sha(teacher), collector_sha256=sha(a.collector),
    selection_sha256=sha(a.corpus/'selection.json'), source=source_identity(Path('.')),
    command=command, documents=[], maximum_pending_raw_documents=2)


def save_manifest():
    temporary = a.out/'manifest.tmp'
    temporary.write_text(json.dumps(metadata, indent=2)+'\n')
    temporary.replace(a.out/'manifest.json')


def compact(document):
    cleanup = a.out/(document['id']+'-cleanup.json')
    with (a.out/(document['id']+'-compaction.log')).open('w') as log:
        subprocess.run([sys.executable, str(Path(__file__).with_name('compact-calibration-captures.py')),
            str(a.out), '--document-id', document['id'], '--cleanup-manifest', str(cleanup)],
            stdout=log, stderr=subprocess.STDOUT, check=True)
        subprocess.run(['pwsh', '-File', str(Path(__file__).with_name('remove-compacted-capture-raw.ps1')),
            '-Root', str(a.out), '-Manifest', str(cleanup)], stdout=log, stderr=subprocess.STDOUT, check=True)


save_manifest()
if shutil.disk_usage(a.out).free < a.minimum_free_gib*1024**3:
    raise RuntimeError('Insufficient free space before teacher startup')
with (a.out/'teacher.log').open('w') as log, ThreadPoolExecutor(max_workers=1) as pool:
    process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=log,
        text=True, bufsize=1, creationflags=subprocess.CREATE_NO_WINDOW if os.name=='nt' else 0)
    pending = None
    try:
        for line in process.stdout:
            if not line.startswith('CAPTURED\t'):
                continue
            index = len(metadata['documents'])
            if index >= len(documents) or line.strip().split('\t')[1] != documents[index]['id']:
                raise RuntimeError('Unexpected document acknowledgement')
            document = documents[index]
            folder = captures/document['id']
            capture = json.loads((folder/'capture.json').read_text())
            if len(capture['tensors']) != 187 or any(t['samples'] < 128 for t in capture['tensors']):
                raise RuntimeError('Incomplete capture')
            if pending is not None:
                pending.result()
            metadata['documents'].append(dict(**document, directory=str(folder)))
            save_manifest()
            pending = pool.submit(compact, document)
            if sha(a.collector) != metadata['collector_sha256']:
                raise RuntimeError('Collector changed during run')
            if shutil.disk_usage(a.out).free < a.minimum_free_gib*1024**3:
                pending.result()
                if shutil.disk_usage(a.out).free < a.minimum_free_gib*1024**3:
                    raise RuntimeError('Free-space reserve reached')
            process.stdin.write(document['id']+'\n')
            process.stdin.flush()
            print('Captured resident:',document['id'],flush=True)
        if pending is not None:
            pending.result()
        if process.wait() or len(metadata['documents']) != len(documents):
            raise RuntimeError('Incomplete resident collection')
    finally:
        if process.poll() is None:
            process.kill()
            process.wait()
        process.stdin.close()
        process.stdout.close()
print('Resident collection and raw cleanup completed.',flush=True)
