#!/usr/bin/env python3
"""Collect qev teacher moments with one resident external teacher and bounded raw storage.

All projection captures are compacted in the correct H128/identity bases. Only
job-generated raw files are removed, after their hashes and aggregate moments
have been recorded. Snapshot fitting statistics at fixed request counts.
"""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import shutil
import struct
import subprocess
import sys
import uuid
import numpy as np
from threadpoolctl import threadpool_limits

sys.path.insert(0, str(Path(__file__).parent))
from evaluation_common import sha
spec = importlib.util.spec_from_file_location('qev_diagonal', Path(__file__).with_name('fit-dot4-quant.py'))
diagonal = importlib.util.module_from_spec(spec)
spec.loader.exec_module(diagonal)


def save_moments(folder, moments, metadata):
    folder.mkdir(parents=True, exist_ok=True)
    temp = folder / 'block-statistics.tmp'
    with temp.open('wb') as stream:
        np.savez(stream, **moments)
    with np.load(temp, allow_pickle=False) as archive:
        if set(archive.files) != set(moments) or any(not np.array_equal(archive[k], moments[k]) for k in moments):
            raise ValueError('Aggregate statistics readback failed')
    temp.replace(folder / 'block-statistics.npz')
    saved = dict(metadata, statistics_sha256=sha(folder / 'block-statistics.npz'))
    temp = folder / 'block-statistics.json.tmp'
    temp.write_text(json.dumps(saved, indent=2)+'\n')
    temp.replace(folder / 'block-statistics.json')


def save_diagonal(folder, sums, counts, basis, source):
    folder.mkdir(parents=True, exist_ok=False)
    tensors = []
    for name in sorted(sums):
        values = sums[name] / counts[name]
        mean = values.mean()
        if mean > 0:
            values /= mean
        values = values.astype('<f4')
        if not np.isfinite(values).all() or np.any(values < 0):
            raise ValueError('Invalid diagonal importance')
        seed = diagonal.SEED if basis[name] else 0
        path = folder / (name+'.cal')
        path.write_bytes(struct.pack('<8sQQIQ', b'Q35CAL1\0', len(values), seed, basis[name], counts[name]) + values.tobytes())
        tensors.append(dict(name=name, columns=len(values), samples=counts[name], basis=basis[name], sha256=sha(path)))
    (folder/'manifest.json').write_text(json.dumps(dict(version=1, recipe='activation-weighted-mse16',
        sign_seed=diagonal.SEED, source=source, tensors=tensors), indent=2)+'\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepared', type=Path, required=True)
    parser.add_argument('--teacher', type=Path, required=True)
    parser.add_argument('--weights', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--snapshots', default='256,1024')
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--minimum-free-gib', type=float, default=15)
    parser.add_argument('--resume', action='store_true', help='Resume a verified aggregate checkpoint')
    args = parser.parse_args()
    snapshots = sorted({int(x) for x in args.snapshots.split(',')})
    jobs = (args.prepared/'jobs.tsv').read_text().splitlines()
    if snapshots[0] < 1 or snapshots[-1] > len(jobs):
        parser.error('Snapshot count outside corpus')
    args.out.mkdir(parents=True, exist_ok=args.resume)
    jobs = jobs[:snapshots[-1]]
    moments, diagonal_sums, counts, bases = {}, {}, {}, {}
    resumed = 0
    info = dict(version=1, sign_seed=diagonal.SEED, teacher_sha256=sha(args.weights),
                executable_sha256=sha(args.teacher), corpus_sha256=sha(args.prepared/'requests.json'),
                captures=[], tensors=[], snapshots=snapshots, minimum_free_gib=args.minimum_free_gib)
    if args.resume:
        folder = args.out/'checkpoint'
        saved = json.loads((folder/'block-statistics.json').read_text())
        if sha(folder/'block-statistics.npz') != saved['statistics_sha256']:
            raise ValueError('Resume statistics checksum mismatch')
        for key in ['sign_seed', 'teacher_sha256', 'executable_sha256', 'corpus_sha256']:
            if saved[key] != info[key]:
                raise ValueError('Resume identity mismatch: '+key)
        with np.load(folder/'block-statistics.npz', allow_pickle=False) as archive:
            moments = {key:archive[key].copy() for key in archive.files}
        for tensor in saved['tensors']:
            name = tensor['name']
            counts[name], bases[name] = tensor['samples'], tensor['basis']
            # Diagonal entries of X^T X are the same sufficient statistic as sum(X^2).
            diagonal_sums[name] = np.diagonal(moments[name],axis1=1,axis2=2).reshape(-1).copy()
        info = saved
        resumed = len(info['captures'])
        if [d['id'] for d in info['captures']] != [j.split('\t')[0] for j in jobs[:resumed]]:
            raise ValueError('Resume corpus order mismatch')
        info.setdefault('resumes', []).append(dict(requests=resumed, diagonal_from_block_moments=True))
        jobs = jobs[resumed:]
        if not jobs:
            raise ValueError('Nothing to resume')
    jobs_path = args.out/(f'jobs-resume-{resumed}.tsv' if resumed else 'jobs.tsv')
    jobs_path.write_text('\n'.join(jobs)+'\n')
    # One capture is at most a few hundred MiB; keep temporary activations in RAM.
    capture_root = Path('/dev/shm') / ('qev-captures-'+uuid.uuid4().hex)
    command = [str(args.teacher.resolve()), '--weights', str(args.weights.resolve()),
               '--jobs-file', str(jobs_path.resolve()),
               '--letters-file', str((args.prepared/'letters.txt').resolve()),
               '--output', str((args.out/(f'option-logits-resume-{resumed}.tsv' if resumed else 'option-logits.tsv')).resolve()),
               '--capture-root', str(capture_root), '--wait-for-ack',
               '--threads', str(args.threads), '--max-context', '32768']
    suffix = f'-resume-{resumed}' if resumed else ''
    (args.out/('command'+suffix+'.json')).write_text(json.dumps(command, indent=2)+'\n')
    with (args.out/('teacher'+suffix+'.stderr.txt')).open('w') as log, threadpool_limits(limits=2):
        process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=log,
                                   text=True, bufsize=1)
        try:
            for index, expected_line in enumerate(jobs, resumed+1):
                ident = expected_line.split('\t')[0]
                line = process.stdout.readline()
                if line.rstrip('\n') != 'DONE\t'+ident:
                    raise RuntimeError('Teacher failed or job mismatch; inspect teacher.stderr.txt: '+repr(line))
                if shutil.disk_usage(args.out).free < args.minimum_free_gib*1024**3:
                    raise RuntimeError('Disk reserve reached')
                folder = capture_root/ident
                capture = json.loads((folder/'capture.json').read_text())
                if capture['version'] != 1 or capture['basis'] != 'identity' or len(capture['tensors']) != 187:
                    raise ValueError('Incomplete projection capture')
                raw_record = dict(id=ident, capture_sha256=sha(folder/'capture.json'), inputs=[])
                names = set()
                for tensor in capture['tensors']:
                    name, basis = diagonal.hf_name(tensor['name'])
                    names.add(name)
                    path = folder/(tensor['name']+'.f32')
                    x = np.fromfile(path, dtype='<f4').reshape(tensor['samples'], tensor['columns'])
                    if len(x) == 0 or x.shape[1]%128 or not np.isfinite(x).all():
                        raise ValueError('Invalid teacher activations')
                    raw_record['inputs'].append(dict(name=name, sha256=sha(path), samples=len(x), columns=x.shape[1]))
                    x = diagonal.rotate(x) if basis else x.astype(np.float64)
                    blocks = x.reshape(len(x), -1, 128).transpose(1, 0, 2)
                    h = blocks.transpose(0, 2, 1) @ blocks
                    d = np.sum(x*x, axis=0)
                    if name not in moments:
                        moments[name], diagonal_sums[name], counts[name], bases[name] = h, d, len(x), basis
                    else:
                        if bases[name] != basis or moments[name].shape != h.shape:
                            raise ValueError('Projection inventory changed')
                        moments[name] += h
                        diagonal_sums[name] += d
                        counts[name] += len(x)
                if names != set(moments):
                    raise ValueError('Projection inventory changed')
                info['captures'].append(raw_record)
                info['tensors'] = [dict(name=k, basis=bases[k], samples=counts[k]) for k in sorted(moments)]
                if index in snapshots:
                    target = args.out/f'n{index}'
                    save_moments(target/'covariance', moments, info)
                    save_diagonal(target/'importance', diagonal_sums, counts, bases,
                                  dict(corpus_sha256=info['corpus_sha256'], requests=index))
                elif index % 64 == 0:
                    save_moments(args.out/'checkpoint', moments, info)
                # This directory contains only the current job's generated raw captures.
                for tensor in capture['tensors']:
                    (folder/(tensor['name']+'.f32')).unlink()
                (folder/'capture.json').unlink()
                folder.rmdir()
                if index % 16 == 0 or index in snapshots:
                    temp = args.out/'manifest.tmp'
                    temp.write_text(json.dumps(info, indent=2)+'\n')
                    temp.replace(args.out/'manifest.json')
                process.stdin.write(ident+'\n')
                process.stdin.flush()
                print(f'Compacted {index}/{len(jobs)+resumed} ({ident})', flush=True)
            if process.wait() != 0:
                raise RuntimeError('Teacher process failed')
            capture_root.rmdir()
        finally:
            if process.poll() is None:
                process.kill()
                process.wait()
            process.stdin.close()
            process.stdout.close()
    print('Completed all calibration snapshots', flush=True)


if __name__ == '__main__':
    main()
