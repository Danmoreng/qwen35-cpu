#!/usr/bin/env python3
"""Quality/compatibility only. Performance uses benchmark-inference-seq.ps1."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import shutil
from evaluation_common import sha, source_identity, select_windows, window_folder

p = argparse.ArgumentParser()
p.add_argument('--out', type=Path, default=Path('benchmarks/native-q4km-2026-09-06/quality'))
p.add_argument('--windows', type=int, default=16)
p.add_argument('--start-window', type=int, default=0)
p.add_argument('--arithmetic', action='store_true')
p.add_argument('--quant', choices=['Q4_K_M','Q4_0-pure'], default='Q4_K_M')
p.add_argument('--manifest', type=Path, default=Path('benchmarks/comparison-2026-09-06/quality-windows.json'))
p.add_argument('--native', type=Path, default=Path('build/qwen35_cpu' + ('.exe' if os.name == 'nt' else '')))
p.add_argument('--llama', type=Path, default=Path('build-llama/bin/llama-fixed-cpu-bench' + ('.exe' if os.name == 'nt' else '')))
p.add_argument('--checkpoint', type=Path)
p.add_argument('--reference-checkpoint', type=Path)
p.add_argument('--teacher', type=Path, default=Path('models/llama-comparison/Qwen3.5-0.8B-BF16.gguf'))
p.add_argument('--model-dir', type=Path, default=Path('models/hf-download-test'))
p.add_argument('--threads', type=int, default=8)
p.add_argument('--max-context', type=int, default=1024)
p.add_argument('--cache-dir', type=Path, help='Content-addressed reference-logit cache; can require tens of GB')
p.add_argument('--ppl-tolerance', type=float, default=.01)
p.add_argument('--kl-tolerance', type=float, default=.002)
p.add_argument('--label', help='Candidate name; defaults to checkpoint stem for q35h files')
a = p.parse_args()
if a.ppl_tolerance < 0 or a.kl_tolerance < 0:
    p.error('Quality tolerances must be nonnegative')
if a.threads <= 0 or a.max_context <= 0:
    p.error('threads and max-context must be positive')
native_label='native-'+(a.label or (a.checkpoint.stem if a.checkpoint and a.checkpoint.suffix == '.q35h' else a.quant))
if any(c not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._-' for c in native_label):
    p.error('Candidate label contains unsafe filename characters')
llama_label='llama-'+a.quant

original = a.manifest.parent
manifest = json.loads(a.manifest.read_text())
windows = select_windows(manifest['windows'], a.start_window, a.windows)
if a.arithmetic:
    windows = [dict(window='arithmetic', folder=str(original/'arithmetic-regression'), scored_tokens=5)]
native = a.native.resolve()
llama = a.llama.resolve()
checkpoint = a.checkpoint or Path(f'models/llama-comparison/Qwen3.5-0.8B-{a.quant}.gguf')
reference_checkpoint = a.reference_checkpoint or checkpoint
for window in windows:
    folder = window_folder(window, a.manifest)
    window['folder'] = str(folder)
    prompt, targets = folder/'prompt.csv', folder/'targets.csv'
    target_ids = [int(x) for x in targets.read_text().strip().split(',')]
    prompt_ids = [int(x) for x in prompt.read_text().strip().split(',')]
    if len(target_ids) != window['scored_tokens'] or not target_ids or not prompt_ids:
        raise ValueError('Token count mismatch or empty input')
    if len(prompt_ids) + len(target_ids) > a.max_context:
        raise ValueError('Window exceeds max-context')
    window.update(prompt_sha256=sha(prompt), targets_sha256=sha(targets),
                  scoring_mask=[True] * len(target_ids))
metadata = dict(native_sha256=sha(native), llama_sha256=sha(llama),
                checkpoint_sha256=sha(checkpoint), reference_checkpoint_sha256=sha(reference_checkpoint),
                teacher_sha256=sha(a.teacher), manifest_sha256=sha(a.manifest),
                windows=windows, commands=[], **source_identity(Path(__file__).resolve().parents[1]),
                upstream=source_identity(Path('.cache/llama.cpp')),
                threads=a.threads, max_context=a.max_context,
                gates=dict(ppl_relative_tolerance=a.ppl_tolerance, kl_absolute_tolerance=a.kl_tolerance,
                           strict_ppl_relative_tolerance=0, strict_kl_absolute_tolerance=0),
                config_sha256=sha(a.model_dir/'config.json'),
                tokenizer_sha256=sha(a.model_dir/'tokenizer.json'))
a.out.mkdir(parents=True, exist_ok=False)
comparison_script = a.out/'compare-logit-dumps.py'
shutil.copyfile(Path(__file__).with_name('compare-logit-dumps.py'), comparison_script)
metadata['comparison_script_sha256'] = sha(comparison_script)
shutil.copyfile(Path(__file__), a.out/'evaluate-native-gguf.py')
shutil.copyfile(Path(__file__).with_name('evaluation_common.py'), a.out/'evaluation_common.py')
def run(command, log):
    metadata['commands'].append(command)
    if command[0] == str(native) and sha(native) != metadata['native_sha256']:
        raise RuntimeError('Native executable changed during evaluation')
    (a.out/'commands.json').write_text(json.dumps(metadata,indent=2)+'\n')
    with log.open('wb') as f:
        subprocess.run(command,stdout=f,stderr=subprocess.STDOUT,check=True,
                       env={**os.environ,'OPENBLAS_NUM_THREADS':'1','MKL_NUM_THREADS':'1'})
for window in windows:
    name=str(window['window'])
    folder=Path(window['folder'])
    dst=a.out/name;dst.mkdir(exist_ok=True)
    prompt=folder/'prompt.csv';targets=folder/'targets.csv'
    for label in ['BF16',llama_label,native_label]:
        dump=dst/(label+'.logits')
        if label.startswith('native'):
            cmd=[str(native),'--model-dir',str(a.model_dir),'--weights',str(checkpoint),
                 '--tokens-file',str(prompt),'--forced-tokens-file',str(targets),'--threads',str(a.threads),
                 '--max-context',str(a.max_context),'--logits-out',str(dump)]
        else:
            model=str(a.teacher if label=='BF16' else reference_checkpoint)
            cmd=[str(llama),'--cpu-gguf',model,'--prompt-tokens-file',str(prompt),
                 '--forced-output-tokens',targets.read_text().strip(),'--cpu-threads',str(a.threads),
                 '--max-context',str(a.max_context),'--max-new-tokens',str(window['scored_tokens']),
                 '--logits-out',str(dump),'--profile-json',str(dst/(label+'-profile.json'))]
        cache = None
        if a.cache_dir and not label.startswith('native'):
            cache_key = dict(binary=metadata['llama_sha256'], model=sha(model),
                             prompt=window['prompt_sha256'], targets=window['targets_sha256'],
                             threads=a.threads, context=a.max_context)
            key = hashlib.sha256(json.dumps(cache_key, sort_keys=True).encode()).hexdigest()
            a.cache_dir.mkdir(parents=True, exist_ok=True)
            cache = a.cache_dir/(key+'.logits')
        if cache and cache.exists() and cache.with_suffix('.sha256').exists():
            if sha(cache) != cache.with_suffix('.sha256').read_text().strip():
                raise ValueError('Reference logit cache checksum mismatch')
            shutil.copyfile(cache, dump)
            metadata.setdefault('cache_hits', []).append(dict(window=name, label=label, key=key))
        else:
            run(cmd,dst/(label+'.log'))
            if cache:
                shutil.copyfile(dump, cache)
                cache.with_suffix('.sha256').write_text(sha(cache)+'\n')
    for reference, candidate, suffix in [('BF16',native_label,'native'),(llama_label,native_label,'native'),('BF16',llama_label,'llama')]:
        stem=dst/(reference+'-vs-'+suffix)
        run([sys.executable,str(comparison_script),'--teacher',str(dst/(reference+'.logits')),
             '--candidate',str(dst/(candidate+'.logits')),'--json',str(stem.with_suffix('.json')),
             '--csv',str(stem.with_suffix('.csv'))],dst/(reference+'-vs-'+suffix+'-compare.log'))
        result=json.loads(stem.with_suffix('.json').read_text())
        print(f"Window {name}, reference {reference}: {candidate} PPL={result['target_nll']['candidate_perplexity']:.6f}, KL={result['kld']['mean']:.8f}",flush=True)
    if a.arithmetic:
        import numpy as np
        fixture=json.loads(Path('configs/arithmetic-regression.json').read_text())['cases'][0]
        dt=np.dtype([('target','<i4'),('logits','<f4',(248320,))])
        records=np.fromfile(dst/(native_label+'.logits'),dtype=dt,offset=24)
        choices=[];past=fixture['prompt_tokens'].copy()
        for rec in records:
            values=rec['logits'].copy();seen=list(set(past));penalty=np.float32(fixture['repetition_penalty'])
            values[seen]=np.where(values[seen]>0,values[seen]/penalty,values[seen]*penalty)
            choices.append(int(values.argmax()));past.append(int(rec['target']))
        (dst/'answer.json').write_text(json.dumps(dict(post_penalty_argmax=choices,
            margin_4_minus_2=float(values[19]-values[17])),indent=2)+'\n')
    for label in ['BF16',llama_label,native_label]:
        (dst/(label+'.logits')).unlink()
(a.out/'commands.json').write_text(json.dumps(metadata,indent=2)+'\n')
print('Quality validation completed.',flush=True)
