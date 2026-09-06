#!/usr/bin/env python3
"""Quality/compatibility only. Performance uses benchmark-inference-seq.ps1."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

p = argparse.ArgumentParser()
p.add_argument('--out', type=Path, default=Path('benchmarks/native-q4km-2026-09-06/quality'))
p.add_argument('--windows', type=int, default=16)
p.add_argument('--start-window', type=int, default=0)
p.add_argument('--arithmetic', action='store_true')
p.add_argument('--quant', choices=['Q4_K_M','Q4_0-pure'], default='Q4_K_M')
a = p.parse_args()
native_label='native-'+a.quant
llama_label='llama-'+a.quant
a.out.mkdir(parents=True, exist_ok=True)
original = Path('benchmarks/comparison-2026-09-06')
manifest = json.loads((original/'quality-windows.json').read_text())
windows = manifest['windows'][a.start_window:a.windows]
if a.arithmetic:
    windows = [dict(window='arithmetic', folder=str(original/'arithmetic-regression'), scored_tokens=5)]
native = Path('build/qwen35_cpu.exe').resolve()
llama = Path('build-llama/bin/llama-fixed-cpu-bench.exe').resolve()
checkpoint = Path(f'models/llama-comparison/Qwen3.5-0.8B-{a.quant}.gguf')
# Python 3.10 compatibility.
def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''): h.update(block)
    return h.hexdigest()
metadata = dict(native_sha256=sha(native), llama_sha256=sha(llama),
                checkpoint_sha256=sha(checkpoint), windows=windows, commands=[],
                revision=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
                diff_sha256=hashlib.sha256(subprocess.check_output(['git','diff','HEAD'])).hexdigest())
def run(command, log):
    metadata['commands'].append(command)
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
            cmd=[str(native),'--model-dir','models/hf-download-test','--weights',str(checkpoint),
                 '--tokens-file',str(prompt),'--forced-tokens-file',str(targets),'--threads','8',
                 '--max-context','1024','--logits-out',str(dump)]
        else:
            model='models/llama-comparison/Qwen3.5-0.8B-BF16.gguf' if label=='BF16' else str(checkpoint)
            cmd=[str(llama),'--cpu-gguf',model,'--prompt-tokens-file',str(prompt),
                 '--forced-output-tokens',targets.read_text().strip(),'--cpu-threads','8',
                 '--max-context','1024','--max-new-tokens',str(window['scored_tokens']),
                 '--logits-out',str(dump),'--profile-json',str(dst/(label+'-profile.json'))]
        run(cmd,dst/(label+'.log'))
    for reference in ['BF16',llama_label]:
        stem=dst/(reference+'-vs-native')
        run([sys.executable,'scripts/compare-logit-dumps.py','--teacher',str(dst/(reference+'.logits')),
             '--candidate',str(dst/(native_label+'.logits')),'--json',str(stem.with_suffix('.json')),
             '--csv',str(stem.with_suffix('.csv'))],dst/(reference+'-compare.log'))
        result=json.loads(stem.with_suffix('.json').read_text())
        print(f"Window {name}, reference {reference}: native PPL={result['target_nll']['candidate_perplexity']:.6f}, KL={result['kld']['mean']:.8f}",flush=True)
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
print('Quality validation completed.',flush=True)
