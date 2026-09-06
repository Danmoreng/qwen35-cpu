#!/usr/bin/env python3
"""Archive the matched calibrated-H128 / llama.cpp README comparison."""
import csv
import hashlib
import json
from pathlib import Path
import statistics
import zipfile
from evaluation_common import sha

root = Path('benchmarks/readme-calibrated-2026-09-06')
out = Path('docs/results/readme-calibrated-2026-09-06')
out.mkdir(parents=True, exist_ok=True)
read = lambda p: json.loads(p.read_text(encoding='utf-8-sig'))
meta = read(root/'runs/metadata.json')
assert meta['runs'] == 3 and meta['warmup_runs'] == 1 and meta['affinity'] == 21845
rows = []
contracts = {}
for i, case in enumerate(meta['matrix']):
    model, batch, threads, prompt, output = case['name'].rsplit('-', 4)
    b, t, p, n = [int(x[1:]) for x in (batch, threads, prompt, output)]
    argument = lambda flag: case['args'][case['args'].index(flag)+1]
    assert t == 8 and argument('--max-context') == '16384'
    contract = (meta['inputs'][str(Path(argument('--prompt-tokens-file')).resolve())]['sha256'],
                argument('--forced-output-tokens'))
    assert contracts.setdefault((b,p,n), contract) == contract
    profiles = [read(root/f'runs/case-{i}-run-{r}.json') for r in (1, 2, 3)]
    for profile in profiles:
        assert profile['cpu_batch'] == b and profile['prompt_tokens'] == b*p
        assert profile['generated_tokens'] == b*n and profile['cpu_kv_cache'] == 'fp16'
        assert profile.get('decode_forwards', profile.get('decode_forward_steps')) == b*(n-1)
        assert not any(profile.get(k, False) for k in ('operation_instrumentation', 'quality_capture',
                                                      'cached_prefix_tokens', 'greedy_lm_head_batches'))
    row = dict(model=model, batch=b, threads=t, prompt_tokens=p, output_tokens=n)
    for label, key in [('prefill', 'prefill_forward_tokens_per_second'), ('decode', 'tokens_per_second')]:
        values = [profile.get(key, profile['prefill_tokens_per_second']) for profile in profiles]
        row.update({label+'_median': statistics.median(values), label+'_min': min(values), label+'_max': max(values)})
    rows.append(row)

old_root = Path('benchmarks/comparison-2026-09-06')
old_inputs = read(old_root/'quality-inputs.json')
cal = read(Path('benchmarks/plan-calibrated-full/commands.json'))
assert cal['teacher_sha256'] == old_inputs['models']['BF16']['sha256']
assert cal['checkpoint_sha256'] == '8c47dffa6cbc77e2af663a79012a6f847142d78479d9d7adcf28c5b300e2d83d'
native_case = next(c for c in meta['matrix'] if c['name'].startswith('calibrated-'))
native_file = Path(native_case['args'][native_case['args'].index('--cpu-q4-h128')+1]).resolve()
assert meta['inputs'][str(native_file)]['sha256'].lower() == cal['checkpoint_sha256']
assert len(cal['windows']) == 16
for w in cal['windows']:
    folder = Path(w['folder'])
    assert sha(folder/'prompt.csv') == w['prompt_sha256']
    assert sha(folder/'targets.csv') == w['targets_sha256']
    assert len(w['scoring_mask']) == 512 and all(w['scoring_mask'])
with zipfile.ZipFile('docs/results/2026-09-06/raw-results.zip') as archived_quality:
    for w in cal['windows']:
        for file, key in [('prompt.csv','prompt_sha256'), ('targets.csv','targets_sha256')]:
            saved = archived_quality.read(f"quality-{w['window']:02d}/{file}")
            assert hashlib.sha256(saved).hexdigest() == w[key]
for name in ('Q4_0-pure', 'Q4_0', 'Q4_K_M', 'IQ4_XS'):
    case = next(c for c in meta['matrix'] if c['name'].startswith(name+'-b'))
    file = str(Path(case['args'][case['args'].index('--cpu-gguf')+1]).resolve())
    assert meta['inputs'][file]['sha256'].lower() == old_inputs['models'][name]['sha256'].lower()
quality = []
with Path('docs/results/2026-09-06/quality-summary.csv').open() as stream:
    for row in csv.DictReader(stream):
        if row['model'] == 'H128-Q4':
            continue
        quality.append(dict(model=row['model'], scored_tokens=int(row['scored_tokens']),
            tensor_bytes=old_inputs['models'][row['model']]['tensor_bytes'],
            perplexity=float(row['perplexity']), mean_kl=float(row['mean_kl'])))
q = read(Path('benchmarks/plan-calibrated-full/summary.json'))
quality.insert(0, dict(model='calibrated', scored_tokens=q['positions'], tensor_bytes=424934656,
                      perplexity=q['candidate_ppl'], mean_kl=q['candidate_kl']))
assert all(r['scored_tokens'] == 8192 for r in quality)

def write_csv(name, rows):
    with (out/name).open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator='\n')
        writer.writeheader(); writer.writerows(rows)
write_csv('performance.csv', rows)
write_csv('quality.csv', quality)
files = list((root/'runs').glob('*')) + list(root.glob('*.json')) + list(root.glob('*.csv'))
with zipfile.ZipFile(out/'raw-results.zip', 'w', zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
    for file in sorted(files):
        assert file.is_file() and file.suffix not in ('.exe', '.dll', '.gguf', '.q35h', '.logits')
        archive.write(file, file.as_posix())
manifest = dict(provenance=read(root/'provenance.json'),
    quality_sources=['docs/results/2026-09-06/raw-results.zip',
                     'docs/results/implementation-plan-2026-09-06/raw-results.zip'],
    quality_contract='Same BF16 teacher hash, 16 article windows, 8192 positions; full scoring masks verified.',
    files={p.name: sha(p) for p in out.iterdir() if p.is_file() and p.name not in ('manifest.json','SHA256SUMS')})
(out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n',encoding='utf-8',newline='\n')
(out/'SHA256SUMS').write_text(''.join(f'{sha(p)}  {p.name}\n' for p in sorted(out.iterdir())
    if p.is_file() and p.name != 'SHA256SUMS'),encoding='utf-8',newline='\n')
print(json.dumps(dict(performance=rows, quality=quality),indent=2))
