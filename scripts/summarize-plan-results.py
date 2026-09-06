#!/usr/bin/env python3
"""Publish compact, auditable plan measurements without weights, articles or logits."""
import argparse
import csv
import json
from pathlib import Path
import statistics
import zipfile
from evaluation_common import sha, source_identity

parser = argparse.ArgumentParser()
parser.add_argument('--benchmarks', type=Path, default=Path('benchmarks'))
parser.add_argument('--out', type=Path, default=Path('docs/results/implementation-plan-2026-09-06'))
args = parser.parse_args()
root, out = args.benchmarks, args.out
out.mkdir(parents=True, exist_ok=True)
read = lambda p: json.loads(p.read_text(encoding='utf-8-sig'))
models = ['legacy', 'mse16', 'calibrated', 'q4pure', 'q4km']
quality = {name: read(root/f'plan-{name}-full/summary.json') for name in models}
heldout = {name: read(root/f'plan-heldout-{name}/summary.json') for name in models if name != 'q4pure'}
baseline = quality['q4km']
reference_manifest = read(root/'plan-q4km-full/commands.json')
for name in models:
    m = read(root/f'plan-{name}-full/commands.json')
    if m['teacher_sha256'] != reference_manifest['teacher_sha256']:
        raise ValueError('Teacher identity mismatch')
    for a, b in zip(m['windows'], reference_manifest['windows']):
        if any(a[key] != b[key] for key in ('prompt_sha256', 'targets_sha256', 'scoring_mask')):
            raise ValueError('Unpaired evaluation windows')
    if quality[name]['positions'] != 8192 or len(m['windows']) != 16:
        raise ValueError('Incomplete original regression suite')
performance = root/'plan-performance'
metadata = read(performance/'metadata.json')
if metadata['runs'] != 3 or metadata['warmup_runs'] != 1 or metadata['affinity'] != 21845:
    raise ValueError('Unexpected benchmark contract')
rows = []
for index, case in enumerate(metadata['matrix']):
    name, batch, threads, prompt, generated = case['name'].rsplit('-', 4)
    batch, threads, prompt, generated = [int(value[1:]) for value in (batch, threads, prompt, generated)]
    profiles = [read(performance/f'case-{index}-run-{run}.json') for run in (1, 2, 3)]
    for p in profiles:
        if (p['cpu_batch'] != batch or p['prompt_tokens'] != prompt*batch or
            p['generated_tokens'] != generated*batch or p['decode_forwards'] != (generated-1)*batch or
            p['cpu_kv_cache'] != 'fp16' or p['greedy_lm_head_batches'] or p['cached_prefix_tokens'] or
            p.get('operation_instrumentation', False) or p.get('quality_capture', False)):
            raise ValueError('Mismatched work or instrumented performance run')
    row = dict(candidate=name, batch=batch, threads=threads, prompt_tokens=prompt, output_tokens=generated)
    for label, field in [('prefill', 'prefill_forward_tokens_per_second'), ('decode', 'tokens_per_second'),
                         ('prefill_ms', 'prefill_forward_time_ms'), ('decode_ms', 'decode_time_ms'),
                         ('peak_rss_bytes', 'peak_rss_bytes')]:
        values = [p[field] for p in profiles]
        row.update({label+'_median': statistics.median(values), label+'_min': min(values), label+'_max': max(values)})
    rows.append(row)
for row in rows:
    base = next(b for b in rows if b['candidate'] == 'legacy' and
                all(b[key] == row[key] for key in ['batch', 'threads', 'prompt_tokens', 'output_tokens']))
    row['prefill_ratio'] = row['prefill_median']/base['prefill_median']
    row['decode_ratio'] = row['decode_median']/base['decode_median']
    relevant = [row['prefill_ratio']] + ([row['decode_ratio']] if row['output_tokens'] > 2 else [])
    row['speed_point_pass'] = min(relevant) >= 1
    row['speed_equivalence_pass'] = min(relevant) >= 1-metadata['speed_gates']['equivalence_tolerance']
summary = []
arithmetic = read(root/'plan-arithmetic/summary.json')
arithmetic_by_name = {r['candidate']: r for r in arithmetic['results']}
followup = []
followup_metadata = read(root/'plan-performance-followup/metadata.json')
assert followup_metadata['runs'] == 6 and followup_metadata['warmup_runs'] == 1
for index, case in enumerate(followup_metadata['matrix']):
    name, batch, threads, prompt, generated = case['name'].rsplit('-', 4)
    profiles = [read(root/f'plan-performance-followup/case-{index}-run-{run}.json') for run in range(1, 7)]
    assert all(not p.get('operation_instrumentation', False) and not p['cached_prefix_tokens']
               and not p['greedy_lm_head_batches'] for p in profiles)
    followup.append(dict(candidate=name, workload='-'.join([batch, threads, prompt, generated]),
        prefill_median=statistics.median(p['prefill_forward_tokens_per_second'] for p in profiles),
        decode_median=statistics.median(p['tokens_per_second'] for p in profiles)))
for row in followup:
    base = next(b for b in followup if b['candidate'] == 'legacy' and b['workload'] == row['workload'])
    row['prefill_ratio'] = row['prefill_median']/base['prefill_median']
    row['decode_ratio'] = row['decode_median']/base['decode_median']
for name in models:
    q = quality[name]
    measured = [r for r in rows if r['candidate'] == name]
    row = dict(candidate=name, regression_ppl=q['candidate_ppl'], regression_kl=q['candidate_kl'],
        reference_ppl=baseline['reference_ppl'], reference_kl=baseline['reference_kl'],
        regression_ppl_ratio=q['candidate_ppl']/baseline['reference_ppl'],
        regression_strict_pass=q['candidate_ppl'] <= baseline['reference_ppl'] and q['candidate_kl'] <= baseline['reference_kl'],
        regression_engineering_pass=q['candidate_ppl'] <= baseline['reference_ppl']*1.01 and q['candidate_kl'] <= baseline['reference_kl']+.002,
        heldout_ppl=heldout[name]['candidate_ppl'] if name in heldout else None,
        heldout_kl=heldout[name]['candidate_kl'] if name in heldout else None,
        heldout_strict_pass=heldout[name]['strict_pass'] if name in heldout else False,
        heldout_engineering_pass=heldout[name]['engineering_pass'] if name in heldout else False,
        speed_point_pass=all(r['speed_point_pass'] for r in measured),
        speed_equivalence_pass=all(r['speed_equivalence_pass'] for r in measured),
        payload_bytes=521555200 if name == 'q4km' else 424934656)
    row['complete_success'] = row['regression_strict_pass'] and row['heldout_strict_pass'] and row['speed_point_pass']
    row['arithmetic_pass'] = (all(arithmetic_by_name[name][k]
        for k in ('free_pass', 'prefix_and_answer_pass')) if name in arithmetic_by_name else None)
    row['practical_quality_pass'] = name in heldout and all(
        suite[name][metric] < suite['legacy'][metric]
        for suite in (quality, heldout) for metric in ('candidate_ppl', 'candidate_kl'))
    row['followup_speed_pass'] = name in ('legacy', 'mse16', 'calibrated') and all(
        min(r['prefill_ratio'], r['decode_ratio']) >= .97 for r in followup if r['candidate'] == name)
    row['revised_user_goal_pass'] = bool(row['practical_quality_pass'] and row['arithmetic_pass']
        and row['speed_equivalence_pass'] and row['followup_speed_pass'])
    summary.append(row)
def write_csv(path, data):
    with path.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(data[0]), lineterminator='\n')
        writer.writeheader();writer.writerows(data)
write_csv(out/'quality-and-gates.csv', summary)
write_csv(out/'performance.csv', rows)
write_csv(out/'performance-followup.csv', followup)
(out/'results.json').write_text(json.dumps(dict(candidates=summary, performance=rows,
    followup=followup, arithmetic=arithmetic, quality=quality, heldout=heldout, gates=dict(strict_ppl_relative=0, strict_kl_absolute=0,
    engineering_ppl_relative=.01, engineering_kl_absolute=.002, speed=metadata['speed_gates']),
    scope='Measured first stages and calibration pilot; remaining plan stages are not claimed complete'), indent=2)+'\n', encoding='utf-8', newline='\n')
files = []
files += list((root/'plan-arithmetic').glob('*'))
for name in models:
    folder = root/f'plan-{name}-full'
    files += [p for p in folder.rglob('*') if p.is_file() and p.suffix in ('.json', '.csv', '.py')]
for name in heldout:
    files += [p for p in (root/f'plan-heldout-{name}').rglob('*') if p.is_file() and p.suffix in ('.json', '.csv', '.py')]
files += list(performance.glob('*')) + list((root/'plan-performance-inputs').glob('*'))
for extra in ('plan-performance-followup', 'plan-diagnostic-runs'):
    if (root/extra).exists():
        files += list((root/extra).glob('*'))
files += [root/'plan-inputs.json', root/'plan-calibration-pilot/manifest.json', root/'plan-calibration-fit/manifest.json',
          root/'plan-heldout-v1/quality-windows.json', root/'plan-ctest.log']
for name in ('plan-ctest-final.log', 'plan-calibration-rejection.log', 'plan-machine.json'):
    if (root/name).exists():
        files.append(root/name)
files += list(root.glob('plan-scheduler*.log')) + list(root.glob('plan-prefix-*.log'))
files += list((root/'plan-tools').glob('*pack.log'))
if (root/'plan-operation-profiles').exists():
    files += list((root/'plan-operation-profiles').glob('*.csv'))
if (root/'plan-attribution').exists():
    files += list((root/'plan-attribution').glob('*.json'))
with zipfile.ZipFile(out/'raw-results.zip', 'w', zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
    for path in sorted(set(files)):
        if not path.is_file() or path.suffix in ('.logits', '.gguf', '.q35h', '.safetensors', '.exe', '.dll', '.f32'):
            raise ValueError('Unexpected archival input '+str(path))
        archive.write(path, path.as_posix())
manifest = dict(source=source_identity(Path('.')), files={p.name: sha(p) for p in out.iterdir()
    if p.is_file() and p.name not in ('manifest.json', 'SHA256SUMS')},
    limitations=['No general quality certification from the six-document held-out screening suite.',
        'The calibration pilot covers four disjoint English prose documents; broader calibration remains open.',
        'Three speed measurements support a variability report, not strong confidence intervals.',
        'Tokenizer count and type checks do not prove external tokenizer identity; input hashes are recorded.',
        'Original baseline sweeps started before comparator snapshotting was added. A later comparator change '
        'only made top-k tie breaking deterministic and fixed overlap denominators for vocabularies below ten; '
        'NLL and KL formulas were unchanged. Subsequent sweeps archive their comparator snapshot.',
        'Mixed precision, reversible K packing, optimized K tiles and full release gates remain open.'])
(out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n', encoding='utf-8', newline='\n')
(out/'SHA256SUMS').write_text(''.join(f'{sha(p)}  {p.name}\n' for p in sorted(out.iterdir()) if p.is_file() and p.name!='SHA256SUMS'), encoding='utf-8', newline='\n')
print(json.dumps(summary,indent=2))
