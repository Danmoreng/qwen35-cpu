#!/usr/bin/env python3
"""Aggregate completed sequential runs and token-level quality data; retain audit inputs."""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics
import sys
import zipfile

p = argparse.ArgumentParser()
p.add_argument('--root', type=Path, default=Path('benchmarks/comparison-2026-09-06'))
p.add_argument('--output', type=Path, default=Path('docs/results/2026-09-06'))
p.add_argument('--quality-only', action='store_true')
args = p.parse_args()
args.output.mkdir(parents=True, exist_ok=True)

def write_csv(name, rows):
    with (args.output / name).open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)

def percentile(values, p):
    values = sorted(values)
    index = (len(values) - 1) * p
    low, high = math.floor(index), math.ceil(index)
    return values[low] + (values[high] - values[low]) * (index - low)

quality = []
window_rows = []
teacher_values = None
for model in ['H128-Q4', 'Q4_0-pure', 'Q4_0', 'Q4_K_M', 'IQ4_XS']:
    positions = []
    for index in range(16):
        path = args.root / 'quality-results' / f'{index:02d}-{model}.csv'
        with path.open(newline='') as f:
            rows = list(csv.DictReader(f))
        assert len(rows) == 512, path
        positions.extend(rows)
        window_rows.append(dict(model=model, window=index,
            teacher_nll=statistics.mean(float(r['target_nll_teacher']) for r in rows),
            candidate_nll=statistics.mean(float(r['target_nll_candidate']) for r in rows),
            kl=statistics.mean(float(r['kld_teacher_candidate']) for r in rows)))
    teacher = [float(r['target_nll_teacher']) for r in positions]
    if teacher_values is None:
        teacher_values = teacher
    assert teacher == teacher_values, 'Teacher changed across candidate evaluations'
    nll = statistics.mean(float(r['target_nll_candidate']) for r in positions)
    kl = [float(r['kld_teacher_candidate']) for r in positions]
    assert all(math.isfinite(v) for v in kl + teacher) and min(kl) >= -1e-10
    quality.append(dict(model=model, scored_tokens=len(positions),
        teacher_perplexity=math.exp(statistics.mean(teacher)), perplexity=math.exp(nll),
        mean_target_nll=nll, mean_kl=statistics.mean(kl), p95_kl=percentile(kl, .95),
        top1_agreement=statistics.mean(r['top1_equal'] == 'True' for r in positions)))
write_csv('quality-summary.csv', quality)
write_csv('quality-windows.csv', window_rows)
if args.quality_only:
    print(json.dumps(quality, indent=2))
    sys.exit(0)

performance = []
inputs = json.loads((args.root / 'inputs.json').read_text())
binary_hashes = {Path(item['path']).name: item['sha256'].lower() for item in inputs['binaries']}
for group in ['single-ccd', 'batch-ccd', 'batch-physical', 'single-physical-ccd']:
    folder = args.root / ('speed-' + group)
    metadata = json.loads((folder / 'metadata.json').read_text(encoding='utf-8-sig'))
    assert metadata['runs'] == 3 and metadata['warmup_runs'] == 1
    for path, sha256 in metadata['binaries'].items():
        assert sha256.lower() == binary_hashes[path.replace('\\', '/').rsplit('/', 1)[-1]]
    assert (folder / 'results.csv').exists(), 'Runner did not complete'
    for index, case in enumerate(metadata['matrix']):
        model, batch, threads, prompt, output = case['name'].rsplit('-', 4)
        batch, threads, prompt, output = (int(v[1:]) for v in (batch, threads, prompt, output))
        profiles = [json.loads((folder / f'case-{index}-run-{run}.json').read_text(encoding='utf-8-sig')) for run in (1, 2, 3)]
        for data in profiles:
            assert data['cpu_batch'] == batch and not data['prefill_only']
            assert data['prompt_tokens'] == batch * prompt
            assert data['generated_tokens'] == batch * output
            assert data.get('decode_forwards', data.get('decode_forward_steps')) == batch * (output - 1)
            assert data['cpu_kv_cache'] == 'fp16'
            assert not data.get('quality_capture', False)
            assert data.get('greedy_lm_head_batches', 0) == 0
            assert data.get('cached_prefix_tokens', 0) == 0
        prefill = [d.get('prefill_forward_tokens_per_second', d['prefill_tokens_per_second']) for d in profiles]
        decode = [d['tokens_per_second'] for d in profiles]
        assert all(math.isfinite(v) and v > 0 for v in prefill + decode)
        performance.append(dict(model=model, batch=batch, threads=threads, prompt_tokens=prompt,
            output_tokens=output, affinity=metadata['affinity'], measured_runs=3,
            prefill_median=statistics.median(prefill), prefill_min=min(prefill), prefill_max=max(prefill),
            decode_median=statistics.median(decode), decode_min=min(decode), decode_max=max(decode)))
write_csv('performance-summary.csv', performance)
(args.output / 'summary.json').write_text(json.dumps(dict(quality=quality, performance=performance), indent=2) + '\n', newline='\n')

# Keep exact profiles, stdout/stderr, matrix commands and bounded per-token
# metrics. No weights, raw source articles, binary files or huge logit dumps.
files = list(args.root.glob('*-matrix.json')) + list(args.root.glob('performance-prompt-*.csv'))
files += [args.root / name for name in ['inputs.json', 'quality-inputs.json', 'quality-windows.json', 'machine.json']]
files += [args.root / name for name in ['llama-build.log', 'ctest.log', 'batch-validation.log',
          'input-record.log', 'quality-full.log', 'quality-smoke.log'] if (args.root / name).exists()]
for folder in args.root.glob('speed-*'):
    if folder.is_dir():
        files += [f for f in folder.iterdir() if f.is_file()]
files += list((args.root / 'quality-results').glob('*'))
for folder in args.root.glob('quality-[0-9][0-9]'):
    files += [f for f in folder.iterdir() if f.name != 'H128-Q4.log' and
              (f.suffix in ('.json', '.log') or f.name in ('prompt.csv', 'targets.csv'))]
files += [args.root / 'batch-validation/summary.json']
arithmetic = args.root / 'arithmetic-regression'
if (arithmetic / 'summary.json').exists():
    files += [f for f in arithmetic.iterdir() if f.suffix in ('.json', '.csv', '.log')]
    (args.output / 'arithmetic-summary.json').write_text((arithmetic / 'summary.json').read_text(), newline='\n')
with zipfile.ZipFile(args.output / 'raw-results.zip', 'w', compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
    archive.writestr('README.md', '# CPU comparison audit data\n\n'
        'See https://github.com/Danmoreng/qwen35-cpu/blob/codex/cpu-standalone/docs/comparison-2026-09-06.md '
        'for methodology and limitations. Timed runs are speed-* only; quality profiles include logit I/O '
        'and must not be used as speed measurements. Raw float32 logits were removed after scoring. '
        'Original PowerShell results.csv files use German decimal commas; JSON profiles and the '
        'published summary CSVs use locale-independent numbers.\n\n'
        'WikiText token-derived evaluation data: attribution to the WikiText dataset authors and '
        'Wikipedia contributors. The pinned dataset card lists CC BY-SA 3.0 and GFDL; those data '
        'licenses apply instead of the engine MIT license. Source: '
        'https://huggingface.co/datasets/Salesforce/wikitext/tree/b08601e04326c79dfdd32d625aee71d232d685c3\n')
    for path in sorted(set(files)):
        if not path.is_file():
            raise FileNotFoundError(path)
        archive.write(path, path.relative_to(args.root).as_posix())
(args.output / 'SHA256SUMS').write_text(''.join(
    f'{hashlib.sha256(path.read_bytes()).hexdigest()}  {path.name}\n'
    for path in sorted(args.output.iterdir()) if path.is_file() and path.name != 'SHA256SUMS'))
print(json.dumps(quality, indent=2))
print(f'Wrote {len(performance)} speed rows and raw-results.zip to {args.output}')
