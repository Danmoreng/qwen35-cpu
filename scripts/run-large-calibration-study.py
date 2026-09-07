#!/usr/bin/env python3
"""Build and evaluate the frozen 40/256-document B and B+C calibration study."""
import argparse
import csv
import json
from pathlib import Path
import re
import statistics
import subprocess
import sys
from evaluation_common import sha, source_identity

ROOT = Path('benchmarks/bc-large-study')
CAPTURE = Path('benchmarks/bc-large-capture')
MODELS = Path('models/qwen3.5-0.8b')
VARIANTS = ['small-b', 'small-bc', 'full-b', 'full-bc']
BASELINE = MODELS / 'model-calibrated-mse16.q35h'
PACK = Path('benchmarks/bc-tools/qwen35_cpu_pack.exe')


def run(command, name):
    ROOT.mkdir(parents=True, exist_ok=True)
    command = [str(x) for x in command]
    (ROOT / (name + '-command.json')).write_text(json.dumps(command, indent=2) + '\n')
    with (ROOT / (name + '.log')).open('w') as log:
        subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)
    print('Completed:', name, flush=True)


def checkpoint(variant):
    if variant == 'baseline':
        return BASELINE
    if variant == 'old-c':
        return MODELS / 'model-calibrated-cov128.q35h'
    return MODELS / ('model-calibrated-large-' + variant + '.q35h')


def build():
    ROOT.mkdir(parents=True, exist_ok=True)
    (ROOT / 'build-provenance.json').write_text(json.dumps(dict(
        converter_sha256=sha(PACK), baseline_sha256=sha(BASELINE),
        source=source_identity(Path('.')), capture_manifest_sha256=sha(CAPTURE / 'manifest.json')),
        indent=2) + '\n')
    # Check the shared converter's default path against the original artifact.
    control = MODELS / 'model-calibrated-large-converter-control.q35h'
    run([PACK, '--hf-model-dir', MODELS, '--output', control,
         '--importance-dir', 'benchmarks/plan-calibration-fit'], 'converter-control')
    control_hash = sha(control)
    if control_hash != sha(BASELINE):
        raise ValueError('Default converter no longer reproduces the original artifact')
    (ROOT / 'converter-control.json').write_text(json.dumps(dict(sha256=control_hash,
        baseline_bit_identical=True), indent=2) + '\n')
    control.unlink()  # Verified duplicate, not an independent experiment artifact.
    for size in ['small', 'full']:
        importance = ROOT / (size + '-importance')
        covariance = ROOT / (size + '-covariance')
        run([sys.executable, 'scripts/fit-dot4-quant.py', CAPTURE / size,
             '--out', importance], size + '-importance')
        run([sys.executable, 'scripts/fit-block-covariance.py', CAPTURE / ('covariance-' + size),
             '--out', covariance], size + '-covariance')
        for method in ['b', 'bc']:
            variant = size + '-' + method
            command = [PACK, '--hf-model-dir', MODELS, '--output', checkpoint(variant),
                       '--importance-dir', importance]
            if method == 'bc':
                command += ['--covariance-dir', covariance]
            run(command, variant + '-convert')
            run([sys.executable, 'scripts/audit-g32-fit-artifact.py', checkpoint(variant),
                 '--out', ROOT / (variant + '-layout.json')], variant + '-layout')


def evaluate(suite, variants):
    settings = {
        'development': ('benchmarks/plan-heldout-v1/quality-windows.json', 6, 8192,
                        'benchmarks/plan-heldout-logit-cache', 'benchmarks/plan-heldout-calibrated'),
        'full': ('benchmarks/comparison-2026-09-06/quality-windows.json', 16, 1024,
                 'benchmarks/plan-logit-cache', 'benchmarks/plan-calibrated-full'),
        'final': ('benchmarks/bc-large-corpus/final/quality-windows.json', 10, 1024,
                  'benchmarks/bc-final-cache', str(ROOT / 'baseline-final')),
    }
    manifest, windows, context, cache, baseline = settings[suite]
    for variant in variants:
        name = variant + '-' + suite
        out = ROOT / name
        run([sys.executable, 'scripts/evaluate-native-gguf.py',
             '--native', 'benchmarks/g16-calibrated-batch/tools-v3/qwen35_cpu.exe',
             '--checkpoint', checkpoint(variant), '--reference-checkpoint',
             'models/llama-comparison/Qwen3.5-0.8B-Q4_K_M.gguf', '--label', variant,
             '--manifest', manifest, '--windows', windows, '--max-context', context,
             '--cache-dir', cache, '--out', out], name)
        run([sys.executable, 'scripts/quality-report.py', out], name + '-report')
        if variant != 'baseline':
            run([sys.executable, 'scripts/compare-quality-pair.py', baseline, out,
                 '--out', ROOT / (name + '-paired.json')], name + '-paired')
    if all(variant in variants for variant in VARIANTS):
        for left, right in [('small-b', 'full-b'), ('small-bc', 'full-bc'),
                            ('small-b', 'small-bc'), ('full-b', 'full-bc')]:
            name = left + '-vs-' + right + '-' + suite
            run([sys.executable, 'scripts/compare-quality-pair.py',
                 ROOT / (left + '-' + suite), ROOT / (right + '-' + suite),
                 '--out', ROOT / (name + '.json')], name)


def compatibility(variants):
    tools = Path('benchmarks/g16-calibrated-batch/tools-v3')
    for variant in variants:
        for test in ['cpu_prefix_model_test', 'cpu_scheduler_model_test']:
            run([tools / (test + '.exe'), 'models/hf-download-test', checkpoint(variant)],
                variant + '-' + test)
        run([tools / 'cpu_scheduler_model_test.exe', 'models/hf-download-test',
             checkpoint(variant), '--pages'], variant + '-scheduler-pages')
        run([sys.executable, 'scripts/evaluate-native-gguf.py', '--native', tools / 'qwen35_cpu.exe',
             '--checkpoint', checkpoint(variant), '--reference-checkpoint',
             'models/llama-comparison/Qwen3.5-0.8B-Q4_K_M.gguf', '--label', variant,
             '--arithmetic', '--max-context', 128, '--cache-dir', 'benchmarks/bc-arithmetic-cache',
             '--out', ROOT / (variant + '-arithmetic')], variant + '-arithmetic')
        fixture = json.loads(Path('configs/arithmetic-regression.json').read_text())['cases'][0]
        answer = json.loads((ROOT / (variant + '-arithmetic') / 'arithmetic/answer.json').read_text())
        expected = fixture['expected_output_prefix'] + [fixture['expected_next_token']]
        if answer['post_penalty_argmax'] != expected:
            raise ValueError('Arithmetic regression failed: ' + variant)


def speed(variants):
    original = json.loads(Path('benchmarks/g16-calibrated-batch/batch-v3.json').read_text())
    cases = []
    for case in original:
        if not case['name'].startswith('calibrated-g32-'):
            continue
        for variant in ['baseline'] + [v for v in variants if v != 'baseline']:
            entry = dict(case, name=case['name'].replace('calibrated-g32', variant), args=list(case['args']))
            entry['args'][entry['args'].index('--cpu-q4-h128') + 1] = str(checkpoint(variant))
            cases.append(entry)
    matrix = ROOT / 'speed-matrix.json'
    matrix.write_text(json.dumps(cases, indent=2) + '\n')
    run(['pwsh', '-File', 'scripts/benchmark-inference-seq.ps1', '-Matrix', matrix,
         '-OutputDir', ROOT / 'speed', '-Runs', 3, '-WarmupRuns', 1, '-Affinity', 21845], 'speed')
    summarize_speed()


def summarize_speed():
    results = {}
    with (ROOT / 'speed/results.csv').open() as stream:
        for row in csv.DictReader(stream):
            if row['warmup'].lower() == 'true':
                continue
            variant, batch = re.fullmatch(r'(.+)-b(\d+)-t8-p512-n128', row['name']).groups()
            item = results.setdefault(batch, {}).setdefault(variant, dict(decode_runs=[], prefill_runs=[]))
            item['decode_runs'].append(float(row['tokens_per_second']))
            item['prefill_runs'].append(float(row['prefill_tokens_per_second']))
    for variants in results.values():
        for item in variants.values():
            if len(item['decode_runs']) != 3:
                raise ValueError('Expected three measured runs')
            for phase in ['decode', 'prefill']:
                item[phase + '_median'] = statistics.median(item[phase + '_runs'])
        for item in variants.values():
            for phase in ['decode', 'prefill']:
                item[phase + '_change_percent'] = 100 * (
                    item[phase + '_median'] / variants['baseline'][phase + '_median'] - 1)
    (ROOT / 'speed-summary.json').write_text(json.dumps(results, indent=2) + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['build', 'development', 'full', 'final', 'compatibility', 'speed', 'summarize-speed'])
    parser.add_argument('--variants', nargs='+', choices=VARIANTS + ['baseline', 'old-c'], default=VARIANTS)
    args = parser.parse_args()
    if args.stage == 'build':
        build()
    elif args.stage == 'compatibility':
        compatibility(args.variants)
    elif args.stage == 'speed':
        speed(args.variants)
    elif args.stage == 'summarize-speed':
        summarize_speed()
    else:
        evaluate(args.stage, args.variants)
