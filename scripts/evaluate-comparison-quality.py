#!/usr/bin/env python3
"""Sequential teacher-forced quality evaluation. Never use its timings as speed data."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

p = argparse.ArgumentParser()
p.add_argument('--root', type=Path, default=Path('benchmarks/comparison-2026-09-06'))
p.add_argument('--threads', type=int, default=8)
p.add_argument('--windows', type=int, default=16)
p.add_argument('--start-window', type=int, default=0)
args = p.parse_args()
manifest = json.loads((args.root / 'quality-windows.json').read_text())
results = args.root / 'quality-results'
results.mkdir(exist_ok=True)
llama = str(Path('build-llama/bin/llama-fixed-cpu-bench.exe').resolve())
native = str(Path('build/qwen35_cpu.exe').resolve())
models = ['H128-Q4', 'Q4_0-pure', 'Q4_0', 'Q4_K_M', 'IQ4_XS']
command_file = results / 'commands.json'
commands = json.loads(command_file.read_text()) if command_file.exists() else []
if args.start_window == 0 and (args.root / 'inputs.json').exists():
    (args.root / 'quality-inputs.json').write_bytes((args.root / 'inputs.json').read_bytes())

def run(command, log):
    commands.append(command)
    with log.open('wb') as output:
        subprocess.run(command, stdout=output, stderr=subprocess.STDOUT, check=True,
                       env={**os.environ, 'OPENBLAS_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1'})


for window in manifest['windows'][args.start_window:args.windows]:
    index = window['window']
    folder = Path(window['folder'])
    teacher = folder / 'teacher.logits'
    candidate = folder / 'candidate.logits'
    targets = (folder / 'targets.csv').read_text().strip()
    common = ['--prompt-tokens-file', str(folder / 'prompt.csv'),
              '--forced-output-tokens', targets, '--max-new-tokens', str(window['scored_tokens']),
              '--cpu-threads', str(args.threads), '--max-context', '1024']
    run([llama, '--cpu-gguf', 'models/llama-comparison/Qwen3.5-0.8B-BF16.gguf',
         *common, '--logits-out', str(teacher), '--profile-json', str(folder / 'teacher-profile.json')],
        folder / 'teacher.log')
    for model in models:
        if model == 'H128-Q4':
            command = [native, '--model-dir', 'models/hf-download-test',
                       '--weights', 'models/hf-download-test/model.q35h',
                       '--tokens-file', str(folder / 'prompt.csv'), '--forced-tokens-file', str(folder / 'targets.csv'),
                       '--threads', str(args.threads), '--max-context', '1024', '--logits-out', str(candidate)]
        else:
            command = [llama, '--cpu-gguf', f'models/llama-comparison/Qwen3.5-0.8B-{model}.gguf',
                       *common, '--logits-out', str(candidate), '--profile-json', str(folder / (model + '-profile.json'))]
        run(command, folder / (model + '.log'))
        name = f'{index:02d}-{model}'
        csv_path, json_path = results / (name + '.csv'), results / (name + '.json')
        run([sys.executable, 'scripts/compare-logit-dumps.py', '--teacher', str(teacher),
             '--candidate', str(candidate), '--csv', str(csv_path), '--json', str(json_path)],
            folder / (model + '-compare.log'))
        summary = json.loads(json_path.read_text())
        print(f"Window {index:02d} {model}: PPL={summary['target_nll']['candidate_perplexity']:.4f}, "
              f"KL={summary['kld']['mean']:.6f}", flush=True)
        # Delete only this script's disposable, explicitly named logit dump.
        candidate.unlink()
    teacher.unlink()
    command_file.write_text(json.dumps(commands, indent=2) + '\n')
print('Quality evaluation completed', flush=True)
