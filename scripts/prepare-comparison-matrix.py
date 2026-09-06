#!/usr/bin/env python3
"""Prepare fixed-token CPU comparison cases; execute with benchmark-inference-seq.ps1."""
import argparse
import json
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument('--output', type=Path, default=Path('benchmarks/comparison-2026-09-06'))
args = p.parse_args()
args.output.mkdir(parents=True, exist_ok=True)
fixture = Path('configs/qwen3_5_0_8b_text_1024x512_tokens.csv')
tokens = [int(v) for v in fixture.read_text().strip().split(',')]
models = ['H128-Q4', 'Q4_0-pure', 'Q4_0', 'Q4_K_M', 'IQ4_XS']

def matrix(batch, threads, workloads):
    cases = []
    for prompt_count, output_count in workloads:
        prompt = [tokens[i % len(tokens)] for i in range(prompt_count)]
        continuation = [tokens[(prompt_count + i) % len(tokens)] for i in range(output_count)]
        prompt_path = args.output / f'performance-prompt-{prompt_count}.csv'
        prompt_path.write_text(','.join(map(str, prompt)) + '\n')
        for thread_count in threads:
            for model in models:
                common = ['--prompt-tokens-file', str(prompt_path), '--forced-output-tokens',
                          ','.join(map(str, continuation)), '--cpu-threads', str(thread_count),
                          '--cpu-batch', str(batch), '--max-context', '8192',
                          '--max-new-tokens', str(output_count)]
                if model == 'H128-Q4':
                    executable = 'build/qwen35_cpu_bench.exe'
                    flags = ['--hf-model-dir', 'models/hf-download-test', '--cpu-q4-h128',
                             'models/hf-download-test/model.q35h', '--cpu-batch-full-logits']
                else:
                    executable = 'build-llama/bin/llama-fixed-cpu-bench.exe'
                    flags = ['--cpu-gguf', f'models/llama-comparison/Qwen3.5-0.8B-{model}.gguf']
                cases.append(dict(name=f'{model}-b{batch}-t{thread_count}-p{prompt_count}-n{output_count}',
                                  executable=executable, args=flags + common))
    return cases

workloads = [(n, 2) for n in (512, 1024, 2048, 4096)] + [(512, n) for n in (128, 256, 512, 1024)]
groups = {'single-ccd': matrix(1, [8, 12], workloads),
          'single-physical-ccd': matrix(1, [8], workloads[:4] + [(512, 128)]),
          'batch-ccd': matrix(4, [8], [(512, 128)]) + matrix(16, [8], [(512, 128)]),
          'batch-physical': matrix(4, [16], [(512, 128)]) + matrix(16, [16], [(512, 128)])}
for group, cases in groups.items():
    path = args.output / (group + '-matrix.json')
    path.write_text(json.dumps(cases, indent=2) + '\n')
    print(f'{path}: {len(cases)} cases')
