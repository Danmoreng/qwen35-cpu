#!/usr/bin/env python3
"""Matched full-logit workloads for the implementation-plan candidates."""
import argparse
import json
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument('--out', type=Path, required=True)
parser.add_argument('--executable', default='benchmarks/plan-final-tools/qwen35_cpu_bench.exe')
args = parser.parse_args()
args.out.mkdir(parents=True, exist_ok=False)
tokens = [int(v) for v in Path('configs/qwen3_5_0_8b_text_1024x512_tokens.csv').read_text().strip().split(',')]
models = [('legacy', 'models/hf-download-test/model.q35h'),
          ('mse16', 'models/qwen3.5-0.8b/model-mse16.q35h'),
          ('calibrated', 'models/qwen3.5-0.8b/model-calibrated-mse16.q35h'),
          ('q4pure', 'models/llama-comparison/Qwen3.5-0.8B-Q4_0-pure.gguf'),
          ('q4km', 'models/llama-comparison/Qwen3.5-0.8B-Q4_K_M.gguf')]
cases = []
for prompt, output, batch in [(512,128,1), (4096,2,1), (512,128,4), (512,128,16), (8192,128,1)]:
    path = args.out/f'prompt-{prompt}.csv'
    path.write_text(','.join(str(tokens[i%len(tokens)]) for i in range(prompt)))
    continuation = ','.join(str(tokens[(prompt+i)%len(tokens)]) for i in range(output))
    for name, checkpoint in models:
        cases.append(dict(name=f'{name}-b{batch}-t8-p{prompt}-n{output}', executable=args.executable,
            args=['--hf-model-dir','models/hf-download-test','--cpu-q4-h128',checkpoint,
                  '--cpu-batch-full-logits','--prompt-tokens-file',str(path),
                  '--forced-output-tokens',continuation,'--cpu-threads','8','--cpu-batch',str(batch),
                  '--max-context','16384','--max-new-tokens',str(output)]))
(args.out/'matrix.json').write_text(json.dumps(cases,indent=2)+'\n')
