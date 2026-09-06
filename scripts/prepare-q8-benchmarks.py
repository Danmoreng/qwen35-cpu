#!/usr/bin/env python3
"""Prepare paired Q8 experiments; run only with the sequential PowerShell runner."""
import argparse
import json
from pathlib import Path

p=argparse.ArgumentParser()
p.add_argument('--candidate',type=Path,required=True)
p.add_argument('--label',required=True)
p.add_argument('--out',type=Path,required=True)
args=p.parse_args()
args.out.mkdir(parents=True,exist_ok=False)
tokens=[int(x) for x in Path('configs/qwen3_5_0_8b_text_1024x512_tokens.csv').read_text().strip().split(',')]
models=[('calibrated',Path('models/qwen3.5-0.8b/model-calibrated-mse16.q35h')),(args.label,args.candidate)]
cases=[]
for batch,prompt,output in [(1,512,128),(1,4096,2),(4,512,128),(16,512,128),(1,8192,128)]:
    file=args.out/f'prompt-{prompt}.csv'
    file.write_text(','.join(str(tokens[i%len(tokens)]) for i in range(prompt)))
    for name,checkpoint in models:
        cases.append(dict(name=f'{name}-b{batch}-t8-p{prompt}-n{output}',
            executable='benchmarks/q8-tools/qwen35_cpu_bench.exe',args=[
                '--hf-model-dir','models/hf-download-test','--cpu-q4-h128',str(checkpoint),
                '--cpu-batch-full-logits','--prompt-tokens-file',str(file),
                '--forced-output-tokens',','.join(str(tokens[(prompt+i)%len(tokens)]) for i in range(output)),
                '--cpu-threads','8','--cpu-batch',str(batch),'--max-context','16384',
                '--max-new-tokens',str(output)]))
(args.out/'matrix.json').write_text(json.dumps(cases,indent=2)+'\n')
(args.out/'contract.json').write_text(json.dumps(dict(
    baseline='calibrated',candidate=args.label,quality='PPL and KL improve on regression and heldout screening',
    speed_equivalence_tolerance=.03,additional_measurements_if_borderline=True,
    experiment_scope=('Explicitly requested head quality/speed experiment; gates remain Q4'
                      if args.label=='q8-head' else 'Selective mixed-precision experiment'),
    promotion='Measure tradeoffs; do not automatically replace the published standard'),indent=2)+'\n')
