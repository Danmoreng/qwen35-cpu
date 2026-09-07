from pathlib import Path
import subprocess,sys,json
root=Path('benchmarks/quality-k-recheck-2026-09-07')
native='benchmarks/g16-calibrated-batch/tools-v3/qwen35_cpu.exe'
for suite,manifest,windows,context,cache in [('full','benchmarks/comparison-2026-09-06/quality-windows.json',16,1024,'benchmarks/plan-logit-cache'),('audit','benchmarks/head-audit-v1/quality-windows.json',6,8192,'benchmarks/head-audit-cache')]:
 for label,checkpoint in [('calibrated-g32','models/qwen3.5-0.8b/model-calibrated-mse16.q35h'),('calibrated-g16','models/qwen3.5-0.8b/model-calibrated-g16-head.q35h')]:
  out=root/f'{label}-{suite}'
  cmd=[sys.executable,'scripts/evaluate-native-gguf.py','--native',native,'--checkpoint',checkpoint,'--reference-checkpoint','models/llama-comparison/Qwen3.5-0.8B-Q4_K_M.gguf','--label',label,'--manifest',manifest,'--windows',str(windows),'--max-context',str(context),'--cache-dir',cache,'--out',str(out)]
  print('Starting',str(out),flush=True)
  subprocess.run(cmd,check=True)
  with (root/f'{label}-{suite}-report.log').open('w') as f: subprocess.run([sys.executable,'scripts/quality-report.py',str(out)],stdout=f,check=True)
print('All calibrated quality evaluations completed.',flush=True)
