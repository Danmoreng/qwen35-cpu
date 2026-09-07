from pathlib import Path
import subprocess,sys,json
root=Path('benchmarks/quality-k-recheck-2026-09-07')
cmd=[sys.executable,'scripts/evaluate-native-gguf.py','--native','benchmarks/g16-calibrated-batch/tools-v3/qwen35_cpu.exe','--checkpoint','models/llama-comparison/Qwen3.5-0.8B-Q4_K_M.gguf','--manifest','benchmarks/comparison-2026-09-06/quality-windows.json','--windows','16','--max-context','1024','--cache-dir','benchmarks/plan-logit-cache','--out',str(root/'q4km-current-full')]
subprocess.run(cmd,check=True)
with (root/'q4km-current-full-report.log').open('w') as f: subprocess.run([sys.executable,'scripts/quality-report.py',str(root/'q4km-current-full')],stdout=f,check=True)
print('Q4_K_M quality complete. Starting sequential speed comparison.',flush=True)
subprocess.run(['pwsh','-File','scripts/benchmark-inference-seq.ps1','-Matrix',str(root/'speed-matrix.json'),'-OutputDir',str(root/'speed'),'-Runs','3','-WarmupRuns','1','-Affinity','21845','-SpeedTolerance','0.05'],check=True)
