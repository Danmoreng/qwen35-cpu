# Run after converting and auditing model-calibrated-q8-head.q35h.
# Output directories must be new. Never run another benchmark concurrently.
$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest
function Check-Exit([string]$stage) {
    if ($LASTEXITCODE -ne 0) { throw "$stage failed ($LASTEXITCODE)" }
}
$checkpoint = 'models/qwen3.5-0.8b/model-calibrated-q8-head.q35h'
$native = 'benchmarks/q8-tools/qwen35_cpu.exe'
python scripts/check-plan-arithmetic.py --native $native --checkpoint $checkpoint --label q8-head --out benchmarks/q8-head-arithmetic *> benchmarks/q8-head-arithmetic.log
Check-Exit 'Arithmetic'
& benchmarks/q8-tools/cpu_scheduler_model_test.exe models/hf-download-test $checkpoint --pages *> benchmarks/q8-head-scheduler.log
Check-Exit 'Scheduler'
& benchmarks/q8-tools/cpu_prefix_model_test.exe models/hf-download-test $checkpoint *> benchmarks/q8-head-prefix.log
Check-Exit 'Prefix cache'
Write-Output 'Q8 head functional checks passed; starting sequential speed measurements.'
& scripts/benchmark-inference-seq.ps1 -Matrix benchmarks/q8-head-speed-inputs/matrix.json -OutputDir benchmarks/q8-head-speed -Runs 3 -WarmupRuns 1 -Affinity 21845
Check-Exit 'Speed'
Write-Output 'Speed measurements complete; starting quality evaluation.'
python scripts/evaluate-native-gguf.py --native $native --checkpoint $checkpoint --reference-checkpoint models/llama-comparison/Qwen3.5-0.8B-Q4_K_M.gguf --label q8-head --manifest benchmarks/plan-heldout-v1/quality-windows.json --windows 6 --max-context 8192 --cache-dir benchmarks/plan-heldout-logit-cache --out benchmarks/q8-head-heldout
Check-Exit 'Held-out evaluation'
python scripts/quality-report.py benchmarks/q8-head-heldout *> benchmarks/q8-head-heldout-report.log
Check-Exit 'Held-out report'
python scripts/evaluate-native-gguf.py --native $native --checkpoint $checkpoint --reference-checkpoint models/llama-comparison/Qwen3.5-0.8B-Q4_K_M.gguf --label q8-head --windows 16 --max-context 1024 --cache-dir benchmarks/plan-logit-cache --out benchmarks/q8-head-full
Check-Exit 'Regression evaluation'
python scripts/quality-report.py benchmarks/q8-head-full *> benchmarks/q8-head-full-report.log
Check-Exit 'Regression report'
python scripts/summarize-q8-experiment.py --label q8-head
Check-Exit 'Summary'
