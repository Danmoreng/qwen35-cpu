# Reproduce the transfer comparison

Run from the repository root. Use the [validated Julia-2 GPU evaluator and checkpoint](../julia2-jev-skills-2026-11-22/reproduce.md), the [Qev BF16 reference and 255 letter IDs](../../benchmarks/decision-quality.md), the complete [NVIDIA When2Call MCQ file and existing Qev predictions](../../benchmarks/when2call.md) and the original Jev Skills items. The Qev GPU teacher is built against the same llama.cpp revision as the Julia evaluator. The Julia Python environment needs the pinned publisher runtime and its dependencies. The Qev environment needs decider-ai 1.9.0, tokenizers and NumPy.

From Bash, set JULIA_PY and QEV_PY to those Python executables.

```bash
set -euo pipefail
raw=benchmarks/julia2-transfer
julia=benchmarks/julia2-jev-skills
study=docs/results/julia2-transfer-2026-11-22
mkdir -p "$raw"
export PYTHONPATH="$julia/model/runtime"
"$JULIA_PY" "$study/when2call.py"
"$QEV_PY" "$study/german.py"
"$QEV_PY" "$study/encode-german.py" qev
"$JULIA_PY" "$study/encode-german.py" julia
"$julia/readout" "$julia/julia-text-f32.gguf" "$julia/model/text/readout.f32" \
    < "$raw/tokens.txt" > "$raw/logits.tsv" 2> "$raw/gpu.log"
"$julia/readout" "$julia/julia-text-f32.gguf" "$julia/model/text/readout.f32" \
    < "$raw/julia-de-jobs.txt" > "$raw/julia-de.tsv" 2> "$raw/julia-de.log"
build-qev-cuda-current/bin/qev-teacher-cuda \
    --weights benchmarks/qev08-ab-2026-11-20/lab-ref/qev-0.8b-bf16.gguf \
    --jobs-file "$raw/qev-de-jobs.txt" --letters-file benchmarks/qev-calibration-2026-10-21/letters.txt \
    --output "$raw/qev-de.tsv" --threads 4 --max-context 16384 > "$raw/qev-de.log" 2>&1
"$QEV_PY" "$study/score.py"
```

Both models receive the same state, instructions and options. Julia uses its released prompt, tokenizer and original FP32 readout. Qev uses its released option encoding and BF16 reference weights. The German inputs contain opaque IDs for scoring. The separate answer key stays out of both inference paths. The bootstrap draws 2,000 paired samples of source IDs for When2Call, domains for German routing, and ticket pairs for the generated German panel.
