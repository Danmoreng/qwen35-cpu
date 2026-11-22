# Reproduce the Julia-2 text quality run

Run from the qwen35-cpu repository root. Use the [frozen Jev Skills final and development panels and Qev comparison outputs](../../benchmarks/decision-quality.md#frozen-final-results). The scoring script reads the retained Qev BF16 GPU and native H128 logits from the original calibration study, recomputes their upstream scores, and checks them against the original reports.

The commands use Bash, a CUDA-enabled build of llama.cpp at `9b56ccf00d755b0bf780140f6352a67db1ac0378`, a C++17 compiler and the CUDA toolkit. Set `llama` to the llama.cpp source checkout and `lib` to its built shared-library directory. Use a Python environment with the llama.cpp conversion dependencies, the publisher's runtime dependencies, NumPy and threadpoolctl.

```bash
set -euo pipefail
raw=benchmarks/julia2-jev-skills
study=docs/results/julia2-jev-skills-2026-11-22
mkdir -p "$raw"
hf download SupersonicLabs/Julia-2 --revision b3e610b47d140cc4fcbd00d7b58b86184a13bcff \
    --local-dir "$raw/model"
python "$study/extract-text.py" "$raw/model"
cp "$raw/model/tokenizer/"{tokenizer.json,tokenizer_config.json,merges.txt,vocab.json} "$raw/model/text/"
python "$llama/convert_hf_to_gguf.py" "$raw/model/text" --outtype f32 --outfile "$raw/julia-text-f32.gguf"
for split in final development; do
    PYTHONPATH="$raw/model/runtime" python "$study/prepare.py" --panel "$split"
done

g++ -std=c++17 -O2 -I"$llama/include" -I"$llama/src" -I"$llama/ggml/include" \
    -I/opt/cuda/include "$study/readout.cpp" -L"$lib" -L/opt/cuda/lib64 \
    -Wl,-rpath,"$lib:/opt/cuda/lib64" -lllama -lggml -lggml-base -lcudart -lcublas -o "$raw/readout"
for split in final development; do
    "$raw/readout" "$raw/julia-text-f32.gguf" "$raw/model/text/readout.f32" \
        < "$raw/$split-tokens.txt" > "$raw/$split-logits.tsv" 2> "$raw/$split.log"
done
OPENBLAS_NUM_THREADS=2 python "$study/score.py"
```

Each input line holds a row ID and Julia token IDs. Each output line holds its ID and 255 pre-softmax FP32 logits. The reader explicitly resets llama.cpp memory between questions. In this pinned Qwen3.5 implementation, the masked `h_nextn` tensor is the **normalized** final hidden state. The GPU readout applies the original row-major weight matrix and bias. The final scoring stage maps fixed codes back to each question's options, with Julia's true/false ordering for `noul`. Gold labels enter only in the scorer.
