# Bounded release and comparison plan

## 1. Independent CPU extraction — complete

CPU-only CMake build, H128/Q4-G32-DOT4 loading/packing, native tokenizer and CLI,
multi-request C++ engine, kernel/model tests, sequential benchmark runner and
full-logit export. No upstream submodule or model weights are shipped.

## 2. Freeze comparable quantization inputs — pending

Use one pinned Qwen3.5-0.8B BF16 checkpoint and record every source shard hash,
model revision, configuration and tokenizer hash. Exclude MTP/draft tensors from
both formats. Pin the llama.cpp commit at measurement time and record compiler,
build flags, CPU, RAM, affinity and power configuration.

Generate candidates from that exact source:

| Engine | Recipe | Purpose |
| --- | --- | --- |
| This project | H128/Q4-G32-DOT4 | Specialized candidate |
| llama.cpp | Q4_0, pure | Uniform baseline |
| llama.cpp | Q4_K_M | Mixed K-quant baseline |
| llama.cpp | IQ4_XS | Importance-aware family baseline |
| llama.cpp | BF16 | Common quality teacher |

Verify recipe availability in the pinned quantizer. If importance matrices are
used or required, record their calibration corpus and generation command and
keep it disjoint from the evaluation corpus. Do not silently substitute recipes.
Report file bytes, actual quantization recipe and memory alongside quality.

The optional adapter in `tools/llama-comparison` accepts an external checkout:

```sh
cmake -S tools/llama-comparison -B build-llama -G Ninja -DCMAKE_BUILD_TYPE=Release -DLLAMA_SOURCE_DIR=/absolute/path/to/llama.cpp
cmake --build build-llama --target llama-fixed-cpu-bench llama-quantize
```

On Windows, use a Visual Studio developer shell. This adapter was inherited from
the historical pinned version; compilation against a fresh upstream revision
must be verified before use. It is single-sequence only today.

## 3. Quality evaluation — pending

Freeze a licensed held-out text corpus spanning prose, code and multilingual
text. Record corpus hash, exact document boundaries, windows, stride, warmup
prefix and scored-token count. Tokenize once and feed identical IDs to all engines.
Do not use the repeated speed fixture as a representative perplexity corpus.

Teacher-force the same real target tokens, with no sampling penalties or EOS
truncation. Export raw full-vocabulary logits for the BF16 teacher and each
candidate. Score only identical positions with identical preceding context.
For each candidate report `exp(mean(target NLL))`, mean/p95
`KL(p_BF16 || p_candidate)`, top-1 agreement and scored-token count. Compare all
candidates to the same teacher; engine arithmetic differences remain part of
this end-to-end measurement. Do not average per-window perplexities directly.

The native CLI exports the same little-endian `Q35LGT1` format as the llama adapter:

```powershell
./build/qwen35_cpu.exe --model-dir models/qwen3.5-0.8b --weights models/qwen3.5-0.8b/model.q35h --tokens-file prompt.csv --forced-tokens-file targets.csv --logits-out candidate.logits
python scripts/compare-logit-dumps.py --teacher teacher.logits --candidate candidate.logits --csv positions.csv --json quality.json
```

The comparison script requires NumPy. Raw logits cost about 0.99 MB per scored
token per model; evaluate in bounded windows, aggregate token-level statistics
and remove disposable dumps after retaining the reproducible results. Logit
export is never included in speed timing. Self-comparison is only a plumbing
test and does not validate model quality.

## 4. Matched speed matrix — pending

Start with P=512/1024/2048/4096 and N=128/256/512/1024, B=1. Use the same
forced continuation and full vocabulary-head work for kernel comparison.
Distinguish head-free prefill, time to first token and N-1 decode forwards.
Align timer boundaries before comparing different harnesses.

Then extend the llama adapter to independent sequence IDs and test B=4/16/32,
the same admission/arrival pattern, resident limit, context lengths and output
count. Measure cold prefixes and warm shared prefixes separately, including
prefix construction when reporting cold total throughput. Report delivered
tokens/s, TTFT, inter-token latency, total latency and peak memory.

Sweep thread counts fairly for both engines, with matched CPU affinity. Show
both equal-thread comparisons and separately tuned configurations. Run one
benchmark process at a time, three measurements after one warmup, alternating
engine order. Retain command lines, raw profiles, CSV, binary/model hashes and
commit IDs. Do not divide aggregate batch throughput by a single-request
llama.cpp result to advertise a speedup.

The runner consumes a JSON array of cases (paths are relative to the caller):

```json
[
  {
    "name": "h128-b1",
    "executable": "build/qwen35_cpu_bench.exe",
    "args": ["--hf-model-dir", "models/qwen3.5-0.8b", "--cpu-q4-h128", "models/qwen3.5-0.8b/model.q35h", "--prompt-tokens-file", "prompt.csv", "--cpu-threads", "8", "--cpu-batch", "1", "--max-new-tokens", "128", "--max-context", "8192", "--temperature", "0", "--repeat-penalty", "1"]
  }
]
```

```powershell
./scripts/benchmark-inference-seq.ps1 -Matrix matrix.json -Runs 3 -WarmupRuns 1
```

Use `--forced-output-tokens` with a CSV string for matched fixed-token speed
comparisons. The runner appends `--profile-json`; do not supply that flag in cases.
The example above is a greedy engine-only run, not a complete llama comparison.

## 5. Publication gates — pending

- Validate this standalone tree on Linux/GCC and the Intel i7-8750H AVX2 machine.
- Initial bounded native HTTP completions adapter implemented and tested on Windows;
  validate the release pipeline and model download on both target platforms.
- Publish fresh quality/speed results with exact scope and reproducible inputs.
- Choose the public repository name and publish the prepared local repository.

Grouped prefix attention, mixed prefill/decode projection batches and H256 are
deferred. They are not required to finish the first focused release.
