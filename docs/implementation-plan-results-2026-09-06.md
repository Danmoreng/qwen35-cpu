# Q4 quantization implementation and measurements — 2026-09-06

The revised practical objective is met on the measured workloads: both new H128
artifacts improve perplexity over Legacy, retain approximately the same speed,
and independently generate `4` for the existing German `2+2` regression.
The calibrated MSE16 artifact is the recommended experimental candidate for the
larger regression suite. Unweighted MSE16 is also useful: it needs no calibration
corpus and has slightly better perplexity on the small held-out screening suite.
The original plan's stronger Q4_K_M quality target at Legacy speed is **not met**.

## Quality

All candidates use the same prompts, targets and scoring masks within each suite.
Perplexity is computed from pooled token NLL, not an average of window perplexities.
KL is measured directly against the BF16 teacher, separately from native versus
llama.cpp same-checkpoint compatibility. Lower is better in both columns.

| Native candidate | Regression PPL (8,192 tokens) | Mean KL to BF16 | Held-out PPL (1,024 tokens) | Held-out mean KL |
|---|---:|---:|---:|---:|
| Legacy H128 | 18.58329 | 0.174841 | 5.16899 | 0.119210 |
| MSE16 H128 | 16.86128 | 0.105864 | **4.81958** | 0.075179 |
| Calibrated MSE16 H128 | **16.04087** | **0.096456** | 4.85143 | **0.071835** |
| Native pure Q4_0 | 18.03818 | 0.144650 | Not measured | Not measured |
| Native Q4_K_M | 14.78432 | 0.034708 | 4.65387 | 0.024812 |
| llama.cpp Q4_K_M reference | 14.75290 | 0.034693 | 4.65945 | 0.025163 |

Calibrated MSE16 reduces regression PPL by 13.68% and mean KL by 44.83%
relative to Legacy. Its held-out PPL decreases by 6.14%. MSE16 reduces regression
PPL by 9.27% and held-out PPL by 6.76%. These are measured corpus results, not
general model-quality guarantees. The six held-out documents cover German,
English, code, math, chat and a longer recurrent context, but remain a small
screening set. Calibration uses four disjoint English prose documents, excluding
evaluation source articles and normalized overlapping paragraphs. Broader
multidomain calibration and full covariance/GPTQ remain future work.

## Speed

Windows, AMD Ryzen 9 9955HX3D, eight threads, affinity mask 21845, FP16 KV,
full-vocabulary logits, fixed output tokens, maximum context 16,384. Each main
case has one warmup and three measured runs. All timed processes ran sequentially
through `scripts/benchmark-inference-seq.ps1`, without simultaneous quality work.
Numbers are medians in tokens/second: prefill forward / decode forward throughput.
Batch throughput is compared only against the same batch size.

| Batch / prompt / output tokens per request | Legacy | MSE16 | Calibrated MSE16 | Native Q4_K_M |
|---|---:|---:|---:|---:|
| 1 / 512 / 128 | 2147.90 / 117.79 | 2150.71 / 117.94 | 2150.69 / 118.35 | 369.82 / 87.31 |
| 1 / 4096 / 2 | 1880.17 / 81.07 | 1882.79 / 82.62 | 1884.19 / 81.13 | 362.09 / 66.38 |
| 4 / 512 / 128 | 2160.53 / 313.34 | 2172.82 / 315.35 | 2174.55 / 314.79 | 370.39 / 203.13 |
| 16 / 512 / 128 | 2157.96 / 605.28 | 2159.69 / 605.64 | 2170.38 / 603.26 | 369.14 / 209.42 |
| 1 / 8192 / 128 | 1624.06 / 65.47 | 1620.82 / 68.49 | 1621.23 / 68.36 | 350.50 / 57.48 |

The two-output-token case has only one decode forward per request; its decode
rate is descriptive and excluded from the speed gate. Prefill remains included.
Both new variants stay within the declared 3% slowdown tolerance on every
relevant main measurement. This is an engineering tolerance, not a statistical
proof of equivalence. Strict point estimates of zero slowdown do not all pass.

An additional, separate series used one warmup and **six** measured runs for
batch 16 and long-context decode, balancing forward/reverse execution order.
Calibrated decode was 598.76 versus Legacy 600.74 tok/s at batch 16 (−0.33%),
and 67.85 versus 67.50 tok/s at long context (+0.52%). MSE16 was 602.06 and
67.19 tok/s respectively. These paired follow-up results support essentially
unchanged speed; the apparent 4% long-context gain in the main series is not
treated as a reliable speed improvement. Full ranges and all five candidates
are available in the machine-readable tables and raw profiles.

## Arithmetic regression

Prompt: `Was ist 2 + 2? Antworte nur mit der Zahl.` using the existing rendered
chat token fixture. Temperature 0, repetition penalty 1.05, top-k 20, top-p 0.8,
seed 123, eight threads, context 128. Free generation has no forced output tokens.

| Candidate | Free answer after empty thinking block | Entire prefix + answer argmax matches | Logit margin, 4 minus 2 |
|---|---|---|---:|
| Legacy | `4` | Yes | 3.22558 |
| MSE16 | `4` | Yes | 6.87702 |
| Calibrated MSE16 | `4` | Yes | 6.70245 |

A separate forced-prefix logit inspection verifies every expected token after
applying the repetition penalty. The independent free-generation result is the
answer check; forcing `4` is not counted as evidence of arithmetic correctness.
This is one explicit regression, not an arithmetic benchmark suite.

## Implemented scope

- A deterministic offline MSE16 Q4 quantizer searches both scale signs and uses
  all codes in `[-8,7]`. It evaluates the final FP16 scale, refines scales by
  least squares, and retains the legacy candidate. Legacy remains the default
  and reproduces the original artifact byte for byte.
- Optional diagonal activation-weighted fitting uses actual BF16 projection
  inputs captured after normalization/nonlinearities. It transforms inputs in
  the same seeded H128 basis as the weights; the tied output head uses identity.
  All 187 projection inputs have versioned, validated calibration records.
  Weighted fitting also retains the unweighted candidate as a fallback.
- The runtime DOT4 layout and kernels are retained. All H128 files are
  424,964,864 bytes, including 424,934,656 bytes of tensor payload. Calibrated
  conversion reproduced identical bytes in a second run. Weights stay local.
- Quality tools now support explicit artifacts, validated window selection,
  token-weighted aggregation, paired gates, document bootstrap, content-addressed
  reference caches, input hashes and evaluator snapshots.
- GGUF loading validates the supported architecture, model dimensions, RoPE,
  recurrent/attention schedule, tied head and tokenizer metadata. Tokenizer
  type and count do not establish identity; external tokenizer hashes are saved.
- Optional operation profiling and captured-projection probes separate weight,
  activation and kernel numerical error for bounded examples. They are excluded
  from performance timing. The sampled kernel RMSE is around 1e-7 while sampled
  weight error is much larger. This is not a full attribution of model-level KL.

The original plan's mixed-precision search, reversible K packing, optimized K
SIMD tiles and complete release gates remain open. The current result reaches
the user's revised quality-at-equal-speed objective through offline quantization.

## Reproduction and validation

Build with the repository's normal CMake configuration. To create the unweighted
candidate from the local BF16 Hugging Face directory:

```powershell
build/qwen35_cpu_pack.exe --hf-model-dir models/qwen3.5-0.8b --output models/qwen3.5-0.8b/model-mse16.q35h --quantizer mse16
```

For the calibrated pilot, build the optional llama comparison collector with
the capture support, then use the local BF16 teacher and WikiText parquet:

```powershell
python scripts/collect-quant-calibration.py --out benchmarks/plan-calibration-pilot
python scripts/fit-dot4-quant.py benchmarks/plan-calibration-pilot --out benchmarks/plan-calibration-fit
build/qwen35_cpu_pack.exe --hf-model-dir models/qwen3.5-0.8b --output models/qwen3.5-0.8b/model-calibrated-mse16.q35h --quantizer mse16 --importance-dir benchmarks/plan-calibration-fit
python scripts/check-plan-arithmetic.py
```

The arithmetic script uses the frozen local `benchmarks/plan-final-tools`
executable from this experiment. Reproduction elsewhere requires that binary
location to be populated from the corresponding build. Calibration and benchmark
matrices record their actual commands rather than relying on these examples.

Validation completed locally:

- 14/14 CTests passed, including quantizer, basis, evaluator and GGUF metadata tests.
- Seven malformed calibration cases rejected cleanly without completed or partial artifacts.
- Scheduler, shared-page scheduler and prefix tests passed for all five artifacts
  (15 model test invocations).
- Legacy and calibrated conversion reproduced exact artifact hashes.
- The final benchmark runtime matched the earlier quality runtime exactly on
  a five-token logit fixture for Legacy, MSE16, pure Q4_0 and Q4_K_M.
- Free arithmetic generation and token-by-token regression passed for all three
  H128 candidates.

The source checkout starts from `d58f3df475498f895d2bf273667ec18e98915f8d`
with the accompanying uncommitted implementation. llama.cpp is pinned to
`73a43d1f69345aee8bb186ef4b3172cef892f2e5`. Frozen binary hashes, checkpoint
hashes, command lines, affinity, machine metadata and raw profiles are preserved
in [the result archive](results/implementation-plan-2026-09-06/raw-results.zip).
See [results.json](results/implementation-plan-2026-09-06/results.json),
[main performance](results/implementation-plan-2026-09-06/performance.csv),
[follow-up performance](results/implementation-plan-2026-09-06/performance-followup.csv)
and [hash manifest](results/implementation-plan-2026-09-06/manifest.json).

The earliest complete quality sweeps began before comparator snapshotting was
added. A subsequent comparator fix changed top-k tie handling and overlap
denominators for tiny vocabularies; NLL and KL formulas were unchanged. This
provenance limitation is retained in the manifest. No weights, executable build
outputs, full articles, captured activations or full logit dumps are published.
