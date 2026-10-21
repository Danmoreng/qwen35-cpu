# Prefill optimization: prepare inputs in the DeltaNet workers

The retained change moves linear-attention preparation into the existing
DeltaNet worker dispatch for chunks of at least 128 tokens. Each worker
calculates gates, normalizes Q/K and copies the inputs for its own heads before
running their recurrent scan. Shorter chunks keep serial preparation. No extra
pool dispatch, weight copy, quantization change or attention tile change is
introduced.

## Measured benefit and limits

The clearest result is **512-token prefill throughput with eight requests
processed serially in one process**. The retained version wins in all three
paired rounds: +3.7%, +8.4% and +3.2%; the ratio of medians is +7.0%.
This is not a batched matrix-projection result or a claim of 7% lower latency
for an isolated cold-process request.

| Prompt length | Serial requests per process | Baseline tokens/s | Retained tokens/s | Median change |
| ---: | ---: | ---: | ---: | ---: |
| 64 | 8 | 1,324.00 | 1,275.13 | -3.7% |
| 512 | 8 | 1,514.36 | 1,620.11 | +7.0% |
| 4,096 | 2 | 1,275.18 | 1,253.45 | -1.7% |

The short- and long-prompt paired results change sign between rounds; neither
supports an improvement claim. Their negative medians are retained above, not
hidden. All series ran in a noisy normal desktop session. In the retained
cold-process B1 series, P512 medians are 1,189.44 versus 1,603.04 tokens/s, but
individual baseline samples range from 891.13 to 1,537.17. P64 also has large
slow outliers in the retained binary. Those samples do not support a reliable
B1 latency claim or exclude all short-request latency regressions. A separate
instrumented P64 check changes the direction again and is diagnostic only.

The serial-request series uses one warmup and three measured runs, alternating
forward/reverse case order. Prompts are identical between requests, each with
one forced output token, no decode forwards and no prefix cache. The benchmark
always completes each request's prefill before starting the next; there are no
multi-request prefill matrix batches. Both implementations use the same request
count. Model loading and tokenization are excluded; request setup and the first
vocabulary projection are included.

A separate retained-binary B1 P512/N512 full-logit check (511 actual decode
forwards, one warmup and three measured runs) gives median decode throughput
**112.71 → 113.00 tokens/s (+0.26%)**. The small difference does not support
a decode speed claim; no decode kernel was changed. Both implementations use
the same forced output sequence. [Decode medians and ranges](results/prefill-optimization-2026-09-27/retained-decode.csv).

## Conditions and provenance

Ryzen 5 7600, Linux, GCC 16.2.1 Release, powersave governor, six physical cores
on CPUs 0–5 (`0x3f`), H128/Q4-G32-DOT4, FP16 KV, maximum context 16,384 and
unchanged automatic chunks. The baseline is `9a44970` with benchmark diagnostic
options added but disabled for throughput timing. Every timed series runs
sequentially through `scripts/benchmark-inference-seq.ps1`. No build or
validation from this task runs concurrently with timing; unrelated desktop
activity is not controlled. An external compiler was observed during an early
broad comparison. Frozen binaries and all inputs are hashed by the runner.

All screening series use one warmup and three measured runs. An intermediate
six-run series tested a separate-dispatch variant and is preserved, but its
samples are not combined with the retained implementation. Original case IDs
containing `greedy` in those intermediate series are misleading: forced output
tokens disable fused greedy, making them duplicate full-logit controls. Earlier
audit reports now explicitly correct that label. Actual unforced greedy
correctness checks are separate.

## Experiments and decision

1. **Q4 weight reuse:** a 16-token by 16-row intrinsic tile shares unpacked
   weights across twice as many tokens. The 5,040-case DOT4 suite passes. Initial
   median changes are only +0.5% at P512 and +0.7% at P4096, with noisy P512
   samples. The added tile was not retained.
2. **Full-attention tiling:** KV tiles of 64 and 128 were compared with the
   existing 32. Neither provides a consistent useful gain; the 128-token KV
   tile is slower in the long-prompt median. The existing policy remains.
   Persistent shared packed-K/V panels were not implemented in this bounded
   experiment.
3. **DeltaNet:** reordering the AVX2 scan for state cache locality gives no
   useful gain. A separate AVX-512 specialization retains two 128-column state
   rows in registers across a chunk and preserves the original eight-lane
   reduction order in each half of a ZMM register. It passes bitwise state/output
   checks but is about 2.1% slower at P4096. Both recurrent variants were removed.
   Parallel preparation shows a useful P512 benefit and is retained by fusing
   it into the existing worker dispatch.

A first preparation variant added a dispatch across tokens from eight tokens
onward. A conservative variant restricted this to 128 tokens after a noisy
short-prompt comparison. The retained version removes the additional dispatch
entirely and partitions preparation by the same heads as the recurrent scan.
Every variant has its own frozen binary/source evidence; none is overwritten.

All vectors are resized before dispatch. Convolution finishes before workers
read its output. Each worker owns disjoint heads in the convolution output and
head-major input arrays, and finishes preparation before consuming those heads'
inputs. Different workers have disjoint recurrent state. Per-head arithmetic
and floating-point reduction order remain unchanged. The profiling kernel label
`deltanet+prepare` indicates that its attention wall interval includes worker
preparation; comparing that interval to the old core alone would be misleading.

## Correctness

- All 14 configured CTests pass for the retained implementation. Optional
  NumPy evaluation tests are unavailable on this host.
- Baseline and retained binaries produce bitwise identical full-vocabulary
  dumps at prompt lengths 1, 4, 7, 8, 12, 31, 32, 63, 64, 65, 127, 128, 129, 511,
  512, 513 and 4,096, each with eight forced positions: **33,771,520 values**.
- Four unforced greedy completions with repetition penalty 1.05 match exactly.
- The retained implementation passes the real-model B4/B8/B12/B16/B20/B32
  full-vocabulary batch-versus-individual comparison bitwise.
- DeltaNet tests now cover production 128-column state, odd value-row tails,
  multiple token counts and disjoint head partitions. Available SIMD backends
  require bitwise batch-versus-sequential state/output identity.

This establishes implementation equivalence on the tested inputs, not a new
PPL/KL evaluation. Calibration, artifacts and published quality scores are
unchanged. Other CPUs and operating systems have not been measured.

## Evidence and reproduction

[Serial-request medians and ranges](results/prefill-optimization-2026-09-27/final.csv),
[cold-process B1 samples](results/prefill-optimization-2026-09-27/cold-b1.csv),
[initial screen](results/prefill-optimization-2026-09-27/screen.csv),
[register experiment](results/prefill-optimization-2026-09-27/register.csv), and
[provenance](results/prefill-optimization-2026-09-27/manifest.json).

Raw profiles, exact commands, hashes, source patches, frozen binaries, failed
experimental source and validation dumps remain under ignored
`benchmarks/prefill-opt-20260926/`. The directory date records the investigation's
start before midnight. Retained raw profiles and matrices also accompany the
report. No upstream inference engine is compared.

```sh
cmake --build build-perf-audit -j 6
ctest --test-dir build-perf-audit --output-on-failure
python3 benchmarks/prefill-opt-20260926/check-fused.py
pwsh -NoProfile -File scripts/benchmark-inference-seq.ps1 -Matrix benchmarks/prefill-opt-20260926/sustained-matrix.json -OutputDir benchmarks/prefill-serial-new -Runs 3 -WarmupRuns 1 -Affinity 63
```

On another checkout, use the recorded commands and matching model/token inputs
and point the matrix to independently built executables. Diagnostic options are
documented in the [preceding prefill audit](prefill-audit-2026-09-26.md).
