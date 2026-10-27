# qev AVX2 tuning and prefill attention boundaries

October 27, 2026, AMD EPYC 9V74 cloud VM. Baseline source:
`816beddaa698c15ef1d672ba2a461b2ec7d02c36`. Model:
`danmoreng/qev-0.8b-H128-Q4-G32-DOT4` at
`fc805be27e1409630b14b233ad7aac1a6dbb213f`.
The checkpoint, calibration and temperatures are unchanged.

## Larger chunks and numerical consistency

The engine automatically uses row attention for the first 128 tokens and tiled
attention afterwards. Previously, an explicit chunk larger than 128 could move
the first queries directly to tiled attention. A faster chunk setting therefore
did not necessarily preserve the evaluated numerical path.

This appeared in the existing eleven-option joint fixture: unrestricted chunk
512 changed the rounded score from **1.37 to 1.36**, along with probabilities.
The choice remained the same, but that was not accepted as a probability-preserving
speedup. The faster unrestricted-chunk trial is retained separately and is not
the source of the selected performance table below.

A common chunk-count helper now preserves the initial 128-token row-attention
interval, then finishes the requested canonical chunks. A 579-token prompt at
chunk 512 runs as **128 + 384 + 67**. Both normal requests and explicit prefix
construction use the helper; prefix restores finish the same boundaries.
Defaults retain their existing 64/128 scheduling. No SIMD kernel, model format,
weight or floating-point reduction formula is changed.

## Final paired measurement

Median HTTP milliseconds from three measured runs after one warmup process.
The baseline uses its existing adaptive chunks; selected uses the corrected
engine with **`--threads 4 --prefill-chunk 512`**, both on CPUs 0–3.

| Input tokens | Adaptive baseline | Corrected chunk 512 | Latency reduction |
| --- | ---: | ---: | ---: |
| 74 | 237.56 ms | 226.34 ms | 4.72% |
| 579 | 1,475.70 ms | 1,450.20 ms | 1.73% |
| 4,079 | 12,017.30 ms | 11,298.82 ms | **5.98%** |

The clearest gain is the long input: baseline samples span
11,894.83–12,155.41 ms and selected samples 11,222.07–11,347.92 ms. Short and
medium timing ranges overlap. Their median differences do not establish a
reliable general speedup. Three runs do not provide a population confidence
interval or a universal optimum for every workload or CPU.

These are **decision latencies**, not decode tokens/s. There is one four-option
`per_question` question per request, zero generated output tokens and zero
decode forwards. Each process starts a fresh server and times its first decision
after readiness. Loading is excluded; HTTP, tokenization, scheduling and response
formatting are included. Every timed response has zero prefix-cache credit.
The 4,079-token fixture repeats history synthetically to measure input scaling;
it is not an accuracy corpus.

Conditions: Debian 13.6, GCC 14.2.0 Release, native AVX2/FMA/F16C; four inference
threads, CPU affinity mask 15, four-CPU cgroup quota and 16 GiB memory limit.
The VM exposes five virtual CPUs. Its cache topology groups CPUs 0/1 and 2/3,
with CPU 4 in another reported cache group; this does not establish host physical
core topology. Compilation, validation and other benchmarks did not run during
the final timed series.

## Correctness

- All **20 CTests** pass, including canonical chunk and resumed-prefix boundaries.
- **41 API fixtures** match baseline answers, rounded probabilities, scores and
  usage exactly. They cover both encodings, choice/noul/score, up to 255 options
  and the three timing inputs. Sixteen more calls at concurrency four also match.
- The model-backed regression compares complete finite vocabulary logits at
  chunk 128 against 512/1024, with cold requests and registered prefixes ending
  at 127, 221 and 511. It passes bitwise with contiguous FP16 KV and shared FP16
  pages: **16 full-vocabulary comparisons** for the 579-token test input.

The API check compares rounded wire values. The separate model check establishes
bitwise equality for its tested input and prefixes, not every possible prompt.
No full task-accuracy, PPL or KL study was repeated. Windows, other CPUs and browser
performance were not measured here.

## Rejected trials and autotuning

Expanded Q4 weight scratch tiles, a lower linear preparation threshold, and an
AVX2 DeltaNet loop interchange did not establish a repeatable overall gain across
the tested shapes. They were restored. Three/five threads and alternative CPU
affinities did not establish an advantage over four threads on CPUs 0–3.
The four-segment SIMD loop was already fully unrolled by GCC; additional forced
unrolling was not measured in this series.

Chunk settings vary with input length as well as CPU. A future calibrator should
measure representative length/concurrency classes, validate allowed numerical
paths and cache its result by CPU/ISA/cache topology, quota, affinity, thread
count, model format/hash and engine binary/version. An explicit chunk setting
should override that profile. Native automatic calibration is not implemented
by this change; the retained local profile is a measured startup preset.

## Reproduction and evidence

Build the corrected source normally, then start:

```sh
taskset -c 0-3 ./build/qwen35_cpu_server \
  --model-dir models/qev-0.8b --threads 4 --prefill-chunk 512
```

**Use the corrected engine for this preset.** Applying unrestricted chunk 512 to
the old engine reproduces the numerical discrepancy above. The source revision
and file hashes in the [manifest](../results/qev-avx2-cloud-2026-10-27/manifest.json)
identify the correction.

The final matrix is retained verbatim in that manifest. From the repository root,
using PowerShell 7 and a new output directory:

```sh
taskset -c 0-3 pwsh -NoProfile -File scripts/benchmark-inference-seq.ps1 \
  -Matrix benchmarks/qev-tuning/boundary-confirmation-matrix.json \
  -OutputDir benchmarks/new-qev-confirmation -Runs 3 -WarmupRuns 1 -Affinity 15
```

Adapt local binary/model paths when moving machines. Keep the baseline executable
from the base revision. [Aggregates](../results/qev-avx2-cloud-2026-10-27/performance.csv),
[all final samples](../results/qev-avx2-cloud-2026-10-27/samples.csv),
[validation summary](../results/qev-avx2-cloud-2026-10-27/validation.json) and the
[manifest](../results/qev-avx2-cloud-2026-10-27/manifest.json) preserve compact evidence.
Raw profiles, source/binary snapshots, trial matrices, logs, the failed
unrestricted-chunk comparison and full validation responses remain under ignored
`benchmarks/qev-tuning/`.
