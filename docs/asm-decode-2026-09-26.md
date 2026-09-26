# Single-request DOT4 assembly experiment

Handwritten AVX-512/VNNI assembly did **not establish a useful, repeatable
single-request decode improvement** over the optimized 16-row intrinsic kernel
from commit `9a44970`. Keep that existing kernel as the production default.
The experimental source and build options were removed after the investigation;
the existing intrinsic kernel remains in production. Local source snapshots and
the patch are preserved with the raw evidence.

## Matched B1 measurements

Ryzen 5 7600, six physical cores pinned to CPUs 0–5 (`0x3f`), Linux, GCC 16.2.1
Release, existing `powersave` governor and normal desktop session. All cases use
the same H128/Q4-G32-DOT4 checkpoint, 512 fixed prompt tokens, 128 fixed output
tokens, FP16 KV, full vocabulary, 16,384 maximum context and 128-token prefill
chunks. Decode counts 127 forwards. Loading, tokenization and HTTP are excluded.

All performance measurements run sequentially through
`scripts/benchmark-inference-seq.ps1`. There is no concurrent compilation or
correctness validation during timing. Case order alternates between rounds.

The initial series has one warmup and three measured runs. Values are median
decode tokens/s. Correction (September 27): cases originally labeled fused
greedy supplied forced tokens and therefore execute full logits. The second
column is a duplicate full-logit control; this report establishes no fused-greedy
speed result. Unforced greedy correctness checks remain valid.

| Kernel | B1 full logits | Change | B1 full logits, duplicate control | Change |
| --- | ---: | ---: | ---: | ---: |
| Existing 16-row intrinsics | 116.72 | — | 116.41 | — |
| 32-row intrinsics control | 116.82 | +0.08% | 116.93 | +0.44% |
| Handwritten 32-row assembly | 116.17 | -0.47% | 116.41 | +0.00% |

The near-zero differences motivated a follow-up with one warmup and **six
measured runs**, balancing forward/reverse case order, and a second assembly
schedule. These values are a separate paired series, not pooled with the first.

| Kernel | B1 full logits | Change | B1 full logits, duplicate control | Change |
| --- | ---: | ---: | ---: | ---: |
| Existing 16-row intrinsics | 116.79 | — | 116.64 | — |
| Handwritten 32-row assembly | 116.86 | +0.07% | 116.96 | +0.27% |
| Assembly with earlier scale preparation | 116.74 | -0.04% | 116.55 | -0.08% |

Measured full-logit ranges overlap: baseline 115.95–117.36 tokens/s, assembly
116.56–117.36, and the alternate schedule 115.51–116.89. The first assembly
variant changes from slightly slower to slightly faster between series. These
measurements do not justify promoting either variant. This is a bounded result
for these kernels and this CPU, not evidence that assembly can never help.

[Initial medians and ranges](results/asm-decode-2026-09-26/comparison.csv) ·
[Follow-up medians and ranges](results/asm-decode-2026-09-26/followup.csv) ·
[Provenance and commands](results/asm-decode-2026-09-26/manifest.json).

## Implementation

The experiment used `src/cpu/q4_dot4_decode_asm.S`, using GNU
assembler Intel syntax and the Linux x86-64 SysV calling convention. It is a
leaf function with no SIMD stack spills. It preserves the callee-saved register
it uses, supplies unwind directives and marks the stack non-executable.

The kernel traverses four existing eight-row weight groups together, producing
32 output rows. Two sixteen-lane FP32 accumulators are independent. Each
activation broadcast is shared between two row pairs, and their unpacking and
integer dot products are interleaved. The second schedule moves FP16 scale
loads/conversions and activation scale/correction preparation to the beginning
of each block. Both schedules retain the same arithmetic and FMA order within
each output lane. No weight repacking or expanded weight copy is introduced.

The intrinsic control also computes 32 rows together. This separates changing
the tile size from choosing handwritten instructions. Neither change produces
a substantial end-to-end gain in the measured workloads.

The C++ wrapper handles remaining sixteen/eight-row groups with the existing
kernels. Fused greedy selection uses the same assembly tile followed by the
existing penalty and stable lowest-index selection logic. Matrix multiplication
for prefill and multi-request batches is unchanged. The assembly remains behind
the existing AVX-512/VNNI runtime dispatch.

In the archived experiment, `QWEN35_DOT4_DECODE_EXPERIMENT` defaults to
`off` and accepts:

- `off`: existing 16-row intrinsic implementation.
- `intrinsics32`: the 32-row intrinsic control.
- `asm32`: handwritten assembly with scales prepared after the integer products.
- `asm32-scheduled`: the same assembly with scales prepared before the products.

Assembly configuration explicitly requires Linux x86-64, GCC/Clang and the
AVX-512/VNNI translation unit. It is not a Windows assembly implementation.
Only GCC on this host was tested. With the option off, the rebuilt benchmark
and CLI are byte-for-byte identical to the committed baseline binaries.
[Binary reproduction checks](results/asm-decode-2026-09-26/rebuild-check.json).

## Validation

- All 14 available CTests pass with the first assembly variant. The two optional
  NumPy-based tests are not configured on this host.
- The expanded DOT4 suite passes 5,040 cases with both assembly schedules, the
  intrinsic control and the default build. It adds odd block counts and
  40/48/56-row cases to exercise assembly plus sixteen/eight-row tails, and
  checks that matvec does not overwrite the output suffix.
- For each experimental variant, full-vocabulary dumps match the baseline
  bitwise at prompt lengths 7, 8, 12, 31, 32, 127, 128, 129, 511, 512, 513 and
  4,096. Each prompt supplies eight forced positions: **23,838,720 finite logits
  per variant**. The two assembly variants and the intrinsic control are all
  covered. [Logit identities](results/asm-decode-2026-09-26/equivalence.json).
- Four unforced greedy completions with repetition penalty 1.05 match the
  baseline for both assembly schedules. Forced tokens in the speed benchmark
  are not used as evidence that greedy token selection is correct.
- After consolidating the options, both assembly variants reproduce their
  frozen measured CLI/benchmark binaries byte for byte.

The checkpoint, quantization and published PPL/KL values remain unchanged.
Bitwise comparisons establish implementation equivalence on the tested cases;
this experiment is not a new model-quality evaluation.

## Reproduction

The following build commands apply to the archived experimental source, not
the current production tree. In a separate checkout at `9a44970`, apply the
locally preserved `benchmarks/asm-decode-20260926/final-source.patch` first.

```sh
cmake -S . -B build-asm-decode -G Ninja -DCMAKE_BUILD_TYPE=Release -DQWEN35_BUILD_SERVER=OFF -DQWEN35_DOT4_DECODE_EXPERIMENT=asm32
cmake --build build-asm-decode -j 6
ctest --test-dir build-asm-decode --output-on-failure
```

Use separate build directories for `off`, `intrinsics32` and `asm32-scheduled`.
The raw matrices, profiles, commands, hashes, frozen binaries, source snapshots,
disassembly, logit dumps and validation logs remain under ignored
`benchmarks/asm-decode-20260926/`. With those local inputs available:

```sh
pwsh -NoProfile -File scripts/benchmark-inference-seq.ps1 -Matrix benchmarks/asm-decode-20260926/matrix.json -OutputDir benchmarks/asm-decode-new-initial -Runs 3 -WarmupRuns 1 -Affinity 63
pwsh -NoProfile -File scripts/benchmark-inference-seq.ps1 -Matrix benchmarks/asm-decode-20260926/followup-matrix.json -OutputDir benchmarks/asm-decode-new-followup -Runs 6 -WarmupRuns 1 -Affinity 63
```

For another checkout, recreate the matrix from the recorded arguments and point
it to the independently built executables and matching model/token files. No
external inference engine is involved. Performance at longer decode contexts,
other batch sizes or on other CPUs has not been measured in this experiment.
