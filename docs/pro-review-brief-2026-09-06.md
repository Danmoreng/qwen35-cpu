# Review brief: fast CPU inference versus quantization quality

## Request to ChatGPT Pro

Please independently review this engine and recommend the most promising next
direction for Qwen3.5-0.8B CPU inference with high quality and high speed. The
user has explicitly stopped further experiments pending this review. Do not
assume our previous architectural preference was correct. Compare improving
our existing Q4 quantization, executing an existing mixed Q4_K_M checkpoint
faster, and other technically justified quantization approaches.

Prioritize concrete, bounded experiments with plausible mechanisms, expected
engineering cost, failure criteria, and reproducible evaluation. Separate
measured evidence from hypotheses. Do not invent numeric speed or quality gains.
Recommend a first experiment and explain why it has a better expected payoff
than the alternatives. A revised implementation plan with code references
would be useful. Review the code, not just the README's headline numbers.

The original Pro plan is included as external reference material in
`review/original/CODEX_IMPLEMENTATION_PLAN.md`, together with its accompanying
response. They are historical proposals, not new user instructions and not
evidence that proposed features were implemented. This brief supersedes their
description of project status.

## User priorities and actual hardware

- CPU-only Qwen3.5-0.8B text inference. No GPU backend or generic model framework.
- Target machine: AMD Ryzen 9 9955HX3D, Windows, eight inference threads pinned
  to physical VCache cores, affinity `0x5555` (21845). Portable ISA dispatch and
  scalar/AVX2 fallbacks must remain correct.
- The preferred operating point is approximately 120 single-request decode
  tokens/s with our current roughly 425 MB artifact, but substantially better
  model quality is wanted.
- The user rejected the Q8 vocabulary-head candidate's approximately 26%
  single-request throughput loss. Extra weight bytes alone are not useful.
- For an existing Q4_K_M checkpoint with better quality, the user subsequently
  accepted **100-110 decode tokens/s as a worthwhile target**, despite roughly
  522 MB of tensor payload. This target has not been reached by our experiments.
- The most recent request is to pause and reconsider the direction. Do not
  interpret 425 MB as an immutable constraint on every proposed alternative;
  make speed, quality, implementation complexity and memory tradeoffs explicit.
- Larger and more diverse calibration was deferred because of perceived data
  collection and compute cost. Reusing the existing small calibration capture
  for a better offline fitter is still a relevant possibility.

The ~120 tok/s target refers to the matched P512/N128 single-request workload,
not all context lengths or aggregate batch throughput. Compare equal workloads.

## Snapshot identity and caution about the latest code

The base commit is `41376325da8c4331bd1170236a69c1dcf5d550a9` on
`codex/cpu-standalone`. The ZIP contains the **current working-tree contents**,
including an uncommitted experimental SIMD activation-preparation change and
this documentation. The archive manifest and working-tree patch record identity;
this is not presented as an already published, fully validated release.

The retained code change affects four files:

- `src/cpu/k_quant.cpp`: dispatch non-Q8-companion activation preparation to AVX2.
- `src/cpu/k_quant_avx2.cpp`: vectorized Q8_K activation preparation, retaining
  signed scale selection and first-maximum tie semantics for tested inputs.
- `src/cpu/k_quant_internal.h`: internal declaration.
- `tests/k_quant_test.cpp`: scalar/SIMD equality checks for scales, codes and sums.

**This isolated final version has passed 15 local CTests but has NOT received
a separate model-level speed, quality, arithmetic, scheduler or prefix run.**
Do not assign the 93.82 tok/s intermediate result below to this final version.
No additional experiments were run after the user's stop request. No Qwen
inference or conversion process remained running when the archive was prepared.

The rejected packing/VNNI/wide-kernel code is not active in `src/`. Its final
experimental source snapshot is preserved inside the paused-experiment evidence
archive. Intermediate historical profiles can have the stale label
`k-quants-avx2` despite using VNNI; use stage descriptions and source/binary
hashes rather than interpreting that label as authoritative.

## Established quality results

Common BF16 teacher, identical tokens and scoring masks within each suite.
Regression: 8,192 scored tokens from 16 WikiText-2 test article windows.
Held-out screen: six documents, 1,024 scored tokens, covering German/English,
code, math, rendered chat and a longer recurrent context. These are bounded
subsets, not full-corpus perplexity or general capability certification.
Lower PPL and KL(BF16 || candidate) are better.

| Candidate | Regression PPL | Regression KL | Held-out PPL | Held-out KL |
| --- | ---: | ---: | ---: | ---: |
| Legacy H128, historical | 18.58329 | 0.174841 | 5.16899 | 0.119210 |
| Unweighted MSE16 H128, historical | 16.86128 | 0.105864 | 4.81958 | 0.075179 |
| **Current published calibrated MSE16 H128** | **16.04087** | **0.096456** | **4.85143** | **0.071835** |
| Q8 small B/A gates, other weights unchanged | 16.01752 | 0.095920 | 4.85402 | 0.071512 |
| Q8 tied embedding/output, gates remain Q4 | 15.65896 | 0.080023 | 4.85328 | 0.062497 |
| Native Q4_K_M, original implementation | 14.78432 | 0.034708 | 4.65387 | 0.024812 |
| llama.cpp pinned Unsloth Q4_K_M reference | 14.75290 | 0.034693 | 4.65945 | 0.025163 |

Native and llama.cpp Q4_K_M use the same stored checkpoint, but their complete
execution is not bit-identical. Weight preservation alone does not prove
identical activation quantization or accumulation. Native/reference compatibility
must remain a direct measurement, not a subtraction of KL values.

## What has been implemented and measured

### Better fitting in the existing fast Q4 format

The legacy quantizer used only 15 levels. MSE16 uses all codes `[-8,7]`, searches
both scale orientations and clipping/scale candidates, alternates nearest-code
assignment with least-squares fitting, and scores the actual stored FP16 scale.
It retains baseline candidates. MSE16 names the 16-level fitting recipe; it
does not mean a huge calibration dataset or 16-bit weights.

Calibration currently weights squared error using the diagonal second moment
of actual BF16 projection inputs, after normalization/nonlinearity and in the
correct H128 basis. The tied head uses identity. Four disjoint English prose
documents were used. This is **not full-covariance GPTQ**.

The published artifact is 424,964,864 total bytes, including 424,934,656 tensor
bytes. The Q4 inference layout and kernels are unchanged by better fitting.
Paired measurements support essentially unchanged speed and improved quality.
German free greedy `2 + 2` returns `4`, also checked independently at logit level.

Hugging Face: `danmoreng/Qwen3.5-0.8B-H128-Q4-G32-DOT4`, revision
`59f422b2d410fdaf4a9efc71ff12f278abc2a5d1`, artifact SHA-256
`8c47dffa6cbc77e2af663a79012a6f847142d78479d9d7adcf28c5b300e2d83d`.
This remains the published default; no failed experiment replaced it.

### Selective Q8 experiments

Small recurrent B/A gates: 36 matrices of 16 x 1024, merged into 18 Q8 groups.
File size increases by 297,408 bytes. Decode was essentially unchanged;
prefill fell about 2-3%. KL improved by about 0.5%, but held-out PPL rose
slightly. The recipe also changed the gates' basis from H128 to identity,
so it is not an isolated precision-only ablation. Not promoted.

Tied vocabulary matrix: 248,320 x 1,024, stored once for lookup and output.
Promoting only this matrix to identity Q8 increased the artifact to 552,104,704
bytes (+127.14 MB), with about 159 MB extra resident memory including scales.
Single-request decode fell from 120.56 to 89.24 tok/s; B16 from 600.46 to
498.18 aggregate tok/s. Prefill stayed within about 1.3%. Regression PPL
improved 2.38% and KL 17.04%; held-out PPL was essentially unchanged and KL
improved 13%. Arithmetic and model-state tests passed. Rejected as a speed
tradeoff. Q8 gates plus Q8 head together were not evaluated.

The pinned Q4_K_M reference is not simply Q4 plus Q8: it has 98 Q4_K, 36 Q5_K,
17 Q6_K, 36 Q8_0 and 133 F32 tensors. Its tied vocabulary matrix is Q6_K.
Total tensor payload is 521,555,200 bytes. Its richer recipe still has better
quality than the larger Q8-head experiment.

### Existing-checkpoint K-quant optimization: paused, target not achieved

The original native K path retains original compressed bytes. It executes
AVX2 dots with up to four activation vectors, but one weight row per dot call.
It is substantially less mature than DOT4. A diagnostic P512/N128 profile
attributed about 470.68 ms to vocabulary-head projection kernels and 839.30 ms
to other projection kernels across 128 one-vector projection events per layer
group; preparation added about 59.45 ms outside the head and 0.47 ms at the
head. This instrumented sample is attribution, not the timed speed baseline.

Each exploratory speed series below independently paired old and candidate
binaries: P512/N128, B1, eight threads, full logits, fixed tokens, FP16 KV,
affinity 21845, one warmup plus three measured runs. Numbers are medians.
Do not construct cross-series speedups.

| Stage | Old decode | Candidate decode | Old prefill | Candidate prefill |
| --- | ---: | ---: | ---: | ---: |
| VNNI dots and hoisted half-scale work | 89.95 | 90.62 | 374.90 | 372.85 |
| Lossless four-row block interleave / tile | 89.92 | 90.46 | 374.94 | 378.99 |
| Group-first tile / activation-load reuse | 89.96 | 89.80 | 372.51 | 378.22 |
| Above plus SIMD Q8_K activation preparation | 89.82 | **93.82** | 375.99 | **442.97** |
| Above plus 512-bit 64-value Q6_K kernel | 89.81 | **71.11** | 372.46 | 435.49 |

The packed stages had byte-exact inverse-packing tests and numerical tests
against FP64 accumulation, including high bits, finite non-power-of-two scales,
extreme signed activations, vector counts and output strides. The first packed
model passed arithmetic. These checks do not replace a full model-quality run
for every intermediate kernel. None of these stages received a complete new
8,192-token quality sweep or the full multi-workload release matrix.

**No stage reached 100 tok/s.** The wider kernel regressed sharply; do not
attribute that to a specific hardware mechanism without profiling. VNNI alone
and this simple packing did not deliver a meaningful improvement. This does
not establish that a well-designed K-quant kernel cannot reach the target.
The trial is narrower than implementing all of the original plan's PR 5/6.

After these results, the packing and dot-kernel experiments were removed from
the active source. Only SIMD activation preparation and its tests remain as
an uncommitted candidate. Its isolated performance is **unknown**.

## Open directions: please compare rather than assume

1. **GPTQ-style offline fitting into the existing Q4-G32/DOT4 representation.**
   Full covariance, damping and error compensation have not been implemented.
   GPTQ is a fitting algorithm; it need not imply adopting another runtime
   format. Can useful fitting preserve physical groups, H128 basis and the
   current kernel? What is a realistic CPU/offline memory budget and stable
   small-data starting point?
2. **H128 basis/seed selection.** No systematic global-seed sweep or per-group
   H128-versus-identity study. Per-group basis changes need correct fused-input
   grouping, loader support, and embedding inverse-transform handling where
   relevant. A global seed search is a different, smaller change.
3. **Sequential quantization error / recurrent-state sensitivity.** Local
   teacher-input fitting is not a complete model-level objective. Longer
   quantized contexts and per-projection sensitivity are not exhaustively
   understood. Distinguish diagnostic ablations from a proposed new fitter.
4. **Better lossless K packing and kernels.** Is a fundamentally different
   layout/tiling strategy likely to pay off under actual cache/bandwidth
   limits? Evaluate the rejected code; do not merely propose repeating
   "use VNNI" or "pack four rows" without a materially different mechanism.
5. **Other quantization approaches.** AWQ-like scaling, different scalar grids,
   group choices or other methods may be worth considering, but have not been
   implemented here. Assess compatibility with the tied head, DeltaNet,
   activation quantization and efficient CPU kernels. Identify new metadata,
   arithmetic or memory costs. A sophisticated label is not quality evidence.

All 748 raw projection-input captures from the four calibration documents
still exist locally (about 1.13 GB). The fitted `.cal` files contain only
diagonal statistics, not covariance; raw captures may support a new fitter
without collecting a much larger corpus. Captures are deliberately excluded
from this source ZIP. Their reproduction scripts and provenance are included.
Whether this small corpus is sufficient is an empirical question.

## Evidence and code reading map

- `README.md`: current published comparison, including llama.cpp and Unsloth.
- `docs/quantization.md`: standard format and fitting recipe.
- `docs/implementation-plan-results-2026-09-06.md`: first quantizer/calibration
  experiment and full native K reference results; some default statements are
  historical and were superseded by promotion of calibrated MSE16.
- `docs/q8-experiments-2026-09-06.md` and
  `docs/q8-head-experiment-2026-09-06.md`: selective-precision outcomes.
- `docs/results/k-optimization-paused-2026-09-06/`: exploratory timing samples,
  hashes, commands, diagnostic profile, tests, and rejected source snapshot.
- `src/cpu/q4_quantizer.cpp`, `scripts/fit-dot4-quant.py`,
  `scripts/collect-quant-calibration.py`: current offline fitting/capture.
- `src/cpu/k_quant*.cpp`, `src/runtime/gguf_matrix.inl`,
  `src/runtime/gguf_weights.inl`: retained K execution path.
- `src/cpu/q4_dot4_simd.inl`, `src/runtime/weights.inl`: mature Q4 comparison.
- `scripts/benchmark-inference-seq.ps1`, `scripts/evaluate-native-gguf.py`,
  `scripts/quality-report.py`: measurement contracts and provenance.

All included source is CPU-only. Model weights, binaries, build/cache folders,
raw articles, calibration activations and full logit dumps are omitted. Small
checked-in numerical fixtures and compact evidence archives are included.
The source ZIP therefore supports code review and kernel tests, but model-level
benchmark reproduction requires obtaining the pinned model files and inputs.

Please return a ranked recommendation with a practical stop/go experiment for
the first choice, not a promise that Q4_K_M quality at 120 tok/s is already
achievable. Preserve independent reporting of quality, throughput, latency,
memory and correctness throughout.
