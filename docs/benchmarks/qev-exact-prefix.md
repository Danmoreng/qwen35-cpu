# Exact-prefix reuse on the native QEV decision endpoint

On `POST /v1/systemone`, an identical question and identical options did not
produce a cache hit when each ticket occupied the leading `state` field.
A second experiment moved a fixed routing policy into `state` and placed the
individual ticket in the question. After registering a shared boundary, all
eight previously unseen tickets restored 121 tokens each.

## One request at a time

Localhost HTTP wall time on an Intel Xeon Platinum 8573C VM, four inference
threads pinned to CPUs 0–3, four CPUs of quota. Native Release server at
`eb0df0ae5601771de628476def338419a43a53e2`, released calibrated qev-0.8b
H128/Q4-G32-DOT4, selected option head, default adaptive prefill chunk, and
no generated tokens. Each panel and cache setting started a fresh server.
Timing began after readiness and includes HTTP, tokenization, any snapshot
publication or restoration, inference and response formatting.

| Request | Input tokens | Restored with auto | Cache off, ms | Cache auto, ms |
| --- | ---: | ---: | ---: | ---: |
| Short README fixture, first call | 74 | 0 | 904.9 | 1,052.9 |
| Same short fixture, repeated | 74 | 74 | 365.1 | 7.9 |
| Long README fixture, first call | 579 | 0 | 1,757.0 | 1,476.1 |
| Same long fixture, repeated | 579 | 579 | 1,696.6 | 7.0 |
| New ticket in state, fixed question and options | 60–70 | 0 | 169.4 | 255.9 |
| New ticket after a registered rubric boundary | 168–176 | 121 | 1,932.2 | 1,134.6 |

One warmup and three measured runs per setting through the sequential benchmark
runner, alternating case order. Each cell is the median of the three within-run
medians. Each run has one first call and four repeats per README fixture,
eight distinct ticket-first calls, and eight new rubric-first calls after
priming. The 579-token input reproduces the README fixture with its repeated
account history. The varying-ticket cases use distinct support requests.
Run-to-run wall time on this shared VM varied substantially; the per-run medians
and every client timing are retained in the results. The published 40.29 and
180.07 ms are first-request measurements on a Ryzen 9
9955HX3D and cannot establish a cached latency on that machine.

For the identical fixtures, removing all 74 or 579 tokens of prefill reduced
resident repeat latency by 97.8% and 99.6% respectively. For new tickets
following the 121-token registered rubric boundary, the reduction was 41.3%
across medians and 27–42% in the individual paired runs. For new ticket-first
requests, zero tokens were restored across all runs. Timing differences on
misses include snapshot writes and VM variability.

## What makes a prefix reusable?

- QEV renders `Context: <state>` before the question, options and answer marker.
  The lookup compares token IDs from position zero through a retained boundary.
  An ordinary single-question call registers the entire input when eligible.
  Two independent single-question calls with the routing policy at the front
  and different ticket suffixes still read zero cached tokens before priming.
- To establish a reusable policy boundary in the experiment, one
  `per_question` call supplied the same policy as the state and two distinct
  tickets as separate questions. The server registered their common initial
  tokens (policy plus shared question text). The eight follow-up calls each
  used one new ticket. The latency row excludes the priming request. Input
  layout here tests reuse; it does not establish classification quality for a
  fine-tune trained on a different layout.
- At the boundary, the engine saves KV for all six full-attention layers and
  the FP32 recurrent matrix, convolution history and ring index for all 18
  GatedDeltaNet layers. It also retains the final normalized hidden vector.
  A partial hit restores request-private recurrent state and runs the suffix.
  A complete hit still restores state and projects the option logits from the
  saved hidden vector, so the endpoint recomputes its answer.

The two-question priming request took 3,470.9 ms with caching enabled on
this VM. Of 87 measured responses with caching
enabled, 87 returned exactly the same full answer
objects as their cache-off counterparts. The maximum absolute difference in
reported probabilities was 0.

[README latency baseline](laptop-refresh-9955hx3d.md#qev-08b-decision-latency) ·
[Snapshot capture and restore](../../src/runtime/weights.inl) ·
[QEV serialization](../../src/systemone.h) · [Measurements, inputs and commands](../results/qev-exact-prefix-2026-11-21)

## Reproduce

From the repository root with the Release server and published QEV package:

```sh
python docs/results/qev-exact-prefix-2026-11-21/make-matrix.py \
  --server build/qwen35_cpu_server --model-dir models/qev --out benchmarks/qev-prefix
pwsh scripts/benchmark-inference-seq.ps1 -Matrix benchmarks/qev-prefix/matrix.json \
  -OutputDir benchmarks/qev-prefix/runs -Affinity 15
```
