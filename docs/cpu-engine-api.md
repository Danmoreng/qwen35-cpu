# CPU engine API

`qwen35x/runtime/cpu_engine.h` exposes immutable `CpuModel` weights and a
single-owner `CpuEngine`. Independent engines may share a loaded model. Each
engine owns one executor and scratch workspace; its methods are called serially.
Callbacks run synchronously and must not reenter the engine.

```cpp
std::string error;
qwen35x::CpuLoadOptions load;
load.model_dir = "models/qwen3.5-0.8b";
load.cpu_q4_h128_path = "models/qwen3.5-0.8b/model.q35h";
auto profile = qwen35x::ProfileLoader::load_from_hf_directory(load.model_dir, error);
if (!profile) throw std::runtime_error(error);
auto model = qwen35x::CpuModel::load(*profile, load, error);
if (!model) throw std::runtime_error(error);
qwen35x::CpuEngineConfig config;
config.threads = 8;
auto engine = qwen35x::CpuEngine::create(model, config, error);
if (!engine) throw std::runtime_error(error);
qwen35x::CpuRequestSpec request;
request.prompt_tokens = prompt_token_ids;
request.sampling.temperature = 0;
auto id = engine->submit(std::move(request), error);
if (!id) throw std::runtime_error(error);
qwen35x::CpuRequestResult result;
do {
  if (!engine->advance(id, error)) throw std::runtime_error(error);
  if (!engine->result(id, result)) throw std::runtime_error("Unknown request");
} while (result.status == qwen35x::CpuRequestStatus::prefill ||
         result.status == qwen35x::CpuRequestStatus::decode);
engine->release(id);
```

Include `qwen35x/compiler/compiler.h`, `qwen35x/runtime/cpu_engine.h` and your
application's standard headers. Link `qwen35x_core`. Supply native token IDs;
use `QwenTokenizer` or your application's tokenizer with the same vocabulary.

## Direct batches

Initialize prompts with `advance()` until requests enter decode, then call
`advance_batch()` with distinct active request IDs. Each batch jointly executes
projections and the vocabulary head. Remove terminal requests before the next
batch. Full-logit callbacks and non-greedy sampling retain the full-head path.
The optimized greedy reduction applies to eligible batches.

`CpuRequestSpec::sampling_constraint` optionally retains a request-local
`CpuSamplingConstraint`. Its `apply()` masks raw logits before penalties and
sampling; `accept()` advances the constraint after each selected token. Both run
on the owner and must report failure when no finite candidate remains. A
constraint disables greedy head shortcuts and cannot be combined with forced
tokens or selected readout. The HTTP server uses this for native JSON grammars;
unconstrained requests retain the existing path.

## Bounded serving

Set `max_queued_requests` nonzero and configure `max_resident_requests`,
`max_decode_batch_size`, `max_buffered_tokens` and `max_context` before creation.
Submit requests, then call `step(progressed, error)` repeatedly. One call performs
at most one admission and one bounded prefill chunk or decode batch. It does not
wait for a full batch. Drain tokens with `read_output()`; a slow reader pauses
its own request when its output buffer fills. Cancellation is request-local.

Completed results remain until `release()`. Release them promptly: the record
limit includes terminal results. A serving terminal request frees its resident
model state. Stop-sequence prefixes are withheld until resolved. Callbacks must
outlive their request. There are no background threads for scheduling and no
HTTP transport inside `CpuEngine`; the executor's workers run CPU operations.
The separate [native server](server.md) owns and drives this API.

## Prefix reuse

`register_prefix(tokens, trust_namespace, error)` builds an immutable hybrid
snapshot: FP16 KV, FP32 DeltaNet state, convolution history and final hidden
state. Attach its handle to a request with matching leading tokens and namespace.
Handles bind to the actual loaded model object and prefill configuration.
Keeping a handle pins its storage against eviction.

For queued requests, `register_prefix_tokens` enables automatic single-flight
construction: matching followers wait for one leader to build the shared prefix.
Cache-budget failure falls back to cold processing. Explicitly isolate unrelated
clients using separate namespaces; this is a caller-enforced trust boundary.

`find_prefix(tokens, trust_namespace)` returns the longest retained exact match
without running inference and updates its LRU position. An attached prefix may
be extended with a larger `register_prefix_tokens` boundary; concurrent matching
extensions retain single-flight behavior. Byte budgets evict unpinned entries.
Results distinguish `cached_prefix_tokens` from `prefix_written_tokens` (new
tokens successfully published, excluding a restored portion of an extension).
`stats()` exposes retained entries/bytes, restore hits/tokens, builds, evictions
and budget fallbacks. The HTTP server uses these mechanisms for automatic chat,
raw-completion and qev prefix reuse.

Set `shared_kv_pages = true` for immutable 256-token FP16 pages with copy-on-write
at a partial tail. Recurrent and convolution state remain private to branches.
`physical_kv_bytes` deduplicates KV page payload; it is not total process memory.
The prefix budget charges snapshot storage conservatively and pinned handles can
prevent further cache admission. Contiguous KV remains the default reference path.

## Validation

`cpu_scheduler_model_test <model-dir> <artifact> --pages` checks the real-model
scheduler, backpressure, cancellation, single-flight and paged-prefix paths.
`cpu_prefix_model_test <model-dir> <artifact>` checks prefix/full-logit parity.
Neither test's execution time is a performance benchmark.

## Experimental GGUF loading

The historical `CpuLoadOptions::cpu_q4_h128_path` field also accepts a Qwen3.5-0.8B
GGUF containing pure Q4_0 matrices or the supported Q4_K_M recipe (Q4_K, Q5_K,
Q6_K, Q8_0 matrices), plus F32 retained tensors.
The loader selects by file magic, validates shapes and retains the packed bytes.
`CpuModel::weight_format()` returns `h128-q4-dot4`, `gguf-q4_0-dot4` or `gguf-k-quants`.
Config/tokenizer handling remains unchanged. See [format support](q4km-native.md).

## Text drafts

Set `CpuRequestSpec::draft_mode` to `CpuDraftMode::ngram` or `CpuDraftMode::input` to opt in. The default is off. The verifier is available for single-stream text decode with native packed Q4 DOT4 weights and AVX2 or a wider x86 backend. Multirequest decode batches, forced tokens, logit callbacks and constrained sampling follow their ordinary execution paths.

The n-gram mode indexes three-token sequences with known successors in the prompt and already generated output. A match proposes up to three continuation tokens from the most recent occurrence. Input mode follows a cursor in the source, realigns locally after edits and uses a three-token lookup to find more distant matches. Supply `draft_source_tokens` for a particular document. An empty source selects the prompt. Include the document in the prompt so the model can read it. The separate source guides the draft lookup.

The request owner verifies one pending token and the candidate block in a single batched target forward. Lookup drafting proposes up to three candidates and MTP up to four. Attention and the recurrent state advance in causal order. Every returned token is selected by the target model using the request's penalties and RNG. Unused draft positions consume no random samples. On a rejection, the engine keeps the selected replacement, restores convolution and DeltaNet state after the pending token, and advances those states using the saved linear-layer projections for the accepted rows. Target attention KV already contains the accepted rows. The engine truncates paged KV to the committed boundary and overwrites unused contiguous rows on the next forward. Stop conditions and available output buffer space bound each round.

After a window of at least twelve proposed tokens, acceptance below one third
pauses draft lookup for the next 16 generated tokens in that request.
The engine emits those tokens with ordinary decode and then tries lookup
again. Each window starts with fresh acceptance counts.

One `advance()` may deliver up to five tokens when MTP drafting is enabled.
`decode_forwards` counts committed decode positions. Engine stats expose
draft rounds, proposed and accepted tokens, replayed positions, and
`draft_cooldowns` for pauses started. The verifier retains one recurrent
checkpoint per active engine (18 MiB for 0.8B and 2B, 48 MiB for 4B, plus
convolution state). It retains the middle verification rows of each linear
input projection for rollback and may hold five vocabulary rows for sampling.
Request-local lookup indices grow with the indexed text.

### Learned MTP drafts for 4B

Load the [4B MTP sidecar](quantization.md#optional-4b-mtp-weights) via
`CpuLoadOptions::mtp_weights_path` and select `CpuDraftMode::mtp`. The
experimental path uses the native 4B Q4 DOT4 output head with an AVX2-capable
backend. `CpuEngineConfig::mtp_max_tokens` defaults to one candidate token
and accepts up to four. Three proposals use one target forward with four
causal rows including the pending token.

The trained block combines the target's normalized hidden state with the
following token's embedding and writes to a request-private FP16 KV cache.
During prefill, it computes only the K and V rows needed for later drafting.
The existing verifier selects output tokens with the target sampler and its
request RNG. Candidate generation consumes zero target RNG draws. The engine
overwrites committed MTP KV rows with the corresponding target features after
verification. A multi-request batch uses ordinary target decoding and advances
the MTP cache so a later single-request turn can resume drafting.

For MTP, pass `CpuDraftMode::mtp` to `register_prefix` and `find_prefix`.
Serving requests that set `register_prefix_tokens` derive the snapshot kind
from the request's draft mode. An MTP snapshot holds committed sidecar
attention KV alongside the target snapshot and its normalized final hidden
row. Restoration copies the sidecar rows into request-local state and uses
the existing final hidden row. The shared prefix budget counts the sidecar
rows. The same prompt and namespace can hold both snapshot kinds.

MTP supports constrained sampling, with the live constraint advancing on each
selected target token. A constraint can
provide a private copy via `fork_for_greedy_draft()` for grammar-guided
proposals. The HTTP JSON and tool-call grammar provides this copy.
Forced tokens and logit callbacks require the ordinary
request path. The benchmark profile records MTP prefill overhead, proposal,
verification and cache-repair times separately. In
`mtp_batches_by_proposals`, the four elements count verification forwards
with one through four proposed tokens.
