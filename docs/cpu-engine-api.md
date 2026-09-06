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
