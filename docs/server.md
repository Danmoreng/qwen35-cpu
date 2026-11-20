# Native CPU HTTP server

`qwen35_cpu_server --model-dir models/qwen3.5-0.8b`

The directory contains `model.q35h`, config and tokenizer files. `--weights`
can override the artifact path, including a supported Q4_0 `--pure` or Q4_K_M GGUF file in source
builds. Config and tokenizer files are still read from `--model-dir`. A GGUF-backed
server reports `Qwen3.5-0.8B-GGUF` from `/v1/models`; omit the request model field
or use that ID. See [Q4_K_M support](q4km-native.md). One owner drives the model/tokenizer and bounded
CpuEngine scheduler. HTTP workers never create separate inference executors.
The process starts listening only after the model has loaded successfully.

## API

- `GET /health`: readiness (`{"status":"ok"}`).
- `GET /v1/models`: the single supported model ID.
- `GET /v1/cache`: prefix-cache counters and bounded storage usage.
- `POST /v1/completions`: raw prompt, JSON or SSE response.
- `POST /v1/chat/completions`: text messages, native Qwen formatting, JSON or SSE response.
- `POST /v1/responses` supports local text generation, function calls and stored response chains. See [Responses](#responses).
- `POST /v1/messages`: Anthropic-compatible text Messages, JSON or SSE response.
- `POST /v1/messages/count_tokens`: logical token count using the native formatter/tokenizer.

```json
{
  "model": "Qwen3.5-0.8B-H128-Q4-G32-DOT4",
  "prompt": "Once upon a time",
  "max_tokens": 128,
  "temperature": 0,
  "top_p": 0.8,
  "top_k": 20,
  "repetition_penalty": 1,
  "seed": -1,
  "stream": false,
  "n": 1
}
```

Only `prompt` is required. Model name is optional; if supplied it must match.
The other fields use the defaults above. Automatic prefix reuse is enabled by
default. The server is single-tenant, with one optional API key; the cache does
not provide a separate authentication boundary between clients.

Responses contain `id`, `object`, `created`, `model`, one text choice,
`finish_reason` and token `usage`. Stop IDs come from the model tokenizer.
Raw prompt text must already contain any desired chat template. Unsupported
fields and multiple completions are rejected explicitly.
The supported compatibility surface is documented below; it does not implement
the entire OpenAI platform API.

## Chat completions

`POST /v1/chat/completions` accepts `messages` instead of `prompt`. Messages use
`system`, `developer`, `user` or `assistant` roles, with string content or arrays
of `{"type":"text","text":"..."}` parts. At least one user query is required.
Leading system/developer messages are combined into the Qwen system turn with
two newlines between them. Such instructions are rejected after the conversation
has begun. Tool messages follow the structured-tool contract below. Images and
unsupported message fields are rejected.

At startup the server compiles the checkpoint's Jinja templates with a native
C++ interpreter. It reads `chat_template.jinja` and named files under
`chat_templates/`. If there are no template files it uses `chat_template` in
`tokenizer_config.json`. For tools it selects `tool_use` when available,
otherwise `default`. Legacy packages without a template use the bundled
official Qwen3.5-0.8B template. The server renders before tokenization and
reuses the compiled template across requests.

Chat uses the checkpoint's own generation prompt and thinking default. The
stock 0.8B and 2B templates disable thinking by default, while 4B enables it.
Set `"chat_template_kwargs": {"enable_thinking": false}` to override that
default. Fine-tunes can read other JSON values from the same object. Assistant
history can include `reasoning_content`. Generated thinking text is returned in
the assistant content. The Responses and Anthropic Messages adapters explicitly
request thinking-disabled generation.

Supported generation fields are `model`, `max_completion_tokens` (or `max_tokens`,
but not both), `temperature`, `top_p`, `n=1`, `stream`, `stream_options.include_usage`,
`stop`, and the existing
`top_k`, `seed`, `repetition_penalty`, `prefix_tokens`, `cache_prompt` extensions,
plus optional `prompt_cache_key`. Output defaults
to 128 tokens. Chat defaults to temperature 1, top_p 1 and top_k 0; raw
completions retain the defaults above. Set temperature 0 for greedy decoding.
Optional numeric fields may be null to use their defaults.

Responses use `object: "chat.completion"`, an assistant `message`, finish reason
and token usage. Unsupported fields fail explicitly with an OpenAI-style error.
Structured output and tool fields are described below.
The OpenAI SDK works with a custom server `base_url` and
the actual model ID from `/v1/models`; API compatibility does not imply OpenAI
model capabilities.

## Responses

Point an OpenAI SDK at the local server and use the ID from `GET /v1/models`.

```python
from openai import OpenAI

client = OpenAI(base_url="http://127.0.0.1:8080/v1", api_key="local")
model = client.models.list().data[0].id
response = client.responses.create(
    model=model,
    instructions="Korrigiere nur Rechtschreibung und Zeichensetzung.",
    input="wir treffen uns morgen früh am bahnhof",
    store=False,
)
print(response.output_text)
```

- `POST /v1/responses` accepts a string or typed text messages, developer/system instructions, assistant history, a single client function call and its output. Stored items can also be passed by `item_reference`. The formatter, tokenizer, scheduler, prefix cache and sampling implementation are shared with chat.
- `text.format` supports plain text, JSON objects and the [JSON Schema subset](#structured-output-and-tools). The engine enforces the grammar at each sampling step. An exhausted output budget returns `status: "incomplete"` and `incomplete_details.reason: "max_output_tokens"`. Check the status before parsing. `text.verbosity` applies a natural-language instruction. `temperature`, `top_p` and `max_output_tokens` control generation.
- `previous_response_id` includes prior input and output in the new request. Top-level instructions apply to the current response. Supply them again to carry them forward. `truncation: "auto"` drops the oldest complete conversation turns while retaining leading instructions and the latest user turn. The body and formatted prompt size limits still apply. An oversized final turn returns HTTP 400.
- `stream: true` sends named SSE lifecycle, item, content and text events with sequential event numbers. Function arguments are emitted after grammar validation. `background: true` requires storage and continues after disconnect. A background response created with streaming supports replay with `GET /v1/responses/{id}?stream=true&starting_after=N`.
- `GET /v1/responses/{id}`, `DELETE /v1/responses/{id}` and `POST /v1/responses/{id}/cancel` retrieve, delete and cancel a background job respectively. Retrieval and deletion also work for stored foreground responses. Finalized responses can be deleted. Input items support cursor pagination at `GET /v1/responses/{id}/input_items` using `limit`, `order` and `after`.
- `POST /v1/responses/input_tokens` counts the locally formatted input with the model tokenizer. Generated responses report cached input tokens and output tokens. Token counting alone does not populate the inference cache.
- By default the single-tenant process retains up to 32 response records in RAM, including input, output, metadata and any streaming journal. On insertion the oldest finished record is evicted when needed. Restarting drops the records. With `store: false` the result is returned to the caller and the ID cannot be retrieved. The separate inference prefix cache has its own budget and policy.

### Compatibility boundaries

| Capability | Current behavior or dependency |
|---|---|
| Tool calls | Client functions using the documented object schemas, up to eight definitions and one call per turn. Parallel calls and hosted tools require additional orchestration. |
| Log probabilities | Positive `top_logprobs` and `message.output_text.logprobs` are unsupported. Full logits exist internally. The server needs the selected token and normalized alternatives from the sampler, including alignment with decoded output and streaming. |
| Reasoning and compaction | This formatter uses thinking-disabled Qwen text generation. Effort budgets, reasoning output and compatible opaque compaction items require different model/runtime behavior. The compact endpoint returns 501. |
| Images, audio and files | The text checkpoint has no vision/audio frontend. File extraction, retrieval, ASR and TTS require separate processing or models. |
| Hosted services | Web/file search, code execution, managed prompts, vector stores, external MCP and cloud service tiers require service integrations. Requests for unsupported options return an error. |
| State and transport | The HTTP/SSE implementation supports process-local response IDs. Durable conversations, cross-replica routing, 24-hour cache retention, WebSocket sessions and webhooks need storage or transport work. |

The optional SDK integration test covers the local contract. See [validation](validation.md#responses-sdk).

## Batches

Upload a UTF-8 JSONL file with `purpose="batch"`, then submit its ID to `POST /v1/batches`. The server queues the job and returns its ID immediately. Every row needs a unique `custom_id`, method `POST`, the batch endpoint and an independent request body. Choose one endpoint per file from `/v1/chat/completions`, `/v1/completions`, `/v1/responses` and the local qev extension `/v1/systemone`. Responses rows take a self-contained history and use `store=false`. Streaming and nested background jobs are unsupported.

```python
import json
import time
from openai import OpenAI

client = OpenAI(base_url="http://127.0.0.1:8080/v1", api_key="local")
model = client.models.list().data[0].id
endpoint = "/v1/chat/completions"
texts = ["wir sehen uns am freitag", "bitte schick die rechnung per mail"]
rows = [
    {"custom_id": str(i), "method": "POST", "url": endpoint, "body": {
        "model": model, "temperature": 0, "messages": [
            {"role": "system", "content": "Korrigiere Schreibweise und Zeichensetzung."},
            {"role": "user", "content": text}]}}
    for i, text in enumerate(texts)
]
data = "\n".join(json.dumps(row) for row in rows).encode()
source = client.files.create(file=("transcripts.jsonl", data), purpose="batch")
job = client.batches.create(input_file_id=source.id, endpoint=endpoint, completion_window="24h")
while job.status in {"validating", "in_progress", "cancelling", "finalizing"}:
    time.sleep(0.5)
    job = client.batches.retrieve(job.id)
for file_id in (job.output_file_id, job.error_file_id):
    if file_id:
        for line in client.files.content(file_id).text.splitlines():
            result = json.loads(line)
            print(result["custom_id"], result["response"])
```

Use `client.batches.list()`, `retrieve(id)` and `cancel(id)` for status and cancellation. Completed rows remain downloadable on cancellation or expiry. Structural validation failures, including duplicate IDs, fail the batch before inference and report a line number. Individual inference or parameter errors go into a separate error file. File endpoints support upload, paginated listing, metadata, content and deletion. Requests use the same authentication as chat. Locally stored transcripts are visible to clients sharing the server key.

The default store is in memory. **For restart recovery**, launch with `--batch-db /path/to/batches.sqlite` on a writable local volume, using a dedicated database per model and server. SQLite commits row results and counts together. A restarted server resumes unfinished rows and preserves existing output files throughout their lifetime. One process owns the database. Delete unneeded files with `client.files.delete(id)`. Inputs in active batches are pinned.

The local limits are 4 MiB and 4096 rows per input file, 256 KiB per line, 32 MiB of results per job and 64 queued or running jobs. The file store allows 128 MiB of payload, including 32 MiB reserved from uploads for results. A worker feeds one batch at a time to the shared inference scheduler. It can run multiple rows concurrently with ordinary API traffic. The 24-hour window includes queueing.

### Automatic CPU selection

The default `--threads auto --batch-width auto` probes the loaded model before listening. It measures short greedy decode batches, or prompt processing with the selected decision readout on qev when enabled. The server tests a shortlist of thread counts and widths up to the resident limit, 16 rows and a memory-headroom estimate. Linux uses process affinity and available cgroup v2 CPU quotas. Windows uses process affinity when available. Each candidate receives a warmup and three measurement windows. The probe selects by median throughput, first comparing thread counts and then widths with those threads. Output length, prompt length, sampling mode and concurrent load can change the best setting.

`GET /v1/batch-config` reports detected CPU capacity, chosen inference threads, simultaneous batch rows and the measured candidates. Use `--threads N --batch-width N` to skip the thread and width probes. The width must fit `--residents`. The server shares one inference executor across batch and interactive work.

## Streaming and stops

Both completion routes accept `stream: true` and return `text/event-stream`.
Chat streams start with the assistant role, followed by content deltas, a finish
reason and `[DONE]`. Raw completions use `text` deltas. All chunks retain the same
ID, model and creation timestamp. `stream_options: {"include_usage": true}` adds
a final usage chunk with empty choices before `[DONE]`; intermediate usage is
null. An interrupted/error stream does not promise a final usage event or `[DONE]`.

`stop` accepts one string or up to four strings of 1..256 UTF-8 bytes on either
route. Matches operate on decoded text, including across token boundaries;
potential stop prefixes are withheld. The stop itself is excluded from visible
output. A partial stop prefix is emitted when generation ends without a match.
Usage counts sampled tokens, including EOS and tokens hidden by a matched stop.
Prompt usage remains the full logical input, including any cached prefix.

UTF-8 characters split across tokens are reassembled before JSON serialization.
Malformed or unfinished generated UTF-8 is replaced with U+FFFD consistently for
JSON and streaming output. Each request has at most 64 queued text events / 256
KiB, plus the engine's bounded token buffer. A slow consumer pauses draining its
own request; the inference owner never waits on a socket. Other requests continue.
Disconnects cancel work; ten-second SSE comments keep an idle stream observable.
The existing five-minute deadline includes preparation, queueing and generation.
Validation errors precede SSE headers; later failures use an error event without
a successful terminal sequence.

## Anthropic Messages

The text subset of `POST /v1/messages` accepts the actual `model` ID, required
`max_tokens`, `messages` with user/assistant roles and string or text-block
content, and optional separate `system` text or text blocks. Consecutive messages
with the same role are combined with two newlines. The final turn must be a user
turn; assistant prefilling is rejected. Generation uses the checkpoint template
with thinking disabled and shares the Chat Completions inference engine.

Use the official Anthropic client with the server root as `base_url`,
`anthropic-version: 2023-06-01` or `2026-10-21`, and `x-api-key: <QWEN35_API_KEY>` when configured.
Bearer authentication also works. Unsupported versions and beta headers are
rejected. This is a Messages adapter, not a Claude model or full Anthropic API.
The models endpoint retains the OpenAI shape.
The pinned Anthropic Python SDK 2.1.0 no longer exposes sampling keywords;
for deterministic local decoding, use `default_headers={"anthropic-version":
"2023-06-01"}` on the client and `extra_body={"temperature": 0}` on the request.
The current version rejects sampling fields, matching the
[upstream version contract](https://platform.claude.com/docs/en/api/versioning).

Version 2023-06-01 accepts `temperature` (0..1, default 1), `top_p`, `top_k`.
Both versions accept
`stop_sequences` (up to four strings), `stream`, `thinking: {"type":"disabled"}`,
and the local `cache_prompt`/`prompt_cache_key` extensions. Images, extended
thinking, cloud cache-control/TTL hints and unknown fields are rejected.
Structured output and tools use the additional fields described below.
Responses contain text blocks, `end_turn`, `max_tokens` or `stop_sequence`, and
the exact matched stop string when applicable. Errors use the Messages error
envelope, including authentication and capacity failures.

Streams use named `message_start`, `content_block_start`, `content_block_delta`,
`content_block_stop`, `message_delta` and `message_stop` events, without `[DONE]`.
They share the same UTF-8 handling, bounded queues, cancellation and deadlines
as OpenAI streams. Initial usage is provisional (zero); the final message delta
supplies authoritative cumulative usage, which the official SDK merges into the
final Message. Post-header failures use an `error` event.

Messages usage partitions the logical prompt into `input_tokens` (uncached),
`cache_read_input_tokens` and `cache_creation_input_tokens`. Creation counts only
newly computed tokens successfully retained; extending a restored prefix does
not count restored tokens twice. These describe the local bounded cache, not
cloud billing or persistence. `output_tokens` includes withheld stop/EOS tokens.
`POST /v1/messages/count_tokens` accepts `model`, `messages`, optional `system`
and `tools`; the local `tool_choice` and `output_config` extensions let clients
count exactly the same formatting as generation. Supply the same values when
using tools or a schema (through SDK `extra_body` where needed). It returns the
full formatted input count without inference or cache mutation.

## Structured output and tools

Chat and raw Completions accept `response_format` with `type: "json_object"`,
`"json_schema"` or `"text"`. JSON Schema mode also requires
`json_schema: {"name":"response","strict":true,"schema":{...}}`; description
and strict are optional. Messages accepts
`output_config: {"format":{"type":"json_schema","schema":{...}}}`.
The schema is enforced even when `strict` is false. Chat/Messages add a formatting
instruction to the Qwen system turn. Raw prompts retain their supplied formatting.

The engine masks tokens before greedy or probabilistic sampling and accepts EOS
only after a complete JSON value. This includes tokens spanning several grammar
symbols, split UTF-8 and escaped Unicode surrogate pairs. Plain text generation
keeps its existing optimized path; constrained requests require full logits.
No Python or grammar service is needed at runtime. Text stop sequences cannot be
combined with constrained output. A length-limited JSON response can be incomplete;
check the finish reason before parsing it. Constraints guarantee structure, not
the factual correctness of generated values.

The supported JSON Schema subset is deliberately explicit:

- Types: `object`, `array`, `string`, `integer`, `number`, `boolean`, `null`.
- Objects require `properties` and `additionalProperties: false`; `required`
  names must be unique and defined. Optional properties may be omitted. Generated
  property order follows the schema. At most 32 properties per object.
- Arrays require an `items` schema. `minItems` defaults to 0 and `maxItems` to 64;
  both must be integers in 0..64.
- Scalar `enum` (1..64 values) and `const` must match the declared type. `anyOf`
  supports 1..16 alternatives, including explicit nullability. `title` and
  `description` are accepted text annotations.
- Limits: 32 KiB schema, 256 schema nodes, depth 12, 128-byte property names and
  1024-byte enum literals. Generated numbers have at most 18 integer digits,
  18 fractional digits and a two-digit exponent, keeping them representable by
  ordinary JSON clients. Enum numeric literals use their explicit JSON spelling.
- Unknown keywords are rejected, including `$ref`, regex patterns, numeric
  ranges, string-length constraints, `allOf`, `oneOf` and arbitrary properties.
  General `json_object` mode supports nested JSON containers up to eight levels.

Chat tools use `tools: [{"type":"function","function":{"name":...,"description":...,
"parameters":...}}]`. Parameters require an object schema from the subset above.
Optional `strict` must be boolean. Tool choice supports `auto`, `none`, `required`
or `{"type":"function","function":{"name":"..."}}`. At most eight uniquely named
tools are accepted; names use 1..64 ASCII letters, digits, underscores or hyphens.
The server returns at most one call per turn. An explicit `parallel_tool_calls`
must be false.

Messages uses `tools: [{"name":...,"description":...,"input_schema":...}]` and
`tool_choice` types `auto`, `none`, `any`, or `tool` with a name. The optional
`disable_parallel_tool_use` boolean is accepted; this engine always emits at
most one call. Tool mode cannot be combined with a separate output schema.

The checkpoint template receives structured tool definitions, assistant calls
and tool results. The output grammar follows the Qwen3.5 `<tool_call>`,
`<function=…>` and `<parameter=…>` format. The server maps a completed call
to OpenAI `tool_calls` with JSON-string arguments or Anthropic `tool_use` blocks
with object input. Required and named choices force a call. Automatic choice
also accepts ordinary text, including text before a call. Fine-tunes with a
different tool output syntax can use unconstrained text generation.

Return the assistant call in history followed by OpenAI `role: "tool"` with its
`tool_call_id`, or an Anthropic user `tool_result` block with `tool_use_id` and
text content. Anthropic `is_error` is preserved in the model-visible result.
Unknown, duplicate and unanswered call IDs are rejected. Calls execute in the
client; the server never executes a function.

Tool-mode streams buffer output until the grammar validates it, then emit text
and any completed call in the requested SDK format. Heartbeats and cancellation
remain active. An incomplete call at the output limit returns
`length`/`max_tokens` without a callable tool.

## Automatic prefix cache

`--prefix-cache auto` is the default; `--prefix-cache off` disables automatic
lookup/builds. `--prefix-cache-mib N` bounds retained hybrid snapshots (default
256 MiB, range 1..4096). This is a cache budget, not a total process-memory limit.
The cache evicts the least recently used unpinned entries. When a request
extends a restored checkpoint, it releases its own old handle so the new
checkpoint can take its place within the same budget. Other handles keep
their entries pinned. A full budget falls back to ordinary processing.

Matching uses exact token IDs, model identity, prefill configuration and namespace.
The server restores the longest retained match. Raw completions may retain the
full prompt. Chat builds a system prefix or the conversation preceding the last
user turn; single-turn requests without system instructions retain that user
turn. Within a tool loop, a request ending in a tool result can checkpoint
the conversation through that result. The next request can restore it and
publish a longer checkpoint.
Automatic builds require at least 32 tokens. Concurrent matching builds share
one leader. Snapshots retain FP16 KV, private FP32 recurrent/convolution state
and the final hidden state; a cached sampled answer is never reused.
MTP snapshots also retain the sidecar's FP16 attention KV. The cache budget
includes those rows. Lookup and single-flight builds match the draft snapshot
kind, so a server can interleave MTP and ordinary requests in one namespace.

On completion routes, `cache_prompt: false` bypasses automatic reuse/builds for
that request. `prefix_tokens` remains an explicit override: a positive value
requests exactly that prefix length, and zero disables caching for the request.
The explicit override remains available when automatic caching is off.
`prompt_cache_key` optionally partitions completion caches using a 1..64-byte
UTF-8 string; different keys cannot share snapshots. Keys are optimization
namespaces, not access credentials. qev has a separate namespace.

`usage.prompt_tokens_details.cached_tokens` reports actual restored tokens;
`prompt_tokens` still counts the full logical prompt. SSE includes these details
in its optional final usage chunk. `/v1/cache` reports the last completed
request's aggregate snapshot: entries, bytes, budget, hits, restored tokens,
builds, evictions and budget fallbacks. This authenticated endpoint contains no
prompt text. Cache entries are process-local and disappear on restart.

## qev decision API

With the calibrated [qev-0.8b model](https://huggingface.co/danmoreng/qev-0.8b-H128-Q4-G32-DOT4),
the same executable also serves **`POST /v1/systemone`**. A supported
`qev_config.json` in the model directory enables the route and validates the
255 label IDs and the publisher's temperatures. The model ID becomes
`qev-0.8b-H128-Q4-G32-DOT4`. A directory without that configuration retains
the original completion behavior and rejects decision requests.

```sh
qwen35_cpu_server --model-dir models/qev --threads 8 --decision-encoding per_question
curl http://127.0.0.1:8080/v1/systemone -H 'Content-Type: application/json' -d '{
  "state": {"ticket": "My card was charged twice. Please refund the duplicate."},
  "encoding": "per_question",
  "questions": {
    "dept": {"type": "choice", "instructions": "Which team should handle this?",
             "criteria": {"billing": "payments and refunds", "tech": "bugs", "sales": null}},
    "urgent": {"type": "noul", "instructions": "Is this urgent?"}
  }}'
```

The request accepts `state` (string/object/array), `questions` (1..32 named
questions), optional `encoding`, and optional `model` (accepted and ignored,
as in the publisher's System One API). Other top-level fields are rejected.
Question and option insertion order is preserved. Structured states use the
publisher's JSON spacing and index annotation for arrays of eight or more
elements. Tokenization uses NFC and the model's Qwen regex variant.

| Question type | Criteria | Answer |
| --- | --- | --- |
| `choice` | 2..255 named options and optional descriptions | `choice`, `probabilities`, `confidence`, `x_p_max`, `certainty` |
| `noul` (`bool` alias) | optional `true`/`false` descriptions | `noul`, the probability of true |
| `score` | 2..10 ordered level descriptions | `score` (expected index), `legend`, `probabilities`, `confidence`, `x_p_max`, `certainty` |

Choice lists and numeric-keyed score legends are also accepted. A noul question
may omit instructions if its true/false criteria describe the question.
Probabilities and confidence fields are rounded to four decimal places; expected
scores to two. `confidence` follows TypeSafe's rescaling formulas and differs
from the top probability `x_p_max`; `certainty` is normalized entropy reduction.
The response contains `model`, `answers`, `usage: {input_tokens, output_tokens: 0}`
and `latency_ms` (including queueing from admission). This endpoint returns
probabilities rather than generated text. It has the same authentication,
deadline, disconnect cancellation and capacity bounds as completions.

Encoding defaults to **`joint`** for compatibility; `--decision-encoding
per_question` changes that default. `per_question` matches the published Q4
evaluation and gives every question its own independent state/question prompt.
`joint` includes previous question/answer-slot prefixes, without inserting
predicted letters. Each answer slot is evaluated separately, with automatic
reuse of the exact common token prefix of the request's prepared rows when it
meets the cache threshold. Matching snapshots can also serve subsequent requests.
This does not read multiple slots in one forward. `cache.read_tokens` reports
the sum of actual prefix restores across the questions.
Joint `usage.input_tokens` counts the full logical prompt
once; per-question usage sums the independent rows. Computed tokens can exceed
logical joint usage because of the separate prefix evaluations.

The native path projects only the required option-letter logits for supported
Q4 DOT4 output weights; it performs **zero decode forwards**. Other supported
head formats retain full-vocabulary projection. `--decision-head full` forces
the original readout for comparison; the default is `auto`. Completion requests
keep their existing vocabulary/sampling behavior.

For a qev server, `--prefill-chunk auto` measures the adaptive schedule and a
fixed chunk of 128 before listening. It uses the selected-option head, one
resident request and the final inference thread count. The probes use 74 and
128 synthetic input tokens (capped at the context limit), one warmup and three
samples for each setting. A fixed chunk wins if its median decisions/s is more
than 3% higher at every probed length. Smaller contexts below 74 tokens and
full-head models retain the adaptive schedule. `GET /v1/batch-config` reports
the rates in `chunk_probes` and the chosen `prefill_chunk` (0 means adaptive).
The selected schedule applies to that server instance. Omitting the flag uses
the adaptive 64/128-token schedule. A numeric value from 1 to 2048 directly
sets the chunk size.

Multiple decision requests share one inference executor. The default processes
one request's chunk per scheduler step. `--prefill-requests N` lets eligible
selected-readout requests share matrix operations during prefill. Cache hits
and the number of simultaneously ready requests affect how often this happens.
See the [decision tuning measurements](benchmarks/decision-tuning-9955hx3d.md)
for the single-request and concurrent profiles on the Ryzen 9 9955HX3D.

The requested prefill chunk is an upper bound. The engine also ends a chunk at
token 128 so that the first queries retain the row-attention kernel used by the
default configuration. For example, a 579-token input with chunk 512 is processed
as 128, 384 and 67 tokens. Explicit prefix construction and restored requests
follow the same boundaries. Larger chunks previously moved the first queries to
tiled attention and could change probabilities and scores. See the
[AVX2 tuning study](benchmarks/qev-avx2-cloud.md) for the regression and measurements.

The existing body/context bounds apply. Rendered state <=64 KiB and evaluated
input tokens <=131,072 per request; context overflow fails explicitly, without
truncation. Images, streaming and chat templates are unsupported for decisions.
The server uses only native CPU code; Python and decider-ai are optional test
dependencies. Build the engine revision named by the model card, since earlier
release binaries predate this endpoint.

## Sampling (v0.1.1+)

The server uses the engine's existing per-request sampler. Defaults above retain
v0.1.0 greedy behavior; a positive temperature enables sampling.

| Field | Valid values | Meaning |
| --- | --- | --- |
| `temperature` | finite, >=0 | <=1e-6 selects greedy; otherwise scales logits |
| `top_k` | integer 0..248320 | 0 disables Top-K filtering |
| `top_p` | finite, (0,1] | nucleus probability cutoff; 1 disables it |
| `repetition_penalty` | finite, >=1 | 1 disables the penalty |
| `seed` | integer -1..4294967295 | -1 uses a random seed; otherwise seeds a private RNG |

The engine applies repetition penalty to tokens in the prompt and generated
history, then temperature, Top-K and Top-P before drawing a token. Requests do
not share RNG state. A fixed seed reproduces sampling with the same inputs,
settings, executable and CPU path; it is not a cross-platform bitwise guarantee.
Mixed greedy/sampling batches use full logits; eligible all-greedy batches keep
the optimized vocabulary reduction. `top_k`, `seed` and `repetition_penalty` are
API extensions, not standard OpenAI completions parameters.

## Bounds and lifecycle

- Default 16 residents and decode batch limit 16; `--residents` accepts 1..64.
- At most 64 outstanding service jobs; HTTP pool has 32 workers and 64 queued sockets.
- Request body <=256 KiB except batch file uploads; raw prompt <=64 KiB; context includes generated inputs.
- Default context 8192, adjustable with `--max-context`.
- Five-minute HTTP request deadline; disconnects and deadlines cancel engine work
  at a scheduler boundary. JSON responses accumulate output up to the requested
  token limit; SSE delivers incremental text. Finished request records are released.
- Process termination drops in-flight requests; there is no persistent queue or
  graceful rolling-restart protocol in this initial server.

Invalid requests return 400, failed authorization 401, service capacity exhaustion
429, runtime failures 500 and deadline expiry 504. A saturated transport queue may
close new sockets before they reach the API. Generated text is not logged.

Default binding is `127.0.0.1:8080`. `--host` and `--port` are configurable.
Set `QWEN35_API_KEY` to require `Authorization: Bearer <key>` on every route.
Non-loopback binding requires that variable. The native listener is plain HTTP;
use a TLS reverse proxy for network deployments. No CORS policy is enabled.

## Test

```sh
python tests/server_smoke.py --server build/qwen35_cpu_server --model-dir models/qwen3.5-0.8b
```

This real-model test checks authorization, invalid input, repeated requests and
exact greedy/seeded-sampling parity for concurrent prefix-sharing requests,
mixed sampling settings, Top-K/Top-P edge cases and CLI/API sampling agreement. It is
a correctness test, not a server throughput measurement.

## Drafting for text completions

OpenAI-style `/v1/chat/completions` and `/v1/completions` accept optional `"draft": "ngram"` or `"draft": "input"`. Omit the field or use `"off"` for ordinary decode. Input mode can additionally receive a `draft_source` string containing the document to follow. Keep the document in the prompt as well. Both methods support greedy and temperature sampling with the usual top-k, top-p, repetition penalty and seed. The native CLI exposes the same modes via `--draft off|ngram|input` and `--draft-source-file FILE`.

For each request, sustained low acceptance pauses lookup for the next 16
generated tokens. Ordinary decode continues during the pause and automatic
lookup resumes afterward. See the [drafting API](cpu-engine-api.md#text-drafts)
for the acceptance threshold and counters.

For experimental learned 4B drafting, start the server with the native 4B
checkpoint and `--mtp-weights FILE`. Select `"draft": "mtp"` in a chat
completion or a raw completion. `--mtp-depth 1|2|3|4` sets the proposal limit
and defaults to one.
Chat function tools and JSON schemas use a private grammar for proposed
tokens and the target grammar for the emitted result. When an MTP sidecar is
loaded, `GET /v1/cache` includes cumulative draft counters and the four
verification counts indexed by one through four proposals.
See the [4B packing instructions](quantization.md#optional-4b-mtp-weights).

In this request the transcript supplies candidate tokens and the target model decides which corrections to make.

```json
{"model":"Qwen3.5-0.8B-H128-Q4-G32-DOT4","messages":[{"role":"user","content":"Korrigiere nur Schreibfehler: Ich schicke dir die Unterlagen per Mehl zu."}],"draft":"input","draft_source":"Ich schicke dir die Unterlagen per Mehl zu.","temperature":0}
```

See the [engine API](cpu-engine-api.md#text-drafts) for dispatch, validation and memory behavior.

See [fine-tune serving](finetunes.md) for `POST /v1/classify`, classifier profiles and shared prefill.
