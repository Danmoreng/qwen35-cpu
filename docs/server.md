# Native CPU HTTP server

`qwen35_cpu_server --model-dir models/qwen3.5-0.8b --threads 8`

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
- `POST /v1/completions`: JSON input, complete JSON response.
- `POST /v1/chat/completions`: text messages, native Qwen formatting, complete JSON response.

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
  "n": 1,
  "prefix_tokens": 0
}
```

Only `prompt` is required. Model name is optional; if supplied it must match.
The other fields use the defaults above. `prefix_tokens` is a custom extension:
it requests automatic single-flight reuse of that many leading prompt tokens.
Matching is by exact token IDs, with private recurrent state per request and
shared immutable KV pages. The server is single-tenant; all requests share the
same trust namespace and optional API key.

Responses contain `id`, `object`, `created`, `model`, one text choice,
`finish_reason` and token `usage`. Stop IDs come from the model tokenizer.
Prompt text must already contain any desired chat template. Unsupported fields,
multiple completions and streaming are rejected explicitly.
The supported compatibility surface is documented below; it does not implement
the entire OpenAI platform API.

## Chat completions

`POST /v1/chat/completions` accepts `messages` instead of `prompt`. Messages use
`system`, `developer`, `user` or `assistant` roles, with string content or arrays
of `{"type":"text","text":"..."}` parts. At least one user query is required.
Leading system/developer messages are combined into the Qwen system turn with
two newlines between them. Such instructions are rejected after the conversation
has begun. Images, tool messages and unsupported message fields are rejected.

The native formatter matches the pinned Qwen3.5 text chat template with thinking
disabled for generation. Assistant history may contain `reasoning_content`;
historical reasoning is handled according to that template. Clients supply
messages, not already formatted prompts. The formatter does not execute model
repository code or require Python/Jinja at runtime.

Supported generation fields are `model`, `max_completion_tokens` (or `max_tokens`,
but not both), `temperature`, `top_p`, `n=1`, `stream=false`, and the existing
`top_k`, `seed`, `repetition_penalty`, `prefix_tokens` extensions. Output defaults
to 128 tokens. Chat defaults to temperature 1, top_p 1 and top_k 0; raw
completions retain the defaults above. Set temperature 0 for greedy decoding.
Optional numeric fields may be null to use their defaults.

Responses use `object: "chat.completion"`, an assistant `message`, finish reason
and token usage. Unsupported fields fail explicitly with an OpenAI-style error.
Stop strings, streaming, structured output, tool calling and the Responses API
are not supported yet. The OpenAI SDK works with a custom server `base_url` and
the actual model ID from `/v1/models`; API compatibility does not imply OpenAI
model capabilities.

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
predicted letters. This initial native implementation evaluates each causal
prefix separately. It does not read multiple slots in one forward or reuse
decision prefixes. Joint `usage.input_tokens` counts the full logical prompt
once; per-question usage sums the independent rows. Computed tokens can exceed
logical joint usage because of the separate prefix evaluations.

The native path projects only the required option-letter logits for supported
Q4 DOT4 output weights; it performs **zero decode forwards**. Other supported
head formats retain full-vocabulary projection. `--decision-head full` forces
the original readout for comparison; the default is `auto`. Completion requests
keep their existing vocabulary/sampling behavior. `--prefill-chunk N` selects
an explicit 1..2048-token chunk; omit it to retain the adaptive 64/128-token
default. Multiple decision
requests share the existing bounded scheduler. Prefill chunks use one executor
and are time-sliced between requests; there is no cross-request prefill batch.
Decode batching therefore does not accelerate this workload. See the
[HTTP and concurrency measurements](https://github.com/Danmoreng/qwen35-cpu/blob/3680efb33bf6975ce6722d64d20c641cb7490b06/docs/qev-systemone-api-2026-10-21.md).

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
- Request body <=256 KiB; raw prompt <=64 KiB; context includes generated inputs.
- Default context 8192, adjustable with `--max-context`; no automatic thread tuning.
- Five-minute HTTP request deadline; disconnects and deadlines cancel engine work
  at a scheduler boundary. Output is accumulated up to the requested token limit
  because responses are non-streaming. Finished request records are released.
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
