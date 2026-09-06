# Native CPU HTTP server

`qwen35_cpu_server --model-dir models/qwen3.5-0.8b --threads 8`

The directory contains `model.q35h`, config and tokenizer files. `--weights`
can override the artifact path. One owner drives the model/tokenizer and bounded
CpuEngine scheduler. HTTP workers never create separate inference executors.
The process starts listening only after the model has loaded successfully.

## API

- `GET /health`: readiness (`{"status":"ok"}`).
- `GET /v1/models`: the single supported model ID.
- `POST /v1/completions`: JSON input, complete JSON response.

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
This is a limited completions API, not complete OpenAI API compatibility.

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
