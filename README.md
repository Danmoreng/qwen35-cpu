# Qwen3.5 CPU

Run Qwen3.5 text models locally on an x86 CPU. The engine ships as a native
HTTP server and a raw text-completion CLI, and can be embedded in a C++20
application. The ready-to-run **0.8B** checkpoint uses **425 MB of weights**.
Source builds also run **2B** and **4B** text models. The engine selects AVX2
or AVX-512 kernels at runtime on supported CPUs.

[Quick start](#download-and-run) · [CPU benchmarks](#cpu-performance) · [Documentation](docs/README.md)

## What you can run

- **Local chat and tools.** Connect a chat client through OpenAI-compatible Chat Completions or Anthropic Messages for streaming text, constrained JSON and tool calls. Current source also supports OpenAI Responses and offline JSONL batches. [API and supported fields](docs/server.md)
- **Concurrent requests.** The C++ engine batches decode across active requests. Exact-prefix caching reuses work when requests share a system prompt or conversation history. [Embedding and scheduling](docs/cpu-engine-api.md)
- **Larger models and fine-tunes.** Pack 2B or 4B text weights in H128/Q4-G32-DOT4 format. The 4B profile supports optional multi-token prediction (MTP) for speculative decoding. Fine-tunes can expose classification or decision endpoints. [Packing guide](docs/quantization.md) · [4B agent benchmarks](docs/benchmarks/qwen35-4b-pi-agent.md) · [Fine-tune guide](docs/finetunes.md)
- **Browser decisions.** Try the separate qev-0.8b fine-tune running on your own CPU via WebAssembly. [Live playground](https://huggingface.co/spaces/danmoreng/qev-0.8b-cpu-playground)

## Download and run

Download and extract the [Windows x64 or Linux x64 release](https://github.com/Danmoreng/qwen35-cpu/releases).
From the extracted directory, fetch the pinned [0.8B model](https://huggingface.co/danmoreng/Qwen3.5-0.8B-H128-Q4-G32-DOT4)
and start the server. The downloader verifies each file. Linux binaries require
glibc 2.35 or newer.

**Linux**

```sh
bash ./download-model.sh danmoreng/Qwen3.5-0.8B-H128-Q4-G32-DOT4 f6cb5cf04a9094670f9041578a3f4e44b94a1395
./qwen35_cpu_server --model-dir models/qwen3.5-0.8b --threads 8
```

Send a chat request from a second terminal.

```sh
curl http://127.0.0.1:8080/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"messages":[{"role":"user","content":"Name three planets."}],"max_tokens":32,"temperature":0}'
```

<details>
<summary>Windows PowerShell commands</summary>

```powershell
./download-model.ps1 -Repo danmoreng/Qwen3.5-0.8B-H128-Q4-G32-DOT4 -Revision f6cb5cf04a9094670f9041578a3f4e44b94a1395
./qwen35_cpu_server.exe --model-dir models/qwen3.5-0.8b --threads 8
```

Send a chat request from a second PowerShell window.

```powershell
Invoke-RestMethod http://127.0.0.1:8080/v1/chat/completions -Method Post -ContentType application/json -Body '{"messages":[{"role":"user","content":"Name three planets."}],"max_tokens":32,"temperature":0}'
```

</details>

<a id="qwen35-08b-performance"></a>

## CPU performance

Measured on the **AMD Ryzen 9 9955HX3D** laptop with eight threads pinned to
physical cores 0–7. Rates are median **tokens/s** from three runs after one
warmup. Prefill measures one prompt. Decode uses 512 input and 128 output
tokens per request. B=16 is total throughput across 16 private requests.

![Decode and prefill throughput for native H128/Q4-G32-DOT4, ik_llama.cpp IQ4_K_R4 and llama.cpp Q4_0 at 0.8B, 2B and 4B.](docs/results/ryzen-large-2026-11-20/cpu-performance.png)

| Model | Engine / checkpoint | Tensor MB | Decode B=1 | Decode B=16 | Prefill 512 | Prefill 4,096 |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| 0.8B | ik_llama.cpp IQ4_KS_R4 | 401.4 | 121.5 | 400.7 | 3,630.8 | 3,144.8 |
| 0.8B | ik_llama.cpp IQ4_K_R4 | 424.9 | 113.4 | 388.9 | 3,629.5 | 3,141.2 |
| 0.8B | **Native H128/Q4-G32-DOT4** | 424.9 | 111.0 | 575.6 | 3,052.4 | 2,670.7 |
| 0.8B | llama.cpp Q4_0 | 424.9 | 101.8 | 456.5 | 1,099.5 | 1,072.2 |
| 0.8B | llama.cpp Unsloth Q4_K_M | 521.6 | 83.3 | 389.8 | 1,106.7 | 959.4 |
| 2B | ik_llama.cpp IQ4_K_R4 | 1,060.4 | 43.6 | 245.7 | 1,737.7 | 1,636.2 |
| 2B | llama.cpp Q4_0 | 1,060.4 | 42.3 | 306.4 | 701.4 | 626.8 |
| 2B | **Native H128/Q4-G32-DOT4** | 1,060.4 | 42.2 | 382.6 | 1,504.9 | 1,414.4 |
| 4B | ik_llama.cpp IQ4_K_R4 | 2,369.0 | 18.9 | 106.5 | 707.8 | 635.6 |
| 4B | **Native H128/Q4-G32-DOT4** | 2,369.0 | 18.4 | 163.4 | 614.4 | 557.5 |
| 4B | llama.cpp Q4_0 | 2,369.0 | 18.2 | 98.3 | 195.9 | 175.7 |

At 2B and 4B, IK led single-request decode and prefill. Native delivered the
highest decode throughput across 16 requests. All three formats have the same
on-disk tensor size within each of these model sizes. Timings use fixed tokens,
FP16 KV and full-vocabulary logits and exclude loading, tokenization and HTTP.
[0.8B runs, November 19](docs/benchmarks/ryzen-9955hx3d.md) · [2B and 4B runs, November 20](docs/benchmarks/ryzen-9955hx3d-large.md)

<a id="quantization-quality"></a>

### 0.8B quantization quality

Perplexity (PPL) measures how well the model predicts the reference text.
KL measures how closely its predictions match the BF16 teacher, in nats.
Lower is better for both. Scores cover 8,192 tokens from 16 English WikiText-2
test windows. MB measures the on-disk tensor payload.

| Scoring engine / checkpoint | Tensor MB | PPL ↓ | KL to BF16 ↓ |
| --- | ---: | ---: | ---: |
| llama.cpp BF16 reference | 1,505.8 | 14.38556 | 0 |
| llama.cpp Unsloth Q4_K_M | 521.6 | 14.77209 | 0.034894 |
| ik_llama.cpp IQ4_K_R4 | 424.9 | 15.77716 | 0.068812 |
| **Native H128/Q4-G32-DOT4** | 424.9 | 15.79649 | 0.060100 |
| llama.cpp Q4_0 | 424.9 | 18.03403 | 0.144830 |
| ik_llama.cpp IQ4_KS_R4 | 401.4 | 15.86784 | 0.084944 |

The November native build reproduced every vocabulary logit from the October
quality run. The comparator scores come from that run with the same binaries
and weights. [Native verification](docs/results/readme-2026-11-19/native-quality.json) · [Quality measurements](docs/benchmarks/cpu.md#quality-and-size)

## Build from source

Use a C++20 compiler, CMake 3.24+ and Ninja. On Linux, run the following from
the repository root.

```sh
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build
ctest --test-dir build --output-on-failure
```

On Windows, run `./scripts/build.ps1` with Visual Studio 2022 C++ Build Tools
and PowerShell 7. For an engine and CLI build, set `-DQWEN35_BUILD_SERVER=OFF`
when configuring CMake. The engine is MIT-licensed. Model weights are distributed
separately under their own licenses.

[Documentation](docs/README.md) · [Model formats and calibration](docs/quantization.md) ·
[0.8B GGUF support](docs/q4km-native.md) · [Validation](docs/validation.md) · [Provenance](PROVENANCE.md)
