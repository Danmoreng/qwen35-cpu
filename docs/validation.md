# Extraction validation — September 6, 2026

Source: qwen35x `f766837`; standalone Windows/MSVC Release build on Ryzen 9
9955HX3D. The CMake project enables C++ only, with no CUDA dependency or GPU
runtime stubs. BF16 source parsing is retained for offline checkpoint conversion.

The eight CTest cases cover Q4/DOT4/H128/Q8 kernels, tiled attention, DeltaNet,
artifact serialization and KV pages. The optional real-model scheduler test
with `--pages` passed, including full-logit parity, queueing, cancellation,
backpressure, single-flight prefix reuse and independent paged branches.

An extraction regression ran both the frozen original `shared-pages.exe` and
the new `qwen35_cpu_bench` through the sequential runner: P=512, N=16, eight
threads, affinity `0xffff`, greedy decoding, repetition penalty 1. B=1 used
cold contiguous state; B=4 used a registered 511-token prefix and shared pages.
All output token arrays matched exactly. One run per configuration was used
for regression checking; these timings are not published as performance evidence.

The text CLI also loaded successfully with only config/tokenizer files and the
packed artifact, with no BF16 shards or safetensors index in its model directory.
The text CLI generated output from a raw prompt. An eight-position teacher-forced
dump contained all 248,320 vocabulary logits per position; self-comparison gave
zero KL divergence and identical target NLL. This checks export/reader alignment,
not accuracy against BF16 or another engine.

Validated artifact SHA-256:
`e73de30bf646dee502dd5e519939221f7d2b60068ccbbe16dc1701597919c42f`.
Model weights and disposable output/logit files remain ignored local assets.
No new quantization quality matrix or llama.cpp multi-request comparison has
been measured in this standalone repository yet.

## Published server and release validation

The [v0.1.0 release workflow](https://github.com/Danmoreng/qwen35-cpu/actions/runs/34033624187)
passed on Windows Server 2022/MSVC and Ubuntu 22.04/GCC 12. Both platforms built
the CPU targets, passed all eight CTest cases, downloaded the public Hugging Face
model with checksum verification, passed the real-model scheduler test and tested
HTTP authentication, input validation and concurrent greedy/prefix parity from
the extracted release archives. The release job verified archive checksums before
publishing the ZIP and tar.gz assets.

The public model revision is `cc7df08da7ef7ac15db62e80b4eda85e19a143da` in
`danmoreng/Qwen3.5-0.8B-H128-Q4-G32-DOT4`. A local test also downloaded this exact
revision and ran the GitHub-built Windows binary against it. Windows dependency
inspection found only `KERNEL32.dll` and `WS2_32.dll` for the server executable.

These are correctness and distribution checks, not new performance or quality
measurements. The Intel i7-8750H laptop was unreachable during this release work;
CI passing does not replace validation on that particular machine.
