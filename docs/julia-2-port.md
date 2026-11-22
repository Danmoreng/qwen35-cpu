# Julia-2 text on the native CPU engine

Julia-2 uses the engine's Qwen3.5-0.8B text architecture. A native
feasibility probe ran the complete text forward pass with the engine's
H128/Q4-G32-DOT4 weights and Julia-2's original FP32 decision head.
Serving needs Julia-specific checkpoint import, prompt formatting and readout.
[Publisher model card](https://huggingface.co/SupersonicLabs/Julia-2/resolve/b3e610b47d140cc4fcbd00d7b58b86184a13bcff/README.md) · [Publisher text configuration](https://huggingface.co/SupersonicLabs/Julia-2/resolve/b3e610b47d140cc4fcbd00d7b58b86184a13bcff/trunk/config.json)

## Architecture and weights

| Component | Parameters |
| --- | ---: |
| Qwen3.5 text trunk | 752,393,024 |
| Decision head | 261,375 |
| Vision | 100,592,896 |
| Audio | 207,814,064 |
| Full checkpoint | 1,061,060,959 |

The text trunk has width 1,024, 18 causal Gated DeltaNet layers and six
gated full-attention layers. The full-attention layers use eight query
heads, two KV heads and head dimension 256. Native prefill implements
these operations. Julia projects the final token's normalized hidden
state through its learned 255 × 1,024 matrix and bias. It assigns option
labels by position and applies softmax to the entries for the supplied
options.
[Publisher prompt and labels](https://huggingface.co/SupersonicLabs/Julia-2/resolve/b3e610b47d140cc4fcbd00d7b58b86184a13bcff/runtime/julia2/prompt.py) · [Publisher trunk and readout](https://huggingface.co/SupersonicLabs/Julia-2/resolve/b3e610b47d140cc4fcbd00d7b58b86184a13bcff/runtime/julia2/trunk.py) · [Native prefill](../src/runtime/prefill.inl)

The probe copied the checkpoint's 320 text tensors byte for byte,
renamed the text tensor prefix to the Qwen convention and used the
publisher's nested Qwen configuration. The existing packer verified a
424,964,864-byte trunk artifact using uncalibrated MSE16. The original
FP32 head uses 1,045,500 bytes. The combined text weights occupy
426.0 MB, compared with 1,505.8 MB of original BF16 trunk and FP32 head
tensor data.
[Trunk packer](../src/q4_h128_convert_main.cpp)

## Local feasibility probe

The i7-8750H probe used six physical cores. The publisher runtime
provided the token IDs. A separate native tokenizer check matched all
30 inputs. The native probe read the final hidden state held by the
engine's prefill snapshot and applied Julia's original FP32 head. Each
prompt ran independently with zero decode steps. The publisher CPU
reference used the BF16 checkpoint expanded to FP32 for computation
and received the same token IDs.

| Sample | Julia FP32 reference correct | Native H128 correct | Same top option | Median largest option change | Maximum option change |
| --- | ---: | ---: | ---: | ---: | ---: |
| Benchmark Heaven subset (68–3,387 tokens) | 13/24 | 13/24 | 23/24 | 1.90 pp | 14.37 pp |
| Publisher fixtures (39–256 tokens) | — | — | 6/6 | 0.59 pp | 3.43 pp |

On the 24 labeled tasks, mean Brier was 0.5645 for the
FP32 reference and 0.5348 for H128. On the six publisher
fixtures, the FP32 reference matched 6 published top options.
Its largest option probability difference against the publisher's stored
values was 0.22 pp.

The sample combines the fixed 24 tasks from the earlier precision study
and two publisher fixture rows per question type. It verifies the
end-to-end forward path and records probability drift. The [complete Jev Skills GPU evaluation](benchmarks/julia2-jev-skills.md)
compares Julia and Qev on 15,154 text questions. Native serving latency remains unmeasured.
[Probe data](results/qev-julia-2-probe-2026-11-21/comparison.json) · [Artifact and binary hashes](results/qev-julia-2-probe-2026-11-21/manifest.json)

## Native serving work

1. Add a Julia-2 text import entry point that maps the trunk tensor
   prefix and verifies the Qwen3.5-0.8B profile. Store the original
   FP32 head with checks for shape and checkpoint identity.
2. Expose the final normalized hidden state after prefill. Apply the
   Julia head and implement its `choice`, `noul` and `score` results,
   including option ordering, expected level and confidence. Reuse
   the System One request envelope.
3. Port the publisher's prompt format and limits. For shared state,
   prefill the state once and fork the question suffixes using the
   existing recurrent and KV snapshots. Test the prompt's tokenization
   boundaries and compare each branch with the publisher's shared-state
   implementation.
4. Collect Julia activations for H128 calibration. Evaluate the reference
   and calibrated Q4 on the full 231-task set. Measure serial and
   shared-state latency with a warmup and repeated runs.

Vision and audio need their own towers and an input path for embeddings.
The text probe uses the engine's existing matrix and attention kernels.
[Native prefix interface](../include/qwen35x/runtime/cpu_engine.h) · [Publisher shared-state inference](https://huggingface.co/SupersonicLabs/Julia-2/resolve/b3e610b47d140cc4fcbd00d7b58b86184a13bcff/runtime/julia2/__init__.py)
