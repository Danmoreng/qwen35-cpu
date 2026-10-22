# Static Hugging Face CPU playground

The Space uses the same custom C++ CPU/WASM engine and calibrated qev model as
the local playground. Hosting serves app files; inference stays in a browser
worker on the visitor's CPU. No Python inference service, GPU backend or remote
decision API is involved.

## Build and preview

```sh
source .cache/emsdk/emsdk_env.sh
bash scripts/build-wasm.sh all
python scripts/prepare-qev-space.py --output dist/hf-qev-space-preview
python scripts/serve-space-preview.py --directory dist/hf-qev-space-preview
```

Open `http://127.0.0.1:8766/`. This preview serves the actual static manifest:
the model is downloaded from its immutable Hugging Face revision, not from
the local model directory. The packager uses local runtime files to verify
their hashes against that remote revision before staging. It requires all
seven runtime variants and their license files, and refuses an existing output
directory. No model weights, ONNX/GGUF comparisons, diagnostic binaries or old
baseline directories are copied into the Space. The current package is about
9.7 MB; the initial model/config/tokenizer download is about 448 MB.

The source UI retains its local-preview labels. The packager produces the
hosted labels explicitly and fails if their source anchors unexpectedly change.
The worker accepts either the original loopback manifest or the pinned public
model URLs. Manifest validation requires the exact qev checkpoint identity,
fixed runtime-file names, sizes, hashes and immutable remote URLs. Configuration
files receive browser SHA-256 checks; model tensors are checked by the C++
reader. The hosted manifest pins the same checkpoint SHA-256 as the native
release, and contains no access tokens.

The static README sets COOP/COEP/CORP through Hugging Face's supported
[`custom_headers`](https://huggingface.co/docs/hub/spaces-config-reference).
Threaded modes require cross-origin isolation. Fused-FMA and SIMD capability
probes select a supported runtime; separate single-thread bundles remain
available. Verify the actual hosted page and embedded Space, since a local
header test alone cannot establish iframe compatibility.

## Publish after validation

Commit and push finished source directly to `main`, then wait for successful
CI before publishing. Prepare a fresh package from that committed source:

```sh
python scripts/prepare-qev-space.py --output dist/hf-qev-space-release
python scripts/publish-qev-space.py --folder dist/hf-qev-space-release \
  --repo danmoreng/qev-0.8b-cpu-playground
```

The publisher requires `huggingface_hub` and local HF authentication. It checks
every staged file against `build-info.json`, rejects uncommitted source and
bundled weights, creates a public static Space and uploads with an optimistic
parent-commit check. Use `--update` only for an existing Space. It never passes
tokens through command arguments or stores them in the package.

`build-info.json` records the source revision and every app/runtime hash.
`model/manifest.json` records the separate immutable model revision, hashes,
sizes and download URLs. Model/license notices are retained. The original qev
model and examples are credited; Snake remains explicitly path-assisted.
Native model-card accuracy is not presented as a new browser accuracy result.

## Validation

`node tests/model_manifest_test.mjs` checks local/remote manifests, immutable
URLs, wrong checkpoint/repository identities, missing/duplicate/path-traversing
file names and invalid byte counts. These checks and the game scenario tests
also run in release CI. The full seven-variant build runs DOT4, DeltaNet and
activation-quantization kernel tests before copying each runtime. Actual
browser checks must additionally cover remote model loading, live inference,
cache reuse, cancellation, capabilities and the public Space headers.
