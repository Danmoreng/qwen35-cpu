# Model and binary publication

The model repository and engine repository are separate. Never add model weights
to GitHub. Hugging Face hosts the `.q35h`, tokenizer, model card, Apache-2.0 model
license, modification NOTICE, quantization provenance and SHA-256 manifest.
GitHub Releases host small Windows/Linux executable archives and checksums.

## Prepare and publish the model

```sh
python scripts/prepare-hf-package.py --source models/qwen3.5-0.8b --artifact models/qwen3.5-0.8b/model-calibrated-large-full-bc.q35h --output dist/huggingface-bc256
```

Preparation allows only the validated artifact hash. It copies a fixed file list,
records BF16 shard hashes and refuses an existing output directory. Test the
prepared directory directly with the release server before upload. The original
BF16 source revision was not retained; source hashes are recorded instead of
inventing a revision. The model card reports the measured 256-document calibrated
quality and speed separately, with corpus limitations. Quantization provenance
includes converter, calibration and covariance sidecars, plus corpus provenance. The format is unchanged.

Authenticate locally, then upload the checked package:

```sh
hf auth login
python scripts/upload-hf-package.py --folder dist/huggingface-bc256 --repo danmoreng/Qwen3.5-0.8B-H128-Q4-G32-DOT4 --update
```

The upload script requires `huggingface_hub`, validates the exact manifest and
model card, and uploads its files in one commit. Without `--update` it creates
a new public repository and refuses to reuse an existing one. With `--update`
it requires an existing repository and checks its parent commit when uploading,
so a concurrent update cannot be silently overwritten. Old artifacts remain
accessible through their immutable revisions.
Do not put access tokens in source files, command arguments or GitHub model vars.

Record the returned model commit, replace `MODEL_COMMIT` in the README and set
GitHub repository variables `HF_MODEL_REPO` and `HF_MODEL_REVISION`. The revision
must be the immutable 40-character Hugging Face commit, not `main`.

## CI and releases

### qev decision package

The qev fine-tune has its own repository and fixed evaluated artifact. Prepare it
with `scripts/prepare-qev-hf-package.py`, passing its source directory,
`model-cal-1024-block128.q35h`, a fresh output directory and `--license-file`
containing the standard Apache-2.0 text. The upstream model declares that license
in its card but does not ship a LICENSE file. The preparation script preserves
the decision configuration and adds source/calibration/evaluation provenance.
It rejects any artifact SHA256 other than the frozen final candidate. No chat
template is needed for its plain System One prompts.

Test the prepared directory with `tests/systemone_server_smoke.py` and the native
server before using the existing uploader with
`--repo danmoreng/qev-0.8b-H128-Q4-G32-DOT4`. The uploader recognizes this exact
qev package profile while preserving the original model's file allowlist.
Use a current source build or a native release that includes the decision API
(v0.2.0 or newer). The base/chat HF model
and its existing published revision are unchanged.

`.github/workflows/build-release.yml` runs on pushes, pull requests and manual
dispatch. It builds and runs CTest on Windows Server 2022/MSVC and Ubuntu
22.04/GCC 12. Model tests run when the public pinned model variables are set.
Release tags require those variables and both platforms' model/HTTP checks.

Successful build jobs create workflow artifacts for review. A pushed `v*` tag
also creates a GitHub Release containing ZIP/tar.gz archives and SHA-256 files,
only after both platforms pass. There is no automatic version bump or tag push.
The release job alone receives `contents: write`; build jobs are read-only and
all third-party actions are pinned to commits. No HF token is needed for CI
because it only downloads the public model.

```sh
# After reviewing a successful main/branch CI run and setting the model vars:
git tag v0.1.0
git push origin v0.1.0
```

To package locally after a build:

```sh
python scripts/package-release.py --version 0.1.0-dev --platform windows-x64
# Linux: use --platform linux-x64
```

Archives include the native server, text CLI, download scripts, documentation,
engine license and dependency licenses. Windows uses `/MT` and requires only OS
DLLs. Linux server builds link libstdc++/libgcc statically and target glibc 2.35+.
CPU ISA-specific translation units retain runtime dispatch; release builds do not
set `-march=native` globally. No weights, CUDA components or Python runtime are
bundled. Release packaging does not claim AVX2 hardware validation solely because
the build succeeded on a GitHub runner.

Sources: [Hugging Face upload guide](https://huggingface.co/docs/huggingface_hub/guides/upload),
[GitHub workflow permissions](https://docs.github.com/en/actions/reference/workflows-and-actions/workflow-syntax).

## Evidence size policy

Keep raw benchmark profiles, per-token dumps, activation manifests and archives
under ignored `benchmarks/`. Commit only code, narrative reports and compact
aggregate CSV/JSON results. Do not copy complete run directories into Git.
The [evidence inventory](results/README.md) lists retained published data;
[Git history](history.md) preserves superseded studies. Runtime archives contain
Markdown guides and reports, excluding `docs/results/`; evidence links are for
the matching source checkout on GitHub.

## Current model publication

The standard H128/Q4-G32-DOT4 checkpoint is published at immutable Hugging Face revision
`f6cb5cf04a9094670f9041578a3f4e44b94a1395`, with model SHA256
`013fbfaa03760e759181301ddaf964bb5c200c50c50617afe72557fd65bcbf0a`.
The public package was downloaded through the documented checksum-verifying
script and passed HTTP validation, greedy/seeded sampling, mixed batches and
prefix parity. README and CI pin this revision. Model metadata is compact;
full activation-input manifests remain local, outside the engine repository.

## Published qev decision model

The calibrated 425 MB qev artifact is public at
[`danmoreng/qev-0.8b-H128-Q4-G32-DOT4`](https://huggingface.co/danmoreng/qev-0.8b-H128-Q4-G32-DOT4),
immutable model revision `fc805be27e1409630b14b233ad7aac1a6dbb213f`. The model card pins
engine source `be5b9fa510b050c99faef4af83e58585093dc3aa`; the server change is
available in [PR #1](https://github.com/Danmoreng/qwen35-cpu/pull/1).
The anonymous public download passed every SHA256SUMS entry. Its artifact hash
is `e7d61e7b191de6c2afb8663e8a1f8c5f4d52315e9492db7a690708d744f8dc69`.
This publication is separate from the existing base/chat model and does not
change its CI model variables or existing binary releases. The decision endpoint is included in native v0.2.0 and newer releases.

## Native v0.2.0 and current qev documentation

The tested native decision server is available in
[v0.2.0](https://github.com/Danmoreng/qwen35-cpu/releases/tag/v0.2.0), with Linux
and Windows archives and SHA-256 files. Source tag: `eaf2425b56d11000ca2744e35e859d9381382a6b`.
Linux and Windows release CI both passed, including full/selected HTTP parity.
Current comparisons use separately confirmed llama.cpp batch tuning; see
[the HTTP study](benchmarks/decision-http.md).

The qev model repository's metadata-only revision
`eb167df11cbd6130513fedfcfdfb2ca11180b2de` updates README, serving evidence in
`quantization.json`, and their checksum entries. The checkpoint, tokenizer,
calibration and decision configuration are unchanged from the earlier immutable
revision. Anonymous download checks verified the new metadata checksums and
unchanged model SHA-256. The model card pins source `0b48efc1aee5a793048822aacffe4ae76d595118`.

Development now validates locally and pushes finished work directly to `main`;
pull requests are only used when explicitly requested. CI runs after each push
and must succeed before publication. Force-push and deletion of `main` remain
blocked. For the subsequent browser deployment, see
[static Space packaging](browser/hosting.md).
