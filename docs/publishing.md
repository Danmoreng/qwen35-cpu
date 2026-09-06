# Model and binary publication

The model repository and engine repository are separate. Never add model weights
to GitHub. Hugging Face hosts the `.q35h`, tokenizer, model card, Apache-2.0 model
license, modification NOTICE, quantization provenance and SHA-256 manifest.
GitHub Releases host small Windows/Linux executable archives and checksums.

## Prepare and publish the model

```sh
python scripts/prepare-hf-package.py --source models/qwen3.5-0.8b --artifact models/qwen3.5-0.8b/model-calibrated-mse16.q35h --output dist/huggingface-mse16
```

Preparation allows only the validated artifact hash. It copies a fixed file list,
records BF16 shard hashes and refuses an existing output directory. Test the
prepared directory directly with the release server before upload. The original
BF16 source revision was not retained; source hashes are recorded instead of
inventing a revision. The model card reports the measured calibrated MSE16
quality and speed separately, with corpus limitations. Quantization provenance
includes the converter sidecar and calibration manifest. The format is unchanged.

Authenticate locally, then upload the checked package:

```sh
hf auth login
python scripts/upload-hf-package.py --folder dist/huggingface-mse16 --repo danmoreng/Qwen3.5-0.8B-H128-Q4-G32-DOT4 --update
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
