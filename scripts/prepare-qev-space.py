#!/usr/bin/env python3
"""Stage an allowlisted static HF Space. No upload and no copied model weights."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import urllib.request

REPO = 'danmoreng/qev-0.8b-H128-Q4-G32-DOT4'
REVISION = 'eb167df11cbd6130513fedfcfdfb2ca11180b2de'
MODEL_SHA = 'e7d61e7b191de6c2afb8663e8a1f8c5f4d52315e9492db7a690708d744f8dc69'
MODEL_FILES = ('model.q35h', 'config.json', 'tokenizer.json', 'tokenizer_config.json',
               'vocab.json', 'merges.txt', 'qev_config.json')
WEB_FILES = ('index.html', 'style.css', 'app.js', 'worker.js', 'capabilities.js',
             'scenarios.js', 'model-manifest.js')
MODES = ('scalar', 'simd', 'threads', 'relaxed', 'relaxed-threads', 'fma', 'fma-threads')
LICENSES = ('qwen35-cpu.txt', 'decider-ai.txt', 'ggml.txt', 'utf8proc.txt',
            'nlohmann-json.txt', 'emscripten.txt')


def sha(path):
    with path.open('rb') as stream: return hashlib.file_digest(stream, 'sha256').hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runtime-dir', type=Path, default=Path('dist/wasm'))
    parser.add_argument('--model-dir', type=Path, default=Path('dist/qev-h128-public-final'))
    parser.add_argument('--model-revision', default=REVISION)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists(): raise ValueError('Choose a new output directory')
    if len(args.model_revision) != 40 or any(c not in '0123456789abcdef' for c in args.model_revision):
        raise ValueError('Pin an immutable model revision')
    base = f'https://huggingface.co/{REPO}/resolve/{args.model_revision}'
    with urllib.request.urlopen(base + '/SHA256SUMS', timeout=60) as response:
        checksums = dict((name, digest) for line in response.read().decode().splitlines()
                         for digest, name in [line.split('  ', 1)])
    if checksums.get('model.q35h') != MODEL_SHA: raise ValueError('Unexpected published checkpoint')
    files = []
    for name in MODEL_FILES:
        path = args.model_dir / name
        digest = sha(path)
        if checksums.get(name) != digest: raise ValueError('Runtime file differs from pinned model: ' + name)
        files.append({'name': name, 'bytes': path.stat().st_size, 'sha256': digest, 'url': base + '/' + name})
    # Never recursively copy dist/wasm: it can contain large benchmark models,
    # diagnostic bundles, old baselines and profiling captures.
    args.output.mkdir(parents=True)
    for name in WEB_FILES: shutil.copyfile(args.runtime_dir / name, args.output / name)
    for mode in MODES:
        (args.output / mode).mkdir()
        for name in ('qev_wasm.js', 'qev_wasm.wasm', 'toolchain.txt'):
            shutil.copyfile(args.runtime_dir / mode / name, args.output / mode / name)
    (args.output / 'licenses').mkdir()
    for name in LICENSES: shutil.copyfile(args.runtime_dir / 'licenses' / name, args.output / 'licenses' / name)
    shutil.copyfile('LICENSE', args.output / 'LICENSE')
    (args.output / 'model').mkdir()
    for name in ('LICENSE', 'NOTICE'): shutil.copyfile(args.model_dir / name, args.output / 'model' / name)
    manifest = {'format': 'qev-local-wasm/1', 'repository': REPO, 'revision': args.model_revision,
                'model_sha256': MODEL_SHA, 'files': files}
    (args.output / 'model' / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    index = args.output / 'index.html'
    html = index.read_text()
    substitutions = {'LOCAL PREVIEW': 'BROWSER CPU',
        'About 425 MB from localhost.': 'About 448 MB from Hugging Face. Inference runs on your CPU.',
        'Local preview · CPU / WASM · nothing published': 'CPU / WASM · decisions stay in your browser'}
    for before, after in substitutions.items():
        if html.count(before) != 1: raise ValueError('Unexpected deployment label: ' + before)
        html = html.replace(before, after)
    index.write_text(html)
    source = subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip()
    dirty = bool(subprocess.check_output(['git', 'status', '--porcelain', '--untracked-files=normal', '--',
                                        'web', 'scripts', 'src', 'include', 'cmake', 'CMakeLists.txt'], text=True).strip())
    (args.output / 'README.md').write_text(f'''---
title: qev CPU decision playground
emoji: 🧭
colorFrom: green
colorTo: yellow
sdk: static
app_file: index.html
fullWidth: true
header: mini
license: mit
models:
- {REPO}
tags:
- cpu
- wasm
- decision-model
custom_headers:
  cross-origin-embedder-policy: require-corp
  cross-origin-opener-policy: same-origin
  cross-origin-resource-policy: cross-origin
---
# qev CPU decision playground

Seven live demos and a JSON editor using our custom C++ CPU engine compiled to
WebAssembly. **Inference runs on your own CPU in the browser.** No server
inference, Python model runtime or GPU backend is used. Decision inputs stay in
the browser; the page downloads app files and the public model.

Click **Load model**, then choose Snake, 2048, Guardrails, Inbox, Support desk,
Tool routing or Moderation. Games also work manually before loading. Change the
inputs and inspect the actual evaluated request and probabilities.

The initial model/runtime-file transfer is about {sum(f['bytes'] for f in files)/1e6:.0f} MB
({files[0]['bytes']/1e6:.0f} MB of weights). Browser caching is optional; Unload releases the
worker and Clear cache removes this demo's stored model. Desktop Chromium is the
tested target. SIMD/single-thread fallbacks are included for other capability
sets; threading depends on cross-origin isolation. Memory use exceeds the model
file size and depends on browser, threads and context settings.

**Snake is path-assisted:** game code computes collision and shortest-open-route
facts around walls and the current body. qev scores each direction independently
and the highest yes score selects the move. These scores do not sum to 100%.
The helper does not simulate future tail motion or guarantee solving every
board; repeated full states pause autoplay. This is a live assisted-control
demo, not an unaided spatial-reasoning benchmark. Tool routing only displays a
choice; it does not call tools or connect to accounts.

Model: [calibrated H128/Q4-G32-DOT4](https://huggingface.co/{REPO}), pinned at
`{args.model_revision}`; checkpoint SHA-256 `{MODEL_SHA}`. The model card's task
quality scores were measured with the native engine, not newly measured here.
The original qev model and decision examples are by
[labqoat](https://app.labqoat.com/e/qev/qev/). This is an independent implementation.
Weights are downloaded from the model repository and are not copied into this
Space. Model license: Apache-2.0; code: MIT; dependency notices are in `licenses/`.

[Engine source](https://github.com/Danmoreng/qwen35-cpu/tree/{source}) ·
[Native benchmarks](https://github.com/Danmoreng/qwen35-cpu/blob/{source}/docs/benchmarks/decision-http.md)
''')
    package_files = {str(p.relative_to(args.output)): {'bytes': p.stat().st_size, 'sha256': sha(p)}
                     for p in sorted(args.output.rglob('*')) if p.is_file()}
    provenance = {'source_revision': source, 'source_dirty': dirty, 'model_revision': args.model_revision,
                  'model_sha256': MODEL_SHA, 'files': package_files, 'weights_bundled': False}
    (args.output / 'build-info.json').write_text(json.dumps(provenance, indent=2) + '\n')
    print(f'Staged {len(package_files) + 1} files in {args.output}; no model weights or upload')


if __name__ == '__main__': main()
