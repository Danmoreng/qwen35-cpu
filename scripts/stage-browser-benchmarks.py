#!/usr/bin/env python3
"""Stage local external-engine benchmarks beside the existing CPU/WASM demo."""
import argparse
import os
import shutil
import urllib.request
from pathlib import Path


def link(source, target):
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() and os.path.samefile(source, target):
        return
    target.unlink(missing_ok=True)
    try:
        os.link(source, target)
    except OSError:
        shutil.copyfile(source, target)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--node-modules', type=Path, default=Path('.cache/wasm-browser/node_modules'))
    parser.add_argument('--onnx-dir', type=Path, default=Path('dist/qev-onnx'))
    parser.add_argument('--model-dir', type=Path, default=Path('models/qev-0.8b'))
    parser.add_argument('--output', type=Path, default=Path('dist/wasm'))
    args = parser.parse_args()
    onnx = args.output / 'onnx-bench'
    llama = args.output / 'llama-bench'
    for folder, worker in [(onnx, 'onnx-browser-worker.js'), (llama, 'llama-browser-worker.js')]:
        folder.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(Path('scripts') / worker, folder / 'worker.js')
        (folder / 'index.html').write_text(
            '<!doctype html><meta charset="utf-8"><link rel="icon" href="data:,">'
            '<title>Local browser engine comparison</title>'
            '<p>Local benchmark fixture. Use scripts/benchmark-browser-decision.mjs.</p>')
    for path in args.onnx_dir.rglob('*'):
        if path.is_file():
            link(path, onnx / 'model' / path.relative_to(args.onnx_dir))
    generation = onnx / 'model/generation_config.json'
    if not generation.exists():
        generation.write_text('{"eos_token_id":248044,"pad_token_id":248044}')
    runtime = args.node_modules / 'onnxruntime-web/dist'
    for suffix in ['', '.jsep', '.asyncify']:
        for extension in ['mjs', 'wasm']:
            name = f'ort-wasm-simd-threaded{suffix}.{extension}'
            link(runtime / name, onnx / 'runtime' / name)
    transformers = args.node_modules / '@huggingface/transformers'
    link(transformers / 'dist/transformers.js', onnx / 'runtime/transformers.js')
    link(transformers / 'LICENSE', onnx / 'runtime/transformers-LICENSE')
    license_path = onnx / 'runtime/onnxruntime-LICENSE'
    if not license_path.exists():
        urllib.request.urlretrieve(
            'https://raw.githubusercontent.com/microsoft/onnxruntime/8d85527a0/LICENSE', license_path)
    shutil.copyfile(args.model_dir / 'qev_config.json', llama / 'qev_config.json')
    for dtype in ['Q8_0', 'Q4_0', 'Q4_K_M']:
        source = args.model_dir / f'qev-0.8b-{dtype}.gguf'
        link(source, llama / 'models' / source.name)
    print('Staged local fixtures under', args.output, '(nothing uploaded)')


if __name__ == '__main__':
    main()
