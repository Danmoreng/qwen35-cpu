#!/usr/bin/env python3
"""Prepare an allowlisted, independently usable model repository. No upload."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as source:
        for data in iter(lambda: source.read(4 * 1024 * 1024), b''):
            h.update(data)
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--source', type=Path, required=True)
    p.add_argument('--artifact', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--engine-url', default='https://github.com/Danmoreng/qwen35-cpu')
    args = p.parse_args()
    required = ['config.json', 'tokenizer.json', 'tokenizer_config.json',
                'vocab.json', 'merges.txt', 'chat_template.jinja', 'LICENSE']
    for name in required:
        if not (args.source / name).is_file():
            raise ValueError(f'Missing source file: {name}')
    if not args.artifact.is_file():
        raise ValueError('Missing packed artifact')
    # Publishing remains restricted to the evaluated standard checkpoint.
    artifact_hash = digest(args.artifact)
    if artifact_hash != '013fbfaa03760e759181301ddaf964bb5c200c50c50617afe72557fd65bcbf0a':
        raise ValueError('Artifact differs from the validated release candidate')
    quantization_path = Path(str(args.artifact)+'.quantization.json')
    calibration_path = Path(str(args.artifact)+'.calibration.json')
    quantization = json.loads(quantization_path.read_text())
    calibration = json.loads(calibration_path.read_text())
    if quantization['quantizer'] != 'mse16' or quantization['importance'] != 'Q35CAL1':
        raise ValueError('Expected calibrated MSE16 provenance')
    covariance_path = Path(str(args.artifact) + '.covariance.json')
    covariance = json.loads(covariance_path.read_text())
    if quantization.get('error_compensation') != 'block128' or covariance['damping'] != 0.01:
        raise ValueError('Expected block128 error compensation with damping 0.01')
    args.output.mkdir(parents=True, exist_ok=False)
    for name in required:
        shutil.copyfile(args.source / name, args.output / name)
    shutil.copyfile(args.artifact, args.output / 'model.q35h')
    revision = subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip()
    source_hashes = {file.name: digest(file) for file in sorted(args.source.glob('*.safetensors'))}
    selection_path = Path('benchmarks/bc-large-corpus/selection.json')
    selection = json.loads(selection_path.read_text())
    compact_calibration = {k: v for k, v in calibration.items() if k != 'inputs'}
    compact_calibration['raw_input_count'] = len(calibration['inputs'])
    compact_calibration['evidence_policy'] = 'Full raw-input hashes retained locally; original sidecar checksum is recorded.'
    compact_corpus = dict(selection_sha256=digest(selection_path),
        sources={name: {k: source[k] for k in ('repository', 'revision', 'file')}
                 for name, source in selection['sources'].items()},
        selection='First eligible independent documents in pinned first-shard order; see preparation script.',
        domain_documents={name: sum(d['domain'] == name and d['split'] == 'calibration'
                                   for d in selection['documents']) for name in selection['sources']})
    recipe = dict(format='q35h', recipe='H128/Q4-G32-DOT4', quantizer='activation-weighted-mse16-block128-compensation',
                  format_changed=False, transform_size=128,
                  scale_group=32, artifact_sha256=artifact_hash,
                  base_model='Qwen/Qwen3.5-0.8B', source_shard_sha256=source_hashes,
                  engine_revision=revision, source_revision='not recorded; use source shard hashes',
                  text_only=True, kv_cache='fp16', recurrent_state='fp32',
                  quantizer_parameters=quantization, calibration_provenance=compact_calibration,
                  calibration_sidecar_sha256=digest(calibration_path),
                  covariance_provenance=covariance, covariance_sidecar_sha256=digest(covariance_path),
                  calibration_documents=256, calibration_tokens=262144,
                  corpus_provenance=compact_corpus,
                  validation_report=args.engine_url+'/blob/'+revision+'/docs/g32-large-calibration-2026-09-07.md')
    (args.output / 'quantization.json').write_text(json.dumps(recipe, indent=2) + '\n', encoding='utf-8')
    notice = ('Derived from Qwen/Qwen3.5-0.8B by Qwen, licensed under Apache-2.0.\n'
              'Modifications: text-only extraction, H128/Q4-G32 quantization and CPU DOT4 packing.\n'
              'Vision and MTP/draft weights are omitted. Original tokenizer/configuration retained.\n')
    if (args.source / 'NOTICE').is_file():
        notice += '\nOriginal NOTICE:\n' + (args.source / 'NOTICE').read_text(encoding='utf-8')
    (args.output / 'NOTICE').write_text(notice, encoding='utf-8')
    card = Path('docs/model-card.md').read_text(encoding='utf-8')
    for key, value in {'ENGINE_URL': args.engine_url, 'ENGINE_REVISION': revision,
                       'SHA256': artifact_hash}.items():
        card = card.replace('@' + key + '@', value)
    (args.output / 'README.md').write_text(card, encoding='utf-8')
    manifest = ''.join(f'{digest(file)}  {file.name}\n' for file in sorted(args.output.iterdir()) if file.is_file())
    (args.output / 'SHA256SUMS').write_bytes(manifest.encode('ascii'))
    print(f'Prepared {args.output}; no files uploaded.')


if __name__ == '__main__':
    main()
