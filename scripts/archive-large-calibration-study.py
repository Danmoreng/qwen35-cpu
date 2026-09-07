#!/usr/bin/env python3
"""Retain evidence locally and publish only a compact aggregate summary to Git."""
from pathlib import Path
import json
import shutil

destination = Path('benchmarks/bc-large-evidence')
destination.mkdir(parents=True, exist_ok=True)


def copy(source, relative):
    source = Path(source)
    if not source.is_file():
        raise FileNotFoundError(source)
    target = destination / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, target)


study = Path('benchmarks/bc-large-study')
for source in study.rglob('*'):
    if source.is_file() and source.suffix in {'.json', '.csv', '.log', '.txt', '.py', '.ps1', '.md'}:
        copy(source, Path('study') / source.relative_to(study))

for name in ['selection.json', 'mixture.json', 'overlap-audit.json', 'final/quality-windows.json']:
    copy(Path('benchmarks/bc-large-corpus') / name, Path('corpus') / name)
capture = Path('benchmarks/bc-large-capture')
for name in ['manifest.json', 'profile.json', 'teacher.log', 'collection-validation.json', 'final-logit-hashes.json',
             'covariance-full/block-statistics.json', 'covariance-small/block-statistics.json']:
    copy(capture / name, Path('capture') / name)
for source in (capture / 'captures').glob('*/*.json'):
    if source.name in {'capture.json', 'diagonal-statistics.json'}:
        copy(source, Path('capture') / source.relative_to(capture))
for name in ['bc-large-corpus.log', 'bc-large-capture.log', 'bc-large-ctest.log']:
    copy(Path('benchmarks') / name, name)
copy('benchmarks/bc-block-compact-validation/result.json', 'covariance-compaction-validation.json')
copy('benchmarks/bc-block-compact-validation-fit/manifest.json', 'covariance-validation-fit-manifest.json')
for name in ['prepare-diverse-calibration.py', 'audit-calibration-corpus.py',
             'collect-resident-calibration.py', 'compact-calibration-captures.py',
             'block_covariance_accumulator.py', 'fit-block-covariance.py', 'fit-dot4-quant.py',
             'remove-compacted-capture-raw.ps1', 'run-large-calibration-study.py',
             'compare-quality-pair.py', 'audit-g32-fit-artifact.py', 'benchmark-inference-seq.ps1',
             'evaluation_common.py', 'archive-large-calibration-study.py']:
    copy(Path('scripts') / name, Path('source/scripts') / name)
for name in ['tools/llama-calibration-cuda/resident.cpp', 'tools/llama-calibration-cuda/CMakeLists.txt',
             'tools/llama-comparison/capture-inputs.h']:
    copy(name, Path('source') / name)
print('Archived text evidence:', destination)
summary = dict(
    artifact_sha256=json.loads((study / 'full-bc-layout.json').read_text())['candidate_sha256'],
    quality={}, speed=json.loads((study / 'speed-summary.json').read_text()),
    candidate_decision=json.loads((study / 'candidate-decision.json').read_text()),
    collection_checks=json.loads((capture / 'collection-validation.json').read_text()),
    local_evidence=str(destination),
    evidence_policy='Raw profiles, per-token results and full manifests are retained locally, not in Git.')
for suite in ['development', 'full', 'final']:
    for variant in ['small-b', 'small-bc', 'full-b', 'full-bc', 'old-c']:
        path = study / (variant + '-' + suite + '-paired.json')
        if path.exists():
            data = json.loads(path.read_text())
            summary['quality'][variant + '-' + suite] = {
                key: data[key] for key in ['overall', 'domains', 'paired_document_bootstrap']}
target = Path('docs/results/g32-large-calibration-2026-09-07/summary.json')
target.parent.mkdir(parents=True, exist_ok=True)
target.write_text(json.dumps(summary, indent=2) + '\n', encoding='utf-8', newline='\n')
