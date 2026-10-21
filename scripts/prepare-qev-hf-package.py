#!/usr/bin/env python3
"""Prepare only the evaluated qev Q4 artifact and its text-only runtime files."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess

ARTIFACT_SHA256 = 'e7d61e7b191de6c2afb8663e8a1f8c5f4d52315e9492db7a690708d744f8dc69'
SOURCE_REVISION = '6334efd2d556cad22a3040049efb2887a60e7fb2'


def sha(path):
    with path.open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,required=True)
    p.add_argument('--artifact',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--license-file',type=Path,required=True,help='Standard Apache-2.0 license text')
    args=p.parse_args()
    if sha(args.artifact)!=ARTIFACT_SHA256:raise ValueError('Artifact is not the evaluated qev release candidate')
    required=['config.json','tokenizer.json','tokenizer_config.json','vocab.json','merges.txt','qev_config.json']
    for name in required:
        if not (args.source/name).is_file():raise ValueError('Missing '+name)
    license_text=args.license_file.read_text()
    if 'Apache License' not in license_text or 'Version 2.0' not in license_text:raise ValueError('Expected Apache-2.0 license text')
    cfg=json.loads((args.source/'qev_config.json').read_text())
    if cfg['format']!='qev-systemone/1' or cfg['temperatures']!=dict(choice=1.7,noul=.9,score=1.25):raise ValueError('Unexpected qev configuration')
    sidecars={name:json.loads(Path(str(args.artifact)+'.'+name+'.json').read_text()) for name in ['quantization','calibration','covariance','precision']}
    if sidecars['quantization'].get('error_compensation')!='block128' or sidecars['covariance']['damping']!=.01:raise ValueError('Unexpected calibration recipe')
    calibration=sidecars['calibration']
    if 'inputs' in calibration:
        calibration['raw_input_count']=len(calibration.pop('inputs'))
        calibration['original_sidecar_sha256']=sha(Path(str(args.artifact)+'.calibration.json'))
    study=Path('docs/results/qev-calibration-2026-10-21');api=Path('docs/results/qev-api-2026-10-21')
    selection=json.loads((study/'selection.json').read_text())['pure']
    if selection['sha256']!=ARTIFACT_SHA256:raise ValueError('Release differs from frozen final selection')
    revision=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()
    args.output.mkdir(parents=True,exist_ok=False)
    for name in required:shutil.copyfile(args.source/name,args.output/name)
    shutil.copyfile(args.artifact,args.output/'model.q35h')
    (args.output/'LICENSE').write_text(license_text)
    (args.output/'NOTICE').write_text(
        'Derived from labqoat/qev-0.8b by labqoat, based on Qwen3.5-0.8B-Base by Qwen.\n'
        'The original model declares Apache-2.0. See its model card for training-data provenance.\n'
        'Modifications: text-only extraction, independent activation calibration,\n'
        'block-128 error compensation, H128/Q4-G32-DOT4 CPU packing.\n'
        'Vision weights are omitted; tokenizer and decision configuration are retained.\n')
    recipe=dict(format='q35h',recipe='H128/Q4-G32-DOT4',artifact_sha256=ARTIFACT_SHA256,
        base_model='labqoat/qev-0.8b',source_revision=SOURCE_REVISION,
        source_safetensors_sha256=sha(args.source/'model.safetensors'),engine_revision=revision,
        text_only=True,calibration_requests=1024,calibration_tokens=252138,
        sidecars=sidecars,calibration_inputs=json.loads((study/'calibration-inputs.json').read_text()),
        source_revisions=json.loads((study/'source-revisions.json').read_text()),
        quality_summary=json.loads((study/'summary.json').read_text())['final_outcomes']['pure'],
        probability_drift=json.loads((study/'probability-drift-detail.json').read_text())['variants']['pure'],
        http_performance_csv=(api/'http-performance.csv').read_text(),
        evidence={'quality_provenance_sha256':sha(study/'provenance.json'),'http_provenance_sha256':sha(api/'provenance.json')})
    (args.output/'quantization.json').write_text(json.dumps(recipe,indent=2)+'\n')
    card=Path('docs/qev-model-card.md').read_text()
    for key,value in dict(ENGINE_URL='https://github.com/Danmoreng/qwen35-cpu',ENGINE_REVISION=revision,SHA256=ARTIFACT_SHA256).items():card=card.replace('@'+key+'@',value)
    if '@ENGINE_' in card:raise ValueError('Unresolved model card fields')
    (args.output/'README.md').write_text(card)
    (args.output/'SHA256SUMS').write_text(''.join(f'{sha(file)}  {file.name}\n' for file in sorted(args.output.iterdir())))
    print('Prepared:',args.output,'engine revision:',revision)


if __name__=='__main__':main()
