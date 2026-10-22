#!/usr/bin/env python3
"""Publish a checked static qev Space using local Hugging Face credentials."""
import argparse
import hashlib
import json
from pathlib import Path
from huggingface_hub import HfApi

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--folder', type=Path, required=True)
parser.add_argument('--repo', required=True)
parser.add_argument('--update', action='store_true')
args = parser.parse_args()
info = json.loads((args.folder / 'build-info.json').read_text())
if info['source_dirty'] or info['weights_bundled']:
    raise ValueError('Publish only committed source without bundled weights')
expected = set(info['files']) | {'build-info.json'}
actual = {str(path.relative_to(args.folder)) for path in args.folder.rglob('*') if path.is_file()}
if actual != expected: raise ValueError('Unexpected package files')
for name, record in info['files'].items():
    path = args.folder / name
    if path.suffix in ('.q35h', '.gguf', '.safetensors'): raise ValueError('Do not upload model weights to the Space')
    if path.stat().st_size != record['bytes'] or hashlib.sha256(path.read_bytes()).hexdigest() != record['sha256']:
        raise ValueError('Package changed: ' + name)
api = HfApi()
api.whoami()
exists = api.repo_exists(args.repo, repo_type='space')
if exists != args.update: raise ValueError('Use --update only for an existing Space; otherwise choose a new name')
if not exists: api.create_repo(args.repo, repo_type='space', space_sdk='static', private=False, exist_ok=False)
parent = api.space_info(args.repo).sha
commit = api.upload_folder(repo_id=args.repo, repo_type='space', folder_path=args.folder,
    parent_commit=parent, commit_message='Publish local CPU decision playground with pinned qev model')
print(json.dumps({'repo': args.repo, 'revision': commit.oid, 'url': f'https://huggingface.co/spaces/{args.repo}',
                  'source_revision': info['source_revision'], 'model_revision': info['model_revision']}, indent=2))
