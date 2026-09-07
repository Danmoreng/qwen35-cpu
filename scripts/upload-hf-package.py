#!/usr/bin/env python3
"""Publish a prepared package using the local hf login (never a CLI token)."""
import argparse
import hashlib
from pathlib import Path
from huggingface_hub import HfApi, ModelCard

p = argparse.ArgumentParser()
p.add_argument('--folder', type=Path, required=True)
p.add_argument('--repo', required=True)
p.add_argument('--update', action='store_true', help='Update an existing repository with an optimistic parent-commit check')
args = p.parse_args()
expected = {'model.q35h', 'config.json', 'tokenizer.json', 'tokenizer_config.json',
            'vocab.json', 'merges.txt', 'chat_template.jinja', 'LICENSE', 'NOTICE',
            'README.md', 'quantization.json'}
if {x.name for x in args.folder.iterdir()} != expected | {'SHA256SUMS'}:
    raise ValueError('Package contains unexpected or missing files')
manifest = {}
for line in (args.folder / 'SHA256SUMS').read_text().splitlines():
    sha, name = line.split('  ', 1)
    if name in manifest or name not in expected:
        raise ValueError('Invalid checksum entry')
    manifest[name] = sha
if manifest.keys() != expected:
    raise ValueError('Incomplete manifest')
for name, sha in manifest.items():
    h = hashlib.sha256()
    with (args.folder / name).open('rb') as source:
        for chunk in iter(lambda: source.read(4*1024*1024), b''):
            h.update(chunk)
    if h.hexdigest() != sha:
        raise ValueError('Checksum mismatch: ' + name)
ModelCard.load(args.folder / 'README.md').validate()
api = HfApi()
api.whoami()  # Fail before creating a public repository if login is unavailable.
parent = None
if args.update:
    parent = api.model_info(args.repo).sha
else:
    api.create_repo(args.repo, repo_type='model', private=False, exist_ok=False)
commit = api.upload_folder(repo_id=args.repo, repo_type='model', folder_path=args.folder,
                         parent_commit=parent,
                         commit_message='Publish calibrated H128/Q4-G32-DOT4 model')
print('Published:', commit.repo_url)
print('Pin this model revision in README and CI:', commit.oid)
