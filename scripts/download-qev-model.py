#!/usr/bin/env python3
"""Download the pinned public qev runtime files and verify their checksums."""
import argparse
import hashlib
from pathlib import Path
import urllib.request

REPOSITORY = 'danmoreng/qev-0.8b-H128-Q4-G32-DOT4'
REVISION = 'fc805be27e1409630b14b233ad7aac1a6dbb213f'
MODEL_SHA256 = 'e7d61e7b191de6c2afb8663e8a1f8c5f4d52315e9492db7a690708d744f8dc69'
FILES = ('model.q35h', 'config.json', 'tokenizer.json', 'tokenizer_config.json',
         'vocab.json', 'merges.txt', 'qev_config.json', 'LICENSE', 'NOTICE')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists(): raise ValueError('Choose a new destination directory')
    base = f'https://huggingface.co/{REPOSITORY}/resolve/{REVISION}'
    with urllib.request.urlopen(base + '/SHA256SUMS', timeout=60) as response:
        manifest_text = response.read().decode()
    manifest = {}
    for line in manifest_text.splitlines():
        digest, name = line.split('  ', 1)
        if name in manifest or len(digest) != 64 or any(c not in '0123456789abcdef' for c in digest):
            raise ValueError('Invalid checksum manifest')
        manifest[name] = digest
    if manifest.get('model.q35h') != MODEL_SHA256 or any(name not in manifest for name in FILES):
        raise ValueError('Unexpected model manifest')
    args.output.mkdir(parents=True)
    for name in FILES:
        target = args.output / name
        temporary = target.with_suffix(target.suffix + '.part')
        digest = hashlib.sha256()
        with urllib.request.urlopen(base + '/' + name, timeout=120) as source, temporary.open('wb') as output:
            while chunk := source.read(1024 * 1024):
                digest.update(chunk); output.write(chunk)
        if digest.hexdigest() != manifest[name]: raise ValueError('Checksum mismatch: ' + name)
        temporary.replace(target)
    (args.output / 'SHA256SUMS').write_text(''.join(f'{manifest[name]}  {name}\n' for name in FILES))
    print(f'Verified {REPOSITORY}@{REVISION} in {args.output}')


if __name__ == '__main__': main()
