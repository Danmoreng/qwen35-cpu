#!/usr/bin/env python3
"""Verify, extract and execute a release archive, optionally with a real model."""
import argparse
import hashlib
from pathlib import Path
import subprocess
import tarfile
import tempfile
import zipfile

p = argparse.ArgumentParser()
p.add_argument('--directory', type=Path, default=Path('dist'))
p.add_argument('--model-dir', type=Path)
args = p.parse_args()
archives = list(args.directory.glob('*.zip')) + list(args.directory.glob('*.tar.gz'))
if len(archives) != 1:
    raise ValueError('Expected exactly one release archive')
archive = archives[0]
sha, name = (archive.parent / (archive.name + '.sha256')).read_text().strip().split('  ')
if name != archive.name or hashlib.sha256(archive.read_bytes()).hexdigest() != sha:
    raise ValueError('Release checksum mismatch')
with tempfile.TemporaryDirectory(prefix='qwen35-release-') as directory:
    root = Path(directory).resolve()
    def validate(name):
        if not (root / name).resolve().is_relative_to(root):
            raise ValueError('Archive path escapes extraction directory')
    if archive.suffix == '.zip':
        with zipfile.ZipFile(archive) as z:
            for member in z.infolist():
                validate(member.filename)
            z.extractall(root)
        suffix = '.exe'
    else:
        with tarfile.open(archive) as t:
            for member in t.getmembers():
                validate(member.name)
                if not member.isfile() and not member.isdir():
                    raise ValueError('Unexpected archive member type')
            t.extractall(root)
        suffix = ''
    servers = list(root.glob('*/qwen35_cpu_server' + suffix))
    if len(servers) != 1:
        raise ValueError('Server missing from release')
    package = servers[0].parent
    for required in ['qwen35_cpu' + suffix, 'download-model.sh', 'download-model.ps1',
                     'LICENSE', 'licenses/cpp-httplib.txt', 'licenses/nlohmann-json.txt']:
        if not (package / required).is_file():
            raise ValueError('Missing release file: ' + required)
    subprocess.run([str(servers[0]), '--help'], check=True)
    subprocess.run([str(package / ('qwen35_cpu' + suffix)), '--help'], check=True)
    if args.model_dir:
        import sys
        subprocess.run([sys.executable, 'tests/server_smoke.py', '--server', str(servers[0]),
                        '--model-dir', str(args.model_dir.resolve())], check=True)
    print('Release checksum, contents and executable checks passed')
