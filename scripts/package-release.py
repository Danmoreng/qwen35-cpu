#!/usr/bin/env python3
"""Install and archive release binaries without model weights or build tools."""
import argparse
import hashlib
import json
from pathlib import Path
import platform
import shutil
import subprocess

p = argparse.ArgumentParser()
p.add_argument('--build', default='build')
p.add_argument('--output', default='dist')
p.add_argument('--version', default='dev')
p.add_argument('--platform', choices=['windows-x64', 'linux-x64'], required=True)
args = p.parse_args()
if not all(c.isalnum() or c in '.-_' for c in args.version):
    raise ValueError('Invalid version')
root = Path(args.output).resolve()
name = f'qwen35-cpu-{args.version}-{args.platform}'
stage = root / name
stage.mkdir(parents=True, exist_ok=False)
subprocess.run(['cmake', '--install', args.build, '--config', 'Release', '--prefix', str(stage)], check=True)
info = dict(version=args.version, platform=args.platform,
            commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
            build_host=platform.platform())
(stage / 'build-info.json').write_text(json.dumps(info, indent=2) + '\n')
fmt = 'zip' if args.platform == 'windows-x64' else 'gztar'
archive = Path(shutil.make_archive(str(root / name), fmt, root_dir=root, base_dir=name))
sha = hashlib.sha256(archive.read_bytes()).hexdigest()
(root / (archive.name + '.sha256')).write_bytes(f'{sha}  {archive.name}\n'.encode('ascii'))
print(archive)
