"""Shared validation and provenance for offline quality measurements."""
import hashlib
from pathlib import Path
import subprocess


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def source_identity(root):
    root = Path(root).resolve()
    try:
        toplevel = subprocess.check_output(
            ['git', '-C', str(root), 'rev-parse', '--show-toplevel'], stderr=subprocess.DEVNULL).decode().strip()
        if Path(toplevel).resolve() != root:
            raise ValueError('Source archive is nested inside another checkout')
        revision = subprocess.check_output(
            ['git', '-C', str(root), 'rev-parse', 'HEAD'], stderr=subprocess.DEVNULL).decode().strip()
        diff = subprocess.check_output(['git', '-C', str(root), 'diff', 'HEAD'], stderr=subprocess.DEVNULL)
    except (OSError, ValueError, subprocess.CalledProcessError):
        revision, diff = None, b''
    # Include untracked implementation files and support source archives.
    files = [p for folder in ('src', 'include', 'scripts', 'tests', 'tools', 'configs')
             for p in (root / folder).rglob('*')
             if p.is_file() and '__pycache__' not in p.parts]
    files += [root / n for n in ('CMakeLists.txt', 'AGENTS.md') if (root / n).exists()]
    tree = hashlib.sha256()
    for path in sorted(files):
        tree.update(path.relative_to(root).as_posix().encode() + b'\0')
        tree.update(bytes.fromhex(sha(path)))
    return dict(revision=revision, diff_sha256=hashlib.sha256(diff).hexdigest(),
                source_tree_sha256=tree.hexdigest())


def select_windows(windows, start, count):
    if start < 0 or count <= 0:
        raise ValueError('Window start must be nonnegative and count positive')
    selected = windows[start:start + count]
    if len(selected) != count:
        raise ValueError(f'Requested {count} windows at {start}, found {len(selected)}')
    if len({str(w['window']) for w in selected}) != len(selected):
        raise ValueError('Duplicate window IDs')
    return selected


def window_folder(window, manifest_path):
    # Older manifests serialized Windows separators and repository-relative paths.
    path = Path(window['folder'].replace('\\', '/'))
    if not path.is_absolute() and not path.is_dir():
        path = Path(manifest_path).parent/path
    return path.resolve()
