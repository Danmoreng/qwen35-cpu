#!/usr/bin/env python3
"""Archive the bounded native-GGUF experiments; never include weights/logits."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import statistics
import subprocess
import zipfile

parser = argparse.ArgumentParser()
parser.add_argument('--root', type=Path, default=Path('benchmarks/native-q4_0-2026-09-06'))
parser.add_argument('--kroot', type=Path, default=Path('benchmarks/native-q4km-2026-09-06'))
parser.add_argument('--out', type=Path, default=Path('docs/results/native-q4_0-2026-09-06'))
args = parser.parse_args()
root, kroot, out = args.root, args.kroot, args.out
out.mkdir(parents=True, exist_ok=True)

def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8*1024*1024), b''): h.update(block)
    return h.hexdigest()

def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))

rows, files = [], []
for folder in [root/'quick', root/'long', kroot/'quick']:
    metadata = read(folder/'metadata.json')
    assert metadata['runs'] == 3 and metadata['warmup_runs'] == 1
    assert metadata['affinity'] == 21845 and (folder/'results.csv').exists()
    for index, case in enumerate(metadata['matrix']):
        model, batch, threads, prompt, output = case['name'].rsplit('-', 4)
        batch, threads, prompt, output = [int(v[1:]) for v in (batch, threads, prompt, output)]
        profiles = [read(folder/f'case-{index}-run-{run}.json') for run in (1, 2, 3)]
        for p in profiles:
            assert p['cpu_batch'] == batch and p['prompt_tokens'] == prompt*batch
            assert p['generated_tokens'] == output*batch
            assert p.get('decode_forwards', p.get('decode_forward_steps')) == batch*(output-1)
            assert p['cpu_kv_cache'] == 'fp16' and not p.get('cached_prefix_tokens', 0)
            assert not p.get('greedy_lm_head_batches', 0) and not p.get('quality_capture', False)
        row = dict(group=folder.as_posix(), model=model, batch=batch, threads=threads,
                   prompt_tokens=prompt, output_tokens=output, affinity=21845)
        for label, values in [
            ('prefill', [p.get('prefill_forward_tokens_per_second', p['prefill_tokens_per_second']) for p in profiles]),
            ('prefill_with_request_init', [p['prefill_tokens_per_second'] for p in profiles]),
            ('decode', [p['tokens_per_second'] for p in profiles])]:
            row.update({label+'_median': statistics.median(values), label+'_min': min(values), label+'_max': max(values)})
        rows.append(row)
    files.extend(f for f in folder.iterdir() if f.is_file())
with (out/'performance-summary.csv').open('w', newline='', encoding='utf-8') as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator="\n"); w.writeheader(); w.writerows(rows)

for folder in [root/'quality-check', root/'arithmetic', kroot/'arithmetic']:
    files.extend(f for f in folder.rglob('*') if f.is_file() and f.suffix in ('.json', '.csv'))
for base in (root, kroot): files.extend(base.glob('*-matrix.json'))
previous = Path('benchmarks/comparison-2026-09-06')
files.extend(previous/name for name in ['inputs.json', 'machine.json', 'quality-windows.json',
    'performance-prompt-512.csv', 'performance-prompt-4096.csv',
    'quality-00/prompt.csv', 'quality-00/targets.csv',
    'arithmetic-regression/prompt.csv', 'arithmetic-regression/targets.csv'])
source_paths = subprocess.check_output(['git', 'ls-files', '--cached', '--others', '--exclude-standard',
    'src', 'include', 'CMakeLists.txt'], text=True).splitlines()
manifest = dict(parent_revision=subprocess.check_output(['git','rev-parse','HEAD'], text=True).strip(),
    note='Measurements were collected before commit. Raw runner metadata records each measured binary hash. '
         'A load-time null guard was added after the Q4_0 measurements; timed kernels were unchanged. '
         'The K-quant trial preceded Q4_0 support. Source hashes below normalize line endings to LF.',
    source_sha256={p: hashlib.sha256(Path(p).read_bytes().replace(b'\r\n', b'\n')).hexdigest() for p in source_paths},
    checkpoints={p: dict(sha256=sha(p), bytes=Path(p).stat().st_size) for p in [
        'models/llama-comparison/Qwen3.5-0.8B-Q4_0-pure.gguf',
        'models/llama-comparison/Qwen3.5-0.8B-Q4_K_M.gguf', 'models/hf-download-test/model.q35h']})
(out/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n', encoding='utf-8', newline='\n')
with zipfile.ZipFile(out/'raw-results.zip', 'w', zipfile.ZIP_DEFLATED, compresslevel=9) as z:
    for f in sorted(set(files)):
        assert f.is_file() and f.suffix not in ('.logits', '.gguf', '.q35h', '.safetensors')
        z.write(f, f.as_posix())
with zipfile.ZipFile(out/'raw-results.zip') as z: assert z.testzip() is None
artifacts = [out/'performance-summary.csv', out/'manifest.json', out/'raw-results.zip']
(out/'SHA256SUMS').write_text(''.join(f'{sha(p)}  {p.name}\n' for p in artifacts), encoding='utf-8', newline='\n')
print(f'Archived {len(rows)} cases and {len(set(files))} raw files in {out}')
