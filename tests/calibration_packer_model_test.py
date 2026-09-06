"""Optional model-backed rejection checks for the offline calibration container."""
import argparse
from pathlib import Path
import struct
import subprocess
import tempfile

parser = argparse.ArgumentParser()
parser.add_argument('--packer', type=Path, default=Path('benchmarks/plan-final-tools/qwen35_cpu_pack.exe'))
parser.add_argument('--model-dir', default='models/qwen3.5-0.8b')
parser.add_argument('--calibration', type=Path, default=Path('benchmarks/plan-calibration-fit'))
args = parser.parse_args()
name = 'model.language_model.embed_tokens.weight.cal'
original = (args.calibration/name).read_bytes()
variants = {}
for label, offset, fmt, value in [('columns',8,'Q',1025), ('seed',16,'Q',1),
                                  ('basis',24,'I',1), ('samples',28,'Q',0), ('negative',36,'f',-1)]:
    data = bytearray(original)
    struct.pack_into('<'+fmt, data, offset, value)
    variants[label] = data
variants['truncated'] = original[:20]
variants['trailing'] = original+b'!'
for label, data in variants.items():
    with tempfile.TemporaryDirectory(prefix='q35-calibration-') as tmp:
        root = Path(tmp)
        (root/'manifest.json').write_text('{}')
        (root/name).write_bytes(data)
        output = root/'invalid.q35h'
        result = subprocess.run([str(args.packer.resolve()),'--hf-model-dir',args.model_dir,
                                 '--output',str(output),'--quantizer','mse16','--importance-dir',str(root)],
                                capture_output=True, text=True)
        if result.returncode != 5 or output.exists() or list(root.glob('*.partial.*')):
            raise AssertionError(f'Invalid calibration was not rejected cleanly: {label}: {result.stderr}')
        print(f'Rejected {label} calibration and removed partial output')
