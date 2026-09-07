#!/usr/bin/env python3
"""Fit diagonal importance in the exact H128 basis; never rotate a diagonal imatrix."""
import argparse
import json
from pathlib import Path
import re
import struct
import numpy as np
from evaluation_common import sha

SEED = 0x5148333548313238
MASK = (1 << 64)-1


def signs(columns, seed=SEED):
    result = []
    for block in range(columns//128):
        for word in range(2):
            x = (seed + block*0x9e3779b97f4a7c15 + word*0xd1b54a32d192ed03 + 0x9e3779b97f4a7c15) & MASK
            x = ((x ^ (x >> 30))*0xbf58476d1ce4e5b9) & MASK
            x = ((x ^ (x >> 27))*0x94d049bb133111eb) & MASK
            x ^= x >> 31
            result.extend(-1.0 if (x >> bit)&1 else 1.0 for bit in range(64))
    return np.array(result)


def rotate(inputs, seed=SEED):
    if inputs.shape[-1] % 128:
        raise ValueError('H128 requires complete blocks')
    result = (np.asarray(inputs, dtype=np.float64)*signs(inputs.shape[-1], seed)).copy()
    blocks = result.reshape(-1, 128)
    for stride in [1, 2, 4, 8, 16, 32, 64]:
        view = blocks.reshape(-1, 2*stride)
        left, right = view[:, :stride].copy(), view[:, stride:].copy()
        view[:, :stride], view[:, stride:] = left+right, left-right
    return result / np.sqrt(128)


def hf_name(name):
    if name in ('token_embd.weight', 'output.weight'):
        return 'model.language_model.embed_tokens.weight', 0
    match = re.fullmatch(r'blk\.(\d+)\.(.+)\.weight', name)
    if not match:
        raise ValueError('Unknown captured projection '+name)
    mapping = dict(ffn_gate='mlp.gate_proj', ffn_up='mlp.up_proj', ffn_down='mlp.down_proj',
        attn_qkv='linear_attn.in_proj_qkv', attn_gate='linear_attn.in_proj_z',
        ssm_alpha='linear_attn.in_proj_a', ssm_beta='linear_attn.in_proj_b', ssm_out='linear_attn.out_proj',
        attn_q='self_attn.q_proj', attn_k='self_attn.k_proj', attn_v='self_attn.v_proj', attn_output='self_attn.o_proj')
    return f'model.language_model.layers.{match[1]}.{mapping[match[2]]}.weight', 1


def fit(root, out, use_compact=True):
    manifest = json.loads((root/'manifest.json').read_text())
    statistics, inputs = {}, []
    names = None
    for document in manifest['documents']:
        folder = Path(document['directory'])
        compact = folder/'diagonal-statistics.json'
        if use_compact and compact.exists():
            saved = json.loads(compact.read_text())
            path = folder/'diagonal-statistics.npz'
            if saved['version'] != 1 or saved['sign_seed'] != SEED or sha(path) != saved['statistics_sha256']:
                raise ValueError('Invalid compacted statistics')
            current = {t['name'] for t in saved['tensors']}
            if names is not None and names != current:
                raise ValueError('Projection inventory changed between documents')
            names = current
            with np.load(path, allow_pickle=False) as archive:
                if set(archive.files) != current:
                    raise ValueError('Compacted projection inventory mismatch')
                for tensor in saved['tensors']:
                    name, basis, count = tensor['name'], tensor['basis'], tensor['samples']
                    sums = archive[name].copy()
                    if not count or sums.ndim != 1 or not np.isfinite(sums).all() or np.any(sums < 0):
                        raise ValueError('Invalid compacted moments')
                    if name not in statistics:
                        statistics[name] = [sums, count, basis]
                    else:
                        statistics[name][0] += sums
                        statistics[name][1] += count
                    inputs.append(dict(path=tensor['path'], sha256=tensor['sha256']))
            continue
        capture = json.loads((folder/'capture.json').read_text())
        if capture['version'] != 1 or capture['basis'] != 'identity':
            raise ValueError('Expected raw teacher inputs in the identity basis')
        current = {hf_name(t['name'])[0] for t in capture['tensors']}
        if names is not None and names != current:
            raise ValueError('Projection inventory changed between documents')
        names = current
        for tensor in capture['tensors']:
            name, basis = hf_name(tensor['name'])
            path = folder/(tensor['name']+'.f32')
            x = np.fromfile(path, dtype='<f4').reshape(tensor['samples'], tensor['columns'])
            if not len(x) or not np.isfinite(x).all():
                raise ValueError('Empty/non-finite projection capture')
            if basis:
                x = rotate(x)
            else:
                x = x.astype(np.float64)
            sums = np.sum(x*x, axis=0)
            if name not in statistics:
                statistics[name] = [sums, len(x), basis]
            else:
                statistics[name][0] += sums
                statistics[name][1] += len(x)
            inputs.append(dict(path=str(path), sha256=sha(path)))
    if len(statistics) != 187:
        raise ValueError(f'Expected all 187 supported projections including tied head, found {len(statistics)}')
    out.mkdir(parents=True, exist_ok=False)
    outputs = []
    for name, (sums, count, basis) in sorted(statistics.items()):
        importance = sums/count
        # Scaling all h by the same positive constant preserves the objective.
        mean = importance.mean()
        if mean > 0:
            importance /= mean
        importance = importance.astype('<f4')
        if not np.isfinite(importance).all():
            raise ValueError('Importance overflow')
        path = out/(name+'.cal')
        path.write_bytes(struct.pack('<8sQQIQ', b'Q35CAL1\0', len(importance), SEED if basis else 0, basis, count)
                         + importance.tobytes())
        outputs.append(dict(name=name, columns=len(importance), samples=count, basis=basis,
                            sha256=sha(path), zero_energy_channels=int(np.count_nonzero(importance == 0))))
    (out/'manifest.json').write_text(json.dumps(dict(version=1, recipe='activation-weighted-mse16',
        basis='R=H128 D/sqrt(128); transformed actual teacher inputs; identity tied head', sign_seed=SEED,
        calibration_manifest_sha256=sha(root/'manifest.json'), inputs=inputs, tensors=outputs), indent=2)+'\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('root', type=Path)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--raw-inputs', action='store_true', help='Read raw captures even if compacted moments exist')
    args = parser.parse_args()
    fit(args.root, args.out, not args.raw_inputs)
