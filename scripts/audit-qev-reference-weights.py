#!/usr/bin/env python3
"""Verify matching HF/GGUF text weights, including Qwen normalization/SSM conversions."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import struct
import sys
import numpy as np


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model-dir', type=Path, required=True)
    p.add_argument('--gguf', type=Path, required=True)
    p.add_argument('--llama-source', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    args = p.parse_args()
    sys.path.insert(0, str(args.llama_source/'gguf-py'))
    from gguf import GGUFReader
    spec = importlib.util.spec_from_file_location('qev_fit', Path(__file__).with_name('fit-dot4-quant.py'))
    fit = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(fit)
    hf = args.model_dir/'model.safetensors'
    with hf.open('rb') as f:
        size = struct.unpack('<Q', f.read(8))[0]
        header = json.loads(f.read(size))
    start = 8+size
    reader = GGUFReader(str(args.gguf))
    projections, auxiliary = [], []
    with hf.open('rb') as f:
        def source(name):
            info = header[name]
            lo,hi = info['data_offsets']
            f.seek(start+lo)
            raw = f.read(hi-lo)
            if len(raw)!=hi-lo:
                raise ValueError('Truncated safetensors payload')
            if info['dtype']=='BF16':
                return (np.frombuffer(raw,dtype='<u2').astype(np.uint32)<<16).view(np.float32)
            if info['dtype']=='F32':
                return np.frombuffer(raw,dtype='<f4')
            raise ValueError('Unsupported source dtype: '+info['dtype'])

        for tensor in reader.tensors:
            try:
                name,_ = fit.hf_name(tensor.name)
            except (ValueError,KeyError):
                continue
            info = header[name]
            lo,hi = info['data_offsets']
            f.seek(start+lo)
            digest = hashlib.sha256()
            remaining = hi-lo
            while remaining:
                block = f.read(min(remaining,16*1024*1024))
                if not block:
                    raise ValueError('Truncated source projection')
                digest.update(block)
                remaining -= len(block)
            raw = np.ascontiguousarray(tensor.data).view(np.uint8).reshape(-1)
            converted = hashlib.sha256(memoryview(raw)).hexdigest()
            projections.append(dict(hf=name,gguf=tensor.name,bytes=hi-lo,
                byte_identical=digest.hexdigest()==converted,hf_sha256=digest.hexdigest(),gguf_sha256=converted))

        mapping = dict(attn_norm=('input_layernorm.weight','offset'),
            post_attention_norm=('post_attention_layernorm.weight','offset'),
            attn_q_norm=('self_attn.q_norm.weight','offset'),
            attn_k_norm=('self_attn.k_norm.weight','offset'),
            ssm_norm=('linear_attn.norm.weight','identity'),
            ssm_conv1d=('linear_attn.conv1d.weight','identity'),
            ssm_a=('linear_attn.A_log','negative-exp'),
            ssm_dt=('linear_attn.dt_bias','identity'))
        for tensor in reader.tensors:
            if tensor.name=='output_norm.weight':
                name,conversion = 'model.language_model.norm.weight','offset'
            else:
                match = re.fullmatch(r'blk\.(\d+)\.([^.]+)(?:\.weight|\.bias)?',tensor.name)
                if not match or match[2] not in mapping:
                    continue
                tail,conversion = mapping[match[2]]
                name = f'model.language_model.layers.{match[1]}.{tail}'
            expected = source(name)
            if conversion=='offset':
                expected = expected+np.float32(1)
            elif conversion=='negative-exp':
                expected = -np.exp(expected.astype(np.float64)).astype(np.float32)
            actual = np.asarray(tensor.data).reshape(-1)
            if expected.shape!=actual.shape:
                raise ValueError('Auxiliary shape mismatch: '+name)
            delta = np.abs(expected.astype(np.float64)-actual)
            relative = np.divide(delta,np.abs(expected),out=np.zeros_like(delta),where=expected!=0)
            exact = np.array_equal(expected,actual)
            # exp implementations may differ by a few FP32 ULPs; other conversions must be exact.
            accepted = bool(np.allclose(expected,actual,rtol=5e-7,atol=0)) if conversion=='negative-exp' else exact
            auxiliary.append(dict(hf=name,gguf=tensor.name,conversion=conversion,exact=exact,
                accepted=accepted,max_absolute_delta=float(delta.max()),max_relative_delta=float(relative.max())))
    result = dict(projections=len(projections),projection_bytes=sum(x['bytes'] for x in projections),
        projection_mismatches=[x['gguf'] for x in projections if not x['byte_identical']],
        auxiliary_tensors=len(auxiliary),auxiliary_mismatches=[x['gguf'] for x in auxiliary if not x['accepted']],
        auxiliary_exact=sum(x['exact'] for x in auxiliary),exp_relative_tolerance=5e-7,
        max_auxiliary_absolute_delta=max(x['max_absolute_delta'] for x in auxiliary),
        max_auxiliary_relative_delta=max(x['max_relative_delta'] for x in auxiliary),
        tensors=projections,auxiliary=auxiliary)
    args.out.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ('tensors','auxiliary')},indent=2))
    if len(projections)!=187 or len(auxiliary)!=133 or result['projection_mismatches'] or result['auxiliary_mismatches']:
        raise ValueError('HF/GGUF text-weight audit failed')


if __name__=='__main__':
    main()
