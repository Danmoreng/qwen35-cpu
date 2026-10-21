#!/usr/bin/env python3
"""Build local text-only qev ONNX fixtures from pinned optimized graph templates.

Requires numpy, onnx and onnxruntime (including its quantization dependencies).
No model weights or generated graphs belong in Git.
"""
import argparse
import copy
import hashlib
import json
import re
import shutil
import struct
import urllib.request
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper
from onnxruntime.quantization.matmul_nbits_quantizer import (
    DefaultWeightOnlyQuantConfig, DefaultWeightOnlyQuantizer,
)

TEMPLATE = 'onnx-community/Qwen3.5-0.8B-ONNX-OPT'
REVISION = '9a4c21d8f724176b0a7edaaed36303044451886e'


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


class Source:
    def __init__(self, path):
        self.path = path
        with path.open('rb') as stream:
            length = struct.unpack('<Q', stream.read(8))[0]
            self.header = json.loads(stream.read(length))
        self.offset = 8 + length

    def read(self, name):
        item = self.header[name]
        lo, hi = item['data_offsets']
        dtype = {'BF16': '<u2', 'F32': '<f4'}[item['dtype']]
        a = np.memmap(self.path, dtype=dtype, mode='r', offset=self.offset + lo,
                      shape=tuple(item['shape']))
        assert a.nbytes == hi - lo
        return ((a.astype(np.uint32) << 16).view(np.float32)
                if item['dtype'] == 'BF16' else np.asarray(a))

    def convert(self, name):
        if name == 'model.embed_tokens.weight':
            return self.read('model.language_model.embed_tokens.weight')
        if name == 'lm_head.MatMul.weight':
            return self.read('model.language_model.embed_tokens.weight').T.copy()
        if name == 'model.layers.24.final_norm_layernorm.weight':
            return self.read('model.language_model.norm.weight') + np.float32(1)
        match = re.fullmatch(r'model.layers.(\d+)\.(.+)', name)
        if not match:
            raise ValueError('Unmapped template weight: ' + name)
        tail = match[2]
        if tail.startswith('attn.'):
            tail = tail.replace('attn.', 'self_attn.', 1)
        else:
            tail = tail.replace('gdn.', 'linear_attn.')
        tail = tail.replace('.MatMul.weight', '.weight').replace('.layernorm.weight', '.weight')
        tail = tail.replace('conv1d.weight_3d', 'conv1d.weight').replace('A_neg_exp', 'A_log')
        value = self.read(f'model.language_model.layers.{match[1]}.{tail}')
        if name.endswith('A_neg_exp'):
            value = -np.exp(value.astype(np.float64)).astype(np.float32)
        if ('layernorm.weight' in name or '.attn.q_norm.' in name or '.attn.k_norm.' in name):
            value = value + np.float32(1)
        if '.MatMul.weight' in name:
            value = value.T.copy()
        return value


def export(source, template, output, bits):
    model = copy.deepcopy(template)
    graph = model.graph
    quantizer = DefaultWeightOnlyQuantizer(DefaultWeightOnlyQuantConfig(
        bits=bits or 4, block_size=32, is_symmetric=False))
    matrix_nodes = {n.input[1]: n for n in graph.node if n.op_type == 'MatMul'}
    old = list(graph.initializer)
    del graph.initializer[:]
    audit = []
    external = output.with_name(output.name + '_data')
    with external.open('wb') as stream:
        def write(name, array):
            a = np.ascontiguousarray(array)
            tensor = TensorProto(name=name, data_type=helper.np_dtype_to_tensor_dtype(a.dtype), dims=a.shape)
            # set_external_data requires raw_data to exist, even when empty.
            tensor.raw_data = b''
            onnx.external_data_helper.set_external_data(tensor, location=external.name,
                offset=stream.tell(), length=a.nbytes)
            tensor.ClearField('raw_data')
            stream.write(memoryview(a).cast('B'))
            graph.initializer.append(tensor)

        for original in old:
            name = original.name
            if not original.external_data:
                graph.initializer.append(original)
                continue
            value = source.convert(name)
            assert list(value.shape) == list(original.dims), (name, value.shape, original.dims)
            audit.append({'name': name, 'shape': list(value.shape)})
            if bits and name in matrix_nodes:
                node = matrix_nodes[name]
                packed, scales, zero = quantizer.qbits_block_quant(value)
                names = [name + suffix for suffix in ['_quant', '_scales', '_zp']]
                for key, array in zip(names, [packed, scales, zero]):
                    write(key, array)
                replacement = helper.make_node('MatMulNBits', [node.input[0], *names], list(node.output),
                    name=node.name, domain='com.microsoft', K=value.shape[0], N=value.shape[1],
                    bits=bits, block_size=32)
                node.CopyFrom(replacement)
            elif bits and name == 'model.embed_tokens.weight':
                # Gather uses adjacent packed nibbles, unlike MatMulNBits' block layout.
                blocks = value.reshape(value.shape[0], -1, 32)
                lo = np.minimum(blocks.min(-1), 0)
                hi = np.maximum(blocks.max(-1), 0)
                scales = (hi - lo) / np.float32((1 << bits) - 1)
                scales[scales == 0] = 1
                zp = np.clip(np.rint(-lo / scales), 0, (1 << bits) - 1).astype(np.uint8)
                q = np.clip(np.rint(blocks / scales[..., None]) + zp[..., None], 0, (1 << bits) - 1).astype(np.uint8).reshape(value.shape)
                if bits == 4:
                    q = q[:, 0::2] | (q[:, 1::2] << 4)
                    zp = zp[:, 0::2] | (zp[:, 1::2] << 4)
                names = [name + suffix for suffix in ['_quant', '_scales', '_zp']]
                for key, array in zip(names, [q, scales, zp]):
                    write(key, array)
                node = graph.node[0]
                node.CopyFrom(helper.make_node('GatherBlockQuantized', [names[0], 'input_ids', *names[1:]],
                    list(node.output), domain='com.microsoft', bits=bits, block_size=32,
                    gather_axis=0, quantize_axis=1))
            else:
                write(name, value)
            del value
    if not any(x.domain == 'com.microsoft' for x in model.opset_import):
        model.opset_import.append(helper.make_opsetid('com.microsoft', 1))
    output.write_bytes(model.SerializeToString())
    return {'graph': output.name, 'sha256': sha(output), 'data_sha256': sha(external),
            'bytes': output.stat().st_size + external.stat().st_size, 'mapped_weights': audit}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model-dir', type=Path, default=Path('models/qev-0.8b'))
    parser.add_argument('--output', type=Path, default=Path('dist/qev-onnx'))
    parser.add_argument('--template-dir', type=Path, default=Path('.cache/qev-onnx-template'))
    parser.add_argument('--dtypes', nargs='+', choices=['fp32', 'q8', 'q4'], default=['q8', 'q4'])
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / 'onnx').mkdir(exist_ok=True)
    args.template_dir.mkdir(parents=True, exist_ok=True)
    source = Source(args.model_dir / 'model.safetensors')
    # AutoModelForCausalLM selects the text-only embed/decoder sessions when
    # loading the checkpoint's original ConditionalGeneration architecture.
    config = json.loads((args.model_dir / 'config.json').read_text())
    config['transformers.js_config'] = {'use_external_data_format': {'embed_tokens': 1, 'decoder_model_merged': 1}}
    (args.output / 'config.json').write_text(json.dumps(config, indent=2) + '\n')
    for name in ['tokenizer.json', 'tokenizer_config.json', 'qev_config.json']:
        shutil.copyfile(args.model_dir / name, args.output / name)
    manifest = {'template': TEMPLATE, 'template_revision': REVISION,
        'source_sha256': sha(source.path), 'quantization': 'ORT asymmetric weight-only, block size 32; FP32 activations',
        'artifacts': []}
    for part in ['embed_tokens', 'decoder_model_merged']:
        path = args.template_dir / (part + '.onnx')
        if not path.exists():
            urllib.request.urlretrieve(f'https://huggingface.co/{TEMPLATE}/resolve/{REVISION}/onnx/{part}.onnx', path)
        template = onnx.load(path, load_external_data=False)
        for dtype in args.dtypes:
            suffix, bits = {'fp32': ('', 0), 'q8': ('_quantized', 8), 'q4': ('_q4', 4)}[dtype]
            target = args.output / 'onnx' / (part + suffix + '.onnx')
            print('Exporting', target, flush=True)
            item = export(source, template, target, bits)
            item.update(dtype=dtype, template_sha256=sha(path))
            manifest['artifacts'].append(item)
            print('Wrote', item['bytes'], 'bytes', flush=True)
    (args.output / 'export.json').write_text(json.dumps(manifest, indent=2) + '\n')


if __name__ == '__main__':
    main()
