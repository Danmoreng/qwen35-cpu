#!/usr/bin/env python3
"""Verify that a mixed-precision experiment preserves all unpromoted Q4 bytes."""
import argparse
import json
from pathlib import Path
import struct
from evaluation_common import sha

def inventory(path):
    with path.open('rb') as f:
        assert f.read(8) == b'Q35H128\0'
        version, endian, header, alignment = struct.unpack('<4I',f.read(16))
        assert (version,endian,header,alignment) == (1,0x01020304,96,64)
        count, directory, _, _, seed = struct.unpack('<5Q',f.read(40))
        f.seek(directory)
        tensors = {}
        for _ in range(count):
            name_len,rank,encoding,transform,group,reserved,sign_seed,elements,offset,size,checksum = struct.unpack('<6I5Q',f.read(64))
            name=f.read(name_len).decode()
            shape=list(struct.unpack('<'+'Q'*rank,f.read(rank*8)))
            tensors[name]=dict(encoding=encoding,transform=transform,group=group,seed=sign_seed,
                              elements=elements,offset=offset,bytes=size,checksum=checksum,shape=shape)
        return tensors

parser=argparse.ArgumentParser()
parser.add_argument('candidate',type=Path)
parser.add_argument('--baseline',type=Path,default=Path('models/qwen3.5-0.8b/model-calibrated-mse16.q35h'))
parser.add_argument('--out',type=Path,required=True)
args=parser.parse_args()
base,candidate=inventory(args.baseline),inventory(args.candidate)
q8=[name for name,t in candidate.items() if t['encoding']==7]
assert q8 and all(name.endswith('linear_attn.in_proj_ba.weight') or name=='model.language_model.embed_tokens.weight' for name in q8)
gate_names=[name for name in q8 if name.endswith('linear_attn.in_proj_ba.weight')]
assert len(gate_names) in (0,18)
changed=[]
with args.baseline.open('rb') as original, args.candidate.open('rb') as experiment:
    for name,t in candidate.items():
        if t['encoding']==7:
            assert t['transform']==0 and t['seed']==0 and t['group']==32
            assert t['shape']==([32,1024] if name in gate_names else [248320,1024])
            assert t['bytes']==t['elements']//32*34
            changed.append(name);continue
        old=base[name]
        assert t['encoding']==old['encoding'] and t['seed']==old['seed']
        is_split=name.replace('in_proj_all.weight','in_proj_ba.weight') in gate_names
        if is_split:
            assert t['shape']==[old['shape'][0]-32,1024]
        else:
            assert t['shape']==old['shape'] and t['bytes']==old['bytes']
        original.seek(old['offset']);experiment.seek(t['offset'])
        remaining=t['bytes']
        while remaining:
            size=min(remaining,1024*1024)
            assert original.read(size)==experiment.read(size),name
            remaining-=size
    assert set(base)-set(candidate) <= {'model.language_model.embed_tokens.weight'}
result=dict(baseline_sha256=sha(args.baseline),candidate_sha256=sha(args.candidate),
    baseline_file_bytes=args.baseline.stat().st_size,candidate_file_bytes=args.candidate.stat().st_size,
    baseline_tensor_bytes=sum(t['bytes'] for t in base.values()),
    candidate_tensor_bytes=sum(t['bytes'] for t in candidate.values()),
    q8_tensors=changed,remaining_q4_bytes_unchanged=True,tensors=candidate)
args.out.parent.mkdir(parents=True,exist_ok=True)
args.out.write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8',newline='\n')
print(json.dumps({k:v for k,v in result.items() if k!='tensors'},indent=2))
