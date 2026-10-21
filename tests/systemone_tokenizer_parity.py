#!/usr/bin/env python3
"""Compare native System One prompt IDs with every frozen HF evaluation row."""
import argparse
import json
from pathlib import Path
import subprocess

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--prepared',type=Path,required=True)
p.add_argument('--protocol',type=Path,required=True)
p.add_argument('--model-dir',type=Path,required=True)
p.add_argument('--output',type=Path,required=True)
args=p.parse_args()
items=json.loads((args.prepared/'items.json').read_text())
rows=json.loads((args.prepared/'rows.json').read_text())
requests=[dict(state=x['state'],questions=x['questions'],encoding='per_question') for x in items]
proc=subprocess.run([str(args.protocol.resolve()),str(args.model_dir.resolve())],
    input=''.join(json.dumps(x,ensure_ascii=False)+'\n' for x in requests),text=True,capture_output=True,check=True)
actual=[json.loads(x) for x in proc.stdout.split('\n') if x]
assert len(actual)==len(items)
expected={}
jobs=dict(line.split('\t') for line in (args.prepared/'jobs.tsv').read_text().splitlines())
for row in rows:
    expected.setdefault(row['item'],[]).append([int(x) for x in Path(jobs[row['id']]).read_text().split(',')])
mismatches=[dict(item=items[i]['id'],index=i) for i,record in enumerate(actual) if record['rows']!=expected[i]]
result=dict(items=len(items),questions=len(rows),mismatched_requests=len(mismatches),mismatches=mismatches)
args.output.write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({k:v for k,v in result.items() if k!='mismatches'}))
if mismatches:raise SystemExit(1)
