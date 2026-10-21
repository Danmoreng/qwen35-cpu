#!/usr/bin/env python3
"""Check qev wire/prompt parity against decider-ai 1.9.0 and real CPU inference.

Requires tokenizers and decider-ai==1.9.0, as used by the offline quality study.
These checks measure correctness, not throughput.
"""
import argparse
import concurrent.futures
import importlib.util
import json
import os
from pathlib import Path
import socket
import subprocess
import tempfile
import time
import urllib.error
import urllib.request
from types import SimpleNamespace
from decider.prompt import build
from decider.systemone import render_question, render_state, format_answer


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--server', type=Path, required=True)
    p.add_argument('--protocol', type=Path, required=True)
    p.add_argument('--model-dir', type=Path, required=True)
    p.add_argument('--weights', type=Path, required=True)
    p.add_argument('--study-logits', type=Path)
    p.add_argument('--native-study', type=Path)
    args = p.parse_args()
    spec = importlib.util.spec_from_file_location('adapter', Path(__file__).parents[1]/'scripts/qev-cpu.py')
    adapter = importlib.util.module_from_spec(spec); spec.loader.exec_module(adapter)
    tok = adapter.TokenizerAdapter(args.model_dir)
    temps = json.loads((args.model_dir/'qev_config.json').read_text())['temperatures']
    smoke = [json.loads(s) for s in Path('configs/qev-smoke.jsonl').read_text().splitlines() if s.strip()]
    fixtures = [{k:r[k] for k in ('state','questions')} for r in smoke]
    fixtures += [dict(state={'z':'Grüße 😀 a,b:c\\"','records':[{'v':i} for i in range(9)]}, questions={
        'z':{'type':'choice','instructions':{'pick':'Select'},'criteria':{'z':None,'a':[1,2],'none':None}},
        'a':{'type':'noul','criteria':{'true':{'allowed':True}}},
        's':{'type':'score','instructions':'Rate','criteria':{'10':'great','-1':'poor','2':'okay'}}})]
    fixtures += [dict(state=['test', {'values':list(range(8))}],questions={'wide':{
        'type':'choice','instructions':'Pick one','criteria':{f'option-{i}':None for i in range(n)}}}) for n in [11,77,255]]
    fixtures += [dict(state='\n  "name": "Test"\n## heading\n.#.#........#.\n \t\n  \n a\u0301 ₹ ١٢ Ⅲ 😀\u2028done',
        questions={'q':{'type':'noul','instructions':'Is this text?'}})]
    fixtures = [{**r,'encoding':e} for r in fixtures for e in ('joint','per_question')]
    proc = subprocess.run([str(args.protocol.resolve()),str(args.model_dir.resolve())],
        input=''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in fixtures), text=True,capture_output=True,check=True)
    rendered = [json.loads(s) for s in proc.stdout.split('\n') if s]
    assert len(rendered)==len(fixtures)
    def oracle(r):
        qs = [render_question(q) for q in r['questions'].values()]
        groups = [[q] for q in qs] if r['encoding']=='per_question' else [qs]
        rows=[]
        for group in groups:
            objects=[]
            for q in group:
                opts=['not listed here' if s.strip().lower().startswith('none of the above') or
                      s.strip().lower() in ('none','n/a','none of these') else s for s in q['options']]
                objects.append(SimpleNamespace(text=q['question'],options=opts,gold=0))
            enc=build(SimpleNamespace(context=render_state(r['state']),qs=objects),tok,adapter.NoShuffle(),
                      max_options=255,max_ctx_tokens=1000000,layout='state_first',chat=None)
            rows += [enc['ids'][:s+1] for s in enc['slots']]
        answers={}
        for key,q in zip(r['questions'],qs):
            logits=[((i*7)%17-8)*.125 for i in range(len(q['names']))]
            answers[key]=format_answer(q,adapter.distribution(logits,temps[q['type']]))
        return rows,answers
    for r,result in zip(fixtures,rendered):
        rows,answers=oracle(r)
        assert result['state']==render_state(r['state']), (r,result['state'])
        assert result['rows']==rows, ('Prompt mismatch',r)
        assert result['answers']==answers, ('Answer mismatch',r,result['answers'],answers)
    print(f'{len(fixtures)} protocol fixtures match upstream tokens and answer formatting',flush=True)
    with socket.socket() as sock:
        sock.bind(('127.0.0.1',0));port=sock.getsockname()[1]
    base=f'http://127.0.0.1:{port}'
    def call(body=None, path='/v1/systemone', authorized=True):
        headers={'Content-Type':'application/json'}
        if authorized:headers['Authorization']='Bearer systemone-test'
        req=urllib.request.Request(base+path,data=None if body is None else json.dumps(body,ensure_ascii=False).encode(),headers=headers)
        try:
            with urllib.request.urlopen(req,timeout=120) as response:return response.status,json.load(response)
        except urllib.error.HTTPError as e:return e.code,json.load(e)
    cmd=[str(args.server.resolve()),'--model-dir',str(args.model_dir.resolve()),'--weights',str(args.weights.resolve()),
         '--threads','8','--port',str(port),'--max-context','8192','--residents','4','--decision-encoding','per_question']
    with tempfile.TemporaryFile(mode='w+b') as log:
        server=subprocess.Popen(cmd,stdout=log,stderr=log,env={**os.environ,'QWEN35_API_KEY':'systemone-test'})
        try:
            for _ in range(600):
                if server.poll() is not None:raise AssertionError('Server exited before readiness')
                try:
                    if call(path='/health')[0]==200:break
                except OSError:pass
                time.sleep(.1)
            else:raise AssertionError('Readiness deadline')
            assert call(fixtures[0],authorized=False)[0]==401
            assert call(path='/v1/models')[1]['data'][0]['id']=='qev-0.8b-H128-Q4-G32-DOT4'
            refs=[]
            for r in smoke:
                body={k:r[k] for k in ('state','questions')};body['encoding']='per_question'
                status,result=call(body);assert status==200,result
                assert result['usage']['output_tokens']==0
                assert result['usage']['input_tokens']==sum(len(x) for x in oracle(body)[0])
                refs.append(result['answers'])
            if args.study_logits:
                values={x[0]:list(map(float,x[1:])) for line in args.study_logits.read_text().splitlines() if (x:=line.split('\t'))}
                index=0;max_delta=0.
                for request,actual in zip(smoke,refs):
                    for key,spec in request['questions'].items():
                        q=render_question(spec);p=adapter.distribution(values[f'smoke-{index:03d}'][:len(q['names'])],temps[q['type']]);index+=1
                        expected=format_answer(q,p)
                        if q['type']=='noul':max_delta=max(max_delta,abs(actual[key]['noul']-expected['noul']))
                        else:
                            for k,v in expected['probabilities'].items():max_delta=max(max_delta,abs(actual[key]['probabilities'][k]-v))
                assert max_delta<=.001, ('Offline study parity',max_delta)
                print('Maximum rounded probability delta to offline study:',max_delta,flush=True)
            mixed=next(r for r in fixtures if len(r['questions'])==3 and r['encoding']=='joint')
            status,reference=call(mixed);assert status==200,reference
            if args.native_study:
                with tempfile.TemporaryDirectory() as folder:
                    root=Path(folder);rows,_=oracle(mixed);jobs=[]
                    for i,ids in enumerate(rows):
                        token_file=root/f'{i}.txt';token_file.write_text(','.join(map(str,ids)))
                        jobs.append(f'row-{i}\t{token_file}\n')
                    (root/'jobs.tsv').write_text(''.join(jobs))
                    cfg=json.loads((args.model_dir/'qev_config.json').read_text())
                    (root/'letters.txt').write_text(','.join(map(str,cfg['letter_token_ids'])))
                    subprocess.run([str(args.native_study.resolve()),'--model-dir',str(args.model_dir),
                        '--weights',str(args.weights),'--jobs-file',str(root/'jobs.tsv'),
                        '--letters-file',str(root/'letters.txt'),'--output',str(root/'logits.tsv'),
                        '--threads','8','--max-context','8192'],check=True,stdout=subprocess.DEVNULL)
                    values=[list(map(float,s.split('\t')[1:])) for s in (root/'logits.tsv').read_text().splitlines()]
                    for (key,spec),xs in zip(mixed['questions'].items(),values):
                        q=render_question(spec);expected=format_answer(q,adapter.distribution(xs[:len(q['names'])],temps[q['type']]))
                        assert reference['answers'][key]==expected,('Joint readout mismatch',key,reference,expected)
                print('Joint multi-question HTTP readout matches independent CPU prefix evaluation',flush=True)
            with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
                outputs=list(pool.map(lambda _:call(mixed),range(4)))
            assert all(status==200 and answer['answers']==reference['answers'] for status,answer in outputs),outputs
            # Joint numbering agrees with independently evaluated prefix rows;
            # changing encoding does not alter a single-question request.
            single={**fixtures[0],'encoding':'joint'}
            assert call(single)[1]['answers']==call({**single,'encoding':'per_question'})[1]['answers']
            invalid=[{}, {'state':'x','questions':{}}, {**single,'images':{}}, {**single,'encoding':'bad'},
                     {**single,'state':'x'*65537}, {**single,'questions':{'q':{'type':'noul'}}},
                     {**single,'questions':{'q':{'type':'score','instructions':'Rate','criteria':['only']}}},
                     {**single,'questions':{'q':{'type':'choice','instructions':'Pick','criteria':['only']}}},
                     {**single,'state':' token'*9000}, {**single,'questions':{'q':{'type':'unknown'}}}]
            for body in invalid:assert call(body)[0]==400,body
            # Generation and decision requests share the bounded owner/scheduler.
            with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
                a=pool.submit(call,single);b=pool.submit(call,{'prompt':'Hello','max_tokens':1},'/v1/completions')
                assert a.result()[0]==200 and b.result()[0]==200
            assert call(single)[0]==200
            print('System One real-model HTTP checks passed',flush=True)
        finally:
            server.terminate()
            try:server.wait(timeout=10)
            except subprocess.TimeoutExpired:server.kill();server.wait()
            if server.returncode not in (-15,0):
                log.seek(0);print(log.read().decode(errors='replace'))


if __name__=='__main__':main()
