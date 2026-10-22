#!/usr/bin/env python3
"""One B1 first HTTP decision; orchestrate only via benchmark-inference-seq.ps1.

Each process starts a fresh local server, waits for readiness, then times one
request. Loading/readiness are excluded; HTTP, tokenization and postprocessing
are included. No cross-request prefix cache or generated continuation is used.
"""
import argparse
import concurrent.futures
import json
import os
from pathlib import Path
import socket
import subprocess
import time
import urllib.request


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--backend', choices=['native','llama'], required=True)
    p.add_argument('--server', type=Path, required=True)
    p.add_argument('--model-dir', type=Path)
    p.add_argument('--weights', type=Path, required=True)
    p.add_argument('--request', type=Path, required=True)
    p.add_argument('--threads', type=int, default=8)
    p.add_argument('--decision-head', choices=['auto', 'full'])
    p.add_argument('--prefill-chunk', type=int, default=0)
    p.add_argument('--concurrency', type=int, default=1)
    p.add_argument('--requests', type=int, help='Total requests in the sample; defaults to concurrency')
    p.add_argument('--distinct-states', action='store_true')
    p.add_argument('--profile-json', type=Path, required=True)
    args = p.parse_args()
    if not 1 <= args.concurrency <= 16:raise ValueError('concurrency must be 1..16')
    count = args.requests if args.requests is not None else args.concurrency
    if not args.concurrency <= count <= 16:raise ValueError('requests must be concurrency..16')
    if args.profile_json.exists():raise ValueError('Use a fresh profile path')
    with socket.socket() as s:s.bind(('127.0.0.1',0));port=s.getsockname()[1]
    base=f'http://127.0.0.1:{port}'
    cmd=[str(args.server.resolve())]
    if args.backend=='native':
        cmd+=['--model-dir',str(args.model_dir),'--weights',str(args.weights),'--threads',str(args.threads),
              '--residents',str(args.concurrency),'--max-context','8192','--port',str(port)]
        if args.decision_head:cmd+=['--decision-head',args.decision_head]
        if args.prefill_chunk:cmd+=['--prefill-chunk',str(args.prefill_chunk)]
    else:
        cmd+=['--model',str(args.weights),'--n-gpu-layers','0','--threads',str(args.threads),
              '--threads-batch',str(args.threads),'--parallel',str(args.concurrency),'--ctx-size',str(8192*args.concurrency),
              '--no-warmup','--cache-ram','0','--host','127.0.0.1','--port',str(port)]
    body=args.request.read_bytes()
    # Concurrency matrices use a different state per request, including B1,
    # so identical prefixes/results cannot turn the test into a cache benchmark.
    if args.distinct_states or args.concurrency > 1:
        request=json.loads(body)
        if not isinstance(request['state'],str):raise ValueError('Concurrency fixtures require string state')
        bodies=[json.dumps({**request,'state':f'{chr(65+i)}: '+request['state']}).encode() for i in range(count)]
    else:bodies=[body]
    with args.profile_json.with_suffix('.server.log').open('w') as log:
        child=subprocess.Popen(cmd,stdout=log,stderr=log,env={**os.environ,'QWEN35_API_KEY':''})
        try:
            for _ in range(1200):
                if child.poll() is not None:raise RuntimeError('Server failed before readiness')
                try:
                    with urllib.request.urlopen(base+'/health',timeout=1) as response:
                        if response.status==200:break
                except OSError:pass
                time.sleep(.05)
            else:raise RuntimeError('Readiness deadline')
            def decide(payload):
                req=urllib.request.Request(base+'/v1/systemone',data=payload,headers={'Content-Type':'application/json'})
                start=time.perf_counter()
                with urllib.request.urlopen(req,timeout=120) as response:raw=response.read()
                return (time.perf_counter()-start)*1000,json.loads(raw)
            start=time.perf_counter()
            if args.concurrency == 1:results=[decide(payload) for payload in bodies]
            else:
                with concurrent.futures.ThreadPoolExecutor(max_workers=args.concurrency) as pool:
                    results=list(pool.map(decide,bodies))
            elapsed=results[0][0] if len(results) == 1 else (time.perf_counter()-start)*1000
            times=[x[0] for x in results];result=results[0][1]
            assert all(x[1]['usage']['output_tokens']==0 for x in results)
            args.profile_json.write_text(json.dumps(dict(cpu_batch=args.concurrency,prompt_tokens=result['usage']['input_tokens'],
                generated_tokens=0,decode_forward_steps=0,decision_latency_ms=elapsed,
                decisions_per_second=1000*len(results)/elapsed,server_command=cmd,response=result,
                concurrency=args.concurrency,completed_requests=len(results),
                request_latencies_ms=times,mean_request_latency_ms=sum(times)/len(times),
                max_request_latency_ms=max(times),
                responses=[x[1] for x in results],
                scope='First B1 HTTP decision after readiness; no warmup inside server; loading excluded'),indent=2)+'\n')
        finally:
            child.terminate()
            try:child.wait(timeout=10)
            except subprocess.TimeoutExpired:child.kill();child.wait()


if __name__=='__main__':main()
