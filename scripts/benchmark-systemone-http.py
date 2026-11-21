#!/usr/bin/env python3
"""Time HTTP decisions through benchmark-inference-seq.ps1.

Each process starts a fresh local server and waits for readiness before timing
requests. The default sends one request. A JSONL panel sends distinct requests
to the resident server and records per-item latencies in input order.
"""
import argparse
import concurrent.futures
import json
import os
import math
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
    p.add_argument('--prefix-cache', choices=['auto', 'off'],
                   help='Explicit native cache mode; records cache counters after the timed panel')
    p.add_argument('--decision-head', choices=['auto', 'full'])
    p.add_argument('--ticket-panel', action='store_true', help='Read labeled, variable-length states from --request')
    p.add_argument('--jsonl-panel', action='store_true',
                   help='Time every {id, request} row on one resident server; results retain input order')
    p.add_argument('--prefill-requests', type=int, choices=range(1, 9),
                   help='Benchmark native shared prefill, with fixed admission width and the prefix cache off')
    p.add_argument('--prefill-chunk', help='Native prefill chunk size or auto')
    p.add_argument('--batch-size', type=int)
    p.add_argument('--ubatch-size', type=int)
    p.add_argument('--concurrency', type=int, default=1)
    p.add_argument('--requests', type=int, help='Total requests in the sample; defaults to concurrency')
    p.add_argument('--distinct-states', action='store_true')
    p.add_argument('--profile-json', type=Path, required=True)
    args = p.parse_args()
    if args.prefix_cache and (args.backend != 'native' or (args.prefill_requests and args.prefix_cache != 'off')):
        p.error('--prefix-cache requires native inference; --prefill-requests benchmarks require off')
    if not 1 <= args.concurrency <= 16:raise ValueError('concurrency must be 1..16')
    count = args.requests if args.requests is not None else args.concurrency
    if args.jsonl_panel and (args.requests is not None or args.ticket_panel or args.distinct_states):
        p.error('--jsonl-panel takes one request per row and supplies its own states and count')
    if not args.concurrency <= count <= 16:raise ValueError('requests must be concurrency..16')
    if args.profile_json.exists():raise ValueError('Use a fresh profile path')
    with socket.socket() as s:s.bind(('127.0.0.1',0));port=s.getsockname()[1]
    base=f'http://127.0.0.1:{port}'
    cmd=[str(args.server.resolve())]
    if args.backend=='native':
        cmd+=['--model-dir',str(args.model_dir),'--weights',str(args.weights),'--threads',str(args.threads),
              '--residents',str(args.concurrency),'--max-context','8192','--port',str(port)]
        if args.prefill_requests:
            cmd += ['--prefill-requests', str(args.prefill_requests), '--batch-width', str(args.concurrency)]
        if args.prefix_cache or args.prefill_requests:
            cmd += ['--prefix-cache', args.prefix_cache or 'off']
        if args.decision_head:cmd+=['--decision-head',args.decision_head]
        if args.prefill_chunk not in (None, '0'):cmd+=['--prefill-chunk',str(args.prefill_chunk)]
    else:
        cmd+=['--model',str(args.weights),'--n-gpu-layers','0','--threads',str(args.threads),
              '--threads-batch',str(args.threads),'--parallel',str(args.concurrency),'--ctx-size',str(8192*args.concurrency),
              '--no-warmup','--cache-ram','0','--host','127.0.0.1','--port',str(port)]
        if args.batch_size:cmd+=['--batch-size',str(args.batch_size)]
        if args.ubatch_size:cmd+=['--ubatch-size',str(args.ubatch_size)]
    body=args.request.read_bytes()
    # Concurrency matrices use a different state per request, including B1,
    # so identical prefixes/results cannot turn the test into a cache benchmark.
    if args.jsonl_panel:
        rows = [json.loads(line) for line in body.splitlines() if line.strip()]
        ids = [row['id'] for row in rows]
        if not ids or len(set(ids)) != len(ids):
            raise ValueError('Panel must contain distinct request IDs')
        if len(ids) < args.concurrency:
            p.error('Panel needs at least one request per concurrent client')
        bodies = [json.dumps(row['request'], ensure_ascii=False).encode() for row in rows]
    elif args.ticket_panel:
        panel = json.loads(body)
        if len(panel['cases']) < count: raise ValueError('Not enough distinct cases')
        bodies = [json.dumps(dict(state=c['fields']['state'], questions={'dept': panel['qev_question']})).encode()
                  for c in panel['cases'][:count]]
    elif args.distinct_states or args.concurrency > 1:
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
            stats = {}
            panel_info = {}
            if args.jsonl_panel:
                panel_info['request_ids'] = ids
                status = Path(f'/proc/{child.pid}/status')
                if status.is_file():
                    fields = dict(line.split(':', 1) for line in status.read_text().splitlines())
                    panel_info['server_peak_rss_bytes'] = int(fields['VmHWM'].split()[0]) * 1024
                    panel_info['server_allowed_cpus'] = fields['Cpus_allowed_list'].strip()
            if args.prefill_chunk == 'auto' and args.backend == 'native':
                with urllib.request.urlopen(base+'/v1/batch-config', timeout=5) as response:
                    panel_info['batch_config'] = json.load(response)
            if args.prefill_requests or args.prefix_cache:
                with urllib.request.urlopen(base+'/v1/cache', timeout=5) as response:
                    stats = json.load(response)
                if args.prefix_cache:
                    panel_info['cache_status'] = stats
            args.profile_json.write_text(json.dumps(dict(cpu_batch=args.concurrency,prompt_tokens=result['usage']['input_tokens'],
                generated_tokens=0,decode_forward_steps=0,decision_latency_ms=elapsed,
                decisions_per_second=1000*len(results)/elapsed,server_command=cmd,response=result,
                concurrency=args.concurrency,completed_requests=len(results),
                request_latencies_ms=times,mean_request_latency_ms=sum(times)/len(times),
                max_request_latency_ms=max(times), p95_ms=sorted(times)[math.ceil(.95*len(times))-1],
                shared_ticks=stats.get('shared_prefill_ticks', 0),
                shared_requests=stats.get('shared_prefill_requests', 0),
                responses=[x[1] for x in results],
                scope='HTTP decisions after readiness; loading excluded', **panel_info),indent=2)+'\n')
        finally:
            child.terminate()
            try:child.wait(timeout=10)
            except subprocess.TimeoutExpired:child.kill();child.wait()


if __name__=='__main__':main()
