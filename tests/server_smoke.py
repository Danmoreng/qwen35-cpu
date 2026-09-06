#!/usr/bin/env python3
"""Real-model HTTP correctness checks. No performance measurements."""
import argparse
import concurrent.futures
import json
import os
from pathlib import Path
import socket
import subprocess
import tempfile
import time
import urllib.error
import urllib.request

p = argparse.ArgumentParser()
p.add_argument('--server', type=Path, required=True)
p.add_argument('--model-dir', type=Path, required=True)
p.add_argument('--weights', type=Path)
p.add_argument('--threads', default='2')
args = p.parse_args()
with socket.socket() as s:
    s.bind(('127.0.0.1', 0))
    port = s.getsockname()[1]
base = f'http://127.0.0.1:{port}'


def call(path, data=None, authorized=True):
    headers = {'Content-Type': 'application/json'}
    if authorized:
        headers['Authorization'] = 'Bearer local-test-key'
    req = urllib.request.Request(base + path,
          data=None if data is None else json.dumps(data).encode(), headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=120) as response:
            return response.status, json.load(response)
    except urllib.error.HTTPError as e:
        return e.code, json.load(e)


cmd = [str(args.server.resolve()), '--model-dir', str(args.model_dir.resolve()),
       '--threads', args.threads, '--port', str(port), '--max-context', '256', '--residents', '4']
if args.weights:
    cmd += ['--weights', str(args.weights.resolve())]
with tempfile.TemporaryFile(mode='w+b') as log:
    process = subprocess.Popen(cmd, stdout=log, stderr=log,
        env={**os.environ, 'QWEN35_API_KEY': 'local-test-key'})
    try:
        for _ in range(600):
            if process.poll() is not None:
                raise AssertionError('Server exited before becoming ready')
            try:
                if call('/health')[0] == 200:
                    break
            except OSError:
                pass
            time.sleep(.1)
        else:
            raise AssertionError('Server startup deadline exceeded')
        assert call('/health', authorized=False)[0] == 401
        assert call('/v1/models')[1]['data'][0]['id'] == 'Qwen3.5-0.8B-H128-Q4-G32-DOT4'
        request = {'prompt': 'The capital of France is', 'max_tokens': 12, 'temperature': 0}
        status, reference = call('/v1/completions', request)
        assert status == 200, reference
        for patch in [{'stream': True}, {'temperature': .7}, {'max_tokens': -1},
                      {'max_tokens': 1.5}, {'model': 'wrong'}, {'unknown': 1}, {'n': 2},
                      {'prefix_tokens': 200}, {'prompt': ''}]:
            assert call('/v1/completions', {**request, **patch})[0] == 400, patch
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
            answers = list(pool.map(lambda _: call('/v1/completions', {**request, 'prefix_tokens': 1}), range(4)))
        for status, answer in answers:
            assert status == 200, answer
            assert answer['choices'] == reference['choices'], (reference, answer)
            assert answer['usage'] == reference['usage']
        assert call('/v1/completions', request)[1]['choices'] == reference['choices']
        print('HTTP authentication, validation, concurrent greedy/prefix parity and reuse passed')
    except BaseException:
        log.seek(0)
        print(log.read().decode(errors='replace'))
        raise
    finally:
        process.terminate()
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill(); process.wait()
