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
        for patch in [{'stream': True}, {'temperature': -1}, {'max_tokens': -1},
                      {'temperature': '0.7'}, {'top_p': 0}, {'top_p': 1.1},
                      {'top_k': -1}, {'top_k': 1.5}, {'top_k': 248321},
                      {'repetition_penalty': .9}, {'seed': -2}, {'seed': 4294967296},
                      {'seed': 18446744073709551615}, {'seed': True},
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
        # top_k=1 exercises the full-logits sampler but must retain greedy output.
        assert call('/v1/completions', {**request, 'temperature': .7, 'top_k': 1,
                    'top_p': 1, 'seed': 42})[1]['choices'] == reference['choices']
        assert call('/v1/completions', {**request, 'temperature': .7, 'top_k': 0,
                    'top_p': 1e-7, 'seed': 42})[1]['choices'] == reference['choices']
        sampled = {**request, 'temperature': 1.2, 'top_p': .9, 'top_k': 40,
                   'repetition_penalty': 1.1, 'seed': 42}
        status, expected = call('/v1/completions', sampled)
        assert status == 200, expected
        assert expected['choices'] != reference['choices'], 'Sampling was ignored'
        assert call('/v1/completions', sampled)[1]['choices'] == expected['choices']
        other = {**sampled, 'seed': 123}
        status, other_expected = call('/v1/completions', other)
        assert status == 200, other_expected
        # Interleave greedy and independently seeded requests, with prefix reuse.
        mixed = [sampled, request, other, sampled]
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
            answers = list(pool.map(lambda body: call('/v1/completions', {**body, 'prefix_tokens': 1}), mixed))
        for (status, answer), wanted in zip(answers, [expected, reference, other_expected, expected]):
            assert status == 200, answer
            assert answer['choices'] == wanted['choices'], (answer, wanted)
        cli = args.server.resolve().with_name('qwen35_cpu' + args.server.suffix)
        weights = args.weights or args.model_dir / 'model.q35h'
        output = subprocess.check_output([str(cli), '--model-dir', str(args.model_dir.resolve()),
            '--weights', str(weights.resolve()), '--prompt', request['prompt'],
            '--max-new-tokens', '12', '--max-context', '256', '--threads', args.threads,
            '--temperature', '1.2', '--top-p', '.9', '--top-k', '40',
            '--repetition-penalty', '1.1', '--seed', '42'], text=True, encoding='utf-8')
        assert output.removesuffix('\n') == expected['choices'][0]['text'], output
        print('HTTP validation, greedy and seeded sampling, mixed batches and prefix parity passed')
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
