#!/usr/bin/env python3
"""Real HTTP regression: selected/full heads, previous server and concurrency.

Uses only the standard library. This is a correctness test, not a benchmark.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
import json
import os
from pathlib import Path
import socket
import subprocess
import tempfile
import time
import urllib.request


@contextmanager
def server(binary, model, weights, head=None, expected_auto_head='selected', threads=8):
    with socket.socket() as sock:
        sock.bind(('127.0.0.1', 0)); port = sock.getsockname()[1]
    command = [str(binary.resolve()), '--model-dir', str(model), '--weights', str(weights),
               '--threads', str(threads), '--port', str(port), '--residents', '16']
    if head: command += ['--decision-head', head]
    with tempfile.TemporaryFile() as log:
        child = subprocess.Popen(command, stdout=log, stderr=log,
                                 env={**os.environ, 'QWEN35_API_KEY': ''})
        base = f'http://127.0.0.1:{port}'
        try:
            for _ in range(600):
                if child.poll() is not None: raise RuntimeError('Server exited before readiness')
                try:
                    with urllib.request.urlopen(base + '/health', timeout=1): break
                except OSError: time.sleep(.05)
            else: raise TimeoutError('Readiness deadline')
            if head:
                log.seek(0)
                expected = expected_auto_head if head == 'auto' else 'full'
                assert f'System One readout: {expected}' in log.read().decode(), 'Unexpected readout dispatch'
            def call(body, path='/v1/systemone'):
                request = urllib.request.Request(base + path, data=json.dumps(body).encode(),
                                                 headers={'Content-Type': 'application/json'})
                with urllib.request.urlopen(request, timeout=120) as response:
                    return json.load(response)
            yield call
        finally:
            child.terminate()
            try: child.wait(timeout=10)
            except subprocess.TimeoutExpired: child.kill(); child.wait()
            if child.returncode not in (-15, 0):
                log.seek(0); print(log.read().decode(errors='replace'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--server', type=Path, required=True)
    parser.add_argument('--reference-server', type=Path)
    parser.add_argument('--model-dir', type=Path, required=True)
    parser.add_argument('--weights', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--expected-auto-head', choices=['selected', 'full'], default='selected')
    parser.add_argument('--threads', type=int, default=8)
    args = parser.parse_args()
    fixtures = [{**{k: json.loads(line)[k] for k in ('state', 'questions')}, 'encoding': encoding}
                for line in Path('configs/qev-smoke.jsonl').read_text().splitlines() if line.strip()
                for encoding in ('joint', 'per_question')]
    # Exercise many selected output tiles and the final supported label, as well
    # as mixed question types and option/question insertion order.
    for count in (2, 11, 77, 255):
        for encoding in ('joint', 'per_question'):
            fixtures.append({'state': {'target': count - 1}, 'encoding': encoding,
                'questions': {'wide': {'type': 'choice', 'instructions': 'Pick the target.',
                    'criteria': {f'item-{i}': f'Target {i}' for i in range(count)}},
                    'yes': {'type': 'noul', 'instructions': 'Is the target a number?'},
                    'score': {'type': 'score', 'instructions': 'How clear is the target?',
                              'criteria': ['unclear', 'clear', 'very clear']}}})
    runs = []
    variants = [('full', args.server, 'full'), ('selected', args.server, 'auto')]
    if args.reference_server: variants.insert(0, ('previous', args.reference_server, None))
    reference = None
    for name, binary, head in variants:
        with server(binary, args.model_dir, args.weights, head, args.expected_auto_head, args.threads) as call:
            outputs = [call(body) for body in fixtures]
            canonical = [{'answers': r['answers'], 'usage': r['usage']} for r in outputs]
            assert all(r['usage']['output_tokens'] == 0 for r in outputs)
            if reference is None: reference = canonical
            else: assert canonical == reference, f'{name}: HTTP answer/usage mismatch'
            parallel_inputs = [{**fixtures[0], 'state': f'{i}: ' + str(fixtures[0]['state'])}
                               for i in range(16)]
            serial = [call(body) for body in parallel_inputs]
            with ThreadPoolExecutor(max_workers=16) as pool:
                parallel = list(pool.map(call, parallel_inputs))
            assert [r['answers'] for r in parallel] == [r['answers'] for r in serial]
            completion = call({'prompt': 'Hello', 'max_tokens': 2, 'temperature': 0}, '/v1/completions')
            recovery = call(fixtures[0])
            assert recovery['answers'] == outputs[0]['answers']
            runs.append({'variant': name, 'fixtures': len(fixtures), 'concurrent_requests': 16,
                         'completion': completion['choices'], 'answers': canonical})
            print(f'{name}: {len(fixtures)} fixture responses, 16 concurrent calls and completion recovery passed', flush=True)
    assert all(run['completion'] == runs[0]['completion'] for run in runs)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({'exact_wire_parity': True, 'runs': runs}, indent=2) + '\n')


if __name__ == '__main__': main()
