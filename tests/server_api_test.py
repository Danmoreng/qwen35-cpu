"""Real-model protocol checks; optional official SDK conformance, no speed claims."""
import argparse
import json
import os
from pathlib import Path
import socket
import subprocess
import tempfile
import time
import urllib.error
import urllib.request

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--server', type=Path, required=True)
p.add_argument('--model-dir', type=Path, required=True)
p.add_argument('--threads', default='2')
p.add_argument('--sdk', action='store_true', help='Also require the official OpenAI Python SDK')
args = p.parse_args()
with socket.socket() as s:
    s.bind(('127.0.0.1', 0))
    port = s.getsockname()[1]
base = f'http://127.0.0.1:{port}'
key = 'protocol-test-key'


def call(path, body=None):
    req = urllib.request.Request(base + path, data=None if body is None else json.dumps(body).encode(),
        headers={'Authorization': 'Bearer ' + key, 'Content-Type': 'application/json'})
    try:
        with urllib.request.urlopen(req, timeout=120) as r:
            return r.status, json.load(r)
    except urllib.error.HTTPError as e:
        return e.code, json.load(e)


command = [str(args.server.resolve()), '--model-dir', str(args.model_dir.resolve()),
           '--threads', args.threads, '--port', str(port), '--max-context', '1024', '--residents', '4']
with tempfile.TemporaryFile(mode='w+b') as log:
    process = subprocess.Popen(command, stdout=log, stderr=log, env={**os.environ, 'QWEN35_API_KEY': key})
    try:
        for _ in range(600):
            assert process.poll() is None, 'Server exited during startup'
            try:
                if call('/health')[0] == 200:
                    break
            except OSError:
                pass
            time.sleep(.1)
        else:
            raise AssertionError('Server did not become ready')
        model = call('/v1/models')[1]['data'][0]['id']
        fixtures = json.loads(Path(__file__).with_name('fixtures').joinpath('qwen-chat.json').read_text())['cases']
        for case in [fixtures[0], fixtures[2], fixtures[5], fixtures[6]]:
            request = dict(model=model, messages=case['messages'], max_completion_tokens=12, temperature=0)
            status, chat = call('/v1/chat/completions', request)
            assert status == 200, chat
            status, raw = call('/v1/completions', dict(model=model, prompt=case['prompt'], max_tokens=12, temperature=0))
            assert status == 200, raw
            choice = chat['choices'][0]
            assert chat['object'] == 'chat.completion' and choice['message']['role'] == 'assistant'
            assert choice['message']['content'] == raw['choices'][0]['text'], (chat, raw)
            assert choice['finish_reason'] == raw['choices'][0]['finish_reason']
            assert chat['usage'] == raw['usage'], (chat, raw)
        if args.sdk:
            import openai
            with openai.OpenAI(api_key=key, base_url=base + '/v1', max_retries=0) as client:
                answer = client.chat.completions.create(**request)
                assert answer.choices[0].message.content == chat['choices'][0]['message']['content']
                assert answer.usage.total_tokens == chat['usage']['total_tokens']
                try:
                    client.chat.completions.create(**{**request, 'model': 'missing'})
                except openai.BadRequestError as e:
                    assert e.body['type'] == 'invalid_request_error'
                else:
                    raise AssertionError('SDK did not receive a typed bad-request error')
                print(f'OpenAI SDK {openai.__version__}: text chat and error conformance passed')
        print('Real-model chat / raw-completion template, output and token-usage parity passed')
    except BaseException:
        log.seek(0)
        print(log.read().decode(errors='replace'))
        raise
    finally:
        process.terminate()
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()
