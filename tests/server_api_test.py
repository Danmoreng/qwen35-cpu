"""Real-model protocol checks; optional official SDK conformance, no speed claims."""
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

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--server', type=Path, required=True)
p.add_argument('--model-dir', type=Path, required=True)
p.add_argument('--threads', default='2')
p.add_argument('--sdk', action='store_true', help='Also require the official OpenAI and Anthropic Python SDKs')
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


def stream_call(path, body):
    req = urllib.request.Request(base + path, data=json.dumps({**body, 'stream': True,
        'stream_options': {'include_usage': True}}).encode(),
        headers={'Authorization': 'Bearer ' + key, 'Content-Type': 'application/json'})
    with urllib.request.urlopen(req, timeout=120) as r:
        assert r.headers['Content-Type'].startswith('text/event-stream')
        records = [line.decode().removeprefix('data: ').strip() for line in r if line.startswith(b'data: ')]
    assert records[-1] == '[DONE]', records
    chunks = [json.loads(r) for r in records[:-1]]
    assert len({c['id'] for c in chunks}) == 1 and len({c['created'] for c in chunks}) == 1
    assert chunks[-1]['choices'] == [] and chunks[-1]['usage'] is not None
    assert all(c.get('usage') is None for c in chunks[:-1])
    assert chunks[-2]['choices'][0]['finish_reason'] in ['length', 'stop']
    if path == '/v1/chat/completions':
        assert chunks[0]['choices'][0]['delta']['role'] == 'assistant'
        text = ''.join(c['choices'][0]['delta'].get('content', '') for c in chunks[:-1])
    else:
        text = ''.join(c['choices'][0]['text'] for c in chunks[:-1])
    return text, chunks[-1]['usage'], chunks[-2]['choices'][0]['finish_reason']


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
        fixtures = json.loads(Path(__file__).with_name('fixtures').joinpath('qwen-chat.json').read_text(encoding='utf-8'))['cases']
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
        for path, body, expected in [('/v1/chat/completions', request, chat),
                ('/v1/completions', dict(model=model, prompt=case['prompt'], max_tokens=12, temperature=0), raw)]:
            text, usage, reason = stream_call(path, body)
            expected_text = expected['choices'][0].get('text', expected['choices'][0].get('message', {}).get('content'))
            assert text == expected_text and usage == expected['usage']
            assert reason == expected['choices'][0]['finish_reason']
            # A stop embedded in the generated text must never leak, including in SSE.
            assert len(text) >= 3, text
            marker = text[1:3]
            status, stopped = call(path, {**body, 'stop': marker})
            assert status == 200, stopped
            stop_text = stopped['choices'][0].get('text', stopped['choices'][0].get('message', {}).get('content'))
            assert stop_text == text[:text.index(marker)] and stopped['choices'][0]['finish_reason'] == 'stop'
            streamed_stop = stream_call(path, {**body, 'stop': marker})
            assert streamed_stop == (stop_text, stopped['usage'], 'stop')
        # Disconnect immediately after headers; a subsequent request still completes.
        req = urllib.request.Request(base + '/v1/chat/completions',
            data=json.dumps({**request, 'stream': True, 'max_completion_tokens': 512}).encode(),
            headers={'Authorization': 'Bearer ' + key, 'Content-Type': 'application/json'})
        with urllib.request.urlopen(req, timeout=120) as stream:
            assert stream.readline().startswith(b'data: ')
        assert call('/v1/chat/completions', request)[0] == 200
        # Validation failures must retain a JSON error status, not start an SSE stream.
        status, bad = call('/v1/chat/completions', {**request, 'stream': True, 'max_completion_tokens': 1024})
        assert status == 400 and 'error' in bad, bad
        # Automatic reuse is exact, private to its key, and optional per request.
        long_system = 'You are concise. Answer with a short fact. ' * 12
        cached_request = dict(model=model, messages=[{'role': 'system', 'content': long_system},
            {'role': 'user', 'content': 'What is two plus two?'}], max_tokens=12, temperature=0,
            prompt_cache_key='cache-contract')
        status, cold = call('/v1/chat/completions', {**cached_request, 'cache_prompt': False})
        assert status == 200, cold
        status, first = call('/v1/chat/completions', cached_request)
        assert status == 200, first
        status, warm = call('/v1/chat/completions', cached_request)
        assert status == 200, warm
        assert cold['choices'] == first['choices'] == warm['choices']
        assert cold['usage']['prompt_tokens_details']['cached_tokens'] == 0
        assert first['usage']['prompt_tokens_details']['cached_tokens'] == 0
        assert warm['usage']['prompt_tokens_details']['cached_tokens'] >= 32
        status, isolated = call('/v1/chat/completions', {**cached_request, 'prompt_cache_key': 'other-key'})
        assert status == 200 and isolated['usage']['prompt_tokens_details']['cached_tokens'] == 0
        assert isolated['choices'] == cold['choices']
        concurrent_request = {**cached_request, 'prompt_cache_key': 'concurrent-key'}
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
            parallel = list(pool.map(lambda _: call('/v1/chat/completions', concurrent_request), range(4)))
        assert all(status == 200 and value['choices'] == cold['choices'] for status, value in parallel)
        assert sum(value['usage']['prompt_tokens_details']['cached_tokens'] > 0 for _, value in parallel) == 3
        # The next turn can restore the system prefix and publish a longer history.
        history = {**cached_request, 'messages': cached_request['messages'] + [
            {'role': 'assistant', 'content': warm['choices'][0]['message']['content']},
            {'role': 'user', 'content': 'And three plus three?'}]}
        _, history_cold = call('/v1/chat/completions', {**history, 'cache_prompt': False})
        _, history_first = call('/v1/chat/completions', history)
        _, history_warm = call('/v1/chat/completions', history)
        assert history_cold['choices'] == history_first['choices'] == history_warm['choices']
        assert history_warm['usage']['prompt_tokens_details']['cached_tokens'] > warm['usage']['prompt_tokens_details']['cached_tokens']
        _, metrics = call('/v1/cache')
        assert metrics['hits'] > 0 and metrics['builds'] > 0 and metrics['bytes'] <= metrics['budget_bytes']
        structured_schema = {'type': 'object', 'properties': {
            'answer': {'type': 'integer', 'enum': [4]},
            'city': {'type': 'string', 'enum': ['Berlin']},
            'flags': {'type': 'array', 'items': {'type': 'boolean'}, 'minItems': 1, 'maxItems': 2}},
            'required': ['answer', 'city', 'flags'], 'additionalProperties': False}
        structured_request = dict(model=model, messages=[{'role': 'user', 'content': 'Two plus two, city Berlin, and one true flag.'}],
            temperature=0, max_tokens=96, response_format={'type': 'json_schema',
                'json_schema': {'name': 'answer', 'strict': True, 'schema': structured_schema}})
        status, structured = call('/v1/chat/completions', structured_request)
        assert status == 200, structured
        assert structured['choices'][0]['finish_reason'] == 'stop', structured
        structured_value = json.loads(structured['choices'][0]['message']['content'])
        assert structured_value['answer'] == 4 and structured_value['city'] == 'Berlin' and 1 <= len(structured_value['flags']) <= 2
        assert all(type(flag) is bool for flag in structured_value['flags'])
        text, usage, reason = stream_call('/v1/chat/completions', structured_request)
        assert json.loads(text) == structured_value and reason == 'stop'
        status, generic = call('/v1/chat/completions', dict(model=model, temperature=0, max_tokens=64,
            messages=[{'role': 'user', 'content': 'Reply exactly with the object {"ok":true}.'}],
            response_format={'type': 'json_object'}))
        assert status == 200 and isinstance(json.loads(generic['choices'][0]['message']['content']), dict), generic
        sampled_schema = {'type': 'object', 'properties': {'letter': {'type': 'string', 'enum': ['A', 'B']}},
            'required': ['letter'], 'additionalProperties': False}
        status, sampled = call('/v1/chat/completions', {**structured_request, 'temperature': .7, 'top_k': 20, 'seed': 17,
            'response_format': {'type': 'json_schema', 'json_schema': {'name': 'letter', 'schema': sampled_schema}}})
        assert status == 200 and json.loads(sampled['choices'][0]['message']['content'])['letter'] in ['A', 'B'], sampled
        bad_schema = {**structured_schema, 'unknown_rule': True}
        status, invalid = call('/v1/chat/completions', {**structured_request, 'stream': True,
            'response_format': {'type': 'json_schema', 'json_schema': {'name': 'bad', 'schema': bad_schema}}})
        assert status == 400 and 'error' in invalid, invalid
        tool_schema = {'type': 'object', 'properties': {'city': {'type': 'string', 'enum': ['Berlin']}},
            'required': ['city'], 'additionalProperties': False}
        offered_tools = [{'type': 'function', 'function': {'name': 'weather', 'description': 'Get weather for a city.',
            'parameters': tool_schema, 'strict': True}}]
        tool_request = dict(model=model, messages=[{'role': 'user', 'content': 'Get the weather in Berlin.'}],
            temperature=0, max_tokens=96, tools=offered_tools, tool_choice='required', parallel_tool_calls=False)
        status, tool_reply = call('/v1/chat/completions', tool_request)
        assert status == 200, tool_reply
        choice = tool_reply['choices'][0]
        assert choice['finish_reason'] == 'tool_calls' and choice['message']['content'] is None, tool_reply
        selected = choice['message']['tool_calls'][0]
        assert selected['function']['name'] == 'weather' and json.loads(selected['function']['arguments']) == {'city': 'Berlin'}
        status, roundtrip = call('/v1/chat/completions', {**tool_request, 'tool_choice': 'none',
            'messages': tool_request['messages'] + [choice['message'],
                {'role': 'tool', 'tool_call_id': selected['id'], 'content': 'Sunny, 18 degrees Celsius.'}]})
        assert status == 200 and isinstance(roundtrip['choices'][0]['message']['content'], str), roundtrip
        status, truncated = call('/v1/chat/completions', {**tool_request, 'max_tokens': 1})
        assert status == 200 and truncated['choices'][0]['finish_reason'] == 'length'
        assert 'tool_calls' not in truncated['choices'][0]['message']
        if args.sdk:
            import openai
            with openai.OpenAI(api_key=key, base_url=base + '/v1', max_retries=0) as client:
                answer = client.chat.completions.create(**request)
                assert answer.choices[0].message.content == chat['choices'][0]['message']['content']
                assert answer.usage.total_tokens == chat['usage']['total_tokens']
                with client.chat.completions.create(**request, stream=True, stream_options={'include_usage': True}) as stream:
                    chunks = list(stream)
                assert ''.join(c.choices[0].delta.content or '' for c in chunks if c.choices) == answer.choices[0].message.content
                assert chunks[-1].usage.total_tokens == answer.usage.total_tokens and not chunks[-1].choices
                try:
                    client.chat.completions.create(**{**request, 'model': 'missing'})
                except openai.BadRequestError as e:
                    assert e.body['type'] == 'invalid_request_error'
                else:
                    raise AssertionError('SDK did not receive a typed bad-request error')
                named_request = {**tool_request, 'tool_choice': {'type': 'function', 'function': {'name': 'weather'}}}
                with client.chat.completions.create(**named_request, stream=True) as stream:
                    chunks = list(stream)
                deltas = [c.choices[0].delta for c in chunks if c.choices]
                calls = [call for d in deltas for call in (d.tool_calls or [])]
                assert calls and calls[0].id and calls[0].function.name == 'weather'
                assert json.loads(''.join(c.function.arguments or '' for c in calls)) == {'city': 'Berlin'}
                assert chunks[-1].choices[0].finish_reason == 'tool_calls'
                assert all(not d.content for d in deltas), 'Internal tool envelope leaked into text'
                print(f'OpenAI SDK {openai.__version__}: text chat, SSE, constrained JSON, tools and errors passed')
            import anthropic
            with anthropic.Anthropic(api_key=key, base_url=base, max_retries=0) as latest:
                current = latest.messages.create(model=model, max_tokens=4,
                    messages=[{'role': 'user', 'content': 'Hello'}])
                assert current.type == 'message' and current.usage.output_tokens > 0
                try:
                    latest.messages.create(model=model, max_tokens=4, messages=[{'role': 'user', 'content': 'Hi'}],
                        extra_body={'temperature': 0})
                except anthropic.BadRequestError:
                    pass
                else:
                    raise AssertionError('Removed sampling field accepted on 2026-10-21')
            with anthropic.Anthropic(api_key=key, base_url=base, max_retries=0,
                    default_headers={'anthropic-version': '2023-06-01'}) as client:
                msg_request = dict(model=model, system='Answer briefly.', messages=[
                    {'role': 'user', 'content': [{'type': 'text', 'text': 'What is two plus two?'}]}],
                    max_tokens=12, extra_body={'temperature': 0})
                answer = client.messages.create(**msg_request)
                assert answer.type == 'message' and answer.role == 'assistant'
                _, equivalent = call('/v1/chat/completions', dict(model=model,
                    messages=[{'role': 'system', 'content': msg_request['system']}] + msg_request['messages'],
                    max_tokens=12, temperature=0))
                assert answer.content[0].text == equivalent['choices'][0]['message']['content']
                with client.messages.stream(**msg_request) as stream:
                    text = ''.join(stream.text_stream)
                    final = stream.get_final_message()
                assert text == answer.content[0].text == final.content[0].text
                assert final.stop_reason == answer.stop_reason and final.usage.output_tokens == answer.usage.output_tokens
                count = client.messages.count_tokens(model=model, system=msg_request['system'], messages=msg_request['messages'])
                assert count.input_tokens == equivalent['usage']['prompt_tokens']
                assert final.usage.input_tokens + final.usage.cache_read_input_tokens + final.usage.cache_creation_input_tokens == count.input_tokens
                marker = text[:1]
                assert marker
                stopped = client.messages.create(**msg_request, stop_sequences=[marker])
                assert stopped.stop_reason == 'stop_sequence' and stopped.stop_sequence == marker
                assert stopped.content[0].text == text[:text.index(marker)]
                with client.messages.stream(**msg_request, stop_sequences=[marker]) as stream:
                    stop_text = ''.join(stream.text_stream)
                    final = stream.get_final_message()
                assert stop_text == stopped.content[0].text and final.stop_sequence == marker
                assert final.usage.output_tokens == stopped.usage.output_tokens
                # Local cache accounting must partition the same logical Qwen prompt.
                long_request = {**msg_request, 'system': 'An Anthropic cache accounting fixture. ' * 16}
                first = client.messages.create(**long_request)
                warm = client.messages.create(**long_request)
                count = client.messages.count_tokens(model=model, system=long_request['system'], messages=long_request['messages'])
                assert first.usage.cache_creation_input_tokens > 0 and first.usage.cache_read_input_tokens == 0
                assert warm.usage.cache_read_input_tokens == first.usage.cache_creation_input_tokens
                assert warm.usage.cache_creation_input_tokens == 0
                for value in [first, warm]:
                    assert value.usage.input_tokens + value.usage.cache_creation_input_tokens + value.usage.cache_read_input_tokens == count.input_tokens
                assert first.content == warm.content
                for extra in [{'model': 'missing'}, {'max_tokens': 1024, 'stream': True}]:
                    try:
                        client.messages.create(**{**msg_request, **extra})
                    except anthropic.BadRequestError as e:
                        assert e.body['type'] == 'error' and e.body['error']['type'] == 'invalid_request_error'
                    else:
                        raise AssertionError('Messages SDK did not receive a typed error')
                for headers in [{'anthropic-version': 'invalid'}, {'x-api-key': 'invalid', 'Authorization': 'Bearer invalid'}]:
                    try:
                        client.messages.create(**msg_request, extra_headers=headers)
                    except (anthropic.BadRequestError, anthropic.AuthenticationError):
                        pass
                    else:
                        raise AssertionError('Messages invalid headers accepted')
                raw_headers = {'x-api-key': key, 'anthropic-version': '2023-06-01', 'Content-Type': 'application/json'}
                req = urllib.request.Request(base + '/v1/messages/count_tokens', headers=raw_headers,
                    data=json.dumps(dict(model=model, messages=msg_request['messages'], system=msg_request['system'])).encode())
                with urllib.request.urlopen(req, timeout=120) as response:
                    assert json.load(response)['input_tokens'] == equivalent['usage']['prompt_tokens']
                structured_args = dict(model=model, messages=structured_request['messages'], max_tokens=96,
                    extra_body={'temperature': 0}, output_config={'format': {'type': 'json_schema', 'schema': structured_schema}})
                with client.messages.stream(**structured_args) as stream:
                    text = ''.join(stream.text_stream)
                    structured_final = stream.get_final_message()
                assert json.loads(text) == structured_value and structured_final.stop_reason == 'end_turn'
                anthropic_tools = [{'name': 'weather', 'description': 'Get weather for a city.', 'input_schema': tool_schema}]
                tool_args = dict(model=model, messages=tool_request['messages'], max_tokens=96,
                    tools=anthropic_tools, tool_choice={'type': 'tool', 'name': 'weather'}, extra_body={'temperature': 0})
                with client.messages.stream(**tool_args) as stream:
                    events = list(stream)
                    tool_final = stream.get_final_message()
                assert tool_final.stop_reason == 'tool_use' and len(tool_final.content) == 1
                tool = tool_final.content[0]
                assert tool.type == 'tool_use' and tool.name == 'weather' and tool.input == {'city': 'Berlin'}
                counted_tool = client.messages.count_tokens(model=model, messages=tool_request['messages'], tools=anthropic_tools,
                    extra_body={'tool_choice': {'type': 'tool', 'name': 'weather'}})
                assert counted_tool.input_tokens == tool_final.usage.input_tokens + tool_final.usage.cache_creation_input_tokens + tool_final.usage.cache_read_input_tokens
                assert any(e.type == 'content_block_delta' and e.delta.type == 'input_json_delta' for e in events)
                history = tool_request['messages'] + [
                    {'role': 'assistant', 'content': [{'type': 'tool_use', 'id': tool.id, 'name': tool.name, 'input': tool.input}]},
                    {'role': 'user', 'content': [{'type': 'tool_result', 'tool_use_id': tool.id, 'content': 'Sunny, 18 degrees Celsius.'}]}]
                follow = client.messages.create(**{**tool_args, 'messages': history, 'tool_choice': {'type': 'none'}})
                assert follow.content[0].type == 'text' and follow.stop_reason in ['end_turn', 'max_tokens']
                partial = client.messages.create(**{**tool_args, 'max_tokens': 1})
                assert partial.stop_reason == 'max_tokens' and partial.content == []
                print(f'Anthropic SDK {anthropic.__version__}: Messages, SSE, tools/results, JSON, stops, cache usage, counting and errors passed')
        print('Real-model template, buffered/streamed output, stops, usage, disconnect and validation checks passed')
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
