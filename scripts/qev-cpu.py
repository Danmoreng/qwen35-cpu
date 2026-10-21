#!/usr/bin/env python3
"""Text-only qev experiment using existing CPU CLIs and per-question encoding.

Requires tokenizers and decider-ai==1.9.0. No PyTorch or GPU runtime is needed.
Each question starts a CLI process; use the sequential benchmark runner for timing.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
from types import SimpleNamespace
import struct
import subprocess

from decider.prompt import build, label_table
from decider.systemone import format_answer, render_question, render_state
from tokenizers import Tokenizer


class TokenizerAdapter:
    def __init__(self, directory):
        self.tokenizer = Tokenizer.from_file(str(directory / 'tokenizer.json'))

    def encode(self, text, add_special_tokens=False):
        return self.tokenizer.encode(text, add_special_tokens=add_special_tokens).ids


class NoShuffle:
    def shuffle(self, values):
        pass

    def sample(self, values, count):
        return values[:count]


def encode(tokenizer, state, question):
    rendered = render_question(question)
    options = []
    for option in rendered['options']:
        key = option.strip().lower()
        options.append('not listed here' if key.startswith('none of the above') or
                       key in ('none', 'n/a', 'none of these') else option)
    example = SimpleNamespace(context=render_state(state), qs=[SimpleNamespace(
        text=rendered['question'], options=options, gold=0)])
    row = build(example, tokenizer, NoShuffle(), max_options=255,
                max_ctx_tokens=1000000, layout='state_first', chat=None)
    if row['slots'] != [len(row['ids']) - 1]:
        raise ValueError('Expected one final answer slot')
    return row['ids'], rendered


def read_logits(path, letters):
    with path.open('rb') as stream:
        header = stream.read(24)
        magic, version, vocab, records = struct.unpack('<8sIIQ', header)
        if magic != b'Q35LGT1\0' or version != 1 or records != 1:
            raise ValueError('Expected one Q35LGT1 logit record')
        if stream.read(4) != struct.pack('<i', letters[0]):
            raise ValueError('Unexpected forced target')
        values = struct.unpack(f'<{vocab}f', stream.read(4 * vocab))
        if stream.read(1) or not all(math.isfinite(x) for x in values):
            raise ValueError('Invalid logit payload')
        return [values[i] for i in letters]


def distribution(logits, temperature):
    # Upstream: option softmax, followed by probability tempering with a 1e-9 floor.
    maximum = max(logits)
    values = [math.exp(x - maximum) for x in logits]
    total = sum(values)
    values = [max(x / total, 1e-9) ** (1 / temperature) for x in values]
    return [x / sum(values) for x in values]


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model-dir', type=Path, required=True)
    parser.add_argument('--requests', type=Path, required=True,
                        help='JSONL: state and questions, optionally id and expected')
    parser.add_argument('--output', type=Path, required=True,
                        help='New directory for tokens, commands, logits and answers')
    parser.add_argument('--backend', choices=['native', 'llama'])
    parser.add_argument('--executable', type=Path)
    parser.add_argument('--weights', type=Path)
    parser.add_argument('--threads', type=int, default=8)
    parser.add_argument('--max-context', type=int, default=8192)
    parser.add_argument('--prepare-only', action='store_true')
    args = parser.parse_args()
    if not args.prepare_only and not all((args.backend, args.executable, args.weights)):
        parser.error('Inference requires --backend, --executable and --weights')
    if args.threads < 1:
        parser.error('--threads must be positive')
    args.output.mkdir(parents=True, exist_ok=False)
    tokenizer = TokenizerAdapter(args.model_dir)
    cfg = json.loads((args.model_dir / 'qev_config.json').read_text())
    letters = label_table(tokenizer)[1]
    if letters != cfg['letter_token_ids']:
        raise ValueError('Tokenizer option IDs differ from qev_config.json')
    results = []
    for line in args.requests.read_text().splitlines():
        if not line.strip():
            continue
        request = json.loads(line)
        result = {'id': request.get('id', len(results)), 'answers': {}, 'rows': []}
        for key, question in request['questions'].items():
            ids, rendered = encode(tokenizer, request['state'], question)
            if len(ids) >= args.max_context:
                raise ValueError('Prompt exceeds context limit')
            folder = args.output / f'row-{sum(len(r["rows"]) for r in results) + len(result["rows"]):04d}'
            folder.mkdir()
            tokens = folder / 'tokens.txt'
            tokens.write_text(','.join(map(str, ids)))
            (folder / 'prompt.txt').write_text(tokenizer.tokenizer.decode(ids))
            row = {'question': key, 'tokens': len(ids), 'tokens_file': str(tokens),
                   'rendered': rendered}
            if not args.prepare_only:
                forced = folder / 'forced.txt'
                forced.write_text(str(letters[0]))
                logits = folder / 'output.logits'
                if args.backend == 'native':
                    command = [str(args.executable.resolve()), '--model-dir', str(args.model_dir),
                               '--weights', str(args.weights), '--tokens-file', str(tokens),
                               '--forced-tokens-file', str(forced), '--logits-out', str(logits),
                               '--threads', str(args.threads), '--max-context', str(args.max_context)]
                else:
                    command = [str(args.executable.resolve()), '--cpu-gguf', str(args.weights),
                               '--prompt-tokens-file', str(tokens), '--forced-output-tokens', str(letters[0]),
                               '--max-new-tokens', '1', '--logits-out', str(logits),
                               '--profile-json', str(folder / 'profile.json'),
                               '--cpu-threads', str(args.threads), '--max-context', str(args.max_context)]
                row['command'] = command
                with (folder / 'stdout.txt').open('w') as out, (folder / 'stderr.txt').open('w') as err:
                    completed = subprocess.run(command, stdout=out, stderr=err)
                if completed.returncode:
                    raise RuntimeError(f'CPU inference failed; see {folder / "stderr.txt"}')
                values = read_logits(logits, letters)[:len(rendered['names'])]
                probs = distribution(values, cfg['temperatures'][rendered['type']])
                row.update(letter_logits=values, probabilities=probs)
                result['answers'][key] = format_answer(rendered, probs)
            result['rows'].append(row)
        if 'expected' in request:
            result['expected'] = request['expected']
        results.append(result)
        print(json.dumps({k: v for k, v in result.items() if k != 'rows'}), flush=True)
    metadata = {'encoding': 'per_question', 'backend': args.backend, 'results': results,
                'input_sha256': sha(args.requests), 'config_sha256': sha(args.model_dir / 'qev_config.json'),
                'tokenizer_sha256': sha(args.model_dir / 'tokenizer.json')}
    if not args.prepare_only:
        metadata.update(weights_sha256=sha(args.weights), executable_sha256=sha(args.executable))
    (args.output / 'results.json').write_text(json.dumps(metadata, indent=2) + '\n')


if __name__ == '__main__':
    main()
