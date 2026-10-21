#!/usr/bin/env python3
"""Extract actual x86-64 TurboFan bytes from V8's Linux perf jitdump.

Format: Linux tools/perf/Documentation/jitdump-specification.txt. This reads
owned diagnostic browser output; it does not attach to any live process.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re
import struct
import subprocess


def extract(path, function_index):
    data = path.read_bytes()
    if len(data) < 40:
        raise ValueError(f'Truncated jitdump header: {path}')
    magic, version, header_size, machine, _, pid, timestamp, flags = struct.unpack_from('<IIIIIIQQ', data)
    if magic != 0x4A695444 or version != 1 or machine != 62 or not 40 <= header_size <= len(data):
        raise ValueError(f'Expected little-endian x86-64 jitdump v1: {path}')
    offset = header_size
    pattern = re.compile(rf'wasm-function\[{function_index}\]-{function_index}-turbofan$')
    while offset + 16 <= len(data):
        kind, size, _ = struct.unpack_from('<IIQ', data, offset)
        if size < 16 or offset + size > len(data):
            raise ValueError(f'Truncated jitdump record at {offset}: {path}')
        if kind == 0 and size >= 56:
            owner, tid, vma, address, length, index = struct.unpack_from('<IIQQQQ', data, offset + 16)
            name_end = data.find(b'\0', offset + 56, offset + size)
            if name_end < 0 or name_end + 1 + length > offset + size:
                raise ValueError(f'Invalid code-load payload at {offset}: {path}')
            name = data[offset + 56:name_end].decode('utf-8', errors='replace')
            if pattern.search(name):
                return data[name_end + 1:name_end + 1 + length], {
                    'jitdump': str(path), 'jitdump_sha256': hashlib.sha256(data).hexdigest(),
                    'pid': owner, 'tid': tid, 'name': name, 'function_index': function_index,
                    'address': hex(address), 'code_size': length, 'code_index': index,
                    'tier': 'TurboFan', 'file_timestamp': timestamp, 'flags': flags,
                }
        offset += size
    return None


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--jitdump', nargs='+', type=Path, required=True)
    p.add_argument('--function-index', type=int, required=True)
    p.add_argument('--output-prefix', type=Path, required=True)
    a = p.parse_args()
    skipped = []
    for path in a.jitdump:
        try:
            found = extract(path, a.function_index)
        except ValueError as error:
            skipped.append(str(error))
            continue
        if found:
            code, metadata = found
            prefix = a.output_prefix
            prefix.parent.mkdir(parents=True, exist_ok=True)
            binary, assembly = Path(str(prefix) + '.bin'), Path(str(prefix) + '.asm')
            binary.write_bytes(code)
            result = subprocess.run(['objdump', '-D', '-b', 'binary', '-m', 'i386:x86-64',
                                     '-M', 'intel', str(binary)], check=True, capture_output=True, text=True)
            assembly.write_text(result.stdout)
            metadata.update(skipped_inputs=skipped, code_sha256=hashlib.sha256(code).hexdigest(),
                            binary=str(binary), assembly=str(assembly),
                            note='Code-load bytes include padding/constant data; inspect loop ranges before claiming spills.')
            Path(str(prefix) + '.json').write_text(json.dumps(metadata, indent=2) + '\n')
            print(json.dumps(metadata))
            return
    raise SystemExit(f'No TurboFan wasm function {a.function_index} found')


if __name__ == '__main__':
    main()
