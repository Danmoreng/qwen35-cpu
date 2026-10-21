#!/usr/bin/env python3
"""Serve only the local demo and an explicit model-file allowlist on loopback."""
import argparse
import hashlib
import json
import mimetypes
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlsplit

FILES = ('model.q35h', 'config.json', 'tokenizer.json', 'tokenizer_config.json',
         'vocab.json', 'merges.txt', 'qev_config.json')
EXPECTED_MODEL = 'e7d61e7b191de6c2afb8663e8a1f8c5f4d52315e9492db7a690708d744f8dc69'

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model-dir', type=Path, default=Path('dist/qev-h128-public-final'))
    parser.add_argument('--directory', type=Path, default=Path('dist/wasm'))
    parser.add_argument('--port', type=int, default=8765)
    args = parser.parse_args()
    root, model = args.directory.resolve(), args.model_dir.resolve()
    files = []
    for name in FILES:
        path = model / name
        with path.open('rb') as stream:
            digest = hashlib.file_digest(stream, 'sha256').hexdigest()
        files.append({'name': name, 'bytes': path.stat().st_size, 'sha256': digest})
    if files[0]['sha256'] != EXPECTED_MODEL:
        raise SystemExit('This local demo expects the published calibrated qev checkpoint.')
    manifest = json.dumps({'format': 'qev-local-wasm/1', 'files': files,
                           'model_sha256': files[0]['sha256']}).encode()
    mimetypes.add_type('application/wasm', '.wasm')

    class Handler(SimpleHTTPRequestHandler):
        def __init__(self, *values, **kw):
            super().__init__(*values, directory=str(root), **kw)

        def end_headers(self):
            self.send_header('Cross-Origin-Opener-Policy', 'same-origin')
            self.send_header('Cross-Origin-Embedder-Policy', 'require-corp')
            self.send_header('Cross-Origin-Resource-Policy', 'same-origin')
            self.send_header('Cache-Control', 'no-cache')
            super().end_headers()

        def do_GET(self):
            path = unquote(urlsplit(self.path).path)
            if path == '/model/manifest.json':
                self.send_response(200)
                self.send_header('Content-Type', 'application/json')
                self.send_header('Content-Length', str(len(manifest)))
                self.end_headers(); self.wfile.write(manifest)
            elif path.startswith('/model/'):
                name = path[len('/model/'):]
                if name not in FILES:
                    self.send_error(404); return
                target = model / name
                self.send_response(200)
                self.send_header('Content-Type', 'application/octet-stream')
                self.send_header('Content-Length', str(target.stat().st_size))
                self.end_headers()
                try:
                    with target.open('rb') as stream:
                        while chunk := stream.read(1024 * 1024):
                            self.wfile.write(chunk)
                except (BrokenPipeError, ConnectionResetError):
                    pass
            else:
                candidate = (root / path.lstrip('/')).resolve()
                if not candidate.is_relative_to(root) or candidate.is_dir() and path != '/':
                    self.send_error(404); return
                super().do_GET()

        def do_HEAD(self):
            # Never expose arbitrary model paths through inherited translation.
            if unquote(urlsplit(self.path).path).startswith('/model/'):
                self.send_error(405); return
            super().do_HEAD()

    print(f'Local CPU/WASM demo: http://127.0.0.1:{args.port}', flush=True)
    print(f'Checkpoint verified: {files[0]["sha256"]}', flush=True)
    ThreadingHTTPServer(('127.0.0.1', args.port), Handler).serve_forever()

if __name__ == '__main__':
    main()
