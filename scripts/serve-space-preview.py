#!/usr/bin/env python3
"""Loopback-only preview of the staged static Space, including its remote model manifest."""
import argparse
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
import mimetypes
from pathlib import Path

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--directory', type=Path, required=True)
parser.add_argument('--port', type=int, default=8766)
args = parser.parse_args()
if not (args.directory / 'model' / 'manifest.json').is_file(): raise ValueError('Missing static model manifest')
mimetypes.add_type('application/wasm', '.wasm')


class Handler(SimpleHTTPRequestHandler):
    def __init__(self, *values, **kwargs):
        super().__init__(*values, directory=str(args.directory.resolve()), **kwargs)

    def end_headers(self):
        self.send_header('Cross-Origin-Opener-Policy', 'same-origin')
        self.send_header('Cross-Origin-Embedder-Policy', 'require-corp')
        self.send_header('Cross-Origin-Resource-Policy', 'cross-origin')
        self.send_header('Cache-Control', 'no-cache')
        super().end_headers()

    def list_directory(self, path):
        self.send_error(404); return None


print(f'Space preview: http://127.0.0.1:{args.port}', flush=True)
ThreadingHTTPServer(('127.0.0.1', args.port), Handler).serve_forever()
