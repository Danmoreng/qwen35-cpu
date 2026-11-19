#!/usr/bin/env python3
"""Run a frozen text panel against one native server via the sequential runner."""

import argparse
import ast
import json
import math
import os
from pathlib import Path
import re
import socket
import statistics
import subprocess
import time
import unicodedata
from urllib.request import Request, urlopen


def unique_object(pairs):
    result = dict(pairs)
    if len(result) != len(pairs):
        raise ValueError("Duplicate JSON key")
    return result


def matches(kind, expected, text):
    if kind == "edit":
        return " ".join(unicodedata.normalize("NFC", text).split()) == " ".join(expected.split())
    if kind == "json":
        try:
            actual = json.loads(text, object_pairs_hook=unique_object)
            return json.dumps(actual, sort_keys=True, allow_nan=False) == json.dumps(expected, sort_keys=True)
        except ValueError:
            return False
    if kind == "python":
        fence = re.fullmatch(r"```(?:python)?\s*\n(.*?)\n```", text.strip(), re.DOTALL)
        try:
            return ast.dump(ast.parse(fence[1] if fence else text.strip())) == ast.dump(ast.parse(expected))
        except (SyntaxError, ValueError):
            return False
    raise ValueError(f"Unexpected task kind {kind}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--server", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--panel", type=Path, required=True)
    parser.add_argument("--threads", type=int, default=6)
    parser.add_argument("--profile-json", type=Path, required=True)
    args = parser.parse_args()
    panel = json.loads(args.panel.read_text(encoding="utf-8"))
    if args.profile_json.exists() or not panel or len({item["id"] for item in panel}) != len(panel):
        parser.error("A fresh profile path and a nonempty panel with unique IDs are required")
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    base = f"http://127.0.0.1:{port}"
    command = [str(args.server.resolve()), "--model-dir", str(args.model_dir.resolve()),
               "--weights", str(args.weights.resolve()), "--threads", str(args.threads),
               "--residents", "1", "--batch-width", "1", "--max-context", "4096",
               "--prefix-cache", "off", "--port", str(port)]
    with args.profile_json.with_suffix(".server.log").open("w") as log:
        server = subprocess.Popen(command, stdout=log, stderr=log, env={**os.environ, "QWEN35_API_KEY": ""})
        try:
            for _ in range(600):
                if server.poll() is not None:
                    raise RuntimeError("Server exited before readiness")
                try:
                    with urlopen(base + "/v1/models", timeout=1) as response:
                        model = json.load(response)["data"][0]["id"]
                    break
                except OSError:
                    time.sleep(0.05)
            else:
                raise RuntimeError("Server readiness deadline exceeded")
            rows = []
            for item in panel:
                prompt = f"<|im_start|>user\n{item['prompt']}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
                body = dict(model=model, prompt=prompt, temperature=0, max_tokens=256)
                request = Request(base + "/v1/completions", data=json.dumps(body).encode(),
                                  headers={"Content-Type": "application/json"})
                started = time.perf_counter()
                with urlopen(request, timeout=60) as response:
                    reply = json.load(response)
                elapsed = (time.perf_counter() - started) * 1000
                choice = reply["choices"][0]
                rows.append(dict(id=item["id"], kind=item["kind"], milliseconds=elapsed, **reply["usage"],
                                 finish_reason=choice["finish_reason"], text=choice["text"],
                                 passed=choice["finish_reason"] != "length" and
                                 matches(item["kind"], item["expected"], choice["text"])))
            peak_rss = None
            process_status = Path(f"/proc/{server.pid}/status")
            if process_status.is_file():
                peak = re.search(r"^VmHWM:\s+(\d+) kB$", process_status.read_text(), re.MULTILINE)
                if peak:
                    peak_rss = int(peak[1]) * 1024
            latencies = sorted(row["milliseconds"] for row in rows)
            result = dict(model=model, cases=len(rows), correct=sum(row["passed"] for row in rows),
                          http_p50_ms=statistics.median(latencies),
                          http_p95_ms=latencies[math.ceil(0.95 * len(rows)) - 1],
                          server_peak_rss_bytes=peak_rss, command=command, observations=rows)
            args.profile_json.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        finally:
            server.terminate()
            try:
                server.wait(timeout=5)
            except subprocess.TimeoutExpired:
                server.kill()
                server.wait()


if __name__ == "__main__":
    main()
