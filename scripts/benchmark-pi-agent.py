#!/usr/bin/env python3
"""Run the installed Pi Coding Agent against a local CPU server or run a fixed prefix probe."""

import argparse
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, HTTPServer
import json
import os
from pathlib import Path
import queue
import socket
import subprocess
import threading
import time
from urllib.error import HTTPError
from urllib.request import Request, urlopen


def json_file(path, data):
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def free_port():
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def get_json(url, timeout=2):
    with urlopen(url, timeout=timeout) as response:
        return json.load(response)


@contextmanager
def child(command, log, **kwargs):
    with log.open("w", encoding="utf-8") as output:
        process = subprocess.Popen(command, stderr=output, **kwargs)
        try:
            yield process
        finally:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=8)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()


def wait_ready(process, base):
    deadline = time.monotonic() + 120
    while process.poll() is None and time.monotonic() < deadline:
        try:
            if get_json(base + "/health").get("status") == "ok":
                return get_json(base + "/v1/models")["data"][0]["id"]
        except OSError:
            time.sleep(0.2)
    raise RuntimeError("Server failed to become ready; check server.log")


class Observer:
    def __init__(self, base, model, config, output, native_log):
        self.base = base
        self.model = model
        self.config = config
        self.output = output
        self.native_log = native_log
        self.requests = []
        self.turn = 0

    def complete(self, path, body, send=None):
        number = len(self.requests)
        request_path = self.output / f"http-{number:03}.json"
        body.update(model=self.model, stream=True, stream_options={"include_usage": True})
        if self.config["engine"] == "native":
            body.update(draft="mtp" if self.config["depth"] else "off")
        elif path == "/v1/chat/completions":
            body["chat_template_kwargs"] = {"enable_thinking": False}

        stats = {"turn": self.turn, "request": number, "path": path}
        self.requests.append(stats)
        json_file(request_path, body)
        response_log = self.output / f"http-{number:03}.sse"
        request = Request(self.base + path, data=json.dumps(body).encode(),
                          headers={"Content-Type": "application/json", "Accept": "text/event-stream"})
        start = time.perf_counter()
        if self.config["engine"] == "native":
            self.native_log.readlines()
        with response_log.open("wb") as raw:
            try:
                response = urlopen(request, timeout=240)
            except HTTPError as error:
                response = error
            with response:
                status = response.status if hasattr(response, "status") else response.code
                stats["status"] = status
                if send:
                    send(status, response.headers.get_content_type())
                for line in response:
                    raw.write(line)
                    if send:
                        send(line)
                    if line.startswith(b"data: ") and line.strip() != b"data: [DONE]":
                        data = json.loads(line[6:])
                        self.observe_event(data, start, stats)
        stats["wall_ms"] = 1000 * (time.perf_counter() - start)
        if self.config["engine"] == "native":
            for line in self.native_log.readlines():
                if line.startswith("BENCH_TIMING "):
                    stats["timings"] = json.loads(line.removeprefix("BENCH_TIMING "))
            cache = get_json(self.base + "/v1/cache")
            stats["draft_total"] = cache.get("draft")
        json_file(self.output / "requests.json", self.requests)
        return stats

    @staticmethod
    def observe_event(data, start, stats):
        if data.get("usage"):
            stats["usage"] = data["usage"]
        if data.get("timings"):
            stats["timings"] = data["timings"]
        if "error" in data:
            stats["error"] = data["error"]
        for choice in data.get("choices", ()):
            delta = choice.get("delta", {})
            if (delta.get("content") or delta.get("tool_calls") or delta.get("reasoning_content")
                    or choice.get("text")):
                ms = 1000 * (time.perf_counter() - start)
                stats.setdefault("first_output_ms", ms)
                stats["last_output_ms"] = ms
            if choice.get("finish_reason"):
                stats["finish"] = choice["finish_reason"]


@contextmanager
def proxy(observer, max_requests_per_turn):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt, *values):
            pass

        def do_POST(self):
            if (self.path != "/v1/chat/completions"
                    or sum(r["turn"] == observer.turn for r in observer.requests) >= max_requests_per_turn):
                self.send_error(429, "Benchmark request limit reached")
                return
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            body.pop("store", None)
            for tool in body.get("tools", ()):
                close_tool_schema(tool["function"]["parameters"])
            body.update(temperature=observer.config.get("temperature", 0), top_k=40, top_p=0.95, seed=83,
                        parallel_tool_calls=False, cache_prompt=True)
            if observer.config["engine"] == "ik":
                body["repeat_penalty"] = 1.0
            else:
                body["repetition_penalty"] = 1.0

            def send(line_or_status, media=None):
                if media:
                    self.send_response(line_or_status)
                    self.send_header("Content-Type", media)
                    self.end_headers()
                else:
                    self.wfile.write(line_or_status)
                    self.wfile.flush()

            observer.complete(self.path, body, send)

    with HTTPServer(("127.0.0.1", 0), Handler) as http:
        worker = threading.Thread(target=http.serve_forever, daemon=True)
        worker.start()
        try:
            yield f"http://127.0.0.1:{http.server_port}/v1"
        finally:
            http.shutdown()
            worker.join()


def close_tool_schema(schema):
    # Pi's file tools only use their declared keys. Supply the explicit JSON
    # Schema setting required by the native engine and send it to both servers.
    if schema.get("type") == "object":
        schema.setdefault("additionalProperties", False)
        for value in schema.get("properties", {}).values():
            close_tool_schema(value)
    elif schema.get("type") == "array":
        close_tool_schema(schema["items"])


def sandbox(shared_network=False):
    command = ["bwrap", "--unshare-all", "--die-with-parent",
               "--ro-bind", "/usr", "/usr", "--symlink", "usr/bin", "/bin",
               "--symlink", "usr/lib", "/lib", "--symlink", "usr/lib", "/lib64",
               "--proc", "/proc", "--dev", "/dev", "--tmpfs", "/tmp",
               "--ro-bind", "/etc", "/etc", "--clearenv", "--setenv", "HOME", "/tmp",
               "--setenv", "PATH", "/usr/bin", "--setenv", "LANG", "C.UTF-8"]
    if shared_network:
        command.append("--share-net")
    return command


def collect_events(process, mailbox):
    for line in process.stdout:
        mailbox.put(line)
    mailbox.put(None)


def run_pi(observer, config, fixture, output):
    workspace = output / "workspace"
    workspace.mkdir()
    settings = output / "pi-settings"
    settings.mkdir()
    json_file(settings / "settings.json", {"compaction": {"enabled": False}, "retry": {"enabled": False}})
    turns = []
    with proxy(observer, max_requests_per_turn=6) as endpoint:
        json_file(settings / "models.json", {"providers": {"benchmark": {
            "baseUrl": endpoint, "apiKey": "local", "api": "openai-completions",
            "compat": {"supportsStore": False, "supportsDeveloperRole": False,
                       "supportsReasoningEffort": False, "maxTokensField": "max_tokens"},
            "models": [{"id": "qwen35", "input": ["text"], "reasoning": False,
                        "contextWindow": 16384, "maxTokens": 1900}]
        }}})
        pi_dir = Path(config["pi"]).resolve()
        command = sandbox(shared_network=True) + [
            "--bind", str(workspace), "/work", "--chdir", "/work",
            "--ro-bind", str(pi_dir), "/pi",
            "--bind", str(settings), "/settings", "--setenv", "PI_CODING_AGENT_DIR", "/settings",
            "/usr/bin/node", "/pi/dist/bundle/cli.js", "--offline", "--mode", "rpc",
            "--model", "benchmark/qwen35", "--thinking", "off", "--no-session",
            "--no-context-files", "--no-skills", "--no-extensions", "--no-mcp",
            "--no-prompt-templates", "--tools", "read,write,edit",
            "--system-prompt", fixture["system"]]
        json_file(output / "pi-command.json", command)
        with child(command, output / "pi.stderr.log",
                   stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True) as pi:
            mailbox = queue.Queue()
            reader = threading.Thread(target=collect_events, args=(pi, mailbox), daemon=True)
            reader.start()
            with (output / "pi-events.jsonl").open("w", encoding="utf-8", buffering=1) as log:
                def receive(timeout):
                    line = mailbox.get(timeout=timeout)
                    if line is None:
                        raise RuntimeError("Pi exited; check pi.stderr.log")
                    log.write(line)
                    event = json.loads(line)
                    if event["type"] == "response" and not event["success"]:
                        raise RuntimeError(f"Pi RPC error: {event}")
                    return event

                pi.stdin.write('{"type":"get_state","id":"ready"}\n')
                pi.stdin.flush()
                while receive(30).get("id") != "ready":
                    pass
                for turn, task in enumerate(fixture["turns"], 1):
                    observer.turn = turn
                    pi.stdin.write(json.dumps({"type": "prompt", "id": f"turn-{turn}",
                                              "message": task["prompt"]}) + "\n")
                    pi.stdin.flush()
                    begin = time.perf_counter()
                    info = {"turn": turn, "tools": []}
                    while True:
                        event = receive(480)
                        if event["type"] == "tool_execution_end":
                            info["tools"].append({"name": event["toolName"], "error": event["isError"]})
                        if event["type"] == "message_end" and event["message"].get("role") == "assistant":
                            message = event["message"]
                            info["last_stop"] = message.get("stopReason")
                            if message.get("errorMessage"):
                                info.setdefault("errors", []).append(message["errorMessage"])
                        if event["type"] == "agent_settled":
                            break
                    info["wall_ms"] = 1000 * (time.perf_counter() - begin)
                    snapshot = output / f"turn-{turn}.html"
                    original = workspace / "index.html"
                    if original.is_file() and not original.is_symlink():
                        snapshot.write_bytes(original.read_bytes())
                    turns.append(info)
                    json_file(output / "turns.json", turns)
    return turns


def run_probe(observer):
    # Fixed, repeatable code-heavy prefix. Requests two and three share the exact text prefix.
    lines = [f"function cell{i}(row, col) {{ return row >= 0 && col >= 0 && row < {i + 5}; }}\n"
             for i in range(92)]
    prompt = ("<|im_start|>system\nYou are a coding assistant. Complete the JavaScript code."
              "<|im_end|>\n<|im_start|>user\nImplement a grid puzzle using this existing module.\n"
              + "".join(lines) + "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n")
    for turn, (cache, suffix) in enumerate([(False, ""), (True, ""), (True, "// follow-up\n")], 1):
        observer.turn = turn
        observer.complete("/v1/completions", {"prompt": prompt + suffix, "max_tokens": 144,
                                                    "temperature": 0, "top_k": 1, "seed": 83,
                                                    "cache_prompt": cache})


def run_context_probe(observer, lengths, generated):
    # Qwen encodes ASCII digits individually. Whole cycles keep the expected
    # continuation identical even when the prompt lengths differ.
    cycle = "01234567"
    expected = (cycle * ((generated + 7) // 8))[:generated]
    for length in lengths:
        if length <= 0 or length % len(cycle):
            raise ValueError("Context length must be a positive multiple of eight")
        stats = observer.complete("/v1/completions", {
            "prompt": cycle * (length // len(cycle)), "max_tokens": generated,
            "temperature": 0, "top_k": 1, "seed": 83, "cache_prompt": False})
        raw = observer.output / f"http-{stats['request']:03}.sse"
        chunks = [json.loads(line[6:]) for line in raw.read_bytes().splitlines()
                  if line.startswith(b"data: ") and line != b"data: [DONE]"]
        output = "".join(choice.get("text", "") for chunk in chunks
                         for choice in chunk.get("choices", []))
        usage = stats.get("usage", {})
        if (output != expected or usage.get("prompt_tokens") != length
                or usage.get("completion_tokens") != generated
                or stats.get("finish") != "length" or stats.get("status") != 200):
            raise RuntimeError(f"Fixed continuation mismatch at {length}: {output!r}")
        stats["decode_tokens_per_second"] = ((generated - 1) * 1000 /
            (stats["last_output_ms"] - stats["first_output_ms"]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    workload = parser.add_mutually_exclusive_group()
    workload.add_argument("--fixture", type=Path)
    workload.add_argument("--synthetic-context", type=int, nargs="+",
                          help="Exact prompt lengths for the matched digit continuation")
    parser.add_argument("--synthetic-output", type=int, default=128)
    parser.add_argument("--profile-json", type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    fixture = json.loads(args.fixture.read_text()) if args.fixture else None
    output = args.profile_json.with_suffix(".raw").resolve()
    output.mkdir()
    port = free_port()
    base = f"http://127.0.0.1:{port}"
    command = [*config["command"], "--port", str(port)]
    json_file(output / "server-command.json", command)
    server_log = output / "server.log"
    with child(command, server_log, cwd=output, stdout=subprocess.DEVNULL,
               env={**os.environ, "QWEN35_API_KEY": ""}) as process:
        model = wait_ready(process, base)
        with server_log.open(encoding="utf-8", errors="replace") as timing_log:
            observer = Observer(base, model, config, output, timing_log)
            if fixture:
                turns = run_pi(observer, config, fixture, output)
            else:
                turns = []
                if args.synthetic_context:
                    if args.synthetic_output < 2:
                        parser.error("--synthetic-output must be at least two")
                    run_context_probe(observer, args.synthetic_context, args.synthetic_output)
                else:
                    run_probe(observer)
    requests = observer.requests
    metrics = {"http_requests": len(requests), "wall_ms": sum(r["wall_ms"] for r in requests),
               "prompt_tokens": sum(r.get("usage", {}).get("prompt_tokens", 0) for r in requests),
               "completion_tokens": sum(r.get("usage", {}).get("completion_tokens", 0) for r in requests),
               "failed_requests": sum(r["status"] != 200 or "error" in r for r in requests),
               "turns": turns, "requests": requests}
    json_file(args.profile_json, metrics)


if __name__ == "__main__":
    main()
