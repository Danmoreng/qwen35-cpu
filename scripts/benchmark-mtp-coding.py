#!/usr/bin/env python3
"""Run coding and non-executing tool-call fixtures on the native HTTP server."""

import argparse
import json
import os
from pathlib import Path
import runpy
import socket
import subprocess
import time
from urllib.request import Request, urlopen


def changes(before, after):
    return {key: ([b - a for a, b in zip(before[key], value)] if isinstance(value, list)
                  else value - before[key]) for key, value in after.items()}


def execute_fixtures(base, model, depth, tasks, tools, matches):
    def get(path):
        with urlopen(base + path, timeout=2) as response:
            return json.load(response)

    def complete(body):
        request = Request(base + "/v1/chat/completions", data=json.dumps(body).encode("utf-8"),
                          headers={"Content-Type": "application/json"})
        started = time.perf_counter()
        with urlopen(request, timeout=75) as response:
            payload = json.load(response)
        return payload, 1000 * (time.perf_counter() - started)

    observations = []
    for item in tasks:
        before = get("/v1/cache")["draft"]
        messages = [{"role": "user", "content": item["prompt"]}]
        body = {"model": model, "messages": messages, "max_tokens": item.get("max_tokens", 96),
                "temperature": 0, "draft": "mtp" if depth else "off"}
        if item.get("sampled"):
            body.update(temperature=0.65, top_k=40, top_p=0.9, repetition_penalty=1.05, seed=83)
        if item.get("tools"):
            body.update(tools=[tools[name] for name in item["tools"]],
                        tool_choice=item.get("choice", "auto"), parallel_tool_calls=False)
        turns = []
        passed = True
        for expected in item.get("calls", []):
            reply, latency = complete(body)
            choice = reply["choices"][0]
            message = choice["message"]
            calls = message.get("tool_calls", [])
            observed = None
            if len(calls) == 1:
                fn = calls[0]["function"]
                observed = {"name": fn["name"], "arguments": json.loads(fn["arguments"])}
            turns.append(dict(milliseconds=latency, usage=reply["usage"],
                              finish=choice["finish_reason"], content=message.get("content"), call=observed))
            wanted = {key: expected[key] for key in ("name", "arguments")}
            if choice["finish_reason"] != "tool_calls" or json.dumps(observed, sort_keys=True) != json.dumps(wanted, sort_keys=True):
                passed = False
                break
            if "result" in expected:
                messages.extend([message, {"role": "tool", "tool_call_id": calls[0]["id"],
                                          "content": expected["result"]}])
        if passed and (not item.get("calls") or "expected" in item):
            if item.get("final_without_tools"):
                body["tool_choice"] = "none"
            reply, latency = complete(body)
            choice = reply["choices"][0]
            message = choice["message"]
            content = message.get("content")
            calls = [c["function"] for c in message.get("tool_calls", [])]
            turns.append(dict(milliseconds=latency, usage=reply["usage"], finish=choice["finish_reason"],
                              content=content, unexpected_calls=calls))
            if item["kind"] == "prose":
                passed = bool(content)
            else:
                passed = not calls and isinstance(content, str) and choice["finish_reason"] == "stop"
                if passed and item.get("expected_kind", item["kind"]) == "python":
                    passed = matches("python", item["expected"], content)
                elif passed:
                    passed = content.strip() == item["expected"]
        observations.append(dict(id=item["id"], kind=item["kind"], sampled=item.get("sampled", False),
                                 passed=passed, milliseconds=sum(turn["milliseconds"] for turn in turns),
                                 turns=turns, draft=changes(before, get("/v1/cache")["draft"])))
    return observations


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--server", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--mtp-weights", type=Path, required=True)
    parser.add_argument("--depth", type=int, choices=range(5), required=True)
    parser.add_argument("--fixtures", type=Path, required=True)
    parser.add_argument("--text-panel", type=Path, required=True)
    parser.add_argument("--profile-json", type=Path, required=True)
    parser.add_argument("--only", help="Run one fixture for a smoke test")
    args = parser.parse_args()
    fixtures = json.loads(args.fixtures.read_text(encoding="utf-8"))
    text_panel = json.loads(args.text_panel.read_text(encoding="utf-8"))
    tasks = [item for item in text_panel if item["kind"] == "python"] + fixtures["tasks"]
    if args.only:
        tasks = [item for item in tasks if item["id"] == args.only]
    if not tasks or len({item["id"] for item in tasks}) != len(tasks) or args.profile_json.exists():
        parser.error("Use unique task IDs, a known smoke fixture and a new profile path")
    matches = runpy.run_path(str(Path(__file__).with_name("benchmark-text-panel.py")))["matches"]
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    base = f"http://127.0.0.1:{port}"
    command = [str(args.server.resolve()), "--model-dir", str(args.model_dir), "--weights", str(args.weights),
               "--mtp-weights", str(args.mtp_weights), "--mtp-depth", str(max(args.depth, 1)),
               "--threads", "6", "--residents", "1", "--batch-width", "1", "--max-context", "4096",
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
            rows = execute_fixtures(base, model, args.depth, tasks, fixtures["tools"], matches)
            result = dict(model=model, depth=args.depth, tasks=len(rows),
                          passed=sum(row["passed"] for row in rows),
                          total_http_ms=sum(row["milliseconds"] for row in rows),
                          server_command=command, observations=rows)
            args.profile_json.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        finally:
            server.terminate()
            try:
                server.wait(timeout=5)
            except subprocess.TimeoutExpired:
                server.kill()
                server.wait()


if __name__ == "__main__":
    main()
