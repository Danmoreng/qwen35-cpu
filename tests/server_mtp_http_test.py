#!/usr/bin/env python3
"""Optional real-4B HTTP regression for long MTP tool replies and byte stops."""

import argparse
import json
import os
from pathlib import Path
import socket
import subprocess
import tempfile
import time
from urllib.request import Request, urlopen


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--server", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--mtp-weights", type=Path, required=True)
    parser.add_argument("--mtp-depth", type=int, choices=(1, 2, 3, 4), default=3)
    args = parser.parse_args()
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    base = f"http://127.0.0.1:{port}"

    def request(path, body=None):
        return urlopen(Request(base + path, data=None if body is None else json.dumps(body).encode(),
                               headers={"Content-Type": "application/json"}), timeout=200)

    def payload(path, body=None):
        with request(path, body) as response:
            return json.load(response)

    def stream(path, body):
        with request(path, {**body, "stream": True, "stream_options": {"include_usage": True}}) as response:
            events = []
            for line in response:
                if not line.startswith(b"data: "):
                    continue
                if line.strip() == b"data: [DONE]":
                    return events
                events.append(json.loads(line[6:]))
        raise AssertionError("No terminal SSE event")

    command = [str(args.server.resolve()), "--model-dir", str(args.model_dir), "--weights", str(args.weights),
               "--mtp-weights", str(args.mtp_weights), "--mtp-depth", str(args.mtp_depth), "--prefix-cache", "auto",
               "--threads", "6", "--residents", "1", "--batch-width", "1", "--port", str(port)]
    with tempfile.TemporaryFile(mode="w+b") as log:
        process = subprocess.Popen(command, stdout=log, stderr=log, env={**os.environ, "QWEN35_API_KEY": ""})
        try:
            for _ in range(600):
                assert process.poll() is None, "Server exited during startup"
                try:
                    if payload("/health").get("status") == "ok":
                        break
                except OSError:
                    time.sleep(0.1)
            else:
                raise AssertionError("Server startup deadline exceeded")
            plain = {"prompt": "The first 20 positive integers are 1, 2, 3,", "temperature": 0, "max_tokens": 36}
            full = payload("/v1/completions", plain)["choices"][0]["text"]
            assert len(full) > 20, f"Byte-stop fixture stopped early: {full!r}"
            marker = full[10:20]
            stopped = payload("/v1/completions", {**plain, "stop": marker})
            draft = stream("/v1/completions", {**plain, "stop": marker, "draft": "mtp"})
            draft_choices = [choice for event in draft for choice in event.get("choices", [])]
            text = "".join(choice.get("text", "") for choice in draft_choices)
            assert text == stopped["choices"][0]["text"] == full.split(marker, 1)[0]
            assert draft_choices[-1]["finish_reason"] == "stop"
            draft_usage = next(event["usage"] for event in draft if event.get("usage"))
            assert draft_usage["completion_tokens"] == stopped["usage"]["completion_tokens"]
            print("Streamed byte-stop parity passed.")

            fixture_path = Path(__file__).with_name("fixtures") / "qwen35-mtp-long-tool.json"
            body = json.loads(fixture_path.read_text(encoding="utf-8"))
            body["chat_template_kwargs"] = {"enable_thinking": False}
            # This is a long-output/backpressure check; pin the offered write
            # independently of the checkpoint template's automatic tool choice.
            body["tool_choice"] = {"type": "function", "function": {"name": "write"}}
            ordinary = payload("/v1/chat/completions", body)
            before = payload("/v1/cache")["draft"]
            events = stream("/v1/chat/completions", {**body, "draft": "mtp"})
            after = payload("/v1/cache")["draft"]
            ordinary_choice = ordinary["choices"][0]
            ordinary_call = ordinary_choice["message"]["tool_calls"][0]["function"]
            choices = [choice for event in events for choice in event.get("choices", [])]
            draft_call = next(call["function"] for choice in choices
                              for call in choice.get("delta", {}).get("tool_calls", []))
            usage = next(event["usage"] for event in events if event.get("usage"))
            assert choices[-1]["finish_reason"] == ordinary_choice["finish_reason"] == "tool_calls"
            assert draft_call["name"] == ordinary_call["name"] == "write"
            assert json.loads(draft_call["arguments"]) == json.loads(ordinary_call["arguments"])
            assert usage["completion_tokens"] == ordinary["usage"]["completion_tokens"]
            assert usage["completion_tokens"] > 512, "Long tool fixture stopped early"
            accepted = after["accepted_tokens"] - before["accepted_tokens"]
            assert accepted > 256, f"Drafting stalled at the server output buffer: {accepted}"
            print(f"MTP tool reply matched across {usage['completion_tokens']} tokens, "
                  f"{accepted} drafts accepted.", flush=True)

        except BaseException:
            log.seek(0)
            print(log.read().decode(errors="replace"))
            raise
        finally:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=8)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()


if __name__ == "__main__":
    main()
