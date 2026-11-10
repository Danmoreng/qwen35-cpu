#!/usr/bin/env python3
"""Run frozen JevBench v1 against native Q4 or published llama.cpp Q8/BF16 qev.

Uses the upstream typesafe adapter and scorer without changes. This collects
quality evidence only; recorded request times are not a performance comparison.
Raw evidence must live outside the JevBench checkout. No model fitting occurs.
"""
import argparse
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time
import urllib.request
from zoneinfo import ZoneInfo

JEVBENCH_REVISION = "1bcc55eb6c8cffde2306b3db03ede39b61c6152a"
MODEL_ID = "qev-0.8b-H128-Q4-G32-DOT4"
MODEL_SHA256 = "e7d61e7b191de6c2afb8663e8a1f8c5f4d52315e9492db7a690708d744f8dc69"
GGUF_SHA256 = {
    "Q8_0": "07856dd5d6c7d1e19a9579ef324e5377649890a6438f7735ea5ac8935f2b2140",
    "BF16": "47a3e93109187add85f7905d507024ade0fc4c957a503183f3cc6106c68a9980",
}


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def git(directory, *args):
    return subprocess.check_output(["git", "-C", str(directory), *args], text=True).strip()


def save(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")


def get(url):
    with urllib.request.urlopen(url, timeout=2) as response:
        return json.load(response)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--harness", type=Path, required=True)
    parser.add_argument("--server", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--backend", choices=["native", "llama"], default="native")
    parser.add_argument("--weights", type=Path, help="Pinned published GGUF for the llama backend")
    parser.add_argument("--gguf-precision", choices=GGUF_SHA256)
    parser.add_argument("--llama-source", type=Path, help="Source checkout used to build llama-server")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--profile-json", type=Path,
                        help="Sequential benchmark runner output; evidence defaults to PROFILE.evidence")
    parser.add_argument("--threads", type=int, default=6)
    parser.add_argument("--timeout-seconds", type=int, default=3600)
    args = parser.parse_args()
    if not args.output and not args.profile_json:
        parser.error("Require --output or --profile-json")
    if args.profile_json and args.profile_json.exists():
        parser.error("Use a new profile path")
    now = datetime.now(ZoneInfo("Europe/Berlin"))
    cutoff = now.replace(hour=22, minute=45, second=0, microsecond=0)
    available = (cutoff - now).total_seconds() - 30
    if available < 120 or args.timeout_seconds < 1 or args.threads < 1:
        parser.error("Insufficient time before 22:45 Europe/Berlin or invalid limits")
    harness = args.harness.resolve()
    model = args.model_dir.resolve()
    server = args.server.resolve()
    out = (args.output or Path(str(args.profile_json) + ".evidence")).resolve()
    if out.is_relative_to(harness):
        parser.error("Raw evidence must be outside the benchmark checkout")
    if git(harness, "rev-parse", "HEAD") != JEVBENCH_REVISION:
        parser.error("JevBench is not at the frozen article-comparison revision")
    if git(harness, "status", "--porcelain", "--untracked-files=no"):
        parser.error("The benchmark checkout has tracked modifications")
    if args.backend == "native":
        if args.weights or args.gguf_precision or args.llama_source:
            parser.error("GGUF options require --backend llama")
        weights, expected_sha, model_id = model / "model.q35h", MODEL_SHA256, MODEL_ID
    else:
        if not all((args.weights, args.gguf_precision, args.llama_source)):
            parser.error("llama requires --weights, --gguf-precision and --llama-source")
        weights = args.weights.resolve()
        expected_sha = GGUF_SHA256[args.gguf_precision]
        model_id = "qev-0.8b-" + args.gguf_precision
    if sha(weights) != expected_sha:
        parser.error("Expected the pinned published qev artifact for this precision")
    out.mkdir(parents=True, exist_ok=False)
    tasks = b"".join((harness / "datasets/public" / (name + ".jsonl")).read_bytes()
                     for name in ("original", "easy", "hard"))
    rows = [json.loads(line) for line in tasks.splitlines() if line.strip()]
    if len(rows) != 231 or len({row["id"] for row in rows}) != 231:
        raise ValueError("Expected exactly 231 unique public tasks")
    (out / "all.jsonl").write_bytes(tasks)
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    url = f"http://127.0.0.1:{port}"
    if args.backend == "native":
        server_command = [str(server), "--model-dir", str(model), "--threads", str(args.threads),
                          "--port", str(port), "--max-context", "4096", "--residents", "1",
                          "--decision-encoding", "per_question", "--prefix-cache", "off"]
    else:
        server_command = [str(server), "--model", str(weights), "--alias", model_id,
                          "--n-gpu-layers", "0", "--threads", str(args.threads),
                          "--threads-batch", str(args.threads), "--parallel", "1",
                          "--ctx-size", "4096", "--batch-size", "2048", "--ubatch-size", "256",
                          "--cache-type-k", "f16", "--cache-type-v", "f16", "--flash-attn", "on",
                          "--no-warmup", "--cache-ram", "0", "--no-cache-prompt",
                          "--host", "127.0.0.1", "--port", str(port)]
    command = [sys.executable, "-m", "jevbench.cli", "run", "--tasks", str(out / "all.jsonl"),
               "--adapter", "typesafe", "--endpoint", url, "--key-env", "", "--model", model_id,
               "--run-label", model_id, "--revision", "sha256:" + expected_sha,
               "--cost-basis", "local_cpu_compute_excluded", "--reserve-usd", "0",
               "--results", str(out / "results.jsonl"), "--raw-dir", str(out / "raw"),
               "--ledger", str(out / "ledger.jsonl"), "--manifest", str(out / "run-manifest.json")]
    project = Path(__file__).resolve().parents[1]
    identity = dict(started=now.isoformat(), jevbench_revision=JEVBENCH_REVISION,
                    tasks_sha256=sha(out / "all.jsonl"), model_sha256=expected_sha,
                    backend=args.backend, weights=str(weights), model_id=model_id,
                    model_files={p.name: sha(p) for p in sorted(model.iterdir()) if p.is_file()},
                    engine_revision=git(project, "rev-parse", "HEAD"),
                    engine_worktree_status=git(project, "status", "--short"),
                    server_sha256=sha(server), script_sha256=sha(__file__),
                    server_command=server_command, benchmark_command=command,
                    affinity=sorted(os.sched_getaffinity(0)), python=sys.version,
                    max_context=4096, prefix_cache=False, encoding="per_question",
                    purpose="Quality only; request times are not a performance benchmark")
    if args.backend == "llama":
        identity.update(llama_revision=git(args.llama_source, "rev-parse", "HEAD"),
                        llama_worktree_status=git(args.llama_source, "status", "--short"))
        # The server entry point is small and dynamically linked; preserve the
        # implementation and tensor-library identities as well as the launcher.
        identity["shared_libraries"] = {str(p): sha(p) for p in sorted(server.parent.glob("*.so*")) if p.is_file()}
    save(out / "identity.json", identity)
    env = dict(os.environ)
    env.pop("QWEN35_API_KEY", None)
    env["PYTHONPATH"] = str(harness)
    process = None
    started = time.monotonic()
    status = {"status": "running"}
    save(out / "status.json", status)
    try:
        with (out / "server.log").open("wb") as log:
            process = subprocess.Popen(server_command, stdout=log, stderr=subprocess.STDOUT, env=env)
            for _ in range(240):
                if process.poll() is not None:
                    raise RuntimeError("Server exited before readiness; see server.log")
                try:
                    health = get(url + "/health")
                    models = get(url + "/v1/models")
                    if health["status"] == "ok" and models["data"][0]["id"] == model_id:
                        break
                except (OSError, KeyError):
                    pass
                time.sleep(.25)
            else:
                raise TimeoutError("Server readiness exceeded 60 seconds")
            save(out / "health.json", dict(health=health, models=models, pid=process.pid))
            remaining = min(args.timeout_seconds, available) - (time.monotonic() - started)
            with (out / "jevbench.log").open("wb") as run_log:
                subprocess.run(command, env=env, stdout=run_log, stderr=subprocess.STDOUT,
                               check=True, timeout=max(1, remaining))
            summary_command = [sys.executable, "-m", "jevbench.cli", "summarize",
                               "--tasks", str(out / "all.jsonl"), "--results", str(out / "results.jsonl"),
                               "--ledger", str(out / "ledger.jsonl"),
                               "--public-export", str(out / "summary.json")]
            with (out / "summarize.log").open("wb") as log:
                subprocess.run(summary_command, env=env, stdout=log, stderr=subprocess.STDOUT,
                               check=True, timeout=30)
            summary = json.loads((out / "summary.json").read_text())
            if not summary["complete"] or summary["n_attempted"] != 231:
                raise RuntimeError("Incomplete run; inspect preserved raw evidence")
            save(out / "health-after.json", get(url + "/health"))
            status = dict(status="complete", n_correct=summary["n_correct"],
                          n_attempted=summary["n_attempted"], accuracy=summary["accuracy"])
            if args.profile_json:
                save(args.profile_json, dict(requests=summary["n_attempted"],
                     n_correct=summary["n_correct"], accuracy=summary["accuracy"],
                     schema_validity_strict=summary["schema_validity_strict"],
                     median_latency_ms=summary["latency"]["p50_s"] * 1000,
                     p95_latency_ms=summary["latency"]["p95_s"] * 1000,
                     brier=summary["brier_mean"], backend=args.backend,
                     evidence=str(out), loading_included=False,
                     scope="Serial HTTP decisions over all 231 JevBench v1 tasks; no prefix cache"))
            print(json.dumps(status), flush=True)
    except BaseException as exc:
        status = dict(status="interrupted", error=str(exc))
        raise
    finally:
        if process is not None and process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
        status["finished"] = datetime.now(ZoneInfo("Europe/Berlin")).isoformat()
        save(out / "status.json", status)


if __name__ == "__main__":
    main()
