#!/usr/bin/env python3
"""Prepare and evaluate frozen NVIDIA When2Call MCQ items through qev's API.

All candidates are scored in one forward pass. This is a decision-model
adaptation, not NVIDIA's full-response likelihood or length-normalized score.
"""
import argparse
from collections import Counter
from datetime import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import socket
import statistics
import subprocess
import sys
import time
import urllib.error
import urllib.request
from zoneinfo import ZoneInfo

DATA_REV = "0582f7749df63a96fdc3070932e83e72396ace53"
CODE_REV = "ecc8d42388e91ab37e7e737d48e16e8ecea3d1dc"
DATA_SHA = "8c3694e583eeeb8dbc297e6cd90da70efc68efa4b6adb7227523e828c6b7b14c"
CATEGORIES = ("direct", "tool_call", "request_for_info", "cannot_answer")
ORDER_SEED = "qev-when2call-2026-11-10-v1"
INSTRUCTION = ("Which candidate assistant response is most appropriate for the user's question, "
               "given only the available tools? Select the response itself, without executing any tool.")
WEIGHTS = {
    "Q4": ("models/qev/model.q35h", "e7d61e7b191de6c2afb8663e8a1f8c5f4d52315e9492db7a690708d744f8dc69"),
    "Q8": ("models/qev-gguf/qev-0.8b-Q8_0.gguf", "07856dd5d6c7d1e19a9579ef324e5377649890a6438f7735ea5ac8935f2b2140"),
    "BF16": ("models/qev-gguf/qev-0.8b-bf16.gguf", "47a3e93109187add85f7905d507024ade0fc4c957a503183f3cc6106c68a9980"),
}


def save(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def records(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def git(*args, cwd=None):
    return subprocess.check_output(["git", *args], cwd=cwd, text=True).strip()


def request_for(item):
    """Expose only the available tools, user question and four candidate texts."""
    assert set(item["answers"]) == set(CATEGORIES)
    assert item["correct_answer"] in CATEGORIES
    order = sorted(CATEGORIES, key=lambda c: hashlib.sha256(
        (ORDER_SEED + ":" + item["uuid"] + ":" + c).encode()).digest())
    mapping = {f"option_{i+1}": category for i, category in enumerate(order)}
    # The default NVIDIA tool formatting preserves each tool JSON string.
    tools = "\n\n".join("<tool>" + tool + "</tool>" for tool in item["tools"])
    state = ("You are a helpful AI assistant.\n"
             "You have access to the following tools described in <tool></tool> "
             "which you can use to answer the user's questions.\n"
             "Only use a tool if it directly answers the user's question.\n\n"
             "To use a tool, return JSON in the following format:\n"
             '{"name": "tool_name", "arguments": {"argument1": "value1", "argument2": "value2", ...}}'
             "\n\n" + tools + "\n\nUser question:\n" + item["question"]).rstrip()
    request = dict(state=state, questions={"decision": dict(type="choice", instructions=INSTRUCTION,
                   criteria={key: item["answers"][category] for key, category in mapping.items()})})
    return dict(id=item["uuid"], source=item["source"], source_id=item["source_id"],
                expected=item["correct_answer"], option_categories=mapping,
                no_available_tools=not item["tools"], request=request)


def prepare(args):
    if sha(args.dataset) != DATA_SHA:
        raise ValueError("Expected the frozen, complete NVIDIA MCQ test file")
    items = records(args.dataset)
    assert len(items) == len({r["uuid"] for r in items}) == 3652
    tasks = [request_for(r) for r in items]
    args.prepared.mkdir(parents=True, exist_ok=False)
    # Native protocol implementation, rather than an approximate token counter.
    proc = subprocess.run([str(args.protocol), "models/qev"], input="".join(
        json.dumps(r["request"]) + "\n" for r in tasks), text=True,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=180, check=True)
    encoded = [json.loads(line) for line in proc.stdout.splitlines()]
    assert len(encoded) == len(tasks)
    for task, row in zip(tasks, encoded):
        assert len(row["rows"]) == 1
        task["native_tokens"] = len(row["rows"][0])
    max_tokens = max(r["native_tokens"] for r in tasks)
    max_context = next(n for n in (4096, 8192, 16384, 32768) if n >= max_tokens)
    (args.prepared / "tasks.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in tasks))
    counts = Counter(r["expected"] for r in tasks)
    save(args.prepared / "manifest.json", dict(dataset_revision=DATA_REV, code_revision=CODE_REV,
         dataset_sha256=DATA_SHA, tasks_sha256=sha(args.prepared / "tasks.jsonl"), items=len(tasks),
         categories=dict(counts), correct_option_positions=dict(Counter(
             next(k for k, v in r["option_categories"].items() if v == r["expected"]) for r in tasks)),
         order_seed=ORDER_SEED, option_order="SHA256(seed:uuid:category), no model outputs used",
         max_context=max_context, min_tokens=min(r["native_tokens"] for r in tasks),
         max_tokens=max_tokens, median_tokens=statistics.median(r["native_tokens"] for r in tasks),
         total_tokens=sum(r["native_tokens"] for r in tasks), protocol_sha256=sha(args.protocol),
         instruction=INSTRUCTION, metadata_in_model_input=False, truncation=False))
    print((args.prepared / "manifest.json").read_text(), flush=True)


def metrics(rows):
    confusion = {c: {p: 0 for p in (*CATEGORIES, "invalid")} for c in CATEGORIES}
    for row in rows:
        confusion[row["expected"]][row["predicted"] or "invalid"] += 1
    per_category = {}
    for category in CATEGORIES:
        tp = confusion[category][category]
        support = sum(confusion[category].values())
        predicted = sum(r[category] for r in confusion.values())
        precision = tp / predicted if predicted else 0.0
        recall = tp / support if support else None
        f1 = 2 * tp / (support + predicted) if support + predicted else 0.0
        per_category[category] = dict(support=support, predicted=predicted, correct=tp,
                                      precision=precision, recall=recall, f1=f1)
    valid = [r for r in rows if r["valid"]]
    latencies = sorted(r["latency_ms"] for r in rows)
    hallucination = [r for r in rows if r["expected"] == "cannot_answer" and r["no_available_tools"]]
    return dict(requests=len(rows), n_correct=sum(r["correct"] for r in rows),
                accuracy=sum(r["correct"] for r in rows) / len(rows), valid=len(valid),
                brier=statistics.mean(r["brier"] for r in valid) if valid else None,
                macro_f1_present=statistics.mean(v["f1"] for v in per_category.values() if v["support"]),
                macro_f1_four=statistics.mean(v["f1"] for v in per_category.values()),
                median_latency_ms=statistics.median(latencies), p95_latency_ms=latencies[math.ceil(.95*len(latencies))-1],
                no_tool_cannot_answer_cases=len(hallucination),
                no_tool_hallucination_rate=sum(r["predicted"] == "tool_call" for r in hallucination)/len(hallucination)
                if hallucination else None,
                top_probability_ties=sum(r.get("top_probability_tie", False) for r in rows),
                api_choice_disagreements=sum(r.get("api_choice_disagrees", False) for r in rows),
                per_category=per_category, confusion=confusion)


def fetch(url):
    with urllib.request.urlopen(url, timeout=2) as response:
        return json.load(response)


def evaluate(args):
    now = datetime.now(ZoneInfo("Europe/Berlin"))
    cutoff = now.replace(hour=22, minute=40, second=0, microsecond=0)
    available = (cutoff - now).total_seconds() - 30
    if available < 120:
        raise ValueError("Insufficient time before the daily shutdown handoff")
    deadline = time.monotonic() + min(available, args.timeout_seconds)
    manifest = json.loads((args.prepared / "manifest.json").read_text())
    assert sha(args.prepared / "tasks.jsonl") == manifest["tasks_sha256"]
    tasks = records(args.prepared / "tasks.jsonl")
    if args.limit < 0 or args.limit > len(tasks) or args.timeout_seconds < 1:
        raise ValueError("Invalid sample size or timeout")
    if args.limit:
        # Frozen input-only sample. Its results are not full-test accuracy.
        tasks = sorted(tasks, key=lambda r: hashlib.sha256(("timing:" + r["id"]).encode()).digest())[:args.limit]
    if args.profile_json.exists():
        raise ValueError("Use a fresh profile path")
    out = Path(str(args.profile_json) + ".evidence")
    out.mkdir(parents=True, exist_ok=False)
    weights, expected_sha = WEIGHTS[args.precision]
    assert sha(weights) == expected_sha
    native = args.precision == "Q4"
    if native and args.backend != "cpu":
        raise ValueError("The native Q4 engine is CPU-only")
    server = Path("build-jevbench/qwen35_cpu_server" if native else
                  ".cache/llama-qev-vulkan/bin/llama-server" if args.backend == "vulkan" else
                  "build-dale-llama/bin/llama-server").resolve()
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0)); port = sock.getsockname()[1]
    model_id = "qev-0.8b-" + args.precision
    command = ([str(server), "--model-dir", "models/qev", "--threads", "6", "--port", str(port),
                "--max-context", str(manifest["max_context"]), "--residents", "1",
                "--decision-encoding", "per_question", "--prefix-cache", "off"] if native else
               [str(server), "--model", weights, "--alias", model_id,
                "--n-gpu-layers", "999" if args.backend == "vulkan" else "0",
                "--threads", "6", "--threads-batch", "6", "--parallel", "1", "--ctx-size", str(manifest["max_context"]),
                "--batch-size", "2048", "--ubatch-size", "256", "--cache-type-k", "f16", "--cache-type-v", "f16",
                "--flash-attn", "on", "--no-warmup", "--cache-ram", "0", "--no-cache-prompt",
                "--host", "127.0.0.1", "--port", str(port)])
    if args.backend == "vulkan":
        command += ["--device", "Vulkan0", "-lv", "4"]
    identity = dict(started=now.isoformat(), precision=args.precision, model_sha256=expected_sha,
                    backend=args.backend,
                    worktree_revision=git("rev-parse", "HEAD"), server_sha256=sha(server), command=command,
                    native_engine_source_revision="04d432123663b701acfd410a5d0fe8208515c183" if native else None,
                    script_sha256=sha(__file__), tasks_sha256=manifest["tasks_sha256"], dataset_revision=DATA_REV,
                    affinity=sorted(os.sched_getaffinity(0)), max_context=manifest["max_context"],
                    selected_ids=[r["id"] for r in tasks], limit=args.limit, prefix_cache=False,
                    shared_libraries={str(p): sha(p) for p in server.parent.glob("*.so*")} if not native else {},
                    llama_revision=git("rev-parse", "HEAD", cwd=".cache/llama-dale") if not native else None)
    if identity["affinity"] != list(range(6)):
        raise ValueError("Run through the sequential runner with affinity 63 (six physical cores)")
    save(out / "identity.json", identity)
    save(out / "status.json", dict(status="running"))
    proc = None
    results = []
    env = dict(os.environ); env.pop("QWEN35_API_KEY", None)
    try:
        with (out / "server.log").open("wb") as log:
            proc = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, env=env)
            url = f"http://127.0.0.1:{port}"
            for _ in range(240):
                if proc.poll() is not None:
                    raise RuntimeError("Server failed; inspect server.log")
                try:
                    if fetch(url + "/health")["status"] == "ok":
                        break
                except (OSError, KeyError):
                    pass
                time.sleep(.25)
            else:
                raise TimeoutError("Server readiness exceeded 60 seconds")
            save(out / "health.json", fetch(url + "/health"))
            model_list = fetch(url + "/v1/models")
            expected_id = "qev-0.8b-H128-Q4-G32-DOT4" if native else model_id
            assert model_list["data"][0]["id"] == expected_id
            save(out / "models.json", model_list)
            with (out / "responses.jsonl").open("w") as raw, (out / "results.jsonl").open("w") as scored:
                for task in tasks:
                    if time.monotonic() > deadline:
                        raise TimeoutError("Benchmark deadline reached; partial results preserved")
                    request = dict(task["request"], model=model_id)
                    started = time.perf_counter_ns()
                    body = json.dumps(request).encode()
                    req = urllib.request.Request(url + "/v1/systemone", data=body,
                                                 headers={"Content-Type": "application/json"})
                    try:
                        with urllib.request.urlopen(req, timeout=min(120, max(1, deadline-time.monotonic()))) as response:
                            data = response.read(); status = response.status
                    except urllib.error.HTTPError as error:
                        data = error.read(); status = error.code
                    answer = json.loads(data)
                    elapsed = (time.perf_counter_ns()-started)/1e6
                    record = dict(id=task["id"], expected=task["expected"], predicted=None, correct=False,
                                  valid=False, latency_ms=elapsed, status=status, no_available_tools=task["no_available_tools"],
                                  response_sha256=hashlib.sha256(data).hexdigest())
                    probs = answer.get("answers", {}).get("decision", {}).get("probabilities", {})
                    valid = (status == 200 and set(probs) == set(task["option_categories"]) and
                             all(type(v) in (float, int) and math.isfinite(v) and 0 <= v <= 1 for v in probs.values()) and
                             abs(sum(probs.values())-1) <= .001)
                    if valid:
                        category_probs = {task["option_categories"][k]: v for k, v in probs.items()}
                        chosen = min(probs, key=lambda k: (-probs[k], k))
                        predicted = task["option_categories"][chosen]
                        usage = answer["usage"]
                        assert usage["output_tokens"] == 0
                        assert 0 < usage["input_tokens"] <= manifest["max_context"]
                        assert usage["input_tokens"] == task["native_tokens"], "Token-count parity changed"
                        assert answer.get("cache", {}).get("read_tokens", 0) == 0
                        api_choice = answer["answers"]["decision"].get("choice")
                        assert api_choice in probs
                        record.update(predicted=predicted, correct=predicted == task["expected"], valid=True,
                                      probs=category_probs, input_tokens=usage["input_tokens"],
                                      api_choice=task["option_categories"][api_choice],
                                      api_choice_disagrees=api_choice != chosen,
                                      top_probability_tie=sum(v == max(probs.values()) for v in probs.values()) > 1,
                                      brier=sum((v-int(k == task["expected"]))**2 for k, v in category_probs.items()))
                    raw.write(json.dumps(dict(id=task["id"], request=request, status=status,
                                             response_text=data.decode()), ensure_ascii=False) + "\n"); raw.flush()
                    scored.write(json.dumps(record) + "\n"); scored.flush()
                    results.append(record)
                    if len(results) % 100 == 0:
                        print(f"{args.precision}: {len(results)}/{len(tasks)}", flush=True)
            summary = metrics(results)
            summary.update(precision=args.precision, loading_included=False, full_test=not args.limit,
                           evidence=str(out), scope="Serial HTTP candidate-choice decisions, no prefix cache")
            save(out / "summary.json", summary)
            profile = {k: v for k, v in summary.items() if not isinstance(v, dict)}
            save(args.profile_json, profile)
            save(out / "status.json", dict(status="complete", finished=datetime.now().isoformat()))
            print(json.dumps(profile), flush=True)
    except BaseException as error:
        save(out / "status.json", dict(status="interrupted", completed=len(results), error=str(error)))
        raise
    finally:
        if proc is not None and proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill(); proc.wait()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    p = sub.add_parser("prepare")
    p.add_argument("--dataset", type=Path, required=True)
    p.add_argument("--prepared", type=Path, required=True)
    p.add_argument("--protocol", type=Path, default=Path("build-jevbench/systemone_protocol_test"))
    p = sub.add_parser("run")
    p.add_argument("--prepared", type=Path, required=True)
    p.add_argument("--precision", choices=WEIGHTS, required=True)
    p.add_argument("--backend", choices=("cpu", "vulkan"), default="cpu",
                   help="Vulkan is an external llama.cpp quality reference, not a native engine backend")
    p.add_argument("--profile-json", type=Path, required=True)
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--timeout-seconds", type=int, default=3600)
    args = parser.parse_args()
    (prepare if args.action == "prepare" else evaluate)(args)


if __name__ == "__main__":
    main()
