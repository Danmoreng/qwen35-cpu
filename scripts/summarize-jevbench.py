#!/usr/bin/env python3
"""Validate repeated JevBench evidence and export CPU/H100 chart aggregates.

Local measurements must contain one warmup and three complete measured passes
for each qev format. Published H100 numbers are explicitly a separate source.
"""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import socket
import statistics


def read(path):
    return json.loads(path.read_text())


def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def jsonl(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def save(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    root, out = a.root, a.output
    perf = root / "cpu-performance"
    metadata = read(perf / "metadata.json")
    assert metadata["runs"] == 3 and metadata["warmup_runs"] == 1
    assert metadata["affinity"] == 63
    baseline = jsonl(root / "qev-h128/results.jsonl")
    ids = {r["task_id"] for r in baseline}
    assert len(ids) == len(baseline) == 231
    tasks = {t["id"]: t for t in jsonl(root / "qev-h128/all.jsonl")}
    qev_models, samples, validations = [], [], []
    labels = [("qev-q4", "H128/Q4-G32-DOT4", "qev Q4", "native CPU"),
              ("qev-q8", "Q8_0", "qev Q8", "llama.cpp CPU"),
              ("qev-bf16", "BF16", "qev BF16", "llama.cpp CPU")]
    reference_usage = {r["task_id"]: r["usage"]["input_tokens"] for r in baseline}
    for case, (key, fmt, label, backend) in enumerate(labels):
        measured, signatures, identities, token_deltas = [], [], [], []
        for run in range(4):
            profile = read(perf / f"case-{case}-run-{run}.json")
            folder = perf / f"case-{case}-run-{run}.json.evidence"
            identity = read(folder / "identity.json")
            assert identity["affinity"] == list(range(6))
            records = jsonl(folder / "results.jsonl")
            assert read(folder / "status.json")["status"] == "complete"
            assert len(records) == 231 and {r["task_id"] for r in records} == ids
            assert all(r["ok"] and r["strict_valid"] and r["valid"] for r in records)
            assert all(r["usage"]["output_tokens"] == 0 for r in records)
            assert all(0 < r["usage"]["input_tokens"] < 4096 for r in records)
            for r in records:
                raw = folder / "raw" / (hashlib.sha256(r["task_id"].encode()).hexdigest()+".json")
                assert sha(raw) == r["raw_sha256"]
                answer = read(raw)["response"]["answers"]["decision"]
                probs = ({"yes": answer["noul"], "no": 1-answer["noul"]}
                         if answer["type"] == "noul" else answer["probabilities"])
                assert probs == r["probs_as_returned"]
                task = tasks[r["task_id"]]
                assert set(probs) == set(task["labels"])
                assert all(math.isfinite(v) and 0 <= v <= 1 for v in probs.values())
                assert abs(sum(probs.values()) - 1) <= .001
                predicted = min(probs, key=lambda k: (-probs[k], k))
                assert predicted == r["predicted"]
                assert (predicted == str(task["expected"])) == r["correct"]
            signature = {r["task_id"]: (r["predicted"], r["correct"], r["probs_as_returned"]) for r in records}
            signatures.append(signature)
            assert profile["n_correct"] == sum(bool(r["correct"]) for r in records)
            assert abs(profile["median_latency_ms"] - statistics.median(r["latency_s"] for r in records)*1000) < 1e-9
            command = identity["server_command"]
            port = int(command[command.index("--port")+1])
            with socket.socket() as sock:
                assert sock.connect_ex(("127.0.0.1", port)) != 0
            delta = {r["task_id"]: r["usage"]["input_tokens"] - reference_usage[r["task_id"]] for r in records
                     if r["usage"]["input_tokens"] != reference_usage[r["task_id"]]}
            token_deltas.append(delta)
            identities.append(identity)
            samples.append(dict(model=key, run=run, warmup=run == 0,
                                n_correct=profile["n_correct"], median_latency_ms=profile["median_latency_ms"],
                                p95_latency_ms=profile["p95_latency_ms"], brier=profile["brier"],
                                profile=str(perf / f"case-{case}-run-{run}.json")))
            if run:
                measured.append(profile)
        assert all(s == signatures[0] for s in signatures), "Answers/probabilities changed across repeated passes"
        assert all(d == token_deltas[0] for d in token_deltas)
        if case == 0:
            assert signatures[0] == {r["task_id"]: (r["predicted"], r["correct"], r["probs_as_returned"]) for r in baseline}
        weights = Path(identities[0].get("weights") or "models/qev/model.q35h")
        medians = [r["median_latency_ms"] for r in measured]
        qev_models.append(dict(id=key, model="qev-0.8b", chart_label=label, format=fmt,
                               source="local_cpu", hardware="AMD Ryzen 5 7600", backend=backend,
                               file_bytes=weights.stat().st_size, model_sha256=sha(weights),
                               correct=measured[0]["n_correct"], tasks=231, accuracy=measured[0]["accuracy"],
                               brier=measured[0]["brier"], median_latency_ms=statistics.median(medians),
                               median_min_ms=min(medians), median_max_ms=max(medians),
                               p95_latency_ms=statistics.median(r["p95_latency_ms"] for r in measured),
                               latency_aggregation="median of three 231-task pass medians; warmup excluded"))
        validations.append(dict(model=key, complete_passes=4, raw_hashes_verified=924,
                                strict_validity=1.0, repeat_probabilities_identical=True,
                                different_token_counts_vs_native=len(token_deltas[0]),
                                token_count_deltas=sorted(set(token_deltas[0].values())),
                                server_stopped=True))
    gemma = []
    for key, label, correct, brier, ms in [("gemma-e2b", "Gemma 4 E2B", 180, .313, 102),
                                         ("gemma-e4b", "Gemma 4 E4B", 187, .243, 139),
                                         ("gemma-12b", "Gemma 4 12B", 200, .173, 136),
                                         ("gemma-26b", "Gemma 4 26B-A4B", 207, .162, 245)]:
        gemma.append(dict(id=key, model="Strands Decider / "+label, chart_label=label,
                          source="aws_strands_report", hardware="NVIDIA H100 80 GB", backend="Strands Decider",
                          correct=correct, tasks=231, accuracy=correct/231, brier=brier, median_latency_ms=ms,
                          reference="https://strandsagents.com/blog/strands-decider-gemma-4-models/"))
    out.mkdir(parents=True, exist_ok=True)
    models = [qev_models[2], qev_models[1], qev_models[0], *gemma]
    measured_date = read(perf / "case-0-run-1.json.evidence/identity.json")["started"][:10]
    save(out / "chart-data.json", dict(benchmark="JevBench v1 public, 231 tasks", measured=measured_date,
                                      cpu_measurement_protocol="one full warmup plus three full measured passes, serial HTTP, loading excluded",
                                      models=models))
    q4 = qev_models[0]
    gap = abs(q4["correct"] - 180)
    caption = (f'qev-0.8B in H128/Q4-G32-DOT4: {q4["file_bytes"]/1e6:.0f} MB, {q4["correct"]}/231 ({100*q4["accuracy"]:.2f}%) on JevBench v1, '
               f'{q4["median_latency_ms"]:.0f} ms median on a Ryzen 5 7600 CPU. Q8/BF16 tested too. '
               "Chart includes AWS's reported Gemma 4/H100 results (E2B: 180/231). "
               f'Different hardware; {gap} task{"s" if gap != 1 else ""} apart.')
    assert len(caption) <= 280
    (out / "twitter-caption.txt").write_text(caption + "\n")
    save(out / "cpu-validation.json", validations)
    with (out / "cpu-samples.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(samples[0]), lineterminator="\n"); w.writeheader(); w.writerows(samples)
    fields = ["id", "model", "format", "source", "hardware", "backend", "file_bytes", "correct", "tasks", "accuracy",
              "brier", "median_latency_ms", "median_min_ms", "median_max_ms", "p95_latency_ms"]
    with (out / "cpu-comparison.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore", lineterminator="\n"); w.writeheader(); w.writerows(models)
    save(root / "cpu-identities.json", {label[0]: [read(perf / f"case-{i}-run-{r}.json.evidence/identity.json")
                                                  for r in range(4)] for i, label in enumerate(labels)})
    for model in qev_models:
        print(f'{model["chart_label"]}: {model["correct"]}/231, median {model["median_latency_ms"]:.3f} ms', flush=True)


if __name__ == "__main__":
    main()
