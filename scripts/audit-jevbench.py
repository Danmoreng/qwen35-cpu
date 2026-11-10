#!/usr/bin/env python3
"""Audit paired public-set results and prepare a quality-only token control.

Use --prepare before running qev-llama-study on jobs.tsv with each GGUF.
Run again afterwards to include those logits. This does not measure latency.
"""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import subprocess


def read(path):
    return json.loads(path.read_text())


def rows(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def save(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n")


def paired(first, second, boundary):
    changed = []
    for key, a in first.items():
        b = second[key]
        if a["predicted"] != b["predicted"]:
            changed.append(dict(task_id=key, first=a["predicted"], second=b["predicted"],
                                first_correct=a["correct"], second_correct=b["correct"],
                                boundary_difference=key in boundary,
                                first_margin=margin(a), second_margin=margin(b)))
    wins = sum(r["first_correct"] and not r["second_correct"] for r in changed)
    losses = sum(r["second_correct"] and not r["first_correct"] for r in changed)
    n = wins + losses
    p = min(1.0, 2 * sum(math.comb(n, i) for i in range(min(wins, losses) + 1)) / 2**n) if n else 1.0
    return dict(first_correct=sum(r["correct"] for r in first.values()),
                second_correct=sum(r["correct"] for r in second.values()),
                first_only_correct=wins, second_only_correct=losses,
                different_predictions=len(changed), exact_mcnemar_two_sided_p=p,
                cases=changed)


def margin(row):
    probs = sorted(row["probs_as_returned"].values(), reverse=True)
    return probs[0] - probs[1]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--prepare", action="store_true")
    p.add_argument("--protocol", type=Path, default=Path("build-jevbench/systemone_protocol_test"))
    p.add_argument("--model", type=Path, default=Path("models/qev"))
    a = p.parse_args()
    config = read(a.model / "qev_config.json")
    audit = a.root / "audit"
    audit.mkdir(exist_ok=True)
    a.output.mkdir(parents=True, exist_ok=True)
    tasks = {r["id"]: r for r in rows(a.root / "qev-h128/all.jsonl")}
    baseline = {}
    for case, name in enumerate(("Q4", "Q8", "BF16")):
        records = rows(a.root / f"cpu-performance/case-{case}-run-0.json.evidence/results.jsonl")
        baseline[name] = {r["task_id"]: r for r in records}
        assert len(records) == len(tasks) == 231
        assert set(baseline[name]) == set(tasks)
    boundary = {r["task_id"] for r in read(a.root / "gguf-prompt-parity.json")["differences"]}
    result = dict(tasks=len(tasks), comparisons={})
    for first, second in (("Q4", "BF16"), ("Q4", "Q8"), ("Q8", "BF16")):
        result["comparisons"][first + "_vs_" + second] = paired(baseline[first], baseline[second], boundary)
    if a.prepare:
        requests = []
        for key in tasks:
            filename = hashlib.sha256(key.encode()).hexdigest() + ".json"
            requests.append(read(a.root / "qev-h128/raw" / filename)["request"])
        output = subprocess.check_output([str(a.protocol), str(a.model)],
                                        input="".join(json.dumps(r) + "\n" for r in requests).encode())
        prepared = [json.loads(line) for line in output.splitlines()]
        assert len(prepared) == len(tasks)
        tokens = audit / "tokens"
        tokens.mkdir(exist_ok=True)
        jobs = []
        for key, row in zip(tasks, prepared):
            assert len(row["rows"]) == 1
            ids = row["rows"][0]
            assert len(ids) == baseline["Q4"][key]["usage"]["input_tokens"]
            path = tokens / (key + ".txt")
            path.write_text(",".join(map(str, ids)) + "\n")
            jobs.append(key + "\t" + str(path.resolve()))
        (audit / "jobs.tsv").write_text("\n".join(jobs) + "\n")
        letters = config["letter_token_ids"]
        (audit / "letters.txt").write_text(",".join(map(str, letters)) + "\n")
        save(audit / "requests.json", requests)
    controlled = {}
    for name in ("BF16", "Q8"):
        path = audit / (name.lower() + "-native-tokens.tsv")
        if not path.exists():
            continue
        requests = read(audit / "requests.json")
        assert len(requests) == len(tasks)
        records = {}
        logits = {line.split("\t")[0]: list(map(float, line.split("\t")[1:]))
                  for line in path.read_text().splitlines()}
        assert set(logits) == set(tasks)
        assert all(len(v) == len(config["letter_token_ids"]) and all(math.isfinite(x) for x in v)
                   for v in logits.values())
        for (key, task), request in zip(tasks.items(), requests):
            question = request["questions"]["decision"]
            if question["type"] == "noul":
                labels = ["no", "yes"]
            elif question["type"] == "score":
                labels = list(map(str, range(len(question["criteria"]))))
            else:
                labels = list(question["criteria"])
            values = logits[key][:len(labels)]
            temp = config["temperatures"][question["type"]]
            weights = [math.exp((v-max(values))/temp) for v in values]
            probs = {k: v/sum(weights) for k, v in zip(labels, weights)}
            assert set(probs) == set(task["labels"])
            predicted = min(probs, key=lambda k: (-probs[k], k))
            records[key] = dict(predicted=predicted, correct=predicted == str(task["expected"]),
                                probs_as_returned=probs)
        controlled[name] = records
    if controlled:
        result["token_control"] = dict(
            purpose="Quality only: exact native tokens, CPU llama.cpp; no latency inference",
            ubatch=512, batch=2048, threads=6, kv="F16", context=4096,
            correct={k: sum(r["correct"] for r in v.values()) for k, v in controlled.items()},
            versus_original={k: paired(v, baseline[k], boundary) for k, v in controlled.items()},
            q4_comparisons={k: paired(baseline["Q4"], v, boundary) for k, v in controlled.items()})
    save(a.output / "precision-audit.json", result)
    with (a.output / "precision-disagreements.csv").open("w", newline="") as stream:
        fieldnames = ["comparison", "task_id", "first", "second", "first_correct", "second_correct",
                      "boundary_difference", "first_margin", "second_margin"]
        writer = csv.DictWriter(stream, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for key, comparison in result["comparisons"].items():
            writer.writerows(dict(comparison=key, **row) for row in comparison["cases"])
    print(json.dumps({k: {f: v for f, v in value.items() if f != "cases"}
                      for k, value in result["comparisons"].items()}, indent=2))
    if controlled:
        print("Native-token control correct:", result["token_control"]["correct"])


if __name__ == "__main__":
    main()
