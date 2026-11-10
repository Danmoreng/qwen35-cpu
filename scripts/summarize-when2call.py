#!/usr/bin/env python3
"""Independently validate When2Call raw responses and export compact evidence."""
import argparse
import csv
from datetime import datetime
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import re
import socket
import statistics

spec = importlib.util.spec_from_file_location("when2call", Path(__file__).with_name("when2call.py"))
bench = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bench)


def read(path):
    return json.loads(path.read_text())


def validate(folder, tasks, expected_ids, partial=False):
    identity = read(folder / "identity.json")
    assert identity["affinity"] == list(range(6))
    assert (identity["selected_ids"][:len(expected_ids)] if partial else identity["selected_ids"]) == expected_ids
    assert read(folder / "status.json")["status"] == ("interrupted" if partial else "complete")
    scored = bench.records(folder / "results.jsonl")
    raw = bench.records(folder / "responses.jsonl")
    assert len(scored) == len(raw) == len(expected_ids)
    assert [r["id"] for r in scored] == [r["id"] for r in raw] == expected_ids
    for record, response in zip(scored, raw):
        task = tasks[record["id"]]
        assert record["expected"] == task["expected"]
        assert response["request"] == dict(task["request"], model="qev-0.8b-" + identity["precision"])
        assert record["response_sha256"] == hashlib.sha256(response["response_text"].encode()).hexdigest()
        assert record["status"] == response["status"]
        answer = json.loads(response["response_text"])
        if record["valid"]:
            assert response["status"] == 200
            probs = answer["answers"]["decision"]["probabilities"]
            assert set(probs) == set(task["option_categories"])
            assert all(type(v) in (int, float) and math.isfinite(v) and 0 <= v <= 1 for v in probs.values())
            assert abs(sum(probs.values())-1) <= .001
            category_probs = {task["option_categories"][k]: v for k, v in probs.items()}
            assert record["probs"] == category_probs
            chosen = min(probs, key=lambda k: (-probs[k], k))
            assert record["predicted"] == task["option_categories"][chosen]
            api_choice = answer["answers"]["decision"]["choice"]
            assert record["api_choice"] == task["option_categories"][api_choice]
            assert probs[api_choice] == max(probs.values())
            assert record["api_choice_disagrees"] == (api_choice != chosen)
            assert record["correct"] == (record["predicted"] == task["expected"])
            assert record["input_tokens"] == answer["usage"]["input_tokens"] == task["native_tokens"]
            assert answer["usage"]["output_tokens"] == 0
            assert answer.get("cache", {}).get("read_tokens", 0) == 0
            assert abs(record["brier"] - sum((v-int(k == task["expected"]))**2 for k, v in category_probs.items())) < 1e-12
        else:
            assert not record["correct"] and record["predicted"] is None
    summary = bench.metrics(scored)
    if not partial:
        original = read(folder / "summary.json")
        assert all(original[key] == value for key, value in summary.items())
    command = identity["command"]
    port = int(command[command.index("--port")+1])
    with socket.socket() as sock:
        assert sock.connect_ex(("127.0.0.1", port)) != 0, "Benchmark server still running"
    return summary, {r["id"]: r for r in scored}, identity


def calibration(rows, bins=10):
    groups = [[] for _ in range(bins)]
    for row in rows:
        if row["valid"]:
            p = max(row["probs"].values())
            groups[min(int(p*bins), bins-1)].append((p, row["correct"]))
    n = sum(map(len, groups))
    return sum(len(g)/n * abs(statistics.mean(p for p, _ in g)-statistics.mean(c for _, c in g))
               for g in groups if g) if n else None


def paired(first, second):
    wins = sum(a["correct"] and not second[key]["correct"] for key, a in first.items())
    losses = sum(not a["correct"] and second[key]["correct"] for key, a in first.items())
    changed = [dict(id=key, expected=a["expected"], first=a["predicted"], second=second[key]["predicted"],
                    first_correct=a["correct"], second_correct=second[key]["correct"],
                    first_probabilities=a.get("probs"), second_probabilities=second[key].get("probs"))
               for key, a in first.items() if a["predicted"] != second[key]["predicted"]]
    return dict(first_only_correct=wins, second_only_correct=losses, different_predictions=len(changed), cases=changed)


def quality_only(values):
    return {key: value for key, value in values.items() if key not in ("median_latency_ms", "p95_latency_ms")}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    tasks = {r["id"]: r for r in bench.records(a.root / "prepared/tasks.jsonl")}
    assert len(tasks) == 3652
    manifest = read(a.root / "prepared/manifest.json")
    assert bench.sha(a.root / "prepared/tasks.jsonl") == manifest["tasks_sha256"]
    sources = read(a.root / "quality-sources.json")
    driver_shas = {bench.sha(path) for path in (a.root / "drivers").glob("when2call*.py")}
    sample_ids = sorted(tasks, key=lambda key: hashlib.sha256(("timing:"+key).encode()).digest())[:128]
    a.output.mkdir(parents=True, exist_ok=True)
    rows, by_precision, per_category, validation, identities = [], {}, [], [], {}
    for case, precision in enumerate(("Q4", "Q8", "BF16")):
        folder = a.root / sources[precision]
        quality_meta = read(folder.parent / "metadata.json")
        assert quality_meta["runs"] == 1 and quality_meta["warmup_runs"] == 0 and quality_meta["affinity"] == 63
        quality, baseline, identity = validate(folder, tasks, list(tasks))
        assert identity["model_sha256"] == bench.WEIGHTS[precision][1]
        assert identity["tasks_sha256"] == manifest["tasks_sha256"] and identity["script_sha256"] in driver_shas
        if identity.get("backend") == "vulkan":
            log = (folder / "server.log").read_text()
            offloaded = re.search(r"offloaded (\d+)/(\d+) layers to GPU", log)
            assert offloaded and int(offloaded[1]) == int(offloaded[2]) > 0
            assert "Vulkan0 (Intel(R) Graphics (BMG G31))" in log
        by_precision[precision] = baseline
        identities[precision] = dict(quality=identity)
        started = datetime.fromisoformat(identity["started"])
        finished = datetime.fromisoformat(read(folder / "status.json")["finished"])
        if finished.tzinfo is None:
            finished = finished.replace(tzinfo=started.tzinfo)
        row = dict(precision=precision, quality_backend=identity.get("backend", "cpu"),
                   requests=quality["requests"], correct=quality["n_correct"],
                   accuracy=quality["accuracy"], brier=quality["brier"], ece_10_bins=calibration(baseline.values()),
                   api_choice_correct=sum(r.get("api_choice") == r["expected"] for r in baseline.values()),
                   macro_f1_present=quality["macro_f1_present"], macro_f1_four=quality["macro_f1_four"],
                   valid=quality["valid"], no_tool_hallucination_rate=quality["no_tool_hallucination_rate"],
                   no_tool_cannot_answer_cases=quality["no_tool_cannot_answer_cases"],
                   top_probability_ties=quality["top_probability_ties"], api_choice_disagreements=quality["api_choice_disagreements"],
                   quality_wall_time_s=(finished-started).total_seconds())
        rows.append(row)
        checked = 3652
        if precision != "Q4":
            sample_folder = a.root / f"cpu-validation/case-{case-1}-run-0.json.evidence"
            sample_meta = read(sample_folder.parent / "metadata.json")
            assert sample_meta["runs"] == 1 and sample_meta["warmup_runs"] == 0 and sample_meta["affinity"] == 63
            _, cpu_sample, cpu_identity = validate(sample_folder, tasks, sample_ids)
            assert cpu_identity.get("backend", "cpu") == "cpu"
            assert cpu_identity["model_sha256"] == identity["model_sha256"]
            assert cpu_identity["llama_revision"] == identity["llama_revision"]
            assert cpu_identity["tasks_sha256"] == manifest["tasks_sha256"]
            assert cpu_identity["script_sha256"] in driver_shas
            agreement = paired(cpu_sample, baseline)
            agreement.update(tasks=len(cpu_sample), first_backend="cpu", second_backend=identity["backend"],
                cpu_correct=sum(r["correct"] for r in cpu_sample.values()),
                gpu_correct=sum(baseline[key]["correct"] for key in cpu_sample),
                max_probability_delta=max(abs(r["probs"][category]-baseline[key]["probs"][category])
                    for key, r in cpu_sample.items() for category in bench.CATEGORIES))
            bench.save(a.output / (precision.lower()+"-cpu-quality-agreement.json"), agreement)
            identities[precision]["cpu_validation"] = cpu_identity
            checked += len(cpu_sample)
        for category, values in quality["per_category"].items():
            per_category.append(dict(checkpoint=precision, category=category, **values))
        bench.save(a.output / (precision.lower()+"-quality.json"), dict(
            **quality_only(quality), ece_10_bins=row["ece_10_bins"], api_choice_correct=row["api_choice_correct"],
            per_source={source: quality_only(bench.metrics([
                baseline[key] for key, task in tasks.items() if task["source"] == source]))
                for source in sorted({r["source"] for r in tasks.values()})}))
        validation.append(dict(precision=precision, full_test_rows=3652,
                               raw_response_hashes_verified=checked,
                               affinity=list(range(6)), server_stopped=True))
    partial_folder = a.root / "quality/case-1-run-0.json.evidence"
    if partial_folder.exists() and read(partial_folder / "status.json")["status"] == "interrupted":
        partial_count = len(bench.records(partial_folder / "results.jsonl"))
        _, partial, partial_identity = validate(partial_folder, tasks, list(tasks)[:partial_count], partial=True)
        assert partial_identity["model_sha256"] == bench.WEIGHTS["Q8"][1]
        assert partial_identity["script_sha256"] in driver_shas
        assert partial_identity["tasks_sha256"] == manifest["tasks_sha256"]
        agreement = paired(partial, by_precision["Q8"])
        agreement.update(tasks=partial_count, first_backend="cpu", second_backend="vulkan",
                         cpu_correct=sum(r["correct"] for r in partial.values()),
                         gpu_correct=sum(by_precision["Q8"][key]["correct"] for key in partial),
                         max_probability_delta=max(abs(r["probs"][category]-by_precision["Q8"][key]["probs"][category])
                             for key, r in partial.items() for category in bench.CATEGORIES),
                         excluded_from_full_test_accuracy=True)
        bench.save(a.output / "q8-partial-cpu-agreement.json", agreement)
        identities["Q8"]["partial_cpu"] = partial_identity
    bench.save(a.output / "comparison.json", dict(dataset="NVIDIA When2Call MCQ test", tasks=3652,
               dataset_revision=bench.DATA_REV, adapted_protocol=True, models=rows))
    comparisons = {}
    for first, second in (("Q4", "BF16"), ("Q4", "Q8"), ("Q8", "BF16")):
        difference = paired(by_precision[first], by_precision[second])
        difference["different_item_ids"] = [row["id"] for row in difference.pop("cases")]
        comparisons[first+"_vs_"+second] = difference
    bench.save(a.output / "paired-differences.json", comparisons)
    bench.save(a.output / "validation.json", validation)
    for name, data in (("comparison.csv", rows), ("per-category.csv", per_category)):
        with (a.output / name).open("w", newline="") as stream:
            w = csv.DictWriter(stream, fieldnames=list(data[0]), lineterminator="\n")
            w.writeheader(); w.writerows(data)
    predictions = []
    for key, task in tasks.items():
        row = dict(id=key, expected=task["expected"])
        for precision, results in by_precision.items():
            record = results[key]
            row[precision + "_prediction"] = record["predicted"]
            row[precision + "_api_choice"] = record.get("api_choice")
            for category in bench.CATEGORIES:
                row[precision + "_p_" + category] = record.get("probs", {}).get(category)
        predictions.append(row)
    with (a.output / "predictions.csv").open("w", newline="") as stream:
        w = csv.DictWriter(stream, fieldnames=list(predictions[0]), lineterminator="\n")
        w.writeheader(); w.writerows(predictions)
    bench.save(a.root / "identities.json", identities)
    compact_identities = {}
    for precision, measurements in identities.items():
        compact_identities[precision] = {}
        for name, identity in measurements.items():
            compact = {key: value for key, value in identity.items() if key != "selected_ids"}
            compact["selected_ids_count"] = len(identity["selected_ids"])
            compact["selected_ids_sha256"] = hashlib.sha256(
                json.dumps(identity["selected_ids"], separators=(",", ":")).encode()).hexdigest()
            compact_identities[precision][name] = compact
    bench.save(a.output / "manifest.json", dict(
        dataset=manifest, dataset_license="CC-BY-4.0", cpu="AMD Ryzen 5 7600", gpu="Intel Arc Pro B70",
        actual_protocol=dict(full_quality_passes_per_model=1, full_quality_items=3652,
            cpu_validation_items_per_reference=128, cpu_validation_passes=1,
            repeated_performance_series=False, generated_tokens=0, truncation=False,
            quality_backends={r["precision"]: r["quality_backend"] for r in rows}),
        gpu_build=read(a.root / "gpu-build.json"),
        llama_revision=identities["Q8"]["quality"]["llama_revision"],
        native_model_repository="danmoreng/qev-0.8b-H128-Q4-G32-DOT4",
        native_model_revision="eb167df11cbd6130513fedfcfdfb2ca11180b2de",
        gguf_model_repository="labqoat/qev-0.8b-GGUF",
        gguf_model_revision="79caef6979f61c4ee4fa6e894424789ce66042db",
        study_plan=read(a.root / "study-plan.json"),
        analysis_driver_sha256=bench.sha(__file__),
        archived_driver_hashes={path.name: bench.sha(path) for path in sorted((a.root / "drivers").glob("*.py"))},
        raw_evidence_root=str(a.root), engine_identities=compact_identities,
        evidence_hashes={str(path.relative_to(a.root)): bench.sha(path)
                        for stage in ("quality", "quality-gpu", "cpu-validation")
                        for path in sorted((a.root / stage).rglob("*"))
                        if path.is_file() and path.suffix in (".json", ".jsonl", ".log", ".txt")}))
    for name in ("quality-sources.json", "quality-cpu-matrix.json", "quality-gpu-matrix.json", "cpu-validation-matrix.json", "gpu-build.json"):
        bench.save(a.output / name, read(a.root / name))
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
