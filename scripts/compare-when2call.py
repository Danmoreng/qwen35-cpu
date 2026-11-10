#!/usr/bin/env python3
"""Re-score frozen qev outputs and public When2Call decision-model references.

No model inference or fitting. Raw downloaded snapshots remain under benchmarks/.
Requires Python 3.11+ and the pinned NVIDIA test file.
"""
import argparse
from collections import Counter
import csv
from datetime import datetime
import hashlib
from html.parser import HTMLParser
import json
from pathlib import Path

import when2call as bench

AA_URL = "https://artificialanalysis.ai/evaluations/when2call"
METHOD_URL = "https://artificialanalysis.ai/methodology/decision-index"
STUDY_REPO = "https://github.com/sammy143/decision-models-eval"
SIZES = {
    "baseten_elephant-9b": 9,
    "baseten_mammoth-27b": 27,
    "cloudflare_clef": 27,
    "perplexity_pplx-decider-flash-v1-2": 4,
    "perplexity_pplx-decider-v1-2": 27,
}
SELECTED = set(SIZES) | {"typesafe_jev-1-14", "liquid-ai_d1-ultra"}
SIZE_SOURCES = {
    "baseten_elephant-9b": "https://artificialanalysis.ai/decision-models/elephant-9b",
    "baseten_mammoth-27b": "https://artificialanalysis.ai/decision-models/mammoth-27b",
    "cloudflare_clef": "https://huggingface.co/Cloudflare/clef/blob/171287b2571042abec9e7b99752aee58216b0ed0/README.md",
    "perplexity_pplx-decider-flash-v1-2": "https://huggingface.co/perplexity-ai/pplx-decider-flash-v1.2/blob/9ff7cf9838e59df461c276b7699d56b244d2af58/README.md",
    "perplexity_pplx-decider-v1-2": "https://huggingface.co/perplexity-ai/pplx-decider-flash-v1.2/blob/9ff7cf9838e59df461c276b7699d56b244d2af58/README.md",
}


class Page(HTMLParser):
    def __init__(self):
        super().__init__()
        self.in_script = False
        self.scripts = []
        self.text = []

    def handle_starttag(self, tag, attrs):
        if tag == "script":
            self.in_script = True

    def handle_endtag(self, tag):
        if tag == "script":
            self.in_script = False

    def handle_data(self, text):
        (self.scripts if self.in_script else self.text).append(text)


def aa_models(html):
    page = Page(); page.feed(html)
    chunks = []
    for script in page.scripts:
        marker = "self.__next_f.push("
        if script.startswith(marker):
            value = json.loads(script[len(marker):-1])
            if len(value) > 1 and isinstance(value[1], str):
                chunks.append(value[1])
    models = {}

    def walk(value):
        if isinstance(value, dict):
            if value.get("kind") == "decisionModel" and "scores" in value:
                models[value["slug"]] = value
            for child in value.values():
                walk(child)
        elif isinstance(value, list):
            for child in value:
                walk(child)

    for line in "".join(chunks).splitlines():
        try:
            value = json.loads(line.split(":", 1)[1])
        except (ValueError, IndexError):
            continue
        walk(value)
    if not models:
        raise ValueError("No AA model scores found in the frozen page")
    return list(models.values())


def aa_exclusions(html):
    page = Page(); page.feed(html)
    text = [line.strip() for line in page.text if line.strip()]
    start = text.index("Task ID")
    excluded = []
    for line in text[start+1:]:
        if line.startswith(("live_multiple_", "live_simple_")):
            excluded.append(line)
        elif excluded:
            break
    if len(excluded) != len(set(excluded)) or not excluded:
        raise ValueError("Invalid AA exclusion list")
    return set(excluded)


def select_cohort(items, excluded):
    # source_id groups several variants. AA excludes only the ambiguous
    # request-for-information variant, not the tool/cannot-answer siblings.
    removed = [item for item in items if item["source_id"] in excluded
               and item["correct_answer"] == "request_for_info"]
    if Counter(item["source_id"] for item in removed) != Counter({key: 1 for key in excluded}):
        raise ValueError("Exclusions must identify exactly one ask-information item each")
    ids = {item["uuid"] for item in removed}
    return [item for item in items if item["uuid"] not in ids], removed


def write_csv(path, rows):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader(); writer.writerows(rows)


def decision_counts(rows, precision):
    return dict(tasks=len(rows),
                correct=sum(row[precision+"_api_choice"] == row["expected"] for row in rows),
                saved_probability_argmax_correct=sum(row[precision+"_prediction"] == row["expected"] for row in rows))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--research", type=Path, required=True)
    p.add_argument("--dataset", type=Path, required=True)
    p.add_argument("--predictions", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    assert bench.sha(args.dataset) == bench.DATA_SHA
    items = bench.records(args.dataset)
    by_id = {item["uuid"]: item for item in items}
    assert len(items) == len(by_id) == 3652
    predictions = list(csv.DictReader(args.predictions.open()))
    assert len(predictions) == 3652 and {row["id"] for row in predictions} == set(by_id)
    assert all(row["expected"] == by_id[row["id"]]["correct_answer"] for row in predictions)
    excluded = aa_exclusions((args.research / "aa-methodology.html").read_text())
    assert len(excluded) == 229
    kept, removed = select_cohort(items, excluded)
    assert len(kept) == 3423 and len(removed) == 229
    kept_ids = {item["uuid"] for item in kept}
    aa = aa_models((args.research / "aa-when2call.html").read_text())
    assert len(aa) == 17
    args.output.mkdir(parents=True, exist_ok=True)
    published = []
    for model in aa:
        accuracy = model["scores"]["when2callAccuracy"] / 100
        correct = round(accuracy * 3423)
        assert abs(correct / 3423 - accuracy) < 1e-12
        published.append(dict(id=model["slug"], label=model["name"], provider=model["creatorName"],
            tasks=3423, correct=correct, accuracy=accuracy, source="aa", source_url=AA_URL,
            parameters_b=SIZES.get(model["slug"]), parameters_source_url=SIZE_SOURCES.get(model["slug"]),
            selected_for_chart=model["slug"] in SELECTED))
    published.sort(key=lambda row: -row["accuracy"])
    local = []
    for precision in ("Q4", "Q8", "BF16"):
        row = dict(id="qev-"+precision.lower(), label="qev-0.8B "+precision, provider="Local",
            parameters_b=.8, source="local_cpu" if precision == "Q4" else "local_gpu",
            hardware="AMD Ryzen 5 7600" if precision == "Q4" else "Intel Arc Pro B70")
        for name, selected in (("full", predictions), ("aa_subset", [r for r in predictions if r["id"] in kept_ids])):
            counts = decision_counts(selected, precision)
            row[name] = dict(**counts, accuracy=counts["correct"]/counts["tasks"])
        local.append(row)
    # Recompute ordinary accuracy from the independently published decisions.
    # The study's headline "skill" is chance-corrected and is not accuracy.
    study_pin = json.loads((args.research / "sammy143/commit.json").read_text())["sha"]
    study_scores = json.loads((args.research / "sammy143/results/exp1/scores.json").read_text())
    external, external_predictions = [], []
    labels = {"A": "direct", "B": "tool_call", "C": "request_for_info", "D": "cannot_answer"}
    for model, label in (("snap", "SubQ Snap preview"), ("jev-1.13", "Jev 1.13"), ("jev-1.14", "Jev 1.14")):
        path = args.research / f"sammy143/runs/exp1/{model}/results.jsonl"
        rows = [r for r in bench.records(path) if r["catalog_id"] == 62]
        assert len(rows) == 3652
        ids = [r["run_id"].rsplit(":", 1)[1] for r in rows]
        assert len(set(ids)) == 3652 and set(ids) == set(by_id)
        correct = 0
        for record, key in zip(rows, ids):
            assert record["track"] == by_id[key]["correct_answer"]
            assert record["status"] == "ok"
            answer = record["response"]["answers"]["q"]
            # Published decimal probabilities can serialize a tied maximum
            # as 0.46 versus 0.45999999999999996. Score the explicit choice.
            assert answer["probabilities"][answer["choice"]] >= max(answer["probabilities"].values()) - 1e-12
            predicted = labels[answer["choice"]]
            correct += predicted == record["track"]
            external_predictions.append(dict(model=model, id=key, expected=record["track"], predicted=predicted))
        score = study_scores["models"][model]["benchmarks"]["62"]
        assert abs(correct/3652 - score["raw"]) <= .00005
        external.append(dict(id=model, label=label, provider="Published study", source="independent",
            tasks=3652, correct=correct, accuracy=correct/3652, parameters_b=None,
            source_url=f"{STUDY_REPO}/blob/{study_pin}/runs/exp1/{model}/results.jsonl"))
    cohorts = []
    for name, n, peers in (("aa_subset", 3423, [r for r in published if r["selected_for_chart"]]),
                            ("full", 3652, external)):
        rows = [*peers, *[{key: value for key, value in row.items() if key not in ("full", "aa_subset")}
                           | row[name] for row in local]]
        rows.sort(key=lambda row: -row["accuracy"])
        cohorts.append(dict(id=name, tasks=n, metric="Explicit API-choice accuracy",
            note="Same task cohort; prompts and option order differ", models=rows))
    bench.save(args.output / "chart-data.json", dict(date=datetime.now().date().isoformat(), cohorts=cohorts))
    bench.save(args.output / "local-rescoring.json", local)
    write_csv(args.output / "aa-reported-results.csv", published)
    write_csv(args.output / "external-full-predictions.csv", external_predictions)
    write_csv(args.output / "aa-excluded-items.csv", [dict(id=r["uuid"], source_id=r["source_id"],
              expected=r["correct_answer"]) for r in sorted(removed, key=lambda row: row["uuid"])])
    bench.save(args.output / "provenance.json", dict(captured=datetime.now().astimezone().isoformat(),
        aa_source=AA_URL, aa_methodology=METHOD_URL, study_revision=study_pin,
        dataset_revision=bench.DATA_REV, dataset_sha256=bench.DATA_SHA,
        local_predictions_sha256=bench.sha(args.predictions), local_predictions_path=str(args.predictions),
        comparison_driver_sha256=bench.sha(__file__), no_new_inference=True,
        selection="Keep all items except request_for_info variants at the 229 AA source IDs",
        comparison_rule="Use explicit API choice; retain the earlier probability-argmax counts separately",
        raw_hashes={str(path.relative_to(args.research)): bench.sha(path)
                    for path in sorted(args.research.rglob("*")) if path.is_file()},
        chart_selection="All five AA models with size metadata here, plus hosted Jev 1.14 and d1-ultra references",
        limitations=["qev was not run by Artificial Analysis", "Prompts and answer orders differ",
                     "Hosted model parameter counts and runtime precision are not assumed",
                     "No likelihood-scored, normalized-accuracy or chance-corrected skill values are mixed in"]))
    print(json.dumps(local, indent=2))


if __name__ == "__main__":
    main()
