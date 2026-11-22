"""Score every Julia answer and pair it with both frozen Qev references."""

import gzip
import importlib.util
import io
import json
import math
import statistics
from pathlib import Path

import numpy as np

raw = Path("benchmarks/julia2-jev-skills")
qev = Path("benchmarks/qev-calibration-2026-10-21")
out = Path("docs/results/julia2-jev-skills-2026-11-22")
out.mkdir(exist_ok=True)


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


scorer = load_module("jev_skills", qev / "upstream/code/score.py")
study = load_module("calibration_study", Path("scripts/summarize-qev-study.py"))
refs = {"qev_bf16": "current-bf16-gpu.tsv", "qev_h128": "selected-pure-all.tsv"}
reference_logits = {
    model: study.study.load_logits(qev / file) for model, file in refs.items()
}


def distribution(values, names, temperature=1):
    prob = study.study.adapter.distribution(values[: len(names)], temperature)
    return dict(zip(names, prob, strict=True))


def answer(kind, prob):
    assert all(0 <= x <= 1 and math.isfinite(x) for x in prob.values())
    assert math.isclose(sum(prob.values()), 1, abs_tol=1e-6)
    return {"noul": prob["true"]} if kind == "noul" else {"probabilities": prob}


def read_panel(split):
    panel = qev / "panels" / split
    items = json.loads((panel / "items.json").read_text())
    rows = json.loads((panel / "rows.json").read_text())
    temps = json.loads((panel / "manifest.json").read_text())["temperatures"]
    mapping_file = "mapping.jsonl" if split == "final" else "development-mapping.jsonl"
    mapping = [
        json.loads(line) for line in (raw / mapping_file).read_text().splitlines()
    ]
    predictions_file = (
        "julia-predictions.jsonl.gz"
        if split == "final"
        else "development-predictions.jsonl.gz"
    )
    datasets = {model: [{} for _ in items] for model in ["julia", *refs]}
    with (
        (raw / f"{split}-logits.tsv").open() as stream,
        gzip.GzipFile(str(out / predictions_file), "wb", mtime=0) as compressed,
        io.TextIOWrapper(compressed, encoding="utf-8") as predictions,
    ):
        for row, meta, line in zip(rows, mapping, stream, strict=True):
            ident, *vals = line.split()
            assert ident == row["id"] == meta["row"] and len(vals) == 255
            ix, key = meta["item"], meta["key"]
            kind = meta["kind"]
            assert (ix, key) == (row["item"], row["question"]) and key not in datasets[
                "julia"
            ][ix]
            names = [
                ("true" if v else "false") if isinstance(v, bool) else str(v)
                for v in row["rendered"]["names"]
            ]
            assert set(names) == set(meta["options"])
            values = np.array(vals[: len(meta["options"])], dtype=np.float32)
            exp = np.exp(values - values.max())
            probs = exp / exp.sum(dtype=np.float32)
            p = dict(zip(meta["options"], map(float, probs), strict=True))
            if kind == "noul":
                total = p["true"] + p["false"]
                p = {k: v / total for k, v in p.items()}
            datasets["julia"][ix][key] = answer(kind, p)
            print(
                json.dumps(
                    {
                        "row": ident,
                        "item": items[ix]["id"],
                        "question": key,
                        "tokens": meta["tokens"],
                        "probabilities": p,
                    },
                    separators=(",", ":"),
                ),
                file=predictions,
            )
            for model in refs:
                pr = distribution(reference_logits[model][ident], names, temps[kind])
                datasets[model][ix][key] = answer(kind, pr)
    for data in datasets.values():
        assert all(
            set(ans) == set(item["questions"])
            for ans, item in zip(data, items, strict=True)
        )
    return items, rows, datasets


items, rows, datasets = read_panel("final")
assert len(items) == 9133 and len(rows) == 13601
reports = {model: scorer.score(items, data) for model, data in datasets.items()}
for model, file in [
    ("qev_bf16", "bf16-final-score.json"),
    ("qev_h128", "selected-pure-final-score.json"),
]:
    old = json.loads((qev / file).read_text())
    assert all(old[key] == value for key, value in reports[model].items()), model


def save(name, obj):
    (out / name).write_text(json.dumps(obj, indent=2) + "\n")


save("scores.json", reports)
for model, result in reports.items():
    briers = [
        s["brier"] for k, s in result["skills"].items() if k != "rules/missing_facts"
    ]
    print(
        model,
        result["pooled_all"],
        statistics.mean(briers),
        result["pooled_by_category"],
        result["extras"]["rules/missing_facts"],
    )

groups = sorted(set(study.group_id(i) for i in items))
skills = sorted(k for k in reports["julia"]["skills"] if k != "rules/missing_facts")
gids = {name: i for i, name in enumerate(groups)}
sids = {name: i for i, name in enumerate(skills)}
counts = np.zeros((len(groups), len(skills)))
diffs = np.zeros((len(refs), 2, *counts.shape))
for row in rows:
    ix, key = row["item"], row["question"]
    item = items[ix]
    target = item["targets"].get(key)
    if not target or item["skill"] not in sids:
        continue
    g, s = gids[study.group_id(item)], sids[item["skill"]]
    counts[g, s] += 1
    gold = max(target, key=target.get)

    def metrics(dataset):
        dist = scorer.to_dist(item["questions"][key], datasets[dataset][ix][key])
        best = max(dist.values())
        top = [name for name, v in dist.items() if v >= best - 1e-9]
        return np.asarray(
            [
                (gold in top) / len(top),
                sum((v - target.get(k, 0)) ** 2 for k, v in dist.items()),
            ]
        )

    julia = metrics("julia")
    for ri, ref in enumerate(refs):
        diffs[ri, :, g, s] += julia - metrics(ref)
delta = (diffs.sum(axis=2) / counts.sum(axis=0)).mean(axis=-1)
rng = np.random.default_rng(20261021)
samples = []
for _ in range(10):
    weights = rng.multinomial(
        len(groups), np.full(len(groups), 1 / len(groups)), size=100
    )
    samples.append(((weights @ diffs) / (weights @ counts)).mean(axis=-1))
interval = np.quantile(np.concatenate(samples, axis=2), [0.025, 0.975], axis=2)
paired = {
    ref: {
        "accuracy_delta_pp": 100 * delta[i, 0],
        "accuracy_ci95_pp": (100 * interval[:, i, 0]).tolist(),
        "brier_delta": delta[i, 1],
        "brier_ci95": interval[:, i, 1].tolist(),
    }
    for i, ref in enumerate(refs)
}
save(
    "paired.json",
    dict(
        method="Julia minus Qev; 1000 paired cluster bootstrap replicates grouped by request, game variants and missing-fact pairs",
        seed=20261021,
        groups=len(groups),
        skills=len(skills),
        labeled_questions=int(counts.sum()),
        comparisons=paired,
    ),
)
print(paired)


development_items, development_rows, development_answers = read_panel("development")
assert len(development_items) == 1035 and len(development_rows) == 1553
all_ids = {i["id"] for i in items + development_items}
published_items = json.loads((qev / "eval/items.json").read_text())
assert len(all_ids) == len(published_items) == 10168
assert all_ids == {i["id"] for i in published_items}
all_rows = {r["id"] for r in rows + development_rows}
published_rows = json.loads((qev / "eval/rows.json").read_text())
assert len(all_rows) == len(published_rows) == 15154
assert all_rows == {r["id"] for r in published_rows}
complete = {}
for model in datasets:
    indexed = {
        i["id"]: a
        for i, a in zip(
            items + development_items,
            datasets[model] + development_answers[model],
            strict=True,
        )
    }
    complete[model] = scorer.score(
        published_items, [indexed[i["id"]] for i in published_items]
    )
save("complete-suite.json", complete)
print("Complete suite:", {k: v["pooled_all"] for k, v in complete.items()})
