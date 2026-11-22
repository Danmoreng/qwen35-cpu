"""Score the same NVIDIA items and both German panels by stable item ID."""

import csv
import gzip
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

out = Path(__file__).parent
raw = Path("benchmarks/julia2-transfer")


def records(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def logits(path):
    by_id = {}
    with path.open() as stream:
        for line in stream:
            ident, *values = line.split()
            assert ident not in by_id and len(values) == 255
            by_id[ident] = np.array(values, dtype=np.float32)
            assert np.isfinite(by_id[ident]).all()
    return by_id


def distribution(values, count):
    x = values[:count]
    exp = np.exp(x - x.max())
    return (exp / exp.sum()).tolist()


def summarize(rows):
    return {
        model: dict(
            correct=sum(r[model] == r["gold"] for r in rows),
            total=len(rows),
            by_target={
                target: dict(
                    correct=sum(
                        r[model] == target for r in rows if r["gold"] == target
                    ),
                    total=sum(r["gold"] == target for r in rows),
                )
                for target in sorted(set(r["gold"] for r in rows))
            },
        )
        for model in ("julia", "qev")
    }


def pairs(rows):
    groups = defaultdict(list)
    for r in rows:
        groups[r["group"]].append(r)
    assert all(len(g) == len(set(r["gold"] for r in g)) == 2 for g in groups.values())
    return {
        model: sum(all(r[model] == r["gold"] for r in g) for g in groups.values())
        for model in ("julia", "qev")
    }


def bootstrap(rows):
    groups = defaultdict(list)
    julia_only = qev_only = 0
    for r in rows:
        diff = int(r["julia"] == r["gold"]) - int(r["qev"] == r["gold"])
        groups[r["group"]].append(diff)
        julia_only += diff == 1
        qev_only += diff == -1
    size = np.array([len(g) for g in groups.values()])
    value = np.array([sum(g) for g in groups.values()])
    rng = np.random.default_rng(20261122)
    samples = []
    for _ in range(20):
        draw = rng.integers(0, len(groups), size=(100, len(groups)))
        samples.extend(100 * value[draw].sum(axis=1) / size[draw].sum(axis=1))
    return dict(
        julia_only=julia_only,
        qev_only=qev_only,
        delta_julia_minus_qev_pp=100 * (julia_only - qev_only) / len(rows),
        source_groups=len(groups),
        paired_cluster_ci95_pp=np.quantile(samples, [0.025, 0.975]).tolist(),
    )


ref_dir = Path("docs/results/when2call-2026-11-10")
with (ref_dir / "predictions.csv").open() as f:
    refs = {r["id"]: r for r in csv.DictReader(f)}
output = logits(raw / "logits.tsv")
when = []
for meta in records(raw / "options.jsonl"):
    ident = meta["id"]
    ref = refs[ident]
    probs = distribution(output.pop(ident), 4)
    cats = meta["categories"]
    gold = ref["expected"]
    when.append(
        dict(
            panel="when2call",
            id=ident,
            group=meta["source"] + ":" + meta["source_id"],
            gold=gold,
            julia=cats[np.argmax(probs)],
            qev=ref["BF16_prediction"],
            qev_h128=ref["Q4_prediction"],
            julia_probabilities=dict(zip(cats, probs)),
            tokens=meta["tokens"],
        )
    )
assert len(when) == len(refs) == len(set(r["id"] for r in when)) == 3652 and not output
nvidia = summarize(when)
assert nvidia["qev"]["correct"] == 1647
nvidia["qev_h128"] = dict(
    correct=sum(r["qev_h128"] == r["gold"] for r in when), total=len(when)
)
assert nvidia["qev_h128"]["correct"] == 1651
nvidia["paired"] = bootstrap(when)


# The two public benchmarks draw on BFCL. Exclude all derivatives of a
# source item if its ID or either user question occurs in the other suite.
def norm(text):
    return " ".join(text.casefold().split())


jev_path = Path("benchmarks/qev-calibration-2026-10-21/eval/items.json")
bfcl = [r for r in json.loads(jev_path.read_text()) if r["source"].startswith("bfcl:")]
ids = {r["id"].removeprefix("x-bfcl-") for r in bfcl}
assert len(ids) == len(bfcl) == 571
assert all("\nUser:\n" in r["state"] for r in bfcl)
texts = {norm(r["state"].rsplit("\nUser:\n", 1)[1]) for r in bfcl}
known = set()
for row in records(Path("benchmarks/decision-gpu-2026-11-20/when2call.jsonl")):
    if row["source_id"] in ids or any(
        isinstance(row.get(k), str) and norm(row[k]) in texts
        for k in ("question", "orig_question")
    ):
        known.add(row["source"] + ":" + row["source_id"])
filtered = [r for r in when if r["group"] not in known]
sensitivity = summarize(filtered)
sensitivity["removed_groups"] = len(known)
sensitivity["removed_rows"] = len(when) - len(filtered)
sensitivity["paired"] = bootstrap(filtered)

key = json.loads((raw / "german-key.json").read_text())
german = {
    i: dict(id=i, group=k["group"], gold=k["expected"], panel=k["panel"])
    for i, k in key.items()
}
for model in ("julia", "qev"):
    output = logits(raw / f"{model}-de.tsv")
    for meta in records(raw / f"{model}-de-options.jsonl"):
        i = meta["id"]
        scores = output.pop(i)[: len(meta["names"])]
        german[i][model] = key[i]["categories"][meta["names"][scores.argmax()]]
    assert not output and all(model in r for r in german.values())
routing = [r for r in german.values() if r["panel"] == "routing"]
fresh = [r for r in german.values() if r["panel"] != "routing"]
assert len(routing) == 64 and len(fresh) == 128
result = dict(
    when2call=nvidia,
    when2call_without_known_jev_sources=sensitivity,
    german64=summarize(routing),
    german64_paired=bootstrap(routing),
    contrast128={
        m: dict(
            correct=sum(r[m] == r["gold"] for r in fresh),
            total=128,
            both_per_pair=pairs(fresh)[m],
            pairs=64,
        )
        for m in ("julia", "qev")
    },
    contrast_by_family={
        family: pairs([r for r in fresh if r["panel"] == family])
        for family in ("deadline", "code")
    },
    contrast_paired=bootstrap(fresh),
)
packed = "".join(
    json.dumps(r, ensure_ascii=False, separators=(",", ":")) + "\n"
    for r in [*when, *german.values()]
)
(out / "predictions.jsonl.gz").write_bytes(gzip.compress(packed.encode(), mtime=0))
(out / "scores.json").write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps(result, indent=2))
