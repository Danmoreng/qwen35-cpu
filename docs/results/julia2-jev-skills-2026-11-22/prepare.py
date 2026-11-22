import argparse
from pathlib import Path
import json
import collections
import hashlib
from julia2 import Tokenizer, prompt

d = Path("benchmarks/julia2-jev-skills")
parser = argparse.ArgumentParser()
parser.add_argument("--panel", choices=["development", "final"], default="final")
split = parser.parse_args().panel
panel = Path("benchmarks/qev-calibration-2026-10-21/panels") / split
tok = Tokenizer(d / "model/tokenizer")
items = json.loads((panel / "items.json").read_text())
rows = json.loads((panel / "rows.json").read_text())
frozen = json.loads(
    Path("docs/results/qev-calibration-2026-10-21/panels.json").read_text()
)[split]
assert len(items) == frozen["items"]
assert len(rows) == len(set(r["id"] for r in rows)) == frozen["rows"]
assert (
    hashlib.sha256("\n".join(item["id"] for item in items).encode()).hexdigest()
    == frozen["ids_sha256"]
)
mapping = "mapping.jsonl" if split == "final" else "development-mapping.jsonl"
lengths = []
counts = collections.Counter()
with (
    (d / f"{split}-tokens.txt").open("w") as jobs,
    (d / mapping).open("w") as meta,
):
    for r in rows:
        item = items[r["item"]]
        q = item["questions"][r["question"]]
        assert isinstance(item["state"], str)
        options = prompt.options_for(q)
        names = [k for k, _ in options]
        ids = tok.encode(
            prompt.render(item["state"], q["type"], q.get("instructions", ""), options)
        )
        assert (
            0 < len(ids) <= 16384
            and len(names) == len(set(names))
            and 1 <= len(names) <= 255
        )
        lengths.append(len(ids))
        counts[q["type"]] += 1
        print(r["id"], *ids, file=jobs)
        print(
            json.dumps(
                dict(
                    row=r["id"],
                    item=r["item"],
                    key=r["question"],
                    kind=q["type"],
                    options=names,
                    tokens=len(ids),
                ),
                ensure_ascii=False,
            ),
            file=meta,
        )
print(len(lengths), min(lengths), max(lengths), sum(lengths), counts)
