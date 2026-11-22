"""Render the frozen When2Call comparison with Julia's published text interface."""

import csv
import hashlib
import json
import runpy
from pathlib import Path

from julia2 import Tokenizer, prompt

raw = Path("benchmarks/julia2-transfer")
suite = runpy.run_path("scripts/when2call.py")
data = Path("benchmarks/decision-gpu-2026-11-20/when2call.jsonl")
assert hashlib.sha256(data.read_bytes()).hexdigest() == suite["DATA_SHA"]
references = Path("docs/results/when2call-2026-11-10/predictions.csv")
with references.open() as stream:
    expected = {r["id"]: r["expected"] for r in csv.DictReader(stream)}
tok = Tokenizer(Path("benchmarks/julia2-jev-skills/model/tokenizer"))
seen = set()
with (raw / "tokens.txt").open("w") as jobs, (raw / "options.jsonl").open("w") as meta:
    for item in suite["records"](data):
        task = suite["request_for"](item)
        ident = task["id"]
        assert ident not in seen and expected[ident] == task["expected"]
        seen.add(ident)
        req = task["request"]
        q = req["questions"]["decision"]
        options = prompt.options_for(q)
        ids = tok.encode(
            prompt.render(req["state"], q["type"], q["instructions"], options)
        )
        assert 0 < len(ids) <= 16384 and len(options) == 4
        print(ident, *ids, file=jobs)
        print(
            json.dumps(
                dict(
                    id=ident,
                    source=item["source"],
                    source_id=item["source_id"],
                    tokens=len(ids),
                    categories=[task["option_categories"][k] for k, _ in options],
                )
            ),
            file=meta,
        )
assert seen == set(expected) and len(seen) == 3652
