"""Use each publisher's ordering, artifacts and text encoder on identical requests."""

import json
import runpy
import sys
from pathlib import Path

raw = Path("benchmarks/julia2-transfer")
backend = sys.argv[1]
if backend == "julia":
    from julia2 import Tokenizer, prompt

    tokenizer = Tokenizer(Path("benchmarks/julia2-jev-skills/model/tokenizer"))
elif backend == "qev":
    adapter = runpy.run_path("scripts/qev-cpu.py")
    tokenizer = adapter["TokenizerAdapter"](Path("models/qev-0.8b"))
    (raw / "qev-ids").mkdir(exist_ok=True)
else:
    raise ValueError(backend)
with (
    (raw / f"{backend}-de-jobs.txt").open("w") as jobs,
    (raw / f"{backend}-de-options.jsonl").open("w") as options,
):
    for line in (raw / "german-inputs.jsonl").read_text().splitlines():
        row = json.loads(line)
        ident, req = row["id"], row["request"]
        (q,) = req["questions"].values()
        if backend == "julia":
            opts = prompt.options_for(q)
            ids = tokenizer.encode(
                prompt.render(req["state"], q["type"], q["instructions"], opts)
            )
            names = [k for k, _ in opts]
            print(ident, *ids, file=jobs)
        else:
            ids, rendered = adapter["encode"](tokenizer, req["state"], q)
            names = rendered["names"]
            dest = raw / "qev-ids" / f"{ident}.txt"
            dest.write_text(",".join(map(str, ids)) + "\n")
            print(ident, dest.resolve(), sep="\t", file=jobs)
        assert set(names) == set(q["criteria"]) and 0 < len(ids) <= 16384
        print(json.dumps(dict(id=ident, names=names, tokens=len(ids))), file=options)
