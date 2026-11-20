"""Regenerate the two qev chunk-boundary fixtures from the frozen HTTP requests."""
from pathlib import Path
import json
import random
import sys

source = Path("docs/results/qev-api-2026-10-21")
out = Path(sys.argv[1])
out.mkdir(parents=True, exist_ok=True)
short = json.loads((source / "short-request.json").read_text())
long = json.loads((source / "long-request.json").read_text())
rows = [
    {"id": str(n), "request": {**short, "state": short["state"] + " x" * (n - 74)}}
    for n in (74, 127, 128, 129)
]
rows.append({"id": "579", "request": long})
with (out / "boundaries.jsonl").open("w") as boundary:
    for row in rows:
        boundary.write(json.dumps(row) + "\n")

rng = random.Random(9955)
with (out / "confirm-lengths.jsonl").open("w") as panel:
    for repeat in range(20):
        order = rows.copy()
        rng.shuffle(order)
        for row in order:
            panel.write(json.dumps({**row, "id": f"{row['id']}-{repeat}"}) + "\n")

tickets = json.loads(Path("tests/data/finetunes/qev.json").read_text())
with (out / "short-routing.jsonl").open("w") as panel:
    for repeat in range(4):
        for i, case in enumerate(tickets["cases"][:16]):
            request = {"state": case["fields"]["state"], "questions": {"dept": tickets["qev_question"]}}
            panel.write(json.dumps({"id": f"ticket-{i}-{repeat}", "request": request}) + "\n")
