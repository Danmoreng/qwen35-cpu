"""Freeze German requests and their key before scoring either checkpoint."""

import json
import random
from pathlib import Path

evidence = Path(__file__).parent
prior = Path("docs/results/decision-followup-2026-11-20")
raw = Path("benchmarks/julia2-transfer")
rng = random.Random(int((evidence / "seed.txt").read_text()))


def rows(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


requests = rows(prior / "routing-de-inputs.jsonl")
key = {
    r["id"]: dict(
        panel="routing",
        group=r["domain"],
        expected=r["expected"],
        categories=r["options"],
    )
    for r in rows(prior / "routing-de-gold.jsonl")
}

for pair in range(64):
    team = [f"Team {n}" for n in rng.sample(range(101, 999), 3)]
    ticket = rng.randrange(1000000, 9999999)
    threshold = rng.randint(20, 180)
    codes = rng.sample(range(10, 99), 2)
    if pair < 32:
        family = "deadline"
        rules = [
            f"Unter {threshold} Minuten Wartezeit übernimmt {team[0]}.",
            f"Ab {threshold} Minuten Wartezeit übernimmt {team[1]}.",
            f"Bei unbekannter Wartezeit übernimmt {team[2]}.",
        ]
        delta = rng.randint(1, 8)
        details = [
            f"Wartezeit laut Ticket: {n} Minuten."
            for n in (threshold - delta, threshold + delta)
        ]
    else:
        family = "code"
        rules = [f"Für Filialcode {c} ist {t} zuständig." for c, t in zip(codes, team)]
        rules.append(f"Alle übrigen Filialcodes gehen an {team[2]}.")
        details = [f"Filialcode laut Ticket: {c}." for c in codes]
    rng.shuffle(rules)
    order = list(team)
    rng.shuffle(order)
    criteria = dict(zip("ABC", order))
    group = f"{family}-{pair:02}"
    for variant, detail in enumerate(details):
        ident = f"{rng.getrandbits(64):016x}"
        state = (
            f"Interne Zuständigkeiten für Vorgang {ticket}:\n"
            + "\n".join(rules)
            + "\n"
            + detail
        )
        request = dict(
            state=state,
            questions={
                "team": dict(
                    type="choice",
                    instructions="Welches Team soll dieses Ticket gemäß der internen Regel übernehmen?",
                    criteria=criteria,
                )
            },
        )
        requests.append(dict(id=ident, request=request))
        key[ident] = dict(
            panel=family, group=group, expected=team[variant], categories=criteria
        )
rng.shuffle(requests)
assert len(requests) == len(key) == len({r["id"] for r in requests}) == 192
(raw / "german-inputs.jsonl").write_text(
    "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in requests)
)
(raw / "german-key.json").write_text(
    json.dumps(key, ensure_ascii=False, indent=2) + "\n"
)
