#!/usr/bin/env python3
"""Normalize Pi benchmark profiles and score each saved app in an offline browser."""

import argparse
import csv
import json
from pathlib import Path
import runpy
import statistics
import subprocess


def load(path):
    return json.loads(path.read_text(encoding="utf-8"))


def requests(profile, case, run):
    previous_draft = None
    for request in profile["requests"]:
        usage = request.get("usage", {})
        timings = request.get("timings", {})
        row = {key: request.get(key) for key in
               ("turn", "request", "status", "finish", "wall_ms", "first_output_ms", "last_output_ms")}
        row.update(case=case, run=run, prompt=usage.get("prompt_tokens"),
                   cached=usage.get("prompt_tokens_details", {}).get("cached_tokens"),
                   output=usage.get("completion_tokens"),
                   prefill=timings.get("prompt_n"), prefill_ms=timings.get("prompt_ms"),
                   decode_ms=timings.get("predicted_ms"), restore_ms=timings.get("prefix_restore_ms"),
                   proposed=timings.get("draft_n", 0), accepted=timings.get("draft_n_accepted", 0))
        draft = request.get("draft_total")
        if draft:
            row["proposed"] = draft["proposed_tokens"] - (previous_draft or {}).get("proposed_tokens", 0)
            row["accepted"] = draft["accepted_tokens"] - (previous_draft or {}).get("accepted_tokens", 0)
        previous_draft = draft
        yield row


def browser_score(profile_path, fixture, playwright, chrome):
    runner = runpy.run_path(str(Path(__file__).with_name("benchmark-pi-agent.py")))
    script = Path(__file__).with_name("score-pi-beacon-switch.mjs").resolve()
    target = profile_path.with_suffix(".raw")
    active_checks = []
    scores = []
    for stage, turn in enumerate(fixture["turns"], 1):
        active_checks.extend(turn["checks"])
        snapshot = (target / f"turn-{stage}.html").resolve()
        if not snapshot.is_file():
            results = {name: {"pass": False, "reason": "index.html missing"} for name in active_checks}
        else:
            command = runner["sandbox"]() + ["--ro-bind", str(playwright.resolve()), "/playwright-core",
                "--ro-bind", str(chrome.resolve().parent), "/chromium",
                "--ro-bind", str(script), "/judge.mjs", "--ro-bind", str(snapshot), "/game.html",
                "/usr/bin/node", "/judge.mjs", "/playwright-core/index.mjs",
                f"/chromium/{chrome.name}", "/game.html", str(stage)]
            results = json.loads(subprocess.run(command, check=True, text=True,
                                                 capture_output=True, timeout=45).stdout)
            if set(results) != set(active_checks):
                raise ValueError(f"Fixture and browser checks differ at stage {stage}")
        scores.append({"turn": stage, "passed": sum(r["pass"] for r in results.values()),
                       "total": len(results), "checks": results})
    return scores


def aggregate(case, run, profile, rows, scores):
    turns = profile["turns"]
    keys = ("prompt", "cached", "output", "prefill", "prefill_ms", "decode_ms",
            "wall_ms", "proposed", "accepted")
    summary = {key: sum(row[key] for row in rows if row[key] is not None) for key in keys}
    summary.update(case=case, run=run, requests=len(rows), failed_requests=profile["failed_requests"],
                   stopped_turns=sum(t["last_stop"] == "stop" for t in turns),
                   agent_ms=sum(t["wall_ms"] for t in turns),
                   tools=sum(len(t["tools"]) for t in turns),
                   tool_errors=sum(tool["error"] for t in turns for tool in t["tools"]),
                   longest_prompt=max(row["prompt"] or 0 for row in rows),
                   longest_output=max(row["output"] or 0 for row in rows))
    if scores:
        summary["checks"] = scores[-1]["passed"]
        summary["max_checks"] = scores[-1]["total"]
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--fixture", type=Path)
    parser.add_argument("--playwright-core", type=Path)
    parser.add_argument("--chromium", type=Path)
    args = parser.parse_args()
    metadata = load(args.run / "metadata.json")
    fixture = load(args.fixture) if args.fixture else None
    if fixture and not (args.playwright_core and args.chromium):
        parser.error("Browser scoring requires --playwright-core and --chromium")
    args.output.mkdir(parents=True, exist_ok=True)
    all_rows, summaries, quality = [], [], []
    for index, case in enumerate(metadata["matrix"]):
        for run in range(metadata["warmup_runs"], metadata["warmup_runs"] + metadata["runs"]):
            profile_path = args.run / f"case-{index}-run-{run}.json"
            profile = load(profile_path)
            rows = list(requests(profile, case["name"], run))
            all_rows.extend(rows)
            scores = browser_score(profile_path, fixture, args.playwright_core, args.chromium) if fixture else []
            if scores:
                quality.append({"case": case["name"], "run": run, "turns": scores})
            summaries.append(aggregate(case["name"], run, profile, rows, scores))
    for name, rows in (("requests", all_rows), ("runs", summaries)):
        with (args.output / f"{name}.csv").open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    if quality:
        (args.output / "quality.json").write_text(json.dumps(quality, indent=2) + "\n", encoding="utf-8")
    for case in metadata["matrix"]:
        rows = [row for row in summaries if row["case"] == case["name"]]
        medians = {name: statistics.median(row[name] for row in rows)
                   for name in rows[0] if name not in ("case", "run")}
        print(case["name"], json.dumps(medians))


if __name__ == "__main__":
    main()
