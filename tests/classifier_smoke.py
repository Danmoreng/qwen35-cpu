#!/usr/bin/env python3
"""Check real full/selected readouts, prefix reuse and invalid classifier requests."""
import argparse
import json
from pathlib import Path
import sys
import urllib.error
import urllib.request

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from finetune_server import running_server


def classify(url, payload):
    request = urllib.request.Request(url + "/v1/classify", data=json.dumps(payload).encode(),
                                     headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=90) as response:
        return json.load(response)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--server", required=True)
    p.add_argument("--model-dir", type=Path, required=True)
    p.add_argument("--panel", type=Path, required=True)
    p.add_argument("--output", type=Path)
    args = p.parse_args()
    panel = json.loads(args.panel.read_text())
    prompts = [panel["prompt_template"].format(**c["fields"]) for c in panel["cases"]]
    results = {}
    for head in ("auto", "full"):
        with running_server(args.model_dir, args.server, 4, ["--decision-head", head, "--prefix-cache", "auto"]) as url:
            try:
                classify(url, {"prompt": ""})
                raise AssertionError("Empty input accepted")
            except urllib.error.HTTPError as error:
                assert error.code == 400
            rows = [classify(url, {"prompt": prompt}) for prompt in prompts]
            hit = classify(url, {"prompt": prompts[-1]})
            assert hit["cache"]["read_tokens"] == rows[-1]["usage"]["input_tokens"]
            assert hit["probabilities"] == rows[-1]["probabilities"]
            assert all(row["usage"]["output_tokens"] == 0 and
                       abs(sum(row["probabilities"].values())-1) < 1e-9 for row in rows)
            results[head] = rows
    diffs = [abs(a["probabilities"][k]-b["probabilities"][k])
             for a, b in zip(results["auto"], results["full"]) for k in a["probabilities"]]
    assert max(diffs) < 1e-5
    evidence = dict(cases=len(prompts), max_probability_difference=max(diffs),
                    top1_matches=sum(a["label"] == b["label"] for a, b in zip(results["auto"], results["full"])))
    print(evidence)
    if args.output:
        args.output.write_text(json.dumps(evidence, indent=2) + "\n")


if __name__ == "__main__":
    main()
