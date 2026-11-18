#!/usr/bin/env python3
"""Compare a native server with the original HF fine-tune on labeled task examples."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import statistics
import urllib.request


def fetch(base, route, body):
    req = urllib.request.Request(base.rstrip("/") + route, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=180) as response:
        return json.load(response)


def error_counts(expected, actual):
    # Levenshtein with a deterministic substitution/deletion/insertion tie break.
    ref, hyp = expected.split(), actual.split()
    prev = [(j, 0, j, 0) for j in range(len(hyp) + 1)]
    for i, word in enumerate(ref, 1):
        row = [(i, i, 0, 0)]
        for j, got in enumerate(hyp, 1):
            sub, delete, insert = prev[j-1], prev[j], row[-1]
            miss = int(word != got)
            row.append(min([(sub[0]+miss, sub[1], sub[2], sub[3]+miss),
                            (delete[0]+1, delete[1]+1, delete[2], delete[3]),
                            (insert[0]+1, insert[1], insert[2]+1, insert[3])], key=lambda x: x[0]))
        prev = row
    return prev[-1]


def metrics(panel, rows, key):
    cases = panel["cases"]
    if panel["task"] == "classification":
        labels = panel["labels"]
        pairs = [(case["expected"], max(row[key], key=row[key].get)) for case, row in zip(cases, rows)]
        f1 = []
        for label in labels:
            positive = sum(y == label for y, _ in pairs)
            predicted = sum(y == label for _, y in pairs)
            tp = sum(y == p == label for y, p in pairs)
            f1.append(2 * tp / (positive + predicted) if positive + predicted else 0)
        out = dict(accuracy=statistics.mean(y == p for y, p in pairs), macro_f1=statistics.mean(f1))
        if "positive" in panel:
            groups = defaultdict(list)
            for case, row in zip(cases, rows):
                groups[case["group"]].append((row[key][panel["positive"]], case["expected"] == panel["positive"]))
            ranks = []
            for group in groups.values():
                ranks.append(next((1/i for i, (_, pos) in enumerate(sorted(group, key=lambda x: -x[0]), 1) if pos), 0))
            out["mrr"] = statistics.mean(ranks)
        return out
    if panel["task"] == "asr":
        counts = [error_counts(c["expected"], r[key]) for c, r in zip(cases, rows)]
        words = sum(len(c["expected"].split()) for c in cases)
        errors = sum(v[0] for v in counts)
        return dict(wer=errors / words if words else errors, insertions=sum(v[2] for v in counts),
                    deletions=sum(v[1] for v in counts), substitutions=sum(v[3] for v in counts))
    correct = fields = valid = 0
    for case, row in zip(cases, rows):
        try:
            obj = json.loads(row[key])
        except ValueError:
            obj = None
        valid += isinstance(obj, dict)
        for field, value in case["expected"].items():
            fields += 1
            correct += isinstance(obj, dict) and field in obj and obj[field] == value
    if not fields:
        raise ValueError("Extraction panel contains no expected fields")
    return dict(field_accuracy=correct / fields, valid_json_object=valid / len(rows))


def main():
    import torch
    from transformers import AutoModelForCausalLM, AutoModelForImageTextToText, AutoTokenizer
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--reference", type=Path, required=True, help="Original local HF checkpoint")
    p.add_argument("--panel", type=Path, required=True)
    p.add_argument("--server", default="http://127.0.0.1:8080")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--threads", type=int, default=3)
    args = p.parse_args()
    panel = json.loads(args.panel.read_text())
    task = panel["task"]
    if task not in ("classification", "asr", "extraction") or not panel["cases"]:
        p.error("Expected a nonempty classification, asr or extraction panel")
    torch.set_num_threads(args.threads)
    config = json.loads((args.reference / "config.json").read_text())
    auto = AutoModelForImageTextToText if "vision_config" in config else AutoModelForCausalLM
    model, info = auto.from_pretrained(args.reference, local_files_only=True, use_safetensors=True,
                                       dtype="auto", output_loading_info=True)
    if any(info.get(k) for k in ("missing_keys", "mismatched_keys", "unexpected_keys", "error_msgs")):
        raise ValueError(f"Checkpoint loading discrepancy: {info}")
    model.eval()
    tok = AutoTokenizer.from_pretrained(args.reference, local_files_only=True)
    if task == "classification":
        tokens = [tok.encode(t, add_special_tokens=False) for t in panel["labels"].values()]
        if any(len(t) != 1 for t in tokens) or len(set(t[0] for t in tokens)) != len(tokens):
            raise ValueError("Class tokens must be distinct single tokens")
    rows = []
    with torch.inference_mode():
        for case in panel["cases"]:
            prompt = panel["prompt_template"].format(**case["fields"])
            ids = tok(prompt, add_special_tokens=False, return_tensors="pt").input_ids
            if task == "classification":
                if "qev_question" in panel:
                    response = fetch(args.server, "/v1/systemone", dict(state=case["fields"]["state"],
                                     questions={"dept": panel["qev_question"]}))
                    native = response["answers"]["dept"]["probabilities"]
                else:
                    response = fetch(args.server, "/v1/classify", {"prompt": prompt})
                    native = response["probabilities"]
                if set(native) != set(panel["labels"]):
                    raise ValueError("Profile labels differ from evaluation panel")
                logits = model(input_ids=ids, logits_to_keep=1, use_cache=False).logits[0, -1].float()
                probs = logits[[t[0] for t in tokens]].double().softmax(0)
                if "qev_question" in panel:
                    probs = probs.clamp_min(1e-9).pow(1 / panel["temperature"])
                    probs /= probs.sum()
                reference = dict(zip(panel["labels"], probs.tolist()))
                count = response["usage"]["input_tokens"]
            else:
                limit = panel.get("max_new_tokens", 64)
                response = fetch(args.server, "/v1/completions",
                                 dict(prompt=prompt, max_tokens=limit, temperature=0))
                native = response["choices"][0]["text"]
                result = model.generate(ids, max_new_tokens=limit, do_sample=False, repetition_penalty=1.0)
                reference = tok.decode(result[0, ids.shape[1]:], skip_special_tokens=True)
                count = response["usage"]["prompt_tokens"]
            if count != ids.shape[1]:
                raise ValueError(f"Prompt token counts differ: native {count}, reference {ids.shape[1]}")
            rows.append(dict(reference=reference, native=native, tokens=count))
    summary = {k: metrics(panel, rows, k) for k in ("reference", "native")}
    if task == "classification":
        deltas = [abs(r["native"][k] - r["reference"][k]) for r in rows for k in r["native"]]
        summary.update(top1_agreement=statistics.mean(max(r["native"], key=r["native"].get) ==
                       max(r["reference"], key=r["reference"].get) for r in rows),
                       probability_mae=statistics.mean(deltas), max_probability_error=max(deltas))
    args.output.write_text(json.dumps(dict(summary=summary, dtype=str(model.dtype), rows=rows),
                                      indent=2, allow_nan=False) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
