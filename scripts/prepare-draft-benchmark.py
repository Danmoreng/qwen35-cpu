"""Prepare the text-drafting prompts and sequential CPU benchmark matrix."""
import argparse
import json
import subprocess
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--cli", type=Path, required=True)
parser.add_argument("--bench", type=Path, required=True)
parser.add_argument("--model-dir", type=Path, required=True)
parser.add_argument("--out", type=Path, required=True)
parser.add_argument("--threads", type=int, default=3)
args = parser.parse_args()
out = args.out.resolve()
out.mkdir(parents=True, exist_ok=False)
model = args.model_dir.resolve()
fixtures = Path(__file__).resolve().parents[1] / "docs/results/text-drafting-2026-11-17/prompts.json"


def tokenize(name, text):
    prompt = out / (name + ".txt")
    ids = out / (name + ".ids")
    prompt.write_text(text, encoding="utf-8")
    subprocess.run([args.cli.resolve(), "--model-dir", model, "--weights", model / "model.q35h",
                    "--prompt-file", prompt, "--tokenize-out", ids], check=True)
    csv = out / (name + ".csv")
    csv.write_text(",".join(ids.read_text().split()), encoding="ascii")
    return str(csv)


matrix = []
for name, job in json.loads(fixtures.read_text(encoding="utf-8")).items():
    source = job.get("source", "")
    user = job["instruction"] + ("\n\n" + source if source else "")
    prompt = tokenize(name, "<|im_start|>user\n" + user + "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n")
    source_ids = tokenize(name + "-source", source) if source else None
    for sampling in (["greedy", "sampled"] if name in ("editing", "prose") else ["greedy"]):
        greedy = sampling == "greedy"
        for mode in ["off", "ngram", "input"]:
            flags = ["--hf-model-dir", str(model), "--cpu-q4-h128", str(model / "model.q35h"),
                     "--cpu-threads", str(args.threads), "--cpu-batch", "1", "--max-context", "2048",
                     "--prompt-tokens-file", prompt, "--max-new-tokens", "128",
                     "--stop-token-ids", "248044,248046", "--draft", mode,
                     "--temperature", "0" if greedy else "0.65", "--top-p", "0.9", "--top-k", "40",
                     "--seed", "83", "--repeat-penalty", "1" if greedy else "1.05"]
            if mode == "input" and source_ids:
                flags.extend(["--draft-source-tokens-file", source_ids])
            matrix.append({"name": f"{name}-{sampling}-{mode}",
                           "executable": str(args.bench.resolve()), "args": flags})
(out / "matrix.json").write_text(json.dumps(matrix, indent=2), encoding="utf-8")
