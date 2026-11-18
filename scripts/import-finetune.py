#!/usr/bin/env python3
"""Build a text-only H128/Q4-G32-DOT4 package from a Qwen3.5-0.8B fine-tune."""
import argparse
import hashlib
import json
import struct
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

from finetune_server import running_server

ASSETS = ("config.json", "tokenizer.json", "tokenizer_config.json", "vocab.json",
          "merges.txt", "special_tokens_map.json", "chat_template.jinja",
          "generation_config.json", "qev_config.json", "LICENSE", "NOTICE")


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def snapshot(source, revision):
    if Path(source).is_dir():
        return Path(source).resolve(), None
    from huggingface_hub import model_info, snapshot_download
    sha = model_info(source, revision=revision).sha
    root = snapshot_download(source, revision=sha, allow_patterns=[
        *ASSETS, "README.md", "LICENSE*", "adapter_config.json", "*.safetensors", "model.safetensors.index.json"])
    return Path(root), sha


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def write(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def validate_head(root):
    """Some exports include a separate lm_head even when config declares a tie."""
    entries = {}
    for path in root.glob("*.safetensors"):
        with path.open("rb") as stream:
            size = struct.unpack("<Q", stream.read(8))[0]
            if size > 100_000_000:
                raise ValueError("Invalid safetensors header length")
            header = json.loads(stream.read(size))
        for name in ("lm_head.weight", "model.embed_tokens.weight", "model.language_model.embed_tokens.weight"):
            if name in header:
                tensor = header[name]
                start, stop = tensor["data_offsets"]
                entries[name] = path, 8 + size + start, stop - start, tensor["dtype"], tensor["shape"]
    embeds = [entry for name, entry in entries.items() if name.endswith("embed_tokens.weight")]
    if len(embeds) != 1:
        raise ValueError("Expected one text embedding tensor")
    if "lm_head.weight" in entries:
        head, embed = entries["lm_head.weight"], embeds[0]
        if head[2:] != embed[2:]:
            raise ValueError("Untied output heads are unsupported")
        with head[0].open("rb") as a, embed[0].open("rb") as b:
            a.seek(head[1])
            b.seek(embed[1])
            for offset in range(0, head[2], 1024 * 1024):
                n = min(1024 * 1024, head[2] - offset)
                x, y = a.read(n), b.read(n)
                if len(x) != n or x != y:
                    raise ValueError("Output head differs from the embedding")


def merge_adapter(base, adapter, dest):
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoModelForImageTextToText
    config = read(base / "config.json")
    auto = AutoModelForImageTextToText if "vision_config" in config else AutoModelForCausalLM
    model, info = auto.from_pretrained(base, dtype=torch.float32, local_files_only=True,
                                      use_safetensors=True, output_loading_info=True)
    if any(info.get(k) for k in ("missing_keys", "mismatched_keys", "unexpected_keys", "error_msgs")):
        raise ValueError(f"Incomplete base checkpoint: {info}")
    model = PeftModel.from_pretrained(model, adapter, local_files_only=True)
    for name, layer in model.named_modules():
        if name.rsplit(".", 1)[-1] in ("embed_tokens", "lm_head") and hasattr(layer, "base_layer"):
            raise ValueError("LoRA updates to tied embeddings or lm_head need independent sequence-level verification")
    merged = model.merge_and_unload(safe_merge=True)
    if not torch.equal(merged.get_input_embeddings().weight, merged.get_output_embeddings().weight):
        raise ValueError("The merged output head differs from the embedding; this engine requires tied weights")
    merged.save_pretrained(dest, max_shard_size="2GB")


def tokenizer_assets(package):
    # Recent Transformers exports only tokenizer.json. The native loader also
    # needs the legacy BPE files and the added-token decoder (including ChatML).
    tok = read(package / "tokenizer.json")
    model = tok["model"]
    if model["type"] != "BPE":
        raise ValueError("Expected a Qwen BPE tokenizer")
    all_ids = list(model["vocab"].values()) + [t["id"] for t in tok["added_tokens"]]
    if min(all_ids) < 0 or max(all_ids) >= 248320:
        raise ValueError("Tokenizer IDs exceed the supported embedding")
    write(package / "vocab.json", model["vocab"])
    with (package / "merges.txt").open("w", encoding="utf-8") as out:
        out.write("#version: 0.2\n")
        for pair in model["merges"]:
            out.write((pair if isinstance(pair, str) else " ".join(pair)) + "\n")
    config = read(package / "tokenizer_config.json")
    config["added_tokens_decoder"] = {str(t["id"]): {k: v for k, v in t.items() if k != "id"}
                                      for t in tok["added_tokens"]}
    write(package / "tokenizer_config.json", config)


def evaluate(package, checkpoint, panel, server, threads):
    with running_server(package, server, threads) as url:
        subprocess.run([sys.executable, str(Path(__file__).with_name("evaluate-finetune.py")),
                        "--reference", str(checkpoint), "--panel", str(panel), "--server", url,
                        "--threads", str(threads), "--output", str(package / "quality.json")], check=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("source", help="Local directory or Hugging Face repository")
    p.add_argument("--revision", default="main")
    p.add_argument("--base", help="Override the adapter's base directory/repository")
    p.add_argument("--base-revision")
    p.add_argument("--packer", required=True)
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--importance-dir", type=Path)
    p.add_argument("--covariance-dir", type=Path)
    p.add_argument("--classifier-profile", type=Path)
    p.add_argument("--eval-panel", type=Path, help="Run a labeled quality comparison before publishing the directory")
    p.add_argument("--server-binary", help="qwen35_cpu_server used with --eval-panel")
    p.add_argument("--threads", type=int, default=4, help="CPU threads for quality evaluation")
    args = p.parse_args()
    if bool(args.eval_panel) != bool(args.server_binary) or args.threads < 1:
        p.error("--eval-panel and --server-binary must be supplied together, with positive --threads")
    if args.output.exists():
        p.error("Output already exists")
    source, revision = snapshot(args.source, args.revision)
    base, base_revision = source, None
    adapter = (source / "adapter_config.json").is_file()
    base_name = None
    if adapter:
        if not (source / "adapter_model.safetensors").is_file():
            p.error("Expected adapter_model.safetensors")
        cfg = read(source / "adapter_config.json")
        if cfg["peft_type"] != "LORA":
            p.error("Expected a PEFT LoRA adapter")
        base_name = args.base or cfg["base_model_name_or_path"]
        base, base_revision = snapshot(base_name,
                                       args.base_revision or cfg.get("revision") or "main")
    elif args.base or args.base_revision:
        p.error("--base and --base-revision apply to adapters")
    cfg = read(base / "config.json")
    tc = cfg.get("text_config", cfg)
    if (tc.get("model_type") != "qwen3_5_text" or
            (tc["hidden_size"], tc["num_hidden_layers"], tc["intermediate_size"], tc["vocab_size"])
            != (1024, 24, 3584, 248320) or not tc.get("tie_word_embeddings", cfg.get("tie_word_embeddings"))):
        p.error("Expected a tied-head Qwen3.5-0.8B text backbone")
    validate_head(base)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=args.output.parent) as tmp:
        work = Path(tmp)
        checkpoint = base
        if adapter:
            checkpoint = work / "merged"
            merge_adapter(base, source, checkpoint)
        package = work / "package"
        package.mkdir()
        for name in ASSETS:
            origin = source / name if (source / name).is_file() else base / name
            if origin.is_file():
                shutil.copyfile(origin, package / name)
        for group, root in ([("BASE", base), ("ADAPTER", source)] if adapter else [("SOURCE", source)]):
            for file in [root / "README.md", *root.glob("LICENSE*"), root / "NOTICE"]:
                if file.is_file():
                    shutil.copyfile(file, package / f"{group}-{file.name}")
        tokenizer_assets(package)
        if adapter:
            for name in ASSETS:
                if name != "config.json" and (package / name).is_file():
                    shutil.copyfile(package / name, checkpoint / name)
        if args.classifier_profile:
            shutil.copyfile(args.classifier_profile, package / "classifier.json")
        command = [args.packer, "--hf-model-dir", str(checkpoint),
                   "--output", str(package / "model.q35h"), "--quantizer", "mse16"]
        for flag, directory in [("--importance-dir", args.importance_dir),
                                ("--covariance-dir", args.covariance_dir)]:
            if directory:
                command.extend([flag, str(directory)])
        subprocess.run(command, check=True)
        if args.eval_panel:
            evaluate(package, checkpoint, args.eval_panel, args.server_binary, args.threads)
        manifest = dict(source=args.source, revision=revision, adapter=adapter,
                        base=base_name,
                        base_revision=base_revision, panel_sha256=digest(args.eval_panel) if args.eval_panel else None,
                        format="H128/Q4-G32-DOT4",
                        quantizer="mse16", merge_dtype="float32" if adapter else None,
                        packer_sha256=digest(Path(shutil.which(args.packer) or args.packer)),
                        sources={})
        for group, origin in [("source", source), ("base", base)] if adapter else [("source", source)]:
            manifest["sources"][group] = {f.name: digest(f) for f in sorted(origin.iterdir())
                                          if f.is_file() and f.suffix in (".json", ".safetensors", ".jinja")}
        manifest["files"] = {f.name: digest(f) for f in sorted(package.iterdir())}
        write(package / "import.json", manifest)
        package.rename(args.output)


if __name__ == "__main__":
    main()
