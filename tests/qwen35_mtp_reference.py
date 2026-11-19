#!/usr/bin/env python3
"""Compare native MTP hidden rows with a PyTorch calculation from the official shards."""

import argparse
import json
from pathlib import Path
import subprocess

import numpy as np
from safetensors import safe_open
import torch
import torch.nn.functional as F


def reference(embeddings, previous, root):
    index = json.loads((root / "model.safetensors.index.json").read_text())["weight_map"]
    weights = {}
    for shard in {shard for name, shard in index.items() if name.startswith("mtp.")}:
        with safe_open(root / shard, framework="pt", device="cpu") as stream:
            for name in stream.keys():
                if name.startswith("mtp."):
                    weights[name.removeprefix("mtp.").removesuffix(".weight")] = stream.get_tensor(name).float()

    def norm(values, name):
        return F.rms_norm(values, (values.shape[-1],), 1 + weights[name], eps=1e-6)

    def project(values, name):
        return F.linear(values, weights[name])

    def rotary(values):
        left, right, tail = values.split((32, 32, 192), dim=-1)
        phase = torch.arange(len(values), dtype=torch.float32)[:, None, None] * (
            10_000_000.0 ** (-torch.arange(32, dtype=torch.float32) / 32)
        )
        return torch.cat((left * phase.cos() - right * phase.sin(),
                          right * phase.cos() + left * phase.sin(), tail), dim=-1)

    block = "layers.0."
    attn = block + "self_attn."
    current = project(torch.cat((norm(embeddings, "pre_fc_norm_embedding"),
                                 norm(previous, "pre_fc_norm_hidden")), dim=-1), "fc")
    attention_input = norm(current, block + "input_layernorm")
    query, gate = project(attention_input, attn + "q_proj").reshape(-1, 16, 512).chunk(2, dim=-1)
    query = rotary(norm(query, attn + "q_norm"))
    key = project(attention_input, attn + "k_proj").reshape(-1, 4, 256)
    key = rotary(norm(key, attn + "k_norm")).half().float().repeat_interleave(4, dim=1)
    value = project(attention_input, attn + "v_proj").reshape(-1, 4, 256)
    value = value.half().float().repeat_interleave(4, dim=1)
    attention = F.scaled_dot_product_attention(query.transpose(0, 1), key.transpose(0, 1),
                                               value.transpose(0, 1), is_causal=True)
    current = current + project((attention.transpose(0, 1) * gate.sigmoid()).flatten(1), attn + "o_proj")
    mlp_input = norm(current, block + "post_attention_layernorm")
    mlp = F.silu(project(mlp_input, block + "mlp.gate_proj")) * project(mlp_input, block + "mlp.up_proj")
    return norm(current + project(mlp, block + "mlp.down_proj"), "norm")


def metrics(expected, observed):
    error = expected - observed
    return dict(rmse=float(error.square().mean().sqrt()), max_abs=float(error.abs().max()),
                min_cosine=float(F.cosine_similarity(expected, observed).min()))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--probe", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(6)
    results = {}
    for variant in ("bf16", "q8", "h128"):
        dump = args.output_dir / (variant + ".f32")
        subprocess.run([args.probe.resolve(), args.model_dir / f"mtp-{variant}.q35h", dump], check=True)
        rows = torch.from_numpy(np.fromfile(dump, dtype="<f4").reshape(22, 2560))
        if variant == "bf16":
            expected = reference(rows[:7], rows[7:14], args.model_dir)
        observed, batched = rows[14:21], rows[21:]
        results[variant] = dict(reference=metrics(expected, observed),
                                batched_prefill=metrics(observed[-1:], batched))
    assert results["bf16"]["reference"]["max_abs"] < 0.005, results
    for variant in results:
        assert results[variant]["batched_prefill"]["min_cosine"] > 0.999, results
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
