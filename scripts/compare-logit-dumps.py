#!/usr/bin/env python3
"""Compare teacher-forced full-vocabulary logit dumps with streaming KLD."""

import argparse
import csv
import json
import math
import pathlib
import struct

import numpy as np


MAGIC = b"Q35LGT1\0"
HEADER = struct.Struct("<8sIIQ")
TARGET = struct.Struct("<i")


class LogitDump:
    def __init__(self, path: pathlib.Path):
        self.path = path
        self.position = 0
        self.stream = path.open("rb")
        header = self.stream.read(HEADER.size)
        if len(header) != HEADER.size:
            raise ValueError(f"Truncated logit header: {path}")
        magic, version, self.vocab, self.positions = HEADER.unpack(header)
        if magic != MAGIC or version != 1:
            raise ValueError(f"Unsupported logit dump format: {path}")
        if self.positions == 0 or self.vocab < 2:
            self.stream.close()
            raise ValueError(f"Empty logit dump or invalid vocabulary: {path}")
        expected = HEADER.size + self.positions * (TARGET.size + self.vocab * 4)
        if path.stat().st_size != expected:
            raise ValueError(
                f"Logit dump size mismatch for {path}: expected {expected}, "
                f"found {path.stat().st_size}")

    def read(self):
        target_raw = self.stream.read(TARGET.size)
        if len(target_raw) != TARGET.size:
            raise ValueError(f"Truncated target record in {self.path}")
        target = TARGET.unpack(target_raw)[0]
        logits = np.fromfile(self.stream, dtype="<f4", count=self.vocab)
        if logits.size != self.vocab:
            raise ValueError(f"Truncated logits record in {self.path}")
        if not np.isfinite(logits).all():
            index = int(np.flatnonzero(~np.isfinite(logits))[0])
            raise ValueError(f"Non-finite raw logit in {self.path} at output position "
                             f"{self.position}, vocabulary index {index}")
        self.position += 1
        return target, logits

    def close(self):
        self.stream.close()


def log_softmax_f64(logits):
    values = logits.astype(np.float64)
    maximum = float(np.max(values))
    return values - (maximum + math.log(float(np.exp(values - maximum).sum())))


def top_indices(logits, count):
    count = min(count, logits.size)
    selected = np.argpartition(logits, -count)[-count:]
    return selected[np.argsort(logits[selected])[::-1]]


def percentile(values, q):
    return float(np.percentile(np.asarray(values, dtype=np.float64), q))


def summarize(rows):
    kld = [row["kld_teacher_candidate"] for row in rows]
    nll_delta = [row["target_nll_delta"] for row in rows]
    return {
        "positions": len(rows),
        "kld": {
            "mean": float(np.mean(kld)),
            "median": percentile(kld, 50),
            "p95": percentile(kld, 95),
            "p99": percentile(kld, 99),
            "max": float(np.max(kld)),
        },
        "target_nll": {
            "teacher_perplexity": math.exp(float(np.mean([row["target_nll_teacher"] for row in rows]))),
            "candidate_perplexity": math.exp(float(np.mean([row["target_nll_candidate"] for row in rows]))),
            "teacher_mean": float(np.mean([row["target_nll_teacher"] for row in rows])),
            "candidate_mean": float(np.mean([row["target_nll_candidate"] for row in rows])),
            "delta_mean": float(np.mean(nll_delta)),
            "delta_p95": percentile(nll_delta, 95),
            "perplexity_ratio": float(math.exp(min(50.0, float(np.mean(nll_delta))))),
        },
        "agreement": {
            "top1": float(np.mean([row["top1_equal"] for row in rows])),
            "teacher_top1_in_candidate_top5": float(np.mean([
                row["teacher_top1_in_candidate_top5"] for row in rows])),
            "teacher_top1_in_candidate_top10": float(np.mean([
                row["teacher_top1_in_candidate_top10"] for row in rows])),
            "top5_overlap": float(np.mean([row["top5_overlap"] for row in rows])),
            "top10_overlap": float(np.mean([row["top10_overlap"] for row in rows])),
        },
        "logits": {
            "centered_rmse_mean": float(np.mean([row["centered_logit_rmse"] for row in rows])),
            "centered_cosine_mean": float(np.mean([row["centered_logit_cosine"] for row in rows])),
        },
        "argmax_flips": int(sum(not row["top1_equal"] for row in rows)),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--teacher", type=pathlib.Path, required=True)
    parser.add_argument("--candidate", type=pathlib.Path, required=True)
    parser.add_argument("--csv", type=pathlib.Path)
    parser.add_argument("--json", type=pathlib.Path)
    args = parser.parse_args()

    teacher = LogitDump(args.teacher)
    candidate = LogitDump(args.candidate)
    try:
        if teacher.vocab != candidate.vocab or teacher.positions != candidate.positions:
            raise ValueError("Teacher and candidate dump dimensions differ")

        rows = []
        for position in range(teacher.positions):
            target_t, logits_t = teacher.read()
            target_c, logits_c = candidate.read()
            if target_t != target_c or target_t < 0 or target_t >= teacher.vocab:
                raise ValueError(f"Target token mismatch at output position {position}")

            logp = log_softmax_f64(logits_t)
            logq = log_softmax_f64(logits_c)
            probabilities = np.exp(logp)
            raw_kld = float(np.sum(probabilities * (logp - logq)))
            if not math.isfinite(raw_kld):
                raise ValueError(f"Non-finite KL at output position {position}: "
                                 f"{args.teacher} versus {args.candidate}")
            kld = max(0.0, raw_kld)

            top_t = top_indices(logits_t, 10)
            top_c = top_indices(logits_c, 10)
            top_t_5 = set(int(value) for value in top_t[:5])
            top_c_5 = set(int(value) for value in top_c[:5])
            top_t_10 = set(int(value) for value in top_t)
            top_c_10 = set(int(value) for value in top_c)
            teacher_top1 = int(top_t[0])
            candidate_top1 = int(top_c[0])

            centered_t = logits_t.astype(np.float64) - float(np.mean(logits_t, dtype=np.float64))
            centered_c = logits_c.astype(np.float64) - float(np.mean(logits_c, dtype=np.float64))
            difference = centered_c - centered_t
            denominator = float(np.linalg.norm(centered_t) * np.linalg.norm(centered_c))
            cosine = float(np.dot(centered_t, centered_c) / denominator) if denominator else 1.0

            rows.append({
                "position": position,
                "target_token": target_t,
                "kld_teacher_candidate": kld,
                "target_nll_teacher": float(-logp[target_t]),
                "target_nll_candidate": float(-logq[target_t]),
                "target_nll_delta": float(logp[target_t] - logq[target_t]),
                "teacher_top1": teacher_top1,
                "candidate_top1": candidate_top1,
                "top1_equal": teacher_top1 == candidate_top1,
                "teacher_margin": float(logits_t[top_t[0]] - logits_t[top_t[1]]),
                "teacher_top1_in_candidate_top5": teacher_top1 in top_c_5,
                "teacher_top1_in_candidate_top10": teacher_top1 in top_c_10,
                "top5_overlap": len(top_t_5 & top_c_5) / 5.0,
                "top10_overlap": len(top_t_10 & top_c_10) / 10.0,
                "centered_logit_rmse": float(np.sqrt(np.mean(difference * difference))),
                "centered_logit_cosine": cosine,
            })

        summary = summarize(rows)
        summary["teacher"] = str(args.teacher)
        summary["candidate"] = str(args.candidate)
        summary["vocabulary_size"] = teacher.vocab

        if args.csv:
            args.csv.parent.mkdir(parents=True, exist_ok=True)
            with args.csv.open("w", newline="", encoding="utf-8") as stream:
                writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
        if args.json:
            args.json.parent.mkdir(parents=True, exist_ok=True)
            args.json.write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        print(json.dumps(summary, indent=2, allow_nan=False))
    finally:
        teacher.close()
        candidate.close()


if __name__ == "__main__":
    main()
