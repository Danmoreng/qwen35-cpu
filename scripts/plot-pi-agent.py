#!/usr/bin/env python3
"""Plot token-weighted phase rates from measured Pi Coding Agent requests."""

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator, PercentFormatter


BANDS = (0, 1024, 2048, 4096, 8192, 12288, 16384)
LABELS = ("<1K", "1–2K", "2–4K", "4–8K", "8–12K", "12–16K")
SERIES = {
    "native-cached": ("Native", "#007873", "s", "--"),
    "native-mtp3-cached": ("Native MTP 3", "#007873", "o", "-"),
    "ik-cached": ("ik_llama.cpp", "#ab510a", "s", "--"),
    "ik-mtp4-cached": ("ik_llama.cpp MTP 4", "#ab510a", "o", "-"),
}


def total(rows, field):
    return sum(float(row[field]) for row in rows if row[field])


def ratio(rows, numerator, denominator, multiplier):
    valid = [row for row in rows if row[numerator] and row[denominator]
             and float(row[denominator]) > 0]
    return total(valid, numerator) / total(valid, denominator) * multiplier if valid else float("nan")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("requests", type=Path, help="Measured requests.csv from summarize-pi-agent.py")
    parser.add_argument("output", type=Path, help="PNG, SVG or PDF")
    parser.add_argument("--preliminary", action="store_true")
    args = parser.parse_args()
    with args.requests.open(encoding="utf-8", newline="") as stream:
        rows = [row for row in csv.DictReader(stream)
                if row["status"] == "200" and row["prompt"] and row["case"] in SERIES]
    if set(row["case"] for row in rows) != set(SERIES):
        parser.error("Expected measured requests from all four configurations")
    for row in rows:
        # Prefill selects the first output token. One-token replies have no
        # decode work, even when a backend reports a timer-resolution floor.
        row["decode_tokens"] = str(int(row["output"]) - 1) if int(row["output"] or 0) > 1 else ""

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11, "text.color": "#172a35",
                         "axes.labelcolor": "#314753", "xtick.color": "#465d68", "ytick.color": "#465d68"})
    figure, axes = plt.subplots(3, 1, sharex=True, figsize=(9.8, 8.8), dpi=170)
    figure.subplots_adjust(left=0.114, right=0.964, bottom=0.132, top=0.808, hspace=0.61)
    figure.text(0.075, 0.951, "Qwen3.5-4B · Pi Coding Agent", fontsize=23, weight="bold")
    figure.text(0.077, 0.906, "Ryzen 5 7600 · six physical cores · one request at a time", color="#465d68")
    x = list(range(len(LABELS)))
    for offset, (case, (name, color, marker, dash)) in zip((-0.09, -0.03, 0.03, 0.09), SERIES.items()):
        relevant = [row for row in rows if row["case"] == case]
        groups = [[row for row in relevant if low <= int(row["prompt"]) < high]
                  for low, high in zip(BANDS, BANDS[1:])]
        metrics = [
            [ratio(group, "prefill", "prefill_ms", 1000) for group in groups],
            [ratio(group, "decode_tokens", "decode_ms", 1000) for group in groups],
            [ratio(group, "cached", "prompt", 100) for group in groups],
        ]
        for axis, values in zip(axes, metrics):
            axis.plot([position + offset for position in x], values, color=color, label=name,
                      marker=marker, linestyle=dash,
                      markersize=6.1, linewidth=1.8,
                      markerfacecolor=color if dash == "-" else "white", markeredgewidth=1.5)

    axes[0].set_title("Prefill · newly processed tokens", loc="left", weight="bold", pad=8)
    axes[1].set_title("Decode · output after the first token", loc="left", weight="bold", pad=8)
    axes[2].set_title("Prefix cache · share of logical input tokens", loc="left", weight="bold", pad=8)
    for axis in axes:
        axis.set_axisbelow(True)
        axis.grid(axis="y", color="#dde6e9", linewidth=0.8)
        axis.spines[["top", "right", "left"]].set_visible(False)
        axis.spines["bottom"].set_color("#d9e3e7")
        axis.tick_params(length=0, pad=5)
        axis.set_xlim(-0.2, len(x) - 0.80)
        axis.yaxis.set_major_locator(MaxNLocator(nbins=4))
    for axis in axes[:2]:
        axis.set_ylabel("tokens/s")
        axis.set_ylim(bottom=0)
    axes[2].set_ylim(0, 102)
    axes[2].set_yticks([0, 25, 50, 75, 100])
    axes[2].yaxis.set_major_formatter(PercentFormatter())
    axes[2].set_xticks(x, LABELS)
    axes[2].set_xlabel("Logical prompt length, tokens", labelpad=9)
    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(handles, labels, ncols=4, frameon=False, loc="upper left", bbox_to_anchor=(0.069, 0.884))
    repetitions = len({row["run"] for row in rows})
    summary = f"{len(rows)} requests · {repetitions} measured pass{'es' if repetitions != 1 else ''} per mode"
    if args.preliminary:
        summary = f"Preliminary · {summary}"
    figure.text(0.078, 0.055, summary, color="#465d68", fontsize=9.4)
    figure.text(0.078, 0.031, "Rates use summed tokens and backend phase times within each context band.",
                color="#465d68", fontsize=9.4)
    figure.savefig(args.output, facecolor="white")


if __name__ == "__main__":
    main()
