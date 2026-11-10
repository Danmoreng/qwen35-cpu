#!/usr/bin/env python3
"""Render the English JevBench accuracy/latency comparison for sharing.

Requires matplotlib. All plotted numbers come from the supplied aggregate JSON;
the chart distinguishes local CPU measurements from published H100 results.
"""
import argparse
from datetime import date
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import FancyBboxPatch
from matplotlib.ticker import MultipleLocator


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True, help="Output filename stem")
    a = p.parse_args()
    data = json.loads(a.data.read_text())
    measured_date = date.fromisoformat(data["measured"]).strftime("%d %b %Y")
    rows = data["models"]
    assert len(rows) == 7 and all(r["tasks"] == 231 for r in rows)
    assert all(abs(r["accuracy"] - r["correct"] / r["tasks"]) < 1e-10 for r in rows)
    by = {r["id"]: r for r in rows}
    q4 = by["qev-q4"]
    bg, ink, muted, grid = "#F8FAFC", "#13233B", "#53647B", "#DFE6EE"
    teal, orange = "#007F80", "#CC6A14"
    plt.rcParams.update({"font.family": "DejaVu Sans", "svg.fonttype": "none",
                         "text.color": ink, "axes.labelcolor": muted,
                         "xtick.color": muted, "ytick.color": muted})
    fig = plt.figure(figsize=(15, 9.8), facecolor=bg)
    fig.text(.065, .936, "Small model. Strong decisions.", fontsize=28, weight="bold")
    fig.text(.065, .888, "qev-0.8B in Q4, Q8 and BF16  /  JevBench v1 · 231 public tasks",
             fontsize=13.5, color=muted)
    handles = [Line2D([], [], color=teal, marker="o", linestyle="", markersize=8,
                      label="Ryzen 5 7600 CPU · measured here"),
               Line2D([], [], color=orange, marker="s", linestyle="", markersize=8,
                      label="H100 80 GB GPU · AWS / Strands report")]
    fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(.06, .848),
               frameon=False, ncol=2, fontsize=11.3, columnspacing=2.4, handletextpad=.5)
    ax = fig.add_axes([.075, .275, .585, .49], facecolor=bg)
    xmax = max(r["median_latency_ms"] for r in rows)
    ax.set_xlim(0, max(300, (int(xmax / 50) + 2) * 50))
    ax.set_ylim(72, 93)
    ax.xaxis.set_major_locator(MultipleLocator(50 if xmax < 450 else 100))
    ax.yaxis.set_major_locator(MultipleLocator(5))
    ax.grid(color=grid, linewidth=.8, zorder=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(length=0, pad=9, labelsize=11)
    ax.set_xlabel("Median decision latency (ms) · lower is better", labelpad=15, fontsize=11.5)
    ax.set_ylabel("Accuracy (%) · higher is better", labelpad=16, fontsize=11.5)
    # Separate close points with fixed label offsets, while keeping every
    # marker at its measured coordinates (no jitter or invented differences).
    offsets = {"qev-q4": (-12, 36), "qev-q8": (15, -33), "qev-bf16": (13, 27),
               "gemma-e2b": (14, -27), "gemma-e4b": (13, 15),
               "gemma-12b": (13, -25), "gemma-26b": (-13, 15)}
    for r in rows:
        local = r["source"] == "local_cpu"
        highlight = r["id"] == "qev-q4"
        x, y = r["median_latency_ms"], 100 * r["accuracy"]
        marker = "*" if highlight else "o" if local else "s"
        if local:
            ax.errorbar(x, y, xerr=[[x-r["median_min_ms"]], [r["median_max_ms"]-x]],
                        color=teal, capsize=3, linewidth=1.4, zorder=3)
        ax.scatter(x, y, marker=marker, s=330 if highlight else 115,
                   color=teal if local else orange, edgecolor=bg, linewidth=1.7, zorder=5)
        dx, dy = offsets[r["id"]]
        label = f'{r["chart_label"]}\n{100*r["accuracy"]:.2f}% · {x:.0f} ms'
        ax.annotate(label, (x, y), xytext=(dx, dy), textcoords="offset points",
                    ha="right" if dx < 0 else "left", va="center", fontsize=10.3,
                    weight="bold" if highlight else "normal", color=teal if local else ink,
                    arrowprops=dict(arrowstyle="-", color=grid, linewidth=1),
                    bbox=dict(boxstyle="round,pad=.25", facecolor=bg, edgecolor="none", alpha=.94), zorder=6)

    fig.text(.725, .752, "qev-0.8B", fontsize=19, weight="bold", color=teal)
    fig.text(.725, .718, "Same fine-tune. Three formats.", fontsize=10.5, color=muted)
    for y, key in zip([.555, .407, .259], ["qev-bf16", "qev-q8", "qev-q4"]):
        r = by[key]
        hi = key == "qev-q4"
        card = FancyBboxPatch((.715, y), .24, .119, boxstyle="round,pad=.008,rounding_size=.012",
                               transform=fig.transFigure, facecolor="#E5F3F0" if hi else "#FFFFFF",
                               edgecolor="#98CFC3" if hi else grid, linewidth=1)
        fig.add_artist(card)
        mb = r["file_bytes"] / 1e6
        title = "Q4 · H128/Q4-G32-DOT4" if hi else r["format"]
        fig.text(.73, y+.09, title, fontsize=11.3, weight="bold", color=teal if hi else ink)
        fig.text(.73, y+.051, f'{100*r["accuracy"]:.2f}% accuracy', fontsize=15.8, weight="bold")
        fig.text(.73, y+.016, f'{mb:,.0f} MB file  ·  {r["median_latency_ms"]:.0f} ms median',
                 fontsize=10.3, color=muted)

    fig.text(.065, .160,
             f'Q4: {q4["correct"]}/231 correct in a {q4["file_bytes"]/1e6:.0f} MB model file. '
             'One task = 0.43 percentage points.', fontsize=11.6, weight="bold")
    fig.text(.065, .120,
             "qev: Ryzen 5 7600, 6 physical cores; native CPU (Q4), llama.cpp CPU (Q8/BF16). "
             "Gemma: Strands Decider fine-tunes on H100.", fontsize=9.1, color=muted)
    fig.text(.065, .092,
             "CPU latency: median of 3 full-pass medians after 1 warmup; serial HTTP; model loading excluded. "
             "Whiskers: range of the 3 medians.", fontsize=9.1, color=muted)
    fig.text(.065, .064,
             "Different hardware and runtimes across families. Gemma results are reported, not rerun here. "
             "Model-file sizes are not RAM usage.", fontsize=9.1, color=muted)
    fig.text(.065, .029,
             "Sources: strandsagents.com/blog/strands-decider-gemma-4-models/  ·  "
             f"github.com/Danmoreng/qwen35-cpu  ·  {measured_date}", fontsize=8.8, color=muted)
    a.output.parent.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "svg"):
        fig.savefig(a.output.with_suffix("."+ext), dpi=160, facecolor=bg)
    svg = a.output.with_suffix(".svg")
    svg.write_text("\n".join(line.rstrip() for line in svg.read_text().splitlines()) + "\n")
    plt.close(fig)


if __name__ == "__main__":
    main()
