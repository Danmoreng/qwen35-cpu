#!/usr/bin/env python3
"""Render English When2Call accuracy comparisons from frozen aggregate data.

Requires matplotlib. Cohorts stay separate; no hosted GPU memory is inferred.
"""
import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import MultipleLocator


def render(cohort, output):
    bg, ink, muted, grid = "#F8FAFC", "#13233B", "#53647B", "#DFE6EE"
    colors = {"local_cpu": "#007F80", "local_gpu": "#347FAD", "published": "#CC6A14"}
    plt.rcParams.update({"font.family": "DejaVu Sans", "svg.fonttype": "none",
                         "text.color": ink, "axes.labelcolor": muted,
                         "xtick.color": muted, "ytick.color": muted})
    fig = plt.figure(figsize=(16, 9), facecolor=bg)
    aa = cohort["id"] == "aa_subset"
    published = "Artificial Analysis · published" if aa else "Independent study · published"
    handles = [Patch(color=colors["published"], label=published),
               Patch(color=colors["local_cpu"], label="AMD Ryzen 5 7600 · Q4"),
               Patch(color=colors["local_gpu"], label="Intel Arc Pro B70 · Q8 / BF16")]
    fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(.045, .985),
               frameon=False, ncol=3, fontsize=11, columnspacing=2.1, handlelength=1.1)
    fig.text(.055, .897, f'When2Call · {cohort["tasks"]:,} tasks · same cohort, different prompts',
             fontsize=11.5, color=muted)
    fig.text(.944, .897, "Parameters", fontsize=11.5, color=muted, ha="right")
    fig.add_artist(Line2D([.87, .87], [.16, .87], transform=fig.transFigure, color=grid, linewidth=1))
    ax = fig.add_axes([.32, .15, .51, .70], facecolor=bg)
    rows = cohort["models"]
    ax.set_xlim(0, 106)
    ax.set_ylim(-.65, len(rows)-.35)
    ax.invert_yaxis()
    ax.xaxis.set_major_locator(MultipleLocator(20))
    ax.grid(axis="x", color=grid, linewidth=.8, zorder=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_yticks([])
    ax.tick_params(axis="x", length=0, pad=10, labelsize=11)
    ax.set_xlabel("Accuracy (%)", fontsize=12, labelpad=15)
    labels = []
    for i, row in enumerate(rows):
        assert row["tasks"] == cohort["tasks"]
        assert abs(row["correct"] / row["tasks"] - row["accuracy"]) < 1e-10
        local = row["source"].startswith("local_")
        color = colors[row["source"] if local else "published"]
        score = 100 * row["accuracy"]
        highlight = row["id"] == "qev-q4"
        ax.barh(i, score, color=color, height=.53, zorder=3)
        labels.append(ax.text(-.035, i-.12, row["label"], transform=ax.get_yaxis_transform(),
                   ha="right", va="center", fontsize=12.2, color=color if local else ink,
                   weight="bold" if highlight else "normal"))
        detail = ("Local re-score" if aa else "Local measurement") if local else row["provider"]
        labels.append(ax.text(-.035, i+.20, detail, transform=ax.get_yaxis_transform(),
                   ha="right", va="center", fontsize=9.5, color=muted))
        labels.append(ax.text(score+1.2, i, f"{score:.2f}%", ha="left", va="center",
                   fontsize=12, weight="bold", color=color if local else ink))
        n = row.get("parameters_b")
        size = f"{n:g}B" if n is not None else "Undisclosed"
        labels.append(ax.text(1.225, i, size, transform=ax.get_yaxis_transform(), ha="right",
                   va="center", fontsize=12 if n is not None else 10.5,
                   color=color if local else ink if n is not None else muted,
                   weight="bold" if local else "normal"))
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    bounds = fig.bbox
    assert all(bounds.contains(text.get_window_extent(renderer).x0, text.get_window_extent(renderer).y0)
               and bounds.contains(text.get_window_extent(renderer).x1, text.get_window_extent(renderer).y1)
               for text in labels), "Chart text falls outside the image"
    output.parent.mkdir(parents=True, exist_ok=True)
    for extension in ("png", "svg"):
        fig.savefig(output.with_suffix("."+extension), dpi=160, facecolor=bg)
    svg = output.with_suffix(".svg")
    svg.write_text("\n".join(line.rstrip() for line in svg.read_text().splitlines())+"\n")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    data = json.loads(args.data.read_text())
    for cohort in data["cohorts"]:
        render(cohort, args.output_dir / ("when2call-"+cohort["id"].replace("_", "-")))
    def sha(path):
        with path.open("rb") as stream:
            return hashlib.file_digest(stream, "sha256").hexdigest()
    manifest = dict(data_sha256=sha(args.data), renderer_sha256=sha(Path(__file__)),
                    matplotlib_version=matplotlib.__version__,
                    files={path.name: sha(path) for path in sorted(args.output_dir.glob("when2call-*"))
                           if path.suffix in (".png", ".svg")})
    (args.output_dir / "chart-manifest.json").write_text(json.dumps(manifest, indent=2)+"\n")


if __name__ == "__main__":
    main()
