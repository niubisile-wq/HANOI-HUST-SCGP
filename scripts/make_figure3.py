from __future__ import annotations

import csv
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "source_data" / "figure3_sealed_confirmation.csv"
OUTPUT = ROOT / "figures" / "figure3_sealed_confirmation_gap"

MODEL_LABELS = {
    "noace_classical": "NOACE classical",
    "source_logistic_l2": "Source Logistic L2",
    "source_mlp_deep": "Source MLP deep",
}
MODEL_COLORS = {
    "noace_classical": "#AAB4BA",
    "source_logistic_l2": "#1F5A7A",
    "source_mlp_deep": "#6E91A8",
}
DARK = "#263640"
GRID = "#D9E0E4"

mpl.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
        "font.size": 7.0,
        "axes.titlesize": 7.8,
        "axes.labelsize": 7.2,
        "axes.linewidth": 0.7,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "legend.frameon": False,
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
    }
)

with SOURCE.open("r", encoding="utf-8", newline="") as handle:
    rows = list(csv.DictReader(handle))

models = [row["model"] for row in rows]
labels = [MODEL_LABELS[model] for model in models]
colors = [MODEL_COLORS[model] for model in models]
auroc = np.array([float(row["mean_component_auroc"]) for row in rows])
exact_set = np.array([float(row["exact_set_accuracy"]) for row in rows])

fig, axes = plt.subplots(
    2, 1, figsize=(3.48, 3.12), sharex=True,
    gridspec_kw={"height_ratios": [1, 1], "hspace": 0.50}
)
fig.subplots_adjust(left=0.39, right=0.965, top=0.91, bottom=0.17)

y = np.arange(len(models))

for ax, values, title, panel in [
    (axes[0], auroc, "Component-level discrimination", "a"),
    (axes[1], exact_set, "Exact compound recovery", "b"),
]:
    ax.barh(
        y, values, height=0.48, color=colors,
        edgecolor="white", linewidth=0.6, zorder=2
    )
    ax.scatter(
        values, y, s=17, color=colors,
        edgecolor="white", linewidth=0.55, zorder=3
    )
    for yi, value in zip(y, values):
        x_text = max(value + 0.025, 0.055)
        ax.text(
            x_text, yi, f"{value:.3f}",
            ha="left", va="center", fontsize=6.8, color=DARK
        )
    ax.set_yticks(y, labels)
    ax.invert_yaxis()
    ax.set_xlim(0.0, 1.0)
    ax.set_xticks([0.0, 0.5, 1.0])
    ax.grid(axis="x", color=GRID, linewidth=0.55, zorder=0)
    ax.tick_params(axis="y", length=0, pad=4, labelsize=6.8)
    ax.tick_params(axis="x", length=2.5, colors="#586870", labelsize=6.6)
    ax.spines["left"].set_visible(False)
    ax.spines["bottom"].set_color("#7C8A92")
    ax.set_title(title, loc="left", fontweight="bold", pad=5)
    ax.text(
        -0.36, 1.08, panel, transform=ax.transAxes,
        fontsize=8.5, fontweight="bold", color=DARK
    )

axes[1].set_xlabel("Metric value on the sealed 14-bearing partition")

fig.text(
    0.39, 0.045,
    "The panels use the same 0–1 scale but quantify different prediction endpoints.",
    ha="left", va="bottom", fontsize=5.9, color="#586870"
)

for suffix, kwargs in {
    ".svg": {},
    ".pdf": {},
    ".png": {"dpi": 600},
    ".tiff": {"dpi": 600},
}.items():
    fig.savefig(OUTPUT.with_suffix(suffix), bbox_inches="tight", facecolor="white", **kwargs)

plt.close(fig)
