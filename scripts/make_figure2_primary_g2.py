from __future__ import annotations

import csv
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "source_data" / "figure2_primary_g2_protocol.csv"
OUTPUT = ROOT / "figures" / "figure2_primary_g2_protocol"

PROTOCOLS = [
    "Record-grouped\nfixed reference",
    "Bearing-grouped\nfixed reference",
    "Bearing-grouped\nnested selection",
]
PROTOCOL_KEYS = [
    "Record-grouped fixed",
    "Bearing-grouped fixed",
    "Bearing-grouped nested",
]
ENDPOINTS = ["Record", "Bearing aggregated"]
COLORS = {"Record": "#1F5A7A", "Bearing aggregated": "#D7792F"}
DARK = "#263640"
GRID = "#D9E0E4"

mpl.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
        "font.size": 7.0,
        "axes.titlesize": 8.0,
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

values = {
    (row["protocol"], row["endpoint"]): {
        "auroc": float(row["mean_component_auroc"]),
        "exact": float(row["exact_set_accuracy"]),
    }
    for row in rows
}

fig, axes = plt.subplots(1, 2, figsize=(7.08, 2.78), sharey=True)
fig.subplots_adjust(left=0.20, right=0.985, top=0.78, bottom=0.22, wspace=0.18)
y = np.arange(len(PROTOCOL_KEYS))

for ax, metric, title, panel in [
    (axes[0], "auroc", "Mean component AUROC", "a"),
    (axes[1], "exact", "Exact-set accuracy", "b"),
]:
    record = np.array([values[(key, "Record")][metric] for key in PROTOCOL_KEYS])
    bearing = np.array(
        [values[(key, "Bearing aggregated")][metric] for key in PROTOCOL_KEYS]
    )
    for yi, left, right in zip(y, record, bearing):
        ax.plot([left, right], [yi, yi], color="#AEB9BF", linewidth=1.1, zorder=1)
    for endpoint, series in [("Record", record), ("Bearing aggregated", bearing)]:
        ax.scatter(
            series,
            y,
            s=31,
            color=COLORS[endpoint],
            edgecolor="white",
            linewidth=0.65,
            label=endpoint,
            zorder=3,
        )
        for yi, value in zip(y, series):
            offset = -0.030 if endpoint == "Record" else 0.030
            align = "right" if endpoint == "Record" else "left"
            ax.text(
                value + offset,
                yi - 0.18,
                f"{value:.3f}",
                ha=align,
                va="center",
                fontsize=6.4,
                color=COLORS[endpoint],
            )
    ax.set_xlim(0.0, 1.0)
    ax.set_xticks(np.linspace(0.0, 1.0, 6))
    ax.set_xlabel("Metric value")
    ax.set_title(title, loc="left", fontweight="bold", pad=5)
    ax.grid(axis="x", color=GRID, linewidth=0.55, zorder=0)
    ax.tick_params(axis="y", length=0, pad=7, labelsize=6.8)
    ax.tick_params(axis="x", length=2.5, colors="#586870", labelsize=6.6)
    ax.spines["left"].set_visible(False)
    ax.spines["bottom"].set_color("#7C8A92")
    ax.text(
        -0.04,
        1.08,
        panel,
        transform=ax.transAxes,
        fontsize=9.0,
        fontweight="bold",
        color=DARK,
    )

axes[0].set_yticks(y, PROTOCOLS)
axes[0].invert_yaxis()
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(
    handles,
    labels,
    loc="upper center",
    bbox_to_anchor=(0.62, 0.965),
    ncol=2,
    handletextpad=0.5,
    columnspacing=1.4,
)
fig.text(
    0.20,
    0.055,
    "Means across 100 registered splits; paired endpoints within a row reuse the same predictions.",
    ha="left",
    va="bottom",
    fontsize=6.1,
    color="#586870",
)

for suffix, kwargs in {
    ".svg": {},
    ".pdf": {},
    ".png": {"dpi": 600},
    ".tiff": {"dpi": 600},
}.items():
    fig.savefig(OUTPUT.with_suffix(suffix), bbox_inches="tight", facecolor="white", **kwargs)

plt.close(fig)
