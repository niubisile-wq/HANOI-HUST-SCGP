"""Build Figure 4 from frozen protocol-sensitivity results.

The figure deliberately separates aggregate score change from foldwise
configuration selection.  The latter is not treated as an uncertainty
interval or as a same-candidate-set leaderboard.
"""

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "source_data"
OUTPUT = ROOT / "figures" / "figure4_protocol_sensitivity"

mpl.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
        "font.size": 7,
        "axes.titlesize": 8,
        "axes.labelsize": 7,
        "xtick.labelsize": 6.5,
        "ytick.labelsize": 6.5,
        "axes.linewidth": 0.7,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "legend.frameon": False,
        "pdf.fonttype": 42,
        "svg.fonttype": "none",
    }
)

BLUE = "#1f5f7a"
LIGHT_BLUE = "#7ea4b7"
ORANGE = "#c9783d"
GREY = "#aeb8bd"
GRID = "#d9e0e3"
TEXT = "#263238"

metrics = pd.read_csv(SOURCE / "figure4_protocol_metrics.csv")
selection = pd.read_csv(SOURCE / "figure4_nested_selection_counts.csv")

fig = plt.figure(figsize=(3.50, 4.25), constrained_layout=False)
grid = fig.add_gridspec(
    2,
    1,
    height_ratios=[1.0, 1.25],
    hspace=0.62,
    left=0.25,
    right=0.98,
    top=0.96,
    bottom=0.10,
)

# Panel a: paired protocol scores.  Each metric occupies its own row so that
# values and deltas remain readable at final single-column size.
ax_a = fig.add_subplot(grid[0])
row_labels = ["Mean component\nAUROC", "Mean balanced\naccuracy", "Exact-set\naccuracy"]
y_positions = [2, 1, 0]
for y_pos, (_, row) in zip(y_positions, metrics.iterrows(), strict=True):
    source_value = row["frozen_source"]
    nested_value = row["nested_physical_unit"]
    ax_a.plot(
        [nested_value, source_value],
        [y_pos, y_pos],
        color=GREY,
        lw=1.3,
        zorder=1,
    )
    ax_a.scatter(
        source_value,
        y_pos,
        s=28,
        color=BLUE,
        edgecolor="white",
        linewidth=0.6,
        zorder=3,
        label="Frozen source" if y_pos == 2 else None,
    )
    ax_a.scatter(
        nested_value,
        y_pos,
        s=28,
        color=ORANGE,
        marker="s",
        edgecolor="white",
        linewidth=0.6,
        zorder=3,
        label="Nested selection" if y_pos == 2 else None,
    )
    ax_a.text(
        source_value,
        y_pos + 0.18,
        f"{source_value:.3f}",
        ha="center",
        va="bottom",
        color=BLUE,
        fontsize=6.1,
    )
    ax_a.text(
        nested_value,
        y_pos - 0.18,
        f"{nested_value:.3f}",
        ha="center",
        va="top",
        color=ORANGE,
        fontsize=6.1,
    )
    ax_a.text(
        0.872,
        y_pos,
        f'Δ {row["delta"]:+.3f}',
        ha="right",
        va="center",
        color=TEXT,
        fontsize=6.1,
        fontweight="bold" if y_pos == 2 else "normal",
    )

ax_a.set_xlim(0.49, 0.88)
ax_a.set_ylim(-0.55, 2.55)
ax_a.set_yticks(y_positions, row_labels)
ax_a.set_xticks([0.5, 0.6, 0.7, 0.8])
ax_a.set_xlabel("Metric value")
ax_a.set_title("Protocol tightening changes the reported score", loc="left", pad=5)
ax_a.grid(axis="x", color=GRID, lw=0.6)
ax_a.set_axisbelow(True)
ax_a.spines["left"].set_visible(False)
ax_a.spines["bottom"].set_color("#839096")
ax_a.tick_params(axis="y", length=0, pad=4)
ax_a.legend(
    loc="upper left",
    bbox_to_anchor=(0.0, 1.01),
    ncol=2,
    fontsize=5.8,
    handletextpad=0.4,
    columnspacing=0.9,
    borderaxespad=0,
)
ax_a.text(
    -0.20,
    1.08,
    "a",
    transform=ax_a.transAxes,
    fontsize=8,
    fontweight="bold",
    color=TEXT,
)

# Panel b: selected configuration frequency across outer folds.
ax_b = fig.add_subplot(grid[1])
selection = selection.sort_values("count", ascending=True).reset_index(drop=True)
labels = []
colors = []
for row in selection.itertuples(index=False):
    family = "Logistic L2" if row.family == "logistic_l2" else "Extra Trees"
    view = {
        "all": "All features",
        "statistics": "Statistics",
        "fixed_log_power": "Fixed log-power",
    }[row.representation]
    labels.append(f"{family} · {view}")
    colors.append(BLUE if row.family == "logistic_l2" else GREY)

y = range(len(selection))
bars = ax_b.barh(y, selection["count"], height=0.64, color=colors)
for bar, count in zip(bars, selection["count"], strict=True):
    ax_b.text(
        count + 0.18,
        bar.get_y() + bar.get_height() / 2,
        f"{count}",
        va="center",
        ha="left",
        fontsize=6.4,
        color=TEXT,
    )

ax_b.set_yticks(list(y), labels)
ax_b.set_xlim(0, 11)
ax_b.set_xticks([0, 2, 4, 6, 8, 10])
ax_b.set_xlabel("Number of outer folds selecting the configuration")
ax_b.set_title("Nested selection is distributed across configurations", loc="left", pad=5)
ax_b.grid(axis="x", color=GRID, lw=0.6)
ax_b.set_axisbelow(True)
ax_b.spines["left"].set_visible(False)
ax_b.spines["bottom"].set_color("#839096")
ax_b.tick_params(axis="y", length=0, pad=3)
ax_b.text(
    -0.20,
    1.08,
    "b",
    transform=ax_b.transAxes,
    fontsize=8,
    fontweight="bold",
    color=TEXT,
)
ax_b.text(
    0.0,
    -0.33,
    "19 leave-one-bearing-out folds; 24 candidate configurations per fold.",
    transform=ax_b.transAxes,
    ha="left",
    va="top",
    fontsize=5.8,
    color="#59686f",
)

fig.savefig(OUTPUT.with_suffix(".pdf"), bbox_inches="tight")
fig.savefig(OUTPUT.with_suffix(".svg"), bbox_inches="tight")
fig.savefig(OUTPUT.with_suffix(".png"), dpi=600, bbox_inches="tight")
fig.savefig(
    OUTPUT.with_suffix(".tiff"),
    dpi=600,
    bbox_inches="tight",
    pil_kwargs={"compression": "tiff_lzw"},
)
plt.close(fig)
