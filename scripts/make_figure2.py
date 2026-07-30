from __future__ import annotations

import csv
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "source_data"
OUTPUT = ROOT / "figures" / "figure2_source_champion"

BLUE = "#1F5A7A"
BLUE_MID = "#6E91A8"
NEUTRAL = "#AAB4BA"
DARK = "#263640"
GRID = "#D9E0E4"

mpl.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
        "font.size": 7.2,
        "axes.titlesize": 8.2,
        "axes.labelsize": 7.4,
        "axes.linewidth": 0.7,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "xtick.labelsize": 6.8,
        "ytick.labelsize": 6.8,
        "legend.frameon": False,
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
    }
)


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def display_name(method: str, representation: str) -> str:
    method_map = {
        "logistic_l2": "Logistic L2",
        "extra_trees": "Extra Trees",
        "rbf_svm": "RBF SVM",
    }
    representation_map = {
        "envelope_log_power": "Envelope log-power",
        "fixed_log_power": "Fixed log-power",
        "statistics": "Statistics",
        "all": "All features",
    }
    return f"{method_map[method]}\n{representation_map[representation]}"


leaderboard = read_rows(SOURCE / "figure2_source_leaderboard.csv")
ablation = read_rows(SOURCE / "figure2_representation_ablation.csv")

leaderboard = sorted(
    leaderboard, key=lambda row: float(row["mean_component_auroc"]), reverse=True
)
labels = [display_name(row["method"], row["representation"]) for row in leaderboard]
auroc = np.array([float(row["mean_component_auroc"]) for row in leaderboard])

representations = [row["representation"] for row in ablation]
metrics = ["AUROC", "BAcc", "AUPR", "Macro-F1", "Exact-set", "Brier"]
raw = np.array([[float(row[metric]) for metric in metrics] for row in ablation])

# Column-wise performance strength. Higher is better except for Brier.
strength = np.empty_like(raw)
for column, metric in enumerate(metrics):
    values = raw[:, column]
    if metric == "Brier":
        values = -values
    span = values.max() - values.min()
    strength[:, column] = 0.5 if span == 0 else (values - values.min()) / span

fig = plt.figure(figsize=(7.2, 3.35), facecolor="white")
grid = fig.add_gridspec(
    1, 2, width_ratios=[1.0, 1.42], left=0.155, right=0.985,
    top=0.87, bottom=0.20, wspace=0.34
)

# Panel a: ranked frozen source-side AUROC.
ax_a = fig.add_subplot(grid[0, 0])
y = np.arange(len(leaderboard))
colors = [BLUE] + [BLUE_MID] + [NEUTRAL] * (len(leaderboard) - 2)
ax_a.hlines(y, 0.70, auroc, color=colors, linewidth=2.4, zorder=2)
ax_a.scatter(auroc, y, s=29, color=colors, edgecolor="white", linewidth=0.6, zorder=3)
for yi, value in zip(y, auroc):
    ax_a.text(value + 0.0027, yi, f"{value:.3f}", va="center", ha="left",
              fontsize=6.7, color=DARK)
ax_a.set_yticks(y, labels)
ax_a.invert_yaxis()
ax_a.set_xlim(0.70, 0.832)
ax_a.set_xticks([0.70, 0.75, 0.80])
ax_a.set_xlabel("Mean component AUROC")
ax_a.set_title("Frozen source-side ranking", loc="left", fontweight="bold", pad=8)
ax_a.grid(axis="x", color=GRID, linewidth=0.55, zorder=0)
ax_a.tick_params(axis="y", length=0, pad=4)
ax_a.spines["left"].set_visible(False)
ax_a.spines["bottom"].set_color("#7C8A92")
ax_a.get_yticklabels()[0].set_fontweight("bold")
ax_a.text(
    auroc[0] - 0.001, 0, " reference ",
    va="center", ha="right", fontsize=6.1, color="white",
    bbox={"boxstyle": "round,pad=0.16", "facecolor": BLUE, "edgecolor": "none"}
)

# Panel b: metric profile of four representations under the same classifier.
ax_b = fig.add_subplot(grid[0, 1])
cmap = LinearSegmentedColormap.from_list(
    "source_strength", ["#F2F5F6", "#B9CEDA", BLUE]
)
image = ax_b.imshow(strength, cmap=cmap, vmin=0, vmax=1, aspect="auto")

rep_labels = [
    "Envelope log-power",
    "All features",
    "Fixed log-power",
    "Statistics",
]
metric_labels = ["AUROC ↑", "BAcc ↑", "AUPR ↑", "Macro-F1 ↑", "Exact-set ↑", "Brier ↓"]
ax_b.set_yticks(np.arange(len(rep_labels)), rep_labels)
ax_b.set_xticks(np.arange(len(metric_labels)), metric_labels)
ax_b.tick_params(axis="x", top=True, bottom=False, labeltop=True, labelbottom=False,
                 length=0, pad=4)
ax_b.tick_params(axis="y", length=0, pad=4)
ax_b.set_title(
    "Representation profile under Logistic L2",
    loc="left", fontweight="bold", pad=8
)
for tick in ax_b.get_yticklabels():
    if tick.get_text() == "Envelope log-power":
        tick.set_fontweight("bold")

for row in range(raw.shape[0]):
    for column in range(raw.shape[1]):
        color = "white" if strength[row, column] > 0.62 else DARK
        ax_b.text(
            column, row, f"{raw[row, column]:.3f}",
            ha="center", va="center", fontsize=6.35, color=color
        )

# White cell boundaries and a restrained outline around the selected reference row.
ax_b.set_xticks(np.arange(-0.5, len(metrics), 1), minor=True)
ax_b.set_yticks(np.arange(-0.5, len(rep_labels), 1), minor=True)
ax_b.grid(which="minor", color="white", linewidth=1.2)
ax_b.tick_params(which="minor", bottom=False, left=False)
for spine in ax_b.spines.values():
    spine.set_visible(False)
ax_b.add_patch(
    mpl.patches.Rectangle(
        (-0.5, -0.5), len(metrics), 1, fill=False,
        edgecolor=BLUE, linewidth=1.2, clip_on=False
    )
)

# Panel labels and compact interpretive footer.
fig.text(0.018, 0.94, "a", fontsize=9.2, fontweight="bold", color=DARK)
fig.text(0.485, 0.94, "b", fontsize=9.2, fontweight="bold", color=DARK)
fig.text(
    0.155, 0.055,
    "Darker cells indicate stronger performance within each metric; arrows show the preferred direction.",
    fontsize=6.4, color="#53636C"
)

for suffix, kwargs in {
    ".svg": {},
    ".pdf": {},
    ".png": {"dpi": 600},
    ".tiff": {"dpi": 600},
}.items():
    fig.savefig(OUTPUT.with_suffix(suffix), bbox_inches="tight", facecolor="white", **kwargs)

plt.close(fig)
