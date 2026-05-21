import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize


# ============================================================
# Data
# Each value is: distance mean RMSE (mm) / angle mean RMSE (deg)
# ============================================================

conditions = ["Clean", "Gaussian 2%", "Bias 1%", "Drift 2%", "Impulse 1%", "Gain 3%"]
condition_labels = ["Clean", "Gaussian-2", "Bias-1", "Drift-2", "Impulse-1", "Gain-3"]

models = [
    "TinyNet",
    "KNN",
    "MobileNetV2",
    "ShuffleNetV2",
    "SqueezeNet",
    "EfficientLite",
]

distance = np.array([
    [0.058, 0.131, 0.053, 0.070, 1.665, 0.075],
    [0.252, 0.285, 0.262, 0.277, 1.473, 0.208],
    [0.120, 0.183, 0.126, 0.125, 1.455, 0.139],
    [0.144, 0.226, 0.145, 0.150, 3.417, 0.144],
    [0.190, 0.161, 0.163, 0.163, 0.737, 0.172],
    [0.089, 0.137, 0.067, 0.082, 3.254, 0.074],
])

angle = np.array([
    [0.164, 0.218, 0.137, 0.206, 1.336, 0.243],
    [0.294, 0.313, 0.285, 0.314, 1.223, 0.293],
    [0.339, 0.405, 0.339, 0.357, 0.670, 0.330],
    [0.337, 0.352, 0.326, 0.356, 2.615, 0.320],
    [0.707, 0.474, 0.399, 0.412, 0.790, 0.442],
    [0.259, 0.254, 0.216, 0.273, 6.416, 0.208],
])


# ============================================================
# Style
# ============================================================

plt.rcParams.update({
    "font.family": "Times New Roman",
    "font.size": 8,
    "axes.labelsize": 8,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "legend.fontsize": 7,
    "axes.linewidth": 0.8,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "xtick.top": True,
    "ytick.right": True,
    "mathtext.fontset": "stix",
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "svg.fonttype": "none",
    "figure.dpi": 300,
    "savefig.dpi": 600,
})

FIG_W_IN = 7.16
FIG_H_IN = 3.05

dist_clip = 1.8
angle_clip = 1.5

# A dimensionless joint robustness score used only for visualization.
# Lower is better. Clipping prevents impulse outliers from flattening
# all other conditions.
dist_score = np.minimum(distance / dist_clip, 1.0)
angle_score = np.minimum(angle / angle_clip, 1.0)
joint_score = 0.5 * (dist_score + angle_score)

cmap = LinearSegmentedColormap.from_list(
    "white_to_blue",
    ["#f7fbff", "#d6e6f2", "#9ecae1", "#4292c6", "#08519c"],
)
norm = Normalize(vmin=0.0, vmax=1.0)


# ============================================================
# Figure
# ============================================================

fig, (ax1, ax2) = plt.subplots(
    1,
    2,
    figsize=(FIG_W_IN, FIG_H_IN),
    gridspec_kw={"width_ratios": [1.08, 1.22]},
)


# ============================================================
# (a) Model-level robustness profile
# ============================================================

x = np.arange(len(conditions))
colors = ["#1f77b4", "#7f7f7f", "#2ca02c", "#9467bd", "#8c564b", "#ff7f0e"]
markers = ["o", "s", "^", "D", "v", "P"]
linestyles = ["-", "--", "-.", ":", (0, (3, 1, 1, 1)), (0, (5, 2))]

for idx, name in enumerate(models):
    is_ours = name == "TinyNet"
    ax1.plot(
        x,
        joint_score[idx],
        color=colors[idx],
        marker=markers[idx],
        linestyle=linestyles[idx],
        linewidth=1.8 if is_ours else 1.15,
        markersize=4.2 if is_ours else 3.4,
        markerfacecolor="white" if not is_ours else colors[idx],
        markeredgewidth=0.8,
        label=name,
        zorder=4 if is_ours else 3,
    )

ax1.set_xlim(-0.25, len(conditions) - 0.75)
ax1.set_ylim(0.0, 1.02)
ax1.set_yticks(np.linspace(0.0, 1.0, 6))
ax1.set_xticks(x)
ax1.set_xticklabels(condition_labels, rotation=0)
ax1.set_ylabel(r"$E_{\mathrm{joint}}$")
ax1.set_xlabel("Disturbance type")
ax1.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.45)
ax1.set_axisbelow(True)
ax1.legend(
    frameon=False,
    loc="upper left",
    bbox_to_anchor=(0.0, 1.02),
    fontsize=6.8,
    ncol=2,
    columnspacing=0.8,
    handlelength=1.7,
    handletextpad=0.4,
)

# ============================================================
# (b) Distance / angle robustness heatmap
# ============================================================

ax2.imshow(joint_score, cmap=cmap, norm=norm, aspect="auto")

ax2.set_xticks(np.arange(len(conditions)))
ax2.set_xticklabels(condition_labels, rotation=0)
ax2.set_yticks(np.arange(len(models)))
ax2.set_yticklabels(models)

ax2.tick_params(
    axis="x",
    top=False,
    bottom=True,
    labeltop=False,
    labelbottom=True,
    pad=3,
)
ax2.tick_params(axis="y", pad=3)

ax2.set_xticks(np.arange(-0.5, len(conditions), 1), minor=True)
ax2.set_yticks(np.arange(-0.5, len(models), 1), minor=True)
ax2.grid(which="minor", color="0.50", linewidth=0.55)
ax2.tick_params(which="minor", bottom=False, left=False)

for spine in ax2.spines.values():
    spine.set_linewidth(0.9)

for i in range(len(models)):
    for j in range(len(conditions)):
        cell_text = f"{distance[i, j]:.2f}/{angle[i, j]:.2f}"
        text_color = "white" if joint_score[i, j] >= 0.58 else "black"
        ax2.text(
            j,
            i,
            cell_text,
            ha="center",
            va="center",
            color=text_color,
            fontsize=6.5,
        )

ax2.set_xlabel("Distance / angle RMSE",)


# ============================================================
# Layout and save
# ============================================================

plt.subplots_adjust(
    left=0.075,
    right=0.985,
    bottom=0.30,
    top=0.94,
    wspace=0.35,
)

pos1 = ax1.get_position()
pos2 = ax2.get_position()
caption_y = 0.115
fig.text(
    pos1.x0 + pos1.width / 2,
    caption_y,
    "(a) Model-level robustness profiles",
    ha="center",
    va="center",
    fontsize=9,
)
fig.text(
    pos2.x0 + pos2.width / 2,
    caption_y,
    "(b) Distance/angle robustness table",
    ha="center",
    va="center",
    fontsize=9,
)

out_base = "Fig_signal_disturbance_distance_angle_TAES"

fig.savefig(out_base + ".pdf", bbox_inches="tight")
fig.savefig(out_base + ".svg", bbox_inches="tight")
fig.savefig(out_base + ".png", bbox_inches="tight")
fig.savefig(out_base + ".tif", bbox_inches="tight")

plt.show()
