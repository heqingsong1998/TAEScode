import numpy as np
import matplotlib.pyplot as plt


# -----------------------------
# Data
# -----------------------------
models = [
    "TinyNet",
    "MobileNetV2-1D",
    "ShuffleNetV2-1D",
    "SqueezeNet-1D",
    "EfficientLite-1D",
]
plot_labels = [
    "TinyNet",
    "MobileNetV2-1D",
    "ShuffleNetV2-1D",
    "SqueezeNet-1D",
    "EfficientLite-1D",
]

ram_kib = np.array([8.25, 50.00, 40.00, 32.00, 80.00])
macs = np.array([167790, 4569030, 697414, 914438, 3987782])
latency_ms = np.array([0.1141, 0.2318, 0.1276, 0.1666, 0.2492])

dmean_rmse = np.array([0.058, 0.120, 0.144, 0.190, 0.089])
pitch_rmse = np.array([0.157, 0.361, 0.395, 0.836, 0.271])
roll_rmse = np.array([0.171, 0.316, 0.279, 0.578, 0.246])
angle_rmse = (pitch_rmse + roll_rmse) / 2.0

# Compact accuracy score for a single trade-off plot.
# Lower is better. TinyNet is normalized to 1.0.
joint_rmse = 0.5 * (
    dmean_rmse / dmean_rmse[0]
    + angle_rmse / angle_rmse[0]
)


# -----------------------------
# Figure style
# -----------------------------
plt.rcParams.update({
    "font.family": "Times New Roman",
    "font.size": 7.5,
    "axes.labelsize": 8.0,
    "xtick.labelsize": 7.0,
    "ytick.labelsize": 7.0,
    "legend.fontsize": 6.5,
    "axes.linewidth": 0.9,
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

FIG_W_IN = 3.50
FIG_H_IN = 2.35


# -----------------------------
# Plot
# -----------------------------
fig, ax = plt.subplots(1, 1, figsize=(FIG_W_IN, FIG_H_IN))

macs_m = macs / 1e6
marker_sizes = 28 + ram_kib * 0.95

scatter = ax.scatter(
    macs_m,
    joint_rmse,
    s=marker_sizes,
    c=latency_ms,
    cmap="viridis_r",
    edgecolor="black",
    linewidth=0.65,
    zorder=3,
)

# Highlight the proposed model without changing the latency color mapping.
ax.scatter(
    macs_m[0],
    joint_rmse[0],
    s=marker_sizes[0] * 1.25,
    facecolor="none",
    edgecolor="#d62728",
    linewidth=1.2,
    zorder=4,
)

ax.set_xscale("log")
ax.set_xlim(0.08, 7.0)
ax.set_ylim(0.85, 4.65)
ax.set_xlabel("MACs (million, log scale)")
ax.set_ylabel(r"$E_{\mathrm{joint}}$")
ax.set_yticks([1.0, 2.0, 3.0, 4.0])
ax.grid(True, which="major", linestyle="--", linewidth=0.6, alpha=0.45)
ax.grid(True, which="minor", axis="x", linestyle=":", linewidth=0.45, alpha=0.28)
ax.set_axisbelow(True)

ax.axhline(1.0, color="0.35", linestyle="--", linewidth=0.8, alpha=0.7)

label_offsets = {
    "TinyNet": (4, 8),
    "MobileNetV2-1D": (-55, 1),
    "ShuffleNetV2-1D": (-56, 1),
    "SqueezeNet-1D": (7, 0),
    "EfficientLite-1D": (-54, -1),
}

for name, x_val, y_val in zip(plot_labels, macs_m, joint_rmse):
    dx, dy = label_offsets.get(name, (6, 0))
    ax.annotate(
        name,
        xy=(x_val, y_val),
        xytext=(dx, dy),
        textcoords="offset points",
        ha="left",
        va="center",
        fontsize=6.7,
    )

cbar = fig.colorbar(scatter, ax=ax, pad=0.025, fraction=0.08)
cbar.set_label("Latency (ms)", rotation=90)
cbar.ax.tick_params(labelsize=6.7)

legend_sizes = [10, 40, 80]
size_handles = [
    plt.scatter(
        [],
        [],
        s=36 + size * 1.15,
        facecolor="white",
        edgecolor="black",
        linewidth=0.65,
    )
    for size in legend_sizes
]
ax.legend(
    size_handles,
    [f"{size} KiB" for size in legend_sizes],
    title="RAM",
    frameon=False,
    loc="upper left",
    bbox_to_anchor=(0.02, 0.98),
    borderaxespad=0.0,
    handletextpad=0.6,
    labelspacing=0.45,
    fontsize=6.5,
    title_fontsize=6.8,
)

plt.subplots_adjust(left=0.15, right=0.88, bottom=0.20, top=0.96)


# -----------------------------
# Save
# -----------------------------
out_base = "Fig_lightweight_model_comparison_TAES"
fig.savefig(out_base + ".pdf", bbox_inches="tight")
fig.savefig(out_base + ".svg", bbox_inches="tight")
fig.savefig(out_base + ".png", bbox_inches="tight")
fig.savefig(out_base + ".tif", bbox_inches="tight")

plt.show()
