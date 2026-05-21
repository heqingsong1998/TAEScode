import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# =========================
# Data from Table 4
# =========================
variants = [
    "C16", "C24", "C32/Ours", "C48", "C64",
    "w/o Diff", "w/o Residual", "No Dilation", "w/o GlobalNorm"
]
variant_tick_labels = [
    "C16", "C24", "C32/Ours", "C48", "C64",
    "w/o Diff", "w/o Residual", "No Dilation", "w/o GlobalNorm"
]

ram_kib = np.array([4.12, 6.19, 8.25, 12.38, 16.50, 8.25, 7.38, 8.25, 8.25])
macs = np.array([51838, 101622, 167790, 349278, 596302, 162986, 163694, 167790, 167790])

rmse_dist = np.array([0.086, 0.073, 0.058, 0.066, 0.059, 0.060, 0.063, 0.064, 0.063])
rmse_angle = np.array([0.224, 0.195, 0.164, 0.215, 0.184, 0.213, 0.271, 0.249, 0.210])

baseline_idx = variants.index("C32/Ours")
base_dist = rmse_dist[baseline_idx]
base_angle = rmse_angle[baseline_idx]

delta_dist = (rmse_dist - base_dist) / base_dist * 100
delta_angle = (rmse_angle - base_angle) / base_angle * 100

normalized_rmse = 0.5 * (rmse_dist / base_dist) + 0.5 * (rmse_angle / base_angle)

idx_no_base = [i for i in range(len(variants)) if i != baseline_idx]
variants_no_base = [variants[i] for i in idx_no_base]
delta_dist_no_base = delta_dist[idx_no_base]
delta_angle_no_base = delta_angle[idx_no_base]

# =========================
# Style
# =========================
plt.rcParams.update({
    "font.family": "Times New Roman",
    "font.size": 8,
    "axes.labelsize": 8,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "legend.fontsize": 7,
    "axes.linewidth": 0.85,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "xtick.major.width": 0.85,
    "ytick.major.width": 0.85,
    "xtick.top": True,
    "ytick.right": True,
    "mathtext.fontset": "stix",
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "svg.fonttype": "none",
    "figure.dpi": 300,
    "savefig.dpi": 600,
})

blue = "#1f77b4"
orange = "#ff7f0e"
green = "#2ca02c"

fig = plt.figure(figsize=(7.16, 4.95))

# =========================
# (a) Distance and angle RMSE
# =========================
ax1 = fig.add_axes([0.075, 0.650, 0.365, 0.350])
ax1b = ax1.twinx()

x = np.arange(len(variants))
width = 0.36

bar1 = ax1.bar(
    x - width / 2, rmse_dist, width,
    color=blue, edgecolor="black", linewidth=0.4,
    label="Distance RMSE"
)
bar2 = ax1b.bar(
    x + width / 2, rmse_angle, width,
    color=orange, edgecolor="black", linewidth=0.4,
    label="Angle RMSE"
)

ax1.axhline(base_dist, color=blue, linestyle="--", linewidth=1.0, alpha=0.65)
ax1b.axhline(base_angle, color=orange, linestyle="--", linewidth=1.0, alpha=0.65)

ax1.set_ylabel("Distance RMSE (mm)", color=blue)
ax1b.set_ylabel("Angle RMSE (deg)", color=orange)
ax1.tick_params(axis="y", labelcolor=blue)
ax1b.tick_params(axis="y", labelcolor=orange)
ax1.tick_params(axis="x", labelsize=6.2, pad=1)
ax1b.tick_params(axis="y", pad=2)

ax1.set_xticks(x)
ax1.set_xticklabels(variant_tick_labels, rotation=30, ha="right")
ax1.set_ylim(0.0, 0.095)
ax1b.set_ylim(0.0, 0.285)
ax1.grid(axis="y", linestyle="--", alpha=0.35)

# Legend inside the figure
leg = ax1.legend(
    [bar1, bar2],
    ["Distance RMSE", "Angle RMSE"],
    loc="upper left",
    bbox_to_anchor=(0.30, 0.99),
    frameon=True,
    framealpha=0.92,
    facecolor="white",
    edgecolor="none",
    fontsize=6.6,
    borderpad=0.25,
    handlelength=1.3,
    handletextpad=0.35,
)
leg.set_zorder(10)

# =========================
# (b) Relative RMSE increase
# =========================
ax2 = fig.add_axes([0.585, 0.650, 0.365, 0.350])

x2 = np.arange(len(variants_no_base))

line1, = ax2.plot(
    x2,
    delta_dist_no_base,
    color=blue,
    marker="o",
    linewidth=1.8,
    markersize=4.6,
    label="Distance",
)
line2, = ax2.plot(
    x2,
    delta_angle_no_base,
    color=orange,
    marker="s",
    linewidth=1.8,
    markersize=4.6,
    label="Angle",
)

ax2.set_ylabel(r"$\Delta$RMSE over C32/Ours (%)")
ax2.set_xticks(x2)
ax2.set_xticklabels([variant_tick_labels[i] for i in idx_no_base], rotation=30, ha="right")
ax2.tick_params(axis="x", labelsize=6.2, pad=1)
ax2.set_ylim(0, max(delta_angle_no_base) * 1.18)
ax2.grid(axis="y", linestyle="--", alpha=0.35)

dist_label_offsets = {
    0: (0.18, 2.6),
    1: (0.20, 4.1),
    2: (0.00, 2.4),
    3: (0.00, 2.7),
    4: (0.00, 2.8),
    5: (-0.12, 2.8),
    6: (-0.12, 2.8),
    7: (0.08, 2.8),
}
angle_label_offsets = {
    0: (0.08, -10.2),
    1: (0.14, -8.5),
    2: (0.00, 2.6),
    3: (0.00, 4.8),
    4: (-0.50, 3.0),
    5: (0.00, 3.0),
    6: (0.08, 2.7),
    7: (-0.60, -1.7),
}

for i, v in enumerate(delta_dist_no_base):
    dx, dy = dist_label_offsets.get(i, (0.0, 2.5))
    ax2.text(
        i + dx,
        v + dy,
        f"{v:.1f}%",
        ha="center",
        va="bottom",
        fontsize=6.2,
        color=blue
    )

for i, v in enumerate(delta_angle_no_base):
    dx, dy = angle_label_offsets.get(i, (0.0, 2.5))
    ax2.text(
        i + dx,
        v + dy,
        f"{v:.1f}%",
        ha="center",
        va="bottom",
        fontsize=6.2,
        color=orange
    )

ax2.legend(
    loc="upper left",
    frameon=True,
    framealpha=0.92,
    facecolor="white",
    edgecolor="none",
    fontsize=6.8,
)

# =========================
# (c) Accuracy-complexity trade-off
# =========================
ax3 = fig.add_axes([0.135, 0.150, 0.685, 0.335])

scatter = ax3.scatter(
    macs / 1e5,
    normalized_rmse,
    s=55 + 2.2 * ram_kib,
    c=ram_kib,
    cmap="viridis",
    edgecolor="black",
    linewidth=0.8,
    zorder=3
)

# Baseline line
ax3.axhline(
    1.0,
    color="gray",
    linestyle="--",
    linewidth=1.0,
    alpha=0.75
)

# Manual label offsets
offsets = {
    "C16": (7, 5),
    "C24": (7, 4),
    "C32/Ours": (5, -6),
    "C48": (7, 5),
    "C64": (7, -4),
    "w/o Diff": (3, -10),
    "w/o Residual": (8, 4),
    "No Dilation": (8, -2),
    "w/o GlobalNorm": (8, -1),
}

for i, name in enumerate(variants):
    dx, dy = offsets[name]
    ax3.annotate(
        name,
        xy=(macs[i] / 1e5, normalized_rmse[i]),
        xytext=(dx, dy),
        textcoords="offset points",
        fontsize=6.7,
        ha="left",
        va="center"
    )

ax3.set_xlabel(r"MACs ($\times 10^5$)")
ax3.set_ylabel(r"$E_{\mathrm{joint}}$")
ax3.set_xlim(0.35, 6.4)
ax3.set_ylim(0.84, max(normalized_rmse) * 1.08)
ax3.grid(True, linestyle="--", alpha=0.35)

cax = fig.add_axes([0.835, 0.150, 0.018, 0.335])
cbar = fig.colorbar(scatter, cax=cax)
cbar.set_label("RAM (KiB)")
cbar.ax.tick_params(labelsize=7)

fig.text(0.257, 0.525, "(a) Distance and angle RMSE", ha="center", va="center", fontsize=8.2)
fig.text(0.767, 0.525, "(b) Relative RMSE increase", ha="center", va="center", fontsize=8.2)
fig.text(0.477, 0.042, "(c) Accuracy-complexity trade-off", ha="center", va="center", fontsize=8.2)

# =========================
# Save
# =========================
out_dir = Path(".")
fig.savefig(out_dir / "Fig5_ablation_updated_v3.png", dpi=600, bbox_inches="tight")
fig.savefig(out_dir / "Fig5_ablation_updated_v3.pdf", bbox_inches="tight")
fig.savefig(out_dir / "Fig5_ablation_updated_v3.svg", bbox_inches="tight")
fig.savefig(out_dir / "Fig5_ablation_updated_v3.tiff", dpi=600, bbox_inches="tight")

plt.show()
