import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# Data
# ============================================================

categories = [
    "Pitch sweep",
    "Roll sweep",
    "Pitch sine",
    "Roll sine",
    "Two-axis sine(0)",
    "Two-axis sine(90)",
    "Pitch-roll coupled",
    "Diagonal same",
    "Diagonal opposite",
]

category_labels = [
    "Pitch\nsweep",
    "Roll\nsweep",
    "Pitch\nsine",
    "Roll\nsine",
    "Sine\n0 deg",
    "Sine\n90 deg",
    "Coupled",
    "Diag.\nsame",
    "Diag.\nopp.",
]

# Distance RMSE, mm
d1 = np.array([0.054, 0.057, 0.068, 0.057, 0.060, 0.059, 0.049, 0.067, 0.051])
d2 = np.array([0.042, 0.055, 0.050, 0.042, 0.050, 0.045, 0.035, 0.046, 0.035])
d3 = np.array([0.046, 0.052, 0.053, 0.060, 0.057, 0.042, 0.045, 0.045, 0.049])
d4 = np.array([0.053, 0.045, 0.057, 0.051, 0.063, 0.048, 0.045, 0.054, 0.053])
d_mean = np.array([0.049, 0.052, 0.057, 0.052, 0.058, 0.049, 0.044, 0.053, 0.047])

# Angle RMSE, deg
pitch = np.array([0.121, 0.079, 0.131, 0.114, 0.118, 0.118, 0.122, 0.073, 0.118])
roll = np.array([0.047, 0.080, 0.061, 0.079, 0.082, 0.078, 0.074, 0.066, 0.051])
angle_mean = np.array([0.084, 0.079, 0.096, 0.096, 0.100, 0.098, 0.098, 0.069, 0.085])


# ============================================================
# Plot helpers
# ============================================================

def closed(values: np.ndarray) -> np.ndarray:
    return np.r_[values, values[0]]


def style_polar_axis(ax, angles, labels, r_ticks, r_lim):
    ax.set_theta_offset(np.pi / 2.0)
    ax.set_theta_direction(-1)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels)
    ax.tick_params(axis="x", pad=6, labelsize=7.0)
    ax.set_yticks(r_ticks)
    ax.set_yticklabels([f"{tick:.2f}" for tick in r_ticks], color="0.35", fontsize=7.0)
    ax.set_ylim(*r_lim)
    ax.set_rlabel_position(48)
    ax.grid(True, linestyle="--", linewidth=0.55, color="0.75", alpha=0.75)
    ax.spines["polar"].set_linewidth(0.85)


# ============================================================
# Figure style
# ============================================================

plt.rcParams.update({
    "font.family": "Times New Roman",
    "font.size": 8,
    "axes.labelsize": 8,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "legend.fontsize": 7,
    "axes.linewidth": 0.85,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "svg.fonttype": "none",
    "figure.dpi": 300,
    "savefig.dpi": 600,
})

blue = "#1f77b4"
orange = "#ff7f0e"
green = "#2ca02c"
purple = "#9467bd"
red = "#d62728"
teal = "#008b8b"

num_axes = len(categories)
angles = np.linspace(0, 2 * np.pi, num_axes, endpoint=False)
angles = np.r_[angles, angles[0]]

fig = plt.figure(figsize=(7.16, 3.25))
gs = fig.add_gridspec(1, 2, left=0.07, right=0.98, bottom=0.29, top=0.89, wspace=0.38)


# ============================================================
# (a) Distance RMSE
# ============================================================

ax1 = fig.add_subplot(gs[0, 0], projection="polar")
style_polar_axis(
    ax1,
    angles,
    category_labels,
    r_ticks=[0.02, 0.04, 0.06, 0.08],
    r_lim=(0.0, 0.08),
)

line_kwargs = dict(linewidth=0.9, linestyle="--", alpha=0.72)
ax1.plot(angles, closed(d1), color=blue, label="d1", **line_kwargs)
ax1.plot(angles, closed(d2), color=orange, label="d2", **line_kwargs)
ax1.plot(angles, closed(d3), color=green, label="d3", **line_kwargs)
ax1.plot(angles, closed(d4), color=purple, label="d4", **line_kwargs)
ax1.plot(angles, closed(d_mean), color=red, linewidth=1.9, marker="o", markersize=2.6, label="Mean")
ax1.fill(angles, closed(d_mean), color=red, alpha=0.12)

ax1.legend(
    loc="upper center",
    bbox_to_anchor=(0.5, -0.11),
    frameon=False,
    ncol=5,
    fontsize=6.8,
    handlelength=1.6,
    columnspacing=0.7,
)


# ============================================================
# (b) Angle RMSE
# ============================================================

ax2 = fig.add_subplot(gs[0, 1], projection="polar")
style_polar_axis(
    ax2,
    angles,
    category_labels,
    r_ticks=[0.05, 0.10, 0.15],
    r_lim=(0.0, 0.15),
)

ax2.plot(angles, closed(pitch), color=blue, linewidth=0.95, linestyle="--", alpha=0.78, label="Pitch")
ax2.plot(angles, closed(roll), color=orange, linewidth=0.95, linestyle="--", alpha=0.78, label="Roll")
ax2.plot(angles, closed(angle_mean), color=teal, linewidth=1.9, marker="o", markersize=2.6, label="Mean")
ax2.fill(angles, closed(angle_mean), color=teal, alpha=0.12)

ax2.legend(
    loc="upper center",
    bbox_to_anchor=(0.5, -0.11),
    frameon=False,
    ncol=3,
    fontsize=6.8,
    handlelength=1.6,
    columnspacing=0.8,
)


pos1 = ax1.get_position()
pos2 = ax2.get_position()
caption_y = 0.135
fig.text(pos1.x0 + pos1.width / 2, caption_y, "(a) Distance RMSE", ha="center", va="center", fontsize=8.5)
fig.text(pos2.x0 + pos2.width / 2, caption_y, "(b) Angle RMSE", ha="center", va="center", fontsize=8.5)


# ============================================================
# Save
# ============================================================

out_base = "Fig_dynamic_trajectory_radar_TAES"
fig.savefig(out_base + ".pdf", bbox_inches="tight")
fig.savefig(out_base + ".svg", bbox_inches="tight")
fig.savefig(out_base + ".png", bbox_inches="tight")
fig.savefig(out_base + ".tif", bbox_inches="tight")

plt.show()
