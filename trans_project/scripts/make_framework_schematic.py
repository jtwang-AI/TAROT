from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np


OUT_DIR = Path("paper/figures")


def setup_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 7,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "pdf.fonttype": 42,
            "svg.fonttype": "none",
        }
    )


def arrow(ax, start, end, color="#3d3d3d", lw=0.9, style="->") -> None:
    ax.annotate(
        "",
        xy=end,
        xytext=start,
        arrowprops=dict(arrowstyle=style, lw=lw, color=color, shrinkA=1.5, shrinkB=1.5),
    )


def rounded_box(ax, xy, wh, text, face="#f5f7f8", edge="#4a4a4a", fontsize=7.0, weight="normal") -> None:
    x, y = xy
    w, h = wh
    box = patches.FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.012,rounding_size=0.018",
        linewidth=0.8,
        edgecolor=edge,
        facecolor=face,
    )
    ax.add_patch(box)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fontsize, fontweight=weight)


def panel_label(ax, label: str) -> None:
    ax.text(-0.04, 1.04, label, transform=ax.transAxes, fontsize=8.5, fontweight="bold", va="top")


def draw_scene(ax) -> None:
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title("Open-team pursuit state", fontsize=7.5, pad=4)

    for x, y, r in [(0.28, 0.72, 0.075), (0.62, 0.68, 0.055), (0.48, 0.30, 0.065), (0.78, 0.40, 0.045)]:
        ax.add_patch(patches.Circle((x, y), r, facecolor="#d9d9d9", edgecolor="#9a9a9a", linewidth=0.5))

    learners = np.array([[0.18, 0.20], [0.16, 0.50]])
    partners = np.array([[0.80, 0.26], [0.82, 0.56]])
    target = np.array([0.50, 0.50])

    for xy in learners:
        ax.scatter(*xy, s=34, marker="o", color="#0f766e", edgecolor="white", linewidth=0.4, zorder=4)
        arrow(ax, xy, target * 0.65 + xy * 0.35, color="#0f766e", lw=0.8)
    for xy in partners:
        ax.scatter(*xy, s=34, marker="s", color="#2563eb", edgecolor="white", linewidth=0.4, zorder=4)
        arrow(ax, xy, target * 0.55 + xy * 0.45, color="#2563eb", lw=0.8)
    ax.scatter(*target, s=48, marker="*", color="#111111", zorder=5)
    ax.text(0.06, 0.92, "unknown partners", color="#2563eb", fontsize=5.8)
    ax.text(0.06, 0.84, "controlled drones", color="#0f766e", fontsize=5.8)
    ax.text(0.06, 0.76, "moving target + clutter", color="#555555", fontsize=5.8)


def draw_pipeline(ax) -> None:
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.set_title("Risk-gated teammate belief", fontsize=7.5, pad=4)

    rounded_box(ax, (0.02, 0.64), (0.22, 0.18), "State\nabstraction", "#eef4f7")
    rounded_box(ax, (0.30, 0.64), (0.22, 0.18), "Partner\nbelief", "#edf7f1")
    rounded_box(ax, (0.58, 0.64), (0.17, 0.18), "Risk\ngate", "#fff5e6")
    rounded_box(ax, (0.81, 0.64), (0.17, 0.18), "Role\nplanner", "#edf7f1")
    rounded_box(ax, (0.58, 0.22), (0.20, 0.18), "Safety\ngovernor", "#f4f0fa")
    rounded_box(ax, (0.30, 0.22), (0.20, 0.18), "Fallback\nrecovery", "#f7f7f7")

    arrow(ax, (0.24, 0.73), (0.30, 0.73))
    arrow(ax, (0.52, 0.73), (0.58, 0.73))
    arrow(ax, (0.75, 0.73), (0.81, 0.73))
    arrow(ax, (0.90, 0.64), (0.71, 0.40))
    arrow(ax, (0.58, 0.66), (0.50, 0.40), color="#9a6b21")
    arrow(ax, (0.50, 0.31), (0.58, 0.31))
    arrow(ax, (0.68, 0.22), (0.68, 0.11))
    ax.text(0.68, 0.06, "safe velocity commands", ha="center", va="center", fontsize=6.2)

    ax.text(0.41, 0.58, r"$p(\tau \mid h_t)$", ha="center", fontsize=6.4, color="#2d6a4f")
    ax.text(0.66, 0.58, "clearance\nconfidence", ha="center", fontsize=5.8, color="#9a6b21")
    ax.text(0.47, 0.45, "low trust or high risk", ha="center", fontsize=5.6, color="#666666")
    ax.text(0.88, 0.45, "chase / flank /\nintercept", ha="center", fontsize=5.8, color="#2d6a4f")


def draw_outputs(ax) -> None:
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.set_title("Claims evaluated", fontsize=7.5, pad=4)
    bars = [
        ("task success", 0.78, "#62b097"),
        ("collisions", 0.36, "#d88c7a"),
        ("shield burden", 0.28, "#9185c6"),
        ("role switches", 0.44, "#b0b0b0"),
    ]
    for i, (label, width, color) in enumerate(bars):
        y = 0.76 - i * 0.16
        ax.text(0.04, y + 0.025, label, ha="left", va="center", fontsize=6.2)
        ax.add_patch(patches.Rectangle((0.42, y), 0.48, 0.055, facecolor="#eeeeee", edgecolor="none"))
        ax.add_patch(patches.Rectangle((0.42, y), 0.48 * width, 0.055, facecolor=color, edgecolor="none"))
    rounded_box(ax, (0.12, 0.06), (0.76, 0.13), "Evidence: ablations,\nrobustness, trajectories", "#f7f7f7", fontsize=5.9)


def main() -> None:
    setup_style()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=(7.2, 2.45))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.05, 1.55, 1.0], wspace=0.18)
    axes = [fig.add_subplot(gs[0, i]) for i in range(3)]
    draw_scene(axes[0])
    draw_pipeline(axes[1])
    draw_outputs(axes[2])
    for ax, lab in zip(axes, "abc"):
        panel_label(ax, lab)
    for ext in ["pdf", "svg", "png", "tiff"]:
        path = OUT_DIR / f"framework_schematic.{ext}"
        if ext in {"png", "tiff"}:
            fig.savefig(path, dpi=600, bbox_inches="tight")
        else:
            fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {OUT_DIR / 'framework_schematic.pdf'}")


if __name__ == "__main__":
    main()
