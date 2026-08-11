"""Draw the publication-ready TAROT architecture overview (Fig. 1).

The artwork uses a wide, two-level architecture layout: a compact system
pipeline across the top, three dashed detail groups in the middle, and the
closed execution loop at the bottom.  PDF/SVG remain vector based; PNG/TIFF
are generated as review and submission fallbacks.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT_DIR = ROOT / "paper" / "figures"

# Okabe--Ito-derived, color-vision-deficiency-friendly semantic palette.
INK = "#26323B"
MUTED = "#66737C"
HAIR = "#AAB4BB"
CONTROL = "#009E73"
PARTNER = "#0072B2"
GATE = "#E69F00"
SAFETY = "#CC79A7"
CAUTION = "#D55E00"
LIGHT_GREEN = "#EAF7F2"
LIGHT_BLUE = "#EAF3F8"
LIGHT_AMBER = "#FFF3D6"
LIGHT_PURPLE = "#F7ECF5"
LIGHT_GRAY = "#F2F5F6"
PANEL = "#FBFCFC"


def setup_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "STIX Two Text", "DejaVu Serif"],
            "font.size": 7.2,
            "mathtext.fontset": "stix",
            "axes.linewidth": 0.7,
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.025,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )


def arrow(
    ax: plt.Axes,
    start: tuple[float, float],
    end: tuple[float, float],
    *,
    color: str = INK,
    width: float = 0.9,
    dashed: bool = False,
    curved: float = 0.0,
    scale: float = 8.0,
    zorder: int = 6,
) -> None:
    ax.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=scale,
            linewidth=width,
            linestyle=(0, (3.0, 2.0)) if dashed else "-",
            color=color,
            connectionstyle=f"arc3,rad={curved}",
            shrinkA=1.0,
            shrinkB=1.0,
            zorder=zorder,
            clip_on=False,
        )
    )


def rounded_box(
    ax: plt.Axes,
    xy: tuple[float, float],
    wh: tuple[float, float],
    *,
    face: str,
    edge: str,
    linewidth: float = 0.9,
    radius: float = 0.012,
    dashed: bool = False,
    zorder: int = 2,
) -> FancyBboxPatch:
    x, y = xy
    w, h = wh
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle=f"round,pad=0.004,rounding_size={radius}",
        facecolor=face,
        edgecolor=edge,
        linewidth=linewidth,
        linestyle=(0, (4.0, 2.4)) if dashed else "-",
        zorder=zorder,
    )
    ax.add_patch(patch)
    return patch


def top_module(
    ax: plt.Axes,
    xy: tuple[float, float],
    wh: tuple[float, float],
    step: str,
    title: str,
    *,
    face: str,
    edge: str,
) -> None:
    x, y = xy
    w, h = wh
    rounded_box(ax, xy, wh, face=face, edge=edge, linewidth=1.0, radius=0.010, zorder=7)
    ax.text(x + 0.010, y + h - 0.018, step, color=edge, fontsize=5.9, fontweight="bold", ha="left", va="top", zorder=8)
    ax.text(x + w / 2, y + 0.038, title, color=INK, fontsize=6.8, fontweight="bold", ha="center", va="center", linespacing=1.0, zorder=8)


def detail_box(
    ax: plt.Axes,
    xy: tuple[float, float],
    wh: tuple[float, float],
    title: str,
    *,
    subtitle: str | None = None,
    face: str,
    edge: str,
    title_size: float = 7.0,
) -> None:
    x, y = xy
    w, h = wh
    rounded_box(ax, xy, wh, face=face, edge=edge, linewidth=0.9, radius=0.010, zorder=3)
    ax.text(x + w / 2, y + h - 0.028, title, color=INK, fontsize=title_size, fontweight="bold", ha="center", va="top", zorder=5)
    if subtitle:
        ax.text(x + w / 2, y + 0.032, subtitle, color=MUTED, fontsize=5.9, ha="center", va="bottom", linespacing=1.05, zorder=5)


def group_panel(
    ax: plt.Axes,
    xy: tuple[float, float],
    wh: tuple[float, float],
    label: str,
    heading: str,
    accent: str,
) -> None:
    x, y = xy
    w, h = wh
    rounded_box(ax, xy, wh, face=PANEL, edge=HAIR, linewidth=0.9, radius=0.010, dashed=True, zorder=0)
    ax.text(x + 0.012, y + h - 0.020, label, color=accent, fontsize=8.2, fontweight="bold", ha="left", va="top", zorder=5)
    ax.text(x + 0.042, y + h - 0.020, heading, color=INK, fontsize=7.8, fontweight="bold", ha="left", va="top", zorder=5)
    ax.plot([x + 0.012, x + w - 0.012], [y + h - 0.058, y + h - 0.058], color=accent, alpha=0.48, linewidth=0.9, zorder=2)


def draw_top_pipeline(ax: plt.Axes) -> None:
    ax.text(0.020, 0.972, "SYSTEM DECISION PIPELINE", color=MUTED, fontsize=6.1, fontweight="bold", ha="left", va="top")
    specs = [
        (0.020, 0.118, "1  STATE", "open-team\nscene", LIGHT_GREEN, CONTROL),
        (0.174, 0.154, "2  INFER", "motion +\nbelief", LIGHT_BLUE, PARTNER),
        (0.365, 0.111, "3  TEST", "risk\ngate", LIGHT_AMBER, GATE),
        (0.514, 0.116, "4  ADAPT", "role\nevent", LIGHT_PURPLE, SAFETY),
        (0.668, 0.120, "5  COMMAND", "candidate\n$u_{i,t}$", LIGHT_BLUE, PARTNER),
        (0.826, 0.153, "6  SAFEGUARD", "predictive safety\nprojection", LIGHT_GREEN, CONTROL),
    ]
    y, h = 0.842, 0.104
    for x, w, step, title, face, edge in specs:
        top_module(ax, (x, y), (w, h), step, title, face=face, edge=edge)
    for (x, w, *_), (nx, *_rest) in zip(specs[:-1], specs[1:]):
        arrow(ax, (x + w, y + h / 2), (nx, y + h / 2), color=CONTROL, width=1.15, scale=8.2, zorder=9)

    # Light vertical associations connect the compact overview to its detail groups.
    for x0, x1, color in [(0.079, 0.079, CONTROL), (0.251, 0.322, PARTNER), (0.420, 0.485, GATE), (0.572, 0.616, SAFETY), (0.902, 0.862, CONTROL)]:
        ax.plot([x0, x1], [0.838, 0.783], color=color, alpha=0.35, linewidth=0.75, linestyle=(0, (2.5, 2.2)), zorder=1)


def draw_scene(ax: plt.Axes) -> None:
    x, y, w, h = 0.015, 0.176, 0.220, 0.604
    group_panel(ax, (x, y), (w, h), "(a)", "Open-team state", CONTROL)

    target = (0.126, 0.435)
    controlled = [(0.050, 0.300), (0.055, 0.500)]
    partners = [(0.202, 0.350), (0.198, 0.560)]
    obstacles = [(0.082, 0.595, 62), (0.176, 0.622, 82), (0.174, 0.260, 118), (0.112, 0.355, 43)]

    for ox, oy, size in obstacles:
        ax.scatter(ox, oy, s=size, marker="o", facecolor="#D9DEE1", edgecolor="#87939A", linewidth=0.65, zorder=2)
    for cx, cy in controlled:
        ax.plot([cx, target[0]], [cy, target[1]], color=CONTROL, alpha=0.28, linewidth=0.65, linestyle=(0, (2, 2)), zorder=1)
        ax.scatter(cx, cy, s=31, marker="o", color=CONTROL, edgecolor="white", linewidth=0.45, zorder=6)
        arrow(ax, (cx + 0.003, cy + 0.008), (cx + 0.050, cy + 0.072), color=CONTROL, width=1.15, scale=7.5)
    for px, py in partners:
        ax.scatter(px, py, s=30, marker="s", color=PARTNER, edgecolor="white", linewidth=0.45, zorder=6)
        arrow(ax, (px - 0.004, py + 0.004), (px - 0.054, py + 0.035), color=PARTNER, width=1.05, scale=7.0)

    ax.scatter(*target, s=75, marker="*", color=INK, zorder=7)
    arrow(ax, (target[0] + 0.006, target[1] - 0.003), (target[0] + 0.045, target[1] - 0.080), color=INK, width=0.8, scale=6.5)
    ax.text(target[0] + 0.046, target[1] - 0.090, "target motion", color=MUTED, fontsize=5.6, ha="center", va="top")

    # Compact legend and observation state, kept inside the group frame.
    ly = 0.218
    ax.scatter(0.035, ly, s=20, marker="o", color=CONTROL, zorder=5)
    ax.text(0.050, ly, "controlled", color=CONTROL, fontsize=5.8, va="center")
    ax.scatter(0.103, ly, s=20, marker="s", color=PARTNER, zorder=5)
    ax.text(0.118, ly, "partner", color=PARTNER, fontsize=5.8, va="center")
    ax.scatter(0.170, ly, s=22, marker="*", color=INK, zorder=5)
    ax.text(0.184, ly, "target", color=INK, fontsize=5.8, va="center")

    rounded_box(ax, (0.035, 0.655), (0.180, 0.050), face="white", edge=HAIR, linewidth=0.7, radius=0.008, zorder=3)
    ax.text(0.125, 0.680, r"state $s_t=\{p_t^i,p_t^j,q_t,o_{1:O}\}$", color=MUTED, fontsize=6.0, ha="center", va="center", zorder=5)


def draw_adaptation(ax: plt.Axes) -> None:
    x, y, w, h = 0.250, 0.176, 0.452, 0.604
    group_panel(ax, (x, y), (w, h), "(b)", "Belief, risk gate, and role event", PARTNER)

    # Motion evidence and recursive categorical belief.
    detail_box(ax, (0.270, 0.345), (0.126, 0.320), "Recursive belief", face=LIGHT_BLUE, edge=PARTNER)
    ax.text(0.333, 0.592, r"$b_t(\tau)\propto b_{t-1}(\tau)^\eta L_t(\tau)$", color=INK, fontsize=6.1, ha="center")
    bars = [0.032, 0.078, 0.122, 0.064, 0.026]
    for idx, height in enumerate(bars):
        ax.add_patch(Rectangle((0.286 + idx * 0.020, 0.426), 0.013, height, facecolor=CONTROL if idx == 2 else PARTNER, edgecolor="none", alpha=0.82, zorder=5))
    ax.plot([0.282, 0.385], [0.425, 0.425], color=MUTED, linewidth=0.55, zorder=5)
    ax.text(0.333, 0.387, "chase · block · flank", color=PARTNER, fontsize=5.8, ha="center")
    ax.text(0.333, 0.359, "lazy · noisy", color=MUTED, fontsize=5.8, ha="center")

    # Confidence-clearance gate.
    detail_box(ax, (0.430, 0.435), (0.112, 0.170), "Risk gate", face=LIGHT_AMBER, edge=GATE)
    ax.text(0.486, 0.522, r"$\gamma_t=\max_\tau b_t(\tau)$", color=INK, fontsize=6.0, ha="center")
    ax.text(0.486, 0.484, r"$d_t^{\min}$ vs. $\alpha r_{\rm near}$", color=GATE, fontsize=6.0, ha="center")
    ax.plot([0.452, 0.520], [0.460, 0.460], color=GATE, linewidth=1.5, solid_capstyle="round", zorder=5)
    ax.scatter(0.494, 0.460, s=14, color=INK, edgecolor="white", linewidth=0.4, zorder=6)

    # Trusted path retains the event-triggered role state; the lower path is explicit recovery.
    detail_box(ax, (0.577, 0.514), (0.105, 0.143), "Role event", face=LIGHT_PURPLE, edge=SAFETY, title_size=6.8)
    ax.scatter(0.600, 0.568, s=56, marker="o", facecolors="white", edgecolors=SAFETY, linewidths=0.8, zorder=5)
    ax.plot([0.600, 0.600], [0.568, 0.589], color=SAFETY, linewidth=0.7, zorder=6)
    ax.plot([0.600, 0.616], [0.568, 0.558], color=SAFETY, linewidth=0.7, zorder=6)
    ax.text(0.639, 0.567, "C   I   F", color=SAFETY, fontsize=6.3, fontweight="bold", ha="center")
    ax.text(0.6295, 0.532, "chase / intercept / flank", color=MUTED, fontsize=5.2, ha="center", va="center")

    detail_box(ax, (0.577, 0.303), (0.105, 0.130), "Fallback", subtitle="direct target pursuit", face=LIGHT_GRAY, edge=MUTED, title_size=6.8)

    arrow(ax, (0.396, 0.520), (0.430, 0.520), color=INK, width=1.0)
    arrow(ax, (0.542, 0.550), (0.577, 0.585), color=CONTROL, width=1.15)
    ax.text(0.558, 0.610, "trusted + clear", color=CONTROL, fontsize=5.7, ha="center", va="bottom")
    arrow(ax, (0.486, 0.435), (0.577, 0.368), color=GATE, width=1.15, curved=0.12)
    ax.text(0.510, 0.384, "uncertain\nor close", color=GATE, fontsize=5.7, ha="center", va="top", linespacing=0.95)

    # Both branches merge into the same candidate command leaving panel (b).
    ax.plot([0.690, 0.690], [0.368, 0.585], color=PARTNER, linewidth=0.9, zorder=4)
    arrow(ax, (0.682, 0.585), (0.690, 0.585), color=SAFETY, width=0.9, scale=6.5)
    arrow(ax, (0.682, 0.368), (0.690, 0.368), color=MUTED, width=0.9, scale=6.5)
    arrow(ax, (0.690, 0.477), (0.716, 0.477), color=PARTNER, width=1.25, scale=8.5)
    ax.text(0.690, 0.690, "24-step event clock; role state persists between events", color=MUTED, fontsize=5.9, ha="right", va="center")


def draw_safeguard(ax: plt.Axes) -> None:
    x, y, w, h = 0.717, 0.176, 0.268, 0.604
    group_panel(ax, (x, y), (w, h), "(c)", "Predictive safety projection", SAFETY)

    start = (0.752, 0.372)
    candidate_end = (0.875, 0.465)
    corrected_end = (0.842, 0.635)
    obstacle = (0.947, 0.495)

    ax.scatter(*obstacle, s=700, marker="o", facecolors="none", edgecolors=SAFETY, linewidths=0.85, linestyle=(0, (3, 2)), zorder=1)
    ax.scatter(*obstacle, s=220, marker="o", facecolor="#D9DEE1", edgecolor="#87939A", linewidth=0.8, zorder=4)
    ax.text(obstacle[0], obstacle[1], "obstacle", color=MUTED, fontsize=5.3, ha="center", va="center", zorder=6)

    ax.scatter(*start, s=39, marker="o", color=CONTROL, edgecolor="white", linewidth=0.45, zorder=7)
    ax.text(start[0], start[1] - 0.040, r"$p_t^i$", color=CONTROL, fontsize=6.2, ha="center")
    arrow(ax, start, candidate_end, color=GATE, width=1.15, dashed=True, scale=8.0)
    ax.text(0.808, 0.443, r"candidate $u_{i,t}$", color=GATE, fontsize=5.9, ha="center")
    ax.scatter(*candidate_end, s=34, marker="x", color=CAUTION, linewidth=1.15, zorder=8)
    ax.text(candidate_end[0] - 0.006, candidate_end[1] - 0.048, r"$p_t^i+\Delta t\,u_{i,t}$", color=CAUTION, fontsize=5.7, ha="center")

    arrow(ax, start, corrected_end, color=CONTROL, width=1.55, scale=9.0)
    ax.text(0.800, 0.594, r"executed $\widetilde u_{i,t}$", color=CONTROL, fontsize=6.2, fontweight="bold", ha="center")

    # Explicit outward/tangential correction components and one-step clearance.
    arrow(ax, candidate_end, (0.835, 0.550), color=SAFETY, width=0.9, scale=6.5)
    arrow(ax, candidate_end, (0.875, 0.590), color=SAFETY, width=0.9, scale=6.5)
    ax.text(0.832, 0.560, "outward", color=SAFETY, fontsize=5.4, ha="right")
    ax.text(0.884, 0.582, "tangential", color=SAFETY, fontsize=5.4, ha="left")
    ax.plot([candidate_end[0], obstacle[0]], [candidate_end[1], obstacle[1]], color=MUTED, linewidth=0.7, linestyle=(0, (2, 2)), zorder=3)
    ax.text(0.913, 0.454, r"$d_{io}^1$", color=MUTED, fontsize=5.8, ha="center")

    # The three safeguard operations echo the high-level pipeline without duplicating prose.
    steps = [
        (0.735, "predict", LIGHT_BLUE, PARTNER),
        (0.818, "check", LIGHT_AMBER, GATE),
        (0.901, "correct", LIGHT_GREEN, CONTROL),
    ]
    for sx, label, face, edge in steps:
        rounded_box(ax, (sx, 0.248), (0.068, 0.055), face=face, edge=edge, linewidth=0.8, radius=0.010, zorder=3)
        ax.text(sx + 0.034, 0.276, label, color=edge, fontsize=5.8, fontweight="bold", ha="center", va="center", zorder=5)
    arrow(ax, (0.803, 0.276), (0.818, 0.276), color=MUTED, width=0.7, scale=5.5)
    arrow(ax, (0.886, 0.276), (0.901, 0.276), color=MUTED, width=0.7, scale=5.5)
    ax.text(0.852, 0.220, r"$\widetilde u_{i,t}=\mathcal{P}_{\rm geom}(u_{i,t};o_{1:O},p_t,\beta)$", color=INK, fontsize=6.1, ha="center")
    ax.text(0.852, 0.179, "empirical one-step guard; no reachability claim", color=CAUTION, fontsize=5.1, ha="center", va="bottom")


def draw_closed_loop(ax: plt.Axes) -> None:
    # Reference-style bottom path separates the command flow from internal model arrows.
    ax.plot([0.850, 0.850], [0.176, 0.104], color=CONTROL, linewidth=1.1, zorder=2)
    ax.plot([0.850, 0.070], [0.104, 0.104], color=CONTROL, linewidth=1.1, zorder=2)
    arrow(ax, (0.070, 0.104), (0.070, 0.176), color=CONTROL, width=1.1, scale=8.0)
    rounded_box(ax, (0.322, 0.072), (0.386, 0.060), face="white", edge=CONTROL, linewidth=0.8, radius=0.012, zorder=3)
    ax.text(0.515, 0.102, r"closed loop: execute $\widetilde u_{i,t}$  $\longrightarrow$  observe $s_{t+1}$  $\longrightarrow$  update $b_{t+1}$", color=CONTROL, fontsize=6.4, fontweight="bold", ha="center", va="center", zorder=5)

    # A restrained key documents the semantic color encoding.
    key_y = 0.035
    entries = [
        (0.260, CONTROL, "main / executed path"),
        (0.455, PARTNER, "belief / candidate"),
        (0.620, GATE, "risk / recovery"),
        (0.775, SAFETY, "safety correction"),
    ]
    for kx, color, label in entries:
        ax.plot([kx, kx + 0.030], [key_y, key_y], color=color, linewidth=1.8, solid_capstyle="round")
        ax.text(kx + 0.036, key_y, label, color=MUTED, fontsize=5.6, ha="left", va="center")


def build_fig1(out_dir: Path | str = DEFAULT_OUT_DIR) -> dict[str, Path]:
    setup_style()
    output = Path(out_dir)
    output.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(7.2, 3.35))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    draw_top_pipeline(ax)
    draw_scene(ax)
    draw_adaptation(ax)
    draw_safeguard(ax)
    draw_closed_loop(ax)
    fig.subplots_adjust(left=0.006, right=0.996, top=0.995, bottom=0.010)

    stem = output / "jksucis_framework"
    pdf_metadata = {
        "Title": "TAROT architecture: belief, risk gating, role events, and predictive safety projection",
        "Author": "Anonymous",
    }
    svg_metadata = {
        "Title": "TAROT architecture: belief, risk gating, role events, and predictive safety projection",
        "Description": "Publication figure showing TAROT's system pipeline, open-team state, teammate belief and risk-gated role events, predictive geometric correction, and closed execution loop.",
    }
    paths = {
        "pdf": stem.with_suffix(".pdf"),
        "svg": stem.with_suffix(".svg"),
        "png": stem.with_suffix(".png"),
        "tiff": stem.with_suffix(".tiff"),
    }
    fig.savefig(paths["pdf"], metadata=pdf_metadata)
    fig.savefig(paths["svg"], metadata=svg_metadata)
    fig.savefig(paths["png"], dpi=300)
    fig.savefig(paths["tiff"], dpi=300, pil_kwargs={"compression": "tiff_lzw"})
    plt.close(fig)
    return paths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR, help="Directory for PDF/SVG/PNG/TIFF outputs")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    paths = build_fig1(args.out_dir)
    for kind, path in paths.items():
        print(f"{kind}: {path}")


if __name__ == "__main__":
    main()
