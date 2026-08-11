from __future__ import annotations

"""Render real trajectories and generate the 3-D validation figure/table."""

import argparse
import json
from math import sqrt
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
import pybullet as bullet

from run_3d_validation import Obstacle3D, Scene3D, scenes


POLICIES = ["tarot_no_safety", "tarot_default", "tarot_tuned_cem", "flat_cem_tuned"]
LABELS = {
    "tarot_no_safety": "TAROT w/o projection",
    "tarot_default": "TAROT",
    "tarot_tuned_cem": "TAROT-CEM",
    "flat_cem_tuned": "Flat CEM",
}
COLORS = {
    "tarot_no_safety": "#B279A2",
    "tarot_default": "#2A9D8F",
    "tarot_tuned_cem": "#1B7F4B",
    "flat_cem_tuned": "#8E7CC3",
}


def setup_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.titlesize": 10.5,
            "axes.labelsize": 9.5,
            "xtick.labelsize": 8.5,
            "ytick.labelsize": 8.5,
            "legend.fontsize": 8.2,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 0.9,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
        }
    )


def wilson(values: pd.Series, z: float = 1.96) -> tuple[float, float, float]:
    n = len(values)
    successes = float(values.sum())
    p = successes / n
    denominator = 1.0 + z * z / n
    center = (p + z * z / (2.0 * n)) / denominator
    half = z * sqrt((p * (1.0 - p) + z * z / (4.0 * n)) / n) / denominator
    return p, center - half, center + half


def paired_effect(
    frame: pd.DataFrame,
    first: str,
    second: str,
    metric: str,
    reduction: bool = False,
    samples: int = 4000,
) -> tuple[float, float, float]:
    key = ["scene", "seed"]
    a = frame.loc[frame.policy == first, key + [metric]].rename(columns={metric: "a"})
    b = frame.loc[frame.policy == second, key + [metric]].rename(columns={metric: "b"})
    paired = a.merge(b, on=key, validate="one_to_one")
    diff = paired.b.to_numpy() - paired.a.to_numpy() if reduction else paired.a.to_numpy() - paired.b.to_numpy()
    rng = np.random.default_rng(20260716)
    boot = np.empty(samples)
    for start in range(0, samples, 100):
        stop = min(start + 100, samples)
        indices = rng.integers(0, len(diff), size=(stop - start, len(diff)))
        boot[start:stop] = diff[indices].mean(axis=1)
    return float(diff.mean()), float(np.quantile(boot, 0.025)), float(np.quantile(boot, 0.975))


def add_body(shape: int, position, color, orientation=(0, 0, 0, 1)) -> int:
    visual = bullet.createVisualShape(shape, radius=0.2, rgbaColor=color) if shape == bullet.GEOM_SPHERE else -1
    return bullet.createMultiBody(baseMass=0, baseVisualShapeIndex=visual, basePosition=position, baseOrientation=orientation)


def create_box(position, half_extents, color) -> int:
    visual = bullet.createVisualShape(bullet.GEOM_BOX, halfExtents=half_extents, rgbaColor=color)
    return bullet.createMultiBody(baseMass=0, baseVisualShapeIndex=visual, basePosition=position)


def create_cylinder(position, radius, height, color) -> int:
    visual = bullet.createVisualShape(bullet.GEOM_CYLINDER, radius=radius, length=height, rgbaColor=color)
    return bullet.createMultiBody(baseMass=0, baseVisualShapeIndex=visual, basePosition=position)


def create_sphere(position, radius, color) -> int:
    visual = bullet.createVisualShape(bullet.GEOM_SPHERE, radius=radius, rgbaColor=color)
    return bullet.createMultiBody(baseMass=0, baseVisualShapeIndex=visual, basePosition=position)


def create_scene_obstacle(obstacle: Obstacle3D) -> None:
    if obstacle.kind == "box":
        create_box((obstacle.x, obstacle.y, obstacle.height / 2), (obstacle.sx, obstacle.sy, obstacle.height / 2), obstacle.color)
        # A thin bright top face improves depth perception in headless renders.
        top = tuple(min(c + 0.10, 1.0) for c in obstacle.color[:3]) + (1.0,)
        create_box((obstacle.x, obstacle.y, obstacle.height + 0.04), (obstacle.sx, obstacle.sy, 0.04), top)
    elif obstacle.kind == "tree":
        create_cylinder((obstacle.x, obstacle.y, obstacle.height / 2), obstacle.sx * 0.44, obstacle.height, obstacle.color)
        leaf = (0.12, 0.40, 0.12, 1.0)
        create_sphere((obstacle.x, obstacle.y, obstacle.height * 0.80), obstacle.sx * 1.75, leaf)
        create_sphere((obstacle.x + 0.45, obstacle.y - 0.20, obstacle.height * 0.94), obstacle.sx * 1.25, (0.16, 0.48, 0.16, 1.0))
    elif obstacle.kind == "tank":
        create_cylinder((obstacle.x, obstacle.y, obstacle.height / 2), obstacle.sx, obstacle.height, obstacle.color)
        create_sphere((obstacle.x, obstacle.y, obstacle.height), obstacle.sx, tuple(min(c + 0.08, 1.0) for c in obstacle.color[:3]) + (1.0,))
    else:
        create_cylinder((obstacle.x, obstacle.y, obstacle.height / 2), obstacle.sx, obstacle.height, obstacle.color)


def create_drone(position: np.ndarray, color: tuple[float, float, float, float], scale: float = 1.0) -> None:
    x, y, z = map(float, position)
    create_sphere((x, y, z), 0.25 * scale, color)
    arm = tuple(max(c * 0.72, 0.02) for c in color[:3]) + (1.0,)
    create_box((x, y, z), (0.52 * scale, 0.055 * scale, 0.045 * scale), arm)
    create_box((x, y, z), (0.055 * scale, 0.52 * scale, 0.045 * scale), arm)
    for dx, dy in ((0.48, 0), (-0.48, 0), (0, 0.48), (0, -0.48)):
        create_cylinder((x + dx * scale, y + dy * scale, z + 0.02), 0.15 * scale, 0.035, (0.10, 0.10, 0.10, 1.0))


def choose_trajectory(scene_name: str, episodes: pd.DataFrame, trajectory_dir: Path) -> tuple[np.ndarray, int]:
    group = episodes.loc[
        (episodes.scene == scene_name) & (episodes.policy == "tarot_tuned_cem")
    ].copy()
    group["path"] = group.seed.map(lambda seed: trajectory_dir / f"{scene_name}__tarot_tuned_cem__{int(seed)}.npz")
    available = group.loc[group.path.map(Path.exists)]
    successful = available.loc[available.success == 1]
    candidates = successful if len(successful) else available.sort_values("steps", ascending=False)
    if not len(candidates):
        raise FileNotFoundError(f"No retained trajectory for {scene_name}")
    target_steps = float(candidates.steps.median())
    selected = candidates.iloc[(candidates.steps - target_steps).abs().argmin()]
    with np.load(selected.path) as bundle:
        trajectory = bundle["trajectory"]
    return trajectory, int(selected.seed)


def render_scene(scene: Scene3D, trajectory: np.ndarray, path: Path) -> None:
    client = bullet.connect(bullet.DIRECT)
    if client < 0:
        raise RuntimeError("PyBullet DIRECT connection failed")
    bullet.resetSimulation()
    bullet.configureDebugVisualizer(bullet.COV_ENABLE_SHADOWS, 1)
    create_box((20, 20, -0.16), (22, 22, 0.16), scene.ground_color)
    # Low boundary strips make the 40 x 40 m workspace legible without acting
    # as simulated obstacles.
    edge = (0.72, 0.72, 0.72, 1.0)
    create_box((20, -0.12, 0.08), (20, 0.12, 0.08), edge)
    create_box((20, 40.12, 0.08), (20, 0.12, 0.08), edge)
    create_box((-0.12, 20, 0.08), (0.12, 20, 0.08), edge)
    create_box((40.12, 20, 0.08), (0.12, 20, 0.08), edge)
    for obstacle in scene.obstacles:
        create_scene_obstacle(obstacle)

    trail_colors = [
        (0.05, 0.55, 0.43, 0.75), (0.15, 0.70, 0.55, 0.75),
        (0.12, 0.35, 0.85, 0.62), (0.30, 0.52, 0.95, 0.62),
        (0.82, 0.18, 0.13, 0.72),
    ]
    stride = max(len(trajectory) // 24, 1)
    for body_index in range(5):
        for point in trajectory[::stride, body_index]:
            create_sphere(point, 0.14 if body_index < 4 else 0.11, trail_colors[body_index])
    final = trajectory[-1]
    create_drone(final[0], (0.03, 0.58, 0.43, 1.0), 1.35)
    create_drone(final[1], (0.10, 0.72, 0.57, 1.0), 1.35)
    create_drone(final[2], (0.10, 0.32, 0.82, 1.0), 1.20)
    create_drone(final[3], (0.25, 0.50, 0.94, 1.0), 1.20)
    create_sphere(final[4], 0.52, (0.86, 0.12, 0.10, 1.0))

    camera = {
        "urban": (38, -36, 50),
        "forest": (40, -39, 42),
        "industrial": (38, -35, 56),
    }[scene.name]
    distance, pitch, yaw = camera
    view = bullet.computeViewMatrixFromYawPitchRoll(
        cameraTargetPosition=(20, 20, 3.0),
        distance=distance,
        yaw=yaw,
        pitch=pitch,
        roll=0,
        upAxisIndex=2,
    )
    projection = bullet.computeProjectionMatrixFOV(fov=52, aspect=16 / 9, nearVal=0.1, farVal=120)
    width, height, rgba, _, _ = bullet.getCameraImage(
        width=1280,
        height=720,
        viewMatrix=view,
        projectionMatrix=projection,
        shadow=1,
        lightDirection=(-0.4, -0.7, -1.0),
        renderer=bullet.ER_TINY_RENDERER,
    )
    image = np.asarray(rgba, dtype=np.uint8).reshape(height, width, 4)[:, :, :3].copy()
    sky = np.all(image > 247, axis=2)
    image[sky] = np.array([235, 243, 247], dtype=np.uint8)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(image).save(path)
    bullet.disconnect()


def make_table(frame: pd.DataFrame, output: Path) -> list[dict]:
    records = []
    lines = [
        r"\begin{table*}[t]",
        r"\caption{Dynamics-aware 3-D validation over 1,200 matched episodes. Collision denotes a contact involving a controlled drone. Successful steps are conditioned on success. Projection activation is normalized by controlled-drone time steps. Brackets show 95\% Wilson intervals for success.}",
        r"\label{tab:dynamics3d}",
        r"\centering",
        r"\small",
        r"\resizebox{\textwidth}{!}{%",
        r"\begin{tabular}{lccccc}",
        r"\toprule",
        r"Method & Success [95\% CI] $\uparrow$ & Collision $\downarrow$ & Timeout $\downarrow$ & Steps $\mid$ success $\downarrow$ & Activation (\%) $\downarrow$ \\",
        r"\midrule",
    ]
    for policy in POLICIES:
        group = frame.loc[frame.policy == policy]
        success, lo, hi = wilson(group.success)
        collision = float(group.collision_episode.mean())
        timeout = float(group.timeout.mean())
        successful = group.loc[group.success == 1, "steps"]
        steps = float(successful.mean()) if len(successful) else None
        activation = float(100 * group.activation_rate.mean())
        records.append(
            {
                "policy": policy,
                "episodes": len(group),
                "success": success,
                "success_ci": [lo, hi],
                "collision": collision,
                "timeout": timeout,
                "successful_steps": steps,
                "activation_percent": activation,
            }
        )
        activation_text = "--" if policy == "tarot_no_safety" else f"{activation:.1f}"
        steps_text = "--" if steps is None else f"{steps:.1f}"
        lines.append(
            f"{LABELS[policy]} & {success:.3f} [{lo:.3f}, {hi:.3f}] & {collision:.3f} & "
            f"{timeout:.3f} & {steps_text} & {activation_text} \\\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}", r"}", r"\end{table*}"])
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines) + "\n")
    return records


def grouped_bars(ax: plt.Axes, frame: pd.DataFrame, metric: str, ylabel: str, percent: bool = False) -> None:
    scene_order = ["urban", "forest", "industrial"]
    x = np.arange(len(scene_order), dtype=float)
    width = 0.19
    for offset, policy in enumerate(POLICIES):
        values = []
        errors = []
        for scene_name in scene_order:
            group = frame.loc[(frame.scene == scene_name) & (frame.policy == policy)]
            if metric in {"success", "collision_episode"}:
                mean, lo, hi = wilson(group[metric])
                values.append(mean)
                errors.append((max(mean - lo, 0.0), max(hi - mean, 0.0)))
            else:
                values.append(float(group[metric].mean()))
                errors.append((0.0, 0.0))
        values = np.asarray(values) * (100.0 if percent else 1.0)
        err = np.asarray(errors).T * (100.0 if percent else 1.0)
        ax.bar(
            x + (offset - 1.5) * width,
            values,
            width=width,
            color=COLORS[policy],
            edgecolor="white",
            linewidth=0.45,
            yerr=err if np.any(err) else None,
            capsize=2.2,
            error_kw={"lw": 0.8, "capthick": 0.8},
            label=LABELS[policy],
        )
    ax.set_xticks(x, ["Urban", "Forest", "Industrial"])
    ax.set_ylabel(ylabel)
    ax.grid(axis="y", color="#E5E5E5", lw=0.7)
    ax.set_axisbelow(True)


def make_composite(frame: pd.DataFrame, renders: dict[str, Path], output_dir: Path) -> None:
    fig = plt.figure(figsize=(14.2, 7.7))
    grid = fig.add_gridspec(2, 3, height_ratios=[1.08, 0.92], hspace=0.23, wspace=0.25)
    for index, scene_name in enumerate(("urban", "forest", "industrial")):
        ax = fig.add_subplot(grid[0, index])
        ax.imshow(Image.open(renders[scene_name]))
        ax.set_axis_off()
        label = scenes()[scene_name].label
        ax.text(0.02, 0.96, f"{chr(97 + index)}", transform=ax.transAxes, fontsize=13, fontweight="bold", va="top", color="white", bbox={"facecolor": "black", "alpha": 0.52, "pad": 2.5, "edgecolor": "none"})
        ax.text(0.98, 0.96, label, transform=ax.transAxes, fontsize=10.2, fontweight="bold", va="top", ha="right", color="white", bbox={"facecolor": "black", "alpha": 0.52, "pad": 3, "edgecolor": "none"})

    axes = [fig.add_subplot(grid[1, i]) for i in range(3)]
    grouped_bars(axes[0], frame, "success", "Episode rate")
    axes[0].set_ylim(0, 1.05)
    axes[0].set_title("d   Capture success (95% Wilson CI)", loc="left", fontweight="bold")
    grouped_bars(axes[1], frame, "collision_episode", "Episode rate")
    axes[1].set_ylim(0, 1.05)
    axes[1].set_title("e   Controlled-drone collision (95% Wilson CI)", loc="left", fontweight="bold")
    grouped_bars(axes[2], frame, "activation_rate", "Activation (%)", percent=True)
    axes[2].set_ylim(0, 105)
    axes[2].set_title("f   Projection activation", loc="left", fontweight="bold")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, frameon=False, bbox_to_anchor=(0.5, -0.005))
    fig.subplots_adjust(bottom=0.10, top=0.99, left=0.055, right=0.985)
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_dir / "dynamics3d_validation.pdf", bbox_inches="tight", facecolor="white", transparent=False)
    fig.savefig(output_dir / "dynamics3d_validation.svg", bbox_inches="tight", facecolor="white", transparent=False)
    png_path = output_dir / "dynamics3d_validation.png"
    # [ADAPTED: JKSUCIS treats this mixed render/vector plate as combination
    # artwork and requests at least 600 dpi for raster delivery.]
    output_dpi = 600
    fig.savefig(png_path, dpi=output_dpi, bbox_inches="tight", facecolor="white", transparent=False)
    rgba = Image.open(png_path).convert("RGBA")
    flattened = Image.new("RGB", rgba.size, "white")
    flattened.paste(rgba, mask=rgba.getchannel("A"))
    flattened.save(png_path, dpi=(output_dpi, output_dpi))
    flattened.save(
        output_dir / "dynamics3d_validation.tiff",
        compression="tiff_lzw",
        dpi=(output_dpi, output_dpi),
    )
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    setup_style()
    frame = pd.read_csv(args.results / "episodes.csv")
    if len(frame) != 1200:
        raise ValueError(f"Expected 1,200 formal episodes, found {len(frame)}")

    render_dir = args.out / "scene_renders"
    render_paths = {}
    selected = {}
    for scene_name, scene in scenes().items():
        trajectory, seed = choose_trajectory(scene_name, frame, args.results / "trajectories")
        path = render_dir / f"{scene_name}_tarot_cem.png"
        render_scene(scene, trajectory, path)
        render_paths[scene_name] = path
        selected[scene_name] = {"seed": seed, "steps": len(trajectory) - 1}

    figure_dir = args.out / "figures"
    make_composite(frame, render_paths, figure_dir)
    table_records = make_table(frame, args.out / "tables" / "jksucis_3d_table.tex")
    success_effect = paired_effect(frame, "tarot_tuned_cem", "flat_cem_tuned", "success")
    activation_reduction = tuple(
        100.0 * value
        for value in paired_effect(
            frame, "tarot_tuned_cem", "flat_cem_tuned", "activation_rate", reduction=True
        )
    )
    summary = {
        "selected_render_trajectories": selected,
        "table": table_records,
        "tarot_cem_minus_flat_success": success_effect,
        "tarot_cem_activation_reduction_percentage_points": activation_reduction,
    }
    (args.out / "dynamics3d_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
