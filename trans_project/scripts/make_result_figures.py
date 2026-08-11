from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


METHOD_NAMES = {
    "greedy": "Greedy",
    "safety_only": "Safety only",
    "tarot_no_safety": "TAROT w/o safety",
    "tarot_default": "TAROT",
    "tarot_tuned_cem": "TAROT-CEM",
    "flat_cem_tuned": "Flat CEM",
}


COLORS = {
    "greedy": "#7a7a7a",
    "safety_only": "#2c7fb8",
    "tarot_no_safety": "#d95f0e",
    "tarot_default": "#1b9e77",
    "tarot_tuned_cem": "#006d2c",
    "flat_cem_tuned": "#756bb1",
}


def rows(path: Path) -> list[dict]:
    with path.open() as f:
        return list(csv.DictReader(f))


def grouped_mean(data: list[dict], key_fields: tuple[str, ...], metrics: tuple[str, ...]) -> dict[tuple[str, ...], dict[str, float]]:
    groups = defaultdict(list)
    for row in data:
        groups[tuple(row[k] for k in key_fields)].append(row)
    out = {}
    for key, items in groups.items():
        out[key] = {m: sum(float(r[m]) for r in items) / len(items) for m in metrics}
        out[key]["n"] = float(len(items))
    return out


def setup_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8,
            "axes.titlesize": 9,
            "axes.labelsize": 8,
            "legend.fontsize": 7,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.dpi": 180,
            "savefig.bbox": "tight",
        }
    )


def make_pareto(opt_rows: list[dict], out_dir: Path) -> None:
    metrics = ("success", "collision_count", "safety_interventions")
    mean = grouped_mean(opt_rows, ("policy",), metrics)
    order = ["greedy", "tarot_no_safety", "safety_only", "tarot_default", "tarot_tuned_cem", "flat_cem_tuned"]
    fig, ax = plt.subplots(figsize=(3.35, 2.35))
    for policy in order:
        rec = mean[(policy,)]
        size = 95 + 120 * max(0.0, 1.0 - rec["collision_count"])
        ax.scatter(
            rec["safety_interventions"],
            rec["success"],
            s=size,
            color=COLORS[policy],
            edgecolor="black",
            linewidth=0.45,
            alpha=0.92,
            label=METHOD_NAMES[policy],
        )
        offsets = {
            "greedy": (5, -14),
            "tarot_no_safety": (5, 8),
            "safety_only": (6, -5),
            "tarot_default": (6, 12),
            "tarot_tuned_cem": (6, 4),
            "flat_cem_tuned": (-58, 6),
        }
        ax.annotate(
            METHOD_NAMES[policy],
            (rec["safety_interventions"], rec["success"]),
            xytext=offsets[policy],
            textcoords="offset points",
            fontsize=6.5,
        )
    ax.set_xlabel("Safety interventions per episode")
    ax.set_ylabel("Success rate")
    ax.set_title("Safety-efficiency operating points")
    ax.grid(True, color="#d9d9d9", linewidth=0.5, alpha=0.7)
    ax.set_xlim(-3, 70)
    ax.set_ylim(0.24, 0.60)
    fig.savefig(out_dir / "pareto_success_shield.pdf")
    fig.savefig(out_dir / "pareto_success_shield.png")
    plt.close(fig)


def budget_value(policy: str) -> float | None:
    if not policy.startswith("tarot_budget_"):
        return None
    return float(policy.removeprefix("tarot_budget_").replace("p", "."))


def make_budget(budget_rows: list[dict], out_dir: Path) -> None:
    metrics = ("success", "collision_count", "safety_interventions")
    mean = grouped_mean(budget_rows, ("policy",), metrics)
    budget_points = []
    for (policy,), rec in mean.items():
        b = budget_value(policy)
        if b is not None:
            budget_points.append((b, rec))
    budget_points.sort(key=lambda x: x[0])
    budgets = [b for b, _ in budget_points]
    success = [r["success"] for _, r in budget_points]
    collision = [r["collision_count"] for _, r in budget_points]
    shield = [r["safety_interventions"] for _, r in budget_points]

    fig, axes = plt.subplots(1, 2, figsize=(6.8, 2.2))
    axes[0].plot(budgets, success, marker="o", color="#1b9e77", label="Success")
    axes[0].plot(budgets, collision, marker="s", color="#d95f0e", label="Collision")
    axes[0].set_xlabel("Safety budget")
    axes[0].set_ylabel("Rate / count")
    axes[0].set_title("Task and risk")
    axes[0].grid(True, color="#d9d9d9", linewidth=0.5, alpha=0.7)
    axes[0].legend(frameon=False)

    axes[1].plot(budgets, shield, marker="^", color="#2c7fb8")
    axes[1].set_xlabel("Safety budget")
    axes[1].set_ylabel("Interventions / episode")
    axes[1].set_title("Shield usage")
    axes[1].grid(True, color="#d9d9d9", linewidth=0.5, alpha=0.7)

    fig.savefig(out_dir / "budget_sweep_trends.pdf")
    fig.savefig(out_dir / "budget_sweep_trends.png")
    plt.close(fig)


def make_scalability(scale_rows: list[dict], out_dir: Path) -> None:
    metrics = ("success", "collision_count", "runtime_ms")
    mean = grouped_mean(scale_rows, ("policy", "num_learners", "num_partners", "num_obstacles"), metrics)
    policies = ["greedy", "safety_only", "tarot_full"]
    labels = {"greedy": "Greedy", "safety_only": "Safety only", "tarot_full": "TAROT"}
    colors = {"greedy": "#7a7a7a", "safety_only": "#2c7fb8", "tarot_full": "#1b9e77"}
    fig, axes = plt.subplots(1, 2, figsize=(6.8, 2.2))
    for policy in policies:
        points = []
        for key, rec in mean.items():
            p, learners, partners, obstacles = key
            if p == policy:
                points.append((int(float(learners) + float(partners)), int(float(obstacles)), rec))
        points.sort(key=lambda x: x[0])
        drones = [p[0] for p in points]
        runtime = [p[2]["runtime_ms"] for p in points]
        success = [p[2]["success"] for p in points]
        axes[0].plot(drones, runtime, marker="o", color=colors[policy], label=labels[policy])
        axes[1].plot(drones, success, marker="o", color=colors[policy], label=labels[policy])
    axes[0].set_xlabel("Total drones")
    axes[0].set_ylabel("Runtime ms / episode")
    axes[0].set_title("Runtime scaling")
    axes[1].set_xlabel("Total drones")
    axes[1].set_ylabel("Success rate")
    axes[1].set_title("Performance scaling")
    for ax in axes:
        ax.grid(True, color="#d9d9d9", linewidth=0.5, alpha=0.7)
        ax.legend(frameon=False)
    fig.savefig(out_dir / "scalability_trends.pdf")
    fig.savefig(out_dir / "scalability_trends.png")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--optimized", type=Path, default=Path("trans_project/results/local_optimized_baselines_eval100/episodes.csv"))
    parser.add_argument("--budget", type=Path, default=Path("trans_project/results/local_budget_eval100/episodes.csv"))
    parser.add_argument("--scalability", type=Path, default=Path("trans_project/results/local_scalability_eval50/episodes.csv"))
    parser.add_argument("--out-dir", type=Path, default=Path("paper/figures"))
    args = parser.parse_args()

    setup_style()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    make_pareto(rows(args.optimized), args.out_dir)
    make_budget(rows(args.budget), args.out_dir)
    make_scalability(rows(args.scalability), args.out_dir)
    print(f"wrote result figures to {args.out_dir}")


if __name__ == "__main__":
    main()
