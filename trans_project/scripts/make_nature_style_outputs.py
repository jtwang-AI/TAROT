from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path
from statistics import mean

import matplotlib

matplotlib.use("Agg")
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np


POLICY_LABELS = {
    "safety_only": "Safety only",
    "tarot_no_teammate": "No teammate",
    "tarot_default": "TAROT",
    "tarot_tuned_cem": "TAROT-CEM",
    "flat_cem_tuned": "Flat CEM",
    "tarot_instant_belief": "Instant belief",
}
COLORS = {
    "safety_only": "#7aa6c2",
    "tarot_no_teammate": "#e8b071",
    "tarot_default": "#62b097",
    "tarot_tuned_cem": "#247a4b",
    "flat_cem_tuned": "#9185c6",
    "tarot_instant_belief": "#b58cc4",
}


def rows(path: Path) -> list[dict]:
    with path.open() as f:
        return list(csv.DictReader(f))


def fval(row: dict, key: str) -> float:
    return float(row[key])


def grouped_mean(data: list[dict], keys: tuple[str, ...], metrics: tuple[str, ...]) -> dict[tuple[str, ...], dict[str, float]]:
    groups = defaultdict(list)
    for row in data:
        groups[tuple(row[k] for k in keys)].append(row)
    out = {}
    for key, items in groups.items():
        out[key] = {m: mean([fval(r, m) for r in items if r[m] != ""]) for m in metrics}
        out[key]["n"] = len(items)
    return out


def setup_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 7,
            "axes.titlesize": 8,
            "axes.labelsize": 7,
            "legend.fontsize": 6,
            "xtick.labelsize": 6,
            "ytick.labelsize": 6,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "pdf.fonttype": 42,
            "svg.fonttype": "none",
            "savefig.bbox": "tight",
        }
    )


def panel_label(ax, label: str) -> None:
    ax.text(-0.08, 1.08, label, transform=ax.transAxes, fontsize=9, fontweight="bold", va="top")


def draw_schematic(ax) -> None:
    ax.axis("off")
    boxes = [
        (0.02, 0.64, "Partner\nmotion"),
        (0.37, 0.64, "Belief\nupdate"),
        (0.72, 0.64, "Risk\ngate"),
        (0.24, 0.20, "Role\nassignment"),
        (0.62, 0.20, "Safety\ngovernor"),
    ]
    for x, y, text in boxes:
        rect = patches.FancyBboxPatch(
            (x, y),
            0.23,
            0.18,
            boxstyle="round,pad=0.02,rounding_size=0.025",
            linewidth=0.8,
            edgecolor="#3b3b3b",
            facecolor="#f2f5f3",
        )
        ax.add_patch(rect)
        ax.text(x + 0.115, y + 0.09, text, ha="center", va="center", fontsize=6.6)
    arrows = [
        ((0.25, 0.73), (0.37, 0.73)),
        ((0.60, 0.73), (0.72, 0.73)),
        ((0.485, 0.64), (0.36, 0.40)),
        ((0.47, 0.29), (0.62, 0.29)),
    ]
    for start, end in arrows:
        ax.annotate("", xy=end, xytext=start, arrowprops=dict(arrowstyle="->", lw=0.9, color="#3b3b3b"))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_title("Mechanism")


def plot_pareto(ax, optimized: list[dict]) -> None:
    mean_by = grouped_mean(optimized, ("policy",), ("success", "collision_count", "safety_interventions"))
    order = ["safety_only", "tarot_default", "tarot_tuned_cem", "flat_cem_tuned"]
    for policy in order:
        r = mean_by[(policy,)]
        ax.scatter(
            r["safety_interventions"],
            r["success"],
            s=80,
            color=COLORS[policy],
            edgecolor="black",
            linewidth=0.4,
            label=POLICY_LABELS[policy],
        )
        ax.annotate(POLICY_LABELS[policy], (r["safety_interventions"], r["success"]), xytext=(4, 4), textcoords="offset points", fontsize=6)
    ax.set_xlabel("Safety interventions")
    ax.set_ylabel("Success")
    ax.set_title("Operating point")
    ax.grid(True, lw=0.4, color="#dddddd")


def plot_partner_speed(ax, reliability: list[dict]) -> None:
    mean_by = grouped_mean(reliability, ("policy", "partner_speed"), ("success",))
    for policy in ["safety_only", "tarot_default", "tarot_tuned_cem", "flat_cem_tuned"]:
        pts = []
        for (p, speed), rec in mean_by.items():
            if p == policy:
                pts.append((float(speed), rec["success"]))
        pts.sort()
        ax.plot([x for x, _ in pts], [y for _, y in pts], marker="o", lw=1.2, ms=3, color=COLORS[policy], label=POLICY_LABELS[policy])
    ax.set_xlabel("Partner speed")
    ax.set_ylabel("Success")
    ax.set_title("Reliability shift")
    ax.grid(True, lw=0.4, color="#dddddd")


def plot_intervention_reduction(ax, reliability: list[dict]) -> None:
    data = [r for r in reliability if r["policy"] in {"tarot_default", "tarot_no_teammate"}]
    mean_by = grouped_mean(data, ("policy", "partner_policy"), ("safety_interventions", "success"))
    partners = ["greedy", "flank", "blocker", "lazy", "noisy", "mixed"]
    reductions = []
    for partner in partners:
        no_tm = mean_by[("tarot_no_teammate", partner)]
        full = mean_by[("tarot_default", partner)]
        reductions.append(no_tm["safety_interventions"] - full["safety_interventions"])
    x = np.arange(len(partners))
    ax.bar(x, reductions, color="#62b097", width=0.62, label="Shield reduction")
    ax.axhline(0, color="#555555", lw=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels(["chase", "flank", "block", "lazy", "noisy", "mixed"], rotation=25, ha="right")
    ax.set_ylabel("Fewer interventions")
    ax.set_title("Teammate-conditioning effect")
    ax.grid(True, axis="y", lw=0.4, color="#dddddd")


def plot_gate_heatmap(ax, belief_steps: list[dict]) -> None:
    data = [r for r in belief_steps if r["policy"] == "tarot_default"]
    mean_by = grouped_mean(data, ("scenario", "perception_noise"), ("risk_gated",))
    scenarios = ["open_team", "standard", "corridor"]
    noises = [0.0, 0.06, 0.12]
    mat = np.array([[mean_by.get((s, str(n)), {"risk_gated": 0.0})["risk_gated"] for n in noises] for s in scenarios])
    im = ax.imshow(mat, cmap="YlGnBu", vmin=0, vmax=max(0.25, float(mat.max())))
    ax.set_xticks(range(len(noises)))
    ax.set_xticklabels([f"{n:.2f}" for n in noises])
    ax.set_yticks(range(len(scenarios)))
    ax.set_yticklabels(["open", "standard", "corridor"])
    ax.set_xlabel("Noise")
    ax.set_title("Risk-gate rate")
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            ax.text(j, i, f"{mat[i, j]:.2f}", ha="center", va="center", fontsize=6)
    return im


def belief_table(belief_steps: list[dict]) -> str:
    rows_default = [r for r in belief_steps if r["policy"] == "tarot_default"]
    mean_by = grouped_mean(rows_default, ("partner_policy",), ("mode_accuracy", "actionable_mode_accuracy", "risk_gated", "uncertain", "confidence", "reliability", "safety_interventions"))
    partners = ["greedy", "flank", "blocker", "lazy", "noisy"]
    lines = [
        "\\begin{table}[t]",
        "\\centering",
        "\\caption{Belief diagnostic for TAROT over 146,659 step-level records. Actionable accuracy excludes risk-gated and uncertain steps.}",
        "\\label{tab:belief_diagnostics}",
        "\\resizebox{\\columnwidth}{!}{%",
        "\\begin{tabular}{lrrrrr}",
        "\\toprule",
        "Partner & Mode acc. & Act. acc. & Gate & Conf. & Rel. \\\\",
        "\\midrule",
    ]
    for partner in partners:
        r = mean_by[(partner,)]
        lines.append(
            f"{partner} & {r['mode_accuracy']:.3f} & {r['actionable_mode_accuracy']:.3f} & "
            f"{r['risk_gated']:.3f} & {r['confidence']:.3f} & {r['reliability']:.3f} \\\\"
        )
    lines.extend(["\\bottomrule", "\\end{tabular}", "}", "\\end{table}", ""])
    return "\n".join(lines)


def make_composite(args) -> None:
    optimized = rows(args.optimized)
    reliability = rows(args.reliability)
    belief_steps = rows(args.belief_steps)
    setup_style()
    fig = plt.figure(figsize=(7.2, 4.9))
    gs = fig.add_gridspec(2, 3, height_ratios=[1.0, 1.05], wspace=0.42, hspace=0.55)
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[0, 2])
    ax_d = fig.add_subplot(gs[1, 0:2])
    ax_e = fig.add_subplot(gs[1, 2])
    draw_schematic(ax_a)
    plot_pareto(ax_b, optimized)
    plot_partner_speed(ax_c, reliability)
    plot_intervention_reduction(ax_d, reliability)
    im = plot_gate_heatmap(ax_e, belief_steps)
    for ax, label in [(ax_a, "a"), (ax_b, "b"), (ax_c, "c"), (ax_d, "d"), (ax_e, "e")]:
        panel_label(ax, label)
    cbar = fig.colorbar(im, ax=ax_e, fraction=0.046, pad=0.04)
    cbar.set_label("Gate rate", fontsize=6)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out_dir / "nature_main_composite.pdf")
    fig.savefig(args.out_dir / "nature_main_composite.svg")
    fig.savefig(args.out_dir / "nature_main_composite.png", dpi=600)
    plt.close(fig)
    args.table_dir.mkdir(parents=True, exist_ok=True)
    (args.table_dir / "belief_diagnostics_table.tex").write_text(belief_table(belief_steps))
    print(f"wrote Nature-style composite to {args.out_dir}")
    print(f"wrote belief diagnostics table to {args.table_dir}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--optimized", type=Path, default=Path("trans_project/results/local_optimized_baselines_eval100/episodes.csv"))
    parser.add_argument("--reliability", type=Path, default=Path("remote_results/remote_partner_reliability_eval50/episodes.csv"))
    parser.add_argument("--belief-steps", type=Path, default=Path("trans_project/results/local_belief_diagnostics_eval50/belief_steps.csv"))
    parser.add_argument("--out-dir", type=Path, default=Path("paper/figures"))
    parser.add_argument("--table-dir", type=Path, default=Path("paper/tables"))
    args = parser.parse_args()
    make_composite(args)


if __name__ == "__main__":
    main()
