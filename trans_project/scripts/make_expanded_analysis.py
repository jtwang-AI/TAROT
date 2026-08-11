from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path
from statistics import mean, pstdev

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


KEY_POLICIES = ["safety_only", "tarot_no_teammate", "tarot_default", "tarot_tuned_cem", "flat_cem_tuned"]
POLICY_LABELS = {
    "greedy": "Greedy",
    "safety_only": "Safety only",
    "tarot_no_safety": "TAROT w/o safety",
    "tarot_no_teammate": "TAROT w/o teammate",
    "tarot_no_risk_gate": "TAROT w/o risk gate",
    "tarot_instant_belief": "TAROT instant",
    "tarot_default": "TAROT",
    "tarot_tuned_cem": "TAROT-CEM",
    "flat_cem_tuned": "Flat CEM",
}
COLORS = {
    "safety_only": "#2c7fb8",
    "tarot_no_teammate": "#fdae6b",
    "tarot_default": "#1b9e77",
    "tarot_tuned_cem": "#006d2c",
    "flat_cem_tuned": "#756bb1",
    "greedy": "#7a7a7a",
}


def read_rows(path: Path) -> list[dict]:
    with path.open() as f:
        return list(csv.DictReader(f))


def fval(row: dict, key: str) -> float:
    return float(row[key])


def mean_ci(vals: list[float]) -> tuple[float, float]:
    if not vals:
        return 0.0, 0.0
    m = mean(vals)
    if len(vals) < 2:
        return m, 0.0
    half = 1.96 * pstdev(vals) / (len(vals) ** 0.5)
    return m, half


def grouped(rows: list[dict], keys: tuple[str, ...]) -> dict[tuple[str, ...], list[dict]]:
    out = defaultdict(list)
    for row in rows:
        out[tuple(row[k] for k in keys)].append(row)
    return dict(out)


def metric_by_policy(rows: list[dict], metric: str) -> dict[str, list[float]]:
    out = defaultdict(list)
    for row in rows:
        out[row["policy"]].append(fval(row, metric))
    return dict(out)


def policy_summary_table(rows: list[dict], caption: str, label: str, policies: list[str]) -> str:
    by_policy = metric_by_policy(rows, "success")
    collisions = metric_by_policy(rows, "collision_count")
    shields = metric_by_policy(rows, "safety_interventions")
    near = metric_by_policy(rows, "near_miss_count")
    lines = [
        "\\begin{table}[t]",
        "\\centering",
        f"\\caption{{{caption}}}",
        f"\\label{{{label}}}",
        "\\resizebox{\\columnwidth}{!}{%",
        "\\begin{tabular}{lrrrr}",
        "\\toprule",
        "Method & Success $\\uparrow$ & Collision $\\downarrow$ & Near miss $\\downarrow$ & Shield $\\downarrow$ \\\\",
        "\\midrule",
    ]
    for policy in policies:
        if policy not in by_policy:
            continue
        succ, succ_ci = mean_ci(by_policy[policy])
        coll, _ = mean_ci(collisions[policy])
        near_m, _ = mean_ci(near[policy])
        shield, _ = mean_ci(shields[policy])
        lines.append(
            f"{POLICY_LABELS[policy]} & {succ:.3f}$\\pm${succ_ci:.3f} & {coll:.3f} & {near_m:.2f} & {shield:.1f} \\\\"
        )
    lines.extend(["\\bottomrule", "\\end{tabular}", "}", "\\end{table}", ""])
    return "\n".join(lines)


def paired_deltas(rows: list[dict], key_fields: tuple[str, ...], base: str, compare: str) -> tuple[tuple[float, float], tuple[float, float]]:
    groups = grouped(rows, key_fields)
    success_d = []
    shield_d = []
    for items in groups.values():
        by_policy = {row["policy"]: row for row in items}
        if base in by_policy and compare in by_policy:
            success_d.append(fval(by_policy[base], "success") - fval(by_policy[compare], "success"))
            shield_d.append(fval(by_policy[base], "safety_interventions") - fval(by_policy[compare], "safety_interventions"))
    return mean_ci(success_d), mean_ci(shield_d)


def paired_delta_table(robust: list[dict], reliability: list[dict], tail: list[dict]) -> str:
    specs = [
        (
            "Robustness",
            robust,
            ("scenario", "target_policy", "partner_policy", "perception_noise", "num_obstacles", "seed"),
        ),
        (
            "Reliability",
            reliability,
            ("scenario", "target_policy", "partner_policy", "partner_speed", "perception_noise", "seed"),
        ),
        (
            "Tail stress",
            tail,
            ("scenario", "target_policy", "partner_policy", "num_obstacles", "perception_noise", "target_speed", "seed"),
        ),
    ]
    comparisons = [
        ("tarot_default", "safety_only", "TAROT - Safety"),
        ("tarot_default", "tarot_no_teammate", "TAROT - no teammate"),
        ("tarot_tuned_cem", "tarot_default", "TAROT-CEM - TAROT"),
        ("flat_cem_tuned", "tarot_default", "Flat CEM - TAROT"),
    ]
    lines = [
        "\\begin{table*}[t]",
        "\\centering",
        "\\caption{Paired effect estimates across expanded evaluations. Positive $\\Delta$Success favors the first method; negative $\\Delta$Shield means the first method uses fewer safety interventions.}",
        "\\label{tab:paired_deltas}",
        "\\begin{tabular}{llrr}",
        "\\toprule",
        "Block & Comparison & $\\Delta$Success $\\uparrow$ & $\\Delta$Shield $\\downarrow$ \\\\",
        "\\midrule",
    ]
    for block, data, keys in specs:
        for base, compare, name in comparisons:
            (ds, ds_ci), (dh, dh_ci) = paired_deltas(data, keys, base, compare)
            lines.append(f"{block} & {name} & {ds:.3f}$\\pm${ds_ci:.3f} & {dh:.1f}$\\pm${dh_ci:.1f} \\\\")
        lines.append("\\midrule")
    lines[-1] = "\\bottomrule"
    lines.extend(["\\end{tabular}", "\\end{table*}", ""])
    return "\n".join(lines)


def speed_bin(speed: float) -> str:
    if speed <= 1.8:
        return "slow"
    if speed >= 3.0:
        return "fast"
    return "nominal"


def reliability_bin_table(rows: list[dict]) -> str:
    by = defaultdict(list)
    for row in rows:
        if row["policy"] in KEY_POLICIES:
            by[(row["policy"], speed_bin(fval(row, "partner_speed")))].append(row)
    lines = [
        "\\begin{table}[t]",
        "\\centering",
        "\\caption{Partner reliability sweep grouped by partner-speed regime. Each cell reports success / shield.}",
        "\\label{tab:partner_reliability_bins}",
        "\\resizebox{\\columnwidth}{!}{%",
        "\\begin{tabular}{lrrr}",
        "\\toprule",
        "Method & Slow & Nominal & Fast \\\\",
        "\\midrule",
    ]
    for policy in KEY_POLICIES:
        cells = []
        for bin_name in ["slow", "nominal", "fast"]:
            items = by.get((policy, bin_name), [])
            succ = mean([fval(r, "success") for r in items]) if items else 0.0
            shield = mean([fval(r, "safety_interventions") for r in items]) if items else 0.0
            cells.append(f"{succ:.3f}/{shield:.1f}")
        lines.append(f"{POLICY_LABELS[policy]} & " + " & ".join(cells) + " \\\\")
    lines.extend(["\\bottomrule", "\\end{tabular}", "}", "\\end{table}", ""])
    return "\n".join(lines)


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


def plot_curve(rows: list[dict], x_key: str, y_key: str, out: Path, title: str, xlabel: str, ylabel: str) -> None:
    by = defaultdict(list)
    for row in rows:
        if row["policy"] in KEY_POLICIES:
            by[(row["policy"], fval(row, x_key))].append(fval(row, y_key))
    fig, ax = plt.subplots(figsize=(3.4, 2.25))
    for policy in KEY_POLICIES:
        points = []
        for (p, x), vals in by.items():
            if p == policy:
                points.append((x, mean(vals)))
        points.sort()
        if not points:
            continue
        ax.plot([x for x, _ in points], [y for _, y in points], marker="o", color=COLORS[policy], label=POLICY_LABELS[policy])
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, color="#d9d9d9", linewidth=0.5, alpha=0.7)
    ax.legend(frameon=False)
    fig.savefig(out)
    fig.savefig(out.with_suffix(".png"))
    plt.close(fig)


def notes(robust: list[dict], reliability: list[dict], tail: list[dict]) -> str:
    def overall(data: list[dict], policy: str, metric: str) -> float:
        vals = [fval(r, metric) for r in data if r["policy"] == policy]
        return mean(vals) if vals else 0.0

    lines = [
        "# Expanded Experiment Notes",
        "",
        "## Row Counts",
        "",
        f"- Robustness sweep: {len(robust)} episode rows.",
        f"- Partner reliability sweep: {len(reliability)} episode rows.",
        f"- Tail stress evaluation: {len(tail)} episode rows.",
        "",
        "## Overall Signals",
        "",
        f"- Robustness TAROT success={overall(robust, 'tarot_default', 'success'):.3f}, shield={overall(robust, 'tarot_default', 'safety_interventions'):.1f}.",
        f"- Robustness Flat CEM success={overall(robust, 'flat_cem_tuned', 'success'):.3f}, shield={overall(robust, 'flat_cem_tuned', 'safety_interventions'):.1f}.",
        f"- Reliability TAROT success={overall(reliability, 'tarot_default', 'success'):.3f}, shield={overall(reliability, 'tarot_default', 'safety_interventions'):.1f}.",
        f"- Tail stress TAROT success={overall(tail, 'tarot_default', 'success'):.3f}, collision={overall(tail, 'tarot_default', 'collision_count'):.3f}.",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--robustness", type=Path, required=True)
    parser.add_argument("--reliability", type=Path, required=True)
    parser.add_argument("--tail", type=Path, required=True)
    parser.add_argument("--table-dir", type=Path, default=Path("paper/tables"))
    parser.add_argument("--figure-dir", type=Path, default=Path("paper/figures"))
    parser.add_argument("--notes", type=Path, default=Path("refine-logs/V0_4_EXPANDED_EXPERIMENTS.md"))
    args = parser.parse_args()

    robust = read_rows(args.robustness)
    reliability = read_rows(args.reliability)
    tail = read_rows(args.tail)
    args.table_dir.mkdir(parents=True, exist_ok=True)
    args.figure_dir.mkdir(parents=True, exist_ok=True)
    setup_style()

    (args.table_dir / "robustness_summary_table.tex").write_text(
        policy_summary_table(robust, "Robustness sweep over perception noise, obstacle density, partner behavior, target behavior, and scenario type.", "tab:robustness_summary", KEY_POLICIES)
    )
    (args.table_dir / "tail_stress_table.tex").write_text(
        policy_summary_table(tail, "Long-tail stress evaluation over dense obstacles, higher perception noise, faster targets, and adverse partners.", "tab:tail_stress", KEY_POLICIES)
    )
    (args.table_dir / "partner_reliability_table.tex").write_text(reliability_bin_table(reliability))
    (args.table_dir / "paired_delta_table.tex").write_text(paired_delta_table(robust, reliability, tail))
    plot_curve(robust, "perception_noise", "success", args.figure_dir / "robustness_noise_success.pdf", "Perception-noise robustness", "Perception noise", "Success rate")
    plot_curve(robust, "num_obstacles", "success", args.figure_dir / "robustness_obstacle_success.pdf", "Obstacle-density robustness", "Obstacle count", "Success rate")
    plot_curve(reliability, "partner_speed", "success", args.figure_dir / "partner_speed_success.pdf", "Partner-speed reliability", "Partner speed", "Success rate")
    plot_curve(tail, "target_speed", "success", args.figure_dir / "tail_target_speed_success.pdf", "Tail stress by target speed", "Target speed", "Success rate")
    args.notes.write_text(notes(robust, reliability, tail))
    print(f"wrote expanded analysis tables to {args.table_dir}")
    print(f"wrote expanded analysis figures to {args.figure_dir}")
    print(f"wrote notes to {args.notes}")


if __name__ == "__main__":
    main()
