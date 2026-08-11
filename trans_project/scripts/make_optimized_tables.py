from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path


METRICS = ["success", "collision_count", "near_miss_count", "steps", "safety_interventions"]


METHOD_NAMES = {
    "greedy": "Greedy pursuit",
    "safety_only": "Safety only",
    "tarot_no_safety": "TAROT w/o safety",
    "tarot_default": "TAROT",
    "tarot_tuned_cem": "TAROT-CEM",
    "flat_cem_tuned": "Flat CEM",
}


OVERALL_ORDER = [
    "greedy",
    "tarot_no_safety",
    "safety_only",
    "tarot_default",
    "tarot_tuned_cem",
    "flat_cem_tuned",
]


SCENARIO_ORDER = [
    ("open_team", "evasive", "lazy", "open/lazy"),
    ("standard", "evasive", "lazy", "standard/lazy"),
    ("corridor", "deceptive", "noisy", "corridor/noisy"),
]


SCENARIO_POLICY_ORDER = ["safety_only", "tarot_default", "tarot_tuned_cem", "flat_cem_tuned"]


def load_rows(path: Path) -> list[dict]:
    with path.open() as f:
        return list(csv.DictReader(f))


def grouped_mean(rows: list[dict], keys: tuple[str, ...]) -> dict[tuple[str, ...], dict[str, float]]:
    groups = defaultdict(list)
    for row in rows:
        groups[tuple(row[key] for key in keys)].append(row)
    out = {}
    for key, items in groups.items():
        out[key] = {metric: sum(float(r[metric]) for r in items) / len(items) for metric in METRICS}
        out[key]["episodes"] = float(len(items))
    return out


def optimized_table(rows: list[dict]) -> str:
    overall = grouped_mean(rows, ("policy",))
    flat_shield = overall[("flat_cem_tuned",)]["safety_interventions"]
    lines = [
        "\\begin{tabular}{lrrrrr}",
        "\\toprule",
        "Method & Success $\\uparrow$ & Collision $\\downarrow$ & Near miss $\\downarrow$ & Shield $\\downarrow$ & Shield red. $\\uparrow$ \\\\",
        "\\midrule",
    ]
    for policy in OVERALL_ORDER:
        r = overall[(policy,)]
        if policy in {"greedy", "tarot_no_safety", "flat_cem_tuned"}:
            reduction = "--"
        elif flat_shield > 0:
            reduction = f"{100.0 * (1.0 - r['safety_interventions'] / flat_shield):.0f}\\%"
        else:
            reduction = "--"
        lines.append(
            f"{METHOD_NAMES[policy]} & {r['success']:.3f} & {r['collision_count']:.3f} & "
            f"{r['near_miss_count']:.2f} & {r['safety_interventions']:.1f} & {reduction} \\\\"
        )
    lines.extend(["\\bottomrule", "\\end{tabular}", ""])
    return "\n".join(lines)


def scenario_table(rows: list[dict]) -> str:
    grouped = grouped_mean(rows, ("scenario", "target_policy", "partner_policy", "policy"))
    lines = [
        "\\begin{tabular}{llrrrr}",
        "\\toprule",
        "Setting & Method & Success $\\uparrow$ & Collision $\\downarrow$ & Steps $\\downarrow$ & Shield $\\downarrow$ \\\\",
        "\\midrule",
    ]
    for scenario, target, partner, label in SCENARIO_ORDER:
        for i, policy in enumerate(SCENARIO_POLICY_ORDER):
            r = grouped[(scenario, target, partner, policy)]
            setting = label if i == 0 else ""
            lines.append(
                f"{setting} & {METHOD_NAMES[policy]} & {r['success']:.3f} & {r['collision_count']:.3f} & "
                f"{r['steps']:.1f} & {r['safety_interventions']:.1f} \\\\"
            )
        if label != SCENARIO_ORDER[-1][3]:
            lines.append("\\midrule")
    lines.extend(["\\bottomrule", "\\end{tabular}", ""])
    return "\n".join(lines)


def result_notes(rows: list[dict]) -> str:
    overall = grouped_mean(rows, ("policy",))
    flat = overall[("flat_cem_tuned",)]
    tarot = overall[("tarot_default",)]
    tuned = overall[("tarot_tuned_cem",)]
    shield_drop = 100.0 * (1.0 - tarot["safety_interventions"] / flat["safety_interventions"])
    tuned_drop = 100.0 * (1.0 - tuned["safety_interventions"] / flat["safety_interventions"])
    return "\n".join(
        [
            "# Optimized Baseline Result Notes",
            "",
            "## Overall",
            "",
            f"- Flat CEM reaches {flat['success']:.3f} success and {flat['collision_count']:.3f} collisions, but uses {flat['safety_interventions']:.1f} safety interventions per episode.",
            f"- Default TAROT reaches {tarot['success']:.3f} success and {tarot['collision_count']:.3f} collisions with {tarot['safety_interventions']:.1f} interventions, a {shield_drop:.0f}% reduction relative to Flat CEM.",
            f"- TAROT-CEM reaches {tuned['success']:.3f} success and {tuned['collision_count']:.3f} collisions with {tuned['safety_interventions']:.1f} interventions, a {tuned_drop:.0f}% reduction relative to Flat CEM.",
            "",
            "## Interpretation",
            "",
            "- The optimized flat controller is a useful high-intervention upper-bound baseline.",
            "- The strongest defensible TAROT claim is a safety-efficiency Pareto tradeoff, not domination of every metric.",
            "- The corridor/noisy setting remains a failure/stress regime and should be reported honestly.",
            "",
        ]
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, default=Path("paper/tables"))
    parser.add_argument("--notes", type=Path, default=Path("refine-logs/V0_3_OPTIMIZED_BASELINES.md"))
    args = parser.parse_args()

    rows = load_rows(args.episodes)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "optimized_baselines_table.tex").write_text(optimized_table(rows))
    (args.out_dir / "optimized_scenario_table.tex").write_text(scenario_table(rows))
    args.notes.write_text(result_notes(rows))
    print(f"wrote optimized tables to {args.out_dir}")
    print(f"wrote notes to {args.notes}")


if __name__ == "__main__":
    main()
