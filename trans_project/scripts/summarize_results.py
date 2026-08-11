from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path


METRICS = [
    "success",
    "task_completion",
    "steps",
    "collision_count",
    "near_miss_count",
    "safety_interventions",
    "energy",
    "role_switches",
]


def mean(rows: list[dict], key: str) -> float:
    return sum(float(r[key]) for r in rows) / max(len(rows), 1)


def load_rows(path: Path) -> list[dict]:
    with path.open() as f:
        return list(csv.DictReader(f))


def group_rows(rows: list[dict], keys: tuple[str, ...]) -> dict[tuple[str, ...], list[dict]]:
    grouped = defaultdict(list)
    for row in rows:
        grouped[tuple(row[k] for k in keys)].append(row)
    return dict(grouped)


def summarize_grouped(grouped: dict[tuple[str, ...], list[dict]], keys: tuple[str, ...]) -> list[dict]:
    out = []
    for group_key, items in sorted(grouped.items()):
        rec = {k: v for k, v in zip(keys, group_key)}
        rec["episodes"] = len(items)
        for metric in METRICS:
            rec[metric] = mean(items, metric)
        out.append(rec)
    return out


def latex_main_table(policy_summary: list[dict]) -> str:
    names = {
        "greedy": "Greedy pursuit",
        "safety_only": "Safety only",
        "teammate_only": "Teammate only",
        "tarot_no_safety": "TAROT w/o safety",
        "tarot_no_risk_gate": "TAROT w/o risk gate",
        "tarot_no_teammate": "TAROT w/o teammate belief",
        "tarot_no_events": "TAROT w/o event gating",
        "tarot_instant_belief": "TAROT w/ instant belief",
        "tarot_full": "TAROT",
    }
    order = [
        "greedy",
        "teammate_only",
        "safety_only",
        "tarot_no_safety",
        "tarot_no_risk_gate",
        "tarot_no_teammate",
        "tarot_no_events",
        "tarot_instant_belief",
        "tarot_full",
    ]
    by_policy = {r["policy"]: r for r in policy_summary}
    lines = [
        "\\begin{tabular}{lrrrrr}",
        "\\toprule",
        "Method & Success $\\uparrow$ & Collision $\\downarrow$ & Near miss $\\downarrow$ & Steps $\\downarrow$ & Shield $\\downarrow$ \\\\",
        "\\midrule",
    ]
    for policy in order:
        if policy not in by_policy:
            continue
        r = by_policy[policy]
        lines.append(
            f"{names.get(policy, policy)} & "
            f"{r['success']:.3f} & {r['collision_count']:.3f} & {r['near_miss_count']:.2f} & "
            f"{r['steps']:.1f} & {r['safety_interventions']:.1f} \\\\"
        )
    lines.extend(["\\bottomrule", "\\end{tabular}", ""])
    return "\n".join(lines)


def markdown_notes(policy_summary: list[dict], scenario_summary: list[dict]) -> str:
    by_policy = {r["policy"]: r for r in policy_summary}
    full = by_policy.get("tarot_full", {})
    greedy = by_policy.get("greedy", {})
    no_safety = by_policy.get("tarot_no_safety", {})
    no_tm = by_policy.get("tarot_no_teammate", {})
    safety = by_policy.get("safety_only", {})
    lines = [
        "# V0.1 Result Notes",
        "",
        "## Overall",
        "",
        f"- TAROT success: {full.get('success', 0):.3f}.",
        f"- Greedy success: {greedy.get('success', 0):.3f}.",
        f"- TAROT without safety success: {no_safety.get('success', 0):.3f}.",
        f"- TAROT without teammate belief success: {no_tm.get('success', 0):.3f}.",
        f"- Safety-only success: {safety.get('success', 0):.3f}.",
        "",
        "## Interpretation",
        "",
        "- The reachability-style safety governor is the dominant validated contribution in v0.1.",
        "- Teammate belief currently helps most when partners are weak or lazy: it reduces steps and safety interventions.",
        "- Event gating is not yet distinguishable from always-updated roles and should not be a primary paper claim yet.",
        "- Corridor/noisy settings remain difficult and should be treated as stress/failure analysis unless improved.",
        "",
        "## Strong Scenario-Level Signals",
        "",
    ]
    interesting = [
        r
        for r in scenario_summary
        if r["policy"] in {"tarot_full", "safety_only", "tarot_no_teammate", "tarot_no_safety"}
        and r["partner_policy"] in {"lazy", "mixed", "noisy"}
    ]
    for r in interesting:
        lines.append(
            f"- {r['scenario']} / {r['partner_policy']} / {r['policy']}: "
            f"success={r['success']:.3f}, collision={r['collision_count']:.3f}, "
            f"steps={r['steps']:.1f}, shield={r['safety_interventions']:.1f}."
        )
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    rows = load_rows(args.episodes)
    args.out.mkdir(parents=True, exist_ok=True)
    policy_summary = summarize_grouped(group_rows(rows, ("policy",)), ("policy",))
    scenario_summary = summarize_grouped(
        group_rows(rows, ("scenario", "target_policy", "partner_policy", "policy")),
        ("scenario", "target_policy", "partner_policy", "policy"),
    )

    (args.out / "policy_summary.json").write_text(json.dumps(policy_summary, indent=2))
    (args.out / "scenario_summary.json").write_text(json.dumps(scenario_summary, indent=2))
    (args.out / "main_table.tex").write_text(latex_main_table(policy_summary))
    (args.out / "RESULT_NOTES.md").write_text(markdown_notes(policy_summary, scenario_summary))
    print(f"summary written to {args.out}")


if __name__ == "__main__":
    main()
