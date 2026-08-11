from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from statistics import mean, pstdev
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tarot_sim import SimConfig, policy_suite, run_episode


METRICS = [
    "success",
    "task_completion",
    "steps",
    "collision_count",
    "near_miss_count",
    "safety_interventions",
    "energy",
    "role_switches",
    "progress",
]


def scenario_grid(episodes: int):
    scenarios = [
        # scenario, target, partner, noise, obstacles, target speed
        ("open_team", "evasive", "blocker", 0.00, 2, 2.35),
        ("open_team", "deceptive", "blocker", 0.00, 2, 2.35),
        ("open_team", "evasive", "lazy", 0.00, 2, 2.20),
        ("standard", "evasive", "mixed", 0.00, 8, 1.80),
        ("standard", "evasive", "blocker", 0.00, 8, 1.80),
        ("standard", "evasive", "lazy", 0.00, 8, 1.80),
        ("standard", "deceptive", "mixed", 0.00, 8, 1.80),
        ("standard", "deceptive", "blocker", 0.00, 8, 1.80),
        ("corridor", "evasive", "mixed", 0.00, 12, 1.80),
        ("standard", "stop_go", "mixed", 0.05, 8, 1.80),
        ("corridor", "deceptive", "noisy", 0.08, 12, 1.80),
    ]
    for scenario, target_policy, partner_policy, noise, obstacles, target_speed in scenarios:
        for ep in range(episodes):
            yield SimConfig(
                seed=10_000 + ep + 1_000 * len(scenario) + 97 * len(target_policy),
                scenario=scenario,
                target_policy=target_policy,
                partner_policy=partner_policy,
                perception_noise=noise,
                target_speed=target_speed,
                num_obstacles=obstacles,
            )


def write_csv(path: Path, rows: list[dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def aggregate(rows: list[dict]) -> list[dict]:
    grouped = {}
    for row in rows:
        key = (row["policy"], row["scenario"], row["target_policy"], row["partner_policy"])
        grouped.setdefault(key, []).append(row)
    out = []
    for (policy, scenario, target_policy, partner_policy), items in sorted(grouped.items()):
        rec = {
            "policy": policy,
            "scenario": scenario,
            "target_policy": target_policy,
            "partner_policy": partner_policy,
            "episodes": len(items),
        }
        for metric in METRICS:
            vals = [float(x[metric]) for x in items]
            rec[f"{metric}_mean"] = mean(vals)
            rec[f"{metric}_std"] = pstdev(vals) if len(vals) > 1 else 0.0
        out.append(rec)
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--out", type=Path, default=Path("trans_project/results/controlled_smoke"))
    args = parser.parse_args()

    policies = policy_suite()
    rows = []
    for cfg in scenario_grid(args.episodes):
        for policy in policies:
            rows.append(run_episode(policy, cfg))

    agg = aggregate(rows)
    args.out.mkdir(parents=True, exist_ok=True)
    write_csv(args.out / "episodes.csv", rows)
    write_csv(args.out / "aggregate.csv", agg)
    (args.out / "aggregate.json").write_text(json.dumps(agg, indent=2))
    print(f"wrote {len(rows)} episode rows")
    print(f"aggregate: {args.out / 'aggregate.csv'}")


if __name__ == "__main__":
    main()
