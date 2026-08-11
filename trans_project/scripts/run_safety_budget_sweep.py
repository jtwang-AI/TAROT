from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from statistics import mean, pstdev
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tarot_sim import GreedyPolicy, SafetyOnlyPolicy, SimConfig, TarotPolicy, run_episode


METRICS = ["success", "steps", "collision_count", "near_miss_count", "safety_interventions", "energy"]


class NamedTarot(TarotPolicy):
    def __init__(self, budget: float):
        super().__init__(safety_budget=budget)
        self.name = f"tarot_budget_{budget:.2f}".replace(".", "p")


def configs(episodes: int):
    cases = [
        ("open_team", "evasive", "lazy", 2, 2.35, 0.0),
        ("standard", "evasive", "mixed", 8, 1.80, 0.0),
        ("standard", "stop_go", "mixed", 8, 1.80, 0.05),
        ("corridor", "deceptive", "noisy", 12, 1.80, 0.08),
    ]
    for scenario, target, partner, obstacles, speed, noise in cases:
        for ep in range(episodes):
            yield SimConfig(
                seed=30_000 + ep + len(scenario) * 907 + len(partner) * 101,
                scenario=scenario,
                target_policy=target,
                partner_policy=partner,
                num_obstacles=obstacles,
                target_speed=speed,
                perception_noise=noise,
            )


def aggregate(rows: list[dict]) -> list[dict]:
    grouped = {}
    for row in rows:
        key = (row["policy"], row["scenario"], row["partner_policy"])
        grouped.setdefault(key, []).append(row)
    out = []
    for (policy, scenario, partner), items in sorted(grouped.items()):
        rec = {"policy": policy, "scenario": scenario, "partner_policy": partner, "episodes": len(items)}
        for metric in METRICS:
            vals = [float(x[metric]) for x in items]
            rec[f"{metric}_mean"] = mean(vals)
            rec[f"{metric}_std"] = pstdev(vals) if len(vals) > 1 else 0.0
        out.append(rec)
    return out


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--out", type=Path, default=Path("trans_project/results/safety_budget_sweep"))
    args = parser.parse_args()

    budgets = [0.0, 0.35, 0.55, 0.75, 0.85, 1.05, 1.25]
    policies = [GreedyPolicy(), SafetyOnlyPolicy()] + [NamedTarot(b) for b in budgets]
    rows = []
    for cfg in configs(args.episodes):
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
