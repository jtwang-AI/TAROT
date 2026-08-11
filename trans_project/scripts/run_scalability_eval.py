from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path
from statistics import mean, pstdev
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tarot_sim import GreedyPolicy, SafetyOnlyPolicy, SimConfig, TarotPolicy, run_episode


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def aggregate(rows: list[dict]) -> list[dict]:
    grouped = {}
    for row in rows:
        key = (row["policy"], row["num_learners"], row["num_partners"], row["num_obstacles"])
        grouped.setdefault(key, []).append(row)
    out = []
    metrics = ["success", "collision_count", "steps", "runtime_ms"]
    for key, items in sorted(grouped.items()):
        policy, learners, partners, obstacles = key
        rec = {
            "policy": policy,
            "num_learners": learners,
            "num_partners": partners,
            "num_obstacles": obstacles,
            "episodes": len(items),
        }
        for metric in metrics:
            vals = [float(x[metric]) for x in items]
            rec[f"{metric}_mean"] = mean(vals)
            rec[f"{metric}_std"] = pstdev(vals) if len(vals) > 1 else 0.0
        out.append(rec)
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--out", type=Path, default=Path("trans_project/results/scalability_eval"))
    args = parser.parse_args()

    sizes = [
        (1, 1, 4),
        (2, 2, 8),
        (3, 3, 12),
        (4, 4, 16),
        (6, 6, 24),
    ]
    policies = [GreedyPolicy(), SafetyOnlyPolicy(), TarotPolicy()]
    rows = []
    for learners, partners, obstacles in sizes:
        for ep in range(args.episodes):
            cfg = SimConfig(
                seed=50_000 + ep + learners * 1009 + obstacles * 17,
                scenario="standard",
                target_policy="evasive",
                partner_policy="mixed",
                num_learners=learners,
                num_partners=partners,
                num_obstacles=obstacles,
                target_speed=1.8,
            )
            for policy in policies:
                t0 = time.perf_counter()
                row = run_episode(policy, cfg)
                row["runtime_ms"] = (time.perf_counter() - t0) * 1000.0
                row["num_obstacles"] = float(obstacles)
                rows.append(row)
    agg = aggregate(rows)
    args.out.mkdir(parents=True, exist_ok=True)
    write_csv(args.out / "episodes.csv", rows)
    write_csv(args.out / "aggregate.csv", agg)
    (args.out / "aggregate.json").write_text(json.dumps(agg, indent=2))
    print(f"wrote {len(rows)} episode rows")
    print(f"aggregate: {args.out / 'aggregate.csv'}")


if __name__ == "__main__":
    main()
