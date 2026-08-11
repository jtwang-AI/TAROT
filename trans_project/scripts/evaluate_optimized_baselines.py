from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from run_controlled_eval import aggregate, scenario_grid, write_csv
from tarot_sim import FlatParamPolicy, GreedyPolicy, SafetyOnlyPolicy, TarotPolicy, run_episode


def load_params(path: Path) -> dict[str, float]:
    payload = json.loads(path.read_text())
    return {k: float(v) for k, v in payload["params"].items()}


def policy_set(tarot_params: dict[str, float], flat_params: dict[str, float]):
    tarot_tuned = TarotPolicy(**tarot_params)
    tarot_tuned.name = "tarot_tuned_cem"
    flat_tuned = FlatParamPolicy(params=flat_params, name="flat_cem_tuned")
    tarot_default = TarotPolicy()
    tarot_default.name = "tarot_default"
    tarot_no_safety = TarotPolicy(use_safety=False)
    tarot_no_safety.name = "tarot_no_safety"
    return [
        GreedyPolicy(),
        SafetyOnlyPolicy(),
        tarot_no_safety,
        tarot_default,
        tarot_tuned,
        flat_tuned,
    ]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--out", type=Path, default=Path("trans_project/results/optimized_baselines_eval100"))
    parser.add_argument(
        "--tarot-policy",
        type=Path,
        default=Path("trans_project/results/local_tarot_cem_v1/best_policy.json"),
    )
    parser.add_argument(
        "--flat-policy",
        type=Path,
        default=Path("trans_project/results/local_flat_cem_v1/best_policy.json"),
    )
    args = parser.parse_args()

    policies = policy_set(load_params(args.tarot_policy), load_params(args.flat_policy))
    rows = []
    for cfg in scenario_grid(args.episodes):
        for policy in policies:
            rows.append(run_episode(policy, cfg))

    args.out.mkdir(parents=True, exist_ok=True)
    write_csv(args.out / "episodes.csv", rows)
    write_csv(args.out / "aggregate.csv", aggregate(rows))
    (args.out / "aggregate.json").write_text(json.dumps(aggregate(rows), indent=2))
    print(f"wrote {len(rows)} episode rows")
    print(f"aggregate: {args.out / 'aggregate.csv'}")


if __name__ == "__main__":
    main()
