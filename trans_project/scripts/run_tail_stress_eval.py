from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from eval_common import save_run, policy_suite
from tarot_sim import SimConfig, run_episode


def configs(episodes: int):
    scenarios = ["corridor", "standard"]
    target_policies = ["evasive", "deceptive", "stop_go"]
    partner_policies = ["mixed", "noisy", "lazy"]
    obstacle_counts = [12, 16, 20, 24]
    noise_levels = [0.08, 0.12, 0.16]
    target_speeds = [1.8, 2.2, 2.6]
    for scenario in scenarios:
        for target in target_policies:
            for partner in partner_policies:
                for obstacles in obstacle_counts:
                    for noise in noise_levels:
                        for speed in target_speeds:
                            for ep in range(episodes):
                                yield SimConfig(
                                    seed=310_000
                                    + ep
                                    + 101 * len(scenario)
                                    + 1009 * len(target)
                                    + 9176 * len(partner)
                                    + obstacles * 37
                                    + int(noise * 10_000)
                                    + int(speed * 1000),
                                    scenario=scenario,
                                    target_policy=target,
                                    partner_policy=partner,
                                    num_obstacles=obstacles,
                                    perception_noise=noise,
                                    target_speed=speed,
                                )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=30)
    parser.add_argument("--out", type=Path, default=Path("trans_project/results/tail_stress_eval30"))
    args = parser.parse_args()

    policies = policy_suite(
        (
            "greedy",
            "safety_only",
            "tarot_no_teammate",
            "tarot_no_risk_gate",
            "tarot_default",
            "tarot_tuned_cem",
            "flat_cem_tuned",
        )
    )
    rows = []
    for cfg in configs(args.episodes):
        for policy in policies:
            row = run_episode(policy, cfg)
            row["perception_noise"] = float(cfg.perception_noise)
            row["num_obstacles"] = float(cfg.num_obstacles)
            row["target_speed"] = float(cfg.target_speed)
            rows.append(row)
    save_run(
        args.out,
        rows,
        ("policy", "scenario", "target_policy", "partner_policy", "num_obstacles", "perception_noise", "target_speed"),
    )


if __name__ == "__main__":
    main()
