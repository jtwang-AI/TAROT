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
    scenarios = ["open_team", "standard", "corridor"]
    target_policies = ["evasive", "deceptive"]
    partner_policies = ["blocker", "lazy", "mixed", "noisy"]
    noise_levels = [0.0, 0.04, 0.08, 0.12]
    obstacle_counts = [2, 8, 14, 20]
    for scenario in scenarios:
        for target in target_policies:
            for partner in partner_policies:
                for noise in noise_levels:
                    for obstacles in obstacle_counts:
                        if scenario == "open_team" and obstacles > 8:
                            continue
                        if scenario == "corridor" and obstacles < 8:
                            continue
                        target_speed = 2.25 if scenario == "open_team" else 1.85
                        for ep in range(episodes):
                            yield SimConfig(
                                seed=110_000
                                + ep
                                + 101 * len(scenario)
                                + 1009 * len(target)
                                + 9176 * len(partner)
                                + int(noise * 10_000)
                                + obstacles * 37,
                                scenario=scenario,
                                target_policy=target,
                                partner_policy=partner,
                                perception_noise=noise,
                                num_obstacles=obstacles,
                                target_speed=target_speed,
                            )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--out", type=Path, default=Path("trans_project/results/robustness_sweep_eval50"))
    args = parser.parse_args()

    policies = policy_suite(
        (
            "greedy",
            "safety_only",
            "tarot_no_safety",
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
        ("policy", "scenario", "target_policy", "partner_policy", "perception_noise", "num_obstacles"),
    )


if __name__ == "__main__":
    main()
