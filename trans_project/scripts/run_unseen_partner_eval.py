from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from eval_common import policy_suite, save_run
from tarot_sim import SimConfig, run_episode


def configs(episodes: int):
    """Evaluation-only configurations with partners outside the belief prototypes."""
    scenarios = ["open_team", "standard"]
    target_policies = ["evasive", "deceptive"]
    partner_policies = ["circling", "switching", "erratic"]
    noise_levels = [0.0, 0.06]
    for scenario in scenarios:
        obstacles = 2 if scenario == "open_team" else 8
        target_speed = 2.25 if scenario == "open_team" else 1.85
        for target in target_policies:
            for partner in partner_policies:
                for noise in noise_levels:
                    for episode in range(episodes):
                        yield SimConfig(
                            seed=(
                                610_000
                                + episode
                                + 10_007 * scenarios.index(scenario)
                                + 20_011 * target_policies.index(target)
                                + 30_013 * partner_policies.index(partner)
                                + 40_009 * noise_levels.index(noise)
                            ),
                            scenario=scenario,
                            target_policy=target,
                            partner_policy=partner,
                            perception_noise=noise,
                            num_obstacles=obstacles,
                            target_speed=target_speed,
                        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("trans_project/results/unseen_partner_eval100"),
    )
    args = parser.parse_args()

    policies = policy_suite(
        (
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
        ("policy", "scenario", "target_policy", "partner_policy", "perception_noise"),
    )


if __name__ == "__main__":
    main()
