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
    scenarios = ["open_team", "standard"]
    target_policies = ["evasive", "deceptive"]
    partner_policies = ["greedy", "flank", "blocker", "lazy", "noisy", "mixed"]
    partner_speeds = [1.4, 1.8, 2.2, 2.6, 3.0, 3.4]
    noise_levels = [0.0, 0.06]
    for scenario in scenarios:
        for target in target_policies:
            for partner in partner_policies:
                for speed in partner_speeds:
                    for noise in noise_levels:
                        obstacles = 2 if scenario == "open_team" else 8
                        target_speed = 2.25 if scenario == "open_team" else 1.85
                        for ep in range(episodes):
                            yield SimConfig(
                                seed=210_000
                                + ep
                                + 101 * len(scenario)
                                + 1009 * len(target)
                                + 9176 * len(partner)
                                + int(speed * 1000)
                                + int(noise * 10_000),
                                scenario=scenario,
                                target_policy=target,
                                partner_policy=partner,
                                perception_noise=noise,
                                num_obstacles=obstacles,
                                target_speed=target_speed,
                                partner_speed=speed,
                            )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--out", type=Path, default=Path("trans_project/results/partner_reliability_eval50"))
    args = parser.parse_args()

    policies = policy_suite(
        (
            "safety_only",
            "tarot_no_teammate",
            "tarot_no_risk_gate",
            "tarot_instant_belief",
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
            row["partner_speed"] = float(cfg.partner_speed)
            row["target_speed"] = float(cfg.target_speed)
            rows.append(row)
    save_run(
        args.out,
        rows,
        ("policy", "scenario", "target_policy", "partner_policy", "partner_speed", "perception_noise"),
    )


if __name__ == "__main__":
    main()
