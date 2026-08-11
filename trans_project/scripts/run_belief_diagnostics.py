from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean, pstdev
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tarot_sim import OpenTeamPursuitEnv, SimConfig, TarotPolicy, run_episode


TRUE_TO_MODE = {
    "greedy": "chaser",
    "flank": "flanker",
    "blocker": "blocker",
    "lazy": "lazy",
    "noisy": "noisy",
}


METRICS = [
    "mode_accuracy",
    "actionable_mode_accuracy",
    "risk_gated",
    "uncertain",
    "confidence",
    "reliability",
    "safety_interventions",
]


def configs(episodes: int):
    scenarios = [
        ("open_team", 2, 2.25),
        ("standard", 8, 1.85),
        ("corridor", 12, 1.85),
    ]
    partner_policies = ["greedy", "flank", "blocker", "lazy", "noisy"]
    noise_levels = [0.0, 0.06, 0.12]
    for scenario, obstacles, speed in scenarios:
        for partner in partner_policies:
            for noise in noise_levels:
                for ep in range(episodes):
                    yield SimConfig(
                        seed=410_000 + ep + 101 * len(scenario) + 9176 * len(partner) + int(noise * 10_000),
                        scenario=scenario,
                        target_policy="evasive",
                        partner_policy=partner,
                        num_obstacles=obstacles,
                        target_speed=speed,
                        perception_noise=noise,
                    )


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def run_trace(policy: TarotPolicy, cfg: SimConfig) -> tuple[list[dict], dict]:
    policy.reset()
    env = OpenTeamPursuitEnv(cfg)
    obs = env.reset()
    done = False
    rows = []
    true_mode = TRUE_TO_MODE[cfg.partner_policy]
    total_interventions = 0
    while not done:
        action, roles, interventions = policy.act(obs, env)
        pred = policy.partner_mode
        actionable = pred not in {"risk_gated", "uncertain", "unknown"}
        rows.append(
            {
                "policy": policy.name,
                "scenario": cfg.scenario,
                "partner_policy": cfg.partner_policy,
                "true_mode": true_mode,
                "pred_mode": pred,
                "seed": float(cfg.seed),
                "t": float(env.t),
                "perception_noise": float(cfg.perception_noise),
                "num_obstacles": float(cfg.num_obstacles),
                "mode_accuracy": float(pred == true_mode),
                "actionable_mode_accuracy": float(pred == true_mode) if actionable else "",
                "risk_gated": float(pred == "risk_gated"),
                "uncertain": float(pred == "uncertain"),
                "confidence": float(policy.last_confidence),
                "reliability": float(policy.last_reliability),
                "safety_interventions": float(interventions),
                "roles": "|".join(roles),
            }
        )
        total_interventions += interventions
        obs, stats, done = env.step(action, roles)
        stats.safety_interventions += interventions
    episode = run_episode(policy, cfg)
    episode["diagnostic_interventions"] = float(total_interventions)
    return rows, episode


def summarize(rows: list[dict]) -> list[dict]:
    groups = defaultdict(list)
    for row in rows:
        groups[(row["policy"], row["scenario"], row["partner_policy"], row["perception_noise"])].append(row)
    out = []
    for key, items in sorted(groups.items()):
        rec = {
            "policy": key[0],
            "scenario": key[1],
            "partner_policy": key[2],
            "perception_noise": key[3],
            "steps": len(items),
        }
        for metric in METRICS:
            vals = []
            for row in items:
                val = row[metric]
                if val != "":
                    vals.append(float(val))
            rec[f"{metric}_mean"] = mean(vals) if vals else 0.0
            rec[f"{metric}_std"] = pstdev(vals) if len(vals) > 1 else 0.0
        out.append(rec)
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--out", type=Path, default=Path("trans_project/results/belief_diagnostics_eval50"))
    args = parser.parse_args()

    policies = []
    full = TarotPolicy()
    full.name = "tarot_default"
    instant = TarotPolicy(belief="instant")
    instant.name = "tarot_instant_belief"
    policies.extend([full, instant])

    step_rows = []
    episode_rows = []
    for cfg in configs(args.episodes):
        for policy in policies:
            trace, episode = run_trace(policy, cfg)
            step_rows.extend(trace)
            episode_rows.append(episode)

    args.out.mkdir(parents=True, exist_ok=True)
    write_csv(args.out / "belief_steps.csv", step_rows)
    write_csv(args.out / "episodes.csv", episode_rows)
    write_csv(args.out / "belief_aggregate.csv", summarize(step_rows))
    (args.out / "belief_aggregate.json").write_text(json.dumps(summarize(step_rows), indent=2))
    print(f"wrote {len(step_rows)} belief step rows")
    print(f"wrote {len(episode_rows)} episode rows")
    print(f"aggregate: {args.out / 'belief_aggregate.csv'}")


if __name__ == "__main__":
    main()
