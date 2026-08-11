from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tarot_sim import SimConfig, TarotPolicy, run_episode


PARAM_NAMES = [
    "safety_budget",
    "reliability_threshold",
    "risk_clearance_scale",
    "intercept_mix",
    "intercept_offset",
    "flank_mix",
    "flank_offset",
]


def policy_from_vec(vec, name="tarot_cem_candidate"):
    vals = dict(zip(PARAM_NAMES, [float(x) for x in vec]))
    vals["safety_budget"] = float(np.clip(vals["safety_budget"], 0.25, 1.5))
    vals["reliability_threshold"] = float(np.clip(vals["reliability_threshold"], 0.05, 0.75))
    vals["risk_clearance_scale"] = float(np.clip(vals["risk_clearance_scale"], 1.0, 4.0))
    vals["intercept_mix"] = float(np.clip(vals["intercept_mix"], 0.0, 0.8))
    vals["intercept_offset"] = float(np.clip(vals["intercept_offset"], 0.4, 4.0))
    vals["flank_mix"] = float(np.clip(vals["flank_mix"], 0.0, 0.9))
    vals["flank_offset"] = float(np.clip(vals["flank_offset"], 0.5, 5.0))
    return TarotPolicy(**vals), vals


def train_configs(episodes: int, seed_offset: int):
    cases = [
        ("open_team", "evasive", "lazy", 2, 2.35, 0.0),
        ("open_team", "deceptive", "blocker", 2, 2.35, 0.0),
        ("standard", "evasive", "mixed", 8, 1.80, 0.0),
        ("standard", "stop_go", "mixed", 8, 1.80, 0.05),
        ("standard", "evasive", "lazy", 8, 1.80, 0.0),
        ("corridor", "deceptive", "noisy", 12, 1.80, 0.08),
    ]
    for scenario, target, partner, obstacles, speed, noise in cases:
        for ep in range(episodes):
            yield SimConfig(
                seed=90_000 + seed_offset + ep + len(scenario) * 101 + len(partner) * 17,
                scenario=scenario,
                target_policy=target,
                partner_policy=partner,
                num_obstacles=obstacles,
                target_speed=speed,
                perception_noise=noise,
            )


def score_policy(policy: TarotPolicy, episodes: int, seed_offset: int) -> float:
    rows = [run_episode(policy, cfg) for cfg in train_configs(episodes, seed_offset)]
    success = np.mean([r["success"] for r in rows])
    collision = np.mean([r["collision_count"] for r in rows])
    near = np.mean([r["near_miss_count"] for r in rows])
    steps = np.mean([r["steps"] for r in rows]) / 240.0
    shield = np.mean([r["safety_interventions"] for r in rows]) / 100.0
    return float(3.2 * success - 1.6 * collision - 0.05 * near - 0.12 * steps - 0.20 * shield)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--generations", type=int, default=12)
    parser.add_argument("--population", type=int, default=48)
    parser.add_argument("--elite-frac", type=float, default=0.2)
    parser.add_argument("--episodes", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, default=Path("trans_project/results/tarot_cem"))
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    mean = np.array([0.85, 0.30, 2.5, 0.35, 1.8, 0.45, 2.5], dtype=float)
    std = np.array([0.35, 0.20, 0.75, 0.20, 0.75, 0.25, 0.9], dtype=float)
    elite_n = max(2, int(args.population * args.elite_frac))
    best_score = -1e9
    best_vec = mean.copy()
    history = []
    for gen in range(args.generations):
        candidates = rng.normal(mean, std, size=(args.population, len(mean)))
        scored = []
        for i, vec in enumerate(candidates):
            policy, _ = policy_from_vec(vec)
            score = score_policy(policy, args.episodes, seed_offset=gen * 1000 + i * 19)
            scored.append((score, vec))
        scored.sort(key=lambda x: x[0], reverse=True)
        elites = np.vstack([v for _, v in scored[:elite_n]])
        mean = elites.mean(axis=0)
        std = elites.std(axis=0) + 1e-3
        if scored[0][0] > best_score:
            best_score, best_vec = scored[0]
        rec = {"generation": gen, "best_score": float(scored[0][0]), "mean_score": float(np.mean([s for s, _ in scored]))}
        history.append(rec)
        print(f"generation={gen} best={rec['best_score']:.4f} mean={rec['mean_score']:.4f}")
    _, params = policy_from_vec(best_vec, name="tarot_cem_best")
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "best_policy.json").write_text(json.dumps({"params": params, "score": best_score}, indent=2))
    (args.out / "history.json").write_text(json.dumps(history, indent=2))
    print(json.dumps({"params": params, "score": best_score}, indent=2))


if __name__ == "__main__":
    main()
