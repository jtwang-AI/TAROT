from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tarot_sim import FlatParamPolicy, SimConfig, run_episode


def train_configs(episodes: int, seed_offset: int):
    cases = [
        ("open_team", "evasive", "lazy", 2, 2.35, 0.0),
        ("standard", "evasive", "mixed", 8, 1.80, 0.0),
        ("standard", "stop_go", "mixed", 8, 1.80, 0.05),
        ("corridor", "deceptive", "noisy", 12, 1.80, 0.08),
    ]
    for scenario, target, partner, obstacles, speed, noise in cases:
        for ep in range(episodes):
            yield SimConfig(
                seed=70_000 + seed_offset + ep + len(scenario) * 101 + len(partner) * 17,
                scenario=scenario,
                target_policy=target,
                partner_policy=partner,
                num_obstacles=obstacles,
                target_speed=speed,
                perception_noise=noise,
            )


def score_policy(policy: FlatParamPolicy, episodes: int, seed_offset: int) -> float:
    rows = [run_episode(policy, cfg) for cfg in train_configs(episodes, seed_offset)]
    success = np.mean([r["success"] for r in rows])
    collision = np.mean([r["collision_count"] for r in rows])
    near = np.mean([r["near_miss_count"] for r in rows])
    steps = np.mean([r["steps"] for r in rows]) / 240.0
    shield = np.mean([r["safety_interventions"] for r in rows]) / 100.0
    return float(3.0 * success - 1.5 * collision - 0.08 * near - 0.15 * steps - 0.05 * shield)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--generations", type=int, default=12)
    parser.add_argument("--population", type=int, default=48)
    parser.add_argument("--elite-frac", type=float, default=0.2)
    parser.add_argument("--episodes", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, default=Path("trans_project/results/flat_cem"))
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    mean = np.array([1.0, 0.0, 0.0, 0.6, 1.0, 0.7], dtype=float)
    std = np.array([0.8, 0.6, 0.5, 0.8, 0.3, 0.4], dtype=float)
    history = []
    best_score = -1e9
    best_vec = mean.copy()
    elite_n = max(2, int(args.population * args.elite_frac))
    for gen in range(args.generations):
        candidates = rng.normal(mean, std, size=(args.population, len(mean)))
        candidates[:, 4] = np.clip(candidates[:, 4], 0.2, 1.5)
        candidates[:, 5] = np.clip(candidates[:, 5], 0.0, 1.5)
        scored = []
        for i, vec in enumerate(candidates):
            policy = FlatParamPolicy.from_vector(vec, name="flat_cem_candidate")
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
    params = {k: float(v) for k, v in zip(FlatParamPolicy.param_names, best_vec)}
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "best_policy.json").write_text(json.dumps({"params": params, "score": best_score}, indent=2))
    (args.out / "history.json").write_text(json.dumps(history, indent=2))
    print(json.dumps({"params": params, "score": best_score}, indent=2))


if __name__ == "__main__":
    main()
