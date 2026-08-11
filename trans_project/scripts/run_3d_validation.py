from __future__ import annotations

"""Dynamics-aware 3-D validation for TAROT.

The high-throughput experiments in :mod:`tarot_sim` deliberately use planar
single-integrator dynamics.  This script keeps the same high-level controller
interface but executes the commands through a delayed, acceleration-limited
3-D vehicle model with wind disturbances.  PyBullet is used for the 3-D scene
geometry and camera rendering.  The experiment is intentionally described as
3-D simulation, not as AirSim or physical-flight evidence.
"""

import argparse
import csv
from dataclasses import dataclass
import json
from pathlib import Path
import sys
from typing import Iterable

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from eval_common import policy_suite
from tarot_sim import SimConfig


@dataclass(frozen=True)
class Obstacle3D:
    kind: str
    x: float
    y: float
    sx: float
    sy: float
    height: float
    color: tuple[float, float, float, float]

    @property
    def controller_radius(self) -> float:
        if self.kind in {"cylinder", "tree", "tank"}:
            return self.sx
        return float(np.hypot(self.sx, self.sy))


@dataclass(frozen=True)
class Scene3D:
    name: str
    label: str
    obstacles: tuple[Obstacle3D, ...]
    wind: tuple[float, float, float]
    latency_steps: int
    observation_noise: float
    partner_mode: str
    ground_color: tuple[float, float, float, float]


def _box(x: float, y: float, hx: float, hy: float, height: float, color) -> Obstacle3D:
    return Obstacle3D("box", x, y, hx, hy, height, color)


def _cylinder(kind: str, x: float, y: float, radius: float, height: float, color) -> Obstacle3D:
    return Obstacle3D(kind, x, y, radius, radius, height, color)


def scenes() -> dict[str, Scene3D]:
    concrete = (0.52, 0.56, 0.61, 1.0)
    brick = (0.55, 0.27, 0.18, 1.0)
    trunk = (0.30, 0.18, 0.08, 1.0)
    blue = (0.16, 0.39, 0.66, 1.0)
    orange = (0.82, 0.34, 0.10, 1.0)
    metal = (0.55, 0.59, 0.62, 1.0)
    return {
        "urban": Scene3D(
            name="urban",
            label="Urban courtyard",
            obstacles=(
                _box(10.0, 28.0, 3.0, 5.5, 12.0, brick),
                _box(17.0, 9.0, 4.2, 2.8, 8.0, concrete),
                _box(27.5, 28.5, 4.0, 3.5, 15.0, concrete),
                _box(31.5, 12.0, 2.5, 5.0, 10.0, brick),
                _box(20.0, 22.0, 2.0, 2.0, 7.0, concrete),
            ),
            wind=(0.22, -0.13, 0.04),
            latency_steps=1,
            observation_noise=0.08,
            partner_mode="switching",
            ground_color=(0.36, 0.39, 0.42, 1.0),
        ),
        "forest": Scene3D(
            name="forest",
            label="Forest corridor",
            obstacles=tuple(
                _cylinder("tree", x, y, r, h, trunk)
                for x, y, r, h in (
                    (8, 11, 1.0, 9), (9, 27, 1.2, 11), (13, 19, 1.0, 10),
                    (17, 6, 1.1, 12), (18, 31, 1.0, 9), (22, 14, 1.2, 12),
                    (24, 25, 1.1, 10), (28, 7, 1.0, 9), (30, 20, 1.2, 12),
                    (33, 31, 1.1, 11), (35, 12, 0.9, 8),
                )
            ),
            wind=(-0.28, 0.19, 0.07),
            latency_steps=2,
            observation_noise=0.12,
            partner_mode="mixed",
            ground_color=(0.18, 0.35, 0.16, 1.0),
        ),
        "industrial": Scene3D(
            name="industrial",
            label="Industrial yard",
            obstacles=(
                _box(9.0, 14.0, 4.2, 1.5, 3.0, blue),
                _box(10.0, 28.0, 4.2, 1.5, 5.5, orange),
                _box(19.0, 34.0, 5.0, 1.7, 3.0, blue),
                _box(29.0, 7.0, 5.0, 1.7, 5.5, orange),
                _cylinder("tank", 26.0, 24.0, 2.7, 8.0, metal),
                _cylinder("tank", 33.0, 29.0, 2.3, 7.0, metal),
                _box(18.0, 27.0, 1.4, 3.2, 6.0, blue),
            ),
            wind=(0.34, 0.08, 0.06),
            latency_steps=1,
            observation_noise=0.10,
            partner_mode="dropout",
            ground_color=(0.42, 0.38, 0.31, 1.0),
        ),
    }


class ControllerAdapter:
    def __init__(self, cfg: SimConfig):
        self.cfg = cfg
        self.t = 0
        self.prev_roles = ["none"] * cfg.num_learners


def _clip_norm(vec: np.ndarray, limit: float) -> np.ndarray:
    norm = float(np.linalg.norm(vec))
    if norm <= limit or norm < 1e-12:
        return vec
    return vec * (limit / norm)


def _unit_to(target: np.ndarray, source: np.ndarray, speed: float) -> np.ndarray:
    return _clip_norm(target - source, speed)


def _obstacle_clearance(position: np.ndarray, obstacle: Obstacle3D, drone_radius: float) -> float:
    if position[2] - drone_radius > obstacle.height:
        return np.inf
    if obstacle.kind in {"cylinder", "tree", "tank"}:
        return float(np.linalg.norm(position[:2] - np.array([obstacle.x, obstacle.y]))) - obstacle.sx - drone_radius
    dx = max(abs(position[0] - obstacle.x) - obstacle.sx, 0.0)
    dy = max(abs(position[1] - obstacle.y) - obstacle.sy, 0.0)
    if dx == 0.0 and dy == 0.0:
        inside = min(obstacle.sx - abs(position[0] - obstacle.x), obstacle.sy - abs(position[1] - obstacle.y))
        return -inside - drone_radius
    return float(np.hypot(dx, dy)) - drone_radius


def _repel_from_obstacles(position: np.ndarray, desired: np.ndarray, scene: Scene3D, gain: float = 2.0) -> np.ndarray:
    out = desired.copy()
    for obstacle in scene.obstacles:
        center = np.array([obstacle.x, obstacle.y])
        radius = obstacle.controller_radius
        rel = position[:2] - center
        distance = float(np.linalg.norm(rel))
        margin = radius + 2.3
        if distance < margin and distance > 1e-9:
            out += rel / distance * gain * (margin - distance)
    return out


class DynamicsEpisode:
    def __init__(self, scene: Scene3D, policy, seed: int, horizon: int = 320):
        self.scene = scene
        self.policy = policy
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self.dt = 0.12
        self.horizon = horizon
        self.drone_radius = 0.38
        self.num_learners = 2
        self.num_partners = 2
        self.cfg = SimConfig(
            seed=seed,
            world_size=40.0,
            num_learners=2,
            num_partners=2,
            num_targets=1,
            num_obstacles=len(scene.obstacles),
            horizon=horizon,
            dt=self.dt,
            drone_speed=3.2,
            partner_speed=2.8,
            target_speed=1.7,
            capture_radius=1.60,
            collision_radius=self.drone_radius,
            near_miss_radius=1.0,
            perception_noise=scene.observation_noise,
            target_policy="deceptive",
            partner_policy=scene.partner_mode,
            scenario=f"3d_{scene.name}",
        )
        self.adapter = ControllerAdapter(self.cfg)
        jitter = self.rng.normal(0.0, 0.35, size=(5, 2))
        self.positions = np.array(
            [
                [3.2, 6.0, 3.1], [3.4, 34.0, 4.0],
                [36.4, 6.5, 3.5], [36.5, 34.0, 4.4],
                [20.0, 20.0, 3.7],
            ],
            dtype=float,
        )
        self.positions[:, :2] += jitter
        self.velocities = np.zeros_like(self.positions)
        # ``latency_steps`` is the sensing delay.  The actuator path adds one
        # further controller-to-execution handoff cycle, so commands are
        # executed after latency_steps + 1 simulation steps.
        self.command_queue = [np.zeros((2, 2), dtype=float) for _ in range(scene.latency_steps + 1)]
        self.history = [self.positions.copy() for _ in range(scene.latency_steps + 1)]
        self.stats = {
            "collision_count": 0,
            "partner_collision_count": 0,
            "near_miss_count": 0,
            "safety_interventions": 0,
            "energy": 0.0,
            "role_switches": 0,
            "min_clearance": np.inf,
        }

    def observation(self) -> dict[str, np.ndarray]:
        sensed = self.history[0].copy()
        sensed[:, :2] += self.rng.normal(0.0, self.scene.observation_noise, size=(5, 2))
        return {
            "learners": sensed[:2, :2],
            "partners": sensed[2:4, :2],
            "targets": sensed[4:5, :2],
            "captured": np.array([False]),
            "obstacles": np.array(
                # A conservative 0.8 m footprint inflation accounts for the
                # delayed dynamic execution that is absent from the 2-D model.
                [[o.x, o.y, o.controller_radius + 0.8] for o in self.scene.obstacles], dtype=float
            ),
            "t": np.array([self.adapter.t], dtype=float),
        }

    def target_command(self) -> np.ndarray:
        target = self.positions[4]
        team = self.positions[:4]
        nearest = team[np.argmin(np.linalg.norm(team - target[None, :], axis=1))]
        flee = _unit_to(target[:2], nearest[:2], self.cfg.target_speed)
        tangent = np.array([-flee[1], flee[0]])
        desired = 0.70 * flee + 0.30 * tangent
        desired = _repel_from_obstacles(target, desired, self.scene, gain=1.4)
        if target[0] < 3.0 or target[0] > 37.0:
            desired[0] *= -1.0
        if target[1] < 3.0 or target[1] > 37.0:
            desired[1] *= -1.0
        return _clip_norm(desired, self.cfg.target_speed)

    def partner_commands(self) -> np.ndarray:
        target = self.positions[4]
        out = []
        for index, partner in enumerate(self.positions[2:4]):
            mode = self.scene.partner_mode
            if mode == "switching":
                mode = ("greedy", "flank", "lazy")[(self.adapter.t // 45 + index) % 3]
            elif mode == "mixed":
                mode = ("greedy", "flank")[index]
            elif mode == "dropout":
                mode = "lazy" if index == 1 and self.adapter.t > self.horizon // 3 else ("greedy", "flank")[index]
            if mode == "lazy":
                desired = 0.22 * _unit_to(target[:2], partner[:2], self.cfg.partner_speed)
            elif mode == "flank":
                center = self.positions[:4, :2].mean(axis=0)
                radial = _unit_to(target[:2], center, 1.0)
                tangent = np.array([-radial[1], radial[0]])
                waypoint = target[:2] + tangent * (2.6 if index == 0 else -2.6)
                desired = _unit_to(waypoint, partner[:2], self.cfg.partner_speed)
            else:
                desired = _unit_to(target[:2], partner[:2], self.cfg.partner_speed)
            desired = _repel_from_obstacles(partner, desired, self.scene, gain=1.8)
            out.append(_clip_norm(desired, self.cfg.partner_speed))
        return np.vstack(out)

    def _advance_group(
        self,
        indices: Iterable[int],
        horizontal_commands: np.ndarray,
        target_altitudes: np.ndarray,
        max_speed: float,
        max_acceleration: float,
        tau: float,
    ) -> None:
        wind_base = np.asarray(self.scene.wind, dtype=float)
        for local_index, body_index in enumerate(indices):
            phase = 0.17 * self.seed + 0.8 * body_index
            gust = np.array(
                [
                    0.30 * np.sin(self.adapter.t / 17.0 + phase),
                    0.22 * np.cos(self.adapter.t / 23.0 + phase),
                    0.08 * np.sin(self.adapter.t / 13.0 + phase),
                ]
            )
            desired_velocity = np.array(
                [
                    horizontal_commands[local_index, 0],
                    horizontal_commands[local_index, 1],
                    np.clip(1.4 * (target_altitudes[local_index] - self.positions[body_index, 2]), -1.2, 1.2),
                ]
            )
            acceleration = (desired_velocity - self.velocities[body_index]) / tau + wind_base + gust
            acceleration = _clip_norm(acceleration, max_acceleration)
            self.velocities[body_index] += acceleration * self.dt
            self.velocities[body_index] = _clip_norm(self.velocities[body_index], max_speed)
            self.positions[body_index] += self.velocities[body_index] * self.dt
            self.stats["energy"] += float(np.dot(acceleration, acceleration) * self.dt)

    def advance(self, learner_command: np.ndarray, roles: list[str], interventions: int) -> tuple[bool, bool]:
        self.command_queue.append(learner_command.copy())
        delayed = self.command_queue.pop(0)
        partner = self.partner_commands()
        target = self.target_command()[None, :]

        self._advance_group(range(0, 2), delayed, np.array([3.1, 4.0]), 3.5, 5.2, 0.36)
        self._advance_group(range(2, 4), partner, np.array([3.5, 4.4]), 3.2, 4.2, 0.48)
        self._advance_group([4], target, np.array([3.7]), 2.0, 3.0, 0.55)

        self.positions[:, 0] = np.clip(self.positions[:, 0], 0.3, 39.7)
        self.positions[:, 1] = np.clip(self.positions[:, 1], 0.3, 39.7)
        self.positions[:, 2] = np.clip(self.positions[:, 2], 1.5, 7.5)
        self.stats["safety_interventions"] += interventions
        self.stats["role_switches"] += sum(a != b for a, b in zip(roles, self.adapter.prev_roles))
        self.adapter.prev_roles = list(roles)

        collision = False
        near_miss = False
        # The primary safety outcome follows the controlled aircraft.  Contacts
        # by externally controlled partners are retained separately because the
        # evaluated policy cannot alter their commands.
        for body_index, position in enumerate(self.positions[:4]):
            for obstacle in self.scene.obstacles:
                clearance = _obstacle_clearance(position, obstacle, self.drone_radius)
                self.stats["min_clearance"] = min(self.stats["min_clearance"], clearance)
                if clearance < 0.0:
                    if body_index < self.num_learners:
                        self.stats["collision_count"] += 1
                        collision = True
                    else:
                        self.stats["partner_collision_count"] += 1
                elif clearance < 0.65:
                    self.stats["near_miss_count"] += 1
                    near_miss = True
        for i in range(4):
            for j in range(i + 1, 4):
                clearance = float(np.linalg.norm(self.positions[i] - self.positions[j])) - 2.0 * self.drone_radius
                self.stats["min_clearance"] = min(self.stats["min_clearance"], clearance)
                if clearance < 0.0:
                    if i < self.num_learners or j < self.num_learners:
                        self.stats["collision_count"] += 1
                        collision = True
                    else:
                        self.stats["partner_collision_count"] += 1
                elif clearance < 0.65:
                    self.stats["near_miss_count"] += 1
                    near_miss = True

        distance = np.linalg.norm(self.positions[:4] - self.positions[4][None, :], axis=1)
        success = bool(np.sum(distance < self.cfg.capture_radius) >= 2 and not collision)
        self.history.append(self.positions.copy())
        self.history.pop(0)
        self.adapter.t += 1
        return success, collision

    def run(self, retain_trajectory: bool = False) -> tuple[dict, np.ndarray | None]:
        self.policy.reset()
        trajectory = [self.positions.copy()] if retain_trajectory else None
        success = False
        collision = False
        while self.adapter.t < self.horizon and not success and not collision:
            obs = self.observation()
            action, roles, interventions = self.policy.act(obs, self.adapter)
            success, collision = self.advance(action, roles, interventions)
            if trajectory is not None:
                trajectory.append(self.positions.copy())
        steps = self.adapter.t
        row = {
            "policy": self.policy.name,
            "scene": self.scene.name,
            "scene_label": self.scene.label,
            "partner_mode": self.scene.partner_mode,
            "seed": self.seed,
            "success": float(success),
            "collision_episode": float(collision),
            "timeout": float(not success and not collision),
            "steps": steps,
            "collision_count": self.stats["collision_count"],
            "partner_collision_count": self.stats["partner_collision_count"],
            "near_miss_count": self.stats["near_miss_count"],
            "safety_interventions": self.stats["safety_interventions"],
            "activation_rate": self.stats["safety_interventions"] / max(steps * self.num_learners, 1),
            "energy": self.stats["energy"],
            "role_switches": self.stats["role_switches"],
            "min_clearance": float(self.stats["min_clearance"]),
            "latency_steps": self.scene.latency_steps,
            "observation_delay_steps": self.scene.latency_steps,
            "command_delay_steps": self.scene.latency_steps + 1,
            "observation_noise": self.scene.observation_noise,
            "wind_magnitude": float(np.linalg.norm(self.scene.wind)),
        }
        return row, None if trajectory is None else np.asarray(trajectory)


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def aggregate(rows: list[dict]) -> list[dict]:
    groups: dict[tuple[str, str], list[dict]] = {}
    for row in rows:
        groups.setdefault((str(row["scene"]), str(row["policy"])), []).append(row)
    out = []
    metrics = [
        "success", "collision_episode", "timeout", "steps", "activation_rate",
        "energy", "role_switches", "min_clearance",
    ]
    for (scene, policy), values in sorted(groups.items()):
        record = {"scene": scene, "policy": policy, "episodes": len(values)}
        for metric in metrics:
            array = np.asarray([float(v[metric]) for v in values])
            record[f"{metric}_mean"] = float(np.mean(array))
            record[f"{metric}_std"] = float(np.std(array))
        successful_steps = [float(v["steps"]) for v in values if float(v["success"]) == 1.0]
        record["successful_steps_mean"] = float(np.mean(successful_steps)) if successful_steps else None
        out.append(record)
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=100, help="matched seeds per scene and policy")
    # The 830k block is disjoint from all development/smoke runs (730k).
    parser.add_argument("--seed-base", type=int, default=830_000)
    parser.add_argument("--horizon", type=int, default=320)
    parser.add_argument("--retain", type=int, default=12, help="seeds per scene/policy retained as trajectories")
    parser.add_argument("--out", type=Path, default=ROOT / "results" / "dynamics3d_eval100")
    args = parser.parse_args()

    selected_policies = policy_suite(
        ("tarot_no_safety", "tarot_default", "tarot_tuned_cem", "flat_cem_tuned")
    )
    all_scenes = scenes()
    rows: list[dict] = []
    trajectory_dir = args.out / "trajectories"
    trajectory_dir.mkdir(parents=True, exist_ok=True)
    for scene_index, scene in enumerate(all_scenes.values()):
        for episode_index in range(args.episodes):
            seed = args.seed_base + 10_003 * scene_index + episode_index
            for policy in selected_policies:
                runner = DynamicsEpisode(scene, policy, seed, horizon=args.horizon)
                row, trajectory = runner.run(retain_trajectory=episode_index < args.retain)
                rows.append(row)
                if trajectory is not None:
                    np.savez_compressed(
                        trajectory_dir / f"{scene.name}__{policy.name}__{seed}.npz",
                        trajectory=trajectory,
                        scene=scene.name,
                        policy=policy.name,
                        seed=seed,
                    )
        print(f"finished {scene.label}: {args.episodes * len(selected_policies)} episodes", flush=True)

    write_csv(args.out / "episodes.csv", rows)
    aggregates = aggregate(rows)
    write_csv(args.out / "aggregate.csv", aggregates)
    (args.out / "aggregate.json").write_text(json.dumps(aggregates, indent=2))
    manifest = {
        "description": "Dynamics-aware 3-D validation with delayed acceleration-limited execution and PyBullet rendering",
        "episodes_per_scene_policy": args.episodes,
        "total_episodes": len(rows),
        "scenes": [scene.name for scene in all_scenes.values()],
        "policies": [policy.name for policy in selected_policies],
        "seed_base": args.seed_base,
        "horizon": args.horizon,
        "dt": 0.12,
        "observation_delay_steps": [1, 2],
        "command_delay_steps": [2, 3],
        "controller_interface": "2-D horizontal velocity commands executed by a 3-D dynamic model",
        "limitations": "No image-based perception, aerodynamic rotor model, AirSim, or physical flight",
    }
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"wrote {len(rows)} episodes to {args.out}", flush=True)


if __name__ == "__main__":
    main()
