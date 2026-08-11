from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple

import numpy as np


Array = np.ndarray


@dataclass
class SimConfig:
    seed: int = 0
    world_size: float = 40.0
    num_learners: int = 2
    num_partners: int = 2
    num_targets: int = 1
    num_obstacles: int = 8
    horizon: int = 240
    dt: float = 0.15
    drone_speed: float = 3.0
    partner_speed: float = 2.6
    target_speed: float = 2.2
    capture_radius: float = 1.2
    collision_radius: float = 0.45
    near_miss_radius: float = 0.8
    obstacle_radius_min: float = 0.8
    obstacle_radius_max: float = 1.8
    perception_noise: float = 0.0
    target_policy: str = "evasive"
    partner_policy: str = "mixed"
    scenario: str = "standard"


@dataclass
class StepStats:
    progress: float
    collisions: int
    near_misses: int
    safety_interventions: int
    energy: float
    role_switches: int


def _norm(vec: Array, eps: float = 1e-9) -> float:
    return float(np.linalg.norm(vec) + eps)


def _clip_norm(vec: Array, max_norm: float) -> Array:
    n = _norm(vec)
    if n <= max_norm:
        return vec
    return vec / n * max_norm


def _unit_to(target: Array, source: Array, speed: float) -> Array:
    return _clip_norm(target - source, speed)


class OpenTeamPursuitEnv:
    """A compact open-teaming pursuit simulator for fast statistical tests.

    The environment is new code. It is not intended to replace AirSim; it is the
    high-throughput layer used to test claims before high-fidelity validation.
    """

    def __init__(self, cfg: SimConfig):
        self.cfg = cfg
        self.rng = np.random.default_rng(cfg.seed)
        self.t = 0
        self.learners = np.zeros((cfg.num_learners, 2), dtype=float)
        self.partners = np.zeros((cfg.num_partners, 2), dtype=float)
        self.targets = np.zeros((cfg.num_targets, 2), dtype=float)
        self.obstacles = np.zeros((cfg.num_obstacles, 3), dtype=float)
        self.prev_actions = np.zeros((cfg.num_learners, 2), dtype=float)
        self.prev_roles: List[str] = ["none"] * cfg.num_learners
        self.captured = np.zeros(cfg.num_targets, dtype=bool)
        self.reset()

    @property
    def team(self) -> Array:
        return np.vstack([self.learners, self.partners])

    def reset(self) -> Dict[str, Array]:
        c = self.cfg
        self.t = 0
        center = c.world_size / 2.0
        start_radius = c.world_size * 0.35
        learner_angles = np.linspace(np.pi * 0.8, np.pi * 1.2, c.num_learners)
        partner_angles = np.linspace(-np.pi * 0.2, np.pi * 0.2, c.num_partners)
        self.learners = np.column_stack(
            [center + start_radius * np.cos(learner_angles), center + start_radius * np.sin(learner_angles)]
        )
        self.partners = np.column_stack(
            [center + start_radius * np.cos(partner_angles), center + start_radius * np.sin(partner_angles)]
        )
        self.learners += self.rng.normal(0.0, 0.7, size=self.learners.shape)
        self.partners += self.rng.normal(0.0, 0.7, size=self.partners.shape)
        self.targets = self.rng.uniform(c.world_size * 0.35, c.world_size * 0.65, size=(c.num_targets, 2))
        protected = np.vstack([self.learners, self.partners, self.targets])

        def valid_obstacles(candidates: Array) -> Array:
            kept = []
            for ox, oy, rad in candidates:
                point = np.array([ox, oy])
                if np.min(np.linalg.norm(protected - point[None, :], axis=1)) < rad + 3.0:
                    continue
                kept.append([ox, oy, rad])
            if not kept:
                return np.empty((0, 3), dtype=float)
            return np.asarray(kept, dtype=float).reshape(-1, 3)

        if c.scenario == "corridor":
            xs = np.linspace(c.world_size * 0.25, c.world_size * 0.75, c.num_obstacles)
            ys = center + self.rng.choice([-1.0, 1.0], size=c.num_obstacles) * self.rng.uniform(2.0, 4.0, c.num_obstacles)
            radii = self.rng.uniform(c.obstacle_radius_min, c.obstacle_radius_max, c.num_obstacles)
            candidates = np.column_stack([xs, ys, radii])
        else:
            candidates = np.empty((0, 3), dtype=float)
            while len(candidates) < c.num_obstacles * 4:
                batch = np.column_stack(
                    [
                        self.rng.uniform(4.0, c.world_size - 4.0, size=c.num_obstacles),
                        self.rng.uniform(4.0, c.world_size - 4.0, size=c.num_obstacles),
                        self.rng.uniform(c.obstacle_radius_min, c.obstacle_radius_max, size=c.num_obstacles),
                    ]
                )
                candidates = np.vstack([candidates, batch])
        filtered = valid_obstacles(candidates)
        if len(filtered) < c.num_obstacles:
            filler = np.column_stack(
                [
                    self.rng.uniform(4.0, c.world_size - 4.0, size=c.num_obstacles * 3),
                    self.rng.uniform(4.0, c.world_size - 4.0, size=c.num_obstacles * 3),
                    self.rng.uniform(c.obstacle_radius_min, c.obstacle_radius_max, size=c.num_obstacles * 3),
                ]
            )
            filtered = np.vstack([filtered, valid_obstacles(filler)])
        self.obstacles = filtered[: c.num_obstacles]
        self.prev_actions = np.zeros((c.num_learners, 2), dtype=float)
        self.prev_roles = ["none"] * c.num_learners
        self.captured = np.zeros(c.num_targets, dtype=bool)
        return self.observe()

    def observe(self) -> Dict[str, Array]:
        noise = self.cfg.perception_noise
        def maybe_noisy(x: Array) -> Array:
            if noise <= 0:
                return x.copy()
            return x + self.rng.normal(0.0, noise, size=x.shape)

        return {
            "learners": maybe_noisy(self.learners),
            "partners": maybe_noisy(self.partners),
            "targets": maybe_noisy(self.targets),
            "captured": self.captured.copy(),
            "obstacles": self.obstacles.copy(),
            "t": np.array([self.t], dtype=float),
        }

    def _target_actions(self) -> Array:
        c = self.cfg
        actions = []
        for target in self.targets:
            if c.target_policy == "static":
                actions.append(np.zeros(2))
                continue
            team = self.team
            nearest = team[np.argmin(np.linalg.norm(team - target[None, :], axis=1))]
            flee = _unit_to(target, nearest, c.target_speed)
            if c.target_policy == "deceptive":
                tangent = np.array([-flee[1], flee[0]])
                act = 0.65 * flee + 0.35 * tangent
            elif c.target_policy == "stop_go":
                act = flee if (self.t // 30) % 2 == 0 else np.zeros(2)
            else:
                act = flee
            actions.append(_clip_norm(act, c.target_speed))
        return np.vstack(actions)

    def _partner_actions(self) -> Array:
        c = self.cfg
        actions = []
        target = self.targets[0]
        for i, partner in enumerate(self.partners):
            mode = c.partner_policy
            if mode == "mixed":
                modes = ["greedy", "flank", "lazy", "noisy"]
                mode = modes[(c.seed + i) % len(modes)]
            elif mode == "switching":
                # Held-out non-stationary teammate: the active behavior changes
                # during an episode and is therefore absent from TAROT's static
                # prototype family.
                modes = ["greedy", "flank", "lazy"]
                mode = modes[((self.t // 30) + i) % len(modes)]
            if mode == "lazy":
                act = 0.35 * _unit_to(target, partner, c.partner_speed)
            elif mode == "flank":
                center = self.team.mean(axis=0)
                radial = _unit_to(target, center, 1.0)
                tangent = np.array([-radial[1], radial[0]])
                desired = target + tangent * (2.5 if i % 2 == 0 else -2.5)
                act = _unit_to(desired, partner, c.partner_speed)
            elif mode == "noisy":
                act = _unit_to(target, partner, c.partner_speed)
                act += self.rng.normal(0.0, c.partner_speed * 0.35, size=2)
            elif mode == "circling":
                # Held-out orbiting behavior. Its direction is tangential to the
                # target rather than any of the five inference prototypes.
                radial = target - partner
                tangent = np.array([-radial[1], radial[0]])
                act = _clip_norm(tangent, c.partner_speed)
            elif mode == "erratic":
                # Held-out unstructured behavior used to test graceful fallback.
                angle = self.rng.uniform(-np.pi, np.pi)
                act = c.partner_speed * np.array([np.cos(angle), np.sin(angle)])
            elif mode == "blocker":
                escape = _unit_to(target, self.team.mean(axis=0), 1.0)
                act = _unit_to(target + escape * 3.0, partner, c.partner_speed)
            else:
                act = _unit_to(target, partner, c.partner_speed)
            actions.append(_clip_norm(act, c.partner_speed))
        return np.vstack(actions) if actions else np.zeros((0, 2))

    def step(self, learner_actions: Array, roles: Iterable[str] | None = None) -> Tuple[Dict[str, Array], StepStats, bool]:
        c = self.cfg
        learner_actions = np.asarray(learner_actions, dtype=float).reshape(c.num_learners, 2)
        learner_actions = np.vstack([_clip_norm(a, c.drone_speed) for a in learner_actions])
        partner_actions = self._partner_actions()
        target_actions = self._target_actions()
        prev_dist = float(np.min(np.linalg.norm(self.team[:, None, :] - self.targets[None, :, :], axis=2)))

        self.learners = np.clip(self.learners + learner_actions * c.dt, 0.0, c.world_size)
        self.partners = np.clip(self.partners + partner_actions * c.dt, 0.0, c.world_size)
        self.targets = np.clip(self.targets + target_actions * c.dt, 0.0, c.world_size)

        team = self.team
        for j, target in enumerate(self.targets):
            close = np.sum(np.linalg.norm(team - target[None, :], axis=1) < c.capture_radius)
            if close >= 2:
                self.captured[j] = True
        new_dist = float(np.min(np.linalg.norm(team[:, None, :] - self.targets[None, :, :], axis=2)))
        progress = max(0.0, prev_dist - new_dist)

        collisions = 0
        near_misses = 0
        for drone in team:
            for ox, oy, rad in self.obstacles:
                d = _norm(drone - np.array([ox, oy]))
                if d < rad + c.collision_radius:
                    collisions += 1
                elif d < rad + c.near_miss_radius:
                    near_misses += 1
        for i in range(len(team)):
            for j in range(i + 1, len(team)):
                d = _norm(team[i] - team[j])
                if d < c.collision_radius:
                    collisions += 1
                elif d < c.near_miss_radius:
                    near_misses += 1

        role_list = list(roles) if roles is not None else ["none"] * c.num_learners
        role_switches = sum(a != b for a, b in zip(role_list, self.prev_roles))
        self.prev_roles = role_list
        energy = float(np.sum(np.linalg.norm(learner_actions, axis=1) ** 2))
        self.prev_actions = learner_actions.copy()
        self.t += 1
        done = bool(np.all(self.captured) or self.t >= c.horizon or collisions > 0)
        stats = StepStats(
            progress=progress,
            collisions=collisions,
            near_misses=near_misses,
            safety_interventions=0,
            energy=energy,
            role_switches=role_switches,
        )
        return self.observe(), stats, done


class Policy:
    name = "policy"

    def reset(self) -> None:
        pass

    def act(self, obs: Dict[str, Array], env: OpenTeamPursuitEnv) -> Tuple[Array, List[str], int]:
        raise NotImplementedError


class GreedyPolicy(Policy):
    name = "greedy"

    def act(self, obs: Dict[str, Array], env: OpenTeamPursuitEnv) -> Tuple[Array, List[str], int]:
        target = obs["targets"][0]
        actions = np.vstack([_unit_to(target, p, env.cfg.drone_speed) for p in obs["learners"]])
        return actions, ["chase"] * env.cfg.num_learners, 0


class SafetyOnlyPolicy(Policy):
    name = "safety_only"

    def act(self, obs: Dict[str, Array], env: OpenTeamPursuitEnv) -> Tuple[Array, List[str], int]:
        target = obs["targets"][0]
        raw = np.vstack([_unit_to(target, p, env.cfg.drone_speed) for p in obs["learners"]])
        safe, interventions = safety_project(raw, obs, env.cfg, budget=1.0)
        return safe, ["safe_chase"] * env.cfg.num_learners, interventions


class TeammateOnlyPolicy(Policy):
    name = "teammate_only"

    def act(self, obs: Dict[str, Array], env: OpenTeamPursuitEnv) -> Tuple[Array, List[str], int]:
        target = obs["targets"][0]
        partner_center = obs["partners"].mean(axis=0) if len(obs["partners"]) else target
        actions = []
        roles = []
        for i, p in enumerate(obs["learners"]):
            offset = target - partner_center
            tangent = np.array([-offset[1], offset[0]])
            if _norm(tangent) < 1e-6:
                tangent = np.array([1.0, 0.0])
            tangent = tangent / _norm(tangent)
            desired = target + tangent * (2.2 if i % 2 == 0 else -2.2)
            actions.append(_unit_to(desired, p, env.cfg.drone_speed))
            roles.append("flank")
        return np.vstack(actions), roles, 0


class FlatParamPolicy(Policy):
    name = "flat_cem"

    param_names = [
        "target_gain",
        "partner_center_gain",
        "spread_gain",
        "obstacle_gain",
        "speed_scale",
        "safety_budget",
    ]

    def __init__(self, params: Dict[str, float] | None = None, use_safety: bool = True, name: str | None = None):
        default = {
            "target_gain": 1.0,
            "partner_center_gain": 0.0,
            "spread_gain": 0.0,
            "obstacle_gain": 0.5,
            "speed_scale": 1.0,
            "safety_budget": 0.6,
        }
        if params:
            default.update({k: float(v) for k, v in params.items() if k in default})
        self.params = default
        self.use_safety = use_safety
        if name is not None:
            self.name = name

    @classmethod
    def from_vector(cls, vec: Array, use_safety: bool = True, name: str | None = None) -> "FlatParamPolicy":
        params = {k: float(v) for k, v in zip(cls.param_names, vec)}
        params["speed_scale"] = float(np.clip(params["speed_scale"], 0.2, 1.5))
        params["safety_budget"] = float(np.clip(params["safety_budget"], 0.0, 1.5))
        return cls(params=params, use_safety=use_safety, name=name)

    def act(self, obs: Dict[str, Array], env: OpenTeamPursuitEnv) -> Tuple[Array, List[str], int]:
        p = self.params
        target = obs["targets"][0]
        learners = obs["learners"]
        partners = obs["partners"]
        team_center = np.vstack([learners, partners]).mean(axis=0)
        partner_center = partners.mean(axis=0) if len(partners) else target
        actions = []
        for i, pos in enumerate(learners):
            target_vec = _unit_to(target, pos, env.cfg.drone_speed)
            partner_vec = _unit_to(partner_center, pos, env.cfg.drone_speed)
            spread_vec = _unit_to(pos, team_center, env.cfg.drone_speed)
            obstacle_vec = np.zeros(2)
            for ox, oy, rad in obs["obstacles"]:
                center = np.array([ox, oy])
                rel = pos - center
                d = _norm(rel)
                if d < rad + 3.0:
                    obstacle_vec += rel / d * (rad + 3.0 - d)
            raw = (
                p["target_gain"] * target_vec
                + p["partner_center_gain"] * partner_vec
                + p["spread_gain"] * spread_vec
                + p["obstacle_gain"] * obstacle_vec
            )
            actions.append(_clip_norm(raw, env.cfg.drone_speed * p["speed_scale"]))
        raw_actions = np.vstack(actions)
        if self.use_safety:
            safe, interventions = safety_project(raw_actions, obs, env.cfg, budget=p["safety_budget"])
        else:
            safe, interventions = raw_actions, 0
        return safe, ["flat"] * env.cfg.num_learners, interventions


class TarotPolicy(Policy):
    name = "tarot_full"

    def __init__(
        self,
        use_teammate: bool = True,
        use_events: bool = True,
        use_safety: bool = True,
        use_risk_gate: bool = True,
        belief: str = "bayes",
        safety_budget: float = 0.85,
        reliability_threshold: float = 0.30,
        risk_clearance_scale: float = 2.5,
        intercept_mix: float = 0.35,
        intercept_offset: float = 1.8,
        flank_mix: float = 0.45,
        flank_offset: float = 2.5,
    ):
        self.use_teammate = use_teammate
        self.use_events = use_events
        self.use_safety = use_safety
        self.use_risk_gate = use_risk_gate
        self.belief = belief
        self.safety_budget = safety_budget
        self.reliability_threshold = reliability_threshold
        self.risk_clearance_scale = risk_clearance_scale
        self.intercept_mix = intercept_mix
        self.intercept_offset = intercept_offset
        self.flank_mix = flank_mix
        self.flank_offset = flank_offset
        self.prev_partners: Array | None = None
        self.prev_target: Array | None = None
        self.partner_mode: str = "unknown"
        self.last_reliability: float = 0.0
        self.last_confidence: float = 0.0
        self.last_roles: List[str] = []
        self.mode_belief: Dict[str, float] = {}
        self.mode_probs: Dict[str, float] = {}
        suffix = []
        if not use_teammate:
            suffix.append("no_teammate")
        if not use_events:
            suffix.append("no_events")
        if not use_safety:
            suffix.append("no_safety")
        if not use_risk_gate:
            suffix.append("no_risk_gate")
        if belief != "bayes":
            suffix.append(f"{belief}_belief")
        if suffix:
            self.name = "tarot_" + "_".join(suffix)

    def reset(self) -> None:
        self.prev_partners = None
        self.prev_target = None
        self.partner_mode = "unknown"
        self.last_reliability = 0.0
        self.last_confidence = 0.0
        self.last_roles = []
        self.mode_belief = {}
        self.mode_probs = {mode: 0.2 for mode in ["chaser", "blocker", "flanker", "lazy", "noisy"]}

    def _partner_reliability(self, obs: Dict[str, Array], env: OpenTeamPursuitEnv) -> Tuple[float, str]:
        if len(obs["partners"]) == 0:
            return 0.0, "none"
        if self.prev_partners is None or self.prev_target is None:
            return 0.25, "unknown"
        target = obs["targets"][0]
        prev_target = self.prev_target
        team_center = np.vstack([obs["learners"], obs["partners"]]).mean(axis=0)
        escape = _unit_to(target, team_center, 1.0)
        tangent = np.array([-escape[1], escape[0]]) / _norm(escape)
        scores = {"chaser": [], "blocker": [], "flanker": [], "lazy": []}
        speed_fracs = []
        for i, partner in enumerate(obs["partners"]):
            prev = self.prev_partners[i]
            vel = (partner - prev) / max(env.cfg.dt, 1e-6)
            speed = _norm(vel)
            speed_fracs.append(min(speed / max(env.cfg.partner_speed, 1e-6), 1.0))
            if speed < 0.15:
                scores["lazy"].append(1.0)
                scores["chaser"].append(0.0)
                scores["blocker"].append(0.0)
                scores["flanker"].append(0.0)
                continue
            chaser_dir = _unit_to(prev_target, prev, 1.0)
            blocker_dir = _unit_to(prev_target + escape * 3.0, prev, 1.0)
            flank_a = _unit_to(prev_target + tangent * 2.5, prev, 1.0)
            flank_b = _unit_to(prev_target - tangent * 2.5, prev, 1.0)
            vhat = vel / speed
            scores["chaser"].append(max(0.0, float(np.dot(vhat, chaser_dir))))
            scores["blocker"].append(max(0.0, float(np.dot(vhat, blocker_dir))))
            scores["flanker"].append(max(0.0, float(max(np.dot(vhat, flank_a), np.dot(vhat, flank_b)))))
            scores["lazy"].append(0.0)
        means = {k: float(np.mean(v)) if v else 0.0 for k, v in scores.items()}
        speed_factor = float(np.mean(speed_fracs)) if speed_fracs else 0.0
        best_aligned = max(means["chaser"], means["blocker"], means["flanker"])
        likelihood = {
            "chaser": float(np.exp(5.0 * means["chaser"]) * (0.25 + speed_factor)),
            "blocker": float(np.exp(5.0 * means["blocker"]) * (0.25 + speed_factor)),
            "flanker": float(np.exp(5.0 * means["flanker"]) * (0.25 + speed_factor)),
            "lazy": float(np.exp(5.0 * means["lazy"]) * (1.15 - 0.55 * speed_factor)),
            "noisy": float(np.exp(4.0 * max(0.0, 0.65 - best_aligned)) * (0.35 + speed_factor)),
        }
        total_like = sum(max(v, 1e-9) for v in likelihood.values())
        instant_probs = {k: max(v, 1e-9) / total_like for k, v in likelihood.items()}
        if self.belief == "instant":
            probs = instant_probs
        else:
            if not self.mode_probs:
                self.mode_probs = {mode: 0.2 for mode in likelihood}
            posterior = {
                k: (max(self.mode_probs.get(k, 0.2), 1e-9) ** 0.75) * max(likelihood[k], 1e-9)
                for k in likelihood
            }
            total = sum(posterior.values())
            probs = {k: v / total for k, v in posterior.items()}
            self.mode_probs = probs
        mode, confidence = max(probs.items(), key=lambda kv: kv[1])
        self.last_confidence = float(confidence)
        if mode == "lazy":
            reliability = 0.20
        elif mode == "noisy":
            reliability = 0.12
        else:
            reliability = float(np.clip(confidence * (0.35 + 0.65 * means[mode]) * (0.4 + 0.6 * speed_factor), 0.0, 1.0))
        if confidence < 0.42:
            return min(reliability, 0.20), "uncertain"
        return reliability, mode

    def act(self, obs: Dict[str, Array], env: OpenTeamPursuitEnv) -> Tuple[Array, List[str], int]:
        target = obs["targets"][0]
        learners = obs["learners"]
        team_center = np.vstack([learners, obs["partners"]]).mean(axis=0)
        reliability, mode = self._partner_reliability(obs, env) if self.use_teammate else (0.0, "disabled")
        if self.use_teammate:
            if self.use_risk_gate and len(obs["obstacles"]):
                min_clearance = min(
                    _norm(p - np.array([ox, oy])) - rad
                    for p in obs["learners"]
                    for ox, oy, rad in obs["obstacles"]
                )
                if min_clearance < env.cfg.near_miss_radius * self.risk_clearance_scale:
                    reliability, mode = 0.0, "risk_gated"
        self.partner_mode = mode
        self.last_reliability = float(reliability)
        event_phase = (env.t % 24 == 0) or not self.use_events
        roles: List[str] = []
        actions = []
        escape = _unit_to(target, team_center, 1.0)
        if _norm(escape) < 1e-6:
            escape = np.array([1.0, 0.0])
        tangent = np.array([-escape[1], escape[0]]) / _norm(escape)
        for i, p in enumerate(learners):
            if event_phase:
                if reliability < self.reliability_threshold:
                    role = "recover"
                elif mode == "blocker":
                    role = "chase"
                elif mode == "chaser" and i > 0:
                    role = "flank"
                elif mode == "flanker" and i == 0:
                    role = "intercept"
                elif i == 0:
                    role = "intercept"
                else:
                    role = "flank"
            else:
                role = env.prev_roles[i] if env.prev_roles[i] != "none" else "intercept"
            if role == "recover":
                desired = target
            elif role == "chase":
                desired = target
            elif role == "intercept":
                desired = (1.0 - self.intercept_mix) * target + self.intercept_mix * (
                    target + escape * self.intercept_offset
                )
            elif role == "backup":
                desired = target - escape * 1.2
            else:
                side = self.flank_offset if i % 2 == 0 else -self.flank_offset
                desired = (1.0 - self.flank_mix) * target + self.flank_mix * (target + tangent * side)
            actions.append(_unit_to(desired, p, env.cfg.drone_speed))
            roles.append(role)
        raw = np.vstack(actions)
        if self.use_safety:
            safe, interventions = safety_project(raw, obs, env.cfg, budget=self.safety_budget)
        else:
            safe, interventions = raw, 0
        self.prev_partners = obs["partners"].copy()
        self.prev_target = target.copy()
        self.last_roles = roles
        return safe, roles, interventions


def safety_project(actions: Array, obs: Dict[str, Array], cfg: SimConfig, budget: float) -> Tuple[Array, int]:
    safe = actions.copy()
    interventions = 0
    learners = obs["learners"]
    team = np.vstack([obs["learners"], obs["partners"]])
    for i, p in enumerate(learners):
        original = safe[i].copy()
        for ox, oy, rad in obs["obstacles"]:
            center = np.array([ox, oy])
            rel = p - center
            d = _norm(rel)
            next_rel = p + safe[i] * cfg.dt - center
            next_d = _norm(next_rel)
            margin = rad + cfg.near_miss_radius * 1.8
            if d < margin or next_d < margin:
                outward = rel / d
                tangent = np.array([-outward[1], outward[0]])
                toward = float(np.dot(safe[i], -outward))
                if toward > 0:
                    safe[i] += outward * toward * 1.25
                side = 1.0 if np.dot(tangent, safe[i]) >= 0 else -1.0
                safe[i] += outward * cfg.drone_speed * budget * max(margin - min(d, next_d), 0.0) / margin
                safe[i] += tangent * side * cfg.drone_speed * 0.35 * budget
        for q in team:
            rel = p - q
            d = _norm(rel)
            if d > 1e-6 and d < cfg.near_miss_radius * 1.3:
                safe[i] += rel / d * cfg.drone_speed * budget * (cfg.near_miss_radius * 1.3 - d)
        safe[i] = _clip_norm(safe[i], cfg.drone_speed)
        if np.linalg.norm(safe[i] - original) > 1e-6:
            interventions += 1
    return safe, interventions


def summarize_episode(cfg: SimConfig, stats: List[StepStats], captured: Array, steps: int) -> Dict[str, float]:
    collisions = sum(s.collisions for s in stats)
    near_misses = sum(s.near_misses for s in stats)
    return {
        "success": float(np.all(captured) and collisions == 0),
        "task_completion": float(np.mean(captured)),
        "steps": float(steps),
        "collision_count": float(collisions),
        "near_miss_count": float(near_misses),
        "safety_interventions": float(sum(s.safety_interventions for s in stats)),
        "energy": float(sum(s.energy for s in stats)),
        "role_switches": float(sum(s.role_switches for s in stats)),
        "progress": float(sum(s.progress for s in stats)),
    }


def run_episode(policy: Policy, cfg: SimConfig) -> Dict[str, float]:
    policy.reset()
    env = OpenTeamPursuitEnv(cfg)
    obs = env.reset()
    done = False
    stats: List[StepStats] = []
    while not done:
        action, roles, interventions = policy.act(obs, env)
        obs, step_stats, done = env.step(action, roles)
        step_stats.safety_interventions += interventions
        stats.append(step_stats)
    row = summarize_episode(cfg, stats, env.captured, env.t)
    row.update(
        {
            "policy": policy.name,
            "scenario": cfg.scenario,
            "target_policy": cfg.target_policy,
            "partner_policy": cfg.partner_policy,
            "seed": float(cfg.seed),
            "num_learners": float(cfg.num_learners),
            "num_partners": float(cfg.num_partners),
        }
    )
    return row


def policy_suite() -> List[Policy]:
    return [
        GreedyPolicy(),
        SafetyOnlyPolicy(),
        TeammateOnlyPolicy(),
        FlatParamPolicy(),
        TarotPolicy(),
        TarotPolicy(use_teammate=False),
        TarotPolicy(use_events=False),
        TarotPolicy(use_safety=False),
        TarotPolicy(use_risk_gate=False),
        TarotPolicy(belief="instant"),
    ]
