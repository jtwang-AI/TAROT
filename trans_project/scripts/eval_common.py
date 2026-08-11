from __future__ import annotations

import csv
import json
from pathlib import Path
from statistics import mean, pstdev
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tarot_sim import FlatParamPolicy, GreedyPolicy, SafetyOnlyPolicy, TarotPolicy


METRICS = [
    "success",
    "task_completion",
    "steps",
    "collision_count",
    "near_miss_count",
    "safety_interventions",
    "energy",
    "role_switches",
    "progress",
]


TAROT_CEM_PARAMS = {
    "safety_budget": 1.2050607471565902,
    "reliability_threshold": 0.29407296212153916,
    "risk_clearance_scale": 3.3569204539802233,
    "intercept_mix": 0.28817201968900946,
    "intercept_offset": 1.4022064042719558,
    "flank_mix": 0.45810280941007436,
    "flank_offset": 2.099336983812863,
}


FLAT_CEM_PARAMS = {
    "target_gain": 1.9068097797202308,
    "partner_center_gain": 0.1334189299794637,
    "spread_gain": 0.2814117237442981,
    "obstacle_gain": 0.15412778637657037,
    "speed_scale": 1.0351498849861112,
    "safety_budget": 1.0265609236530422,
}


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def aggregate(rows: list[dict], keys: tuple[str, ...]) -> list[dict]:
    grouped = {}
    for row in rows:
        key = tuple(row[k] for k in keys)
        grouped.setdefault(key, []).append(row)
    out = []
    for group_key, items in sorted(grouped.items()):
        rec = {k: v for k, v in zip(keys, group_key)}
        rec["episodes"] = len(items)
        for metric in METRICS:
            vals = [float(x[metric]) for x in items]
            rec[f"{metric}_mean"] = mean(vals)
            rec[f"{metric}_std"] = pstdev(vals) if len(vals) > 1 else 0.0
        out.append(rec)
    return out


def save_run(out: Path, rows: list[dict], aggregate_keys: tuple[str, ...]) -> None:
    agg = aggregate(rows, aggregate_keys)
    out.mkdir(parents=True, exist_ok=True)
    write_csv(out / "episodes.csv", rows)
    write_csv(out / "aggregate.csv", agg)
    (out / "aggregate.json").write_text(json.dumps(agg, indent=2))
    print(f"wrote {len(rows)} episode rows")
    print(f"aggregate: {out / 'aggregate.csv'}")


def policy_suite(names: tuple[str, ...] | None = None):
    tarot_default = TarotPolicy()
    tarot_default.name = "tarot_default"
    tarot_tuned = TarotPolicy(**TAROT_CEM_PARAMS)
    tarot_tuned.name = "tarot_tuned_cem"
    no_safety = TarotPolicy(use_safety=False)
    no_safety.name = "tarot_no_safety"
    no_teammate = TarotPolicy(use_teammate=False)
    no_teammate.name = "tarot_no_teammate"
    no_risk = TarotPolicy(use_risk_gate=False)
    no_risk.name = "tarot_no_risk_gate"
    instant = TarotPolicy(belief="instant")
    instant.name = "tarot_instant_belief"
    flat = FlatParamPolicy(params=FLAT_CEM_PARAMS, name="flat_cem_tuned")
    policies = [
        GreedyPolicy(),
        SafetyOnlyPolicy(),
        no_safety,
        no_teammate,
        no_risk,
        instant,
        tarot_default,
        tarot_tuned,
        flat,
    ]
    if names is None:
        return policies
    wanted = set(names)
    return [policy for policy in policies if policy.name in wanted]
