from __future__ import annotations

import json
from math import sqrt
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Rectangle
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
PAPER = ROOT / "paper"
FIGURES = PAPER / "figures"
TABLES = PAPER / "tables"

DATASETS = {
    "Main grid": ROOT / "trans_project/results/local_optimized_baselines_eval100/episodes.csv",
    "Robustness": ROOT / "remote_results/remote_robustness_eval50_v2/episodes.csv",
    "Reliability shift": ROOT / "remote_results/remote_partner_reliability_eval50/episodes.csv",
    "Tail stress": ROOT / "remote_results/remote_tail_stress_eval30_v2/episodes.csv",
    "Unseen partners": ROOT / "trans_project/results/unseen_partner_eval100/episodes.csv",
}

BUDGET = ROOT / "trans_project/results/local_budget_eval100/episodes.csv"
SCALABILITY = ROOT / "trans_project/results/local_scalability_eval50/episodes.csv"
DYNAMICS3D = ROOT / "trans_project/results/dynamics3d_eval100/episodes.csv"
DYNAMICS3D_SUMMARY = ROOT / "trans_project/results/dynamics3d_artifacts/dynamics3d_summary.json"

POLICY_LABELS = {
    "greedy": "Greedy",
    "safety_only": "Safety only",
    "tarot_no_safety": "TAROT w/o projection",
    "tarot_no_teammate": "TAROT w/o teammate",
    "tarot_no_risk_gate": "TAROT w/o risk gate",
    "tarot_default": "TAROT",
    "tarot_tuned_cem": "TAROT-CEM",
    "flat_cem_tuned": "Flat CEM",
}

COLORS = {
    "greedy": "#7A7A7A",
    "safety_only": "#4C78A8",
    "tarot_no_safety": "#B279A2",
    "tarot_no_teammate": "#72B7B2",
    "tarot_no_risk_gate": "#F2CF5B",
    "tarot_default": "#2A9D8F",
    "tarot_tuned_cem": "#1B7F4B",
    "flat_cem_tuned": "#8E7CC3",
}


def setup_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 8.5,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 0.9,
            "lines.linewidth": 1.8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
        }
    )


def enrich(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    frame["collision_episode"] = (frame["collision_count"] > 0).astype(float)
    frame["timeout"] = 1.0 - frame["success"] - frame["collision_episode"]
    frame["intervention_rate"] = (
        frame["safety_interventions"]
        / (frame["steps"].clip(lower=1) * frame["num_learners"].clip(lower=1))
    )
    frame["successful_steps"] = frame["steps"].where(frame["success"] == 1)
    return frame


def wilson_interval(successes: float, n: int, z: float = 1.96) -> tuple[float, float]:
    p = successes / n
    denominator = 1.0 + z * z / n
    center = (p + z * z / (2.0 * n)) / denominator
    half = z * sqrt((p * (1.0 - p) + z * z / (4.0 * n)) / n) / denominator
    return center - half, center + half


def paired_effect(
    frame: pd.DataFrame,
    first: str,
    second: str,
    metric: str,
    reduction: bool = False,
    samples: int = 2000,
    seed: int = 20260716,
) -> tuple[float, float, float]:
    key_columns = [
        c
        for c in frame.columns
        if c
        in {
            "scenario",
            "target_policy",
            "partner_policy",
            "seed",
            "num_learners",
            "num_partners",
            "perception_noise",
            "num_obstacles",
            "target_speed",
            "partner_speed",
        }
    ]
    a = frame.loc[frame.policy == first, key_columns + [metric]].rename(columns={metric: "a"})
    b = frame.loc[frame.policy == second, key_columns + [metric]].rename(columns={metric: "b"})
    paired = a.merge(b, on=key_columns, how="inner", validate="one_to_one")
    diff = paired["b"].to_numpy() - paired["a"].to_numpy() if reduction else paired["a"].to_numpy() - paired["b"].to_numpy()
    rng = np.random.default_rng(seed)
    boot = np.empty(samples, dtype=float)
    n = len(diff)
    for start in range(0, samples, 100):
        stop = min(samples, start + 100)
        indices = rng.integers(0, n, size=(stop - start, n))
        boot[start:stop] = diff[indices].mean(axis=1)
    return float(diff.mean()), float(np.quantile(boot, 0.025)), float(np.quantile(boot, 0.975))


def save_figure(fig: plt.Figure, stem: str) -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    for suffix in ("pdf", "svg"):
        fig.savefig(FIGURES / f"{stem}.{suffix}", bbox_inches="tight")
    fig.savefig(FIGURES / f"{stem}.png", dpi=350, bbox_inches="tight")
    plt.close(fig)


def panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(-0.12, 1.06, label, transform=ax.transAxes, fontsize=13, fontweight="bold", va="top")


def _make_framework_legacy() -> None:
    # Draw at the manuscript's physical text width so fonts are not silently
    # halved when LaTeX scales the figure down to \textwidth.
    fig, axes = plt.subplots(
        1,
        3,
        figsize=(7.2, 2.55),
        gridspec_kw={"width_ratios": [1.0, 1.75, 1.35]},
    )
    ink = "#30343B"
    teal = "#16877B"
    blue = "#2F6BDE"
    green = "#1B7F4B"
    amber = "#A66A12"
    red = "#A64B3C"

    def prepare_panel(ax: plt.Axes, label: str, title: str) -> None:
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis("off")
        ax.text(0.0, 0.985, label, fontsize=8.5, fontweight="bold", va="top")
        ax.text(0.13, 0.985, title, fontsize=6.8, fontweight="semibold", va="top")
        ax.plot([0.13, 0.98], [0.925, 0.925], color="#D7DBE0", lw=0.65, clip_on=False)

    def box(
        ax: plt.Axes,
        x: float,
        y: float,
        w: float,
        h: float,
        label: str,
        face: str,
        fontsize: float = 5.5,
        weight: str = "normal",
    ) -> None:
        ax.add_patch(
            FancyBboxPatch(
                (x, y),
                w,
                h,
                boxstyle="round,pad=0.008,rounding_size=0.012",
                facecolor=face,
                edgecolor=ink,
                lw=0.7,
            )
        )
        ax.text(
            x + w / 2,
            y + h / 2,
            label,
            ha="center",
            va="center",
            fontsize=fontsize,
            fontweight=weight,
            linespacing=1.12,
        )

    def flow_arrow(
        ax: plt.Axes,
        start: tuple[float, float],
        end: tuple[float, float],
        color: str = ink,
        lw: float = 0.8,
    ) -> None:
        ax.add_patch(
            FancyArrowPatch(
                start,
                end,
                arrowstyle="-|>",
                mutation_scale=7.5,
                lw=lw,
                color=color,
                shrinkA=1.2,
                shrinkB=1.2,
            )
        )

    # a: pursuit scene and legend.
    ax = axes[0]
    prepare_panel(ax, "a", "Open-team pursuit state")
    ax.text(0.13, 0.855, "controlled drones", color=teal, fontsize=5.8, va="top")
    ax.text(0.13, 0.795, "uncontrolled partners", color=blue, fontsize=5.8, va="top")
    ax.text(0.13, 0.735, "moving target and clutter", color="#555B62", fontsize=5.8, va="top")

    obstacle_xy = np.array([[0.28, 0.63], [0.69, 0.63], [0.50, 0.23], [0.43, 0.51]])
    ax.scatter(
        obstacle_xy[:, 0],
        obstacle_xy[:, 1],
        s=[105, 92, 175, 70],
        marker="o",
        facecolor="#D8DADD",
        edgecolor="#858A90",
        linewidth=0.65,
        zorder=1,
    )
    target = (0.47, 0.45)
    learners = [(0.24, 0.18), (0.28, 0.39)]
    partners = [(0.82, 0.27), (0.84, 0.49)]
    for x, y in learners:
        ax.scatter(x, y, s=25, marker="o", color=teal, edgecolor="white", linewidth=0.45, zorder=4)
        flow_arrow(ax, (x + 0.015, y + 0.012), (x + 0.17, y + 0.12), color=teal, lw=1.0)
    for x, y in partners:
        ax.scatter(x, y, s=27, marker="s", color=blue, edgecolor="white", linewidth=0.45, zorder=4)
        flow_arrow(ax, (x - 0.015, y + 0.005), (x - 0.16, y + 0.06), color=blue, lw=1.0)
    ax.scatter(*target, marker="*", s=63, color="#17191C", zorder=5)
    flow_arrow(ax, (target[0] + 0.025, target[1] + 0.01), (target[0] + 0.11, target[1] - 0.07), color=ink, lw=0.7)

    # b: belief, confidence/risk gate, role selection, and fallback path.
    ax = axes[1]
    prepare_panel(ax, "b", "Teammate-aware role adaptation")
    ax.text(0.76, 0.865, "chase / intercept / flank", color=green, fontsize=5.7, ha="center")
    y_top, h_top = 0.61, 0.17
    specs = {
        "features": (0.01, y_top, 0.20, h_top, "motion +\nclearance", "#EDF3F6"),
        "belief": (0.26, y_top, 0.22, h_top, "recursive belief\n$b_t(\\tau)$", "#E8F3EC"),
        "gate": (0.53, y_top, 0.18, h_top, "confidence +\nrisk gate", "#FFF1DA"),
        "roles": (0.76, y_top, 0.23, h_top, "event-triggered\nrole update", "#E8F3EC"),
        "fallback": (0.53, 0.20, 0.18, 0.17, "fallback\nrecovery", "#F4F4F4"),
        "candidate": (0.76, 0.20, 0.23, 0.17, "candidate velocity\ncommands", "#EDF3F6"),
    }
    for x, y, w, h, label, color in specs.values():
        box(ax, x, y, w, h, label, color, fontsize=5.25)
    flow_arrow(ax, (0.21, 0.695), (0.26, 0.695))
    flow_arrow(ax, (0.48, 0.695), (0.53, 0.695))
    flow_arrow(ax, (0.71, 0.695), (0.76, 0.695))
    flow_arrow(ax, (0.875, 0.61), (0.875, 0.37))
    flow_arrow(ax, (0.62, 0.61), (0.62, 0.37), color=amber)
    flow_arrow(ax, (0.71, 0.285), (0.76, 0.285))
    ax.text(
        0.59,
        0.475,
        "low confidence\nor low clearance",
        color=amber,
        fontsize=4.9,
        ha="right",
        va="center",
        linespacing=1.1,
    )

    # c: the final one-step geometric safeguard.
    ax = axes[2]
    prepare_panel(ax, "c", "One-step execution safeguard")
    box(ax, 0.01, 0.61, 0.23, 0.18, "candidate\n$u_{i,t}$", "#EDF3F6", fontsize=5.5)
    box(ax, 0.34, 0.49, 0.64, 0.33, "", "#F0ECF7")
    ax.text(
        0.66,
        0.745,
        "predictive safety projection",
        ha="center",
        va="center",
        fontsize=6.0,
        fontweight="bold",
    )
    ax.text(
        0.66,
        0.615,
        "one-step clearance check\nremove inward component\nadd outward / tangential correction",
        ha="center",
        va="center",
        fontsize=5.0,
        linespacing=1.35,
    )
    box(ax, 0.48, 0.13, 0.40, 0.14, "executed command\n$\\widetilde{u}_{i,t}$", "#E8F3EC", fontsize=5.4)
    flow_arrow(ax, (0.24, 0.70), (0.34, 0.70))
    flow_arrow(ax, (0.68, 0.49), (0.68, 0.27))
    ax.text(
        0.02,
        0.385,
        "Geometric, model-light filter;\nno formal forward-invariance claim",
        color=red,
        fontsize=5.0,
        ha="left",
        va="center",
        linespacing=1.12,
    )

    fig.subplots_adjust(left=0.018, right=0.995, top=0.985, bottom=0.035, wspace=0.16)
    save_figure(fig, "jksucis_framework")


def make_framework() -> None:
    """Generate Fig. 1 through its dedicated, reproducible figure script."""
    try:
        from trans_project.scripts.make_fig1_framework import build_fig1
    except ModuleNotFoundError:
        from make_fig1_framework import build_fig1

    build_fig1(FIGURES)


def make_main_table(main: pd.DataFrame) -> list[dict]:
    policies = ["greedy", "safety_only", "tarot_no_safety", "tarot_default", "tarot_tuned_cem", "flat_cem_tuned"]
    rows = []
    tex = [
        r"\begin{table*}[t]",
        r"\caption{Main-grid results over 6,600 episodes. Success and collision are episode-level rates; capture time is conditioned on success. Projection activation is the fraction of controlled-agent time steps whose proposed command was modified. Brackets show 95\% Wilson intervals for success.}",
        r"\label{tab:main_results}",
        r"\centering",
        r"\small",
        r"\resizebox{\textwidth}{!}{%",
        r"\begin{tabular}{lccccc}",
        r"\toprule",
        r"Method & Success [95\% CI] $\uparrow$ & Collision $\downarrow$ & Timeout $\downarrow$ & Steps $\mid$ success $\downarrow$ & Activation (\%) $\downarrow$ \\",
        r"\midrule",
    ]
    for policy in policies:
        group = main.loc[main.policy == policy]
        n = len(group)
        success = float(group.success.mean())
        lo, hi = wilson_interval(float(group.success.sum()), n)
        collision = float(group.collision_episode.mean())
        timeout = float(group.timeout.mean())
        capture_steps = float(group.successful_steps.mean())
        activation = float(100.0 * group.intervention_rate.mean()) if policy not in {"greedy", "tarot_no_safety"} else None
        rows.append(
            {
                "policy": policy,
                "episodes": n,
                "success": success,
                "success_ci": [lo, hi],
                "collision": collision,
                "timeout": timeout,
                "successful_steps": capture_steps,
                "activation_percent": activation,
            }
        )
        activation_text = "--" if activation is None else f"{activation:.1f}"
        tex.append(
            f"{POLICY_LABELS[policy]} & {success:.3f} [{lo:.3f}, {hi:.3f}] & {collision:.3f} & {timeout:.3f} & {capture_steps:.1f} & {activation_text} \\\\"
        )
    tex.extend([r"\bottomrule", r"\end{tabular}", r"}", r"\end{table*}"])
    (TABLES / "jksucis_main_table.tex").write_text("\n".join(tex) + "\n")
    return rows


def make_cross_suite_table(frames: dict[str, pd.DataFrame]) -> list[dict]:
    suite_names = ["Robustness", "Reliability shift", "Tail stress", "Unseen partners"]
    records = []
    tex = [
        r"\begin{table*}[t]",
        r"\caption{Cross-suite evidence for TAROT-CEM. Paired effects use matched scenario--seed episodes. Positive activation reduction means TAROT-CEM invokes the safety projection less often than the comparator. Confidence intervals are paired bootstrap 95\% intervals.}",
        r"\label{tab:cross_suite}",
        r"\centering",
        r"\small",
        r"\resizebox{\textwidth}{!}{%",
        r"\begin{tabular}{lrrrrr}",
        r"\toprule",
        r"Suite & Episodes & Success & Activation (\%) & $\Delta$ success vs. Flat & Activation reduction vs. Flat \\",
        r"\midrule",
    ]
    for suite in suite_names:
        frame = frames[suite]
        group = frame.loc[frame.policy == "tarot_tuned_cem"]
        success = float(group.success.mean())
        activation = float(100.0 * group.intervention_rate.mean())
        delta = paired_effect(frame, "tarot_tuned_cem", "flat_cem_tuned", "success")
        reduction = paired_effect(frame, "tarot_tuned_cem", "flat_cem_tuned", "intervention_rate", reduction=True)
        reduction = tuple(100.0 * x for x in reduction)
        records.append(
            {
                "suite": suite,
                "episodes": len(frame),
                "per_policy": len(group),
                "success": success,
                "activation_percent": activation,
                "delta_success_vs_flat": delta,
                "activation_reduction_vs_flat": reduction,
            }
        )
        tex.append(
            f"{suite} & {len(frame):,} & {success:.3f} & {activation:.1f} & "
            f"{delta[0]:+.3f} [{delta[1]:+.3f}, {delta[2]:+.3f}] & "
            f"{reduction[0]:+.1f} [{reduction[1]:+.1f}, {reduction[2]:+.1f}] \\\\"
        )
    tex.extend([r"\bottomrule", r"\end{tabular}", r"}", r"\end{table*}"])
    (TABLES / "jksucis_cross_suite_table.tex").write_text("\n".join(tex) + "\n")
    return records


def make_unseen_table(unseen: pd.DataFrame) -> list[dict]:
    partners = ["circling", "switching", "erratic"]
    policies = ["safety_only", "tarot_no_teammate", "tarot_default", "tarot_tuned_cem", "flat_cem_tuned"]
    records = []
    tex = [
        r"\begin{table}[t]",
        r"\caption{Success / projection-activation percentage for partner behaviors excluded from the belief prototype set. Each cell aggregates 800 episodes.}",
        r"\label{tab:unseen_partners}",
        r"\centering",
        r"\small",
        r"\begin{tabular}{lccc}",
        r"\toprule",
        r"Method & Circling & Switching & Erratic \\",
        r"\midrule",
    ]
    for policy in policies:
        cells = []
        for partner in partners:
            group = unseen.loc[(unseen.policy == policy) & (unseen.partner_policy == partner)]
            success = float(group.success.mean())
            activation = float(100.0 * group.intervention_rate.mean())
            records.append({"policy": policy, "partner": partner, "success": success, "activation_percent": activation})
            cells.append(f"{success:.3f} / {activation:.1f}")
        tex.append(f"{POLICY_LABELS[policy]} & " + " & ".join(cells) + r" \\")
    tex.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}"])
    (TABLES / "jksucis_unseen_table.tex").write_text("\n".join(tex) + "\n")
    return records


def make_evidence_figure(frames: dict[str, pd.DataFrame]) -> list[dict]:
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 7.6))
    main = frames["Main grid"]
    policies = ["greedy", "safety_only", "tarot_no_safety", "tarot_default", "tarot_tuned_cem", "flat_cem_tuned"]
    y = np.arange(len(policies))
    means, low, high = [], [], []
    for policy in policies:
        group = main.loc[main.policy == policy]
        means.append(group.success.mean())
        lo, hi = wilson_interval(group.success.sum(), len(group))
        low.append(lo)
        high.append(hi)
    ax = axes[0, 0]
    for i, policy in enumerate(policies):
        ax.errorbar(means[i], i, xerr=[[means[i] - low[i]], [high[i] - means[i]]], fmt="o", ms=7, color=COLORS[policy], capsize=3)
    ax.set_yticks(y, [POLICY_LABELS[p] for p in policies])
    ax.invert_yaxis()
    ax.set_xlabel("Episode success rate")
    ax.set_xlim(0.24, 0.60)
    ax.grid(axis="x", color="#E6E6E6", lw=0.8)
    ax.set_title("Main-grid task outcome")
    panel_label(ax, "a")

    ax = axes[0, 1]
    active_policies = ["safety_only", "tarot_default", "tarot_tuned_cem", "flat_cem_tuned"]
    values = [100 * main.loc[main.policy == p, "intervention_rate"].mean() for p in active_policies]
    bars = ax.bar(np.arange(len(active_policies)), values, color=[COLORS[p] for p in active_policies], width=0.68)
    ax.set_xticks(np.arange(len(active_policies)), [POLICY_LABELS[p] for p in active_policies], rotation=18, ha="right")
    ax.set_ylabel("Projection activation (% agent-steps)")
    ax.set_ylim(0, 105)
    ax.grid(axis="y", color="#E6E6E6", lw=0.8)
    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 2.0, f"{value:.1f}", ha="center", va="bottom", fontsize=8.5)
    ax.set_title("Intervention burden")
    panel_label(ax, "b")

    effect_records = []
    suites = ["Robustness", "Reliability shift", "Tail stress", "Unseen partners"]
    comparisons = [("tarot_tuned_cem", "flat_cem_tuned"), ("tarot_tuned_cem", "tarot_no_teammate")]
    ax = axes[1, 0]
    labels, points, errors_low, errors_high, effect_colors = [], [], [], [], []
    for suite in suites:
        for first, second in comparisons:
            effect = paired_effect(frames[suite], first, second, "success")
            labels.append(f"{suite}: {POLICY_LABELS[first]} vs {POLICY_LABELS[second]}")
            points.append(effect[0])
            errors_low.append(effect[0] - effect[1])
            errors_high.append(effect[2] - effect[0])
            effect_colors.append(COLORS[second])
            effect_records.append({"suite": suite, "first": first, "second": second, "success_effect": effect})
    yy = np.arange(len(labels))
    for i in range(len(labels)):
        ax.errorbar(points[i], i, xerr=[[errors_low[i]], [errors_high[i]]], fmt="o", color=effect_colors[i], capsize=2.5, ms=5.5)
    ax.axvline(0, color="#444444", ls="--", lw=1)
    ax.set_yticks(yy, labels)
    ax.invert_yaxis()
    ax.set_xlabel("Paired $\\Delta$ success")
    ax.grid(axis="x", color="#E6E6E6", lw=0.8)
    ax.set_title("Generalization effects (95% paired bootstrap CI)")
    panel_label(ax, "c")

    ax = axes[1, 1]
    budget = enrich(pd.read_csv(BUDGET))
    budget = budget.loc[budget.policy.str.startswith("tarot_budget")].copy()
    budget["budget"] = budget.policy.str.extract(r"(\d+p\d+)")[0].str.replace("p", ".", regex=False).astype(float)
    curve = budget.groupby("budget", as_index=False).agg(success=("success", "mean"), collision=("collision_episode", "mean"), activation=("intervention_rate", "mean"))
    ax.plot(curve.budget, curve.success, marker="o", color=COLORS["tarot_default"], label="success")
    ax.plot(curve.budget, curve.collision, marker="s", color="#D65F5F", label="collision episode")
    ax.set_xlabel("Safety-projection budget")
    ax.set_ylabel("Episode rate")
    ax.set_ylim(0.25, 0.75)
    ax.grid(color="#E6E6E6", lw=0.8)
    ax.legend(frameon=False, loc="best")
    ax.set_title("Safety-budget sensitivity")
    panel_label(ax, "d")
    fig.subplots_adjust(hspace=0.42, wspace=0.48)
    save_figure(fig, "jksucis_evidence_summary")
    return effect_records


def make_generalization_figure(frames: dict[str, pd.DataFrame]) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(12.4, 3.55))
    policies = ["safety_only", "tarot_default", "tarot_tuned_cem", "flat_cem_tuned"]

    robustness = frames["Robustness"]
    ax = axes[0]
    for policy in policies:
        curve = robustness.loc[robustness.policy == policy].groupby("num_obstacles", as_index=False).success.mean()
        ax.plot(curve.num_obstacles, curve.success, marker="o", ms=4.5, color=COLORS[policy], label=POLICY_LABELS[policy])
    ax.set_xlabel("Number of obstacles")
    ax.set_ylabel("Success rate")
    ax.grid(color="#E6E6E6", lw=0.8)
    ax.set_title("Obstacle-density shift")
    panel_label(ax, "a")

    reliability = frames["Reliability shift"]
    ax = axes[1]
    for policy in policies:
        curve = reliability.loc[reliability.policy == policy].groupby("partner_speed", as_index=False).success.mean()
        ax.plot(curve.partner_speed, curve.success, marker="o", ms=4.5, color=COLORS[policy], label=POLICY_LABELS[policy])
    ax.set_xlabel("Partner speed")
    ax.set_ylabel("Success rate")
    ax.grid(color="#E6E6E6", lw=0.8)
    ax.set_title("Partner-reliability shift")
    panel_label(ax, "b")

    unseen = frames["Unseen partners"]
    ax = axes[2]
    partners = ["circling", "switching", "erratic"]
    x = np.arange(len(partners))
    width = 0.19
    for j, policy in enumerate(policies):
        vals = [unseen.loc[(unseen.policy == policy) & (unseen.partner_policy == p), "success"].mean() for p in partners]
        ax.bar(x + (j - 1.5) * width, vals, width=width, color=COLORS[policy], label=POLICY_LABELS[policy])
    ax.set_xticks(x, [p.capitalize() for p in partners])
    ax.set_ylabel("Success rate")
    ax.set_ylim(0.30, 0.95)
    ax.grid(axis="y", color="#E6E6E6", lw=0.8)
    ax.set_title("Out-of-prototype partners")
    panel_label(ax, "c")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 1.05), ncol=4, frameon=False)
    fig.subplots_adjust(wspace=0.32, top=0.80)
    save_figure(fig, "jksucis_generalization")


def make_scalability_table() -> None:
    frame = enrich(pd.read_csv(SCALABILITY))
    tex = [
        r"\begin{table}[t]",
        r"\caption{Scalability of TAROT over 50 episodes per configuration. Runtime is wall-clock time per complete episode on the local CPU used for the original experiment.}",
        r"\label{tab:scalability}",
        r"\centering",
        r"\small",
        r"\begin{tabular}{rrrrr}",
        r"\toprule",
        r"Drones & Obstacles & Success & Collision & Runtime (ms) \\",
        r"\midrule",
    ]
    tarot = frame.loc[frame.policy == "tarot_full"]
    for (learners, partners), group in tarot.groupby(["num_learners", "num_partners"]):
        obstacles = int(group.num_obstacles.iloc[0]) if "num_obstacles" in group else -1
        runtime = float(group.runtime_ms.mean()) if "runtime_ms" in group else float("nan")
        tex.append(
            f"{int(learners + partners)} & {obstacles} & {group.success.mean():.3f} & {group.collision_episode.mean():.3f} & {runtime:.1f} \\\\"
        )
    tex.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}"])
    (TABLES / "jksucis_scalability_table.tex").write_text("\n".join(tex) + "\n")


def main() -> None:
    setup_style()
    FIGURES.mkdir(parents=True, exist_ok=True)
    TABLES.mkdir(parents=True, exist_ok=True)
    frames = {name: enrich(pd.read_csv(path)) for name, path in DATASETS.items()}
    main_rows = make_main_table(frames["Main grid"])
    cross_rows = make_cross_suite_table(frames)
    unseen_rows = make_unseen_table(frames["Unseen partners"])
    effects = make_evidence_figure(frames)
    make_generalization_figure(frames)
    make_framework()
    make_scalability_table()
    dynamics3d_rows = len(pd.read_csv(DYNAMICS3D))
    dynamics3d_summary = json.loads(DYNAMICS3D_SUMMARY.read_text())
    episode_totals = {name: len(frame) for name, frame in frames.items()}
    episode_totals["Dynamics-aware 3-D"] = dynamics3d_rows
    payload = {
        "episode_totals": episode_totals,
        "reported_total_including_budget_and_scalability": int(
            sum(len(frame) for frame in frames.values())
            + len(pd.read_csv(BUDGET))
            + len(pd.read_csv(SCALABILITY))
            + dynamics3d_rows
        ),
        "main": main_rows,
        "cross_suite": cross_rows,
        "unseen": unseen_rows,
        "paired_effects": effects,
        "dynamics3d": dynamics3d_summary,
    }
    (PAPER / "jksucis_statistics.json").write_text(json.dumps(payload, indent=2))
    print(json.dumps(payload["episode_totals"], indent=2))
    print(f"reported total: {payload['reported_total_including_budget_and_scalability']:,}")


if __name__ == "__main__":
    main()
