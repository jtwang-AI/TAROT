# TAROT Supplementary Material

This document describes the anonymous supplementary bundle. The Git repository contains the simulator, evaluation scripts, tuned parameter files, aggregate results, and compact validation artifacts. Large raw traces are regenerated from the tracked scripts, seeds, and policies rather than versioned directly.

## Reported evaluation data

| Dataset | Episode rows |
|---|---:|
| Main optimized grid | 6,600 |
| Robustness | 115,200 |
| Reliability shift | 100,800 |
| Tail stress | 136,080 |
| Out-of-prototype partners | 14,400 |
| Safety-budget sweep | 3,600 |
| Scalability sweep | 750 |
| Dynamics-aware 3-D validation | 1,200 |

The manuscript total is 378,630 episodes. Step-level belief diagnostics and excluded development smoke runs are not added to this total.

## Reproduction

The core simulator requires Python, NumPy, pandas, and Matplotlib; 3-D rendering additionally requires Pillow and PyBullet 3.2.7. From the project root:

```text
python3 trans_project/scripts/run_unseen_partner_eval.py --episodes 100 --out trans_project/results/unseen_partner_eval100
python3 trans_project/scripts/make_jksucis_artifacts.py
python3 trans_project/scripts/run_3d_validation.py --episodes 100 --retain 12 --seed-base 830000 --out trans_project/results/dynamics3d_eval100
python3 trans_project/scripts/render_3d_validation.py --results trans_project/results/dynamics3d_eval100 --out trans_project/results/dynamics3d_artifacts
```

The first command reproduces the out-of-prototype evaluation. The second regenerates the planar journal figures, tables, paired-bootstrap effects, and `paper/jksucis_statistics.json`. The final two commands reproduce the independently seeded 3-D episodes and the PyBullet-rendered validation figure/table. The reported 3-D seed block is disjoint from the 730,000-series development smoke runs.

## Dependencies and scope

The planar analyses require Python, NumPy, pandas, and Matplotlib. The 3-D renderer additionally uses Pillow and PyBullet 3.2.7. The code implements a planar single-integrator simulator, a delayed acceleration-limited 3-D execution layer, and a one-step geometric safety projection. PyBullet supplies scene construction and rendering; the package does not implement Hamilton--Jacobi reachability, a control-barrier certificate, AirSim perception/rotor dynamics, or physical flight control.
