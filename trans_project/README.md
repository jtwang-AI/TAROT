# TAROT Experimental Project

This directory contains new experimental code for the TAROT paper plan.

Rules for this project:

- Do not use previous method or training code.
- Do not modify the server system environment.
- Keep dependencies minimal.
- Treat AirSim/Colosseum as a future perception/rotor-dynamics backend; do not label the present 3-D layer as AirSim.

Current backend:

- `src/tarot_sim.py`: a lightweight multi-drone open-teaming pursuit simulator and policy suite.
- `scripts/run_controlled_eval.py`: controlled evaluation runner for baselines and TAROT variants.
- `scripts/run_3d_validation.py`: delayed, acceleration-limited 3-D execution in urban, forest, and industrial geometry.
- `scripts/render_3d_validation.py`: PyBullet headless rendering and data-driven 3-D figure/table generation.

The planar simulator supports large-scale statistics. The 3-D layer adds dynamics stress and reproducible scene renders while remaining explicit about the missing AirSim-class perception, rotor dynamics, and physical flight stack.
