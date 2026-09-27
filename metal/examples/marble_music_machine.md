# Robotic marble music machine

Use the [shared source setup](../INSTALL.md#development-source), with its Python
environment active, and run these commands from the repository root.

This example showcases the integrated native Metal Euler physics pipeline (`integrated_euler_v1`),
combining motor actuation, fixed-joint tendons, passive spring/damping resonators,
fluid drag, pyramidal sphere-plane and sphere-sphere collision contacts, joint limits,
dry frictionloss, polynomial joint equality, and live current-state sensor queries—all
solved in one coupled constraint solve per Euler step.

The machine features:
- A tilted soundboard ramp where marbles roll under gravity and inertia-box fluid drag.
- A primary motorized selector gate driven by a rhythmic harmonic control torque.
- A secondary counter-rotating follower gate linked by a polynomial joint equality constraint.
- Resonant chime bars mounted on passive spring/damper pivots that ring when struck by descending marbles.
- A fixed-joint tendon coupling the selector gate to chime 1, providing rhythmic mechanical feedback.
- Pyramidal friction contacts (condim=3) between marbles, track, gates, and chimes.
- Current-state sensor queries tracking gate positions and chime vibrations during GPU execution.

Run a CPU reference smoke test:

```bash
PYTHONPATH=metal python metal/examples/marble_music_machine.py --mode cpu --headless --steps 500
```

Run the native Metal comparison check on an Apple Silicon Mac:

```bash
MUJOCO_METAL_RUN_GPU=1 PYTHONPATH=metal python metal/examples/marble_music_machine.py --headless --check --steps 100
```

Record the side-by-side native Metal and CPU MuJoCo renders:

```bash
PYTHONPATH=metal python metal/examples/marble_music_machine.py --record metal/examples/assets/marble_music_machine.gif --steps 400
```

![Native Metal robotic marble music machine beside CPU MuJoCo](assets/marble_music_machine.gif)

The 400-step native run exercises all integrated feature families simultaneously. Maximum qpos/qvel
differences from CPU were `3.72e-3` / `1.83e-1`; sensor values remained within `0.0` error.
Coupled constraints prevent the numerical divergence that occurs when contact and joint constraints are solved independently.
