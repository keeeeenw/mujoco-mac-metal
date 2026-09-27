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
MUJOCO_METAL_RUN_GPU=1 PYTHONPATH=metal python metal/examples/marble_music_machine.py --headless --check --steps 400
```

Record the side-by-side native Metal and CPU MuJoCo renders:

```bash
PYTHONPATH=metal python metal/examples/marble_music_machine.py --record metal/examples/assets/marble_music_machine.gif --steps 400
```

![Native Metal robotic marble music machine beside CPU MuJoCo](assets/marble_music_machine.gif)

The 400-step native run exercises all integrated feature families simultaneously. Maximum qpos/qvel
differences from CPU were `8.94e-6` / `1.97e-4`; stage sensor error was `0.0`, and trajectory sensor error was `8.57e-6`.
The run recorded 211 active joint limit constraint steps (238 steps near limits), chime oscillations of `0.091 rad` (chime 1) and `0.066 rad` (chime 2), and 4 distinct contact pairs (`chime1_sphere <-> marble1_sphere`, `chime2_sphere <-> marble2_sphere`, `gate1_bar <-> marble1_sphere`, and `gate2_bar <-> marble2_sphere`).
Coupled constraints prevent the numerical divergence that occurs when contact and joint constraints are solved independently.

