# Friction laboratory

Three identical marbles start with the same sideways speed on a level, high-contrast checker floor. Their sliding friction coefficients are 0.01, 0.12, and 1.00. The different deceleration and spin-up make sliding turn into rolling at visibly different rates.

The demo uses the bounded `friction_contact_euler_v1` native profile: model-derived plane/sphere and sphere/sphere contacts, pyramidal `condim=3`, and the MuJoCo 3.10 contact law. It compares every native step with an independent CPU `mj_step` trajectory.

```bash
PYTHONPATH=metal/examples:metal python metal/examples/friction_laboratory.py --headless --check
PYTHONPATH=metal/examples:metal python metal/examples/friction_laboratory.py --mode cpu --headless
```

To save the side-by-side native and CPU rendering as a GIF, pass `--record metal/examples/assets/friction_laboratory.gif`.

![Native Metal and CPU MuJoCo friction laboratory](assets/friction_laboratory.gif)

The recorded 600-step run kept all three contacts active. Maximum absolute
qpos/qvel differences were `7.52e-6` / `4.80e-6`. These are fixture-specific
correctness results, not throughput measurements. The bounded solver retains
an explicit nonconvergence status; the sliding-to-rolling case is a regression
test for its contact-block refinement.
