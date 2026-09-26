# Clockwork automaton

This compact counter-rotating gear display shows polynomial joint equality, dry
friction, and bounded slide-joint behavior in the `joint_constraints_euler_v1`
profile. A four-state host sequencer winds, coasts, reverses, and releases a
fixed-gain scalar motor while a small applied-force program moves the visible
escapement pawl. The side-by-side recorder compares native motion with an
independent MuJoCo CPU rollout and prints the measured gear-ratio residual.

The toothed wheels are visual geometry. Contacts are disabled in the model, so
there is no tooth collision, ratchet impact, ray casting, or hidden rigid-body
gear-contact claim; the counter-rotation comes from a polynomial joint equality.

Run a CPU reference smoke test:

```bash
PYTHONPATH=metal python metal/examples/clockwork_automaton.py --mode cpu --headless --steps 800
```

Run the native comparison on an MPS Mac:

```bash
MUJOCO_METAL_RUN_GPU=1 PYTHONPATH=metal python metal/examples/clockwork_automaton.py --headless --check --steps 200
```

Record the actual native and CPU renders:

```bash
PYTHONPATH=metal python metal/examples/clockwork_automaton.py --record metal/examples/assets/clockwork_automaton.gif --steps 1800
```

![Native Metal clockwork beside CPU MuJoCo](assets/clockwork_automaton.gif)

The 1,800-step native run covered all four sequencer states. Maximum qpos/qvel
differences from CPU were `7.23e-6` / `2.63e-5`; the measured gear relation
residual stayed below `9.60e-6` radians. This is a fixture check, not a general
constraint accuracy guarantee or speed benchmark.
