# Spin-and-Grip arcade

This tabletop exhibit uses three different contact effects: a rolling sphere
that loses travel to rolling resistance, a spinning sphere whose spin is damped
by torsional friction, and an actuated press that holds then releases a sliding
block. These effects arise from contact forces, inertia and the press motor;
there is no scripted object animation.

The native scene uses `integrated_euler_v1`, the elliptic cone, condim 4 for
torsional spin friction and condim 6 for rolling resistance. The press/object
pair uses condim 3 and demonstrates sliding friction. The comparison on the
right is a separate MuJoCo CPU rollout from the same initial state. The GIF is
rendered at presentation speed; it does not represent physics throughput.

Use the isolated source environment described in the
[development setup](../INSTALL.md#development-source). From the repository root,
run the native numerical check with MPS fallback disabled:

```sh
PYTORCH_ENABLE_MPS_FALLBACK=0 PYTHONPATH=metal .venv-metal/bin/python \
  metal/examples/spin_and_grip.py --headless --check --steps 400
```

The check compares each native step against CPU `mj_step`, asserts a sustained
press contact and hold/release behavior, then runs CPU low-friction
counterfactuals for rolling and torsional friction. Its current validation
metrics on the M1 Max with 32 GB unified memory are:

- Maximum native/CPU absolute `qpos` difference: `4.75e-6` (mixed coordinate
  units); maximum `qvel` difference: `6.27e-4` (mixed units).
- High rolling resistance traveled `0.2040 m`; the low-friction CPU
  counterfactual traveled `1.0507 m`.
- High torsional friction ended at `2.057 rad/s`; the low-friction CPU
  counterfactual ended at `5.774 rad/s`.
- The pad first contacted the block at step 42 and maintained contact for 139
  steps in the CPU reference rollout. Hold drift was `0 m`; after release the
  block traveled `0.0972 m` in that same CPU reference. The current check uses
  these CPU event values to qualify the demo's contact timing and release; they
  are not yet asserted as native contact-event diagnostics.

These are fixture-specific numerical checks, not universal tolerances. Run the
CPU baseline separately with:

```sh
PYTHONPATH=metal .venv-metal/bin/python metal/examples/spin_and_grip.py \
  --mode cpu --headless --steps 400
```

Record a side-by-side comparison GIF (Pillow and a working MuJoCo OpenGL
context are required):

```sh
PYTORCH_ENABLE_MPS_FALLBACK=0 PYTHONPATH=metal .venv-metal/bin/python \
  metal/examples/spin_and_grip.py --headless --steps 400 \
  --record metal/examples/assets/spin_and_grip.gif
```

To run the targeted native contact qualification matrix:

```sh
MUJOCO_METAL_RUN_GPU=1 PYTORCH_ENABLE_MPS_FALLBACK=0 PYTHONPATH=metal \
  .venv-metal/bin/python -m pytest -q metal/tests/test_coupled_constraints.py \
  -k 'native_contact_cone_dimension_matrix or native_friction_contact_capsule_and_box or native_condim6_contact_engages_rolling_friction or native_explicit_pair_anisotropic or native_zero_and_near_zero_friction or native_contact_final_valid_row_capacity'
```

The native friction increment is on the Problem 003 development branch. It is
not included in the released 0.4.0 wheel. The current qualification covers the
four accepted primitive shapes across the prior collision matrix and tests
plane contacts against spheres, capsules and boxes for higher dimensions. This
does not establish every shape-pair/cone/dimension combination, arbitrary
models or full MuJoCo compatibility. The integrated pipeline still enforces
the documented candidate pair, contact-slot, row and generalized-velocity
limits. Contact assembly, solving and integration remain separate from
rendering, which uses MuJoCo OpenGL.

## Recorded comparison

![Spin-and-Grip arcade: native Metal beside CPU MuJoCo](assets/spin_and_grip.gif)
