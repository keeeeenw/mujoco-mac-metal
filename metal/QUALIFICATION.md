# Development qualification — 2026-10-05

Current `main` publishes the experimental development backend. See
[milestone status](STATUS.md): **0/16 milestones 005–020 have full-composition
sign-off**. Earlier bounded capabilities and individual regression groups retain
their recorded evidence. This is source publication, not a new PyPI release.

## Publication scope

The source contains coherent dynamics, lifecycle, collision, solver, deformable
and API changes plus their original regression fixtures. It also corrects an
undefined mask in public spatial-armature assembly. Final native replay and
broader armature composition checks remain open. Newer mesh-SDF arithmetic
experiments have not been included.

Some host test utilities require Torch: a no-Torch full-suite attempt stopped
with six collection errors and is not passing qualification. On a Mac, merely
setting `MUJOCO_METAL_RUN_GPU=0` does not prevent every historical test from
checking actual MPS availability. The publication host run therefore disables
MPS availability for collection. This is a test guard, not a change to backend
physics or production dependencies. Host checks exercise contracts, lowering,
algorithm references and collection; skipped native tests are not GPU passes.

## Selected recorded native evidence

| Group | Recorded result | Qualification boundary |
|---|---|---|
| Newton precision and original slope regression | 94 passed | Includes the original 300-step slope fixture; not the full solver-option matrix |
| Contact override | 69 passed | Selected rigid/flex, integrator and lifecycle fixtures |
| Retained SDF arithmetic/source trace | 11 passed | Numeric and trace-inertness checks; not full SDF contact correctness |
| Original rigid-SDF group | Incomplete at 300 seconds, with 3 failures | Mesh witness multiplicity and bowl/torus acceleration or trajectory gates remain open |
| More recent combined SDF/armature/flex check | Incomplete at 180 seconds | Different, unpublished source; no completed suite summary |

These native results belong to their recorded development compositions and are
not a fresh full GPU run of this publication. Other terrain, flex, stage and API
checks are described in [development coverage](DEVELOPMENT.md), with the same
scope restriction. No numerical tolerance or solver budget was relaxed to obtain
these results. A screenshot or successful compile is not physics evidence;
no new speedup is claimed.

## Reproduce on the pinned source runtime

Use Python 3.12, MuJoCo 3.10.0 and Torch 2.9.1 in the isolated source environment.
Record the commit, module/shader paths and hashes, flags, exact command, exit
code and pass/fail/skip counts. Keep GPU checks serialized.

Host reference check with an explicit test-availability guard:

```sh
MUJOCO_METAL_RUN_GPU=0 PYTHONPATH=metal python - <<'PY'
import torch
import pytest
torch.backends.mps.is_available = lambda: False
raise SystemExit(pytest.main(["metal/tests", "-q", "-p", "no:cacheprovider"]))
PY
```

Native gates (the original SDF cases currently include known failures):

```sh
MUJOCO_METAL_RUN_GPU=1 PYTORCH_ENABLE_MPS_FALLBACK=0 \
  PYTORCH_MPS_FAST_MATH=0 PYTHONPATH=metal \
  python -m pytest -q metal/tests/test_sdf_contact_013.py \
  metal/tests/test_bundled_sdf_rigid_oracle_019.py
```

Clean-wheel installation, the complete native suite and requalification of all
current demos remain pending. Gallery clips are recorded fixtures, not release
qualification or performance measurements.

---

# Historical bounded qualification — 2026-09-29

## Scope

The recorded September 29 bounded source included connect/weld constraints and runtime activation,
mocap/keyframe lifecycle, stateful actuators, spatial tendon wrapping,
site-only tendon armature bias and ball-joint limits. These extend the bounded
integrated Euler profile; they do not establish unrestricted MuJoCo compatibility.
Cylinder/ellipsoid collision support and the eccentric-roller demo were not
included in that qualification. They are now development paths on main, with
full qualification still open.

Reference/runtime: MuJoCo 3.10.0, Python 3.12, Torch 2.9.1, Apple Silicon MPS
float32; CPU MuJoCo provides the numerical oracle. Native checks use
`PYTORCH_ENABLE_MPS_FALLBACK=0`. Qualification targets the M1 Max with
32 GB unified memory; this is not broad Mac or Linux qualification.

## Validation results

| Check | Result | Scope |
|---|---|---|
| Full suite with native GPU execution | 518 passed, 336.75 s | Recorded physics qualification |
| Focused armature/spatial/coupled suite | 105 passed | Native regression checks |
| Gripper and drawbridge native headless gates | Exit 0 for both | Bounded demo behavior and CPU comparisons |
| Full suite without Torch | 215 passed, 304 skipped, 2.57 s | Final publication check |
| Shader source inventory, including missing-file negative control | Included in the passing CPU suite | Source completeness |
| Built-wheel contents and installation for this increment | Pending | Source checks are not a wheel-install test |
| Interactive display and broader hardware/OS behavior | Not requalified | Separate validation required |

The publication check reran the CPU suite; it did not repeat the recorded GPU
qualification. Skipped CPU tests are not GPU passes. Algebra checks, compilation
and demo images are not substitutes for native physics qualification. Passing
the bounded test matrix does not establish every admitted feature combination.
No new performance measurement or speedup is claimed.

## Reproduce from a clean checkout

Follow [source installation](INSTALL.md#development-source) in a dedicated
Python 3.12 environment. For no-Torch qualification, install the pinned MuJoCo,
NumPy, pytest and absl-py dependencies into a separate environment without Torch,
and point `PYTHONPATH` at this source tree. Do not uninstall Torch from a working
GPU environment.

```sh
PYTHONPATH=metal python -m pytest metal/tests -q -p no:cacheprovider
```

On an idle supported Apple GPU in the source-install environment:

```sh
PYTORCH_ENABLE_MPS_FALLBACK=0 MUJOCO_METAL_RUN_GPU=1 PYTHONPATH=metal \
  python -m pytest metal/tests -q -p no:cacheprovider

PYTORCH_ENABLE_MPS_FALLBACK=0 PYTHONPATH=metal:metal/examples \
  python metal/examples/muscle_gripper.py --headless --check --steps 3000

PYTORCH_ENABLE_MPS_FALLBACK=0 PYTHONPATH=metal:metal/examples \
  python metal/examples/cable_drawbridge.py --headless --check --steps 1400
```

Record the commit, imported module path, shader hashes, dependency versions,
exit codes and raw output for any new qualification. Current `main` source
features are not a new PyPI release; the published 0.4.0 package retains its
previous scope. See [development coverage](DEVELOPMENT.md), the
[support inventory](COVERAGE.md), and individual demo guides for numerical bounds.
