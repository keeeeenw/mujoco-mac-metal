# Development qualification — 2026-10-11

Current `main` publishes the experimental development backend. See
[milestone status](STATUS.md): **0/16 milestones 005–020 have full-composition
sign-off**. Earlier bounded capabilities and individual regression groups retain
their recorded evidence. This is source publication, not a new PyPI release.

## Local source regression review — 2026-10-11

Revision `fc4cced35657f9bc24bf579fb0c39fd840cd95d9` on
`implementation/mac-milestone-completion` adds owned compensated force scratch,
pre-allocation memory admission, cached Jacobian/row ownership repairs,
selected-world recovery coverage and route-specific collision support fixes.
The prepared API host fixture now uses the real accumulator and preserves its
residual. Dense solver outputs start with defined values so a first masked
solve cannot expose allocator contents from an inactive world.

The final host run at that revision passed **2,419 tests, 99 skipped,
2,201 deselected, in 99.92 seconds**. It explicitly disabled MPS availability
for collection and excluded tests marked `gpu`; it is host regression
evidence, not native physics evidence. The host command below can be used
with pytest arguments `-m 'not gpu'` to reproduce this selection.

The independently executed native regression group passed **134 tests,
10 deselected, in 137.78 seconds**. It covers dense/awake solves, compensated
forces, retained Jacobians, masked recovery, cached constraint families, mesh
contacts/forces and cylinder support callbacks. The callback checks compare
both native CCD and libccd routes, with two rotations, against the installed
MuJoCo 3.10.0 binary. Their sampled bitwise matches do not qualify every
direction or other MuJoCo/compiler builds. Poisoned-allocation tests cover
first masked solves and retention of an inactive world's prior failure status.

A separate native spot check finished with **1 passed and 2 failed in
131.79 seconds**. Transformed/scaled heightfield frames passed. The original
PGS/Euler strict contact-J gate still fails (maximum violating error
`9.39097133e-11`, with `rtol=atol=2e-12`). The original zero-margin alternate
cylinder-heightfield trajectory also still fails its acceleration comparison
(maximum violating error `1.85741866`, with `rtol=atol=2e-3`). No tolerances
were relaxed. The remaining recorded zero-margin terrain cases were not
repeated in this spot check.

These results use Python 3.12.13, MuJoCo 3.10.0 and Torch 2.9.1 on Apple
Silicon, with MPS fallback and fast math disabled. All imports and shaders
resolve to this source checkout. They establish the covered regressions,
not full simulation parity, milestone completion, wheel qualification,
demo requalification or a performance improvement. The strict solver and
terrain gates, broader SDF/deformable/integrator/API combinations and final
distribution/demo gates remain open.

Reproduce the native regression group using the pinned source runtime.
`MUJOCO_METAL_TRACE_ROOT` must point to a directory containing
`.upstream/mujoco-3.10.0` for the callback ABI test; a skip is not a pass:

```sh
MUJOCO_METAL_RUN_GPU=1 PYTORCH_ENABLE_MPS_FALLBACK=0 \
PYTORCH_MPS_FAST_MATH=0 PYTHONPATH=metal:metal/tests \
python -m pytest -q -m gpu -p no:cacheprovider \
  metal/tests/test_solve.py metal/tests/test_awake_smooth_017.py \
  metal/tests/test_component_smooth_force_pair.py \
  metal/tests/test_cached_jacobian_capture.py \
  metal/tests/test_rhs_accumulate_owned.py \
  metal/tests/test_masked_recovery_production.py \
  metal/tests/test_cached_row_family_closure.py \
  metal/tests/test_mesh_contact_011.py metal/tests/test_mesh_force_011.py \
  metal/tests/test_common_ccd_support_source.py
```

Reproduce the two remaining numerical spot failures without changing their
original acceptance thresholds:

```sh
MUJOCO_METAL_RUN_GPU=1 PYTORCH_ENABLE_MPS_FALLBACK=0 \
PYTORCH_MPS_FAST_MATH=0 PYTHONPATH=metal:metal/tests \
python -m pytest -q -p no:cacheprovider \
  'metal/tests/test_contact_operator_precision.py::test_actual_contact_jacobian_reference_and_inverse_mass_rows[solver0-integrated_euler_v1]' \
  'metal/tests/test_primitive_hfield_full_manifold.py::test_full_primitive_terrain_contacts_forces_trajectory_and_restore[0.0-False-True-cylinder-.25 .15]'
```

## Publication scope

The host publication suite passed **2,163 tests, with 1,615 native-dependent
tests skipped, in 94.60 seconds** at source revision `61e415e8b` (exit 0).
Tracked source was verified unchanged before and after execution. Subsequent
publication edits record this result and correct documentation links only.
See the [machine-readable record](validation/publication-20261005.json).
This result does not establish native correctness for the skipped cases.

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
