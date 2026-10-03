# Physics qualification — 2026-09-29

## Scope

Current source includes connect/weld constraints and runtime activation,
mocap/keyframe lifecycle, stateful actuators, spatial tendon wrapping,
site-only tendon armature bias and ball-joint limits. These extend the bounded
integrated Euler profile; they do not establish unrestricted MuJoCo compatibility.
Cylinder/ellipsoid collision support and the eccentric-roller demo remain in
development and are not included in this qualification.

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
