# Experimental MuJoCo Metal package

> **Development branch:** the published 0.3.0 release has the bounded
> contact-free Euler features documented below. New source profiles add RK4,
> rigid passive forces, stateless joint/tendon servos, current-state sensors,
> body-fluid drag, bounded implicitfast, sphere contact and joint constraints. See [development coverage](DEVELOPMENT.md) and the
> [creative demo gallery](examples/demo_gallery.md) for scope and evidence.
> These increments do not establish full MuJoCo support.


**Current status: experimental bounded native stepping, not a complete simulation backend.**
This generalized branch computes kinematics, `M(q)`, and inertial/gravity bias,
solves dense SPD systems, integrates state, and provides a narrowly qualified
native `contact_free_euler_v1` simulation profile. It does not replace `mj_step`
or support the full MuJoCo model and force system. The earlier robot-specific
implementation is preserved on
[`archive/metal-microduck-v1`](https://github.com/keeeeenw/mujoco-mac-metal/tree/archive/metal-microduck-v1);
its robot-specific contact pipeline is not part of this generalized package.

This optional package targets Python 3.12 and MuJoCo **3.10.0**. The surrounding MuJoCo source checkout is 3.14.1; that version is not a target. Use an isolated environment so the pinned package does not alter the surrounding checkout:

```sh
cd metal
python3.12 -m venv .venv-metal
source .venv-metal/bin/activate
python -m pip install -e '.[metal,test]'
```

Torch installs automatically on Apple Silicon macOS. On other platforms, `.[test]` installs CPU utilities without Torch. Importing `mujoco_metal` and running preflight do not import Torch or initialize MPS.

Run `python -m mujoco_metal preflight --model path/to/model.xml --json --inventory` to inspect runtime version, model dimensions, package and all six shader paths/hashes, capability boundaries, and the versioned feature/API inventory. The overall `gpu_qualified` field remains false because the full backend is not qualified. Inventory completeness is explicitly false because the captured enum and Python binding inventory is not exhaustive.

`load_model(xml_or_path)` returns an immutable, dimension-derived descriptor. `descriptor.forward_kinematics(qpos)` is a CPU reference for body, inertial, geom, site, and joint-anchor/axis world poses across hinge, slide, ball, and free joints. `MetalKinematics(descriptor).run(qpos_batch)` computes the same pose fields through the batched native Metal kinematics kernel. GPU checks have passed on an Apple M1 for empty and fixed worlds, mixed hinge/slide/ball/free models with off-center joints and multiple free roots, world-attached sites/geoms, and a 32-DOF chain. This is a narrow correctness qualification, not a general model-support or performance claim. Constructing `MetalKinematics` initializes MPS and compiles the bundled shader.

`smooth_dynamics(descriptor, qpos, qvel)` is the CPU reference returning a dense joint-space mass matrix and inertial/gravity bias forces. `MetalSmoothDynamics(descriptor).run(qpos_batch, qvel_batch)` accepts host NumPy state batches and returns those two outputs as MPS tensors. Native GPU checks have passed on an Apple M1 across empty/fixed, mixed-joint, rotated-inertia, massless-ancestor, disabled-gravity, and 32-DOF-chain fixtures. `MetalDenseSolve(nv, batch_size, nrhs=1).run_device(mass, rhs)` performs a dimension-derived dense Cholesky factorization and one or multiple right-hand-side solves. `MetalEulerIntegration(descriptor, batch_size, timestep).run_device(qpos, qvel, qacc, time, solve_status)` updates hinge, slide, ball, and free coordinates on MPS. Both primitives have narrow GPU correctness qualifications against synthetic systems or MuJoCo 3.10; neither alone establishes unrestricted dynamics support. These device primitives require contiguous float32 MPS tensors; their outputs are borrowed reusable views, overwritten by the next call.

For persistent GPU inputs, construct `MetalKinematics` or `MetalSmoothDynamics`
with `batch_size=B`, then call `run_device` with contiguous float32 MPS tensors.
The host-array `run` methods remain available for reference comparisons and tools.
`MetalSimulation` uses the device methods internally, without a per-step host
state transfer. Device outputs are borrowed and overwritten by later calls; copy
results when retaining them beyond the next invocation.

`ModelLifecycle` performs transactional CPU body-mass updates through MuJoCo `mj_setConst`. `BatchedConstants` maintains per-environment body masses and derived `body_invweight0` rows with atomic recomputation/restore and seeded mass randomization. `KinematicsBatchState` tracks explicit environment rows with generation-based FK cache invalidation, snapshots, restore, and tangent-space joint randomization. These are CPU lifecycle utilities and do not advance physics.

`MetalSimulation(model, batch_size=1, qpos=None, qvel=None)` connects persistent MPS state, smooth dynamics, the native dense solve, and semi-implicit Euler for the bounded `contact_free_euler_v1` profile. A local Apple M1 GPU qualification passed on four rigid-body fixtures, three initial states, 1,000-step 1 ms rollouts, and reset/restore resume. This remains a narrow qualification, not general model support. The model must explicitly disable contacts and use Euler integration. The profile supports gravity and joint armature, assumes zero applied generalized force, and rejects actuators, tendons, limits, friction loss, passive/fluid forces, sensors, equality constraints, flexes, plugins, damping, springs, callbacks, mocap, and non-Euler integrators. A model may be accepted for kinematics or mass/bias queries and still be rejected for stepping. Construction validates the profile before initializing MPS.

`simulation.step()` returns a borrowed MPS status vector. Solver and integration failures are per-world; a failed world keeps its state and its first nonzero status remains sticky until reset or restore. State views are copies; snapshots and resets cross the host/device boundary and belong outside the hot step loop. Position, velocity, acceleration, and simulation time use float32 on device. In particular, time's representable increment gets coarser as elapsed time grows.

Run the opt-in GPU correctness tests only on an available Apple GPU with the pinned Torch extra installed: `MUJOCO_METAL_RUN_GPU=1 python -m pytest -m gpu`. Ordinary `python -m pytest` runs CPU tests and skips the GPU cases. `preflight` remains CPU-only: shader hashes and stage labels are inventory, not a device probe. The overall GPU-qualified field stays false because full-library physics coverage remains incomplete. The standalone source tree carries the Apache 2.0 license and notices.

## Applied forces, damping and basic motors

Select a profile explicitly; the original `contact_free_euler_v1` remains the
unforced baseline. The published 0.3.0 profiles require contacts disabled and Euler integration.

| Profile | Additional supported behavior |
| --- | --- |
| `contact_free_forces_euler_v1` | Per-call generalized forces and nonnegative linear joint damping. |
| `contact_free_motor_euler_v1` | The force profile plus stateless, fixed-gain, no-bias motors transmitted to hinge/slide joints. |

```python
simulation = MetalSimulation(model, batch_size=B,
                             profile="contact_free_motor_euler_v1")
simulation.step(ctrl=controls, qfrc_applied=forces)
```

Controls have shape `[B, nu]`; generalized forces have shape `[B, nv]`. Supply
finite host arrays or contiguous float32 MPS tensors. Inputs are held across
`step(steps=N, ...)` and expire at the next call; omitted values mean zero.
Inputs are copied and never modified. Host input validation/upload occurs before
the device loop; device inputs avoid host readback. Active invalid device inputs
propagate to per-world failure handling. Reset/restore clears sticky failures.

Damping follows MuJoCo's implicit Euler effective-mass solve, including the
`DAMPER` and `EULERDAMP` disable flags. Stored `qacc` remains physical acceleration;
Euler uses its separate effective acceleration to advance state. Motors support
control/actuator force clipping, summed moments and global/group disable. Other
transmissions, activation dynamics, non-fixed gains/biases, nonzero actuator
armature/damping and joint-level actuator-force limits are rejected.

Independent M1 checks covered 36 force/damping and 15 motor trajectories of
1,000 steps against CPU MuJoCo. Maximum observed qpos/qvel/physical-qacc errors
were approximately 4.83e-6 / 8.75e-6 / 2.54e-4 for force/damping, and
2.48e-6 / 5.15e-6 / 1.69e-5 for motors. These are fixture-specific results, not
universal error bounds. Public regressions cover malformed inputs, row failures,
reset/restore, changing inputs and the device-only stepping path.

Smooth M/bias queries can now accept models with actuators only if their actuator
armature is zero; they still do not compute actuator forces. Simulation uses the
stricter motor-profile validation. This distinction prevents unsupported inertia
terms from silently disappearing.

Try the [spacecraft force-control demo](examples/space_docking.md), or use the
[installation and diagnostic guide](INSTALL.md) for `pip install mujoco-mac-metal`
and `mujoco-metal doctor --gpu`. The experimental 0.3.0 release is on PyPI.

## Achievements and measured scaling

The generalized package now has a **native contact-free stepping loop**: persistent
MPS state, forward kinematics, mass/bias computation, dense acceleration solve,
and quaternion-aware semi-implicit Euler integration. Physics stepping does not
call CPU MuJoCo, NumPy solves, or read state back to the host. Reset/checkpoint
ownership and per-world failure handling are covered by tests. These are new
backend implementations of existing dynamics methods, not new physics algorithms.

Local qualification includes **133 passing tests with GPU execution enabled**, plus
12 independent 1,000-step trajectories against MuJoCo across slide, hinge,
free-body, and mixed-joint fixtures. The native pendulum also passed headless and
offscreen checks. Interactive native playback still needs qualification with an
active macOS display; the earlier hybrid viewer was checked separately.

A short **M1 Max / 32 GB** pendulum benchmark now includes measured eight-thread
CPU comparisons at every tested batch through **524,288 worlds**. At that batch,
Metal measured **3.20x faster** than eight-thread CPU rollout (5.907 s versus
18.905 s for 200 steps), reaching **17.75 million world-steps/s**. CPU wins at
small batches. See the
[complete CPU8/Metal results, raw trials, and reproduction commands](benchmarks/README.md).
These are contact-free physics API measurements, not walking or PPO results.
CPU rollout writes float64 trajectories while Metal retains float32 device state;
that difference is included in the reported comparison. No full-library or
cross-hardware speedup is claimed.

## Published 0.3.0 boundaries

For the additional source-branch capabilities, see [DEVELOPMENT.md](DEVELOPMENT.md).

| Stage | Status in this generalized package |
| --- | --- |
| Kinematics, dense mass matrix, inertial/gravity bias | Native Metal; qualified on the documented small fixtures. |
| Dense SPD factorization and multiple-RHS solve | Implemented and narrowly GPU-qualified on synthetic scaled/conditioned systems; dense float32 numerical limits remain. |
| Hinge/slide/ball/free semi-implicit Euler integration | Implemented and narrowly GPU-qualified against MuJoCo 3.10. |
| `contact_free_euler_v1` native simulation profile | Implemented and narrowly GPU-qualified on the local Apple M1: four rigid-body model fixtures, three initial states, 1,000 steps at 1 ms, and reset/restore resume. The largest observed absolute component differences were `1.26e-5` in position and `7.32e-5` in velocity; quaternion norm error was at most `1.2e-7`. This is not broad model or hardware qualification. |
| Applied forces, passive forces, actuators and tendon dynamics | Applied generalized forces, linear joint damping and bounded scalar-joint motors are supported in explicit profiles. Cartesian forces, other passive/actuator classes and all tendons remain unsupported. |
| Collision/contact generation, joint limits, equality constraints, friction and constraint solvers | Unsupported. The qualified profile requires contact disabled and rejects limits and constraints, even when they are inactive in the initial state. |
| Device reset/checkpoint lifecycle and per-environment model randomization | Persistent MPS state with host reset/checkpoint and row reset is connected; per-environment model randomization is not connected to native stepping. |
| Sensors, remaining integrators, flexes/plugins, broad API and precision compatibility | Unimplemented or unqualified; full MuJoCo coverage is not established. |
| Native rendering and end-to-end training integration | Outside the implemented scope. |

## Runnable Mac demo

![Metal hybrid pendulum alongside CPU MuJoCo](examples/assets/pendulum.gif)

Left: Metal mass/bias with CPU solve and integration. Right: CPU MuJoCo.
The GIF plays three seconds of simulation at a fixed presentation rate; it
does not show measured execution speed. Both use OpenGL rendering.

The [side-by-side chaotic pendulum demo](examples/README.md) keeps its recorded
GIF labeled as a **hybrid** rollout: Metal mass/bias followed by CPU solve and
integration, alongside independent CPU MuJoCo. The same example now has an
explicit `--mode metal` that advances its contact-free model with native MPS
dynamics, solve and integration. OpenGL still renders both sides; it is not a
Metal renderer. The native example has no CPU physics fallback and is not a
performance result. See the example instructions for launch, numerical checks
and limitations.

## FAQ: MuJoCo, Metal, and Apple Silicon

Checked on **2026-09-26**. These answers distinguish upstream documentation,
version-specific issue reports, community projects, and this package's local
validation. External projects were not installed or benchmarked for this FAQ.
Our numerical contract remains **MuJoCo 3.10.0**; newer upstream source and
documentation do not automatically extend this package's support.

### Does installing MuJoCo on a Mac enable Metal physics?

The standard C engine's `mj_step` runs CPU physics. GPU simulation uses a
separate implementation; upstream documents MJX and MuJoCo Warp alongside the
C engine. Installing this experimental package does not replace `mj_step` or
redirect arbitrary MuJoCo applications to Metal. See the
[upstream overview](https://mujoco.readthedocs.io/en/stable/overview.html) and
[our explicit native entry points](mujoco_metal/smooth_metal.py).

Physics, rendering, and neural-network training are separate workloads. A GPU
viewer does not demonstrate GPU physics, and GPU PPO can accompany CPU physics.
Likewise, a Metal physics kernel does not make a viewer use Metal.

### Does the classic visualizer require OpenGL 3.3 or newer?

The documented requirement is **OpenGL 1.5 compatibility functionality**, with
`ARB_framebuffer_object` and `ARB_vertex_buffer_object`. The classic renderer
uses fixed-function OpenGL. Forcing a forward-compatible 3.3 **core** context
can remove functionality it needs; that is not a general macOS fix. See
[MuJoCo's OpenGL requirements](https://mujoco.readthedocs.io/en/stable/programming/visualization.html#using-opengl),
also present in the
[3.10.0 documentation source](https://github.com/google-deepmind/mujoco/blob/3.10.0/doc/programming/visualization.rst).

Apple's OpenGL deprecation does not itself make existing OpenGL applications
Metal-native or prevent them from running. Context/profile behavior also
depends on macOS and GLFW versions; consult
[GLFW's macOS notes](https://www.glfw.org/docs/latest/compat_guide.html#compat_osx).

### How should I investigate `OpenGL ARB_framebuffer_object required`?

Start with MuJoCo's own viewer or example for the installed version. In custom
windowing code, verify window/context creation succeeded and make that context
current on the calling thread **before** `mjr_makeContext`. Check the actual GL
version and extension support, and remove incompatible core-profile hints.
`mjrContext` manages MuJoCo rendering resources; it does not create the operating
system's OpenGL context. Offscreen rendering still needs a suitable context.
See [context setup](https://mujoco.readthedocs.io/en/stable/programming/visualization.html#context-and-gpu-resources).

For Python's `mujoco.viewer.launch_passive` on macOS, use
`mjpython your_viewer_script.py` from the intended environment to satisfy its
main-thread requirement. This applies to that viewer API, not every Python
simulation script. See the
[passive-viewer documentation](https://mujoco.readthedocs.io/en/stable/python.html#passive-viewer).

The author of [discussion #2970](https://github.com/google-deepmind/mujoco/discussions/2970)
ultimately identified a `glutin` problem. The thread's suggested 3.3-core-context
recipe should not override MuJoCo's renderer requirements.

### What about upstream Filament or other Metal renderers?

Renderer support must be checked separately for each release and build.
The surrounding upstream-derived source includes a Filament integration, and
[its documentation](../doc/programming/visualization.rst) notes that Filament
itself supports Metal. However, the
[backend selector in this source snapshot](../src/render/filament/core/filament_platform_factory.cc)
selects OpenGL or Vulkan; it does not expose a direct Metal selection.
Filament's capabilities alone therefore do not establish a working native
Metal MuJoCo viewer. This package supplies no renderer, and we have not
qualified that Filament integration on macOS.

### Can MJX use the Apple GPU through JAX-Metal?

MJX-JAX needs a backend that can compile and execute the operations used by the
chosen model and solver. Seeing an Apple GPU in JAX, or successfully loading a
model, does not establish that `mjx.step` works. Current
[MJX documentation](https://mujoco.readthedocs.io/en/stable/mjx.html) distinguishes
JAX and Warp implementations; the older 3.1.5 documentation is not a current
compatibility matrix. Apple's
[JAX-Metal page](https://developer.apple.com/metal/jax/) explains its compiler
path and version requirements.

There are concrete failure reports, with different scopes:

- [stac-mjx #126](https://github.com/talmolab/stac-mjx/issues/126) reports
  `mhlo.cholesky` legalization failure with `jax-metal 0.1.1` and JAX 0.5.0,
  and a separate compatibility failure with JAX 0.7.2. It describes
  `mhlo.triangular_solve` as a possible additional blocker, not a demonstrated
  failure in that report.
- [stretch_mujoco #40](https://github.com/hello-robot/stretch_mujoco/issues/40)
  reports `mhlo.reduce` failure and separate model/rendering limitations.

These are useful reproductions to investigate, not proof that every version,
model, or future backend fails. Record the exact macOS, JAX, jaxlib, plugin,
MuJoCo and model versions when reproducing them. We have not requalified those
JAX-Metal combinations here.

### Are Cholesky factorization and triangular solves impossible on Metal/MPS?

No. Apple exposes
[MPSMatrixDecompositionCholesky](https://developer.apple.com/documentation/metalperformanceshaders/mpsmatrixdecompositioncholesky)
and [MPSMatrixSolveTriangular](https://developer.apple.com/documentation/metalperformanceshaders/mpsmatrixsolvetriangular).
A missing compiler lowering in a JAX plugin does not mean the underlying
hardware or MPS library lacks the operation. Metal, MPS, MPSGraph, JAX-Metal,
PyTorch's MPS backend, and MLX have distinct APIs and operation coverage.
This package now includes a small dense Cholesky and triangular-solve kernel
through Torch's MPS shader API. It has narrow correctness qualification for
synthetic systems and the bounded contact-free simulation profile; it is not a
general MuJoCo sparse or constrained solver. Numerical stability, supported
layouts, and workload-specific performance still require assessment.

### How do the community alternatives compare?

These projects are worth evaluating against a specific model and workload.
The descriptions below summarize their own documentation, not a compatibility
or speed certification from this project.

| Project | Documented approach | Qualification to keep in mind |
| --- | --- | --- |
| [RobotFlow-Labs/Mujoco-mlx](https://github.com/RobotFlow-Labs/Mujoco-mlx) | Ports MJX operations from JAX to Apple MLX; reports working stepping and simple physics comparisons. | Its README also mentions moving linear algebra to a CPU stream. MLX usage alone does not establish that every operation runs on GPU, or that all MuJoCo features match. |
| [genesisinteractive/MuJoCo-MLX-Cpp](https://github.com/genesisinteractive/MuJoCo-MLX-Cpp) | C++/MLX physics with a batched Metal pipeline and a dual CPU/GPU API. The supplied `arghyasur1991` URL redirects here. | Review its [conformance report](https://github.com/genesisinteractive/MuJoCo-MLX-Cpp/blob/main/CONFORMANCE.md), scalar versus batched paths, and model coverage. Its hardware/workload-specific benchmark numbers are not measurements of this package. |
| [Genesis World](https://github.com/Genesis-Embodied-AI/genesis-world) | A separate simulator whose [installation documentation](https://genesis-world.readthedocs.io/en/latest/user_guide/overview/installation.html) lists Apple Silicon simulation through `gs.metal`. | Current source describes the Quadrants compiler, forked from Taichi. Treat older Taichi-only descriptions as historical. Metal availability does not establish MuJoCo API, contact-dynamics, or trained-policy equivalence; rendering and optional components have their own requirements. |

### Does this package use JAX or MLX, and what is actually validated?

It uses custom Metal Shading Language kernels launched through
`torch.mps.compile_shader`, with MPS tensors for outputs. JAX and MLX are not
dependencies of this package. The
[launcher](mujoco_metal/smooth_metal.py), [dependency pins](pyproject.toml), and
[GPU tests](tests/test_gpu.py) document that path.

The current validation covers batched kinematics, dense mass matrices,
inertial/gravity bias, dense SPD solves, joint-coordinate integration, and the
bounded `contact_free_euler_v1` pipeline on local M1 fixtures compared with
MuJoCo 3.10. Additional development profiles cover bounded springs, fluids,
servos/fixed tendons, RK4/implicitfast, sphere contact, joint constraints and
current-state sensors. [Development coverage](DEVELOPMENT.md) lists each subset,
its CPU-reference evidence and the substantial remaining gaps. Native rendering,
training and full-library coverage remain unsupported. Features outside the selected profile are rejected.
Validation of a separate robot-specific backend does not extend this
generalized package's coverage.

### Will it work on every M1–M5 Mac, and is it faster than CPU physics?

Support across every chip generation is not established. Apple Silicon is the
intended hardware family, but macOS, framework versions and memory requirements
still need qualification. The [measured pendulum comparison](benchmarks/README.md)
shows a batch-dependent advantage over eight-thread CPU rollout on one M1 Max.
It also shows CPU winning at small batches. Precision and output costs differ,
and these short measurements establish no training-throughput advantage.

Use `python -m mujoco_metal preflight --json --inventory` to record this
installation's module/shader paths, hashes, version and declared support.
Preflight is CPU-only; it is not a runtime GPU probe. After the isolated install
above, run `MUJOCO_METAL_RUN_GPU=1 python -m pytest -q tests/test_gpu.py` from
`metal/` for the scoped GPU checks. Benchmark separately on idle hardware with
matching physics, precision, environment counts and synchronization, including
state transfers and the full workload being claimed.
