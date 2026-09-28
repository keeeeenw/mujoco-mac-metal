# FAQ: MuJoCo, Metal, and Apple Silicon

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
MuJoCo 3.10. Additional 0.4.0 profiles cover bounded springs, fluids,
servos/fixed tendons, RK4/implicitfast, sphere contact, joint constraints and
current-state sensors. Current main additionally qualifies integrated Euler with
primitive collision manifolds, pyramidal/elliptic condim1/3/4/6 friction and
coupled scalar constraints. Those additions require the
[source installation](INSTALL.md#development-source); they are not included in
the published 0.4.0 wheel. Connect/weld equalities and runtime equality activation
remain unsupported. [Development coverage](DEVELOPMENT.md) lists each subset,
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
Preflight is CPU-only; it is not a runtime GPU probe. After the [isolated source install](INSTALL.md#development-source), run `MUJOCO_METAL_RUN_GPU=1 PYTORCH_ENABLE_MPS_FALLBACK=0 python -m pytest -q tests`
from `metal/` for the full package suite. `tests/test_gpu.py` alone covers only
the earlier kinematics/mass/bias checks, not all stepping profiles. Benchmark separately on idle hardware with
matching physics, precision, environment counts and synchronization, including
state transfers and the full workload being claimed.
