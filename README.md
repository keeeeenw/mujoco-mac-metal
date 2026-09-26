# MuJoCo Mac Metal

Experimental native Apple GPU physics for MuJoCo, maintained by
[keeeeenw](https://github.com/keeeeenw). This community project builds on
[Google DeepMind's MuJoCo](https://github.com/google-deepmind/mujoco) and adds an
optional [Metal package](metal/README.md). It is not an official MuJoCo release.

**Native contact-free stepping works for a bounded model profile. Full MuJoCo
Metal support remains a work in progress.** The ordinary MuJoCo build remains
available alongside the optional package.

## Our contributions

- **Native Metal physics pipeline:** model-derived forward kinematics, dense mass
  matrices and inertial/gravity bias, a dense Cholesky acceleration solve, and
  quaternion-aware semi-implicit Euler integration. Hinge, slide, ball and free
  joints are supported within the explicit `contact_free_euler_v1` profile.
- **Persistent GPU simulation state:** batched stepping without CPU physics or
  per-step host state readback, per-world failure handling, selected-row reset,
  and checkpoint ownership/restore. Host-side lifecycle utilities also cover
  model-constant recomputation and invalidation.
- **Validation and reproducible measurements:** 133 tests passed with GPU execution
  enabled; independent CPU-reference trajectory and solve checks supplement the
  suite. Benchmarks include actual eight-thread CPU measurements at every tested
  batch through 524,288 worlds.
- **Forces and controls:** explicit generalized-force and linear-damping support,
  plus bounded hinge/slide motors with clipping and disable flags. Separate
  profiles preserve the original unforced baseline.
- **Tools and documentation:** spacecraft and pendulum comparisons, capability inventory,
  runtime/shader provenance, Apple Silicon FAQ, and a portable benchmark runner.
- **Upstream regression coverage:** an `mj_setConst` inertial-update roundtrip test
  and documentation clarifying compiled simple-body/sparsity limitations.

These contributions implement and qualify existing dynamics and numerical
methods on Metal; MuJoCo's physics methods and PPO/RL algorithms are not our
inventions. See [LICENSE](LICENSE) and [Metal attribution](metal/NOTICE).

## Measured performance on M1 Max

For the contact-free four-DOF pendulum on an **M1 Max with 32 GB unified memory**,
200 steps at 524,288 environments took **18.905 s on eight-thread CPU rollout
versus 5.907 s on Metal: 3.20x faster**, reaching **17.75 million world-steps/s**.
CPU wins at small batches; gains level off at large batches.

This is a short physics API benchmark, not a walking or PPO training result.
CPU rollout records float64 trajectories; Metal retains float32 device state.
Those different precision and output costs are included in the comparison.
[Complete results, raw trials and reproduction commands](metal/benchmarks/README.md)
explain the measurement boundaries.

## Install and check

In an isolated Python 3.12 environment, install directly from this repository:

```sh
python -m pip install "mujoco-mac-metal[metal] @ git+https://github.com/keeeeenw/mujoco-mac-metal.git@main#subdirectory=metal"
PYTORCH_ENABLE_MPS_FALLBACK=0 mujoco-metal doctor --gpu
```

The proposed `pip install mujoco-mac-metal` release is not on PyPI yet.
See [installation and diagnostics](metal/INSTALL.md) for requirements and status.

## Try the Mac demos

![Native Metal spacecraft approach beside CPU MuJoCo](metal/examples/assets/space_docking.gif)

The [spacecraft approach demo](metal/examples/space_docking.md) uses three free
bodies, applied forces/torques and different damping values, with an independent
CPU reference. Targets are visual only: no docking contacts or latching are
modeled. Its controller runs on the host; native physics runs on Metal.

The original pendulum comparison is also available:

![Hybrid Metal pendulum compared with CPU MuJoCo](metal/examples/assets/pendulum.gif)

The [demo guide](metal/examples/README.md) provides installation and playback
commands. Select `--mode metal` for native contact-free physics, `--mode metal-hybrid` for Metal mass/bias with CPU solve/integration, or `--mode cpu`.
**The GIF shows the hybrid mode at a fixed playback rate, not native benchmark
speed.** Rendering uses MuJoCo's OpenGL visualizer. Native headless/offscreen
checks passed; native interactive playback still needs qualification with an
active macOS display.

## Current limits and next steps

The optional package targets Python 3.12, MuJoCo **3.10.0** and Torch **2.9.1**;
use its isolated installation instructions rather than treating the newer
surrounding MuJoCo source version as the qualified runtime.

All current profiles require contacts disabled and Euler integration. Explicit
profiles support generalized forces, linear damping and basic scalar-joint motors;
other actuation, passive/fluid forces, tendons, contacts and
constraints, sensors, other integrators, native rendering and training integration
remain unsupported. Per-environment model randomization is not connected to
native stepping. Linux/CUDA integration and other Apple hardware/OS combinations
have not been validated for this Metal package. Full upstream-core and
single-precision compatibility are not established.

Next milestones are contacts and constraints, broader force/actuator support,
then broader model/API coverage and sustained application-level qualification.
See the [support inventory and FAQ](metal/README.md) for precise boundaries.

## Contribute

Community testing, code review and pull requests are welcome—especially numerical
edge cases, other Apple hardware, better matched CPU/GPU benchmarks, and ideas
for code structure. Follow [CONTRIBUTING.md](CONTRIBUTING.md) and include the model,
software versions and reproduction commands with reports.

Upstream proposals: [native Metal support #3626](https://github.com/google-deepmind/mujoco/pull/3626)
and [inertial-update regression #3624](https://github.com/google-deepmind/mujoco/pull/3624).
The broader request is [MuJoCo issue #95](https://github.com/google-deepmind/mujoco/issues/95).
The older robot-specific implementation remains on
[`archive/metal-microduck-v1`](https://github.com/keeeeenw/mujoco-mac-metal/tree/archive/metal-microduck-v1).

## Upstream MuJoCo

The following documentation describes the underlying upstream project. Its
installation and feature descriptions do not imply Metal acceleration.

<h1>
  <a href="#"><img alt="MuJoCo" src="banner.png" width="100%"/></a>
</h1>

<p>
  <a href="https://github.com/google-deepmind/mujoco/actions/workflows/build.yml?query=branch%3Amain" alt="GitHub Actions">
    <img src="https://img.shields.io/github/actions/workflow/status/google-deepmind/mujoco/build.yml?branch=main">
  </a>
  <a href="https://mujoco.readthedocs.io/" alt="Documentation">
    <img src="https://readthedocs.org/projects/mujoco/badge/?version=latest">
  </a>
  <a href="https://github.com/google-deepmind/mujoco/blob/main/LICENSE" alt="License">
    <img src="https://img.shields.io/github/license/google-deepmind/mujoco">
  </a>
</p>

**MuJoCo** stands for **Mu**lti-**Jo**int dynamics with **Co**ntact. It is a
general purpose physics engine that aims to facilitate research and development
in robotics, biomechanics, graphics and animation, machine learning, and other
areas which demand fast and accurate simulation of articulated structures
interacting with their environment.

Upstream MuJoCo is maintained by [Google DeepMind](https://www.deepmind.com/).

MuJoCo has a C API and is intended for researchers and developers. The runtime
simulation module is tuned to maximize performance and operates on low-level
data structures that are preallocated by the built-in XML compiler. The library
includes interactive visualization with a native GUI, rendered in OpenGL. MuJoCo
further exposes a large number of utility functions for computing
physics-related quantities.

We also provide [Python bindings] and a plug-in for the [Unity] game engine.

## Documentation

MuJoCo's documentation can be found at [mujoco.readthedocs.io]. Upcoming
features due for the next release can be found in the [changelog] in the
"latest" branch.

## Getting Started

There are two easy ways to get started with MuJoCo:

1. **Run `simulate` on your machine.**
[This video](https://www.youtube.com/watch?v=P83tKA1iz2Y) shows a screen capture
of `simulate`, MuJoCo's native interactive viewer. Follow the steps described in
the [Getting Started] section of the documentation to get `simulate` running on
your machine.

2. **Explore our online IPython notebooks.**
If you are a Python user, you might want to start with our tutorial notebooks
running on Google Colab:

 - The **introductory** tutorial teaches MuJoCo basics:
   [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/google-deepmind/mujoco/blob/main/python/tutorial.ipynb)
 - The **Model Editing** tutorial shows how to create and edit models procedurally:
   [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/google-deepmind/mujoco/blob/main/python/mjspec.ipynb)
 - The **rollout** tutorial shows how to use the multithreaded `rollout` module:
   [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/google-deepmind/mujoco/blob/main/python/rollout.ipynb)
 - The **LQR** tutorial synthesizes a linear-quadratic controller, balancing a
   humanoid on one leg:
   [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/google-deepmind/mujoco/blob/main/python/LQR.ipynb)
 - The **least-squares** tutorial explains how to use the Python-based nonlinear
   least-squares solver:
   [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/google-deepmind/mujoco/blob/main/python/least_squares.ipynb)
 - The **MJX** tutorial provides usage examples of
   [MuJoCo XLA](https://mujoco.readthedocs.io/en/stable/mjx.html), a branch of MuJoCo written in JAX:
   [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/google-deepmind/mujoco/blob/main/mjx/tutorial.ipynb)
 - The **differentiable physics** tutorial trains locomotion policies with
   analytical gradients automatically derived from MuJoCo's physics step:
   [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/google-deepmind/mujoco/blob/main/mjx/training_apg.ipynb)

## Installation

### Prebuilt binaries

Versioned releases are available as precompiled binaries from the GitHub
[releases page], built for Linux (x86-64 and AArch64), Windows (x86-64 only),
and macOS (universal). This is the recommended way to use the software.

### Building from source

Users who wish to build MuJoCo from source should consult the [build from
source] section of the documentation. However, note that the commit at
the tip of the `main` branch may be unstable.

### Python (>= 3.10)

The native Python bindings, which come pre-packaged with a copy of MuJoCo, can
be installed from [PyPI] via:

```bash
pip install mujoco
```

Note that Pre-built Linux wheels target `manylinux2014`, see
[here](https://github.com/pypa/manylinux) for compatible distributions. For more
information such as building the bindings from source, see the [Python bindings]
section of the documentation.

## Versioning

We aim to release MuJoCo in the first week of each month. Our versioning
standards changed to modified Semantic Versioning in 3.5.0,
see [versioning](VERSIONING.md) for details.

## Contributing

We welcome community engagement: questions, requests for help, bug reports and
feature requests. To read more about bug reports, feature requests and more
ambitious contributions, please see our [contributors guide](CONTRIBUTING.md)
and [style guide](STYLEGUIDE.md).

## Asking Questions

Questions and requests for help are welcome as a GitHub
["Asking for Help" Discussion](https://github.com/google-deepmind/mujoco/discussions/categories/asking-for-help)
and should focus on a specific problem or question.

## Bug reports and feature requests

GitHub [Issues](https://github.com/google-deepmind/mujoco/issues) are reserved
for bug reports, feature requests and other development-related subjects.

## Related software
MuJoCo is the backbone for numerous environment packages. Below we list several
bindings and converters.

### Bindings

These packages give users of various languages access to MuJoCo functionality:

#### First-party bindings:

- [Python bindings](https://mujoco.readthedocs.io/en/stable/python.html)
  - [dm_control](https://github.com/google-deepmind/dm_control), Google
    DeepMind's related environment stack, includes
    [PyMJCF](https://github.com/google-deepmind/dm_control/blob/main/dm_control/mjcf/README.md),
    a module for procedural manipulation of MuJoCo models.
- [JavaScript bindings and WebAssembly support](/wasm/README.md) (inspired [stillonearth](https://github.com/stillonearth) and [zalo](https://github.com/zalo)'s community projects; [mjswan](https://github.com/ttktjmt/mjswan) extends these with real-time policy control, interactive force
application, and more).
- [C# bindings and Unity plug-in](https://mujoco.readthedocs.io/en/stable/unity.html)

#### Third-party bindings:

- **MATLAB Simulink**: [Simulink Blockset for MuJoCo Simulator](https://github.com/mathworks-robotics/mujoco-simulink-blockset)
  by [Manoj Velmurugan](https://github.com/vmanoj1996).
- **Swift**: [swift-mujoco](https://github.com/liuliu/swift-mujoco)
- **Java**: [mujoco-java](https://github.com/CommonWealthRobotics/mujoco-java)
- **Julia**: [MuJoCo.jl](https://github.com/JamieMair/MuJoCo.jl)
- **Rust**: [MuJoCo-rs](https://github.com/davidhozic/mujoco-rs)

### Converters

- **OpenSim**: [MyoConverter](https://github.com/MyoHub/myoconverter) converts
  OpenSim models to MJCF.
- **SDFormat**: [gz-mujoco](https://github.com/gazebosim/gz-mujoco/) is a
  two-way SDFormat <-> MJCF conversion tool.
- **OBJ**: [obj2mjcf](https://github.com/kevinzakka/obj2mjcf)
  a script for converting composite OBJ files into a loadable MJCF model.
- **onshape**: [Onshape to Robot](https://github.com/rhoban/onshape-to-robot)
  Converts [onshape](https://www.onshape.com/en/) CAD assemblies to MJCF.

## Citation

If you use MuJoCo for published research, please cite:

```
@inproceedings{todorov2012mujoco,
  title={MuJoCo: A physics engine for model-based control},
  author={Todorov, Emanuel and Erez, Tom and Tassa, Yuval},
  booktitle={2012 IEEE/RSJ International Conference on Intelligent Robots and Systems},
  pages={5026--5033},
  year={2012},
  organization={IEEE},
  doi={10.1109/IROS.2012.6386109}
}
```

## License and Disclaimer

Copyright 2021 DeepMind Technologies Limited.

Box collision code ([`engine_collision_box.c`](https://github.com/google-deepmind/mujoco/blob/main/src/engine/engine_collision_box.c))
is Copyright 2016 Svetoslav Kolev.

ReStructuredText documents, images, and videos in the `doc` directory are made
available under the terms of the Creative Commons Attribution 4.0 (CC BY 4.0)
license. You may obtain a copy of the License at
https://creativecommons.org/licenses/by/4.0/legalcode.

Source code is licensed under the Apache License, Version 2.0. You may obtain a
copy of the License at https://www.apache.org/licenses/LICENSE-2.0.

This is not an officially supported Google product.

[build from source]: https://mujoco.readthedocs.io/en/latest/programming#building-from-source
[Getting Started]: https://mujoco.readthedocs.io/en/latest/programming#getting-started
[Unity]: https://unity.com/
[releases page]: https://github.com/google-deepmind/mujoco/releases
[mujoco.readthedocs.io]: https://mujoco.readthedocs.io
[changelog]: https://mujoco.readthedocs.io/en/latest/changelog.html
[Python bindings]: https://mujoco.readthedocs.io/en/stable/python.html#python-bindings
[PyPI]: https://pypi.org/project/mujoco/
