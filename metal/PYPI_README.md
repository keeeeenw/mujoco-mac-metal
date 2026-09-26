# mujoco-mac-metal

Experimental native Apple GPU physics profiles for MuJoCo 3.10.0, exposed as
`mujoco_metal` and the `mujoco-metal` diagnostic command. This source tree is
**0.4.0.dev0**, an unreleased development version. The published PyPI release
remains **0.3.0**, with contact-free Euler forces, damping and scalar motors.

Development profiles add bounded RK4/implicitfast integration, passive and
inertia-box fluid forces, fixed-joint tendon servos/dynamics, sphere contact,
scalar joint constraints and selected current-state sensors. This is not full
MuJoCo compatibility or a replacement for upstream `mj_step`. Visualization
still uses MuJoCo OpenGL. Detailed support boundaries and CPU-reference evidence
are in [development coverage](https://github.com/keeeeenw/mujoco-mac-metal/blob/develop/metal-physics-coverage/metal/DEVELOPMENT.md).
For unreleased profiles and demos, follow the
[source installation guide](https://github.com/keeeeenw/mujoco-mac-metal/blob/develop/metal-physics-coverage/metal/INSTALL.md#development-source).

Install the published release on macOS arm64 with Python 3.12 using
`pip install mujoco-mac-metal`. The package selects Torch 2.9.1 on that platform.
See the [installation and diagnostic guide](https://github.com/keeeeenw/mujoco-mac-metal/blob/main/metal/INSTALL.md).
