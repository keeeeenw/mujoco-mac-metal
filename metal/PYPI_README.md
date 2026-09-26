# mujoco-mac-metal

Experimental native Apple GPU physics profiles for MuJoCo 3.10.0, exposed as
`mujoco_metal` and the `mujoco-metal` diagnostic command.

Version **0.4.0** adds bounded RK4/implicitfast integration, including eligible
free-body midpoint; passive and inertia-box fluid forces; fixed-joint tendon
servos/dynamics; sphere contact and friction; scalar joint constraints; and
selected current-state sensors. Persistent GPU state supports row reset,
checkpoint replay and per-world failure handling.

These are explicitly validated model subsets, not full MuJoCo compatibility or
a transparent replacement for `mj_step`. Rendering still uses MuJoCo OpenGL.
The package uses custom Metal shaders through PyTorch MPS; it does not use JAX
or MLX. See [feature coverage and remaining limits](https://github.com/keeeeenw/mujoco-mac-metal/blob/main/metal/DEVELOPMENT.md).

On macOS arm64 with Python 3.12:

```sh
python -m pip install --upgrade mujoco-mac-metal
PYTORCH_ENABLE_MPS_FALLBACK=0 mujoco-metal doctor --gpu
```

Dependencies pin MuJoCo 3.10.0 and Torch 2.9.1 on that platform. The diagnostic
GPU smoke test covers one small force/Euler fixture, not every feature. The
physics source checkpoint passed 257 tests with native GPU execution enabled
on an M1 Max with 32 GB unified memory; broader hardware validation is pending.
No speed claim for the new profiles follows from those checks.

The wheel includes all 15 shader resources. Demo scripts, XML models and GIFs
are in the [repository gallery](https://github.com/keeeeenw/mujoco-mac-metal/blob/main/metal/examples/demo_gallery.md).
See the [installation guide](https://github.com/keeeeenw/mujoco-mac-metal/blob/main/metal/INSTALL.md)
for source setup, recording dependencies and qualification details.

This community project builds on Google DeepMind's MuJoCo and uses the
Apache-2.0 license. It is not an official MuJoCo release.
