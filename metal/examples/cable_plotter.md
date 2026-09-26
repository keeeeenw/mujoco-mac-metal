# Crossed-cable calligraphy plotter

The mini gantry has two orthogonal sliders and two fixed-joint tendons. One
tendon measures `x + y`, the other `x - y`; two stateless position servos
drive those lengths along a smooth figure-eight path. The MuJoCo model has
contacts disabled and no actuator state.

Run the native comparison on Apple Silicon with the pinned MuJoCo 3.10.0 and
Python 3.12 environment:

```sh
PYTHONPATH=metal PYTORCH_ENABLE_MPS_FALLBACK=0 \
  .venv-demo/bin/python metal/examples/cable_plotter.py \
  --mode metal --headless --check --steps 200
```

The check compares each native state with an independent CPU `mj_step`
reference. A CPU-only run checks the model without initializing MPS:

```sh
PYTHONPATH=metal .venv-demo/bin/python \
  metal/examples/cable_plotter.py --mode cpu --headless --steps 200
```

To record the actual two rendered simulations side by side, install Pillow and
run with a working OpenGL context:

```sh
PYTHONPATH=metal .venv-demo/bin/python metal/examples/cable_plotter.py \
  --mode metal --steps 1200 --record metal/examples/assets/cable_plotter.gif
```

The native demo composes the smooth dynamics, stateless transmission force,
dense solve, and semi-implicit Euler primitives directly. It demonstrates
fixed-joint tendon transmission and affine-bias servo forces; it does not
establish support for general actuator models, tendons as passive elements, or
contact physics.
