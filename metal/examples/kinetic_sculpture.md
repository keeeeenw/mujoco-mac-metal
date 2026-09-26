# Kinetic sculpture

Use the [shared source setup](../INSTALL.md#development-source), with its Python
environment active, and run these commands from the repository root.

Four spring-driven petals fold around a fixed stem. Joint stiffness and
damping set the motion, while body gravity compensation carries most of each
petal's weight. A gentle alternating force and torque act at two petal centers
to add a breeze. The model disables contacts and uses Euler integration.

Run a native MPS comparison on Apple Silicon with the pinned MuJoCo 3.10.0 and
Python 3.12 environment:

```sh
PYTHONPATH=metal PYTORCH_ENABLE_MPS_FALLBACK=0 \
  python metal/examples/kinetic_sculpture.py \
  --mode metal --headless --check --steps 200
```

The script compares every native state against a separate CPU `mj_step`
reference. It also has a CPU-only run for checking the model and input stream
without initializing MPS:

```sh
PYTHONPATH=metal python \
  metal/examples/kinetic_sculpture.py --mode cpu --headless --steps 200
```

This example exercises the `contact_free_passive_euler_v1` profile with four
hinge springs, linear dampers, body gravity compensation, and per-step body
wrenches. It is a targeted demonstration of those features; the overall
backend remains experimental and does not support general contacts or
constraints.

## Recorded native simulation

![Spring flower: native Metal and CPU reference](assets/kinetic_sculpture.gif)

```sh
PYTHONPATH=metal PYTORCH_ENABLE_MPS_FALLBACK=0 python metal/examples/kinetic_sculpture.py --mode metal --headless --steps 1000 --record metal/examples/assets/kinetic_sculpture.gif
```

Recording uses Pillow and MuJoCo OpenGL. The 1,000-step M1 Max native run
observed maximum qpos error 3.84e-6 and qvel error 1.63e-5 against CPU MuJoCo.
The animation shows simulation time, not computation speed.
