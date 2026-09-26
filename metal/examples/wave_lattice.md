# Mechanical wave lattice

A fixed-gain motor drives the first colored link of an articulated hinge/ball
lattice. Joint springs and damping carry the traveling motion through the
coupled mechanism. The native `contact_free_implicitfast_v1` profile forms
MuJoCo's damping-modified mass matrix and solves it natively; the second render
panel is an independent CPU `mj_step` reference. The frame caption reports the
measured lattice excursion and current applied torque.

This bounded profile supports the non-free rigid-body chain used here. Free-body
midpoint integration and contact/constraint dynamics remain outside this demo;
all geom contacts are disabled explicitly.

CPU smoke run:

```bash
PYTHONPATH=metal python metal/examples/wave_lattice.py --mode cpu --headless --steps 1600
```

Native short comparison on an MPS Mac:

```bash
MUJOCO_METAL_RUN_GPU=1 PYTHONPATH=metal python metal/examples/wave_lattice.py --headless --check --steps 200
```

Record the actual native/CPU renders:

```bash
PYTHONPATH=metal python metal/examples/wave_lattice.py --record metal/examples/assets/wave_lattice.gif --steps 2400
```
