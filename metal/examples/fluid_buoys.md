# Current-driven inertia-box buoys

Use the [shared source setup](../INSTALL.md#development-source), with its Python
environment active, and run these commands from the repository root.

Three colored free bodies start in a uniform current with different angular
velocities. The current and rotational damping change their measured positions
and orientations over time. This demo exercises MuJoCo 3.10's inertia-box
fluid forces from density, viscosity, and wind. It uses the behavior implemented
by MuJoCo; it does not model buoyancy, lifting surfaces, or geom-level
ellipsoid interactions.

Run a short native-versus-CPU check on Apple Silicon:

```sh
PYTHONPATH=metal PYTORCH_ENABLE_MPS_FALLBACK=0 \
  python metal/examples/fluid_buoys.py \
  --mode metal --headless --check --steps 200
```

Select `--integrator RK4` to use the RK4 profile. A CPU-only rollout does not
initialize MPS:

```sh
PYTHONPATH=metal python metal/examples/fluid_buoys.py \
  --mode cpu --headless --steps 600
```

To record side-by-side rendered rollouts, install Pillow and use a working
OpenGL context:

```sh
PYTHONPATH=metal python metal/examples/fluid_buoys.py \
  --mode metal --steps 1200 --record metal/examples/assets/fluid_buoys.gif
```

![Native inertia-box current drift beside CPU MuJoCo](assets/fluid_buoys.gif)
