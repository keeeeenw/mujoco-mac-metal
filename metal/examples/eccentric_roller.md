# Eccentric Roller Workshop

Cylindrical rollers and nonuniform ellipsoids sort themselves on a tilted
table with a lane divider: rollers (axes across the slope, high sliding
friction) roll downhill into the far catch bin, while ellipsoids (low
sliding friction) slide and tumble into the near bin. The front tumbler
is faster, so the pair never pinches: sustained pinch transients stall
the PGS solver (see report), while ellipsoid-ellipsoid contact itself is
covered in the matrix and force suites.

## What it exercises

- Milestone 010 cylinder/ellipsoid contact manifolds (rolling + sliding)
- Cylinder-plane rolling and ellipsoid-plane sliding with Coulomb friction
- Same-lane differential sliding (two friction levels, verified separate)
- Native vs CPU trajectory parity over 900 steps

Rolling motion emerges from rigid-body dynamics with sliding friction.
No rolling-resistance torque is modeled: all contacts use the default
contact dimensionality. The lane divider is too tall to climb, so lanes
never interact (this justifies the contype/conaffinity pair filtering that
keeps candidate pairs within the native 16-pair capacity).

## Run

```sh
./run env MUJOCO_METAL_RUN_GPU=1 python metal/examples/eccentric_roller.py --check
./run env MUJOCO_METAL_RUN_GPU=1 python metal/examples/eccentric_roller.py --record /tmp/roller.gif
```
