# Clockwork parcel sorter

Use the [shared source setup](../INSTALL.md#development-source), with its Python
environment active, and run these commands from the repository root.
This integrated demo requires current main source; it is not supported by the
published 0.4.0 wheel alone.

This example showcases complete primitive collision support in the integrated native Metal Euler pipeline (`integrated_euler_v1`),
combining multi-contact manifolds across planes, spheres, capsules, and boxes with motor actuation,
polynomial joint equality, fixed tendons, joint limits, frictionloss, and live sensor queries—all
solved in one coupled Delassus PGS constraint solve per step.

The sorter features:
- A solid box feed ramp where mixed parcels descend under gravity and Coulomb friction.
- A mixed parcel feed comprising real box parcels (40x40x30 mm), capsule parcels, and sphere parcels.
- An actuated swinging diverter gate driven by a rhythmic clockwork motor torque.
- A follower diverter gate synchronized symmetrically via a polynomial joint equality constraint.
- A passive spring/damper lever coupled to the diverter mechanism via a fixed tendon.
- Multi-contact primitive manifolds in the demo mechanism: box-box face/edge contact (up to 8 points), capsule-box (2 points), capsule-capsule (2 points), sphere-box (1 point), and sphere-plane (1 point). (The complete qualification test suite in `metal/tests/test_primitive_collision_qualification.py` verifies all 9 valid primitive collision pairs, including plane-capsule, plane-box, sphere-sphere, and sphere-capsule; plane-plane does not generate contacts).
- Dynamic parcel-to-parcel interactions (`box_parcel_geom <-> cap_parcel_geom`) demonstrating stacking and collision deflection.
- Actual physical routing directing the box parcel to the left chute ($Y < -0.1$), the capsule parcel to the right chute ($Y > 0.1$), and the sphere parcel along the center track ($|Y| < 0.05$).

Run a CPU reference smoke test:

```bash
PYTHONPATH=metal python metal/examples/clockwork_parcel_sorter.py --mode cpu --headless --steps 600
```

Run the native Metal comparison check on an Apple Silicon Mac:

```bash
PYTORCH_ENABLE_MPS_FALLBACK=0 PYTHONPATH=metal:metal/examples python metal/examples/clockwork_parcel_sorter.py --headless --check --steps 600
```

Record the side-by-side native Metal and CPU MuJoCo renders:

```bash
PYTORCH_ENABLE_MPS_FALLBACK=0 PYTHONPATH=metal:metal/examples python metal/examples/clockwork_parcel_sorter.py --record metal/examples/assets/clockwork_parcel_sorter.gif --steps 600
```

![Native Metal clockwork parcel sorter beside CPU MuJoCo](assets/clockwork_parcel_sorter.gif)

The 600-step native run exercises 9 active contact pairs across 5 primitive collision combinations simultaneously (plane-sphere, sphere-box, capsule-box, capsule-capsule, and box-box).
Pre-impact mixed-coordinate generalized coordinate errors against CPU MuJoCo were `3.74e-6` (`qpos`) and `1.30e-4` (`qvel`), corresponding to decoupled parcel translation error $< 3.2 \times 10^{-7}$ m and hinge angle error $< 1.83 \times 10^{-6}$ rad; stage sensor error was `0.0`, and equality residual matched CPU within `3.2e-5` rad.
Across the full 600 steps (1.2 seconds of multi-contact impacts, chaotic bouncing, and sorting), unit-aware physical errors remain tightly bounded:
- Parcel translation error: `1.64e-2` m (1.64 cm, verified bounded by $\le 0.03$ m)
- Parcel linear velocity error: `4.79e-2` m/s (verified bounded by $\le 0.10$ m/s)
- Hinge joint angle error: `2.07e-3` rad (verified bounded by $\le 0.005$ rad)
- Hinge joint velocity error: `4.00e-1` rad/s (verified bounded by $\le 0.5$ rad/s)
- Parcel rotation geodesic error: `5.44e-1` rad (verified bounded by $\le 1.0$ rad)
- Parcel angular velocity error: `1.82` rad/s (verified bounded by $\le 2.5$ rad/s)
The run recorded 536 native active contact steps (537 CPU steps), peak 7 simultaneous contacts, 443 active joint limit steps, and 9 unique contact pairs (`box_parcel_geom <-> cap_parcel_geom`, `box_parcel_geom <-> chute_left`, `box_parcel_geom <-> diverter1_flap`, `box_parcel_geom <-> ramp`, `cap_parcel_geom <-> chute_right`, `cap_parcel_geom <-> diverter2_flap`, `cap_parcel_geom <-> ramp`, `floor <-> sph_parcel_geom`, and `ramp <-> sph_parcel_geom`).
Physical routing successfully dispatched the box parcel to the left channel ($Y = -0.405$ m), the capsule parcel to the right channel ($Y = +0.402$ m), and the sphere parcel down the center ($Y = 1.38 \times 10^{-9}$ m).
