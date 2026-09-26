# Native physics development coverage

This branch extends the published **0.3.0** package. Use the source checkout
with `PYTHONPATH=metal`; installing 0.3.0 does not provide these new profiles.
The numerical reference remains MuJoCo **3.10.0**, Python **3.12**, Torch
**2.9.1**, MPS float32, with CPU fallback disabled. These are bounded feature
increments, not full MuJoCo compatibility or new performance results.

## Qualified increments

| Capability | Explicit simulation profile | Evidence and creative demo |
|---|---|---|
| Quaternion-aware RK4 | `contact_free_rk4_v1`, force/motor/passive/sensor/transmission RK4 variants | Mixed-joint trajectories; [tumbling toys](examples/tumbling_toys.md) |
| Rigid joint springs, polynomial damping, body gravity compensation and Cartesian body forces | `contact_free_passive_euler_v1`, `contact_free_passive_rk4_v1` | Euler/RK4 trajectories, disable flags and applied-wrench checks; [spring flower](examples/kinetic_sculpture.md) |
| Stateless scalar servos, fixed/affine gain and affine bias, fixed-joint tendon transmissions | `contact_free_transmission_euler_v1`, `contact_free_transmission_rk4_v1` | Force stage and 400-step trajectories against CPU; [cable plotter](examples/cable_plotter.md) |
| Joint, quaternion, frame, clock, gyro and velocimeter sensor queries | `contact_free_sensor_euler_v1`, `contact_free_sensor_rk4_v1` | Current-state queries, reset/restore and CPU sensor oracles; [scanning rig](examples/scanning_rig.md) |
| Plane–sphere and sphere–sphere normal contact | `normal_contact_euler_v1` | Coupled contact forces, 500-step drop/stack and checkpoint replay; [marble cascade](examples/marble_cascade.md) |

Profiles deliberately reject unimplemented combinations. For example, the
normal-contact profile does not thereby support sensors, fixed-tendon servos,
joint limits or equality constraints. Contact capacity is currently limited to
16 candidate pairs and 32 generalized velocities; unsupported shapes and
capacities fail at construction.

`sensor_values()` performs an explicit forward query from the current state.
It does **not** reproduce MuJoCo `mj_step`'s stored pre-integration sensor timing.
Unsupported sensor types, delay/history/noise features and callbacks are
rejected. Native stepping remains separate from OpenGL visualization.

## Demo evidence

The clips are actual native/CPU simulations at a presentation playback rate,
not throughput measurements. [Explore the gallery](examples/demo_gallery.md).

- Tumbling toys: 1,000 steps, maximum absolute qpos/qvel differences
  `2.00e-6` / `7.34e-6`.
- Spring flower: 1,000 steps, maximum qpos/qvel differences
  `3.84e-6` / `1.63e-5`.
- Cable plotter: 1,200 steps, maximum qpos/qvel differences
  `1.36e-7` / `1.51e-7`.
- Marble cascade: 600 steps, 560 steps with CPU-reference contacts, peak four
  contacts; maximum qpos/qvel differences `1.67e-5` / `2.76e-5`.
- Sensor scanning rig: 2,000 steps (four simulated seconds); maximum qpos/qvel
  differences `6.53e-6` / `6.64e-6`, maximum current sensor difference `7.49e-5`
  across the fixture's mixed sensor units, including accumulated float32 time.

These measured errors describe the fixtures, not universal tolerances.

## Still required for full coverage

Remaining work includes wider collision geometry, friction and constraint
families, warm starting and solver settings, spatial/wrapped tendon dynamics,
stateful and muscle actuators, fluids, implicit integrators, remaining sensor
families and exact stage/history semantics, mocap, flex, plugins, mutable model
and broader API behavior. Each needs its own source-derived implementation,
negative capability guards and CPU-reference numerical qualification. A small
working contact or sensor subset does not complete these categories.

RL integration and validation on other Mac hardware are deferred. No new
speedup claim follows from the feature checks here.
