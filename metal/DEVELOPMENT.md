# Native physics development coverage

Version **0.4.0** provides the bounded profiles below. Upgrade the installed package, and use the [source setup](INSTALL.md#development-source)
for demo scripts and assets.
The numerical reference remains MuJoCo **3.10.0**, Python **3.12**, Torch
**2.9.1**, MPS float32, with CPU fallback disabled. These are bounded feature
increments, not full MuJoCo compatibility or new performance results.

The current regression checkpoint passed **282 tests with native GPU execution
enabled**. A separate environment without Torch passed **155 CPU tests**, with
127 GPU checks skipped. These counts describe the source snapshot and do not
establish coverage of every MuJoCo feature.

The source checkpoint is `feature/problem-001-integrated-euler`; documentation-only revisions do not change
these results. Tests ran on an M1 Max with 32 GB unified memory. Numerical
qualification is separate from performance qualification: the older pendulum
benchmark has not been rerun for the additional profiles.

## Qualified increments

| Capability | Explicit simulation profile | Evidence and creative demo |
|---|---|---|
| Integrated Euler physics pipeline: unified coupled constraint solve (plane–sphere and sphere–sphere contacts, condim 1 and pyramidal condim 3, scalar limits, dry frictionloss, polynomial joint equality) with motor actuation, fixed-joint tendons, rigid passive forces, fluid drag, and live current-state sensor queries | `integrated_euler_v1` | Coupled Delassus PGS solve, multi-batch rollouts, CPU oracle parity, and snapshot replay; [robotic marble music machine](examples/marble_music_machine.md) |
| Quaternion-aware RK4 | `contact_free_rk4_v1`, force/motor/passive/sensor/transmission RK4 variants | Mixed-joint trajectories; [tumbling toys](examples/tumbling_toys.md) |
| Rigid joint springs, polynomial damping, body gravity compensation and Cartesian body forces | `contact_free_passive_euler_v1`, `contact_free_passive_rk4_v1` | Euler/RK4 trajectories, disable flags and applied-wrench checks; [spring flower](examples/kinetic_sculpture.md) |
| Stateless scalar servos, fixed/affine gain and affine bias, fixed-joint tendon transmissions plus spring/damping/armature | `contact_free_transmission_euler_v1`, `contact_free_transmission_rk4_v1` | Force stage and 400-step trajectories against CPU; [cable plotter](examples/cable_plotter.md) |
| Joint, quaternion, frame, clock, gyro and velocimeter sensor queries | `contact_free_sensor_euler_v1`, `contact_free_sensor_rk4_v1` | Current-state queries, reset/restore and CPU sensor oracles; [scanning rig](examples/scanning_rig.md) |
| Inertia-box body fluid drag, viscosity and wind | `contact_free_fluid_euler_v1`, `contact_free_fluid_rk4_v1` | Rotated articulated-body forces and 400-step trajectories; [current-driven bodies](examples/fluid_buoys.md) |
| Implicitfast with constant DOF damping and eligible free-body midpoint | `contact_free_implicitfast_v1` | Mixed ball/hinge/slide and standalone/articulated free-body trajectories, source-exact qacc and replay; [mechanical wave lattice](examples/wave_lattice.md) and [off-center balance workshop](examples/offcenter_balance.md) |
| Scalar joint limits, frictionloss and polynomial equality | `joint_constraints_euler_v1` | Coupled forces, disabled/active rows, 300-step trajectories and replay; [clockwork automaton](examples/clockwork_automaton.md) |
| Pyramidal condim3 sphere contact | `friction_contact_euler_v1` | Rotated/offset direct-force oracle, sliding/re-impact/separation trajectories; bounded block refinement for sliding-to-rolling transitions; [friction laboratory](examples/friction_laboratory.md) |
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

Implicitfast uses midpoint correction only for source-eligible standalone free
bodies. Other free trees retain ordinary implicitfast integration. Its reported
`qacc` follows MuJoCo 3.10: midpoint DOFs store the velocity difference divided
by the timestep; other DOFs keep the forward acceleration.

The `integrated_euler_v1` profile implements a model-derived execution plan (`sim.execution_plan`)
as the runtime authority for stage enablement, subsystem construction, and dependency validation.
The pipeline consists of ten sequential integration stages plus an explicit on-demand sensor query:
1. `smooth_dynamics`: Forward kinematics, body inertias, CRBA generalized mass matrix, and bias forces.
2. `passive_forces`: Joint springs, polynomial damping, gravity compensation, and Cartesian body wrenches.
3. `fluid_forces`: Inertia-box fluid drag, viscosity, and wind forces.
4. `fixed_tendons`: Fixed-joint tendon lengths, Jacobian, spring/damping forces, and tendon armature matrix.
5. `actuation`: Stateless scalar motor actuators and transmissions with `ctrlrange` clipping.
6. `smooth_assembly`: Aggregates unconstrained forces and tendon armature.
7. `unconstrained_solve`: Unconstrained acceleration solve $M \hat{a} = \tau_{\text{smooth}}$.
8. `coupled_constraints`: Unified Delassus Projected Gauss-Seidel (PGS) constraint solve simultaneously coupling plane–sphere and sphere–sphere contacts (condim 1 and pyramidal condim 3), scalar joint limits, dry joint frictionloss, and polynomial equality constraints.
9. `euler_damping`: Semi-implicit velocity damping solve $(M + h D) v^+ = M v^*$.
10. `euler_integration`: Semi-implicit Euler state integration with sticky failure rollback.

On-demand queries:
- `sensor_query`: Explicit stateless current-state forward sensor evaluations (`sim.sensor_values()`) on MPS.

The coupled solver enforces an explicit, validated convergence contract:
- Iteration settings: Bound to $[1, 2048]$ (values $\le 0$ or $> 2048$ raise `ValueError`).
- Tolerance settings: Must be finite and positive; floored at single-precision float32 hardware precision floor $10^{-6}$.
- Contract exposure: Both requested and effective iterations/tolerances, maximum refinement sweeps (64), and convergence metric (`max_normalized_projected_gradient`) are explicitly exposed on `CoupledSolverSettings` via `sim.solver_settings`.
- Convergence metric: $\max_i |\text{proj}_i - \lambda_i| \cdot D_{ii} / s_i \le \text{tolerance}$.
- Solver status codes: 0 = converged, 2 = non-finite / divergence, 3 = iteration exhaustion / non-convergence. If any world fails to converge, its state stickily rolls back to its previous valid state without advancing, while healthy worlds continue. Recovery is achieved via selective per-world reset (`sim.state.reset(env_ids, qpos=..., qvel=...)`).
- Device residency & buffer audit: Hot path physics execute on MPS with preallocated workspace buffers (`workspace_J`, `workspace_debug`, `contact_row_data`, `contact_jacobian`, `out_force`, `out_acc`, etc.) and persistent preallocated default equality state, avoiding CPU physics and host-device state synchronization. Intermediate PyTorch MPS operations (e.g. status mask selection) execute within device memory. Buffer specifications are grounded in real allocations via `sim.buffer_audit()`. On the 100-step coupled verification fixture, native Metal matches CPU MuJoCo with maximum position error `1.50e-7`, velocity error `1.67e-6`, and sensor error `1.07e-6`.
- Full assembled system qualification: Tested via `test_integrated_simulation_assembled_system_matches_cpu` across both deterministic mixed 3D fixture (nontrivial 3D rotational and tangential Jacobians with non-axial hinge axes) and axis-aligned fixture. Compares complete active $J$ (max error $< 10^{-6}$), effective $M$ with tendon armature ($< 10^{-6}$), regularizer $R$ ($< 10^{-6}$), $a_r$ ($< 10^{-3}$), $\text{rhs}$ ($< 10^{-3}$), assembled Delassus $W = J M^{-1} J^T + R$ ($< 10^{-6}$), nonzero cross-coupling blocks ($W_{cj}$ norm $0.5902$), zero inactive rows, and host KKT projected residual ($< 10^{-4}$).
- Capacity boundaries vs. admission guards: Distinguishes admission guards (CPU lowering and overflow rejection: $nv=33, nc=17, nr=97$ rejected with `ValueError`) from GPU capacity execution ($nv=32, nc=16, nr=96$ executed on GPU with status 0, isolated worlds, finite float32 outputs, slot 95 exercised via active contact pair 15, and verified against CPU MuJoCo references).
- Control clipping: Evaluates same-time physical effect of `mjDSBL_CLAMPCTRL` over identical 50 steps from identical initial states ($|q_{\text{noclamp}} - q_{\text{clamp}}| = 0.03632 > 0.02$, with status 0 and CPU parity for both).
- Test suite: **288 passed** with GPU enabled (`MUJOCO_METAL_RUN_GPU=1`), **150 passed** (135 skipped) in real no-Torch environment.

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
- Friction laboratory: 600 sustained-contact steps; maximum qpos/qvel
  differences `7.52e-6` / `4.80e-6`.
- Clockwork automaton: 1,800 steps; maximum qpos/qvel differences
  `7.23e-6` / `2.63e-5`.
- Inertia-box current drift: 1,200 steps; maximum qpos/qvel differences
  `5.43e-6` / `5.00e-7`.
- Mechanical wave lattice: 2,400 steps; maximum qpos/qvel differences
  `8.51e-6` / `2.65e-5`.
- Off-center balance workshop: 1,800 steps; maximum qpos/qvel differences
  `3.20e-5` / `2.58e-5`, with independently measured CPU/native COM and origin trails.
- Sensor scanning rig: 2,000 steps (four simulated seconds); maximum qpos/qvel
  differences `6.53e-6` / `6.64e-6`, maximum current sensor difference `7.49e-5`
  across the fixture's mixed sensor units, including accumulated float32 time.
- Robotic marble music machine: 400 steps; maximum absolute qpos/qvel differences
  `8.94e-6` / `1.97e-4`; stage sensor difference `0.0`, trajectory sensor difference `8.57e-6`; exercises all Euler feature families simultaneously in one coupled Delassus solve (211 active joint limit steps, 238 near limit steps, chime oscillations 0.091 rad / 0.066 rad, 4 active contact pairs).

These measured errors describe the fixtures, not universal tolerances.

## Still required for full coverage

Remaining work includes wider collision geometry, friction and constraint
families, warm starting and solver settings, spatial/wrapped tendon dynamics,
stateful and muscle actuators, geom-level fluid/lift models, full implicit and wider velocity derivatives, remaining sensor
families and exact stage/history semantics, mocap, flex, plugins, mutable model
and broader API behavior. Each needs its own source-derived implementation,
negative capability guards and CPU-reference numerical qualification. A small
working contact or sensor subset does not complete these categories.

RL integration and validation on other Mac hardware are deferred. No new
speedup claim follows from the feature checks here.
