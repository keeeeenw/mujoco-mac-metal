# Native physics development coverage

The released **0.4.0** package provides the bounded profiles below. The in-progress
`feature/problem-003-contact-friction` branch additionally develops condim-1/3/4/6
contact friction; this unreleased change requires review before it ships. Use the
[source setup](INSTALL.md#development-source) for branch demos and assets.
The numerical reference remains MuJoCo **3.10.0**, Python **3.12**, Torch
**2.9.1**, MPS float32, with CPU fallback disabled. These are bounded feature
increments, not full MuJoCo compatibility or new performance results.

The full source suite passed **402 tests with native MPS execution enabled** and
fallback disabled. An isolated environment with Torch removed passed **175 tests**
and skipped **224 GPU-dependent tests**. These results describe this source
snapshot and do not establish coverage of every MuJoCo feature.

This branch is based on accepted source checkpoint `370606776`; Problem 003 changes
are isolated on `feature/problem-003-contact-friction`. Tests ran on an M1 Max with 32 GB unified memory. Numerical
qualification is separate from performance qualification: the older pendulum
benchmark has not been rerun for the additional profiles.

## Qualified increments

| Capability | Explicit simulation profile | Evidence and creative demo |
|---|---|---|
| Integrated Euler physics pipeline: complete primitive collision support (all 9 valid pairs among planes, spheres, capsules, boxes, multi-contact manifolds up to 8 points per pair), condim 1/3/4/6 with pyramidal and elliptic cones, anisotropic sliding/torsional/rolling friction, plus motor actuation, fixed-joint tendons, rigid passive forces, fluid drag, joint limits/frictionloss, polynomial joint equality, and live sensor queries | `integrated_euler_v1` | Independent CPU contact force/torque and trajectory comparisons, projected cone residual checks, final admitted row layouts and snapshot replay; [Spin-and-Grip arcade](examples/spin_and_grip.md), [clockwork parcel sorter](examples/clockwork_parcel_sorter.md), and [robotic marble music machine](examples/marble_music_machine.md) |
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
8. `coupled_constraints`: Unified Delassus projected solve simultaneously coupling complete primitive collision manifolds across planes, spheres, capsules, and boxes (condim 1/3/4/6 with pyramidal or elliptic cones), scalar joint limits, dry frictionloss, and polynomial equality constraints. Pyramidal contacts allocate dimension-dependent edge rows; elliptic contacts solve normal, slide, torsion and rolling components in a coupled block.
9. `euler_damping`: Semi-implicit velocity damping solve $(M + h D) v^+ = M v^*$.
10. `euler_integration`: Semi-implicit Euler state integration with sticky failure rollback.

On-demand queries:
- `sensor_query`: Explicit stateless current-state forward sensor evaluations (`sim.sensor_values()`) on MPS.

The coupled solver enforces an explicit, validated convergence contract:
- Iteration settings: Bound to $[1, 2048]$ (values $\le 0$ or $> 2048$ raise `ValueError`).
- Tolerance settings: Must be finite and positive; floored at single-precision float32 hardware precision floor $10^{-6}$.
- Contract exposure: Both requested and effective iterations/tolerances, maximum refinement sweeps (256), and convergence metric (`max_normalized_projected_gradient`) are explicitly exposed on `CoupledSolverSettings` via `sim.solver_settings`.
- Convergence metric: $\max_i |\text{proj}_i - \lambda_i| \cdot D_{ii} / s_i \le \text{tolerance}$.
- Solver status codes: 0 = converged, 2 = non-finite / divergence, 3 = iteration exhaustion / non-convergence. If any world fails to converge, its state stickily rolls back to its previous valid state without advancing, while healthy worlds continue. Recovery is achieved via selective per-world reset (`sim.state.reset(env_ids, qpos=..., qvel=...)`).
- Device residency & buffer audit: Hot path physics execute on MPS with preallocated workspace buffers (`workspace_J`, `workspace_debug`, `contact_row_data`, `contact_jacobian`, `out_force`, `out_acc`, etc.) and persistent preallocated default equality state, avoiding CPU physics and host-device state synchronization. Intermediate PyTorch MPS operations (e.g. status mask selection) execute within device memory. Buffer specifications are grounded in real allocations via `sim.buffer_audit()`. On the 100-step coupled verification fixture, native Metal matches CPU MuJoCo with maximum position error `1.50e-7`, velocity error `1.67e-6`, and sensor error `1.07e-6`.
- Full assembled system qualification: Tested via `test_integrated_simulation_assembled_system_matches_cpu` across both deterministic mixed 3D fixture (nontrivial 3D rotational and tangential Jacobians with non-axial hinge axes) and axis-aligned fixture. Compares complete active $J$ (max error $< 10^{-6}$), effective $M$ with tendon armature ($< 10^{-6}$), regularizer $R$ ($< 10^{-6}$), $a_r$ ($< 10^{-3}$), $\text{rhs}$ ($< 10^{-3}$), assembled Delassus $W = J M^{-1} J^T + R$ ($< 10^{-6}$), nonzero cross-coupling blocks ($W_{cj}$ norm $0.5902$), zero inactive rows, and host KKT projected residual ($< 10^{-4}$).
- Capacity boundaries vs. admission guards: Distinguishes admission guards (CPU lowering and overflow rejection: $nv=33, npairs=17, ncontacts=25, nr=97$ rejected with `ValueError`) from GPU capacity execution ($nv=32, npairs=16, ncontacts=24, nr=96$ executed on GPU with status 0, isolated worlds, finite float32 outputs, slot 15 exercised for pairs, slot 23 for contacts, row 95 for constraint rows, and verified against CPU MuJoCo references).
- Control clipping: Evaluates same-time physical effect of `mjDSBL_CLAMPCTRL` over identical 50 steps from identical initial states ($|q_{\text{noclamp}} - q_{\text{clamp}}| = 0.03632 > 0.02$, with status 0 and CPU parity for both).

## Problem 003 contact-friction qualification

This is an unreleased development-branch result, run on an **M1 Max with 32 GB
unified memory**, MuJoCo 3.10.0, Python 3.12 and Torch 2.9.1. `PYTORCH_ENABLE_MPS_FALLBACK=0`
was set for native checks. The eight cone/condim cases use a transformed plane
and sphere, nonzero sliding/angular velocities, asserted contact engagement,
an independently reconstructed CPU contact Jacobian, physical wrench comparison
and host-computed projected cone residual.

| Cone | condim 1 max absolute qacc error | condim 3 | condim 4 | condim 6 |
|---|---:|---:|---:|---:|
| Pyramidal | `5.02e-6` | `5.86e-5` | `1.05e-4` | `1.05e-4` |
| Elliptic | `5.02e-6` | `4.68e-5` | `2.53e-3` (`6.19e-6` relative) | `4.78e-3` (`1.14e-5` relative) |

Across that matrix, the largest absolute generalized constraint-force error was
`0.0296` for the high-impulse elliptic condim-6 case. The projected KKT/cone
residual assertion is `2e-5`; CPU contact force/torque checks use scaled
float32 tolerances. Capsule-plane and box-plane condim-4/6 fixtures pass under
both cones. Together with the earlier condim-1/3 primitive-family suite, this
covers the accepted plane/sphere/capsule/box families without claiming every
shape-pair/cone/dimension cross-product was tested.

R2 completion adds the missing interacting-friction checks. Each accepted
non-plane primitive pair now has a condim-4/6 representative; six native cases
compare both moving bodies' rotational Jacobians, physical contact wrenches,
mapped contact rows, full Delassus operators and CPU accelerations. All six W
comparisons enforce the same `rtol=5e-3, atol=3e-3`. The largest observed
box-box W difference was `2.60e-3` (relative maximum `3.67e-3`); the other pairs
had smaller observed differences, but do not use tighter assertion bounds.

| Original acceptance item | Executable qualification | Bound or observed result |
|---|---|---|
| Friction directions participate with articulated scalar constraints | `test_high_dimensional_friction_couples_with_articulated_constraints` | 4 cone/condim cases; rotational Jacobians, torsion/roll moments, active equality/limit/frictionloss and nonzero cross blocks; host projected residual `<=2e-5`; `qacc` `rtol=1e-3, atol=2e-2` |
| Failure rollback is isolated beside active and empty peers | `test_friction_failure_isolated_from_active_and_empty_worlds` | 4 cone/condim cases; statuses `[3,0,0]`; failed `qpos/qvel/time` exact; healthy contact and empty worlds match CPU; reset clears stale force and exact replay succeeds |
| Non-plane high-dimensional primitive friction | `test_high_dimensional_nonplane_pairs_compare_both_moving_bodies` | 6 pair families; both bodies' rotational Jacobians nonzero and CPU-matched; physical moments asserted; box-box max `J` difference `5.84e-6`, max `W` difference `2.60e-3` |
| Dynamic spin/roll/slip and contact events | `test_high_dimensional_spin_slip_separation_and_reimpact` | 16 cone/condim/timestep/spin-sign cases; `dt=1,4 ms`; all show contact → separation → re-impact; max bounds `qpos<2e-5`, `qvel<2e-4`, `qacc<3e-2`, constraint-force `<8e-2` |

These cases complement the earlier eight-case cone/dimension matrix, transformed
plane contact matrix, capacity boundaries, anisotropic and near-zero coefficient
checks, and Spin-and-Grip demo. They close the specific R2 qualification gaps but
do not form an exhaustive Cartesian product of every contact and model option.

The final admitted row-layout checks run ten elliptic condim-6 contacts (60
contact rows plus 30 reserved joint rows, 90 total) and seven pyramidal
condim-6 contacts (70 contact rows plus 21 joint rows, 91 total), both compared
with CPU references. A further contact over the row budget is rejected during
lowering.

The articulated friction matrix covers both cones and condim 4/6. It engages
rotational contact friction with a scalar equality coupling hinges `j1` and
`j2`, an active limit on `j1` only, one frictionloss row on `j1`, and six
time-varying controls; mapped rows, cross-coupling, physical wrenches and the
short native trajectory are checked against CPU MuJoCo. Failure isolation is
tested in a three-world batch: a failing condim-4/6 contact, a successful
condim-1 normal-contact peer, and a contact-free peer. Statuses are `[3,0,0]`;
the failed world's state and time roll back exactly while both peers advance
against CPU references. A targeted reset, cleared diagnostics and replay are
also checked. The successful peer validates active-contact isolation, not
high-dimensional moments.

A review-discovered four-contact elliptic box case now checks projected
stationarity at the retained global multipliers after every coupled iteration.
This replaces the previous convergence check, which re-optimized temporary
contact blocks and could report success for a different point. Elliptic systems
now use globally coupled cone-projected FISTA with an adaptive restart; the
pyramidal PGS update path is preserved. On the saved four-contact box fixture,
elliptic condim 3/4/6 converges in 31/118/136 iterations, respectively;
host-projected residual maxima across the four contacts are `2.48e-6`,
`3.44e-6`, and `3.86e-6`, with maximum absolute `qacc` differences of
`1.27e-3`, `2.90e-3`, and `2.90e-3`. These are bounded float32 fixture results.
No speedup has been measured for this solver change.

The separate MuJoCo no-slip post-solver remains unsupported. Models with
nonzero `noslip_iterations` are rejected during lowering; `noslip_tolerance`
has no effect while the no-slip solver is disabled. `solver="CG"` is rejected;
the native profile chooses its own projected algorithms (pyramidal PGS or
elliptic global FISTA), regardless of the accepted default Newton or PGS CPU
selector. `iterations` and `tolerance` feed the native coupled solver, and the
effective float32 tolerance floor is exposed through `CoupledSolverSettings`.

The [Spin-and-Grip guide](examples/spin_and_grip.md) records a 400-step matched
native/CPU run: max qpos difference `4.75e-6`, max qvel difference `3.37e-4`,
rolling travel `0.2040 m` versus `1.0507 m` in the low-friction CPU case, and
torsional spin `2.057 rad/s` versus `5.774 rad/s`. The press held with zero
measured drift and the released block moved `0.0972 m`. These are demo-specific
measurements, not general tolerance guarantees. No performance comparison was
run for this feature. The four-contact convergence correction required 31 to
136 outer iterations for elliptic cases in the bounded fixture; this is
convergence evidence, not a performance claim.

**Release status:** these friction extensions are not present in the 0.4.0
PyPI wheel. They remain bounded to `integrated_euler_v1`, primitive contacts,
condim 1/3/4/6, pyramidal/elliptic cones, scalar joint constraints and the
documented workspace caps. This does not complete full MuJoCo contact or solver
support.

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
- Clockwork parcel sorter: 600 steps; pre-impact mixed-coordinate generalized coordinate errors against CPU MuJoCo `3.74e-6` (`qpos`) and `1.30e-4` (`qvel`), with decoupled parcel translation error $< 3.2 \times 10^{-7}$ m and hinge angle error $< 1.83 \times 10^{-6}$ rad; stage sensor difference `0.0`, equality residual match within `3.2e-5` rad; full-run physical errors bounded within 0.0164 m translation, 0.048 m/s linear velocity, 0.00207 rad hinge angle, 0.400 rad/s hinge velocity, 0.544 rad rotation, and 1.82 rad/s angular velocity; exercises 9 active contact pairs across 5 collision combinations (plane-sphere, sphere-box, capsule-box, capsule-capsule, box-box; all 9 canonical pairs verified in the qualification suite) with 536 native active contact steps, peak 7 simultaneous contacts, 443 active joint limit steps; demonstrates actual physical routing sending box parcels to the left chute ($Y = -0.405$ m), capsule parcels to the right chute ($Y = +0.402$ m), and sphere parcels down the center ($Y = 0.000$ m).
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
