# Native Metal API and state contracts

For installation, see [INSTALL.md](INSTALL.md). For all 0.4.0 profiles and their
limits, see [DEVELOPMENT.md](DEVELOPMENT.md). The default profile described below
is intentionally narrower than the complete set of opt-in profiles.

## Inspection and computation primitives

Run `python -m mujoco_metal preflight --model path/to/model.xml --json --inventory` to inspect runtime version, model dimensions, package and all 18 bundled shader paths/hashes, capability boundaries, and the versioned feature/API inventory. The overall `gpu_qualified` field remains false because the full backend is not qualified. Inventory completeness is explicitly false because the captured enum and Python binding inventory is not exhaustive.

`load_model(xml_or_path)` returns an immutable, dimension-derived descriptor. `descriptor.forward_kinematics(qpos)` is a CPU reference for body, inertial, geom, site, and joint-anchor/axis world poses across hinge, slide, ball, and free joints. `MetalKinematics(descriptor).run(qpos_batch)` computes the same pose fields through the batched native Metal kinematics kernel. GPU checks have passed on an Apple M1 for empty and fixed worlds, mixed hinge/slide/ball/free models with off-center joints and multiple free roots, world-attached sites/geoms, and a 32-DOF chain. This is a narrow correctness qualification, not a general model-support or performance claim. Constructing `MetalKinematics` initializes MPS and compiles the bundled shader.

`smooth_dynamics(descriptor, qpos, qvel)` is the CPU reference returning a dense joint-space mass matrix and inertial/gravity bias forces. `MetalSmoothDynamics(descriptor).run(qpos_batch, qvel_batch)` accepts host NumPy state batches and returns those two outputs as MPS tensors. Native GPU checks have passed on an Apple M1 across empty/fixed, mixed-joint, rotated-inertia, massless-ancestor, disabled-gravity, and 32-DOF-chain fixtures. `MetalDenseSolve(nv, batch_size, nrhs=1).run_device(mass, rhs)` performs a dimension-derived dense Cholesky factorization and one or multiple right-hand-side solves. `MetalEulerIntegration(descriptor, batch_size, timestep).run_device(qpos, qvel, qacc, time, solve_status)` updates hinge, slide, ball, and free coordinates on MPS. Both primitives have narrow GPU correctness qualifications against synthetic systems or MuJoCo 3.10; neither alone establishes unrestricted dynamics support. These device primitives require contiguous float32 MPS tensors; their outputs are borrowed reusable views, overwritten by the next call.

For persistent GPU inputs, construct `MetalKinematics` or `MetalSmoothDynamics`
with `batch_size=B`, then call `run_device` with contiguous float32 MPS tensors.
The host-array `run` methods remain available for reference comparisons and tools.
`MetalSimulation` uses the device methods internally, without a per-step host
state transfer. Device outputs are borrowed and overwritten by later calls; copy
results when retaining them beyond the next invocation.

`ModelLifecycle` performs transactional CPU body-mass updates through MuJoCo `mj_setConst`. `BatchedConstants` maintains per-environment body masses and derived `body_invweight0` rows with atomic recomputation/restore and seeded mass randomization. `KinematicsBatchState` tracks explicit environment rows with generation-based FK cache invalidation, snapshots, restore, and tangent-space joint randomization. These are CPU lifecycle utilities and do not advance physics.

`MetalSimulation(model, batch_size=1, qpos=None, qvel=None)` connects persistent MPS state, smooth dynamics, the native dense solve, and semi-implicit Euler for the bounded `contact_free_euler_v1` profile. A local Apple M1 GPU qualification passed on four rigid-body fixtures, three initial states, 1,000-step 1 ms rollouts, and reset/restore resume. This remains a narrow qualification, not general model support. The model must explicitly disable contacts and use Euler integration. The profile supports gravity and joint armature, assumes zero applied generalized force, and rejects actuators, tendons, limits, friction loss, passive/fluid forces, sensors, equality constraints, flexes, plugins, damping, springs, callbacks, mocap, and non-Euler integrators. A model may be accepted for kinematics or mass/bias queries and still be rejected for stepping. Construction validates the profile before initializing MPS.

`simulation.step()` returns a borrowed MPS status vector. Solver and integration failures are per-world; a failed world keeps its state and its first nonzero status remains sticky until reset or restore. State views are copies; snapshots and resets cross the host/device boundary and belong outside the hot step loop. Position, velocity, acceleration, and simulation time use float32 on device. In particular, time's representable increment gets coarser as elapsed time grows.

Run the opt-in GPU correctness tests only on an available Apple GPU with the pinned Torch extra installed: `MUJOCO_METAL_RUN_GPU=1 PYTORCH_ENABLE_MPS_FALLBACK=0 python -m pytest -m gpu`. Ordinary `python -m pytest` runs CPU tests and skips the GPU cases. `preflight` remains CPU-only: shader hashes and stage labels are inventory, not a device probe. The overall GPU-qualified field stays false because full-library physics coverage remains incomplete. The standalone source tree carries the Apache 2.0 license and notices.

## Applied forces, damping and basic motors

Select a profile explicitly; the original `contact_free_euler_v1` remains the
unforced baseline. These three Euler profiles require contacts disabled. Version 0.4.0 also
provides additional profiles listed in [development coverage](DEVELOPMENT.md).

| Profile | Additional supported behavior |
| --- | --- |
| `contact_free_forces_euler_v1` | Per-call generalized forces and nonnegative linear joint damping. |
| `contact_free_motor_euler_v1` | The force profile plus stateless, fixed-gain, no-bias motors transmitted to hinge/slide joints. |

```python
simulation = MetalSimulation(model, batch_size=B,
                             profile="contact_free_motor_euler_v1")
simulation.step(ctrl=controls, qfrc_applied=forces)
```

Controls have shape `[B, nu]`; generalized forces have shape `[B, nv]`. Supply
finite host arrays or contiguous float32 MPS tensors. Inputs are held across
`step(steps=N, ...)` and expire at the next call; omitted values mean zero.
Inputs are copied and never modified. Host input validation/upload occurs before
the device loop; device inputs avoid host readback. Active invalid device inputs
propagate to per-world failure handling. Reset/restore clears sticky failures.

Damping follows MuJoCo's implicit Euler effective-mass solve, including the
`DAMPER` and `EULERDAMP` disable flags. Stored `qacc` remains physical acceleration;
Euler uses its separate effective acceleration to advance state. Motors support
control/actuator force clipping, summed moments and global/group disable. Other
transmissions, activation dynamics, non-fixed gains/biases, nonzero actuator
armature/damping and joint-level actuator-force limits are rejected.

Independent M1 checks covered 36 force/damping and 15 motor trajectories of
1,000 steps against CPU MuJoCo. Maximum observed qpos/qvel/physical-qacc errors
were approximately 4.83e-6 / 8.75e-6 / 2.54e-4 for force/damping, and
2.48e-6 / 5.15e-6 / 1.69e-5 for motors. These are fixture-specific results, not
universal error bounds. Public regressions cover malformed inputs, row failures,
reset/restore, changing inputs and the device-only stepping path.

Smooth M/bias queries can now accept models with actuators only if their actuator
armature is zero; they still do not compute actuator forces. Simulation validates the selected profile and actuator family; the published
motor profile remains limited to fixed-gain scalar motors. This distinction prevents unsupported inertia
terms from silently disappearing.

## Integrated Euler physics pipeline

The `integrated_euler_v1` stepping profile composes the native Metal Euler physics features into one unified pipeline:
- **Actuation:** Stateless scalar motors, servos, and affine gain/bias transmissions.
- **Fixed-joint tendons:** Multi-joint fixed wrap tendons with stiffness, damping, and armature.
- **Passive forces:** Rigid joint springs, linear and polynomial joint damping, body gravity compensation, and applied body wrenches (`xfrc_applied`).
- **Fluid forces:** Inertia-box body fluid drag, fluid viscosity, and wind.
- **Sensors:** Stateless current-state sensor queries (`sim.sensor_values()`) evaluated on GPU.
- **Unified coupled constraint solve:** Combines complete primitive collision manifolds across all 9 valid pairs among planes, spheres, capsules, and boxes, scalar joint limits, dry frictionloss, and joint/connect/weld equality constraints in one coupled Delassus system $W = J M^{-1} J^T + R$. Joint equality reserves 1 row, connect reserves 3 rows, and weld reserves 6 rows per equality, in arbitrary mixed order with deterministic row mapping (`eq_rowadr`/`eq_rownum`, total equality rows `n_eq_rows`). Body-body, body-world (world body 0) and site-site forms are supported for connect and weld, including transformed local anchors/sites, nontrivial weld relative poses and `torquescale` (zero torque scale reserves 6 rows with zero rotational rows). In the current source checkout, integrated Euler lowers condim 1/3/4/6 for pyramidal and elliptic cones, with anisotropic sliding, torsional and rolling friction. Pyramidal cone rows are expanded by dimension; elliptic contact blocks are solved together with their normal row. Both use contact-frame rotational Jacobians and reconstruct physical contact force/torque. This independently qualified subset is available from source on main; it is not in the released 0.4.0 wheel.
- Contact friction lowering expands MuJoCo geom friction `[slide, torsion, rolling]` to per-axis `[slide1, slide2, torsion, roll1, roll2]`; `<pair>` overrides retain their five coefficients. Standard geom mixing and pair priority are applied before lowering. `solreffriction` is honored for elliptic cones; MuJoCo defines it as ineffective for pyramidal cones. Unsupported dimensions and capacity overflow fail during construction.
- The separate MuJoCo no-slip post-solver is not implemented. `integrated_euler_v1` rejects models with nonzero `noslip_iterations`; keep it at its default zero. The MuJoCo `CG` solver selection is rejected; the native profile selects its own projected algorithm (pyramidal PGS or elliptic global FISTA), independent of the accepted default Newton or PGS selector. MuJoCo `iterations` and `tolerance` map to the native outer iteration budget and projected residual threshold, with the documented float32 minimum tolerance.
- **Full primitive collision support:** Real multi-contact manifolds generated entirely on GPU via Separating Axis Theorem (SAT 15 candidate axes), Sutherland-Hodgman polygon clipping, and closest segment points. Box-box produces up to 8 contact points, plane-box up to 4, capsule-box / capsule-capsule / plane-capsule up to 2, and sphere pairs 1 point. Both $(A, B)$ and $(B, A)$ geometry orderings, explicit `<pair>`, `<exclude>`, and parent/welded body filtering are fully respected.

Solving contact and joint constraints in one coupled Delassus system prevents the numerical divergence (exceeding 15% error) that occurs when constraints are solved sequentially or decoupled.

```python
sim = MetalSimulation(model, batch_size=B, profile="integrated_euler_v1")
sim.step(steps=N, ctrl=controls, qfrc_applied=forces, xfrc_applied=wrenches)
sensor_data = sim.sensor_values()
```

### Capacity limits and guards
The integrated Euler pipeline enforces explicit hardware-tailored capacity bounds:
- Generalized velocities: $nv \le 32$.
- Candidate collision pairs: $npairs \le 16$.
- Candidate contact point slots: $ncontacts \le 24$.
- Total candidate constraint rows: $nr \le 96$.
- Equality row spans: joint equality 1 row, connect 3 rows, weld 6 rows per equality (`n_eq_rows` total, deterministic `eq_rowadr`/`eq_rownum` mapping). The activity array keeps one entry per equality, not per row.
- Contact row counts depend on cone and `condim`: pyramidal condim 1/3/4/6 expands to 1/4/6/10 rows; elliptic condim 1/3/4/6 uses 1/3/4/6 coupled rows. Lowering calculates row offsets and rejects models that exceed the total row cap before GPU execution.
- Supported primitive geometries: plane, sphere, capsule, box. Models exceeding these bounds or requesting non-primitive geometries (meshes, cylinders, ellipsoids, heightfields, SDFs), non-Euler integrators, flex, or plugins are rejected cleanly during profile validation before GPU execution.
- Supported equality types: joint, connect, weld (body-body, body-world, site-site, including mocap bodies as kinematic anchors; both-mocap equalities reserve rows but assemble zero rows/forces, matching MuJoCo skipping its empty Jacobian). Tendon/flex equalities, ball limits and non-finite `torquescale` are rejected, even when initially inactive.
- Mocap bodies: jointless direct children of world only (pinned compiler restriction, re-validated at lowering). Prescribed per-environment poses via `sim.set_mocap`.

### Equality activation, reset, snapshots and failure

```python
sim.set_equality_active(values, env_ids=None)  # persistent per-env, per-equality booleans
sim.set_mocap(pos, quat=None, env_ids=None)  # prescribed per-env mocap poses
sim.reset(env_ids=None, qpos=None, qvel=None, eq_active=None, mocap_pos=None, mocap_quat=None)
sim.reset_to_keyframe(key_id, env_ids=None)  # pinned mj_resetDataKeyframe semantics
sim.copy_environment(src, dst)
snapshot = sim.state.snapshot()  # schema v3 carries eq_active + mocap when present
sim.state.restore(snapshot)
```

- `set_equality_active` accepts host boolean/integer arrays of shape `(neq,)` (broadcast to the selected worlds) or `(len(env_ids), neq)`, or contiguous int32/bool MPS tensors of the same shapes. Device tensors are copied, never borrowed or modified. `env_ids=None` selects all worlds. Validation is atomic: malformed input leaves every world unchanged. Changes take effect on the next step or assembly, invalidate cached assembly/diagnostics, and never clear sticky per-world failure status. Releasing an equality clears its stale rows, multipliers and generalized force on recomputation; reattaching reuses the compiled reference anchors/poses.
- `reset` with `eq_active=None` restores compiled `eq_active0` defaults for the selected worlds; an explicit `eq_active` array overrides them. `mocap_pos`/`mocap_quat=None` restore compiled reference frames; explicit arrays override. Unselected worlds are untouched. Reset clears sticky failure status for the selected worlds only. `sim.reset` additionally clears held per-call inputs (`ctrl`, `qfrc_applied`, `xfrc_applied`) for the selected worlds and invalidates cached assembly.
- `set_mocap` accepts host arrays or contiguous float32 MPS tensors: `(nmocap, 3/4)` broadcast to the selected worlds or `(len(env_ids), nmocap, 3/4)` per world, or a single `(…, nmocap, 7)` posquat array with `quat=None`. Inputs are copied, validation is atomic, sticky failure is preserved, changes take effect on the next step/assembly.
- `reset_to_keyframe` mirrors pinned `mj_resetDataKeyframe`: time to `key_time`, qpos/qvel/mocap to the key values, qacc/status cleared, equality activity to compiled defaults, held controls to `key_ctrl` for the selected worlds. Invalid key ids fail atomically.
- `copy_environment` copies every state row (qpos/qvel/qacc/time/status/eq/mocap) src to dst and bumps generation.
- Snapshots use schema version 3 when the model has mocap bodies (`nmocap`, `mpos`, `mquat` included), version 2 when it has equalities but no mocap, and version 1 otherwise. Restoring a version-1 snapshot into a model with equalities, or a version <3 snapshot into a model with mocap bodies, is rejected rather than silently dropping activity/poses; model/profile/dimension mismatches are rejected; restores are atomic.
- Model changes are shared by the batch, not per environment: `ModelLifecycle.recompute_body_masses` (via `mj_setConst`) and `update_geom_contact` (friction/solref/solimp/margin/gap) are the supported runtime-mutable paths with transactional validation; reference poses/shapes/types are recompile-required (build a new lifecycle/simulation). Actuator activation state, warmstarts, history, userdata and plugins have no native storage and are rejected at admission; controls and applied forces are per-call held inputs, not persistent state.
- A failed physics step rolls back physical state/time for that world and stays sticky until reset/restore; an accepted activity change is persistent input state, not a partially applied step.

Try the [magnetic crane demo](examples/magnetic_crane.md), the [latch-and-release cargo bridge demo](examples/cargo_bridge.md), the [clockwork parcel sorter demo](examples/clockwork_parcel_sorter.md), the [robotic marble music machine demo](examples/marble_music_machine.md) or the [spacecraft force-control demo](examples/space_docking.md), or use the
[installation and diagnostic guide](INSTALL.md) for `pip install mujoco-mac-metal`
and `mujoco-metal doctor --gpu`. Install version 0.4.0 for the additional bounded profiles.
