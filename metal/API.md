# Native Metal API and state contracts

These APIs include experimental development paths. Consult [status](STATUS.md)
and [qualification](QUALIFICATION.md) before treating an admitted path as qualified.

For installation, see [INSTALL.md](INSTALL.md). For all 0.4.0 profiles and their
limits, see [DEVELOPMENT.md](DEVELOPMENT.md). The default profile described below
is intentionally narrower than the complete set of opt-in profiles.

## Inspection and computation primitives

Run `python -m mujoco_metal preflight --model path/to/model.xml --json --inventory` to inspect runtime version, model dimensions, package and every bundled shader's path/hash, capability boundaries, and the versioned feature/API inventory. The overall `gpu_qualified` field remains false because the full backend is not qualified. Inventory completeness is explicitly false because the captured enum and Python binding inventory is not exhaustive.

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

Run the opt-in GPU correctness tests only on an available Apple GPU with the pinned Torch extra installed: `MUJOCO_METAL_RUN_GPU=1 PYTORCH_ENABLE_MPS_FALLBACK=0 python -m pytest -m gpu`. CPU reference tests may require Torch on the host; disabling the opt-in flag alone is not a universal GPU exclusion for every historical test. Use an explicitly CPU-only test environment or availability guard for CPU-only qualification. `preflight` remains CPU-only: shader hashes and stage labels are inventory, not a device probe. The overall GPU-qualified field stays false because full-library physics coverage remains incomplete. The standalone source tree carries the Apache 2.0 license and notices.

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
- **Fluid forces:** Inertia-box body fluid drag plus per-geom ellipsoid fluid (added-mass, Magnus/Kutta lift, blunt/slender/angular viscous terms) with pinned body/geom selection, fluid viscosity, and wind.
- **Sensors:** Stateless current-state sensor queries (`sim.sensor_values()`) evaluated on GPU.
- **Unified coupled constraint solve:** Combines complete primitive collision manifolds across all 9 valid pairs among planes, spheres, capsules, and boxes, scalar joint limits, dry frictionloss, and joint/connect/weld equality constraints in one coupled Delassus system $W = J M^{-1} J^T + R$. Joint equality reserves 1 row, connect reserves 3 rows, and weld reserves 6 rows per equality, in arbitrary mixed order with deterministic row mapping (`eq_rowadr`/`eq_rownum`, total equality rows `n_eq_rows`). Body-body, body-world (world body 0) and site-site forms are supported for connect and weld, including transformed local anchors/sites, nontrivial weld relative poses and `torquescale` (zero torque scale reserves 6 rows with zero rotational rows). In the current source checkout, integrated Euler lowers condim 1/3/4/6 for pyramidal and elliptic cones, with anisotropic sliding, torsional and rolling friction. Pyramidal cone rows are expanded by dimension; elliptic contact blocks are solved together with their normal row. Both use contact-frame rotational Jacobians and reconstruct physical contact force/torque. This independently qualified subset is available from source on main; it is not in the released 0.4.0 wheel.
- Contact friction lowering expands MuJoCo geom friction `[slide, torsion, rolling]` to per-axis `[slide1, slide2, torsion, roll1, roll2]`; `<pair>` overrides retain their five coefficients. Standard geom mixing and pair priority are applied before lowering. `solreffriction` is honored for elliptic cones; MuJoCo defines it as ineffective for pyramidal cones. Unsupported dimensions and capacity overflow fail during construction.
- Solver development source contains distinct PGS, CG and Newton paths and a separate no-slip stage. Their complete option/contact/integrator composition is still under qualification. This is not a promise that selecting a solver merely aliases another algorithm, or that contact refinement substitutes for the pinned no-slip stage. `solver_diagnostics[0]` reports the final retained-system projected dual residual after no-slip, while `[1]` reports executed iterations. `solver_history` contains coarse samples of the optimizer's stopping metric, which can differ from the final dual certificate. `assembled_system()` exposes retained row data for independent checks. Warm-start state and rollback are owned by the simulation. The component-block sparse route currently admits PGS; sparse CG/Newton remain guarded pending the complete native operator and qualification.
- **Full primitive collision support:** Real multi-contact manifolds generated entirely on GPU via Separating Axis Theorem (SAT 15 candidate axes), Sutherland-Hodgman polygon clipping, and closest segment points. Box-box produces up to 8 contact points, plane-box up to 4, capsule-box / capsule-capsule / plane-capsule up to 2, and sphere pairs 1 point. Both $(A, B)$ and $(B, A)$ geometry orderings, explicit `<pair>`, `<exclude>`, and parent/welded body filtering are fully respected.
- **Convex cylinder/ellipsoid manifolds (multiCCD port):** capsule-cylinder, cylinder-cylinder and cylinder-box pairs use the pinned `mjc_Convex` multiCCD construction: one primary GJK+MPR witness, then frame perturbations (both geometries rotated in opposite directions about the primary contact point, tangent axes, $\pm 10^{-3}$ rad) with re-solve and distinctness tolerance $10^{-3}\min(\text{rbound})$; new witnesses inherit the primary penetration. Extra contacts are skipped with `MULTICCD` disabled and for sphere/ellipsoid involvement, matching pinned gating. Capsule-involved singles carry the pinned `mjc_fixNormal` analytic normal refinement (libccd-family MPR facet correction; wider application was measured to regress currently-exact pairs because the correction evaluates at the witness). Per-geom bounding radii (`geom_rbound`) feed the relative tolerance.

Solving contact and joint constraints in one coupled Delassus system prevents the numerical divergence (exceeding 15% error) that occurs when constraints are solved sequentially or decoupled.

```python
sim = MetalSimulation(model, batch_size=B, profile="integrated_euler_v1")
sim.step(steps=N, ctrl=controls, qfrc_applied=forces, xfrc_applied=wrenches)
sensor_data = sim.sensor_values()
```

### Capacity limits and guards

Development `CapacityLimits()` defaults are `max_nv=32`, `max_pairs=32`,
`max_slots=50`, `max_rows=640`, `max_batch=16`, and a 1 GiB memory budget.
The 50 contact slots accommodate the pinned `mjMAXCONPAIR` manifold capacity.
Limits are admission budgets; they do not alter the model's solver iteration
count or tolerance. Callers can retain smaller budgets explicitly, including
`CapacityLimits(max_slots=48, max_rows=96)`.

- The dense row route applies when `nr <= 96` and `nv <= 32`. The larger
  block-row route and component mass storage use their selected layout's
  estimate. `integrated_scalable_v1` supplies defaults of `max_pairs=64`,
  `max_slots=64`, `max_rows=256` and the model's velocity count when no explicit
  limits are provided. Explicit limits remain authoritative. Every physical
  backing allocation must fit signed 32-bit indexing and the memory budget;
  overflow raises `CapacityOverflow` before GPU allocation.
- Contact row counts depend on cone and `condim`: pyramidal dimensions1/3/4/6
  require1/4/6/10 rows; elliptic dimensions1/3/4/6 require1/3/4/6 rows. Compaction
  reports active candidates separately from allocated capacity.
- Compiled joint, connect, weld, tendon, flex edge, flex vertex and flex strain
  equality row inventories are admitted. Flex and integrated implicit
  combinations remain under broader qualification; admission is not a claim
  of complete trajectory parity.
- Rigid collision lowering admits plane, sphere, capsule, box, cylinder,
  ellipsoid, compiled convex mesh hulls and heightfields. A concave source mesh
  uses its compiler-produced convex collision hull; original triangle surfaces
  remain separate for ray and mesh-SDF queries. Mesh bounds are64 compiled hull
  vertices per geom,256 total hull vertices and256 total hull faces.
  Heightfields retain at most64 rows/columns,4,096 data samples and8 geom
  instances. Mesh-heightfield uses the pinned common-CCD prism pipeline with
  up to50 contact witnesses. Heightfield-heightfield remains rejected.
- SDF lowering admits plugin-free compiled octrees and the five pinned bundled
  bolt/bowl/gear/nut/torus descriptors, including analytic-SDF, mesh-SDF and
  SDF-SDF producers within their configured node and initpoint budgets.
  Third-party SDF plugin classes remain rejected. Complete contact, Jacobian
  and trajectory qualification is still open for several SDF cases.
- Spatial tendon paths, limits, friction and equality are admitted by the
  integrated profiles, with source/compiler guards on unsupported armature
  combinations. Built-in USER actuator defaults and registered device-native
  USER extensions are admitted. Host Python numerical actuator callbacks are
  a separate unsupported route.
- Native force, actuator and sensor extensions execute their device-stage
  contracts within `MetalSimulation`; init/reset/snapshot/restore ownership is
  explicit. Full extension and deformable/contact/integrator composition
  qualification remains open.
- Mocap bodies follow the pinned compiler restriction: jointless direct
  children of world. Prescribed per-environment poses use `sim.set_mocap`.

These are development-source admission and API contracts. They do not change
published wheel contents or establish complete physics coverage.

### Subtree angular momentum query

Development `api.mj_angmomMat(sim, body)` returns an owned float32 MPS matrix
`[batch,3,nv]`. Its world-axis rows map generalized velocity to angular momentum
about the selected body's subtree center of mass (kg m²/s). The query combines
rotated principal inertia with the moment of body linear momentum; it preserves
simulation state and does not advance time. An optional validated `dynamics`
record follows the other spatial queries' batched contract.

Focused native tests pass for dense and component profiles across free, ball,
hinge, slide and welded bodies, off-center/rotated inertia, two environments and
state mutation. Independent CPU matrix and physical momentum-sum tests also
cover world subtrees and fixed zero-DOF models. These checks do not establish
complete flex, extension, solver or distribution qualification.

Development `api.mj_subtreeVel(sim)` also returns owned `[batch,nbody,3]`
`subtree_linvel` (m/s) and `subtree_angmom` (kg m²/s), in world axes. It evaluates
all supplied body motion without changing stored sensors or sleep scheduling.
This differs explicitly from upstream's persistent, sleep-filtered subtree sensor
cache update. Independent CPU comparisons and native dense/component tests pass
for the same mixed-joint cases, including state mutation and output ownership.
The package surface catalogue records this query contract separately from exact
C/Python signature compatibility.

Development `api.mj_local2Global(sim, pos, quat, body, sameframe=0)` returns
owned world position `[batch,3]` and rotation `[batch,3,3]` arrays without
stepping or changing the supplied state. All five pinned `mjtSameFrame`
alignments are covered: rotation-only alignment still transforms position
through the regular body frame. Local quaternion inputs are used as supplied,
without normalization. A `None` input omits its corresponding output, matching
the pinned C null-pointer behavior (the upstream Python binding requires arrays).
CPU oracle tests cover every alignment and optional-output combination;
native dense/component tests cover distinct worlds, articulated and fixed
bodies, state mutation/restoration and output ownership. These focused checks
do not qualify the remaining solver, flex or distribution work.

### Native stages and state APIs

Development integrators distinguish the forward acceleration from the temporary
acceleration used to advance velocity. As in MuJoCo 3.10, implicit flex stiffness
correction changes the integration solve without replacing the public `qacc` or
its next-step warm start. Stored acceleration sensors sample the forward stage.
Eligible implicitfast midpoint DOFs are the explicit exception: MuJoCo overwrites
their final acceleration from the velocity increment. This is the intended state
contract; complete flex trajectory and restore qualification is still in progress.

- Development stage entry points: `mj_fwdPosition`, `mj_fwdVelocity`, `mj_fwdActuation`, `mj_fwdAcceleration` (unconstrained acceleration), and `mj_fwdConstraint` (coupled acceleration) consume generation-scoped prepared records. `mj_forward` and `mj_forwardSkip` orchestrate those stages. CPU checks cover publication order, control/applied-force overrides, foreign records and input mutation. Selected native prepared-chain tests pass; broader model and integrator compositions remain open. These functions have explicit batched return values and are not signature-compatible replacements for upstream functions.
- Development inverse dynamics: `mj_inverse` runs the complete prepared inverse stages; `mj_inverseSkip` can reuse a validated POS/VEL prefix. Both compute $M qacc + qfrc\_bias - qfrc\_passive - qfrc\_constraint$, with integrator-specific acceleration conversion when `INVDISCRETE` is enabled. The result is the generalized force required to produce the supplied acceleration; current actuator force is not subtracted. Explicit inverse queries preserve borrowed forward-stage storage and cache records, including on failure, using device snapshots. This incurs allocation overhead at the query boundary. `mj_inverseSkip(return_details=True)` also returns owned status and numerical intermediates, excluding the transient stage record. `mj_compareFwdInv` returns owned `[batch,2]` residuals from a validated constraint-stage record. Selected native legacy/integrated inverse rows, component implicit conversion and query restoration tests pass; full solver, flex and extension compositions remain open.
- Native state selectors: `mj_getState` and `mj_setState` conform to pinned `mjtState` bitmasks (TIME=1, QPOS=2, QVEL=4, ACT=8, HISTORY=16, WARMSTART=32, CTRL=64, QFRC_APPLIED=128, XFRC_APPLIED=256, EQ_ACTIVE=512, MOCAP_POS=1024, MOCAP_QUAT=2048, USERDATA=4096, PLUGIN=8192; groups PHYSICS=30, USER=8128, FULLPHYSICS=8223, INTEGRATION=16383) with atomic validation and per-world selection.

For a split evaluation in development source:

```python
from mujoco_metal import native_api as api
record = api.mj_fwdPosition(sim, return_record=True)
dynamics = api.mj_fwdVelocity(sim, record=record)
force = api.mj_fwdActuation(sim, record=record)
acceleration, status = api.mj_fwdAcceleration(sim, record=record)
acceleration, status, dynamics = api.mj_fwdConstraint(sim, record=record)
```

Each output is borrowed unless otherwise documented. A new position stage, reset,
restore or relevant input mutation invalidates cached reuse. Device input tensors
must identify captured storage; repeated host arrays are checked for matching
values at the explicit boundary. `mj_fwdConstraint` requires a published ACC;
it consumes that acceleration without another unconstrained solve. The selected
native chain test verifies this sequence; it does not qualify every model or
integrator combination.

Development split stepping is available as `api.mj_step1(sim)` followed by
`api.mj_step2(sim, ctrl=controls)`. Step1 evaluates POS/VEL, their sensors and
enabled energy diagnostics; step2 evaluates ACT/ACC/constraints and advances
one time step. Controls and applied forces may be installed between them.
Inputs are explicitly batched: controls `[batch,nu]`, generalized applied forces
`[batch,nv]`, and body wrenches `[batch,nbody,6]`. An invalid input rejects before
producer writes. Replacing POS or resetting/restoring state invalidates a pending
prefix. As in upstream split stepping, RK4 models use Euler for step2; implicit
models keep implicit integration.

`mj_step1(control_callback=...)` is an optional host extension. A Python callback
runs after the POS/VEL prefix and may return controls for step2; it is not GPU
native code. With `mjENBL_ENERGY` enabled, development `sim.energy` returns owned
`[batch,2]` potential/kinetic samples from the latest evaluated stages. Selected
native split-step and energy tests pass; broad integrator-composition
qualification remains pending.

Explicit inverse queries also preserve pending split-step records, sleep
scheduling, stored sensor status and plugin device state. Registered plugins
must implement device snapshot/restore methods for this rollback contract;
host snapshots are not used as an implicit fallback.

Development standalone inverse stages are also available:

```python
record = api.mj_invPosition(sim, return_record=True)
dynamics = api.mj_invVelocity(sim, record=record)
forces = api.mj_invConstraint(sim, qacc=requested_acceleration, record=record)
```

The first two calls publish borrowed POS/VEL stages and replace or extend the
current prepared record. They skip sensors, do not advance state and do not run
actuation or a forward constraint optimizer. The pinned engine uses the same
velocity computations for forward and inverse dynamics. `mj_invConstraint`
returns owned `[batch,nv]` generalized constraint forces from the cached cost
gradient; `return_details=True` also returns an owned status. This primitive
uses the supplied continuous acceleration directly. Integrator-specific
`INVDISCRETE` conversion belongs to the complete `mj_inverse` pipeline.
Constraint-query scratch and borrowed stages are preserved even after a failed
reduction. Stale or mutated prefixes are rejected before physics writes.
Selected dense/component and legacy native stage comparisons, both contact cone
types, both equality/friction force signs, ownership, sensor preservation and
failure recovery pass in [`test_native_inverse_split_019.py`](tests/test_native_inverse_split_019.py).
This is development source; complete flex, sleep and plugin combinations still
require qualification.

Independent development checks now pass 41 native cases covering the prepared
forward chain, legacy and integrated inverse rows, all tested contact cone
dimensions, both equality-force signs, mocap input ownership, split stepping,
energy and forward/inverse comparisons. See
[`test_native_inverse_stages.py`](tests/test_native_inverse_stages.py) and
[`test_forward_stages_019.py`](tests/test_forward_stages_019.py) for the exact
fixtures and tolerances. This qualifies those cases on the pinned MuJoCo 3.10.0
target; broad integrator, solver and extension compositions remain open.

### Custom native forces

`NativePlugin` force callbacks consume the current native smooth stage through
the `dynamics` argument. `PluginType.FORCE` contributes to the passive force
bucket; `PluginType.ACTUATOR` contributes to the actuator bucket. Each contributes
once to the forward force sum. Prepared and ordinary forward paths use the same
classification. This protocol is separate from upstream bundled plugins.

`CustomMagneticForcePlugin(charge=..., b_field=(Bx, By, Bz), body=...)` applies
the Cartesian Lorentz force at a body's inertial COM and projects it through
the translational Jacobian. The field uses world axes; `body` may be a compiled
ID, body name, or `None` for all non-world bodies. The plugin requires a complete
VEL-stage `dynamics` record. It does not reinterpret angular or slide joint
coordinates as Cartesian velocity. Initialization prepares model topology;
runtime force evaluation uses device tensors without CPU physics or readback.
Its native projection, two-world 20-step trajectory and exact checkpoint replay
checks pass in [`test_magnetic_force_projection_019.py`](tests/test_magnetic_force_projection_019.py).
The optional `run_host` route requires explicit host-callback opt-in and uses
upstream COM velocity and force projection; it is a host route.

### Device-native MuJoCo USER actuator callbacks

`NativeActuatorUserPlugin` implements the `mjDYN_USER`, `mjGAIN_USER`, and
`mjBIAS_USER` callback slots without invoking Python numerical callbacks from
the simulation. Register an unbound subclass with `default_registry`; declare
the compiled actuator IDs it owns in its `dynamics`, `gain`, and `bias`
bindings. Its `run_user_actuator_device` method writes only those output slots
into borrowed device tensors: activation derivatives have shape
`[batch, max(na, 1)]`, and gain and bias each have shape
`[batch, max(nu, 1)]`. Dynamics values use activation-units per second; gain
and bias use the actuator-force units defined by MuJoCo. `compute_mask` is a
boolean device vector selecting worlds whose rows must be overwritten on this
evaluation. The callback is a pure ACT-stage evaluator and must not advance
persistent state because it can run repeatedly at RK stages or during a
velocity query. Use `advance_user_actuator_device` once per accepted step for
stateful logic. Declare additional persistent device storage with
`device_workspace_bytes`, which participates in Simulation capacity preflight.

The implicit actuator velocity derivative follows pinned MuJoCo 3.10 behavior:
its USER gain/bias derivative contribution is zero, even when the registered
force callback reads actuator velocity. The callback protocol does not accept
derivative planes and does not claim an exact implicit Jacobian for a custom
velocity-dependent USER law. Other supported built-in actuator families keep
their pinned derivatives.

An unregistered USER dynamics slot keeps MuJoCo's zero-derivative behavior;
unregistered USER gain and bias slots use unit gain and zero bias. Registrations
are checked against compiled actuator types and IDs before device initialization.
This Python oracle/contract is exercised in
[`test_actuator_user_callbacks_019.py`](tests/test_actuator_user_callbacks_019.py);
the registered native callback subset has passed its selected Apple MPS
qualification matrix (activation dimensions 1/2/3, Euler/RK4/implicit, defaults,
replay and accepted-world failure isolation). This does not translate arbitrary
Python callbacks or qualify every custom plugin composition.

### Mass queries in development source

`mj_fullM` returns an owned `[batch,nv,nv]` mass including joint/tendon armature;
this explicitly requested conversion may expand component storage. `mj_mulM`
applies dense or component mass to `[batch,nv]` or `[batch,nrhs,nv]` without
expanding sparse mass. `mj_mulM2`, `mj_solveM`, and `mj_solveM2` return owned
`(value,status)` tensors. Status is `[batch]` int32; non-SPD/nonfinite/overflow
worlds report 2 and return zero. The factor convention is pinned descending
$M=L^T D L$: `mj_mulM2` applies $\sqrt D L$, and `mj_solveM2` applies
$\sqrt{D^{-1}}L^{-T}$. These are not symmetric principal matrix square roots.
Queries preserve borrowed stage storage, with allocation at the query boundary.
`mj_factorM(sim, *, dynamics=None)` returns owned factors without modifying
the simulation's prepared mass factor or stages. Its result contains `nv`,
`status`, and `blocks`; each block has device `dof_ids`, unit lower-triangular
`L`, diagonal `D`, and `Dinv`. Component input retains local blocks, including
tendon armature, without expanding a global dense matrix. A failure in any
component zeros every factor block for that world and reports status 2;
healthy worlds remain independent. This backend counterpart returns explicit
factors rather than writing upstream `MjData.qLD`.
Pinned CPU fixtures cover multiple worlds, free/ball/hinge/slide chains,
independent components, tendon armature and several RHS. Selected dense and
component native query checks now pass, including the public query's stage
preservation and pinned armature mass. Broader compositions remain under
qualification; these additions are not in the published wheel.
The factor query is checked against the pinned engine's actual CSR `qLD`
and `qLDiagInv`, including multiple worlds, component-local storage, and owned
output mutation. The focused mass-query packet passes 16 cases (nine CPU,
seven native MPS), including both public dense/component stage-preservation
fixtures, native failure isolation, zero DOFs and empty RHS batches. This does
not qualify every sleep, contact or plugin composition.

### Spatial queries in development source

The following device query implementations are available from
`mujoco_metal.native_api` in development source. Pinned CPU comparisons cover
free/ball/hinge/slide chains, welded/static bodies, world/local frames and all
five camera modes. Selected native smooth-stage Jacobian and Jacobian-dot
checks now pass, including distinct worlds and indexed MPS buffers under the
default MPS device. Remaining compositions are under qualification; this is
not an additional claim for the published 0.4.0 wheel.

- `mj_jac`, `mj_jacDot`, `mj_jacBody`, `mj_jacBodyCom`, `mj_jacGeom`,
  `mj_jacSite`: return owned translation/rotation Jacobians, each `[batch,3,nv]`,
  in world axes. Body origin and inertial COM are distinct. `mj_jacSubtreeCom`
  returns the mass-weighted subtree COM translation Jacobian.
- `mj_jacPointAxis(sim, point, axis, body)` returns owned point and axis
  Jacobians `[batch,3,nv]`. Both inputs use world coordinates; the axis is not
  normalized. Each axis column is the angular Jacobian column crossed with
  that vector. Focused CPU and native checks cover all joint families, static
  bodies, zero and nonunit axes, independent tangent differences, and prepared
  simulation preservation.
- `mj_contactForce(sim, contact_id, record=None)`: consumes a completed coherent
  native forward record and returns owned `[batch,6]` force/torque in the contact
  frame. The integer selects a native candidate slot; CPU compact contact indices
  have a different ordering. Invalid slots, inactive contacts and failed worlds
  return zero. Both coupled cone outputs and legacy condim 1/3 pyramidal outputs
  are handled on the device. This query does not detect contacts or run a solver.
  CPU ownership/source-decode checks and ten native public forward fixtures pass,
  covering both cones, condim 1/3/4/6 and the two legacy contact profiles. Broader
  flex, sparse-runtime and release-wheel compositions remain under qualification.
  Reset, restore or mutation invalidates a prepared record.
- `mj_objectVelocity`, `mj_objectAcceleration`: return owned `[batch,6]` values
  in angular-then-linear order, centered on a body inertial/regular frame,
  geom, site or camera. `flg_local=0` uses world axes; `1` uses object axes.
  Acceleration uses current velocity and supplied/owned `qacc`, the pinned RNE
  gravity convention and rotating-frame correction; it does not solve forward
  constraints. Supply matching `qvel`/`qacc` when querying an explicit stage.
- `mj_applyFT(sim, force, torque, point, body)`: returns the generalized wrench
  contribution `[batch,nv]`; it does not mutate held applied forces. Force and
  torque are world vectors `[batch,3]`, and either may be `None` for zero.

Queries evaluate current smooth state by default. An optional `dynamics=` stage
avoids that evaluation; its borrowed inputs must still describe the requested
state. Host arrays are staged at the explicit input boundary. Math remains on
the simulation device, and temporary forward storage is restored after queries.

### Global scene rays in development source

`native_api.mj_ray` accepts one origin/direction `[B,3]` per world;
`mj_multiRay` accepts origins `[B,3]` and directions `[B,N,3]`.
Both return an owned dictionary of MPS tensors: `distance`, `geomid`, `normal`
and `status`. Missing hits have distance/ID `-1` and zero normal. Status `1`
marks a nonfinite ray/pose or near-zero direction; status `2` marks invalid
SDF arithmetic. Failure is per ray and does not advance simulation state.
The signatures and returned tensors differ from upstream output pointers.
The direction thresholds preserve MuJoCo 3.10's distinction: `mj_ray` checks
norm against `mjMINVAL` (`1e-15`), while `mj_multiRay` checks squared norm.
Primitive quadratic intersections also use `mjMINVAL` for their coefficient
test. Scaling a direction can therefore change eligibility, not only the
returned ray parameter.

The scene includes visual geoms even when their collision masks are zero.
Six-entry `geomgroup`, `flg_static`, body exclusion and material/geom visibility
follow the pinned engine filters. `mj_multiRay`'s optional `cutoff` eliminates
geom bounding spheres; it is not a maximum intersection distance. Directions
need not be unit. The pinned SDF march normalizes its direction internally and
returns physical distance, while analytic intersections retain ray parameters.
`mj_rayMesh` and `mj_rayHfield` query a specific asset without visual filters.
`mju_rayGeom(sim, pos, mat, size, pnt, vec, geomtype)` queries explicit primitive
poses independently of scene and forward-stage storage. Inputs are `[B,3]`
except row-major matrices `[B,3,3]`; one plane, sphere, capsule, ellipsoid,
cylinder or box type applies to the batch. It returns owned `distance`,
`normal` and `status` tensors, with per-world invalid-input isolation.

`ray_queries.MetalRayQueries` is also usable with compiled model assets and
device poses independently of the stepping profile. Triangle/heightfield assets
are packed dynamically; non-convex mesh rays do not require convex contact
hulls. Model compilation and asset preprocessing remain host operations.
The constructor captures model constants; rebuild it after model edits. Runtime
rays and nearest-hit decisions execute on Metal without CPU physics fallback.
The query memory budget covers its captured assets, owned outputs and shader
parameter bindings. Caller inputs, simulation storage and temporary saved
forward workspaces are additional memory. Source gates compare default and
provided-stage queries, asset/filter behavior, non-convex mesh holes, SDFs,
small directions, heightfield side hits and capacity rejection with the pinned
CPU engine; they also check prepared-stage and state preservation.
These development APIs require source qualification separately from release
wheel coverage. They do not add a Metal renderer.

### Flex and skin rays in development source

`mj_rayFlex(sim, flex_layer, flg_vert, flg_edge, flg_face, flg_skin, flexid,
pnt, vec)` queries one compiled flex at the current simulation state. Origins
and directions are `[B,3]`. The position pipeline and ray intersections execute
on device while saved prepared stages and accepted simulation state are
preserved. Vertex spheres, edge capsules and triangle/tetrahedron faces follow
the pinned visibility flags and tetrahedron layer selection. Nearest vertex IDs
are local to the selected flex; source face order and hit/vertex tie rules are
retained. This query does not imply qualification of deformable collision or
complete deformable dynamics.

`mju_raySkin(sim, face, vert, pnt, vec)` takes explicit host integer topology
`face[nface,3]` and current float32 world positions `vert[B,nvert,3]`; the counts
are derived from their shapes. This utility queries supplied geometry without
evaluating skin animation or advancing physics. Both APIs return owned MPS
`distance`, `vertid`, `normal` and `status`; the normal is an additional output
compared with upstream `mju_raySkin`. Misses have distance/vertex ID `-1` and
zero normal. Status `1` isolates invalid rays/nonfinite vertices; `2` isolates
invalid arithmetic. Empty skin topology returns an ordinary miss.

`MetalRaySurface` also accepts immutable topology from `lower_ray_flex` or
`lower_ray_skin` and explicit contiguous MPS vertex positions. Its budget
covers captured topology, owned outputs and shader bindings; caller vertices,
saved simulation workspaces and other query objects are additional memory.
Topology and radius are captured constants: rebuild the program after model
edits. Tests compare all visibility flag combinations, deformed batched inputs,
tetrahedron layers, normals, local vertex selection, empty/degenerate faces,
invalid-world isolation and prepared-state preservation against MuJoCo 3.10.

### Position manifold utilities in development source

`mj_integratePos(sim, qpos, qvel, dt)`,
`mj_differentiatePos(sim, qpos1, qpos2, dt)` and
`mj_normalizeQuat(sim, qpos)` execute joint-coordinate mathematics on Metal.
Positions are `[B,nq]` and tangent velocities are `[B,nv]`; host arguments are
explicitly staged as float32 MPS tensors. Every call returns an owned `qpos`
or `qvel` tensor and an int32 `status[B]`. Unlike upstream in-place output
pointers, these APIs preserve caller tensors, accepted simulation state and
prepared forward records. They also work through `MetalPositionQueries`
without constructing a simulation or invoking CPU physics.

Free-joint translation, hinge/slide scalars and ball/free local quaternion
rotations follow MuJoCo 3.10. Quaternion signs are preserved by normalization;
zero or sub-`mjMINVAL` quaternions become identity. Finite float32 `dt` may be
negative. Integration accepts zero `dt` and still normalizes quaternions;
differentiation requires a nonzero value and chooses the source shortest signed
rotation. Nonfinite inputs produce status `1`; arithmetic overflow produces
status `2`. Failed position rows preserve the input and failed velocity rows
return zero. Invalid shapes, devices and nonrepresentable `dt` are rejected
before dispatch. The query memory budget covers static joint metadata,
dispatch bindings and one result/status allocation; retained results and
caller/simulation storage are additional memory.

### Finite-difference derivatives in development source

`native_api.mjd_stepFD(sim, eps=1e-6, flg_centered=False)` returns owned
device tensors `DyDq/DyDv/DyDa/DyDu` and `DsDq/DsDv/DsDa/DsDu`, plus
`status[B]`. Step Jacobians are transposed by input coordinate: state outputs
have trailing size `2*nv+na`, and sensor outputs have trailing size
`nsensordata`. The first dimension is the batch, so a result such as
`DyDq[b, i]` is the next-state derivative for tangent DOF `i` in world `b`.
Position inputs are perturbed in joint tangent coordinates, including free and
ball-joint quaternions. Controls use the pinned range-aware forward/backward
stencil independently per world; other inputs use forward differences by
default or centered differences when requested. Epsilon is rounded to
float32 for the native backend. The pinned CPU engine uses binary64, so small
epsilon values can amplify native output rounding; the derivative precision
contract and cross-feature qualification are still under review. `mjd_stepFD`
is a package API corresponding to the source's internal step derivative
routine; it is not a same-named upstream Python binding.

`native_api.mjd_transitionFD(sim, eps=1e-6, flg_centered=False)` returns
control-theory `A[B,nx,nx]`, `B[B,nx,nu]`, `C[B,nsensordata,nx]` and
`D[B,nsensordata,nu]`, where `nx=2*nv+na`, plus `status[B]`.
`native_api.mjd_inverseFD(sim, eps=1e-6, flg_actuation=False)` returns
transposed force derivatives `DfDq/DfDv/DfDa[B,nv,nv]`, sensor derivatives
`DsDq/DsDv/DsDa[B,nv,nsensordata]`, the source packed mass derivative
`DmDq[B,nv,nM]`, and `status[B]`. The packed mass output follows MuJoCo's
`qM` diagonal-then-ancestor order. `flg_actuation=True` subtracts the native
ACT-stage generalized force before differentiating.

Every perturbation restarts from the same full device checkpoint. State,
held controls/forces, histories, warm starts, sensors, sleep state, plugin
state, mutable NumPy bookkeeping, cache contents and prepared-record identity
are rolled back after success and exceptions. Plugin extensions must supply
`device_snapshot()`, `device_snapshot_bytes()` and `restore_device(payload)`
hooks for derivatives. The byte bound must cover the complete rollback payload
before snapshot allocation. Stateless extensions inherit the zero-byte contract.
Failed worlds receive zero derivative rows and retain a nonzero status; healthy
worlds continue. The query budget is checked for owned derivative outputs,
perturbation buffers and reusable rollback storage before those allocations.
These source APIs perform no CPU MuJoCo physics or device-value readback in the
finite-difference loop. As in the pinned engine, `mjd_stepFD` permits RK4 but
rejects actuator history; `mjd_transitionFD` also rejects RK4;
`mjd_inverseFD` rejects RK4 and noslip iterations.
They remain development APIs requiring native qualification and do not add
release-wheel coverage.

### Analytical quaternion derivatives in development source

`mjd_subQuat(qa, qb)` accepts contiguous float32 MPS tensors `[B,4]` and returns
owned `Da` and `Db` matrices `[B,3,3]` for local rotational perturbations.
`mjd_quatIntegrate(vel, scale)` accepts `[B,3]` angular velocities and a finite
float32 scalar and returns owned `Dquat`, `Dvel` `[B,3,3]` and `Dscale` `[B,3]`.
Following MuJoCo 3.10, `Dvel` differentiates **scaled** velocity; multiply by
`scale` to differentiate the original velocity. These analytical Metal kernels
operate independently of a simulation and do not call CPU physics.

Both utilities return int32 `status[B]`: `0` success, `1` nonfinite inputs,
`2` arithmetic overflow. Failed rows contain zero derivatives. Public tensor
admission and memory-budget checks precede device allocation; inputs are not
mutated. The budget covers program bindings plus one result/status allocation;
caller inputs and previously retained results are additional memory. Native
qualification is still in progress. These functions do not establish complete
transition or inverse finite-difference analysis support.

### Equality activation, reset, snapshots and failure

`mj_stateSize(model, sig)` computes the flat vector width from compiled model
metadata. `mj_extractState(model, source, srcsig, dstsig)` extracts an owned
batched tensor with pinned upstream field ordering, including history, equality
activity, mocap and plugin-state widths. These two utilities use upstream
`mjtState` masks. They are distinct from the backend dictionary interface and
its `StateSpec` masks in `mj_getState`/`mj_setState`. Extraction copies even
nonfinite values and does not admit them as simulation state. CPU tests compare
every valid mask's size and individual/composite extraction with pinned MuJoCo;
native extraction passes its selected batched ownership and device checks.
Broader state/API composition remains under qualification.

```python
sim.set_equality_active(values, env_ids=None)  # persistent per-env, per-equality booleans
sim.set_mocap(pos, quat=None, env_ids=None)  # prescribed per-env mocap poses
sim.reset(env_ids=None, qpos=None, qvel=None, eq_active=None, mocap_pos=None, mocap_quat=None)
sim.reset_to_keyframe(key_id, env_ids=None)  # pinned mj_resetDataKeyframe semantics
sim.copy_environment(src, dst)
state_rows = sim.state.snapshot()  # narrow device-state snapshot
sim.state.restore(state_rows)
checkpoint = sim.snapshot()  # full trajectory checkpoint, including solver seeds
sim.restore(checkpoint)
```

- `set_equality_active` accepts host boolean/integer arrays of shape `(neq,)` (broadcast to the selected worlds) or `(len(env_ids), neq)`, or contiguous int32/bool MPS tensors of the same shapes. Device tensors are copied, never borrowed or modified. `env_ids=None` selects all worlds. Validation is atomic: malformed input leaves every world unchanged. Changes take effect on the next step or assembly, invalidate cached assembly/diagnostics, and never clear sticky per-world failure status. Releasing an equality clears its stale rows, multipliers and generalized force on recomputation; reattaching reuses the compiled reference anchors/poses.
- `reset` with `eq_active=None` restores compiled `eq_active0` defaults for the selected worlds; an explicit `eq_active` array overrides them. `mocap_pos`/`mocap_quat=None` restore compiled reference frames; explicit arrays override. Unselected worlds are untouched. Reset clears sticky failure status for the selected worlds only. `sim.reset` additionally clears held per-call inputs (`ctrl`, `qfrc_applied`, `xfrc_applied`) for the selected worlds and invalidates cached assembly.
- `set_mocap` accepts host arrays or contiguous float32 MPS tensors: `(nmocap, 3/4)` broadcast to the selected worlds or `(len(env_ids), nmocap, 3/4)` per world, or a single `(…, nmocap, 7)` posquat array with `quat=None`. Inputs are copied, validation is atomic, sticky failure is preserved, changes take effect on the next step/assembly.
- `reset_to_keyframe` mirrors pinned `mj_resetDataKeyframe`: time to `key_time`, qpos/qvel/mocap to the key values, qacc/status cleared, equality activity to compiled defaults, held controls to `key_ctrl` for the selected worlds. Invalid key ids fail atomically.
- `copy_environment` copies every state row (qpos/qvel/qacc/time/status/eq/mocap) src to dst and bumps generation.
- `DeviceState.snapshot()` captures device-state rows and is not a full trajectory checkpoint. Use `MetalSimulation.snapshot()` and `restore()` for exact replay: they also carry acceleration and constraint warm starts, held inputs, sensor/history storage, delay rings and extension state. Restoring only kinematic rows cannot reproduce the rounding of a later solve with different retained seeds.
- Development device-state snapshots use schema version 6, including warning statistics and reset epochs, plus equality/mocap/activation fields when present. Older schemas remain validated on restore: a version-1 snapshot cannot restore a model with equalities, versions below 3 cannot restore mocap bodies, and versions below 4 cannot restore activation storage. Schemas below 5 restore zero warning history and schemas below 6 restore legacy reset epochs. Model/profile/dimension mismatches are rejected before writes. Simulation-level snapshots have their own schema and include the nested device-state snapshot.
- Model changes are shared by the batch, not per environment: `ModelLifecycle.recompute_body_masses` (via `mj_setConst`) and `update_geom_contact` (friction/solref/solimp/margin/gap) are host preparation utilities with transactional validation; reference poses/shapes/types are recompile-required. `MetalSimulation.apply_lifecycle` adopts a rebuilt lifecycle/model into a live simulation atomically (structural-count compatibility required): device stages are rebuilt and swapped, live state and held inputs preserved, fingerprints/generation advance, caches invalidate. On any failure the simulation is untouched and keeps stepping. Development device state owns activation, history, userdata, warm-start acceleration, controls and applied forces. Registered extensions must provide the documented lifecycle and device rollback hooks; this does not imply support for arbitrary upstream plugins. Explicit state/query boundaries may synchronize for validation. Profile/model admission and source qualification remain separate from state storage.
- A failed physics step rolls back physical state/time for that world and stays sticky until reset/restore; an accepted activity change is persistent input state, not a partially applied step.

Try the [magnetic crane demo](examples/magnetic_crane.md), the [latch-and-release cargo bridge demo](examples/cargo_bridge.md), the [clockwork parcel sorter demo](examples/clockwork_parcel_sorter.md), the [robotic marble music machine demo](examples/marble_music_machine.md) or the [spacecraft force-control demo](examples/space_docking.md), or use the
[installation and diagnostic guide](INSTALL.md) for `pip install mujoco-mac-metal`
and `mujoco-metal doctor --gpu`. Install version 0.4.0 for the additional bounded profiles.

### Sparse query boundaries in development source

`mj_kinematics`, `mj_comPos` and `mj_comVel` expose owned, batched current-state
Cartesian fields from the native smooth pipeline. Kinematics returns body,
inertial, geom and site positions/rotation matrices, body quaternions and joint
anchors/axes. `mj_comPos` returns `subtree_com[B,nbody,3]`, spatial inertia
`cinert[B,nbody,10]` and motion DOFs `cdof[B,nv,6]`; zero-mass subtrees use the
pinned inertial-position fallback. `mj_comVel` returns `cvel[B,nbody,6]` and
`cdof_dot[B,nv,6]` in rotation:translation convention. These full-state queries
preserve simulation state and prepared records rather than updating upstream
sleep-filtered caches. A validated `dynamics=` smooth stage may be supplied.
Additional output/scratch and rollback storage is checked against the query
memory budget before allocation. CPU oracle checks and native public queries
pass selected dense/component fixtures with CPU fallback forbidden, prepared
records preserved and output retention through reset. These checks do not
establish full sleep, mocap or model-lifecycle composition qualification.

`mj_transmission(sim)` returns owned `actuator_length[B,nu]`, dense
`actuator_moment[B,nu,nv]` and `status[B]` from the production native POS stage.
The dense moment representation expands MuJoCo's packed moment rows; actuator
velocity belongs to VEL and is not returned by this query. The active profile
must admit the model's transmission family. Outputs survive reset and subsequent
queries. Prepared stage owners, plugin state and workspaces are restored on
success and exception. Preflight accounts for rollback storage, temporary POS
contexts and owned outputs. Focused native CPU-oracle checks pass the admitted
integrated transmission families, the legacy scalar motor profile, empty
outputs, reset/restore, prepared-owner retention and failed-plugin rollback.
These checks cover the query contract and selected fixtures, separately from
full transmission/solver/integrator composition qualification.

The native runtime check implementations mirror `mj_checkPos`, `mj_checkVel`
and `mj_checkAcc`: they report the first bad coordinate per world, update
warning counts and last-info indices, filter velocity/acceleration checks to
awake DOFs, and apply `mjDSBL_AUTORESET` on device. A mixed-world reset carries
an explicit row-valid mask through recovery producers and prepared ACC and
CONSTRAINT consumers; invalid cached rows return status 3 and zero numerical
outputs, while healthy rows retain their borrowed stage owners. `mj_checkAcc`
records the warning before its selected-world forward recovery. CPU source and
contract tests pass. Native MPS mixed-world checks for dense, component and
legacy contact routes are pending independent qualification; this is not yet a
full physics-closure claim.

`mj_rnePostConstraint(sim, forward_result=None, qacc=None)` returns owned
`cacc`, `cfrc_ext`, `cfrc_int` `[B,nbody,6]`, `subtree_com[B,nbody,3]`,
`qacc[B,nv]` and `status[B]` from native post-constraint RNE. A supplied
forward bundle must be coherent and belong to this simulation. Without one,
native forward stages are evaluated inside a transaction that restores state,
prepared records and plugin workspaces. A float32 MPS acceleration override
isolates post-stage math from the forward solver and omits its low residual.
Failed worlds return zero fields with their status. Budget admission covers
rollback, query scratch, output and peak temporary storage. Sensor-free models
allocate a query-local RNE workspace.

External spatial forces follow pinned MuJoCo: applied body wrenches, rigid
geom contacts and connect/weld equality forces. The upstream routine skips flex
contacts and joint/tendon/flex equality contributions to `cfrc_ext`; their
dynamical effects still enter the acceleration and internal force calculation.
Selected native post-stage, legacy contact, sleep and rollback checks pass.
Full end-to-end constraint-solver composition remains under qualification;
a shared-acceleration oracle does not establish forward-solver correctness.

`mj_rne(sim, flg_acc=1, qvel=None, qacc=None, dynamics=None)` returns owned
batched rigid-body recursive Newton–Euler generalized forces. It includes
gravity and Coriolis terms; `flg_acc=0` omits acceleration. As in pinned MuJoCo,
joint/tendon armature and applied, passive, actuation and constraint forces are
excluded. This full-state query evaluates every body without updating upstream's
sleep-filtered persistent output. CPU oracle checks cover free, ball, hinge and
slide chains with rotated inertia and gravity disabled. Native dense/component
query fixtures pass with CPU physics forbidden and prepared state preserved;
this interface does not establish full dynamics coverage.

Prepared `mj_mulJacVec` and `mj_mulJacTVec` consume a coherent CONSTRAINT
record. Inputs have explicit batched dimensions; outputs own their storage.
The row dimension uses the backend's reserved canonical rows, rather than the
CPU engine's compact active rows. Inactive rows and failed worlds contribute
zero. State changes invalidate the prepared record. These queries consume
packed CSR directly when supplied; they do not assemble rows or rerun the
forward solver. Native dense fixtures qualify both cones and condim1/3/4/6.
Integrated sparse producers and their full physics composition remain under
qualification.

The component-block profile uses `mass_blocks`, immutable
`mass_block_layout` metadata and, when nonzero, `tendon_armature_blocks`.
Explicit inverse queries multiply these blocks directly; they do not reconstruct
a global dense mass matrix. Fixed and spatial tendon Jacobians are prepared
before armature projection. Supplied stage dictionaries must match the compiled
model layout, device, dtype, shape and finite-value contract. Supplying both
`poses` and `dynamics` requires their position-record identity to match.

Inverse constraint queries assemble canonical rows without running the forward
optimizer. Reserved inactive rows contribute no inverse force even if their
impedance storage is positive. Pinned CPU tests cover bilateral activity and
pyramidal/elliptic contact cost gradients for condim3/4/6, including all three
elliptic cost regions. Selected native inverse row and producer checks now
pass; complete solver, integrator and flex compositions remain open.
These changes describe development source and do not change the published wheel.
