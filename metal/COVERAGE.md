# Pinned-version support contract

Full milestone acceptance remains open; see [current status](STATUS.md).
The generated rows describe scoped contracts, not full-backend completion.

This table is generated from `mujoco_metal/registry.py` (`REQUIREMENTS`) by
`python -m mujoco_metal coverage`. Do not edit the table by hand: the drift
test `test_coverage_contract.py::test_docs_match_generated_table` fails unless
this file contains the generated output verbatim between the markers. Change
the contract in `registry.py`, regenerate, and paste the result here.

```sh
PYTHONPATH=metal python -m mujoco_metal coverage
```

Owner milestone identifiers track implementation and qualification groups. `baseline`/`004` rows
are implemented; `out-of-scope` rows are explicit project exclusions (RL/PPO,
deployment targets, broad hardware testing, extra MuJoCo versions, Metal
renderer), not future physics work.

Implementation and qualification are separate. `native_gpu/unqualified` can
describe an integrated device path with focused passing tests whose full
composition gate remains open. `gpu_qualified` describes the cited fixtures
and stated restrictions, not every combination of model features, solvers,
integrators and runtime flags. Test filenames identify evidence to inspect;
their presence in this table is not a claim that every test in them passed.
The pinned inventory is still being reconciled (`INVENTORY_COMPLETE=False`);
this table is not a declaration of complete MuJoCo compatibility.

Development `preflight --inventory --json` also exposes `pinned_enums`, a separate
catalogue of all 70 public `mjt` enum groups and 529 members from MuJoCo 3.10.0.
It records pinned header references, numeric values, domains and milestone
owners, and detects new or changed binding groups and values. This extends
the earlier selected-feature inventory to sleep policies, flex self-contact,
constraint states, extension capabilities and host compilation/UI/rendering
enums. A matching catalogue establishes specification coverage only;
`gpu_qualified` and overall `inventory_complete` remain false. The feature
table below remains the implementation and fixture-qualification contract.

`pinned_binding_surface` additionally records 809 public fields across `MjModel`,
`MjData`, `MjOption`, `MjContact` and `MjStatistic`; 69 exported non-enum Python
classes and their public member sets; all 317 public free-function bindings;
all 29 option fields; and an exact map of the 67 package `native_api`
counterparts. Each remaining upstream function has its own host boundary or
subsystem-owner row; internal/prepared counterparts are distinguished from
same-named wrappers. It retains 812 field declarations and 527 function
declarations from the pinned headers separately, so C-only storage and APIs are
not silently counted as Python exposure. Added or removed fields, class members,
functions, or wrapper names fail the drift checks. This is a specification and
ownership crosswalk; complete feature-to-execution mapping and native
qualification remain open (`feature_mapping_complete=False`,
`function_mapping_complete=False`).

<!-- COVERAGE_TABLE_BEGIN -->
| ID | Capability | Execution | Status | Owner | Limitation | Admission | Tests |
|---|---|---|---|---|---|---|---|
| REQ-MOD-001 | pinned MuJoCo 3.10.0 model lowering | host | lowered/cpu_oracle | baseline | requires MuJoCo 3.10.0; other versions rejected | accept 3.10.0, reject others | test_model.py |
| REQ-MOD-002 | bounded stepping profile validation | host | lowered/cpu_oracle | baseline | unknown profiles and unsupported combinations rejected | accept listed profiles, reject others | test_stepping.py |
| REQ-MOD-003 | pinned MjOption lowering | host | lowered/cpu_oracle | 005 | all 29 pinned MjOption fields are explicitly assigned to physics, solver, collision, extension, or visualization requirements; unlisted option combinations remain rejected by profile validation | field-level requirement crosswalk plus bounded profile admission | test_binding_inventory.py, test_stepping.py |
| REQ-JNT-001 | hinge/slide/ball/free kinematics | device | native_gpu/gpu_qualified | baseline | one immutable model per batch | accept all four joint types | test_model.py |
| REQ-JNT-002 | scalar hinge/slide joint limits | device | native_gpu/gpu_qualified | baseline | ball/tendon limits rejected | accept limited hinge/slide, reject limited ball | test_coupled_constraints.py |
| REQ-JNT-003 | ball joint limits | device | native_gpu/gpu_qualified | 009 | pinned axis-angle rows in reserved slots (integrated profile; legacy joint-constraints profile stays scalar-only) | ball branch in joint-limit loop | test_rigid_constraints_009.py |
| REQ-GEO-001 | plane/sphere/capsule/box collision | device | native_gpu/gpu_qualified | baseline | all 9 valid unordered pairs; plane-plane yields nothing | accept the four primitives, reject others | test_primitive_collision_qualification.py |
| REQ-GEO-002 | cylinder/ellipsoid collision | device | native_gpu/unqualified | 010 | 010 implements plane/sphere-cylinder analytic pairs + convex MPR pairs | admit cylinder/ellipsoid pairs with pinned slot counts | test_coupled_constraints.py |
| REQ-GEO-003 | convex mesh collision | device | native_gpu/unqualified | 011 | 011 implements convex-hull vertex-support GJK/MPR singles + face-snap | admit convex mesh pairs (<=64 verts) with singles slot counts | test_mesh_contact_011.py |
| REQ-GEO-004 | heightfield collision | device | native_gpu/unqualified | 012 | 012 implements per-prism terrain collision vs sphere/capsule/box/cylinder/ellipsoid; mesh-hfield and hfield-hfield rejected, plane-hfield yields nothing | admit heightfield pairs within dim/data caps with per-prism slot counts | test_hfield_contact_012.py |
| REQ-GEO-005 | SDF collision | device | native_gpu/unqualified | 013 | 013 implements plugin-free mesh-octree SDF vs analytic/SDF-SDF via Halton/descent; mesh-SDF gap, plugin SDFs per 019 contract | admit SDF pairs within oct/node/initpoint caps with per-seed slot counts | test_sdf_contact_013.py |
| REQ-GEO-006 | visual-only geometry decorations | upstream_host | upstream_cpu_only/unqualified | out-of-scope | renderer labels/decorations, never collision geometry | never admitted as collision geometry | — |
| REQ-GEO-007 | mjtGeom count sentinel | none | not_implemented/unqualified | 005 | mjNGEOMTYPES is a count, not a selectable type | never admitted | — |
| REQ-CON-001 | condim 1/3/4/6 with pyramidal/elliptic cones | device | native_gpu/unqualified | baseline | selected both-cone force and sensor trajectories pass after packed-J consumer correction; complete option/integrator/contact composition remains unqualified; other condim values rejected | accept 1/3/4/6, reject others | test_coupled_constraints.py |
| REQ-CON-002 | nine valid primitive contact pairs | device | native_gpu/gpu_qualified | baseline | plane-plane yields nothing; cylinder/ellipsoid/mesh/hfield/SDF pairs per owning milestones | accept the 9 primitive pairs | test_primitive_collision_qualification.py |
| REQ-CON-003 | anisotropic sliding/torsional/rolling friction | device | native_gpu/unqualified | baseline | five-coefficient friction expansion and pair mixing implemented; current development anisotropic and rolling-friction numerical gates remain open | accept finite friction rows | test_coupled_constraints.py |
| REQ-INT-001 | Euler/RK4/implicitfast bounded profiles | device | native_gpu/gpu_qualified | baseline | per-profile supported combinations only | accept listed profile/integrator pairs | test_integration.py |
| REQ-INT-002 | implicit integrator (bounded contact-free) | device | native_gpu/gpu_qualified | 015 | nonsymmetric LU solve with automatic passive, tendon, Coriolis and FD fluid derivatives (eps 1e-3); coupled contacts/equalities/flex guarded | admit mjINT_IMPLICIT in contact_free_implicit_v1 and integrated_implicit_v1; integrated combinations remain under REQ-INT-003 qualification and explicit composition guards | test_implicit_015.py, test_implicit_derivative_repair.py |
| REQ-INT-003 | component implicit solve and flex integration | device | native_gpu/unqualified | 018 | native GMRES and all six selected implicit/implicitfast flex material/damping profiles pass 25-step trajectories and exact checkpoint replay; broader contact/material compositions remain unqualified | development integrated implicit profiles; no CPU physics fallback | test_implicit_effective_operator_017.py, test_flex_implicit_simulation.py, test_sparse_implicit_public_019.py |
| REQ-INT-004 | integrated RK4 constraints and sleep lifecycle | device | native_gpu/unqualified | 017 | selected native constraint/sleep fixtures pass trajectories, checkpoint replay and reset lifecycle; complete contact/sleep compositions remain unqualified | development integrated RK4 profile; bounded contact-free profiles are separate | test_runge_kutta.py, test_integrated_scalable_public_017.py |
| REQ-JAC-001 | dense Jacobians | device | native_gpu/gpu_qualified | baseline | native pipeline is dense | accept dense | test_coupled_constraints.py |
| REQ-JAC-002 | sparse/auto Jacobians | device | native_gpu/unqualified | 017 | direct canonical CSR producers and packed consumers pass selected rigid both-cone and flex plane/sphere fixtures; complete sparse solver and feature composition remain unqualified | model-derived dense/CSR layout with checked canonical row spans | test_constraint_jacobian_pattern_017.py |
| REQ-SOL-001 | native projected Gauss-Seidel solver | device | native_gpu/gpu_qualified | baseline | recorded PGS subset qualified; complete cone, budget and cross-feature composition gates remain open | PGS selects the projected solver | test_coupled_constraints.py |
| REQ-SOL-002 | native primal conjugate-gradient solver | device | native_gpu/unqualified | 014 | native primal CG with configured search budgets; selected mixed-island force and acceleration comparisons pass, but exact iteration counts and full composition remain unqualified | CG selects native primal CG | test_solver_completion_014.py, test_solver_islands_014.py |
| REQ-SOL-003 | native primal Newton solver | device | native_gpu/unqualified | 014 | native Newton/PCG Hessian path; selected warm elliptic and mixed mass/contact-island gates pass; complete composition qualification remains open | Newton selects native primal Newton | test_solver_completion_014.py, test_solver_islands_014.py |
| REQ-EQ-001 | joint/connect/weld equalities with per-env activity | device | native_gpu/gpu_qualified | 004 | 1/3/6 rows, body/world/site forms incl. mocap anchors, torquescale, schema-2/3 activity state | accept joint/connect/weld body/site/world/mocap, reject others even if inactive | test_connect_weld.py, test_equality_activity.py, test_equality_qualification.py, test_mocap_state.py |
| REQ-EQ-002 | tendon equality | device | native_gpu/gpu_qualified | 008 | cubic tendon-length coupling as coupled rows (single or paired tendons) | tendon_constraint_rows equality branch | test_spatial_tendons_008.py |
| REQ-EQ-003 | flex distance equality | device | native_gpu/gpu_qualified | 018 | flex edge distance equalities on GPU | mjEQ_FLEX distance constraints coupled with block-row solver | test_flex_018.py |
| REQ-EQ-005 | flex vertex and strain equalities | device | native_gpu/unqualified | 018 | compiled vertex/strain row producers and coupled assembly implemented; full flex/contact/integrator qualification remains open | validate compiled equality row spans and flex topology | test_coupled_flex_equality_rows.py |
| REQ-EQ-004 | removed distance equality | none | not_implemented/unqualified | 009 | removed upstream in MuJoCo 2.2.2; rejected | reject distance equalities | — |
| REQ-TEN-001 | fixed-joint tendons with spring/damping/armature | device | native_gpu/gpu_qualified | baseline | fixed joint paths only; no wrapping, limits, friction or equality | accept fixed joint tendons | test_tendons.py |
| REQ-TEN-002 | spatial tendons, wrapping, limits, friction, equality | device | native_gpu/gpu_qualified | 008 | pulley/site/sphere/cylinder paths; tendon limits, friction loss and tendon equality as coupled rows; site-only armature with Jdot bias (wrapped armature rejected upstream) | spatial kinematics + tendon_constraint_rows + armature_dots | test_spatial_tendons_008.py |
| REQ-TRN-001 | joint/jointinparent/tendon transmissions | device | native_gpu/gpu_qualified | baseline | stateless fixed/affine scalar scope | accept listed transmissions | test_transmissions.py |
| REQ-TRN-002 | slidercrank/site/body transmissions | device | native_gpu/gpu_qualified | 007 | rigid transmissions incl. ball/free gears; spatial-tendon paths stay in 008 | general kinematics + BODY adhesion from same-step candidates | test_actuators_007.py |
| REQ-TRN-003 | undefined transmission sentinel | none | not_implemented/unqualified | 005 | never admitted | reject undefined | — |
| REQ-DYN-001 | stateless (none) actuator dynamics | device | native_gpu/gpu_qualified | baseline | no activation state | accept dyntype none | test_actuation.py |
| REQ-DYN-002 | stateful activation dynamics (built-in) | device | native_gpu/gpu_qualified | 007 | built-in integrator/filter/filterexact/muscle/dcmotor only | act_dot switch + exact-slot advance, schema-4 state | test_actuators_007.py |
| REQ-DYN-003 | user-callback actuator dynamics | device | native_gpu/gpu_qualified | 019 | requires an explicitly registered NativeActuatorUserPlugin; unbound USER dynamics use pinned zero-derivative behavior; actnum=1 consumes the callback scalar return and higher-order activation consumes the written act_dot slice | registered callback writes activation derivative and force-law outputs; Euler/RK4/implicit native parity, accepted-step state, reset/failure/replay tests | test_actuator_user_callbacks_019.py |
| REQ-GAIN-001 | fixed/affine gains | device | native_gpu/gpu_qualified | baseline | stateless scalar scope | accept fixed/affine | test_actuation.py |
| REQ-GAIN-002 | muscle/DC-motor gains (built-in) | device | native_gpu/gpu_qualified | 007 | built-in muscle/dcmotor only | muscle FLV + DC resistance/voltage paths | test_actuators_007.py |
| REQ-GAIN-003 | user-callback actuator gains | device | native_gpu/gpu_qualified | 019 | requires an explicitly registered NativeActuatorUserPlugin; unbound USER tags use pinned unit-gain default; implicit USER velocity derivative is zero per MuJoCo 3.10 | registered callback writes USER gain plane; native Euler/RK4/implicit step tests | test_actuator_user_callbacks_019.py |
| REQ-BIAS-001 | none/affine biases | device | native_gpu/gpu_qualified | baseline | stateless scalar scope | accept none/affine | test_actuation.py |
| REQ-BIAS-002 | muscle/DC-motor biases (built-in) | device | native_gpu/gpu_qualified | 007 | built-in muscle passive + DC back-EMF/cogging/LuGre | bias switch + post-clamp DC mechanics | test_actuators_007.py |
| REQ-BIAS-003 | user-callback actuator biases | device | native_gpu/gpu_qualified | 019 | requires an explicitly registered NativeActuatorUserPlugin; unbound USER tags use pinned zero-bias default; implicit USER velocity derivative is zero per MuJoCo 3.10 | registered callback writes USER bias plane; native Euler/RK4/implicit step tests | test_actuator_user_callbacks_019.py |
| REQ-SENS-001 | current-state kinematic/clock/gyro/velocimeter queries | device | native_gpu/gpu_qualified | baseline | explicit current-state query; stored forward-stage samples and compiled delay/interval history use separate APIs; pinned noise metadata does not introduce stochastic samples | accept the 14 listed types | test_sensors.py |
| REQ-SENS-002 | remaining sensor families (016) | device | native_gpu/gpu_qualified | 016 | touch/accel/force/torque/magnetometer/range/tendon/actuator/limit/frame-acc/subtree/insidesite/geomdist/contact/energy; cutoff honored; compiled delay/interval history is separate from current-state queries; SDF and extension combinations require their own qualification | accept listed built-in types subject to profile and query contracts; extension types have separate requirements | test_sensors.py, test_sensor_families_016.py, test_sensor_force_016.py, test_sensor_spatial_016.py, test_sensor_remaining_016.py |
| REQ-SENS-003 | sensor extensions (016/019) | device | native_gpu/gpu_qualified | 019 | camera-projection, tactile, plugin, and user sensors supported natively | accept sensor extensions | test_sensor_extensions_016.py, test_extensions_019.py |
| REQ-STATE-001 | time/qpos/qvel/eq_active/mocap state ownership | device | native_gpu/gpu_qualified | 006 | persistent per-env poses/masks, schema-3 snapshots, keyframe reset, copy | accept matching snapshots, reject v1-into-neq and <v3-into-mocap | test_device_state.py, test_equality_activity.py, test_mocap_state.py |
| REQ-STATE-002 | control/applied-force state ownership | device | native_gpu/gpu_qualified | baseline | per-call held inputs validated before the device loop | accept finite host/device inputs | test_simulation.py |
| REQ-STATE-003 | mocap position/quaternion inputs | device | native_gpu/gpu_qualified | 006 | jointless world-child mocap bodies; per-env poses, keyframe reset, schema-3 snapshots | accept valid mocap, reject non-world-child | test_mocap_state.py |
| REQ-STATE-004 | actuator activation state | device | native_gpu/gpu_qualified | 007 | schema-4 act storage, reset/keyframe/restore/copy, exact-slot advance | na rows with actearly/actrange semantics | test_actuators_007.py |
| REQ-STATE-005 | pinned acceleration warmstart state | device | native_gpu/unqualified | 014 | canonical qacc_warmstart state is distinct from the legacy retained-multiplier API; scalar primal cost-gated initialization implemented; complete cone/budget/composition qualification pending | finite device state with shape (batch, nv); WARMSTART disable honored by solver | test_solver_completion_014.py, test_native_api_transactions.py |
| REQ-STATE-006 | history state | device | native_gpu/unqualified | 015 | canonical compiled history storage, interpolation, delay/interval sampling and selected lifecycle implemented; full integrator/sleep/extension composition qualification pending | validate compiled addresses, sizes and interpolation; reject invalid or overlapping layouts | test_history_parity.py, test_native_api_transactions.py |
| REQ-STATE-007 | userdata/plugin state | device | native_gpu/unqualified | 019 | compiled userdata/plugin-state tensors and selected state lifecycle implemented; native callback-owned state requires explicit device rollback methods; arbitrary upstream plugins are not automatically ported | validate compiled state dimensions; stateful native callbacks require device snapshot, full restore and masked restore | test_native_api_transactions.py |
| REQ-STATE-008 | getState group selectors | device | native_gpu/gpu_qualified | 019 | mj_getState and mj_setState group API supported on device | native state get/set selectors | test_extensions_019.py |
| REQ-STATE-009 | state count sentinel | none | not_implemented/unqualified | 005 | mjNSTATE is a count, not selectable state | never admitted | — |
| REQ-DSBL-001 | honored disable flags | device | native_gpu/gpu_qualified | baseline | constraint/equality/frictionloss/limit/contact/spring/damper/gravity/clampctrl/warmstart/filterparent/actuation/refsafe/sensor/midphase/eulerdamp/autoreset honored per stage | accept listed flags | test_simulation.py |
| REQ-DSBL-002 | island/ccd disable flags | device | native_gpu/gpu_qualified | 017 | island disable flag supported; ccd flags rejected | native island disable | test_scalable_017.py |
| REQ-DSBL-003 | disable-bit count sentinel | none | not_implemented/unqualified | 005 | mjNDISABLE is a count, not a flag | never admitted | — |
| REQ-ENBL-001 | override/energy enable flags | device | native_gpu/unqualified | 020 | global contact overrides and retained energy paths are implemented; complete option and feature composition remains unqualified | validate compiled override payload and profile-specific energy paths | test_contact_override_020.py, test_public_enable_history_plugin_composition_019.py |
| REQ-ENBL-002 | forward/inverse enable flags | device | native_gpu/unqualified | 019 | prepared forward/inverse comparison and Euler/implicit discrete conversion implemented; full constrained/flex/extension compositions remain under qualification | validate supported integration profile and prepared-stage/query ownership | test_native_inverse_stages.py, test_native_prepared_api_019.py, test_sparse_implicit_public_019.py, test_sparse_implicit_query_transactions.py |
| REQ-ENBL-003 | sleep enable flag | device | native_gpu/unqualified | 017 | selected native island and sleep/wake fixtures pass; full integrator, reset, constraint and flex combinations remain unqualified | native sleep advance | test_scalable_017.py |
| REQ-ENBL-004 | exact-diagonal enable flag | device | native_gpu/unqualified | 014 | integrated dense/component paths compute exact diag(J M^-1 J^T) and cone-specific impedance; broad solver, contact, sleep and lifecycle compositions remain under qualification | assembly-only rows, exact mass solves, then preassembled-row solve | test_constraint_impedance_public_020.py, test_constraint_impedance_020.py |
| REQ-ENBL-005 | enable-bit count sentinel | none | not_implemented/unqualified | 005 | mjNENABLE is a count, not a flag | never admitted | — |
| REQ-PROF-001 | contact_free_euler_v1 baseline | device | native_gpu/gpu_qualified | baseline | unforced Euler baseline | accept contact-disabled Euler models | test_simulation.py |
| REQ-PROF-002 | forces/motor/transmission/fluid/passive/sensor Euler profiles | device | native_gpu/gpu_qualified | baseline | per-profile bounded subsets | accept listed profile/model pairs | test_forces.py, test_transmissions.py, test_fluid.py, test_passive.py, test_sensors.py, test_sensor_families_016.py, test_sensor_force_016.py, test_sensor_spatial_016.py, test_sensor_remaining_016.py, test_integrators_015.py |
| REQ-PROF-003 | joint_constraints_euler_v1 scalar stage | device | native_gpu/gpu_qualified | baseline | scalar-only stage; connect/weld explicitly not admitted | accept scalar joint content only | test_joint_constraints.py |
| REQ-PROF-004 | normal/friction contact Euler profiles | device | native_gpu/gpu_qualified | baseline | bounded normal/pyramidal sphere slices | accept listed contact content only | test_contact.py |
| REQ-PROF-005 | integrated_euler_v1 unified pipeline | device | native_gpu/gpu_qualified | 004 | nv<=32, pairs<=16, slots<=24, rows<=96 | accept validated integrated models | test_integrated_simulation.py, test_connect_weld.py, test_equality_activity.py, test_equality_qualification.py |
| REQ-PROF-006 | RK4 variants | device | native_gpu/gpu_qualified | baseline | qualified contact-free RK4 subsets; development integrated RK4 has separate, incomplete lifecycle qualification | accept listed RK4 profile/model pairs | test_runge_kutta.py |
| REQ-PROF-007 | implicitfast profile | device | native_gpu/gpu_qualified | baseline | constant damping + eligible free-body midpoint | accept eligible models | test_implicit.py |
| REQ-API-001 | host XML compilation/model editing/serialization | upstream_host | upstream_cpu_only/unqualified | 005 | upstream host route, not a GPU rewrite | host only | — |
| REQ-API-002 | host OpenGL visualization | upstream_host | upstream_cpu_only/unqualified | out-of-scope | OpenGL only; no Metal renderer | host only | — |
| REQ-API-003 | native public stage and dynamics wrappers | device | native_gpu/unqualified | 019 | package wrappers operate on MetalSimulation; similarly named mujoco bindings remain upstream CPU APIs, and wrapper coverage is limited to the explicit pinned entrypoint list in pinned_surface.json | explicit package counterpart map and per-entrypoint stage/query ownership | test_native_prepared_api_019.py, test_native_inverse_stages.py, test_mass_queries_019.py, test_native_api_sparse_019.py |
| REQ-REL-001 | wheel build/install qualification and coverage closure | none | not_implemented/unqualified | 020 | clean workspace-local wheel install, bundled shaders/assets, CLI, viewer startup, reconciled matrix | build/install in workspace-local env | — |
| REQ-PLUG-001 | mujoco.pid | device | native_gpu/unqualified | 019 | compiled PID adapter and stateful actuator integration implemented; selected filter/actearly and lifecycle fixtures pass, complete trajectory/composition qualification remains open | validate compiled bundled configuration and profile-specific native composition; reject unknown plugins | test_bundled_plugin_inventory.py, test_bundled_plugins_019.py |
| REQ-PLUG-002 | mujoco.elasticity.cable | device | native_gpu/unqualified | 019 | compiled cable lowering and native force producer integrated; complete force, trajectory and lifecycle qualification remains open | validate compiled bundled configuration and profile-specific native composition; reject unknown plugins | test_bundled_plugin_inventory.py, test_bundled_cable_019.py |
| REQ-PLUG-003 | mujoco.sensor.touch_grid | device | native_gpu/unqualified | 019 | compiled touch-grid lowering and native sensor producer integrated; complete contact/filter/stage/lifecycle qualification remains open | validate compiled bundled configuration and profile-specific native composition; reject unknown plugins | test_bundled_plugin_inventory.py, test_bundled_touch_grid_019.py |
| REQ-PLUG-004 | mujoco.sdf.bolt | device | native_gpu/unqualified | 019 | compiled native SDF provider and sampled default/custom query fixtures implemented; full collision/ray/lifecycle composition remains unqualified | validate compiled bundled configuration and profile-specific native composition; reject unknown plugins | test_bundled_plugin_inventory.py, test_bundled_plugins_019.py |
| REQ-PLUG-005 | mujoco.sdf.bowl | device | native_gpu/unqualified | 019 | compiled native SDF provider and sampled default/custom query fixtures implemented; full collision/ray/lifecycle composition remains unqualified | validate compiled bundled configuration and profile-specific native composition; reject unknown plugins | test_bundled_plugin_inventory.py, test_bundled_plugins_019.py |
| REQ-PLUG-006 | mujoco.sdf.gear | device | native_gpu/unqualified | 019 | compiled native SDF provider and sampled default/custom query fixtures implemented; full collision/ray/lifecycle composition remains unqualified | validate compiled bundled configuration and profile-specific native composition; reject unknown plugins | test_bundled_plugin_inventory.py, test_bundled_plugins_019.py |
| REQ-PLUG-007 | mujoco.sdf.nut | device | native_gpu/unqualified | 019 | compiled native SDF provider and sampled default/custom query fixtures implemented; full collision/ray/lifecycle composition remains unqualified | validate compiled bundled configuration and profile-specific native composition; reject unknown plugins | test_bundled_plugin_inventory.py, test_bundled_plugins_019.py |
| REQ-PLUG-008 | mujoco.sdf.torus | device | native_gpu/unqualified | 019 | compiled native SDF provider and sampled default/custom query fixtures implemented; full collision/ray/lifecycle composition remains unqualified | validate compiled bundled configuration and profile-specific native composition; reject unknown plugins | test_bundled_plugin_inventory.py, test_bundled_plugins_019.py |
<!-- COVERAGE_TABLE_END -->
