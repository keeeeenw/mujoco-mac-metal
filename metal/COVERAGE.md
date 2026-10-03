# Pinned-version support contract

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

<!-- COVERAGE_TABLE_BEGIN -->
| ID | Capability | Execution | Status | Owner | Limitation | Admission | Tests |
|---|---|---|---|---|---|---|---|
| REQ-MOD-001 | pinned MuJoCo 3.10.0 model lowering | host | lowered/cpu_oracle | baseline | requires MuJoCo 3.10.0; other versions rejected | accept 3.10.0, reject others | test_model.py |
| REQ-MOD-002 | bounded stepping profile validation | host | lowered/cpu_oracle | baseline | unknown profiles and unsupported combinations rejected | accept listed profiles, reject others | test_stepping.py |
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
| REQ-CON-001 | condim 1/3/4/6 with pyramidal/elliptic cones | device | native_gpu/gpu_qualified | baseline | other condim values rejected | accept 1/3/4/6, reject others | test_coupled_constraints.py |
| REQ-CON-002 | nine valid primitive contact pairs | device | native_gpu/gpu_qualified | baseline | plane-plane yields nothing; cylinder/ellipsoid/mesh/hfield/SDF pairs per owning milestones | accept the 9 primitive pairs | test_primitive_collision_qualification.py |
| REQ-CON-003 | anisotropic sliding/torsional/rolling friction | device | native_gpu/gpu_qualified | baseline | five-coefficient friction expansion with pair mixing | accept finite friction rows | test_coupled_constraints.py |
| REQ-INT-001 | Euler/RK4/implicitfast bounded profiles | device | native_gpu/gpu_qualified | baseline | per-profile supported combinations only | accept listed profile/integrator pairs | test_integration.py |
| REQ-INT-002 | full implicit integrator | none | not_implemented/unqualified | 015 | constant-damping implicitfast shortcut must not cover it | reject mjINT_IMPLICIT outside guards | — |
| REQ-JAC-001 | dense Jacobians | device | native_gpu/gpu_qualified | baseline | native pipeline is dense | accept dense | test_coupled_constraints.py |
| REQ-JAC-002 | sparse/auto Jacobians | none | not_implemented/unqualified | 017 | no sparse path; auto must not silently select one | reject sparse | — |
| REQ-SOL-001 | PGS/Newton selector mapping | device | native_gpu/gpu_qualified | baseline | accepted names run the native projected solver, documented as a mapping | accept PGS/Newton as mapped | test_coupled_constraints.py |
| REQ-SOL-002 | CG solver selection | device | native_gpu/unqualified | 014 | accepted as mapped to the native projected solver (same as Newton); selection name does not change execution | accept CG as mapped | test_solver_completion_014.py |
| REQ-EQ-001 | joint/connect/weld equalities with per-env activity | device | native_gpu/gpu_qualified | 004 | 1/3/6 rows, body/world/site forms incl. mocap anchors, torquescale, schema-2/3 activity state | accept joint/connect/weld body/site/world/mocap, reject others even if inactive | test_connect_weld.py, test_equality_activity.py, test_equality_qualification.py, test_mocap_state.py |
| REQ-EQ-002 | tendon equality | device | native_gpu/gpu_qualified | 008 | cubic tendon-length coupling as coupled rows (single or paired tendons) | tendon_constraint_rows equality branch | test_spatial_tendons_008.py |
| REQ-EQ-003 | flex equalities | none | not_implemented/unqualified | 018 | rejected at lowering | reject flex/flexvert/flexstrain | — |
| REQ-EQ-004 | removed distance equality | none | not_implemented/unqualified | 009 | removed upstream in MuJoCo 2.2.2; rejected | reject distance equalities | — |
| REQ-TEN-001 | fixed-joint tendons with spring/damping/armature | device | native_gpu/gpu_qualified | baseline | fixed joint paths only; no wrapping, limits, friction or equality | accept fixed joint tendons | test_tendons.py |
| REQ-TEN-002 | spatial tendons, wrapping, limits, friction, equality | device | native_gpu/gpu_qualified | 008 | pulley/site/sphere/cylinder paths; tendon limits, friction loss and tendon equality as coupled rows; site-only armature with Jdot bias (wrapped armature rejected upstream) | spatial kinematics + tendon_constraint_rows + armature_dots | test_spatial_tendons_008.py |
| REQ-TRN-001 | joint/jointinparent/tendon transmissions | device | native_gpu/gpu_qualified | baseline | stateless fixed/affine scalar scope | accept listed transmissions | test_transmissions.py |
| REQ-TRN-002 | slidercrank/site/body transmissions | device | native_gpu/gpu_qualified | 007 | rigid transmissions incl. ball/free gears; spatial-tendon paths stay in 008 | general kinematics + BODY adhesion from same-step candidates | test_actuators_007.py |
| REQ-TRN-003 | undefined transmission sentinel | none | not_implemented/unqualified | 005 | never admitted | reject undefined | — |
| REQ-DYN-001 | stateless (none) actuator dynamics | device | native_gpu/gpu_qualified | baseline | no activation state | accept dyntype none | test_actuation.py |
| REQ-DYN-002 | stateful activation dynamics (built-in) | device | native_gpu/gpu_qualified | 007 | built-in integrator/filter/filterexact/muscle/dcmotor only | act_dot switch + exact-slot advance, schema-4 state | test_actuators_007.py |
| REQ-DYN-003 | user-callback actuator dynamics | none | not_implemented/unqualified | 019 | arbitrary callbacks cannot execute as Metal kernels | reject dyntype user (019 extension contract) | — |
| REQ-GAIN-001 | fixed/affine gains | device | native_gpu/gpu_qualified | baseline | stateless scalar scope | accept fixed/affine | test_actuation.py |
| REQ-GAIN-002 | muscle/DC-motor gains (built-in) | device | native_gpu/gpu_qualified | 007 | built-in muscle/dcmotor only | muscle FLV + DC resistance/voltage paths | test_actuators_007.py |
| REQ-GAIN-003 | user-callback actuator gains | none | not_implemented/unqualified | 019 | arbitrary callbacks cannot execute as Metal kernels | reject gaintype user (019 extension contract) | — |
| REQ-BIAS-001 | none/affine biases | device | native_gpu/gpu_qualified | baseline | stateless scalar scope | accept none/affine | test_actuation.py |
| REQ-BIAS-002 | muscle/DC-motor biases (built-in) | device | native_gpu/gpu_qualified | 007 | built-in muscle passive + DC back-EMF/cogging/LuGre | bias switch + post-clamp DC mechanics | test_actuators_007.py |
| REQ-BIAS-003 | user-callback actuator biases | none | not_implemented/unqualified | 019 | arbitrary callbacks cannot execute as Metal kernels | reject biastype user (019 extension contract) | — |
| REQ-SENS-001 | current-state kinematic/clock/gyro/velocimeter queries | device | native_gpu/gpu_qualified | baseline | explicit query, not stored mj_step timing; no delay/noise/history | accept the 14 listed types | test_sensors.py |
| REQ-SENS-002 | remaining sensor families (016) | device | native_gpu/gpu_qualified | 016 | touch/accel/force/torque/magnetometer/range/tendon/actuator/limit/frame-acc/subtree/insidesite/geomdist/contact/energy; cutoff honored; delay/interval/noise/history rejected; SDF rays/geomdist + camprojection/tactile/plugin/user per 019; single contact slot; stored step-stage sample vs explicit query | accept the 31 listed types; reject the 4 deferred types | test_sensors.py, test_sensor_families_016.py, test_sensor_force_016.py, test_sensor_spatial_016.py, test_sensor_remaining_016.py |
| REQ-SENS-003 | deferred sensor extensions (019) | none | not_implemented/unqualified | 019 | camera-projection/tactile/plugin/user sensors rejected | reject deferred sensor types | — |
| REQ-STATE-001 | time/qpos/qvel/eq_active/mocap state ownership | device | native_gpu/gpu_qualified | 006 | persistent per-env poses/masks, schema-3 snapshots, keyframe reset, copy | accept matching snapshots, reject v1-into-neq and <v3-into-mocap | test_device_state.py, test_equality_activity.py, test_mocap_state.py |
| REQ-STATE-002 | control/applied-force state ownership | device | native_gpu/gpu_qualified | baseline | per-call held inputs validated before the device loop | accept finite host/device inputs | test_simulation.py |
| REQ-STATE-003 | mocap position/quaternion inputs | device | native_gpu/gpu_qualified | 006 | jointless world-child mocap bodies; per-env poses, keyframe reset, schema-3 snapshots | accept valid mocap, reject non-world-child | test_mocap_state.py |
| REQ-STATE-004 | actuator activation state | device | native_gpu/gpu_qualified | 007 | schema-4 act storage, reset/keyframe/restore/copy, exact-slot advance | na rows with actearly/actrange semantics | test_actuators_007.py |
| REQ-STATE-005 | warmstart state | device | native_gpu/unqualified | 014 | retained multipliers seed the next solve with cost-gated fallback; get/set/clear API; reset/keyframe clear; WARMSTART disable honored | warm lam retention + cost check | test_solver_completion_014.py |
| REQ-STATE-006 | history state | none | not_implemented/unqualified | 015 | no sensor/actuator history storage | reject history-dependent models | — |
| REQ-STATE-007 | userdata/plugin state | none | not_implemented/unqualified | 019 | no userdata/plugin state ownership | reject stateful plugins | — |
| REQ-STATE-008 | getState group selectors | none | not_implemented/unqualified | 019 | no mj_getState/mj_setState group API | not exposed | — |
| REQ-STATE-009 | state count sentinel | none | not_implemented/unqualified | 005 | mjNSTATE is a count, not selectable state | never admitted | — |
| REQ-DSBL-001 | honored disable flags | device | native_gpu/gpu_qualified | baseline | constraint/equality/frictionloss/limit/contact/spring/damper/gravity/clampctrl/warmstart/filterparent/actuation/refsafe/sensor/midphase/eulerdamp/autoreset honored per stage | accept listed flags | test_simulation.py |
| REQ-DSBL-002 | island/ccd disable flags | none | not_implemented/unqualified | 017 | no island/ccd execution paths | reject nativeccd/island/multiccd | — |
| REQ-DSBL-003 | disable-bit count sentinel | none | not_implemented/unqualified | 005 | mjNDISABLE is a count, not a flag | never admitted | — |
| REQ-ENBL-001 | override/energy enable flags | host | cpu_reference/cpu_oracle | baseline | override rejected for contacts; energy is metadata-only | reject override | test_coupled_constraints.py |
| REQ-ENBL-002 | forward/inverse enable flags | none | not_implemented/unqualified | 019 | no fwdinv/invdiscrete execution paths | reject fwdinv/invdiscrete | — |
| REQ-ENBL-003 | sleep enable flag | none | not_implemented/unqualified | 017 | sleep rejected; no sleeping execution path | reject sleep | — |
| REQ-ENBL-004 | exact-diagonal enable flag | none | not_implemented/unqualified | 014 | diagapprox used; diagexact not executed | ignore diagexact | — |
| REQ-ENBL-005 | enable-bit count sentinel | none | not_implemented/unqualified | 005 | mjNENABLE is a count, not a flag | never admitted | — |
| REQ-PROF-001 | contact_free_euler_v1 baseline | device | native_gpu/gpu_qualified | baseline | unforced Euler baseline | accept contact-disabled Euler models | test_simulation.py |
| REQ-PROF-002 | forces/motor/transmission/fluid/passive/sensor Euler profiles | device | native_gpu/gpu_qualified | baseline | per-profile bounded subsets | accept listed profile/model pairs | test_forces.py, test_transmissions.py, test_fluid.py, test_passive.py, test_sensors.py, test_sensor_families_016.py, test_sensor_force_016.py, test_sensor_spatial_016.py, test_sensor_remaining_016.py, test_integrators_015.py |
| REQ-PROF-003 | joint_constraints_euler_v1 scalar stage | device | native_gpu/gpu_qualified | baseline | scalar-only stage; connect/weld explicitly not admitted | accept scalar joint content only | test_joint_constraints.py |
| REQ-PROF-004 | normal/friction contact Euler profiles | device | native_gpu/gpu_qualified | baseline | bounded normal/pyramidal sphere slices | accept listed contact content only | test_contact.py |
| REQ-PROF-005 | integrated_euler_v1 unified pipeline | device | native_gpu/gpu_qualified | 004 | nv<=32, pairs<=16, slots<=24, rows<=96 | accept validated integrated models | test_integrated_simulation.py, test_connect_weld.py, test_equality_activity.py, test_equality_qualification.py |
| REQ-PROF-006 | RK4 variants | device | native_gpu/gpu_qualified | baseline | contact-free RK4 subsets; integrated RK4 deferred | accept listed RK4 profile/model pairs | test_runge_kutta.py |
| REQ-PROF-007 | implicitfast profile | device | native_gpu/gpu_qualified | baseline | constant damping + eligible free-body midpoint | accept eligible models | test_implicit.py |
| REQ-API-001 | host XML compilation/model editing/serialization | upstream_host | upstream_cpu_only/unqualified | 005 | upstream host route, not a GPU rewrite | host only | — |
| REQ-API-002 | host OpenGL visualization | upstream_host | upstream_cpu_only/unqualified | out-of-scope | OpenGL only; no Metal renderer | host only | — |
| REQ-REL-001 | wheel build/install qualification and coverage closure | none | not_implemented/unqualified | 020 | clean workspace-local wheel install, bundled shaders/assets, CLI, viewer startup, reconciled matrix | build/install in workspace-local env | — |
<!-- COVERAGE_TABLE_END -->
