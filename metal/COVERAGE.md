# Pinned-version support contract

This table is generated from `mujoco_metal/registry.py` (`REQUIREMENTS`) by
`python -m mujoco_metal coverage`. Do not edit the table by hand: the drift
test `test_coverage_contract.py::test_docs_match_generated_table` fails unless
this file contains the generated output verbatim between the markers. Change
the contract in `registry.py`, regenerate, and paste the result here.

```sh
./run --cpu python -m mujoco_metal coverage
```

Owner milestones `006`–`020` are defined in the full-coverage roadmap; each
owns the implementation and qualification of its rows. `baseline`/`004` rows
are implemented; `out-of-scope` rows are explicit project exclusions (RL/PPO,
deployment targets, broad hardware testing, extra MuJoCo versions, Metal
renderer), not future physics work.

<!-- COVERAGE_TABLE_BEGIN -->
| ID | Capability | Execution | Status | Owner | Admission | Tests |
|---|---|---|---|---|---|---|
| REQ-MOD-001 | pinned MuJoCo 3.10.0 model lowering | host | lowered/cpu_oracle | baseline | accept 3.10.0, reject others | test_model.py |
| REQ-MOD-002 | bounded stepping profile validation | host | lowered/cpu_oracle | baseline | accept listed profiles, reject others | test_stepping.py |
| REQ-JNT-001 | hinge/slide/ball/free kinematics | device | native_gpu/gpu_qualified | baseline | accept all four joint types | test_model.py |
| REQ-JNT-002 | scalar hinge/slide joint limits | device | native_gpu/gpu_qualified | baseline | accept limited hinge/slide, reject limited ball | test_coupled_constraints.py |
| REQ-JNT-003 | ball joint limits | none | not_implemented/unqualified | 009 | reject limited ball joints | — |
| REQ-GEO-001 | plane/sphere/capsule/box collision | device | native_gpu/gpu_qualified | baseline | accept the four primitives, reject others | test_primitive_collision_qualification.py |
| REQ-GEO-002 | cylinder/ellipsoid collision | none | not_implemented/unqualified | 010 | reject cylinders and ellipsoids | — |
| REQ-GEO-003 | convex mesh collision | none | not_implemented/unqualified | 011 | reject meshes | — |
| REQ-GEO-004 | heightfield collision | none | not_implemented/unqualified | 012 | reject heightfields | — |
| REQ-GEO-005 | SDF collision | none | not_implemented/unqualified | 013 | reject SDFs | — |
| REQ-GEO-006 | visual-only geometry decorations | upstream_host | upstream_cpu_only/unqualified | out-of-scope | never admitted as collision geometry | — |
| REQ-GEO-007 | mjtGeom count sentinel | none | not_implemented/unqualified | 005 | never admitted | — |
| REQ-CON-001 | condim 1/3/4/6 with pyramidal/elliptic cones | device | native_gpu/gpu_qualified | baseline | accept 1/3/4/6, reject others | test_coupled_constraints.py |
| REQ-CON-002 | nine valid primitive contact pairs | device | native_gpu/gpu_qualified | baseline | accept the 9 primitive pairs | test_primitive_collision_qualification.py |
| REQ-CON-003 | anisotropic sliding/torsional/rolling friction | device | native_gpu/gpu_qualified | baseline | accept finite friction rows | test_coupled_constraints.py |
| REQ-INT-001 | Euler/RK4/implicitfast bounded profiles | device | native_gpu/gpu_qualified | baseline | accept listed profile/integrator pairs | test_integration.py |
| REQ-INT-002 | full implicit integrator | none | not_implemented/unqualified | 015 | reject mjINT_IMPLICIT outside guards | — |
| REQ-JAC-001 | dense Jacobians | device | native_gpu/gpu_qualified | baseline | accept dense | test_coupled_constraints.py |
| REQ-JAC-002 | sparse/auto Jacobians | none | not_implemented/unqualified | 017 | reject sparse | — |
| REQ-SOL-001 | PGS/Newton selector mapping | device | native_gpu/gpu_qualified | baseline | accept PGS/Newton as mapped | test_coupled_constraints.py |
| REQ-SOL-002 | CG solver selection | none | not_implemented/unqualified | 014 | reject CG | — |
| REQ-EQ-001 | joint/connect/weld equalities with per-env activity | device | native_gpu/gpu_qualified | 004 | accept joint/connect/weld body/site/world/mocap, reject others even if inactive | test_connect_weld.py, test_equality_activity.py, test_equality_qualification.py, test_mocap_state.py |
| REQ-EQ-002 | tendon equality | none | not_implemented/unqualified | 009 | reject tendon equalities | — |
| REQ-EQ-003 | flex equalities | none | not_implemented/unqualified | 018 | reject flex/flexvert/flexstrain | — |
| REQ-EQ-004 | removed distance equality | none | not_implemented/unqualified | 009 | reject distance equalities | — |
| REQ-TEN-001 | fixed-joint tendons with spring/damping/armature | device | native_gpu/gpu_qualified | baseline | accept fixed joint tendons | test_tendons.py |
| REQ-TEN-002 | spatial tendons, wrapping, limits, friction, equality | none | not_implemented/unqualified | 008 | reject spatial/wrapped/limited tendons | — |
| REQ-TRN-001 | joint/jointinparent/tendon transmissions | device | native_gpu/gpu_qualified | baseline | accept listed transmissions | test_transmissions.py |
| REQ-TRN-002 | slidercrank/site/body transmissions | device | native_gpu/gpu_qualified | 007 | general kinematics + BODY adhesion from same-step candidates | test_actuators_007.py |
| REQ-TRN-003 | undefined transmission sentinel | none | not_implemented/unqualified | 005 | reject undefined | — |
| REQ-DYN-001 | stateless (none) actuator dynamics | device | native_gpu/gpu_qualified | baseline | accept dyntype none | test_actuation.py |
| REQ-DYN-002 | stateful activation dynamics | device | native_gpu/gpu_qualified | 007 | act_dot switch + exact-slot advance, schema-4 state | test_actuators_007.py |
| REQ-GAIN-001 | fixed/affine gains | device | native_gpu/gpu_qualified | baseline | accept fixed/affine | test_actuation.py |
| REQ-GAIN-002 | muscle/DC-motor/user gains | device | native_gpu/gpu_qualified | 007 | muscle FLV + DC resistance/voltage paths | test_actuators_007.py |
| REQ-BIAS-001 | none/affine biases | device | native_gpu/gpu_qualified | baseline | accept none/affine | test_actuation.py |
| REQ-BIAS-002 | muscle/DC-motor/user biases | device | native_gpu/gpu_qualified | 007 | bias switch + post-clamp DC mechanics | test_actuators_007.py |
| REQ-SENS-001 | current-state kinematic/clock/gyro/velocimeter queries | device | native_gpu/gpu_qualified | baseline | accept the 14 listed types | test_sensors.py |
| REQ-SENS-002 | remaining sensor families | none | not_implemented/unqualified | 016 | reject unlisted sensor types | — |
| REQ-STATE-001 | time/qpos/qvel/eq_active/mocap state ownership | device | native_gpu/gpu_qualified | 006 | accept matching snapshots, reject v1-into-neq and <v3-into-mocap | test_device_state.py, test_equality_activity.py, test_mocap_state.py |
| REQ-STATE-002 | control/applied-force state ownership | device | native_gpu/gpu_qualified | baseline | accept finite host/device inputs | test_simulation.py |
| REQ-STATE-003 | mocap position/quaternion inputs | device | native_gpu/gpu_qualified | 006 | accept valid mocap, reject non-world-child | test_mocap_state.py |
| REQ-STATE-004 | actuator activation state | device | native_gpu/gpu_qualified | 007 | na rows with actearly/actrange semantics | test_actuators_007.py |
| REQ-STATE-005 | warmstart state | none | not_implemented/unqualified | 014 | ignore warmstart content | — |
| REQ-STATE-006 | history state | none | not_implemented/unqualified | 015 | reject history-dependent models | — |
| REQ-STATE-007 | userdata/plugin state | none | not_implemented/unqualified | 019 | reject stateful plugins | — |
| REQ-STATE-008 | getState group selectors | none | not_implemented/unqualified | 019 | not exposed | — |
| REQ-STATE-009 | state count sentinel | none | not_implemented/unqualified | 005 | never admitted | — |
| REQ-DSBL-001 | honored disable flags | device | native_gpu/gpu_qualified | baseline | accept listed flags | test_simulation.py |
| REQ-DSBL-002 | island/ccd disable flags | none | not_implemented/unqualified | 017 | reject nativeccd/island/multiccd | — |
| REQ-DSBL-003 | disable-bit count sentinel | none | not_implemented/unqualified | 005 | never admitted | — |
| REQ-ENBL-001 | override/energy enable flags | host | cpu_reference/cpu_oracle | baseline | reject override | test_coupled_constraints.py |
| REQ-ENBL-002 | forward/inverse enable flags | none | not_implemented/unqualified | 019 | reject fwdinv/invdiscrete | — |
| REQ-ENBL-003 | sleep enable flag | none | not_implemented/unqualified | 017 | reject sleep | — |
| REQ-ENBL-004 | exact-diagonal enable flag | none | not_implemented/unqualified | 014 | ignore diagexact | — |
| REQ-ENBL-005 | enable-bit count sentinel | none | not_implemented/unqualified | 005 | never admitted | — |
| REQ-PROF-001 | contact_free_euler_v1 baseline | device | native_gpu/gpu_qualified | baseline | accept contact-disabled Euler models | test_simulation.py |
| REQ-PROF-002 | forces/motor/transmission/fluid/passive/sensor Euler profiles | device | native_gpu/gpu_qualified | baseline | accept listed profile/model pairs | test_forces.py, test_transmissions.py, test_fluid.py, test_passive.py, test_sensors.py |
| REQ-PROF-003 | joint_constraints_euler_v1 scalar stage | device | native_gpu/gpu_qualified | baseline | accept scalar joint content only | test_joint_constraints.py |
| REQ-PROF-004 | normal/friction contact Euler profiles | device | native_gpu/gpu_qualified | baseline | accept listed contact content only | test_contact.py |
| REQ-PROF-005 | integrated_euler_v1 unified pipeline | device | native_gpu/gpu_qualified | 004 | accept validated integrated models | test_integrated_simulation.py, test_connect_weld.py, test_equality_activity.py, test_equality_qualification.py |
| REQ-PROF-006 | RK4 variants | device | native_gpu/gpu_qualified | baseline | accept listed RK4 profile/model pairs | test_runge_kutta.py |
| REQ-PROF-007 | implicitfast profile | device | native_gpu/gpu_qualified | baseline | accept eligible models | test_implicit.py |
| REQ-API-001 | host XML compilation/model editing/serialization | upstream_host | upstream_cpu_only/unqualified | 005 | host only | — |
| REQ-API-002 | host OpenGL visualization | upstream_host | upstream_cpu_only/unqualified | out-of-scope | host only | — |
| REQ-REL-001 | wheel build/install qualification and coverage closure | none | not_implemented/unqualified | 020 | build/install in workspace-local env | — |
<!-- COVERAGE_TABLE_END -->
