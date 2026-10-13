# Development milestone status

The source backend is experimental. **5 of the 16 milestones 005–020 are accepted: 005 (classification/support
contract), 006 (state/model lifecycle), 007 (built-in actuation and rigid
transmissions) and 008 (spatial tendon mechanics under the admitted-model and
integrated Euler contracts), and 009 (rigid constraints and mixed composition).**
Milestones 010–020 remain open; acceptance of state ownership does not qualify
their remaining numerical feature compositions. Earlier bounded profiles and selected
native fixtures have passed independent checks.

| Milestone | Area | Current status |
|---|---|---|
| 005 | Pinned inventory and support contract | Qualified: complete classification, direct header/binding drift and compiled admission matrix; native physics qualification remains separate |
| 006 | State, model and lifecycle | Qualified: 170 host + 191 native checks, full crane/replay and bounded Mac viewer; shared-model and recompile boundaries documented |
| 007 | Actuators and transmissions | Qualified: 181 host + 198 native checks, full gripper/counterfactual, exact replay and working Mac viewer; extensions and remaining integrators retain later owners |
| 008 | Spatial tendons | Qualified: 113 host + 66 native checks; wrapping/transition/FD/armature/composition, full drawbridge/counterfactual, exact replay and working Mac viewer |
| 009 | Rigid constraints | Qualified: 54 host + 76 native checks; genuine both-cone shared coupling, retained global residuals/forces, full suspension/level hold, exact replay and reviewed Mac viewer |
| 010 | Analytic collision families | Primitive and additional convex-shape paths implemented; complete dispatcher/trajectory gate open |
| 011 | Mesh collisions | Native hull and contact paths implemented; full irregular/degenerate/composition gate open |
| 012 | Heightfields | Source-precision and pose/cache checks pass on selected fixtures; full terrain/integrator gate open |
| 013 | SDF collisions | Seven original bundled rigid-SDF oracle fixtures pass; complete pair, search and feature-composition gates open |
| 014 | Solvers | Selected Newton, warm-start and coupled regressions pass; full solver-option matrix open |
| 015 | Integrators and forces | Euler/RK4/implicit paths and derivatives implemented; full force/contact/material matrix open |
| 016 | Sensors and queries | Expanded native stages and queries implemented; complete timing/history/frame matrix open |
| 017 | Capacity, sparse execution and sleep | Model-derived storage and larger profiles implemented; all-boundary/native composition gate open |
| 018 | Deformables | Material, interpolation, attachment and collision paths implemented; complete geometry/integrator/lifecycle matrix open |
| 019 | APIs and extensions | Expanded package APIs and bundled adapters implemented; full native/host and extension composition gate open |
| 020 | Distribution and demos | Source demos and packaging support present; final clean-wheel/native/demo gates open |

## Evidence and limitations

Standalone POS/VEL/ACC publication now has actual source-stage, ownership,
history and rollback regressions. Additional nonzero contact/equality cases
qualified the component implicitfast midpoint correction and both monolithic
Newton/CG solver layouts, across dense/sparse Jacobians and island modes.
These are scoped corrections; the full milestone acceptance count remains
unchanged. See [API](API.md) and [qualification](QUALIFICATION.md).

Raw sensor and adapter diagnostics now participate in selected checkpoint,
reset, keyframe and environment-copy operations. Actual post-write and late
publication restore faults have passing rollback checks, including prepared
record and old-buffer ownership. The reviewed port passes a combined 209-check
native/mixed gate. This does not close the full lifecycle or API milestones.

Later coherent correction gates pass 1,100 affected checks and the full host
selection (3,008 passed, 99 skipped); the exact managed source port passes
495 checks. They cover complete late reset/keyframe/copy rollback, retained
sleeping pose words and awake lists, source callback cutoffs, invalid sensor
row ownership and typed actuator payload rejection before physical recovery.
Normal warning/AUTORESET controls remain tested. These overlapping scopes are
not a completed full native, wheel or demo qualification. Milestones 009–020
remain open.

Prepared potential/kinetic energy APIs, computed-stage metadata and current flex
queries have separate passing native correction gates. A 100-case source-order
matrix checks five integrators, automatic energy and sensors enabled/disabled,
interval/delay modes, raw history buffers and exact checkpoint replay. Batched
optional sensor payloads now cover every tendon/actuator index; their real
allocation and physical regression selection passes 35 checks. The complete host
selection at `51513bd7d` passes 2,996 checks with 99 skips; the skipped
native cases remain unqualified. See [producing revisions and exact remaining
scope](QUALIFICATION.md#prepared-energy-apis-and-sensor-storage). This does not
change the full milestone count.

The public selected-state copy contract passes 73 host/crosswalk checks and
94 affected native copy/lifecycle checks, including retained words, overlap,
failure atomicity, ordered worlds and exact replay. Split implicit integration
also passes its 25-case affected group after including solved constraint force.
See [exact scope and producing revisions](QUALIFICATION.md). These close
specific contracts; the full milestone count remains unchanged.

The latest complete local development host selection at `51513bd7d` passed
**2,996 tests, 99 skipped, 2,705 deselected in 102.38 seconds**;
it does not qualify the skipped native cases. The historical host publication
check passed **2,163 tests, 1,615 skipped** at its separately recorded revision;
see [producing revisions and scope](QUALIFICATION.md).

A local line-search correction passed 43 focused native checks and the original
alternate-cylinder heightfield trajectory/restore case without altered bounds
or solver budgets. The earlier complete terrain matrix finished with **34 passed,
6 failed** at its separately recorded producing revision.
Neither heightfields nor the full solver milestone is accepted from this result.

A subsequent native-prism publication correction passed seven affected native
checks and the original native-CCD cylinder trajectory/restore case. A separate
constraint-row consumer correction then passed 14 dense/component and alternate
CCD operator checks and the original native-CCD box trajectory. The complete
40-case terrain matrix at `10d9bf136` then finished with **38 passed, 2 failed**.
All former native-CCD cylinder/box failures pass; zero-margin alternate-CCD box
trajectories in both geom orders were correction gates at that revision and
subsequently pass after the retained-velocity correction described below. The four rejected-seed
warm-selection failures were subsequently corrected by retaining equality
regularizers and comparing original-operator paired objectives. All eight native
accepted/rejected boundary cases now pass; further groups passed 85 native
solver/island/line cases and 10 original solver/warm/slope compositions. See the qualification record for the
separate source revisions and their limits.

Selected source compositions passed 94 Newton regression checks (including the
original 300-step slope fixture), 69 contact-override checks and 11 retained
SDF arithmetic/source-trace checks. These are distinct regression groups, not a
complete backend suite. Earlier terrain/cache, flex and API results retain their
producing-composition limits; see [development coverage](DEVELOPMENT.md).

The remaining 26-file native API group at `baf24b80c` passed **117 tests,
280 deselected**, with no failures or skips. The earlier 16 failures were
corrected in RNE dispatch selectors, retained-state projection, masked force
publication and a stale site-feedback fixture. The complementary 35-file
native API group was repeated at `f59d1224f` and passed **304 tests** with no
failures, skips or warnings. The complete original finite-difference gate at
that revision separately passed **14 native tests**, including the rich
three-world inverse-force fixture and exact query rollback.
The later force/recovery fixes passed 39 and 58 affected checks; these combine
host contracts with real native physics and lifecycle execution.

Additional inverse-derivative residual correction passes 82 affected host/native
checks at `88aed2063`. The original material/integrator/equality native group
passes 90 cases at that revision. Stronger active 2D/3D elastic invariants exposed
false forces after rigid transformation; `15787fa29` corrects their retained
geometry, passing all four new physical cases and 90 affected host/native checks.
These demonstrate nonzero carrier reaction, momentum balance and replay, not
complete deformable coverage. Both alternate-CCD box terrain cases failed
at that source with their original trajectory bounds.

The retained-velocity bias correction at `ff659a24e` passes 71 affected
host/native checks and the complete CPU selection (2,455 passed, 99 skipped).
Both original alternate-CCD box trajectories now pass their unchanged gates;
the complete original 40-case terrain matrix now passes at that source,
including both CCD algorithms, both margins and geom orders, a separated peer,
forces, four-step trajectories and exact replay. No bounds or budgets changed.
Separately, a
34-case active simplex work/energy/stability gate at `15787fa29` covers two
stiffnesses, two timesteps and all four integrators with deformed free carriers.
Articulated COM publication and stable nonzero material damping corrections at
`abf0bde7c`/`ed74d3fa0` pass 16 native damping cases and 27 native
COM/spatial/recovery/invariant regressions. The original 90-case material/
integrator/equality matrix plus 32 independent native energy cases also pass
at `4d5497758` (122 passed, no failures or skips). Original force, derivative,
trajectory and replay bounds remain unchanged.
Stored energy copy/checkpoint ownership at `ab7c360a2` also passes the
complete 19-case native lifecycle file and 71 affected native transaction,
split-stage, delay and recovery cases. These preserve nonzero source energy,
selected worlds, malformed rejection, legacy compatibility and exact replay.
These are source-specific results; full heightfield/deformable composition,
final wheel/demo checks and milestones 009–020 remain open.

Current checkpoint and owned-keyframe regression fixtures at `fe644d658` /
`70136b887` pass the complete 19-case native actuator/device/mocap/equality
selection. They retain legacy activation migration and strengthen current
velocity/precision replay and corrupt-payload rejection. The preceding full
native discovery stopped after 115 passes on a stale checkpoint-version test;
it is not a completed backend suite.

The original rigid-SDF deadline and mesh/bowl/torus failures are historical
evidence. The seven original bundled rigid-SDF fixtures now pass in the
117-test group, including strict five-provider force/trajectory/replay,
mesh rows and query reuse. This closes those fixtures, not the complete SDF
search, pair, lifecycle or extension composition scope.

[Qualification](QUALIFICATION.md) separates publication checks from historical
native evidence. [Coverage](COVERAGE.md) describes admitted paths and restrictions.
No full compatibility, current-wheel qualification or new performance claim is
made. MuJoCo's dynamics and numerical methods are upstream work; this project
implements those methods for the optional Metal backend.

## Sequential milestone qualification

Milestone 005 passes its complete targeted infrastructure gate: **1,292 passed
in 37.04 seconds**, with no skips. The crosswalk covers 529 enum members,
813 distinct binding/header runtime fields, 554 distinct binding/header functions,
69 Python classes, 29 option fields, 79 package wrappers and 1,167 combination
rows. Direct installed-header checks found and classified three previously omitted
C-only SDF functions. No full repository regression was required for this
infrastructure gate. Next is Milestone 006: complete state/model lifecycle and
its magnetic-crane demonstration.

Milestone 006 passes 170 distinct host and 191 native checks, including full
crane release/attached counterfactual, exact checkpoint replay and mocap-driven
flex lifecycle. The refreshed crane GIF shows actual native/CPU frames. Its
main-thread GLFW viewer advances native physics and closes under a wall-time
limit; no interpreter or global environment changes were needed. Milestone 007
actuator qualification and its full gripper demo are next.

## Sequential actuation qualification — 2026-10-13 UTC

Milestone 007 passes 181 host and 198 native checks without failures or skips:
nonzero rigid transmission frames, moving slider-cranks, optional DC states,
aggregate limits/gravity routing, and changing adhesion forces under contact.
The source corrections preserve DC resistance, axis derivatives, rotated
reference-site conventions and retained geometry. The crane replay still passes.

The original gripper and always-open episodes each complete 22,000 native steps.
Maximum native penetration is 0.880637 mm against the unchanged 1 mm bound,
both payloads reach their receivers, and full checkpoint replay is exact. The
new comparison GIF and GLFW viewer were visually reviewed. The demo guide
records remaining long-horizon rotational differences.

The following spatial-tendon qualification closes Milestone 008. These
actuation results alone do not close
all collision/solver options, other integrators, extensions, distribution,
broad hardware or performance qualification.

## Sequential spatial tendon qualification — 2026-10-13 UTC

Milestone 008 passes 113 host and 66 native checks without failures or skips.
Compensated native site geometry consumes retained FK words, including each
joint's ordered anchor/axis. The complete armature family tests caught and
corrected an earlier-joint frame error before acceptance. Position-cache ABI,
physical storage and overflow guards include the new residual planes.

Independent finite differences cover multiple pulleys, ball/free/hinge paths,
sphere/cylinder inside and outside routes, both side selections and actual
wrap transitions. Limits, friction, polynomial equalities, passive/actuated
forces, 70-DOF assembly, masked worlds, reset and full replay pass.

The unchanged drawbridge completes 1,400 primary and 1,400 constant-release
native steps. Full checkpoint replay is exact during release and re-haul.
The fresh comparison GIF and Mac GLFW viewer were reviewed. Default soft
contacts briefly compress 18.31 mm native versus 18.57 mm CPU; the original
checks pass, but this does not qualify a 1 mm impact bound. The payload slides
off during release and the deck re-raises empty. See the [demo guide](examples/cable_drawbridge.md).

Milestone 009 is next: complete rigid-constraint families and genuine mixed
coupling under both contact cones, followed by the full suspension platform.
Solver options, other integrators, extensions, distribution and broad hardware
retain their later gates. No performance claim follows from these checks.

## Sequential rigid constraint qualification — 2026-10-13 UTC

Milestone 009 passes 54 host and 76 native checks without failures or skips.
A shared articulated fixture demonstrates all ten nonzero cross-block couplings
between equalities, dry friction, ball/tendon limits and contacts under both
cones. Independent mass/Jacobian/force references, retained global residuals,
soft redundancy, changing contacts, failed peers and full replay pass.

The original 4,800-step tilt and 4,800-step level episodes pass unchanged bounds.
Native/CPU maximum linear-coordinate difference is 0.215 mm; strut-axis error
is 0.04819 rad against the original 0.08 rad limit. Full checkpoint replay is
exact at two loaded-motion points. Both nominal stop overruns are zero, and the
strut/bob stay clear of the platform and cargo. Cargo contacts remain compliant;
this does not claim all contact penetration is below 1 mm. The fresh comparison
GIF and bounded native Mac viewer were reviewed. See the [demo guide](examples/suspension_platform.md).

A general source-rounded COM/CRBA/RNE correction resolves the preceding failed
axis trajectory without changing the model or weakening its gates. Its separate
31-check affected group covers native operator accuracy, dense/component storage,
retained position/velocity ownership, masks and explicit queries. Counts overlap
and are not additive. No performance or unrestricted binary64 claim follows.
Prior qualification stays attached to its producing revision. Milestone 010 is
next; complete collision, solver, integrator and installed-wheel scope remains open.
