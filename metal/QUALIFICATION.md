# Development qualification — updated 2026-10-13 UTC

Historical entries retain their producing revisions and original open gates.
Use the latest sequential entries and [milestone status](STATUS.md) for current
acceptance; overlapping counts are not additive.

## Complete sensor and extension correction composition

The coherent source at `203b59ca5` passes the complete **39-file affected
selection: 1,100 checks in 888.90 seconds**, with no failures or skips and MPS
CPU fallback disabled. The full host selection at that source passes
**3,008 tests, 99 skipped, 3,294 deselected in 106.96 seconds** with MPS disabled.
After porting the exact commits, verifying all 599 producing file hashes and
checking actual managed module/shader imports, the combined **13-file managed
selection passes 495 checks in 167.28 seconds**, including the original coupled
constraint file. These overlapping counts must not be added together or called
an unrestricted GPU test count.

The reviewable corrections are `7d201f2ba` (complete reset/keyframe/copy
rollback), `ad3dc31e4` (post-callback cutoff and source awake-policy checks),
`3416b5fc8` (retained sleeping pose/list copies), `ece21dc59` (typed actuator
payload rejection) and `203b59ca5` (complete failed-world sensor ownership).
Actual baseline runs reproduced each affected ownership/failure path; fixture
errors involving tuple checkpoint keys, source sleep behavior and compiler
admission were corrected independently of production code. No original
physics tolerance, iteration budget or source recovery behavior was relaxed.

New cases compare source callback cutoffs, selected samples/state, subsequent
CPU motion, actual post-write failures, initialized/lazy bindings, reusable
storage, derivative-query ownership, prepared-record coherence and exact replay.
Typed actuator checks cover NaN/Infinity in force and activation derivatives
across five profiles and ordinary/split/prepared forward routes, with independent
healthy-world CPU references. Separate finite-large-force controls retain
configured BADQACC warning and AUTORESET behavior. Extension rejection remains
an explicit backend contract, not arbitrary upstream callback conversion.

The preceding complete backend discovery at `f65806d51` stopped after
**1,121 passes and one failure in 2,075.11 seconds** on an obsolete empty-pair
header assertion. `ec23523d0` distinguishes logical offsets from the world-mask
pointer and entries, and checks actual empty-contact native acceleration and
zero constraint force against CPU MuJoCo. Its focused gate passes **4 checks in
1.70 seconds**. The failed discovery did not run the rest of the repository.

Complete repository-native qualification, remaining API contracts, clean-wheel
installation and full-timeline creative demo gates remain open. No performance
claim or full milestone acceptance follows from these correction gates.

## Prepared energy APIs and sensor storage

Runtime revision `2f0c6ded4`, with tests through `7145fe3d9`, passes the complete
eight-file affected selection: **192 checks in 113.76 seconds, comprising
141 native checks and 51 host checks**, with no failures or skips and MPS CPU
fallback disabled. The complete host selection at the same runtime passes
**2,983 tests, 99 skipped, 2,544 deselected in 113.64 seconds**. Skips and
deselections do not qualify native physics.

Actual public `mj_energyPos`/`mj_energyVel` calls match independent pinned
MuJoCo with automatic energy enabled or disabled. Tests cover dense/component
mass, five integrators, real scalar/free/ball joints, nonlinear joint springs,
fixed/spatial tendon springs, active one-dimensional flex and zero-DOF models.
Prepared calls forbid geometry/mass redispatch and CPU physics. Selected rows,
other-column preservation, owned outputs, injected publication failures,
copy/reset/checkpoint replay and compatible model rebuilds are checked.

New correction gates exposed typed flex-context access, malformed enabled
checkpoint acceptance, disabled-forward clearing, POS kinetic sensors reading
zero velocity, and current queries replacing stored samples. Corrected query
rollback is exercised after actual native computation as well as on success.
Sensor-driven buffers are charged in host allocation admission.

Timed sensors cover automatic energy on/off across all five integrators, cutoff,
interval/delay storage and exact checkpoint replay. A failing-before RK4 case
showed potential-energy error up to `0.00731576`: delayed samples require the
last RK substage geometry with restored starting inputs. The correction keeps
the original `rtol=2e-5`, `atol=2e-6` energy bounds and exact replay assertions.

This is scoped qualification. Complete sleeping/failed-peer, gravity/spring
disable, pulley/armature, retained-precision and lifecycle boundary combinations,
the full native suite, clean-wheel execution and final demos remain open.
Milestones 005–020 remain unaccepted.

The later prepared-record owner binding correction at `4552436df`, with tests
through `aafda0e3f`, passes a separate **192-case** energy/stage/state-copy gate
in **96.13 seconds**: **152 native checks and 40 host checks**, no failures or
skips. State-bound tensors cannot be replaced while an old record is reused;
explicitly prepared external inputs remain valid. Real tensor protocol controls
cover position, velocity and both mocap fields. The complete host selection
passes **2,988 tests, 99 skipped, 2,565 deselected in 111.77 seconds**.

Additional native energy cases at this runtime pass the complete direct-energy
file: **108 passed in 19.82 seconds**, without failures or skips. They cover
gravity/spring disables, nonlinear scalar and fixed-tendon springs, fixed and
spatial pulley armature, true sleeping trees/static gravity, failed/stale peers,
lazy selected checkpoint restoration under inference mode, all five integrators
with zero-DOF energy sensors, and rejection of MPS inputs lacking mutation
counters. Sleeping fixed/spatial/static tendon springs have a separate nonzero
CPU counterfactual (`0.51` potential energy); the unchanged native bounds are
`rtol=2e-5`, `atol=2e-6`. Sleep cases use the admitted scalable Euler and RK4
profiles; historical small dense Euler rejects sleep by design.

Computed-energy metadata is implemented at `7c6ac50d3`. Its final affected gate
passes **135 checks in 22.06 seconds: 117 native cases and 18 host checks**.
Owned per-world flags participate in selected reset/copy/restore, compatible
model rebuilds and publication rollback. Malformed checkpoint flags reject
before mutation; legacy checkpoints initialize them to zero. The earlier
eight-file history/composition gate passes **241 checks in 117.16 seconds**.
The complete host selection passes **2,988 tests, 99 skipped, 2,588 deselected
in 104.79 seconds**. These overlapping selections are separate evidence and
must not be added together.

Prepared sensor publication is implemented in `b943aa49e`, following cached
mass admission correction `0af0fc76e`. Its complete affected selection passes
**574 checks in 322.01 seconds**, without failures or skips. The complete host
selection passes **2,996 tests, 99 skipped, 2,863 deselected in 102.39 seconds**.
The stage tests compare actual calls against pinned MuJoCo, with CPU physics
forbidden in native execution. They cover mixed address sentinels, dense/component
profiles, populated delayed/interval histories, energy metadata, real sleeping
trees, selected healthy/failed worlds, input mutation/replacement, reset/restore,
owned outputs, admitted storage and actual evaluation/publication/plugin faults.

A further contact/equality composition gate exposed missing component
implicitfast midpoint initialization: orientation differed by `0.00020559`
after two updates while velocity matched the source. A diagnostic reproduced
the same defect without sensor publication. The correction is `e6c0acdce`.
Expanded island-disabled cases then exposed monolithic Newton/CG stopping
scaled by an empty island inertia; `73e2cf55a` restores the pinned global scale.

The expanded native source comparison passes all **120** force-composition cases:
three standalone sensor stages, five integrated configurations, two solver
families, dense/sparse Jacobians and islands enabled/disabled. It checks nonzero
contact/equality reactions and force/torque readings, subsequent CPU trajectories,
actual midpoint eligibility, unchanged history and exact checkpoint replay.
The same selection also passes the complete 13-case midpoint file before a new
CPU protocol fixture failed for missing batch metadata; correcting that fixture
requires no production change. The subsequent complete admission/allocation
selection passes **24 checks in 10.31 seconds**, including actual dense/component
program tensors and reusable mask execution. Earlier midpoint/force selection:
**28 passed in 19.18 seconds**. These overlapping results must not be summed.

No tolerances or iteration limits were relaxed. The broader backend regression,
remaining public stage contracts, final clean-wheel and creative demo gates
remain open. These results do not establish performance or complete milestones.

The complete backend discovery at `35bb85072` stopped after **1,001 passes**
on a contact-sensor fixture supplying an obsolete 42-word record layout.
`dcf382464` uses the canonical production layout and rejects malformed views
before dispatch. The original complete 20-case query matrix passes unchanged
physics bounds. A subsequent RNE fixture read preserved previous-step scratch
after a current query; `f669bc2d7` instead compares the owned public query with
the independent CPU oracle at full retained inputs and asserts the old buffers
stay unchanged. The complete seven-file selection passes **288 checks in
407.62 seconds**; the explicit native-only RNE guard passes **1 in 48.18 seconds**.
Full host checks pass **3,000 tests, 99 skipped, 2,974 deselected in 102.80 seconds**.

Raw sensor and adapter diagnostic lifecycle ownership is corrected in
`3a4bf4a76`. A stronger test reproduced partial physical restoration after
a post-write exception, with position error up to `0.1157316`. The correction
adds admitted rollback across owned state, caches, storage bindings and adapter
payloads, without running physics for recovery. **34 new native checks pass
in 19.86 seconds**, including actual post-write/publication faults, exact
restoration, old buffer identity and valid/stale prepared-record classification.
The full ten-file affected selection passes **326 in 215.98 seconds**, with no
failures or skips. Full host checks pass **3,000 tests, 99 skipped, 3,004
deselected in 107.23 seconds**. After porting the exact source, a combined
contact/RNE/sensor/lifecycle gate passes **209 checks in 178.66 seconds**.

These scopes overlap and must not be summed. The complete backend discovery
was a failed run, not full qualification; all original milestone, API, wheel
and full-timeline creative demo acceptance remains required.

Current flex-energy queries are corrected at `ba8c2097c`. Real tensor mutation
previously left flex geometry stale, producing a potential-energy error of
`0.704` in a retained fixture. Eight new native cases cover dense/component
mass, changed/unchanged inputs and success/injected failure after real native
evaluation. They verify stored energy/flags/history, borrowed mass/geometry and
record validity survive inspection, then compare three subsequent CPU/native
steps and exact checkpoint replay. The full nine-file affected selection passes
**249 checks in 119.26 seconds**, with no failures or skips; its complete host
selection passes **2,988 tests, 99 skipped, 2,596 deselected in 103.91 seconds**.

These scoped gates do not establish all retained-precision/lifecycle
compositions, a full native suite or milestone completion. The earlier
192-case selections and the later metadata/query selections are distinct,
overlapping evidence and must not be counted twice.

Lazy POS/VEL energy ordering is corrected in `ef814cfa1`. A new 100-case
matrix compares full forward and five consecutive updates against pinned
MuJoCo across five integrators, automatic energy enabled/disabled, sensors
enabled/disabled and direct/history/interval/delay modes. It checks diagnostic
values and flags, stored samples, complete raw history buffers and exact cursor
and checkpoint replay. An earlier ten-file candidate selection passed **350
checks in 180.12 seconds**. The subsequent optional-owner recovery correction
passes the complete stage-order and state-check selection: **140 checks in
65.27 seconds**, with no failures or skips.

The wider gate also exposed an undersized optional sensor payload: later worlds
could read beyond its backing, occasionally returning NaN tendon samples.
`51513bd7d` sizes it from admitted batched tendon/actuator counts. A deterministic
binding check, the original 50-step physical/reset/replay test, exact owned
memory admission and zero-width/tendon-only/actuator-only boundaries pass
**35 checks in 5.30 seconds**. The complete host suite at this final source
passes **2,996 tests, 99 skipped, 2,705 deselected in 102.38 seconds**. The
350-, 140- and 35-case selections overlap and are not cumulative counts. These
results still do not qualify the full native suite or close milestones 005–020.

## Rigid convex oracle input precision

The complete public rigid-convex regression file passes **20 checks in 49.24
seconds: 14 native cases and six host checks**, with fallback disabled and no
failures or skips. A symmetric mesh–box test had compared a float32 CPU pose
with a native constructor pose that now retains float64 residual words. Pinned
CPU itself chooses an X-normal manifold at the original pose and a Z-normal
manifold after projection. The test now verifies represented initialization
exactly and exercises both inputs in both geometry orders, with CPU physics
forbidden during native contact assembly.

The pair-matrix and reset/trajectory protocols also use matching declared
inputs on both systems. Original contact count, position (`2e-4`), distance
(`2e-5`), normal/Jacobian, reference/regularization, friction, force and replay
bounds remain unchanged. No production physics or solver budget was changed.
The preceding full native discovery stopped at **289 passes, one failure and
3,063 deselected in 1,198.32 seconds**; it is not a complete native suite or a
milestone acceptance result.

## Public selected state copy

The batched `mj_copyState` entrypoint copies selected pinned state components
without evaluating physics. Its complete host/crosswalk group passes **73 tests
in 2.37 seconds**. The complete native copy, transaction, device-state, energy
and split-step selection passes **94 tests, 19 deselected in 78.44 seconds**,
with no failures or skips and CPU fallback disabled.

The full host suite at `50d3bdad0` passes **2,976 tests, 87 skipped,
2,470 deselected in 111.99 seconds**, with MPS availability disabled only in
that host-test process. Skipped/deselected tests do not establish native physics
qualification.

The copy file includes all individual bits and composite masks, actual contact
and connect-equality forces, activation and compiled history, retained position/
velocity, changed/replaced source tensors with unavailable mutation counters,
ordered selection, different batch sizes, overlapping rows, prepared-context
invalidation and exact replay. An injected failure after real writes restores
storage and coherent records. Literal nonfinite/signed-zero values and invalid
mocap normalization are copied without setter admission; they do not constitute
valid stepping inputs. Immutable inference destination storage is rejected
before any writes, following a failing-before rollback regression. Empty
compiled plugin state is covered; arbitrary plugin contracts are not implied.

Initial native admission also caught the distinction between unindexed `mps`
metadata and actual `mps:0` tensor devices. Admission now uses the actual indexed
device. These checks retain original physics bounds and independent pinned CPU
oracles; they qualify this selected-copy contract, not `mj_copyData`, whole
milestones 006/019 or final distribution qualification.

## Split-step integration with solved constraints

Split-step integration now includes the solved constraint force in its total
force RHS, after prepared acceleration sensors and forward/inverse diagnostics
consume their stage results. The complete affected split-stage, inverse,
observational-query and component-mocap selection passes **25 native tests,
26 deselected in 11.75 seconds**, with CPU fallback disabled.

Six new cases compare three consecutive split updates and exact checkpoint
replay against pinned MuJoCo, with active contact, connect equality, moving
mocap, delayed controls/sensors and retained state. Euler, implicit and
implicitfast cover dense and sparse Jacobian settings; both implicit profiles
use component mass independently of Jacobian layout. Original `rtol=4e-5`,
`atol=6e-6` physics bounds and exact replay remain unchanged. Before correction,
Euler passed while both sparse implicit cases exposed missing constraint force
in integration despite correct forward acceleration. The initial dense-Jacobian
implicit fixtures also contained an incorrect mass-layout assumption, corrected
separately. This qualifies the defect and affected selection, not a complete
milestone or whole-suite pass.

## Component assembly with mocap

Component assembly now passes prepared poses to smooth dynamics without
also passing the mocap inputs already represented by those poses. The complete
component-mocap, observational-query and cached-solver selection passes
**65 native tests in 52.39 seconds**, with fallback disabled. New scalable
Euler and component implicitfast cases exercise active plane contact, a real
connect equality, changed mocap poses, retained positions/velocities and a
pending split-step record. Current acceleration and constraint forces match
pinned MuJoCo within the unchanged `rtol=4e-5`, `atol=6e-6` bounds; querying
preserves complete checkpoint bytes. No CPU physics is called by native
assembly. This closes the supplied-pose/mocap composition defect, not an entire
milestone or full-suite qualification.

## Cached solver input completeness

The two complete cached Jacobian-capture and row-family native files pass
**59 tests in 50.12 seconds**, with fallback disabled and their original
cached/fresh and CPU-oracle bounds retained. Their input helper now preserves
the force and mass residual words that production passes to the solver, plus
an owned root-COM copy. Contact dimensions, both friction cones, equality and
limit rows, dense/component and packed layouts, disturbed workspaces and the
large-row route remain covered.

A new negative control deliberately drops the residual arguments and detects
the resulting acceleration difference on both profiles. The preceding complete
native discovery stopped after **243 passes and one failure in 1,100.48 seconds**
on this incomplete-input fixture. Including both produced words reproduces the
fresh rotational acceleration exactly in the isolated diagnostic; no solver or
physics code, tolerance or iteration budget was changed. This is a corrected
regression gate, not a full-suite pass. The last complete CPU gate passed
**2,907 tests, 99 skipped and 2,406 deselected in 101.03 seconds** at `58989de9e`.
Skipped GPU tests are not qualification evidence.

## Observational assembly, DC admission and contact-bound APIs

Revisions `a4ca52ab1`, `42eaa5be1` and `12cc24ffd` add owned observational
assembly, correct zero-state DC dynamics admission and expose the model-only
`mj_maxContact` utility. The affected native actuation/query/plugin selection
passes **60 tests, 37 deselected in 54.57 seconds** with CPU fallback disabled.
This includes every individual force in a 20-combination gain/bias fixture,
real neighboring activation states, four integrators and exact precision replay.
Positive state layouts and DC-gain layouts remain validated. The native shader
already skips unallocated dynamics; the admission correction follows that same
pinned source rule. Independent CPU forward and eight-step execution accepted
all 220 compile-valid templates previously rejected by that admission check.

Observational-query fixtures cover dense Euler and component implicitfast,
malformed input, an injected post-assembly failure, actual borrowed producer
storage, a pending split-step record, stored energy/sensors, held inputs and
checkpoint/replay preservation. They were additionally strengthened with
independent pinned current-state acceleration and constraint-force comparisons;
all **four tests pass in 3.49 seconds**, with native CPU-physics calls forbidden.
The earlier SDF error-norm-only noninterference check missed changes to retained
multiplier bytes. Its stronger complete-state checks pass for both bowl and
torus through the new observational query. The producer's documented mutable
workspace behavior is retained.

The actual public model-only contact-bound utility passes **441 host tests**;
its combined binding/catalogue selection passes **457 tests in 1.88 seconds**.
Both geometry orders, all nine geometry types, CCD flags, SDF seed counts,
explicit margins, model override/pair/geom precedence and argument boundaries
are compared with pinned MuJoCo 3.10.0. No simulation state or physics is
dispatched. These results qualify the utility, not contact physics. Full native,
distribution, demo and milestone closure remains pending.

## Current checkpoint and owned keyframe regression gates — 2026-10-11

Revisions `fe644d658` and `70136b887` update native lifecycle fixtures to the
current schema-7 checkpoint and the prepared model ownership contract. Schema-6
activation migration remains tested; current checkpoints additionally preserve
owned precision planes. Invalid keyframe tests corrupt the actual prepared
payload, while constructor and public inspection copies are independently shown
to leave that payload unchanged.

The complete actuator, device-state, mocap and equality native selection passes
**19 tests, 32 deselected in 28.27 seconds**, with no failures or skips. Original
activation, selected reset, invalid-input and attach/release trajectory bounds
remain, with stronger exact velocity/precision replay and rejection atomicity
checks. These commits change tests only, not production physics. The preceding
full native discovery stopped after 115 passes on the stale schema assertion;
that discovery is not a full-suite pass. Full composition, wheel and demo
qualification remains open.

## Stored energy copy and checkpoint ownership — 2026-10-11

Revision `ab7c360a2` includes enabled stored energy in full simulation copies
and checkpoints. Full and selected restoration preserves the saved stage
samples without running extra physics. Energy payloads are validated before
state writes or extension callbacks. Legacy schema-1 checkpoints without the
optional field clear selected stored energy; disabled diagnostics carry None.

The new complete native lifecycle file passes **19 tests in 4.33 seconds**,
covering all five dense/scalable Euler/RK4/implicit profiles, nonzero independent
CPU energy comparisons, copy and result ownership, full/selected/legacy restore,
malformed atomic rejection, disabled energy, empty-selection rejection and exact
replay. All CPU physics stage calls are forbidden during native evaluation.
The complete affected transaction, split-stage, actuator-delay, energy and
recovery group passes **71 native tests, 69 deselected in 49.59 seconds**,
with no failures, skips or warnings. Revision `88f297e75` repairs its stale
force-failure injection to the executed paired solver, verifies injection calls,
and preserves every original failed/healthy state, sensor and warm-start bound.

The full CPU suite at `ab7c360a2` passes **2,460 tests, 99 skipped,
2,398 deselected in 99.48 seconds**. This does not qualify skipped native cases.
The scoped lifecycle results do not establish full milestones 006/019/020,
standalone energy-stage API coverage, final installed-wheel physics or demos.

## Articulated COM and nonzero elastic damping — 2026-10-11

Revision `abf0bde7c` publishes each articulated root's center of mass into its
awake descendants' slots, matching the point-velocity consumer's body indexing.
Previously those descendant slots were zero, producing false edge extension
under rigid motion. Revision `ed74d3fa0` evaluates the pinned squared-length
Rayleigh term as `damping * rate * (2 * length - timestep * rate)`, avoiding
cancellation without changing the material law, derivative or solver budget.

At that combined source, **16 native cases passed in 94.16 seconds**: 2D/3D
simplexes, damping 0.1/0.4 and all four integrators. These compare nonzero passive
forces, dissipative work, independent velocity derivatives, four-step states
and exact checkpoint replay with the pinned CPU oracle. CPU physics calls are
forbidden during native evaluation. A separate dense/sparse COM, prepared
POS/VEL, spatial-query, sleep, selected reset, recovery and invariant group
passed **27 native tests in 16.68 seconds**. No bounds were relaxed.

The complete CPU selection at `ed74d3fa0` passed **2,460 tests, 99 skipped,
2,379 deselected in 99.93 seconds**, with MPS availability hidden. This does not
qualify skipped GPU cases. The earlier 90-case material/integrator/equality
matrix was repeated at `4d5497758` together with the 32 native independent
energy cases: **122 passed, 109 deselected in 2,670.23 seconds**, with no
failures, skips or warnings. The original force, derivative, trajectory, replay
bounds and budgets were unchanged. Broader compositions, long-horizon behavior,
demos and final wheel remain required. These results
are scoped correctness evidence, not full milestone closure or a speed claim.

## Retained velocity in inertial bias — 2026-10-11

Revision `ff659a24e` feeds the simulator's represented high/low/tail velocity
into compensated comVel/RNE calculations, including cached velocity stages and
RK4 stage state. Standalone device tensors have zero residuals; the explicit
host-input entry retains supplied velocity residuals. The native dispatch keeps
its 31-buffer binding limit, and memory admission includes the packed payload.

A two-world public simulator fixture with binary-exact pose and inertia
reproduced a bias error of `4.19095164e-9`. Its corrected result agrees with the
independent CPU oracle within `2e-13`, including actual tensor mutation,
replacement, selected reset and checkpoint restoration. CPU forward/step calls
are forbidden during native evaluation. This isolates velocity loss; it does
not establish source-rounded precision for general articulated inertia/CDOF.

The affected native/host group passed **71 tests in 66.47 seconds**, including
the complete original derivative file, cached POS/VEL, retained-state lifecycle,
invalid-metadata ownership and actual workspace accounting. The complete CPU
selection passed **2,455 tests, 99 skipped, 2,359 deselected in 99.54 seconds**,
with MPS availability hidden. Skips do not establish native qualification.

Both original zero-margin alternate-CCD box terrain orders then passed at that
revision: **2 passed, 48 deselected in 50.04 seconds**, preserving their original
force, four-step trajectory, checkpoint bounds and solver budgets. Their earlier
failures below are historical. The complete unchanged matrix then passed
**40 tests, 10 deselected in 1,893.40 seconds** at the same production revision.
This covers five primitive families, both CCD algorithms, both geom orders,
zero/positive margin, a contact-free peer, forces, four-step trajectories and
checkpoint replay. Complete heightfield composition and milestone acceptance
remain open; no terrain bounds or solver budgets changed.

## Public API corrections and retained state — 2026-10-11

The complete remaining 26-file native API selection at producing revision
`baf24b80c7f1ae88f2fb3228ffab2ef3549c5cf5` passed **117 tests, 280 deselected
in 440.95 seconds**, with no failures or skips. The same selection previously
passed 101 and failed 16 at `783ce0d6d`. Its original force, trajectory,
checkpoint, recovery and ownership bounds were retained. This includes the
seven original bundled rigid-SDF oracle fixtures: query reuse, five provider
force/trajectory/replay cases and mesh contact rows. Those fixtures pass;
the complete SDF pair, search and feature-composition scope remains open.

Three production corrections and one stale fixture correction cleared the
observed failures:

- `c3aad7ef6` supplies the per-world selector suffix required by both native
  post-constraint RNE dispatches and admits its storage. The complete RNE file
  passed **34 checks in 5.22 seconds**.
- `5ced2d594` publishes the float32 projection of the complete retained scalar,
  including a tail that decides a halfway tie, and canonicalizes residuals with
  scaled arithmetic that preserves subnormal storage. The strict checkpoint
  validator is unchanged. The writer and original 120-step site-feedback
  trajectory/reset/replay passed **3 checks in 9.72 seconds**; the affected
  integration/quaternion/lifecycle/capacity group passed **42 checks in
  230.02 seconds**.
- `c371443a7` aligns the site-feedback fixture with the existing public velocity
  return contract and its independent oracle's target parameters. It preserves
  the nonzero selected force comparison and unselected-world sentinel.
- `baf24b80c` publishes masked compensated force words through predicated native
  stores, preserving borrowed high/low RHS versions during healthy checks.
  The force/site-feedback/owner group passed **39 checks in 13.60 seconds**;
  complete state-check, production recovery and admission groups passed
  **58 checks in 6.17 seconds**. These include host contracts and actual native
  execution; they are not all GPU tests.

The complementary 35-file forward/inverse/query/derivative/plugin native
selection was repeated at `f59d1224feb872fc084aa011772f6a66de1b58a0`:
**304 tests passed, 384 deselected in 151.61 seconds**, with no failures,
skips or warnings. Its earlier immutable-index ownership warning was corrected
in `783ce0d6d`; the affected flex inverse/lifecycle group separately passed
**10 tests, 8 deselected in 25.18 seconds**, treating that warning as an error.
Together the two complementary selections exercise 417 native cases; their
runtime sources are identical, with only inventory and documentation changes
between their producing revisions.

The complete original `test_finite_difference_native_020.py` at `f59d1224f`
then passed **14 native tests in 55.55 seconds**. This separately covers the
rich three-world contact/equality/mocap/activation and stored-sensor fixture,
including its original inverse-force derivative bounds, public transition and
inverse matrices, exact query rollback, checkpoint/reset behavior,
integrators, sleeping worlds and nonfinite-world isolation. Earlier derivative
failures remain historical evidence; this result does not cover every possible
model or derivative combination.

The full CPU selection at `f59d1224f` passed **2,442 tests, 99 skipped,
2,320 deselected in 99.38 seconds**, with MPS availability hidden. It does not
qualify the skipped native cases.

Full API and milestone acceptance still require the execution inventory,
remaining feature combinations, final wheel and demo gates. The later complete
40-case primitive-terrain result is recorded above; broader heightfield
composition remains open. No new performance result is claimed.

## Derivative residuals and active elastic materials — 2026-10-11

Revision `88aed2063` retains the low word of the inverse mass-product term
when forming finite-difference force derivatives. The complete native derivative
file, a new nonbinary-inertia public inverse-derivative regression, and affected
CPU algorithm/inverse contracts passed **82 tests in 56.49 seconds**. This is
a mixed host/native group. The new native case compares two nonzero acceleration
worlds with the independent pinned CPU derivative and requires exact query
rollback; helpers alone are not its physics evidence.

The five-file original material/integrator/equality native group at that revision
passed **90 tests, 107 deselected in 2,435.07 seconds**, with no failures or
skips. It includes all four integrators, two timesteps, Q1/Q2 volume and shell
materials, and dense/sparse flex equalities. The stiff larger-step Q2 RK4 fixture
checks pinned warning/reset behavior; it is not stable-integration evidence.

Stronger active-material checks then exposed false elastic forces after rigid
transformation: **three failed and one passed**. Revision `15787fa29` preserves
retained vertex and compiled rest-length residuals in stretch forces and uses
the same geometry for their analytic tangents. All **four physical invariant
cases passed in 7.29 seconds**; the affected material, baseline flex, memory
accounting and invariant group passed **90 tests in 39.16 seconds**. That second
90-test result combines host and native execution and is distinct from the
earlier all-native group.

The new fixtures exercise active 2D/3D elasticity at two stiffnesses,
undeformed/deformed and translated/rotated worlds, nonzero free-carrier reactions,
linear/angular momentum balance and exact replay. The rest-length residual
planes and conservative temporary geometry bound are included in capacity
accounting, with a test of actual allocated storage. These results do not close
the full deformable geometry, contact, energy and stability scope.

The added `test_active_material_energy_018.py` passed **34 checks in 236.36
seconds** with the same production source: two independent CPU generalized-work
gradient cases and 32 native trajectories. The matrix spans triangular/tetrahedral
materials, stiffnesses 1,000/4,000, timesteps 20/200 microseconds and all four
integrators, with deformed free carriers in two frames and exact replay.
Over six steps, maximum relative total-energy drift was **0.09235%** in both
engines; maximum native/CPU energy disagreement relative to initial energy was
**5.01e-10**. These are short-window correctness/stability measurements, not
long-horizon stability or performance results. Elastic plus kinetic energy is
evaluated by an independent test oracle from the represented native state;
this does not add elasticity to upstream's potential-energy sensor contract.
No native simulation path receives oracle forces, states or energy values.

The two unchanged alternate-CCD box terrain cases were repeated at `15787fa29`:
**both failed, 48 tests deselected in 52.10 seconds**, with the same maximum
acceleration violation of `2.81290692`. The later retained-velocity correction
above clears both original cases; this failed result belongs to its source.

## Earlier incomplete primitive-terrain matrix — 2026-10-11

The unchanged 40-case native terrain matrix at producing revision
`10d9bf13676b766ddb936b19876ffeab8b9827af` finished with **38 passed,
2 failed, 10 host tests deselected in 1,863.01 seconds**, exit 1. It covers
sphere, capsule, ellipsoid, cylinder and box, both CCD algorithms, both geom
orders, zero/positive margin, a contact-free neighboring environment, four
trajectory steps and checkpoint replay. The original `rtol=atol=2e-3`
trajectory bounds and solver budgets were retained.

All four previously failing native-CCD cylinder/box cases now pass. The
remaining failures at that producing revision are the zero-margin
**alternate-CCD box** cases in both geom orders; rotational acceleration differs
by up to `2.81290692` on the
original trajectory. This is not complete heightfield qualification or full
milestone closure. The preceding 34-pass/6-failure result below belongs to
its earlier producing revision.

An observer of the ordinary step sequence reproduces this failure. At the
identical represented state, native witnesses match the pinned CPU engine
and native acceleration agrees to about `1.2e-11`; the independent CPU
trajectory later selects a different alternate-CCD contact result after a
tiny accumulated state difference. That evidence identifies a precision and
trajectory gate subsequently corrected by consuming retained velocity words.
It does not justify removing the
original gate or replacing trajectory qualification with same-state checks.

```sh
MUJOCO_METAL_RUN_GPU=1 PYTORCH_ENABLE_MPS_FALLBACK=0 \
PYTORCH_MPS_FAST_MATH=0 PYTHONPATH=metal:metal/tests \
python -m pytest -q -m gpu -p no:cacheprovider \
  metal/tests/test_primitive_hfield_full_manifold.py
```

## Original-operator warm selection — 2026-10-11

Local producing revision `c77851c9072295d014f1c015c8c9efa36a0da476`
compares cold and warm acceleration objectives using retained original mass,
Jacobian, reference and regularizer words. It evaluates quadratic, unilateral,
Huber and elliptic cone costs with paired arithmetic, preserving the cold
acceleration residual for DOFs outside the current island. The preceding
regularizer commit `eb173e5a393837261108e9c989876260fdb79b8c` retains compiled
DOF/body inverse-weight residuals in existing constant bindings, accounts the
additional storage, and publishes joint/connect/weld regularizer low words.

The failing-first regression has a warm seed which the pinned source rejects
but rounded operator/cost arithmetic accepts. Its adjacent float32 seed must
be accepted. All **8 native boundary cases** now pass across CG/Newton and
dense/component routes, with CPU physics forbidden; the complete boundary
group passed **12 tests in 33.31 seconds**. The original rejected-seed run failed
all four native cases. No epsilon, blanket seed rejection or tolerance change
was used to fix the decision.

Further qualification passed:

- **85 native tests, 119 deselected in 13.09 seconds**: actual equality
  regularizers, existing primal solver, islands and line-coefficient cases.
- **10 native tests, 67 deselected in 50.09 seconds**: original solver-selection,
  finite-budget, warm-start and 300-step slope compositions.
- **9 tests in 1.94 seconds**: compiled residuals/capacity accounting and actual
  native joint/connect/weld regularizers, packed constant sizes and repeated
  inactive-world transitions. Regularizers match pinned rows at `2e-12`.
- **2,436 host tests, 99 skipped, 2,319 deselected in 99.46 seconds**. Skips do
  not qualify native physics. All producing source/tests are now committed.

Complete terrain, solver-option and other feature compositions remain open,
including Huber force-zone predicates and generic articulated mass precision.
This is no claim of whole-milestone, current-wheel or performance qualification.

```sh
MUJOCO_METAL_RUN_GPU=1 PYTORCH_ENABLE_MPS_FALLBACK=0 \
PYTORCH_MPS_FAST_MATH=0 PYTHONPATH=metal:metal/tests \
python -m pytest -q -p no:cacheprovider \
  metal/tests/test_primal_objective_pairs.py \
  metal/tests/test_equality_regularizer_pairs.py \
  metal/tests/test_primal_solver_math.py \
  metal/tests/test_solver_islands_014.py \
  metal/tests/test_primal_line_coefficient_residuals.py
```

## Retained native terrain constraint rows — 2026-10-11

Local producing revision `fdcff2d80f9ee0159f3df29ceb76b6e5f6c259d7`
enables the retained contact-row consumer for both terrain CCD algorithms.
The preceding publication correction preserved accurate native witnesses, but
the host enabled precise Jacobian/reference/regularizer assembly only when
native CCD was disabled. The same-state box trace isolated that consumer loss.

The dense/component native witness and existing alternate-CCD PGS/CG/Newton
operator group passed **14 tests in 360.15 seconds**, exit 0. The native
cylinder/box witness cases cover both geom orders, actual canonical Jacobians
and regularizers at `2e-12`, reference accelerations at `rtol=2e-11,
atol=2e-10`, and an adjacent contact-free world. CPU forward/mass calls are
forbidden during native assembly. The unchanged original native box trajectory
and four strict component cases passed **5 tests in 57.31 seconds**, exit 0.
No model, solver budget or original trajectory bound was altered.

The full host selection passed **2,433 tests, 99 skipped, 2,313 deselected in
99.98 seconds**. Four host checks belong to the uncommitted warm-boundary
regression under development; the committed host subset is 2,429 checks.
The complete terrain matrix has not been rerun at this revision. The separate
scalar-equality regularizer and initial warm-objective defects remain open.
This is scoped numerical qualification, not whole-milestone completion.

```sh
MUJOCO_METAL_RUN_GPU=1 PYTORCH_ENABLE_MPS_FALLBACK=0 \
PYTORCH_MPS_FAST_MATH=0 PYTHONPATH=metal:metal/tests \
python -m pytest -q -p no:cacheprovider \
  metal/tests/test_hfield_native_prism_precision.py \
  metal/tests/test_contact_operator_precision.py
```

## Native prism witness publication — 2026-10-11

Local producing revision `d97f75a5e9615e380e2c091b10394ab7f66a7284`
preserves the position, normal and distance words already computed by native
prism GJK/EPA. Previously the heightfield publisher transformed projected
float32 witnesses and left their precision sidecar zero. Native prism witnesses
now reach the world transform, frame construction and existing owned contact
payload without that projection loss. No collision algorithm, model, tolerance,
solver budget, shader binding or allocation was added or changed.

The affected native publication/pose group passed **7 tests, 4 deselected in
32.16 seconds**, exit 0. Four new cases compare actual retained cylinder/box
contact position, normal and distance with pinned CPU contacts at `2e-12`,
in both geom orders and with a second contact-free world. CPU forward/mass
calls are forbidden during the native publication. Existing retained pose and
world-transform tests also pass. An initial test reconstructed the distance
from incorrect record indices; the corrected test uses the canonical high/low/
tail fields and retains its original bounds.

The unchanged original native-CCD cylinder trajectory/restore case passes.
At this producing revision the native-CCD box trajectory still failed, with a maximum violating velocity
error of `0.01710173`; the expanded terrain matrix has not been rerun at this
revision. Initial witness precision passing is not full trajectory or terrain
qualification. Native sphere/capsule shrink-distance precision and complete
solver compositions also remain open.

The full development host run passed **2,433 tests, 99 skipped, 2,309 deselected
in 99.25 seconds**. This includes 2,429 checks in the producing commit and four
additional CPU boundary-oracle checks under development. The latter do not
qualify native warm selection: the rejected-seed native matrix separately
failed all four dense/component CG/Newton cases at the preceding source. A
scalar-equality regularizer precision gap and rounded objective selection need
correction. No whole milestone, wheel or performance claim is made.

Reproduce the committed native publication group with the pinned runtime and
fallback disabled:

```sh
MUJOCO_METAL_RUN_GPU=1 PYTORCH_ENABLE_MPS_FALLBACK=0 \
PYTORCH_MPS_FAST_MATH=0 PYTHONPATH=metal:metal/tests \
python -m pytest -q -m gpu -p no:cacheprovider \
  metal/tests/test_hfield_native_prism_precision.py \
  metal/tests/test_hfield_geom_xmat_transport_018.py \
  metal/tests/test_hfield_fk_geom_xmat_source_witness.py \
  metal/tests/test_hfield_represented_state_019.py
```

## Consistent primal line coefficients — 2026-10-11

Local producing revision `8f105983b9e52c6a2afbfd119256355c81e736be`
retains regularizer and anisotropic friction words through scalar and elliptic
line evaluation, matching the represented operands used by gradient/Hessian
updates. It also corrects the missing polynomial constant when an initially
satisfied unilateral row enters its active quadratic zone. Test-only shader
wrappers exercise the production functions against pinned `mj_constraintUpdate`
costs and independent derivatives; no CPU physics is used in the native solve.

The complete host selection passed **2,429 tests, 99 skipped, 2,297 deselected
in 101.87 seconds**, exit 0. The focused native line group passed **43 tests,
34 deselected in 7.07 seconds**, exit 0. It covers scalar equality, unilateral
and Huber transitions plus elliptic dimensions 3/4/6 and cone-zone transitions,
with cost/derivative/curvature bounds of `2e-12`.

The original zero-margin alternate cylinder-heightfield trajectory and restore
case now **passes in 57.77 seconds**, with its model, `rtol=atol=2e-3`, solver
budget, independent four-step rollout and bitwise replay unchanged. The complete
40-case terrain matrix finished with **34 passed, 6 failed, 10 host tests
deselected in 1,803.19 seconds**, exit 1. The failures are zero-margin cylinder
trajectories with native CCD and zero-margin box trajectories with either CCD
mode, each in both geom orders. Initial contacts, frames and forces passed;
later velocity or acceleration comparisons failed. This result does **not**
qualify all terrain shapes, CCD modes, margins or geom orders. Initial cold/warm objectives
and Huber force-zone predicates also retain precision gaps; complete solver
qualification remains open. The earlier failures below belong to their recorded
producing revisions, rather than proving the corrected case still fails.

This is source-only scoped qualification, not milestone closure, a current-wheel
qualification or a performance claim. Python 3.12.13, MuJoCo 3.10.0 and Torch
2.9.1 were used, with fallback/fast math disabled and serialized native checks.
Reproduce the focused line group:

```sh
MUJOCO_METAL_RUN_GPU=1 PYTORCH_ENABLE_MPS_FALLBACK=0 \
PYTORCH_MPS_FAST_MATH=0 PYTHONPATH=metal:metal/tests \
python -m pytest -q -m gpu -p no:cacheprovider \
  metal/tests/test_primal_line_coefficient_residuals.py \
  metal/tests/test_primal_elliptic_line_pair_014.py \
  metal/tests/test_primal_line_alpha_014.py
```

The full terrain matrix uses the same environment and GPU flags:

```sh
MUJOCO_METAL_RUN_GPU=1 PYTORCH_ENABLE_MPS_FALLBACK=0 \
PYTORCH_MPS_FAST_MATH=0 PYTHONPATH=metal:metal/tests \
python -m pytest -q -m gpu -p no:cacheprovider \
  metal/tests/test_primitive_hfield_full_manifold.py
```

## Original-mass residuals in coupled solvers — 2026-10-11

Local producing revision `cb2901278b0c59eb51c83cc820a306333fb73303`
connects the compiled inertia residual producer to fresh and retained-position
PGS, CG and Newton matrix operators. Original high/low mass is retained before
rounded factor scratch is reused. Component tendon-armature high/low inputs
are included in the original operator. Refinement consumes mass residuals even
when the force or Jacobian low word is zero. Dense primal initial acceleration,
paired gradient/Hessian products and paired line polynomials use these words.
The debug tail and complete solver dimensions, including the world mask, are
included in admission; the coupled shader retains its 31-buffer ABI.

Empty contact descriptors now have zero logical slots while device upload
retains physical sentinels. Exact-impedance cone validation accepts its inert
sentinel. Component exact row packing now dispatches its final zero RHS row;
previously that allocated row could retain uninitialized values. Native tests
poison the backing before initial and repeated recomputation. Cached component
PGS solves, status merging and component armature staging preserve selected
world ownership.

At the producing revision, the complete host selection passed **2,426 tests,
99 skipped, 2,270 deselected in 100.78 seconds**, exit 0. The affected native
selection passed **124 tests, 129 deselected in 36.53 seconds**, exit 0. This
includes 45 new native cases across fresh/cached dense/component PGS/CG/Newton:
actual nonbinary MuJoCo inertia, friction constraints, armature residuals,
invalid metadata, masked nonfinite isolation, failure/recovery, actual backing
admission and component exact inverse rows. The equations use independent
`mj_fullM` oracles, with CPU forward/mass calls forbidden during the selected
native coupled execution. These checks retain their original tight bounds.

The original dense-PGS terrain contact-Jacobian/reference/inverse-mass gate
passed separately in a fresh process: **1 passed in 118.15 seconds**, exit 0.
It retains source Jacobian and regularizer bounds of `2e-12`, reference bounds
of `rtol=2e-11, atol=2e-10`, and inverse-row bounds of `2e-11`. Its mass oracle
uses both published original-operator words. A preceding combined invocation
reported a Metal command-buffer error and missing contacts; that failure is
not erased by the isolated pass. Reliability across long mixed native test
batches remains unproven. The existing Hessian witness was also corrected to
use its canonical scalar-row `1/R` weight; the previous inconsistent fixture
failed identically against the preceding shader revision.

The unchanged zero-margin alternate cylinder-heightfield **trajectory still
fails** at its original `rtol=atol=2e-3`, with maximum violating acceleration
error `0.05186221`. Initial contact/frame/generalized-force checks passed.
Its latest combined terrain/strict run failed both cases in 35.50 seconds;
the missing mass-result field that stopped that strict run was subsequently
fixed, and its fresh-process result is recorded above. No terrain fixture,
tolerance or solver budget was relaxed.

General articulated CRBA residuals, dense exact-impedance row precision,
effective implicit precision, scalar cold/warm objective predicates and full
solver/contact/integrator compositions remain open. Component exact inverse
rows passing does not establish complete binary64 regularizer precision.
This is **scoped operator qualification**, not full milestone closure, terrain
parity, a wheel qualification or a speedup claim. Earlier sections below record
historical producing revisions rather than the current residual wiring.

Python 3.12.13, MuJoCo 3.10.0 and Torch 2.9.1 were used. All native tests ran
with fallback disabled and serialized GPU work; runtime source remained
unchanged during the final checks. Reproduce the affected native selection:

```sh
MUJOCO_METAL_RUN_GPU=1 PYTORCH_ENABLE_MPS_FALLBACK=0 \
PYTORCH_MPS_FAST_MATH=0 PYTHONPATH=metal:metal/tests \
python -m pytest -q -m gpu -p no:cacheprovider \
  metal/tests/test_coupled_mass_residual.py \
  metal/tests/test_compiled_mass_residual_producer.py \
  metal/tests/test_compiled_simple_mass.py \
  metal/tests/test_component_matrix_residual.py \
  metal/tests/test_awake_smooth_017.py \
  metal/tests/test_smooth_sparse_mass.py \
  metal/tests/test_dense_pair_mass_solve_019.py \
  metal/tests/test_cached_position_015.py \
  metal/tests/test_position_context_geom_residual_018.py \
  metal/tests/test_forward_stages_019.py \
  metal/tests/test_native_inverse_split_019.py \
  metal/tests/test_inverse_skip_019.py \
  metal/tests/test_mass_queries_019.py \
  metal/tests/test_component_smooth_force_pair.py \
  metal/tests/test_cached_position_solver_015.py \
  metal/tests/test_primal_hessian_precision.py \
  metal/tests/test_constraint_impedance_020.py
```

Run the original gates separately, quoting each complete node ID:

```sh
MUJOCO_METAL_RUN_GPU=1 PYTORCH_ENABLE_MPS_FALLBACK=0 \
PYTORCH_MPS_FAST_MATH=0 PYTHONPATH=metal:metal/tests \
python -m pytest -q -p no:cacheprovider \
  'metal/tests/test_contact_operator_precision.py::test_actual_contact_jacobian_reference_and_inverse_mass_rows[solver0-integrated_euler_v1]'

MUJOCO_METAL_RUN_GPU=1 PYTORCH_ENABLE_MPS_FALLBACK=0 \
PYTORCH_MPS_FAST_MATH=0 PYTHONPATH=metal:metal/tests \
python -m pytest -q -p no:cacheprovider \
  'metal/tests/test_primitive_hfield_full_manifold.py::test_full_primitive_terrain_contacts_forces_trajectory_and_restore[0.0-False-True-cylinder-.25 .15]'
```

## Compiled inertia residual producer — 2026-10-11

Local revision `d3636c169a5dcd1fc430f56a210f6b539573b2fc` lowers the
residual of each binary64 compiled `dof_M0` constant into a separate float32
word. The native simple-row postpass publishes that word alongside dense or
component mass storage. Selected worlds and awake rows are updated; sleeping
rows and unselected worlds retain their previous backing. Generic CRBA rows
currently publish zero residuals. This preserves the represented constant pair;
it does not establish binary64 fidelity for general articulated inertia.

Retained POS stages own both mass planes and validate their metadata before
velocity-stage writes. The standalone `MetalSmoothDynamics.mass_matrix_pair`
query returns owned high/low dense matrices. Native component smooth-force
solves and dense/component discrete inverse mass products and solves consume
the producer's residuals. Lowering uses host compilation constants; stepping
and these native operators do not call CPU forward dynamics or a CPU mass
query. Device admission includes the constant residual, mass residual backing
and retained POS residual, including physical empty-model sentinels.

At the exact revision above, the complete host selection passed **2,424 tests,
99 skipped, 2,225 deselected in 99.45 seconds**, exit 0. The affected native
regression group passed **58 tests, 83 deselected in 9.56 seconds**, exit 0.
This group covers the new producer, compiled simple mass, paired component
matrix, awake/sparse mass, dense pair solve, position/cache validation, split
forward/inverse stages, inverse skipping and mass query fixtures. New native
checks compare actual source constant residuals and inverse operations, reject
corrupted cached metadata before output writes, refresh recomputed constants,
invalidate stale shortcuts, preserve masked/sleeping backing and verify actual
empty-model allocations against admission. Native producer/inverse tests also
forbid CPU `mj_forward` and `mj_fullM` calls during execution.

The unchanged original zero-margin alternate cylinder-heightfield trajectory
was rerun separately and **failed in 10.91 seconds**, exit 1. Initial contacts
and constraint forces passed; the rollout acceleration discrepancy remains
`1.85741866` at the original `rtol=atol=2e-3`. The coupled constraint APIs do
not yet consume the mass residual planes. Remaining work includes that
integration, generic source-order CRBA, factor/refinement conditioning,
effective implicit operators, remaining compositions and demo/wheel gates.
**No full milestone closure, terrain parity or speedup is claimed.**

These checks used Python 3.12.13, MuJoCo 3.10.0 and Torch 2.9.1 with the
same MPS and host guards recorded below. Tracked source stayed unchanged.
Reproduce the affected native selection with the pinned runtime:

```sh
MUJOCO_METAL_RUN_GPU=1 PYTORCH_ENABLE_MPS_FALLBACK=0 \
PYTORCH_MPS_FAST_MATH=0 PYTHONPATH=metal:metal/tests \
python -m pytest -q -m gpu -p no:cacheprovider \
  metal/tests/test_compiled_mass_residual_producer.py \
  metal/tests/test_compiled_simple_mass.py \
  metal/tests/test_component_matrix_residual.py \
  metal/tests/test_awake_smooth_017.py \
  metal/tests/test_smooth_sparse_mass.py \
  metal/tests/test_dense_pair_mass_solve_019.py \
  metal/tests/test_cached_position_015.py \
  metal/tests/test_position_context_geom_residual_018.py \
  metal/tests/test_forward_stages_019.py \
  metal/tests/test_native_inverse_split_019.py \
  metal/tests/test_inverse_skip_019.py \
  metal/tests/test_mass_queries_019.py
```

## Component matrix residuals and cached diagonal ownership — 2026-10-11

Local revision `dccfbe5c59396f50679e018e40c55f6751361814` adds optional
`mass_blocks_low` and `tendon_armature_blocks_low` inputs to the native component
mass solver. Fresh and cached solves retain the original operator's residuals;
both matrix-vector routes consume them. Cached factorization owns both matrix
planes and its diagonal addition, so caller mutation and subsequent scratch
reuse cannot silently change the equation used by refinement. Device admission
includes the retained residual and diagonal buffers, including empty-model
sentinels. Invalid residual metadata is rejected before retained state changes.

The complete host selection passed **2,423 tests, 99 skipped, 2,215 deselected
in 100.46 seconds**, exit 0. The focused component group passed **41 tests in
1.19 seconds**. The expanded native regression group passed **151 tests,
34 deselected in 149.23 seconds**, exit 0. These runs used the identical source
subsequently committed at the revision above, Python 3.12.13, MuJoCo 3.10.0 and
Torch 2.9.1, with MPS fallback and fast math disabled. No test ran concurrently
with an edit to its runtime source.

Native execution covers positive-definite and indefinite component equations,
noncontiguous component DOF layouts, multiple worlds and right-hand sides,
caller mutation after factorization, invalid input metadata, selected-world
failure isolation and recovery, diagonal ownership, and both matrix-vector
routes. An actual MuJoCo inertia matrix supplies a test-only oracle residual:
the previous quantized solve differs by more than `1e-8`; the retained-matrix
solve matches the full reference at `rtol=atol=3e-13`. CPU oracle matrices are
used only by tests, not by production stepping.

**This is a solver-interface correction, not full simulation precision.** The
production mass producer and coupled constraint routes do not yet supply these
matrix residuals. Factor storage remains float32 and is used as the refinement
preconditioner; the tests do not establish convergence for every ill-conditioned
operator or an operator whose rounded factor is singular. The original failing
terrain trajectory was not rerun for this interface-only change and remains
unqualified. No additional full milestone, demo, wheel or speedup is claimed.

Reproduce the focused component gate with the pinned source runtime:

```sh
MUJOCO_METAL_RUN_GPU=1 PYTORCH_ENABLE_MPS_FALLBACK=0 \
PYTORCH_MPS_FAST_MATH=0 PYTHONPATH=metal:metal/tests \
python -m pytest -q -p no:cacheprovider \
  metal/tests/test_component_matrix_residual.py \
  metal/tests/test_component_mass_solver.py \
  metal/tests/test_component_paired_rhs.py \
  metal/tests/test_component_solve_017.py
```

The expanded native selection adds these four files to the ten-file local
source regression command recorded below, and uses `-m gpu`. Supply the pinned
callback source root as described there; skips are not native passes.

## Compiled simple-inertia correction — 2026-10-11

Local revision `3d38fd35f` on `implementation/mac-milestone-completion`
retains MuJoCo's compiled `dof_simplenum` and `dof_M0` constants. Its native
mass stage uses the source constant diagonal and exact zero couplings for
simple rows, before the separate tendon-armature projection. Dense, component
and standalone query routes follow the same rule. The standalone query now
also supplies the complete mass-layout ABI, including the world-mask words.

The constants are owned and included in device memory admission. Lifecycle
recomputation supplies fresh source constants; direct descriptor edits invalidate
the shortcut and continue through generic native CRBA. No stepping operation
calls CPU physics. Native tests execute arbitrary free/ball rotations, hinge and
slide rows mixed with a nonsimple body, inherited motor armature, masked worlds,
sleeping rows and stale-constant invalidation.

The complete host selection passed **2,422 tests, 99 skipped, 2,208 deselected
in 100.60 seconds**. The final native group passed **5 tests, 3 deselected in
118.67 seconds**, including the original strict PGS/Euler contact-J rollout
gate and the four new mass tests. A separate existing awake/sparse/force group
plus the original zero-margin cylinder-terrain trajectory finished with
**21 passed and 1 failed in 35.56 seconds**. The failing trajectory retains its
original maximum violating acceleration difference `1.85741866`; no tolerance
or solver budget was changed.

Same-state diagnostics now show exact zero off-diagonal mass entries for the
isolated simple body. Its diagonal still has a float32 rounding difference
`2.9802322762395583e-10`, and the initial full retained acceleration differs by
approximately `2.5e-9`. Independently integrated trajectories still choose
different terrain witnesses. This is a structural mass correction, **not full
mass/solver precision, trajectory parity, milestone closure or speed evidence**.
Previous test counts below retain their original producing revisions.

Reproduce the new native execution tests and unchanged strict gate:

```sh
MUJOCO_METAL_RUN_GPU=1 PYTORCH_ENABLE_MPS_FALLBACK=0 \
PYTORCH_MPS_FAST_MATH=0 PYTHONPATH=metal:metal/tests \
python -m pytest -q -m gpu -p no:cacheprovider \
  metal/tests/test_compiled_simple_mass.py \
  'metal/tests/test_contact_operator_precision.py::test_actual_contact_jacobian_reference_and_inverse_mass_rows[solver0-integrated_euler_v1]'
```

Current `main` publishes the experimental development backend. See
[milestone status](STATUS.md): **0/16 milestones 005–020 have full-composition
sign-off**. Earlier bounded capabilities and individual regression groups retain
their recorded evidence. This is source publication, not a new PyPI release.

## Local source regression review — 2026-10-11

Revision `fc4cced35657f9bc24bf579fb0c39fd840cd95d9` on
`implementation/mac-milestone-completion` adds owned compensated force scratch,
pre-allocation memory admission, cached Jacobian/row ownership repairs,
selected-world recovery coverage and route-specific collision support fixes.
The prepared API host fixture now uses the real accumulator and preserves its
residual. Dense solver outputs start with defined values so a first masked
solve cannot expose allocator contents from an inactive world.

The final host run at that revision passed **2,419 tests, 99 skipped,
2,201 deselected, in 99.92 seconds**. It explicitly disabled MPS availability
for collection and excluded tests marked `gpu`; it is host regression
evidence, not native physics evidence. The host command below can be used
with pytest arguments `-m 'not gpu'` to reproduce this selection.

The independently executed native regression group passed **134 tests,
10 deselected, in 137.78 seconds**. It covers dense/awake solves, compensated
forces, retained Jacobians, masked recovery, cached constraint families, mesh
contacts/forces and cylinder support callbacks. The callback checks compare
both native CCD and libccd routes, with two rotations, against the installed
MuJoCo 3.10.0 binary. Their sampled bitwise matches do not qualify every
direction or other MuJoCo/compiler builds. Poisoned-allocation tests cover
first masked solves and retention of an inactive world's prior failure status.

A separate native spot check at that revision finished with **1 passed and 2 failed in
131.79 seconds**. Transformed/scaled heightfield frames passed. The original
PGS/Euler strict contact-J gate still fails (maximum violating error
`9.39097133e-11`, with `rtol=atol=2e-12`). The original zero-margin alternate
cylinder-heightfield trajectory also still fails its acceleration comparison
(maximum violating error `1.85741866`, with `rtol=atol=2e-3`). No tolerances
were relaxed. The remaining recorded zero-margin terrain cases were not
repeated in this spot check.

These results use Python 3.12.13, MuJoCo 3.10.0 and Torch 2.9.1 on Apple
Silicon, with MPS fallback and fast math disabled. All imports and shaders
resolve to this source checkout. They establish the covered regressions,
not full simulation parity, milestone completion, wheel qualification,
demo requalification or a performance improvement. At that revision the strict solver and
terrain gates, broader SDF/deformable/integrator/API combinations and final
distribution/demo gates remain open.

Reproduce the native regression group using the pinned source runtime.
`MUJOCO_METAL_TRACE_ROOT` must point to a directory containing
`.upstream/mujoco-3.10.0` for the callback ABI test; a skip is not a pass:

```sh
MUJOCO_METAL_RUN_GPU=1 PYTORCH_ENABLE_MPS_FALLBACK=0 \
PYTORCH_MPS_FAST_MATH=0 PYTHONPATH=metal:metal/tests \
python -m pytest -q -m gpu -p no:cacheprovider \
  metal/tests/test_solve.py metal/tests/test_awake_smooth_017.py \
  metal/tests/test_component_smooth_force_pair.py \
  metal/tests/test_cached_jacobian_capture.py \
  metal/tests/test_rhs_accumulate_owned.py \
  metal/tests/test_masked_recovery_production.py \
  metal/tests/test_cached_row_family_closure.py \
  metal/tests/test_mesh_contact_011.py metal/tests/test_mesh_force_011.py \
  metal/tests/test_common_ccd_support_source.py
```

Reproduce the two remaining numerical spot failures without changing their
original acceptance thresholds:

```sh
MUJOCO_METAL_RUN_GPU=1 PYTORCH_ENABLE_MPS_FALLBACK=0 \
PYTORCH_MPS_FAST_MATH=0 PYTHONPATH=metal:metal/tests \
python -m pytest -q -p no:cacheprovider \
  'metal/tests/test_contact_operator_precision.py::test_actual_contact_jacobian_reference_and_inverse_mass_rows[solver0-integrated_euler_v1]' \
  'metal/tests/test_primitive_hfield_full_manifold.py::test_full_primitive_terrain_contacts_forces_trajectory_and_restore[0.0-False-True-cylinder-.25 .15]'
```

## Near-axis libccd correction — 2026-10-11

Revision `7db8b94ac` preserves source precision when a cylinder support direction
is almost parallel to its axis. Its tiny radial components are scaled by an exact
power of two before their squared length is formed, then the scale is undone on
the root. Both libccd adapters use this path and retain the original source
`mjMINVAL` comparison. No collision epsilon, solver budget or test tolerance was
relaxed. The native-CCD cylinder branch is unchanged.

Independent same-state diagnostics found that FK and the local support direction
matched CPU exactly; the first loss was the tiny radial length, before the square
root's existing refinement. At the original failing state, all 36 terrain contact
positions now match the CPU oracle exactly, with normals matching within
roundoff. This fixes a collision defect rather than attributing the same-state
Jacobian discrepancy to divergence between two rollouts.

The native correction group passed **9 tests, 1 deselected in 139.30 seconds**.
It includes tiny/large radial arithmetic, both actual libccd adapters against the
installed MuJoCo callback near the axis and threshold, the complete frozen
terrain witness set with a separated peer, existing sqrt/callback regressions,
and the original PGS/Euler strict-J rollout gate at its unchanged tolerance.
The complete host selection passed **2,419 tests, 99 skipped, 2,204 deselected
in 102.06 seconds**. Runtime and flags are the same as above.

The original zero-margin alternate cylinder-heightfield trajectory **still
failed** in a separate 11.13-second run, with maximum violating acceleration
difference `1.85741866` at the unchanged `rtol=atol=2e-3`. Its initial contacts
and forces passed. The other original solver/profile and terrain trajectory
combinations are not qualified by the repaired strict-J node. No full milestone,
wheel, demo or performance claim follows from these bounded results.

Reproduce the correction group with the pinned callback source root supplied:

```sh
MUJOCO_METAL_RUN_GPU=1 PYTORCH_ENABLE_MPS_FALLBACK=0 \
PYTORCH_MPS_FAST_MATH=0 PYTHONPATH=metal:metal/tests \
python -m pytest -q -m gpu -p no:cacheprovider \
  metal/tests/test_libccd_radial_precision.py \
  metal/tests/test_common_ccd_sqrt_source.py \
  metal/tests/test_common_ccd_support_source.py \
  'metal/tests/test_contact_operator_precision.py::test_actual_contact_jacobian_reference_and_inverse_mass_rows[solver0-integrated_euler_v1]'
```

## Publication scope

The host publication suite passed **2,163 tests, with 1,615 native-dependent
tests skipped, in 94.60 seconds** at source revision `61e415e8b` (exit 0).
Tracked source was verified unchanged before and after execution. Subsequent
publication edits record this result and correct documentation links only.
See the [machine-readable record](validation/publication-20261005.json).
This result does not establish native correctness for the skipped cases.

The source contains coherent dynamics, lifecycle, collision, solver, deformable
and API changes plus their original regression fixtures. It also corrects an
undefined mask in public spatial-armature assembly. Final native replay and
broader armature composition checks remain open. Newer mesh-SDF arithmetic
experiments have not been included.

Some host test utilities require Torch: a no-Torch full-suite attempt stopped
with six collection errors and is not passing qualification. On a Mac, merely
setting `MUJOCO_METAL_RUN_GPU=0` does not prevent every historical test from
checking actual MPS availability. The publication host run therefore disables
MPS availability for collection. This is a test guard, not a change to backend
physics or production dependencies. Host checks exercise contracts, lowering,
algorithm references and collection; skipped native tests are not GPU passes.

## Selected recorded native evidence

| Group | Recorded result | Qualification boundary |
|---|---|---|
| Newton precision and original slope regression | 94 passed | Includes the original 300-step slope fixture; not the full solver-option matrix |
| Contact override | 69 passed | Selected rigid/flex, integrator and lifecycle fixtures |
| Retained SDF arithmetic/source trace | 11 passed | Numeric and trace-inertness checks; not full SDF contact correctness |
| Original rigid-SDF group | Incomplete at 300 seconds, with 3 failures | Mesh witness multiplicity and bowl/torus acceleration or trajectory gates remain open |
| More recent combined SDF/armature/flex check | Incomplete at 180 seconds | Different, unpublished source; no completed suite summary |

These native results belong to their recorded development compositions and are
not a fresh full GPU run of this publication. Other terrain, flex, stage and API
checks are described in [development coverage](DEVELOPMENT.md), with the same
scope restriction. No numerical tolerance or solver budget was relaxed to obtain
these results. A screenshot or successful compile is not physics evidence;
no new speedup is claimed.

## Reproduce on the pinned source runtime

Use Python 3.12, MuJoCo 3.10.0 and Torch 2.9.1 in the isolated source environment.
Record the commit, module/shader paths and hashes, flags, exact command, exit
code and pass/fail/skip counts. Keep GPU checks serialized.

Host reference check with an explicit test-availability guard:

```sh
MUJOCO_METAL_RUN_GPU=0 PYTHONPATH=metal python - <<'PY'
import torch
import pytest
torch.backends.mps.is_available = lambda: False
raise SystemExit(pytest.main(["metal/tests", "-q", "-p", "no:cacheprovider"]))
PY
```

Native gates (the original SDF cases currently include known failures):

```sh
MUJOCO_METAL_RUN_GPU=1 PYTORCH_ENABLE_MPS_FALLBACK=0 \
  PYTORCH_MPS_FAST_MATH=0 PYTHONPATH=metal \
  python -m pytest -q metal/tests/test_sdf_contact_013.py \
  metal/tests/test_bundled_sdf_rigid_oracle_019.py
```

Clean-wheel installation, the complete native suite and requalification of all
current demos remain pending. Gallery clips are recorded fixtures, not release
qualification or performance measurements.

---

# Historical bounded qualification — 2026-09-29

## Scope

The recorded September 29 bounded source included connect/weld constraints and runtime activation,
mocap/keyframe lifecycle, stateful actuators, spatial tendon wrapping,
site-only tendon armature bias and ball-joint limits. These extend the bounded
integrated Euler profile; they do not establish unrestricted MuJoCo compatibility.
Cylinder/ellipsoid collision support and the eccentric-roller demo were not
included in that qualification. They are now development paths on main, with
full qualification still open.

Reference/runtime: MuJoCo 3.10.0, Python 3.12, Torch 2.9.1, Apple Silicon MPS
float32; CPU MuJoCo provides the numerical oracle. Native checks use
`PYTORCH_ENABLE_MPS_FALLBACK=0`. Qualification targets the M1 Max with
32 GB unified memory; this is not broad Mac or Linux qualification.

## Validation results

| Check | Result | Scope |
|---|---|---|
| Full suite with native GPU execution | 518 passed, 336.75 s | Recorded physics qualification |
| Focused armature/spatial/coupled suite | 105 passed | Native regression checks |
| Gripper and drawbridge native headless gates | Exit 0 for both | Bounded demo behavior and CPU comparisons |
| Full suite without Torch | 215 passed, 304 skipped, 2.57 s | Final publication check |
| Shader source inventory, including missing-file negative control | Included in the passing CPU suite | Source completeness |
| Built-wheel contents and installation for this increment | Pending | Source checks are not a wheel-install test |
| Interactive display and broader hardware/OS behavior | Not requalified | Separate validation required |

The publication check reran the CPU suite; it did not repeat the recorded GPU
qualification. Skipped CPU tests are not GPU passes. Algebra checks, compilation
and demo images are not substitutes for native physics qualification. Passing
the bounded test matrix does not establish every admitted feature combination.
No new performance measurement or speedup is claimed.

## Reproduce from a clean checkout

Follow [source installation](INSTALL.md#development-source) in a dedicated
Python 3.12 environment. For no-Torch qualification, install the pinned MuJoCo,
NumPy, pytest and absl-py dependencies into a separate environment without Torch,
and point `PYTHONPATH` at this source tree. Do not uninstall Torch from a working
GPU environment.

```sh
PYTHONPATH=metal python -m pytest metal/tests -q -p no:cacheprovider
```

On an idle supported Apple GPU in the source-install environment:

```sh
PYTORCH_ENABLE_MPS_FALLBACK=0 MUJOCO_METAL_RUN_GPU=1 PYTHONPATH=metal \
  python -m pytest metal/tests -q -p no:cacheprovider

PYTORCH_ENABLE_MPS_FALLBACK=0 PYTHONPATH=metal:metal/examples \
  python metal/examples/muscle_gripper.py --headless --check --steps 3000

PYTORCH_ENABLE_MPS_FALLBACK=0 PYTHONPATH=metal:metal/examples \
  python metal/examples/cable_drawbridge.py --headless --check --steps 1400
```

Record the commit, imported module path, shader hashes, dependency versions,
exit codes and raw output for any new qualification. Current `main` source
features are not a new PyPI release; the published 0.4.0 package retains its
previous scope. See [development coverage](DEVELOPMENT.md), the
[support inventory](COVERAGE.md), and individual demo guides for numerical bounds.

## Sequential qualification: Milestone 005 — 2026-10-12

The complete targeted inventory/support-contract gate passes **1,292 tests in
37.04 seconds**, with no failures or skips. This closes Milestone 005's
infrastructure scope; it is not native physics or whole-backend qualification.
The remaining milestones are qualified in order using their complete required
checks, without requiring a full repository regression between milestones.

The crosswalk classifies all 529 pinned public enum members, 813 distinct fields
across the five runtime binding/header structures, 554 distinct binding/header
functions, 69 Python classes, 29 option fields and 79 package wrappers. Direct
MuJoCo 3.10.0 header checks independently validate C enum membership, function
declarations and field types/extents. They exposed three previously omitted
C-only SDF APIs; those retain explicit subsystem ownership and upstream/API
boundaries. Internal/prepared stages never imply same-named standalone API
equivalence. Missing or new rows, owners, test references and declaration drift
are exercised by negative controls. Generated public tables are checked verbatim.

The combination matrix contains 45 unordered geometry pairs, 1,050 actuator
transmission/dynamics/gain/bias tuples and 72 integrator/solver/cone/Jacobian
configurations. All are exercised through actual pinned compilation and applicable
CPU lowering/profile admission. The actuator fixture admits 870 tuples; 180
parameter configurations fail the pinned DC-motor activation-slot consistency
check. This is an explicit expected rejection for those parameters, not a
family-wide rejection or a skip. Heightfield/heightfield admission rejects the
absent source dispatcher; other absent-dispatch primitive combinations retain
empty contact capacity. None of these checks establishes native numerical
parity or arbitrary Cartesian-product feature support. All combination rows
retain their later numerical qualification owners.

```sh
PYTHONPATH=metal:metal/tests PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 \
python -m pytest -m 'not gpu' -q -x -p no:cacheprovider \
  metal/tests/test_support_inventory_005.py \
  metal/tests/test_combination_admission_005.py \
  metal/tests/test_binding_inventory.py \
  metal/tests/test_coverage_contract.py \
  metal/tests/test_counterpart_source_contract_005.py \
  metal/tests/test_admission_contract_consistency_005.py \
  metal/tests/test_model.py metal/tests/test_stepping.py \
  metal/tests/test_bundled_plugin_inventory.py metal/tests/test_doctor.py
```

The classification explorer is available through `coverage --inventory` and
`coverage --inventory --json`. Milestone 006's state/model lifecycle and magnetic
crane are next. The known full-timeline muscle-gripper failure remains open under
its owning physics/demo gates; no original numerical bound was relaxed.

## Sequential qualification: Milestone 006 — 2026-10-12

Milestone 006 closes its complete state/model ownership and admitted lifecycle
contract: **170 distinct host tests and 191 native tests**, with no skips or
failures. These include the 22-file core gate (**160 host in 5.21 seconds**,
**188 native in 166.93 seconds**), eight additional crane host/CLI tests and
one full native crane test, plus two host and two native mocap/flex lifecycle
checks. Overlapping smoke repeats are not added to the distinct counts.

The core gate covers model ownership, shared mass/inertia/contact/reference
updates and atomic rebuild/adoption; actual constant recomputation and late
failure/restore/randomization coherence; device state, equality/mocap, history,
activation, controls/forces, warm starts, raw/stored sensor diagnostics, energy,
sleep, retained precision and registered extension lifecycle. Full/selected
reset, keyframe, environment copy and checkpoint operations have source oracles,
healthy-peer preservation, real mutation/replacement, cache-owner checks,
late-fault rollback and exact subsequent replay. Known field/API/storage ABI
boundaries remain explicit; no arbitrary upstream callback or per-environment
live model randomization claim is made.

The full **900-step crane** and always-attached native counterfactual preserve
the original model, solver settings and physical bounds. New evidence adds
actual native bin-contact counts and exact complete-checkpoint replay from
transport through release and landing. Native/CPU contact counts are **218/218**;
attached counterfactual counts are **0/0**. Maximum full-run qpos/qvel errors are
**1.13e-7 / 1.86e-6**. Native weld force changes from **3.930** to **0.0**. The
refreshed 45-frame GIF was inspected across carrying, release and landing.

The added demo/CLI suite passes **9 tests in 112.15 seconds** with dtype- and
literal-bit-exact replay, including signed-zero negative controls. The previous
recording gate passed eight checks in 114.85 seconds; it is overlapping evidence.
Mocap/flex passes **2 native checks in 2.50 seconds**, with independent
source vertices, selected reset/keyframe/restore and exact following steps.
A main-thread GLFW/OpenGL viewer smoke run completes **66 actual native steps**
and exits normally under `--viewer-seconds 3`. The existing standalone runtime’s
upstream `mjpython` trampoline had library-lookup/crash failures; the demo uses
main-thread GLFW without changing any installed runtime or global environment.

These milestone checks do not qualify the full backend, packaging, broad Mac
hardware or later numerical feature matrices. Milestone 007 is next; the known
full-timeline gripper penetration failure remains its explicit open gate. No
repository-wide regression or performance benchmark was run.

## Sequential qualification: Milestone 007 — 2026-10-13 UTC

The final scoped acceptance set passes **181 host tests in 1.00 seconds** and
**198 native tests in 693.53 seconds**, with no failures or skips. Source bytes
remained unchanged. The nine original gate files are augmented by optional DC
states, aggregate limits/gravity/contact composition, nonzero rigid transmissions,
retained analytic contacts, and affected crane and contact-precision checks.
This closes admitted built-in actuation/rigid transmissions under integrated
Euler. Extensions remain owned by 019 and other integration stages by 015;
these results do not establish arbitrary Cartesian-product model support.

Corrections preserve temperature-dependent DC current resistance, the world
slider axis and its derivative, and the pinned rotational reference-site
quaternion convention. Tests compare nonzero state/velocity, changing commands,
parent-frame gears, common ancestry, infeasible crank branches, antagonist
forces, activation derivatives and complete checkpoint replay with independent
MuJoCo 3.10.0 stages/trajectories. Native stage checks forbid CPU physics.
All 32 optional DC-state allocations run under three controller modes.
Existing muscle transition, activation/force-limit and lifecycle checks remain.

The original gripper and always-open episodes each pass all 22,000 native steps
with unchanged XML, controls, solver configuration and bounds. Actual peak
contact penetration is **0.880637 mm native versus 0.880697 mm CPU**, below 1 mm.
Native support contacts occur on 14,299 steps and red-bin contacts on 12,436;
the always-open counterfactual has zero bin hits. Both objects lift above 0.309 m
and reach their receivers. A full checkpoint during the second carry replays
exactly, including dtype/literal bits, precision and solver/actuator history.

The earlier 28.01 mm capsule penetration failure is resolved by source-rounded
capsule/box manifolds from retained poses. A new retained lateral adhesion case
also exposed box/plane depth/impedance rounding under tiny rotation: its angular
acceleration error 9.023e-5 exceeded the unchanged 5e-5 absolute bound. Both
lateral and normal-motion cases now pass after source-rounded corner/depth
production and paired assembly. Direct depth, corner-order, tiny-rotation and
empty-world checks accompany the fix. No failed case was removed or bound
relaxed. Complete collision/solver-option qualification remains separate.

The reviewed GIF contains 88 actual native/CPU frame pairs at 2 fps across the
44-second grasp at 1x playback. Both deliveries, lighting, framing and clipping
were reviewed. Main-thread GLFW runs ordinary Python on macOS: 20 native steps
in three wall seconds, with CPU stepping forbidden. CPU forward evaluation only
mirrors native state for OpenGL. The rollout functions and XML are unchanged
from full-demo source 61bdb3611; later physics corrections affect transmission
and box/plane branches absent from that four-slide-muscle model.

Long delivery does not imply identical rotational trajectories: maximum
translation difference 0.986 mm, geodesic rotation 0.115 rad, and angular velocity
2.35 rad/s during the second pickup. Pre-contact position/velocity errors are
5.37e-8 and 1.83e-6; activation error stays below 3.22e-7.
See the [gripper guide](examples/muscle_gripper.md).

Reproduce the scoped selection in the pinned development environment:

```sh
PYTHONPATH=metal:metal/tests:metal/examples MUJOCO_METAL_RUN_GPU=1 \
PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 \
PYTORCH_ENABLE_MPS_FALLBACK=0 PYTORCH_MPS_FAST_MATH=0 \
python -m pytest -q -x -p no:cacheprovider \
  metal/tests/test_actuation.py \
  metal/tests/test_actuators_007.py \
  metal/tests/test_transmissions.py \
  metal/tests/test_transmission_simulation.py \
  metal/tests/test_actuator_inheritance_repair.py \
  metal/tests/test_dcmotor_unallocated_activations.py \
  metal/tests/test_actuator_velocity_derivatives.py \
  metal/tests/test_actuator_workspace_scaling.py \
  metal/tests/test_actuator_delay_repair.py \
  metal/tests/test_actuator_complete_007.py \
  metal/tests/test_actuator_limits_composition_007.py \
  metal/tests/test_rigid_transmission_complete_007.py \
  metal/tests/test_capsule_box_contact_source_007.py \
  metal/tests/test_magnetic_crane_006.py \
  metal/tests/test_contact_precision.py \
  metal/tests/test_contact_operator_precision.py
```

Run separately with -m 'not gpu' and -m gpu for the two phases. The prepared host
gate also disables MPS availability. The full original gripper is a separate
acceptance run, not part of the 198 count. No repository-wide regression,
benchmark, publication or release was performed. Milestone 008 is next.

## Sequential qualification: Milestone 008 — 2026-10-13 UTC

The complete 19-file scoped gate passes **113 host tests in 3.69 seconds** and
**66 native tests in 65.99 seconds**, without failures or skips. Native tests
use actual MPS, MuJoCo 3.10.0 and this source's modules/shaders with CPU fallback
disabled. The source stayed frozen through the checks. Marker exclusions divide
host/native phases; they are not skipped acceptance cases.

The original pulley-armature acceleration test first failed its unchanged
1e-4 absolute bound. Native site-only length/velocity/J now consume retained
FK residuals in compensated arithmetic. Complete armature family checks then
caught ordered same-body joints: final body orientation cannot reconstruct an
earlier joint's frame. Native FK now retains that joint's actual source anchor
and axis low/tail planes before later joints move. Source-word reconstruction,
position-cache ABI, physical memory counts and exact overflow guards pass.
The public outputs remain float32; this is not an unrestricted binary64 claim.
Wrapped and fixed tendon kernels retain their existing paths.

The selection includes the original spatial/fixed tendon mechanics, nonzero
motor/filter/muscle forces, slack/rest damping, tendon limits/friction/equalities,
contact composition and site armature dots/bias/mass/qacc. Independent finite
differences cover multiple pulleys, hinge/ball/free chains and both sphere and
cylinder wrap sides, inside routes and one-sided transition behavior. Actual
wrap metadata and six-step boundary trajectories are checked. Real mutation,
replacement, prevalidation, masked worlds, reset and complete snapshot replay
pass. The 70-DOF CSR assembly test explicitly materializes its validation view;
runtime stepping does not gain an implicit dense allocation.

The full original drawbridge and constant-release counterfactual each pass
**1,400 native steps**, with unchanged XML, controls and all original bounds.
Full snapshots replay literally for 20 prescribed steps at release step 900
and re-haul step 1,200, including retained precision/solver/actuator history.
Maximum linear-coordinate, scalar-joint and normalized rotation differences are
14.63 mm, 0.00707 rad and 0.10724 rad; activation differs by 4.05e-7.
Default soft-contact impact compression is 18.31 mm native versus 18.57 mm CPU;
this is not a 1 mm penetration qualification. The crate lifts to 0.535 m, slides
off at release and the deck re-raises empty. The [demo guide](examples/cable_drawbridge.md)
records these boundaries rather than claiming receiver delivery.

The new GIF has 70 actual comparison pairs over 2.8 simulated seconds at 1x.
A main-thread Mac GLFW viewer completes 45 native steps in three wall seconds
with CPU stepping forbidden; source CPU forward queries only mirror native
poses for upstream OpenGL visualization. Both visuals were reviewed.

Reproduce the complete native selection in the pinned source environment:

```sh
PYTHONPATH=metal:metal/tests:metal/examples MUJOCO_METAL_RUN_GPU=1 \
PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 \
PYTORCH_ENABLE_MPS_FALLBACK=0 PYTORCH_MPS_FAST_MATH=0 \
python -m pytest -q -x -p no:cacheprovider -m gpu \
  metal/tests/test_spatial_tendons_008.py \
  metal/tests/test_tendons.py \
  metal/tests/test_tendon_equality_metadata_ownership.py \
  metal/tests/test_contact_free_spatial_composition_015.py \
  metal/tests/test_tendon_dynamic_nv_017.py \
  metal/tests/test_spatial_shader_dependencies.py \
  metal/tests/test_spatial_limit_firstsolve_019.py \
  metal/tests/test_spatial_force_recovery_019.py \
  metal/tests/test_spatial_site_precision_008.py \
  metal/tests/test_spatial_wrap_transitions_008.py \
  metal/tests/test_site_armature_families.py \
  metal/tests/test_site_armature_closeout.py \
  metal/tests/test_fk_source_rounding.py \
  metal/tests/test_fk_position_words.py \
  metal/tests/test_fk_source_identity_precision.py \
  metal/tests/test_capacity_allocation_repair.py \
  metal/tests/test_capacity_017.py \
  metal/tests/test_position_context_geom_residual_018.py \
  metal/tests/test_forward_stages_019.py
```

The host phase uses the same files with `-m "not gpu"`, disabling MPS
availability before pytest collection. The full demo command is in its guide.
These results close the admitted spatial-tendon milestone; complete rigid
constraint composition is next in 009. Other solver options/integrators retain
014/015; APIs/extensions retain 019 and distribution/broad hardware retain 020.
No full repository regression, performance benchmark, publication or release
was performed.
