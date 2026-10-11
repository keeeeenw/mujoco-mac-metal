# Development milestone status

The source backend is experimental. **0 of the 16 full-coverage milestones
005–020 are accepted as complete.** This is a full-composition acceptance count,
not a count of functioning features. Earlier bounded profiles and selected
native fixtures have passed independent checks.

| Milestone | Area | Current status |
|---|---|---|
| 005 | Pinned inventory and support contract | Catalogue implemented; execution mapping and contract reconciliation open |
| 006 | State, model and lifecycle | Transactional and cache regressions pass in selected compositions; final full lifecycle gate open |
| 007 | Actuators and transmissions | Built-in and extension paths implemented; complete combination and demo qualification open |
| 008 | Spatial tendons | Wrapping and admitted armature paths implemented; public assembly and final composition gate open |
| 009 | Rigid constraints | Coupled constraint paths implemented; full mixed-feature qualification open |
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

The latest local development host selection at `ed74d3fa0` passed **2,460 tests, 99 skipped**;
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
COM/spatial/recovery/invariant regressions. Original force, derivative,
trajectory and replay bounds remain unchanged.
These are source-specific results; full heightfield/deformable composition,
final wheel/demo checks and milestones 005–020 remain open.

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
