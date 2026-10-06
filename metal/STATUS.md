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
| 013 | SDF collisions | Native providers and search paths implemented; original mesh/bowl/torus correctness gates open |
| 014 | Solvers | Selected Newton, warm-start and coupled regressions pass; full solver-option matrix open |
| 015 | Integrators and forces | Euler/RK4/implicit paths and derivatives implemented; full force/contact/material matrix open |
| 016 | Sensors and queries | Expanded native stages and queries implemented; complete timing/history/frame matrix open |
| 017 | Capacity, sparse execution and sleep | Model-derived storage and larger profiles implemented; all-boundary/native composition gate open |
| 018 | Deformables | Material, interpolation, attachment and collision paths implemented; complete geometry/integrator/lifecycle matrix open |
| 019 | APIs and extensions | Expanded package APIs and bundled adapters implemented; full native/host and extension composition gate open |
| 020 | Distribution and demos | Source demos and packaging support present; final clean-wheel/native/demo gates open |

## Evidence and limitations

The current host publication check passed **2,163 tests, 1,615 skipped**.
The skipped native cases remain unqualified by this run; see
[the recorded scope](QUALIFICATION.md).

Selected source compositions passed 94 Newton regression checks (including the
original 300-step slope fixture), 69 contact-override checks and 11 retained
SDF arithmetic/source-trace checks. These are distinct regression groups, not a
complete backend suite. Earlier terrain/cache, flex and API results retain their
producing-composition limits; see [development coverage](DEVELOPMENT.md).

The original rigid-SDF group remained incomplete after a 300-second deadline,
with failures in mesh witness multiplicity and bowl/torus force or trajectory
comparisons. More recent mesh arithmetic changes are still under qualification
and are not included in this publication. The newer 180-second combined check
also did not finish; a timeout is not a passing result.

[Qualification](QUALIFICATION.md) separates publication checks from historical
native evidence. [Coverage](COVERAGE.md) describes admitted paths and restrictions.
No full compatibility, current-wheel qualification or new performance claim is
made. MuJoCo's dynamics and numerical methods are upstream work; this project
implements those methods for the optional Metal backend.
