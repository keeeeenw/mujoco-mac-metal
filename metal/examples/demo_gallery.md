# Explore the Mac physics demos

Each demo introduces a different physical capability. These are actual
simulations with CPU MuJoCo reference checks; the clips are not benchmarks.

| Demo | What to watch | Native capability |
|---|---|---|
| [Spacecraft approach](space_docking.md) | Three craft chase moving markers with different damping | Applied generalized forces and Euler damping |
| [Spring flower](kinetic_sculpture.md) | Four petals fold and respond to a breeze | Springs, damping, gravity compensation and Cartesian forces |
| [Tumbling toys](tumbling_toys.md) | Colorful asymmetric objects spin around three different axes | Quaternion-aware RK4 |
| [Marble cascade](marble_cascade.md) | Three marbles collide on a tilted plane | Normal plane–sphere and sphere–sphere contact |
| [Friction laboratory](friction_laboratory.md) | Low, medium, and high friction turn sliding into rolling at different rates | Pyramidal condim-3 contact |
| [Spin-and-Grip arcade](spin_and_grip.md) | A rolling sphere slows, a spinning sphere loses spin, and an actuated press holds and releases a sliding block | Integrated elliptic condim-4/6 torsional and rolling friction; condim-3 press contact |
| [Sensor scanning rig](scanning_rig.md) | A pan/tilt scanner plots its measured pose and direction | Current-state frame, gyro and velocity sensors |
| [Cable plotter](cable_plotter.md) | Crossed tendons guide an XY pen around a figure eight | Fixed-joint tendon servos |
| [Clockwork automaton](clockwork_automaton.md) | Counter-rotating gears and a sequenced pawl | Polynomial joint equality, limits and dry joint friction |
| [Current-driven bodies](fluid_buoys.md) | Colorful shapes drift and rotate in a current | MuJoCo inertia-box fluid drag and viscosity |
| [Mechanical wave lattice](wave_lattice.md) | A motor drives a chain of colored articulated links | Bounded implicitfast integration |
| [Off-center balance workshop](offcenter_balance.md) | L-shaped tools spin around stationary centers of mass with measured orbit plots | Free-body implicitfast midpoint, including offset COMs |
| [Robotic marble music machine](marble_music_machine.md) | Dual selector gates and resonant chime bars struck by descending marbles | Integrated Euler pipeline: coupled constraints, contacts, limits, equality, tendons, actuation, fluid, and sensors |
| [Clockwork parcel sorter](clockwork_parcel_sorter.md) | Real box parcels, capsule rollers, and swinging diverter gates routing parcels to bins | Complete primitive collisions: multi-contact manifolds across planes, spheres, capsules, boxes, coupled Euler solve |
| [Latch-and-release cargo bridge](cargo_bridge.md) | Two deck sections joined by a connect constraint unlatch a weld brace and spill a payload into a tray | Connect/weld equalities with per-environment activation, coupled contact solve |
| [Magnetic crane](magnetic_crane.md) | A mocap hook carries a welded cargo box to a bin and releases it | Per-environment mocap inputs, weld attach/release, keyframe reset |
| [Muscle-powered gripper](muscle_gripper.md) | Palm-cage grasp, bin delivery and pedestal placement | Stateful actuation and contact-rich manipulation |
| [Cable drawbridge](cable_drawbridge.md) | A routed cable hauls, goes slack and re-tensions a loaded deck | Spatial tendon wrapping, pulley division and spring/damping forces |
| [Suspension platform](suspension_platform.md) | A suspended platform and payload | Tendon constraints and ball-joint limits |
| [Chaotic pendulum](README.md) | Four connected arms swing and tumble | Generalized rigid-body mass and bias |

Use current `main` and the [demo/source setup](../INSTALL.md#development-source).
Spin-and-Grip, the parcel sorter and the marble music machine require the source
physics package; the published 0.4.0 wheel does not include integrated Euler.
Demo scripts, XML models and GIFs live in the repository.
Follow the command in each guide. Most new demos produce numerical reports
and optional GIFs; an interactive viewer is available only where documented. Rendering currently uses MuJoCo OpenGL.
