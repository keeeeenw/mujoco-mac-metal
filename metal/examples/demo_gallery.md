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
| [Sensor scanning rig](scanning_rig.md) | A pan/tilt scanner plots its measured pose and direction | Current-state frame, gyro and velocity sensors |
| [Cable plotter](cable_plotter.md) | Crossed tendons guide an XY pen around a figure eight | Fixed-joint tendon servos |
| [Clockwork automaton](clockwork_automaton.md) | Counter-rotating gears and a sequenced pawl | Polynomial joint equality, limits and dry joint friction |
| [Current-driven bodies](fluid_buoys.md) | Colorful shapes drift and rotate in a current | MuJoCo inertia-box fluid drag and viscosity |
| [Mechanical wave lattice](wave_lattice.md) | A motor drives a chain of colored articulated links | Bounded implicitfast integration |
| [Off-center balance workshop](offcenter_balance.md) | L-shaped tools spin around stationary centers of mass with measured orbit plots | Free-body implicitfast midpoint, including offset COMs |
| [Chaotic pendulum](README.md) | Four connected arms swing and tumble | Generalized rigid-body mass and bias |

Use the [development source setup](../INSTALL.md#development-source) for the
unreleased 0.4.0.dev0 demos; PyPI 0.3.0 does not include the new profiles.
Follow the command in each guide. Most new demos produce numerical reports
and optional GIFs; an interactive viewer is available only where documented. Rendering currently uses MuJoCo OpenGL.
