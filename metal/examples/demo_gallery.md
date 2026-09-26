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
| [Chaotic pendulum](README.md) | Four connected arms swing and tumble | Generalized rigid-body mass and bias |

The development demos may require newer source than the latest PyPI release.
Follow the command in each guide. Rendering currently uses MuJoCo OpenGL.
