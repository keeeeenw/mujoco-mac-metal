# Explore the Mac physics demos

Each demo introduces a different physical capability. These are actual
simulations with CPU MuJoCo reference checks; the clips are not benchmarks.

| Demo | What to watch | Native capability |
|---|---|---|
| [Spacecraft approach](space_docking.md) | Three craft chase moving markers with different damping | Applied generalized forces and Euler damping |
| [Tumbling toys](tumbling_toys.md) | Colorful asymmetric objects spin around three different axes | Quaternion-aware RK4 |
| [Chaotic pendulum](README.md) | Four connected arms swing and tumble | Generalized rigid-body mass and bias |

The development demos may require newer source than the latest PyPI release.
Follow the command in each guide. Rendering currently uses MuJoCo OpenGL.
