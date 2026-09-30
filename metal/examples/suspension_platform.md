# Suspension platform (mixed rigid constraints)

Use the [shared source setup](../INSTALL.md#development-source), with its Python
environment active, and run these commands from the repository root.
This integrated demo requires current main source; it is not supported by the
published 0.4.0 wheel alone.

A mocap carrier tilts sinusoidally (±0.35 rad, 0.8 s period). A platform hangs
from four spatial-tendon cables (one travel-limited, one pair coupled by a
tendon equality) with a loose cargo crate riding aboard on frictional contact.
A ball-jointed pendulum strut swings into its angular-stop limit. Ball limits,
tendon limits, tendon equality, dry contact and mocap motion all participate
in the same coupled solve. A paired level-hold run shows the limits never
spuriously engage. The simulation starts from the compiled `start` keyframe
via `sim.reset_to_keyframe`. Rendering uses MuJoCo OpenGL; playback speed is
presentation only, not a physics throughput claim.

Run a CPU reference smoke test:

```bash
PYTHONPATH=metal python metal/examples/suspension_platform.py --mode cpu --headless --steps 1200
```

Run the native Metal comparison check on an Apple Silicon Mac:

```bash
PYTORCH_ENABLE_MPS_FALLBACK=0 PYTHONPATH=metal:metal/examples python metal/examples/suspension_platform.py --headless --check --steps 1200
```

Record the side-by-side native Metal and CPU MuJoCo renders (left panel native,
right CPU):

```bash
PYTORCH_ENABLE_MPS_FALLBACK=0 PYTHONPATH=metal:metal/examples python metal/examples/suspension_platform.py --record metal/examples/assets/suspension_platform.gif --steps 1200
```

Launch the interactive viewer on a Mac:

```bash
PYTORCH_ENABLE_MPS_FALLBACK=0 PYTHONPATH=metal:metal/examples mjpython metal/examples/suspension_platform.py --steps 1200
```

![Suspension platform](assets/suspension_platform.gif)

The `--headless --check` run asserts actual native behavior: native/CPU pose
parity, ball-stop engagement (strut past 0.28 rad), cable-limit engagement
(cable length past 0.66 m), cargo contact on a majority of steps, platform
sway in the tilt run and near-stillness in the level run.

Measured CPU results on the qualified run (MuJoCo 3.10.0):

- Strut peak angle: `0.30` rad (stop at `0.28` rad, `396` engaged steps).
- Cable peak length: `0.68` m (limit at `0.66` m, `359` engaged steps).
- Cargo contact steps: `1170` of `1200`; cargo stays aboard (peak z `1.07` m).
- Swing-run platform sway vs level-run: `0.029` m vs `0.021` m lateral.
