# Suspension platform (mixed rigid constraints)

Use the [shared source setup](../INSTALL.md#development-source), with its Python
environment active, and run these commands from the repository root.
This integrated demo requires current main source; it is not supported by the
published 0.4.0 wheel alone.

A mocap carrier tilts sinusoidally (±0.35 rad, 0.8 s period). A platform hangs
from four spatial-tendon cables (one travel-limited, one pair coupled by a
tendon equality) with a loose cargo crate riding aboard on frictional contact.
A ball-jointed pendulum strut, mounted alongside the platform, swings into its
angular-stop limit. Its offset mount keeps the strut and bob clear of the cargo. Ball limits,
tendon limits, tendon equality, dry contact and mocap motion all participate
in the same coupled solve. A paired level-hold run shows the limits never
spuriously engage. The simulation starts from the compiled `start` keyframe
via `sim.reset_to_keyframe`. Rendering uses MuJoCo OpenGL; playback speed is
presentation only, not a physics throughput claim. The 0.5 ms step keeps the
2.4 s demonstration duration and carrier period unchanged while limiting soft
stop excursion at the strut tip and cable.

Run a CPU reference smoke test:

```bash
PYTHONPATH=metal python metal/examples/suspension_platform.py --mode cpu --headless --steps 4800
```

Run the native Metal comparison check on an Apple Silicon Mac:

```bash
PYTORCH_ENABLE_MPS_FALLBACK=0 PYTHONPATH=metal:metal/examples python metal/examples/suspension_platform.py --headless --check --steps 4800
```

Record the side-by-side native Metal and CPU MuJoCo renders (left panel native,
right CPU):

```bash
PYTORCH_ENABLE_MPS_FALLBACK=0 PYTHONPATH=metal:metal/examples python metal/examples/suspension_platform.py --record metal/examples/assets/suspension_platform.gif --steps 4800
```

![Suspension platform](assets/suspension_platform.gif)

The `--headless --check` run asserts actual native behavior: native/CPU pose
parity, ball-stop engagement, cable-limit engagement, cargo contact on a majority of steps, platform
sway in the tilt run and near-stillness in the level run. Orientation parity
is split by physics: platform/cargo quats and the strut-axis tilt are gated
tightly, while ball-joint twist about the strut long axis is reported only
(it drifts between float32/float64 with no observable effect on the
axisymmetric strut/bob hardware). Native stop engagement is asserted on the
native strut peak itself, not just the CPU reference. Geometry-distance
checks verify that the strut and bob stay clear of both the platform and cargo
through the complete native trajectory. The reported stop overruns are also
measured independently: the strut's angular excess is converted to its 0.5 m
tip travel, and the cable's length is compared with the nominal 0.66 m limit.
The configured 0.2788 rad and 0.6578 m soft stops pre-compensate the CPU-measured
compliance; both measured nominal-limit overruns must stay under 1 mm.
For the native cable measurement, the demo copies the accepted Metal `qpos`
and current mocap pose into a scratch `MjData`, then runs only MuJoCo's
kinematics and tendon-length query. It does not advance that scratch state.
The JSON reports this `nat_cable_peak`/native overrun separately from the CPU
reference peak. Geometry overlap is also computed from the strict measured
distance at accepted native poses; the conservative nudge lower bound remains
a separate report.

Measured CPU trial results on MuJoCo 3.10.0 (native requalification pending):

- Strut peak angle: `0.27992` rad (configured soft stop `0.2788` rad; `0.58` mm
  tip overrun relative to the nominal `0.28` rad stop).
- Cable peak length: `0.65998` m (configured soft limit `0.6578` m; less than
  `0.1` mm overrun relative to the nominal `0.66` m limit).
- Cargo contact steps: `4702` of `4800`; cargo stays aboard (peak z `1.05` m).
- Swing-run platform sway vs level-run: `0.031` m vs `0.021` m lateral.

Lighting and the dark checkerboard match the robotic marble music machine.
