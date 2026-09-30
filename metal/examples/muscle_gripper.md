# Muscle gripper (antagonistic muscle actuation)

Use the [shared source setup](../INSTALL.md#development-source), with its Python
environment active, and run these commands from the repository root.
This integrated demo requires current main source; it is not supported by the
published 0.4.0 wheel alone.

A mocap gantry carries two muscle-driven fingers. Each hinge joint is driven by
an antagonistic flexor/extensor pair using MuJoCo muscle dynamics
(`dyntype="muscle"` with activation state), force-length-velocity gains
(`gaintype="muscle"`) and passive force biases (`biastype="muscle"`).
Phase A grasps a red sphere (r=0.055, 0.25 kg), carries it over a walled bin
and sets it down inside. Phase B grasps a blue sphere (r=0.055, 0.15 kg),
carries it to a pedestal and sets it down on top. Gantry motion and finger
schedules are deterministic open-loop inputs identical in CPU/native runs
(2700 steps, dt=0.002 s). A paired always-open run shows the physical effect
of grasping. The palm pad is a contact surface (paired with both balls),
so the grasp is a three-sided cage (two fingers + palm) rather than a
friction-only pinch: this keeps the long open-loop carry deterministic under
millimeter perturbations on both CPU and native. The simulation starts from the compiled `start` keyframe via
`sim.reset_to_keyframe`. Rendering uses MuJoCo OpenGL; playback speed is
presentation only, not a physics throughput claim.

Run a CPU reference smoke test:

```bash
PYTHONPATH=metal python metal/examples/muscle_gripper.py --mode cpu --headless --steps 2700
```

Run the native Metal comparison check on an Apple Silicon Mac:

```bash
PYTORCH_ENABLE_MPS_FALLBACK=0 PYTHONPATH=metal:metal/examples python metal/examples/muscle_gripper.py --headless --check --steps 2700
```

Record the side-by-side native Metal and CPU MuJoCo renders (left panel native,
right CPU):

```bash
PYTORCH_ENABLE_MPS_FALLBACK=0 PYTHONPATH=metal:metal/examples python metal/examples/muscle_gripper.py --record metal/examples/assets/muscle_gripper.gif --steps 2700
```

Launch the interactive viewer on a Mac (requires a display; headless
verification uses `--headless --check` plus the GIF below):

```bash
./run mjpython metal/examples/muscle_gripper.py --steps 2700
```

Add `--viewer-seconds 20` to auto-close after 20 s of wall time for explicit
visual checks. The viewer steps the native simulation with the deterministic
schedule and reports native/CPU divergence live.

![Muscle gripper](assets/muscle_gripper.gif)

The `--headless --check` run asserts actual native behavior: muscle activation
state advances on device, finger contacts occur on grasp steps, the released
ball lands inside the bin footprint while the always-open counterfactual stays
outside, and the second ball lands on the pedestal top.

Measured results on the qualified palm-cage run (MuJoCo 3.10.0, native
`--headless --check --steps 2700`, exit 0):

- Ball delivered `(x, z)`: (`0.673`, `0.135`) inside bin footprint; always-open `(x, z)`: (`-3.667`, `0.055`) on floor.
- Second ball placed `(x, z)`: (`-0.133`, `0.255`) on pedestal top (z=0.2); always-open stays at (`0.300`, `0.055`).
- Ball-bin contact steps: `1600`; finger contact steps: `1463`.
- Pre-grasp native parity: qpos `7.4e-08`, qvel `1.4e-05`.
- Full-run native/CPU parity on focused actuator fixtures (see milestone 007
  report): integrator/filter/filter-exact qpos `1.6e-07`, muscle `2e-03`.

Superseded gap (preserved in the private archive): the pre-palm-cage
friction-only pinch diverged during the contact-rich carry (GPU ejected the
first ball; `results/milestone-007/failures/gpu-demo-divergence.log`). The
palm contact cage (three-sided grasp) resolved the chaos: CPU ±2 mm
perturbations now deliver identically. The demo is CPU- and
natively qualified; `assets/muscle_gripper.gif` shows the passing run.
