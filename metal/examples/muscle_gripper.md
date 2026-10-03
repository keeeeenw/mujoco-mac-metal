# Muscle gripper (antagonistic muscle actuation)

Use the [shared source setup](../INSTALL.md#development-source), with its Python
environment active, and run these commands from the repository root.
This integrated demo requires current main source; it is not supported by the
published 0.4.0 wheel alone.

A mocap gantry carries two muscle-driven fingers. Each slide joint is driven by
an antagonistic flexor/extensor pair using MuJoCo muscle dynamics
(`dyntype="muscle"` with activation state), force-length-velocity gains
(`gaintype="muscle"`) and passive force biases (`biastype="muscle"`).
Phase A grasps a red sphere (r=0.055, 0.25 kg), carries it over a walled bin
and sets it down inside. Phase B grasps a blue sphere (r=0.055, 0.15 kg),
carries it to a pedestal and sets it down on top. Gantry motion and finger
schedules are deterministic open-loop inputs identical in CPU/native runs
(3000 steps, dt=0.002 s). A paired always-open run shows the physical effect
of grasping. The parallel jaws have small supporting ledges, so carrying does
not depend on a friction-only pinch. Release heights clear the bin walls and
pedestal, and the pedestal sits outside the first pickup's jaw sweep. Its
sphere contact includes rolling resistance to keep the deposited ball from
rolling off. The simulation starts from the compiled `start` keyframe via
`sim.reset_to_keyframe`. Rendering uses MuJoCo OpenGL; playback speed is
presentation only, not a physics throughput claim.

Run a CPU reference smoke test:

```bash
PYTHONPATH=metal python metal/examples/muscle_gripper.py --mode cpu --headless --steps 3000
```

Run the native Metal comparison check on an Apple Silicon Mac:

```bash
PYTORCH_ENABLE_MPS_FALLBACK=0 PYTHONPATH=metal:metal/examples python metal/examples/muscle_gripper.py --headless --check --steps 3000
```

Record the side-by-side native Metal and CPU MuJoCo renders (left panel native,
right CPU):

```bash
PYTORCH_ENABLE_MPS_FALLBACK=0 PYTHONPATH=metal:metal/examples python metal/examples/muscle_gripper.py --record metal/examples/assets/muscle_gripper.gif --steps 3000
```

Launch the interactive viewer on a Mac (requires a display; headless
verification uses `--headless --check` plus the GIF below):

```bash
PYTORCH_ENABLE_MPS_FALLBACK=0 PYTHONPATH=metal:metal/examples mjpython metal/examples/muscle_gripper.py --steps 3000
```

Add `--viewer-seconds 20` to auto-close after 20 s of wall time for explicit
visual checks (without it, the viewer runs the finite `steps` rollout, then
closes; interactive visual validation additionally requires a display and is
otherwise unverified here). The viewer steps the native simulation with the deterministic
schedule and reports native/CPU divergence live.

![Muscle gripper](assets/muscle_gripper.gif)

The `--headless --check` run asserts actual native behavior: muscle activation
state advances on device, finger contacts occur on grasp steps, the released
ball lands inside the bin footprint while the always-open counterfactual stays
outside, and the second ball lands on the pedestal top.

The demo checks the moving palm, jaws and ledges against the floor, bin and
pedestal throughout the native rollout, using geometry-distance queries on a
separate CPU model. Those queries also cover pairs omitted from collision
filtering; they do not advance the native physics. The CPU regression suite
replays the complete schedule and verifies both deliveries and furniture
clearance. Jaw self-pairs (finger-finger, ledge-ledge, ledge-opposite-jaw)
clear by 29 mm or more at every sampled step; the symmetric fully-closed
empty pose would intersect and is outside the schedule envelope (muscle
closure is always asymmetric or ball-separated). Attached jaw parts
intentionally meet at their mechanical joints. Slide limits act as compliant
end-stops: the grasp transient overshoots the soft limits by up to ~88 mm
of travel, inside the verified 150 mm envelope; this is limit-force
equilibrium, not runaway.

Lighting and the dark checkerboard match the robotic marble music machine.
This is a deterministic open-loop manipulation demo, not a qualified general
purpose grasp controller or a friction-only pinch benchmark.
