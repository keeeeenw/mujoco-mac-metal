# Muscle gripper (antagonistic muscle actuation)

Use the [shared source setup](../INSTALL.md#development-source), with its Python
environment active, and run the commands below from the repository root.
This integrated demo requires development source; the published 0.4.0 wheel
alone does not provide it.

A mocap gantry carries two muscle-driven slide jaws. Each joint uses an
antagonistic flexor/extensor pair with MuJoCo muscle activation dynamics,
force-length-velocity gains and passive force biases. Thin, symmetric support
ledges carry a 20 g sphere and a 20 g capsule; this is a supported grasp, not a friction-only
pinch. Phase A delivers the red sphere to the green bin. Phase B delivers the
blue capsule to a pedestal in the second pickup lane. Both objects have a
55 mm radius; the horizontal capsule has a 40 mm centerline segment. Their positions advance through contact physics, with no payload
teleportation, weld or attachment toggle.

The 44-second schedule (22,000 steps at dt=0.002 s) acquires each object while
stationary, lifts it before transport, lowers it over its receiver, and then
relaxes the muscles. Ten-second horizontal carries limit acceleration to
0.048 m/s². The wider bin and separate pickup lanes keep the empty jaws clear
of the furniture. Slide travel is [-0.018, 0.04] m, giving the pads a 2 mm radial
gap around a centered payload at closure. The compiled start keyframe is shared
by the CPU and native paths. Capsule manifolds reserve 128 constraint rows;
the demo explicitly requests the 256-row block-solver budget rather than
truncating contacts to fit the small-model default.

Run the complete CPU reference delivery and geometry check:

```bash
PYTHONPATH=metal:metal/examples python metal/examples/muscle_gripper.py --mode cpu --headless --check
```

Run the complete native Metal comparison check on an Apple Silicon Mac:

```bash
PYTORCH_ENABLE_MPS_FALLBACK=0 PYTHONPATH=metal:metal/examples python metal/examples/muscle_gripper.py --headless --check
```

Record genuine side-by-side native Metal and CPU MuJoCo renders:

```bash
PYTORCH_ENABLE_MPS_FALLBACK=0 PYTHONPATH=metal:metal/examples python metal/examples/muscle_gripper.py --record metal/examples/assets/muscle_gripper.gif
```

Launch the native interactive viewer (requires a display):

```bash
PYTORCH_ENABLE_MPS_FALLBACK=0 PYTHONPATH=metal:metal/examples python metal/examples/muscle_gripper.py
```

Add `--viewer-seconds 20` to close after 20 seconds of wall time. Shorter
`--steps` values can inspect part of the motion, but `--check` requires the
complete schedule and refuses to certify an incomplete rollout. Rendering uses
MuJoCo OpenGL; presentation speed is not a physics throughput measurement.
Lighting and the checkerboard match the robotic marble music machine.

The complete native and CPU schedules pass delivery, furniture clearance and
unchanged 1 mm contact-penetration checks. Both native episodes ran all 22,000
steps, including the always-open counterfactual. Native peak contact penetration
was 0.880637 mm; CPU peak was 0.880697 mm. Native support contacts occurred on
14,299 steps and the red sphere reached its bin on 12,436 steps; the always-open
counterfactual recorded zero bin hits. Both payloads rose above 0.309 m. A full
checkpoint restored during the second carry replayed exactly, including retained
precision, actuator and solver state. Empty-jaw checks exercise all 16 corners
of the four-muscle [0,1] control box.

The comparison GIF contains 88 actual rendered frame pairs at 2 frames per
simulated second, covering the complete 44-second grasp schedule at 1x playback.
The main-thread GLFW viewer was separately checked for three wall seconds and
advanced 20 actual native steps without CPU physics stepping. Drag to orbit or
pan, scroll to zoom, and press Escape to close; a regular Python interpreter
runs this viewer without the mjpython launcher.

This delivery check does not establish identical long contact trajectories.
The maximum payload translation difference was 0.986 mm; capsule rotation
reached 0.115 rad and angular-velocity difference reached 2.35 rad/s during the
second pickup. Before contact, position error was 5.37e-8 and velocity error
1.83e-6. Activation error stayed below 3.22e-7. These comparisons separate
functional success and clipping checks from rotational trajectory agreement;
complete solver-option qualification remains a later milestone.

The report separates geometry-distance queries from actual solver contact
distances. Queries run on a separate CPU model and cover even pairs omitted
from collision filtering; they do not advance native physics. Output also
separates the unnudged query, a conservative lower bound, and the pose-sampling
uncertainty (at most 0.5 mm). Functional grasp/rest contacts must penetrate less
than 1 mm; compliant joint-limit overshoot is reported separately. Attached
parts intentionally meet at their mechanical joints.

In native mode, contact counts and contact penetration are read from the
actual native solver. The CPU oracle's contact counts are reported separately.
Checks require both payloads to lift and arrive at their receivers, and compare
with a paired always-open rollout that cannot make the deliveries. This is a
deterministic open-loop manipulation demonstration, not a qualified general
purpose grasp controller.
