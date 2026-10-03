# Latch-and-release cargo bridge

Use the [shared source setup](../INSTALL.md#development-source), with its Python
environment active, and run these commands from the repository root.
This integrated demo requires current main source; it is not supported by the
published 0.4.0 wheel alone.

Two rigid deck sections are joined through a ball-like connect constraint at
midspan. DeckA is hinged to a fixed frame; deckB is free and latched to the
world frame through a weld brace. A payload sphere rests on the decks and
contacts the decks. A solid landing stop supports the released deck. At a deterministic release
step the brace weld is deactivated through the public
`sim.set_equality_active` API; the connected sections articulate under gravity
and the deck lands on the stop while the payload shifts along the bridge. A paired always-latched run shows the
physical effect of release. The release schedule is identical in CPU and native
runs. Rendering uses MuJoCo OpenGL; playback speed is presentation only, not a
physics throughput claim.

Run a CPU reference smoke test:

```bash
PYTHONPATH=metal python metal/examples/cargo_bridge.py --mode cpu --headless --steps 1200
```

Run the native Metal comparison check on an Apple Silicon Mac:

```bash
PYTORCH_ENABLE_MPS_FALLBACK=0 PYTHONPATH=metal:metal/examples python metal/examples/cargo_bridge.py --headless --check --steps 1200
```

Record the side-by-side native Metal and CPU MuJoCo renders (labels show
`LATCHED`/`RELEASED` state and payload position; left panel native, right CPU):

```bash
PYTORCH_ENABLE_MPS_FALLBACK=0 PYTHONPATH=metal:metal/examples python metal/examples/cargo_bridge.py --record metal/examples/assets/cargo_bridge.gif --steps 1200
```

![Latch-and-release cargo bridge](assets/cargo_bridge.gif)

The `--headless --check` run asserts native behavior: the weld carries load
while latched and is removed after release; connect forces remain engaged;
the deck contacts its landing stop; and the payload drops with the articulated
bridge while the always-latched comparison stays elevated. Tight pose parity
is checked over the stable latched horizon; full-run contact-release divergence
is reported separately.

The stop has an explicit deck contact pair. The previous receiving-tray layout
allowed the deck to pass through the tray; that is no longer treated as payload
delivery. A geometry-distance check covers the complete native trajectory,
allowing at most 7 mm of transient compliant impact penetration into the stop.
The CPU regression suite independently exercises the same schedule and bound.
This is a soft-contact simulation, not a claim of exact zero penetration.

Lighting and the dark checkerboard match the robotic marble music machine.
