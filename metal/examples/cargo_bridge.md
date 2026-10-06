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
PYTHONPATH=metal python metal/examples/cargo_bridge.py --mode cpu --headless --steps 2400
```

Run the native Metal comparison check on an Apple Silicon Mac:

```bash
PYTORCH_ENABLE_MPS_FALLBACK=0 PYTHONPATH=metal:metal/examples python metal/examples/cargo_bridge.py --headless --check --steps 2400
```

Record the side-by-side native Metal and CPU MuJoCo renders (labels show
`LATCHED`/`RELEASED` state and payload position; left panel native, right CPU):

```bash
PYTORCH_ENABLE_MPS_FALLBACK=0 PYTHONPATH=metal:metal/examples python metal/examples/cargo_bridge.py --record metal/examples/assets/cargo_bridge.gif --steps 2400
```

![Latch-and-release cargo bridge](assets/cargo_bridge.gif)

The `--headless --check` run asserts native behavior: the weld carries load
while latched and is removed after release; connect forces remain engaged;
the deck contacts its landing stop; and the payload drops with the articulated
bridge while the always-latched comparison stays elevated. Tight pose parity
is checked over the stable latched horizon; full-run contact-release divergence
is reported separately.

The stop has an explicit deck contact pair. The bridge now uses a 1 ms step
and a 2 ms normal-contact time constant; release remains at 0.1 s (step 100).
The CPU trajectory reaches the stop with 0.88 mm of measured geometric
penetration in the CPU trial. The independent conservative distance lower
bound includes the monitor's 0.5 mm uncertainty and is gated at 1.5 mm.
An independently checked isolated development composition now passes the full
2,400-step native schedule and always-latched comparison: maximum overlap at
accepted native poses is 0.883 mm, with the conservative lower bound at
−1.383 mm. The reported latch-row force norm falls from 107.99 before release to zero
after release, while the connect constraint continues carrying load. This norm
combines weld row components and is not a force in newtons. The original
numerical and outcome bounds are retained. This result does not qualify every
current development change or refresh the displayed GIF. The rollout remains
2.4 s long, so event timing and scene duration are unchanged.
The JSON reports `max_actual_deck_tray_overlap_m` from the strict geometry
distance at each accepted state (native qpos in Metal mode); this metric stays
separate from the uncertainty-adjusted lower bound.

Lighting and the dark checkerboard match the robotic marble music machine.
