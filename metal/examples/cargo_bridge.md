# Latch-and-release cargo bridge

Use the [shared source setup](../INSTALL.md#development-source), with its Python
environment active, and run these commands from the repository root.
This integrated demo requires current main source; it is not supported by the
published 0.4.0 wheel alone.

Two rigid deck sections are joined through a ball-like connect constraint at
midspan. DeckA is hinged to a fixed frame; deckB is free and latched to the
world frame through a weld brace. A payload sphere rests on the decks and
contacts the decks, a receiving tray and the floor. At a deterministic release
step the brace weld is deactivated through the public
`sim.set_equality_active` API; the connected sections articulate under gravity
and the payload slides into the tray. A paired always-latched run shows the
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

Launch the interactive viewer on a Mac:

```bash
./run mjpython metal/examples/cargo_bridge.py --steps 900
```

![Latch-and-release cargo bridge](assets/cargo_bridge.gif)

The `--headless --check` run asserts actual native behavior, not compilation:
latch (weld) force carries load while latched and is removed after release,
connect force stays engaged throughout, payload contacts the deck/tray, the
released payload reaches the tray while the always-latched counterfactual stays
on deck, and pre-release native/CPU pose parity holds. Full-run contact-release
trajectories diverge as documented for contact-rich scenes; the check asserts
physical outcomes for the full run and tight parity only over the stable
latched horizon.

Measured native results on the qualified run (MuJoCo 3.10.0, MPS float32):

- Pre-release parity (50 stable latched steps): qpos `1.93e-07`, qvel `8.77e-06`.
- Full-run maxima (1200 steps incl. release): qpos `1.33e-03`, qvel `3.75e-01`.
- Native latch force before release: `109.95`; after release: `0.0`.
- Native connect force before/after: `86.19` / `44.01`.
- Payload contact steps: `1075`; payload-tray contact steps released/latched: `1011` / `0`.
- Released payload `(x, z)`: (`0.699`, `0.230`) inside tray footprint; latched payload `(x, z)`: (`0.730`, `0.603`) up on deck.
- Native connect/weld anchor coincidence: `5.95e-04` m before release, `4.62e-04` m after (connect stays constrained); native/CPU anchor parity `1.15e-05` m.
