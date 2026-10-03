# Gravity assembly puzzle (convex mesh contact)

Use the [shared source setup](../INSTALL.md#development-source), with its Python
environment active, and run these commands from the repository root.
This integrated demo requires current main source; it is not supported by the
published 0.4.0 wheel alone.

Three procedural convex hulls (tetrahedron, brick, octahedron; authored
vertex lists) drop into a two-lane walled tray under gravity with Coulomb
friction: tetra settles in the left lane, brick in the right lane, octa
stacks on top of the brick. Piece/tray, piece/wall, piece/divider and
brick/octa contacts exercise the native bounded mesh manifolds (up to 4
face witnesses with sorted slot identity) against boxes and other meshes.
Six contact pairs x 4 slots fill the 24-slot/96-row budget exactly (see the
pair-budget comment in `assembly_puzzle.xml`); the pyramidal cone needs 120
rows on this scene and is rejected at admission, so the elliptic cone is a
capacity requirement (CPU lands identically under both cones). Rest
equilibria with friction are non-unique and tumbling orientations diverge
chaotically, so the check gates lane containment (measured max orientation
envelope 0.173 on the qualifying revision), impact-transient and settle
envelopes plus pre-impact parity, not exact rest poses. Omitted pairs
(octa/floor, tetra/divider and others the choreography never approaches)
are guarded by monitored trajectory bounds (tetra/divider gap, octa floor
clearance) that fail loudly instead of tunneling silently. Rendering uses
MuJoCo OpenGL; playback speed is presentation only, not a physics
throughput claim.

Run a CPU reference smoke test:

```bash
PYTHONPATH=metal python metal/examples/assembly_puzzle.py --mode cpu --headless --check --steps 600
```

Run the native Metal comparison check on an Apple Silicon Mac:

```bash
PYTORCH_ENABLE_MPS_FALLBACK=0 PYTHONPATH=metal:metal/examples python metal/examples/assembly_puzzle.py --headless --check --steps 600
```

Record the side-by-side native Metal and CPU MuJoCo renders (left panel native,
right CPU):

```bash
PYTORCH_ENABLE_MPS_FALLBACK=0 PYTHONPATH=metal:metal/examples python metal/examples/assembly_puzzle.py --record metal/examples/assets/assembly_puzzle.gif --steps 600
```

![Assembly puzzle](assets/assembly_puzzle.gif)

The `--headless --check` run asserts actual native behavior: all three pieces
settle inside their tray lanes below rim height, pre-impact flight parity is
exact, settled end positions agree to lane precision, and piece penetration
stays within the compliant impact bound. Multi-basin deep-rotated mesh-mesh
overlap is excluded from tight parity (matrix documents the regime).

Lighting and the dark checkerboard match the robotic marble music machine.
