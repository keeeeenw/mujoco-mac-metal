# Deformable cable cradle demo (Milestone 018)

This demo exercises native deformable physics on Apple Silicon MPS:
- 1D cable lowering from compiled `mjModel` flex structures.
- Bilateral distance equality constraints (`mjEQ_FLEX`) with exact analytic compliance and reference acceleration.
- Delassus block-row constraint solve on MPS coupled with generalized coordinate kinematics.
- Live state tracking (`flexedge_length`, `flexvert_xpos`) with zero CPU fallback during device stepping.

## Model description

A 6-element flexible cable (`count="6 1 1"`) is suspended from a rigid gantry frame. The cable vertices are initialized with a lateral offset to induce a wide swinging arc under gravity. As the cable swings, tension and bilateral equality constraints preserve edge rest lengths while accommodating multi-axis rotation and deformation.

## Verification

Run the headless verification check against the pinned MuJoCo CPU oracle:

```sh
./run python metal/examples/deformable_cable_cradle.py --headless --check
```

This verifies:
- Maximum generalized coordinate error $\le 2.0 \times 10^{-3}$ m across 300 steps.
- Genuine large-amplitude swinging dynamics ($> 0.08$ m excursion).
- Cable edge length compliance matching CPU reference within $\pm 0.015$ m.
