# Scalable marble feed

This fixture runs fifteen isolated marble lanes in one model. Each rolling
marble stays in its own three-sided chute; lane-specific collision groups
exclude cross-lane pairs while retaining every physically possible floor and
rail contact. The 1 mm contact margin supplies stable contact rows at exact
geometric tangency, with no initial penetration.

The compiled model has 90 DOFs, 45 broad-phase pairs, 45 contact slots, and 390
allocated constraint rows. A 30-step pinned CPU run reaches 45 active pairs,
45 contacts, and 270 active rows. These counts clear the former 64-DOF,
16-pair, 24-slot, and 256-row limits without the quadratic candidate and row
growth of an all-to-all marble pile. The current conservative block-sparse
workspace estimate is 2,084,968 bytes, within the demo's explicit 1 GiB limit.

Run the CPU oracle and capacity assertions with:

```sh
python metal/examples/scalable_marble_factory.py --steps 30 --check
```

Use `--mode metal` to run the same public integrated profile on an authorized
native backend. That mode checks native broad-phase/contact/active-row peaks
separately from the pinned counts, reports position, velocity, acceleration,
and orientation errors separately, and can verify exact replay with
`--restore-check`. It has not yet been native-qualified.
