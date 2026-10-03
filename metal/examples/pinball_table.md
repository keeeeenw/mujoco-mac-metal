# Pinball inspection table

Instrumented pinball table for milestone 016: two free balls drop onto force
pads while motor-driven capsule flippers sweep. Touch pads report contact
forces, body sites report acceleration, and a downward range array scans ball
heights. The overlay displays live `touchA`, body-A acceleration magnitude,
and range-0 distance every frame.

Model: `pinball_table.xml` (integrated_euler_v1, pyramidal cone, 14 DoF:
2 hinges + 2 free balls). Sensors: 2 touch, 2 accelerometers, 3 rangefinders,
2 jointpos, 1 actuatorfrc, 1 jointlimitfrc. Walls are decorative
(contype 0); contact masks keep candidate rows within the 96-row budget
(floor/balls/flippers only, no flipper-flipper pairs).

Verification (600 steps, dt 0.002):
- CPU (`--mode cpu`): qpos/qvel/sensor errors 0 by construction; activity
  gates (touch peak 82.4, accel peak 274.7, range hits 100%).
- Native (`--mode metal`): max qpos error 3.7e-07, max qvel error 7.7e-06,
  max stored-sensor error 1.2e-03; same activity gates.
- Stored `step_sensordata()` matches `mj_step` timing; `sensor_values()`
  re-evaluates at the post-step state for the overlay (see REPORT).

Run:
```sh
./run --cpu python metal/examples/pinball_table.py --headless --check --steps 600 --mode cpu
./run python metal/examples/pinball_table.py --headless --check --steps 600 --mode metal
./run python metal/examples/pinball_table.py --headless --steps 300 --mode metal --record results/milestone-016/demo-evidence/pinball_table.gif
```

Known gaps: the FORCE body-wrench sensor is exercised by
`test_sensor_force_016.py`, not this demo (see REPORT for a multi-pair
cfrc assembly gap tracked as a follow-up). SDF rays/geomdist, camera/
tactile/plugin/user sensors stay 019-owned.
