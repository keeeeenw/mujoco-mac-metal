# Control and sensor histories

The development backend uses the compiled MuJoCo 3.10.0 `mjData.history`
layout for actuator controls and sensor samples. History is device resident;
there is no second simulation-owned delay ring. This implements existing MuJoCo
semantics, rather than a new filtering method.

Compile `nsample`, interpolation, delay and sensor interval/phase through XML or
`MjSpec` before creating a simulation. Changing the sample count or address on an
already compiled model does not allocate storage and is rejected.

Controls with positive delay read the canonical ring at the evaluation time minus
delay. Controls with zero delay use the live held control, including when history
recording is enabled. Sensor histories support zero-order hold, linear and cubic
interpolation, delayed reads and interval sampling. Successful steps record held
controls and raw forward-stage sensor samples at the pre-integration timestamp.
Failed worlds do not advance their histories. Snapshot/restore, environment copy,
selected reset and `native_api.mj_setState(..., {"history": ...})` share this storage.

`step_sensordata()` returns the stored step-stage sample. `sensor_values()` performs
a current-state forward query with the configured delayed/held read semantics;
queries do not insert history. Ordinary queries do not advance the simulation.

History values and public state timestamps are float32. Compiled interval/phase
and delay constants retain a low part for boundary decisions: simply rounding an
interval can otherwise insert a sample one step early. Qualification uses the same
float32 input timestamp for the pinned CPU and native stage. This does not establish
identical sampling decisions between two long trajectories with different clock
rounding. Pinned 3.10.0 retains `sensor_noise` metadata but its engine does not add
random sensor noise; this backend follows that behavior.

Run the focused qualification from the source package:

```sh
MUJOCO_METAL_RUN_GPU=1 python -m pytest -q \
  tests/test_history_parity.py tests/test_actuator_delay_repair.py
```

The development implementation passed 34 focused checks on the available Apple
Silicon host with GPU opt-in enabled. They include pinned interpolation/recording,
interval phase boundaries, integration and RK4 delay onset, reset/copy/replay,
invalid snapshots and failed-world isolation. This is focused evidence; the full
sensor, extension and cross-feature qualification remains a separate gate.
