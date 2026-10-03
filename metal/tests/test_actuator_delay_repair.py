# Copyright 2026 The MuJoCo Metal contributors
# Licensed under the Apache License, Version 2.0.
"""R06c repair: actuator control-history (delay line) mechanism.

Failing-first: a pinned-exact host delay line (mju_historyInit/Insert/Read
plus the mj_readCtrl selection rule) with snapshot ownership. Step-loop
wiring and admission stay pending coordination with in-flight simulation
state work; guards are unchanged by this commit.
"""

import numpy as np
import pytest

from mujoco_metal.stateful_actuation import (
    ActuatorDelayLine, delay_reference,
)


def _fill(line, pairs):
  for t, v in pairs:
    line.insert(t, v)
  return line


def test_zero_state_reads_newest_cpu():
  line = ActuatorDelayLine(4, 0)
  assert line.read(0.0) == 0.0
  assert line.read(1.0) == 0.0


def test_zoh_exact_sequence_cpu():
  line = _fill(ActuatorDelayLine(4, 0), [(0.0, 1.0), (0.002, 3.0), (0.004, 5.0)])
  # A single insert into the zero-state buffer leaves stale zero slots;
  # pinned returns the oldest LOGICAL slot (still zero) for reads at or
  # before the oldest stamp, exactly like the C ring.
  assert _fill(ActuatorDelayLine(4, 0), [(0.002, 3.0)]).read(-1.0) == 0.0
  # Exact hit returns the sample.
  assert line.read(0.002) == 3.0
  # Hold: most recent sample <= t.
  assert line.read(0.003) == 3.0
  assert line.read(0.004) == 5.0
  # After newest: newest value.
  assert line.read(100.0) == 5.0


def test_linear_interpolation_cpu():
  line = _fill(ActuatorDelayLine(4, 1), [(0.0, 0.0), (0.004, 4.0)])
  assert line.read(0.001) == pytest.approx(1.0)
  assert line.read(0.003) == pytest.approx(3.0)


def test_cubic_hand_computed_cpu():
  # Samples (0,0),(1,1),(2,2),(3,3) fill slots 1..3 (slot 0 stays zero):
  # read at 1.5 brackets i=2 over [1,2]; slopes m_lo=(2-0)/2=1 (via the
  # stale slot), m_hi=(3-1)/2=1; alpha=0.5 weights (0.5,-0.375,0.5,-0.125):
  # 0.5*1 - 0.375*1 + 0.5*2 - 0.125*1 = 1.0, exactly like the C ring.
  line = _fill(ActuatorDelayLine(4, 2), [(0.0, 0.0), (1.0, 1.0),
                                         (2.0, 2.0), (3.0, 3.0)])
  assert line.read(1.5) == pytest.approx(1.0)
  # Step samples (0,0),(1,0),(2,10),(3,10): read at 1.5 gives
  # 0.5*0 - 0.375*5 + 0.5*10 - 0.125*5 = 2.5 (Catmull-Rom, verified
  # against the pinned Hermite basis by hand).
  line2 = _fill(ActuatorDelayLine(4, 2), [(0.0, 0.0), (1.0, 0.0),
                                          (2.0, 10.0), (3.0, 10.0)])
  assert line2.read(1.5) == pytest.approx(2.5)


def test_wraparound_and_eviction_cpu():
  line = ActuatorDelayLine(3, 0)
  for k, v in enumerate([1.0, 2.0, 3.0, 4.0, 5.0]):
    line.insert(0.001 * k, v)
  # Capacity 3 keeps the newest three samples.
  assert line.read(0.004) == 5.0
  assert line.read(0.002) == 3.0
  assert line.read(-1.0) == 3.0  # oldest retained


def test_out_of_order_insert_cpu():
  line = _fill(ActuatorDelayLine(4, 1), [(0.0, 0.0), (0.004, 4.0)])
  line.insert(0.002, 9.0)  # lands between, shifting out the oldest slot
  assert line.read(0.002) == pytest.approx(9.0)
  assert line.read(0.003) == pytest.approx((9.0 + 4.0) / 2)


def test_duplicate_time_overwrites_cpu():
  line = _fill(ActuatorDelayLine(4, 0), [(0.0, 1.0), (0.002, 2.0)])
  line.insert(0.002, 7.0)
  assert line.read(0.002) == 7.0
  assert line.read(100.0) == 7.0


def test_snapshot_restore_atomicity_cpu():
  line = _fill(ActuatorDelayLine(4, 1), [(0.0, 1.0), (0.002, 3.0)])
  snap = line.snapshot()
  before = (line.read(0.001), line.read(0.003))
  with pytest.raises(ValueError):
    line.restore({"nsample": 4, "interp": 1, "cursor": 0,
                  "times": np.full(4, np.nan), "values": np.zeros(4)})
  with pytest.raises(ValueError):
    line.restore({"nsample": 3, "interp": 1, "cursor": 0,
                  "times": np.zeros(3), "values": np.zeros(3)})
  # Rejections leave state untouched.
  assert (line.read(0.001), line.read(0.003)) == before
  line.insert(0.004, 9.0)
  assert line.read(0.004) == pytest.approx(9.0)
  line.restore(snap)
  assert (line.read(0.001), line.read(0.003)) == before
  fresh = ActuatorDelayLine(4, 1)
  fresh.restore(snap)
  assert (fresh.read(0.001), fresh.read(0.003)) == before


def test_delay_reference_shifts_signal_cpu():
  script = [(0.002 * k, 1.0 if k >= 2 else 0.0) for k in range(6)]
  # Zero-order hold, 2-step (4 ms) delay: output lags the step input.
  got = delay_reference(6, 0, 0.004, script)
  assert got == [0.0, 0.0, 0.0, 0.0, 1.0, 1.0]
  # No buffer reads live control.
  assert delay_reference(0, 0, 0.004, script) == [0.0, 0.0, 1.0, 1.0, 1.0, 1.0]


def test_invalid_config_cpu():
  with pytest.raises(ValueError):
    ActuatorDelayLine(-1, 0)
  with pytest.raises(ValueError):
    ActuatorDelayLine(4, 3)
  with pytest.raises(ValueError):
    ActuatorDelayLine(0, 0).read(0.0)
  with pytest.raises(ValueError):
    ActuatorDelayLine(4, 0).insert(np.nan, 1.0)
  with pytest.raises(ValueError):
    ActuatorDelayLine(4, 0).read(np.inf)


def _needs_gpu():
  import os
  return pytest.mark.skipif(os.getenv("MUJOCO_METAL_RUN_GPU") != "1", reason="opt-in GPU")


@_needs_gpu()
@pytest.mark.parametrize("interp", [0, 1, 2])
def test_device_delay_matches_host_reference_gpu(interp):
  import os
  assert os.getenv("MUJOCO_METAL_RUN_GPU") == "1"
  from mujoco_metal.stateful_actuation import MetalDelayLine
  import torch
  nu, b, n = 3, 2, 6
  dev = MetalDelayLine([n] * nu, [interp] * nu, [0.004, 0.0, 0.008], batch_size=b)
  rng = np.random.default_rng(11)
  script = [(0.002 * k, rng.normal(size=(b, nu))) for k in range(10)]
  host = [ActuatorDelayLine(n, interp) for _ in range(b * nu)]
  for t, u in script:
    dev.record(u, t)
    for w in range(b):
      for i in range(nu):
        host[w * nu + i].insert(t, float(u[w, i]))
    got = dev.read(u, t).cpu().numpy()
    for w in range(b):
      for i, dly in enumerate([0.004, 0.0, 0.008]):
        want = host[w * nu + i].read(t - dly)
        assert got[w, i] == pytest.approx(want, abs=1e-5), (interp, t, w, i)


@_needs_gpu()
def test_device_delay_snapshot_restore_gpu():
  import os
  assert os.getenv("MUJOCO_METAL_RUN_GPU") == "1"
  from mujoco_metal.stateful_actuation import MetalDelayLine
  dev = MetalDelayLine([4, 0], [1, 0], [0.002, 0.0], batch_size=1)
  dev.record(np.array([[1.0, 2.0]]), 0.0)
  dev.record(np.array([[3.0, 4.0]]), 0.002)
  snap = dev.snapshot()
  ref = dev.read(np.array([[3.0, 4.0]]), 0.002).cpu().numpy().copy()
  dev.record(np.array([[9.0, 9.0]]), 0.004)
  assert not np.allclose(dev.read(np.array([[9.0, 9.0]]), 0.004).cpu().numpy(), ref)
  dev.restore(snap)
  np.testing.assert_allclose(dev.read(np.array([[3.0, 4.0]]), 0.002).cpu().numpy(), ref)
  bad = dict(snap)
  bad["values"] = np.full_like(snap["values"], np.nan)
  with pytest.raises(ValueError):
    dev.restore(bad)
  # Rejection preserves device state.
  np.testing.assert_allclose(dev.read(np.array([[3.0, 4.0]]), 0.002).cpu().numpy(), ref)
  dev.reset()
  np.testing.assert_allclose(dev.read(np.array([[0.0, 7.0]]), 0.0).cpu().numpy(),
                             np.array([[0.0, 7.0]]), atol=1e-6)
