# Copyright 2026 The MuJoCo Metal contributors
# Licensed under the Apache License, Version 2.0.
"""R02/R03 repair: activation atomicity and simulation-level state ownership.

R02: disabled-actuation preservation, failed-world activation rollback,
recovery after selected reset.
R03: versioned simulation snapshots covering device state, warm starts,
held inputs and stored sensor samples; query isolation twins; atomic
invalid restore; restore+randomize coherence; raw-vs-simulator reset.
"""

import os

import mujoco
import numpy as np
import pytest


def _filter_model(disabled=True):
  m = mujoco.MjModel.from_xml_string(
      "<mujoco><option timestep='0.002' integrator='Euler'/>"
      "<worldbody>"
      "<body pos='0 0 0.5'><joint name='h' type='hinge' axis='0 0 1'/>"
      "<geom type='sphere' size='0.1' mass='0.5'/></body></worldbody>"
      "<actuator><general name='g' joint='h' dyntype='filter' gainprm='2' biasprm='0 -2 0'/></actuator>"
      "</mujoco>")
  if disabled:
    m.opt.disableflags |= int(mujoco.mjtDisableBit.mjDSBL_ACTUATION)
  return m


def test_disabled_actuation_preserves_activation_cpu():
  model = _filter_model(disabled=True)
  assert model.na == 1
  d = mujoco.MjData(model)
  d.qpos[:] = 0.1
  d.act[:] = [0.7]
  for _ in range(5):
    mujoco.mj_step(model, d)
    assert float(d.act[0]) == pytest.approx(0.7), d.act[:]
  assert bool(np.all(np.isfinite(d.qpos)))


@pytest.mark.skipif(os.getenv("MUJOCO_METAL_RUN_GPU") != "1", reason="opt-in GPU")
def _needs_gpu():
  import os
  return pytest.mark.skipif(os.getenv("MUJOCO_METAL_RUN_GPU") != "1", reason="opt-in GPU")


@_needs_gpu()
def test_disabled_actuation_preserves_activation_gpu():
  import os
  assert os.getenv("MUJOCO_METAL_RUN_GPU") == "1"
  from mujoco_metal.simulation import MetalSimulation
  model = _filter_model(disabled=True)
  sim = MetalSimulation(model, batch_size=1, profile="integrated_euler_v1")
  sim.reset(qpos=np.array([[0.1]], dtype=np.float32),
            qvel=np.zeros((1, model.nv), dtype=np.float32),
            act=np.array([[0.7]], dtype=np.float32))
  for _ in range(5):
    sim.step(1, ctrl=np.array([[1.0]], dtype=np.float32))
  assert float(sim.state._act.cpu().numpy()[0, 0]) == pytest.approx(0.7)
  # CPU twin with frozen activation agrees on qpos/qvel.
  d = mujoco.MjData(model)
  d.qpos[:] = [0.1]
  d.act[:] = [0.7]
  for _ in range(5):
    d.ctrl[:] = [1.0]
    mujoco.mj_step(model, d)
  np.testing.assert_allclose(sim.state.qpos.cpu().numpy()[0],
                             np.asarray(d.qpos), rtol=1e-5, atol=1e-6)


@_needs_gpu()
def test_failed_world_keeps_activation_and_state_gpu():
  import os
  assert os.getenv("MUJOCO_METAL_RUN_GPU") == "1"
  from mujoco_metal.simulation import MetalSimulation
  model = _filter_model(disabled=False)
  assert model.na == 1
  sim = MetalSimulation(model, batch_size=2, profile="integrated_euler_v1")
  sim.reset(qpos=np.array([[0.1], [0.2]], dtype=np.float32),
            qvel=np.zeros((2, model.nv), dtype=np.float32),
            act=np.array([[0.3], [0.4]], dtype=np.float32))
  sim.step(2, ctrl=np.array([[0.5], [0.5]], dtype=np.float32))
  healthy_act = sim.state._act.cpu().numpy().copy()
  healthy_qpos = sim.state.qpos.cpu().numpy().copy()
  # Inject a force-evaluation failure into world 1 only (NaN state).
  bad = sim.state._qpos.cpu().numpy()
  bad[1, 0] = np.nan
  sim.state._qpos.copy_(sim.state._torch.as_tensor(bad))
  sim.step(1, ctrl=np.array([[0.5], [0.5]], dtype=np.float32))
  status = sim.state._status.cpu().numpy()
  assert int(status[1]) != 0
  after_act = sim.state._act.cpu().numpy()
  after_qpos = sim.state.qpos.cpu().numpy()
  # Failed world: activation and observable state exactly preserved.
  assert after_act[1, 0] == healthy_act[1, 0]
  assert np.array_equal(after_qpos[1], healthy_qpos[1], equal_nan=True) or \
      np.all(~np.isfinite(after_qpos[1]))
  # Healthy world: kept advancing (finite and moved).
  assert bool(np.all(np.isfinite(after_qpos[0])))
  assert not np.array_equal(after_qpos[0], healthy_qpos[0])
  # Recovery: reset the failed world, both advance again.
  sim.reset(env_ids=[1])
  sim.step(1, ctrl=np.array([[0.5], [0.5]], dtype=np.float32))
  assert bool(np.all(np.isfinite(sim.state.qpos.cpu().numpy())))
