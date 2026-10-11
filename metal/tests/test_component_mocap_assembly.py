# Copyright 2026 The MuJoCo Metal contributors
# Licensed under the Apache License, Version 2.0.
"""Supplied component poses include mocap exactly once at native assembly."""
import os
import mujoco
import numpy as np
import pytest


@pytest.mark.gpu
@pytest.mark.skipif(os.getenv('MUJOCO_METAL_RUN_GPU') != '1', reason='native assembly')
@pytest.mark.parametrize('profile', ('integrated_scalable_v1', 'integrated_implicitfast_v1'))
def test_component_mocap_assembly_matches_pinned_force_and_preserves_query_state(profile, monkeypatch):
  from mujoco_metal import MetalSimulation
  from test_observational_assembly_query import _assert_snapshot_bytes
  model = mujoco.MjModel.from_xml_string('''<mujoco>
    <option timestep=".001" solver="PGS" iterations="80" jacobian="sparse"/>
    <worldbody><geom type="plane" size="2 2 .1" condim="1"/>
      <body name="load" pos="0 0 .099"><joint name="j" type="slide" axis="0 0 1"/>
        <geom type="sphere" size=".1" mass="1" condim="1"/></body>
      <body name="target" mocap="true" pos="0 0 .2"/>
    </worldbody>
    <equality><connect body1="load" body2="target" anchor="0 0 .1"/></equality>
    </mujoco>''')
  if profile == 'integrated_implicitfast_v1':
    model.opt.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
  sim = MetalSimulation(model, 2, profile=profile)
  assert sim._component_mass_enabled
  sim.reset(qpos=np.array([[-.0012345678901234], [-.0023456789012345]]),
            qvel=np.array([[.0123456789012345], [-.0234567890123456]]),
            eq_active=np.array([[1], [0]]),
            mocap_pos=np.array([[[0., 0., .2001]], [[0., 0., .1998]]]),
            mocap_quat=np.tile(np.array([[[1., 0., 0., 0.]]]), (2, 1, 1)))
  # The producer was the original failing operation. Exercise it first,
  # then the observational wrapper on a real pending forward record.
  producer = sim.assembled_system(recompute=True)
  assert producer['status'].cpu().tolist() == [0, 0]
  sim.step1()
  before = sim.snapshot()
  state = before['device']
  words = state.precision_words.astype(float)
  q = state.qpos.astype(float) + words[:, :1] + words[:, 1:2]
  v = state.qvel.astype(float) + words[:, 2:3] + words[:, 3:4]
  oracle = []
  for row in range(2):
    data = mujoco.MjData(model)
    data.qpos[:], data.qvel[:] = q[row], v[row]
    data.mocap_pos[:], data.mocap_quat[:] = state.mpos[row], state.mquat[row]
    data.eq_active[:] = state.eq_active[row]
    data.qacc_warmstart[:] = before['native_state']['qacc_warmstart'][row]
    mujoco.mj_forward(model, data)
    assert data.ncon == 1 and np.linalg.norm(data.qfrc_constraint) > .01
    oracle.append(data)
  calls = []
  original = sim._smooth.run_device
  def capture(*args, **kwargs):
    if kwargs.get('poses') is not None:
      assert args[2] is None and args[3] is None
      assert kwargs.get('_trusted_internal_poses') is True
      calls.append(True)
    return original(*args, **kwargs)
  monkeypatch.setattr(sim._smooth, 'run_device', capture)
  def forbid(*args, **kwargs):
    raise AssertionError('native assembly must not call CPU physics')
  for name in ('mj_step', 'mj_forward', 'mj_collision', 'mj_fwdPosition'):
    monkeypatch.setattr(mujoco, name, forbid)
  result = sim.query_assembled_system()
  assert calls
  for name in ('qacc', 'qfrc_constraint'):
    np.testing.assert_allclose(result[name].cpu().numpy(),
                               np.array([getattr(data, name) for data in oracle]),
                               rtol=4e-5, atol=6e-6)
  _assert_snapshot_bytes(before, sim.snapshot())
  assert sim.step2().cpu().tolist() == [0, 0]
