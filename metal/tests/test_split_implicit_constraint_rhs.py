# Copyright 2026 The MuJoCo Metal contributors
# Licensed under the Apache License, Version 2.0.
"""Split integration consumes total force, including solved constraints."""
import os
import mujoco
import numpy as np
import pytest


def _model(integrator, sparse):
  spec = mujoco.MjSpec.from_string('''<mujoco>
    <option timestep=".0001" solver="PGS" iterations="80"/>
    <worldbody><geom type="plane" size="2 2 .1" condim="1"/>
      <body name="load" pos="0 0 .099"><joint name="j" type="slide" axis="0 0 1"/>
        <geom type="sphere" size=".1" mass="1" condim="1"/></body>
      <body name="target" mocap="true" pos="0 0 .2"/>
    </worldbody>
    <equality><connect body1="load" body2="target" anchor="0 0 .1"/></equality>
    <actuator><general name="drive" joint="j" dyntype="filterexact" dynprm=".01"/></actuator>
    <sensor><jointpos name="position" joint="j"/><jointvel joint="j"/></sensor>
    </mujoco>''')
  for entry in (spec.actuator('drive'), spec.sensor('position')):
    entry.nsample, entry.interp, entry.delay = 4, 1, .0002
  model = spec.compile()
  model.opt.integrator = getattr(mujoco.mjtIntegrator, 'mjINT_' + integrator)
  model.opt.jacobian = mujoco.mjtJacobian.mjJAC_SPARSE if sparse else mujoco.mjtJacobian.mjJAC_DENSE
  return model


def _cpu_state(model, sim):
  from mujoco_metal.native_api import mj_getState
  state = sim.snapshot()['device']
  values = mj_getState(sim)
  words = state.precision_words.astype(float)
  q = state.qpos.astype(float) + words[:, :1] + words[:, 1:2]
  v = state.qvel.astype(float) + words[:, 2:3] + words[:, 3:4]
  data = []
  for row in range(2):
    d = mujoco.MjData(model)
    d.qpos[:], d.qvel[:], d.time = q[row], v[row], float(state.time[row])
    for source, destination in (('act', 'act'), ('history', 'history'),
        ('warmstart', 'qacc_warmstart'), ('ctrl', 'ctrl'), ('qfrc_applied', 'qfrc_applied'),
        ('xfrc_applied', 'xfrc_applied'), ('eq_active', 'eq_active'),
        ('mocap_pos', 'mocap_pos'), ('mocap_quat', 'mocap_quat')):
      getattr(d, destination)[:] = values[source][row].cpu().numpy()
    data.append(d)
  return data


@pytest.mark.gpu
@pytest.mark.skipif(os.getenv('MUJOCO_METAL_RUN_GPU') != '1', reason='native split integration')
@pytest.mark.parametrize('integrator', ('EULER', 'IMPLICITFAST', 'IMPLICIT'))
@pytest.mark.parametrize('sparse', (False, True))
def test_split_constraint_force_reaches_implicit_integration_and_replays(integrator, sparse, monkeypatch):
  from mujoco_metal import MetalSimulation
  from test_observational_assembly_query import _assert_snapshot_bytes
  model = _model(integrator, sparse)
  profile = {'EULER': ('integrated_scalable_v1' if sparse else 'integrated_euler_v1'),
             'IMPLICITFAST': 'integrated_implicitfast_v1',
             'IMPLICIT': 'integrated_implicit_v1'}[integrator]
  sim = MetalSimulation(model, 2, profile=profile)
  # Integrated implicit methods always use component mass; the model's
  # dense/sparse Jacobian option independently chooses constraint storage.
  assert sim._component_mass_enabled is (sparse or integrator != 'EULER')
  sim.reset(qpos=np.array([[-.00234567890123456], [-.00123456789012345]]),
            qvel=np.array([[.0212345678901234], [-.0152345678901234]]),
            act=np.array([[.2], [.3]]), eq_active=np.array([[1], [0]]),
            mocap_pos=np.array([[[0., 0., .2001]], [[0., 0., .2002]]]),
            mocap_quat=np.tile(np.array([[[1., 0., 0., 0.]]]), (2, 1, 1)))
  wrench = np.zeros((2, model.nbody, 6), dtype=np.float32)
  wrench[:, 1, 2] = [.011, -.012]
  assert sim.step(2, ctrl=np.array([[.4], [.6]], dtype=np.float32),
                  qfrc_applied=np.array([[.025], [-.035]], dtype=np.float32),
                  xfrc_applied=wrench).cpu().tolist() == [0, 0]
  checkpoint = sim.snapshot()
  oracle = _cpu_state(model, sim)
  observed = []
  for _ in range(3):
    for d in oracle:
      mujoco.mj_step1(model, d)
      mujoco.mj_step2(model, d)
      assert d.ncon == 1 and np.linalg.norm(d.qfrc_constraint) > .01
    with monkeypatch.context() as guard:
      def forbid(*args, **kwargs):
        raise AssertionError('native split stepping must not call CPU physics')
      for name in ('mj_step', 'mj_step1', 'mj_step2', 'mj_forward', 'mj_collision'):
        guard.setattr(mujoco, name, forbid)
      sim.step1()
      assert sim.step2().cpu().tolist() == [0, 0]
    for name in ('qpos', 'qvel', 'qacc', 'act'):
      np.testing.assert_allclose(getattr(sim.state, '_' + name).cpu().numpy(),
                                 np.array([getattr(d, name) for d in oracle]),
                                 rtol=4e-5, atol=6e-6)
    observed.append(sim.snapshot())
  sim.restore(checkpoint)
  for expected in observed:
    sim.step1()
    assert sim.step2().cpu().tolist() == [0, 0]
    _assert_snapshot_bytes(expected, sim.snapshot(), replay=True)
