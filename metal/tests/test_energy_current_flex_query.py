# Copyright 2026 The MuJoCo Metal contributors
# Licensed under the Apache License, Version 2.0.
"""Current flex energy queries must evaluate fresh, observational geometry."""
import os
import mujoco
import numpy as np
import pytest

from mujoco_metal import MetalSimulation
from mujoco_metal.native_api import mj_forward
from mujoco_metal.forward_stages import ForwardStage
from test_energy_lifecycle_006 import _native_only

pytestmark = [pytest.mark.gpu, pytest.mark.skipif(
    os.getenv('MUJOCO_METAL_RUN_GPU') != '1', reason='native opt-in')]


def _host(value):
  return value.detach().cpu().numpy().copy()


@pytest.mark.parametrize('profile', ['integrated_euler_v1', 'integrated_scalable_v1'])
@pytest.mark.parametrize('changed', [False, True])
@pytest.mark.parametrize('failure', [False, True])
def test_current_flex_energy_matches_source_and_restores_prepared_buffers(
    profile, changed, failure, monkeypatch):
  model = mujoco.MjModel.from_xml_string('''<mujoco>
    <option timestep=".001"><flag energy="enable"/></option>
    <worldbody><flexcomp name="rope" type="grid" count="3 1 1"
      spacing=".2 .2 .2" dim="1" pos="0 0 1" radius=".01" mass="1">
      <edge stiffness="20" damping="0"/>
      <contact contype="0" conaffinity="0"/></flexcomp></worldbody>
    <sensor><e_potential/><e_kinetic/></sensor></mujoco>''')
  q = np.tile(model.qpos0, (3, 1)).astype(np.float32)
  q[:, 3] += [.01, .02, .03]
  v = np.tile(np.linspace(.02, .04, model.nv), (3, 1)).astype(np.float32)
  sim = MetalSimulation(model, 3, qpos=q, qvel=v, profile=profile)
  record = _native_only(monkeypatch, lambda: mj_forward(sim))['record']
  if changed:
    sim.state._qpos[:, 3].add_(sim.state._torch.tensor(
        [.08, .12, .16], dtype=sim.state._torch.float32, device=sim.state._device))
  q, v = _host(sim.state._qpos), _host(sim.state._qvel)
  refs = []
  for pos, vel in zip(q, v):
    data = mujoco.MjData(model)
    data.qpos[:], data.qvel[:] = pos, vel
    mujoco.mj_forward(model, data)
    refs.append(data)
  before = sim.snapshot()
  generation, epoch = sim.state.generation, sim._forward_stages.epoch
  geometry = {name: _host(getattr(sim.flex, name)) for name in
              ('_flexvert_xpos', '_flexedge_length', '_flexedge_J')}
  context = sim.flex._latest_position_context
  kin_generation = sim.flex._kinematics_generation
  dynamics = record.values[ForwardStage.POS]['dynamics']
  mass = dynamics['mass_blocks' if sim._component_mass_enabled else 'mass_matrix']
  before_mass = _host(mass)
  if failure:
    original, calls = sim._sensor_values_impl, []
    def fail_after_native_query():
      result = original()
      assert result is not None
      calls.append(True)
      raise RuntimeError('completed flex energy query fault')
    monkeypatch.setattr(sim, '_sensor_values_impl', fail_after_native_query)
    with pytest.raises(RuntimeError, match='flex energy query fault'):
      _native_only(monkeypatch, sim.sensor_values)
    assert calls == [True]
  else:
    result = _native_only(monkeypatch, sim.sensor_values)
    np.testing.assert_allclose(_host(result), [d.sensordata for d in refs],
                               rtol=2e-5, atol=2e-6)
    if changed:
      assert np.min(np.array([d.energy[0] for d in refs]) - before['energy'][:, 0]) > .1
    result.fill_(-10)
  after = sim.snapshot()
  for name in ('energy', 'energy_computed', 'sensordata'):
    np.testing.assert_array_equal(after[name], before[name])
  np.testing.assert_array_equal(after['native_state']['history'],
                                 before['native_state']['history'])
  for name, value in geometry.items():
    np.testing.assert_array_equal(_host(getattr(sim.flex, name)), value)
  np.testing.assert_array_equal(_host(mass), before_mass)
  assert sim.flex._latest_position_context is context
  assert sim.flex._kinematics_generation == kin_generation
  assert sim._forward_stages._record is record
  assert (sim.state.generation, sim._forward_stages.epoch) == (generation, epoch)
  if changed:
    with pytest.raises(ValueError, match='mutated'):
      sim.validate_forward_stage_record(record, 'VEL')
  else:
    assert sim.validate_forward_stage_record(record, 'VEL') is record
  for _ in range(3):
    for data in refs:
      mujoco.mj_step(model, data)
    np.testing.assert_array_equal(_host(_native_only(monkeypatch, sim.step)), 0)
    np.testing.assert_allclose(_host(sim.state._qpos), [d.qpos for d in refs],
                               rtol=2e-5, atol=2e-6)
    np.testing.assert_allclose(_host(sim.state._qvel), [d.qvel for d in refs],
                               rtol=2e-5, atol=2e-6)
    np.testing.assert_allclose(_host(sim.state._qacc), [d.qacc for d in refs],
                               rtol=2e-5, atol=2e-4)
  final = sim.snapshot()
  _native_only(monkeypatch, lambda: sim.restore(before))
  for _ in range(3):
    _native_only(monkeypatch, sim.step)
  replay = sim.snapshot()
  for name in ('qpos', 'qvel', 'qacc', 'precision_words', 'time', 'status'):
    np.testing.assert_array_equal(getattr(replay['device'], name),
                                   getattr(final['device'], name))
  for name in ('energy', 'energy_computed', 'sensordata'):
    np.testing.assert_array_equal(replay[name], final[name])
