# Copyright 2026 The MuJoCo Metal contributors
# Licensed under the Apache License, Version 2.0.
"""CPU storage protocols only; actual energy physics requires native fixtures."""
from types import SimpleNamespace, MethodType
import numpy as np
import pytest
import torch
import mujoco

from test_state_copy_api import _sim
from mujoco_metal.forward_stages import ForwardStage
from mujoco_metal.native_api import mj_energyPos, mj_energyVel
from mujoco_metal import MetalSimulation
import mujoco_metal.energy as energy


def _prepared():
  sim = _sim()
  sim._energy = None
  sim._energy_manual = False
  sim._spatial_tendons = None
  sim.state._reset_mask_i32 = torch.zeros(3, dtype=torch.int32)
  sim._smooth = SimpleNamespace(_workspace={
      'mass_sparse_product': torch.full((3,), 31.),
      'mass_matvec_dims': torch.full((7,), 17, dtype=torch.int32),
      'energy_velocity_dims': torch.full((6,), 19, dtype=torch.int32)})
  dynamics = {'status': torch.zeros(3, dtype=torch.int32)}
  record = sim._forward_stages.begin(generation=sim.state.generation,
      qpos=sim.state._qpos, position={'dynamics': dynamics, 'context': {}})
  sim._forward_stages.publish(record, ForwardStage.VEL,
      {'dynamics': dynamics, 'qvel': sim.state._qvel})
  sim._step1_record = record
  def potential(self, qpos, dyn, context, *, world_mask, manual):
    assert manual
    self._energy[world_mask, 0] = torch.tensor([3., 5., 7.])[world_mask]
    self._smooth._workspace['mass_sparse_product'].fill_(99.)
    self.state._reset_mask_i32.fill_(1)
  def kinetic(self, velocity, *, world_mask, manual):
    assert manual
    self._energy[world_mask, 1] = torch.tensor([11., 13., 17.])[world_mask]
    self._smooth._workspace['energy_velocity_dims'].fill_(88)
  sim._update_energy_position = MethodType(potential, sim)
  sim._update_energy_velocity = MethodType(kinetic, sim)
  return sim, record


def test_public_selected_columns_and_prepared_record_preservation():
  sim, record = _prepared()
  source_qpos, generation = sim.state._qpos.clone(), sim.state._generation
  before = {n: v.clone() for n, v in sim._smooth._workspace.items()}
  mask = sim.state._reset_mask_i32.clone()
  result = mj_energyPos(sim, record, env_ids=[2, 0])
  assert torch.equal(result, torch.tensor([3., 0., 7.]))
  assert torch.equal(sim.energy[:, 1], torch.zeros(3))
  output = result.clone()
  result.fill_(-10.)
  assert torch.equal(sim.energy[:, 0], output)
  mj_energyVel(sim, record, env_ids=[1])
  assert torch.equal(sim.energy[:, 1], torch.tensor([0., 13., 0.]))
  assert torch.equal(sim.energy[:, 0], output)
  for name, value in before.items():
    assert torch.equal(value, sim._smooth._workspace[name])
  assert torch.equal(mask, sim.state._reset_mask_i32)
  assert sim.validate_forward_stage_record(record, 'VEL') is record
  assert sim._step1_record is record and sim.state._generation == generation
  assert torch.equal(sim.state._qpos, source_qpos)


@pytest.mark.parametrize('existing', (False, True))
def test_actual_post_publication_failure_restores_storage_and_record(existing, monkeypatch):
  sim, record = _prepared()
  if existing:
    mj_energyPos(sim, record)
  old_fields = {k: v for k, v in vars(sim).items() if k.startswith('_energy')}
  before = None if sim._energy is None else sim._energy.clone()
  scratch = {k: v.clone() for k, v in sim._smooth._workspace.items()}
  calls = []
  original = energy._publish_energy
  def fail(*args):
    calls.append(True)
    original(*args)
    raise RuntimeError('actual diagnostic publication failure')
  monkeypatch.setattr(energy, '_publish_energy', fail)
  with pytest.raises(RuntimeError, match='publication failure'):
    mj_energyVel(sim, record)
  assert calls == [True]
  now = {k: v for k, v in vars(sim).items() if k.startswith('_energy')}
  assert now.keys() == old_fields.keys()
  assert all(now[k] is old_fields[k] for k in now)
  if before is not None:
    assert torch.equal(before, sim._energy)
  assert sim.validate_forward_stage_record(record, 'VEL') is record
  for k, v in scratch.items():
    assert torch.equal(v, sim._smooth._workspace[k])


def test_explicit_energy_does_not_enable_real_automatic_producers():
  sim, record = _prepared()
  mj_energyPos(sim, record)
  before = sim.energy
  # Real producer gating, not the storage fixture's synthetic dispatch.
  assert MetalSimulation._update_energy_position(sim, record.qpos, {}, {}) is None
  assert MetalSimulation._update_energy_velocity(sim, {}) is None
  # Automatic disabled sampling does not evaluate either formula; the
  # position sweep follows pinned forward's clearing of stored diagnostics.
  assert torch.equal(sim.energy, torch.zeros_like(before))


@pytest.mark.parametrize('bad_ids', ([], [3], [0, 0], [.5]))
def test_bad_selection_is_rejected_before_lazy_allocation(bad_ids):
  sim, record = _prepared()
  with pytest.raises((ValueError, TypeError, IndexError)):
    mj_energyPos(sim, record, env_ids=bad_ids)
  assert sim._energy is None and not sim._energy_manual
  assert sim.validate_forward_stage_record(record, 'VEL') is record


def test_stale_input_or_budget_rejection_preserves_lazy_metadata():
  from mujoco_metal.capacity import CapacityLimits, CapacityOverflow
  sim, record = _prepared()
  sim.state._qpos.add_(.1)
  with pytest.raises(ValueError, match='mutated'):
    mj_energyPos(sim, record)
  assert sim._energy is None and not sim._energy_manual
  sim, record = _prepared()
  sim.limits = CapacityLimits(memory_budget_bytes=1)
  with pytest.raises(CapacityOverflow):
    mj_energyPos(sim, record)
  assert sim._energy is None and sim.validate_forward_stage_record(record, 'VEL') is record


def test_failed_and_stale_rows_do_not_publish_diagnostics():
  sim, record = _prepared()
  sim.state._status[1] = 1
  record.row_valid[2] = False
  mj_energyPos(sim, record)
  assert torch.equal(sim.energy[:, 0], torch.tensor([3., 0., 0.]))


def test_inference_caller_does_not_make_persistent_energy_immutable():
  sim, record = _prepared()
  with torch.inference_mode():
    mj_energyPos(sim, record)
  assert not torch.is_inference(sim._energy)
  assert not torch.is_inference(sim._energy_stage)
  assert isinstance(sim._energy._version, int)
  sim._energy.copy_(torch.full_like(sim._energy, 3.))


def test_previous_borrowed_energy_view_survives_explicit_publication():
  sim, record = _prepared()
  mj_energyPos(sim, record)
  earlier = sim._energy[:, 1]
  saved = earlier.clone()
  record.values[ForwardStage.POS]['energy_pos'] = sim._energy[:, 0]
  mj_energyVel(sim, record)
  assert torch.equal(earlier, saved)
  assert torch.equal(sim.energy[:, 1], torch.tensor([11., 13., 17.]))
  assert sim.validate_forward_stage_record(record, 'VEL') is record


@pytest.mark.parametrize('enabled', [False, True])
@pytest.mark.parametrize('sensors', ['', '<e_potential/>', '<e_kinetic/>'])
def test_energy_sensor_allocation_is_charged_in_host_admission(enabled, sensors):
  from mujoco_metal.capacity import estimate_capacity
  model = mujoco.MjModel.from_xml_string(f'''<mujoco><option>
    <flag energy="{'enable' if enabled else 'disable'}"/></option>
    <worldbody><body><joint/><geom type="sphere" size=".1"/></body></worldbody>
    <sensor>{sensors}</sensor></mujoco>''')
  parts = dict(estimate_capacity(model, 3, 0, 0, 0).memory_breakdown)
  active = enabled or bool(sensors)
  assert parts['simulation.energy_output'] == (24 if active else 0)
  assert parts['simulation.energy_stage'] == (24 if active else 0)
  assert parts['simulation.energy_position_dims'] == (60 if active else 0)
