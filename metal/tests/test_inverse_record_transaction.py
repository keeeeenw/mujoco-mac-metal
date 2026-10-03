# Copyright 2026 The MuJoCo Metal contributors
# Licensed under the Apache License, Version 2.0.
"""Explicit device-query rollback preserves coherent record storage/tokens."""
from types import SimpleNamespace

import pytest


@pytest.mark.parametrize('fail', [False, True])
def test_query_restores_record_values_owners_tokens_and_bookkeeping(fail):
  torch = pytest.importorskip('torch')
  from mujoco_metal.forward_stages import ForwardStage as Stage, ForwardStageCoordinator
  from mujoco_metal.native_api import _inverse_query_workspaces
  sim = SimpleNamespace(_forward_position_epoch=12,
      _last_actuation_kin={'force': torch.tensor([[.7]])},
      _rhs=torch.tensor([[.5]]))
  sim._forward_stages = ForwardStageCoordinator(sim)
  qpos, qvel = torch.tensor([[.2]]), torch.tensor([[.3]])
  record = sim._forward_stages.begin(generation=0, qpos=qpos,
                                   position={'mass': torch.tensor([[[2.]]])})
  sim._forward_stages.publish(record, Stage.VEL, {'qvel': qvel})
  original_force = sim._last_actuation_kin['force']
  def query():
    with _inverse_query_workspaces(sim):
      qpos.add_(9)
      qvel.add_(8)
      sim._forward_stages.invalidate()
      sim._forward_position_epoch = 99
      sim._last_actuation_kin['force'] = torch.tensor([[99.]])
      sim._rhs.add_(10)
      if fail:
        raise RuntimeError('query failed')
  if fail:
    with pytest.raises(RuntimeError, match='query failed'):
      query()
  else:
    query()
  assert sim._forward_position_epoch == 12
  assert sim._last_actuation_kin['force'] is original_force
  torch.testing.assert_close(original_force, torch.tensor([[.7]]))
  torch.testing.assert_close(sim._rhs, torch.tensor([[.5]]))
  assert sim._forward_stages._record is record
  assert record.qpos is qpos and record.values[Stage.VEL]['qvel'] is qvel
  torch.testing.assert_close(qpos, torch.tensor([[.2]]))
  torch.testing.assert_close(qvel, torch.tensor([[.3]]))
  sim._forward_stages.validate(record, generation=0, minimum=Stage.VEL)


def test_query_never_rehabilitates_a_record_already_stale_on_entry():
  torch = pytest.importorskip('torch')
  from mujoco_metal.forward_stages import ForwardStageCoordinator
  from mujoco_metal.native_api import _inverse_query_workspaces
  sim = SimpleNamespace()
  sim._forward_stages = ForwardStageCoordinator(sim)
  qpos = torch.tensor([[.2]])
  record = sim._forward_stages.begin(generation=0, qpos=qpos, position={'mass': object()})
  qpos.add_(1)
  with _inverse_query_workspaces(sim):
    pass
  with pytest.raises(ValueError, match='mutated'):
    sim._forward_stages.validate(record, generation=0)


@pytest.mark.parametrize('fail', [False, True])
@pytest.mark.parametrize('allocated', [False, True])
def test_query_restores_energy_values_and_owner_even_if_buffer_is_replaced(
    fail, allocated):
  torch = pytest.importorskip('torch')
  from mujoco_metal.native_api import _inverse_query_workspaces
  owner = torch.tensor([[2., .5], [3., .7]]) if allocated else None
  sim = SimpleNamespace(_energy=owner)
  saved = owner.clone() if allocated else None
  def query():
    with _inverse_query_workspaces(sim):
      if allocated:
        sim._energy.add_(9)
      sim._energy = torch.zeros(2, 2)
      if fail:
        raise RuntimeError('energy query failed')
  if fail:
    with pytest.raises(RuntimeError, match='energy query failed'):
      query()
  else:
    query()
  assert sim._energy is owner
  if allocated:
    torch.testing.assert_close(owner, saved, rtol=0, atol=0)
