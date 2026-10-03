# Copyright 2026 The MuJoCo Metal contributors
# Licensed under the Apache License, Version 2.0.
"""Ownership, invalidation, and publication contracts without GPU dispatch."""
import pytest

from mujoco_metal.forward_stages import ForwardStage as Stage, ForwardStageCoordinator


class Versioned:
  def __init__(self):
    self._version = 0


def _prepared():
  owner = object()
  stages = ForwardStageCoordinator(owner)
  qpos, mpos, mquat = Versioned(), Versioned(), Versioned()
  record = stages.begin(generation=3, qpos=qpos, mocap_pos=mpos,
      mocap_quat=mquat, position={'poses': object()})
  return stages, record


def test_stage_order_ownership_generation_and_replacement():
  stages, record = _prepared()
  with pytest.raises(ValueError, match='cannot skip'):
    stages.publish(record, Stage.ACT, {'force': object()})
  with pytest.raises(ValueError, match='needs VEL'):
    stages.consume(record, Stage.VEL, generation=3)
  foreign = ForwardStageCoordinator(object())
  with pytest.raises(ValueError, match='stale'):
    foreign.validate(record, generation=3)
  with pytest.raises(ValueError, match='stale'):
    stages.validate(record, generation=4)
  stages.begin(generation=3, qpos=Versioned(), position={'poses': object()})
  with pytest.raises(ValueError, match='stale'):
    stages.validate(record, generation=3)


@pytest.mark.parametrize('name', ['qpos', 'mocap_pos', 'mocap_quat'])
@pytest.mark.parametrize('action', ['mutate', 'replace'])
def test_captured_position_input_mutation_or_replacement_rejects_reuse(name, action):
  stages, record = _prepared()
  if action == 'mutate':
    getattr(record, name)._version += 1
  else:
    setattr(record, name, Versioned())
  with pytest.raises(ValueError, match='mutated|replaced'):
    stages.consume(record, Stage.POS, generation=3)


def test_velocity_mutation_rejects_vel_reuse_but_allows_pos_refresh():
  stages, record = _prepared()
  qvel = Versioned()
  stages.publish(record, Stage.VEL, {'qvel': qvel, 'bias': 1})
  stages.publish(record, Stage.ACT, {'force': 2})
  stages.publish(record, Stage.ACC, {'acc': 3})
  qvel._version += 1
  with pytest.raises(ValueError, match='qvel was mutated'):
    stages.consume(record, Stage.VEL, generation=3)
  stages.consume(record, Stage.POS, generation=3)
  stages.publish(record, Stage.VEL, {'qvel': qvel, 'bias': 4})
  assert record.stage == Stage.VEL
  assert set(record.values) == {Stage.POS, Stage.VEL}
  assert stages.consume(record, Stage.VEL, generation=3)['bias'] == 4


def test_invalid_publication_is_atomic_and_invalidated_record_rejects_reuse():
  stages, record = _prepared()
  old = (record.stage, dict(record.values), dict(record.input_tensors),
         dict(record.input_versions))
  for stage, value, error in ((Stage.VEL, ['not', 'a', 'dict'], TypeError),
                              (Stage.ACC, {'qvel': Versioned()}, ValueError)):
    with pytest.raises(error):
      stages.publish(record, stage, value)
    assert (record.stage, record.values, record.input_tensors, record.input_versions) == old
  stages.invalidate()
  with pytest.raises(ValueError, match='stale'):
    stages.validate(record, generation=3)


def test_missing_mutation_counters_allow_fresh_pos_but_reject_cached_reuse():
  stages = ForwardStageCoordinator(object())
  record = stages.begin(generation=0, qpos=object(), position={'poses': object()})
  assert record.stage == Stage.POS
  with pytest.raises(ValueError, match='mutation counter'):
    stages.consume(record, Stage.POS, generation=0)


def test_real_torch_mutation_and_inference_tensor_counters():
  torch = pytest.importorskip('torch')
  stages = ForwardStageCoordinator(object())
  qpos = torch.zeros(1, 2)
  record = stages.begin(generation=0, qpos=qpos, position={'poses': object()})
  qpos.add_(1)
  with pytest.raises(ValueError, match='mutated'):
    stages.consume(record, Stage.POS, generation=0)
  with torch.inference_mode():
    untracked = torch.zeros(1, 2)
  record = stages.begin(generation=0, qpos=untracked, position={'poses': object()})
  with pytest.raises(ValueError, match='mutation counter'):
    stages.consume(record, Stage.POS, generation=0)
