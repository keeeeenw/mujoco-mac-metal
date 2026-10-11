# Copyright 2026 The MuJoCo Metal contributors
# Licensed under the Apache License, Version 2.0.
"""CPU metadata/ownership checks; native numerical evidence is separate."""
from types import SimpleNamespace

import mujoco
import pytest

torch = pytest.importorskip('torch')

from mujoco_metal.smooth_metal import MetalSmoothDynamics


def _program():
  program = MetalSmoothDynamics.__new__(MetalSmoothDynamics)
  program.model = SimpleNamespace(nq=3, nv=2)
  program._torch = torch
  program._workspace = {'batch_size': 2,
                         'bias_velocity_words': torch.full((12,), -17.)}
  return program


def test_velocity_payload_contains_current_planes_and_clears_stale_residuals():
  program = _program()
  qvel = torch.arange(4, dtype=torch.float32).reshape(2, 2)
  words = torch.arange(20, dtype=torch.float32).reshape(2, 10)
  program._velocity_words_provider = lambda _: words
  actual = program._pack_bias_velocity(qvel).reshape(3, 2, 2)
  torch.testing.assert_close(actual[0], qvel, rtol=0, atol=0)
  torch.testing.assert_close(actual[1], words[:, 6:8], rtol=0, atol=0)
  torch.testing.assert_close(actual[2], words[:, 8:10], rtol=0, atol=0)
  program._velocity_words_provider = lambda _: None
  qvel.add_(.5)
  actual = program._pack_bias_velocity(qvel).reshape(3, 2, 2)
  torch.testing.assert_close(actual[0], qvel, rtol=0, atol=0)
  assert torch.count_nonzero(actual[1:]) == 0


@pytest.mark.parametrize('invalid', ['shape', 'dtype', 'layout', 'device', 'type'])
def test_invalid_velocity_payload_metadata_preserves_owned_scratch(invalid):
  program = _program()
  words = torch.zeros((2, 10))
  if invalid == 'shape':
    words = words[:, :-1].contiguous()
  elif invalid == 'dtype':
    words = words.to(torch.float64)
  elif invalid == 'layout':
    words = torch.zeros((2, 20))[:, ::2]
  elif invalid == 'device':
    words = torch.empty((2, 10), device='meta')
  else:
    words = []
  program._velocity_words_provider = lambda _: words
  before = program._workspace['bias_velocity_words'].clone()
  version = program._workspace['bias_velocity_words']._version
  with pytest.raises(ValueError, match='retained velocity words'):
    program._pack_bias_velocity(torch.zeros((2, 2)))
  torch.testing.assert_close(program._workspace['bias_velocity_words'], before, rtol=0, atol=0)
  assert program._workspace['bias_velocity_words']._version == version


@pytest.mark.parametrize('moving', [False, True])
def test_velocity_payload_capacity_charges_planes_or_empty_sentinel(moving):
  from mujoco_metal.capacity import _runtime_buffer_sizes
  body = '<body><freejoint/><geom type="sphere" size=".1"/></body>' if moving else ''
  model = mujoco.MjModel.from_xml_string('<mujoco><worldbody>'+body+'</worldbody></mujoco>')
  assert dict(_runtime_buffer_sizes(model, 3))['smooth.bias_velocity_words'] == max(9*model.nv, 1)
