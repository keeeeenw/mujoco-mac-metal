# Copyright 2026 The MuJoCo Metal contributors
# Licensed under the Apache License, Version 2.0.
"""Pinned frame transform paths, optional outputs, and real native execution."""
import os
import ctypes
from types import SimpleNamespace

import mujoco
import numpy as np
import pytest

from tests.test_spatial_queries_019 import _fixture, _stages


def _inputs():
  return (np.array([[.13, -.21, .07], [-.08, .05, .19]], np.float32),
          # Deliberately nonunit: mj_local2Global must not normalize it.
          np.array([[.9, .2, -.3, .1], [.5, -.6, .7, -.4]], np.float32))


def _oracle(worlds, position, quaternion, body, sameframe):
  positions, matrices = [], []
  # Python bindings require all four arrays; the exported pinned C API also
  # defines null input/output semantics, which this batched API exposes.
  source = ctypes.CDLL(mujoco._structs.__file__).mj_local2Global
  double_ptr = ctypes.POINTER(ctypes.c_double)
  source.argtypes = [ctypes.c_void_p, double_ptr, double_ptr, double_ptr,
                     double_ptr, ctypes.c_int, ctypes.c_ubyte]
  source.restype = None
  def pointer(value):
    return None if value is None else value.ctypes.data_as(double_ptr)
  for world, data in enumerate(worlds):
    xp = None if position is None else np.zeros(3)
    xm = None if quaternion is None else np.zeros(9)
    local_pos = None if position is None else position[world].astype(np.float64)
    local_quat = None if quaternion is None else quaternion[world].astype(np.float64)
    source(data._address, pointer(xp), pointer(xm), pointer(local_pos),
           pointer(local_quat), body, sameframe)
    if xp is not None:
      positions.append(xp)
    if xm is not None:
      matrices.append(xm.reshape(3, 3))
  return (None if position is None else np.stack(positions),
          None if quaternion is None else np.stack(matrices))


@pytest.mark.parametrize('sameframe', list(range(5)))
@pytest.mark.parametrize('outputs', ['both', 'position', 'orientation', 'neither'])
def test_local_to_global_all_pinned_alignment_paths_and_optional_outputs(sameframe, outputs):
  torch = pytest.importorskip('torch')
  from mujoco_metal import native_api
  model = _fixture()
  worlds = [_stages(model, seed, 'cpu') for seed in (17, 31)]
  dynamics = {key: torch.cat([world[1][key] for world in worlds])
              for key in ('root_com', 'cvel', 'cdof', 'cdof_dot')}
  dynamics['poses'] = {key: torch.cat([world[1]['poses'][key] for world in worlds])
                       for key in worlds[0][1]['poses']}
  sim = SimpleNamespace(model=model, _mjmodel=model, batch_size=2,
                        state=SimpleNamespace(_device=torch.device('cpu')))
  pos, quat = _inputs()
  if outputs not in ('both', 'position'):
    pos = None
  if outputs not in ('both', 'orientation'):
    quat = None
  before = {key: value.clone() for key, value in dynamics['poses'].items()}
  for body in range(model.nbody):
    expected = _oracle([world[0] for world in worlds], pos, quat, body, sameframe)
    actual = native_api.mj_local2Global(sim, pos, quat, body, sameframe, dynamics=dynamics)
    for value, reference in zip(actual, expected):
      if reference is None:
        assert value is None
      else:
        np.testing.assert_allclose(value.numpy(), reference, atol=3e-6, rtol=3e-5)
    retained = tuple(None if value is None else value.clone() for value in actual)
    other = native_api.mj_local2Global(sim, pos, quat, body, sameframe, dynamics=dynamics)
    for value in other:
      if value is not None:
        value.zero_()
    for value, saved in zip(actual, retained):
      assert value is None if saved is None else torch.equal(value, saved)
  for key, tensor in dynamics['poses'].items():
    assert torch.equal(tensor, before[key])


def test_local_to_global_rejects_invalid_ids_layouts_and_nonfinite_inputs():
  torch = pytest.importorskip('torch')
  from mujoco_metal import native_api
  model = _fixture()
  _, dynamics, _, _ = _stages(model, 17, 'cpu')
  sim = SimpleNamespace(model=model, _mjmodel=model, batch_size=1,
                        state=SimpleNamespace(_device=torch.device('cpu')))
  pos, quat = _inputs()
  for body, sameframe in ((-1, 0), (model.nbody, 0), (1, 5), (1, -1)):
    with pytest.raises(ValueError):
      native_api.mj_local2Global(sim, pos[:1], quat[:1], body, sameframe, dynamics=dynamics)
  for body, sameframe in ((True, 0), (1, False), (1, 1.5)):
    with pytest.raises(TypeError):
      native_api.mj_local2Global(sim, pos[:1], quat[:1], body, sameframe, dynamics=dynamics)
  with pytest.raises(ValueError):
    native_api.mj_local2Global(sim, pos, quat[:1], 1, dynamics=dynamics)
  invalid = pos[:1].copy()
  invalid[0, 0] = np.inf
  with pytest.raises(ValueError):
    native_api.mj_local2Global(sim, invalid, quat[:1], 1, dynamics=dynamics)


@pytest.mark.gpu
@pytest.mark.skipif(os.getenv('MUJOCO_METAL_RUN_GPU') != '1', reason='native opt-in')
@pytest.mark.parametrize('profile', ['integrated_euler_v1', 'integrated_scalable_v1'])
def test_native_local_to_global_public_profiles_owned_outputs_and_state_changes(profile, monkeypatch):
  import torch
  from mujoco_metal import native_api
  from mujoco_metal.simulation import MetalSimulation
  model = _fixture()
  worlds = [_stages(model, seed, 'cpu')[0] for seed in (17, 31)]
  qp = np.stack([data.qpos.copy() for data in worlds]).astype(np.float32)
  qv = np.stack([data.qvel.copy() for data in worlds]).astype(np.float32)
  for world, data in enumerate(worlds):
    data.qpos[:] = qp[world]
    data.qvel[:] = qv[world]
    mujoco.mj_forward(model, data)
  pos, quat = _inputs()
  expected = {(body, sf): _oracle(worlds, pos, quat, body, sf)
               for body in range(model.nbody) for sf in range(5)}
  sim = MetalSimulation(model, batch_size=2, qpos=qp, qvel=qv, profile=profile)
  def forbidden(*args, **kwargs):
    raise AssertionError('CPU frame/physics fallback invoked')
  for name in ('mj_local2Global', 'mj_forward', 'mj_kinematics', 'mj_comPos', 'mj_comVel'):
    monkeypatch.setattr(mujoco, name, forbidden)
  retained = []
  before_qp, before_qv = sim.state._qpos.clone(), sim.state._qvel.clone()
  for (body, sf), reference in expected.items():
    actual = native_api.mj_local2Global(sim, pos, quat, body, sf)
    for value, oracle in zip(actual, reference):
      assert value.device.type == 'mps'
      np.testing.assert_allclose(value.cpu().numpy(), oracle, atol=8e-6, rtol=5e-5)
    retained.append((actual, tuple(value.clone() for value in actual)))
  assert torch.equal(sim.state._qpos, before_qp)
  assert torch.equal(sim.state._qvel, before_qv)
  assert native_api.mj_local2Global(sim, None, None, 1) == (None, None)
  sim.state._qpos[:, 0].add_(.13)
  shifted, matrix = native_api.mj_local2Global(sim, pos, quat, 1)
  np.testing.assert_allclose(shifted.cpu().numpy(),
      expected[1, 0][0] + np.array([.13, 0, 0]), atol=8e-6, rtol=5e-5)
  np.testing.assert_allclose(matrix.cpu().numpy(), expected[1, 0][1], atol=8e-6, rtol=5e-5)
  sim.state._qpos.copy_(before_qp)
  restored = native_api.mj_local2Global(sim, pos, quat, 1)
  for value, oracle in zip(restored, expected[1, 0]):
    np.testing.assert_allclose(value.cpu().numpy(), oracle, atol=8e-6, rtol=5e-5)
  for result, saved in retained:
    for value, original in zip(result, saved):
      assert torch.equal(value, original)
