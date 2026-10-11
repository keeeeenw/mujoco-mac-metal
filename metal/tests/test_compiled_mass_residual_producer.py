# Copyright 2026 The MuJoCo Metal contributors
# Licensed under the Apache License, Version 2.0.
"""Execute source-constant residual lowering, native production and retention."""
from dataclasses import replace
import os

import mujoco
import numpy as np
import pytest

from mujoco_metal.model import load_model
from mujoco_metal.smooth_metal import position_context_shapes

_XML = '''<mujoco><option gravity="0 0 0"><flag contact="disable"/></option>
  <worldbody><body><freejoint/><inertial pos="0 0 0" mass="1.3"
    diaginertia=".023125 .024125 .03125"/></body></worldbody></mujoco>'''
_GPU = pytest.mark.skipif(os.getenv("MUJOCO_METAL_RUN_GPU") != "1",
                          reason="native mass residual producer")


def test_residual_producer_and_position_cache_admission():
  from mujoco_metal.capacity import _runtime_buffer_sizes
  for xml in (_XML, '<mujoco/>'):
    source = mujoco.MjModel.from_xml_string(xml)
    for storage in ("dense", "block_sparse"):
      parts = dict(_runtime_buffer_sizes(source, 2, mass_storage=storage))
      assert parts["model.smooth.dof_M0_low"] == max(source.nv, 1)
      expected = 2 * source.nv ** 2
      assert parts["smooth.mass_low"] == max(expected, 1)
      key = "mass_matrix_low" if storage == "dense" else "mass_blocks_low"
      assert parts[f"smooth.position_cache.{key}"] == max(expected, 1)
      shapes = position_context_shapes(source, 2, mass_storage=storage,
                                       component_nnz=source.nv ** 2)
      assert int(np.prod(shapes[key])) == expected


def _reference(source):
  data = mujoco.MjData(source)
  mujoco.mj_forward(source, data)
  mass = np.empty((source.nv, source.nv))
  mujoco.mj_fullM(source, data, mass)
  return mass


@pytest.mark.gpu
@_GPU
@pytest.mark.parametrize("storage", ["dense", "block_sparse", "query"])
def test_native_mass_retains_actual_source_constant_residuals(storage):
  import torch
  from mujoco_metal.smooth_metal import MetalSmoothDynamics
  source = mujoco.MjModel.from_xml_string(_XML)
  expected = _reference(source)
  smooth = MetalSmoothDynamics(load_model(source), 2,
      mass_storage="dense" if storage == "query" else storage)
  from mujoco_metal.capacity import _runtime_buffer_sizes
  parts = dict(_runtime_buffer_sizes(source, 2,
      mass_storage="dense" if storage == "query" else storage))
  assert parts["smooth.mass_low"] == smooth._workspace["mass_low"].numel()
  assert parts["model.smooth.dof_M0_low"] == smooth._arrays["dof_M0_low"].numel()
  q = torch.tensor(np.tile(source.qpos0, (2, 1)), dtype=torch.float32, device="mps")
  v = torch.zeros((2, source.nv), device="mps")
  if storage == "query":
    high, low = smooth.mass_matrix_pair(np.tile(source.qpos0, (2, 1)))
  else:
    result = smooth.run_device(q, v)
    key = "mass_matrix" if storage == "dense" else "mass_blocks"
    high, low = result[key], result[f"{key}_low"]
  actual = (high.cpu().numpy().astype(float) + low.cpu().numpy()).reshape(2, 6, 6)
  assert np.any(low.cpu().numpy() != 0)
  np.testing.assert_allclose(actual, np.tile(expected, (2, 1, 1)),
                             rtol=0, atol=2e-15)
  if storage != "query":
    high.fill_(123.25)
    low.fill_(.003125)
    lists = smooth._workspace["all_awake_lists"]
    lists["counts"][1].zero_()
    smooth.run_device(q, v, awake_lists=lists,
        world_mask=torch.tensor([0, 1], dtype=torch.int32, device="mps"))
    np.testing.assert_array_equal(high.cpu().numpy(), 123.25)
    np.testing.assert_array_equal(low.cpu().numpy(), np.float32(.003125))


@pytest.mark.gpu
@_GPU
@pytest.mark.parametrize("storage", ["dense", "block_sparse"])
def test_position_capture_owns_mass_residual_and_checks_metadata_before_velocity(storage):
  import torch
  from mujoco_metal.smooth_metal import MetalSmoothDynamics
  source = mujoco.MjModel.from_xml_string(_XML)
  smooth = MetalSmoothDynamics(load_model(source), 2, mass_storage=storage)
  q = torch.tensor(np.tile(source.qpos0, (2, 1)), dtype=torch.float32, device="mps")
  v = torch.zeros((2, source.nv), device="mps")
  result = smooth.run_device(q, v)
  key = "mass_matrix_low" if storage == "dense" else "mass_blocks_low"
  wanted = result[key].clone()
  context = smooth.position_context(result)
  result[key].fill_(99)
  cached = smooth.run_velocity_device(context, v)
  np.testing.assert_array_equal(cached[key].cpu().numpy(), wanted.cpu().numpy())
  # Invalid retained mass must be rejected before any bias or output writes.
  before = smooth._workspace["bias"].clone()
  corrupt = dict(context, **{key: torch.zeros((1,), device="mps")})
  with pytest.raises(ValueError, match=key):
    smooth.run_velocity_device(corrupt, v)
  np.testing.assert_array_equal(smooth._workspace["bias"].cpu().numpy(), before.cpu().numpy())
  smooth.prepare_workspace(2, mass_storage=storage)
  with pytest.raises(ValueError, match="another or replaced"):
    smooth.run_velocity_device(context, v)


@pytest.mark.gpu
@_GPU
@pytest.mark.parametrize("profile", ["integrated_euler_v1", "integrated_scalable_v1"])
def test_production_mass_pair_inverse_uses_native_producer_without_cpu_feedback(profile, monkeypatch):
  import torch
  from mujoco_metal.simulation import MetalSimulation
  source = mujoco.MjModel.from_xml_string(_XML)
  matrix = _reference(source)
  sim = MetalSimulation(source, 2, profile=profile)
  high = np.linspace(-.4, .5, 12, dtype=np.float32).reshape(2, 6)
  low = np.linspace(1e-8, -3e-8, 12, dtype=np.float32).reshape(2, 6)
  rhs = high.astype(float) + low
  expected = np.linalg.solve(matrix, rhs.T).T

  def forbidden(*args, **kwargs):
    raise AssertionError("CPU mass/forward feedback during native execution")

  monkeypatch.setattr(mujoco, "mj_forward", forbidden)
  monkeypatch.setattr(mujoco, "mj_fullM", forbidden)
  dynamics = sim._smooth.run_device(sim.state._qpos, sim.state._qvel)
  force = torch.tensor(high, device="mps")
  residual = torch.tensor(low, device="mps")
  result_hi, result_lo, status = sim._inverse_discrete_mass_pair_solve(
      dynamics, force, residual)
  np.testing.assert_array_equal(status.cpu().numpy(), 0)
  actual = result_hi.cpu().numpy().astype(float) + result_lo.cpu().numpy()
  np.testing.assert_allclose(actual, expected, rtol=3e-13, atol=3e-13)
  product_hi, product_low = sim._inverse_discrete_mass_pair_product(
      dynamics, force, residual)
  product = product_hi.cpu().numpy().astype(float) + product_low.cpu().numpy()
  np.testing.assert_allclose(product, np.einsum("ij,bj->bi", matrix, rhs),
                             rtol=3e-13, atol=3e-13)


@pytest.mark.gpu
@_GPU
def test_recompiled_constants_refresh_and_invalidated_simple_rows_have_zero_residuals():
  from mujoco_metal.smooth_metal import MetalSmoothDynamics
  source = mujoco.MjModel.from_xml_string(_XML)
  original = load_model(source)
  source.body_mass[1] = 2.7
  source.body_inertia[1] = [.042, .053, .066]
  mujoco.mj_setConst(source, mujoco.MjData(source))
  fresh = MetalSmoothDynamics(load_model(source))
  high, low = fresh.mass_matrix_pair(source.qpos0)
  np.testing.assert_allclose(high.cpu().numpy().astype(float) + low.cpu().numpy(),
                             _reference(source)[None], rtol=0, atol=2e-15)
  edited = replace(original, body_mass=np.asarray(original.body_mass) * 2)
  invalidated = MetalSmoothDynamics(edited)
  assert not np.any(invalidated.model.dof_simplenum)
  _, low = invalidated.mass_matrix_pair(source.qpos0)
  np.testing.assert_array_equal(low.cpu().numpy(), 0)


@pytest.mark.gpu
@_GPU
@pytest.mark.parametrize("storage", ["dense", "block_sparse"])
def test_empty_model_residual_backings_match_actual_admission(storage):
  import torch
  from mujoco_metal.capacity import _runtime_buffer_sizes
  from mujoco_metal.smooth_metal import MetalSmoothDynamics
  source = mujoco.MjModel.from_xml_string('<mujoco/>')
  smooth = MetalSmoothDynamics(load_model(source), 2, mass_storage=storage)
  parts = dict(_runtime_buffer_sizes(source, 2, mass_storage=storage))
  assert parts["smooth.mass_low"] == smooth._workspace["mass_low"].numel() == 1
  assert parts["model.smooth.dof_M0_low"] == smooth._arrays["dof_M0_low"].numel() == 1
  result = smooth.run_device(torch.zeros((2, 0), device="mps"),
                            torch.zeros((2, 0), device="mps"))
  key = "mass_matrix_low" if storage == "dense" else "mass_blocks_low"
  assert result[key].numel() == 0
  context = smooth.position_context(result)
  assert smooth.run_velocity_device(context, torch.zeros((2, 0), device="mps"))[key].numel() == 0
