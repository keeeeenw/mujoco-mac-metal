# Copyright 2026 The MuJoCo Metal contributors
# Licensed under the Apache License, Version 2.0.
"""Signed Euler effective matrices and their pinned integration witness."""
import os
import sys

import mujoco
import numpy as np
import pytest

from mujoco_metal.smooth_solve import (
    MetalSymmetricLDLSolve, symmetric_ldl_workspace_elements)


def _model():
  return mujoco.MjModel.from_xml_string('''<mujoco>
    <option gravity="0 0 0" timestep=".1"><flag contact="disable"/></option>
    <worldbody>
      <body><joint type="slide" axis="1 0 0" damping="2"/>
        <geom size=".1" mass="1" contype="0" conaffinity="0"/></body>
      <body pos="1 0 0"><joint type="slide" axis="1 0 0" damping="-20"/>
        <geom size=".1" mass="1" contype="0" conaffinity="0"/></body>
    </worldbody></mujoco>''')


def test_pinned_euler_accepts_signed_effective_matrix():
  model = _model()
  data = mujoco.MjData(model)
  data.qvel[:] = [.4, -.2]
  data.qfrc_applied[:] = [2, -3]
  mujoco.mj_forward(model, data)
  mass = np.zeros((model.nv, model.nv))
  mujoco.mj_fullM(model, data, mass)
  effective = mass + model.opt.timestep * np.diag(model.dof_damping)
  np.testing.assert_allclose(effective, np.diag([1.2, -1]))
  # Cholesky rejection here is not a physics error: the actual MuJoCo Euler
  # path factors this signed matrix with descending L' D L.
  with pytest.raises(np.linalg.LinAlgError):
    np.linalg.cholesky(effective)
  rhs = (data.qfrc_smooth + data.qfrc_constraint).copy()
  acceleration = np.linalg.solve(effective, rhs)
  old_velocity = data.qvel.copy()
  old_position = data.qpos.copy()
  mujoco.mj_Euler(model, data)
  np.testing.assert_allclose(data.qvel,
      old_velocity + model.opt.timestep * acceleration, atol=1e-15)
  np.testing.assert_allclose(data.qpos,
      old_position + model.opt.timestep * data.qvel, atol=1e-15)
  np.testing.assert_allclose(data.qHDiagInv, [1 / 1.2, -1])


def test_signed_solve_capacity_preflight_precedes_torch(monkeypatch):
  monkeypatch.setitem(sys.modules, "torch", None)
  counts = symmetric_ldl_workspace_elements(0, 2)
  assert counts["default_dof_ids"] == 2
  assert counts["default_counts"] == 6
  for args in ((-1, 1), (1, 0), (65536, 1),
               (0, ((1 << 32) - 1) // 3 + 1)):
    with pytest.raises(ValueError):
      MetalSymmetricLDLSolve(*args)
  for args in ((True, 1), (1, 1.5)):
    with pytest.raises(TypeError):
      MetalSymmetricLDLSolve(*args)


native = pytest.mark.skipif(os.getenv("MUJOCO_METAL_RUN_GPU") != "1",
                            reason="native opt-in")


@pytest.mark.gpu
@native
@pytest.mark.parametrize("nrhs", [1, 3])
def test_native_signed_solve_reverse_order_multiple_rhs_and_reuse(nrhs):
  import torch
  # First world needs a negative final Schur pivot, despite a zero original
  # top-left diagonal. The others exercise scale without a positive cutoff.
  matrices = np.stack([[[0., 1.], [1., 1.]],
                       [[1.2, 0.], [0., -1.]],
                       [[1.e-25, 0.], [0., -2.e-25]]]).astype(np.float32)
  shape = (3, 2) if nrhs == 1 else (3, 2, nrhs)
  rhs = np.arange(1, np.prod(shape) + 1, dtype=np.float32).reshape(shape)
  rhs[2] *= 1.e-25
  def tensor(value):
    return torch.tensor(value, dtype=torch.float32, device="mps").contiguous()
  mass, right = tensor(matrices), tensor(rhs)
  stage = MetalSymmetricLDLSolve(2, 3, nrhs)
  solution_pointer = None
  for scale in (1., -.5, 2.):
    right.copy_(tensor(scale * rhs))
    actual, status = stage.run_device(mass, right)
    np.testing.assert_array_equal(status.cpu().numpy(), 0)
    expected = np.stack([np.linalg.solve(m, r)
                         for m, r in zip(matrices, scale * rhs)])
    np.testing.assert_allclose(actual.cpu().numpy(), expected, rtol=3e-6, atol=3e-6)
    np.testing.assert_array_equal(mass.cpu().numpy(), matrices)
    np.testing.assert_array_equal(right.cpu().numpy(), scale * rhs)
    if solution_pointer is not None:
      assert actual.data_ptr() == solution_pointer
    solution_pointer = actual.data_ptr()


@pytest.mark.gpu
@native
def test_native_signed_solve_failures_sleep_and_zero_dofs():
  import torch
  def tensor(value, dtype=torch.float32):
    return torch.tensor(value, dtype=dtype, device="mps").contiguous()
  matrices = np.array([[[1.2, 0], [0, -1]], [[1, 0], [0, 0]],
                       [[1, .1], [0, 1]], [[1, 0], [0, np.nan]]], np.float32)
  stage = MetalSymmetricLDLSolve(2, 4)
  mass = tensor(matrices)
  actual, status = stage.run_device(mass, tensor(np.ones((4, 2))))
  np.testing.assert_array_equal(status.cpu().numpy(), [0, 3, 2, 1])
  np.testing.assert_allclose(actual.cpu().numpy()[0], [1 / 1.2, -1], rtol=2e-6)
  np.testing.assert_array_equal(actual.cpu().numpy()[1:], 0)
  # Sleeping NaNs are outside the selected principal system. One world is
  # entirely asleep; another solves a valid negative awake pivot.
  awake = {"dof_ids": tensor([[1, -1], [-1, -1], [0, -1], [0, -1]], torch.int32),
           "counts": tensor([[0, 0, 1], [0, 0, 0], [0, 0, 1], [0, 0, 1]], torch.int32)}
  retained = tensor(np.full((4, 2), 7))
  right = tensor([[np.nan, 2], [np.nan, np.nan], [3, np.nan], [4, np.nan]])
  actual, status = stage.run_device(mass, right, awake_lists=awake, retained=retained)
  np.testing.assert_array_equal(status.cpu().numpy(), 0)
  np.testing.assert_allclose(actual.cpu().numpy(), [[7, -2], [7, 7], [3, 7], [4, 7]])
  empty = MetalSymmetricLDLSolve(0, 2)
  actual, status = empty.run_device(torch.empty((2, 0, 0), device="mps"),
                                    torch.empty((2, 0), device="mps"))
  assert actual.shape == (2, 0)
  np.testing.assert_array_equal(status.cpu().numpy(), 0)
