# Copyright 2026 The MuJoCo Metal contributors
# Licensed under the Apache License, Version 2.0.
"""Full compiled cloth metric and retained-coordinate invariant qualification."""
import os

import mujoco
import numpy as np
import pytest

torch = pytest.importorskip("torch")

from mujoco_metal.flex import MetalFlex
from mujoco_metal.capacity import _runtime_buffer_sizes


def _model():
  return mujoco.MjModel.from_xml_string('''<mujoco>
    <option gravity="0 0 0" jacobian="dense"/>
    <worldbody><flexcomp name="cloth" type="grid" count="2 2 1"
        spacing=".1 .1 .1" mass="1" dim="2">
      <contact contype="0" conaffinity="0"/>
    </flexcomp></worldbody>
    <equality><flexvert flex="cloth"/></equality>
  </mujoco>''')


def _run(device):
  model = _model()
  program = MetalFlex(model, batch_size=3, device=device)
  coordinates, point_jacobians, positions, jacobians = [], [], [], []
  for translation, perturbation in ((0.0, 0.0), (0.0, 3e-9), (1024.0, 3e-4)):
    data = mujoco.MjData(model)
    data.qpos[:] = model.qpos0 + translation + np.linspace(
        -perturbation, perturbation, model.nq)
    mujoco.mj_forward(model, data)
    coordinates.append(data.flexvert_xpos.copy())
    point_jac = np.zeros((model.nflexvert, 3, model.nv))
    for v, body in enumerate(model.flex_vertbodyid):
      mujoco.mj_jac(model, data, point_jac[v], None,
                    data.flexvert_xpos[v], int(body))
    point_jacobians.append(point_jac)
    positions.append(data.flexvert_length.copy())
    dense_jac = np.zeros((model.nflexvert, 2, model.nv))
    for v in range(model.nflexvert):
      for invariant in range(2):
        begin = int(model.flexvert_J_rowadr[v, invariant])
        count = int(model.flexvert_J_rownnz[v, invariant])
        cols = np.asarray(model.flexvert_J_colind).reshape(-1)[begin:begin+count]
        values = np.asarray(data.flexvert_J).reshape(-1)[begin:begin+count]
        dense_jac[v, invariant, cols] = values
    jacobians.append(dense_jac)
  full = np.asarray(coordinates, dtype=np.float64)
  high = full.astype(np.float32)
  residual = full - high.astype(np.float64)
  low = residual.astype(np.float32)
  tail = (residual - low.astype(np.float64)).astype(np.float32)
  assert np.any(low != 0)
  tensor = lambda x: torch.tensor(np.asarray(x), dtype=torch.float32, device=device)
  actual_pos, actual_jac = program._compute_flexvert_equalities(
      tensor(high), tensor(point_jacobians), tensor(low), tensor(tail))
  np.testing.assert_allclose(actual_pos.cpu().numpy(), positions,
                             rtol=3e-6, atol=2e-10)
  np.testing.assert_allclose(actual_jac.cpu().numpy(), jacobians,
                             rtol=3e-6, atol=6e-6)
  np.testing.assert_allclose(actual_pos.cpu().numpy()[0], 0.0, atol=2e-14)
  original = actual_pos.clone()
  wrong_pos, _ = program._compute_flexvert_equalities(
      tensor(high), tensor(point_jacobians), torch.zeros_like(tensor(low)),
      torch.zeros_like(tensor(tail)))
  assert np.max(np.abs(wrong_pos.cpu().numpy() - original.cpu().numpy())) > 1e-6
  inventory = dict(_runtime_buffer_sizes(model, 3))
  assert inventory["flex.vertex_equality_operands"] == program._vertex_equality_operands.numel()
  assert inventory["flex.vertex_equality_topology"] == program._vertex_equality_topology.numel()


def test_host_cloth_reference_precision_and_actual_owned_capacity():
  _run("cpu")


def test_native_cloth_reference_precision_and_actual_owned_capacity():
  if os.environ.get("MUJOCO_METAL_RUN_GPU") != "1" or not torch.backends.mps.is_available():
    pytest.skip("explicit native Metal qualification required")
  _run("mps")
