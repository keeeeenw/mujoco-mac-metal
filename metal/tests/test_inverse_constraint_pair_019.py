# Copyright 2026 The MuJoCo Metal contributors
# Licensed under the Apache License, Version 2.0.
"""Two-float inverse-row reduction tests for retained aref residuals."""
from types import SimpleNamespace

import numpy as np
import pytest
import mujoco


def test_inverse_constraint_pair_preserves_subulp_aref_directional_force():
  torch = pytest.importorskip("torch")
  from mujoco_metal.inverse_constraints import inverse_constraint_force

  # Captured equality row 2 from the pinned rich inverse-FD fixture, at the
  # base and qpos[2] + 2e-4 samples.  The aref high words alone lose most of
  # this directional change; the residual words restore it.
  eps = np.float64(0.00020000338554382324)
  row_r = np.float32(0.018821497)
  aref_hi = np.array([-263.7594, -263.23306], dtype=np.float32)
  aref_low = np.array([6.806302e-6, -5.164822e-6], dtype=np.float32)
  descriptor = SimpleNamespace(
      nr=1, nv=1, n_eq_rows=1, ncontacts_max=0, cone_type=0)
  rows = {
      "J": torch.tensor([[[-1.]], [[-1.]]], dtype=torch.float32),
      "R": torch.full((2, 1), float(row_r), dtype=torch.float32),
      "ar": torch.from_numpy(aref_hi[:, None].copy()),
      "ar_low": torch.from_numpy(aref_low[:, None].copy()),
      "lo": torch.full((2, 1), -float("inf"), dtype=torch.float32),
      "hi": torch.full((2, 1), float("inf"), dtype=torch.float32),
      "active": torch.ones((2, 1), dtype=torch.float32),
  }
  qacc = torch.zeros((2, 1), dtype=torch.float32)
  high, low = inverse_constraint_force(
      rows, qacc, descriptor, return_low=True)

  # CPU oracle evaluates the same pinned row map and impedance from the
  # captured float32 inputs in binary64.  J=-1 and qacc=0, so J.T*(-jar/R)
  # reduces to -aref/R exactly.
  source_aref = aref_hi.astype(np.float64) + aref_low.astype(np.float64)
  expected = -source_aref / np.float64(row_r)
  actual_direction = (
      (high[1, 0] - high[0, 0]) + (low[1, 0] - low[0, 0])) / float(eps)
  expected_direction = (expected[1] - expected[0]) / eps
  assert abs(float(actual_direction) - expected_direction) < 0.02

  high_only_direction = float((high[1, 0] - high[0, 0]) / float(eps))
  assert abs(high_only_direction - expected_direction) > 0.5

  # The ordinary API remains the same high-word tensor shape and input rows
  # stay borrowed/read-only.
  ordinary = inverse_constraint_force(rows, qacc, descriptor)
  torch.testing.assert_close(ordinary, high, rtol=0, atol=0)
  assert tuple(low.shape) == tuple(high.shape) == (2, 1)


def test_inverse_constraint_force_consumes_paired_acceleration_input():
  torch = pytest.importorskip("torch")
  from mujoco_metal.inverse_constraints import inverse_constraint_force
  descriptor = SimpleNamespace(
      nr=1, nv=1, n_eq_rows=1, ncontacts_max=0, cone_type=0)
  rows = {
      "J": torch.ones((1, 1, 1), dtype=torch.float32),
      "R": torch.ones((1, 1), dtype=torch.float32),
      "ar": torch.zeros((1, 1), dtype=torch.float32),
      "ar_low": torch.zeros((1, 1), dtype=torch.float32),
      "lo": torch.full((1, 1), -float("inf"), dtype=torch.float32),
      "hi": torch.full((1, 1), float("inf"), dtype=torch.float32),
      "active": torch.ones((1, 1), dtype=torch.float32),
  }
  high, low = inverse_constraint_force(
      rows, torch.ones((1, 1), dtype=torch.float32), descriptor,
      qacc_low=torch.tensor([[2.0**-24]], dtype=torch.float32),
      return_low=True)
  represented = float(high.item()) + float(low.item())
  assert abs(represented + (1.0 + 2.0**-24)) < 1e-14


def test_pair_products_scale_extreme_operands_and_mask_invalid_inactive_rows():
  torch = pytest.importorskip("torch")
  from mujoco_metal.inverse_constraints import (
      _two_prod, inverse_constraint_force)

  left = torch.tensor([1e35, 1e-35, 1e20, 1e-20], dtype=torch.float32)
  right = torch.tensor([1e-20, 1e20, 1e-20, 1e20], dtype=torch.float32)
  product, residual = _two_prod(left, right)
  expected = left.double() * right.double()
  torch.testing.assert_close(product.double() + residual.double(), expected,
                             rtol=2e-15, atol=0)

  rows = {
      "J": torch.tensor([[[1.0]], [[float("nan")]]], dtype=torch.float32),
      "R": torch.tensor([[.5], [0.]], dtype=torch.float32),
      "ar": torch.zeros((2, 1), dtype=torch.float32),
      "ar_low": torch.zeros((2, 1), dtype=torch.float32),
      "lo": torch.full((2, 1), -float("inf"), dtype=torch.float32),
      "hi": torch.full((2, 1), float("inf"), dtype=torch.float32),
      "active": torch.tensor([[1.], [0.]], dtype=torch.float32),
  }
  descriptor = SimpleNamespace(
      nr=1, nv=1, n_eq_rows=0, ncontacts_max=0, cone_type=0)
  high, low = inverse_constraint_force(
      rows, torch.tensor([[2.], [float("nan")]]), descriptor,
      return_low=True)
  assert torch.isfinite(high).all() and torch.isfinite(low).all()
  torch.testing.assert_close(high, torch.tensor([[-4.], [0.]]), rtol=0, atol=0)


@pytest.mark.parametrize("storage", ["dense", "csr"])
@pytest.mark.parametrize("side", ["top", "bottom"])
@pytest.mark.parametrize("condim", [3, 4, 6])
def test_elliptic_region_boundary_uses_full_aref_pair(storage, side, condim):
  torch = pytest.importorskip("torch")
  from mujoco_metal.inverse_constraints import inverse_constraint_force
  friction = np.array([.7, .03, .02, .01, .005], dtype=np.float32)
  coeff = torch.from_numpy(friction[:condim-1].copy())
  mu = torch.tensor(friction[0], dtype=torch.float32)
  tangent = torch.linspace(.4, 1.1, condim-1, dtype=torch.float32)
  tangent_u = tangent * coeff
  T = torch.linalg.vector_norm(tangent_u)
  normal = T.clone() if side == "top" else -T / mu.square()
  if side == "bottom":
    # Make the high-word-only test land strictly in the bottom region, then
    # choose aref_low so the represented row residual is in the middle.
    for _ in range(8):
      N = mu * (mu * normal)
      margin = N + T
      if float(margin) < 0:
        break
      normal = torch.nextafter(normal, torch.tensor(-float("inf")))
    assert float(margin) < 0
    normal_low = (-float(margin) / float(mu.square())) * 1.5 + 1e-5
  else:
    # Exactly on the top boundary in high words.  A negative residual low
    # moves the true represented point into the smooth middle region.
    normal_low = -1e-5

  jar_hi = torch.cat((normal.reshape(1), tangent))
  jar_low = torch.zeros_like(jar_hi)
  jar_low[0] = normal_low
  ar = -jar_hi
  ar_low = -jar_low
  J = torch.eye(condim, dtype=torch.float32)[None]
  rows = dict(J=J, R=torch.ones((1, condim)), ar=ar[None],
      ar_low=ar_low[None], lo=torch.full((1, condim), -float("inf")),
      hi=torch.full((1, condim), float("inf")),
      active=torch.ones((1, condim)), contact_mask=torch.ones((1, 1)),
      contact_friction=torch.from_numpy(friction[None].copy()))
  if storage == "csr":
    from mujoco_metal.constraint_jacobian import (
        ConstraintJacobianPattern, PACKED_J_CSR,
        packed_jacobian_device_storage, packed_jacobian_scatter_rows_torch)
    offsets = np.arange(condim+1, dtype=np.int32)
    pattern = ConstraintJacobianPattern(
        offsets, np.arange(condim, dtype=np.int32), condim, condim)
    packed, layout = packed_jacobian_device_storage(
        torch, "cpu", 1, pattern, mode=PACKED_J_CSR)
    packed_jacobian_scatter_rows_torch(
        torch, packed, 1, layout, pattern, J, 0, row_count=condim,
        active=rows["active"])
    rows.pop("J")
    rows.update(J_packed=packed, jacobian_layout=layout,
                jacobian_pattern=pattern)
  descriptor = SimpleNamespace(
      nr=condim, nv=condim, n_eq_rows=0, ncontacts_max=1,
      nr_joint=0, cone_type=int(mujoco.mjtCone.mjCONE_ELLIPTIC),
      contact_condim_packed=np.array(
          [[condim, 0, int(mujoco.mjtCone.mjCONE_ELLIPTIC)]], np.int32))
  high, low = inverse_constraint_force(
      rows, torch.zeros((1, condim)), descriptor, return_low=True)

  actual_jar = jar_hi.double().numpy() + jar_low.double().numpy()
  coeff64 = friction[:condim-1].astype(np.float64)
  mu64 = float(friction[0])
  U = np.r_[mu64*actual_jar[0], coeff64*actual_jar[1:]]
  N64, T64 = U[0], np.linalg.norm(U[1:])
  assert N64 < mu64*T64
  assert mu64*N64 + T64 > 0
  scale = -(N64-mu64*T64)*mu64 / (mu64**2 * (1+mu64**2))
  expected = np.r_[scale, -coeff64*scale*U[1:]/T64]
  represented = (high.double() + low.double()).numpy()[0]
  np.testing.assert_allclose(represented, expected, atol=2e-6, rtol=2e-5)
