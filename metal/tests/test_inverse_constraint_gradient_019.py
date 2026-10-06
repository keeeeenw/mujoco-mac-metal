# Copyright 2026 The MuJoCo Metal contributors
# Licensed under the Apache License, Version 2.0.
"""Actual pinned MuJoCo oracle for canonical inverse cost gradients."""
import os
from types import SimpleNamespace

import mujoco
import numpy as np
import pytest


@pytest.mark.parametrize('device', [
    'cpu', pytest.param('mps', marks=[pytest.mark.gpu,
        pytest.mark.skipif(os.getenv('MUJOCO_METAL_RUN_GPU') != '1', reason='native opt-in')])])
@pytest.mark.parametrize('condim', [3, 4, 6])
@pytest.mark.parametrize('cone', ['pyramidal', 'elliptic'])
@pytest.mark.parametrize('region', [
    'top', 'bottom', 'middle', 'near-top', 'near-bottom'])
@pytest.mark.parametrize('jacobian_storage', ['dense', 'csr'])
def test_inverse_gradient_matches_pinned_contact_cost(
    device, condim, cone, region, jacobian_storage):
  torch = pytest.importorskip('torch')
  from mujoco_metal.inverse_constraints import inverse_constraint_force
  model = mujoco.MjModel.from_xml_string(f'''<mujoco>
    <option gravity="0 0 0" cone="{cone}" impratio="3"/>
    <default><geom condim="{condim}" friction=".7 .03 .02"/></default>
    <worldbody><geom type="plane" size="2 2 .1"/>
      <body pos="0 0 .09"><freejoint/><geom size=".1" mass="1"/></body>
    </worldbody></mujoco>''')
  data = mujoco.MjData(model)
  data.qvel[:] = [.03, -.02, -.3, .5, -.1, .2]
  mujoco.mj_inverse(model, data)
  nr, nv = data.nefc, model.nv
  assert data.ncon == 1 and nr > 0
  J = data.efc_J.reshape(nr, nv).copy()
  if region == 'top':
    jar = np.full(nr, 1000.)
  elif region == 'bottom':
    jar = np.full(nr, -1000.)
  elif region in ('near-top', 'near-bottom') and cone == 'elliptic':
    jar = np.zeros(nr)
    tangent = np.linspace(0.4, 1.1, condim - 1)
    friction = np.asarray(data.contact[0].friction)
    mu = float(friction[0])
    T = np.linalg.norm(tangent * friction[:condim-1])
    # Put the represented residual just to either side of the cone boundary.
    jar[1:condim] = tangent
    jar[0] = (T * (1 + 2e-7) if region == 'near-top'
              else -T/mu * (1 - 2e-7))
  else:
    jar = (-5 + np.tile([-20., 20.], nr//2) if cone == 'pyramidal'
           else np.linspace(-10, 30, nr))
  if cone == 'elliptic':
    if region not in ('middle', 'near-top', 'near-bottom'):
      jar[1:] = 0
    elif region == 'middle':
      jar[0] = 0
  data.qacc[:] = np.linalg.lstsq(J, jar+data.efc_aref, rcond=None)[0]
  mujoco.mj_inverse(model, data)
  if region == 'top' or (region == 'near-top' and cone == 'elliptic'):
    assert np.linalg.norm(data.qfrc_constraint) < 1e-8
  else:
    assert np.linalg.norm(data.qfrc_constraint) > 1
  def tensor(value):
    return torch.tensor(np.asarray(value).copy()[None], dtype=torch.float32, device=device)
  aref_hi = np.asarray(data.efc_aref, dtype=np.float32)
  aref_low = (np.asarray(data.efc_aref, dtype=np.float64)
              - aref_hi.astype(np.float64)).astype(np.float32)
  lo = np.full(nr, 0 if cone == 'pyramidal' else -np.inf)
  rows = dict(J=tensor(J), R=tensor(data.efc_R), ar=tensor(aref_hi),
      ar_low=tensor(aref_low),
      lo=tensor(lo), hi=tensor(np.full(nr, np.inf)), active=tensor(np.ones(nr)),
      contact_mask=tensor([1.]), contact_friction=tensor(data.contact[0].friction))
  if jacobian_storage == 'csr':
    from mujoco_metal.constraint_jacobian import (
        ConstraintJacobianPattern, PackedJacobianLayout, PACKED_J_CSR,
        packed_jacobian_device_storage, packed_jacobian_scatter_rows_torch)
    row_columns = [np.flatnonzero(J[row] != 0).astype(np.int32)
                   for row in range(nr)]
    row_offsets = np.zeros(nr + 1, dtype=np.int32)
    for row, support in enumerate(row_columns):
      row_offsets[row + 1] = row_offsets[row] + len(support)
    columns = (np.concatenate(row_columns) if row_offsets[-1]
               else np.zeros((0,), dtype=np.int32))
    pattern = ConstraintJacobianPattern(row_offsets, columns, nv, nr)
    assert pattern.nnz < nr * nv
    packed, layout = packed_jacobian_device_storage(
        torch, device, 1, pattern, mode=PACKED_J_CSR)
    packed_jacobian_scatter_rows_torch(
        torch, packed, 1, layout, pattern, tensor(J), 0,
        row_count=nr, active=rows['active'])
    rows.pop('J')
    rows.update(J_packed=packed, jacobian_layout=layout,
                jacobian_pattern=pattern)
  desc = SimpleNamespace(nr=nr, nv=nv, n_eq_rows=0, ncontacts_max=1,
      nr_joint=0, cone_type=int(model.opt.cone),
      contact_condim_packed=np.array([[condim, 0, int(model.opt.cone)]], np.int32))
  original = {name: value.clone() for name, value in rows.items()
              if hasattr(value, 'clone')}
  result = inverse_constraint_force(rows, tensor(data.qacc), desc)
  assert result.device.type == device
  np.testing.assert_allclose(result.detach().cpu().numpy()[0], data.qfrc_constraint,
                             atol=4e-3, rtol=3e-5)
  paired_hi, paired_low = inverse_constraint_force(
      rows, tensor(data.qacc), desc, return_low=True)
  paired = (paired_hi.detach().cpu().double()
            + paired_low.detach().cpu().double()).numpy()[0]
  np.testing.assert_allclose(paired, data.qfrc_constraint,
                             atol=4e-3, rtol=3e-5)
  for name, value in original.items():
    torch.testing.assert_close(rows[name], value, rtol=0, atol=0)


def test_inverse_gradient_ignores_inactive_rows_without_clipping_valid_small_impedance():
  torch = pytest.importorskip('torch')
  from mujoco_metal.inverse_constraints import inverse_constraint_force
  rows = dict(J=torch.eye(2)[None], R=torch.tensor([[1e-16, .25]]),
      ar=torch.zeros((1, 2)), lo=torch.full((1, 2), -float('inf')),
      hi=torch.full((1, 2), float('inf')), active=torch.tensor([[1., 0.]]))
  desc = SimpleNamespace(nr=2, nv=2, n_eq_rows=2, ncontacts_max=0, cone_type=0)
  result = inverse_constraint_force(rows, torch.tensor([[1., 100.]]), desc)
  torch.testing.assert_close(result, torch.tensor([[-1e16, 0.]]), rtol=1e-6, atol=0)


def test_inverse_gradient_handles_empty_dofs_and_rejects_foreign_row_shapes():
  torch = pytest.importorskip('torch')
  from mujoco_metal.inverse_constraints import inverse_constraint_force
  desc = SimpleNamespace(nr=0, nv=0)
  result = inverse_constraint_force({}, torch.zeros((3, 0)), desc)
  assert result.shape == (3, 0)
  desc = SimpleNamespace(nr=1, nv=2, n_eq_rows=1, ncontacts_max=0, cone_type=0)
  with pytest.raises(ValueError, match='rows.J'):
    inverse_constraint_force({'J': torch.zeros((2, 1, 2))}, torch.zeros((1, 2)), desc)
