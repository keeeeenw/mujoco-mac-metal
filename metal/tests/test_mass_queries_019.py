# Copyright 2026 The MuJoCo Metal contributors
# Licensed under the Apache License, Version 2.0.
"""Actual pinned mass-query oracles; native math dispatch separately opt-in."""
import os

import mujoco
import numpy as np
import pytest


def _fixture():
  return mujoco.MjModel.from_xml_string('''<mujoco>
    <option gravity="0 0 0"><flag contact="disable"/></option>
    <worldbody><body><freejoint/><geom mass="2" size=".2" pos=".1 .05 .03"/>
      <body pos=".4 .1 .05"><joint type="ball" armature=".1"/>
        <geom size=".1" mass="1" pos=".1 .1 .04"/>
        <body pos=".3 -.1 .05"><joint name="h" axis="0 0 1"/>
          <geom size=".1" mass=".7" pos=".1 .04 .02"/>
          <body pos=".2 .1 .03"><joint name="s" type="slide" axis="0 1 0"/>
            <geom size=".1" mass=".6"/>
          </body></body></body></body>
      <body pos="2 0 0"><joint type="slide" axis="1 0 0"/>
        <geom size=".1" mass="1.3"/></body></worldbody>
    <tendon><fixed armature=".3"><joint joint="h" coef="1"/>
      <joint joint="s" coef="-.7"/></fixed></tendon>
  </mujoco>''')


def _stage(sparse, device):
  torch = pytest.importorskip('torch')
  model = _fixture()
  rng = np.random.default_rng(1903)
  masses, vectors, full, product, root, solve, half = [], [], [], [], [], [], []
  for _ in range(3):
    data = mujoco.MjData(model)
    data.qpos[:] += rng.normal(0, .12, model.nq)
    mujoco.mj_normalizeQuat(model, data.qpos)
    mujoco.mj_forward(model, data)
    mass = np.empty((model.nv, model.nv))
    mujoco.mj_fullM(model, data, mass)
    vector = rng.normal(0, .4, (2, model.nv))
    a, b, c, e = [np.empty_like(vector) for _ in range(4)]
    for row in range(2):
      mujoco.mj_mulM(model, data, a[row], vector[row])
      mujoco.mj_mulM2(model, data, b[row], vector[row])
    mujoco.mj_solveM(model, data, c, vector)
    mujoco.mj_solveM2(model, data, e, vector, np.sqrt(data.qLDiagInv))
    masses.append(mass); vectors.append(vector)
    full.append(mass); product.append(a); root.append(b); solve.append(c); half.append(e)
  tensor = lambda value: torch.tensor(np.asarray(value), dtype=torch.float32, device=device)
  if sparse:
    # Independent compiled-tree partition, not the production packer: the
    # tendon joins only DOFs in the first tree, and the second slide is free.
    tree = np.asarray(model.body_treeid)[model.dof_bodyid]
    groups = [np.flatnonzero(tree == value) for value in np.unique(tree)]
    widths = np.asarray([len(ids) for ids in groups], dtype=np.int32)
    layout = dict(ncomponent=len(groups), nnz=int(np.sum(widths**2)),
        component_dofnum=widths, component_dof_ids=np.concatenate(groups),
        component_dof_offsets=np.r_[0, np.cumsum(widths)],
        component_mass_offsets=np.r_[0, np.cumsum(widths**2)],
        dof_component=tree)
    def pack(values):
      result = np.zeros((3, layout['nnz']))
      for component, width in enumerate(layout['component_dofnum']):
        begin = int(layout['component_dof_offsets'][component])
        ids = layout['component_dof_ids'][begin:begin+int(width)]
        offset = int(layout['component_mass_offsets'][component])
        result[:, offset:offset+int(width)**2] = np.asarray(values)[:, ids[:, None], ids[None, :]].reshape(3, -1)
      return tensor(result)
    armature = np.zeros_like(masses)
    j = np.zeros(model.nv)
    j[model.jnt_dofadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, 'h')]] = 1
    j[model.jnt_dofadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, 's')]] = -.7
    armature[:] = .3 * np.outer(j, j)
    stage = {'mass_blocks': pack(np.asarray(masses)-armature),
             'tendon_armature_blocks': pack(armature), 'mass_block_layout': layout}
  else:
    stage = {'mass_matrix': tensor(masses)}
  return stage, tensor(vectors), dict(full=full, product=product, sqrt=root,
                                     solve=solve, half_solve=half)


@pytest.mark.parametrize('sparse', [False, True])
@pytest.mark.parametrize('device', ['cpu', pytest.param('mps', marks=[
    pytest.mark.gpu, pytest.mark.skipif(os.getenv('MUJOCO_METAL_RUN_GPU') != '1', reason='opt-in native query math')])])
def test_mass_query_products_factors_and_multi_rhs_match_actual_pinned_engine(sparse, device):
  from mujoco_metal.mass_queries import full_mass, mass_product, mass_factor_query
  stage, vectors, expected = _stage(sparse, device)
  def host(value):
    return value.detach().cpu().numpy()
  np.testing.assert_allclose(host(full_mass(stage)), expected['full'], atol=2e-6, rtol=2e-6)
  np.testing.assert_allclose(host(mass_product(stage, vectors)), expected['product'], atol=2e-6, rtol=2e-6)
  for operation in ('sqrt', 'solve', 'half_solve'):
    value, status = mass_factor_query(stage, vectors, operation)
    assert not host(status).any()
    np.testing.assert_allclose(host(value), expected[operation], atol=8e-5, rtol=8e-5)
    # Two-dimensional input is one RHS per world, not a cross-world batch.
    single, status = mass_factor_query(stage, vectors[:, 0], operation)
    np.testing.assert_allclose(host(single), np.asarray(expected[operation])[:, 0], atol=8e-5, rtol=8e-5)
  # Distinguish the pinned lower-triangular sqrt factor from the symmetric
  # principal square root; the fixture must exercise nonzero off-diagonals.
  assert np.max(np.abs(np.asarray(expected['full'])[:, 0, 3:])) > .1


@pytest.mark.parametrize('operation', ['sqrt', 'solve', 'half_solve'])
def test_factor_queries_isolate_non_spd_nonfinite_and_overflow_worlds(operation):
  torch = pytest.importorskip('torch')
  from mujoco_metal.mass_queries import mass_factor_query
  mass = torch.tensor([[[2., .1], [.1, 1.]], [[-1., 0], [0, 1]],
                       [[1., float('nan')], [0, 1]], [[1., .3], [.1, 1.]]])
  vector = torch.ones((4, 2))
  value, status = mass_factor_query({'mass_matrix': mass}, vector, operation)
  assert status.tolist() == [0, 2, 2, 2]
  assert torch.isfinite(value).all()
  assert not value[1:].any()
  assert value[0].abs().sum() > .1
  result, status = mass_factor_query({'mass_matrix': mass[:1]}, torch.tensor([[float('inf'), 0]]), operation)
  assert status.tolist() == [2] and not result.any()


def test_zero_dof_query_and_invalid_rhs_metadata():
  torch = pytest.importorskip('torch')
  from mujoco_metal.mass_queries import mass_factor_query, full_mass, mass_product
  stage = {'mass_matrix': torch.zeros(2, 0, 0)}
  empty = torch.zeros(2, 3, 0)
  assert full_mass(stage).shape == (2, 0, 0)
  assert mass_product(stage, empty).shape == (2, 3, 0)
  value, status = mass_factor_query(stage, empty, 'solve')
  assert value.shape == empty.shape and status.tolist() == [0, 0]
  with pytest.raises(ValueError, match='vector must'):
    mass_product(stage, torch.zeros(1, 0))
  with pytest.raises(ValueError, match='unknown'):
    mass_factor_query(stage, empty, 'bogus')
