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


def _pinned_factors(model, data):
  """Decode the pinned engine's CSR qLD, including its stored diagonal D."""
  lower = np.eye(model.nv)
  diagonal = np.empty(model.nv)
  for dof in range(model.nv):
    begin = int(model.M_rowadr[dof])
    count = int(model.M_rownnz[dof])
    columns = np.asarray(model.M_colind[begin:begin+count])
    assert columns[-1] == dof
    lower[dof, columns[:-1]] = data.qLD[begin:begin+count-1]
    diagonal[dof] = data.qLD[begin+count-1]
  return lower, diagonal, data.qLDiagInv.copy()


def _stage(sparse, device):
  torch = pytest.importorskip('torch')
  model = _fixture()
  rng = np.random.default_rng(1903)
  masses, vectors, full, product, root, solve, half = [], [], [], [], [], [], []
  lower_factors, factor_diagonals, inverse_diagonals = [], [], []
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
    # Decode actual pinned CSR qLD; its diagonal entries are D, while its
    # off-diagonal entries are the unit-lower L. Do not derive this oracle
    # by factorizing the test matrix with the implementation under test.
    lower, diagonal, inverse_diagonal = _pinned_factors(model, data)
    lower_factors.append(lower)
    factor_diagonals.append(diagonal)
    inverse_diagonals.append(inverse_diagonal)
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
      solve=solve, half_solve=half, lower=lower_factors,
      diagonal=factor_diagonals, inverse_diagonal=inverse_diagonals)


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


@pytest.mark.parametrize('sparse', [False, True])
@pytest.mark.parametrize('device', ['cpu', pytest.param('mps', marks=[
    pytest.mark.gpu, pytest.mark.skipif(os.getenv('MUJOCO_METAL_RUN_GPU') != '1',
                                      reason='opt-in native factor query')])])
def test_owned_factorM_blocks_match_pinned_qLD_and_qLDiagInv(
    sparse, device, monkeypatch):
  torch = pytest.importorskip('torch')
  from mujoco_metal import mass_queries
  stage, _, expected = _stage(sparse, device)
  saved = {key: value.clone() for key, value in stage.items()
           if isinstance(value, torch.Tensor)}
  def forbid_expansion(*args, **kwargs):
    raise AssertionError('factor query must not expand global component mass')
  monkeypatch.setattr(mass_queries, 'full_mass', forbid_expansion)
  result = mass_queries.factor_mass(stage)
  assert result['nv'] == _fixture().nv
  assert result['status'].dtype == torch.int32
  assert result['status'].device == stage[next(iter(saved))].device
  np.testing.assert_array_equal(result['status'].cpu().numpy(), [0, 0, 0])
  assert len(result['blocks']) == (2 if sparse else 1)
  seen = []
  for block in result['blocks']:
    ids = block['dof_ids'].cpu().numpy()
    seen.extend(ids.tolist())
    expected_L = np.asarray(expected['lower'])[:, ids[:, None], ids[None, :]]
    np.testing.assert_allclose(block['L'].cpu().numpy(), expected_L,
                               rtol=8e-5, atol=8e-5)
    for name, oracle in (('D', 'diagonal'), ('Dinv', 'inverse_diagonal')):
      np.testing.assert_allclose(block[name].cpu().numpy(),
          np.asarray(expected[oracle])[:, ids], rtol=8e-5, atol=8e-5)
  assert sorted(seen) == list(range(result['nv']))
  # Owned query outputs cannot corrupt the input mass or a later factor call.
  result['blocks'][0]['L'].zero_()
  result['blocks'][0]['D'].zero_()
  for key, original in saved.items():
    torch.testing.assert_close(stage[key], original, rtol=0, atol=0)
  replay = mass_queries.factor_mass(stage)
  assert replay['blocks'][0]['L'].abs().sum() > 0


@pytest.mark.parametrize('device', ['cpu', pytest.param('mps', marks=[
    pytest.mark.gpu, pytest.mark.skipif(os.getenv('MUJOCO_METAL_RUN_GPU') != '1',
                                      reason='opt-in native factor failure isolation')])])
def test_factorM_component_failures_zero_all_blocks_and_keep_healthy_peer(device):
  torch = pytest.importorskip('torch')
  from mujoco_metal.mass_queries import factor_mass, mass_factor_query
  # A failure in the second component must zero the successful first one too.
  # The last world has finite positive mass but an overflowing inverse D.
  blocks = torch.tensor([[2., 3.], [2., -1.], [2., float('nan')], [2., 1e-40]],
                        device=device)
  layout = dict(ncomponent=2, component_dofnum=np.array([1, 1]),
      component_dof_offsets=np.array([0, 1, 2]),
      component_mass_offsets=np.array([0, 1, 2]),
      component_dof_ids=np.array([0, 1]), dof_component=np.array([0, 1]))
  result = factor_mass({'mass_blocks': blocks, 'mass_block_layout': layout})
  assert result['status'].tolist() == [0, 2, 2, 2]
  for block in result['blocks']:
    for name in ('L', 'D', 'Dinv'):
      assert torch.isfinite(block[name]).all()
      assert not block[name][1:].any()
      assert block[name][0].abs().sum() > 0
  zero = factor_mass({'mass_matrix': torch.zeros(2, 0, 0, device=device)})
  assert zero['nv'] == 0 and zero['status'].tolist() == [0, 0]
  assert zero['blocks'][0]['L'].shape == (2, 0, 0)
  for operation in ('solve', 'sqrt', 'half_solve'):
    output, status = mass_factor_query(
        {'mass_matrix': torch.zeros(2, 0, 0, device=device)},
        torch.zeros(2, 3, 0, device=device), operation)
    assert output.shape == (2, 3, 0) and status.tolist() == [0, 0]
    output, status = mass_factor_query(
        {'mass_matrix': torch.eye(2, device=device).expand(2, 2, 2)},
        torch.zeros(2, 0, 2, device=device), operation)
    assert output.shape == (2, 0, 2) and status.tolist() == [0, 0]


@pytest.mark.gpu
@pytest.mark.skipif(os.getenv('MUJOCO_METAL_RUN_GPU') != '1', reason='opt-in full native mass API')
@pytest.mark.parametrize('profile', ['integrated_euler_v1', 'integrated_scalable_v1'])
def test_public_native_mass_queries_match_pinned_armature_and_preserve_stage(profile):
  import torch
  from mujoco_metal import MetalSimulation
  from mujoco_metal import native_api as api

  model = _fixture()
  rng = np.random.default_rng(1919)
  qpos = np.tile(model.qpos0, (2, 1))
  qpos += rng.normal(0, .1, qpos.shape)
  for q in qpos:
    mujoco.mj_normalizeQuat(model, q)
  qvel = rng.normal(0, .2, (2, model.nv)).astype(np.float32)
  source = rng.normal(0, .3, (2, model.nv)).astype(np.float32)
  expected = {key: [] for key in (
      'full', 'product', 'sqrt', 'solve', 'half', 'L', 'D', 'Dinv')}
  for world in range(2):
    data = mujoco.MjData(model)
    data.qpos[:], data.qvel[:] = qpos[world], qvel[world]
    mujoco.mj_forward(model, data)
    matrix = np.empty((model.nv, model.nv))
    mujoco.mj_fullM(model, data, matrix)
    expected['full'].append(matrix)
    for key, value in zip(('L', 'D', 'Dinv'), _pinned_factors(model, data)):
      expected[key].append(value)
    product, root = np.empty(model.nv), np.empty(model.nv)
    rhs = source[world:world+1].astype(np.float64)
    solve, half = np.empty_like(rhs), np.empty_like(rhs)
    mujoco.mj_mulM(model, data, product, rhs[0])
    mujoco.mj_mulM2(model, data, root, rhs[0])
    mujoco.mj_solveM(model, data, solve, rhs)
    mujoco.mj_solveM2(model, data, half, rhs, np.sqrt(data.qLDiagInv))
    for key, value in (('product', product), ('sqrt', root), ('solve', solve[0]), ('half', half[0])):
      expected[key].append(value)
  sim = MetalSimulation(model, 2, qpos=qpos.astype(np.float32), qvel=qvel, profile=profile)
  record = api.mj_fwdPosition(sim, return_record=True)
  stage = api.mj_fwdVelocity(sim, record=record)
  saved = {key: (value.data_ptr(), value.detach().cpu().numpy().copy())
           for key, value in stage.items() if isinstance(value, torch.Tensor)}
  vector = torch.as_tensor(source, device='mps')
  for supplied in (stage, None):
    np.testing.assert_allclose(api.mj_fullM(sim, dynamics=supplied).cpu().numpy(), expected['full'], atol=4e-5, rtol=4e-4)
    np.testing.assert_allclose(api.mj_mulM(sim, vector, dynamics=supplied).cpu().numpy(), expected['product'], atol=4e-5, rtol=4e-4)
    factor = api.mj_factorM(sim, dynamics=supplied)
    assert not factor['status'].cpu().numpy().any()
    for block in factor['blocks']:
      ids = block['dof_ids'].cpu().numpy()
      reference_L = np.asarray(expected['L'])[:, ids[:, None], ids[None, :]]
      np.testing.assert_allclose(block['L'].cpu().numpy(), reference_L,
                                 atol=8e-5, rtol=8e-4)
      for name in ('D', 'Dinv'):
        np.testing.assert_allclose(block[name].cpu().numpy(),
            np.asarray(expected[name])[:, ids], atol=8e-5, rtol=8e-4)
    for function, key in ((api.mj_mulM2, 'sqrt'), (api.mj_solveM, 'solve'), (api.mj_solveM2, 'half')):
      result, status = function(sim, vector, dynamics=supplied)
      assert not status.cpu().numpy().any()
      np.testing.assert_allclose(result.cpu().numpy(), expected[key], atol=8e-5, rtol=8e-4)
    for key, (pointer, original) in saved.items():
      assert stage[key].data_ptr() == pointer
      np.testing.assert_array_equal(stage[key].cpu().numpy(), original)
    sim.validate_forward_stage_record(record, 'VEL')
