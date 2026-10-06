# Copyright 2026 The MuJoCo Metal contributors
# Licensed under the Apache License, Version 2.0.
"""Public sparse API orchestration against pinned CPU stage data.

These tests qualify layout selection, ownership and input contracts. The
separate opt-in native tests qualify the actual Metal producers and consumers.
"""
import copy
import os
from types import SimpleNamespace

import mujoco
import numpy as np
import pytest


def _fixture():
  return mujoco.MjModel.from_xml_string('''<mujoco>
    <option gravity="0 0 0"><flag contact="disable"/></option>
    <worldbody><body><joint name="a" axis="0 0 1"/>
      <geom size=".1" mass="1" pos=".2 0 0"/>
      <body pos=".4 .1 .2"><joint name="b" axis="0 1 0"/>
        <geom size=".08" mass=".7" pos=".1 .2 .15"/>
      </body></body></worldbody>
    <tendon><fixed armature=".3" limited="false">
      <joint joint="a" coef="1"/><joint joint="b" coef="-.6"/>
    </fixed></tendon></mujoco>''')


def _cpu_sim():
  torch = pytest.importorskip('torch')
  from mujoco_metal.smooth_metal import compile_tree_mass_layout
  model = _fixture()
  data = mujoco.MjData(model)
  data.qpos[:] = [.35, -.4]
  data.qvel[:] = [.7, -.8]
  data.qacc[:] = [.4, -.7]
  mujoco.mj_inverse(model, data)
  tensor = lambda value: torch.tensor(np.asarray(value).copy()[None], dtype=torch.float32)
  mass = np.empty((model.nv, model.nv))
  mujoco.mj_fullM(model, data, mass)
  layout = compile_tree_mass_layout(model.body_treeid, model.dof_bodyid)
  def pack(values):
    output = np.empty(layout['nnz'])
    for component, n in enumerate(layout['component_dofnum']):
      start = int(layout['component_dof_offsets'][component])
      ids = layout['component_dof_ids'][start:start+int(n)]
      offset = int(layout['component_mass_offsets'][component])
      output[offset:offset+int(n)**2] = values[np.ix_(ids, ids)].reshape(-1)
    return tensor(output)
  tendon_J = np.array([[1., -.6]])
  armature = .3 * tendon_J.T @ tendon_J
  from tests.pinned_pose_abi import pinned_pose_abi
  poses = pinned_pose_abi(data)
  stage = {'poses': poses, 'mass_blocks': pack(mass-armature),
      'tendon_armature_blocks': pack(armature), 'mass_block_layout': layout,
      **{name: tensor(value) for name, value in (
          ('qfrc_bias', data.qfrc_bias), ('cdof', data.cdof),
          ('cdof_dot', data.cdof_dot), ('cvel', data.cvel), ('root_com', data.subtree_com))}}
  seen = []
  def smooth(qpos, qvel, **kwargs):
    assert kwargs['poses'] is poses
    torch.testing.assert_close(kwargs['tendon_J'], tensor(tendon_J))
    assert kwargs['awake_lists'] is lists
    seen.append('smooth')
    return stage
  def fk(qpos, **kwargs):
    seen.append('fk')
    return poses
  def project(blocks, vector, tendon_armature_blocks):
    # Independent validation math only, not a fallback in the native code.
    torch.testing.assert_close(blocks, stage['mass_blocks'])
    torch.testing.assert_close(tendon_armature_blocks, stage['tendon_armature_blocks'])
    return vector @ tensor(mass)[0].T
  lists = {'dof_ids': torch.arange(model.nv, dtype=torch.int32)[None],
           'counts': torch.tensor([[0, model.nv, 1]], dtype=torch.int32)}
  state = SimpleNamespace(_qpos=tensor(data.qpos), _qvel=tensor(data.qvel),
      _qacc=tensor(data.qacc), _device=torch.device('cpu'), _nmocap=0,
      _mpos=torch.zeros((1, 0, 3)), _mquat=torch.zeros((1, 0, 4)))
  sim = SimpleNamespace(model=model, _mjmodel=model, state=state, batch_size=1,
      _component_mass_enabled=True, _passive=None, _spatial_tendons=None,
      _component_tendon_jacobian=lambda velocity, position: tensor(tendon_J),
      _spatial_cache_key=('old',),
      _tendons=SimpleNamespace(run_device=lambda *_: (torch.zeros((1, model.nv)), None, None)),
      _smooth=SimpleNamespace(run_device=smooth, _fk=SimpleNamespace(run_device=fk),
          mass_blocks_matvec_device=project, mass_block_layout=layout, _has_tendon_armature=True,
          _workspace={'all_awake_lists': lists}))
  return sim, data, stage, seen


def test_stateless_inverse_uses_sparse_mass_and_tendon_armature_without_dense_stage():
  from mujoco_metal.native_api import _inverse_impl, _inverse_query_workspaces
  sim, data, stage, seen = _cpu_sim()
  assert 'mass_matrix' not in stage
  with _inverse_query_workspaces(sim):
    result = _inverse_impl(sim, None, None, None, None, None)
  np.testing.assert_allclose(result.numpy()[0], data.qfrc_inverse, atol=2e-6, rtol=2e-5)
  assert seen == ['fk', 'smooth']
  assert sim._spatial_cache_key == ('old',)


def test_sparse_query_accepts_matching_stage_and_rejects_wrong_layout():
  from mujoco_metal.native_api import _query_smooth, _stage_dynamics
  sim, _, stage, seen = _cpu_sim()
  poses = stage['poses']
  dynamics = _query_smooth(sim, sim.state._qpos, sim.state._qvel, poses=poses)
  assert seen == ['smooth']  # Supplied position record avoids FK.
  assert _stage_dynamics(sim, dynamics) is dynamics
  wrong = dict(stage, mass_block_layout=copy.deepcopy(stage['mass_block_layout']))
  wrong['mass_block_layout']['component_dof_ids'][0] = 999
  with pytest.raises(ValueError, match='does not match'):
    _stage_dynamics(sim, wrong)
  with pytest.raises(ValueError, match='missing tendon_armature'):
    _stage_dynamics(sim, {name: value for name, value in stage.items() if name != 'tendon_armature_blocks'})


@pytest.mark.parametrize('field', ['mass_blocks', 'tendon_armature_blocks', 'cdof', 'root_com'])
def test_sparse_stage_rejects_nonfinite_consumed_inputs(field):
  from mujoco_metal.native_api import _stage_dynamics
  sim, _, stage, _ = _cpu_sim()
  stage[field] = stage[field].clone()
  stage[field].reshape(-1)[0] = float('nan')
  with pytest.raises(ValueError, match='nonfinite|invalid native'):
    _stage_dynamics(sim, stage)


def test_spatial_query_consumes_supplied_sparse_motion_with_checked_contract():
  from mujoco_metal.native_api import mj_jacBody, mj_objectVelocity
  sim, data, stage, seen = _cpu_sim()
  jp, jr = mj_jacBody(sim, 2, dynamics=stage)
  expected_p, expected_r = np.empty((3, sim.model.nv)), np.empty((3, sim.model.nv))
  mujoco.mj_jacBody(sim.model, data, expected_p, expected_r, 2)
  np.testing.assert_allclose(jp.numpy()[0], expected_p, atol=2e-6, rtol=2e-5)
  np.testing.assert_allclose(jr.numpy()[0], expected_r, atol=2e-6, rtol=2e-5)
  assert seen == []
  with pytest.raises(TypeError, match='objtype'):
    mj_objectVelocity(sim, float(mujoco.mjtObj.mjOBJ_BODY), 2, dynamics=stage)
  malformed = dict(stage, cvel=stage['cvel'][:, :-1])
  with pytest.raises(ValueError, match='cvel'):
    mj_objectVelocity(sim, mujoco.mjtObj.mjOBJ_BODY, 2, dynamics=malformed)


@pytest.mark.gpu
@pytest.mark.skipif(os.getenv('MUJOCO_METAL_RUN_GPU') != '1', reason='opt-in native gate')
def test_native_sparse_inverse_queries_and_split_velocity_match_pinned_tendon_mass():
  from mujoco_metal import MetalSimulation
  from mujoco_metal.native_api import mj_inverse, mj_fwdPosition, mj_fwdVelocity, mj_jacBody
  model = _fixture()
  data = mujoco.MjData(model)
  data.qpos[:] = [.35, -.4]
  data.qvel[:] = [.7, -.8]
  data.qacc[:] = [.4, -.7]
  mujoco.mj_inverse(model, data)
  sim = MetalSimulation(model, profile='integrated_scalable_v1',
      qpos=data.qpos[None].astype(np.float32), qvel=data.qvel[None].astype(np.float32))
  result = mj_inverse(sim, qacc=data.qacc[None].astype(np.float32))
  np.testing.assert_allclose(result.detach().cpu().numpy()[0], data.qfrc_inverse, atol=3e-5, rtol=3e-4)
  poses = mj_fwdPosition(sim)
  dynamics = mj_fwdVelocity(sim, poses=poses)
  assert 'mass_matrix' not in dynamics and 'tendon_armature_blocks' in dynamics
  jp, jr = mj_jacBody(sim, 2, dynamics=dynamics)
  expected_p, expected_r = np.empty((3, model.nv)), np.empty((3, model.nv))
  mujoco.mj_jacBody(model, data, expected_p, expected_r, 2)
  np.testing.assert_allclose(jp.detach().cpu().numpy()[0], expected_p, atol=3e-5, rtol=3e-4)
  np.testing.assert_allclose(jr.detach().cpu().numpy()[0], expected_r, atol=3e-5, rtol=3e-4)


def test_public_sparse_mass_queries_match_pinned_engine_and_preserve_inputs():
  torch = pytest.importorskip('torch')
  from mujoco_metal.native_api import mj_fullM, mj_mulM, mj_mulM2, mj_solveM, mj_solveM2
  sim, data, stage, seen = _cpu_sim()
  vector = torch.tensor([[.8, -.3]], dtype=torch.float32)
  before = {key: value.clone() for key, value in stage.items() if isinstance(value, torch.Tensor)}
  expected_mass = np.empty((sim.model.nv, sim.model.nv))
  mujoco.mj_fullM(sim.model, data, expected_mass)
  np.testing.assert_allclose(mj_fullM(sim, dynamics=stage).numpy()[0], expected_mass, atol=2e-7)
  np.testing.assert_allclose(mj_mulM(sim, vector, dynamics=stage).numpy()[0], expected_mass @ vector.numpy()[0], atol=2e-7)
  for function, oracle in ((mj_mulM2, 'root'), (mj_solveM, 'solve'), (mj_solveM2, 'half')):
    expected = np.empty((1, sim.model.nv))
    source = vector.numpy().astype(np.float64)
    if oracle == 'root':
      mujoco.mj_mulM2(sim.model, data, expected[0], source[0])
    elif oracle == 'solve':
      mujoco.mj_solveM(sim.model, data, expected, source)
    else:
      mujoco.mj_solveM2(sim.model, data, expected, source, np.sqrt(data.qLDiagInv))
    value, status = function(sim, vector, dynamics=stage)
    np.testing.assert_allclose(value.numpy(), expected, atol=2e-6, rtol=2e-5)
    assert not status.any()
  assert seen == []
  for key, value in before.items():
    torch.testing.assert_close(stage[key], value, rtol=0, atol=0)
  with pytest.raises(ValueError, match='finite'):
    mj_solveM(sim, torch.tensor([[float('nan'), 0.]]), dynamics=stage)
