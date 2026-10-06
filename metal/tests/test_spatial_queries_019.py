# Copyright 2026 The MuJoCo Metal contributors
# Licensed under the Apache License, Version 2.0.
"""Independent pinned spatial query math; native public gates are separate."""
import os

import mujoco
import numpy as np
import pytest


@pytest.mark.gpu
@pytest.mark.skipif(os.getenv('MUJOCO_METAL_RUN_GPU') != '1', reason='native opt-in')
def test_native_query_accepts_default_mps_device_with_indexed_stage_tensors():
  """Exercise actual batched query math without Simulation construction."""
  import torch
  from types import SimpleNamespace
  from mujoco_metal.native_api import _validate_spatial_stage
  from mujoco_metal.spatial_queries import DeviceSpatialQueries
  model = _fixture()
  worlds = [_stages(model, seed, 'mps') for seed in (17, 31)]
  dynamics = {name: torch.cat([world[1][name] for world in worlds])
              for name in ('root_com', 'cvel', 'cdof', 'cdof_dot')}
  dynamics['poses'] = {
      name: torch.cat([world[1]['poses'][name] for world in worlds])
      for name in worlds[0][1]['poses']}
  sim = SimpleNamespace(_mjmodel=model, batch_size=2,
                        state=SimpleNamespace(_device=torch.device('mps')))
  _validate_spatial_stage(sim, dynamics)
  program = DeviceSpatialQueries(model, 2, 'mps')
  body = model.nbody - 2
  point = (dynamics['poses']['body_pos'][:, body]
           + dynamics['cdof'].new_tensor([[.1, -.2, .05]]))
  for method, oracle in ((program.jac, mujoco.mj_jac),
                         (program.jac_dot, mujoco.mj_jacDot)):
    jp, jr = method(dynamics, point, body)
    expected_p, expected_r = [], []
    host_points = point.cpu().numpy()
    for world, (data, *_unused) in enumerate(worlds):
      p, r = np.empty((3, model.nv)), np.empty((3, model.nv))
      oracle(model, data, p, r, host_points[world].astype(np.float64), body)
      expected_p.append(p); expected_r.append(r)
    np.testing.assert_allclose(jp.cpu().numpy(), expected_p, atol=3e-6, rtol=3e-5)
    np.testing.assert_allclose(jr.cpu().numpy(), expected_r, atol=3e-6, rtol=3e-5)


def _fixture(disable_gravity=False):
  flag = '<flag contact="disable"' + (' gravity="disable"' if disable_gravity else '') + '/>'
  cameras = ''.join(f'<camera name="c{i}" mode="{mode}" pos=".4 -.7 .5"' +
      (' target="tip"' if mode.startswith('target') else '') + '/>'
      for i, mode in enumerate(('fixed', 'track', 'trackcom', 'targetbody', 'targetbodycom')))
  return mujoco.MjModel.from_xml_string(f'''<mujoco>
    <option gravity=".1 -.3 -9.81">{flag}</option>
    <worldbody><body name="root" pos=".1 .2 .6"><freejoint/>
      <inertial pos=".1 -.05 .03" quat=".9238795325 .3826834324 0 0" mass="2" diaginertia=".1 .2 .3"/>
      <geom type="box" size=".1 .12 .08" pos=".07 .02 .06"/>{cameras}
      <site pos=".2 -.1 .3" quat=".9659258263 0 .2588190451 0"/>
      <body pos=".3 .1 .2"><joint type="ball"/>
        <geom type="capsule" size=".05" fromto="0 0 0 .4 .1 .1"/>
        <body pos=".4 .1 .1"><joint type="hinge" axis=".2 .3 1"/>
          <geom size=".08" pos=".2 0 .1"/>
          <body name="tip" pos=".1 .2 .3"><joint type="slide" axis="1 .2 .1"/>
            <geom size=".07"/><site pos=".2 -.1 .1"/>
          </body>
          <body pos=".2 .3 -.1"><geom size=".04"/></body>
        </body>
      </body>
    </body><body pos="-1 0 0"><geom size=".1"/><site/></body></worldbody>
  </mujoco>''')


def _stages(model, seed, device):
  import torch
  rng = np.random.default_rng(seed)
  data = mujoco.MjData(model)
  mujoco.mj_integratePos(model, data.qpos, rng.normal(size=model.nv), .15)
  data.qvel[:] = rng.normal(size=model.nv)*.4
  mujoco.mj_forward(model, data)
  data.qacc[:] = rng.normal(size=model.nv)*.7
  mujoco.mj_rnePostConstraint(model, data)
  def tensor(values):
    return torch.tensor(np.asarray(values).copy()[None], dtype=torch.float32, device=device)
  from tests.pinned_pose_abi import pinned_pose_abi
  poses = pinned_pose_abi(data, device)
  dynamics = {name: tensor(value) for name, value in (
      ('cdof', data.cdof), ('cdof_dot', data.cdof_dot), ('cvel', data.cvel),
      ('root_com', data.subtree_com))}
  dynamics['poses'] = poses
  return data, dynamics, tensor(data.qvel), tensor(data.qacc)


@pytest.mark.parametrize('seed', [17, 31])
@pytest.mark.parametrize('disable_gravity', [False, True])
def test_spatial_queries_match_actual_pinned_body_joint_and_camera_frames(seed, disable_gravity):
  pytest.importorskip('torch')
  from mujoco_metal.spatial_queries import DeviceSpatialQueries
  model = _fixture(disable_gravity)
  data, dynamics, qvel, qacc = _stages(model, seed, 'cpu')
  program = DeviceSpatialQueries(model, 1, 'cpu')
  for body in range(model.nbody):
    point = dynamics['poses']['body_pos'][:, body] + qvel.new_tensor([[.1, -.2, .05]])
    for method, oracle in ((program.jac, mujoco.mj_jac), (program.jac_dot, mujoco.mj_jacDot)):
      jp, jr = method(dynamics, point, body)
      expected_p, expected_r = np.empty((3, model.nv)), np.empty((3, model.nv))
      oracle(model, data, expected_p, expected_r, point.numpy()[0].astype(np.float64), body)
      np.testing.assert_allclose(jp.numpy()[0], expected_p, atol=3e-6, rtol=3e-5)
      np.testing.assert_allclose(jr.numpy()[0], expected_r, atol=3e-6, rtol=3e-5)
    jp, jr = program.jac(dynamics, point, body)
    force, torque = qvel.new_tensor([[1., -.7, .4]]), qvel.new_tensor([[.2, .3, -.1]])
    expected = np.zeros(model.nv)
    mujoco.mj_applyFT(model, data, force.numpy()[0].astype(np.float64),
        torque.numpy()[0].astype(np.float64), point.numpy()[0].astype(np.float64), body, expected)
    np.testing.assert_allclose(program.apply_ft(dynamics, force, torque, point, body).numpy()[0], expected,
                               atol=3e-6, rtol=3e-5)
    if model.body_subtreemass[body] > 0:
      expected_com = np.empty((3, model.nv))
      mujoco.mj_jacSubtreeCom(model, data, expected_com, body)
      np.testing.assert_allclose(program.jac_subtree_com(dynamics, body).numpy()[0], expected_com,
                                 atol=3e-6, rtol=3e-5)
  np.testing.assert_allclose(program.subtree_com(dynamics).numpy()[0], data.subtree_com,
                             atol=3e-6, rtol=3e-5)
  for kind, count in ((mujoco.mjtObj.mjOBJ_BODY, model.nbody),
      (mujoco.mjtObj.mjOBJ_XBODY, model.nbody), (mujoco.mjtObj.mjOBJ_GEOM, model.ngeom),
      (mujoco.mjtObj.mjOBJ_SITE, model.nsite), (mujoco.mjtObj.mjOBJ_CAMERA, model.ncam)):
    for objid in range(count):
      for local in (False, True):
        expected_v, expected_a = np.zeros(6), np.zeros(6)
        mujoco.mj_objectVelocity(model, data, kind, objid, expected_v, int(local))
        mujoco.mj_objectAcceleration(model, data, kind, objid, expected_a, int(local))
        np.testing.assert_allclose(program.object_velocity(dynamics, kind, objid, local).numpy()[0],
                                   expected_v, atol=4e-6, rtol=4e-5)
        np.testing.assert_allclose(program.object_acceleration(dynamics, qvel, qacc, kind, objid, local).numpy()[0],
                                   expected_a, atol=6e-6, rtol=5e-5)


def test_public_spatial_query_wrappers_use_device_stages_without_mutating_inputs():
  torch = pytest.importorskip('torch')
  from types import SimpleNamespace
  from mujoco_metal import native_api
  model = _fixture()
  data, dynamics, qvel, qacc = _stages(model, 17, 'cpu')
  state = SimpleNamespace(_device=torch.device('cpu'), _qpos=torch.tensor(data.qpos[None], dtype=torch.float32),
      _qvel=qvel, _qacc=qacc)
  sim = SimpleNamespace(model=model, _mjmodel=model, batch_size=1, state=state,
      _smooth=SimpleNamespace(run_device=lambda *_args: dynamics))
  for method, oracle, objid in ((native_api.mj_jacBody, mujoco.mj_jacBody, 1),
      (native_api.mj_jacBodyCom, mujoco.mj_jacBodyCom, 1),
      (native_api.mj_jacGeom, mujoco.mj_jacGeom, 0),
      (native_api.mj_jacSite, mujoco.mj_jacSite, 0)):
    jp, jr = method(sim, objid)
    expected_p, expected_r = np.zeros((3, model.nv)), np.zeros((3, model.nv))
    oracle(model, data, expected_p, expected_r, objid)
    np.testing.assert_allclose(jp.numpy()[0], expected_p, atol=3e-6, rtol=3e-5)
    np.testing.assert_allclose(jr.numpy()[0], expected_r, atol=3e-6, rtol=3e-5)
  point = np.array([[.2, .3, .4]], np.float32)
  for method, oracle in ((native_api.mj_jac, mujoco.mj_jac), (native_api.mj_jacDot, mujoco.mj_jacDot)):
    jp, jr = method(sim, point, 3)
    expected_p, expected_r = np.zeros((3, model.nv)), np.zeros((3, model.nv))
    oracle(model, data, expected_p, expected_r, point[0].astype(np.float64), 3)
    np.testing.assert_allclose(jp.numpy()[0], expected_p, atol=3e-6, rtol=3e-5)
    np.testing.assert_allclose(jr.numpy()[0], expected_r, atol=3e-6, rtol=3e-5)
  force = np.array([[1., -.3, .2]], np.float32)
  expected_force = np.zeros(model.nv)
  mujoco.mj_applyFT(model, data, force[0].astype(np.float64), np.zeros(3), point[0].astype(np.float64), 3, expected_force)
  applied = native_api.mj_applyFT(sim, force, None, point, 3)
  np.testing.assert_allclose(applied.numpy()[0], expected_force, atol=3e-6, rtol=3e-5)
  for method, oracle in ((native_api.mj_objectVelocity, mujoco.mj_objectVelocity),
      (native_api.mj_objectAcceleration, mujoco.mj_objectAcceleration)):
    expected = np.zeros(6)
    oracle(model, data, mujoco.mjtObj.mjOBJ_CAMERA, 4, expected, 1)
    actual = method(sim, mujoco.mjtObj.mjOBJ_CAMERA, 4, 1)
    np.testing.assert_allclose(actual.numpy()[0], expected, atol=6e-6, rtol=5e-5)
    with pytest.raises(ValueError, match='flg_local'):
      method(sim, mujoco.mjtObj.mjOBJ_SITE, 0, 2)
  np.testing.assert_array_equal(state._qvel.numpy(), qvel.numpy())
  np.testing.assert_array_equal(state._qacc.numpy(), qacc.numpy())


def test_spatial_queries_keep_worlds_distinct_and_handle_zero_dofs():
  torch = pytest.importorskip('torch')
  from mujoco_metal.spatial_queries import DeviceSpatialQueries
  model = _fixture()
  data0, stage0, v0, a0 = _stages(model, 17, 'cpu')
  data1, stage1, v1, a1 = _stages(model, 31, 'cpu')
  def concatenate(left, right):
    return ({name: concatenate(value, right[name]) for name, value in left.items()}
            if isinstance(left, dict) else torch.cat((left, right)))
  stage = concatenate(stage0, stage1)
  program = DeviceSpatialQueries(model, 2, 'cpu')
  actual = program.object_acceleration(stage, torch.cat((v0, v1)), torch.cat((a0, a1)),
      mujoco.mjtObj.mjOBJ_CAMERA, 4, True).numpy()
  expected = np.zeros((2, 6))
  for i, data in enumerate((data0, data1)):
    mujoco.mj_objectAcceleration(model, data, mujoco.mjtObj.mjOBJ_CAMERA, 4, expected[i], 1)
  np.testing.assert_allclose(actual, expected, atol=6e-6, rtol=5e-5)
  assert np.linalg.norm(actual[0]-actual[1]) > .1
  static_model = mujoco.MjModel.from_xml_string('<mujoco><worldbody><geom size=".1"/><site/></worldbody></mujoco>')
  _data, static, velocity, acceleration = _stages(static_model, 1, 'cpu')
  zero = DeviceSpatialQueries(static_model, 1, 'cpu')
  jp, jr = zero.jac(static, static['poses']['body_pos'][:, 0], 0)
  assert jp.shape == jr.shape == (1, 3, 0)
  np.testing.assert_array_equal(zero.object_acceleration(static, velocity, acceleration,
      mujoco.mjtObj.mjOBJ_SITE, 0).numpy(), 0)


@pytest.mark.parametrize('field', ['cvel', 'cdof', 'cdof_dot', 'root_com', 'poses'])
def test_public_spatial_queries_validate_supplied_stage_before_computation(field):
  torch = pytest.importorskip('torch')
  from types import SimpleNamespace
  from mujoco_metal import native_api
  model = _fixture()
  data, dynamics, qvel, qacc = _stages(model, 17, 'cpu')
  state = SimpleNamespace(_device=torch.device('cpu'), _qpos=torch.tensor(data.qpos[None], dtype=torch.float32),
      _qvel=qvel, _qacc=qacc)
  sim = SimpleNamespace(model=model, _mjmodel=model, batch_size=1, state=state)
  expected = np.empty(6)
  mujoco.mj_objectVelocity(model, data, mujoco.mjtObj.mjOBJ_SITE, 0, expected, 0)
  actual = native_api.mj_objectVelocity(sim, mujoco.mjtObj.mjOBJ_SITE, 0, dynamics=dynamics)
  np.testing.assert_allclose(actual.numpy()[0], expected, atol=4e-6, rtol=4e-5)
  with pytest.raises(TypeError, match='objtype'):
    native_api.mj_objectVelocity(sim, float(mujoco.mjtObj.mjOBJ_SITE), 0, dynamics=dynamics)
  malformed = dict(dynamics)
  if field == 'poses':
    malformed[field] = dict(dynamics[field])
    malformed[field]['joint_axis'] = dynamics[field]['joint_axis'].clone()
    malformed[field]['joint_axis'].reshape(-1)[0] = float('nan')
  else:
    malformed[field] = dynamics[field].clone()
    malformed[field].reshape(-1)[0] = float('nan')
  with pytest.raises(ValueError, match='nonfinite|invalid native'):
    native_api.mj_objectVelocity(sim, mujoco.mjtObj.mjOBJ_SITE, 0, dynamics=malformed)


@pytest.mark.gpu
@pytest.mark.skipif(os.getenv('MUJOCO_METAL_RUN_GPU') != '1', reason='native opt-in')
def test_spatial_queries_execute_on_mps_with_native_smooth_stages():
  import torch
  from mujoco_metal import MetalSimulation
  from mujoco_metal.spatial_queries import DeviceSpatialQueries
  from mujoco_metal import native_api
  model = _fixture()
  data, _cpu_stage, _qv, _qa = _stages(model, 17, 'cpu')
  sim = MetalSimulation(model, qpos=data.qpos[None].astype(np.float32),
      qvel=data.qvel[None].astype(np.float32), profile='integrated_euler_v1')
  dynamics = sim._smooth.run_device(sim.state._qpos, sim.state._qvel)
  program = DeviceSpatialQueries(model, 1, 'mps')
  qacc = torch.tensor(data.qacc[None], dtype=torch.float32, device='mps')
  for kind, objid in ((mujoco.mjtObj.mjOBJ_SITE, 0), (mujoco.mjtObj.mjOBJ_CAMERA, 4)):
    for local in (False, True):
      expected = np.zeros(6)
      mujoco.mj_objectAcceleration(model, data, kind, objid, expected, int(local))
      actual = program.object_acceleration(dynamics, sim.state._qvel, qacc, kind, objid, local)
      assert actual.device.type == 'mps'
      np.testing.assert_allclose(actual.cpu().numpy()[0], expected, atol=2e-4, rtol=2e-4)
      public = native_api.mj_objectAcceleration(sim, kind, objid, int(local), qacc=qacc)
      assert public.device.type == 'mps'
      np.testing.assert_allclose(public.cpu().numpy()[0], expected, atol=2e-4, rtol=2e-4)


@pytest.mark.parametrize('seed', [17, 31])
def test_subtree_angular_momentum_matrix_matches_pinned_and_independent_momentum(seed):
  torch = pytest.importorskip('torch')
  from mujoco_metal.spatial_queries import DeviceSpatialQueries
  model = _fixture()
  data, dynamics, qvel, _ = _stages(model, seed, 'cpu')
  program = DeviceSpatialQueries(model, 1, 'cpu')
  for body in range(model.nbody):
    expected = np.empty((3, model.nv))
    mujoco.mj_angmomMat(model, data, expected, body)
    actual = program.angmom_matrix(dynamics, body)
    np.testing.assert_allclose(actual.numpy()[0], expected, atol=3e-6, rtol=3e-5)
    # Independent physical momentum sum uses CPU object COM velocities, not
    # either implementation's angular-momentum matrix or Jacobian routine.
    momentum = np.zeros(3)
    for child in range(body, model.nbody):
      if child > body and model.body_parentid[child] < body:
        break
      velocity = np.empty(6)
      mujoco.mj_objectVelocity(model, data, mujoco.mjtObj.mjOBJ_BODY,
                               child, velocity, 0)
      rotation = data.ximat[child].reshape(3, 3)
      inertia = rotation @ np.diag(model.body_inertia[child]) @ rotation.T
      momentum += inertia @ velocity[:3]
      momentum += np.cross(data.xipos[child]-data.subtree_com[body],
                           model.body_mass[child]*velocity[3:])
    np.testing.assert_allclose((actual @ qvel.unsqueeze(-1)).numpy()[0, :, 0],
                               momentum, atol=3e-6, rtol=3e-5)
    saved = actual.clone()
    program.angmom_matrix(dynamics, 0).zero_()
    assert torch.equal(actual, saved)
  for bad in (-1, model.nbody):
    with pytest.raises(ValueError):
      program.angmom_matrix(dynamics, bad)
  for bad in (True, 1.0):
    with pytest.raises(TypeError):
      program.angmom_matrix(dynamics, bad)


def test_fixed_world_angular_momentum_has_zero_width_and_owned_result():
  torch = pytest.importorskip('torch')
  from mujoco_metal.spatial_queries import DeviceSpatialQueries
  model = mujoco.MjModel.from_xml_string(
      '<mujoco><worldbody><body><geom size=".1"/></body></worldbody></mujoco>')
  data, dynamics, _, _ = _stages(model, 17, 'cpu')
  program = DeviceSpatialQueries(model, 1, 'cpu')
  for body in range(model.nbody):
    expected = np.empty((3, 0))
    mujoco.mj_angmomMat(model, data, expected, body)
    actual = program.angmom_matrix(dynamics, body)
    assert actual.shape == (1, 3, 0)
    assert actual.dtype == torch.float32


@pytest.mark.parametrize('seed', [17, 31])
def test_full_state_subtree_motion_matches_pinned_com_momentum_and_velocity(seed):
  torch = pytest.importorskip('torch')
  from mujoco_metal.spatial_queries import DeviceSpatialQueries
  model = _fixture()
  data, dynamics, qvel, _ = _stages(model, seed, 'cpu')
  mujoco.mj_subtreeVel(model, data)
  program = DeviceSpatialQueries(model, 1, 'cpu')
  result = program.subtree_velocity(dynamics)
  for name in ('subtree_linvel', 'subtree_angmom'):
    np.testing.assert_allclose(result[name].numpy()[0], getattr(data, name),
                               atol=3e-6, rtol=3e-5)
  for body in range(model.nbody):
    momentum = (program.angmom_matrix(dynamics, body) @ qvel.unsqueeze(-1)).squeeze(-1)
    torch.testing.assert_close(result['subtree_angmom'][:, body], momentum,
                               atol=3e-6, rtol=3e-5)
  saved = {name: tensor.clone() for name, tensor in result.items()}
  next_result = program.subtree_velocity(dynamics)
  for name in result:
    next_result[name].zero_()
    assert torch.equal(result[name], saved[name])


@pytest.mark.gpu
@pytest.mark.skipif(os.getenv('MUJOCO_METAL_RUN_GPU') != '1', reason='native opt-in')
@pytest.mark.parametrize('profile', ['integrated_euler_v1', 'integrated_scalable_v1'])
def test_native_public_angular_momentum_dense_and_component_with_state_changes(profile, monkeypatch):
  import torch
  from mujoco_metal.simulation import MetalSimulation
  from mujoco_metal import native_api as api
  model = _fixture()
  worlds = [_stages(model, seed, 'cpu')[0] for seed in (17, 31)]
  qp = np.stack([data.qpos.copy() for data in worlds]).astype(np.float32)
  qv = np.stack([data.qvel.copy() for data in worlds]).astype(np.float32)
  # Oracle starts from the exact float32 device inputs, not their original
  # binary64 randomizations. All runtime CPU physics is then forbidden.
  for world, data in enumerate(worlds):
    data.qpos[:] = qp[world]
    data.qvel[:] = qv[world]
    mujoco.mj_forward(model, data)
  expected = []
  for data in worlds:
    mujoco.mj_subtreeVel(model, data)
  expected_motion = {name: np.stack([getattr(data, name).copy() for data in worlds])
                     for name in ('subtree_linvel', 'subtree_angmom')}
  for body in range(model.nbody):
    rows = []
    for data in worlds:
      matrix = np.empty((3, model.nv))
      mujoco.mj_angmomMat(model, data, matrix, body)
      rows.append(matrix)
    expected.append(np.stack(rows))
  sim = MetalSimulation(model, batch_size=2, qpos=qp, qvel=qv, profile=profile)
  def forbidden(*args, **kwargs):
    raise AssertionError('CPU physics/query fallback invoked')
  for name in ('mj_forward', 'mj_step', 'mj_kinematics', 'mj_comPos',
               'mj_comVel', 'mj_jac', 'mj_angmomMat', 'mj_subtreeVel'):
    monkeypatch.setattr(mujoco, name, forbidden)
  before_qpos = sim.state._qpos.clone()
  before_qvel = sim.state._qvel.clone()
  retained = []
  for body in range(model.nbody):
    actual = api.mj_angmomMat(sim, body)
    assert actual.device.type == 'mps'
    np.testing.assert_allclose(actual.cpu().numpy(), expected[body],
                               atol=8e-6, rtol=5e-5)
    retained.append((actual, actual.clone()))
  assert torch.equal(sim.state._qpos, before_qpos)
  assert torch.equal(sim.state._qvel, before_qvel)
  motion = api.mj_subtreeVel(sim)
  retained_motion = {name: tensor.clone() for name, tensor in motion.items()}
  for name, tensor in motion.items():
    assert tensor.device.type == 'mps'
    np.testing.assert_allclose(tensor.cpu().numpy(), expected_motion[name],
                               atol=8e-6, rtol=5e-5)
  sim.state._qpos[:, 0].add_(.13)
  shifted = api.mj_angmomMat(sim, 1)
  # Translating this free root translates its whole subtree. World subtree
  # 0 also contains a separate fixed body, so its COM/matrix can change.
  np.testing.assert_allclose(shifted.cpu().numpy(), expected[1],
                             atol=8e-6, rtol=5e-5)
  for actual, saved in retained:
    assert torch.equal(actual, saved)
  shifted_motion = api.mj_subtreeVel(sim)
  for name, tensor in motion.items():
    assert torch.equal(tensor, retained_motion[name])
    np.testing.assert_allclose(shifted_motion[name][:, 1].cpu().numpy(),
                               expected_motion[name][:, 1], atol=8e-6, rtol=5e-5)
