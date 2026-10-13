# Copyright 2026 The MuJoCo Metal contributors
# Licensed under the Apache License, Version 2.0.
"""Nonzero rigid transmissions: pinned stages, trajectories and full replay."""
import os
from pathlib import Path
import mujoco
import numpy as np
import pytest

CASES = ('ball', 'ball_parent', 'free', 'free_parent', 'site', 'refsite', 'refsite_shared', 'crank', 'crank_infeasible')


def _model(kind):
  parent = kind.endswith('_parent')
  if kind.startswith('ball'):
    body = '<body quat=".9 .1 .2 .3"><joint name="j" type="ball" damping=".1"/><geom size=".12" mass="2"/></body>'
    transmission = f'{"jointinparent" if parent else "joint"}="j" gear=".4 -.7 1.2"'
  elif kind.startswith('free'):
    body = '<body pos=".1 .2 .4" quat=".9 .1 .2 .3"><freejoint name="j"/><geom size=".12" mass="2"/></body>'
    transmission = f'{"jointinparent" if parent else "joint"}="j" gear=".2 -.3 .4 .4 -.7 1.2"'
  else:
    lookup = 'refsite' if kind == 'refsite_shared' else 'crank' if kind.startswith('crank') else kind
    body = ('<site name="ref" pos="-.1 .03 .15" quat=".9 .1 .2 .3"/>'
            '<body pos="0 0 .2"><joint name="j" axis="0 1 0" damping=".1"/>'
            '<geom size=".1" pos=".2 0 0" mass="2"/>'
            '<site name="moving" pos=".2 .04 .08" quat=".9 .1 .2 .3"/></body>')
    transmission = {'site':'site="moving" gear=".2 -.3 .4 .4 -.7 1.2"',
                    'refsite':'site="moving" refsite="ref" gear=".2 -.3 .4 .4 -.7 1.2"',
                    'crank':'cranksite="moving" slidersite="ref" cranklength=".6" gear="-1.3"'}[lookup]
  if kind == "refsite_shared":
    body = '<body pos=".03 -.06 .1"><joint name="common" axis="0 0 1" damping=".1"/><geom size=".1" mass="1"/>' + body + '</body>'
  if kind.startswith("crank"):
    body = body.replace('<site name="ref" pos="-.1 .03 .15" quat=".9 .1 .2 .3"/>',
                        '<body pos="-.1 .03 .15"><joint name="slider_axis" axis="1 0 0" damping=".1"/><geom size=".1" mass="2"/><site name="ref" quat=".9 .1 .2 .3"/></body>')
  if kind == "crank_infeasible":
    transmission = transmission.replace('cranklength=".6"', 'cranklength=".05"')
  # Antagonistic actuators exercise both gear and force signs, nonzero filters
  # and affine length/velocity-dependent forces without contact masking errors.
  model = mujoco.MjModel.from_xml_string(f'''<mujoco>
    <option gravity="0 0 0" timestep=".0005"><flag contact="disable"/></option>
    <worldbody>{body}</worldbody><actuator>
      <general {transmission} dyntype="filterexact" dynprm=".04"
        actearly="true" gaintype="affine" gainprm="1.2 .1 -.03"
        biastype="affine" biasprm=".4 -.15 -.04"
        ctrllimited="true" ctrlrange="-.5 .5" forcelimited="true" forcerange="-.8 .9"/>
      <general {transmission} gainprm="-.7"/>
    </actuator></mujoco>''')
  return model


def _states(model):
  q = np.tile(model.qpos0, (2, 1))
  for world in range(2):
    delta = np.linspace(.13, .31, model.nv) * (1 if world == 0 else -.7)
    mujoco.mj_integratePos(model, q[world], delta, 1)
  v = np.stack([np.linspace(.09, .23, model.nv), np.linspace(-.21, -.07, model.nv)])
  return q.astype(np.float32), v.astype(np.float32), np.asarray([[.22], [-.17]], np.float32)


def _reference(model, q, v, act, ctrl):
  data = mujoco.MjData(model)
  data.qpos[:], data.qvel[:], data.act[:], data.ctrl[:] = q, v, act, ctrl
  mujoco.mj_forward(model, data)
  return data


def _moment(model, data):
  result = np.zeros((model.nu, model.nv))
  for row in range(model.nu):
    start, count = data.moment_rowadr[row], data.moment_rownnz[row]
    result[row, data.moment_colind[start:start+count]] = data.actuator_moment[start:start+count]
  return result


@pytest.mark.parametrize('kind', CASES)
def test_rigid_transmission_oracles_are_nonzero_and_directional(kind):
  model = _model(kind)
  q, v, act = _states(model)
  for world in range(2):
    data = _reference(model, q[world], v[world], act[world], [.4, -.3])
    moment = _moment(model, data)
    assert np.linalg.norm(moment) > .01
    assert np.linalg.norm(data.actuator_velocity) > .001
    assert np.linalg.norm(data.qfrc_actuator) > .01
    assert abs(data.act_dot[0]) > 1
    np.testing.assert_allclose(data.actuator_velocity, moment @ v[world], atol=1e-12)
    # Translation-only site and slider-crank have ordinary scalar length
    # derivatives; ball/free rotation conventions are checked against source.
    if kind.startswith('crank'):
      plus, minus = q[world].astype(float), q[world].astype(float)
      direction = np.ones(model.nv)
      mujoco.mj_integratePos(model, plus, direction, 1e-6)
      mujoco.mj_integratePos(model, minus, direction, -1e-6)
      dp = _reference(model, plus, v[world], act[world], [.4, -.3])
      dm = _reference(model, minus, v[world], act[world], [.4, -.3])
      np.testing.assert_allclose((dp.actuator_length-dm.actuator_length)/2e-6,
                                 moment @ direction, rtol=2e-6, atol=1e-7)


@pytest.mark.gpu
@pytest.mark.skipif(os.getenv('MUJOCO_METAL_RUN_GPU') != '1', reason='native GPU opt-in')
@pytest.mark.parametrize('kind', CASES)
def test_rigid_transmission_nonzero_stages_euler_and_checkpoint(kind, monkeypatch):
  import torch
  from mujoco_metal import MetalSimulation
  monkeypatch.syspath_prepend(str(Path(__file__).parents[1] / "examples"))
  from magnetic_crane import _same_checkpoint
  model = _model(kind)
  q, v, act = _states(model)
  sim = MetalSimulation(model, 2, profile='integrated_euler_v1')
  sim.reset(qpos=q, qvel=v, act=act)
  initial_ctrl = np.asarray([[.8, -.3], [-.8, .4]], np.float32)
  refs = [_reference(model, q[w], v[w], act[w], initial_ctrl[w]) for w in range(2)]
  def forbidden(*_args, **_kwargs):
    raise AssertionError('native transmission used CPU physics')
  names = ('mj_forward', 'mj_step', 'mj_fwdPosition', 'mj_fwdVelocity',
           'mj_fwdActuation', 'mj_fwdAcceleration', 'mj_fwdConstraint')
  with monkeypatch.context() as context:
    for name in names:
      context.setattr(mujoco, name, forbidden)
    dyn = sim._smooth.run_device(sim.state._qpos, sim.state._qvel)
    kin = sim._actuators.run_kinematics(sim.state._qpos, sim.state._qvel, dyn['poses'])
    force = sim._actuators.run_forces(torch.as_tensor(initial_ctrl, device='mps'), sim.state._act, kin)
  for key, expected in [('length', [d.actuator_length for d in refs]),
                        ('velocity', [d.actuator_velocity for d in refs]),
                        ('moment', [_moment(model, d) for d in refs])]:
    np.testing.assert_allclose(kin[key].cpu().numpy(), expected, rtol=3e-5, atol=3e-6)
  for key, field in [('force','actuator_force'), ('qfrc','qfrc_actuator'), ('act_dot','act_dot')]:
    np.testing.assert_allclose(force[key].cpu().numpy(), [getattr(d,field) for d in refs], rtol=3e-5, atol=3e-6)
  for step in range(16):
    ctrl = initial_ctrl if step < 8 else -.8*initial_ctrl
    with monkeypatch.context() as context:
      for name in names:
        context.setattr(mujoco, name, forbidden)
      np.testing.assert_array_equal(sim.step(ctrl=ctrl).cpu().numpy(), [0, 0])
    for world, data in enumerate(refs):
      data.ctrl[:] = ctrl[world]
      mujoco.mj_step(model, data)
    for field in ('qpos','qvel','qacc','act'):
      np.testing.assert_allclose(getattr(sim.state,field).cpu().numpy(),
                                 [getattr(d,field) for d in refs], rtol=3e-5, atol=3e-6)
  before = sim.snapshot()
  sim.step(3, ctrl=initial_ctrl)
  after = sim.snapshot()
  sim.restore(before)
  sim.step(3, ctrl=initial_ctrl)
  _same_checkpoint(after, sim.snapshot())


def _adhesion_model():
  return mujoco.MjModel.from_xml_string('''<mujoco>
    <option timestep=".0005" gravity="0 0 -9.81" iterations="120"/>
    <worldbody><geom name="floor" type="plane" size="2 2 .1"/>
      <body name="box" pos="0 0 .0499"><freejoint/>
        <geom name="box_geom" type="box" size=".05 .05 .05" mass=".5"/>
      </body></worldbody><contact><pair geom1="box_geom" geom2="floor"/></contact>
    <actuator><adhesion body="box" gain="1.2" ctrlrange="0 2"/></actuator>
  </mujoco>''')


def test_body_adhesion_oracle_has_contact_and_control_dependent_force():
  model = _adhesion_model()
  data = mujoco.MjData(model)
  data.qvel[2] = -.001
  forces = []
  for ctrl in (.4, 1.3):
    data.ctrl[0] = ctrl
    mujoco.mj_forward(model, data)
    assert data.ncon > 0 and np.linalg.norm(data.qfrc_constraint) > .01
    assert np.linalg.norm(data.qfrc_actuator) > .01
    forces.append(data.qfrc_actuator.copy())
  np.testing.assert_allclose(forces[1], forces[0]*(1.3/.4), atol=1e-12)


@pytest.mark.gpu
@pytest.mark.skipif(os.getenv('MUJOCO_METAL_RUN_GPU') != '1', reason='native GPU opt-in')
@pytest.mark.parametrize("velocity_axis", [0, 2])
def test_body_adhesion_changing_control_contact_stage_and_full_replay(monkeypatch, velocity_axis):
  from mujoco_metal import MetalSimulation, native_api as api
  from mujoco_metal.forward_stages import ForwardStage
  monkeypatch.syspath_prepend(str(Path(__file__).parents[1] / "examples"))
  from magnetic_crane import _same_checkpoint
  model = _adhesion_model()
  sim = MetalSimulation(model, 1, profile='integrated_euler_v1')
  data = mujoco.MjData(model)
  data.qvel[velocity_axis] = -.001 if velocity_axis == 2 else .001
  # The independent engine starts from the exact public float32 reset inputs.
  # Comparing its unrounded model qpos0 to rounded native inputs is not an
  # aligned contact/force oracle, even when their positions look identical.
  data.qpos[:] = data.qpos.astype(np.float32)
  data.qvel[:] = data.qvel.astype(np.float32)
  sim.reset(qpos=data.qpos.astype(np.float32)[None], qvel=data.qvel.astype(np.float32)[None])
  for step in range(12):
    ctrl = np.asarray([[.4 if step < 6 else 1.3]], np.float32)
    data.ctrl[:] = ctrl[0]
    mujoco.mj_forward(model, data)
    assert data.ncon > 0
    api.mj_setState(sim, {'ctrl':ctrl})
    with monkeypatch.context() as context:
      def forbidden(*_args, **_kwargs):
        raise AssertionError('native adhesion used CPU physics')
      for name in ('mj_forward','mj_step','mj_fwdPosition','mj_fwdVelocity',
                   'mj_fwdActuation','mj_fwdAcceleration','mj_fwdConstraint'):
        context.setattr(mujoco,name,forbidden)
      out = api.mj_forward(sim, skipsensor=True)
      np.testing.assert_allclose(out['record'].values[ForwardStage.ACT]['qfrc_actuator'].cpu().numpy()[0],
                                 data.qfrc_actuator, rtol=3e-5, atol=3e-6)
      np.testing.assert_allclose(sim._actuators._ws['force'].cpu().numpy()[0],
                                 data.actuator_force[0], rtol=3e-5, atol=3e-6)
      np.testing.assert_array_equal(sim.step(ctrl=ctrl).cpu().numpy(), [0])
    mujoco.mj_step(model,data)
    for field in ('qpos','qvel','qacc'):
      np.testing.assert_allclose(getattr(sim.state,field).cpu().numpy()[0],
                                 getattr(data,field), rtol=3e-5, atol=5e-5)
  before = sim.snapshot()
  sim.step(3,ctrl=ctrl)
  after = sim.snapshot()
  sim.restore(before)
  sim.step(3,ctrl=ctrl)
  _same_checkpoint(after,sim.snapshot())
