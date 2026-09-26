# Copyright 2026 The MuJoCo Metal contributors
# Licensed under the Apache License, Version 2.0.
"""Native servo and tendon stepping compared with CPU MuJoCo."""

import os
import mujoco
import numpy as np
import pytest
from mujoco_metal.simulation import MetalSimulation
from mujoco_metal.stepping import validate_stepping_profile

XML = """<mujoco><compiler angle="radian"/><option timestep=".002" gravity="0 0 0"><flag contact="disable"/></option>
<worldbody><body><joint name="x" type="slide" axis="1 0 0" damping=".1" ref=".1"/><geom type="box" size=".1 .1 .1" mass="1"/>
<body><joint name="y" type="slide" axis="0 1 0" damping=".2"/><geom type="sphere" size=".1" mass="1"/></body></body></worldbody>
<tendon><fixed name="diagonal"><joint joint="x" coef="1"/><joint joint="y" coef="-.5"/></fixed></tendon>
<actuator><position joint="x" kp="5" kv=".3"/><general tendon="diagonal" gaintype="affine" gainprm="2 .1 -.1" biastype="affine" biasprm=".1 -.2 -.3" ctrllimited="true" ctrlrange="-1 1" forcelimited="true" forcerange="-2 2"/></actuator></mujoco>"""


@pytest.mark.parametrize(
    'attribute,value', [('frictionloss', '.1'), ('limited', 'true')]
)
def test_tendon_dynamics_guard(attribute, value):
  xml = XML.replace(
      '<fixed name="diagonal">',
      f'<fixed name="diagonal" {attribute}="{value}" range="-1 1">',
  )
  m = mujoco.MjModel.from_xml_string(xml)
  with pytest.raises(ValueError, match='tendon'):
    validate_stepping_profile(m, profile='contact_free_transmission_euler_v1')


@pytest.mark.gpu
@pytest.mark.skipif(
    os.getenv('MUJOCO_METAL_RUN_GPU') != '1', reason='opt-in GPU'
)
@pytest.mark.parametrize('integrator', ['Euler', 'RK4'])
def test_transmission_trajectory(integrator):
  m = mujoco.MjModel.from_xml_string(
      XML.replace(
          'timestep=".002"', f'timestep=".002" integrator="{integrator}"'
      )
  )
  q = np.tile(m.qpos0, (2, 1)).astype('float32')
  q[1] += [0.2, -0.1]
  v = np.array([[0.1, -0.2], [-0.1, 0.2]], dtype='float32')
  sim = MetalSimulation(
      m, 2, q, v, profile=f'contact_free_transmission_{integrator.lower()}_v1'
  )
  refs = [mujoco.MjData(m) for _ in range(2)]
  for i, d in enumerate(refs):
    d.qpos[:] = q[i]
    d.qvel[:] = v[i]
  for step in range(400):
    ctrl = np.array(
        [[0.2 * np.sin(step * 0.02), 1.5], [-0.3, -1.5]], dtype='float32'
    )
    sim.step(ctrl=ctrl)
    for i, d in enumerate(refs):
      d.ctrl[:] = ctrl[i]
      mujoco.mj_step(m, d)
  snap = sim.state.snapshot()
  assert not np.any(snap.status)
  for i, d in enumerate(refs):
    np.testing.assert_allclose(snap.qpos[i], d.qpos, atol=2e-5, rtol=1e-4)
    np.testing.assert_allclose(snap.qvel[i], d.qvel, atol=3e-5, rtol=1e-4)
    np.testing.assert_allclose(snap.qacc[i], d.qacc, atol=8e-5, rtol=1e-4)
  saved = sim.state.snapshot()
  sim.step()
  first = sim.state.snapshot()
  sim.state.restore(saved)
  sim.step()
  second = sim.state.snapshot()
  np.testing.assert_array_equal(first.qpos, second.qpos)


@pytest.mark.gpu
@pytest.mark.skipif(
    os.getenv('MUJOCO_METAL_RUN_GPU') != '1', reason='opt-in GPU'
)
@pytest.mark.parametrize('integrator', ['Euler', 'RK4'])
@pytest.mark.parametrize(
    'disable',
    [
        0,
        int(mujoco.mjtDisableBit.mjDSBL_EULERDAMP),
        int(mujoco.mjtDisableBit.mjDSBL_DAMPER),
        int(mujoco.mjtDisableBit.mjDSBL_SPRING),
    ],
)
def test_coupled_tendon_dynamics_trajectory(integrator, disable):
  xml = XML.replace(
      'timestep=".002"', f'timestep=".002" integrator="{integrator}"'
  )
  xml = xml.replace(
      '<fixed name="diagonal">',
      '<fixed name="diagonal" stiffness="2" damping=".6" armature=".3" springlength=".05 .15">',
  )
  m = mujoco.MjModel.from_xml_string(xml)
  m.opt.disableflags |= disable
  m.tendon_stiffnesspoly[:] = [0.1, 0.2]
  m.tendon_dampingpoly[:] = [0.15, 0.1]
  q = np.array([[0.4, -0.3], [-0.3, 0.2]], dtype='float32')
  v = np.array([[0.4, -0.2], [-0.4, 0.6]], dtype='float32')
  sim = MetalSimulation(
      m, 2, q, v, profile=f'contact_free_transmission_{integrator.lower()}_v1'
  )
  refs = [mujoco.MjData(m) for _ in range(2)]
  for row, d in enumerate(refs):
    d.qpos[:] = q[row]
    d.qvel[:] = v[row]
  for step in range(400):
    ctrl = np.array([[0.1, 0.2], [-0.1, -0.4]], dtype='float32')
    sim.step(ctrl=ctrl)
    for row, d in enumerate(refs):
      d.ctrl[:] = ctrl[row]
      mujoco.mj_step(m, d)
  snap = sim.state.snapshot()
  assert not np.any(snap.status)
  for row, d in enumerate(refs):
    np.testing.assert_allclose(snap.qpos[row], d.qpos, atol=3e-5, rtol=1e-4)
    np.testing.assert_allclose(snap.qvel[row], d.qvel, atol=4e-5, rtol=1e-4)
    np.testing.assert_allclose(snap.qacc[row], d.qacc, atol=1e-4, rtol=1e-4)
