# Copyright 2026 The MuJoCo Metal contributors
# Licensed under the Apache License, Version 2.0.
"""Active elastic material invariance and free-carrier reaction witnesses.

All numerical oracles are independent CPU data; none is supplied to the
native pipeline.
"""
import os

import mujoco
import numpy as np
import pytest

pytestmark = [pytest.mark.gpu, pytest.mark.skipif(
    os.getenv('MUJOCO_METAL_RUN_GPU') != '1', reason='native opt-in')]


def _model(dim, young):
  count = '3 3 1' if dim == 2 else '2 2 2'
  membrane = ' thickness=".02" elastic2d="stretch"' if dim == 2 else ''
  return mujoco.MjModel.from_xml_string(f'''<mujoco>
    <option gravity="0 0 0" timestep=".00002"/>
    <worldbody><body name="carrier" pos="0 0 .5">
      <freejoint name="carrier_joint"/>
      <inertial pos=".02 -.03 .04" mass="2" diaginertia=".2 .25 .3"/>
      <flexcomp name="material" type="grid" count="{count}" dim="{dim}"
          spacing=".1 .1 .1" mass="1">
        <pin id="0"/>
        <contact contype="0" conaffinity="0" selfcollide="none"/>
        <edge stiffness="0" damping="0"/>
        <elasticity young="{young}" poisson=".23" damping="0"{membrane}/>
      </flexcomp>
    </body></worldbody>
  </mujoco>''')


def _state(model):
  qpos = np.tile(model.qpos0.astype(np.float32), (4, 1))
  root = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, 'carrier_joint')
  root_q = int(model.jnt_qposadr[root])
  vertex = 4  # An unpinned material vertex in both fixtures.
  body = int(model.flex_vertbodyid[vertex])
  first = int(model.body_jntadr[body])
  assert int(model.body_jntnum[body]) == 3
  deformation_q = int(model.jnt_qposadr[first + 2])
  qpos[2:, deformation_q] += .012
  for world in (1, 3):
    qpos[world, root_q:root_q + 3] = [2., -1., 3.]
    # Exact quaternion cyclically permutes XYZ, avoiding a trigonometric oracle.
    qpos[world, root_q + 3:root_q + 7] = [.5, .5, .5, .5]
  return qpos, int(model.jnt_dofadr[root])


def _momentum_rate_operators(model, data):
  linear = np.zeros((3, model.nv))
  angular = np.zeros_like(linear)
  for body in range(1, model.nbody):
    jp, jr = np.zeros_like(linear), np.zeros_like(linear)
    mujoco.mj_jacBodyCom(model, data, jp, jr, body)
    mass_jp = float(model.body_mass[body]) * jp
    rot = np.asarray(data.ximat[body]).reshape(3, 3)
    inertia = rot @ np.diag(model.body_inertia[body]) @ rot.T
    linear += mass_jp
    angular += np.cross(data.xipos[body], mass_jp.T).T + inertia @ jr
  return linear, angular


@pytest.mark.parametrize('dim', [2, 3])
@pytest.mark.parametrize('young', [1000., 4000.])
def test_active_material_rigid_invariance_and_carrier_reaction(dim, young):
  from mujoco_metal import MetalSimulation
  qpos, root_dof = _state(model := _model(dim, young))
  refs = []
  operators = []
  for world in range(4):
    data = mujoco.MjData(model)
    data.qpos[:] = qpos[world]
    data.qvel[:] = 0.
    mujoco.mj_forward(model, data)
    refs.append(data)
    operators.append(_momentum_rate_operators(model, data))
  source_force = np.stack([d.qfrc_passive.copy() for d in refs])
  source_acc = np.stack([d.qacc.copy() for d in refs])
  assert np.linalg.norm(source_force[2]) > 1e-3
  assert np.linalg.norm(source_force[3]) > 1e-3
  np.testing.assert_allclose(source_force[:2], 0., rtol=0, atol=1e-8)
  # Material stiffness must matter, rather than every force check being zero.
  base_model = _model(dim, 1000.)
  base = mujoco.MjData(base_model)
  base.qpos[:] = qpos[2]
  mujoco.mj_forward(base_model, base)
  np.testing.assert_allclose(source_force[2], base.qfrc_passive * (young / 1000.),
                             rtol=2e-10, atol=1e-8)
  sim = MetalSimulation(model, batch_size=4, qpos=qpos,
                        qvel=np.zeros((4, model.nv), np.float32),
                        profile='integrated_scalable_v1')
  host = lambda tensor: tensor.detach().cpu().numpy().copy()
  system = sim.assembled_system(recompute=True)
  np.testing.assert_array_equal(host(system['status']), 0)
  force = host(sim._flex._qfrc_passive)
  np.testing.assert_allclose(force, source_force, rtol=7e-4, atol=5e-6)
  np.testing.assert_allclose(force[:2], 0., rtol=0, atol=5e-6)
  assert np.linalg.norm(force[2]) > 1e-3
  assert np.linalg.norm(force[3]) > 1e-3
  # Every internal sliding coordinate is expressed in the carrier frame.
  np.testing.assert_allclose(force[2, root_dof+6:], force[3, root_dof+6:],
                             rtol=7e-4, atol=5e-6)
  checkpoint = sim.snapshot()
  np.testing.assert_array_equal(host(sim.step()), 0)
  acceleration = host(sim.state._qacc)
  np.testing.assert_allclose(acceleration, source_acc, rtol=8e-4, atol=1e-5)
  for world in (2, 3):
    # A pin attached to a free carrier must accelerate that carrier. Merely
    # returning vertex forces while leaving the rigid body fixed cannot pass.
    assert np.linalg.norm(acceleration[world, root_dof:root_dof+6]) > 1e-5
    linear, angular = operators[world]
    for operator in (linear, angular):
      pieces = operator * acceleration[world][None, :]
      rate = pieces.sum(axis=1)
      scale = max(1., float(np.abs(pieces).sum()))
      np.testing.assert_allclose(rate, 0., rtol=0., atol=2e-5 * scale)
  final = {name: host(getattr(sim.state, '_' + name))
           for name in ('qpos', 'qvel', 'qacc')}
  sim.restore(checkpoint)
  np.testing.assert_array_equal(host(sim.step()), 0)
  for name, expected in final.items():
    np.testing.assert_array_equal(host(getattr(sim.state, '_' + name)), expected)
