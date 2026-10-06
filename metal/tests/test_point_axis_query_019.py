# Copyright 2026 The MuJoCo Metal contributors
# Licensed under the Apache License, Version 2.0.
"""Pinned and independent tangent checks for the point/axis query."""
import os

import mujoco
import numpy as np
import pytest

def _fixture():
  return mujoco.MjModel.from_xml_string('''<mujoco>
    <option><flag contact="disable"/></option>
    <worldbody><body name="root" pos=".1 .2 .6"><freejoint/>
      <geom type="box" size=".1 .12 .08" pos=".07 .02 .06"/>
      <body pos=".3 .1 .2"><joint type="ball"/>
        <geom type="capsule" size=".05" fromto="0 0 0 .4 .1 .1"/>
        <body pos=".4 .1 .1"><joint type="hinge" axis=".2 .3 1"/>
          <geom size=".08" pos=".2 0 .1"/>
          <body pos=".1 .2 .3"><joint type="slide" axis="1 .2 .1"/>
            <geom size=".07"/>
          </body>
          <body pos=".2 .3 -.1"><geom size=".04"/></body>
        </body>
      </body>
    </body><body pos="-1 0 0"><geom size=".1"/></body></worldbody>
  </mujoco>''')


def _stages(model, seed, device):
  import torch
  rng = np.random.default_rng(seed)
  data = mujoco.MjData(model)
  mujoco.mj_integratePos(model, data.qpos, rng.normal(size=model.nv), .15)
  data.qvel[:] = rng.normal(size=model.nv) * .4
  mujoco.mj_forward(model, data)
  def tensor(value):
    return torch.tensor(np.asarray(value).copy()[None], dtype=torch.float32,
                        device=device)
  dynamics = {name: tensor(value) for name, value in (
      ("cdof", data.cdof), ("root_com", data.subtree_com))}
  dynamics["poses"] = {"body_pos": tensor(data.xpos)}
  return data, dynamics, None, None


@pytest.mark.parametrize("device", ["cpu", pytest.param("mps", marks=[
    pytest.mark.gpu, pytest.mark.skipif(os.getenv("MUJOCO_METAL_RUN_GPU") != "1",
                                      reason="native opt-in")])])
@pytest.mark.parametrize("seed", [17, 31])
@pytest.mark.parametrize("axis", [[.3, -1.7, 2.1], [0., 0., 0.]])
def test_point_axis_queries_match_pinned_and_independent_tangents(device, seed, axis):
  torch = pytest.importorskip("torch")
  from mujoco_metal.spatial_queries import DeviceSpatialQueries
  model = _fixture()
  data, dynamics, _, _ = _stages(model, seed, device)
  program = DeviceSpatialQueries(model, 1, device)
  axis = torch.tensor([axis], dtype=torch.float32, device=device)
  # Include world, all joint families, and a welded leaf.
  for body in range(model.nbody):
    point = dynamics["poses"]["body_pos"][:, body] + axis.new_tensor([[.1, -.2, .05]])
    saved_point, saved_axis = point.clone(), axis.clone()
    jp, ja = program.jac_point_axis(dynamics, point, axis, body)
    p, a = point.cpu().numpy()[0].astype(np.float64), axis.cpu().numpy()[0].astype(np.float64)
    expected_p, expected_a = np.zeros((3, model.nv)), np.zeros((3, model.nv))
    mujoco.mj_jacPointAxis(model, data, expected_p, expected_a, p, a, body)
    np.testing.assert_allclose(jp.cpu().numpy()[0], expected_p, rtol=3e-5, atol=3e-6)
    np.testing.assert_allclose(ja.cpu().numpy()[0], expected_a, rtol=3e-5, atol=3e-6)
    torch.testing.assert_close(point, saved_point, rtol=0, atol=0)
    torch.testing.assert_close(axis, saved_axis, rtol=0, atol=0)
    # The axis magnitude is preserved, including zero; finite differences
    # rotate a body-fixed vector and move a body-fixed point independently.
    direction = np.linspace(-.4, .6, model.nv)
    rotation = data.xmat[body].reshape(3, 3)
    local_axis = rotation.T @ a
    local_point = rotation.T @ (p - data.xpos[body])
    nudged = []
    eps = 1e-6
    for sign in (-1, 1):
      next_data = mujoco.MjData(model)
      next_data.qpos[:] = data.qpos
      mujoco.mj_integratePos(model, next_data.qpos, direction, sign * eps)
      mujoco.mj_forward(model, next_data)
      next_rotation = next_data.xmat[body].reshape(3, 3)
      nudged.append((next_data.xpos[body] + next_rotation @ local_point,
                     next_rotation @ local_axis))
    np.testing.assert_allclose(jp.cpu().numpy()[0] @ direction,
                               (nudged[1][0] - nudged[0][0]) / (2 * eps),
                               rtol=3e-5, atol=3e-6)
    np.testing.assert_allclose(ja.cpu().numpy()[0] @ direction,
                               (nudged[1][1] - nudged[0][1]) / (2 * eps),
                               rtol=3e-5, atol=3e-6)
    retained_p, retained_a = jp.clone(), ja.clone()
    program.jac_point_axis(dynamics, point + 1, axis * 2, body)
    torch.testing.assert_close(jp, retained_p, rtol=0, atol=0)
    torch.testing.assert_close(ja, retained_a, rtol=0, atol=0)


@pytest.mark.gpu
@pytest.mark.skipif(os.getenv("MUJOCO_METAL_RUN_GPU") != "1", reason="native opt-in")
def test_public_point_axis_query_preserves_prepared_simulation_and_validates_inputs():
  import torch
  from mujoco_metal import MetalSimulation
  from mujoco_metal.native_api import mj_forward, mj_jacPointAxis
  model = _fixture()
  sim = MetalSimulation(model, batch_size=2, profile="integrated_euler_v1")
  mj_forward(sim)
  record = sim._forward_stages._record
  qpos, qvel = sim._state._qpos.clone(), sim._state._qvel.clone()
  generation = sim.state.generation
  point = torch.tensor([[.4, -.1, .7], [-.2, .3, .1]], device="mps")
  axis = torch.tensor([[.2, .4, 2.], [0., 0., 0.]], device="mps")
  jp, ja = mj_jacPointAxis(sim, point, axis, model.nbody - 2)
  assert jp.shape == ja.shape == (2, 3, model.nv)
  for w in range(2):
    data = mujoco.MjData(model)
    data.qpos[:] = qpos[w].cpu().numpy()
    data.qvel[:] = qvel[w].cpu().numpy()
    mujoco.mj_forward(model, data)
    p, a = np.empty((3, model.nv)), np.empty((3, model.nv))
    mujoco.mj_jacPointAxis(model, data, p, a,
                         point[w].cpu().numpy().astype(np.float64),
                         axis[w].cpu().numpy().astype(np.float64), model.nbody - 2)
    np.testing.assert_allclose(jp[w].cpu().numpy(), p, rtol=3e-5, atol=3e-6)
    np.testing.assert_allclose(ja[w].cpu().numpy(), a, rtol=3e-5, atol=3e-6)
  with pytest.raises(ValueError):
    mj_jacPointAxis(sim, point, axis[:, :2], 1)
  with pytest.raises((ValueError, TypeError)):
    mj_jacPointAxis(sim, point, axis, True)
  torch.testing.assert_close(sim._state._qpos, qpos, rtol=0, atol=0)
  torch.testing.assert_close(sim._state._qvel, qvel, rtol=0, atol=0)
  assert sim.state.generation == generation
  assert sim._forward_stages._record is record
