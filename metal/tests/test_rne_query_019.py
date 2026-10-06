# Copyright 2026 The MuJoCo Metal contributors
# Licensed under the Apache License, Version 2.0.
"""Full-state RNE force queries against the independent pinned CPU engine."""
import os
from types import SimpleNamespace

import mujoco
import numpy as np
import pytest

from tests.test_spatial_queries_019 import _fixture, _stages


@pytest.mark.parametrize("gravity_disabled", [False, True])
@pytest.mark.parametrize("include_acceleration", [0, 1])
def test_rne_all_joint_chains_and_rotated_inertia_against_pinned_cpu(
    gravity_disabled, include_acceleration):
  torch = pytest.importorskip("torch")
  from mujoco_metal import native_api
  model = _fixture(gravity_disabled)
  model.dof_armature[:] = .07
  worlds = [_stages(model, seed, "cpu") for seed in (17, 31)]
  dynamics = {key: torch.cat([world[1][key] for world in worlds])
              for key in ("root_com", "cvel", "cdof", "cdof_dot")}
  dynamics["poses"] = {key: torch.cat([world[1]["poses"][key] for world in worlds])
                       for key in worlds[0][1]["poses"]}
  velocity = torch.cat([world[2] for world in worlds])
  acceleration = torch.cat([world[3] for world in worlds])
  sim = SimpleNamespace(model=model, _mjmodel=model, batch_size=2,
      state=SimpleNamespace(_device=torch.device("cpu"), _qvel=velocity,
                            _qacc=acceleration))
  expected = np.empty((2, model.nv))
  for world, (data, *_rest) in enumerate(worlds):
    mujoco.mj_rne(model, data, include_acceleration, expected[world])
  before = {name: value.clone() for name, value in dynamics.items()
            if isinstance(value, torch.Tensor)}
  actual = native_api.mj_rne(sim, include_acceleration, dynamics=dynamics)
  np.testing.assert_allclose(actual.numpy(), expected, rtol=3e-5, atol=4e-6)
  if include_acceleration:
    data = worlds[0][0]
    bias, mass_product = np.empty(model.nv), np.empty(model.nv)
    mujoco.mj_rne(model, data, 0, bias)
    mujoco.mj_mulM(model, data, mass_product, data.qacc)
    np.testing.assert_allclose(mass_product + bias - expected[0],
                               model.dof_armature * data.qacc,
                               rtol=2e-10, atol=2e-12)
    assert np.max(np.abs(mass_product + bias - expected[0])) > .005
  retained = actual.clone()
  native_api.mj_rne(sim, include_acceleration, dynamics=dynamics).zero_()
  assert torch.equal(actual, retained)
  for key in before:
    assert torch.equal(dynamics[key], before[key])
  for bad in (-1, 2, 1.0, "yes"):
    with pytest.raises(ValueError):
      native_api.mj_rne(sim, bad, dynamics=dynamics)
  with pytest.raises(ValueError):
    native_api.mj_rne(sim, qacc=acceleration[:, :-1], dynamics=dynamics)


def test_rne_zero_dof_world_has_owned_empty_output():
  torch = pytest.importorskip("torch")
  from mujoco_metal.spatial_queries import DeviceSpatialQueries
  model = mujoco.MjModel.from_xml_string("<mujoco><worldbody/></mujoco>")
  empty = torch.zeros((2, 0), dtype=torch.float32)
  stage = {"poses": {"inertial_pos": torch.zeros((2, 1, 3)),
                     "inertial_quat": torch.tensor([[[1., 0., 0., 0.]]]).expand(2, -1, -1)},
           "root_com": torch.zeros((2, 1, 3))}
  result = DeviceSpatialQueries(model, 2, "cpu").recursive_newton_euler(
      stage, empty, empty, True)
  assert result.shape == (2, 0)
  assert result is not empty


@pytest.mark.gpu
@pytest.mark.skipif(os.getenv("MUJOCO_METAL_RUN_GPU") != "1", reason="opt-in native RNE")
@pytest.mark.parametrize("profile", ["integrated_euler_v1", "integrated_scalable_v1"])
def test_native_public_rne_owned_query_and_state_restoration(profile, monkeypatch):
  torch = pytest.importorskip("torch")
  if not torch.backends.mps.is_available():
    pytest.skip("native RNE requires MPS")
  from mujoco_metal import native_api
  from mujoco_metal.simulation import MetalSimulation
  model = _fixture()
  model.dof_armature[:] = .07
  worlds = [_stages(model, seed, "cpu") for seed in (17, 31)]
  qp = np.stack([world[0].qpos for world in worlds]).astype(np.float32)
  qv = np.stack([world[0].qvel for world in worlds]).astype(np.float32)
  qa = np.stack([world[0].qacc for world in worlds]).astype(np.float32)
  expected = {}
  for flag in (0, 1):
    expected[flag] = np.empty((2, model.nv))
    for world, (data, *_rest) in enumerate(worlds):
      data.qpos[:] = qp[world]
      data.qvel[:] = qv[world]
      mujoco.mj_forward(model, data)
      data.qacc[:] = qa[world]
      mujoco.mj_rne(model, data, flag, expected[flag][world])
  sim = MetalSimulation(model, batch_size=2, qpos=qp, qvel=qv, profile=profile)
  sim.state._qacc.copy_(torch.tensor(qa, device=sim.device))
  from mujoco_metal.forward_stages import ForwardStage
  velocity_record = native_api.mj_fwdPosition(sim, return_record=True, skipsensor=True)
  native_api.mj_fwdVelocity(sim, record=velocity_record, skipsensor=True)
  # Preserve exact prepared workspace identities and values through queries.
  saved_qp, saved_qv = sim.state._qpos.clone(), sim.state._qvel.clone()
  def forbidden(*args, **kwargs):
    raise AssertionError("CPU RNE/physics fallback invoked")
  for name in ("mj_rne", "mj_forward", "mj_kinematics", "mj_comPos", "mj_comVel"):
    monkeypatch.setattr(mujoco, name, forbidden)
  retained = []
  for flag in (0, 1):
    result = native_api.mj_rne(sim, flag)
    assert result.device.type == "mps"
    np.testing.assert_allclose(result.cpu().numpy(), expected[flag], atol=8e-6, rtol=5e-5)
    retained.append((result, result.clone()))
  assert torch.equal(sim.state._qpos, saved_qp)
  assert torch.equal(sim.state._qvel, saved_qv)
  assert sim._forward_stages.current(
      generation=sim.state.generation, minimum=ForwardStage.VEL) is velocity_record
  sim.state._qacc.zero_()
  np.testing.assert_allclose(native_api.mj_rne(sim).cpu().numpy(), expected[0],
                             atol=8e-6, rtol=5e-5)
  for result, saved in retained:
    assert torch.equal(result, saved)
