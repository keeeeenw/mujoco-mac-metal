# Copyright 2026 The MuJoCo Metal contributors
# Licensed under the Apache License, Version 2.0.
"""Owned full-state Cartesian stage outputs against pinned MuJoCo 3.10."""
import os
from types import SimpleNamespace

import mujoco
import numpy as np
import pytest

from tests.test_spatial_queries_019 import _fixture, _stages


def _batch_stage(model, device="cpu"):
  torch = pytest.importorskip("torch")
  worlds = [_stages(model, seed, device) for seed in (17, 31)]
  stage = {key: torch.cat([world[1][key] for world in worlds])
           for key in ("root_com", "cvel", "cdof", "cdof_dot")}
  stage["poses"] = {key: torch.cat([world[1]["poses"][key] for world in worlds])
                    for key in worlds[0][1]["poses"]}
  return worlds, stage


def _assert_fields(result, worlds, atol=7e-6):
  for name, value in result.items():
    expected = np.stack([np.asarray(getattr(world[0], name)) for world in worlds])
    if name in ("xmat", "ximat", "geom_xmat", "site_xmat"):
      expected = expected.reshape(expected.shape[:-1] + (3, 3))
    np.testing.assert_allclose(value.cpu().numpy(), expected,
                               rtol=4e-5, atol=atol, err_msg=name)


def test_cartesian_stage_cpu_oracle_all_joint_chains_and_owned_outputs():
  torch = pytest.importorskip("torch")
  from mujoco_metal import native_api
  model = _fixture()
  worlds, stage = _batch_stage(model)
  sim = SimpleNamespace(model=model, _mjmodel=model, batch_size=2,
      state=SimpleNamespace(_device=torch.device("cpu")))
  for operation in (native_api.mj_kinematics, native_api.mj_comPos,
                    native_api.mj_comVel):
    result = operation(sim, dynamics=stage)
    _assert_fields(result, worlds)
    retained = {name: value.clone() for name, value in result.items()}
    another = operation(sim, dynamics=stage)
    for value in another.values():
      value.zero_()
    for name, saved in retained.items():
      assert torch.equal(result[name], saved)


def test_com_position_zero_mass_subtree_uses_inertial_frame_not_body_frame():
  torch = pytest.importorskip("torch")
  from mujoco_metal.spatial_queries import DeviceSpatialQueries
  model = mujoco.MjModel.from_xml_string('''<mujoco><worldbody>
    <body pos="1 2 3"><inertial pos=".2 -.3 .4" mass="0"
        diaginertia="0 0 0"/><site pos=".1 0 0"/></body>
  </worldbody></mujoco>''')
  worlds, stage = _batch_stage(model)
  assert model.body_subtreemass[1] == 0
  assert not torch.equal(stage["poses"]["inertial_pos"][:, 1],
                         stage["poses"]["body_pos"][:, 1])
  result = DeviceSpatialQueries(model, 2, "cpu").center_of_mass_position(stage)
  _assert_fields(result, worlds, atol=8e-7)
  assert torch.count_nonzero(result["cinert"]) == 0
  assert result["cdof"].shape == (2, 0, 6)


def test_cartesian_stage_budget_rejects_before_program_or_query_mutation(monkeypatch):
  from mujoco_metal import native_api
  model = _fixture()
  sim = SimpleNamespace(model=model, batch_size=2,
                         limits=SimpleNamespace(memory_budget_bytes=1))
  def forbidden(*args, **kwargs):
    raise AssertionError("query executed before memory admission")
  monkeypatch.setattr(native_api, "_spatial_query", forbidden)
  for query in (native_api.mj_kinematics, native_api.mj_comPos, native_api.mj_comVel):
    with pytest.raises(ValueError, match="budget"):
      query(sim)
  sim.batch_size = 1 << 31
  with pytest.raises(ValueError, match="int32"):
    native_api.mj_comPos(sim)


@pytest.mark.gpu
@pytest.mark.skipif(os.getenv("MUJOCO_METAL_RUN_GPU") != "1",
                    reason="native Cartesian stage qualification requires opt-in")
@pytest.mark.parametrize("profile", ("integrated_euler_v1", "integrated_scalable_v1"))
def test_native_cartesian_queries_preserve_prepared_state_without_cpu_fallback(
    profile, monkeypatch):
  torch = pytest.importorskip("torch")
  if not torch.backends.mps.is_available():
    pytest.skip("native Cartesian queries require MPS")
  from mujoco_metal import native_api
  from mujoco_metal.forward_stages import ForwardStage
  from mujoco_metal.simulation import MetalSimulation
  model = _fixture()
  worlds, _ = _batch_stage(model)
  qp = np.stack([world[0].qpos for world in worlds]).astype(np.float32)
  qv = np.stack([world[0].qvel for world in worlds]).astype(np.float32)
  for world, (data, *_unused) in enumerate(worlds):
    data.qpos[:] = qp[world]
    data.qvel[:] = qv[world]
    mujoco.mj_forward(model, data)
  sim = MetalSimulation(model, batch_size=2, qpos=qp, qvel=qv, profile=profile)
  record = native_api.mj_fwdPosition(sim, return_record=True, skipsensor=True)
  native_api.mj_fwdVelocity(sim, record=record, skipsensor=True)
  saved_state = tuple(t.clone() for t in (sim.state._qpos, sim.state._qvel))
  def forbidden(*args, **kwargs):
    raise AssertionError("CPU Cartesian stage fallback invoked")
  for name in ("mj_forward", "mj_kinematics", "mj_comPos", "mj_comVel", "mj_rne"):
    monkeypatch.setattr(mujoco, name, forbidden)
  retained = []
  for query in (native_api.mj_kinematics, native_api.mj_comPos, native_api.mj_comVel):
    result = query(sim)
    assert all(value.device.type == "mps" for value in result.values())
    _assert_fields(result, worlds)
    retained.extend((value, value.clone()) for value in result.values())
  for value, saved in zip((sim.state._qpos, sim.state._qvel), saved_state):
    assert torch.equal(value, saved)
  assert sim._forward_stages.current(
      generation=sim.state.generation, minimum=ForwardStage.VEL) is record
  sim.reset(qpos=qp, qvel=np.zeros_like(qv))
  native_api.mj_comVel(sim)
  for value, saved in retained:
    assert torch.equal(value, saved)


@pytest.mark.gpu
@pytest.mark.skipif(os.getenv("MUJOCO_METAL_RUN_GPU") != "1",
                    reason="native mocap/zero-DOF Cartesian qualification")
def test_native_cartesian_mocap_and_massless_subtree_source_fields(monkeypatch):
  torch = pytest.importorskip("torch")
  from mujoco_metal import native_api
  from mujoco_metal.simulation import MetalSimulation
  model = mujoco.MjModel.from_xml_string('''<mujoco>
    <worldbody><body name="target" mocap="true" pos="1 .2 .4">
      <geom type="box" size=".1 .05 .04" contype="0" conaffinity="0"/>
      <site pos=".13 -.07 .03" quat=".92387953 .38268343 0 0"/>
      <body pos=".2 .1 -.1"><inertial pos=".03 -.02 .05" mass="0"
        diaginertia="0 0 0"/><site/></body>
    </body></worldbody></mujoco>''')
  assert model.nv == 0 and model.nmocap == 1
  positions = np.asarray([[[1.1, -.3, .7]], [[-.4, .6, 1.2]]], np.float32)
  rotations = np.asarray([[[1., 0., 0., 0.]],
                          [[np.sqrt(.5), 0., np.sqrt(.5), 0.]]], np.float32)
  worlds = []
  for world in range(2):
    data = mujoco.MjData(model)
    data.mocap_pos[:] = positions[world]
    data.mocap_quat[:] = rotations[world]
    mujoco.mj_forward(model, data)
    worlds.append((data,))
  sim = MetalSimulation(model, batch_size=2, profile="integrated_euler_v1")
  sim.set_mocap(positions, rotations)
  retained_inputs = (sim.state._mpos.clone(), sim.state._mquat.clone())
  def forbidden(*args, **kwargs):
    raise AssertionError("CPU mocap kinematics fallback invoked")
  for name in ("mj_forward", "mj_kinematics", "mj_comPos", "mj_comVel"):
    monkeypatch.setattr(mujoco, name, forbidden)
  for query in (native_api.mj_kinematics, native_api.mj_comPos, native_api.mj_comVel):
    _assert_fields(query(sim), worlds, atol=9e-7)
  for value, saved in zip((sim.state._mpos, sim.state._mquat), retained_inputs):
    assert torch.equal(value, saved)


@pytest.mark.gpu
@pytest.mark.skipif(os.getenv("MUJOCO_METAL_RUN_GPU") != "1",
                    reason="native zero-DOF full-step qualification")
@pytest.mark.parametrize("integrator,profile", [
    ("Euler", "integrated_euler_v1"),
    ("RK4", "integrated_rk4_v1"),
    ("implicitfast", "integrated_implicitfast_v1"),
    ("implicit", "integrated_implicit_v1"),
])
def test_native_zero_dof_full_step_sensors_and_checkpoint_replay(
    integrator, profile, monkeypatch):
  torch = pytest.importorskip("torch")
  from mujoco_metal.simulation import MetalSimulation
  model = mujoco.MjModel.from_xml_string(f'''<mujoco>
    <option timestep=".002" integrator="{integrator}">
      <flag contact="disable"/></option>
    <worldbody><body name="display" mocap="true" pos=".2 -.1 .4">
      <geom type="sphere" size=".1" contype="0" conaffinity="0"/>
      <site name="probe" pos=".03 -.02 .05"/>
    </body></worldbody>
    <sensor><framepos objtype="site" objname="probe"/>
      <gyro site="probe"/><accelerometer site="probe"/></sensor>
  </mujoco>''')
  assert model.nq == model.nv == model.nu == 0
  cpu = mujoco.MjData(model)
  for _ in range(3):
    mujoco.mj_step(model, cpu)
  sim = MetalSimulation(model, batch_size=2, profile=profile)
  initial = sim.snapshot()
  def forbidden(*args, **kwargs):
    raise AssertionError("CPU fallback in zero-DOF step")
  for name in ("mj_step", "mj_forward", "mj_rnePostConstraint"):
    monkeypatch.setattr(mujoco, name, forbidden)
  sim.step(3)
  np.testing.assert_array_equal(sim.state._status.cpu().numpy(), [0, 0])
  np.testing.assert_allclose(sim.state._time.cpu().numpy(), cpu.time,
                             rtol=2e-6, atol=1e-8)
  np.testing.assert_allclose(sim._sensordata.cpu().numpy(),
      np.repeat(cpu.sensordata[None], 2, axis=0), rtol=2e-5, atol=2e-6)
  saved = sim._sensordata.clone()
  sim.restore(initial)
  sim.step(3)
  torch.testing.assert_close(sim._sensordata, saved, rtol=0, atol=0)
  np.testing.assert_array_equal(sim.state._status.cpu().numpy(), [0, 0])
