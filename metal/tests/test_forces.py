# Copyright 2026 The MuJoCo Metal contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Opt-in GPU qualification for explicit generalized force and damping."""

import os

import mujoco
import numpy as np
import pytest

from mujoco_metal.simulation import MetalSimulation

pytestmark = pytest.mark.gpu
_requires_gpu = pytest.mark.skipif(
    os.environ.get("MUJOCO_METAL_RUN_GPU") != "1",
    reason="requires explicit MUJOCO_METAL_RUN_GPU=1 and idle GPU",
)

_MIXED = """<mujoco><option timestep='.001' gravity='0 0 -9.81'>
  <flag contact='disable'/></option><worldbody>
  <body><freejoint/><geom type='sphere' size='.05'/>
    <body pos='0 0 .3'><joint type='hinge' axis='0 1 0' damping='.17'/>
      <geom type='box' size='.1 .2 .3'/>
      <body pos='.2 0 0'><joint type='slide' axis='1 0 0' damping='.43'/>
        <geom type='sphere' size='.04'/>
        <body><joint type='ball' damping='.08'/>
          <inertial pos='.1 0 0' mass='.4' diaginertia='.01 .02 .03'/>
        </body>
      </body>
    </body>
  </body>
</worldbody></mujoco>"""


def _oracle_step(model, qpos, qvel, force, count=1):
  data = mujoco.MjData(model)
  data.qpos[:] = qpos
  data.qvel[:] = qvel
  for _ in range(count):
    data.qfrc_applied[:] = force
    mujoco.mj_step(model, data)
  return data


@_requires_gpu
def test_mixed_joint_force_damping_matches_mujoco_and_force_is_per_call():
  import torch

  model = mujoco.MjModel.from_xml_string(_MIXED)
  batch = 3
  qpos = np.tile(model.qpos0.astype(np.float32), (batch, 1))
  qvel = np.stack(
      [
          np.linspace(-0.4, 0.3, model.nv),
          np.linspace(0.2, -0.5, model.nv),
          np.linspace(0.1, 0.6, model.nv),
      ]
  ).astype(np.float32)
  qpos[1, 0] = 0.05
  qpos[2, 0] = -0.08
  simulation = MetalSimulation(
      model,
      batch_size=batch,
      qpos=qpos,
      qvel=qvel,
      profile="contact_free_forces_euler_v1",
  )
  oracle = [mujoco.MjData(model) for _ in range(batch)]
  for i, data in enumerate(oracle):
    data.qpos[:] = qpos[i]
    data.qvel[:] = qvel[i]
  rng = np.random.default_rng(8301)
  force = rng.normal(size=(batch, model.nv)).astype(np.float32)
  for step in range(1000):
    force[:] = rng.normal(size=force.shape).astype(np.float32)
    for i, data in enumerate(oracle):
      data.qfrc_applied[:] = force[i]
      mujoco.mj_step(model, data)
    device_force = torch.tensor(force, device="mps")
    force_copy = device_force.clone()
    status = simulation.step(qfrc_applied=device_force)
    assert torch.equal(device_force, force_copy)
    assert torch.equal(status, torch.zeros_like(status))
  actual = simulation.state.snapshot()
  for i, data in enumerate(oracle):
    np.testing.assert_allclose(actual.qpos[i], data.qpos, rtol=3e-5, atol=5e-6)
    np.testing.assert_allclose(actual.qvel[i], data.qvel, rtol=3e-5, atol=1e-5)
    np.testing.assert_allclose(actual.qacc[i], data.qacc, rtol=2e-4, atol=4e-4)

  # Omitting the next force means zero applied force, not reuse of prior input.
  for data in oracle:
    data.qfrc_applied[:] = 0
    mujoco.mj_step(model, data)
  simulation.step()
  actual = simulation.state.snapshot()
  for i, data in enumerate(oracle):
    np.testing.assert_allclose(actual.qpos[i], data.qpos, rtol=3e-5, atol=5e-6)
    np.testing.assert_allclose(actual.qvel[i], data.qvel, rtol=3e-5, atol=1e-5)
    np.testing.assert_allclose(actual.qacc[i], data.qacc, rtol=2e-4, atol=4e-4)


@_requires_gpu
@pytest.mark.parametrize(
    "disabled",
    [
        (),
        (mujoco.mjtDisableBit.mjDSBL_EULERDAMP,),
        (mujoco.mjtDisableBit.mjDSBL_DAMPER,),
        (mujoco.mjtDisableBit.mjDSBL_GRAVITY,),
    ],
)
def test_damping_and_gravity_disable_flags_match_mujoco(disabled):
  model = mujoco.MjModel.from_xml_string(_MIXED)
  for flag in disabled:
    model.opt.disableflags |= int(flag)
  qpos = np.tile(model.qpos0.astype(np.float32), (2, 1))
  qvel = np.stack(
      [np.linspace(-0.3, 0.5, model.nv), np.linspace(0.2, -0.1, model.nv)]
  ).astype(np.float32)
  force = np.full((2, model.nv), 0.23, dtype=np.float32)
  simulation = MetalSimulation(
      model,
      batch_size=2,
      qpos=qpos,
      qvel=qvel,
      profile="contact_free_forces_euler_v1",
  )
  references = [
      _oracle_step(model, qpos[i], qvel[i], force[i]) for i in range(2)
  ]
  simulation.step(qfrc_applied=force)
  actual = simulation.state.snapshot()
  for i, data in enumerate(references):
    np.testing.assert_allclose(actual.qpos[i], data.qpos, rtol=3e-5, atol=5e-6)
    np.testing.assert_allclose(actual.qvel[i], data.qvel, rtol=3e-5, atol=1e-5)
    np.testing.assert_allclose(actual.qacc[i], data.qacc, rtol=2e-4, atol=4e-4)


@_requires_gpu
def test_bad_host_force_is_rejected_before_state_mutation():
  model = mujoco.MjModel.from_xml_string(_MIXED)
  simulation = MetalSimulation(
      model, batch_size=2, profile="contact_free_forces_euler_v1"
  )
  before = simulation.state.snapshot()
  for force in (
      np.zeros((1, model.nv), dtype=np.float32),
      np.full((2, model.nv), np.nan, dtype=np.float32),
      np.full((2, model.nv), np.finfo(np.float64).max),
  ):
    with pytest.raises(ValueError, match="qfrc_applied"):
      simulation.step(qfrc_applied=force)
    assert simulation.state.generation == 0
    after = simulation.state.snapshot()
    np.testing.assert_array_equal(after.qpos, before.qpos)
    np.testing.assert_array_equal(after.qvel, before.qvel)


@_requires_gpu
def test_mps_nonfinite_force_fails_only_its_world_and_status_stays_sticky():
  import torch

  model = mujoco.MjModel.from_xml_string(_MIXED)
  simulation = MetalSimulation(
      model, batch_size=2, profile="contact_free_forces_euler_v1"
  )
  force = torch.zeros((2, model.nv), dtype=torch.float32, device="mps")
  force[0, 0] = float("nan")
  initial = simulation.state.snapshot()
  status = simulation.step(qfrc_applied=force).clone()
  assert status[0].item() != 0 and status[1].item() == 0
  halfway = simulation.state.snapshot()
  np.testing.assert_array_equal(halfway.qpos[0], initial.qpos[0])
  np.testing.assert_array_equal(halfway.qvel[0], initial.qvel[0])
  assert not np.array_equal(halfway.qpos[1], initial.qpos[1])

  force.zero_()
  status = simulation.step(qfrc_applied=force).clone()
  assert status[0].item() != 0 and status[1].item() == 0
  final = simulation.state.snapshot()
  np.testing.assert_array_equal(final.qpos[0], halfway.qpos[0])
  np.testing.assert_array_equal(final.qvel[0], halfway.qvel[0])
  assert not np.array_equal(final.qpos[1], halfway.qpos[1])


@_requires_gpu
def test_force_input_does_not_mutate_and_restore_repeats_force_rollout():
  import torch

  model = mujoco.MjModel.from_xml_string(_MIXED)
  simulation = MetalSimulation(
      model, batch_size=2, profile="contact_free_forces_euler_v1"
  )
  checkpoint = simulation.state.snapshot()
  force = torch.full((2, model.nv), 0.125, dtype=torch.float32, device="mps")
  original = force.clone()
  simulation.step(steps=5, qfrc_applied=force)
  first = simulation.state.snapshot()
  assert torch.equal(force, original)
  simulation.state.restore(checkpoint)
  simulation.step(steps=5, qfrc_applied=force)
  second = simulation.state.snapshot()
  for name in ("qpos", "qvel", "qacc", "time", "status"):
    np.testing.assert_array_equal(getattr(first, name), getattr(second, name))
  host_force = np.full((2, model.nv), -0.075, dtype=np.float32)
  host_copy = host_force.copy()
  simulation.step(qfrc_applied=host_force)
  np.testing.assert_array_equal(host_force, host_copy)


@_requires_gpu
def test_zero_dof_world_accepts_empty_force_batch():
  import torch

  model = mujoco.MjModel.from_xml_string(
      "<mujoco><option><flag contact='disable'/></option><worldbody/></mujoco>"
  )
  simulation = MetalSimulation(
      model, batch_size=2, profile="contact_free_forces_euler_v1"
  )
  force = torch.empty((2, 0), dtype=torch.float32, device="mps")
  status = simulation.step(qfrc_applied=force)
  assert torch.equal(status, torch.zeros_like(status))


@_requires_gpu
def test_device_force_hotloop_uses_no_host_physics_or_readback(monkeypatch):
  import torch

  model = mujoco.MjModel.from_xml_string(_MIXED)
  simulation = MetalSimulation(
      model, batch_size=2, profile="contact_free_forces_euler_v1"
  )
  force = torch.full((2, model.nv), 0.05, dtype=torch.float32, device="mps")

  def forbidden(*_args, **_kwargs):
    raise AssertionError("host physics or readback entered native step")

  with monkeypatch.context() as block:
    block.setattr(mujoco, "mj_step", forbidden)
    block.setattr(np.linalg, "solve", forbidden)
    block.setattr(torch.Tensor, "cpu", forbidden)
    for _ in range(10):
      status = simulation.step(qfrc_applied=force)
    assert tuple(status.shape) == (2,)
