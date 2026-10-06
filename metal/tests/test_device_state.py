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

"""CPU lifecycle tests for owned device state and immutable checkpoints."""

import os
from dataclasses import replace

import mujoco
import numpy as np
import pytest

from mujoco_metal.device_state import DeviceState
from mujoco_metal.device_state import StateSnapshot
from mujoco_metal.model import load_model
from mujoco_metal.stepping import validate_stepping_profile


@pytest.fixture(autouse=True)
def _require_torch_state_runtime():
  pytest.importorskip("torch")


def _model(mass="1", gravity="0 0 -9.81", timestep=".001"):
  return mujoco.MjModel.from_xml_string(
      f"""<mujoco><option timestep='{timestep}' gravity='{gravity}'>
      <flag contact='disable'/></option><worldbody>
      <body><joint type='hinge' armature='.1'/>
      <geom type='sphere' size='.1' mass='{mass}'/></body>
      </worldbody></mujoco>"""
  )


def _state(model=None, batch_size=3, **kwargs):
  model = _model() if model is None else model
  profile = validate_stepping_profile(model)
  return DeviceState(model, profile, batch_size, device="cpu", **kwargs)


def test_initial_state_owns_inputs_and_readers_return_detached_copies():
  model = _model()
  initial_qpos = np.zeros((3, model.nq), dtype=np.float32)
  initial_qvel = np.full((3, model.nv), 0.25, dtype=np.float32)
  state = _state(model, qpos=initial_qpos, qvel=initial_qvel)
  initial_qpos[:] = 123
  initial_qvel[:] = 456
  np.testing.assert_array_equal(state.qpos.numpy(), np.zeros((3, model.nq)))
  np.testing.assert_array_equal(state.qvel.numpy(), np.full((3, model.nv), 0.25))

  returned = state.qpos
  returned[0, 0] = 17
  assert not np.array_equal(returned.numpy(), state.qpos.numpy())
  assert state.generation == 0
  assert state.device == "cpu"


def test_selected_reset_clears_only_requested_rows_and_commits_generation():
  model = _model()
  state = _state(model)
  state._qacc.fill_(2)
  state._time.fill_(3)
  state._status.fill_(7)
  old_qpos = state.qpos
  qpos = np.array([[0.4]], dtype=np.float32)
  qvel = np.array([[0.7]], dtype=np.float32)
  assert state.reset([1], qpos=qpos, qvel=qvel) == 1
  qpos[:] = 8
  qvel[:] = 9

  np.testing.assert_allclose(state.qpos[1].numpy(), [0.4])
  np.testing.assert_allclose(state.qvel[1].numpy(), [0.7])
  np.testing.assert_array_equal(state.qpos[[0, 2]].numpy(), old_qpos[[0, 2]].numpy())
  np.testing.assert_array_equal(state.qacc[:, 0].numpy(), [2, 0, 2])
  np.testing.assert_array_equal(state.time.numpy(), [3, 0, 3])
  np.testing.assert_array_equal(state.status.numpy(), [7, 0, 7])
  assert state.reset([]) == 1


def test_masked_reset_cpu_adapter_is_row_scoped_and_empty_mask_is_noop():
  import torch

  state = _state(_model(), batch_size=2)
  state._qpos.copy_(torch.tensor([[0.2], [-0.3]], dtype=torch.float32))
  state._qvel.copy_(torch.tensor([[1.0], [2.0]], dtype=torch.float32))
  state._qacc.copy_(torch.tensor([[3.0], [4.0]], dtype=torch.float32))
  state._time.copy_(torch.tensor([0.1, 0.2], dtype=torch.float32))
  state._row_reset_epoch.copy_(torch.tensor([4, 9], dtype=torch.int32))
  generation = state.generation
  persistent = (state._qpos, state._qvel, state._qacc, state._time,
                state._status, state._row_reset_epoch)
  before = tuple(value.clone() for value in persistent)

  state.reset_mask(torch.tensor([False, False], dtype=torch.bool))
  assert state.generation == generation
  assert all(left is right for left, right in zip(
      persistent, (state._qpos, state._qvel, state._qacc, state._time,
                   state._status, state._row_reset_epoch)))
  assert all(torch.equal(expected, actual)
             for expected, actual in zip(before, persistent))

  state.reset_mask(torch.tensor([True, False], dtype=torch.bool))
  assert state.generation == generation
  assert state._row_reset_epoch.tolist() == [5, 9]
  np.testing.assert_array_equal(
      state._qpos[:, 0].numpy(), np.asarray([0.0, -0.3], dtype=np.float32))
  np.testing.assert_array_equal(
      state._qvel[:, 0].numpy(), np.asarray([0.0, 2.0], dtype=np.float32))
  np.testing.assert_array_equal(
      state._qacc[:, 0].numpy(), np.asarray([0.0, 4.0], dtype=np.float32))
  np.testing.assert_array_equal(
      state._time.numpy(), np.asarray([0.0, 0.2], dtype=np.float32))
  assert all(left is right for left, right in zip(
      persistent, (state._qpos, state._qvel, state._qacc, state._time,
                   state._status, state._row_reset_epoch)))


@pytest.mark.parametrize(
    "env_ids, qpos, qvel",
    [
        ([1, 1], None, None),
        ([3], None, None),
        ([1.0], None, None),
        ([1], np.array([[np.nan]]), None),
        ([1], np.array([[0.0, 1.0]]), None),
        ([1], None, np.array([[np.inf]])),
    ],
)
def test_invalid_reset_is_atomic(env_ids, qpos, qvel):
  state = _state()
  state._qacc.fill_(2)
  before = state.snapshot()
  with pytest.raises(ValueError):
    state.reset(env_ids, qpos=qpos, qvel=qvel)
  after = state.snapshot()
  assert state.generation == 0
  for name in ("qpos", "qvel", "qacc", "time", "status"):
    np.testing.assert_array_equal(getattr(after, name), getattr(before, name))


def test_validate_snapshot_preflights_without_mutating_live_state():
  state = _state(batch_size=2,
      qpos=np.array([[.2], [.4]], dtype=np.float32),
      qvel=np.array([[.7], [.9]], dtype=np.float32))
  before = state.snapshot()

  assert state.validate_snapshot(before)
  bad = replace(before)
  object.__setattr__(bad, "qpos", np.array([[.2], [np.nan]], dtype=np.float64))
  with pytest.raises(ValueError, match="finite"):
    state.validate_snapshot(bad, env_ids=[1])

  after = state.snapshot()
  assert state.generation == 0
  for name in ("qpos", "qvel", "qacc", "time", "status"):
    np.testing.assert_array_equal(getattr(after, name), getattr(before, name))


def test_restore_then_seeded_host_reset_is_reproducible():
  state = _state()
  checkpoint = state.snapshot()
  rng = np.random.default_rng(1234)
  first = np.asarray(rng.uniform(-0.5, 0.5, size=(3, 1)), dtype=np.float32)
  state.reset(np.arange(3), qpos=first)
  expected = state.qpos.numpy().copy()
  state.restore(checkpoint)

  rng = np.random.default_rng(1234)
  repeated = np.asarray(rng.uniform(-0.5, 0.5, size=(3, 1)), dtype=np.float32)
  state.reset(np.arange(3), qpos=repeated)
  np.testing.assert_array_equal(state.qpos.numpy(), expected)


def test_rejects_nonunit_free_joint_quaternion_before_reset_mutation():
  model = mujoco.MjModel.from_xml_string(
      """<mujoco><option><flag contact='disable'/></option><worldbody>
      <body><freejoint/><geom type='sphere' size='.1'/></body>
      </worldbody></mujoco>"""
  )
  state = _state(model, batch_size=2)
  before = state.snapshot()
  bad_qpos = np.zeros((1, model.nq), dtype=np.float32)
  with pytest.raises(ValueError, match="quaternions must be unit"):
    state.reset([1], qpos=bad_qpos)
  after = state.snapshot()
  for name in ("qpos", "qvel", "qacc", "time", "status"):
    np.testing.assert_array_equal(getattr(after, name), getattr(before, name))


def test_snapshot_is_immutable_and_restore_round_trips_every_state_field():
  state = _state()
  state._qpos[0, 0] = 0.8
  state._qvel[:, 0].copy_(state._torch.tensor([0.1, 0.2, 0.3]))
  state._qacc[:, 0].copy_(state._torch.tensor([1, 2, 3]))
  state._time.copy_(state._torch.tensor([0.01, 0.02, 0.03]))
  state._status.copy_(state._torch.tensor([0, 2, 0], dtype=state._torch.int32))
  saved = state.snapshot()
  with pytest.raises(ValueError):
    saved.qpos[0, 0] = 3

  state.reset()
  assert state.generation == 1
  assert state.restore(saved) == 2
  restored = state.snapshot()
  for name in ("qpos", "qvel", "qacc", "time", "status"):
    np.testing.assert_array_equal(getattr(restored, name), getattr(saved, name))


def test_descriptor_input_is_snapshotted_instead_of_borrowed():
  from dataclasses import replace

  model = _model()
  descriptor = load_model(model)
  borrowed_body_pos = np.array(descriptor.body_pos, copy=True)
  mutable_descriptor = replace(descriptor, body_pos=borrowed_body_pos)
  profile = validate_stepping_profile(model)
  state = DeviceState(mutable_descriptor, profile, 2, device="cpu")
  borrowed_body_pos[1, 0] = 99
  assert state._model.body_pos[1, 0] == 0
  with pytest.raises(ValueError):
    state._model.body_pos[1, 0] = 1


def test_rejects_foreign_models_even_when_dimensions_match():
  model = _model()
  profile = validate_stepping_profile(model)
  state = DeviceState(model, profile, 2, device="cpu")
  snapshot = state.snapshot()
  assert (_model(mass="2").nq, _model(mass="2").nv) == (model.nq, model.nv)
  for foreign_model in (_model(mass="2"), _model(gravity="0 0 -4")):
    foreign_profile = validate_stepping_profile(foreign_model)
    foreign = DeviceState(foreign_model, foreign_profile, 2, device="cpu")
    before = foreign.snapshot()
    with pytest.raises(ValueError, match="profile does not match"):
      DeviceState(foreign_model, profile, 2, device="cpu")
    with pytest.raises(ValueError, match="do not match"):
      foreign.restore(snapshot)
    np.testing.assert_array_equal(foreign.snapshot().qpos, before.qpos)

  changed = load_model(model)
  changed_body_mass = np.array(changed.body_mass, copy=True)
  changed_body_mass[1] *= 2
  from dataclasses import replace

  with pytest.raises(ValueError, match="descriptor"):
    DeviceState(replace(changed, body_mass=changed_body_mass), profile, 2, device="cpu")

  other_dt = validate_stepping_profile(model, timestep=0.0005)
  other_dt_state = DeviceState(model, other_dt, 2, device="cpu")
  with pytest.raises(ValueError, match="do not match"):
    other_dt_state.restore(snapshot)


def test_rejects_bad_snapshots_and_accepts_static_empty_world():
  model = mujoco.MjModel.from_xml_string(
      "<mujoco><option><flag contact='disable'/></option><worldbody/></mujoco>"
  )
  state = _state(model, batch_size=2)
  assert state.qpos.shape == (2, 0)
  assert state.qvel.shape == (2, 0)
  saved = state.snapshot()
  assert saved.qpos.shape == (2, 0)
  assert state.restore(saved) == 1

  with pytest.raises(ValueError, match="shape"):
    StateSnapshot(
        model_fingerprint=saved.model_fingerprint,
        profile_fingerprint=saved.profile_fingerprint,
        timestep=saved.timestep,
        nq=0,
        nv=0,
        batch_size=2,
        qpos=np.zeros((3, 0)),
        qvel=np.zeros((2, 0)),
        qacc=np.zeros((2, 0)),
        time=np.zeros(2),
        status=np.zeros(2, dtype=np.int32),
    )


def test_warning_statistics_reset_copy_snapshot_and_legacy_restore():
  torch = pytest.importorskip("torch")
  model = mujoco.MjModel.from_xml_string(
      "<mujoco><option><flag contact='disable'/></option><worldbody>"
      "<body><joint name='j'/><geom type='sphere' size='.1'/></body>"
      "</worldbody></mujoco>"
  )
  state = _state(model, batch_size=2)
  assert state.snapshot().schema_version == 6
  badqpos = int(mujoco.mjtWarning.mjWARN_BADQPOS)
  badqvel = int(mujoco.mjtWarning.mjWARN_BADQVEL)
  state._warning_number[0, badqpos] = 3
  state._warning_lastinfo[0, badqpos] = 7
  state._warning_number[1, badqvel] = 9
  state._warning_lastinfo[1, badqvel] = 2
  state._row_reset_epoch.copy_(torch.tensor([3, 8], dtype=torch.int32))
  snap = state.snapshot()
  np.testing.assert_array_equal(snap.row_reset_epoch, [3, 8])
  np.testing.assert_array_equal(snap.warning_number[0, badqpos], 3)
  np.testing.assert_array_equal(snap.warning_lastinfo[0, badqpos], 7)

  state.reset(env_ids=[0])
  assert int(state._row_reset_epoch[0]) == 4
  assert int(state._warning_number[0, badqpos]) == 0
  assert int(state._warning_number[1, badqvel]) == 9
  state.copy_environment(1, 0)
  assert int(state._row_reset_epoch[0]) == 8
  assert int(state._warning_number[0, badqvel]) == 9
  assert int(state._warning_lastinfo[0, badqvel]) == 2

  state._warning_number.zero_()
  state._warning_lastinfo.fill_(11)
  state.restore(snap)
  np.testing.assert_array_equal(state._row_reset_epoch.numpy(), [3, 8])
  assert int(state._warning_number[0, badqpos]) == 3
  assert int(state._warning_lastinfo[0, badqpos]) == 7
  legacy = replace(snap, schema_version=5)
  state.restore(legacy)
  np.testing.assert_array_equal(state._row_reset_epoch.numpy(), [0, 0])
  assert int(state._warning_number[1, badqvel]) == 9

  # Legacy schemas remain readable and explicitly start with empty warning
  # history because they did not serialize the MuJoCo warning statistics.
  legacy = StateSnapshot(
      model_fingerprint=snap.model_fingerprint,
      profile_fingerprint=snap.profile_fingerprint,
      timestep=snap.timestep,
      nq=snap.nq,
      nv=snap.nv,
      batch_size=snap.batch_size,
      qpos=snap.qpos,
      qvel=snap.qvel,
      qacc=snap.qacc,
      time=snap.time,
      status=snap.status,
      schema_version=1,
  )
  state.restore(legacy)
  assert not bool(torch.any(state._warning_number))
  assert not bool(torch.any(state._warning_lastinfo))


def test_device_state_snapshot_is_not_a_full_trajectory_checkpoint():
  """The narrow state API deliberately omits acceleration warmstart state."""
  state = _state(batch_size=2)
  state._qacc_warmstart.fill_(17)
  snapshot = state.snapshot()
  assert not hasattr(snapshot, "qacc_warmstart")
  assert not hasattr(snapshot, "native_state")
  # Full replay is provided by MetalSimulation.snapshot(), which owns the
  # native-state payload (including qacc_warmstart), not StateSnapshot.


def test_device_masked_reset_resets_selected_rows_without_host_ids():
  torch = pytest.importorskip("torch")
  model = mujoco.MjModel.from_xml_string(
      "<mujoco><option><flag contact='disable'/></option><worldbody>"
      "<body><joint name='j'/><geom type='sphere' size='.1'/></body>"
      "</worldbody></mujoco>")
  state = _state(model, batch_size=2)
  state._qpos.copy_(torch.tensor([[0.3], [0.7]], dtype=torch.float32))
  state._qvel.fill_(2)
  state._qacc.fill_(3)
  state._time.fill_(4)
  state._status.fill_(5)
  state._warning_number.fill_(6)
  state._warning_lastinfo.fill_(7)
  state._qacc_warmstart.fill_(8)
  selected = torch.tensor([True, False], dtype=torch.bool)
  state.reset_mask(selected)
  np.testing.assert_allclose(state._qpos.numpy(), [[0.0], [0.7]], atol=0)
  np.testing.assert_array_equal(state._qvel.numpy(), [[0.0], [2.0]])
  np.testing.assert_array_equal(state._qacc.numpy(), [[0.0], [3.0]])
  np.testing.assert_array_equal(state._time.numpy(), [0.0, 4.0])
  np.testing.assert_array_equal(state._status.numpy(), [0, 5])
  assert not bool(torch.any(state._warning_number[0]))
  assert bool(torch.all(state._warning_number[1] == 6))
  assert bool(torch.all(state._warning_lastinfo[1] == 7))
  assert bool(torch.all(state._qacc_warmstart[0] == 0))
  assert bool(torch.all(state._qacc_warmstart[1] == 8))
  with pytest.raises(ValueError, match=r"bool\[batch_size\]"):
    state.reset_mask(torch.tensor([1, 0], dtype=torch.int32))


@pytest.mark.gpu
@pytest.mark.skipif(
    os.environ.get("MUJOCO_METAL_RUN_GPU") != "1",
    reason="requires explicit MUJOCO_METAL_RUN_GPU=1 and an idle Apple GPU",
)
def test_mps_state_reset_restore_and_model_identity():
  torch = pytest.importorskip("torch")
  if not torch.backends.mps.is_available():
    pytest.skip("PyTorch MPS is unavailable")

  model = _model()
  profile = validate_stepping_profile(model)
  state = DeviceState(model, profile, 2)
  state.reset([1], qpos=np.array([[0.25]], dtype=np.float32))
  checkpoint = state.snapshot()
  state.reset()
  state.restore(checkpoint)
  np.testing.assert_allclose(state.qpos.cpu().numpy()[1], [0.25])

  foreign_model = _model(mass="2")
  with pytest.raises(ValueError, match="profile does not match"):
    DeviceState(foreign_model, profile, 2)
