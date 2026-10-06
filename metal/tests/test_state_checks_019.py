# Copyright 2026 The MuJoCo Metal contributors
# Licensed under the Apache License, Version 2.0.
"""CPU source-contract and device-tensor tests for native check-state APIs."""

from types import SimpleNamespace
import os

import mujoco
import numpy as np
import pytest
torch = pytest.importorskip("torch")

from mujoco_metal.state_checks import mj_checkAcc
from mujoco_metal.state_checks import mj_checkPos
from mujoco_metal.state_checks import mj_checkVel


def _model(disable_autoreset):
  flag = ' contact="disable"' + (
      ' autoreset="disable"' if disable_autoreset else "")
  return mujoco.MjModel.from_xml_string(
      '<mujoco><option><flag' + flag + '/></option><worldbody>'
      '<body><joint name="j"/><geom type="sphere" size=".1"/></body>'
      '</worldbody></mujoco>')


def _fake_sim(model, qpos=None, qvel=None, qacc=None, tree_awake=None):
  b, nq, nv = 2, int(model.nq), int(model.nv)
  state = SimpleNamespace(
      _torch=torch,
      _device=torch.device("cpu"),
      _qpos=torch.zeros((b, nq), dtype=torch.float32),
      _qvel=torch.zeros((b, nv), dtype=torch.float32),
      _qacc=torch.zeros((b, nv), dtype=torch.float32),
      _warning_number=torch.zeros((b, int(mujoco.mjtWarning.mjNWARNING)),
                                  dtype=torch.int32),
      _warning_lastinfo=torch.zeros((b, int(mujoco.mjtWarning.mjNWARNING)),
                                    dtype=torch.int32),
  )
  for tensor, data in ((state._qpos, qpos), (state._qvel, qvel),
                       (state._qacc, qacc)):
    if data is not None:
      tensor.copy_(torch.as_tensor(data, dtype=torch.float32))
  sim = SimpleNamespace(model=model, state=state, _sleep_schedule=None,
                        _sleep_dof_treeid=None, reset_masks=[])
  if tree_awake is not None:
    sim._sleep_schedule = SimpleNamespace(
        tree_awake=torch.as_tensor(tree_awake, dtype=torch.int32))
    sim._sleep_dof_treeid = torch.as_tensor(
        np.asarray(model.dof_treeid, dtype=np.int64), dtype=torch.int64)

  def reset(mask, *, forward_acc=False, warning_commit=None):
    sim.reset_masks.append((mask.clone(), forward_acc))
    mask = mask[:, None]
    state._qpos.copy_(torch.where(mask, torch.as_tensor(
        np.broadcast_to(model.qpos0, tuple(state._qpos.shape)).copy(),
        dtype=torch.float32), state._qpos))
    state._qvel.copy_(torch.where(mask, torch.zeros_like(state._qvel),
                                  state._qvel))
    state._qacc.copy_(torch.where(mask, torch.zeros_like(state._qacc),
                                  state._qacc))
    state._warning_number.copy_(torch.where(
        mask, torch.zeros_like(state._warning_number), state._warning_number))
    state._warning_lastinfo.copy_(torch.where(
        mask, torch.zeros_like(state._warning_lastinfo),
        state._warning_lastinfo))
    if warning_commit is not None:
      warning_commit()
  sim._reset_from_check = reset
  return sim


def _source_warning(model, kind, value, *, dof=None):
  data = mujoco.MjData(model)
  if kind == "qpos":
    data.qpos[dof] = value
    mujoco.mj_checkPos(model, data)
    warning = mujoco.mjtWarning.mjWARN_BADQPOS
  elif kind == "qvel":
    data.qvel[dof] = value
    mujoco.mj_checkVel(model, data)
    warning = mujoco.mjtWarning.mjWARN_BADQVEL
  else:
    data.qacc[dof] = value
    mujoco.mj_checkAcc(model, data)
    warning = mujoco.mjtWarning.mjWARN_BADQACC
  return int(data.warning[warning].number), int(data.warning[warning].lastinfo)


@pytest.mark.parametrize(
    "check,kind,warning",
    [(mj_checkPos, "qpos", mujoco.mjtWarning.mjWARN_BADQPOS),
     (mj_checkVel, "qvel", mujoco.mjtWarning.mjWARN_BADQVEL),
     (mj_checkAcc, "qacc", mujoco.mjtWarning.mjWARN_BADQACC)],
)
def test_check_state_disabled_autoreset_matches_source_first_bad(check, kind,
                                                                 warning):
  model = _model(disable_autoreset=True)
  bad_index = 0
  values = np.zeros((2, int(getattr(model, {"qpos": "nq", "qvel": "nv",
                                             "qacc": "nv"}[kind]))),
                    dtype=np.float32)
  values[0, bad_index] = np.nan
  sim = _fake_sim(
      model,
      qpos=values if kind == "qpos" else None,
      qvel=values if kind == "qvel" else None,
      qacc=values if kind == "qacc" else None)
  result = check(sim)
  expected_number, expected_index = _source_warning(
      model, kind, np.nan, dof=bad_index)
  assert expected_number == 2
  assert expected_index == bad_index
  assert result["bad_world"].tolist() == [True, False]
  assert result["first_bad_index"].tolist() == [bad_index, -1]
  assert result["warning_number"].tolist() == [expected_number, 0]
  assert result["warning_lastinfo"].tolist() == [expected_index, 0]
  assert not sim.reset_masks
  assert int(sim.state._warning_number[0, int(warning)]) == 2


@pytest.mark.parametrize(
    "check,kind,warning",
    [(mj_checkPos, "qpos", mujoco.mjtWarning.mjWARN_BADQPOS),
     (mj_checkVel, "qvel", mujoco.mjtWarning.mjWARN_BADQVEL),
     (mj_checkAcc, "qacc", mujoco.mjtWarning.mjWARN_BADQACC)],
)
def test_check_state_autoreset_matches_source_and_masks_worlds(check, kind,
                                                                warning):
  model = _model(disable_autoreset=False)
  bad_index = 0
  width = int(getattr(model, {"qpos": "nq", "qvel": "nv",
                              "qacc": "nv"}[kind]))
  values = np.zeros((2, width), dtype=np.float32)
  values[0, bad_index] = np.float32(1.0e11)
  sim = _fake_sim(
      model,
      qpos=values if kind == "qpos" else None,
      qvel=values if kind == "qvel" else None,
      qacc=values if kind == "qacc" else None)
  if kind == "qpos":
    original = sim.state._qpos[1].clone()
  elif kind == "qvel":
    original = sim.state._qvel[1].clone()
  else:
    original = sim.state._qacc[1].clone()
  result = check(sim)
  expected_number, expected_index = _source_warning(
      model, kind, 1.0e11, dof=bad_index)
  assert expected_number == 1
  assert expected_index == bad_index
  assert result["warning_number"].tolist() == [1, 0]
  assert result["warning_lastinfo"].tolist() == [bad_index, 0]
  assert len(sim.reset_masks) == 1
  assert sim.reset_masks[0][0].tolist() == [True, False]
  assert sim.reset_masks[0][1] == (kind == "qacc")
  assert torch.equal(original, getattr(
      sim.state, {"qpos": "_qpos", "qvel": "_qvel", "qacc": "_qacc"}[kind])[1])
  assert int(sim.state._warning_number[0, int(warning)]) == 1


def test_velocity_check_uses_awake_dof_membership_and_empty_shapes():
  model = mujoco.MjModel.from_xml_string(
      '<mujoco><option><flag autoreset="disable"/></option><worldbody>'
      '<body><joint name="j0"/><geom type="sphere" size=".1"/></body>'
      '<body><joint name="j1"/><geom type="sphere" size=".1"/></body>'
      '</worldbody></mujoco>')
  assert model.nv == 2
  qvel = np.array([[np.nan, 1e11], [np.inf, 0]], dtype=np.float32)
  # The source awake list contains DOF 1 only in world zero and none in world
  # one; the bad DOF 0 is ignored while the bad DOF 1 is reported.
  sim = _fake_sim(model, qvel=qvel, tree_awake=[[0, 1], [0, 0]])
  result = mj_checkVel(sim)
  assert result["bad_world"].tolist() == [True, False]
  assert result["first_bad_index"].tolist() == [1, -1]

  empty = mujoco.MjModel.from_xml_string(
      '<mujoco><option><flag autoreset="disable"/></option><worldbody/></mujoco>')
  empty_sim = _fake_sim(empty)
  result = mj_checkAcc(empty_sim)
  assert result["bad_world"].tolist() == [False, False]
  assert result["first_bad_index"].tolist() == [-1, -1]


@pytest.mark.parametrize(
    "check,field,warning",
    [(mj_checkPos, "_qpos", mujoco.mjtWarning.mjWARN_BADQPOS),
     (mj_checkVel, "_qvel", mujoco.mjtWarning.mjWARN_BADQVEL),
     (mj_checkAcc, "_qacc", mujoco.mjtWarning.mjWARN_BADQACC)],
)
def test_check_state_uses_strict_pinned_max_threshold(check, field, warning):
  model = _model(disable_autoreset=True)
  at_limit = _fake_sim(model)
  getattr(at_limit.state, field)[0, 0].fill_(_MAXVAL_FOR_TEST)
  result = check(at_limit)
  assert result["bad_world"].tolist() == [False, False]
  assert at_limit.state._warning_number[:, int(warning)].tolist() == [0, 0]

  over_limit = _fake_sim(model)
  getattr(over_limit.state, field)[0, 0].fill_(float(np.nextafter(
      np.float32(_MAXVAL_FOR_TEST), np.float32(np.inf))))
  result = check(over_limit)
  assert result["bad_world"].tolist() == [True, False]
  assert result["first_bad_index"].tolist() == [0, -1]


_MAXVAL_FOR_TEST = 1.0e10


def test_autoreset_uses_real_device_state_masked_reset():
  from mujoco_metal.device_state import DeviceState
  from mujoco_metal.simulation import MetalSimulation
  from mujoco_metal.stepping import validate_stepping_profile

  model = _model(disable_autoreset=False)
  state = DeviceState(model, validate_stepping_profile(model), 2, device="cpu")
  state._qpos.copy_(torch.tensor([[float("nan")], [0.25]]))
  state._qvel.copy_(torch.tensor([[4.0], [2.0]]))
  sim = SimpleNamespace(model=model, state=state, _state=state, batch_size=2,
                        _native_plugins=(), _coupled_constraints=None,
                        _delay=None, _flex=None, _sleep_schedule=None,
                        _sensordata=None, _raw_sensordata=None,
                        _assembled_system_valid=True, _accepted_step="old",
                        _last_coupled="old", _last_coupled_generation=9,
                        _step1_record="old")
  sim._reset_from_check = lambda mask, **kw: MetalSimulation._reset_from_check(
      sim, mask, **kw)
  result = mj_checkPos(sim)
  assert result["bad_world"].tolist() == [True, False]
  assert torch.equal(state._qpos, torch.tensor([[0.0], [0.25]]))
  assert torch.equal(state._qvel, torch.tensor([[0.0], [2.0]]))
  warning = int(mujoco.mjtWarning.mjWARN_BADQPOS)
  assert state._warning_number[:, warning].tolist() == [1, 0]
  assert state._warning_lastinfo[:, warning].tolist() == [0, 0]
  assert sim._assembled_system_valid is True
  assert sim._accepted_step == "old"
  assert sim._last_coupled == "old"
  assert sim._step1_record == "old"


def test_bad_acceleration_autoreset_runs_owned_forward_and_preserves_neighbor():
  from mujoco_metal.device_state import DeviceState
  from mujoco_metal.simulation import MetalSimulation
  from mujoco_metal.stepping import validate_stepping_profile

  model = _model(disable_autoreset=False)
  state = DeviceState(model, validate_stepping_profile(model), 2, device="cpu")
  state._qpos.copy_(torch.tensor([[0.4], [0.2]]))
  state._qvel.copy_(torch.tensor([[3.0], [1.0]]))
  state._qacc.copy_(torch.tensor([[1.0e11], [0.25]]))
  calls = []
  sim = SimpleNamespace(
      model=model, state=state, _state=state, batch_size=2,
      _native_plugins=(), _coupled_constraints=None, _delay=None,
      _flex=None, _sleep_schedule=None, _sensordata=None,
      _raw_sensordata=None, _assembled_system_valid=True, _accepted_step=None,
      _last_coupled=None, _last_coupled_generation=None, _step1_record=None)
  sim._reset_from_check = lambda mask, **kw: MetalSimulation._reset_from_check(
      sim, mask, **kw)
  sim._acceleration = lambda qpos, qvel, **kw: (
      calls.append((qpos.clone(), qvel.clone(), kw)) or
      (torch.full_like(qvel, 0.5), torch.zeros((2,), dtype=torch.int32), {}))
  sim._update_energy_position = lambda *args, **kwargs: None
  sim._update_energy_velocity = lambda *args, **kwargs: None
  sim._state._on_masked_reset = None
  result = mj_checkAcc(sim)
  assert result["bad_world"].tolist() == [True, False]
  assert len(calls) == 1
  assert torch.equal(calls[0][0], torch.tensor([[0.0], [0.2]]))
  assert torch.equal(calls[0][1], torch.tensor([[0.0], [1.0]]))
  assert torch.equal(state._qacc, torch.tensor([[0.5], [0.25]]))
  assert torch.equal(state._qpos, torch.tensor([[0.0], [0.2]]))
  assert torch.equal(state._qvel, torch.tensor([[0.0], [1.0]]))


def test_sleep_schedule_masked_reset_preserves_healthy_rows():
  from mujoco_metal.sleep_schedule import DeviceSleepScheduler

  batch = 2
  initial = torch.tensor([[-11, 0]], dtype=torch.int32)
  scheduler = SimpleNamespace(
      batch_size=batch, device=torch.device("cpu"),
      _initial_tree_state=initial,
      tree_state=torch.tensor([[0, -8], [0, -9]], dtype=torch.int32),
      tree_awake=torch.tensor([[0, 1], [0, 1]], dtype=torch.int32),
      links=torch.tensor([[[3, 4], [5, 6]], [[7, 8], [9, 10]]],
                         dtype=torch.int32),
      constraints_active=torch.tensor([1, 2], dtype=torch.int32),
      links_overflow=torch.tensor([1, 1], dtype=torch.int32),
      status=torch.tensor([3, 1], dtype=torch.int32),
      eq_active_default=torch.tensor([[1, 0], [0, 1]], dtype=torch.int32),
      eq_active=torch.zeros((batch, 2), dtype=torch.int32),
      _zero_qfrc=torch.ones((batch, 2), dtype=torch.float32),
      _zero_xfrc=torch.ones((batch, 2, 6), dtype=torch.float32),
      build_calls=0)
  scheduler._build_awake_lists = lambda: setattr(
      scheduler, "build_calls", scheduler.build_calls + 1)
  DeviceSleepScheduler.reset_masked(
      scheduler, torch.tensor([True, False], dtype=torch.bool))
  assert scheduler.tree_state.tolist() == [[-11, 0], [0, -9]]
  assert scheduler.tree_awake.tolist() == [[1, 0], [0, 1]]
  assert scheduler.links[0].tolist() == [[-1, -1], [-1, -1]]
  assert scheduler.links[1].tolist() == [[7, 8], [9, 10]]
  assert scheduler.constraints_active.tolist() == [0, 2]
  assert scheduler.links_overflow.tolist() == [0, 1]
  assert scheduler.status.tolist() == [0, 1]
  assert scheduler.eq_active.tolist() == [[1, 0], [0, 0]]
  assert not bool(torch.any(scheduler._zero_xfrc[0]))
  assert bool(torch.all(scheduler._zero_xfrc[1] == 1))
  assert scheduler.build_calls == 1


def test_stateful_plugin_masked_reset_is_device_only_and_transactional():
  from mujoco_metal.extensions import NativePlugin
  from mujoco_metal.extensions import PluginType
  from mujoco_metal.simulation import _reset_plugins_device_masked

  class ResetPlugin(NativePlugin):
    def __init__(self, name, fail=False):
      super().__init__(name, PluginType.FORCE)
      self.fail = fail

    def init(self, model, batch_size=1, device=None):
      super().init(model, batch_size, device)
      self.value = torch.arange(1, batch_size + 1, dtype=torch.float32,
                                device=self.device)

    def snapshot(self):
      return self.value.detach().cpu().numpy().copy()

    def device_snapshot(self):
      return self.value.clone()

    def device_snapshot_bytes(self):
      return self.value.numel() * self.value.element_size()

    def restore_device(self, snap):
      self.value.copy_(snap)

    def restore_masked(self, snap, accepted_mask):
      self.value.copy_(torch.where(accepted_mask, self.value, snap))

    def reset_masked(self, reset_mask):
      self.value.copy_(torch.where(reset_mask, torch.zeros_like(self.value),
                                   self.value))
      if self.fail:
        raise RuntimeError("masked reset failed after mutation")

  model = _model(disable_autoreset=False)
  first = ResetPlugin("first")
  second = ResetPlugin("second", fail=True)
  for plugin in (first, second):
    plugin.init(model, 2, device=torch.device("cpu"))
  mask = torch.tensor([True, False], dtype=torch.bool)
  with pytest.raises(RuntimeError, match="masked reset failed"):
    _reset_plugins_device_masked((first, second), mask)
  assert first.value.tolist() == [1.0, 2.0]
  assert second.value.tolist() == [1.0, 2.0]
  second.fail = False
  _reset_plugins_device_masked((first, second), mask)
  assert first.value.tolist() == [0.0, 2.0]
  assert second.value.tolist() == [0.0, 2.0]


@pytest.mark.gpu
@pytest.mark.skipif(os.getenv("MUJOCO_METAL_RUN_GPU") != "1",
                    reason="opt-in native MPS integration")
@pytest.mark.parametrize("kind", ["qpos", "qvel", "qacc"])
def test_native_mps_check_state_and_integrated_step_autorecovery(kind):
  from mujoco_metal.simulation import MetalSimulation

  model = _model(disable_autoreset=False)
  sim = MetalSimulation(model, batch_size=2, profile="integrated_euler_v1")
  tensor_name = {"qpos": "_qpos", "qvel": "_qvel", "qacc": "_qacc"}[kind]
  target = getattr(sim.state, tensor_name)
  owned_device = target.device
  target[0, 0] = float("nan") if kind == "qpos" else 1.0e11
  check = {"qpos": mj_checkPos, "qvel": mj_checkVel,
           "qacc": mj_checkAcc}[kind]
  warning = int({"qpos": mujoco.mjtWarning.mjWARN_BADQPOS,
                 "qvel": mujoco.mjtWarning.mjWARN_BADQVEL,
                 "qacc": mujoco.mjtWarning.mjWARN_BADQACC}[kind])
  if kind == "qpos":
    # The production step runs checkPos/checkVel before its forward pipeline.
    sim.step()
  else:
    check(sim)
  assert sim.state._warning_number[:, warning].detach().cpu().tolist() == [1, 0]
  assert sim.state._warning_lastinfo[:, warning].detach().cpu().tolist() == [0, 0]
  assert bool(torch.isfinite(sim.state._qpos).all())
  assert bool(torch.isfinite(sim.state._qvel).all())
  assert bool(torch.isfinite(sim.state._qacc).all())
  for state_tensor in (sim.state._qpos, sim.state._qvel, sim.state._qacc,
                       sim.state._warning_number,
                       sim.state._warning_lastinfo):
    assert state_tensor.device == owned_device
  # Full checkpoints and row copies retain the source warning statistics.
  checkpoint = sim.snapshot()
  sim.reset(env_ids=[0])
  assert int(sim.state._warning_number[0, warning]) == 0
  sim.restore(checkpoint)
  assert sim.state._warning_number[:, warning].detach().cpu().tolist() == [1, 0]
  sim.copy_environment(0, 1)
  assert sim.state._warning_number[:, warning].detach().cpu().tolist() == [1, 1]


@pytest.mark.gpu
@pytest.mark.skipif(os.getenv("MUJOCO_METAL_RUN_GPU") != "1",
                    reason="opt-in native MPS integration")
def test_native_all_healthy_check_preserves_cached_owners_and_samples():
  from mujoco_metal.simulation import MetalSimulation

  model = mujoco.MjModel.from_xml_string(
      '<mujoco><option><flag contact="disable"/></option><worldbody>'
      '<body><joint name="j"/><geom type="sphere" size=".1"/>'
      '<site name="sensor_site"/></body>'
      '</worldbody><sensor><accelerometer site="sensor_site"/></sensor></mujoco>')
  sim = MetalSimulation(model, batch_size=2, profile="integrated_euler_v1")
  control = MetalSimulation(model, batch_size=2, profile="integrated_euler_v1")
  _prepare_native_chain(sim)
  _prepare_native_chain(control)
  record = sim._forward_stages._record
  record_valid = record.row_valid
  owners = {
      "qacc": sim.state._qacc,
      "rhs": sim._rhs,
      "stored_sensor": sim._sensordata,
      "raw_sensor": sim._raw_sensordata,
      "energy": sim._energy,
      "fk_valid": sim._smooth._fk._workspace["outputs"]["cache_valid"],
      "warning_number": sim.state._warning_number,
  }
  contents = {name: value.clone() for name, value in owners.items()
              if value is not None}
  versions = {name: value._version for name, value in owners.items()
              if value is not None}
  warning = int(mujoco.mjtWarning.mjWARN_BADQACC)
  result = mj_checkAcc(sim)
  assert result["bad_world"].detach().cpu().tolist() == [False, False]
  assert sim._forward_stages._record is record
  assert record.row_valid is record_valid
  assert record.row_valid.detach().cpu().tolist() == [True, True]
  assert sim.state._warning_number[:, warning].detach().cpu().tolist() == [0, 0]
  live = {
      "qacc": sim.state._qacc,
      "rhs": sim._rhs,
      "stored_sensor": sim._sensordata,
      "raw_sensor": sim._raw_sensordata,
      "energy": sim._energy,
      "fk_valid": sim._smooth._fk._workspace["outputs"]["cache_valid"],
      "warning_number": sim.state._warning_number,
  }
  for name, value in owners.items():
    if value is None:
      continue
    assert live[name] is value
    assert value._version == versions[name]
    assert torch.equal(value, contents[name])

  # An all-healthy check must leave the subsequent real step bitwise equal
  # to a control that skipped the check, not merely preserve selected buffers.
  sim.step()
  control.step()
  for name in ("_qpos", "_qvel", "_qacc", "_time", "_status"):
    assert torch.equal(getattr(sim.state, name), getattr(control.state, name))
  assert torch.equal(sim._sensordata, control._sensordata)


@pytest.mark.gpu
@pytest.mark.skipif(os.getenv("MUJOCO_METAL_RUN_GPU") != "1",
                    reason="opt-in native MPS integration")
def test_native_mixed_autoreset_masks_flex_recovery_rows():
  """Bad qacc recovery recomputes flex only for the selected world."""
  from mujoco_metal.simulation import MetalSimulation

  model = mujoco.MjModel.from_xml_string("""
    <mujoco><option timestep=".002" gravity="0 0 -9.81">
      <flag contact="disable"/>
    </option><worldbody>
      <flexcomp name="cloth" type="grid" count="2 2 1"
                spacing=".1 .1 .1" mass="1" radius=".01" dim="2">
        <contact contype="0" conaffinity="0"/>
        <elasticity young="100" poisson=".2" damping=".2"
                    thickness=".01" elastic2d="stretch"/>
      </flexcomp>
    </worldbody></mujoco>
  """)
  sim = MetalSimulation(model, batch_size=2, profile="integrated_euler_v1")
  sim.step()
  flex = sim._flex
  assert flex is not None
  healthy = {
      "qpos": sim.state._qpos[1].clone(),
      "qvel": sim.state._qvel[1].clone(),
      "qacc": sim.state._qacc[1].clone(),
      "flex_force": flex._qfrc_passive[1].clone(),
      "flex_tangent": flex._stiffness_tangent[1].clone(),
      "rhs": sim._rhs[1].clone(),
      "sensor": (None if sim._sensordata is None
                 else sim._sensordata[1].clone()),
  }
  bad_qacc = sim.state._qacc.clone()
  bad_qacc[0, 0] = float("inf")
  result = mj_checkAcc(sim, qacc=bad_qacc)
  assert result["bad_world"].detach().cpu().tolist() == [True, False]
  assert torch.equal(sim.state._qpos[1], healthy["qpos"])
  assert torch.equal(sim.state._qvel[1], healthy["qvel"])
  assert torch.equal(sim.state._qacc[1], healthy["qacc"])
  assert torch.equal(flex._qfrc_passive[1], healthy["flex_force"])
  assert torch.equal(flex._stiffness_tangent[1], healthy["flex_tangent"])
  assert torch.equal(sim._rhs[1], healthy["rhs"])
  if healthy["sensor"] is not None:
    assert torch.equal(sim._sensordata[1], healthy["sensor"])


def test_masked_auxiliary_clear_preserves_unselected_rows_cpu_adapter():
  from mujoco_metal.device_state import DeviceState
  from mujoco_metal.stepping import validate_stepping_profile

  model = _model(disable_autoreset=False)
  state = DeviceState(model, validate_stepping_profile(model), 2, device="cpu")
  floating = torch.arange(12, dtype=torch.float32).reshape(2, 2, 3)
  integer = torch.arange(8, dtype=torch.int32).reshape(2, 4)
  float_healthy, int_healthy = floating[1].clone(), integer[1].clone()
  mask = torch.tensor([True, False])
  state.clear_masked_rows(floating, mask)
  state.clear_masked_rows(integer, mask)
  inverted = torch.arange(6, dtype=torch.float32).reshape(2, 3)
  inverted_before = inverted[0].clone()
  state.clear_masked_rows(inverted, mask, invert=True)
  assert torch.equal(inverted[0], inverted_before)
  assert torch.count_nonzero(inverted[1]) == 0
  target = torch.zeros((2, 3), dtype=torch.float32)
  source = torch.ones_like(target)
  state.update_masked_rows(target, source, mask, add=True)
  state.update_masked_rows(target, source, mask, add=False, sign=-1)
  assert torch.equal(target[0], torch.full((3,), -1.0))
  assert torch.equal(target[1], torch.zeros(3))
  assert torch.count_nonzero(integer[0]) == 0
  assert torch.equal(floating[1], float_healthy)
  assert torch.equal(integer[1], int_healthy)


def test_masked_scaled_accumulator_preserves_unselected_rows_cpu_adapter():
  from mujoco_metal.device_state import DeviceState
  from mujoco_metal.stepping import validate_stepping_profile

  model = _model(disable_autoreset=False)
  state = DeviceState(model, validate_stepping_profile(model), 2, device="cpu")
  target = torch.tensor([[10.0], [20.0]])
  source = torch.tensor([[2.0], [3.0]])
  scale = torch.tensor([.25])
  state.add_masked_scaled_rows(
      target, source, scale, torch.tensor([True, False]), sign=-1)
  assert target.tolist() == [[9.5], [20.0]]


def test_masked_packed_row_copy_preserves_unselected_world_and_rows():
  from mujoco_metal.device_state import DeviceState
  from mujoco_metal.stepping import validate_stepping_profile

  model = _model(disable_autoreset=False)
  state = DeviceState(model, validate_stepping_profile(model), 2, device="cpu")
  target = torch.arange(30, dtype=torch.float32).reshape(2, 5, 3)
  source = torch.tensor([[[101., 102., 103.], [104., 105., 106.]],
                         [[201., 202., 203.], [204., 205., 206.]]])
  before = target.clone()
  state.copy_masked_packed_rows(
      target, source, torch.tensor([True, False]), row_offset=2)
  assert torch.equal(target[0, :2], before[0, :2])
  assert target[0, 2:4].tolist() == source[0].tolist()
  assert torch.equal(target[0, 4], before[0, 4])
  assert torch.equal(target[1], before[1])


def test_state_reset_clear_kernel_supports_guarded_inverse_selector():
  from pathlib import Path

  source = (Path(__file__).parents[1] / "mujoco_metal" / "shaders" /
            "state_reset.metal").read_text()
  for entry in ("kernel void clear_float_rows(",
                "kernel void clear_int_rows("):
    body = source[source.index(entry):]
    assert "if (dims[2]!=0) selected=!selected;" in body
    assert body.index("if (world>=uint(batch)) return;") < body.index(
        "if (!selected) return;")
    assert body.index("if (!selected) return;") < body.index("values[")


def test_invalid_stage_zeroing_uses_row_selector_inversion_cpu_adapter():
  from mujoco_metal.device_state import DeviceState
  from mujoco_metal.simulation import _zero_invalid_rows
  from mujoco_metal.stepping import validate_stepping_profile

  model = _model(disable_autoreset=False)
  state = DeviceState(model, validate_stepping_profile(model), 2, device="cpu")
  value = torch.arange(6, dtype=torch.float32).reshape(2, 3)
  valid = torch.tensor([True, False])
  healthy = value[0].clone()
  _zero_invalid_rows(value, valid, state)
  assert torch.equal(value[0], healthy)
  assert torch.count_nonzero(value[1]) == 0


def test_legacy_constraint_msl_world_guards_precede_row_outputs():
  from pathlib import Path

  root = Path(__file__).parents[1] / "mujoco_metal" / "shaders"
  contact = (root / "contact.metal").read_text()
  joint = (root / "joint_constraints.metal").read_text()
  cases = (
      (contact, "kernel void contact_normal(",
       "if (dims[6 + world] == 0) return;", "int a=geom1[slot]"),
      (contact, "kernel void solve_normal_contacts(",
       "if (dims[6 + int(world)] == 0) return;", "status[world]=0;"),
      (joint, "kernel void solve_joint_constraints(",
       "if (dims[11 + int(world)] == 0) return;", "out_status[world]=0;"),
  )
  for source, entry, guard, first_write in cases:
    body = source[source.index(entry):]
    assert body.index(guard) < body.index(first_write)


def test_recovery_mask_guards_tendon_and_cable_producers_before_row_reads():
  from pathlib import Path

  root = Path(__file__).parents[1] / "mujoco_metal" / "shaders"
  tendon = (root / "tendons.metal").read_text()
  cable = (root / "bundled_cable.metal").read_text()
  checks = (
      (tendon, "kernel void fixed_tendon_dynamics(",
       "if (dims[8 + int(world)] == 0) return;", "uint qbase=world"),
      (tendon, "kernel void spatial_tendon_kinematics(",
       "if (dims[8 + int(world)] == 0) return;", "uint vbase=uint(world)"),
      (tendon, "kernel void spatial_tendon_velocity(",
       "dims[8 + int(world)]==0) return;", "uint vbase=world"),
      (tendon, "kernel void spatial_tendon_actuator_state(",
       "dims[5+int(world)]==0) return;", "int tbase=int(world)"),
      (tendon, "kernel void spatial_tendon_forces(",
       "dims[8 + int(world)]==0) return;", "uint vb=world"),
      (tendon, "kernel void spatial_armature_dots(",
       "dims[8 + int(world)]==0) return;", "uint vbase=uint(world)"),
      (tendon, "kernel void spatial_armature_bias(",
       "dims[8 + int(world)]==0) return;", "uint vb=world"),
      (cable, "kernel void bundled_cable_force(",
       "if (dims[5 + world] == 0) return;", "uint qbase"),
  )
  for source, entry, guard_fragment, first_access in checks:
    start = source.index(entry)
    body = source[start:]
    guard = body.index(guard_fragment)
    assert guard < body.index(first_access)


def test_recovery_mask_guards_position_energy_before_physics_reads():
  from pathlib import Path

  source = (Path(__file__).parents[1] / "mujoco_metal" / "shaders" /
            "smooth_mass.metal").read_text()
  start = source.index("kernel void masked_energy_position(")
  body = source[start:]
  assert body.index("if (int(world)>=batch) return;") < body.index(
      "if (dims[12+int(world)]==0) return;")
  guard = body.index("if (dims[12+int(world)]==0) return;")
  assert guard < body.index("float value=0.0f;")
  assert guard < body.index("inertial_pos[")


def test_recovery_mask_guards_typed_actuator_projection_before_force_reads():
  from pathlib import Path

  source = (Path(__file__).parents[1] / "mujoco_metal" / "shaders" /
            "actuation.metal").read_text()
  start = source.index("kernel void actuator_plugin_project_qfrc(")
  body = source[start:]
  guard = body.index("if (int(world)>=batch || dims[3+int(world)]==0) return;")
  assert guard < body.index("int fbase=int(world)*nu;")
  assert guard < body.index("qfrc[int(world)*qwidth+dof]=value;")


@pytest.mark.parametrize("autoreset,expected", [(True, [1, 7]),
                                                  (False, [2, 7])])
def test_warning_commit_updates_selected_rows_only_cpu_adapter(autoreset,
                                                               expected):
  from mujoco_metal.device_state import DeviceState
  from mujoco_metal.stepping import validate_stepping_profile

  model = _model(disable_autoreset=not autoreset)
  state = DeviceState(model, validate_stepping_profile(model), 2, device="cpu")
  warning = int(mujoco.mjtWarning.mjWARN_BADQPOS)
  state._warning_number[1, warning] = 7
  state._warning_lastinfo[1, warning] = 19
  healthy_number = state._warning_number[1].clone()
  healthy_info = state._warning_lastinfo[1].clone()
  state.record_warning_rows(torch.tensor([True, False]),
                            torch.tensor([3, -1], dtype=torch.int32),
                            warning, autoreset=autoreset)
  assert state._warning_number[:, warning].tolist() == expected
  assert state._warning_lastinfo[0, warning].item() == 3
  assert torch.equal(state._warning_number[1], healthy_number)
  assert torch.equal(state._warning_lastinfo[1], healthy_info)


def test_masked_strided_row_copy_preserves_other_rows_cpu_adapter():
  from mujoco_metal.device_state import DeviceState
  from mujoco_metal.stepping import validate_stepping_profile

  model = _model(disable_autoreset=True)
  state = DeviceState(model, validate_stepping_profile(model), 2, device="cpu")
  target = torch.arange(24, dtype=torch.float32).reshape(2, 12)
  source = torch.tensor([[101.0, 102.0], [201.0, 202.0]])
  original = target.clone()
  state.copy_masked_strided_rows(target, source,
                                 torch.tensor([True, False]), offset=4)
  assert target[0, 4:6].tolist() == [101.0, 102.0]
  assert torch.equal(target[0, :4], original[0, :4])
  assert torch.equal(target[0, 6:], original[0, 6:])
  assert torch.equal(target[1], original[1])


def test_masked_update_and_copy_republish_shared_selector_cpu_adapter():
  """Every row kernel gets the mask for its own call, not stale shared data."""
  from mujoco_metal.device_state import DeviceState
  from mujoco_metal.stepping import validate_stepping_profile

  model = _model(disable_autoreset=True)
  state = DeviceState(model, validate_stepping_profile(model), 2,
                      device="cpu")
  # The CPU adapter does not allocate MPS dispatch metadata; supply the exact
  # fixed native ABI buffers so fake kernels can verify each call's selector.
  state._reset_mask_i32 = torch.tensor([1, 0], dtype=torch.int32)
  state._row_operation_mask_i32 = torch.zeros((2,), dtype=torch.int32)
  state._clear_rows_dims = torch.zeros((3,), dtype=torch.int32)
  state._update_rows_dims = torch.zeros((4,), dtype=torch.int32)
  observed = []

  def selected(mask, batch):
    observed.append(tuple(mask.tolist()))
    return mask.to(dtype=torch.bool).reshape(batch, 1)

  def clear(mask, value, dims, *, threads, group_size):
    del threads, group_size
    batch, width, invert = map(int, dims.tolist())
    chosen = selected(mask, batch)
    if invert:
      chosen = ~chosen
    value.reshape(batch, width)[chosen.expand(batch, width)] = 0

  def update(mask, target, source, dims, *, threads, group_size):
    del threads, group_size
    batch, width, mode, sign = map(int, dims.tolist())
    chosen = selected(mask, batch).expand(batch, width)
    dst, src = target.reshape(batch, width), source.reshape(batch, width)
    if mode == 0:
      dst[chosen] += sign * src[chosen]
    else:
      dst[chosen] = sign * src[chosen]

  def copy_int(mask, target, source, dims, *, threads, group_size):
    del threads, group_size
    batch, width = map(int, dims[:2].tolist())
    chosen = selected(mask, batch).expand(batch, width)
    target.reshape(batch, width)[chosen] = source.reshape(batch, width)[chosen]

  state._clear_rows_shader = clear
  state._update_float_rows_shader = update
  state._copy_int_rows_shader = copy_int

  value = torch.tensor([[1., 2.], [3., 4.]])
  state.clear_masked_rows(value, torch.tensor([False, True]))
  assert state._row_operation_mask_i32.tolist() == [0, 1]
  assert state._reset_mask_i32.tolist() == [1, 0]
  target = torch.zeros((2, 2))
  source = torch.tensor([[5., 6.], [7., 8.]])
  state.update_masked_rows(target, source, torch.tensor([True, False]),
                           add=False)
  assert state._row_operation_mask_i32.tolist() == [1, 0]
  assert state._reset_mask_i32.tolist() == [1, 0]
  assert target.tolist() == [[5., 6.], [0., 0.]]

  # An all-false call must also clear a previous true selector.
  prior = target.clone()
  state.update_masked_rows(target, source, torch.tensor([False, False]))
  assert state._row_operation_mask_i32.tolist() == [0, 0]
  assert state._reset_mask_i32.tolist() == [1, 0]
  assert torch.equal(target, prior)

  # Prime the opposite selection before an int copy to catch the same bug on
  # its separate native kernel path.
  state.clear_masked_rows(value, torch.tensor([True, False]))
  dst = torch.zeros((2, 1), dtype=torch.int32)
  src = torch.tensor([[11], [22]], dtype=torch.int32)
  state.copy_masked_rows(dst, src, torch.tensor([False, True]))
  assert state._row_operation_mask_i32.tolist() == [0, 1]
  assert state._reset_mask_i32.tolist() == [1, 0]
  assert dst.tolist() == [[0], [22]]


@pytest.mark.gpu
@pytest.mark.skipif(os.getenv("MUJOCO_METAL_RUN_GPU") != "1",
                    reason="opt-in mixed-world native checkAcc recovery")
def test_allhealthy_and_mixed_badqacc_preserve_stage_sensor_and_force_rows():
  from pathlib import Path
  from mujoco_metal.extensions import (
      CustomMagneticForcePlugin, PluginType, default_registry)
  from mujoco_metal.site_feedback import register_site_feedback
  from mujoco_metal.simulation import MetalSimulation

  model = mujoco.MjModel.from_xml_path(
      str(Path(__file__).parents[1] / "examples" / "haptic_calligraphy.xml"))
  qpos = np.array([[.2, -.4], [-.25, .48]], dtype=np.float32)
  qvel = np.array([[.05, -.03], [-.02, .06]], dtype=np.float32)
  default_registry.register(CustomMagneticForcePlugin(
      name="checkacc_magnetic_mask", charge=.3, b_field=(.2, -.4, .8)))
  register_site_feedback(
      "checkacc_feedback_mask", site="pen_tip", kp=1.1, kd=.4,
      center=(.8, 0., .9), amplitude=(.04, 0., .03),
      frequency=(.7, 0., 1.2), phase=(0., 0., .5))
  try:
    sim = MetalSimulation(model, batch_size=2, qpos=qpos, qvel=qvel,
                          profile="integrated_euler_v1")
    control = MetalSimulation(model, batch_size=2, qpos=qpos, qvel=qvel,
                              profile="integrated_euler_v1")
    record = sim.prepare_forward_position()
    velocity = sim.prepare_forward_velocity(record)
    # Compare stage-owned outputs only after giving the control the same
    # explicit POS/VEL lifecycle. A cold ordinary step need not populate the
    # cached-stage scratch that this test deliberately inspects.
    control_record = control.prepare_forward_position()
    control.prepare_forward_velocity(control_record)
    assert velocity["status"].shape == (2,)
    plugin_rows = [p._queries._plugin_force[1].clone()
                   for p in sim._force_plugins]
    sensor_row = sim._sensordata[1].clone()
    passive_row = sim._forward_stage_passive_force[1].clone()
    assert bool(torch.count_nonzero(passive_row)), passive_row
    torch.testing.assert_close(control._forward_stage_passive_force[1],
                                passive_row, rtol=0, atol=0)
    state_qacc = sim.state._qacc
    generation = sim.state.generation
    stage_owner = sim._forward_stages._record
    sensor_qacc = getattr(sim, "_last_sensor_qacc", None)
    sensor_qacc_low = getattr(sim, "_last_sensor_qacc_low", None)

    healthy = mj_checkAcc(sim, qacc=sim.state._qacc)
    assert healthy["bad_world"].cpu().tolist() == [False, False]
    assert sim.state.generation == generation
    assert sim._forward_stages._record is stage_owner
    assert stage_owner.row_valid.cpu().tolist() == [True, True]
    assert sim.state._qacc is state_qacc
    assert getattr(sim, "_last_sensor_qacc", None) is sensor_qacc
    assert getattr(sim, "_last_sensor_qacc_low", None) is sensor_qacc_low
    torch.testing.assert_close(sim._sensordata[1], sensor_row, rtol=0, atol=0)
    torch.testing.assert_close(
        sim._forward_stage_passive_force[1], passive_row, rtol=0, atol=0)
    for plugin, before in zip(sim._force_plugins, plugin_rows):
      torch.testing.assert_close(plugin._queries._plugin_force[1], before,
                                  rtol=0, atol=0)

    candidate = sim.state._qacc.clone()
    candidate[0, 0] = 1.0e11
    mixed = mj_checkAcc(sim, qacc=candidate)
    assert mixed["bad_world"].cpu().tolist() == [True, False]
    assert sim.state.generation == generation
    assert sim._forward_stages._record is stage_owner
    assert stage_owner.row_valid.cpu().tolist() == [False, True]
    torch.testing.assert_close(sim._sensordata[1], sensor_row, rtol=0, atol=0)
    torch.testing.assert_close(
        sim._forward_stage_passive_force[1], passive_row, rtol=0, atol=0)
    for plugin, before in zip(sim._force_plugins, plugin_rows):
      torch.testing.assert_close(plugin._queries._plugin_force[1], before,
                                  rtol=0, atol=0)
    assert sim.state.qvel[0].cpu().tolist() == [0.0, 0.0]

    # The healthy neighbor's next physical step must match an untouched
    # control, including stored sensors and both built-in spatial FORCE paths.
    sim.step()
    control.step()
    for name in ("_qpos", "_qvel", "_qacc", "_time", "_status"):
      torch.testing.assert_close(getattr(sim.state, name)[1],
                                  getattr(control.state, name)[1],
                                  rtol=0, atol=0)
    torch.testing.assert_close(sim._sensordata[1], control._sensordata[1],
                                rtol=0, atol=0)
    torch.testing.assert_close(sim._forward_stage_passive_force[1],
                                control._forward_stage_passive_force[1],
                                rtol=0, atol=0)
    for plugin, reference in zip(sim._force_plugins, control._force_plugins):
      torch.testing.assert_close(plugin._queries._plugin_force[1],
                                  reference._queries._plugin_force[1],
                                  rtol=0, atol=0)
  finally:
    default_registry.unregister("checkacc_magnetic_mask", PluginType.FORCE)
    default_registry.unregister("checkacc_feedback_mask", PluginType.FORCE)


def _prepared_recovery_model():
  """Two-row equality model used to exercise real prepared ACC/CC stages."""
  return mujoco.MjModel.from_xml_string('''<mujoco>
    <option timestep=".001" gravity="0 0 -9.81" solver="PGS"
            iterations="40">
      <flag contact="disable"/>
    </option>
    <worldbody>
      <body name="left" pos="0 0 1" gravcomp="1">
        <joint name="left_slide" type="slide" axis="1 0 0" damping=".2"/>
        <geom type="sphere" size=".1" mass="1"/>
        <site name="left_site"/>
      </body>
      <body name="right" pos="1 0 1">
        <joint name="right_slide" type="slide" axis="0 1 0" damping=".1"/>
        <geom type="sphere" size=".1" mass="2"/>
      </body>
    </worldbody>
    <equality><joint name="tie" joint1="left_slide" joint2="right_slide"
      polycoef="0 1 0 0 0" solref=".02 1"/></equality>
    <actuator><motor joint="left_slide"/></actuator>
    <sensor><accelerometer site="left_site"/></sensor>
  </mujoco>''')


def _legacy_contact_recovery_model(condim):
  return mujoco.MjModel.from_xml_string(f'''<mujoco>
    <option timestep=".001" gravity="0 0 -9.81" cone="pyramidal"
            solver="PGS" iterations="40"/>
    <worldbody><geom type="plane" size="2 2 .1" condim="{condim}"/>
      <body pos="0 0 .095"><freejoint/>
        <geom type="sphere" size=".1" mass="1" condim="{condim}"
              friction=".7 .1 .05"/>
      </body></worldbody>
  </mujoco>''')


def test_native_recovery_xml_fixtures_compile_with_source_profiles():
  """Compile every MPS check-state XML fixture before GPU gate admission."""
  from pathlib import Path
  from mujoco_metal.stepping import validate_stepping_profile

  simple = _model(disable_autoreset=False)
  assert validate_stepping_profile(
      simple, profile="integrated_euler_v1").name == "integrated_euler_v1"
  mixed_stage = _prepared_recovery_model()
  assert validate_stepping_profile(
      mixed_stage, profile="integrated_euler_v1").name == "integrated_euler_v1"
  assert validate_stepping_profile(
      mixed_stage, profile="integrated_scalable_v1").name == "integrated_scalable_v1"
  haptic = mujoco.MjModel.from_xml_path(
      str(Path(__file__).parents[1] / "examples" / "haptic_calligraphy.xml"))
  assert validate_stepping_profile(
      haptic, profile="integrated_euler_v1").name == "integrated_euler_v1"
  for profile, condim in (("normal_contact_euler_v1", 1),
                          ("friction_contact_euler_v1", 3)):
    model = _legacy_contact_recovery_model(condim)
    assert validate_stepping_profile(model, profile=profile).name == profile
    data = mujoco.MjData(model)
    data.qvel[:] = [.1, -.2, -.05, .3, .1, -.2]
    mujoco.mj_forward(model, data)
    assert data.ncon == 1


def test_device_validation_accepts_canonical_unindexed_allocated_aliases():
  from mujoco_metal.device_state import _device_matches as state_device_matches
  from mujoco_metal.simulation import _device_matches as sim_device_matches

  device = lambda kind, index=None: SimpleNamespace(type=kind, index=index)
  cases = (
      (device("mps", 0), device("mps"), True),
      (device("mps", 0), device("mps", 1), False),
      (device("cpu"), device("cpu"), True),
      (device("cuda", 1), device("cuda", 1), True),
      (device("cuda", 0), device("cuda", 1), False),
      (device("mps", 0), device("mps", 1), False),
      (device("cuda", 0), device("mps"), False),
  )
  for actual, expected, result in cases:
    assert state_device_matches(actual, expected) is result
    assert sim_device_matches(actual, expected) is result


def _prepare_native_chain(sim):
  record = sim.prepare_forward_position()
  sim.prepare_forward_velocity(record)
  sim.prepare_forward_actuation(record)
  return record


@pytest.mark.gpu
@pytest.mark.skipif(os.getenv("MUJOCO_METAL_RUN_GPU") != "1",
                    reason="opt-in native prepared-stage recovery")
@pytest.mark.parametrize("profile,component", [
    ("integrated_euler_v1", False),
    ("integrated_scalable_v1", True),
])
def test_native_mixed_reset_continues_same_prepared_acc_constraint_record(
    profile, component):
  """Masked recovery preserves the healthy row through the original record."""
  from mujoco_metal.simulation import MetalSimulation

  model = _prepared_recovery_model()
  qpos = np.asarray([[.03, -.01], [-.02, .04]], dtype=np.float32)
  qvel = np.asarray([[.2, -.1], [-.15, .23]], dtype=np.float32)
  sim = MetalSimulation(model, batch_size=2, qpos=qpos, qvel=qvel,
                        profile=profile)
  control = MetalSimulation(model, batch_size=2, qpos=qpos, qvel=qvel,
                            profile=profile)
  def assert_qacc_finite(label, owner):
    actual = owner.state._qacc
    assert torch.isfinite(actual).all(), (label, "owned state qacc", actual)
    public = owner.state.qacc
    assert torch.isfinite(public).all(), (label, "public qacc", public)
    torch.testing.assert_close(public, actual, rtol=0, atol=0)

  assert_qacc_finite("construction", sim)
  assert_qacc_finite("construction", control)
  assert sim._component_mass_enabled is component
  record = _prepare_native_chain(sim)
  assert_qacc_finite("sim POS/VEL/ACT", sim)
  control_record = _prepare_native_chain(control)
  assert_qacc_finite("control POS/VEL/ACT", control)
  control_acc = control.prepare_forward_acceleration(control_record)
  assert torch.isfinite(control_acc["qacc_smooth"]).all(), control_acc
  assert_qacc_finite("control ACC", control)
  control_constraint = control.prepare_forward_constraint(control_record)
  assert torch.isfinite(control_constraint["qacc"]).all(), control_constraint
  assert_qacc_finite("control CONSTRAINT", control)
  assert int(sim._coupled_constraints.descriptor.nr) > 0

  # Keep the record, cached owners, and healthy row alive across selected-row
  # AUTORESET. This explicitly tests the prepared kernels' mask ABI instead
  # of preparing a new all-valid record in sim.step().
  record_identity = sim._forward_stages._record
  row_valid_identity = record.row_valid
  saved_stage_owner = record.values[next(iter(record.values))]
  bad = sim.state._qacc.clone()
  bad[0, 0] = float("inf")
  checked = mj_checkAcc(sim, qacc=bad)
  assert_qacc_finite("after checkAcc recovery", sim)
  assert checked["bad_world"].detach().cpu().tolist() == [True, False]
  assert sim._forward_stages._record is record_identity
  assert record.row_valid is row_valid_identity
  assert record.row_valid.detach().cpu().tolist() == [False, True]
  assert record.values[next(iter(record.values))] is saved_stage_owner

  acceleration = sim.prepare_forward_acceleration(record)
  assert torch.isfinite(acceleration["qacc_smooth"]).all(), acceleration
  assert_qacc_finite("sim ACC", sim)
  assert acceleration["status"].detach().cpu().tolist() == [3, 0]
  assert torch.count_nonzero(acceleration["qacc_smooth"][0]) == 0
  torch.testing.assert_close(
      acceleration["qfrc_smooth"][1], control_acc["qfrc_smooth"][1],
      rtol=0, atol=0)
  torch.testing.assert_close(
      acceleration["qacc_smooth"][1], control_acc["qacc_smooth"][1],
      rtol=0, atol=0)

  constrained = sim.prepare_forward_constraint(record)
  assert torch.isfinite(constrained["qacc"]).all(), constrained
  assert_qacc_finite("sim CONSTRAINT", sim)
  assert constrained["status"].detach().cpu().tolist() == [3, 0]
  assert torch.count_nonzero(constrained["qacc"][0]) == 0
  torch.testing.assert_close(constrained["qacc"][1],
                             control_constraint["qacc"][1], rtol=0, atol=0)
  assert torch.isfinite(sim._sensordata).all()
  torch.testing.assert_close(sim._sensordata[1], control._sensordata[1],
                             rtol=0, atol=0)

  # A complete public checkpoint is the supported replay boundary: it
  # invalidates the borrowed split-stage record, restores persistent state,
  # and allows the same prepared query to be repeated without stale rows.
  checkpoint = sim.snapshot()
  assert_qacc_finite("before snapshot", sim)
  saved_state = {name: getattr(sim.state, "_" + name).clone()
                 for name in ("qpos", "qvel", "qacc", "time", "status")}
  sim.reset(env_ids=[0])
  sim.restore(checkpoint)
  for name, expected in saved_state.items():
    torch.testing.assert_close(getattr(sim.state, "_" + name), expected,
                               rtol=0, atol=0)
  assert sim._forward_stages._record is None
  repeated = _prepare_native_chain(sim)
  repeated_acc = sim.prepare_forward_acceleration(repeated)
  repeated_cc = sim.prepare_forward_constraint(repeated)
  assert torch.isfinite(repeated_acc["qacc_smooth"]).all()
  assert torch.isfinite(repeated_cc["qacc"]).all()


@pytest.mark.gpu
@pytest.mark.skipif(os.getenv("MUJOCO_METAL_RUN_GPU") != "1",
                    reason="opt-in native legacy-constraint recovery")
@pytest.mark.parametrize("profile,condim", [
    ("normal_contact_euler_v1", 1),
    ("friction_contact_euler_v1", 3),
])
def test_native_legacy_constraint_autoreset_keeps_healthy_contact_row(
    profile, condim):
  """The nonsplit legacy contact solver also guards selected reset rows."""
  from mujoco_metal.simulation import MetalSimulation

  model = _legacy_contact_recovery_model(condim)
  qvel = np.asarray([[.1, -.2, -.05, .3, .1, -.2],
                     [-.15, .1, -.03, -.1, -.3, .2]], dtype=np.float32)
  sim = MetalSimulation(model, batch_size=2, qvel=qvel, profile=profile)
  control = MetalSimulation(model, batch_size=2, qvel=qvel, profile=profile)
  oracle = mujoco.MjData(model)
  oracle.qpos[:] = model.qpos0
  oracle.qvel[:] = qvel[1]
  mujoco.mj_forward(model, oracle)
  assert oracle.ncon == 1
  sim.step()
  control.step()
  assert sim._contact is not None
  # Legacy profiles do not admit the public standalone POS constructor. The
  # production FWDINV path nevertheless captures its real completed stages;
  # use that owned record to exercise the same ACC/CONSTRAINT consumers.
  sim._acceleration(sim.state._qpos, sim.state._qvel, capture_fwdinv=True)
  control._acceleration(control.state._qpos, control.state._qvel,
                        capture_fwdinv=True)
  record = sim._forward_stages._record
  control_record = control._forward_stages._record
  assert record is not None and control_record is not None
  assert int(record.stage) == 5 and int(control_record.stage) == 5
  healthy = {name: getattr(sim.state, "_" + name)[1].clone()
             for name in ("qpos", "qvel", "qacc", "status")}
  candidate = sim.state._qacc.clone()
  candidate[0, 0] = 1.0e11
  result = mj_checkAcc(sim, qacc=candidate)
  assert result["bad_world"].detach().cpu().tolist() == [True, False]
  assert record.row_valid.detach().cpu().tolist() == [False, True]
  for name, expected in healthy.items():
    torch.testing.assert_close(getattr(sim.state, "_" + name)[1], expected,
                               rtol=0, atol=0)
  assert torch.isfinite(sim.state._qacc).all()
  sim.prepare_forward_actuation(record)
  control.prepare_forward_actuation(control_record)
  acceleration = sim.prepare_forward_acceleration(record)
  control_acceleration = control.prepare_forward_acceleration(control_record)
  assert acceleration["status"].detach().cpu().tolist() == [3, 0]
  torch.testing.assert_close(
      acceleration["qacc_smooth"][1],
      control_acceleration["qacc_smooth"][1], rtol=0, atol=0)
  constrained = sim.prepare_forward_constraint(record)
  control_constrained = control.prepare_forward_constraint(control_record)
  assert constrained["status"].detach().cpu().tolist() == [3, 0]
  assert torch.count_nonzero(constrained["qacc"][0]) == 0
  torch.testing.assert_close(constrained["qacc"][1],
                             control_constrained["qacc"][1], rtol=0, atol=0)
  sim.step()
  control.step()
  for name in ("qpos", "qvel", "qacc", "time", "status"):
    torch.testing.assert_close(getattr(sim.state, "_" + name)[1],
                               getattr(control.state, "_" + name)[1],
                               rtol=0, atol=0)


def test_empty_autorecovery_does_not_replace_borrowed_acc_status_or_low():
  """A healthy check can run the masked recovery path with aliased outputs."""
  from types import MethodType
  from mujoco_metal.device_state import DeviceState
  from mujoco_metal.simulation import MetalSimulation
  from mujoco_metal.stepping import validate_stepping_profile

  model = _model(disable_autoreset=False)
  state = DeviceState(model, validate_stepping_profile(model), 1, device="cpu")
  candidate_acc = torch.tensor([[7.25]], dtype=torch.float32)
  candidate_status = torch.tensor([4], dtype=torch.int32)
  candidate_low = torch.tensor([[0.125]], dtype=torch.float32)
  sim = SimpleNamespace(
      model=model, _mjmodel=model, state=state, _state=state, batch_size=1,
      _native_plugins=(), _coupled_constraints=None, _delay=None, _flex=None,
      _sleep_schedule=None, _sensordata=None, _raw_sensordata=None,
      _control=None, _body_wrench=None, _applied_force=None, _energy=None,
      _next_qacc=torch.empty_like(state._qacc),
      _next_status=torch.empty_like(state._status),
      _check_qacc_low=torch.empty_like(state._qacc),
      _last_recovery_qacc_low=None, _assembled_system_valid=False,
      _accepted_step=None, _last_coupled=None, _last_coupled_generation=None,
      _step1_record=None)
  sim._qacc_low_for_sensor = MethodType(
      MetalSimulation._qacc_low_for_sensor, sim)
  sim._cache_sensor_acceleration = MethodType(
      MetalSimulation._cache_sensor_acceleration, sim)

  def masked_recovery_acceleration(*_args, **_kwargs):
    # The real solver's outputs are workspace views. Re-entering a healthy
    # forward can therefore overwrite exactly the tensors supplied to check.
    candidate_acc.zero_()
    candidate_status.zero_()
    candidate_low.fill_(99.0)
    sim._last_recovery_qacc_low = torch.full_like(candidate_low, -9.0)
    return candidate_acc, candidate_status, {}

  sim._acceleration = masked_recovery_acceleration
  sim._reset_from_check = lambda mask, **kwargs: (
      MetalSimulation._reset_from_check(sim, mask, **kwargs))
  sim._update_energy_position = lambda *_args, **_kwargs: None
  sim._update_energy_velocity = lambda *_args, **_kwargs: None
  sim._next_qvel = torch.empty_like(state._qvel)
  sim._cache_sensor_acceleration(candidate_acc, candidate_low)

  generation = state.generation
  acceleration, qvel, status, mask = MetalSimulation._check_step_acceleration(
      sim, candidate_acc, state._qvel, candidate_status)
  assert mask.tolist() == [False]
  assert state.generation == generation
  assert acceleration.tolist() == [[7.25]]
  assert status.tolist() == [4]
  assert torch.equal(qvel, state._qvel)
  restored_low = sim._qacc_low_for_sensor(acceleration)
  assert restored_low.tolist() == [[0.125]]
