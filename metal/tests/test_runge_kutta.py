# Copyright 2026 The MuJoCo Metal contributors
# Licensed under the Apache License, Version 2.0.
"""RK4 tests against MuJoCo 3.10, including quaternion stage semantics."""

import os
import mujoco
import numpy as np
import pytest
from mujoco_metal.stepping import validate_stepping_profile

XML = """<mujoco><option integrator='RK4' timestep='.002'><flag contact='disable'/></option>
<worldbody><body pos='0 0 1'><freejoint/><geom type='box' size='.1 .2 .3'/></body>
<body pos='1 0 1'><joint type='ball' damping='.1'/><geom type='capsule' size='.08 .3' pos='.1 .2 0'/></body>
<body pos='2 0 1'><joint name='j' axis='0 1 0' damping='.2'/><geom type='box' size='.1 .2 .3' pos='.1 0 -.2'/></body>
</worldbody><actuator><motor joint='j' gear='2'/></actuator></mujoco>"""


def test_rk4_contract():
  m = mujoco.MjModel.from_xml_string(XML)
  profile = validate_stepping_profile(m, profile='contact_free_motor_rk4_v1')
  assert not profile.implicit_euler_damping
  assert profile.passive_damping_enabled
  with pytest.raises(ValueError, match='Euler'):
    validate_stepping_profile(m, profile='contact_free_motor_euler_v1')
  m.opt.integrator = mujoco.mjtIntegrator.mjINT_EULER
  with pytest.raises(ValueError, match='RK4'):
    validate_stepping_profile(m, profile='contact_free_motor_rk4_v1')


def test_integrated_rk4_sleep_is_admitted_with_scalable_position_context():
  model = mujoco.MjModel.from_xml_string('''<mujoco>
    <option integrator="RK4"><flag sleep="enable"/></option>
    <worldbody><body><joint type="slide" damping=".1"/>
      <geom type="sphere" size=".1"/></body></worldbody>
    </mujoco>''')
  profile = validate_stepping_profile(model, profile="integrated_rk4_v1")
  assert profile.name == "integrated_rk4_v1"
  assert "sleep mode" not in profile.rejected


def _rk4_sleep_model():
  return mujoco.MjModel.from_xml_string('''<mujoco>
    <option integrator="RK4" timestep=".002" gravity="0 0 -9.81"
      sleep_tolerance=".1">
      <flag sleep="enable"/>
    </option>
    <worldbody>
      <geom name="floor" type="plane" size="2 2 .1"/>
      <body name="drop" pos="0 0 .1001" sleep="allowed">
        <joint name="drop_joint" type="slide" axis="0 0 1" damping=".1"/>
        <geom name="drop_geom" type="sphere" size=".1" mass="1"/>
      </body>
      <body name="drive" pos="1 0 .5" sleep="allowed">
        <joint name="drive_joint" type="slide" axis="1 0 0" damping=".1"/>
        <geom type="sphere" size=".05" mass="1"/>
      </body>
    </worldbody>
    <actuator><general name="filter" joint="drive_joint"
      dyntype="filter" dynprm=".05" gainprm="0" biastype="none"/></actuator>
    <sensor><jointpos joint="drop_joint"/><jointpos joint="drive_joint"/></sensor>
  </mujoco>''')


def test_rk4_sleep_public_profile_preserves_descriptor_across_state_validation():
  """Simulation and DeviceState validators admit the same sleep profile."""
  import mujoco_metal.device_state as device_state_module

  model = _rk4_sleep_model()
  profile = validate_stepping_profile(model, profile="integrated_rk4_v1")
  assert profile.name == "integrated_rk4_v1"
  state_profile = device_state_module.validate_stepping_profile(
      model, profile.timestep, profile=profile.name)
  assert state_profile == profile


def test_cpu_rk4_sleep_native_fixture_has_contact_sensor_and_activation_inputs():
  model = _rk4_sleep_model()
  assert model.nv == 2 and model.na == 1 and model.nsensor == 2
  data = mujoco.MjData(model)
  data.qvel[0] = -.2
  data.qvel[1] = .01
  data.ctrl[0] = .2
  mujoco.mj_step(model, data)
  assert int(data.ncon) > 0
  assert np.any(data.act != 0)


def test_cpu_rk4_sleep_keeps_last_stage_pose_for_next_cycle_wake():
  """mj_advance leaves RK4 X3 pose caches for the next kinematics1 pass."""
  model = _rk4_sleep_model()
  data = mujoco.MjData(model)
  data.qvel[:] = [.05, .01]
  # These are awake countdown values. Below-tolerance qvel can reach sleep at
  # the end of this RK4 step, after all four forward stages.
  data.tree_asleep[:] = [-2, -2]
  data.ctrl[0] = .2
  mujoco.mj_forward(model, data)
  np.testing.assert_array_equal(data.tree_asleep, [-2, -2])

  mujoco.mj_step(model, data)
  np.testing.assert_array_equal(data.tree_asleep, [0, 1])
  np.testing.assert_array_equal(data.qpos, [0, 0])
  np.testing.assert_array_equal(data.qvel, [0, 0])
  retained_xpos = data.xpos.copy()

  # mj_advance did not recompute kinematics after integrating X0. A fresh
  # forward at the same public qpos detects this real cached-X3 mismatch and
  # wakes the complete sleeping cycle.
  recomputed = mujoco.MjData(model)
  mujoco.mj_copyData(recomputed, model, data)
  mujoco.mj_forward(model, recomputed)
  np.testing.assert_array_equal(recomputed.tree_asleep, [-11, -11])
  assert np.max(np.abs(retained_xpos - recomputed.xpos)) > 5e-5


def test_cpu_simulation_reset_restores_fresh_state_and_sleep_fk_ownership():
  """Exercise Simulation.reset with real DeviceState and CPU-owned fixtures."""
  torch = pytest.importorskip("torch")
  from mujoco_metal.device_state import DeviceState
  from mujoco_metal.simulation import MetalSimulation
  from mujoco_metal.sleep_schedule import initial_sleep_state

  model = _rk4_sleep_model()
  profile = validate_stepping_profile(model, profile="integrated_rk4_v1")
  batch = 2
  sim = object.__new__(MetalSimulation)
  sim._mjmodel = model
  sim.batch_size = batch
  sim._state = DeviceState(model, profile, batch, device="cpu")
  sim._energy = torch.full((batch, 2), 7.0)
  sim._control = torch.full((batch, model.nu), 2.0)
  sim._applied_force = torch.full((batch, model.nv), 3.0)
  sim._body_wrench = torch.full((batch, model.nbody, 6), 4.0)
  sim._sensordata = torch.full((batch, model.nsensordata), 5.0)
  sim._native_plugins = ()
  sim._coupled_constraints = None
  sim._delay = None
  sim._islands = None
  sim._flex = None
  sim._assembled_system_valid = True
  sim._accepted_step = object()
  sim._last_coupled = object()
  sim._last_coupled_generation = 4
  sim._step1_record = object()

  # Seed every reset-sensitive state group with a value distinguishable from
  # construction defaults, then run the actual DeviceState/Simulation reset.
  state = sim._state
  state._qpos.fill_(0.03)
  state._qvel.fill_(0.04)
  state._qacc.fill_(0.05)
  state._qacc_warmstart.fill_(0.06)
  state._time.fill_(0.07)
  state._status.fill_(2)
  state._act.fill_(0.08)
  if state._history.numel():
    state._history.fill_(0.09)

  class FK:
    def __init__(self):
      self._workspace = {"outputs": {
          "body_pos": torch.full((batch, model.nbody * 3), -1.0),
          "body_quat": torch.full((batch, model.nbody * 4), -1.0),
          "cache_valid": torch.ones((batch,), dtype=torch.float32),
      }}
      self.invalidated = []

    def invalidate_cache(self, env_ids=None):
      self.invalidated.append(None if env_ids is None else tuple(env_ids))
      if env_ids is None:
        self._workspace["outputs"]["cache_valid"].zero_()
      elif len(env_ids):
        self._workspace["outputs"]["cache_valid"][list(env_ids)] = 0

  class Sleep:
    def __init__(self):
      self.tree_state = torch.zeros((batch, model.ntree), dtype=torch.int32)
      self.tree_awake = torch.zeros_like(self.tree_state)
      self.reset_calls = []

    def reset(self, env_ids=None):
      rows = (list(range(batch)) if env_ids is None else
              np.asarray(env_ids, dtype=np.int64).tolist())
      self.reset_calls.append(tuple(rows))
      initial = torch.tensor(
          np.asarray(initial_sleep_state(model)).copy(), dtype=torch.int32)
      for world in rows:
        self.tree_state[world].copy_(initial)
        self.tree_awake[world].copy_((initial < 0).to(torch.int32))

  fk, sleep = FK(), Sleep()
  sim._smooth = type("Smooth", (), {"_fk": fk})()
  sim._sleep_schedule = sleep
  reset_data = mujoco.MjData(model)
  sim._sleep_reset_body_pos = torch.as_tensor(
      np.asarray(reset_data.xpos, dtype=np.float32).copy())
  sim._sleep_reset_body_quat = torch.as_tensor(
      np.asarray(reset_data.xquat, dtype=np.float32).copy())

  def on_state_reset(env_ids=None):
    fk.invalidate_cache(env_ids=env_ids)
    sleep.reset(env_ids=env_ids)
    sim._seed_sleep_fk_reference(env_ids=env_ids)
  state._on_reset = on_state_reset

  sim.reset()

  fresh = DeviceState(model, profile, batch, device="cpu")
  for name in ("_qpos", "_qvel", "_qacc", "_qacc_warmstart", "_time",
               "_status", "_act", "_history"):
    np.testing.assert_array_equal(getattr(state, name).numpy(),
                                  getattr(fresh, name).numpy(), err_msg=name)
  np.testing.assert_array_equal(sim._energy.numpy(), 0)
  np.testing.assert_array_equal(sim._control.numpy(), 0)
  np.testing.assert_array_equal(sim._applied_force.numpy(), 0)
  np.testing.assert_array_equal(sim._body_wrench.numpy(), 0)
  np.testing.assert_array_equal(sim._sensordata.numpy(), 0)
  np.testing.assert_array_equal(sleep.tree_state.numpy(),
                                np.tile(initial_sleep_state(model), (batch, 1)))
  np.testing.assert_array_equal(sleep.tree_awake.numpy(), 1)
  np.testing.assert_array_equal(fk._workspace["outputs"]["cache_valid"].numpy(), 0)
  np.testing.assert_array_equal(
      fk._workspace["outputs"]["body_pos"].numpy().reshape(batch, model.nbody, 3),
      np.tile(np.asarray(reset_data.xpos, dtype=np.float32), (batch, 1, 1)))
  np.testing.assert_array_equal(
      fk._workspace["outputs"]["body_quat"].numpy().reshape(batch, model.nbody, 4),
      np.tile(np.asarray(reset_data.xquat, dtype=np.float32), (batch, 1, 1)))
  # DeviceState.reset normalizes the full reset to explicit world IDs before
  # invoking lifecycle callbacks, so invalidation is selected-row scoped.
  assert fk.invalidated == [(np.int64(0), np.int64(1))]
  # Scheduler state resets in the DeviceState reset transaction and is not
  # redundantly cleared by the outer Simulation.reset call.
  assert sleep.reset_calls == [(0, 1)]
  assert sim._accepted_step is None and sim._last_coupled is None
  assert sim._last_coupled_generation is None and sim._step1_record is None
  assert not sim._assembled_system_valid


def test_cpu_reset_orchestration_matches_pinned_cold_defaults_without_torch():
  """Run Simulation.reset itself against a CPU state/reset oracle double."""
  from mujoco_metal.simulation import MetalSimulation
  from mujoco_metal.sleep_schedule import initial_sleep_state

  model = _rk4_sleep_model()
  batch = 2

  class TensorView:
    def __init__(self, array):
      self.array = np.asarray(array)

    def view(self, *shape):
      return TensorView(self.array.reshape(*shape))

    def expand_as(self, other):
      return TensorView(np.broadcast_to(self.array, other.array.shape))

    def copy_(self, other):
      self.array[...] = (other.array if isinstance(other, TensorView) else other)
      return self

    def zero_(self):
      self.array.fill(0)
      return self

  class FakeScheduler:
    def __init__(self):
      self.tree_state = np.zeros((batch, model.ntree), dtype=np.int32)
      self.tree_awake = np.zeros_like(self.tree_state)
      self.calls = []

    def reset(self, env_ids=None):
      rows = range(batch) if env_ids is None else env_ids
      rows = tuple(int(row) for row in rows)
      self.calls.append(rows)
      for row in rows:
        self.tree_state[row] = initial_sleep_state(model)
        self.tree_awake[row] = (self.tree_state[row] < 0).astype(np.int32)

  class FakeState:
    def __init__(self, callback):
      self.generation = 7
      self._on_reset = callback
      self._qpos = np.full((batch, model.nq), 0.03, dtype=np.float32)
      self._qvel = np.full((batch, model.nv), 0.04, dtype=np.float32)
      self._qacc = np.full((batch, model.nv), 0.05, dtype=np.float32)
      self._qacc_warmstart = np.full((batch, model.nv), 0.06, dtype=np.float32)
      self._time = np.full((batch,), 0.07, dtype=np.float32)
      self._status = np.full((batch,), 2, dtype=np.int32)
      self._act = np.full((batch, model.na), 0.08, dtype=np.float32)
      self._history = np.full((batch, 1), 0.09, dtype=np.float32)

    def prepare_reset(self, **kwargs):
      return (np.arange(batch, dtype=np.int64),)

    def reset(self, *, _prepared):
      data = mujoco.MjData(model)
      mujoco.mj_resetData(model, data)
      for name, field in (("_qpos", "qpos"), ("_qvel", "qvel"),
                          ("_act", "act")):
        getattr(self, name)[:] = getattr(data, field)
      self._qacc.fill(0)
      self._qacc_warmstart.fill(0)
      self._time.fill(0)
      self._status.fill(0)
      self._history.fill(0)
      self.generation += 1
      self._on_reset(env_ids=None)
      return self.generation

  sim = object.__new__(MetalSimulation)
  sim._mjmodel = model
  sim.batch_size = batch
  sim._energy = TensorView(np.full((batch, 2), 7.0, dtype=np.float32))
  sim._control = TensorView(np.full((batch, model.nu), 2.0, dtype=np.float32))
  sim._applied_force = TensorView(np.full((batch, model.nv), 3.0, dtype=np.float32))
  sim._body_wrench = TensorView(np.full((batch, model.nbody, 6), 4.0, dtype=np.float32))
  sim._sensordata = TensorView(np.full((batch, model.nsensordata), 5.0, dtype=np.float32))
  sim._native_plugins = ()
  sim._coupled_constraints = None
  sim._delay = None
  sim._islands = None
  sim._flex = None
  sim._assembled_system_valid = True
  sim._accepted_step = object()
  sim._last_coupled = object()
  sim._last_coupled_generation = 4
  sim._step1_record = object()
  fk_outputs = {
      "body_pos": TensorView(np.full((batch, model.nbody * 3), -1.0, dtype=np.float32)),
      "body_quat": TensorView(np.full((batch, model.nbody * 4), -1.0, dtype=np.float32)),
      "cache_valid": TensorView(np.ones((batch,), dtype=np.float32)),
  }

  class FK:
    def __init__(self):
      self._workspace = {"outputs": fk_outputs}
      self.invalidations = 0

    def invalidate_cache(self, env_ids=None):
      self.invalidations += 1
      fk_outputs["cache_valid"].zero_()

  fk, scheduler = FK(), FakeScheduler()
  sim._smooth = type("Smooth", (), {"_fk": fk})()
  sim._sleep_schedule = scheduler
  reset_data = mujoco.MjData(model)
  sim._sleep_reset_body_pos = TensorView(
      np.asarray(reset_data.xpos, dtype=np.float32).copy())
  sim._sleep_reset_body_quat = TensorView(
      np.asarray(reset_data.xquat, dtype=np.float32).copy())
  sim._state = FakeState(lambda env_ids=None: (
      fk.invalidate_cache(env_ids), scheduler.reset(env_ids),
      sim._seed_sleep_fk_reference(env_ids=env_ids)))

  generation = sim.reset()
  assert generation == 8
  reset_oracle = mujoco.MjData(model)
  mujoco.mj_resetData(model, reset_oracle)
  np.testing.assert_array_equal(sim._state._qpos,
                                np.tile(reset_oracle.qpos, (batch, 1)))
  np.testing.assert_array_equal(sim._state._qvel, 0)
  np.testing.assert_array_equal(sim._state._qacc, 0)
  np.testing.assert_array_equal(sim._state._qacc_warmstart, 0)
  np.testing.assert_array_equal(sim._state._time, 0)
  np.testing.assert_array_equal(sim._state._status, 0)
  np.testing.assert_array_equal(sim._state._act,
                                np.tile(reset_oracle.act, (batch, 1)))
  np.testing.assert_array_equal(sim._energy.array, 0)
  np.testing.assert_array_equal(sim._control.array, 0)
  np.testing.assert_array_equal(sim._applied_force.array, 0)
  np.testing.assert_array_equal(sim._body_wrench.array, 0)
  np.testing.assert_array_equal(sim._sensordata.array, 0)
  np.testing.assert_array_equal(
      scheduler.tree_state,
      np.tile(initial_sleep_state(model), (batch, 1)))
  np.testing.assert_array_equal(scheduler.tree_awake, 1)
  np.testing.assert_array_equal(fk_outputs["cache_valid"].array, 0)
  np.testing.assert_allclose(
      fk_outputs["body_pos"].array.reshape(batch, model.nbody, 3),
      np.tile(reset_oracle.xpos, (batch, 1, 1)), rtol=0, atol=1e-7)
  np.testing.assert_array_equal(
      fk_outputs["body_quat"].array.reshape(batch, model.nbody, 4),
      np.tile(reset_oracle.xquat, (batch, 1, 1)))
  assert fk.invalidations == 1 and scheduler.calls == [(0, 1)]
  assert not sim._assembled_system_valid and sim._accepted_step is None
  assert sim._last_coupled is None and sim._last_coupled_generation is None
  assert sim._step1_record is None


@pytest.mark.gpu
@pytest.mark.skipif(os.getenv("MUJOCO_METAL_RUN_GPU") != "1", reason="opt-in GPU")
def test_native_integrated_rk4_sleep_cached_position_lifecycle_matches_mj_step():
  """Exercise public RK4 sleep with cached X3 position and restored X0 state."""
  import torch
  from mujoco_metal import MetalSimulation
  from mujoco_metal.sleep_schedule import initial_sleep_state

  model = _rk4_sleep_model()

  batch, dt = 2, float(model.opt.timestep)
  qpos = np.tile(np.asarray(model.qpos0, dtype=np.float32), (batch, 1))
  qvel = np.asarray([[-.2, .01], [.05, .01]], dtype=np.float32)
  sim = MetalSimulation(model, batch, qpos=qpos, qvel=qvel,
                        profile="integrated_rk4_v1")
  refs = [mujoco.MjData(model) for _ in range(batch)]
  tree_state = np.tile(initial_sleep_state(model), (batch, 1))
  tree_state[1, :model.ntree] = -2
  sim._sleep_schedule.tree_state.copy_(torch.as_tensor(tree_state, device="mps"))
  sim._sleep_schedule.tree_awake.copy_(
      (sim._sleep_schedule.tree_state < 0).to(dtype=torch.int32))
  sim._sleep_schedule._build_awake_lists()
  for world, data in enumerate(refs):
    data.qpos[:] = qpos[world]
    data.qvel[:] = qvel[world]
    data.tree_asleep[:] = tree_state[world, :model.ntree]
    data.ctrl[0] = .2
    mujoco.mj_forward(model, data)

  captured = []
  refreshed = []
  sleep_transition_trace = []
  capture = sim._capture_forward_position
  refresh = sim._refresh_sleep_velocity
  prepare_sleep = sim._prepare_sleep_integration

  def record_capture(stage_qpos, dynamics):
    token = capture(stage_qpos, dynamics)
    captured.append((stage_qpos.detach().cpu().numpy().copy(), token))
    return token

  def record_refresh(stage_qpos, stage_qvel, acceleration, accepted,
                     position_context=None):
    refreshed.append(position_context)
    result = refresh(stage_qpos, stage_qvel, acceleration, accepted,
                     position_context=position_context)
    sleep_transition_trace.append({
        "stage": "refresh",
        "qpos": stage_qpos.detach().cpu().numpy().copy(),
        "qvel": stage_qvel.detach().cpu().numpy().copy(),
        "acceleration": acceleration.detach().cpu().numpy().copy(),
        "accepted": accepted.detach().cpu().numpy().copy(),
        "result": result.detach().cpu().numpy().copy(),
        "tree_state": sim._sleep_schedule.tree_state.detach().cpu().numpy().copy(),
    })
    return result

  def record_prepare(stage_qvel, acceleration, active_worlds):
    before = sim._sleep_schedule.tree_state.detach().cpu().numpy().copy()
    result = prepare_sleep(stage_qvel, acceleration, active_worlds)
    sleep_transition_trace.append({
        "stage": "prepare", "qvel": stage_qvel.detach().cpu().numpy().copy(),
        "acceleration": acceleration.detach().cpu().numpy().copy(),
        "active_worlds": active_worlds.detach().cpu().numpy().copy(),
        "tree_before": before,
        "tree_after": sim._sleep_schedule.tree_state.detach().cpu().numpy().copy(),
    })
    return result

  sim._capture_forward_position = record_capture
  sim._refresh_sleep_velocity = record_refresh
  sim._prepare_sleep_integration = record_prepare

  def assert_matches_reference(label):
    state = sim.state.snapshot()
    np.testing.assert_array_equal(state.status, np.zeros(batch, dtype=np.int32))
    for world, data in enumerate(refs):
      try:
        np.testing.assert_allclose(state.qpos[world], data.qpos,
                                   rtol=3e-4, atol=3e-5)
      except AssertionError as error:
        raise AssertionError(
            f"{label} world {world} qpos mismatch; "
            f"native_qpos={state.qpos[world]!r}, cpu_qpos={data.qpos!r}, "
            f"native_qvel={state.qvel[world]!r}, cpu_qvel={data.qvel!r}, "
            f"native_qacc={state.qacc[world]!r}, cpu_qacc={data.qacc!r}, "
            f"native_tree={sim._sleep_schedule.tree_state[world].detach().cpu().numpy()!r}, "
            f"cpu_tree={data.tree_asleep!r}, "
            f"trace={sleep_transition_trace[-2:]!r}") from error
      try:
        np.testing.assert_allclose(state.qvel[world], data.qvel,
                                   rtol=4e-4, atol=4e-5)
      except AssertionError as error:
        raise AssertionError(
            f"{label} world {world} qvel mismatch; qpos={state.qpos[world]!r}, "
            f"native_qvel={state.qvel[world]!r}, cpu_qvel={data.qvel!r}, "
            f"native_tree={sim._sleep_schedule.tree_state[world].detach().cpu().numpy()!r}, "
            f"cpu_tree={data.tree_asleep!r}, trace="
            f"{sleep_transition_trace[-2:]!r}") from error
      np.testing.assert_allclose(state.qacc[world], data.qacc,
                                 rtol=8e-4, atol=8e-4)
      np.testing.assert_allclose(state.time[world], data.time,
                                 rtol=0, atol=2e-7)
      np.testing.assert_allclose(sim._state._act[world].cpu().numpy(),
                                 data.act, rtol=3e-4, atol=3e-6)
    np.testing.assert_allclose(sim.step_sensordata(),
                               np.stack([d.sensordata for d in refs]),
                               rtol=2e-4, atol=2e-6)
    np.testing.assert_array_equal(
        sim._sleep_schedule.tree_state.cpu().numpy()[:, :model.ntree],
        np.stack([d.tree_asleep for d in refs]))

  start_qpos = qpos.copy()
  for step in range(4):
    ctrl = np.asarray([[.2], [.2]], dtype=np.float32)
    sim.step(ctrl=ctrl)
    for data in refs:
      data.ctrl[:] = ctrl[0]
      mujoco.mj_step(model, data)
    assert_matches_reference(f"initial trajectory step {step}")
  assert len(captured) == 4 and len(refreshed) == 4
  assert all(refreshed[i] is captured[i][1] for i in range(4))
  # The final callback captured X3, while the public state began at X0. This
  # confirms the refresh receives the stage context rather than recomputing
  # all position data from the restored state.
  assert np.max(np.abs(captured[0][0] - start_qpos)) > 1e-6
  assert int(refs[0].ncon) > 0

  checkpoint = sim.snapshot()
  cpu_checkpoint = [mujoco.MjData(model) for _ in range(batch)]
  for saved, data in zip(cpu_checkpoint, refs):
    mujoco.mj_copyData(saved, model, data)
  for _ in range(2):
    sim.step(ctrl=np.full((batch, 1), .2, dtype=np.float32))
    for data in refs:
      data.ctrl[0] = .2
      mujoco.mj_step(model, data)
  replay = sim.state.snapshot()
  sim.restore(checkpoint)
  for data, saved in zip(refs, cpu_checkpoint):
    mujoco.mj_copyData(data, model, saved)
  for _ in range(2):
    sim.step(ctrl=np.full((batch, 1), .2, dtype=np.float32))
    for data in refs:
      data.ctrl[0] = .2
      mujoco.mj_step(model, data)
  np.testing.assert_array_equal(sim.state.snapshot().qpos, replay.qpos)
  np.testing.assert_array_equal(sim.state.snapshot().qvel, replay.qvel)
  assert_matches_reference("restored trajectory")

  sim.reset()
  for data in refs:
    mujoco.mj_resetData(model, data)
  sim.step(ctrl=np.full((batch, 1), .2, dtype=np.float32))
  for data in refs:
    data.ctrl[0] = .2
    mujoco.mj_step(model, data)
  assert_matches_reference("reset trajectory")

  before = sim.state.snapshot()
  invalid_force = torch.zeros((batch, model.nv), dtype=torch.float32,
                              device="mps")
  invalid_force[1, 0] = float("nan")
  failed_status = sim.step(qfrc_applied=invalid_force)
  after = sim.state.snapshot()
  assert int(failed_status[0].item()) == 0
  assert int(failed_status[1].item()) != 0
  np.testing.assert_array_equal(after.qpos[1], before.qpos[1])
  np.testing.assert_array_equal(after.qvel[1], before.qvel[1])
  np.testing.assert_array_equal(after.time[1], before.time[1])
  refs[0].qfrc_applied[:] = 0
  mujoco.mj_step(model, refs[0])
  sim.reset(env_ids=[1])
  mujoco.mj_resetData(model, refs[1])
  sim.step(qfrc_applied=torch.zeros((batch, model.nv), dtype=torch.float32,
                                    device="mps"),
           ctrl=np.full((batch, 1), .2, dtype=np.float32))
  expected_ctrl = np.full((batch, 1), .2, dtype=np.float64)
  for world, data in enumerate(refs):
    data.ctrl[:] = expected_ctrl[world]
  np.testing.assert_array_equal(
      np.stack([np.asarray(data.ctrl) for data in refs]), expected_ctrl)
  for data in refs:
    mujoco.mj_step(model, data)
  assert_matches_reference("failed world reset trajectory")


def _cpu_stage_orchestrator():
  """Exercise actual callback ordering without claiming native physics parity."""
  torch = pytest.importorskip("torch")
  from mujoco_metal.runge_kutta import MetalRungeKutta
  program = object.__new__(MetalRungeKutta)
  program._torch, program._na, program.dt = torch, 0, .002
  program._zero = torch.zeros((1, 1))
  program._initial_qvel = torch.empty((1, 1), dtype=torch.float32)
  class Position:
    def run_device(self, q, v, a, t, status):
      return q+.002*v, v, t+.002, status
  program._position = Position()
  return program, (torch.zeros((1,1)),torch.ones((1,1)),None,
                   torch.zeros(1),torch.zeros(1,dtype=torch.int32))


def test_rk4_reuses_outer_forward_as_first_stage_and_preserves_three_argument_callback():
  program, args = _cpu_stage_orchestrator()
  calls = []
  def callback(q,v,a):
    calls.append(1)
    return program._torch.zeros_like(v),None,args[-1]
  initial=(program._torch.zeros_like(args[1]),None,args[-1])
  program.run_device(*args,callback,initial_stage=initial)
  assert len(calls)==3


def test_rk4_does_not_retry_callback_body_typeerror():
  program,args=_cpu_stage_orchestrator()
  calls=[]
  def callback(q,v,a,time=None):
    calls.append(1)
    raise TypeError("callback body failed")
  with pytest.raises(TypeError,match="callback body failed"):
    program.run_device(*args,callback)
  assert len(calls)==1


def test_rk4_can_defer_final_integration_and_return_final_stage_context():
  program, args = _cpu_stage_orchestrator()
  calls = []
  context = {"position_cache": object()}
  def callback(q, v, a, time=None):
    calls.append((q.clone(), v.clone()))
    acceleration = program._torch.full_like(v, 2.0)
    return acceleration, None, args[-1], context

  result = program.run_device(*args, callback, defer_integration=True)
  assert len(calls) == 4
  assert result["final_context"] is context
  # The runner used only the three half/full-step state integrations between
  # stages; the caller now owns the final update after its sleep transition.
  torch = program._torch
  torch.testing.assert_close(result["weighted_acceleration"],
                             torch.full_like(args[1], 2.0))
  torch.testing.assert_close(result["next_velocity"], args[1] + .004)
  torch.testing.assert_close(result["position_velocity"], args[1] + .002)
  torch.testing.assert_close(result["time"], args[3] + .002)
  assert result["status"].item() == 0


def test_rk4_keeps_owned_x0_velocity_when_callbacks_reuse_input_scratch():
  program, args = _cpu_stage_orchestrator()
  qpos, borrowed_x0, act, time, status = args
  borrowed_x0.fill_(1.0)
  calls = []

  def callback(q, v, a, time=None):
    calls.append(v.clone())
    # Model Simulation._sleep_mask_velocity reusing the caller's backing.
    borrowed_x0.fill_(100.0)
    return program._torch.zeros_like(v), None, status

  initial = (program._torch.zeros_like(borrowed_x0), None, status)
  result = program.run_device(
      qpos, borrowed_x0, act, time, status, callback,
      initial_stage=initial, defer_integration=True)
  assert len(calls) == 3
  np.testing.assert_array_equal(result["position_velocity"].numpy(), [[1.0]])
  np.testing.assert_array_equal(result["weighted_acceleration"].numpy(), [[0.0]])
  np.testing.assert_array_equal(program._initial_qvel.numpy(), [[1.0]])


def test_pinned_forward_skip_pos_retains_final_stage_geometry_after_qpos_restore():
  """Pin the mixed X0 state / X3 position-cache rule used by RK4 sleep."""
  model = mujoco.MjModel.from_xml_string('''<mujoco><option gravity="0 0 0"/>
    <worldbody><body><joint type="slide" axis="1 0 0"/>
      <geom type="sphere" size=".1"/></body></worldbody></mujoco>''')
  data = mujoco.MjData(model)
  data.qpos[0] = .05  # X0
  mujoco.mj_forward(model, data)
  x0_geom = data.geom_xpos.copy()
  data.qpos[0] = .4  # final RK substage X3
  mujoco.mj_forward(model, data)
  x3_geom = data.geom_xpos.copy()
  assert abs(x3_geom[0, 0] - x0_geom[0, 0]) > .3

  # mj_RungeKutta restores X0 then mj_advance's forwardSkip(POS) updates
  # velocity/acceleration caches while preserving the X3 position caches.
  data.qpos[0] = .05
  data.qvel[0] = 0.0
  mujoco.mj_forwardSkip(model, data, mujoco.mjtStage.mjSTAGE_POS, 0)
  np.testing.assert_array_equal(data.geom_xpos, x3_geom)

  # A full recomputation at the restored X0 is observably different; merely
  # supplying X0 to the current monolithic forward path cannot implement the
  # pinned RK4 sleep transition.
  mujoco.mj_forward(model, data)
  assert abs(data.geom_xpos[0, 0] - x3_geom[0, 0]) > .3


def test_pinned_rk4_forward_skip_retains_actuator_length_and_contact_position():
  """Position-dependent caches belong to X3 during the post-sleep refresh."""
  model = mujoco.MjModel.from_xml_string('''<mujoco>
    <option gravity="0 0 0"/>
    <worldbody><geom name="floor" type="plane" size="1 1 .1"/>
      <body><joint name="lift" type="slide" axis="0 0 1"/>
        <geom name="ball" type="sphere" size=".1"/></body>
    </worldbody><actuator><motor joint="lift"/></actuator>
    </mujoco>''')
  data = mujoco.MjData(model)
  data.qpos[0] = .7  # X0: clear of the floor.
  mujoco.mj_forward(model, data)
  x0_length = data.actuator_length.copy()
  assert data.ncon == 0

  data.qpos[0] = .05  # X3: actuator length and contact manifold differ.
  mujoco.mj_forward(model, data)
  x3_length = data.actuator_length.copy()
  x3_ncon = int(data.ncon)
  assert x3_ncon > 0
  assert abs(x3_length[0] - x0_length[0]) > .5

  data.qpos[0] = .7
  mujoco.mj_forwardSkip(model, data, mujoco.mjtStage.mjSTAGE_POS, 0)
  assert int(data.ncon) == x3_ncon
  np.testing.assert_array_equal(data.actuator_length, x3_length)
  mujoco.mj_forward(model, data)
  assert int(data.ncon) == 0
  assert abs(data.actuator_length[0] - x3_length[0]) > .5


def test_pinned_forward_skip_gravcomp_uses_cached_final_stage_pose():
  model = mujoco.MjModel.from_xml_string('''<mujoco>
    <option gravity="0 0 -9.81"/><worldbody>
      <body gravcomp="1"><joint type="hinge" axis="0 1 0"/>
        <inertial pos=".5 0 0" mass="1" diaginertia=".1 .1 .1"/>
        <geom type="sphere" pos=".5 0 0" size=".1" mass="1"/>
      </body></worldbody></mujoco>''')
  data = mujoco.MjData(model)
  data.qpos[0] = 0.0  # X0
  mujoco.mj_forward(model, data)
  x0_gravcomp = data.qfrc_gravcomp.copy()
  data.qpos[0] = np.pi / 2  # X3
  mujoco.mj_forward(model, data)
  x3_gravcomp = data.qfrc_gravcomp.copy()
  assert abs(x0_gravcomp[0] - x3_gravcomp[0]) > 4.0
  data.qpos[0] = 0.0
  mujoco.mj_forwardSkip(model, data, mujoco.mjtStage.mjSTAGE_POS, 0)
  np.testing.assert_array_equal(data.qfrc_gravcomp, x3_gravcomp)


@pytest.mark.gpu
@pytest.mark.skipif(os.getenv("MUJOCO_METAL_RUN_GPU") != "1", reason="opt-in GPU")
def test_rk4_sensor_outer_forward_calls_force_callback_four_times_and_error_once():
  import torch
  from mujoco_metal import MetalSimulation
  from mujoco_metal.extensions import NativePlugin, PluginType, default_registry

  class Counter(NativePlugin):
    def __init__(self):
      super().__init__("rk4_counter", PluginType.FORCE)
      self.attempts = 0
      self.fail = False

    def init(self, model, batch_size=1, device=None):
      super().init(model, batch_size, device)
      self.count = torch.zeros(batch_size, dtype=torch.int32, device=self.device)

    def run_device(self, state, **kwargs):
      self.attempts += 1  # Diagnostic invocation count, outside physics state.
      self.count.add_(1)
      if self.fail:
        raise TypeError("force callback body failed")
      return torch.zeros((self.batch_size, self.model.nv), device=self.device)

    def snapshot(self):
      return self.count.cpu().numpy().copy()

    def device_snapshot(self):
      return self.count.clone()

    def device_snapshot_bytes(self):
      return self.count.numel() * self.count.element_size()

    def restore_device(self, snapshot):
      self.count.copy_(snapshot)

    def restore_masked(self, snapshot, accepted_mask):
      self.count.copy_(torch.where(accepted_mask, self.count, snapshot))

  model = mujoco.MjModel.from_xml_string('''<mujoco><option integrator="RK4"
    gravity="0 0 0"><flag contact="disable"/></option><worldbody><body>
    <joint name="j"/><geom type="sphere" size=".1" mass="1"/>
    </body></worldbody><sensor><jointpos joint="j"/></sensor></mujoco>''')
  default_registry.register_factory("rk4_counter", PluginType.FORCE, Counter)
  try:
    sim = MetalSimulation(model, 2, profile="integrated_rk4_v1")
    plugin = sim._native_plugins[0]
    sim.step()
    np.testing.assert_array_equal(plugin.snapshot(), [4, 4])
    assert plugin.attempts == 4
    before = sim.state.snapshot()
    plugin.fail = True
    with pytest.raises(TypeError, match="force callback body failed"):
      sim.step()
    assert plugin.attempts == 5
    np.testing.assert_array_equal(plugin.snapshot(), [4, 4])
    after = sim.state.snapshot()
    np.testing.assert_array_equal(after.qpos, before.qpos)
    np.testing.assert_array_equal(after.time, before.time)
  finally:
    default_registry.clear()


@pytest.mark.gpu
@pytest.mark.skipif(
    os.getenv('MUJOCO_METAL_RUN_GPU') != '1', reason='opt-in GPU'
)
def test_rk4_trajectory_restore_and_row_failure():
  import torch
  from mujoco_metal.simulation import MetalSimulation

  m = mujoco.MjModel.from_xml_string(XML)
  rng = np.random.default_rng(321)
  q = np.tile(m.qpos0, (3, 1)).astype('float32')
  v = rng.normal(0, 0.4, (3, m.nv)).astype('float32')
  sim = MetalSimulation(m, 3, q, v, profile='contact_free_motor_rk4_v1')
  refs = [mujoco.MjData(m) for _ in range(3)]
  for i, d in enumerate(refs):
    d.qpos[:] = q[i]
    d.qvel[:] = v[i]
  for step in range(500):
    ctrl = np.full((3, 1), 0.1 * np.sin(step / 50), dtype='float32')
    force = np.zeros((3, m.nv), dtype='float32')
    force[:, :3] = [0.2, -0.1, 0.3]
    sim.step(ctrl=ctrl, qfrc_applied=force)
    for i, d in enumerate(refs):
      d.ctrl[:] = ctrl[i]
      d.qfrc_applied[:] = force[i]
      mujoco.mj_step(m, d)
  snap = sim.state.snapshot()
  assert not np.any(snap.status)
  for i, d in enumerate(refs):
    np.testing.assert_allclose(snap.qpos[i], d.qpos, atol=6e-5, rtol=2e-4)
    np.testing.assert_allclose(snap.qvel[i], d.qvel, atol=8e-5, rtol=2e-4)
    np.testing.assert_allclose(snap.qacc[i], d.qacc, atol=1e-3, rtol=3e-4)
  sim.step(3)
  replay = sim.state.snapshot()
  sim.state.restore(snap)
  sim.step(3)
  np.testing.assert_array_equal(sim.state.snapshot().qpos, replay.qpos)
  force = torch.zeros((3, m.nv), device='mps')
  force[1, 0] = float('nan')
  before = sim.state.snapshot()
  sim.step(qfrc_applied=force)
  after = sim.state.snapshot()
  assert after.status[1] != 0
  np.testing.assert_array_equal(after.qpos[1], before.qpos[1])
  assert after.status[0] == 0 and after.status[2] == 0
