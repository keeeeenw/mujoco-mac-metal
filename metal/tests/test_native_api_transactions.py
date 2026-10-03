# Copyright 2026 The MuJoCo Metal contributors
# Licensed under the Apache License, Version 2.0.
"""Pinned mjtState ownership, transaction, and cache-coherence regressions."""

import os

import mujoco
import numpy as np
import pytest

from mujoco_metal import MetalSimulation
from mujoco_metal.extensions import NativePlugin, PluginType, default_registry
from mujoco_metal.native_api import StateSpec, mj_getState, mj_setState

pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(os.getenv("MUJOCO_METAL_RUN_GPU") != "1", reason="opt-in GPU"),
]


def _model():
  return mujoco.MjModel.from_xml_string("""<mujoco>
    <option gravity="0 0 0" timestep=".002" solver="PGS" iterations="100"/>
    <worldbody><body><joint name="j" type="slide" axis="1 0 0"/>
      <geom type="sphere" size=".1" mass="1" contype="0" conaffinity="0"/>
    </body></worldbody></mujoco>""")


def _host(t):
  return t.detach().cpu().numpy().copy()


class StatefulTestPlugin(NativePlugin):
  """Small per-world plugin state used to qualify simulation transactions."""

  def __init__(self, name, role, reject_negative=False, fail_output=False,
               fail_reset=False, fail_restore_at=None):
    super().__init__(name, role)
    self.reject_negative = reject_negative
    self.fail_output = fail_output
    self.fail_reset = fail_reset
    self.fail_restore_at = fail_restore_at
    self.restore_calls = 0
    self.calls = None

  def init(self, model, batch_size=1, device=None):
    super().init(model, batch_size, device)
    import torch
    self.calls = torch.zeros((batch_size,), dtype=torch.int32, device=self.device)

  def run_device(self, state, sensordata=None, sensor_dim=1, **kwargs):
    import torch
    self.calls.add_(1)
    if self.plugin_type == PluginType.SENSOR:
      return self.calls.to(torch.float32).reshape(-1, 1).expand(-1, sensor_dim).contiguous()
    if self.fail_output:
      return torch.full((self.batch_size, self.model.nv), float("nan"),
                        dtype=torch.float32, device=self.device)
    return torch.zeros((self.batch_size, self.model.nv), dtype=torch.float32,
                       device=self.device)

  def snapshot(self):
    return self.calls.detach().cpu().numpy().copy()

  def restore(self, snap, env_ids=None):
    self.restore_calls += 1
    values = np.asarray(snap)
    if values.shape != (self.batch_size,):
      raise ValueError("invalid test plugin snapshot")
    rows = np.arange(self.batch_size) if env_ids is None else np.asarray(env_ids)
    self.calls[rows.tolist()] = self.calls.new_tensor(values[rows], dtype=self.calls.dtype)
    # Deliberately reject after mutation to verify late plugin failure rollback.
    if ((self.reject_negative and np.any(values[rows] < 0)) or
        self.restore_calls == self.fail_restore_at):
      raise ValueError("reject test plugin state")

  def reset(self, env_ids=None):
    if env_ids is None:
      self.calls.zero_()
    else:
      self.calls[np.asarray(env_ids).tolist()] = 0
    if self.fail_reset:
      raise ValueError("reject test plugin reset")


def test_applied_force_selector_is_owned_and_drives_step():
  model = _model()
  sim = MetalSimulation(model, profile="integrated_euler_v1")
  mj_setState(sim, {"qfrc_applied": np.array([[2]], np.float32)})
  assert "qfrc_applied" in mj_getState(sim, StateSpec.QFRC_APPLIED)
  sim.step()
  np.testing.assert_allclose(_host(sim.state.qacc), [[2]], atol=1e-6)
  # Omitted arguments retain installed native state; explicit zero clears it.
  sim.step()
  np.testing.assert_allclose(_host(sim.state.qacc), [[2]], atol=1e-6)
  sim.step(qfrc_applied=np.zeros((1, model.nv), np.float32))
  np.testing.assert_allclose(_host(sim.state.qacc), [[0]], atol=1e-6)


def test_setstate_is_atomic_and_invalidates_assembled_cache():
  model = mujoco.MjModel.from_xml_string("""<mujoco><option timestep=".002" solver="PGS" iterations="100"/>
    <worldbody><geom type="plane" size="2 2 .1"/><body pos="0 0 .09">
      <joint type="slide" axis="0 0 1"/><geom type="sphere" size=".1" mass="1" condim="1"/>
    </body></worldbody></mujoco>""")
  sim = MetalSimulation(model, profile="integrated_euler_v1")
  sim.step()
  before = _host(sim.state.qpos)
  generation = sim.state.generation
  bad = np.full((1, model.nv), np.nan, np.float32)
  with pytest.raises(ValueError):
    mj_setState(sim, {"qpos": np.array([[.7]], np.float32), "warmstart": bad})
  np.testing.assert_array_equal(_host(sim.state.qpos), before)
  assert sim.state.generation == generation

  mj_setState(sim, {"qpos": np.array([[1.0]], np.float32)})
  assert sim.state.generation == generation + 1
  assert not sim._assembled_system_valid
  np.testing.assert_allclose(_host(sim.assembled_system()["qfrc_constraint"]), 0, atol=1e-5)


def test_selected_full_state_groups_copy_reset_and_restore():
  model = _model()
  sim = MetalSimulation(model, batch_size=2, profile="integrated_euler_v1")
  values = {
      "time": np.array([.25, .5], np.float32),
      "qpos": np.array([[.1], [.2]], np.float32),
      "qvel": np.array([[.3], [.4]], np.float32),
      "qacc": np.array([[.5], [.6]], np.float32),
      "qacc_warmstart": np.array([[.7], [.8]], np.float32),
      "qfrc_applied": np.array([[1.], [2.]], np.float32),
      "userdata": np.empty((2, 0), np.float32),
      "plugin_state": np.empty((2, 0), np.float32),
  }
  mj_setState(sim, values)
  state = mj_getState(sim, StateSpec.INTEGRATION)
  assert set(state) == {
      "time", "qpos", "qvel", "act", "history", "warmstart", "ctrl",
      "qfrc_applied", "xfrc_applied", "eq_active", "mocap_pos",
      "mocap_quat", "userdata", "plugin_state",
  }
  assert state["warmstart"].shape == (2, model.nv)
  np.testing.assert_allclose(_host(state["warmstart"]), values["qacc_warmstart"])
  np.testing.assert_allclose(_host(state["qfrc_applied"]), values["qfrc_applied"])
  roundtrip = mj_getState(sim, StateSpec.ALL)
  mj_setState(sim, roundtrip)
  snap = sim.snapshot()

  mj_setState(sim, {"qpos": np.array([[9.]], np.float32)}, env_ids=[1])
  np.testing.assert_allclose(_host(sim.state.qpos), [[.1], [9.]])
  np.testing.assert_allclose(_host(sim._applied_force), [[1.], [2.]])
  sim.restore(snap)
  np.testing.assert_allclose(_host(sim.state.qpos), [[.1], [.2]])
  np.testing.assert_allclose(_host(sim.state._qacc_warmstart), [[.7], [.8]])

  sim.copy_environment(0, 1)
  np.testing.assert_allclose(_host(sim._applied_force), [[1.], [1.]])
  sim.reset(env_ids=[1])
  np.testing.assert_allclose(_host(sim.state._qacc_warmstart), [[.7], [0.]])


def test_rejects_wrong_device_nonfinite_and_invalid_quaternion_before_write():
  import torch
  model = _model()
  sim = MetalSimulation(model, profile="integrated_euler_v1")
  before = _host(sim.state.qpos)
  with pytest.raises(ValueError, match="finite"):
    mj_setState(sim, {"qpos": np.array([[np.inf]], np.float32)})
  with pytest.raises(ValueError, match="duplicates"):
    mj_setState(sim, {"qpos": np.array([[1.], [2.]], np.float32)}, env_ids=[0, 0])
  wrong_device = torch.tensor([[1.]], dtype=torch.float32, device="cpu")
  with pytest.raises(ValueError, match="must be on"):
    mj_setState(sim, {"qpos": wrong_device})
  np.testing.assert_array_equal(_host(sim.state.qpos), before)


def test_bool_eq_active_and_duplicate_aliases_are_handled_explicitly():
  model = mujoco.MjModel.from_xml_string("""<mujoco>
    <option gravity="0 0 0" solver="PGS" iterations="100"/>
    <worldbody>
      <body><joint name="a" type="slide" axis="1 0 0"/>
        <geom type="sphere" size=".1" mass="1" contype="0" conaffinity="0"/></body>
      <body pos="1 0 0"><joint name="b" type="slide" axis="1 0 0"/>
        <geom type="sphere" size=".1" mass="1" contype="0" conaffinity="0"/></body>
    </worldbody><equality><joint joint1="a" joint2="b" polycoef="0 1 0 0 0"/></equality>
    </mujoco>""")
  sim = MetalSimulation(model, profile="integrated_euler_v1")
  mj_setState(sim, {"eq_active": np.array([True], dtype=np.bool_)})
  np.testing.assert_array_equal(_host(mj_getState(sim, StateSpec.EQ_ACTIVE)["eq_active"]), [[1]])
  with pytest.raises(ValueError, match="duplicate state field alias"):
    mj_setState(sim, {"warmstart": np.zeros((1, model.nv), np.float32),
                      "qacc_warmstart": np.ones((1, model.nv), np.float32)})


def test_stateful_force_sensor_plugins_reset_copy_restore_and_late_failure():
  model = mujoco.MjModel.from_xml_string("""<mujoco>
    <option gravity="0 0 0" timestep=".002" solver="PGS" iterations="100"/>
    <worldbody><body><joint type="slide" axis="1 0 0"/>
      <geom type="sphere" size=".1" mass="1" contype="0" conaffinity="0"/>
    </body></worldbody>
    <sensor><user name="stateful_sensor" dim="1"/></sensor></mujoco>""")
  sensor = StatefulTestPlugin("stateful_sensor", PluginType.SENSOR)
  force = StatefulTestPlugin("stateful_force", PluginType.FORCE)
  late_failure = StatefulTestPlugin("late_failure", PluginType.FORCE,
                                    reject_negative=True, fail_output=True)
  default_registry.register(sensor)
  default_registry.register(force)
  default_registry.register(late_failure)
  try:
    sim = MetalSimulation(model, batch_size=2, profile="integrated_euler_v1")
    sim._native_plugins[-1].fail_output = False
    sim.step()
    np.testing.assert_array_equal(sim._native_plugins[0].snapshot(), [1, 1])
    np.testing.assert_array_equal(sim._native_plugins[1].snapshot(), [1, 1])
    snap = sim.snapshot()
    sim.reset(env_ids=[1])
    np.testing.assert_array_equal(sim._native_plugins[0].snapshot(), [1, 0])
    np.testing.assert_array_equal(sim._native_plugins[1].snapshot(), [1, 0])
    sim.copy_environment(0, 1)
    np.testing.assert_array_equal(sim._native_plugins[0].snapshot(), [1, 1])
    np.testing.assert_array_equal(sim._native_plugins[1].snapshot(), [1, 1])
    sim.restore(snap)
    np.testing.assert_array_equal(sim._native_plugins[0].snapshot(), [1, 1])

    # Selected restore updates plugin and kinematic rows without replacing
    # state owned by the neighboring world.
    mj_setState(sim, {"qpos": np.array([[.3]], np.float32)}, env_ids=[0])
    mj_setState(sim, {"qpos": np.array([[.4]], np.float32)}, env_ids=[1])
    sim._native_plugins[0].calls[0] = 3
    sim._native_plugins[1].calls[0] = 4
    sim.restore(snap, env_ids=[1])
    np.testing.assert_allclose(_host(sim.state.qpos), [[.3], [0.]])
    np.testing.assert_array_equal(sim._native_plugins[0].snapshot(), [3, 1])
    np.testing.assert_array_equal(sim._native_plugins[1].snapshot(), [4, 1])

    before_qpos = _host(sim.state.qpos)
    before_sensor = sim._native_plugins[0].snapshot()
    bad = sim.snapshot()
    bad["plugins"][(PluginType.FORCE.value, "late_failure")] = np.array([-1, -1], np.int32)
    with pytest.raises(ValueError, match="reject test plugin state"):
      sim.restore(bad)
    np.testing.assert_array_equal(_host(sim.state.qpos), before_qpos)
    np.testing.assert_array_equal(sim._native_plugins[0].snapshot(), before_sensor)
    # A payload can pass the mutating preflight restore and fail only on the
    # real commit call. Native rows and every plugin instance must still roll
    # back exactly.
    late = sim.snapshot()
    key = (PluginType.FORCE.value, "late_failure")
    late["plugins"][key] = np.array([7, 8], np.int32)
    runtime_late = sim._native_plugins[-1]
    runtime_late.fail_restore_at = runtime_late.restore_calls + 3
    before_late = runtime_late.snapshot()
    with pytest.raises(ValueError, match="reject test plugin state"):
      sim.restore(late)
    np.testing.assert_array_equal(_host(sim.state.qpos), before_qpos)
    np.testing.assert_array_equal(runtime_late.snapshot(), before_late)
    before_forces = [p.snapshot() for p in sim._native_plugins[1:]]
    sim._native_plugins[-1].fail_output = True
    with pytest.raises(ValueError, match="invalid force tensor"):
      sim.step()
    np.testing.assert_array_equal(_host(sim.state.qpos), before_qpos)
    for plugin, before in zip(sim._native_plugins, [before_sensor, *before_forces]):
      np.testing.assert_array_equal(plugin.snapshot(), before)
  finally:
    default_registry.unregister("stateful_sensor", PluginType.SENSOR)
    default_registry.unregister("stateful_force", PluginType.FORCE)
    default_registry.unregister("late_failure", PluginType.FORCE)


def test_reset_and_copy_rollback_when_late_plugin_lifecycle_callback_rejects():
  model = _model()
  reset_plugin = StatefulTestPlugin("late_reset", PluginType.FORCE, fail_reset=True)
  default_registry.register(reset_plugin)
  try:
    sim = MetalSimulation(model, batch_size=2, profile="integrated_euler_v1")
    mj_setState(sim, {"qpos": np.array([[.2], [.4]], np.float32),
                      "qvel": np.array([[.1], [.3]], np.float32),
                      "qfrc_applied": np.array([[1.], [2.]], np.float32)})
    runtime = sim._native_plugins[0]
    runtime.calls[:] = runtime.calls.new_tensor([3, 4])
    before = {"qpos": _host(sim.state.qpos), "qvel": _host(sim.state.qvel),
              "force": _host(sim._applied_force), "plugin": runtime.snapshot(),
              "generation": sim.state.generation}
    with pytest.raises(ValueError, match="reject test plugin reset"):
      sim.reset()
    np.testing.assert_array_equal(_host(sim.state.qpos), before["qpos"])
    np.testing.assert_array_equal(_host(sim.state.qvel), before["qvel"])
    np.testing.assert_array_equal(_host(sim._applied_force), before["force"])
    np.testing.assert_array_equal(runtime.snapshot(), before["plugin"])
    assert sim.state.generation == before["generation"]
  finally:
    default_registry.unregister("late_reset", PluginType.FORCE)

  copy_plugin = StatefulTestPlugin("late_copy", PluginType.FORCE, fail_restore_at=3)
  default_registry.register(copy_plugin)
  try:
    sim = MetalSimulation(_model(), batch_size=2, profile="integrated_euler_v1")
    mj_setState(sim, {"qpos": np.array([[.2], [.4]], np.float32),
                      "qvel": np.array([[.1], [.3]], np.float32),
                      "qfrc_applied": np.array([[1.], [2.]], np.float32)})
    runtime = sim._native_plugins[0]
    runtime.calls[:] = runtime.calls.new_tensor([3, 4])
    before = {"qpos": _host(sim.state.qpos), "qvel": _host(sim.state.qvel),
              "force": _host(sim._applied_force), "plugin": runtime.snapshot(),
              "generation": sim.state.generation}
    with pytest.raises(ValueError, match="reject test plugin state"):
      sim.copy_environment(0, 1)
    np.testing.assert_array_equal(_host(sim.state.qpos), before["qpos"])
    np.testing.assert_array_equal(_host(sim.state.qvel), before["qvel"])
    np.testing.assert_array_equal(_host(sim._applied_force), before["force"])
    np.testing.assert_array_equal(runtime.snapshot(), before["plugin"])
    assert sim.state.generation == before["generation"]
  finally:
    default_registry.unregister("late_copy", PluginType.FORCE)


def test_keyframe_reset_resets_only_selected_plugin_rows():
  model = mujoco.MjModel.from_xml_string("""<mujoco>
    <option gravity="0 0 0" timestep=".002" solver="PGS" iterations="100"/>
    <worldbody><body><joint name="j" type="slide" axis="1 0 0"/>
      <geom type="sphere" size=".1" mass="1" contype="0" conaffinity="0"/>
    </body></worldbody>
    <keyframe><key name="rest" qpos=".25" qvel="0"/></keyframe>
  </mujoco>""")
  plugin = StatefulTestPlugin("keyframe_plugin", PluginType.FORCE)
  default_registry.register(plugin)
  try:
    sim = MetalSimulation(model, batch_size=2, profile="integrated_euler_v1")
    runtime = sim._native_plugins[0]
    runtime.calls[:] = runtime.calls.new_tensor([5, 6])
    sim.reset_to_keyframe(0, env_ids=[1])
    np.testing.assert_array_equal(runtime.snapshot(), [5, 0])
    np.testing.assert_allclose(_host(sim.state.qpos), [[0], [.25]])
  finally:
    default_registry.unregister("keyframe_plugin", PluginType.FORCE)
