# Copyright 2026 The MuJoCo Metal contributors
# Licensed under the Apache License, Version 2.0.
"""CPU tensor-contract tests for the typed actuator extension boundary."""

from types import SimpleNamespace

import pytest
import os
import mujoco
import numpy as np

torch = pytest.importorskip("torch")

from mujoco_metal.extensions import ActuatorPluginOutput, NativeActuatorPlugin
from mujoco_metal.simulation import MetalSimulation
from mujoco_metal.extensions import PluginType, default_registry


class _TypedActuator(NativeActuatorPlugin):
  def __init__(self):
    super().__init__("typed-test")
    self.force = None
    self.derivative = None
    self.advances = []
    self.advance_states = []

  def run_actuator_device(self, state, *, control, kinematics, dynamics,
                          act=None, act_dot=None, compute_mask=None):
    return ActuatorPluginOutput(self.force, self.derivative)

  def advance_actuator_device(self, state, *, accepted_mask, timestep, output):
    self.advances.append((accepted_mask.clone(), float(timestep), output))
    self.advance_states.append(state)


def _simulation_stub(plugin):
  sim = object.__new__(MetalSimulation)
  state = SimpleNamespace(
      _torch=torch,
      _device=torch.device("cpu"),
      _na=2,
      _qpos=torch.tensor([[0.1, -0.2]], dtype=torch.float32),
      _qvel=torch.tensor([[0.3, 0.4]], dtype=torch.float32),
      _qacc=torch.tensor([[1.0, -1.0]], dtype=torch.float32),
      _act=torch.tensor([[0.5, 0.6]], dtype=torch.float32),
      _time=torch.tensor([0.25], dtype=torch.float32),
      _reset_mask_i32=torch.zeros((1,), dtype=torch.int32),
  )
  def clear_rows(value, mask):
    if tuple(state._reset_mask_i32.shape) == tuple(mask.shape):
      state._reset_mask_i32.copy_(mask)
    value[mask] = 0
  def update_rows(target, source, mask):
    target[mask] += source[mask]
  state.clear_masked_rows = clear_rows
  state.update_masked_rows = update_rows
  sim._state = state
  sim._mjmodel = SimpleNamespace(nu=2, nv=2)
  sim.batch_size = 1
  sim.profile = SimpleNamespace(timestep=0.01)
  sim._native_actuator_plugins = (plugin,)
  def project_plugin_force(force, moment, out, *, world_mask=None):
    projected = torch.bmm(moment.transpose(1, 2), force.unsqueeze(2)).squeeze(2)
    if world_mask is None:
      out.copy_(projected)
    else:
      out[world_mask.to(dtype=torch.bool)] = projected[
          world_mask.to(dtype=torch.bool)]
    return out
  sim._actuators = SimpleNamespace(project_plugin_force=project_plugin_force)
  sim._last_actuation_kin = {
      "moment": torch.tensor([[[1.0, 2.0], [-1.0, 3.0]]], dtype=torch.float32)
  }
  sim._native_actuator_force = torch.zeros((1, 2), dtype=torch.float32)
  sim._native_actuator_qfrc = torch.zeros((1, 2), dtype=torch.float32)
  sim._act_dot = torch.tensor([[0.25, -0.5]], dtype=torch.float32)
  sim._last_native_actuator_outputs = {}
  return sim


def test_typed_actuator_projects_force_and_adds_activation_derivative():
  plugin = _TypedActuator()
  plugin.force = torch.tensor([[2.0, -3.0]], dtype=torch.float32)
  plugin.derivative = torch.tensor([[0.75, 1.25]], dtype=torch.float32)
  sim = _simulation_stub(plugin)
  bucket = torch.zeros((1, 2), dtype=torch.float32)
  state = SimpleNamespace()

  output = sim._run_native_actuator_plugins(
      state, torch.zeros((1, 2)), {}, torch.zeros((1, 2)), None,
      force_target=bucket)

  expected_qfrc = torch.tensor([[5.0, -5.0]], dtype=torch.float32)
  torch.testing.assert_close(bucket, expected_qfrc)
  torch.testing.assert_close(output["force"], plugin.force)
  torch.testing.assert_close(
      sim._act_dot, torch.tensor([[1.0, 0.75]], dtype=torch.float32))
  assert sim._last_native_actuator_outputs[plugin] is not None


def test_typed_actuator_masked_projection_preserves_unselected_rows():
  plugin = _TypedActuator()
  plugin.force = torch.tensor([[2.0, -3.0], [70.0, 80.0]],
                              dtype=torch.float32)
  plugin.derivative = torch.tensor([[0.75, 1.25], [30.0, 40.0]],
                                  dtype=torch.float32)
  sim = _simulation_stub(plugin)
  sim.batch_size = 2
  sim._mjmodel = SimpleNamespace(nu=2, nv=2)
  sim._state._reset_mask_i32 = torch.zeros((2,), dtype=torch.int32)
  sim._state._act = torch.tensor([[0.5, 0.6], [0.7, 0.8]], dtype=torch.float32)
  sim._native_actuator_force = torch.tensor([[9.0, 8.0], [3.0, 4.0]])
  sim._native_actuator_qfrc = torch.tensor([[1.0, 2.0], [6.0, 7.0]])
  sim._act_dot = torch.tensor([[0.25, -0.5], [5.0, 6.0]])
  sim._last_actuation_kin = {
      "moment": sim._last_actuation_kin["moment"].expand(2, -1, -1).clone()}
  sim._state._qpos = sim._state._qpos.expand(2, -1).clone()
  sim._state._qvel = sim._state._qvel.expand(2, -1).clone()
  sim._state._qacc = sim._state._qacc.expand(2, -1).clone()
  mask = torch.tensor([True, False])
  bucket = torch.tensor([[0.0, 0.0], [7.0, 8.0]])

  output = sim._run_native_actuator_plugins(
      SimpleNamespace(), torch.zeros((2, 2)), {}, torch.zeros((2, 2)), None,
      force_target=bucket, compute_mask=mask)

  torch.testing.assert_close(bucket[0], torch.tensor([5.0, -5.0]))
  torch.testing.assert_close(bucket[1], torch.tensor([7.0, 8.0]))
  torch.testing.assert_close(output["force"][1], torch.tensor([3.0, 4.0]))
  torch.testing.assert_close(sim._act_dot[1], torch.tensor([5.0, 6.0]))


def test_typed_actuator_state_advances_once_with_accepted_world_mask():
  plugin = _TypedActuator()
  sim = _simulation_stub(plugin)
  output = ActuatorPluginOutput(torch.ones((1, 2), dtype=torch.float32))
  sim._last_native_actuator_outputs = {plugin: output}
  accepted = torch.tensor([True], dtype=torch.bool)

  sim._advance_native_actuator_plugins(accepted)

  assert len(plugin.advances) == 1
  saved_mask, timestep, saved_output = plugin.advances[0]
  torch.testing.assert_close(saved_mask, accepted)
  assert timestep == pytest.approx(0.01)
  assert saved_output is output
  assert sim._last_native_actuator_outputs == {}


def test_typed_actuator_advance_receives_tentative_state_before_commit():
  plugin = _TypedActuator()
  sim = _simulation_stub(plugin)
  old_state = sim._state
  old_qpos = old_state._qpos.clone()
  candidate = SimpleNamespace(
      qpos=torch.tensor([[0.11, -0.19]], dtype=torch.float32),
      qvel=torch.tensor([[0.31, 0.42]], dtype=torch.float32),
      qacc=torch.tensor([[0.7, -0.8]], dtype=torch.float32),
      act=torch.tensor([[0.52, 0.63]], dtype=torch.float32),
      time=torch.tensor([0.26], dtype=torch.float32))

  sim._advance_native_actuator_plugins(
      torch.tensor([True]), qpos=candidate.qpos, qvel=candidate.qvel,
      qacc=candidate.qacc, act=candidate.act, time=candidate.time)

  assert sim._state is old_state
  torch.testing.assert_close(sim._state._qpos, old_qpos)
  observed = plugin.advance_states[0]
  for name in ("_qpos", "_qvel", "_qacc", "_act", "_time"):
    torch.testing.assert_close(getattr(observed, name),
                               getattr(candidate, name[1:]))


def test_typed_actuator_failed_advance_discards_borrowed_force_output():
  class _Failing(_TypedActuator):
    def advance_actuator_device(self, state, *, accepted_mask, timestep,
                                output):
      self.advance_states.append(state)
      raise RuntimeError("injected accepted-step failure")

  plugin = _Failing()
  sim = _simulation_stub(plugin)
  sim._last_native_actuator_outputs = {
      plugin: ActuatorPluginOutput(torch.ones((1, 2), dtype=torch.float32))}
  with pytest.raises(RuntimeError, match="injected accepted-step failure"):
    sim._advance_native_actuator_plugins(torch.tensor([True]))
  assert sim._last_native_actuator_outputs == {}


class _TransactionalStepPlugin(NativeActuatorPlugin):
  """Small stateful plugin for opt-in whole-step lifecycle qualification."""

  def __init__(self, name):
    super().__init__(name)
    self.fail_advance = False
    self.fail_world_one = False

  def init(self, model, batch_size=1, device=None):
    super().init(model, batch_size, device)
    self.ticks = torch.zeros((batch_size,), dtype=torch.float32,
                             device=self.device)
    self.last_act = torch.zeros((batch_size, int(model.na)),
                                dtype=torch.float32, device=self.device)
    self.force = torch.empty((batch_size, int(model.nu)), dtype=torch.float32,
                             device=self.device)
    self.derivative = torch.empty_like(self.last_act)

  def run_actuator_device(self, state, *, control, kinematics, dynamics,
                          act=None, act_dot=None, compute_mask=None):
    self.force.zero_()
    self.force.add_(0.05)
    if self.fail_world_one and self.force.shape[0] > 1:
      self.force[1].fill_(float("nan"))
    self.derivative.zero_()
    if self.derivative.numel():
      self.derivative.add_(0.2)
    return ActuatorPluginOutput(self.force, self.derivative)

  def advance_actuator_device(self, state, *, accepted_mask, timestep, output):
    self.ticks.add_(accepted_mask.to(dtype=torch.float32))
    if getattr(state, "_act", None) is not None:
      self.last_act.copy_(state._act)
    if self.fail_advance:
      raise RuntimeError("injected typed-plugin advance failure")

  def snapshot(self):
    if not hasattr(self, "ticks"):
      return None
    return {"ticks": self.ticks.detach().cpu().numpy().copy(),
            "last_act": self.last_act.detach().cpu().numpy().copy()}

  def reset(self, env_ids=None):
    if not hasattr(self, "ticks"):
      return
    if env_ids is None:
      self.ticks.zero_()
      self.last_act.zero_()
    else:
      ids = torch.as_tensor(env_ids, dtype=torch.int64, device=self.device)
      self.ticks[ids] = 0
      self.last_act[ids] = 0

  def restore(self, snap, env_ids=None):
    if snap is None:
      return
    if env_ids is None:
      self.ticks.copy_(torch.as_tensor(snap["ticks"], device=self.device))
      self.last_act.copy_(torch.as_tensor(snap["last_act"], device=self.device))
    else:
      ids = torch.as_tensor(env_ids, dtype=torch.int64, device=self.device)
      self.ticks[ids] = torch.as_tensor(snap["ticks"], device=self.device)[ids]
      self.last_act[ids] = torch.as_tensor(snap["last_act"], device=self.device)[ids]

  def device_snapshot(self):
    return {"ticks": self.ticks.clone(), "last_act": self.last_act.clone()}

  def device_snapshot_bytes(self):
    return (self.ticks.numel() * self.ticks.element_size()
            + self.last_act.numel() * self.last_act.element_size())

  def restore_device(self, snap):
    self.ticks.copy_(snap["ticks"])
    self.last_act.copy_(snap["last_act"])

  def restore_masked(self, snap, accepted_mask):
    mask = accepted_mask[:, None]
    self.ticks.copy_(torch.where(accepted_mask, self.ticks, snap["ticks"]))
    self.last_act.copy_(torch.where(mask, self.last_act, snap["last_act"]))

  def reset_masked(self, reset_mask):
    if (not isinstance(reset_mask, torch.Tensor)
        or reset_mask.device != self.device
        or reset_mask.dtype != torch.bool
        or tuple(reset_mask.shape) != (self.batch_size,)):
      raise ValueError("invalid reset mask")
    self.ticks.copy_(torch.where(reset_mask, torch.zeros_like(self.ticks),
                                 self.ticks))
    self.last_act.copy_(torch.where(reset_mask[:, None],
                                    torch.zeros_like(self.last_act),
                                    self.last_act))


def _transaction_model(profile="integrated_euler_v1"):
  integrator = {
      "integrated_euler_v1": mujoco.mjtIntegrator.mjINT_EULER,
      "integrated_rk4_v1": mujoco.mjtIntegrator.mjINT_RK4,
      "integrated_implicit_v1": mujoco.mjtIntegrator.mjINT_IMPLICIT,
  }[profile]
  model = mujoco.MjModel.from_xml_string("""
    <mujoco><option timestep="0.002" gravity="0 0 -9.81"/>
      <worldbody><body pos="0 0 0">
        <joint name="j" type="hinge" axis="0 1 0"/>
        <geom type="capsule" size=".04" fromto="0 0 0 .3 0 0" mass="1"/>
      </body></worldbody>
      <actuator><general joint="j" dyntype="integrator" gainprm="1"
        biastype="none" actdim="1"/></actuator>
      <sensor><jointpos joint="j"/></sensor>
    </mujoco>
  """)
  model.opt.integrator = integrator
  return model


def _register_step_plugin(name):
  default_registry.register_factory(
      name, PluginType.ACTUATOR, lambda: _TransactionalStepPlugin(name))


def _unregister_step_plugin(name):
  default_registry.unregister(name, PluginType.ACTUATOR)


def _copy_native_step_state(sim):
  state = sim._state
  copy_tensor = lambda value: (None if value is None else
                               value.detach().cpu().clone())
  return {
      "qpos": copy_tensor(state._qpos), "qvel": copy_tensor(state._qvel),
      "qacc": copy_tensor(state._qacc), "act": copy_tensor(state._act),
      "time": copy_tensor(state._time),
      "warmstart": copy_tensor(state._qacc_warmstart),
      "sensordata": copy_tensor(sim._sensordata),
      "generation": state.generation,
  }


def _assert_native_step_state_equal(a, b):
  for key in ("qpos", "qvel", "qacc", "act", "time", "warmstart",
              "sensordata"):
    if a[key] is None or b[key] is None:
      assert a[key] is b[key], key
    else:
      torch.testing.assert_close(a[key], b[key], rtol=0, atol=0, msg=key)


@pytest.mark.parametrize("profile", [
    "integrated_euler_v1", "integrated_rk4_v1", "integrated_implicit_v1",
])
@pytest.mark.skipif(__import__("os").getenv("MUJOCO_METAL_RUN_GPU") != "1",
                    reason="explicit serialized full-step MPS qualification")
def test_native_typed_actuator_step_commit_checkpoint_replay(profile):
  model = _transaction_model(profile)
  name = f"typed_step_replay_{profile}"
  _register_step_plugin(name)
  try:
    qpos = np.asarray([[0.1], [-0.2]], dtype=np.float32)
    qvel = np.asarray([[0.3], [-0.1]], dtype=np.float32)
    sim = MetalSimulation(model, batch_size=2, qpos=qpos, qvel=qvel,
                          profile=profile)
    status = sim.step()
    np.testing.assert_array_equal(status.cpu().numpy(), [0, 0])
    plugin = sim._native_actuator_plugins[0]
    torch.testing.assert_close(plugin.ticks.cpu(), torch.ones(2))
    assert torch.any(plugin.last_act != 0)

    checkpoint = sim.snapshot()
    generation_before_restore = sim._state.generation
    sim.step()
    expected = _copy_native_step_state(sim)
    expected_ticks = plugin.ticks.clone()
    sim.restore(checkpoint)
    assert sim._state.generation > generation_before_restore
    # Restore advances the monotonic generation to invalidate stage records;
    # replay equality applies to physical state and plugin history.
    assert sim.accepted_step is None
    np.testing.assert_array_equal(plugin.ticks.cpu().numpy(), [1, 1])
    sim.step()
    _assert_native_step_state_equal(_copy_native_step_state(sim), expected)
    torch.testing.assert_close(plugin.ticks, expected_ticks, rtol=0, atol=0)
  finally:
    _unregister_step_plugin(name)


@pytest.mark.parametrize("profile", [
    "integrated_euler_v1", "integrated_rk4_v1", "integrated_implicit_v1",
])
@pytest.mark.skipif(__import__("os").getenv("MUJOCO_METAL_RUN_GPU") != "1",
                    reason="explicit serialized full-step MPS qualification")
def test_native_typed_actuator_failure_rolls_back_step_and_replays(profile):
  model = _transaction_model(profile)
  name = f"typed_step_failure_{profile}"
  _register_step_plugin(name)
  try:
    qpos = np.asarray([[0.1], [-0.2]], dtype=np.float32)
    qvel = np.asarray([[0.3], [-0.1]], dtype=np.float32)
    sim = MetalSimulation(model, batch_size=2, qpos=qpos, qvel=qvel,
                          profile=profile)
    control = MetalSimulation(model, batch_size=2, qpos=qpos, qvel=qvel,
                              profile=profile)
    sim.step()
    control.step()
    before = _copy_native_step_state(sim)
    before_plugin = sim._native_actuator_plugins[0].device_snapshot()
    before_accepted = sim.accepted_step
    failing = sim._native_actuator_plugins[0]
    failing.fail_advance = True
    with pytest.raises(RuntimeError, match="injected typed-plugin advance failure"):
      sim.step()
    failing.fail_advance = False
    _assert_native_step_state_equal(_copy_native_step_state(sim), before)
    torch.testing.assert_close(failing.ticks, before_plugin["ticks"],
                               rtol=0, atol=0)
    torch.testing.assert_close(failing.last_act, before_plugin["last_act"],
                               rtol=0, atol=0)
    assert sim.accepted_step is not None
    assert sim.accepted_step["generation"] == before_accepted["generation"]

    # The rejected attempt must not change the next accepted trajectory.
    sim.step()
    control.step()
    _assert_native_step_state_equal(_copy_native_step_state(sim),
                                    _copy_native_step_state(control))
    torch.testing.assert_close(
        sim._native_actuator_plugins[0].ticks,
        control._native_actuator_plugins[0].ticks, rtol=0, atol=0)
  finally:
    _unregister_step_plugin(name)


@pytest.mark.skipif(__import__("os").getenv("MUJOCO_METAL_RUN_GPU") != "1",
                    reason="explicit serialized failed-world plugin qualification")
def test_native_typed_actuator_advances_only_accepted_worlds():
  model = _transaction_model()
  name = "typed_step_failed_world_mask"
  _register_step_plugin(name)
  try:
    qpos = np.asarray([[0.1], [-0.2]], dtype=np.float32)
    qvel = np.asarray([[0.3], [-0.1]], dtype=np.float32)
    sim = MetalSimulation(model, batch_size=2, qpos=qpos, qvel=qvel,
                          profile="integrated_euler_v1")
    plugin = sim._native_actuator_plugins[0]
    before = _copy_native_step_state(sim)
    plugin.fail_world_one = True
    status = sim.step()
    assert int(status[0].item()) == 0
    assert int(status[1].item()) != 0
    torch.testing.assert_close(plugin.ticks.cpu(), torch.tensor([1.0, 0.0]),
                               rtol=0, atol=0)
    after = _copy_native_step_state(sim)
    assert not torch.equal(after["qpos"][0], before["qpos"][0])
    for key in ("qpos", "qvel", "qacc", "act", "time", "warmstart",
                "sensordata"):
      if before[key] is not None:
        torch.testing.assert_close(after[key][1], before[key][1],
                                   rtol=0, atol=0, msg=f"failed world {key}")
    # A selected reset clears the rejected world without erasing the healthy
    # world's accepted plugin history; subsequent healthy stepping resumes.
    plugin.fail_world_one = False
    sim.reset(env_ids=[1])
    torch.testing.assert_close(plugin.ticks.cpu(), torch.tensor([1.0, 0.0]),
                               rtol=0, atol=0)
    sim.step()
    torch.testing.assert_close(plugin.ticks.cpu(), torch.tensor([2.0, 1.0]),
                               rtol=0, atol=0)
  finally:
    _unregister_step_plugin(name)


def test_typed_actuator_rejects_bad_accepted_mask_before_callback():
  plugin = _TypedActuator()
  sim = _simulation_stub(plugin)
  with pytest.raises(ValueError, match="accepted_mask"):
    sim._advance_native_actuator_plugins(torch.tensor([1], dtype=torch.int32))
  assert not plugin.advances
