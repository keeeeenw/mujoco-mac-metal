# Copyright 2026 The MuJoCo Metal contributors
# Licensed under the Apache License, Version 2.0.
"""CPU orchestration contracts for generation-scoped forward stage records."""

import pytest
import os
from types import SimpleNamespace

from mujoco_metal.forward_stages import ForwardStage, ForwardStageCoordinator


def test_split_step2_uses_captured_passive_damping_and_missing_is_an_error():
  from mujoco_metal.simulation import _select_euler_damping_input

  captured = object()
  stale = object()
  assert _select_euler_damping_input(
      True, {"passive_damping": captured}, stale, None) is captured
  assert _select_euler_damping_input(True, None, stale, None) is stale
  assert _select_euler_damping_input(False, None, None, captured) is captured
  with pytest.raises(RuntimeError, match="VEL-stage derivative is unavailable"):
    _select_euler_damping_input(True, {"passive_damping": None}, None, None)


class _Owner:
  pass


class _Tracked:
  """Minimal mutable object with the tensor version-counter protocol."""

  def __init__(self):
    self._version = 0

  def mutate(self):
    self._version += 1


def test_forward_stages_require_published_pinned_order_and_current_generation():
  owner = _Owner()
  stages = ForwardStageCoordinator(owner)
  qpos = _Tracked()
  pos = {"pose": object(), "mass": object()}
  record = stages.begin(generation=4, qpos=qpos, position=pos)
  assert stages.consume(record, ForwardStage.POS, generation=4) is pos
  with pytest.raises(ValueError, match="needs VEL"):
    stages.consume(record, ForwardStage.VEL, generation=4)
  with pytest.raises(ValueError, match="cannot skip"):
    stages.publish(record, ForwardStage.ACT, {"force": object()})

  vel = {"bias": object()}
  stages.publish(record, ForwardStage.VEL, vel, generation=4)
  stages.publish(record, ForwardStage.ACT, {"force": object()}, generation=4)
  stages.publish(record, ForwardStage.ACC, {"qacc": object()}, generation=4)
  assert stages.consume(record, ForwardStage.VEL, generation=4) is vel
  with pytest.raises(ValueError, match="stale"):
    stages.validate(record, generation=5)


def test_simulation_device_check_accepts_unindexed_backend_alias_only():
  from mujoco_metal.simulation import _device_matches
  from types import SimpleNamespace

  device = lambda kind, index=None: SimpleNamespace(type=kind, index=index)
  assert _device_matches(device("mps", 0), device("mps"))
  assert _device_matches(device("cpu"), device("cpu"))
  assert not _device_matches(device("cuda", 0), device("mps"))
  assert _device_matches(device("cuda", 1), device("cuda", 1))
  assert not _device_matches(device("cuda", 0), device("cuda", 1))


def test_force_plugin_roles_partition_before_runtime_initialization():
  from types import SimpleNamespace
  from mujoco_metal.simulation import _partition_force_plugins

  passive = SimpleNamespace(plugin_type=SimpleNamespace(value="force"))
  actuator = SimpleNamespace(plugin_type=SimpleNamespace(value="actuator"))
  sensor = SimpleNamespace(plugin_type=SimpleNamespace(value="sensor"))
  assert _partition_force_plugins((passive, actuator, sensor)) == (
      (passive,), (actuator,))


def test_force_plugins_add_once_to_their_canonical_force_buckets():
  torch = pytest.importorskip("torch")
  from mujoco_metal.extensions import NativePlugin, PluginType
  from mujoco_metal.simulation import (
      MetalSimulation, _partition_force_plugins)

  class ConstantForcePlugin(NativePlugin):
    def __init__(self, name, role, value):
      super().__init__(name, role)
      self.value = value
      self.calls = 0

    def run_device(self, state, **kwargs):
      self.calls += 1
      return torch.full((2, 1), self.value, dtype=torch.float32,
                        device="cpu")

  force = ConstantForcePlugin("passive", PluginType.FORCE, 3.25)
  actuator = ConstantForcePlugin("actuator", PluginType.ACTUATOR, -2.5)
  passive_plugins, actuator_plugins = _partition_force_plugins(
      (force, actuator))
  assert passive_plugins == (force,)
  assert actuator_plugins == (actuator,)

  sim = object.__new__(MetalSimulation)
  sim._state = SimpleNamespace(_torch=torch, _device=torch.device("cpu"))
  sim._force_plugins = passive_plugins
  sim._actuator_plugins = actuator_plugins
  qpos = torch.zeros((2, 1), dtype=torch.float32)
  qvel = torch.zeros_like(qpos)
  dynamics = {}
  passive = torch.zeros_like(qpos)
  actuator_force = torch.zeros_like(qpos)
  rhs = torch.zeros_like(qpos)
  state = SimpleNamespace()

  sim._accumulate_plugin_forces(
      sim._force_plugins, state, qpos, qvel, dynamics, passive,
      rhs_target=rhs)
  sim._accumulate_plugin_forces(
      sim._actuator_plugins, state, qpos, qvel, dynamics, actuator_force,
      rhs_target=rhs)

  assert torch.equal(passive, torch.full_like(passive, 3.25))
  assert torch.equal(actuator_force, torch.full_like(actuator_force, -2.5))
  assert torch.equal(rhs, torch.full_like(rhs, 0.75))
  assert (force.calls, actuator.calls) == (1, 1)


def test_forward_stages_replace_and_explicitly_invalidate_records():
  owner = _Owner()
  stages = ForwardStageCoordinator(owner)
  first = stages.begin(generation=0, qpos=_Tracked(), position={"pos": 1})
  second = stages.begin(generation=0, qpos=_Tracked(), position={"pos": 2})
  with pytest.raises(ValueError, match="stale"):
    stages.validate(first, generation=0)
  assert stages.consume(second, ForwardStage.POS, generation=0) == {"pos": 2}
  stages.invalidate()
  with pytest.raises(ValueError, match="stale"):
    stages.validate(second, generation=0)


def test_fwdinv_consumes_the_published_forward_result_without_solving_again():
  torch = pytest.importorskip("torch")
  from mujoco_metal.simulation import MetalSimulation

  class FakeSimulation:
    compare_forward_inverse = MetalSimulation.compare_forward_inverse
    validate_forward_stage_record = MetalSimulation.validate_forward_stage_record

    def __init__(self):
      self._state = SimpleNamespace(generation=5, _torch=torch, _device="cpu")
      self.batch_size = 1
      self._forward_stages = ForwardStageCoordinator(self)
      self._contact = self._joint_constraints = None
      self._coupled_constraints = SimpleNamespace(
          descriptor=SimpleNamespace(nr=0))

    def forward_skip(self, *args, **kwargs):
      raise AssertionError("FWDINV must not rerun forward stages")

  sim = FakeSimulation()
  record = sim._forward_stages.begin(
      generation=5, qpos=_Tracked(), position={"context": object()})
  actuation = {"qfrc_actuator": torch.zeros((1, 1))}
  constraint = {"qacc": torch.zeros((1, 1)),
                "qfrc_constraint": torch.zeros((1, 1))}
  sim._forward_stages.publish(record, ForwardStage.VEL, {"qvel": _Tracked()})
  sim._forward_stages.publish(record, ForwardStage.ACT, actuation)
  sim._forward_stages.publish(record, ForwardStage.ACC, {"qacc_smooth": object()})
  sim._forward_stages.publish(record, ForwardStage.CONSTRAINT, constraint)
  out = sim.compare_forward_inverse(record)
  torch.testing.assert_close(out, torch.zeros((1, 2)))
  with pytest.raises(ValueError, match="this record's published"):
    sim.compare_forward_inverse(record, {
        "record": record, "constraint": constraint,
        "actuation": {"qfrc_actuator": torch.zeros((1, 1))}})


def test_step1_control_callback_runs_after_position_and_velocity(monkeypatch):
  import numpy as np
  from mujoco_metal.simulation import MetalSimulation

  # Check-state behavior is exercised by its own focused suite; this fake
  # validates only split-stage ordering and does not implement DeviceState.
  monkeypatch.setattr("mujoco_metal.state_checks.mj_checkPos", lambda sim: None)
  monkeypatch.setattr("mujoco_metal.state_checks.mj_checkVel", lambda sim: None)

  calls = []

  class FakeSimulation:
    step1 = MetalSimulation.step1
    validate_forward_stage_record = MetalSimulation.validate_forward_stage_record

    def __init__(self):
      self._state = SimpleNamespace(generation=2, _qpos=_Tracked(),
                                    _qvel=_Tracked(), _time=0.0)
      self._forward_stages = ForwardStageCoordinator(self)
      self._sleep_schedule = None
      self._step1_record = None

    def _prepare_sleep_schedule(self, qpos, qvel):
      calls.append("sleep")

    def prepare_forward_position(self):
      calls.append("POS")
      return self._forward_stages.begin(
          generation=2, qpos=self._state._qpos, position={"context": object()})

    def prepare_forward_velocity(self, record, *, qvel):
      calls.append("VEL")
      self._forward_stages.publish(record, ForwardStage.VEL, {"qvel": qvel})

    def _prepare_control(self, control):
      calls.append(("control", np.asarray(control).copy()))

  sim = FakeSimulation()
  monkeypatch.setattr(
      "mujoco_metal.simulation._plugin_stage_state",
      lambda state, qpos, qvel, time: SimpleNamespace(qpos=qpos, qvel=qvel))
  control = np.asarray([[.25]], dtype=np.float32)

  def callback(stage_state, record):
    assert stage_state.qpos is sim._state._qpos
    assert record.stage == ForwardStage.VEL
    calls.append("callback")
    return control

  record = sim.step1(control_callback=callback)
  assert record.stage == ForwardStage.VEL
  assert [c if isinstance(c, str) else c[0] for c in calls] == [
      "POS", "VEL", "callback", "control"]
  with pytest.raises(RuntimeError, match="already pending"):
    sim.step1()


def test_step2_consumes_one_current_record_and_clears_pending_state():
  from mujoco_metal.simulation import MetalSimulation

  class FakeSimulation:
    step2 = MetalSimulation.step2
    validate_forward_stage_record = MetalSimulation.validate_forward_stage_record

    def __init__(self):
      self._state = SimpleNamespace(generation=3)
      self._forward_stages = ForwardStageCoordinator(self)
      self._step1_record = self._forward_stages.begin(
          generation=3, qpos=_Tracked(), position={})
      self._forward_stages.publish(
          self._step1_record, ForwardStage.VEL, {"qvel": _Tracked()})
      self.calls = []

    def step(self, steps, *, qfrc_applied=None, ctrl=None, xfrc_applied=None):
      self.calls.append((steps, qfrc_applied, ctrl))
      return "integrated"

  sim = FakeSimulation()
  assert sim.step2(ctrl="new-control", qfrc_applied="new-force") == "integrated"
  assert sim.calls == [(1, "new-force", "new-control")]
  assert sim._step1_record is None
  with pytest.raises(ValueError, match="requires a successful step1"):
    sim.step2()


def test_republishing_a_stage_discards_only_its_downstream_results():
  stages = ForwardStageCoordinator(_Owner())
  record = stages.begin(generation=2, qpos=_Tracked(), position={"pos": 1})
  first_velocity = {"bias": 2}
  stages.publish(record, ForwardStage.VEL, first_velocity)
  stages.publish(record, ForwardStage.ACT, {"force": 3})
  stages.publish(record, ForwardStage.ACC, {"qacc": 4})
  refreshed_velocity = {"bias": 5}
  stages.publish(record, ForwardStage.VEL, refreshed_velocity)
  assert record.stage == ForwardStage.VEL
  assert stages.consume(record, ForwardStage.POS, generation=2) == {"pos": 1}
  assert stages.consume(record, ForwardStage.VEL, generation=2) is refreshed_velocity
  with pytest.raises(ValueError, match="needs ACT"):
    stages.consume(record, ForwardStage.ACT, generation=2)
  stages.publish(record, ForwardStage.ACT, {"force": 6})
  stages.publish(record, ForwardStage.ACC, {"qacc": 7})
  assert stages.consume(record, ForwardStage.ACC, generation=2) == {"qacc": 7}


def test_owned_state_mutation_clears_pending_step2_record():
  """A state callback cannot strand step1 behind a stale record."""
  from mujoco_metal.forward_stages import ForwardStage
  from mujoco_metal.simulation import MetalSimulation

  class FakeSimulation:
    _invalidate_forward_stage_record = MetalSimulation._invalidate_forward_stage_record

    def __init__(self):
      self._forward_stages = ForwardStageCoordinator(self)
      self._step1_record = self._forward_stages.begin(
          generation=7, qpos=_Tracked(), position={"pose": object()})
      self._forward_stages.publish(
          self._step1_record, ForwardStage.VEL, {"qvel": _Tracked()})

  sim = FakeSimulation()
  previous_epoch = sim._forward_stages.epoch
  sim._invalidate_forward_stage_record()
  assert sim._step1_record is None
  assert sim._forward_stages.epoch == previous_epoch + 1
  with pytest.raises(ValueError, match="no forward-stage record"):
    sim._forward_stages.current(generation=7, minimum=ForwardStage.VEL)


@pytest.mark.gpu
@pytest.mark.skipif(os.getenv("MUJOCO_METAL_RUN_GPU") != "1",
                    reason="opt-in native state lifecycle")
def test_native_state_mutations_clear_split_step_record():
  import mujoco
  import numpy as np
  from mujoco_metal import MetalSimulation

  model = mujoco.MjModel.from_xml_string("""<mujoco>
    <option timestep=".002" gravity="0 0 0" solver="PGS" iterations="20"/>
    <worldbody>
      <body><joint name="slide" type="slide" axis="1 0 0"/>
        <geom type="sphere" size=".1" mass="1" contype="0" conaffinity="0"/>
      </body>
      <body name="mocap" mocap="true" pos="0 1 0"/>
    </worldbody>
    <keyframe><key name="reset" qpos=".1" qvel="0"/></keyframe>
  </mujoco>""")
  sim = MetalSimulation(model, batch_size=2, profile="integrated_euler_v1")

  sim.step1()
  assert sim._step1_record is not None
  sim.state.set_mocap(np.array([[.2, 1, 0]], dtype=np.float32),
                      np.array([[1, 0, 0, 0]], dtype=np.float32),
                      env_ids=[1])
  assert sim._step1_record is None
  with pytest.raises(ValueError, match="requires a successful step1"):
    sim.step2()

  sim.step1()
  sim.reset_to_keyframe(0, env_ids=[1])
  assert sim._step1_record is None
  with pytest.raises(ValueError, match="requires a successful step1"):
    sim.step2()

  sim.step1()
  state = sim.state.snapshot()
  sim.state.restore(state)
  assert sim._step1_record is None
  with pytest.raises(ValueError, match="requires a successful step1"):
    sim.step2()

  sim.step1()
  sim.state.copy_environment(0, 1)
  assert sim._step1_record is None
  with pytest.raises(ValueError, match="requires a successful step1"):
    sim.step2()


def test_actuation_control_override_is_temporary_even_on_stage_failure():
  torch = pytest.importorskip("torch")
  from types import MethodType, SimpleNamespace
  from mujoco_metal.simulation import MetalSimulation

  fake = SimpleNamespace(_control=torch.tensor([[1.0]]))

  def install_control(self, value):
    self._control.copy_(value)

  def evaluate(self, _record, *, act=None, time=None):
    assert act is None and time is None
    return self._control.clone()

  fake._prepare_control = MethodType(install_control, fake)
  fake._prepare_forward_actuation_current = MethodType(evaluate, fake)
  result = MetalSimulation.prepare_forward_actuation(
      fake, object(), ctrl=torch.tensor([[7.0]]))
  torch.testing.assert_close(result, torch.tensor([[7.0]]))
  torch.testing.assert_close(fake._control, torch.tensor([[1.0]]))

  def fail(self, _record, *, act=None, time=None):
    raise RuntimeError("stage failed")

  fake._prepare_forward_actuation_current = MethodType(fail, fake)
  with pytest.raises(RuntimeError, match="stage failed"):
    MetalSimulation.prepare_forward_actuation(
        fake, object(), ctrl=torch.tensor([[9.0]]))
  torch.testing.assert_close(fake._control, torch.tensor([[1.0]]))


def test_prepared_actuation_rebuilds_acc_sensor_force_buckets_each_publish():
  """ACT sensor inputs must be replaced, not accumulated across queries."""
  torch = pytest.importorskip("torch")
  from types import MethodType, SimpleNamespace
  from mujoco_metal.forward_stages import ForwardStage
  from mujoco_metal.simulation import MetalSimulation

  class Stages:
    def __init__(self, velocity):
      self.velocity = velocity
      self.published = None

    def consume(self, record, stage, *, generation):
      assert record is token and stage == ForwardStage.VEL and generation == 4
      return self.velocity

    def publish(self, record, stage, value):
      assert record is token and stage == ForwardStage.ACT
      self.published = value

  token = SimpleNamespace(qpos=torch.zeros((1, 1)))
  state = SimpleNamespace(
      _torch=torch, generation=4, _time=torch.tensor([0.0]),
      _qpos=torch.zeros((1, 1)), _qvel=torch.zeros((1, 1)),
      _qacc=torch.zeros((1, 1)), _act=torch.zeros((1, 1)))
  dynamics = {"poses": {}}
  velocity = {"dynamics": dynamics, "qvel": torch.zeros((1, 1)),
              "position_context": {}}
  fake = SimpleNamespace(
      _state=state, _forward_stages=Stages(velocity), _sensors=object(),
      _has_acc_sensors=True, _forward_stage_actuator_force=torch.zeros((1, 1)),
      _sen_act_force=torch.tensor([[0.2, 0.4]]),
      _sen_qfrc_act=torch.tensor([[0.7]]), _actuators=object(), _motor=None,
      _transmissions=None, _passive=None, _control=torch.zeros((1, 1)),
      _legacy_actuator_plugins=(), _native_actuator_plugins=(),
      _last_actuation_kin=None, _act_dot=torch.zeros((1, 1)))

  fake.validate_forward_stage_record = MethodType(
      lambda self, record, stage: record, fake)
  fake._delayed_control = MethodType(lambda self, time: torch.ones((1, 1)), fake)
  fake._lazy_sensor_scratch = MethodType(lambda self: None, fake)
  # Mirror the actual producer contract: the actuator program contributes to
  # sensor-force scratch during its ACT evaluation.
  def actuation(self, *args, **kwargs):
    self._sen_act_force.add_(torch.tensor([[0.1, 0.3]]))
    self._sen_qfrc_act.add_(torch.tensor([[0.5]]))
    return torch.tensor([[0.5]])
  fake._actuation_force = MethodType(actuation, fake)
  fake._snapshot = MethodType(lambda self: None, fake)
  fake._accumulate_plugin_forces = MethodType(lambda *args, **kwargs: None, fake)
  fake._run_native_actuator_plugins = MethodType(
      lambda *args, **kwargs: None, fake)

  first = MetalSimulation._prepare_forward_actuation_current(fake, token)
  torch.testing.assert_close(fake._sen_act_force, torch.tensor([[0.1, 0.3]]))
  torch.testing.assert_close(fake._sen_qfrc_act, torch.tensor([[0.5]]))
  torch.testing.assert_close(first["qfrc_actuator"], torch.tensor([[0.5]]))
  # Simulate a caller retaining a prior sample in the reusable arrays; the
  # next publication must still describe only the current ACT stage.
  fake._sen_act_force.fill_(9.0)
  fake._sen_qfrc_act.fill_(11.0)
  second = MetalSimulation._prepare_forward_actuation_current(fake, token)
  torch.testing.assert_close(fake._sen_act_force, torch.tensor([[0.1, 0.3]]))
  torch.testing.assert_close(fake._sen_qfrc_act, torch.tensor([[0.5]]))
  torch.testing.assert_close(second["qfrc_actuator"], torch.tensor([[0.5]]))

def test_forward_stage_record_cannot_cross_simulations():
  first_owner, second_owner = _Owner(), _Owner()
  first = ForwardStageCoordinator(first_owner)
  second = ForwardStageCoordinator(second_owner)
  record = first.begin(generation=0, qpos=_Tracked(), position={"pos": True})
  with pytest.raises(ValueError, match="stale"):
    second.validate(record, generation=0)


def test_forward_skip_reuses_only_pinned_prefix_and_republishes_suffix():
  import mujoco
  from mujoco_metal.simulation import MetalSimulation

  class FakeSimulation:
    forward_skip = MetalSimulation.forward_skip
    validate_forward_stage_record = MetalSimulation.validate_forward_stage_record

    def __init__(self):
      self._state = SimpleNamespace(generation=3)
      self._forward_stages = ForwardStageCoordinator(self)
      self.calls = []

    def prepare_forward_position(self, qpos=None, *, mocap_pos=None,
                                 mocap_quat=None, skipsensor=False):
      self.calls.append("POS")
      return self._forward_stages.begin(
          generation=self._state.generation, qpos=qpos,
          mocap_pos=mocap_pos, mocap_quat=mocap_quat,
          position={"skipsensor": skipsensor})

    def prepare_forward_velocity(self, record, qvel=None, *, skipsensor=False):
      self.calls.append("VEL")
      result = {"qvel": qvel, "skipsensor": skipsensor}
      self._forward_stages.publish(record, ForwardStage.VEL, result)
      return result

    def prepare_forward_actuation(self, record, *, ctrl=None):
      self.calls.append("ACT")
      result = {"ctrl": ctrl}
      self._forward_stages.publish(record, ForwardStage.ACT, result)
      return result

    def prepare_forward_acceleration(self, record):
      self.calls.append("ACC")
      result = {"qacc_smooth": object()}
      self._forward_stages.publish(record, ForwardStage.ACC, result)
      return result

    def prepare_forward_constraint(self, record, *, skipsensor=False):
      self.calls.append("CONSTRAINT")
      result = {"qacc": object(), "qfrc_constraint": object(),
                "status": object(), "skipsensor": skipsensor}
      self._forward_stages.publish(record, ForwardStage.CONSTRAINT, result)
      return result

  sim = FakeSimulation()
  qpos, qvel = _Tracked(), _Tracked()
  result = sim.forward_skip(mujoco.mjtStage.mjSTAGE_NONE,
                            qpos=qpos, qvel=qvel)
  assert sim.calls == ["POS", "VEL", "ACT", "ACC", "CONSTRAINT"]
  assert result["record"].stage == ForwardStage.CONSTRAINT

  sim.calls.clear()
  pos_record = sim.prepare_forward_position(qpos)
  sim.calls.clear()
  sim.forward_skip(mujoco.mjtStage.mjSTAGE_POS,
                   record=pos_record, qvel=qvel)
  assert sim.calls == ["VEL", "ACT", "ACC", "CONSTRAINT"]

  sim.calls.clear()
  vel_record = sim.prepare_forward_position(qpos)
  sim.prepare_forward_velocity(vel_record, qvel)
  sim.calls.clear()
  sim.forward_skip(mujoco.mjtStage.mjSTAGE_VEL,
                   record=vel_record, qvel=qvel)
  assert sim.calls == ["ACT", "ACC", "CONSTRAINT"]

  sim.calls.clear()
  refreshed_record = sim.prepare_forward_position(qpos)
  mutable_qvel = _Tracked()
  sim.prepare_forward_velocity(refreshed_record, mutable_qvel)
  mutable_qvel.mutate()
  sim.calls.clear()
  sim.forward_skip(mujoco.mjtStage.mjSTAGE_POS,
                   record=refreshed_record, qvel=mutable_qvel)
  assert sim.calls == ["VEL", "ACT", "ACC", "CONSTRAINT"]


def test_forward_skip_requires_cached_prefix_and_matching_velocity_input():
  import mujoco
  from mujoco_metal.simulation import MetalSimulation

  class FakeSimulation:
    forward_skip = MetalSimulation.forward_skip
    validate_forward_stage_record = MetalSimulation.validate_forward_stage_record

    def __init__(self):
      self._state = SimpleNamespace(generation=1)
      self._forward_stages = ForwardStageCoordinator(self)

    def prepare_forward_position(self, qpos=None, *, mocap_pos=None,
                                 mocap_quat=None, skipsensor=False):
      return self._forward_stages.begin(
          generation=1, qpos=qpos, mocap_pos=mocap_pos,
          mocap_quat=mocap_quat, position={})

    def prepare_forward_velocity(self, record, qvel=None, *, skipsensor=False):
      self._forward_stages.publish(record, ForwardStage.VEL, {"qvel": qvel})
      return record.values[ForwardStage.VEL]

  sim = FakeSimulation()
  with pytest.raises(ValueError, match="no forward-stage record"):
    sim.forward_skip(mujoco.mjtStage.mjSTAGE_POS)
  record = sim.prepare_forward_position()
  sim.prepare_forward_velocity(record, _Tracked())
  with pytest.raises(ValueError, match="does not match"):
    sim.forward_skip(mujoco.mjtStage.mjSTAGE_VEL,
                     record=record, qvel=_Tracked())


def test_cached_stage_reuse_rejects_in_place_qpos_and_qvel_mutation():
  stages = ForwardStageCoordinator(_Owner())
  qpos, qvel = _Tracked(), _Tracked()
  record = stages.begin(generation=9, qpos=qpos, position={"pose": 1})
  stages.publish(record, ForwardStage.VEL, {"qvel": qvel})
  stages.validate(record, generation=9, minimum=ForwardStage.VEL)
  qpos.mutate()
  with pytest.raises(ValueError, match="qpos was mutated"):
    stages.validate(record, generation=9, minimum=ForwardStage.POS)

  record = stages.begin(generation=10, qpos=_Tracked(), position={"pose": 1})
  qvel = _Tracked()
  stages.publish(record, ForwardStage.VEL, {"qvel": qvel})
  qvel.mutate()
  with pytest.raises(ValueError, match="qvel was mutated"):
    stages.validate(record, generation=10, minimum=ForwardStage.VEL)
  # Republish VEL after POS uses the current qvel version instead of being
  # blocked by the stale downstream VEL input.
  stages.publish(record, ForwardStage.VEL, {"qvel": qvel})
  stages.validate(record, generation=10, minimum=ForwardStage.VEL)

  record = stages.begin(generation=11, qpos=_Tracked(), position={"pose": 1})
  record.qpos = _Tracked()
  with pytest.raises(ValueError, match="qpos input identity was replaced"):
    stages.validate(record, generation=11, minimum=ForwardStage.POS)

  record = stages.begin(generation=12, qpos=_Tracked(), position={"pose": 1})
  qvel = _Tracked()
  stages.publish(record, ForwardStage.VEL, {"qvel": qvel})
  record.values[ForwardStage.VEL]["qvel"] = _Tracked()
  with pytest.raises(ValueError, match="qvel input identity was replaced"):
    stages.validate(record, generation=12, minimum=ForwardStage.VEL)


def test_cached_stage_reuse_fails_closed_without_mutation_counter():
  stages = ForwardStageCoordinator(_Owner())
  record = stages.begin(generation=2, qpos=object(), position={"pose": 1})
  with pytest.raises(ValueError, match="mutation counter"):
    stages.validate(record, generation=2, minimum=ForwardStage.POS)


@pytest.mark.parametrize("bad_stage", [True, 1.5, "1"])
def test_skip_apis_reject_nonintegral_stage_values_before_work(bad_stage):
  import mujoco
  from mujoco_metal.simulation import MetalSimulation

  class FakeSimulation:
    forward_skip = MetalSimulation.forward_skip
    inverse_skip = MetalSimulation.inverse_skip

    def prepare_forward_position(self, *args, **kwargs):
      pytest.fail("invalid stage must be rejected before physics")

  sim = FakeSimulation()
  with pytest.raises(TypeError, match="stage integer"):
    sim.forward_skip(bad_stage)
  with pytest.raises(TypeError, match="stage integer"):
    sim.inverse_skip(bad_stage)
  # Pybind stage enums remain an accepted input despite not necessarily being
  # registered as numbers.Integral by Python's runtime.
  assert isinstance(mujoco.mjtStage.mjSTAGE_NONE, mujoco.mjtStage)


def test_failed_stage_publish_does_not_change_record_or_bindings():
  stages = ForwardStageCoordinator(_Owner())
  record = stages.begin(generation=6, qpos=_Tracked(), position={"pos": 1})
  snapshot = (record.stage, dict(record.values), dict(record.input_tensors),
              dict(record.input_versions))
  with pytest.raises(ValueError, match="cannot skip"):
    stages.publish(record, ForwardStage.ACT, {"force": 1})
  with pytest.raises(TypeError, match="dictionary"):
    stages.publish(record, ForwardStage.VEL, ["bad"])
  assert record.stage == snapshot[0]
  assert record.values == snapshot[1]
  assert record.input_tensors == snapshot[2]
  assert record.input_versions == snapshot[3]


@pytest.mark.gpu
@pytest.mark.skipif(os.getenv("MUJOCO_METAL_RUN_GPU") != "1",
                    reason="requires explicit native stage qualification")
def test_public_prepared_stage_chain_matches_pinned_forward():
  import mujoco
  import numpy as np
  from mujoco_metal import MetalSimulation

  model = mujoco.MjModel.from_xml_string('''<mujoco>
    <option timestep=".001" gravity="0 0 -9.81" solver="PGS" iterations="40">
      <flag contact="disable"/>
    </option>
    <worldbody>
      <body name="left" pos="0 0 1">
        <joint name="left_slide" type="slide" axis="1 0 0" damping=".2"/>
        <geom type="sphere" size=".1" mass="1"/>
      </body>
      <body name="right" pos="1 0 1">
        <joint name="right_slide" type="slide" axis="0 1 0" damping=".1"/>
        <geom type="sphere" size=".1" mass="2"/>
      </body>
    </worldbody>
    <equality><joint name="tie" joint1="left_slide" joint2="right_slide"
      polycoef="0 1 0 0 0" solref=".02 1"/></equality>
  </mujoco>''')
  reference = mujoco.MjData(model)
  reference.qpos[:] = [.03, -.01]
  reference.qvel[:] = [.2, -.1]
  mujoco.mj_forward(model, reference)
  sim = MetalSimulation(
      model, batch_size=1, qpos=reference.qpos[None].astype(np.float32),
      qvel=reference.qvel[None].astype(np.float32),
      profile="integrated_scalable_v1")

  record = sim.prepare_forward_position()
  pos = sim.validate_forward_stage_record(record, "POS")
  assert "context" in pos.values[next(iter(pos.values))]
  velocity = sim.prepare_forward_velocity(record)
  actuation = sim.prepare_forward_actuation(record)
  acceleration = sim.prepare_forward_acceleration(record)
  constrained = sim.prepare_forward_constraint(record)
  assert actuation["qfrc_actuator"].shape == (1, model.nv)
  pos_result = pos.values[ForwardStage.POS]
  np.testing.assert_allclose(
      pos_result["dynamics"]["mass_blocks"].cpu().numpy()[0],
      _pack_cpu_mass(model, reference), rtol=8e-5, atol=8e-5)
  np.testing.assert_allclose(
      velocity["qfrc_bias"].cpu().numpy()[0], reference.qfrc_bias,
      rtol=8e-5, atol=8e-5)
  np.testing.assert_allclose(
      acceleration["qacc_smooth"].cpu().numpy()[0], reference.qacc_smooth,
      rtol=2e-4, atol=2e-5)
  np.testing.assert_allclose(
      constrained["qacc"].cpu().numpy()[0], reference.qacc,
      rtol=3e-4, atol=3e-5)
  assert sim.validate_forward_stage_record(record, "CONSTRAINT").stage == ForwardStage.CONSTRAINT


@pytest.mark.gpu
@pytest.mark.skipif(os.getenv("MUJOCO_METAL_RUN_GPU") != "1",
                    reason="requires explicit native FWDINV qualification")
def test_step_fwdinv_flag_matches_pinned_two_worlds():
  import mujoco
  import numpy as np
  from mujoco_metal import MetalSimulation

  model = mujoco.MjModel.from_xml_string('''<mujoco>
    <option timestep=".001" gravity="0 0 -9.81" solver="PGS" iterations="40">
      <flag contact="disable"/>
    </option>
    <worldbody>
      <body name="left" pos="0 0 1">
        <joint name="left_slide" type="slide" axis="1 0 0" damping=".2"/>
        <geom type="sphere" size=".1" mass="1"/>
      </body>
      <body name="right" pos="1 0 1">
        <joint name="right_slide" type="slide" axis="0 1 0" damping=".1"/>
        <geom type="sphere" size=".1" mass="2"/>
      </body>
    </worldbody>
    <equality><joint name="tie" joint1="left_slide" joint2="right_slide"
      polycoef="0 1 0 0 0" solref=".02 1"/></equality>
  </mujoco>''')
  model.opt.enableflags |= int(mujoco.mjtEnableBit.mjENBL_FWDINV)
  qpos = np.asarray([[.03, -.01], [-.02, .04]], dtype=np.float32)
  qvel = np.asarray([[.2, -.1], [-.15, .23]], dtype=np.float32)
  expected = []
  for i in range(2):
    reference = mujoco.MjData(model)
    reference.qpos[:] = qpos[i]
    reference.qvel[:] = qvel[i]
    mujoco.mj_step(model, reference)
    expected.append(np.asarray(reference.solver_fwdinv, dtype=np.float32))
  sim = MetalSimulation(model, batch_size=2, qpos=qpos, qvel=qvel,
                        profile="integrated_scalable_v1")
  sim.step()
  np.testing.assert_allclose(sim.solver_fwdinv.cpu().numpy(), expected,
                             rtol=2e-3, atol=2e-4)


@pytest.mark.gpu
@pytest.mark.skipif(os.getenv("MUJOCO_METAL_RUN_GPU") != "1",
                    reason="requires explicit native step1/step2 qualification")
def test_step1_step2_energy_and_callback_order_match_pinned():
  import mujoco
  import numpy as np
  from mujoco_metal import MetalSimulation

  model = mujoco.MjModel.from_xml_string('''<mujoco>
    <option timestep=".001" gravity="0 0 -9.81" solver="PGS" iterations="40">
      <flag contact="disable" energy="enable"/>
    </option>
    <worldbody><body name="body" pos="0 0 1">
      <joint name="hinge" type="hinge" axis="0 1 0" damping=".15"
             stiffness="2" springref=".1"/>
      <geom type="sphere" size=".1" mass="1"/>
    </body></worldbody>
    <actuator><motor name="motor" joint="hinge" gear="1"/></actuator>
  </mujoco>''')
  qpos = np.asarray([[.3], [-.2]], dtype=np.float32)
  qvel = np.asarray([[.4], [-.15]], dtype=np.float32)
  sim = MetalSimulation(model, batch_size=2, qpos=qpos, qvel=qvel,
                        profile="integrated_scalable_v1")
  callback_order = []
  controls = np.asarray([[.7], [-.4]], dtype=np.float32)

  def callback(stage_state, record):
    callback_order.append(("control", record.stage))
    assert record.stage == ForwardStage.VEL
    assert stage_state.qpos.shape == (2, 1)
    return controls

  refs = []
  for world in range(2):
    data = mujoco.MjData(model)
    data.qpos[:] = qpos[world]
    data.qvel[:] = qvel[world]
    mujoco.mj_step1(model, data)
    refs.append(data)
  record = sim.step1(control_callback=callback)
  assert callback_order == [("control", ForwardStage.VEL)]
  assert record.stage == ForwardStage.VEL
  expected_energy = np.asarray([data.energy.copy() for data in refs])
  np.testing.assert_allclose(sim.energy.cpu().numpy(), expected_energy,
                             rtol=4e-5, atol=3e-6)
  for world, data in enumerate(refs):
    data.ctrl[:] = controls[world]
    mujoco.mj_step2(model, data)
  sim.step2()
  np.testing.assert_allclose(sim.state.qpos.cpu().numpy(),
                             np.asarray([d.qpos for d in refs]),
                             rtol=4e-4, atol=4e-5)
  np.testing.assert_allclose(sim.state.qvel.cpu().numpy(),
                             np.asarray([d.qvel for d in refs]),
                             rtol=5e-4, atol=5e-5)
  assert sim.energy.shape == (2, 2)


def _pack_cpu_mass(model, data):
  import mujoco
  import numpy as np
  from mujoco_metal.smooth_metal import compile_tree_mass_layout
  layout = compile_tree_mass_layout(model.body_treeid, model.dof_bodyid)
  full = np.zeros((model.nv, model.nv), dtype=np.float64)
  mujoco.mj_fullM(model, data, full)
  packed = np.empty(int(layout["nnz"]))
  for component, width in enumerate(layout["component_dofnum"]):
    begin = int(layout["component_dof_offsets"][component])
    dofs = np.asarray(layout["component_dof_ids"][begin:begin + int(width)])
    offset = int(layout["component_mass_offsets"][component])
    packed[offset:offset + int(width)**2] = full[np.ix_(dofs, dofs)].reshape(-1)
  return packed


def test_sleep_presolve_forwards_supplied_mocap_to_fk_without_relowering():
  torch = pytest.importorskip("torch")
  from types import SimpleNamespace
  from mujoco_metal.simulation import MetalSimulation

  supplied_pos = torch.tensor([[[.2, -.1, .3]]], dtype=torch.float32)
  supplied_quat = torch.tensor([[[1., 0., 0., 0.]]], dtype=torch.float32)
  state = SimpleNamespace(
      _mpos=torch.zeros_like(supplied_pos),
      _mquat=supplied_quat.clone(),
      _eq_active=None,
      _status=torch.zeros((1,), dtype=torch.int32))
  scheduler = SimpleNamespace(tree_awake=torch.ones((1, 1), dtype=torch.int32))
  captured = {}
  poses = {"body_pos": object()}

  def fk(qpos, mocap_pos, mocap_quat, *, tree_awake):
    captured.update(qpos=qpos, mocap_pos=mocap_pos,
                    mocap_quat=mocap_quat, tree_awake=tree_awake)
    return poses

  def wake(self, *args, **kwargs):
    captured["wake"] = kwargs
    captured.setdefault("wake_calls", []).append(kwargs)

  sim = MetalSimulation.__new__(MetalSimulation)
  sim._state = state
  sim._sleep_schedule = scheduler
  mismatch = torch.zeros((1, 1), dtype=torch.int32)
  sim._smooth = SimpleNamespace(_fk=SimpleNamespace(
      run_device=fk, pose_mismatch=lambda: mismatch))
  sim._coupled_constraints = None
  sim._flex = None
  sim._applied_force = None
  sim._body_wrench = None
  sim._mjmodel = SimpleNamespace(opt=SimpleNamespace(disableflags=0))
  scheduler.wake_before_solve = wake.__get__(scheduler, SimpleNamespace)
  result = MetalSimulation._prepare_sleep_schedule(
      sim, torch.zeros((1, 0), dtype=torch.float32),
      torch.zeros((1, 0), dtype=torch.float32),
      mocap_pos=supplied_pos, mocap_quat=supplied_quat)
  assert result is poses
  assert captured["mocap_pos"] is supplied_pos
  assert captured["mocap_quat"] is supplied_quat
  assert captured["tree_awake"] is scheduler.tree_awake
  assert any(call.get("pose_mismatch") is mismatch for call in captured["wake_calls"])
