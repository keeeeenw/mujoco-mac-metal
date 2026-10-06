# Copyright 2026 The MuJoCo Metal contributors
# Licensed under the Apache License, Version 2.0.
"""Pinned forwardSkip(POS) witnesses and actual native stage consumers."""

import os

import mujoco
import numpy as np
import pytest


XML = """<mujoco><option gravity="0 -1 -9.81"/>
  <worldbody><body pos=".2 .1 .3">
    <joint type="hinge" axis="0 1 0"/>
    <geom type="box" pos=".2 0 .1" size=".1 .05 .08" mass="2"/>
    <body pos=".4 .1 .2"><joint type="hinge" axis="1 0 0"/>
      <geom type="box" pos="0 .2 .1" size=".05 .1 .08" mass="1"/>
    </body></body></worldbody></mujoco>"""


def _reference():
  model = mujoco.MjModel.from_xml_string(XML)
  data = mujoco.MjData(model)
  data.qpos[:] = [.75, -.4]
  data.qvel[:] = [.6, -.3]
  mujoco.mj_forward(model, data)
  cached_mass = np.zeros((model.nv, model.nv))
  mujoco.mj_fullM(model, data, cached_mass)
  cached_poses = data.geom_xpos.copy()
  # RK4 restores X0 while keeping the final substage's POS workspaces.
  data.qpos[:] = [0, .1]
  data.qvel[:] = [.1, -.7]
  mujoco.mj_forwardSkip(model, data, mujoco.mjtStage.mjSTAGE_POS, 0)
  return model, data, cached_mass, cached_poses


def test_pinned_velocity_refresh_uses_cached_positions_and_mass():
  model, data, mass, poses = _reference()
  np.testing.assert_array_equal(data.geom_xpos, poses)
  after = np.zeros_like(mass)
  mujoco.mj_fullM(model, data, after)
  np.testing.assert_array_equal(after, mass)
  bias = data.qfrc_bias.copy()
  mujoco.mj_forward(model, data)
  assert np.max(np.abs(data.geom_xpos - poses)) > .1
  assert np.max(np.abs(data.qfrc_bias - bias)) > .1


def test_position_context_rejects_foreign_and_replaced_workspace_before_device_use():
  from mujoco_metal.smooth_metal import MetalSmoothDynamics

  smooth = MetalSmoothDynamics.__new__(MetalSmoothDynamics)
  smooth._position_context_epoch = 2
  for context in ({"_owner": object(), "_epoch": 2},
                  {"_owner": smooth, "_epoch": 1}):
    with pytest.raises(ValueError, match="another or replaced workspace"):
      smooth.run_velocity_device(context, None)


def test_cached_compaction_map_storage_preserves_logical_empty_and_batch_maps():
  from mujoco_metal.coupled_constraints import _copy_compaction_map_storage
  from mujoco_metal.row_compaction import CompactionMap

  empty = CompactionMap(
      packed_to_logical=np.empty((2, 0), dtype=np.int32),
      logical_to_packed=np.empty((2, 0), dtype=np.int32),
      active_count=np.zeros(2, dtype=np.int32),
      overflow=np.zeros(2, dtype=np.int32))
  packed_cache = np.asarray([9], dtype=np.int32)
  reverse_cache = np.asarray([8], dtype=np.int32)
  _copy_compaction_map_storage(packed_cache, empty.packed_to_logical)
  _copy_compaction_map_storage(reverse_cache, empty.logical_to_packed)
  np.testing.assert_array_equal(packed_cache, [-1])
  np.testing.assert_array_equal(reverse_cache, [-1])
  # Restoring an empty logical view from a physically padded cache is a no-op.
  _copy_compaction_map_storage(empty.packed_to_logical, packed_cache[:0])
  _copy_compaction_map_storage(empty.logical_to_packed, reverse_cache[:0])
  assert empty.packed_to_logical.shape == (2, 0)
  assert empty.logical_to_packed.shape == (2, 0)

  maps = CompactionMap(
      packed_to_logical=np.asarray([[1, -1], [0, 1]], dtype=np.int32),
      logical_to_packed=np.asarray([[-1, 0], [0, 1]], dtype=np.int32),
      active_count=np.asarray([1, 2], dtype=np.int32),
      overflow=np.zeros(2, dtype=np.int32))
  packed_cache = np.full((4,), -1, dtype=np.int32)
  reverse_cache = np.full((4,), -1, dtype=np.int32)
  _copy_compaction_map_storage(packed_cache, maps.packed_to_logical)
  _copy_compaction_map_storage(reverse_cache, maps.logical_to_packed)
  restored_packed = np.full((4,), -2, dtype=np.int32).reshape(2, 2)
  restored_reverse = np.full((4,), -2, dtype=np.int32).reshape(2, 2)
  _copy_compaction_map_storage(restored_packed, packed_cache[:4].reshape(2, 2))
  _copy_compaction_map_storage(restored_reverse, reverse_cache[:4].reshape(2, 2))
  np.testing.assert_array_equal(restored_packed, maps.packed_to_logical)
  np.testing.assert_array_equal(restored_reverse, maps.logical_to_packed)


def test_prepare_position_adds_fixed_and_spatial_armature_before_record_capture():
  """Exercise the public POS orchestrator with array-backed stage stand-ins."""
  from types import MethodType, SimpleNamespace
  from mujoco_metal.forward_stages import ForwardStage, ForwardStageCoordinator
  from mujoco_metal.simulation import MetalSimulation

  class Tensor:
    def __init__(self, value):
      self._value = np.asarray(value, dtype=np.float32)
      self._version = 0
      self.dtype = "float32"
      self.device = SimpleNamespace(type="cpu", index=None)
      self.ndim = self._value.ndim
    @property
    def shape(self): return self._value.shape
    def is_contiguous(self): return self._value.flags.c_contiguous
    def __getitem__(self, key):
      view = Tensor(self._value[key])
      view._value = self._value[key]
      view._version = self._version
      return view
    def view(self, *shape):
      view = Tensor(self._value.reshape(*shape))
      view._value = self._value.reshape(*shape)
      view._version = self._version
      return view
    def reshape(self, *shape): return self.view(*shape)
    def zero_(self): self._value.fill(0); self._version += 1; return self
    def add_(self, other):
      self._value += other._value
      self._version += 1
      return self
    def numpy(self): return self._value

  torch = SimpleNamespace(Tensor=Tensor, float32="float32")

  simulation = MetalSimulation.__new__(MetalSimulation)
  pose = {"body_pos": Tensor(np.zeros((1, 1, 3)))}
  fixed_armature = Tensor([[[.5]]])
  spatial_armature = Tensor([[[.25]]])
  simulation._state = SimpleNamespace(
      _torch=torch, _device=SimpleNamespace(type="cpu", index=None),
      _qpos=Tensor([[0.]]), _qvel=Tensor([[0.]]),
      _mpos=None, _mquat=None, generation=0)
  simulation._mjmodel = SimpleNamespace(nq=1, nv=1)
  simulation.batch_size = 1
  simulation._component_mass_enabled = False
  simulation._forward_stages = ForwardStageCoordinator(simulation)
  simulation._forward_position_epoch = 0
  simulation._sleep_schedule = None
  simulation._spatial_cache_key = None
  simulation._prepare_sleep_schedule = MethodType(
      lambda self, _qpos, _qvel: pose, simulation)
  simulation._component_tendon_jacobian = MethodType(
      lambda self, _qvel, _poses: None, simulation)
  def run_position_device(self, *_args, **_kwargs):
    # A fresh base mass models the real POS producer on every invocation.
    return {"mass_matrix": Tensor([[[2.]]]), "poses": pose,
            "cvel": Tensor(np.zeros((1, 1, 6))),
            "root_com": Tensor(np.zeros((1, 1, 3))),
            "cdof": Tensor(np.zeros((1, 1, 6))),
            "cdof_dot": Tensor(np.zeros((1, 1, 6)))}
  simulation._smooth = SimpleNamespace(
      _fk=SimpleNamespace(run_device=lambda *_args, **_kwargs: pose),
      _workspace={"qvel": Tensor([0.])},
      run_position_device=MethodType(run_position_device, object()))
  simulation._tendons = SimpleNamespace(
      run_device=lambda _qpos, _qvel: (Tensor([[0.]]), None,
                                       fixed_armature),
      _last_length=Tensor([[0.]]))
  simulation._spatial_tendons = SimpleNamespace(
      run_kinematics=lambda _qvel, _poses: {"length": Tensor([[0.]]),
                                            "jacobian": Tensor([[[0.]]])},
      run_forces=lambda _kin, *, include_armature: (
          Tensor([[0.]]), None,
          spatial_armature if include_armature else None))
  simulation._actuators = None
  simulation._transmissions = None
  simulation._coupled_constraints = None
  simulation._flex = None
  simulation._contact = None
  simulation._joint_constraints = None
  simulation._native_plugins = ()
  simulation._sensors = None
  simulation._raw_sensordata = None
  simulation._energy = None
  simulation._capture_forward_position = MethodType(
      lambda self, qpos, dynamics: {"smooth": {}, "fixed_tendon_length":
                                    self._tendons._last_length,
                                    "spatial_tendons": self._spatial_kin},
      simulation)

  first = simulation.prepare_forward_position()
  np.testing.assert_allclose(
      first.values[ForwardStage.POS]["dynamics"]["mass_matrix"].numpy(),
      [[[2.75]]])
  second = simulation.prepare_forward_position()
  np.testing.assert_allclose(
      second.values[ForwardStage.POS]["dynamics"]["mass_matrix"].numpy(),
      [[[2.75]]])


ARMATURE_STAGE_XML = """<mujoco><option gravity="0 0 0"/>
  <worldbody>
    <site name="world_anchor" pos="0 0 0"/>
    <body pos=".3 .2 .1"><joint name="hinge" type="hinge" axis="0 1 0"/>
      <geom type="capsule" fromto="0 0 0 .2 0 .1" size=".04" mass="1.3"/>
      <site name="body_anchor" pos=".25 .1 .15"/>
    </body>
  </worldbody>
  <tendon>
    <fixed name="fixed" armature=".17"><joint joint="hinge" coef="1.4"/></fixed>
    <spatial name="spatial" armature=".23">
      <site site="world_anchor"/><site site="body_anchor"/>
    </spatial>
  </tendon></mujoco>"""


@pytest.mark.gpu
@pytest.mark.skipif(os.getenv("MUJOCO_METAL_RUN_GPU") != "1",
                    reason="requires explicit native prepared-tendon qualification")
@pytest.mark.parametrize(("profile", "sparse"), [
    ("contact_free_transmission_euler_v1", False),
    ("integrated_scalable_v1", True),
])
def test_native_prepared_position_mass_contains_fixed_and_spatial_armature(
    profile, sparse):
  from mujoco_metal import MetalSimulation
  from mujoco_metal.forward_stages import ForwardStage
  from mujoco_metal.smooth_metal import compile_tree_mass_layout

  model = mujoco.MjModel.from_xml_string(ARMATURE_STAGE_XML)
  model.opt.disableflags |= int(mujoco.mjtDisableBit.mjDSBL_CONTACT)
  data = mujoco.MjData(model)
  data.qpos[:] = [.41]
  data.qvel[:] = [-.32]
  mujoco.mj_forward(model, data)
  dense = np.zeros((model.nv, model.nv), dtype=np.float64)
  mujoco.mj_fullM(model, data, dense)
  sim = MetalSimulation(
      model, batch_size=1, qpos=np.asarray(data.qpos, dtype=np.float32)[None],
      qvel=np.asarray(data.qvel, dtype=np.float32)[None], profile=profile)
  record = sim.prepare_forward_position()
  dynamics = record.values[ForwardStage.POS]["dynamics"]
  if not sparse:
    actual = dynamics["mass_matrix"][0].cpu().numpy()
    np.testing.assert_allclose(actual, dense, rtol=6e-5, atol=6e-6)
    return
  layout = compile_tree_mass_layout(model.body_treeid, model.dof_bodyid)
  packed = (dynamics["mass_blocks"] + dynamics["tendon_armature_blocks"])[0]
  actual = np.zeros_like(dense, dtype=np.float64)
  for component, width in enumerate(layout["component_dofnum"]):
    begin = int(layout["component_dof_offsets"][component])
    dofs = np.asarray(layout["component_dof_ids"][begin:begin + int(width)])
    offset = int(layout["component_mass_offsets"][component])
    values = packed[offset:offset + int(width) ** 2].cpu().numpy()
    actual[np.ix_(dofs, dofs)] = values.reshape(int(width), int(width))
  np.testing.assert_allclose(actual, dense, rtol=8e-5, atol=8e-6)


@pytest.mark.gpu
@pytest.mark.skipif(os.getenv("MUJOCO_METAL_RUN_GPU") != "1",
                    reason="requires explicit dense kinetic-sensor qualification")
def test_native_dense_kinetic_sensor_uses_fixed_and_spatial_tendon_armature():
  from mujoco_metal import MetalSimulation

  xml = ARMATURE_STAGE_XML.replace(
      "</tendon></mujoco>", "</tendon><sensor><e_kinetic/></sensor></mujoco>")
  model = mujoco.MjModel.from_xml_string(xml)
  model.opt.disableflags |= int(mujoco.mjtDisableBit.mjDSBL_CONTACT)
  data = mujoco.MjData(model)
  data.qpos[:] = [.41]
  data.qvel[:] = [-.32]
  mujoco.mj_forward(model, data)
  sim = MetalSimulation(
      model, batch_size=1, qpos=np.asarray(data.qpos, dtype=np.float32)[None],
      qvel=np.asarray(data.qvel, dtype=np.float32)[None],
      profile="integrated_euler_v1")
  actual = sim.sensor_values()[0].cpu().numpy()
  np.testing.assert_allclose(actual, data.sensordata, rtol=5e-5, atol=5e-6)


@pytest.mark.parametrize("invalid", ["foreign", "superseded", "restore", "missing"])
def test_simulation_cached_stage_rejects_invalid_context_before_any_device_write(invalid):
  from types import SimpleNamespace
  from mujoco_metal.simulation import MetalSimulation

  simulation = MetalSimulation.__new__(MetalSimulation)
  # Deliberately omit the torch runtime, scheduler and outputs. A bad context
  # must be rejected before any of those can be accessed or mutated.
  simulation._state = SimpleNamespace(generation=3)
  simulation._forward_position_epoch = 2
  for name in ("_tendons", "_spatial_tendons", "_actuators", "_transmissions",
               "_coupled_constraints", "_flex"):
    setattr(simulation, name, None)
  context = {"_owner": simulation, "_epoch": 2, "_generation": 3, "smooth": {}}
  if invalid == "foreign":
    context["_owner"] = object()
  elif invalid == "superseded":
    context["_epoch"] = 1
  elif invalid == "restore":
    context["_generation"] = 2
  else:
    simulation._tendons = object()
  with pytest.raises(ValueError, match="context"):
    simulation._acceleration(None, None, position_context=context,
                             skip_sleep_prepare=True)


def test_failed_position_capture_invalidates_previous_token():
  from types import SimpleNamespace
  from mujoco_metal.simulation import MetalSimulation

  simulation = MetalSimulation.__new__(MetalSimulation)
  simulation._state = SimpleNamespace(generation=3)
  simulation._forward_position_epoch = 2
  for name in ("_tendons", "_spatial_tendons", "_actuators", "_transmissions",
               "_flex", "_contact", "_joint_constraints"):
    setattr(simulation, name, None)
  def fail_capture():
    raise RuntimeError("injected row-cache failure")
  simulation._coupled_constraints = SimpleNamespace(capture_position_context=fail_capture)
  simulation._smooth = SimpleNamespace(position_context=lambda _dynamics: {})
  previous = {"_owner": simulation, "_epoch": 2, "_generation": 3,
              "smooth": {}, "coupled": {}}
  simulation._validate_forward_position(previous)
  with pytest.raises(RuntimeError, match="row-cache failure"):
    simulation._capture_forward_position(None, {})
  with pytest.raises(ValueError, match="stale"):
    simulation._validate_forward_position(previous)


COMPOSED_XML = """<mujoco><option gravity="0 -1 -9.81"/>
  <worldbody><site name="anchor" pos=".6 .4 .8"/>
  <body pos=".2 .1 .3" gravcomp="1">
    <joint name="h1" type="hinge" axis="0 1 0" stiffness=".3" damping=".2"
      actuatorgravcomp="true"/>
    <geom type="box" pos=".2 0 .1" size=".1 .05 .08" mass="2" contype="0" conaffinity="0"/>
    <body pos=".4 .1 .2"><joint name="h2" type="hinge" axis="1 0 0"/>
      <geom type="box" pos="0 .2 .1" size=".05 .1 .08" mass="1" contype="0" conaffinity="0"/>
      <site name="moving" pos=".1 .2 .3"/>
    </body></body></worldbody>
  <tendon><fixed name="fixed" stiffness="3" damping=".4" armature=".15">
    <joint joint="h1" coef="1.4"/><joint joint="h2" coef="-.5"/></fixed>
    <spatial name="path" stiffness="2" damping=".3" armature=".1">
      <site site="anchor"/><site site="moving"/></spatial></tendon>
  <actuator><general name="filtered" joint="h1" dyntype="filter" dynprm=".05"
    gaintype="fixed" gainprm="1" biastype="affine" biasprm="0 -8 -.5"/>
    <motor name="path_motor" tendon="path" gear="2"/></actuator>
  <sensor><tendonpos tendon="path"/><tendonvel tendon="fixed"/>
    <tendonvel tendon="path"/><actuatorpos actuator="filtered"/>
    <actuatorvel actuator="filtered"/><actuatorvel actuator="path_motor"/>
    <actuatorfrc actuator="filtered"/></sensor>
  </mujoco>"""


def _composed_reference():
  model = mujoco.MjModel.from_xml_string(COMPOSED_XML)
  data = mujoco.MjData(model)
  data.qpos[:] = [.6, -.4]
  data.qvel[:] = [.3, .5]
  data.act[:] = [.1]
  data.ctrl[:] = [.2, -.15]
  mujoco.mj_forward(model, data)
  position = {
      "mass": np.zeros((model.nv, model.nv)),
      "tendon_length": data.ten_length.copy(),
      "tendon_J": data.ten_J.copy(),
      "actuator_length": data.actuator_length.copy(),
      "geom_pos": data.geom_xpos.copy(),
  }
  mujoco.mj_fullM(model, data, position["mass"])
  data.qpos[:] = [.1, .2]
  data.qvel[:] = [-.4, .1]
  data.act[:] = [.25]
  mujoco.mj_forwardSkip(model, data, mujoco.mjtStage.mjSTAGE_POS, 0)
  return model, data, position


def test_pinned_composed_velocity_refresh_retains_mass_paths_and_transmissions():
  model, data, position = _composed_reference()
  mass = np.zeros_like(position["mass"])
  mujoco.mj_fullM(model, data, mass)
  np.testing.assert_array_equal(mass, position["mass"])
  np.testing.assert_array_equal(data.ten_length, position["tendon_length"])
  np.testing.assert_array_equal(data.ten_J, position["tendon_J"])
  np.testing.assert_array_equal(data.actuator_length, position["actuator_length"])
  np.testing.assert_array_equal(data.geom_xpos, position["geom_pos"])
  # POS samples retain the original stage while VEL/ACC sensors see the
  # refreshed velocity and activation. These are actual engine samples.
  np.testing.assert_allclose(data.sensordata[[0, 3]],
                             [position["tendon_length"][1],
                              position["actuator_length"][0]])
  np.testing.assert_allclose(data.sensordata[[1, 2]], data.ten_velocity)
  np.testing.assert_allclose(data.sensordata[[4, 5]], data.actuator_velocity)
  np.testing.assert_allclose(data.sensordata[6], data.actuator_force[0])
  force = data.qfrc_passive + data.qfrc_actuator - data.qfrc_bias
  np.testing.assert_allclose(mass @ data.qacc, force, atol=1e-12)
  old_force = force.copy()
  mujoco.mj_forward(model, data)
  new_force = data.qfrc_passive + data.qfrc_actuator - data.qfrc_bias
  assert np.max(np.abs(new_force - old_force)) > .5


@pytest.mark.gpu
@pytest.mark.skipif(os.getenv("MUJOCO_METAL_RUN_GPU") != "1", reason="opt-in GPU")
def test_native_simulation_velocity_refresh_uses_one_composed_position_stage(monkeypatch):
  import torch
  from mujoco_metal.simulation import MetalSimulation

  model, oracle, position = _composed_reference()
  simulation = MetalSimulation(model, 1, qpos=oracle.qpos[None],
                               qvel=oracle.qvel[None], profile="integrated_euler_v1")
  simulation._prepare_control(oracle.ctrl[None])
  def tensor(value):
    return torch.tensor([value], dtype=torch.float32, device="mps")
  q3, v3 = tensor([.6, -.4]), tensor([.3, .5])
  initial_acc, _, dynamics = simulation._acceleration(q3, v3, act_override=tensor([.1]))
  simulation._store_step_sensors(q3, v3, dynamics, qacc=initial_acc,
                                 act_override=tensor([.1]))
  def allocation_during_capture(*_args, **_kwargs):
    raise AssertionError("POS capture allocated new device storage")
  with monkeypatch.context() as allocation_guard:
    allocation_guard.setattr(torch.Tensor, "clone", allocation_during_capture)
    allocation_guard.setattr(torch, "empty", allocation_during_capture)
    allocation_guard.setattr(torch, "zeros", allocation_during_capture)
    context = simulation._capture_forward_position(q3, dynamics)
  q0, v0 = tensor([.1, .2]), tensor([-.4, .1])
  # Overwrite all borrowed position and force workspaces before replaying X3.
  simulation._acceleration(q0, v3, act_override=tensor([.3]))
  def forbidden(*_args, **_kwargs):
    raise AssertionError("velocity-only refresh dispatched a position stage")
  monkeypatch.setattr(simulation._smooth, "run_device", forbidden)
  monkeypatch.setattr(simulation._actuators, "run_kinematics", forbidden)
  monkeypatch.setattr(simulation._spatial_tendons, "run_kinematics", forbidden)
  monkeypatch.setattr(simulation._coupled_constraints, "generate_candidates", forbidden)
  qacc, status, refreshed = simulation._acceleration(
      q0, v0, act_override=tensor([.25]), position_context=context,
      skip_sleep_prepare=True)
  np.testing.assert_array_equal(status.cpu().numpy(), [0])
  np.testing.assert_allclose(refreshed["mass_matrix"][0].cpu().numpy(),
                             position["mass"], atol=2e-6, rtol=3e-5)
  np.testing.assert_allclose(qacc[0].cpu().numpy(), oracle.qacc,
                             atol=2e-4, rtol=5e-5)
  expected_force = oracle.qfrc_passive + oracle.qfrc_actuator - oracle.qfrc_bias
  np.testing.assert_allclose(simulation._rhs[0].cpu().numpy(), expected_force,
                             atol=2e-5, rtol=5e-5)
  simulation._store_step_sensors(
      q0, v0, refreshed, qacc=qacc,
      stages=(mujoco.mjtStage.mjSTAGE_VEL, mujoco.mjtStage.mjSTAGE_ACC),
      act_override=tensor([.25]), position_context=context)
  np.testing.assert_allclose(simulation.step_sensordata()[0], oracle.sensordata,
                             atol=3e-5, rtol=5e-5)


ACTUATOR_XML = """<mujoco><worldbody><body>
  <joint name="hinge" type="hinge" axis="0 1 0"/>
  <geom type="box" pos=".2 0 .1" size=".1 .05 .08" mass="1"/>
  </body></worldbody><actuator><general joint="hinge" dyntype="filter"
  dynprm=".05" gaintype="fixed" gainprm="1" biastype="affine"
  biasprm="0 -8 -.5"/></actuator></mujoco>"""


def _actuator_reference():
  model = mujoco.MjModel.from_xml_string(ACTUATOR_XML)
  data = mujoco.MjData(model)
  data.qpos[:] = .4
  data.qvel[:] = .3
  data.act[:] = .15
  data.ctrl[:] = .2
  mujoco.mj_forward(model, data)
  data.qpos[:] = 0
  data.qvel[:] = -.1
  data.act[:] = .25
  mujoco.mj_forwardSkip(model, data, mujoco.mjtStage.mjSTAGE_POS, 0)
  return model, data


def test_pinned_actuator_force_uses_cached_length_and_new_velocity_activation():
  model, data = _actuator_reference()
  np.testing.assert_allclose(data.actuator_length, [.4])
  np.testing.assert_allclose(data.actuator_velocity, [-.1])
  np.testing.assert_allclose(data.actuator_force, [.25 - 8 * .4 + .05])
  cached_force = data.actuator_force.copy()
  mujoco.mj_forward(model, data)
  assert np.max(np.abs(data.actuator_force - cached_force)) > 3


@pytest.mark.gpu
@pytest.mark.skipif(os.getenv("MUJOCO_METAL_RUN_GPU") != "1", reason="opt-in GPU")
def test_native_actuator_velocity_refresh_preserves_cached_length_and_moment(monkeypatch):
  import torch
  from mujoco_metal.model import load_model
  from mujoco_metal.smooth_metal import MetalSmoothDynamics
  from mujoco_metal.stateful_actuation import MetalActuators

  model, ref = _actuator_reference()
  smooth = MetalSmoothDynamics(load_model(model), batch_size=1)
  actuators = MetalActuators(model, batch_size=1)
  q3 = torch.tensor([[.4]], dtype=torch.float32, device="mps")
  v3 = torch.tensor([[.3]], dtype=torch.float32, device="mps")
  poses = smooth.run_device(q3, v3)["poses"]
  kin = {name: value.clone() for name, value in
         actuators.run_kinematics(q3, v3, poses).items()}

  def unexpected_position_dispatch(*_args, **_kwargs):
    raise AssertionError("velocity refresh recomputed transmission positions")

  monkeypatch.setattr(actuators, "_kin_kernel", unexpected_position_dispatch)
  v0 = torch.tensor([[-.1]], dtype=torch.float32, device="mps")
  updated = actuators.run_velocity_kinematics(kin, v0)
  np.testing.assert_array_equal(updated["moment"].cpu().numpy(), kin["moment"].cpu().numpy())
  np.testing.assert_allclose(updated["length"][0].cpu().numpy(), ref.actuator_length)
  np.testing.assert_allclose(updated["velocity"][0].cpu().numpy(), ref.actuator_velocity)
  result = actuators.run_forces(
      torch.tensor([[.2]], dtype=torch.float32, device="mps"),
      torch.tensor([[.25]], dtype=torch.float32, device="mps"), updated)
  np.testing.assert_allclose(result["force"][0].cpu().numpy(), ref.actuator_force,
                             atol=2e-6, rtol=3e-5)
  np.testing.assert_allclose(result["act_dot"][0].cpu().numpy(), ref.act_dot,
                             atol=2e-6, rtol=3e-5)


TENDON_XML = """<mujoco><option gravity="0 0 0"/><worldbody><body>
  <joint name="slider" type="slide" axis="1 0 0"/>
  <geom type="sphere" size=".1" mass="1"/></body></worldbody>
  <tendon><fixed stiffness="6" damping=".5">
    <joint joint="slider" coef="2"/></fixed></tendon></mujoco>"""


def _tendon_reference():
  model = mujoco.MjModel.from_xml_string(TENDON_XML)
  data = mujoco.MjData(model)
  data.qpos[:] = .4
  data.qvel[:] = .3
  mujoco.mj_forward(model, data)
  data.qpos[:] = 0
  data.qvel[:] = -.1
  mujoco.mj_forwardSkip(model, data, mujoco.mjtStage.mjSTAGE_POS, 0)
  return model, data


def test_pinned_tendon_passive_force_uses_cached_length_and_new_speed():
  model, ref = _tendon_reference()
  np.testing.assert_allclose(ref.ten_length, [.8])
  np.testing.assert_allclose(ref.ten_velocity, [-.2])
  np.testing.assert_allclose(ref.qfrc_passive, [-9.4])
  cached = ref.qfrc_passive.copy()
  mujoco.mj_forward(model, ref)
  assert np.max(np.abs(ref.qfrc_passive - cached)) > 9


@pytest.mark.gpu
@pytest.mark.skipif(os.getenv("MUJOCO_METAL_RUN_GPU") != "1", reason="opt-in GPU")
def test_native_fixed_tendon_refresh_uses_cached_length():
  import torch
  from mujoco_metal.tendons import MetalFixedTendonDynamics

  model, ref = _tendon_reference()
  program = MetalFixedTendonDynamics(model, batch_size=1)
  q3 = torch.tensor([[.4]], dtype=torch.float32, device="mps")
  v3 = torch.tensor([[.3]], dtype=torch.float32, device="mps")
  program.run_device(q3, v3)
  length = program._last_length.clone()
  q0 = torch.zeros_like(q3)
  v0 = torch.tensor([[-.1]], dtype=torch.float32, device="mps")
  program.run_device(q0, v0)  # Deliberately overwrite borrowed stage outputs.
  force, _, _ = program.run_device(q0, v0, length_override=length)
  np.testing.assert_allclose(force[0].cpu().numpy(), ref.qfrc_passive,
                             atol=2e-6, rtol=3e-5)
  np.testing.assert_allclose(program._last_length[0].cpu().numpy(), ref.ten_length)


SPATIAL_XML = """<mujoco><option gravity="0 0 0"/><worldbody>
  <site name="anchor" pos="0 0 0"/>
  <body pos=".3 .2 .1"><joint type="slide" axis="1 0 0"/>
    <geom type="sphere" size=".05" mass="1"/><site name="end"/>
  </body></worldbody><tendon><spatial stiffness="6" damping=".5">
    <site site="anchor"/><site site="end"/></spatial></tendon></mujoco>"""


def _spatial_reference():
  model = mujoco.MjModel.from_xml_string(SPATIAL_XML)
  ref = mujoco.MjData(model)
  ref.qpos[:] = .4
  ref.qvel[:] = .3
  mujoco.mj_forward(model, ref)
  ref.qpos[:] = 0
  ref.qvel[:] = -.1
  mujoco.mj_forwardSkip(model, ref, mujoco.mjtStage.mjSTAGE_POS, 0)
  return model, ref


def test_pinned_spatial_velocity_uses_cached_path_jacobian():
  model, ref = _spatial_reference()
  np.testing.assert_allclose(ref.ten_length, [np.sqrt(.7**2 + .2**2 + .1**2)])
  np.testing.assert_allclose(ref.ten_velocity, [-.1 * .7 / ref.ten_length[0]])
  cached_velocity = ref.ten_velocity.copy()
  mujoco.mj_forward(model, ref)
  assert np.max(np.abs(ref.ten_velocity - cached_velocity)) > .01


@pytest.mark.gpu
@pytest.mark.skipif(os.getenv("MUJOCO_METAL_RUN_GPU") != "1", reason="opt-in GPU")
def test_native_spatial_velocity_reuses_cached_path_and_jacobian(monkeypatch):
  import torch
  from mujoco_metal.model import load_model
  from mujoco_metal.smooth_metal import MetalSmoothDynamics
  from mujoco_metal.spatial_tendons import MetalSpatialTendonDynamics

  model, ref = _spatial_reference()
  smooth = MetalSmoothDynamics(load_model(model), batch_size=1)
  program = MetalSpatialTendonDynamics(model, batch_size=1)
  q3 = torch.tensor([[.4]], dtype=torch.float32, device="mps")
  v3 = torch.tensor([[.3]], dtype=torch.float32, device="mps")
  kin = {name: value.clone() for name, value in program.run_kinematics(
      v3, smooth.run_device(q3, v3)["poses"]).items()}

  def unexpected_path_dispatch(*_args, **_kwargs):
    raise AssertionError("velocity refresh recomputed tendon path")

  monkeypatch.setattr(program, "_kin_kernel", unexpected_path_dispatch)
  v0 = torch.tensor([[-.1]], dtype=torch.float32, device="mps")
  updated = program.run_velocity_kinematics(kin, v0)
  force, _, _ = program.run_forces(updated)
  for name, expected in (("length", ref.ten_length), ("velocity", ref.ten_velocity)):
    np.testing.assert_allclose(updated[name][0].cpu().numpy(), expected,
                               atol=2e-6, rtol=3e-5)
  np.testing.assert_allclose(force[0].cpu().numpy(), ref.qfrc_passive,
                             atol=2e-6, rtol=3e-5)


@pytest.mark.gpu
@pytest.mark.skipif(os.getenv("MUJOCO_METAL_RUN_GPU") != "1", reason="opt-in GPU")
def test_native_velocity_refresh_uses_frozen_pos_without_fk_or_crba(monkeypatch):
  import torch
  from mujoco_metal.model import load_model
  from mujoco_metal.smooth_metal import MetalSmoothDynamics

  model, ref, mass, poses = _reference()
  smooth = MetalSmoothDynamics(load_model(model), batch_size=1)
  q3 = torch.tensor([[.75, -.4]], dtype=torch.float32, device="mps")
  v3 = torch.tensor([[.6, -.3]], dtype=torch.float32, device="mps")
  stage = smooth.run_device(q3, v3)
  context = smooth.position_context(stage)
  # Other stage users may overwrite every borrowed Smooth/FK output.
  q0 = torch.tensor([[0, .1]], dtype=torch.float32, device="mps")
  v0 = torch.tensor([[.1, -.7]], dtype=torch.float32, device="mps")
  smooth.run_device(q0, v0)

  def unexpected_position_dispatch(*_args, **_kwargs):
    raise AssertionError("velocity refresh recomputed a position stage")

  monkeypatch.setattr(smooth._fk, "run_device", unexpected_position_dispatch)
  monkeypatch.setattr(smooth, "_kernel", unexpected_position_dispatch)
  result = smooth.run_velocity_device(context, v0)
  for field, expected in (("qfrc_bias", ref.qfrc_bias),
                          ("cvel", ref.cvel), ("cdof_dot", ref.cdof_dot),
                          ("mass_matrix", mass)):
    np.testing.assert_allclose(result[field][0].cpu().numpy(), expected,
                               atol=5e-6, rtol=3e-5, err_msg=field)
  np.testing.assert_allclose(result["poses"]["geom_pos"][0].cpu().numpy(),
                             poses, atol=2e-7, rtol=3e-5)
  np.testing.assert_array_equal(result["pose_status"].cpu().numpy(), [0])
