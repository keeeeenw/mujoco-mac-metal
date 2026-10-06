# Copyright 2026 The MuJoCo Metal contributors
# Licensed under the Apache License, Version 2.0.
"""CPU contract and pinned-source oracle for native actuator USER callbacks."""

from types import SimpleNamespace
import os

import mujoco
import numpy as np
import pytest

torch = pytest.importorskip("torch")

from mujoco_metal.capacity import _runtime_buffer_sizes
from mujoco_metal.extensions import (
    ExtensionRegistry,
    NativeActuatorUserPlugin,
    NativePlugin,
    PluginType,
    validate_actuator_user_bindings,
)
from mujoco_metal.stateful_actuation import (
    ActuatorModel,
    act_dot_reference,
    force_reference,
)


def _xml(actuators):
  return ("<mujoco><option timestep='.003' gravity='0 0 -9.81'/><worldbody>"
          "<body pos='0 0 1'><joint name='j' type='hinge' axis='0 1 0'/>"
          "<geom type='sphere' size='.1' mass='1'/></body></worldbody>"
          f"<actuator>{actuators}</actuator></mujoco>")


def _user_model(actdim=2):
  return mujoco.MjModel.from_xml_string(_xml(
      f'<general name="user" joint="j" dyntype="user" actdim="{actdim}" '
      'gaintype="user" biastype="user" actearly="true" '
      'actlimited="true" actrange="-2 2"/>'
      '<motor name="built_in" joint="j" gear=".4" '
      'ctrllimited="true" ctrlrange="-1 1"/>'))


def _set_cpu_integrator(model, profile):
  name = {
      "integrated_euler_v1": "mjINT_EULER",
      "integrated_rk4_v1": "mjINT_RK4",
      "integrated_implicit_v1": "mjINT_IMPLICIT",
  }[profile]
  model.opt.integrator = getattr(mujoco.mjtIntegrator, name)


def test_unregistered_user_types_keep_pinned_no_callback_defaults():
  model = _user_model()
  meta = ActuatorModel(model)
  data = mujoco.MjData(model)
  data.qpos[0] = .23
  data.qvel[0] = -.17
  data.act[:] = [.3, -.2]
  data.ctrl[:] = [.8, 1.7]
  callbacks = (mujoco.get_mjcb_act_dyn(), mujoco.get_mjcb_act_gain(),
               mujoco.get_mjcb_act_bias())
  try:
    mujoco.set_mjcb_act_dyn(None)
    mujoco.set_mjcb_act_gain(None)
    mujoco.set_mjcb_act_bias(None)
    mujoco.mj_forward(model, data)
    np.testing.assert_array_equal(data.act_dot, np.zeros(model.na))
    # USER gain/bias are source defaults one/zero when callbacks are absent.
    np.testing.assert_allclose(data.actuator_force[0], data.act[1], atol=1e-12)
    np.testing.assert_allclose(data.actuator_force[1], 1.0, atol=1e-12)
    expected_dot = act_dot_reference(
        meta, data.ctrl, data.act, data.actuator_length,
        data.actuator_velocity)
    np.testing.assert_array_equal(expected_dot, data.act_dot)
    reference = force_reference(
        meta, data.ctrl, data.act, data.actuator_length,
        data.actuator_velocity, data.actuator_moment.reshape(model.nu, model.nv))
    np.testing.assert_allclose(reference["force"], data.actuator_force,
                               rtol=0, atol=2e-12)
    np.testing.assert_allclose(reference["qfrc"], data.qfrc_actuator,
                               rtol=0, atol=2e-12)
  finally:
    mujoco.set_mjcb_act_dyn(callbacks[0])
    mujoco.set_mjcb_act_gain(callbacks[1])
    mujoco.set_mjcb_act_bias(callbacks[2])


def test_registered_user_callbacks_match_pinned_cpu_callback_order():
  model = _user_model()
  meta = ActuatorModel(model)
  data = mujoco.MjData(model)
  data.qpos[0] = -.31
  data.qvel[0] = .27
  data.act[:] = [.21, -.08]
  data.ctrl[:] = [.63, -.4]
  called = []

  def dyn(m, d, actuator_id):
    called.append(("dyn", actuator_id))
    first = int(m.actuator_actadr[actuator_id])
    count = int(m.actuator_actnum[actuator_id])
    d.act_dot[first:first + count] = (
        .5 * d.ctrl[actuator_id] + d.qvel[0],
        -.25 * d.ctrl[actuator_id] + .2,
    )
    return 0.0  # Python bindings require a return even for higher-order USER.

  def gain(m, d, actuator_id):
    called.append(("gain", actuator_id))
    return 1.7 + .2 * d.actuator_length[actuator_id] + .1 * d.actuator_velocity[actuator_id]

  def bias(m, d, actuator_id):
    called.append(("bias", actuator_id))
    return -.3 + .5 * d.actuator_length[actuator_id] - .2 * d.actuator_velocity[actuator_id]

  callbacks = (mujoco.get_mjcb_act_dyn(), mujoco.get_mjcb_act_gain(),
               mujoco.get_mjcb_act_bias())
  try:
    mujoco.set_mjcb_act_dyn(dyn)
    mujoco.set_mjcb_act_gain(gain)
    mujoco.set_mjcb_act_bias(bias)
    mujoco.mj_forward(model, data)

    adr = int(model.actuator_actadr[0])
    user_dot = data.act_dot[adr:adr + 2].copy()
    length = np.asarray(data.actuator_length).copy()
    velocity = np.asarray(data.actuator_velocity).copy()
    user_gain = {0: 1.7 + .2 * length[0] + .1 * velocity[0]}
    user_bias = {0: -.3 + .5 * length[0] - .2 * velocity[0]}
    meta_dot = act_dot_reference(
        meta, data.ctrl, data.act, length, velocity,
        user_dynamics={0: user_dot})
    np.testing.assert_allclose(meta_dot, data.act_dot, rtol=0, atol=1e-12)
    reference = force_reference(
        meta, data.ctrl, data.act, length, velocity,
        data.actuator_moment.reshape(model.nu, model.nv),
        user_gain=user_gain, user_bias=user_bias,
        user_dynamics={0: user_dot})
    np.testing.assert_allclose(reference["force"], data.actuator_force,
                               rtol=0, atol=2e-12)
    np.testing.assert_allclose(reference["qfrc"], data.qfrc_actuator,
                               rtol=0, atol=2e-12)
    assert set(called) == {("dyn", 0), ("gain", 0), ("bias", 0)}
  finally:
    mujoco.set_mjcb_act_dyn(callbacks[0])
    mujoco.set_mjcb_act_gain(callbacks[1])
    mujoco.set_mjcb_act_bias(callbacks[2])


def test_user_dynamics_callback_is_evaluated_at_each_rk4_stage():
  # The shared XML helper intentionally uses Euler by default; construct the
  # same model with RK4 so this test observes the pinned four-stage contract.
  model = mujoco.MjModel.from_xml_string(
      _xml('<general name="user" joint="j" dyntype="user" actdim="2" '
           'gaintype="fixed" biastype="none"/>').replace(
               "<option timestep='.003'", "<option integrator='RK4' timestep='.01'")
  )
  data = mujoco.MjData(model)
  data.act[:] = [.1, .2]
  data.ctrl[0] = .7
  observed = []

  def dyn(m, d, actuator_id):
    observed.append((float(d.time), np.asarray(d.act).copy(), actuator_id))
    d.act_dot[:] = (.7 + d.act[1], d.act[0] - .1)
    return 0.0

  previous = mujoco.get_mjcb_act_dyn()
  try:
    mujoco.set_mjcb_act_dyn(dyn)
    mujoco.mj_step(model, data)
    assert len(observed) == 4
    np.testing.assert_allclose([item[0] for item in observed],
                               [0.0, .005, .005, .01], rtol=0, atol=1e-15)
    assert all(item[2] == 0 for item in observed)
    # Independently apply classical RK4 to the registered two-state callback.
    initial = np.array([.1, .2], dtype=np.float64)
    def f(x):
      return np.array([.7 + x[1], x[0] - .1])
    k1 = f(initial)
    k2 = f(initial + .005 * k1)
    k3 = f(initial + .005 * k2)
    k4 = f(initial + .01 * k3)
    expected = initial + (.01 / 6) * (k1 + 2*k2 + 2*k3 + k4)
    np.testing.assert_allclose(data.act, expected, rtol=0, atol=2e-8)
    np.testing.assert_allclose(observed[0][1], initial, rtol=0, atol=0)
    np.testing.assert_allclose(observed[1][1], initial + .005*k1,
                               rtol=0, atol=1e-8)
  finally:
    mujoco.set_mjcb_act_dyn(previous)


class _Writer(NativeActuatorUserPlugin):
  def __init__(self, name="writer", bindings=None):
    bindings = bindings or {"dynamics": (0,), "gain": (0,), "bias": (0,)}
    super().__init__(name, dynamics=bindings.get("dynamics", ()),
                     gain=bindings.get("gain", ()),
                     bias=bindings.get("bias", ()),
                     device_workspace_bytes=13)
    self.accepted = None

  def run_user_actuator_device(self, state, *, control, activation,
                               activation_derivative, kinematics, time,
                               compute_mask, out_activation_derivative,
                               out_gain, out_bias, dynamics_ids, gain_ids,
                               bias_ids):
    del state, activation, kinematics, time
    mask = torch.ones(control.shape[0], dtype=torch.bool) if compute_mask is None else compute_mask
    for actuator_id in dynamics_ids:
      first = int(self.model.actuator_actadr[actuator_id])
      count = int(self.model.actuator_actnum[actuator_id])
      out_activation_derivative[mask, first:first + count] = control[mask, actuator_id:actuator_id + 1] + torch.arange(count, dtype=control.dtype)[None, :]
    for actuator_id in gain_ids:
      out_gain[mask, actuator_id] = 2.0 + control[mask, actuator_id]
    for actuator_id in bias_ids:
      out_bias[mask, actuator_id] = -1.0 + .5 * control[mask, actuator_id]
    return None

  def advance_user_actuator_device(self, state, *, accepted_mask, timestep,
                                   output):
    del state, timestep
    self.accepted = accepted_mask.clone()
    assert output is not None
    self.output_gain_on_advance = output["gain"][accepted_mask].clone()


def test_registry_preflight_and_batched_writer_contract_cpu():
  model = _user_model()
  registry = ExtensionRegistry()
  registry.register(_Writer())
  snapshot = registry.registration_snapshot()
  validate_actuator_user_bindings(model, snapshot)
  plugins = registry.instantiate(model, batch_size=2, device="cpu",
                                 registrations=snapshot)
  plugin, = plugins
  control = torch.tensor([[.2, .7], [.4, -.3]], dtype=torch.float32)
  act_dot = torch.zeros((2, model.na), dtype=torch.float32)
  gain = torch.ones((2, model.nu), dtype=torch.float32)
  bias = torch.zeros((2, model.nu), dtype=torch.float32)
  mask = torch.tensor([True, False])
  output = plugin.run_user_actuator_device(
      SimpleNamespace(), control=control,
      activation=torch.zeros((2, model.na)),
      activation_derivative=act_dot, kinematics={}, time=torch.zeros(2),
      compute_mask=mask, out_activation_derivative=act_dot,
      out_gain=gain, out_bias=bias,
      dynamics_ids=(0,), gain_ids=(0,), bias_ids=(0,))
  assert output is None
  torch.testing.assert_close(act_dot[0], torch.tensor([.2, 1.2]))
  torch.testing.assert_close(act_dot[1], torch.zeros(model.na))
  torch.testing.assert_close(gain[:, 0], torch.tensor([2.2, 1.0]))
  torch.testing.assert_close(bias[:, 0], torch.tensor([-.9, 0.0]))
  plugin.advance_user_actuator_device(
      SimpleNamespace(), accepted_mask=mask, timestep=.003,
      output={"gain": gain})
  torch.testing.assert_close(plugin.accepted, mask)
  torch.testing.assert_close(plugin.output_gain_on_advance,
                             torch.tensor([[2.2, 1.0]], dtype=torch.float32))


def test_user_binding_type_overlap_factory_and_capacity_guards_cpu():
  model = _user_model()
  registry = ExtensionRegistry()
  registry.register(_Writer("a"))
  registry.register(_Writer("b"))
  with pytest.raises(ValueError, match="multiple registered"):
    validate_actuator_user_bindings(model, registry.registration_snapshot())

  bad_factory_registry = ExtensionRegistry()
  from mujoco_metal.extensions import NativePlugin
  bad_factory_registry.register_factory(
      "bad", PluginType.ACTUATOR,
      lambda: NativePlugin("bad", PluginType.ACTUATOR),
      actuator_user_bindings={"gain": (0,)})
  with pytest.raises(ValueError, match="fresh NativeActuatorUserPlugin"):
    validate_actuator_user_bindings(
        model, bad_factory_registry.registration_snapshot())

  built_in = mujoco.MjModel.from_xml_string(_xml(
      '<general joint="j" gaintype="fixed"/>'))
  wrong_type_registry = ExtensionRegistry()
  wrong_type_registry.register(_Writer(bindings={"gain": (0,)}))
  with pytest.raises(ValueError, match="requires gain USER type"):
    validate_actuator_user_bindings(
        built_in, wrong_type_registry.registration_snapshot())

  parts = dict(_runtime_buffer_sizes(
      model, 2, native_actuator_user_workspace_bytes=13))
  assert parts["actuator.user_gain"] == 2 * max(model.nu, 1)
  assert parts["actuator.user_bias"] == 2 * max(model.nu, 1)
  assert parts["actuator.user_plugin_workspace_elements"] == 4


@pytest.mark.parametrize("role", ["dynamics", "gain", "bias"])
def test_user_binding_roles_validate_independently_cpu(role):
  tags = {"dynamics": "dyntype", "gain": "gaintype", "bias": "biastype"}
  for selected_role in ("dynamics", "gain", "bias"):
    actdim = ' actdim="1"' if selected_role == "dynamics" else ""
    model = mujoco.MjModel.from_xml_string(_xml(
        f'<general joint="j" {tags[selected_role]}="user"{actdim}/>'))
    plugin = _Writer(f"role-{role}-{selected_role}",
                    bindings={role: (0,)})
    registry = ExtensionRegistry()
    registry.register(plugin)
    try:
      if role == selected_role:
        validate_actuator_user_bindings(
            model, registry.registration_snapshot())
      else:
        with pytest.raises(ValueError, match=f"requires {role} USER type"):
          validate_actuator_user_bindings(
              model, registry.registration_snapshot())
    finally:
      registry.unregister(plugin.name, PluginType.ACTUATOR)


def test_invalid_user_factory_rejected_before_device_state(monkeypatch):
  from mujoco_metal import simulation as simulation_module
  from mujoco_metal.extensions import default_registry
  model = _user_model()
  name = "user-factory-preflight-test"
  default_registry.register_factory(
      name, PluginType.ACTUATOR,
      lambda: NativePlugin(name, PluginType.ACTUATOR),
      actuator_user_bindings={"gain": (0,)})
  try:
    def must_not_initialize(*args, **kwargs):
      del args, kwargs
      pytest.fail("unsupported USER profile reached DeviceState/MPS initialization")
    monkeypatch.setattr(simulation_module, "DeviceState", must_not_initialize)
    with pytest.raises(ValueError, match="fresh NativeActuatorUserPlugin"):
      simulation_module.MetalSimulation(model, profile="integrated_euler_v1")
  finally:
    default_registry.unregister(name, PluginType.ACTUATOR)


def test_actuator_model_direct_user_binding_ids_are_not_coerced():
  model = _user_model()
  for invalid, error in ((-1, ValueError), (True, TypeError), (0.5, TypeError)):
    with pytest.raises(error):
      ActuatorModel(
          model, actuator_user_bindings=(("direct", (("gain", (invalid,)),)),))
  with pytest.raises(TypeError):
    ActuatorModel(model, actuator_user_bindings=(("direct", (("gain", 0),)),))


class _VelocityUserWriter(NativeActuatorUserPlugin):
  def __init__(self, name):
    super().__init__(name, dynamics=(0,), gain=(0,), bias=(0,))
    self.ticks = None
    self.fail_world_one = False

  def init(self, model, batch_size=1, device=None):
    super().init(model, batch_size=batch_size, device=device)
    self.ticks = torch.zeros((batch_size,), dtype=torch.float32,
                             device=self.device)

  def reset(self, env_ids=None):
    if env_ids is None:
      self.ticks.zero_()
    else:
      self.ticks[env_ids] = 0.0

  def reset_masked(self, reset_mask):
    self.ticks.masked_fill_(reset_mask, 0.0)

  def snapshot(self):
    return {"ticks": self.ticks.detach().cpu().numpy().copy()}

  def restore(self, snap, env_ids=None):
    values = torch.as_tensor(snap["ticks"], dtype=torch.float32,
                             device=self.device)
    if env_ids is None:
      self.ticks.copy_(values)
    else:
      self.ticks[env_ids] = values[env_ids]

  def device_snapshot(self):
    return {"ticks": self.ticks.clone()}

  def device_snapshot_bytes(self):
    return self.batch_size * 4

  def restore_device(self, snapshot):
    self.ticks.copy_(snapshot["ticks"])

  def restore_masked(self, snapshot, accepted_mask):
    self.ticks.copy_(torch.where(accepted_mask, self.ticks,
                                 snapshot["ticks"]))

  def advance_user_actuator_device(self, state, *, accepted_mask, timestep,
                                   output):
    del state, timestep, output
    self.ticks.add_(accepted_mask.to(dtype=torch.float32))

  def run_user_actuator_device(self, state, *, control, activation,
                               activation_derivative, kinematics, time,
                               compute_mask, out_activation_derivative,
                               out_gain, out_bias, dynamics_ids, gain_ids,
                               bias_ids):
    del state, activation, activation_derivative, time
    mask = (torch.ones(control.shape[0], dtype=torch.bool, device=control.device)
            if compute_mask is None else compute_mask)
    row_mask = mask[:, None]
    for actuator_id in dynamics_ids:
      first = int(self.model.actuator_actadr[actuator_id])
      count = int(self.model.actuator_actnum[actuator_id])
      velocity = kinematics["velocity"][:, actuator_id:actuator_id + 1]
      ctrl = control[:, actuator_id:actuator_id + 1]
      values = (.5 * ctrl + velocity, -.25 * ctrl + .2,
                .1 * ctrl - .3 * velocity + .4)
      computed = torch.cat(values[:count], dim=1)
      current = out_activation_derivative[:, first:first + count]
      out_activation_derivative[:, first:first + count] = torch.where(
          row_mask, computed, current)
    for actuator_id in gain_ids:
      length = kinematics["length"][:, actuator_id]
      velocity = kinematics["velocity"][:, actuator_id]
      computed = (1.7 + .2 * length + .1 * velocity
                  + .01 * self.ticks)
      out_gain[:, actuator_id] = torch.where(
          mask, computed, out_gain[:, actuator_id])
    for actuator_id in bias_ids:
      length = kinematics["length"][:, actuator_id]
      velocity = kinematics["velocity"][:, actuator_id]
      computed = (-.3 + .5 * length - .2 * velocity
                  + .02 * self.ticks)
      out_bias[:, actuator_id] = torch.where(
          mask, computed, out_bias[:, actuator_id])
    world_ids = torch.arange(control.shape[0], device=control.device)
    fail_mask = mask & (world_ids == 1) & self.fail_world_one
    if control.shape[0] > 1:
      out_gain[:, 0] = torch.where(
          fail_mask, torch.full_like(out_gain[:, 0], float("nan")),
          out_gain[:, 0])


def test_user_plugin_masked_reset_resets_only_selected_worlds_cpu():
  plugin = _VelocityUserWriter("masked-reset-contract")
  plugin.init(_user_model(), batch_size=3, device="cpu")
  plugin.ticks.copy_(torch.tensor([2.0, 3.0, 5.0]))
  mask = torch.tensor([False, True, False])
  plugin.reset_masked(mask)
  torch.testing.assert_close(plugin.ticks, torch.tensor([2.0, 0.0, 5.0]),
                             rtol=0, atol=0)


@pytest.mark.parametrize("actdim", [1, 2, 3])
def test_user_dynamics_callback_overwrites_each_activation_dimension_cpu(actdim):
  model = _user_model(actdim=actdim)
  plugin = _VelocityUserWriter(f"activation-dim-{actdim}")
  plugin.init(model, batch_size=2, device="cpu")
  control = torch.tensor([[.4, -.2], [.7, .1]])
  qvel = torch.tensor([[-.3], [.5]])
  out = torch.full((2, model.na), float("nan"))
  plugin.run_user_actuator_device(
      SimpleNamespace(), control=control,
      activation=torch.zeros((2, model.na)),
      activation_derivative=torch.zeros_like(out),
      kinematics={"velocity": torch.cat((qvel, qvel), dim=1),
                  "length": torch.zeros((2, model.nu))},
      time=torch.zeros(2), compute_mask=torch.tensor([True, False]),
      out_activation_derivative=out,
      out_gain=torch.ones((2, model.nu)),
      out_bias=torch.zeros((2, model.nu)), dynamics_ids=(0,),
      gain_ids=(0,), bias_ids=(0,))
  expected = torch.tensor([
      .5 * control[0, 0] + qvel[0, 0],
      -.25 * control[0, 0] + .2,
      .1 * control[0, 0] - .3 * qvel[0, 0] + .4,
  ])[:actdim]
  torch.testing.assert_close(out[0, :actdim], expected, rtol=0, atol=0)
  assert torch.isnan(out[1]).all()


def _set_user_cpu_callbacks(model):
  def dyn(m, d, actuator_id):
    first = int(m.actuator_actadr[actuator_id])
    count = int(m.actuator_actnum[actuator_id])
    slots = (.5 * d.ctrl[actuator_id] + d.qvel[0],
             -.25 * d.ctrl[actuator_id] + .2,
             .1 * d.ctrl[actuator_id] - .3 * d.qvel[0] + .4)
    if count:
      d.act_dot[first:first + count] = slots[:count]
    # MuJoCo's scalar callback return owns the one-slot case.  For higher
    # order activation, the callback writes the complete act_dot slice.
    return float(slots[0]) if count == 1 else 0.0

  def gain(m, d, actuator_id):
    return (1.7 + .2 * d.actuator_length[actuator_id]
            + .1 * d.actuator_velocity[actuator_id]
            + .01 * getattr(d, "_user_plugin_tick", 0.0))

  def bias(m, d, actuator_id):
    return (-.3 + .5 * d.actuator_length[actuator_id]
            - .2 * d.actuator_velocity[actuator_id]
            + .02 * getattr(d, "_user_plugin_tick", 0.0))

  old = (mujoco.get_mjcb_act_dyn(), mujoco.get_mjcb_act_gain(),
         mujoco.get_mjcb_act_bias())
  mujoco.set_mjcb_act_dyn(dyn)
  mujoco.set_mjcb_act_gain(gain)
  mujoco.set_mjcb_act_bias(bias)
  return old


@pytest.mark.parametrize("actdim", [1, 2, 3])
def test_cpu_user_callback_scalar_and_vector_activation_outputs_are_pinned(actdim):
  model = _user_model(actdim=actdim)
  old = _set_user_cpu_callbacks(model)
  try:
    data = mujoco.MjData(model)
    data.qpos[0] = .17
    data.qvel[0] = -.23
    data.act[:] = np.linspace(.1, .1 * model.na, model.na)
    data.ctrl[:] = [.8, -.35]
    mujoco.mj_forward(model, data)
    user_adr = int(model.actuator_actadr[0])
    slots = (.5 * data.ctrl[0] + data.qvel[0],
             -.25 * data.ctrl[0] + .2,
             .1 * data.ctrl[0] - .3 * data.qvel[0] + .4)
    np.testing.assert_allclose(data.act_dot[user_adr:user_adr + actdim],
                               slots[:actdim], rtol=0, atol=1e-12)
    before = data.act.copy()
    mujoco.mj_step(model, data)
    assert np.all(np.isfinite(data.act))
    assert not np.array_equal(data.act, before)
  finally:
    (mujoco.set_mjcb_act_dyn(old[0]), mujoco.set_mjcb_act_gain(old[1]),
     mujoco.set_mjcb_act_bias(old[2]))


def test_cpu_user_force_oracle_is_velocity_dependent():
  """The pinned implicit USER derivative remains zero despite this force law."""
  model = _user_model()
  old = _set_user_cpu_callbacks(model)
  try:
    forces = []
    epsilon = 1e-5
    for velocity in (-epsilon, epsilon):
      data = mujoco.MjData(model)
      data.qpos[0] = .17
      data.qvel[0] = velocity
      data.act[:] = [.3, -.2]
      data.ctrl[:] = [.8, .4]
      mujoco.mj_forward(model, data)
      forces.append(float(data.actuator_force[0]))
    derivative = (forces[1] - forces[0]) / (2 * epsilon)
    assert abs(derivative) > 1e-3

    # qDeriv is the CPU effective operator consumed by implicit integration.
    # Pinned mjd_actuator_vel handles affine, muscle and DC-motor cases, but
    # has no USER gain/bias branch; even this velocity-dependent callback
    # therefore contributes exactly zero to the implicit operator.
    model.opt.integrator = mujoco.mjtIntegrator.mjINT_IMPLICIT
    data = mujoco.MjData(model)
    data.qpos[0] = .17
    data.qvel[0] = .23
    data.act[:] = [.3, -.2]
    data.ctrl[:] = [.8, .4]
    mujoco.mj_step(model, data)
    np.testing.assert_array_equal(data.qDeriv, np.zeros_like(data.qDeriv))
  finally:
    (mujoco.set_mjcb_act_dyn(old[0]), mujoco.set_mjcb_act_gain(old[1]),
     mujoco.set_mjcb_act_bias(old[2]))


@pytest.mark.parametrize("profile", [
    "integrated_euler_v1", "integrated_rk4_v1", "integrated_implicit_v1",
])
@pytest.mark.parametrize("actdim", [1, 2, 3])
@pytest.mark.gpu
@pytest.mark.skipif(os.getenv("MUJOCO_METAL_RUN_GPU") != "1",
                    reason="serialized native MPS actuator USER qualification")
def test_native_user_velocity_callbacks_match_cpu_and_replay(profile, actdim):
  from mujoco_metal.simulation import MetalSimulation
  from mujoco_metal.extensions import default_registry

  model = _user_model(actdim)
  _set_cpu_integrator(model, profile)
  # Ensure the USER actuator is evaluated from a nonzero velocity and a
  # nontrivial two-slot activation state in each batched world.
  qpos = np.array([[.23], [-.31]], dtype=np.float32)
  qvel = np.array([[-.17], [.27]], dtype=np.float32)
  act = np.array([[.3, -.2, .1], [.21, -.08, -.17]], dtype=np.float32)[:, :actdim]
  ctrl = np.array([[.8, 1.7], [.63, -.4]], dtype=np.float32)
  registration_name = f"user-velocity-{profile}"
  prototype = _VelocityUserWriter(registration_name)
  default_registry.register(prototype)
  try:
    sim = MetalSimulation(model, batch_size=2, qpos=qpos, qvel=qvel,
                          profile=profile)
    runtime_plugin, = sim._actuator_user_plugins
    assert runtime_plugin is not prototype
    assert prototype.ticks is None
    sim.reset(qpos=qpos, qvel=qvel, act=act)
    status = sim.step(ctrl=ctrl)
    np.testing.assert_array_equal(status.cpu().numpy(), [0, 0])
    torch.testing.assert_close(runtime_plugin.ticks.cpu(), torch.tensor([1.0, 1.0]),
                               rtol=0, atol=0)

    old = _set_user_cpu_callbacks(model)
    try:
      expected = []
      for world in range(2):
        data = mujoco.MjData(model)
        data.qpos[:] = qpos[world]
        data.qvel[:] = qvel[world]
        data.act[:] = act[world]
        data.ctrl[:] = ctrl[world]
        mujoco.mj_step(model, data)
        expected.append((data.qpos.copy(), data.qvel.copy(),
                         data.qacc.copy(), data.act.copy()))
    finally:
      (mujoco.set_mjcb_act_dyn(old[0]), mujoco.set_mjcb_act_gain(old[1]),
       mujoco.set_mjcb_act_bias(old[2]))
    for field, index in (("_qpos", 0), ("_qvel", 1),
                         ("_qacc", 2), ("_act", 3)):
      actual = getattr(sim._state, field).detach().cpu().numpy()
      np.testing.assert_allclose(actual, np.stack([x[index] for x in expected]),
                                 rtol=2e-5, atol=2e-6)

    checkpoint = sim.snapshot()
    sim.step(ctrl=ctrl)
    target = [getattr(sim._state, field_name).clone() for field_name in
              ("_qpos", "_qvel", "_qacc", "_act")]
    sim.restore(checkpoint)
    replay_status = sim.step(ctrl=ctrl)
    np.testing.assert_array_equal(replay_status.cpu().numpy(), [0, 0])
    for field_name, reference in zip(("_qpos", "_qvel", "_qacc", "_act"), target):
      torch.testing.assert_close(getattr(sim._state, field_name), reference,
                                 rtol=0, atol=0, msg=field_name)
    torch.testing.assert_close(runtime_plugin.ticks.cpu(), torch.tensor([2.0, 2.0]),
                               rtol=0, atol=0)
  finally:
    default_registry.unregister(registration_name, PluginType.ACTUATOR)


@pytest.mark.skipif(os.getenv("MUJOCO_METAL_RUN_GPU") != "1",
                    reason="serialized native MPS USER defaults qualification")
@pytest.mark.parametrize("profile", [
    "integrated_euler_v1", "integrated_implicit_v1",
])
@pytest.mark.gpu
def test_native_unregistered_user_callback_defaults_match_cpu(profile):
  from mujoco_metal.simulation import MetalSimulation
  model = _user_model()
  _set_cpu_integrator(model, profile)
  qpos = np.array([[.23], [-.31]], dtype=np.float32)
  qvel = np.array([[-.17], [.27]], dtype=np.float32)
  act = np.array([[.3, -.2], [.21, -.08]], dtype=np.float32)
  ctrl = np.array([[.8, 1.7], [.63, -.4]], dtype=np.float32)
  sim = MetalSimulation(model, batch_size=2, qpos=qpos, qvel=qvel,
                        profile=profile)
  sim.reset(qpos=qpos, qvel=qvel, act=act)
  actual = sim.forward_skip(ctrl=ctrl)
  expected_dot, expected_force, expected_qfrc = [], [], []
  expected_states = []
  callbacks = (mujoco.get_mjcb_act_dyn(), mujoco.get_mjcb_act_gain(),
               mujoco.get_mjcb_act_bias())
  for world in range(2):
    data = mujoco.MjData(model)
    data.qpos[:] = qpos[world]
    data.qvel[:] = qvel[world]
    data.act[:] = act[world]
    data.ctrl[:] = ctrl[world]
    mujoco.set_mjcb_act_dyn(None)
    mujoco.set_mjcb_act_gain(None)
    mujoco.set_mjcb_act_bias(None)
    try:
      mujoco.mj_forward(model, data)
      expected_dot.append(data.act_dot.copy())
      expected_force.append(data.actuator_force.copy())
      expected_qfrc.append(data.qfrc_actuator.copy())
      mujoco.mj_step(model, data)
      expected_states.append((data.qpos.copy(), data.qvel.copy(),
                              data.qacc.copy(), data.act.copy()))
    finally:
      (mujoco.set_mjcb_act_dyn(callbacks[0]),
       mujoco.set_mjcb_act_gain(callbacks[1]),
       mujoco.set_mjcb_act_bias(callbacks[2]))
  np.testing.assert_allclose(actual["actuation"]["act_dot"].cpu().numpy(),
                             np.asarray(expected_dot), rtol=0, atol=2e-6)
  np.testing.assert_allclose(sim._actuators._ws["force"].reshape(sim.batch_size, model.nu).cpu().numpy(),
                             np.asarray(expected_force), rtol=0, atol=2e-6)
  np.testing.assert_allclose(actual["actuation"]["qfrc_actuator"].cpu().numpy(),
                             np.asarray(expected_qfrc), rtol=0, atol=2e-6)
  status = sim.step(ctrl=ctrl)
  np.testing.assert_array_equal(status.cpu().numpy(), [0, 0])
  for field_name, index in (("_qpos", 0), ("_qvel", 1),
                            ("_qacc", 2), ("_act", 3)):
    observed = getattr(sim._state, field_name).detach().cpu().numpy()
    np.testing.assert_allclose(observed,
                               np.stack([sample[index] for sample in expected_states]),
                               rtol=2e-5, atol=2e-6)


@pytest.mark.skipif(os.getenv("MUJOCO_METAL_RUN_GPU") != "1",
                    reason="serialized native USER accepted-world rollback qualification")
@pytest.mark.gpu
def test_native_user_callback_failure_reset_advances_only_accepted_worlds():
  from mujoco_metal.simulation import MetalSimulation
  from mujoco_metal.extensions import default_registry
  model = _user_model()
  registration_name = "user-world-rollback"
  prototype = _VelocityUserWriter(registration_name)
  default_registry.register(prototype)
  try:
    qpos = np.array([[.1], [-.2]], dtype=np.float32)
    qvel = np.array([[.3], [-.1]], dtype=np.float32)
    sim = MetalSimulation(model, batch_size=2, qpos=qpos, qvel=qvel,
                          profile="integrated_euler_v1")
    runtime_plugin, = sim._actuator_user_plugins
    assert runtime_plugin is not prototype
    assert prototype.ticks is None
    runtime_plugin.fail_world_one = True
    status = sim.step(ctrl=np.array([[.2, .7], [.4, -.3]], dtype=np.float32))
    assert int(status[0].item()) == 0
    assert int(status[1].item()) != 0
    torch.testing.assert_close(runtime_plugin.ticks.cpu(), torch.tensor([1.0, 0.0]),
                               rtol=0, atol=0)
    runtime_plugin.fail_world_one = False
    sim.reset(env_ids=[1])
    torch.testing.assert_close(runtime_plugin.ticks.cpu(), torch.tensor([1.0, 0.0]),
                               rtol=0, atol=0)
    status = sim.step(ctrl=np.array([[.2, .7], [.4, -.3]], dtype=np.float32))
    np.testing.assert_array_equal(status.cpu().numpy(), [0, 0])
    torch.testing.assert_close(runtime_plugin.ticks.cpu(), torch.tensor([2.0, 1.0]),
                               rtol=0, atol=0)
  finally:
    default_registry.unregister(registration_name, PluginType.ACTUATOR)
