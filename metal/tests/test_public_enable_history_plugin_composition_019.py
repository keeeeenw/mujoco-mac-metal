# Copyright 2026 The MuJoCo Metal contributors
# Licensed under the Apache License, Version 2.0.
"""Combined public enable/history/plugin/state transaction regression.

The focused suites cover each subsystem separately. This fixture deliberately
composes the public APIs so stale history, inverse-enable handling, plugin
checkpointing, and selected-world state updates are exercised in one model.
"""

import os
import json

import mujoco
import numpy as np
import pytest


def _host(value):
  return value.detach().cpu().numpy().copy()


def _model():
  spec = mujoco.MjSpec.from_string('''<mujoco>
    <size nuserdata="2"/>
    <option timestep=".002" gravity="0 0 0" integrator="Euler"
            solver="PGS" iterations="80">
      <flag contact="disable"/>
    </option>
    <worldbody><body><joint name="slide" type="slide" axis="1 0 0"
        damping=".15"/><geom type="sphere" size=".1" mass="1"
        contype="0" conaffinity="0"/></body></worldbody>
    <actuator><motor name="motor" joint="slide" gear="1.7"/></actuator>
    <sensor><jointpos name="position" joint="slide"/></sensor>
  </mujoco>''')
  motor = spec.actuator("motor")
  motor.nsample, motor.interp, motor.delay = 8, 1, .003
  sensor = spec.sensor("position")
  sensor.nsample, sensor.interp, sensor.delay = 8, 1, .003
  sensor.interval = [.004, 0]
  model = spec.compile()
  model.opt.enableflags |= int(mujoco.mjtEnableBit.mjENBL_FWDINV)
  model.opt.enableflags |= int(mujoco.mjtEnableBit.mjENBL_INVDISCRETE)
  return model


def _cpu_split_step(model, data, control):
  """Use MuJoCo's public split boundary with controls installed before ACC."""
  data.ctrl[:] = control
  mujoco.mj_step1(model, data)
  mujoco.mj_step2(model, data)


def test_cpu_enable_history_split_and_state_roundtrip_oracle():
  """Pin the combined source semantics without requiring a Metal device."""
  model = _model()
  refs = [mujoco.MjData(model), mujoco.MjData(model)]
  for world, data in enumerate(refs):
    data.userdata[:] = np.asarray([.125 * (world + 1), -.25], dtype=np.float64)
  controls = ([.3, -.5], [-.2, .7], [.6, -.1], [.1, .4], [-.4, .2], [.8, -.3])
  for step, values in enumerate(controls):
    for world, data in enumerate(refs):
      # Identical float32 controls/timestamps isolate integration/history
      # semantics from differences in the native clock representation.
      data.ctrl[0] = np.float32(values[world])
      _cpu_split_step(model, data, np.asarray([np.float32(values[world])]))
      assert np.isfinite(data.qacc).all()
      assert np.isfinite(data.solver_fwdinv).all()
      assert data.history.shape == (model.nhistory,)
      assert data.time == pytest.approx((step + 1) * model.opt.timestep)
  # mjtState's compiled history and warmstart groups round-trip together.
  data = refs[0]
  spec = int(mujoco.mjtState.mjSTATE_INTEGRATION)
  size = mujoco.mj_stateSize(model, spec)
  state = np.empty(size, dtype=np.float64)
  mujoco.mj_getState(model, data, state, spec)
  expected = state.copy()
  expected_qpos = data.qpos.copy()
  expected_history = data.history.copy()
  data.qpos[:] += .25
  data.history[:] = 0
  mujoco.mj_setState(model, data, state, spec)
  np.testing.assert_array_equal(state, expected)
  np.testing.assert_allclose(data.qpos, expected_qpos, atol=0, rtol=0)
  np.testing.assert_allclose(data.history, expected_history, atol=0, rtol=0)
  # FWDINV is published by a full step; INVDISCRETE is exercised by inverse
  # after replacing acceleration with an independently chosen finite vector.
  inverse = mujoco.MjData(model)
  inverse.qpos[:] = data.qpos
  inverse.qvel[:] = data.qvel
  inverse.qacc[:] = .37
  requested = inverse.qacc.copy()
  mujoco.mj_inverse(model, inverse)
  np.testing.assert_array_equal(inverse.qacc, requested)
  discrete_force = inverse.qfrc_inverse.copy()
  model.opt.enableflags &= ~int(mujoco.mjtEnableBit.mjENBL_INVDISCRETE)
  continuous = mujoco.MjData(model)
  continuous.qpos[:] = data.qpos
  continuous.qvel[:] = data.qvel
  continuous.qacc[:] = requested
  mujoco.mj_inverse(model, continuous)
  assert np.linalg.norm(discrete_force - continuous.qfrc_inverse) > 1e-5


def _register_stateful_force(name):
  import torch
  from mujoco_metal.extensions import NativePlugin, PluginType

  class CounterForce(NativePlugin):
    def __init__(self):
      super().__init__(name, PluginType.FORCE)

    def init(self, model, batch_size=1, device=None):
      super().init(model, batch_size, device)
      self.calls = torch.zeros((batch_size,), dtype=torch.int32, device=self.device)

    def run_device(self, state, *, qpos, qvel, dynamics, compute_mask=None, **_):
      if compute_mask is None:
        self.calls.add_(1)
      else:
        self.calls.add_(compute_mask.to(dtype=torch.int32))
      return torch.zeros((self.batch_size, self.model.nv),
                         dtype=torch.float32, device=self.device)

    def snapshot(self):
      return self.calls.detach().cpu().numpy().copy()

    def restore(self, snapshot, env_ids=None):
      values = np.asarray(snapshot, dtype=np.int32)
      if values.shape != (self.batch_size,):
        raise ValueError("invalid counter checkpoint")
      if env_ids is None:
        self.calls.copy_(torch.as_tensor(values, dtype=torch.int32,
                                         device=self.calls.device))
      else:
        ids = torch.as_tensor(np.asarray(env_ids), dtype=torch.int64,
                              device=self.calls.device)
        source = torch.as_tensor(values, dtype=torch.int32,
                                  device=self.calls.device)
        self.calls[ids] = source[ids]

    def device_snapshot(self):
      return self.calls.clone()

    def device_snapshot_bytes(self):
      return self.calls.numel() * self.calls.element_size()

    def restore_device(self, snapshot):
      if (not isinstance(snapshot, torch.Tensor) or snapshot.device != self.calls.device
          or snapshot.dtype != self.calls.dtype or snapshot.shape != self.calls.shape):
        raise ValueError("invalid counter snapshot")
      self.calls.copy_(snapshot)

    def restore_masked(self, snapshot, accepted_mask):
      self.calls.copy_(torch.where(accepted_mask, self.calls, snapshot))

    def reset_masked(self, reset_mask):
      self.calls.copy_(torch.where(reset_mask, torch.zeros_like(self.calls), self.calls))

    def reset(self, env_ids=None):
      if env_ids is None:
        self.calls.zero_()
      else:
        ids = torch.as_tensor(np.asarray(env_ids), dtype=torch.int64, device=self.calls.device)
        self.calls[ids] = 0

  from mujoco_metal.extensions import default_registry
  default_registry.register(CounterForce())
  return default_registry, CounterForce


@pytest.mark.gpu
@pytest.mark.skipif(os.getenv("MUJOCO_METAL_RUN_GPU") != "1", reason="opt-in native test")
def test_public_enable_history_plugin_state_and_inverse_composition():
  """Compose public APIs; the zero-force plugin qualifies state ownership only."""
  import torch
  from mujoco_metal import MetalSimulation
  from mujoco_metal.extensions import PluginType
  from mujoco_metal.native_api import (StateSpec, mj_getState, mj_setState,
      mj_inverse, mj_fwdPosition, mj_fwdVelocity, mj_fwdAcceleration,
      mj_fwdConstraint)

  model = _model()
  name = "state-inverse-composition-019"
  registry, plugin_type = _register_stateful_force(name)
  try:
    batch = 2
    qpos = np.array([[.1], [-.15]], dtype=np.float32)
    qvel = np.array([[.2], [-.3]], dtype=np.float32)
    sim = MetalSimulation(model, batch_size=batch, qpos=qpos, qvel=qvel,
                          profile="integrated_euler_v1")
    user_values = np.array([[.125, -.25], [.25, -.25]], dtype=np.float32)
    mj_setState(sim, {"userdata": user_values})
    runtime = next(p for p in sim._native_plugins if p.name == name)
    prototype = registry.get(name, PluginType.FORCE)
    assert runtime is not prototype
    assert runtime.model is model and runtime.batch_size == batch
    assert tuple(runtime.calls.shape) == (batch,)
    refs = [mujoco.MjData(model), mujoco.MjData(model)]
    for world, data in enumerate(refs):
      data.qpos[:] = qpos[world]
      data.qvel[:] = qvel[world]
      data.userdata[:] = user_values[world]
    for step, control in enumerate(([.3, -.5], [-.2, .7], [.6, -.1], [.1, .4])):
      ctrl = np.asarray(control, dtype=np.float32).reshape(batch, 1)
      sim.step(ctrl=ctrl)
      for world, data in enumerate(refs):
        _cpu_split_step(model, data, ctrl[world])
        np.testing.assert_allclose(sim.state._qpos.detach().cpu().numpy()[world],
                                   data.qpos, atol=3e-5, rtol=3e-5)
        np.testing.assert_allclose(sim.state._qvel.detach().cpu().numpy()[world],
                                   data.qvel, atol=3e-5, rtol=3e-5)
        np.testing.assert_allclose(sim.state._history.detach().cpu().numpy()[world],
                                   data.history, atol=5e-5, rtol=5e-5)
        np.testing.assert_allclose(sim.step_sensordata()[world], data.sensordata,
                                   atol=5e-5, rtol=5e-5)
        np.testing.assert_allclose(sim.solver_fwdinv.detach().cpu().numpy()[world],
                                   data.solver_fwdinv, atol=4e-3, rtol=4e-3)
      assert prototype.model is None and prototype.device is None

    # The callback actually ran. Its zero force is deliberate: this fixture
    # tests state ownership and transaction semantics, while force parity is
    # covered by the dedicated extension physics fixtures.
    assert np.all(_host(runtime.calls) > 0)

    # State groups are copied at the public boundary; replacing history and
    # warmstart changes the next trajectory without mutating the returned copy.
    saved = mj_getState(sim, StateSpec.HISTORY | StateSpec.WARMSTART |
                        StateSpec.USERDATA)
    old_history = saved["history"].clone()
    old_userdata = saved["userdata"].clone()
    replacement = old_history.clone()
    replacement[:, 0] = 0.125
    userdata_replacement = old_userdata + .5
    mj_setState(sim, {"history": replacement, "warmstart": saved["warmstart"],
                      "userdata": userdata_replacement})
    torch.testing.assert_close(saved["history"], old_history, rtol=0, atol=0)
    torch.testing.assert_close(saved["userdata"], old_userdata, rtol=0, atol=0)
    torch.testing.assert_close(mj_getState(sim, StateSpec.HISTORY)["history"],
                               replacement, rtol=0, atol=0)
    torch.testing.assert_close(mj_getState(sim, StateSpec.USERDATA)["userdata"],
                               userdata_replacement, rtol=0, atol=0)

    # Enabled inverse flags must coexist with the split public stages. The
    # supplied acceleration is kept as the inverse target under INVDISCRETE.
    poses = mj_fwdPosition(sim)
    dynamics = mj_fwdVelocity(sim, poses=poses)
    force, status = mj_fwdAcceleration(sim, poses=poses, dynamics=dynamics)
    assert not status.any().item()
    constrained, constraint_status, reused_dynamics = mj_fwdConstraint(
        sim, poses=poses, dynamics=dynamics)
    assert reused_dynamics is dynamics
    assert not constraint_status.any().item()
    assert torch.isfinite(constrained).all().item()
    acceleration = torch.full((batch, model.nv), .37, device="mps")
    result = mj_inverse(sim, qacc=acceleration)
    assert torch.isfinite(result).all().item()
    assert torch.isfinite(force).all().item()
    for world in range(batch):
      data = mujoco.MjData(model)
      data.qpos[:] = _host(sim.state._qpos)[world]
      data.qvel[:] = _host(sim.state._qvel)[world]
      data.qacc[:] = .37
      mujoco.mj_inverse(model, data)
      np.testing.assert_allclose(_host(result)[world], data.qfrc_inverse,
                                 atol=3e-3, rtol=3e-3)

    # Public snapshot/replay includes history and callback-owned state. A
    # selected reset affects one world and leaves its neighbor unchanged.
    checkpoint = sim.snapshot()
    plugin_state = runtime.snapshot()
    sim.step(ctrl=np.array([[.8], [-.4]], dtype=np.float32))
    expected = sim.snapshot()
    sim.restore(checkpoint)
    sim.step(ctrl=np.array([[.8], [-.4]], dtype=np.float32))
    np.testing.assert_array_equal(sim._state._history.detach().cpu().numpy(),
                                  expected["native_state"]["history"])
    np.testing.assert_array_equal(_host(sim.state._userdata),
                                  expected["native_state"]["userdata"])
    np.testing.assert_array_equal(sim.step_sensordata(), expected["sensordata"])
    np.testing.assert_array_equal(runtime.snapshot(), expected["plugins"][("force", name)])
    before = runtime.snapshot()
    sim.reset(env_ids=[1])
    after = runtime.snapshot()
    np.testing.assert_array_equal(after[0], before[0])
    assert after[1] == 0
    np.testing.assert_array_equal(sim.state._history.detach().cpu().numpy()[1],
                                  mujoco.MjData(model).history.astype(np.float32))
    np.testing.assert_array_equal(_host(sim.state._userdata)[1], np.zeros(2, np.float32))
    # A checkpoint made before replay still restores both plugin rows.
    sim.restore(checkpoint)
    np.testing.assert_array_equal(runtime.snapshot(), plugin_state)
    np.testing.assert_array_equal(_host(sim.state._userdata),
                                  checkpoint["native_state"]["userdata"])
  finally:
    registry.unregister(name, PluginType.FORCE)


def _constrained_model():
  """Two linked articulated bodies exercise equality rows in both layouts."""
  spec = mujoco.MjSpec.from_string('''<mujoco>
    <size nuserdata="2"/>
    <option timestep=".002" gravity="0 0 0" integrator="Euler"
            solver="PGS" iterations="80" cone="elliptic">
      <flag contact="disable"/>
    </option>
    <worldbody>
      <body name="left"><joint name="left_slide" type="slide" axis="1 0 0" damping=".1"/>
        <geom type="sphere" size=".04" mass="1" contype="0" conaffinity="0"/></body>
      <body name="right"><joint name="right_slide" type="slide" axis="1 0 0" damping=".2"/>
        <geom type="sphere" size=".04" mass="1.5" contype="0" conaffinity="0"/></body>
    </worldbody>
    <equality><joint name="tie" joint1="left_slide" joint2="right_slide"
                      polycoef="0 1 0 0 0" solref=".02 1"/></equality>
    <actuator><motor name="drive" joint="left_slide" gear="1.3"/></actuator>
    <sensor><jointpos name="left_position" joint="left_slide"/></sensor>
  </mujoco>''')
  sensor = spec.sensor("left_position")
  sensor.nsample, sensor.interp, sensor.delay = 8, 1, .003
  sensor.interval = [.004, 0]
  model = spec.compile()
  model.opt.enableflags |= int(mujoco.mjtEnableBit.mjENBL_FWDINV)
  model.opt.enableflags |= int(mujoco.mjtEnableBit.mjENBL_INVDISCRETE)
  return model


def _flex_equality_model():
  model = mujoco.MjModel.from_xml_string('''<mujoco>
    <option timestep=".002" gravity="0 0 0" integrator="Euler"
            solver="PGS" iterations="50"/>
    <worldbody><body name="anchor" pos="0 0 1">
      <geom type="sphere" size=".02"/>
      <flexcomp name="cable" type="grid" count="4 1 1"
                spacing=".1 .1 .1" mass=".5" radius=".01" dim="1">
        <edge equality="true" solref=".02 1" solimp=".9 .95 .001 .5 2"/>
        <contact contype="0" conaffinity="0"/>
      </flexcomp>
    </body></worldbody>
  </mujoco>''')
  model.opt.enableflags |= int(mujoco.mjtEnableBit.mjENBL_FWDINV)
  model.opt.enableflags |= int(mujoco.mjtEnableBit.mjENBL_INVDISCRETE)
  return model


def _set_jacobian_layout_for_profile(model, profile):
  """Select the compiled MuJoCo layout explicitly; profiles do not override it."""
  model.opt.jacobian = (
      mujoco.mjtJacobian.mjJAC_SPARSE
      if profile == "integrated_scalable_v1"
      else mujoco.mjtJacobian.mjJAC_DENSE)


def test_cpu_constrained_inverse_history_fixture_has_live_equality_rows():
  model = _constrained_model()
  assert model.neq == 1 and model.nhistory > 0
  datas = [mujoco.MjData(model) for _ in range(2)]
  for world, data in enumerate(datas):
    data.qpos[:] = np.asarray([.12 * (world + 1), -.03], dtype=np.float64)
    data.qvel[:] = np.asarray([.4, -.1], dtype=np.float64)
    data.userdata[:] = [.25 + world, -.5]
  for step in range(4):
    for world, data in enumerate(datas):
      ctrl = np.asarray([.6 - .1 * step + world * .2], dtype=np.float64)
      _cpu_split_step(model, data, ctrl)
      assert data.nefc >= 1
      assert np.isfinite(data.efc_force[:data.nefc]).all()
      assert np.isfinite(data.solver_fwdinv).all()
      assert data.history.shape == (model.nhistory,)
    # Both flags must have observable behavior: the forward diagnostic is
    # populated, and inverse-discrete force differs from its continuous form.
    assert np.any(datas[0].solver_fwdinv != 0)
  original = model.opt.enableflags
  data = mujoco.MjData(model)
  data.qpos[:] = datas[0].qpos
  data.qvel[:] = datas[0].qvel
  data.qacc[:] = [.3, -.2]
  mujoco.mj_inverse(model, data)
  discrete = data.qfrc_inverse.copy()
  model.opt.enableflags = original & ~int(mujoco.mjtEnableBit.mjENBL_INVDISCRETE)
  continuous = mujoco.MjData(model)
  continuous.qpos[:] = datas[0].qpos
  continuous.qvel[:] = datas[0].qvel
  continuous.qacc[:] = [.3, -.2]
  mujoco.mj_inverse(model, continuous)
  assert np.linalg.norm(discrete - continuous.qfrc_inverse) > 1e-5


@pytest.mark.gpu
@pytest.mark.skipif(os.getenv("MUJOCO_METAL_RUN_GPU") != "1", reason="opt-in native test")
@pytest.mark.parametrize("profile", ["integrated_euler_v1", "integrated_scalable_v1"])
def test_public_equality_history_both_inverse_flags_dense_and_sparse(profile):
  """Actual public stepping/inverse composition with equality and history."""
  import torch
  from mujoco_metal import MetalSimulation
  from mujoco_metal.native_api import mj_inverse, mj_getState, StateSpec

  model = _constrained_model()
  _set_jacobian_layout_for_profile(model, profile)
  batch = 2
  qpos = np.asarray([[.12, -.03], [-.08, .025]], dtype=np.float32)
  qvel = np.asarray([[.4, -.1], [-.25, .08]], dtype=np.float32)
  sim = MetalSimulation(model, batch_size=batch, qpos=qpos, qvel=qvel,
                        profile=profile)
  # Assert we entered the requested storage implementation, not a silent dense
  # fallback under the scalable profile.
  assert sim._coupled_constraints is not None
  assert (sim._coupled_constraints._jacobian_layout.mode ==
          (1 if profile == "integrated_scalable_v1" else 0))
  refs = [mujoco.MjData(model) for _ in range(batch)]
  for w, ref in enumerate(refs):
    ref.qpos[:] = qpos[w]
    ref.qvel[:] = qvel[w]
    ref.userdata[:] = [.25 + w, -.5]
  for k in range(5):
    ctrl = np.asarray([[.6 - .1 * k], [.8 - .1 * k]], dtype=np.float32)
    status = sim.step(ctrl=ctrl)
    assert not status.detach().cpu().numpy().any()
    for w, ref in enumerate(refs):
      ref.ctrl[:] = ctrl[w]
      mujoco.mj_step(model, ref)
    for label, actual, expected, atol, rtol in (
        ("qpos", sim.state._qpos, [r.qpos for r in refs], 3e-4, 3e-4),
        ("qvel", sim.state._qvel, [r.qvel for r in refs], 4e-4, 4e-4),
        ("qacc", sim.state._qacc, [r.qacc for r in refs], 3e-3, 3e-3),
        ("history", sim.state._history, [r.history for r in refs], 5e-4, 5e-4),
        ("fwdinv", sim.solver_fwdinv, [r.solver_fwdinv for r in refs], 5e-3, 5e-3),
    ):
      np.testing.assert_allclose(_host(actual), np.stack(expected).astype(np.float32),
                                 atol=atol, rtol=rtol, err_msg=label)
  state = mj_getState(sim, StateSpec.HISTORY | StateSpec.WARMSTART)
  assert torch.isfinite(state["history"]).all().item()
  qacc = torch.tensor([[.3, -.2], [-.15, .27]], dtype=torch.float32, device="mps")
  inverse = mj_inverse(sim, qacc=qacc)
  for w, ref in enumerate(refs):
    ref.qacc[:] = _host(qacc)[w]
    mujoco.mj_inverse(model, ref)
    np.testing.assert_allclose(_host(inverse)[w], ref.qfrc_inverse,
                               atol=4e-3, rtol=4e-3)


def test_cpu_flex_equality_inverse_fixture_has_compiled_constraint_state():
  model = _flex_equality_model()
  assert model.nflex == 1 and model.neq > 0 and model.npluginstate == 0
  data = mujoco.MjData(model)
  data.qpos[:] += np.linspace(0, .015, model.nq)
  data.qvel[:] = np.linspace(-.2, .3, model.nv)
  mujoco.mj_step(model, data)
  assert data.nefc > 0
  assert np.isfinite(data.solver_fwdinv).all()
  target = data.qacc.copy()
  mujoco.mj_inverse(model, data)
  np.testing.assert_array_equal(data.qacc, target)
  assert np.isfinite(data.qfrc_inverse).all()


def test_admitted_bundled_plugins_compile_with_upstream_zero_plugin_state():
  """Check actual compiled state sizes for every admitted bundled class.

  The upstream source contract is explicit: cable and touch-grid define an
  ``nstate`` callback returning zero; PID's ``StateSize`` returns zero; each
  of the five SDF registrations returns zero. The compiled model arrays below
  guard the installed pinned libraries and actual attribute/config lowering.
  NativePlugin-owned transaction state is a separate device-side facility.
  Pinned MuJoCo 3.10 source references: ``plugin/actuator/pid.cc:226``
  (``Pid::StateSize`` returns zero; registration at 256),
  ``plugin/elasticity/cable.cc:289``, ``plugin/sensor/touch_grid.cc:503``,
  and ``plugin/sdf/{bolt,bowl,gear,nut,torus}.cc`` at lines 133, 132, 212,
  133, and 82 respectively. ``src/user/user_model.cc:3120-3144`` calls each
  ``nstate`` callback and packs ``plugin_statenum``/``plugin_stateadr`` into
  ``npluginstate``; these tests validate the resulting model arrays.
  """
  mujoco.mj_loadAllPluginLibraries(mujoco.PLUGINS_DIR)
  # Compile cable, PID, and touch-grid in one model. This uses the same
  # production mapping (body-attached cable and actuator/sensor instances).
  mixed = mujoco.MjModel.from_xml_string('''<mujoco>
    <extension>
      <plugin plugin="mujoco.elasticity.cable"><instance name="rod">
        <config key="twist" value="800"/><config key="bend" value="1200"/>
      </instance></plugin>
      <plugin plugin="mujoco.pid"><instance name="pid">
        <config key="kp" value="8"/><config key="ki" value="2"/>
        <config key="kd" value=".5"/><config key="imax" value=".2"/>
        <config key="slewmax" value="20"/>
      </instance></plugin>
      <plugin plugin="mujoco.sensor.touch_grid"/>
    </extension>
    <worldbody>
      <body name="rod"><freejoint/><geom type="capsule" size=".025 .2" mass=".2"/>
        <plugin instance="rod"/></body>
      <body name="pid_body"><joint name="pid_joint" type="slide" axis="0 0 1"/>
        <geom type="box" size=".04 .04 .04" mass=".2"/></body>
      <body name="pad"><freejoint/><geom type="box" size=".12 .12 .04" mass="1"/>
        <site name="touch" size=".1 .1 .1"/></body>
    </worldbody>
    <actuator><plugin joint="pid_joint" plugin="mujoco.pid" instance="pid" actdim="2"/></actuator>
    <sensor><plugin name="grid" plugin="mujoco.sensor.touch_grid" objtype="site" objname="touch">
      <config key="nchannel" value="3"/><config key="size" value="4 3"/>
      <config key="fov" value="150 80"/><config key="gamma" value=".2"/>
    </plugin></sensor>
  </mujoco>''')
  assert mixed.nplugin == 3
  assert mixed.npluginstate == 0
  np.testing.assert_array_equal(mixed.plugin_statenum, np.zeros(3, dtype=np.int32))
  np.testing.assert_array_equal(mixed.plugin_stateadr, np.zeros(3, dtype=np.int32))

  for shape in ("bolt", "bowl", "gear", "nut", "torus"):
    plugin = f"mujoco.sdf.{shape}"
    xml = f'''<mujoco>
      <extension><plugin plugin="{plugin}"><instance name="shape"/></plugin></extension>
      <asset><mesh name="shape_mesh"><plugin instance="shape"/></mesh></asset>
      <worldbody><geom type="sdf" mesh="shape_mesh"><plugin instance="shape"/></geom></worldbody>
    </mujoco>'''
    model = mujoco.MjModel.from_xml_string(xml)
    assert model.nplugin == 1 and model.npluginstate == 0, shape
    np.testing.assert_array_equal(model.plugin_statenum, [0], err_msg=shape)
    np.testing.assert_array_equal(model.plugin_stateadr, [0], err_msg=shape)


@pytest.mark.gpu
@pytest.mark.skipif(os.getenv("MUJOCO_METAL_RUN_GPU") != "1", reason="opt-in native test")
@pytest.mark.parametrize("profile", ["integrated_euler_v1", "integrated_scalable_v1"])
def test_public_flex_equality_forward_inverse_composition(profile):
  """Exercise flex equality constraint rows through stepping and inverse."""
  import torch
  from mujoco_metal import MetalSimulation
  from mujoco_metal.native_api import mj_inverse

  model = _flex_equality_model()
  _set_jacobian_layout_for_profile(model, profile)
  qpos = model.qpos0.astype(np.float32).copy()
  qpos[0] += .01
  qvel = np.linspace(-.2, .3, model.nv, dtype=np.float32)
  sim = MetalSimulation(model, batch_size=1, qpos=qpos[None, :],
                        qvel=qvel[None, :], profile=profile)
  assert (sim._coupled_constraints._jacobian_layout.mode ==
          (1 if profile == "integrated_scalable_v1" else 0))
  ref = mujoco.MjData(model)
  ref.qpos[:] = qpos
  ref.qvel[:] = qvel
  for step in range(3):
    status = sim.step()
    assert not status.detach().cpu().numpy().any()
    mujoco.mj_step(model, ref)
    np.testing.assert_allclose(_host(sim.state._qpos)[0], ref.qpos,
                               atol=8e-4, rtol=8e-4)
    np.testing.assert_allclose(_host(sim.state._qvel)[0], ref.qvel,
                               atol=1e-3, rtol=1e-3)
    np.testing.assert_allclose(_host(sim.solver_fwdinv)[0], ref.solver_fwdinv,
                               atol=7e-3, rtol=7e-3)
  # First compare against a fresh CPU inverse evaluation at the exact native
  # public-stage inputs. Keep the rolling CPU trajectory comparison below as
  # an independent end-to-end gate so input drift is reported, not hidden.
  native_qpos = _host(sim.state._qpos)[0]
  native_qvel = _host(sim.state._qvel)[0]
  native_qacc = _host(sim.state._qacc)[0]
  # Preserve the original failing target (CPU rolling-step qacc) while
  # recording the native qacc separately. The matched CPU inverse uses this
  # exact original target at exact native qpos/qvel.
  target_qacc = ref.qacc.copy()
  desired = torch.as_tensor(target_qacc[None, :], dtype=torch.float32, device="mps")
  force = mj_inverse(sim, qacc=desired)
  matched = mujoco.MjData(model)
  matched.qpos[:] = native_qpos
  matched.qvel[:] = native_qvel
  # The device input is float32. Match its represented value rather than
  # evaluating the fresh CPU oracle at the original float64 rolling target.
  represented_target = _host(desired)[0]
  matched.qacc[:] = represented_target
  mujoco.mj_inverse(model, matched)
  matched_force = matched.qfrc_inverse.copy()
  ref.qacc[:] = desired.cpu().numpy()[0]
  mujoco.mj_inverse(model, ref)
  rolling_force = ref.qfrc_inverse.copy()
  native_force = _host(force)[0]
  detail = {
      "profile": profile,
      "jacobian_mode": int(sim._coupled_constraints._jacobian_layout.mode),
      "native_qpos": native_qpos.tolist(),
      "native_qvel": native_qvel.tolist(),
      "native_qacc": native_qacc.tolist(),
      "qacc_target": target_qacc.tolist(),
      "represented_qacc_target": represented_target.tolist(),
      "native_inverse_force": native_force.tolist(),
      "fresh_cpu_force_same_native_state": matched_force.tolist(),
      "rolling_cpu_force": rolling_force.tolist(),
      "max_abs_error_fresh_cpu": float(np.max(np.abs(native_force - matched_force))),
      "max_abs_error_rolling_cpu": float(np.max(np.abs(native_force - rolling_force))),
  }
  print("flex-inverse-oracle " + json.dumps(detail, sort_keys=True))
  capture_path = os.getenv("MUJOCO_METAL_INVERSE_CAPTURE")
  if capture_path:
    capture_root, capture_ext = os.path.splitext(capture_path)
    capture_path = f"{capture_root}-{profile}{capture_ext or '.npz'}"
    np.savez(capture_path, native_qpos=native_qpos, native_qvel=native_qvel,
             native_qacc=native_qacc, qacc_target=target_qacc,
             represented_qacc_target=represented_target,
             native_inverse_force=native_force,
             fresh_cpu_force=matched_force, rolling_cpu_force=rolling_force)
  np.testing.assert_allclose(
      native_force, matched_force, atol=8e-3, rtol=8e-3,
      err_msg="inverse force against fresh CPU oracle at exact native state")
  # Preserve the original rolling-trajectory force assertion as well.
  np.testing.assert_allclose(native_force, rolling_force,
                             atol=8e-3, rtol=8e-3)
