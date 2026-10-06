# Copyright 2026 The MuJoCo Metal contributors
# Licensed under the Apache License, Version 2.0.
"""Opt-in pinned-engine oracle coverage for generalized device derivatives."""

import os

import mujoco
import numpy as np
import pytest
import torch

from mujoco_metal import MetalSimulation
from mujoco_metal.native_api import mjd_inverseFD, mjd_stepFD, mjd_transitionFD

pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(os.getenv("MUJOCO_METAL_RUN_GPU") != "1", reason="opt-in GPU"),
]


def _model(integrator="Euler"):
  return mujoco.MjModel.from_xml_string("""<mujoco>
    <option integrator="INTEGRATOR" timestep=".002" gravity="0 0 -9.81" solver="PGS" iterations="100"/>
    <worldbody>
      <body name="root">
        <freejoint name="free"/>
        <inertial pos="0 0 0" mass="2" diaginertia=".3 .4 .5"/>
        <body name="ball_body" pos="0 0 .4">
          <joint name="ball" type="ball" damping=".02"/>
          <inertial pos="0 0 0" mass="1" diaginertia=".1 .12 .14"/>
          <body name="hinge_body" pos="0 0 .3">
            <joint name="hinge" type="hinge" axis="0 1 0" damping=".03"/>
            <inertial pos="0 0 0" mass=".7" diaginertia=".04 .05 .06"/>
            <body name="slide_body" pos="0 0 .2">
              <joint name="slide" type="slide" axis="1 0 0" damping=".01"/>
              <inertial pos="0 0 0" mass=".3" diaginertia=".02 .025 .03"/>
            </body>
          </body>
        </body>
      </body>
    </worldbody>
    <actuator><motor joint="hinge" ctrllimited="true" ctrlrange="-1 1" gear="2"/></actuator>
    <sensor><jointpos joint="hinge"/></sensor>
  </mujoco>""".replace("INTEGRATOR", integrator))


def _rich_model():
  """Contact/equality/mocap, stateful activation and stored-stage sensors."""
  return mujoco.MjModel.from_xml_string('''<mujoco>
    <option timestep=".002" gravity="0 0 -9.81" solver="PGS" iterations="100"/>
    <worldbody>
      <geom name="floor" type="plane" size="2 2 .1"/>
      <body name="anchor" mocap="true" pos="0 0 .12">
        <geom type="sphere" size=".1" contype="0" conaffinity="0"/>
      </body>
      <body name="root" pos="0 0 .18">
        <freejoint name="free"/>
        <inertial pos="0 0 0" mass="2" diaginertia=".3 .4 .5"/>
        <geom name="root_geom" type="sphere" size=".1" mass="2"/>
        <body name="link" pos="0 0 .25">
          <joint name="hinge" type="hinge" axis="0 1 0" damping=".03"/>
          <inertial pos="0 0 0" mass=".7" diaginertia=".04 .05 .06"/>
          <geom type="capsule" size=".06 .18" mass=".7"/>
          <site name="imu" pos="0 0 .1"/>
          <body name="slider" pos=".1 0 .2">
            <joint name="slide" type="slide" axis="1 0 0" damping=".01"/>
            <inertial pos="0 0 0" mass=".3" diaginertia=".02 .025 .03"/>
            <geom type="sphere" size=".06" mass=".3"/>
          </body>
        </body>
      </body>
    </worldbody>
    <equality><weld name="mocap_weld" body1="anchor" body2="root" active="true"/></equality>
    <actuator>
      <general name="filter" joint="hinge" dyntype="filter" dynprm=".05"
        gaintype="fixed" gainprm="1" biastype="none"
        ctrllimited="true" ctrlrange="-1 1"/>
      <general name="actuator_state" joint="slide" dyntype="filter"
        dynprm=".08" gaintype="fixed" gainprm=".5" biastype="none"
        ctrllimited="true" ctrlrange="0 1"/>
    </actuator>
    <sensor>
      <jointpos joint="hinge"/>
      <jointvel joint="slide"/>
      <accelerometer site="imu"/>
      <jointactuatorfrc joint="hinge"/>
      <jointactuatorfrc joint="slide"/>
    </sensor>
  </mujoco>''')


def _host(value):
  return value.detach().cpu().numpy().copy()


def _device_snapshot(sim):
  state = sim._state
  names = ("_qpos", "_qvel", "_qacc", "_time", "_status", "_act",
           "_history", "_qacc_warmstart", "_userdata", "_plugin_state",
           "_eq_active", "_mpos", "_mquat")
  rows = {name: (None if getattr(state, name, None) is None else
                 _host(getattr(state, name))) for name in names}
  for name in ("_control", "_applied_force", "_body_wrench", "_sensordata"):
    value = getattr(sim, name, None)
    rows[name] = None if value is None else _host(value)
  rows["generation"] = state.generation
  rows["state_tensors"] = {
      name: _host(value) for name, value in vars(state).items()
      if isinstance(value, torch.Tensor)}
  rows["simulation_tensors"] = {
      name: _host(value) for name, value in vars(sim).items()
      if isinstance(value, torch.Tensor)}
  sensor_program = getattr(sim, "_sensors", None)
  sensor_low = getattr(sensor_program, "_s_state_out_low", None)
  rows["sensor_output_low"] = (None if sensor_low is None else
                               _host(sensor_low))
  scheduler = getattr(sim, "_sleep_schedule", None)
  for name in ("tree_state", "tree_awake", "status", "eq_active"):
    value = getattr(scheduler, name, None) if scheduler is not None else None
    rows[f"sleep_{name}"] = None if value is None else _host(value)
  rows["sleep_epoch"] = (None if scheduler is None else
                         getattr(scheduler, "epoch", None))
  return rows


def _assert_snapshots_equal(actual, expected):
  assert actual.keys() == expected.keys()
  for name, value in expected.items():
    if isinstance(value, np.ndarray):
      np.testing.assert_array_equal(actual[name], value, err_msg=name)
    elif isinstance(value, dict):
      assert actual[name].keys() == value.keys(), name
      for nested, expected_value in value.items():
        if isinstance(expected_value, np.ndarray):
          np.testing.assert_array_equal(actual[name][nested], expected_value,
                                        err_msg=f"{name}.{nested}")
        else:
          assert actual[name][nested] == expected_value, f"{name}.{nested}"
    else:
      assert actual[name] == value, name


def _assert_persistent_snapshots_equal(actual, expected):
  """Compare checkpoint state, excluding only MSL launch descriptors.

  The dimensions tensors below are transient kernel arguments populated by
  each DeviceState row operation; they are not checkpointed simulation state.
  Selector contents and all physical/warning/cache tensors remain compared.
  Exact query rollback continues to use _assert_snapshots_equal and includes
  every launch descriptor.
  """
  ignored = {"generation", "simulation_tensors"}
  actual_persistent = {key: value for key, value in actual.items()
                       if key not in ignored}
  expected_persistent = {key: value for key, value in expected.items()
                         if key not in ignored}
  transient_launch_descriptors = {
      "_clear_rows_dims", "_strided_copy_dims", "_copy_packed_dims",
      "_update_rows_dims", "_warning_rows_dims",
  }
  for snapshot in (actual_persistent, expected_persistent):
    state_tensors = snapshot.get("state_tensors")
    if state_tensors is not None:
      snapshot["state_tensors"] = {
          name: value for name, value in state_tensors.items()
          if name not in transient_launch_descriptors}
  _assert_snapshots_equal(actual_persistent, expected_persistent)


def _cpu_data(model, qpos, qvel, ctrl):
  data = mujoco.MjData(model)
  data.qpos[:] = qpos
  data.qvel[:] = qvel
  data.ctrl[:] = ctrl
  return data


def _cpu_step_output(model, qpos, qvel, act, ctrl, *, mocap_pos=None,
                     mocap_quat=None, eq_active=None):
  data = mujoco.MjData(model)
  data.qpos[:] = qpos
  data.qvel[:] = qvel
  if model.na:
    data.act[:] = act
  data.ctrl[:] = ctrl
  if model.nmocap and mocap_pos is not None:
    data.mocap_pos[:] = mocap_pos
    data.mocap_quat[:] = mocap_quat
  if model.neq and eq_active is not None:
    data.eq_active[:] = eq_active
  mujoco.mj_step(model, data)
  return data.qpos.copy(), data.qvel.copy(), data.act.copy(), data.sensordata.copy()


def _accelerometer_sample_probe(sim, model, qpos, qvel, act, ctrl, field,
                                index, eps, *, mocap_pos, mocap_quat,
                                eq_active, sensor_address):
  """Capture failed ACC sample stencils on device and source for diagnosis."""
  from mujoco_metal.finite_difference import DeviceQueryTransaction, _invalidate_for_query
  from mujoco_metal.native_api import mj_integratePos
  sensor_id = int(np.searchsorted(np.asarray(model.sensor_adr),
                                  sensor_address, side="right") - 1)
  site_id = int(model.sensor_objid[sensor_id])
  body_id = int(model.site_bodyid[site_id])

  def cpu(delta):
    q = np.array(qpos, dtype=np.float64, copy=True)
    v = np.array(qvel, dtype=np.float64, copy=True)
    a = np.array(act, dtype=np.float64, copy=True)
    if field == "qpos":
      direction = np.zeros(model.nv)
      direction[index] = 1
      mujoco.mj_integratePos(model, q, direction, delta)
    else:
      v[index] += delta
    data = _cpu_data(model, q, v, ctrl)
    if model.na:
      data.act[:] = a
    data.mocap_pos[:] = mocap_pos
    data.mocap_quat[:] = mocap_quat
    data.eq_active[:] = eq_active
    mujoco.mj_step(model, data)
    return {
        "sensor": float(data.sensordata[sensor_address]),
        "qacc": np.asarray(data.qacc).copy().tolist(),
        "cacc": np.asarray(data.cacc).reshape(model.nbody, 6)[body_id].tolist(),
        "cvel": np.asarray(data.cvel).reshape(model.nbody, 6)[body_id].tolist(),
        "scom": np.asarray(data.subtree_com).reshape(model.nbody, 3)[
            int(model.body_rootid[body_id])].tolist(),
        "site_pos": np.asarray(data.site_xpos).reshape(model.nsite, 3)[
            site_id].tolist(),
        "site_mat": np.asarray(data.site_xmat).reshape(model.nsite, 9)[
            site_id].tolist(),
    }

  def device(delta):
    transaction.restore()
    if field == "qpos":
      tangent = torch.zeros((sim.batch_size, model.nv), dtype=torch.float32,
                            device=sim.device)
      tangent[:, index] = delta
      integrated = mj_integratePos(sim, sim._state._qpos, tangent, 1.0)
      sim._state._qpos.copy_(integrated["qpos"])
    else:
      sim._state._qvel[:, index].add_(delta)
    _invalidate_for_query(sim)
    acc_inputs = {}
    original_run_acc = sim._run_acc_into

    def capture_acc_inputs(qpos_arg, qvel_arg, qacc_arg, poses_arg,
                           dynamics_arg, out, *, qacc_low=None, program=None,
                           world_mask=None):
      acc_inputs.update({
          "qacc": qacc_arg.detach().clone(),
          "qacc_low": (qacc_low.detach().clone()
                       if qacc_low is not None else None),
          "site_pos": poses_arg["site_pos"].detach().clone(),
          "site_quat": poses_arg["site_quat"].detach().clone(),
          "root_com": poses_arg["root_com"].detach().clone(),
          "cvel": poses_arg["cvel"].detach().clone(),
      })
      result = original_run_acc(
          qpos_arg, qvel_arg, qacc_arg, poses_arg, dynamics_arg, out,
          qacc_low=qacc_low, program=program, world_mask=world_mask)
      acc_inputs["cacc"] = sim._sensors._rne_cacc.detach().clone()
      acc_inputs["cacc_low"] = sim._sensors._rne_cacc_low.detach().clone()
      acc_inputs["rne_scom"] = sim._sensors._rne_scom.detach().clone()
      return result

    sim._run_acc_into = capture_acc_inputs
    try:
      sim.step()
    finally:
      sim._run_acc_into = original_run_acc
    return {
        "sensor": float(_host(sim._sensordata)[0, sensor_address]),
        "sensor_low": float(_host(sim._sensors._s_state_out_low)[
            0, sensor_address]),
        "qacc": _host(acc_inputs["qacc"])[0].tolist(),
        "qacc_low": (_host(acc_inputs["qacc_low"])[0].tolist()
                     if acc_inputs["qacc_low"] is not None else
                     [0.0] * model.nv),
        "cacc": _host(acc_inputs["cacc"].reshape(
            sim.batch_size, model.nbody, 6))[0, body_id].tolist(),
        "cacc_low": _host(acc_inputs["cacc_low"].reshape(
            sim.batch_size, model.nbody, 6))[0, body_id].tolist(),
        "cvel": _host(acc_inputs["cvel"].reshape(
            sim.batch_size, model.nbody, 6))[0, body_id].tolist(),
        "scom": _host(acc_inputs["rne_scom"].reshape(
            sim.batch_size, model.nbody, 3))[0, body_id].tolist(),
        "root_com": _host(acc_inputs["root_com"])[0, body_id].tolist(),
        "site_pos": _host(acc_inputs["site_pos"])[0, site_id].tolist(),
        "site_quat": _host(acc_inputs["site_quat"])[0, site_id].tolist(),
    }

  with DeviceQueryTransaction(sim) as transaction:
    device_samples = {label: device(delta) for label, delta in (
        ("minus", -eps), ("center", 0.), ("plus", eps))}
    transaction.restore()
  return {
      "field": field, "index": index,
      "device": device_samples,
      "cpu": {label: cpu(delta) for label, delta in (
          ("minus", -eps), ("center", 0.), ("plus", eps))},
  }


def _cpu_step_fd(model, qpos, qvel, act, ctrl, eps, centered, *,
                 mocap_pos=None, mocap_quat=None, eq_active=None):
  """Independent finite differences built from public mj_step and integratePos."""
  bstate = _cpu_step_output(model, qpos, qvel, act, ctrl,
                            mocap_pos=mocap_pos, mocap_quat=mocap_quat,
                            eq_active=eq_active)
  nq, nv, na, nu, ns = (model.nq, model.nv, model.na, model.nu,
                        model.nsensordata)
  ndx = 2 * nv + na
  names = ("DyDq", "DyDv", "DyDa", "DyDu", "DsDq", "DsDv", "DsDa", "DsDu")
  widths = (nv, nv, na, nu, nv, nv, na, nu)
  result = {name: np.zeros((width, ndx if n < 4 else ns), dtype=np.float64)
            for n, (name, width) in enumerate(zip(names, widths))}

  def sample(field, index, delta):
    q = np.array(qpos, dtype=np.float64, copy=True)
    v = np.array(qvel, dtype=np.float64, copy=True)
    a = np.array(act, dtype=np.float64, copy=True)
    u = np.array(ctrl, dtype=np.float64, copy=True)
    if field == "qpos":
      direction = np.zeros(nv)
      direction[index] = 1
      mujoco.mj_integratePos(model, q, direction, delta)
    elif field == "qvel":
      v[index] += delta
    elif field == "act":
      a[index] += delta
    else:
      u[index] += delta
    return _cpu_step_output(model, q, v, a, u,
                            mocap_pos=mocap_pos, mocap_quat=mocap_quat,
                            eq_active=eq_active)

  fields = (("qpos", nv), ("qvel", nv), ("act", na), ("ctrl", nu))
  for field, width in fields:
    for i in range(width):
      plus_ok = minus_ok = True
      if field == "ctrl" and model.actuator_ctrllimited[i]:
        lo, hi = model.actuator_ctrlrange[i]
        plus_ok = lo <= ctrl[i] <= hi and lo <= ctrl[i] + eps <= hi
        minus_ok = lo <= ctrl[i] <= hi and lo <= ctrl[i] - eps <= hi
        minus_ok &= centered or not plus_ok
      plus = sample(field, i, eps) if plus_ok else None
      minus = sample(field, i, -eps) if minus_ok and centered or (
          minus_ok and not plus_ok) else None
      if plus is not None and minus is not None:
        first, second, step = minus, plus, 2 * eps
      elif plus is not None:
        first, second, step = bstate, plus, eps
      elif minus is not None:
        first, second, step = minus, bstate, eps
      else:
        continue
      tangent = np.empty(nv)
      mujoco.mj_differentiatePos(model, tangent, step, first[0], second[0])
      state_diff = np.concatenate((tangent, (second[1] - first[1]) / step,
                                   (second[2] - first[2]) / step))
      index_by_field = {"qpos": 0, "qvel": 1, "act": 2, "ctrl": 3}[field]
      result[names[index_by_field]][i] = state_diff
      result[names[index_by_field + 4]][i] = (second[3] - first[3]) / step
  return result


def _inverse_position_force_probe(sim, model, qpos, qvel, ctrl, qacc, dof, eps):
  """Diagnostic source/device force samples used only after a failed oracle."""
  from mujoco_metal.finite_difference import DeviceQueryTransaction, _invalidate_for_query
  from mujoco_metal.native_api import mj_integratePos

  def capture_device_stages():
    workspace = sim._smooth._workspace
    names = ("cdof", "cvel", "cdof_dot", "cacc", "body_force",
             "cvel_low", "cdof_dot_low", "cacc_low", "body_force_low")
    return {name: _host(workspace[name].reshape(sim.batch_size, -1))[0].copy()
            for name in names}

  def summarize_stage_delta(first, second):
    result = {}
    for name in first:
      delta = (second[name].astype(np.float64) -
               first[name].astype(np.float64)) / eps
      nonzero = np.flatnonzero(delta)
      result[name] = {
          "linf": float(np.max(np.abs(delta), initial=0)),
          "changed": int(nonzero.size),
          "first8": delta[nonzero[:8]].tolist(),
      }
    return result

  with DeviceQueryTransaction(sim):
    base_details = sim.inverse_skip(skipsensor=True, return_components=True)
    base = base_details["qfrc_inverse"].clone()
    device_base_stages = capture_device_stages()
    device_base_parts = {
        key: float(_host(value)[0, 2])
        for key, value in base_details["qfrc_inverse_components"].items()}
    transaction = DeviceQueryTransaction(sim)
    tangent = torch.zeros((sim.batch_size, model.nv), dtype=torch.float32,
                          device=sim.device)
    tangent[:, dof] = eps
    integrated = mj_integratePos(sim, sim._state._qpos, tangent, 1.0)
    sim._state._qpos.copy_(integrated["qpos"])
    _invalidate_for_query(sim)
    plus_details = sim.inverse_skip(skipsensor=True, return_components=True)
    plus = plus_details["qfrc_inverse"].clone()
    device_plus_stages = capture_device_stages()
    device_plus_parts = {
        key: float(_host(value)[0, 2])
        for key, value in plus_details["qfrc_inverse_components"].items()}
    device_base, device_plus = _host(base)[0], _host(plus)[0]
    transaction.restore()

  data = _cpu_data(model, qpos, qvel, ctrl)
  if model.na:
    data.act[:] = _host(sim._state._act)[0]
  data.qacc[:] = qacc
  mujoco.mj_inverse(model, data)
  host_base = data.qfrc_inverse.copy()
  host_base_stages = {
      "cdof": np.asarray(data.cdof).reshape(-1).copy(),
      "cvel": np.asarray(data.cvel).reshape(-1).copy(),
      "cdof_dot": np.asarray(data.cdof_dot).reshape(-1).copy(),
      "cacc": np.asarray(data.cacc).reshape(-1).copy(),
  }
  host_base_parts = {
      "bias": float(data.qfrc_bias[2]),
      "passive": float(data.qfrc_passive[2]),
      "constraint": float(data.qfrc_constraint[2]),
  }
  mass_qacc = np.empty(model.nv, dtype=np.float64)
  mujoco.mj_mulM(model, data, mass_qacc, data.qacc)
  host_base_parts["mass_qacc"] = float(mass_qacc[2])
  host_qpos = np.array(qpos, dtype=np.float64, copy=True)
  direction = np.zeros(model.nv)
  direction[dof] = 1
  mujoco.mj_integratePos(model, host_qpos, direction, eps)
  data.qpos[:] = host_qpos
  mujoco.mj_inverse(model, data)
  host_plus = data.qfrc_inverse.copy()
  host_plus_stages = {
      "cdof": np.asarray(data.cdof).reshape(-1).copy(),
      "cvel": np.asarray(data.cvel).reshape(-1).copy(),
      "cdof_dot": np.asarray(data.cdof_dot).reshape(-1).copy(),
      "cacc": np.asarray(data.cacc).reshape(-1).copy(),
  }
  mass_qacc_plus = np.empty(model.nv, dtype=np.float64)
  mujoco.mj_mulM(model, data, mass_qacc_plus, data.qacc)
  host_plus_parts = {
      "mass_qacc": float(mass_qacc_plus[2]),
      "bias": float(data.qfrc_bias[2]),
      "passive": float(data.qfrc_passive[2]),
      "constraint": float(data.qfrc_constraint[2]),
  }
  return {"eps": eps,
          "device_base_force_2": float(device_base[2]),
          "device_plus_force_2": float(device_plus[2]),
          "device_fd_force_2": float((device_plus[2] - device_base[2]) / eps),
          "host_base_force_2": float(host_base[2]),
          "host_plus_force_2": float(host_plus[2]),
          "host_fd_force_2": float((host_plus[2] - host_base[2]) / eps),
          "device_base_components_2": device_base_parts,
          "device_plus_components_2": device_plus_parts,
          "host_base_components_2": host_base_parts,
          "host_plus_components_2": host_plus_parts,
          "device_stage_fd_linf": summarize_stage_delta(
              device_base_stages, device_plus_stages),
          "host_stage_fd_linf": summarize_stage_delta(
              host_base_stages, host_plus_stages)}


def test_step_transition_and_inverse_derivatives_match_pinned_engine_and_restore_state():
  model = _model()
  sim = MetalSimulation(model, batch_size=1, profile="integrated_euler_v1")
  qpos = np.asarray(model.qpos0, dtype=np.float32).copy()
  qvel = np.linspace(-.12, .18, model.nv, dtype=np.float32)
  ctrl = np.asarray([.35], dtype=np.float32)
  sim.reset(qpos=qpos, qvel=qvel)
  sim._control.copy_(torch.as_tensor(ctrl[None], dtype=torch.float32,
                                     device=sim.device))
  before = _device_snapshot(sim)
  generation = sim.state.generation
  record = getattr(getattr(sim, "_forward_stages", None), "_record", None)

  eps = 2e-4
  expected_data = _cpu_data(model, qpos, qvel, ctrl)
  nx = 2 * model.nv + model.na
  expected = {
      "A": np.zeros((nx, nx)), "B": np.zeros((nx, model.nu)),
      "C": np.zeros((model.nsensordata, nx)),
      "D": np.zeros((model.nsensordata, model.nu)),
  }
  mujoco.mjd_transitionFD(model, expected_data, eps, 0,
                          expected["A"], expected["B"],
                          expected["C"], expected["D"])
  result = mjd_transitionFD(sim, eps=eps, flg_centered=False)
  for key in expected:
    np.testing.assert_allclose(_host(result[key])[0], expected[key],
                               rtol=3e-3, atol=3e-3, err_msg=key)
  assert _host(result["status"]).tolist() == [0]
  _assert_snapshots_equal(_device_snapshot(sim), before)
  assert sim.state.generation == generation
  assert getattr(getattr(sim, "_forward_stages", None), "_record", None) is record

  step = mjd_stepFD(sim, eps=eps, flg_centered=True)
  assert step["DyDq"].shape == (1, model.nv, nx)
  assert step["DyDu"].shape == (1, model.nu, nx)
  assert step["DsDq"].shape == (1, model.nv, model.nsensordata)
  _assert_snapshots_equal(_device_snapshot(sim), before)

  control = MetalSimulation(model, batch_size=1, profile="integrated_euler_v1")
  control.reset(qpos=qpos, qvel=qvel)
  control._control.copy_(torch.as_tensor(ctrl[None], dtype=torch.float32,
                                         device=control.device))
  sim.step()
  control.step()
  for field in ("_qpos", "_qvel", "_qacc", "_time", "_status",
                "_qacc_warmstart"):
    np.testing.assert_allclose(_host(getattr(sim._state, field)),
                               _host(getattr(control._state, field)),
                               rtol=1e-6, atol=1e-6, err_msg=field)

  inverse_data = _cpu_data(model, qpos, qvel, ctrl)
  inverse_data.qacc[:] = 0.1
  sim.reset(qpos=qpos, qvel=qvel)
  sim._control.copy_(torch.as_tensor(ctrl[None], dtype=torch.float32,
                                     device=sim.device))
  sim._state._qacc.fill_(.1)
  inverse_before = _device_snapshot(sim)
  expected_inverse = [
      np.zeros((model.nv, model.nv)), np.zeros((model.nv, model.nv)),
      np.zeros((model.nv, model.nv)),
      np.zeros((model.nv, model.nsensordata)), np.zeros((model.nv, model.nsensordata)),
      np.zeros((model.nv, model.nsensordata)), np.zeros((model.nv, model.nC)),
  ]
  mujoco.mjd_inverseFD(model, inverse_data, eps, 0, *expected_inverse)
  inverse = mjd_inverseFD(sim, eps=eps, flg_actuation=False)
  for key, expected_value in zip(
      ("DfDq", "DfDv", "DfDa", "DsDq", "DsDv", "DsDa", "DmDq"),
      expected_inverse):
    try:
      np.testing.assert_allclose(_host(inverse[key])[0], expected_value,
                                 rtol=4e-3, atol=4e-3, err_msg=key)
    except AssertionError as error:
      if key == "DfDq":
        eps_sweep = [
            {"dof": dof, **_inverse_position_force_probe(
                sim, model, qpos, qvel, ctrl, inverse_data.qacc, dof, step)}
            for dof in (3, 6, 8, 9, 10)
            for step in (1e-3, 5e-4, 2e-4, 1e-4)]
        raise AssertionError(f"{error}\nforce/eps probe: {eps_sweep}") from error
      raise
  assert _host(inverse["status"]).tolist() == [0]
  _assert_snapshots_equal(_device_snapshot(sim), inverse_before)


def test_step_control_column_uses_one_sided_stencil_at_limit():
  model = _model()
  sim = MetalSimulation(model, batch_size=1, profile="integrated_euler_v1")
  qpos = np.asarray(model.qpos0, dtype=np.float32).copy()
  qvel = np.zeros(model.nv, dtype=np.float32)
  ctrl = np.asarray([1.0], dtype=np.float32)
  sim.reset(qpos=qpos, qvel=qvel)
  sim._control.copy_(torch.as_tensor(ctrl[None], dtype=torch.float32,
                                     device=sim.device))
  eps = 2e-4
  expected_data = _cpu_data(model, qpos, qvel, ctrl)
  transition = np.zeros((2 * model.nv + model.na, model.nu))
  mujoco.mjd_transitionFD(model, expected_data, eps, 1,
                          None, transition, None, None)
  actual = mjd_transitionFD(sim, eps=eps, flg_centered=True)
  np.testing.assert_allclose(_host(actual["B"])[0], transition,
                             rtol=4e-3, atol=4e-3)


def test_accelerometer_pair_residual_applies_site_offset_once_native():
  model = mujoco.MjModel.from_xml_string('''<mujoco>
    <option timestep=".002" gravity="0 0 -9.81"/>
    <worldbody><body name="rotor" pos="0 0 1">
      <freejoint name="free"/>
      <inertial pos="0 0 0" mass="1" diaginertia=".2 .3 .4"/>
      <site name="imu" pos=".23 -.11 .17"/>
    </body></worldbody>
    <sensor><accelerometer site="imu"/>
      <accelerometer site="imu" cutoff=".5"/>
    </sensor>
  </mujoco>''')
  sim = MetalSimulation(model, batch_size=1, profile="integrated_euler_v1")
  sim.reset(qvel=np.asarray([[.3, -.2, .1, .25, -.15, .2]], np.float32))
  body_id, site_id, sensor_address = 1, 0, 0
  sim._body_wrench[0, body_id, 4] = 1.7  # nonzero torque about y
  captured = {}
  original_run_acc = sim._run_acc_into

  def capture(qpos, qvel, qacc, poses, dynamics, out, *, qacc_low=None,
              program=None, world_mask=None):
    result = original_run_acc(qpos, qvel, qacc, poses, dynamics, out,
                              qacc_low=qacc_low,
                              program=program, world_mask=world_mask)
    sensor = sim._sensors
    captured.update({
        "qacc": qacc.detach().clone(),
        "qacc_low": (qacc_low.detach().clone()
                     if qacc_low is not None else None),
        "site_pos": poses["site_pos"].detach().clone(),
        "site_quat": poses["site_quat"].detach().clone(),
        "cvel": poses["cvel"].detach().clone(),
        "cacc": sensor._rne_cacc.detach().clone(),
        "cacc_low": sensor._rne_cacc_low.detach().clone(),
        "scom": sensor._rne_scom.detach().clone(),
    })
    return result

  sim._run_acc_into = capture
  try:
    sim.step()
  finally:
    sim._run_acc_into = original_run_acc
  cacc = (_host(captured["cacc"])[0, body_id].astype(np.float64)
          + _host(captured["cacc_low"])[0, body_id].astype(np.float64))
  cvel = _host(captured["cvel"])[0, body_id].astype(np.float64)
  scom = _host(captured["scom"])[0, body_id].astype(np.float64)
  site_pos = _host(captured["site_pos"])[0, site_id].astype(np.float64)
  site_quat = _host(captured["site_quat"])[0, site_id].astype(np.float32)
  quat_norm = np.sqrt(np.sum(site_quat * site_quat, dtype=np.float32))
  site_quat /= quat_norm
  offset = site_pos - scom
  angular_acc = cacc[:3]
  point_shift = np.cross(offset, angular_acc)
  assert np.linalg.norm(offset) > .2
  assert np.linalg.norm(angular_acc) > .1
  assert np.linalg.norm(point_shift) > .1
  at_site = cacc[3:] - point_shift
  point_velocity = cvel[3:] + np.cross(cvel[:3], offset)
  # Quaternion rotation matrix for scalar-first q, evaluated independently
  # in binary64 from the actual device transform input.
  w, x, y, z = site_quat.astype(np.float64)
  rotation = np.asarray([
      [1-2*(y*y+z*z), 2*(x*y-w*z), 2*(x*z+w*y)],
      [2*(x*y+w*z), 1-2*(x*x+z*z), 2*(y*z-w*x)],
      [2*(x*z-w*y), 2*(y*z+w*x), 1-2*(x*x+y*y)],
  ])
  exact = rotation.T @ (at_site + np.cross(cvel[:3], point_velocity))
  wrong_double_shift = rotation.T @ (
      (cacc[3:] - 2*point_shift) + np.cross(cvel[:3], point_velocity))
  high = _host(sim._sensordata)[0, sensor_address:sensor_address + 3].astype(np.float64)
  low = _host(sim._sensors._s_state_out_low)[
      0, sensor_address:sensor_address + 3].astype(np.float64)
  np.testing.assert_allclose(high + low, exact, rtol=2e-6, atol=2e-5)
  assert np.linalg.norm(high + low - wrong_double_shift) > .1
  capped_high = _host(sim._sensordata)[0, 3:6].astype(np.float64)
  capped_low = _host(sim._sensors._s_state_out_low)[0, 3:6].astype(np.float64)
  capped_exact = np.clip(exact, -.5, .5)
  assert np.any(np.abs(exact) > .5)
  np.testing.assert_allclose(capped_high + capped_low, capped_exact,
                             rtol=2e-6, atol=2e-5)
  assert np.all(capped_low[np.abs(exact) > .5] == 0.0)


def test_accelerometer_pair_resolves_sub_ulp_stencil_from_qacc_low_native():
  """A paired qacc residual remains visible under a large sensor baseline."""
  model = mujoco.MjModel.from_xml_string('''<mujoco>
    <option gravity="0 0 0"/>
    <worldbody><body pos="0 0 1"><freejoint/>
      <site name="imu" pos="0 .001 0"/>
      <geom type="sphere" size=".1" mass="1"/>
    </body></worldbody>
    <sensor><accelerometer site="imu"/></sensor>
  </mujoco>''')
  sim = MetalSimulation(model, batch_size=1, profile="integrated_euler_v1")
  sim.reset()
  state = sim._state
  dynamics = sim._smooth.run_device(state._qpos, state._qvel)
  poses = dict(dynamics["poses"], cvel=dynamics["cvel"],
               root_com=dynamics["root_com"])
  qacc = torch.zeros((1, model.nv), dtype=torch.float32, device=sim.device)
  qacc[0, 0] = 259.0
  eps = np.float32(2e-4)
  samples = {}
  for label, delta in (("minus", -float(eps)), ("center", 0.0),
                       ("plus", float(eps))):
    qacc_low = torch.zeros_like(qacc)
    qacc_low[0, 5] = delta
    high = sim._sensors.run_acc_device(
        state._qpos, state._qvel, qacc, poses, qacc_low=qacc_low).clone()
    low = sim._sensors._s_state_out_low.clone()
    samples[label] = (_host(high)[0, :3].astype(np.float64),
                      _host(low)[0, :3].astype(np.float64))

  # Independent source kinematics for a free body at zero velocity: site
  # linear acceleration is qacc_translation - r x qacc_rotation.
  assert np.asarray(model.site_pos).shape == (1, 3)
  offset_y = float(model.site_pos[0, 1])
  assert offset_y == pytest.approx(.001, abs=1e-7)
  expected = {label: np.asarray([259.0 - offset_y * delta, 0.0, 0.0])
              for label, delta in (("minus", -float(eps)),
                                   ("center", 0.0),
                                   ("plus", float(eps)))}
  for label in samples:
    high, low = samples[label]
    np.testing.assert_allclose(high + low, expected[label],
                               rtol=0.0, atol=2e-6)
  assert np.array_equal(samples["minus"][0], samples["center"][0])
  assert np.array_equal(samples["plus"][0], samples["center"][0])
  fd = ((samples["plus"][0] + samples["plus"][1])
        - (samples["minus"][0] + samples["minus"][1])) / (2 * float(eps))
  fd_high_only = (samples["plus"][0] - samples["minus"][0]) / (2 * float(eps))
  np.testing.assert_allclose(fd[0], -offset_y, rtol=2e-3, atol=2e-6)
  assert fd_high_only[0] == 0.0
  assert fd[0] != fd_high_only[0]


def test_contact_free_sensor_profile_acc_sensor_reset_step_and_query_native():
  """The admitted contact-free sensor route keeps ACC sensors operational."""
  model = mujoco.MjModel.from_xml_string('''<mujoco>
    <option timestep=".002" gravity="0 0 -9.81" solver="PGS" iterations="40"/>
    <worldbody>
      <body name="ball" pos="0 0 .15"><freejoint/>
        <geom type="sphere" size=".1" contype="0" conaffinity="0" mass="1"/>
        <site name="imu" pos=".01 0 .02"/>
      </body>
    </worldbody>
    <sensor><accelerometer site="imu"/></sensor>
  </mujoco>''')
  model.opt.disableflags |= int(mujoco.mjtDisableBit.mjDSBL_CONTACT)
  sim = MetalSimulation(model, batch_size=1,
                        profile="contact_free_sensor_euler_v1")
  assert sim._coupled_constraints is None
  assert sim._sensors is not None
  assert sim._has_acc_sensors
  sim.reset(qvel=np.asarray([[0., 0., .1, .2, 0., -.1]], dtype=np.float32))
  sim.step()
  assert sim._last_sensor_qacc_low is None
  stepped = _host(sim._sensordata)
  assert np.all(np.isfinite(stepped))
  queried = _host(sim.sensor_values())
  assert queried.shape == (1, model.nsensordata)
  assert np.all(np.isfinite(queried))


def test_sensor_low_words_clear_on_partial_reset_restore_and_keyframe():
  model = mujoco.MjModel.from_xml_string('''<mujoco>
    <worldbody><body pos="0 0 1"><freejoint/>
      <site name="imu" pos=".2 0 .1"/>
      <geom type="sphere" size=".1" mass="1"/>
    </body></worldbody>
    <sensor><accelerometer site="imu"/></sensor>
    <keyframe><key name="home" qpos="0 0 1 1 0 0 0"/></keyframe>
  </mujoco>''')
  sim = MetalSimulation(model, batch_size=2, profile="integrated_euler_v1")
  sim.reset()
  sim.step()
  low = sim._sensors._s_state_out_low
  low.copy_(torch.tensor([[.11, .12, .13], [.21, .22, .23]],
                         dtype=torch.float32, device=sim.device))
  sim.reset(env_ids=[0])
  np.testing.assert_array_equal(_host(low)[0], np.zeros(3, np.float32))
  np.testing.assert_array_equal(_host(low)[1], np.asarray([.21, .22, .23], np.float32))

  checkpoint = sim.snapshot()
  low[0] = torch.tensor([.31, .32, .33], device=sim.device)
  low[1] = torch.tensor([.41, .42, .43], device=sim.device)
  sim.restore(checkpoint, env_ids=[1])
  np.testing.assert_array_equal(_host(low)[0], np.asarray([.31, .32, .33], np.float32))
  np.testing.assert_array_equal(_host(low)[1], np.asarray([.21, .22, .23], np.float32))

  low[0] = torch.tensor([.51, .52, .53], device=sim.device)
  low[1] = torch.tensor([.61, .62, .63], device=sim.device)
  sim.copy_environment(0, 1)
  np.testing.assert_array_equal(_host(low)[1], np.asarray([.51, .52, .53], np.float32))

  invalid = sim.snapshot()
  invalid["sensordata_low"] = np.asarray(
      invalid["sensordata_low"], dtype=np.float32).copy()
  invalid["sensordata_low"][1, 0] = np.nan
  before_invalid = _device_snapshot(sim)
  with pytest.raises(ValueError, match="sensordata_low.*nonfinite"):
    sim.restore(invalid)
  _assert_snapshots_equal(_device_snapshot(sim), before_invalid)

  legacy = sim.snapshot()
  legacy.pop("sensordata_low")
  legacy.pop("raw_sensordata_low")
  sim._sensors._s_state_out_low[1, 0] = .71
  sim.restore(legacy, env_ids=[1])
  np.testing.assert_array_equal(_host(low)[1], np.zeros(3, np.float32))

  low[0] = torch.tensor([.51, .52, .53], device=sim.device)
  low[1] = torch.tensor([.61, .62, .63], device=sim.device)
  sim.reset_to_keyframe(0, env_ids=[0])
  np.testing.assert_array_equal(_host(low)[0], np.zeros(3, np.float32))
  np.testing.assert_array_equal(_host(low)[1], np.asarray([.61, .62, .63], np.float32))


@pytest.mark.parametrize("integrator,profile", [
    ("Euler", "integrated_euler_v1"),
    ("RK4", "integrated_rk4_v1"),
    ("implicit", "integrated_implicit_v1"),
    ("implicitfast", "integrated_implicitfast_v1"),
])
def test_batch_step_and_source_permitted_derivatives_cover_integrators(integrator, profile):
  model = _model(integrator)
  batch = 2
  qpos = np.repeat(np.asarray(model.qpos0, dtype=np.float32)[None, :], batch, axis=0)
  qvel = np.stack([
      np.linspace(-.12, .18, model.nv, dtype=np.float32),
      np.linspace(.08, -.05, model.nv, dtype=np.float32)])
  ctrl = np.asarray([[.35], [-.25]], dtype=np.float32)
  sim = MetalSimulation(model, batch_size=batch, profile=profile)
  sim.reset(qpos=qpos, qvel=qvel)
  sim._control.copy_(torch.as_tensor(ctrl, dtype=torch.float32, device=sim.device))
  before = _device_snapshot(sim)
  eps = 2e-4
  actual = mjd_stepFD(sim, eps=eps, flg_centered=True)
  ndx = 2 * model.nv + model.na
  expected_names = (("DyDq", model.nv, ndx), ("DyDv", model.nv, ndx),
                    ("DyDa", model.na, ndx), ("DyDu", model.nu, ndx),
                    ("DsDq", model.nv, model.nsensordata),
                    ("DsDv", model.nv, model.nsensordata),
                    ("DsDa", model.na, model.nsensordata),
                    ("DsDu", model.nu, model.nsensordata))
  assert actual["status"].shape == (batch,)
  for key, rows, cols in expected_names:
    assert actual[key].shape == (batch, rows, cols), key
  for world in range(batch):
    expected = _cpu_step_fd(model, qpos[world], qvel[world],
                            np.zeros(model.na), ctrl[world], eps, True)
    for key, _, _ in expected_names:
      np.testing.assert_allclose(_host(actual[key])[world], expected[key],
                                 rtol=5e-3, atol=5e-3, err_msg=key)
  np.testing.assert_array_equal(_host(actual["status"]), 0)
  _assert_snapshots_equal(_device_snapshot(sim), before)

  if integrator == "RK4":
    with pytest.raises(ValueError, match="does not support RK4"):
      mjd_transitionFD(sim, eps=eps)
    with pytest.raises(ValueError, match="does not support RK4"):
      mjd_inverseFD(sim, eps=eps)
    return

  transition = mjd_transitionFD(sim, eps=eps, flg_centered=True)
  for world in range(batch):
    data = _cpu_data(model, qpos[world], qvel[world], ctrl[world])
    matrices = [np.zeros((2 * model.nv + model.na, 2 * model.nv + model.na)),
                np.zeros((2 * model.nv + model.na, model.nu)),
                np.zeros((model.nsensordata, 2 * model.nv + model.na)),
                np.zeros((model.nsensordata, model.nu))]
    mujoco.mjd_transitionFD(model, data, eps, 1, *matrices)
    for key, reference in zip(("A", "B", "C", "D"), matrices):
      np.testing.assert_allclose(_host(transition[key])[world], reference,
                                 rtol=5e-3, atol=5e-3, err_msg=key)
  _assert_snapshots_equal(_device_snapshot(sim), before)


def test_batch_three_contact_equality_mocap_activation_and_sensor_stages():
  model = _rich_model()
  batch = 3
  qpos = np.repeat(np.asarray(model.qpos0, dtype=np.float32)[None, :], batch, axis=0)
  qpos[:, 2] = .08  # root sphere is penetrating the plane at the query point
  qvel = np.stack([np.linspace(-.02, .03, model.nv, dtype=np.float32),
                   np.linspace(.01, -.025, model.nv, dtype=np.float32),
                   np.zeros(model.nv, dtype=np.float32)])
  controls = np.asarray([[.7, .8], [1., 0.], [-.4, .5]], dtype=np.float32)
  for world in range(batch):
    data = _cpu_data(model, qpos[world], qvel[world], controls[world])
    data.mocap_pos[0, 2] += [0., .01, -.01][world]
    data.eq_active[:] = 1
    mujoco.mj_forward(model, data)
    assert data.ncon > 0
    assert data.eq_active[0] == 1
  sim = MetalSimulation(model, batch_size=batch, profile="integrated_scalable_v1")
  sim.reset(qpos=qpos, qvel=qvel)
  sim._control.copy_(torch.as_tensor(controls, dtype=torch.float32,
                                    device=sim.device))
  sim._state._mpos[:, 0, 2].add_(torch.tensor([0., .01, -.01], device=sim.device))
  sim._state._act.copy_(torch.tensor([[.1, .2], [.15, .25], [.05, .1]],
                                     dtype=torch.float32, device=sim.device))
  before = _device_snapshot(sim)
  eps = 2e-4
  transition = mjd_transitionFD(sim, eps=eps, flg_centered=True)
  assert tuple(transition["A"].shape) == (batch, 2 * model.nv + model.na,
                                          2 * model.nv + model.na)
  assert tuple(transition["C"].shape) == (batch, model.nsensordata,
                                          2 * model.nv + model.na)
  for world in range(batch):
    data = _cpu_data(model, qpos[world], qvel[world], controls[world])
    data.mocap_pos[:] = _host(sim._state._mpos)[world]
    data.mocap_quat[:] = _host(sim._state._mquat)[world]
    data.act[:] = _host(sim._state._act)[world]
    data.eq_active[:] = _host(sim._state._eq_active)[world]
    matrices = [np.zeros((2 * model.nv + model.na, 2 * model.nv + model.na)),
                np.zeros((2 * model.nv + model.na, model.nu)),
                np.zeros((model.nsensordata, 2 * model.nv + model.na)),
                np.zeros((model.nsensordata, model.nu))]
    mujoco.mjd_transitionFD(model, data, eps, 1, *matrices)
    for key, reference in zip(("A", "B", "C", "D"), matrices):
      try:
        np.testing.assert_allclose(_host(transition[key])[world], reference,
                                   rtol=8e-3, atol=8e-3, err_msg=key)
      except AssertionError as error:
        if key == "C":
          actual = _host(transition[key])[world]
          delta = np.abs(actual - reference)
          bad = delta > 8e-3 + 8e-3 * np.abs(reference)
          rows = []
          for adr in np.flatnonzero(bad.any(axis=1)):
            sensor = int(np.searchsorted(model.sensor_adr, adr, side="right") - 1)
            name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_SENSOR, sensor)
            rows.append({"adr": int(adr), "sensor": name,
                         "actual": actual[adr].tolist(),
                         "expected": reference[adr].tolist()})
          qpos_world = _host(sim._state._qpos)[world]
          qvel_world = _host(sim._state._qvel)[world]
          act_world = _host(sim._state._act)[world]
          ctrl_world = _host(sim._control)[world]
          mocap_pos = _host(sim._state._mpos)[world]
          mocap_quat = _host(sim._state._mquat)[world]
          eq_active = _host(sim._state._eq_active)[world]
          acc_probe = [
              _accelerometer_sample_probe(
                  sim, model, qpos_world, qvel_world, act_world, ctrl_world,
                  field, index, eps, mocap_pos=mocap_pos,
                  mocap_quat=mocap_quat, eq_active=eq_active,
                  sensor_address=4)
              for field, index in (("qpos", 4), ("qpos", 7), ("qvel", 3))]
          raise AssertionError(
              f"{error}\nC sensor rows: {rows}\nACC raw samples: {acc_probe}") from error
        raise
  _assert_snapshots_equal(_device_snapshot(sim), before)

  inverse = mjd_inverseFD(sim, eps=eps, flg_actuation=True)
  assert inverse["DmDq"].shape == (batch, model.nv, model.nM)
  assert inverse["DsDq"].shape == (batch, model.nv, model.nsensordata)
  for world in range(batch):
    data = _cpu_data(model, qpos[world], qvel[world], controls[world])
    data.act[:] = _host(sim._state._act)[world]
    data.mocap_pos[:] = _host(sim._state._mpos)[world]
    data.mocap_quat[:] = _host(sim._state._mquat)[world]
    data.eq_active[:] = _host(sim._state._eq_active)[world]
    data.qacc[:] = _host(sim._state._qacc)[world]
    arrays = [np.zeros((model.nv, model.nv)) for _ in range(3)]
    arrays += [np.zeros((model.nv, model.nsensordata)) for _ in range(3)]
    arrays += [np.zeros((model.nv, model.nM))]
    mujoco.mjd_inverseFD(model, data, eps, 1, *arrays)
    for key, reference in zip(
        ("DfDq", "DfDv", "DfDa", "DsDq", "DsDv", "DsDa", "DmDq"),
        arrays):
      np.testing.assert_allclose(_host(inverse[key])[world], reference,
                                 rtol=8e-3, atol=8e-3, err_msg=key)
  _assert_snapshots_equal(_device_snapshot(sim), before)


def test_query_exception_and_checkpoint_reset_restore_fullstep_state():
  model = _model()
  sim = MetalSimulation(model, batch_size=2, profile="integrated_euler_v1")
  sim.reset(qvel=np.stack((np.linspace(-.1, .1, model.nv, dtype=np.float32),
                           np.linspace(.07, -.05, model.nv, dtype=np.float32))))
  checkpoint = sim.snapshot()
  before = _device_snapshot(sim)
  original_step = sim.step
  calls = 0

  def fail_after_mutating(*args, **kwargs):
    nonlocal calls
    calls += 1
    result = original_step(*args, **kwargs)
    if calls == 2:
      raise RuntimeError("injected step exception")
    return result

  sim.step = fail_after_mutating
  with pytest.raises(RuntimeError, match="injected step exception"):
    mjd_stepFD(sim, eps=2e-4)
  sim.step = original_step
  _assert_snapshots_equal(_device_snapshot(sim), before)
  sim.step()
  after_query = _device_snapshot(sim)

  control = MetalSimulation(model, batch_size=2, profile="integrated_euler_v1")
  control.restore(checkpoint)
  control.step()
  for name in ("_qpos", "_qvel", "_qacc", "_act", "_time", "_status",
               "_qacc_warmstart"):
    actual_value = getattr(sim._state, name)
    expected_value = getattr(control._state, name)
    if actual_value is None or expected_value is None:
      assert actual_value is expected_value, name
    else:
      np.testing.assert_allclose(_host(actual_value), _host(expected_value),
                                 rtol=1e-6, atol=1e-6, err_msg=name)
  sim.restore(checkpoint)
  restored = _device_snapshot(sim)
  # Public restore intentionally advances generation so prepared forward
  # records from before the restore cannot be reused.  It restores persistent
  # numerical state; lazy query scratch may have been allocated since capture.
  assert restored["generation"] > before["generation"]
  _assert_persistent_snapshots_equal(restored, before)


def test_sleeping_world_query_preserves_sleep_wake_checkpoint_and_next_step():
  model = mujoco.MjModel.from_xml_string('''<mujoco>
    <option integrator="RK4" timestep=".002" gravity="0 0 -9.81"
      sleep_tolerance=".1"><flag sleep="enable"/></option>
    <worldbody><geom name="floor" type="plane" size="2 2 .1"/>
      <body name="drop" pos="0 0 .1001" sleep="init">
        <joint name="drop_joint" type="slide" axis="0 0 1" damping=".1"/>
        <site name="drop_imu" pos=".04 0 .02"/>
        <geom type="sphere" size=".1" mass="1"/></body>
      <body name="drive" pos="1 0 .5" sleep="allowed">
        <joint name="drive_joint" type="slide" axis="1 0 0" damping=".1"/>
        <geom type="sphere" size=".05" mass="1"/></body>
    </worldbody><actuator><general name="filter" joint="drive_joint"
      dyntype="filter" dynprm=".05" gainprm="0" biastype="none"/></actuator>
    <sensor><accelerometer site="drop_imu"/>
      <jointpos joint="drop_joint"/><jointpos joint="drive_joint"/></sensor>
  </mujoco>''')
  batch = 3
  sim = MetalSimulation(model, batch_size=batch, profile="integrated_rk4_v1")
  qvel = np.asarray([[0., 0.], [.2, -.1], [0., 0.]], dtype=np.float32)
  sim.reset(qvel=qvel)
  controls = np.asarray([[0.], [.2], [0.]], dtype=np.float32)
  sim._control.copy_(torch.as_tensor(controls, dtype=torch.float32,
                                    device=sim.device))
  for _ in range(3):
    sim.step()
  drop_body = int(model.site_bodyid[0])
  drop_tree = int(model.body_treeid[drop_body])
  tree_awake = _host(sim._sleep_schedule.tree_awake)
  assert tree_awake.shape[1] > drop_tree
  assert not bool(tree_awake[0, drop_tree])
  sleeping_low = np.asarray([.031, .032, .033], np.float32)
  sim._sensors._s_state_out_low[0, :3].copy_(
      torch.as_tensor(sleeping_low, dtype=torch.float32, device=sim.device))
  retained_high = _host(sim._sensordata)[0, :3]
  sim.step()
  np.testing.assert_array_equal(_host(sim._sensors._s_state_out_low)[0, :3],
                                sleeping_low)
  np.testing.assert_array_equal(_host(sim._sensordata)[0, :3], retained_high)
  checkpoint = sim.snapshot()
  before = _device_snapshot(sim)
  result = mjd_stepFD(sim, eps=2e-4, flg_centered=True)
  assert tuple(result["DyDq"].shape) == (batch, model.nv, 2 * model.nv + model.na)
  _assert_snapshots_equal(_device_snapshot(sim), before)
  sim.step()
  control = MetalSimulation(model, batch_size=batch, profile="integrated_rk4_v1")
  control.restore(checkpoint)
  control.step()
  for name in ("_qpos", "_qvel", "_qacc", "_act", "_time", "_status",
               "_qacc_warmstart"):
    np.testing.assert_allclose(_host(getattr(sim._state, name)),
                               _host(getattr(control._state, name)),
                               rtol=2e-5, atol=2e-6, err_msg=name)
  actual, expected = _device_snapshot(sim), _device_snapshot(control)
  assert actual["generation"] != expected["generation"]
  _assert_persistent_snapshots_equal(actual, expected)


def test_nonfinite_world_fails_independently_and_zeroes_all_derivative_rows():
  model = _model()
  qpos = np.repeat(np.asarray(model.qpos0, dtype=np.float32)[None, :], 2, axis=0)
  qvel = np.zeros((2, model.nv), dtype=np.float32)
  sim = MetalSimulation(model, batch_size=2, profile="integrated_euler_v1")
  sim.reset(qpos=qpos, qvel=qvel)
  sim._state._qpos[0, 0] = float("nan")
  result = mjd_stepFD(sim, eps=2e-4, flg_centered=True)
  statuses = _host(result["status"])
  assert statuses[0] != 0
  assert statuses[1] == 0
  for key, value in result.items():
    if key != "status":
      assert np.isfinite(_host(value)).all(), key
      assert not _host(value)[0].any(), key
  assert any(_host(value)[1].any() for key, value in result.items()
             if key != "status" and _host(value)[1].size)
