# Copyright 2026 The MuJoCo Metal contributors
# Licensed under the Apache License, Version 2.0.
"""Milestone 016: tendon/actuator/limit/subtree/insidesite/energy/magnetometer.

CPU oracle parity (float64, independent formulas over pinned MjData inputs),
native query parity, stored step-stage semantics, and lifecycle. ACC and
spatial families arrive in follow-up commits on this branch.
"""

import os

import mujoco
import numpy as np
import pytest

from mujoco_metal.model import load_model
from mujoco_metal.sensors import lower_sensors, sensor_oracle

_XML = '''<mujoco model="state_families">
  <option timestep="0.002" integrator="Euler" gravity="0 0 -9.81" magnetic="0.3 -0.5 0.2"/>
  <worldbody>
    <site name="pad" pos="0.4 0 0.1"/>
    <site name="eye" pos="0 0 1"/>
    <body name="base" pos="0 0 0.5">
      <joint name="h" type="hinge" axis="0 0 1" limited="true" range="-1 1" stiffness="2"/>
      <geom name="g" type="sphere" size="0.1" mass="0.5"/>
      <site name="s2"/>
      <body name="tip" pos="0.2 0 0">
        <joint name="s" type="slide" axis="1 0 0" damping="0.3"/>
        <geom name="g2" type="box" size="0.05 0.05 0.05" mass="0.2"/>
      </body>
    </body>
  </worldbody>
  <tendon><fixed name="t" limited="true" range="-0.5 0.5"><joint joint="h" coef="1"/></fixed></tendon>
  <actuator><motor name="m" joint="h" gear="2"/></actuator>
  <sensor>
    <tendonpos tendon="t"/><tendonvel tendon="t"/>
    <actuatorpos actuator="m"/><actuatorvel actuator="m"/>
    <jointlimitpos joint="h"/><jointlimitvel joint="h"/>
    <tendonlimitpos tendon="t"/><tendonlimitvel tendon="t"/>
    <magnetometer site="s2"/>
    <subtreecom body="base"/><subtreelinvel body="base"/><subtreeangmom body="base"/>
    <insidesite site="s2" objtype="geom" objname="g"/>
    <e_potential/><e_kinetic/>
    <jointpos joint="h"/><clock/>
  </sensor></mujoco>'''


def _batch(model, descriptor, batch=3):
  qpos = np.tile(np.asarray(model.qpos0, dtype=np.float64), (batch, 1))
  qvel = np.zeros((batch, model.nv))
  for w in range(batch):
    qpos[w, int(descriptor.jnt_qposadr[0])] += 0.35 * (w + 1)
    qpos[w, 0 if model.nq == 0 else int(descriptor.jnt_qposadr[1])] += 0.05 * w
    qvel[w] = np.linspace(-0.6, 0.8, model.nv) + 0.2 * w
  times = 0.4 + np.arange(batch) * 0.85
  return qpos, qvel, times


def _feed(model, descriptor, qpos, qvel, times):
  poses = {key: [] for key in (
      "body_pos", "body_quat", "geom_pos", "geom_quat", "site_pos",
      "site_quat", "inertial_pos", "inertial_quat", "cvel", "root_com")}
  extra_lists = {k: [] for k in (
      "ten_length", "ten_velocity", "actuator_length", "actuator_velocity",
      "subtree_com", "subtree_linvel", "subtree_angmom", "energy",
      "efc_type", "efc_id", "efc_pos", "efc_vel", "efc_margin")}
  expected = []
  nefc_max, ne, nf = 0, 0, 0
  datas = []
  for w in range(len(qpos)):
    data = mujoco.MjData(model)
    data.qpos[:] = qpos[w]
    data.qvel[:] = qvel[w]
    data.time = times[w]
    mujoco.mj_forward(model, data)
    mujoco.mj_subtreeVel(model, data)
    mujoco.mj_energyPos(model, data)
    mujoco.mj_energyVel(model, data)
    expected.append(data.sensordata.copy())
    datas.append(data)
    fk = descriptor.forward_kinematics(qpos[w])
    for key in poses:
      poses[key].append(fk[key] if key in fk else
                        data.cvel.copy() if key == "cvel"
                        else data.subtree_com.copy())
    extra_lists["ten_length"].append(np.asarray(data.ten_length).copy())
    extra_lists["ten_velocity"].append(np.asarray(data.ten_velocity).copy())
    extra_lists["actuator_length"].append(np.asarray(data.actuator_length).copy())
    extra_lists["actuator_velocity"].append(np.asarray(data.actuator_velocity).copy())
    extra_lists["subtree_com"].append(np.asarray(data.subtree_com).copy())
    extra_lists["subtree_linvel"].append(np.asarray(data.subtree_linvel).copy())
    extra_lists["subtree_angmom"].append(np.asarray(data.subtree_angmom).copy())
    extra_lists["energy"].append(np.asarray(data.energy).copy())
    nefc_max = max(nefc_max, int(data.nefc))
    ne, nf = int(data.ne), int(data.nf)
    for k, src in (("efc_type", data.efc_type), ("efc_id", data.efc_id),
                   ("efc_pos", data.efc_pos), ("efc_vel", data.efc_vel),
                   ("efc_margin", data.efc_margin)):
      extra_lists[k].append(np.asarray(src).copy())
  poses = {k: np.asarray(v) for k, v in poses.items()}
  width = max(nefc_max, 1)
  efc = {}
  for k in ("efc_type", "efc_id", "efc_pos", "efc_vel", "efc_margin"):
    mat = np.zeros((len(qpos), width), dtype=np.float64)
    for w, row in enumerate(extra_lists[k]):
      mat[w, :len(row)] = row
    efc[k] = mat
  extra = {
      "ten_length": np.asarray(extra_lists["ten_length"]),
      "ten_velocity": np.asarray(extra_lists["ten_velocity"]),
      "actuator_length": np.asarray(extra_lists["actuator_length"]),
      "actuator_velocity": np.asarray(extra_lists["actuator_velocity"]),
      "subtree_com": np.asarray(extra_lists["subtree_com"]),
      "subtree_linvel": np.asarray(extra_lists["subtree_linvel"]),
      "subtree_angmom": np.asarray(extra_lists["subtree_angmom"]),
      "energy": np.asarray(extra_lists["energy"]),
      "ne": ne, "nf": nf, **efc,
  }
  return poses, extra, np.asarray(expected)


def test_state_oracle_matches_mujoco_forward():
  model = mujoco.MjModel.from_xml_string(_XML)
  descriptor = load_model(model)
  qpos, qvel, times = _batch(model, descriptor)
  poses, extra, expected = _feed(model, descriptor, qpos, qvel, times)
  actual = sensor_oracle(model, qpos, qvel, times, poses, extra=extra)
  np.testing.assert_allclose(actual, expected, rtol=3e-8, atol=2e-8)


def test_state_oracle_needs_extra_without_it():
  model = mujoco.MjModel.from_xml_string(_XML)
  descriptor = load_model(model)
  qpos, qvel, times = _batch(model, descriptor)
  poses, _, _ = _feed(model, descriptor, qpos, qvel, times)
  with pytest.raises(ValueError, match="extra"):
    sensor_oracle(model, qpos, qvel, times, poses)


@pytest.mark.gpu
@pytest.mark.skipif(os.getenv("MUJOCO_METAL_RUN_GPU") != "1", reason="opt-in GPU")
def test_native_state_query_matches_forward_gpu():
  import torch
  from mujoco_metal.simulation import MetalSimulation
  model = mujoco.MjModel.from_xml_string(_XML)
  descriptor = load_model(model)
  qpos, qvel, times = _batch(model, descriptor)
  sim = MetalSimulation(model, batch_size=len(qpos), profile="integrated_euler_v1")
  sim.reset(qpos=qpos.astype(np.float32), qvel=qvel.astype(np.float32))
  # Drive one world into its limit so limit rows are active.
  qp = qpos.copy()
  qp[0, 0] = 1.4
  sim.reset(qpos=qp.astype(np.float32), qvel=qvel.astype(np.float32))
  got = sim.sensor_values().cpu().numpy()
  times0 = np.zeros(len(qp))
  poses, extra, _ = _feed(model, descriptor, qp, qvel, times0)
  want = sensor_oracle(model, qp, qvel, times0, poses, extra=extra)
  np.testing.assert_allclose(got, want, rtol=2e-4, atol=2e-4)


@pytest.mark.gpu
@pytest.mark.skipif(os.getenv("MUJOCO_METAL_RUN_GPU") != "1", reason="opt-in GPU")
def test_stored_step_sample_matches_mj_step_gpu():
  import torch
  from mujoco_metal.simulation import MetalSimulation
  model = mujoco.MjModel.from_xml_string(_XML)
  descriptor = load_model(model)
  qpos, qvel, _ = _batch(model, descriptor, batch=2)
  sim = MetalSimulation(model, batch_size=2, profile="integrated_euler_v1")
  sim.reset(qpos=qpos.astype(np.float32), qvel=qvel.astype(np.float32))
  cpus = []
  for w in range(2):
    d = mujoco.MjData(model)
    d.qpos[:] = qpos[w]
    d.qvel[:] = qvel[w]
    cpus.append(d)
  for _ in range(50):
    sim.step(1)
    for d in cpus:
      mujoco.mj_step(model, d)
  stored = sim.step_sensordata()
  for w, d in enumerate(cpus):
    np.testing.assert_allclose(
        stored[w], np.asarray(d.sensordata), rtol=2e-4, atol=2e-4,
        err_msg=f"world {w}")
  # Query differs from stored: it re-evaluates at the post-step state.
  query = sim.sensor_values().cpu().numpy()
  assert np.any(np.abs(query - stored) > 1e-6)
  # Reset zeroes the stored sample; restore leaves it until the next step.
  sim.reset()
  np.testing.assert_array_equal(sim.step_sensordata(),
                                np.zeros_like(stored))
  for _ in range(10):
    sim.step(1)
  snap = sim.state.snapshot()
  snap_qpos = sim.state.qpos.cpu().numpy().copy()
  snap_qvel = sim.state.qvel.cpu().numpy().copy()
  snap_time = float(sim.state.time.cpu().numpy()[0])
  sim.step(1)
  stale = sim.step_sensordata().copy()
  sim.state.restore(snap)
  np.testing.assert_array_equal(sim.step_sensordata(), stale)
  sim.step(1)
  d = mujoco.MjData(model)
  d.qpos[:] = snap_qpos[0]
  d.qvel[:] = snap_qvel[0]
  d.time = snap_time
  mujoco.mj_step(model, d)
  np.testing.assert_allclose(sim.step_sensordata()[0], np.asarray(d.sensordata),
                             rtol=2e-4, atol=2e-4)
