# Copyright 2026 The MuJoCo Metal contributors
# Licensed under the Apache License, Version 2.0.
"""Milestone 016: ACC force families via post-constraint RNE.

Touch, accelerometer, force, torque, actuator/joint/tendon forces,
limit forces and frame accelerations against the pinned oracle
(MjData fields + mj_contactForce), native query parity, and stored
step-stage parity. Contact-sensor and ray families arrive in the
spatial commit on this branch.
"""

import os

import mujoco
import numpy as np
import pytest

from mujoco_metal.model import load_model
from mujoco_metal.sensors import sensor_oracle

_XML = '''<mujoco model="force_families">
  <option timestep="0.002" integrator="Euler" gravity="0 0 -9.81"/>
  <worldbody>
    <geom name="floor" type="plane" size="5 5 0.1"/>
    <site name="pad" pos="0.3 0 0.02" size="0.15 0.15 0.01"/>
    <body name="swing" pos="0 0 0.6">
      <joint name="h" type="hinge" axis="0 1 0" limited="true" range="-0.5 0.5" damping="0.2"/>
      <geom name="arm" type="capsule" size="0.04 0.25" mass="0.4"/>
      <site name="tip" pos="0 -0.3 0"/>
      <site name="gauge" pos="0 0.1 0"/>
    </body>
    <body name="ball" pos="0.3 0 0.35">
      <freejoint name="freeB"/>
      <geom name="ball_geom" type="sphere" size="0.09" mass="0.3"/>
    </body>
  </worldbody>
  <actuator><motor name="drive" joint="h" gear="1.5"/></actuator>
  <sensor>
    <touch site="pad"/>
    <accelerometer site="tip"/>
    <force site="gauge"/>
    <torque site="gauge"/>
    <actuatorfrc actuator="drive"/>
    <jointactuatorfrc joint="h"/>
    <jointlimitfrc joint="h"/>
    <framelinacc objtype="body" objname="swing"/>
    <frameangacc objtype="body" objname="swing"/>
    <jointpos joint="h"/><clock/>
  </sensor></mujoco>'''


def _feed(model, qpos, qvel, times, ctrl=None):
  poses = {key: [] for key in (
      "body_pos", "body_quat", "geom_pos", "geom_quat", "site_pos",
      "site_quat", "inertial_pos", "inertial_quat", "cvel", "root_com")}
  descriptor = load_model(model)
  extra_lists = {k: [] for k in (
      "cacc", "cfrc_int", "subtree_com", "actuator_force", "qfrc_actuator",
      "efc_type", "efc_id", "efc_force", "contact_geom", "contact_frame",
      "contact_pos", "contact_dist", "contact_force", "contact_efc", "ncon")}
  expected, nefc_max, ncon_max, ne, nf = [], 0, 0, 0, 0
  for w in range(len(qpos)):
    data = mujoco.MjData(model)
    data.qpos[:] = qpos[w]
    data.qvel[:] = qvel[w]
    data.time = times[w]
    if ctrl is not None:
      data.ctrl[:] = ctrl[w]
    mujoco.mj_forward(model, data)
    expected.append(data.sensordata.copy())
    fk = descriptor.forward_kinematics(qpos[w])
    for key in poses:
      poses[key].append(fk[key] if key in fk else
                        data.cvel.copy() if key == "cvel"
                        else data.subtree_com.copy())
    extra_lists["cacc"].append(np.asarray(data.cacc).copy())
    extra_lists["cfrc_int"].append(np.asarray(data.cfrc_int).copy())
    extra_lists["subtree_com"].append(np.asarray(data.subtree_com).copy())
    extra_lists["actuator_force"].append(np.asarray(data.actuator_force).copy())
    extra_lists["qfrc_actuator"].append(np.asarray(data.qfrc_actuator).copy())
    nefc_max = max(nefc_max, int(data.nefc))
    ncon_max = max(ncon_max, int(data.ncon))
    ne, nf = int(data.ne), int(data.nf)
    for k, src in (("efc_type", data.efc_type), ("efc_id", data.efc_id),
                   ("efc_force", data.efc_force)):
      extra_lists[k].append(np.asarray(src).copy())
    cg = np.zeros((data.ncon, 2))
    cd = np.zeros((data.ncon,))
    cf = np.zeros((data.ncon, 3, 3))
    cp = np.zeros((data.ncon, 3))
    cf6 = np.zeros((data.ncon, 6))
    ce = np.zeros((data.ncon,), dtype=np.int64)
    for j in range(data.ncon):
      c = data.contact[j]
      cg[j] = [c.geom[0], c.geom[1]]
      cd[j] = float(c.dist)
      cf[j] = np.asarray(c.frame).reshape(3, 3)
      cp[j] = np.asarray(c.pos)
      f = np.zeros(6)
      mujoco.mj_contactForce(model, data, j, f)
      cf6[j] = f
      ce[j] = int(c.efc_address)
    extra_lists["contact_geom"].append(cg)
    extra_lists["contact_frame"].append(cf)
    extra_lists["contact_pos"].append(cp)
    extra_lists["contact_dist"].append(cd)
    extra_lists["contact_force"].append(cf6)
    extra_lists["contact_efc"].append(ce)
    extra_lists["ncon"].append(int(data.ncon))
  poses = {k: np.asarray(v) for k, v in poses.items()}
  width = max(nefc_max, 1)
  efc = {}
  for k in ("efc_type", "efc_id", "efc_force"):
    mat = np.zeros((len(qpos), width))
    for w, row in enumerate(extra_lists[k]):
      mat[w, :len(row)] = row
    efc[k] = mat
  cw = max(ncon_max, 1)
  con = {}
  for k, s in (("contact_geom", (cw, 2)), ("contact_frame", (cw, 3, 3)),
               ("contact_pos", (cw, 3)), ("contact_dist", (cw,)),
               ("contact_force", (cw, 6))):
    mat = np.full((len(qpos),) + s, -1.0)
    for w, row in enumerate(extra_lists[k]):
      mat[w, :len(row)] = row
    con[k] = mat
  ce = np.full((len(qpos), cw), -1, dtype=np.int64)
  for w, row in enumerate(extra_lists["contact_efc"]):
    ce[w, :len(row)] = row
  nw = np.zeros(len(qpos), dtype=np.int64)
  for w in range(len(qpos)):
    nw[w] = len(extra_lists["contact_geom"][w])
  extra = {"cacc": np.asarray(extra_lists["cacc"]),
           "cfrc_int": np.asarray(extra_lists["cfrc_int"]),
           "subtree_com": np.asarray(extra_lists["subtree_com"]),
           "actuator_force": np.asarray(extra_lists["actuator_force"]),
           "qfrc_actuator": np.asarray(extra_lists["qfrc_actuator"]),
           "ne": ne, "nf": nf, **efc, **con,
           "contact_efc": ce, "ncon": nw}
  return poses, extra, np.asarray(expected)


def _states(model, descriptor, batch=3):
  qpos = np.tile(np.asarray(model.qpos0, dtype=np.float64), (batch, 1))
  qvel = np.zeros((batch, model.nv))
  for w in range(batch):
    qpos[w, 0] = 0.3 * (w + 1) - 0.6
    qvel[w] = np.linspace(-1.2, 1.5, model.nv) + 0.3 * w
  return qpos, qvel, 0.4 + np.arange(batch) * 0.85


def test_force_oracle_matches_mujoco_forward():
  model = mujoco.MjModel.from_xml_string(_XML)
  descriptor = load_model(model)
  qpos, qvel, times = _states(model, descriptor)
  ctrl = np.stack([np.full(model.nu, 0.4 * (w + 1)) for w in range(len(qpos))])
  poses, extra, expected = _feed(model, qpos, qvel, times, ctrl)
  acc = (int(mujoco.mjtStage.mjSTAGE_POS), int(mujoco.mjtStage.mjSTAGE_VEL),
         int(mujoco.mjtStage.mjSTAGE_ACC))
  actual = sensor_oracle(model, qpos, qvel, times, poses, extra=extra, stages=acc)
  np.testing.assert_allclose(actual, expected, rtol=3e-8, atol=2e-8)


@pytest.mark.gpu
@pytest.mark.skipif(os.getenv("MUJOCO_METAL_RUN_GPU") != "1", reason="opt-in GPU")
def test_native_force_query_matches_forward_gpu():
  from mujoco_metal.simulation import MetalSimulation
  model = mujoco.MjModel.from_xml_string(_XML)
  descriptor = load_model(model)
  qpos, qvel, times = _states(model, descriptor)
  ctrl = np.stack([np.full(model.nu, 0.4 * (w + 1)) for w in range(len(qpos))])
  sim = MetalSimulation(model, batch_size=len(qpos), profile="integrated_euler_v1")
  sim.reset(qpos=qpos.astype(np.float32), qvel=qvel.astype(np.float32))
  sim.step(1, ctrl=ctrl.astype(np.float32))
  # Query at the post-step state against a matching forward oracle.
  gq = sim.state.qpos.cpu().numpy().astype(np.float64)
  gv = sim.state.qvel.cpu().numpy().astype(np.float64)
  # Controls are held; query with the same ctrl via a stepped reference.
  got = sim.sensor_values().cpu().numpy()
  times0 = np.full(len(qpos), float(sim.state.time.cpu().numpy()[0]))
  poses, extra, _ = _feed(model, gq, gv, times0, ctrl)
  acc = (int(mujoco.mjtStage.mjSTAGE_POS), int(mujoco.mjtStage.mjSTAGE_VEL),
         int(mujoco.mjtStage.mjSTAGE_ACC))
  want = sensor_oracle(model, gq, gv, times0, poses, extra=extra, stages=acc)
  np.testing.assert_allclose(got, want, rtol=2e-4, atol=2e-4)


@pytest.mark.gpu
@pytest.mark.skipif(os.getenv("MUJOCO_METAL_RUN_GPU") != "1", reason="opt-in GPU")
def test_stored_force_sample_matches_mj_step_gpu():
  from mujoco_metal.simulation import MetalSimulation
  model = mujoco.MjModel.from_xml_string(_XML)
  descriptor = load_model(model)
  qpos, qvel, _ = _states(model, descriptor, batch=2)
  ctrl = np.stack([np.full(model.nu, 0.3 * (w + 1)) for w in range(2)])
  sim = MetalSimulation(model, batch_size=2, profile="integrated_euler_v1")
  sim.reset(qpos=qpos.astype(np.float32), qvel=qvel.astype(np.float32))
  cpus = []
  for w in range(2):
    d = mujoco.MjData(model)
    d.qpos[:] = qpos[w]
    d.qvel[:] = qvel[w]
    d.ctrl[:] = ctrl[w]
    cpus.append(d)
  for _ in range(60):
    sim.step(1, ctrl=ctrl.astype(np.float32))
    for w, d in enumerate(cpus):
      d.ctrl[:] = ctrl[w]
      mujoco.mj_step(model, d)
  stored = sim.step_sensordata()
  for w, d in enumerate(cpus):
    np.testing.assert_allclose(
        stored[w], np.asarray(d.sensordata), rtol=2e-4, atol=2e-4,
        err_msg=f"world {w}")
