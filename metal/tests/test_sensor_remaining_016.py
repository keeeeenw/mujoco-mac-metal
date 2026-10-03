# Copyright 2026 The MuJoCo Metal contributors
# Licensed under the Apache License, Version 2.0.
"""Milestone 016: remaining sensor types + query semantics.

Covers tendonactfrc/tendonlimitfrc, geomnormal/geomfromto, and query
order/repeated/batch semantics against the pinned oracle and native.
"""
import os
import mujoco
import numpy as np
import pytest
from mujoco_metal.model import load_model
from mujoco_metal.sensors import sensor_oracle

ACC = (int(mujoco.mjtStage.mjSTAGE_POS), int(mujoco.mjtStage.mjSTAGE_VEL),
       int(mujoco.mjtStage.mjSTAGE_ACC))

_XML = '''<mujoco model="remaining">
  <option timestep="0.002" integrator="Euler" gravity="0 0 -9.81"/>
  <worldbody>
    <geom name="floor" type="plane" size="5 5 0.1"/>
    <body name="hub" pos="0 0 0.5">
      <joint name="h" type="hinge" axis="0 0 1" limited="true" range="-1 1"/>
      <geom name="ga" type="sphere" size="0.1" mass="0.3"/>
      <body name="arm" pos="0.4 0 0">
        <joint name="s" type="slide" axis="1 0 0"/>
        <geom name="gb" type="sphere" size="0.08" mass="0.2"/>
      </body>
    </body>
  </worldbody>
  <tendon>
    <fixed name="t" limited="true" range="-0.5 0.5"><joint joint="h" coef="1.5"/><joint joint="s" coef="0.5"/></fixed>
  </tendon>
  <actuator>
    <motor name="mj" joint="h" gear="1"/>
    <motor name="mt" tendon="t" gear="2"/>
  </actuator>
  <sensor>
    <tendonactuatorfrc tendon="t"/>
    <tendonlimitfrc tendon="t"/>
    <jointlimitfrc joint="h"/>
    <distance geom1="ga" geom2="gb" cutoff="3"/>
    <normal geom1="ga" geom2="gb"/>
    <fromto geom1="ga" geom2="gb"/>
    <jointpos joint="h"/>
  </sensor></mujoco>'''


def _feed(model, qpos, qvel, times, ctrl):
  desc_load = load_model(model)
  poses = {k: [] for k in ("body_pos", "body_quat", "geom_pos", "geom_quat",
                           "site_pos", "site_quat", "inertial_pos",
                           "inertial_quat", "cvel", "root_com")}
  ray = np.full((len(qpos), model.nsensor, 8), -1.0)
  geo = np.zeros((len(qpos), model.nsensor, 7))
  acc = {k: [] for k in ("cacc", "subtree_com", "actuator_force",
                         "qfrc_actuator", "efc_type", "efc_id", "efc_force")}
  ncon_max, nefc_max = 0, 0
  con = {k: [] for k in ("contact_geom", "contact_frame", "contact_pos",
                         "contact_dist", "contact_force", "contact_efc", "ncon")}
  expected = []
  for w in range(len(qpos)):
    d = mujoco.MjData(model)
    d.qpos[:] = qpos[w]; d.qvel[:] = qvel[w]; d.time = times[w]
    d.ctrl[:] = ctrl[w]
    mujoco.mj_forward(model, d)
    mujoco.mj_rnePostConstraint(model, d)
    expected.append(d.sensordata.copy())
    fk = desc_load.forward_kinematics(qpos[w])
    for k in poses:
      poses[k].append(fk[k] if k in fk else d.cvel.copy() if k == "cvel" else d.subtree_com.copy())
    acc["cacc"].append(np.asarray(d.cacc).copy())
    acc["subtree_com"].append(np.asarray(d.subtree_com).copy())
    acc["actuator_force"].append(np.asarray(d.actuator_force).copy())
    acc["qfrc_actuator"].append(np.asarray(d.qfrc_actuator).copy())
    nefc_max = max(nefc_max, int(d.nefc))
    for k, src in (("efc_type", d.efc_type), ("efc_id", d.efc_id), ("efc_force", d.efc_force)):
      acc[k].append(np.asarray(src).copy())
    # geom pairs for oracle
    for i in range(model.nsensor):
      t = int(model.sensor_type[i])
      if t in (39, 40, 41):
        otype, oid = int(model.sensor_objtype[i]), int(model.sensor_objid[i])
        rtype, rid = int(model.sensor_reftype[i]), int(model.sensor_refid[i])
        gl = ([oid] if otype == 5 else list(range(int(model.body_geomadr[oid]), int(model.body_geomadr[oid]) + int(model.body_geomnum[oid]))))
        gr = ([rid] if rtype == 5 else list(range(int(model.body_geomadr[rid]), int(model.body_geomadr[rid]) + int(model.body_geomnum[rid]))))
        best, seg = float(model.sensor_cutoff[i]), np.zeros(6)
        for a in gl:
          for b in gr:
            if a == b: continue
            ft = np.zeros(6)
            dd = mujoco.mj_geomDistance(model, d, a, b, float(model.sensor_cutoff[i]), ft)
            if dd < best: best, seg = dd, ft.copy()
        geo[w, i, 0] = best; geo[w, i, 1:7] = seg
    ncon_max = max(ncon_max, int(d.ncon))
    cg = np.full((d.ncon, 2), -1); cd = np.full((d.ncon,), -1.0)
    cf = np.full((d.ncon, 3, 3), -1.0); cp = np.full((d.ncon, 3), -1.0)
    cf6 = np.full((d.ncon, 6), -1.0); ce = np.full((d.ncon,), -1, dtype=np.int64)
    for j in range(d.ncon):
      c = d.contact[j]
      cg[j] = [c.geom[0], c.geom[1]]; cd[j] = float(c.dist)
      cf[j] = np.asarray(c.frame).reshape(3, 3); cp[j] = np.asarray(c.pos)
      f = np.zeros(6); mujoco.mj_contactForce(model, d, j, f); cf6[j] = f; ce[j] = int(c.efc_address)
    con["contact_geom"].append(cg); con["contact_frame"].append(cf)
    con["contact_pos"].append(cp); con["contact_dist"].append(cd)
    con["contact_force"].append(cf6); con["contact_efc"].append(ce)
    con["ncon"].append(int(d.ncon))
  poses = {k: np.asarray(v) for k, v in poses.items()}
  width = max(nefc_max, 1)
  efc = {}
  for k in ("efc_type", "efc_id", "efc_force"):
    m = np.zeros((len(qpos), width)); 
    for w2, r in enumerate(acc[k]): m[w2, :len(r)] = r
    efc[k] = m
  cw = max(ncon_max, 1)
  cn = {}
  for k, s in (("contact_geom", (cw, 2)), ("contact_frame", (cw, 3, 3)), ("contact_pos", (cw, 3)), ("contact_dist", (cw,)), ("contact_force", (cw, 6))):
    m = np.full((len(qpos),) + s, -1.0)
    for w2, r in enumerate(con[k]): m[w2, :len(r)] = r
    cn[k] = m
  ce = np.full((len(qpos), cw), -1, dtype=np.int64)
  for w2, r in enumerate(con["contact_efc"]): ce[w2, :len(r)] = r
  extra = {"ray": ray, "geom": geo, "cacc": np.asarray(acc["cacc"]),
           "subtree_com": np.asarray(acc["subtree_com"]),
           "actuator_force": np.asarray(acc["actuator_force"]),
           "qfrc_actuator": np.asarray(acc["qfrc_actuator"]),
           "ne": 0, "nf": 0, **efc, **cn, "contact_efc": ce,
           "ncon": np.array(con["ncon"], dtype=np.int64)}
  # actuator trn for tendonactfrc oracle
  extra["actuator_trntype"] = np.asarray(model.actuator_trntype).reshape(-1)
  extra["actuator_trnid"] = np.asarray(model.actuator_trnid).reshape(-1, 2)
  return poses, extra, np.asarray(expected)


def test_remaining_oracle_matches_forward():
  model = mujoco.MjModel.from_xml_string(_XML)
  qpos = np.tile(np.asarray(model.qpos0, float), (2, 1))
  qpos[:, 0] = [0.4, -0.6]; qpos[:, 1] = [0.1, -0.2]
  qvel = np.array([[-0.5, 0.7], [0.3, -0.4]], float)
  times = np.array([0.1, 0.2])
  ctrl = np.array([[0.5, 0.3], [-0.2, 0.8]], float)
  poses, extra, expected = _feed(model, qpos, qvel, times, ctrl)
  actual = sensor_oracle(model, qpos, qvel, times, poses, extra=extra, stages=ACC)
  np.testing.assert_allclose(actual, expected, rtol=3e-8, atol=3e-8)


@pytest.mark.gpu
@pytest.mark.skipif(os.getenv("MUJOCO_METAL_RUN_GPU") != "1", reason="opt-in GPU")
def test_remaining_native_matches_and_query_semantics_gpu():
  from mujoco_metal.simulation import MetalSimulation
  model = mujoco.MjModel.from_xml_string(_XML)
  qpos = np.tile(np.asarray(model.qpos0, float), (2, 1))
  qpos[:, 0] = [0.4, -0.6]; qpos[:, 1] = [0.1, -0.2]
  qvel = np.array([[-0.5, 0.7], [0.3, -0.4]], float)
  ctrl = np.array([[0.5, 0.3], [-0.2, 0.8]], float)
  sim = MetalSimulation(model, batch_size=2, profile="integrated_euler_v1")
  sim.reset(qpos=qpos.astype(np.float32), qvel=qvel.astype(np.float32))
  sim.step(5, ctrl=ctrl.astype(np.float32))
  gq = sim.state.qpos.cpu().numpy().astype(float)
  gv = sim.state.qvel.cpu().numpy().astype(float)
  got = sim.sensor_values().cpu().numpy()
  times0 = np.full(2, float(sim.state.time.cpu().numpy()[0]))
  poses, extra, _ = _feed(model, gq, gv, times0, ctrl)
  want = sensor_oracle(model, gq, gv, times0, poses, extra=extra, stages=ACC)
  # activity: tendonactfrc live, geomdist live
  assert np.any(np.abs(got[:, 0]) > 1e-6), got[:, 0]
  assert np.any(got[:, 3] < 3.0), got[:, 3]
  np.testing.assert_allclose(got, want, rtol=2e-4, atol=2e-4)
  # repeated queries identical; stage order preserved
  again = sim.sensor_values().cpu().numpy()
  np.testing.assert_array_equal(got, again)
  # batch independence: worlds differ
  assert not np.allclose(got[0], got[1])
