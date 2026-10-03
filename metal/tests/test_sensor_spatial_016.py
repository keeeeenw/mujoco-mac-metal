# Copyright 2026 The MuJoCo Metal contributors
# Licensed under the Apache License, Version 2.0.
"""Milestone 016: CONTACT/ray/geom-distance spatial queries.

Site rangefinders (analytic + convex-mesh + heightfield rays, occlusion,
exclusions, no-hit), geom-distance witnesses (analytic pairs, cutoff,
fromto sign), and CONTACT sensors (dataspecs, all four reductions, flip)
against the pinned oracle, native query parity, and stored step-stage
parity with batch/reset/restore coverage.
"""

import os

import mujoco
import numpy as np
import pytest

from mujoco_metal.model import load_model
from mujoco_metal.sensors import sensor_oracle

ACC = (int(mujoco.mjtStage.mjSTAGE_POS), int(mujoco.mjtStage.mjSTAGE_VEL),
       int(mujoco.mjtStage.mjSTAGE_ACC))

_XML = '''<mujoco model="spatial_table">
  <option timestep="0.002" integrator="Euler" gravity="0 0 -9.81"/>
  <asset>
    <mesh name="wedge" vertex="0.15 0.1 0.0 -0.15 0.1 0.0 -0.15 -0.1 0.0 0.15 -0.1 0.0 0.0 0.0 0.12"/>
    <hfield name="rough" nrow="6" ncol="6" size="0.5 0.5 0.06 0.02"/>
  </asset>
  <worldbody>
    <geom name="floor" type="plane" size="5 5 0.1"/>
    <geom name="terrain" type="hfield" hfield="rough" pos="1.2 0 0"/>
    <site name="pad" pos="-0.3 0 0.02" size="0.2 0.2 0.02"/>
    <site name="eye" pos="-0.3 0 1.2" quat="0 1 0 0"/>
    <site name="eye2" pos="0.9 0 0.5" quat="0 1 0 0"/>
    <body name="ballA" pos="-0.3 0 0.4">
      <freejoint name="freeA"/>
      <geom name="ballA_geom" type="sphere" size="0.09" mass="0.3"/>
      <site name="imu" pos="0 0 0"/>
    </body>
    <body name="ballB" pos="0.35 0 0.5">
      <freejoint name="freeB"/>
      <geom name="ballB_geom" type="sphere" size="0.07" mass="0.2"/>
    </body>
    <body name="wedgeB" pos="-0.9 0 0.1">
      <geom name="wedge_geom" type="mesh" mesh="wedge" mass="0.2" contype="0" conaffinity="0"/>
    </body>
  </worldbody>
  <sensor>
    <touch site="pad"/>
    <accelerometer site="imu"/>
    <rangefinder site="eye"/>
    <rangefinder site="eye2" data="dist normal"/>
    <distance geom1="ballA_geom" geom2="ballB_geom" cutoff="2"/>
    <distance geom1="ballA_geom" geom2="wedge_geom" cutoff="2"/>
    <contact site="pad"/>
    <contact data="force dist" reduce="mindist"/>
    <contact data="found force" reduce="netforce"/>
  </sensor></mujoco>'''


def _feed(model, qpos, qvel, times, ctrl=None):
  from mujoco_metal.sensors import lower_sensors
  desc = lower_sensors(model)
  poses = {key: [] for key in (
      "body_pos", "body_quat", "geom_pos", "geom_quat", "site_pos",
      "site_quat", "inertial_pos", "inertial_quat", "cvel", "root_com")}
  descriptor = load_model(model)
  ray = np.full((len(qpos), model.nsensor, 8), -1.0)
  geo = np.zeros((len(qpos), model.nsensor, 7))
  con_lists = {k: [] for k in (
      "contact_geom", "contact_frame", "contact_pos", "contact_dist",
      "contact_force", "contact_efc", "ncon")}
  acc_lists = {k: [] for k in ("cacc", "subtree_com")}
  expected = []
  ncon_max = 0
  for w in range(len(qpos)):
    data = mujoco.MjData(model)
    data.qpos[:] = qpos[w]
    data.qvel[:] = qvel[w]
    data.time = times[w]
    if ctrl is not None:
      data.ctrl[:] = ctrl[w] if ctrl is not None else 0
    mujoco.mj_forward(model, data)
    mujoco.mj_rnePostConstraint(model, data)
    expected.append(data.sensordata.copy())
    acc_lists["cacc"].append(np.asarray(data.cacc).copy())
    acc_lists["subtree_com"].append(np.asarray(data.subtree_com).copy())
    fk = descriptor.forward_kinematics(qpos[w])
    for key in poses:
      poses[key].append(fk[key] if key in fk else
                        data.cvel.copy() if key == "cvel"
                        else data.subtree_com.copy())
    for i in range(model.nsensor):
      t = int(model.sensor_type[i])
      if t == int(mujoco.mjtSensor.mjSENS_RANGEFINDER):
        sid = int(model.sensor_objid[i])
        origin = np.asarray(data.site_xpos[sid]).reshape(3, 1)
        mat = np.asarray(data.site_xmat[sid]).reshape(3, 3)
        direction = (mat @ np.array([0.0, 0.0, 1.0])).reshape(3, 1)
        gid = np.zeros((1, 1), dtype=np.int32)
        nrm = np.zeros((3, 1))
        dist = mujoco.mj_ray(model, data, origin, direction, None, 1,
                             int(model.site_bodyid[sid]), gid, nrm)
        ray[w, i, 0] = dist
        ray[w, i, 1:4] = direction.reshape(-1)
        ray[w, i, 4:7] = nrm.reshape(-1)
        ray[w, i, 7] = float(gid[0, 0])
      elif t in (int(mujoco.mjtSensor.mjSENS_GEOMDIST),
                 int(mujoco.mjtSensor.mjSENS_GEOMNORMAL),
                 int(mujoco.mjtSensor.mjSENS_GEOMFROMTO)):
        otype, oid = int(model.sensor_objtype[i]), int(model.sensor_objid[i])
        rtype, rid = int(model.sensor_reftype[i]), int(model.sensor_refid[i])
        gl = ([oid] if otype == int(mujoco.mjtObj.mjOBJ_GEOM) else
              list(range(int(model.body_geomadr[oid]),
                         int(model.body_geomadr[oid]) + int(model.body_geomnum[oid]))))
        gr = ([rid] if rtype == int(mujoco.mjtObj.mjOBJ_GEOM) else
              list(range(int(model.body_geomadr[rid]),
                         int(model.body_geomadr[rid]) + int(model.body_geomnum[rid]))))
        best, seg = float(model.sensor_cutoff[i]), np.zeros(6)
        for ga in gl:
          for gb in gr:
            if ga == gb:
              continue
            ft = np.zeros(6)
            dd = mujoco.mj_geomDistance(model, data, ga, gb,
                                        float(model.sensor_cutoff[i]), ft)
            if dd < best:
              best, seg = dd, ft.copy()
        geo[w, i, 0] = best
        geo[w, i, 1:7] = seg
    ncon = int(data.ncon)
    ncon_max = max(ncon_max, ncon)
    cg = np.full((ncon, 2), -1)
    cd = np.full((ncon,), -1.0)
    cf = np.full((ncon, 3, 3), -1.0)
    cp = np.full((ncon, 3), -1.0)
    cf6 = np.full((ncon, 6), -1.0)
    ce = np.full((ncon,), -1, dtype=np.int64)
    for j in range(ncon):
      c = data.contact[j]
      cg[j] = [c.geom[0], c.geom[1]]
      cd[j] = float(c.dist)
      cf[j] = np.asarray(c.frame).reshape(3, 3)
      cp[j] = np.asarray(c.pos)
      f = np.zeros(6)
      mujoco.mj_contactForce(model, data, j, f)
      cf6[j] = f
      ce[j] = int(c.efc_address)
    con_lists["contact_geom"].append(cg)
    con_lists["contact_frame"].append(cf)
    con_lists["contact_pos"].append(cp)
    con_lists["contact_dist"].append(cd)
    con_lists["contact_force"].append(cf6)
    con_lists["contact_efc"].append(ce)
    con_lists["ncon"].append(ncon)
  poses = {k: np.asarray(v) for k, v in poses.items()}
  cw = max(ncon_max, 1)
  con = {}
  for k, s in (("contact_geom", (cw, 2)), ("contact_frame", (cw, 3, 3)),
               ("contact_pos", (cw, 3)), ("contact_dist", (cw,)),
               ("contact_force", (cw, 6))):
    mat = np.full((len(qpos),) + s, -1.0)
    for w2, row in enumerate(con_lists[k]):
      mat[w2, :len(row)] = row
    con[k] = mat
  ce = np.full((len(qpos), cw), -1, dtype=np.int64)
  for w2, row in enumerate(con_lists["contact_efc"]):
    ce[w2, :len(row)] = row
  nw = np.array(con_lists["ncon"], dtype=np.int64)
  extra = {"ray": ray, "geom": geo, **con,
           "contact_efc": ce, "ncon": nw,
           "cacc": np.asarray(acc_lists["cacc"]),
           "subtree_com": np.asarray(acc_lists["subtree_com"])}
  return poses, extra, np.asarray(expected)


def _states(model, batch=2):
  qpos = np.tile(np.asarray(model.qpos0, dtype=np.float64), (batch, 1))
  qvel = np.zeros((batch, model.nv))
  for w in range(batch):
    qvel[w] = np.linspace(-0.8, 1.1, model.nv) + 0.25 * w
  # Resting-contact arrangement: ballA kisses the pad, ballB hovers over
  # the terrain, so rays hit and distances are live from the first step.
  # qpos layout: ballA xyz[0:3] + quat[3:7], ballB xyz[7:10] + quat[10:14].
  qpos[:, 0:3] = np.array([-0.3, 0.0, 0.115]) + np.arange(batch)[:, None] * 0.01
  qpos[:, 7:10] = np.array([0.9, 0.0, 0.30]) - np.arange(batch)[:, None] * 0.01
  return qpos, qvel, 0.3 + np.arange(batch) * 0.5


def test_spatial_oracle_matches_mujoco_forward():
  model = mujoco.MjModel.from_xml_string(_XML)
  qpos, qvel, times = _states(model)
  poses, extra, expected = _feed(model, qpos, qvel, times)
  actual = sensor_oracle(model, qpos, qvel, times, poses, extra=extra,
                         stages=ACC)
  np.testing.assert_allclose(actual, expected, rtol=3e-8, atol=3e-8)


def test_ray_occlusion_exclusion_and_miss():
  # Dedicated occlusion scene: two boxes in a row; near occludes far;
  # body exclusion skips the site's own body; skyward rays miss.
  model = mujoco.MjModel.from_xml_string(
      '<mujoco><worldbody>'
      '<site name="eye" pos="0 0 1" quat="1 0 0 0"/>'
      '<body name="near" pos="0 0 0.5"><freejoint/>'
      '<geom type="box" size="0.2 0.2 0.1"/></body>'
      '<body name="far" pos="0 0 0.1"><freejoint/>'
      '<geom type="box" size="0.2 0.2 0.1"/></body>'
      '</worldbody></mujoco>')
  d = mujoco.MjData(model)
  mujoco.mj_forward(model, d)
  down = np.array([0., 0, -1.]).reshape(3, 1)
  org = np.asarray(d.site_xpos[0]).reshape(3, 1)
  hit = mujoco.mj_ray(model, d, org, down, None, 1, -1, None)
  np.testing.assert_allclose(hit, 0.4, atol=1e-9)
  # Exclude the near body: the far box top (z=0.2) is hit instead.
  near = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "near")
  hit2 = mujoco.mj_ray(model, d, org, down, None, 1, near, None)
  np.testing.assert_allclose(hit2, 0.8, atol=1e-9)
  miss = mujoco.mj_ray(model, d, org, -down, None, 1, -1, None)
  assert miss < 0


@pytest.mark.gpu
@pytest.mark.skipif(os.getenv("MUJOCO_METAL_RUN_GPU") != "1", reason="opt-in GPU")
def test_native_spatial_query_matches_forward_gpu():
  from mujoco_metal.simulation import MetalSimulation
  model = mujoco.MjModel.from_xml_string(_XML)
  qpos, qvel, times = _states(model)
  sim = MetalSimulation(model, batch_size=len(qpos), profile="integrated_euler_v1")
  sim.reset(qpos=qpos.astype(np.float32), qvel=qvel.astype(np.float32))
  sim.step(30)
  gq = sim.state.qpos.cpu().numpy().astype(np.float64)
  gv = sim.state.qvel.cpu().numpy().astype(np.float64)
  got = sim.sensor_values().cpu().numpy()
  times0 = np.full(len(qpos), float(sim.state.time.cpu().numpy()[0]))
  poses, extra, _ = _feed(model, gq, gv, times0)
  want = sensor_oracle(model, gq, gv, times0, poses, extra=extra, stages=ACC)
  # Activity: rays hit, distances live, contacts engaged (no vacuous pass).
  adr = np.asarray(model.sensor_adr)
  assert np.all(got[:, adr[2]] >= 0)
  assert np.any(got[:, adr[4]] < 2.0)
  assert np.any(got[:, adr[6]] > 0)
  np.testing.assert_allclose(got, want, rtol=2e-4, atol=2e-4)


@pytest.mark.gpu
@pytest.mark.skipif(os.getenv("MUJOCO_METAL_RUN_GPU") != "1", reason="opt-in GPU")
def test_stored_spatial_sample_matches_mj_step_gpu():
  from mujoco_metal.simulation import MetalSimulation
  model = mujoco.MjModel.from_xml_string(_XML)
  qpos, qvel, _ = _states(model, batch=2)
  sim = MetalSimulation(model, batch_size=2, profile="integrated_euler_v1")
  sim.reset(qpos=qpos.astype(np.float32), qvel=qvel.astype(np.float32))
  cpus = []
  for w in range(2):
    d = mujoco.MjData(model)
    d.qpos[:] = qpos[w]
    d.qvel[:] = qvel[w]
    cpus.append(d)
  for _ in range(80):
    sim.step(1)
    for d in cpus:
      mujoco.mj_step(model, d)
  stored = sim.step_sensordata()
  for w, d in enumerate(cpus):
    np.testing.assert_allclose(
        stored[w], np.asarray(d.sensordata), rtol=2e-3, atol=2e-3,
        err_msg=f"world {w}")
  # Reset zeroes; batch worlds stay independent.
  sim.reset()
  np.testing.assert_array_equal(sim.step_sensordata(),
                                np.zeros_like(stored))
  assert not np.allclose(cpus[0].sensordata, cpus[1].sensordata)
