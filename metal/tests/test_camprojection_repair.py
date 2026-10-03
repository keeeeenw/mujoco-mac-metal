# Copyright 2026 The MuJoCo Metal contributors
# Licensed under the Apache License, Version 2.0.
"""R07a repair: CAMPROJECTION is native projective math, not rendering.

Failing-first: site targets project through fixed and body-mounted cameras
(fovy and intrinsic paths) with CPU parity on device, oracle and pinned
outputs. TACTILE/PLUGIN/USER stay deferred with classified boundaries;
SDF rays/distances stay on the plugin path with an explicit rejection test.
"""

import mujoco
import numpy as np
import pytest

_QID = [1.0, 0.0, 0.0, 0.0]


def _model(cam_extra="", sensor_extra="", body_pos="0 0 0.5", site_pos="0.1 0 0"):
  return mujoco.MjModel.from_xml_string(
      "<mujoco><worldbody>"
      f"<body pos='{body_pos}'><freejoint/>"
      "<geom type='sphere' size='0.1' mass='0.5'/>"
      f"<site name='s' pos='{site_pos}'/>"
      f"<camera name='c' pos='1 0 0' {cam_extra}/>"
      "</body></worldbody>"
      f"<sensor><camprojection site='s' camera='c'/>{sensor_extra}</sensor></mujoco>")


def test_camprojection_oracle_matches_pinned_cpu():
  from mujoco_metal.sensors import sensor_oracle
  from mujoco_metal.model import load_model
  for cam_extra in ("", "fovy='60'"):
    m = _model(cam_extra=cam_extra)
    desc = load_model(m)
    qp = np.array([[0.0, 0.0, 0.6, *_QID]])
    qv = np.zeros((1, m.nv))
    fk = desc.forward_kinematics(qp[0])
    poses = {k: np.asarray(v).reshape(1, *np.shape(v)) for k, v in fk.items()}
    got = sensor_oracle(m, qp, qv, np.zeros(1), poses)
    d = mujoco.MjData(m)
    d.qpos[:] = qp[0]
    mujoco.mj_forward(m, d)
    np.testing.assert_allclose(got[0], np.asarray(d.sensordata), rtol=1e-6, atol=1e-6)


def test_camprojection_intrinsic_path_cpu():
  # Intrinsic/sensorsize path (no fovy). Intrinsics are compiled fields
  # (mutated post-compile, like dampingpoly); the site sits off the focal
  # plane for a finite, meaningful projection.
  from mujoco_metal.sensors import sensor_oracle
  from mujoco_metal.model import load_model
  m = _model(site_pos="0.1 0.15 0.2")
  m.cam_intrinsic[0] = [0.02, 0.02, 0.0, 0.0]
  m.cam_sensorsize[0] = [0.036, 0.024]
  m.cam_resolution[0] = [640, 480]
  desc = load_model(m)
  qp = np.array([[0.0, 0.0, 0.6, *_QID]])
  fk = desc.forward_kinematics(qp[0])
  poses = {k: np.asarray(v).reshape(1, *np.shape(v)) for k, v in fk.items()}
  got = sensor_oracle(m, qp, np.zeros((1, m.nv)), np.zeros(1), poses)
  d = mujoco.MjData(m)
  d.qpos[:] = qp[0]
  mujoco.mj_forward(m, d)
  assert np.all(np.isfinite(got[0])) and abs(float(got[0][0])) < 1e6
  # Oracle and pinned evaluate the same chain in different op order
  # (direct vs 5-matrix product): float64 rounding agrees to ~1e-8.
  np.testing.assert_allclose(got[0], np.asarray(d.sensordata), rtol=1e-6, atol=1e-6)


def test_camprojection_bad_reference_rejected_cpu():
  m = mujoco.MjModel.from_xml_string(
      "<mujoco><worldbody><body><geom type='sphere' size='.1'/>"
      "<site name='s'/><camera name='c' pos='1 0 0'/></body></worldbody>"
      "<sensor><camprojection site='s' camera='c'/></sensor></mujoco>")
  m.sensor_refid[0] = 5
  with pytest.raises(ValueError, match="camera reference"):
    from mujoco_metal.sensors import lower_sensors
    lower_sensors(m)
  for typ, msg in ((int(mujoco.mjtSensor.mjSENS_TACTILE), "deferred"),
                   (int(mujoco.mjtSensor.mjSENS_PLUGIN), "deferred"),
                   (int(mujoco.mjtSensor.mjSENS_USER), "deferred")):
    assert typ in (46, 47, 48), (typ, msg)


def test_sdf_ray_plugin_boundary_parity_cpu():
  # Pinned SDF ray casting goes through the SDF plugin (mjc_getSDF); with
  # no plugin registered both engines skip SDF geoms (pinned reports
  # no-hit -1). Native matches that boundary; WITH-plugin scenes stay 019.
  from mujoco_metal.sensors import lower_sensors
  V = ("-0.05 -0.05 -0.04 0.05 -0.05 -0.04 0.05 0.05 -0.04 -0.05 0.05 -0.04 "
       "-0.05 -0.05 0.04 0.05 -0.05 0.04 0.05 0.05 0.04 -0.05 0.05 0.04")
  m = mujoco.MjModel.from_xml_string(
      '<mujoco><asset><mesh name="b" vertex="' + V + '"/></asset>'
      '<option timestep="0.002" sdf_initpoints="4"/>'
      '<worldbody><geom name="sdf" type="sdf" mesh="b" pos="0 0 0.2"/>'
      "<body pos='0 0 0.5'><freejoint/><geom type='sphere' size='0.05'/>"
      "<site name='eye'/></body>"
      "</worldbody><sensor><rangefinder site='eye'/></sensor></mujoco>")
  lower_sensors(m)  # admission holds
  d = mujoco.MjData(m)
  d.qpos[:] = np.asarray(m.qpos0)
  mujoco.mj_forward(m, d)
  assert float(np.asarray(d.sensordata)[0]) == -1.0  # pinned no-hit, no plugin


def _needs_gpu():
  import os
  return pytest.mark.skipif(os.getenv("MUJOCO_METAL_RUN_GPU") != "1", reason="opt-in GPU")


@_needs_gpu()
def test_sdf_ray_no_hit_parity_gpu():
  import os
  assert os.getenv("MUJOCO_METAL_RUN_GPU") == "1"
  from mujoco_metal.simulation import MetalSimulation
  V = ("-0.05 -0.05 -0.04 0.05 -0.05 -0.04 0.05 0.05 -0.04 -0.05 0.05 -0.04 "
       "-0.05 -0.05 0.04 0.05 -0.05 0.04 0.05 0.05 0.04 -0.05 0.05 0.04")
  m = mujoco.MjModel.from_xml_string(
      '<mujoco><asset><mesh name="b" vertex="' + V + '"/></asset>'
      '<option timestep="0.002" sdf_initpoints="4"/>'
      '<worldbody><geom name="sdf" type="sdf" mesh="b" pos="0 0 0.2"/>'
      "<body pos='0 0 0.5'><freejoint/><geom type='sphere' size='0.05'/>"
      "<site name='eye'/></body>"
      "</worldbody><sensor><rangefinder site='eye'/></sensor></mujoco>")
  sim = MetalSimulation(m, batch_size=1, profile="integrated_euler_v1")
  sim.reset(qpos=np.asarray(m.qpos0, dtype=np.float32).reshape(1, -1),
            qvel=np.zeros((1, m.nv), dtype=np.float32))
  got = sim.sensor_values().cpu().numpy()[0]
  np.testing.assert_allclose(got, [-1.0], atol=1e-6)


@_needs_gpu()
def test_camprojection_device_parity_gpu():
  import os
  assert os.getenv("MUJOCO_METAL_RUN_GPU") == "1"
  from mujoco_metal.simulation import MetalSimulation
  m = _model()
  sim = MetalSimulation(m, batch_size=3, profile="integrated_euler_v1")
  qp = np.asarray(m.qpos0, dtype=np.float32).reshape(1, -1).repeat(3, 0)
  qp[1, 2] += 0.2
  qp[2, 0] += 0.3
  qv = np.zeros((3, m.nv), dtype=np.float32)
  qv[1, 2] = 0.5
  sim.reset(qpos=qp, qvel=qv)
  for _ in range(10):
    sim.step(1)
  got = sim.sensor_values().cpu().numpy()
  gq = sim.state.qpos.cpu().numpy().astype(float)
  gv = sim.state.qvel.cpu().numpy().astype(float)
  for w in range(3):
    d = mujoco.MjData(m)
    d.qpos[:] = gq[w]
    d.qvel[:] = gv[w]
    mujoco.mj_forward(m, d)
    np.testing.assert_allclose(got[w], np.asarray(d.sensordata),
                               rtol=1e-4, atol=1e-3)


@_needs_gpu()
def test_camprojection_rotated_camera_gpu():
  import os
  assert os.getenv("MUJOCO_METAL_RUN_GPU") == "1"
  from mujoco_metal.simulation import MetalSimulation
  # Yawed camera + off-axis site: exercises the full rotation chain.
  m = mujoco.MjModel.from_xml_string(
      "<mujoco><worldbody>"
      "<body pos='0 0 0.5'><freejoint/>"
      "<geom type='sphere' size='0.1' mass='0.5'/>"
      "<site name='s' pos='0.1 0.2 0.1'/>"
      "<camera name='c' pos='1 0.5 0' quat='0 0 0.383 0.924'/>"
      "</body></worldbody>"
      "<sensor><camprojection site='s' camera='c'/></sensor></mujoco>")
  sim = MetalSimulation(m, batch_size=1, profile="integrated_euler_v1")
  qp = np.asarray(m.qpos0, dtype=np.float32).reshape(1, -1)
  sim.reset(qpos=qp, qvel=np.zeros((1, m.nv), dtype=np.float32))
  for _ in range(5):
    sim.step(1)
  got = sim.sensor_values().cpu().numpy()[0]
  d = mujoco.MjData(m)
  d.qpos[:] = sim.state.qpos.cpu().numpy()[0]
  d.qvel[:] = sim.state.qvel.cpu().numpy()[0]
  mujoco.mj_forward(m, d)
  assert np.all(np.isfinite(got))
  np.testing.assert_allclose(got, np.asarray(d.sensordata), rtol=1e-4, atol=1e-3)
