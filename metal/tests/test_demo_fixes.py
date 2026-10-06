# Copyright 2026 The MuJoCo Metal contributors
# Licensed under the Apache License, Version 2.0.
"""CPU regression checks for demo geometry and scheduled outcomes (no GPU).

Covers the demo-fixes scope: gripper furniture clearance plus both
deliveries, bridge landing on a solid stop, suspension strut clearance,
lighting consistency, and negative controls proving the clearance monitor
catches the historic overlaps (unclamped jaw self-intersection, missing
deck/stop contact, centered strut) instead of passing silently.
"""

import importlib
import itertools
from pathlib import Path

import mujoco
import numpy as np
import pytest

EXAMPLES = Path(__file__).resolve().parents[1] / "examples"


def _import(name, monkeypatch):
  monkeypatch.syspath_prepend(str(EXAMPLES))
  return importlib.import_module(name)


def test_gripper_schedule_clears_furniture_and_delivers_both_balls(monkeypatch):
  module = _import("muscle_gripper", monkeypatch)
  result = module.run(mode="cpu", check=True)
  clears = result["minimum_geometry_distance"]
  assert min(clears.values()) >= -1e-5
  assert min(result["conservative_geometry_distance_lower_bound"].values()) >= -6e-4
  assert max(result["geometry_distance_uncertainty_bound"].values()) <= 5e-4 + 1e-10
  # Jaw tips stay apart, and the two support ledges remain geometrically split.
  assert clears["fingerL_geom/fingerR_geom"] > 0.05
  assert clears["ledgeL/fingerR_geom"] >= -1e-5
  assert clears["ledgeR/fingerL_geom"] >= -1e-5
  # Functional grasp/rest penetration stays within the compliant bound.
  assert max(result["max_functional_penetration"].values()) < 0.001
  assert result["max_actual_contact_penetration_m"]["cpu"] < 0.001
  assert min(result["peak_payload_heights_m"].values()) > 0.28
  assert abs(result["released_ball_xyz"][1]) < 0.04
  assert abs(result["block_rel_xyz"][1] - 0.30) < 0.04
  x, z = result["released_ball"]
  assert 0.45 < x < 0.75 and 0.12 < z < 0.15
  x, z = result["block_rel"]
  assert abs(x + 0.5) < 0.1 and abs(z - 0.255) < 0.03
  assert result["ball_bin_hits"] > 100
  assert result["latched_bin_hits"] == 0
  assert result["support_contact_steps_cpu"] > 100
  assert result["first_red_ball_support_contact_step_cpu"] is not None
  assert result["first_red_ball_bin_contact_step"] is not None
  assert result["first_red_ball_support_contact_step_cpu"] < result["first_red_ball_bin_contact_step"]
  assert result["first_large_divergence_steps"]["translation_step_1cm"] is None
  assert set(result["max_state_error_components"]) == {
      "translation_m", "orientation_geodesic_rad", "linear_velocity_m_s",
      "angular_velocity_rad_s", "scalar_joint_position", "scalar_joint_velocity",
  }


def test_gripper_quaternion_error_is_sign_invariant_and_unit_separated(monkeypatch):
  module = _import("muscle_gripper", monkeypatch)
  model = module._load_model()
  data = mujoco.MjData(model)
  mujoco.mj_resetDataKeyframe(model, data, 0)
  qpos_a = data.qpos.copy()
  qvel_a = data.qvel.copy()
  qpos_b = qpos_a.copy()
  qvel_b = qvel_a.copy()
  free = int(mujoco.mjtJoint.mjJNT_FREE)
  free_joint = next(j for j in range(model.njnt) if model.jnt_type[j] == free)
  qa = int(model.jnt_qposadr[free_joint])
  qpos_b[qa + 3:qa + 7] *= -1.0
  qpos_b[qa] += 0.02
  va = int(model.jnt_dofadr[free_joint])
  qvel_b[va:va + 3] += (0.0, 0.03, 0.0)
  errors = module._state_error_breakdown(model, qpos_a, qvel_a, qpos_b, qvel_b)
  assert errors["translation_m"] == pytest.approx(0.02)
  assert errors["orientation_geodesic_rad"] == pytest.approx(0.0, abs=1e-7)
  assert errors["linear_velocity_m_s"] == pytest.approx(0.03)


def test_gripper_supported_joint_endpoints_clear_empty_jaws(monkeypatch):
  """The split supporting jaws stop before crossing through each other."""
  module = _import("muscle_gripper", monkeypatch)
  model = module._load_model()
  grip_l = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "gripL")
  grip_r = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "gripR")
  for joint in (grip_l, grip_r):
    assert model.jnt_limited[joint]
    assert model.jnt_range[joint, 0] == pytest.approx(-0.018)
  monitor = module._clearance_monitor(model)
  data = mujoco.MjData(model)
  mujoco.mj_resetDataKeyframe(model, data, 0)
  data.qpos[2] = 2.0
  data.qpos[9] = 3.0
  mujoco.mj_forward(model, data)
  for _ in range(600):
    data.ctrl[:] = (1.0, 0.0, 1.0, 0.0)
    mujoco.mj_step(model, data)
    monitor.sample(data.qpos, data.mocap_pos, data.mocap_quat)
  monitor.check()
  assert max(data.qpos[:2]) < -0.017, data.qpos[:2]
  assert monitor.minimum["ledgeL/ledgeR"] > 0.0


def test_gripper_actuator_control_corners_do_not_clip_empty_hardware(monkeypatch):
  """All 16 corners of the admitted four-muscle control box remain clear."""
  module = _import("muscle_gripper", monkeypatch)
  model = module._load_model()
  assert model.actuator_ctrllimited.all()
  assert (model.actuator_ctrlrange == (0.0, 1.0)).all()
  for control in itertools.product((0.0, 1.0), repeat=model.nu):
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, 0)
    # Remove both spheres: this explicitly exercises empty-jaw closure.
    data.qpos[2] = 2.0
    data.qpos[9] = 3.0
    mujoco.mj_forward(model, data)
    monitor = module._clearance_monitor(model)
    for _ in range(300):
      data.ctrl[:] = control
      mujoco.mj_step(model, data)
      monitor.sample(data.qpos, data.mocap_pos, data.mocap_quat)
    monitor.check()
    assert monitor.minimum["ledgeL/ledgeR"] > 0.0


def test_gripper_slab_proves_far_pair_separation(monkeypatch):
  """The AABB-slab path defeats the box-box GJK face degeneracy.

  At the start keyframe the ledgeL/fingerR_geom pair is ~0.19 m apart but
  raw MuJoCo GJK returns exactly 0.0; the monitor must still report
  separation above 0.1 m.
  """
  module = _import("muscle_gripper", monkeypatch)
  model = module._load_model()
  from demo_clearance import ClearanceMonitor
  monitor = ClearanceMonitor(model, [("ledgeL", "fingerR_geom")])
  data = mujoco.MjData(model)
  mujoco.mj_resetDataKeyframe(model, data, 0)
  mujoco.mj_forward(model, data)
  monitor.sample(data.qpos, data.mocap_pos, data.mocap_quat)
  assert monitor.reported["ledgeL/fingerR_geom"] > 0.1


def test_bridge_lands_on_stop_instead_of_passing_through(monkeypatch):
  module = _import("cargo_bridge", monkeypatch)
  result = module.run(mode="cpu", check=True)
  assert result["support_contact_steps_cpu"] > 100
  assert result["minimum_geometry_distance"]["deckB_geom/tray"] >= -0.001
  assert result["conservative_geometry_distance_lower_bound"]["deckB_geom/tray"] >= -0.0015
  assert result["latched_payload_z"] - result["released_payload_z"] > 0.25
  assert result["released_payload_z"] < 0.35
  assert result["payload_contacts"] > 50


def test_bridge_stop_pair_is_explicit(monkeypatch):
  """Guards the historic pass-through: the deck/stop pair must exist."""
  text = (EXAMPLES / "cargo_bridge.xml").read_text()
  assert 'geom1="deckB_geom" geom2="tray"' in text


def test_suspension_strut_clears_cargo_through_complete_rollout(monkeypatch):
  module = _import("suspension_platform", monkeypatch)
  result = module.run(mode="cpu", check=True)
  assert min(result["minimum_geometry_distance"].values()) > 0.1
  assert result["cargo_contacts"] > 100
  assert result["strut_hits"] > 0 and result["cable_hits"] > 0
  assert result["strut_tip_overrun_m"] <= 0.001
  assert result["cable_limit_overrun_m"] <= 0.001


def test_suspension_native_state_tendon_query_matches_cpu_kinematic_query(monkeypatch):
  module = _import("suspension_platform", monkeypatch)
  model = module._load_model()
  qpos = model.qpos0.copy()
  mocap_pos = np.asarray(model.key_mpos).reshape(-1, 3)[0]
  mocap_quat = np.asarray(model.key_mquat).reshape(-1, 4)[0]
  expected = mujoco.MjData(model)
  expected.qpos[:] = qpos
  expected.mocap_pos[0] = mocap_pos
  expected.mocap_quat[0] = mocap_quat
  mujoco.mj_forward(model, expected)
  queried = module._tendon_length_at(
      model, mujoco.MjData(model), qpos, np.asarray([mocap_pos]),
      np.asarray([mocap_quat]), 3)
  assert queried == pytest.approx(float(expected.ten_length[3]), abs=1e-12)


def test_suspension_strut_mount_is_offset(monkeypatch):
  """Guards the historic centered-strut crossing: the mount sits at y=-0.5."""
  model = mujoco.MjModel.from_xml_path(
      str(EXAMPLES / "suspension_platform.xml"))
  body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "strut")
  pos = model.body_pos[body]
  assert abs(float(pos[1]) + 0.5) < 1e-6, list(pos)


def test_demo_lighting_matches_gallery():
  """All five demos share the marble-machine light and checker floor."""
  light = ('<light directional="true" pos="0 -2 5" dir="0 0.3 -1" '
           'diffuse="0.9 0.9 0.9" specular="0.3 0.3 0.3"/>')
  texture = 'name="floor_grid" type="2d" builtin="checker"'
  for name in ("muscle_gripper", "cargo_bridge", "suspension_platform",
               "magnetic_crane", "cable_drawbridge"):
    text = (EXAMPLES / f"{name}.xml").read_text()
    assert light in text, name
    assert texture in text, name
    assert 'material="floor_material"' in text, name


def test_clearance_monitor_detects_overlap_without_collision_pair(monkeypatch):
  monitor_type = _import("demo_clearance", monkeypatch).ClearanceMonitor
  model = mujoco.MjModel.from_xml_string('''<mujoco><worldbody>
    <geom name="fixed" type="box" size="0.1 0.1 0.1" contype="0" conaffinity="0"/>
    <body pos="0 0 0.05"><freejoint/><geom name="moving" type="sphere" size="0.05"
      contype="0" conaffinity="0"/></body>
  </worldbody></mujoco>''')
  monitor = monitor_type(model, [("fixed", "moving")])
  data = mujoco.MjData(model)
  mujoco.mj_forward(model, data)
  monitor.sample(data.qpos)
  with pytest.raises(AssertionError, match="Geometry overlap"):
    monitor.check()


def _box_pair_model(second_pos, second_quat=None):
  quat = f"quat='{second_quat}'" if second_quat else ""
  return mujoco.MjModel.from_xml_string(
      f'''<mujoco><worldbody>
    <geom name="a" type="box" size="0.1 0.1 0.1" contype="0" conaffinity="0"/>
    <body pos="{second_pos}"><freejoint/><geom name="b" type="box"
      size="0.1 0.1 0.1" contype="0" conaffinity="0" {quat}/></body>
  </worldbody></mujoco>''')


def test_clearance_monitor_catches_coincident_boxes(monkeypatch):
  """R09: exactly coincident volumes read GJK 0.0; SAT must still fail."""
  monitor_type = _import("demo_clearance", monkeypatch).ClearanceMonitor
  model = _box_pair_model("0 0 0")
  monitor = monitor_type(model, [("a", "b")])
  data = mujoco.MjData(model)
  mujoco.mj_forward(model, data)
  assert float(mujoco.mj_geomDistance(model, data, 0, 1, 1.0, None)) == 0.0
  monitor.sample(data.qpos)
  with pytest.raises(AssertionError, match="Geometry overlap"):
    monitor.check()


def test_clearance_monitor_catches_rotated_box_intersection(monkeypatch):
  """R09: 30-degree yawed overlap reads GJK 0.0 or positive; SAT fails it."""
  monitor_type = _import("demo_clearance", monkeypatch).ClearanceMonitor
  model = _box_pair_model("0.05 0 0", "0 0 0.259 0.966")
  monitor = monitor_type(model, [("a", "b")])
  data = mujoco.MjData(model)
  mujoco.mj_forward(model, data)
  monitor.sample(data.qpos)
  with pytest.raises(AssertionError, match="Geometry overlap"):
    monitor.check()


def test_clearance_monitor_passes_near_and_flush_boxes(monkeypatch):
  """R09: no false positives — 1 mm gap and flush-face touch both pass."""
  monitor_type = _import("demo_clearance", monkeypatch).ClearanceMonitor
  for pos in ("0 0 0.201", "0 0 0.2"):
    model = _box_pair_model(pos)
    monitor = monitor_type(model, [("a", "b")])
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    monitor.sample(data.qpos)
    monitor.check()  # must not raise


def test_clearance_lower_bound_is_not_reported_as_measured_distance(monkeypatch):
  monitor_type = _import("demo_clearance", monkeypatch).ClearanceMonitor
  model = mujoco.MjModel.from_xml_string('''<mujoco><worldbody>
    <geom name="floor" type="plane" size="2 2 0.1"/>
    <body pos="0 0 0.1"><freejoint/><geom name="box" type="box" size="0.1 0.1 0.1"/></body>
  </worldbody></mujoco>''')
  monitor = monitor_type(model, [("floor", "box")])
  data = mujoco.MjData(model)
  mujoco.mj_forward(model, data)
  monitor.sample(data.qpos)
  assert abs(monitor.minimum["floor/box"]) < 1e-10
  assert monitor.reported["floor/box"] < -1e-5
  assert 1e-5 < monitor.uncertainty["floor/box"] <= 5e-4 + 1e-10
  monitor.check()


def test_gripper_incomplete_schedule_cannot_claim_delivery_check(monkeypatch):
  module = _import("muscle_gripper", monkeypatch)
  with pytest.raises(ValueError, match="complete demo checks"):
    module.run(mode="cpu", check=True, steps=3000)


def test_gripper_delivers_distinct_payload_shapes(monkeypatch):
  module = _import("muscle_gripper", monkeypatch)
  model = module._load_model()
  red=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_GEOM,"ball_geom")
  blue=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_GEOM,"ball2_geom")
  assert model.geom_type[red]==int(mujoco.mjtGeom.mjGEOM_SPHERE)
  assert model.geom_type[blue]==int(mujoco.mjtGeom.mjGEOM_CAPSULE)
