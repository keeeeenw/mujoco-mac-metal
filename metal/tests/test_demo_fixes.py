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
from pathlib import Path

import mujoco
import pytest

EXAMPLES = Path(__file__).resolve().parents[1] / "examples"


def _import(name, monkeypatch):
  monkeypatch.syspath_prepend(str(EXAMPLES))
  return importlib.import_module(name)


def test_gripper_schedule_clears_furniture_and_delivers_both_balls(monkeypatch):
  module = _import("muscle_gripper", monkeypatch)
  result = module.run(mode="cpu", check=True)
  clears = result["minimum_geometry_distance"]
  # run(check=True) already enforces the strict -1e-5 overlap gate on base
  # readings; reported minima additionally subtract the 0.5 mm ensemble
  # bound, so the floor here is bound-aware, not penetration.
  assert min(clears.values()) >= -6e-4
  # Jaw self-pairs clear by centimeters through the whole trajectory (the
  # fully-closed empty pose would intersect; the schedule never goes there).
  assert clears["ledgeL/ledgeR"] > 0.01, clears["ledgeL/ledgeR"]
  assert clears["fingerL_geom/fingerR_geom"] > 0.05
  assert clears["ledgeL/fingerR_geom"] > 0.01
  assert clears["ledgeR/fingerL_geom"] > 0.01
  # Functional grasp/rest penetration stays within the compliant bound.
  assert max(result["max_functional_penetration"].values()) < 0.015
  x, z = result["released_ball"]
  assert 0.45 < x < 0.75 and 0.12 < z < 0.15
  x, z = result["block_rel"]
  assert abs(x + 0.5) < 0.1 and abs(z - 0.255) < 0.03
  assert result["ball_bin_hits"] > 100
  assert result["latched_bin_hits"] == 0
  assert result["finger_contacts"] > 100


def test_gripper_monitor_rejects_fully_closed_empty_pose(monkeypatch):
  """Negative control: symmetric full close with no ball intersects 16 mm.

  Proves the jaw self-pair coverage is live: the same monitor that passes
  the delivery schedule fails the unreachable fully-closed empty pose.
  """
  module = _import("muscle_gripper", monkeypatch)
  model = module._load_model()
  from demo_clearance import ClearanceMonitor
  monitor = ClearanceMonitor(
      model, [("ledgeL", "ledgeR"), ("ledgeL", "fingerR_geom"),
              ("fingerL_geom", "fingerR_geom")])
  data = mujoco.MjData(model)
  mujoco.mj_resetDataKeyframe(model, data, 0)
  mujoco.mj_forward(model, data)
  data.qpos[0] = -0.065
  data.qpos[1] = -0.065
  monitor.sample(data.qpos, data.mocap_pos, data.mocap_quat)
  with pytest.raises(AssertionError, match="Geometry overlap"):
    monitor.check()


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
  assert result["minimum_geometry_distance"]["deckB_geom/tray"] >= -0.007
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
