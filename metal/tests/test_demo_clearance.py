# Copyright 2026 The MuJoCo Metal contributors
# Licensed under the Apache License, Version 2.0.
"""CPU regression checks for demo geometry and complete scheduled outcomes."""

import importlib
from pathlib import Path

import mujoco
import pytest

EXAMPLES = Path(__file__).resolve().parents[1] / "examples"


def test_gripper_schedule_clears_furniture_and_delivers_both_balls(monkeypatch):
  monkeypatch.syspath_prepend(str(EXAMPLES))
  result = importlib.import_module("muscle_gripper").run(mode="cpu", check=True)
  assert min(result["minimum_geometry_distance"].values()) >= -1e-5
  x, z = result["released_ball"]
  assert 0.45 < x < 0.75 and 0.12 < z < 0.15
  x, z = result["block_rel"]
  assert abs(x + 0.5) < 0.1 and abs(z - 0.255) < 0.01
  assert result["ball_bin_hits"] > 100
  assert result["latched_bin_hits"] == 0


def test_bridge_lands_on_stop_instead_of_passing_through(monkeypatch):
  monkeypatch.syspath_prepend(str(EXAMPLES))
  result = importlib.import_module("cargo_bridge").run(mode="cpu", check=True)
  assert result["support_contact_steps_cpu"] > 100
  assert result["minimum_geometry_distance"]["deckB_geom/tray"] >= -0.007
  assert result["latched_payload_z"] - result["released_payload_z"] > 0.25


def test_suspension_strut_clears_cargo_through_complete_rollout(monkeypatch):
  monkeypatch.syspath_prepend(str(EXAMPLES))
  result = importlib.import_module("suspension_platform").run(mode="cpu", check=True)
  assert min(result["minimum_geometry_distance"].values()) > 0.1
  assert result["cargo_contacts"] > 100
  assert result["strut_hits"] > 0 and result["cable_hits"] > 0


def test_clearance_monitor_detects_overlap_without_collision_pair(monkeypatch):
  monkeypatch.syspath_prepend(str(EXAMPLES))
  monitor_type = importlib.import_module("demo_clearance").ClearanceMonitor
  model = mujoco.MjModel.from_xml_string('''<mujoco><worldbody>
    <geom name="fixed" type="box" size="0.1 0.1 0.1" contype="0" conaffinity="0"/>
    <body><freejoint/><geom name="moving" type="sphere" size="0.05"
      contype="0" conaffinity="0"/></body>
  </worldbody></mujoco>''')
  monitor = monitor_type(model, [("fixed", "moving")])
  monitor.sample(model.qpos0)
  with pytest.raises(AssertionError, match="Geometry overlap"):
    monitor.check()
