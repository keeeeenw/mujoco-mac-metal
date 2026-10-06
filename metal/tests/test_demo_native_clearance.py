# Copyright 2026 The MuJoCo Metal contributors
# Licensed under the Apache License, Version 2.0.
"""Full native demo qualification, including geometry at accepted GPU poses.

These long tests retain each demo's complete prescribed schedule and independent
CPU comparisons. CPU-only geometry and lighting checks live in test_demo_fixes.
"""

import importlib
import json
import os
from pathlib import Path

import pytest


@pytest.mark.gpu
@pytest.mark.skipif(os.getenv("MUJOCO_METAL_RUN_GPU") != "1",
                    reason="full native demo qualification requires GPU opt-in")
@pytest.mark.parametrize("name,steps", [
    ("muscle_gripper", 22000),
    ("cargo_bridge", 2400),
    ("suspension_platform", 4800),
])
def test_full_native_demo_clearance_and_outcomes(name, steps, monkeypatch):
  monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "examples"))
  result = importlib.import_module(name).run(
      steps=steps, mode="metal", check=True)
  # Keep the complete measurements in the qualification log, including fields
  # whose checks occur in the demo rather than reducing the result to a pass.
  print(json.dumps({"demo": name, "metrics": result}, sort_keys=True))
  assert result["steps"] == steps
  assert result["minimum_geometry_distance"]
  if name == "muscle_gripper":
    assert min(result["minimum_geometry_distance"].values()) >= -1e-5
    assert result["max_actual_contact_penetration_m"]["metal"] < .001
    assert result["native_contact_counts"]["payload_support_steps"] > 100
    assert result["native_contact_counts"]["red_bin_steps"] > 20
  elif name == "cargo_bridge":
    assert result["max_actual_deck_tray_overlap_m"] <= .001
    assert result["native_latch_before"] > 1
    assert result["native_latch_after"] < 1e-4
  else:
    assert result["max_actual_hardware_overlap_m"] == 0
    assert result["native_strut_tip_overrun_m"] <= .001
    assert result["native_cable_limit_overrun_m"] <= .001
    assert result["nat_strut_hits"] > 100
