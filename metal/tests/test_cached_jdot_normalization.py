# Copyright 2026 The MuJoCo Metal contributors
# Licensed under the Apache License, Version 2.0.
"""Cached equality reference source contract for POS-to-VEL transport."""
from pathlib import Path


def test_captured_position_residual_removes_old_jdot_exactly_once():
  source = (Path(__file__).parents[1] / "mujoco_metal"
            / "coupled_constraints.py").read_text()
  capture = source.split("def capture_position_context(", 1)[1]
  capture = capture.split("def _diagnostic_views", 1)[0]
  assert "position_context[:, :, 3].add_(old_jdot_aref)" in capture
  assert "position_context[:, :, 3].sub_(old_jdot_aref)" not in capture

  shader = (Path(__file__).parents[1] / "mujoco_metal" / "shaders"
            / "coupled_constraints.metal").read_text()
  helper = shader.split("kernel void capture_cached_position_context(", 1)[1]
  helper = helper.split("kernel void ", 1)[0]
  assert "position_context[context_base + 3] = -(ar + B * jv);" in helper
  jdot = shader.split("kernel void refresh_cached_equality_jdot(", 1)[1]
  jdot = jdot.split("kernel void ", 1)[0]
  assert "extra_aref[outbase+row] = -transl.x;" in jdot
