# Copyright 2026 The MuJoCo Metal contributors
# Licensed under the Apache License, Version 2.0.
"""Spatial sensor queries compile the same retained SDF dependencies."""


def test_spatial_sdf_precision_helpers_precede_their_actual_users():
  from mujoco_metal.sensors import _spatial_shader_source, _SHADER
  source = _spatial_shader_source()
  assert source.index("struct SdfWide") < source.index("struct SdfMatrixWide")
  root = _SHADER.parent
  for name in ("collision_primitives.metal", "convex_narrowphase.metal",
               "plugin_sdf.metal", "sdf_narrowphase.metal", "sensors_spatial.metal"):
    assert (root / name).read_text() in source
