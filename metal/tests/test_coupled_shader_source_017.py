# Copyright 2026 The MuJoCo Metal contributors
# Licensed under the Apache License, Version 2.0.
"""Verify native witnesses cannot silently compile a foreign shader unit."""
from pathlib import Path

import pytest


def test_coupled_shader_source_requires_its_own_root_before_reading(monkeypatch):
  import mujoco_metal.coupled_constraints as module
  own = Path(__file__).parents[1] / "mujoco_metal" / "shaders"
  assert module._SHADER.resolve().parent == own.resolve()
  # A foreign imported backend must fail before any shader file is read.
  def unexpected_read(*args, **kwargs):
    raise AssertionError("read a shader before checking source ownership")
  monkeypatch.setattr(Path, "read_text", unexpected_read)
  with pytest.raises(AssertionError, match="outside the expected source root"):
    module._coupled_shader_source(expected_root=own / "foreign")


def test_production_coupled_shader_unit_has_complete_dependency_order():
  from mujoco_metal.coupled_constraints import _coupled_shader_source
  own = Path(__file__).parents[1] / "mujoco_metal" / "shaders"
  source = _coupled_shader_source(expected_root=own)
  # Actual helper definition must precede its collision-stage use. The two
  # equality compilation modes keep standalone and coupled entrypoints apart.
  definition = source.index("inline bool common_ccd_production_uses_mjc_convex(")
  call = source.index("bool common_rigid_convex = common_ccd_production_uses_mjc_convex(")
  assert definition < call
  enable = source.index("#define MUJOCO_METAL_EQUALITY_HELPERS_ONLY")
  disable = source.index("#undef MUJOCO_METAL_EQUALITY_HELPERS_ONLY")
  solver = source.index("inline int solve_primal_accel_scalar(")
  assert enable < disable < solver
