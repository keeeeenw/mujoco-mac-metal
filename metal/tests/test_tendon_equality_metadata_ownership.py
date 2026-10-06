# Copyright 2026 The MuJoCo Metal contributors
# Licensed under the Apache License, Version 2.0.
"""Source contracts for row-family ownership and disabled equality bounds."""
from pathlib import Path


def test_tendon_refresh_clears_only_tendon_owned_rows():
  shader = (Path(__file__).parents[1] / "mujoco_metal" / "shaders"
            / "coupled_constraints.metal").read_text()
  tendon = shader.split("kernel void tendon_constraint_rows(", 1)[1]
  tendon = tendon.split("kernel void ", 1)[0]
  assert "int ten_stop = min(nr, ten_base + max(dims[17], 0));" in tendon
  assert "for (int r=max(0, ten_base);r<ten_stop;++r)" in tendon
  assert "for (int r=0;r<nr;++r)" not in tendon
  assert "if (eq_type[e] == 3)" in tendon
  assert "dbg[nr*nr+6*nr+row]=0.0f;" in tendon


def test_dense_and_block_solver_preserve_disabled_equality_bounds():
  shader = (Path(__file__).parents[1] / "mujoco_metal" / "shaders"
            / "coupled_constraints.metal").read_text()
  dense = shader.split("kernel void solve_coupled_constraints(", 1)[1]
  dense = dense.split("kernel void ", 1)[0]
  block = shader.split("kernel void solve_coupled_constraints_block(", 1)[1]
  block = block.split("kernel void ", 1)[0]
  lo = "dbg[nr * nr + 4 * nr + r]"
  hi = "dbg[nr * nr + 5 * nr + r]"
  assert f"lo[r] = {lo};" in dense and f"hi[r] = {hi};" in dense
  assert f"lo[r] = {lo};" in block and f"hi[r] = {hi};" in block
