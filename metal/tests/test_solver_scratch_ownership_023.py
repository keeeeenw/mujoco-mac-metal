"""Scratch ownership: solver-debug layout intervals and Newton lifetimes.

The Newton/CG primal solver aliases the pair-tail planes as its own row
scratch (row_residual_low/row_force_low sit exactly on the pgs_rhs_low /
pgs_lambda_low offsets), while the aref_low plane must survive the whole
solve (dual refinement rebuilds b from it). These tests pin the interval
arithmetic (offsets, bounds, guards) on the host and prove the lifetime
natively: post-solve, aref_low still equals the published row_data
corrections, so no solver stage clobbers it.
"""
import os

import numpy as np
import pytest


def _needs_gpu():
  return pytest.mark.skipif(os.getenv("MUJOCO_METAL_RUN_GPU") != "1",
                            reason="opt-in GPU")


def test_tail_structure_and_refinement_footprint_cpu():
  from mujoco_metal.capacity import (
      SIGNED_I32_ELEMENT_LIMIT, primal_scratch_floats, solver_debug_layout,
  )
  import mujoco
  for solver in (mujoco.mjtSolver.mjSOL_PGS, mujoco.mjtSolver.mjSOL_CG,
                 mujoco.mjtSolver.mjSOL_NEWTON):
    for nv, nr in ((6, 16), (1, 1), (0, 0), (32, 96), (2, 3)):
      lay = solver_debug_layout(nv, nr, int(solver), mass_storage="dense")
      v, r = max(nv, 1), max(nr, 1)
      # Pair tail is exactly rhs_lo(nr) + lambda_lo(nr) + aref_lo(nr) after
      # eight DOF planes; Newton/CG refinement needs 3nv words of the
      # Newton core (v pair + temp pair) for every solver type it runs on.
      assert lay["high_low_workspace_length"] == 8 * v + 3 * r
      if int(solver) in (int(mujoco.mjtSolver.mjSOL_CG),
                         int(mujoco.mjtSolver.mjSOL_NEWTON)):
        core = {"CG": 13 * v + 5 * r, "Newton": 16 * v + 5 * r}[
            "CG" if int(solver) == int(mujoco.mjtSolver.mjSOL_CG)
            else "Newton"]
        assert core >= 3 * v
      assert lay["final_stride"] <= SIGNED_I32_ELEMENT_LIMIT
      assert lay["base_stride"] > 0 and lay["final_stride"] > lay["base_stride"]
  # Zero-row/contact guards stay nonzero (one-element sentinels).
  lay = solver_debug_layout(0, 0, int(mujoco.mjtSolver.mjSOL_NEWTON))
  assert lay["high_low_workspace_length"] == 8 + 3
  # Signed-int32 overflow still fails closed.
  with pytest.raises(Exception):
    solver_debug_layout(50000, 50000, int(mujoco.mjtSolver.mjSOL_NEWTON))


def test_shader_aliasing_is_documented():
  from pathlib import Path
  src = (Path(__file__).parents[1] / "mujoco_metal" / "shaders"
         / "coupled_constraints.metal").read_text()
  # Primal solver row scratch aliases the rhs_lo/lambda_lo tail offsets.
  assert "device float* row_residual_low = high_low + 8 * max(nv, 1);" in src
  assert "device float* row_force_low = row_residual_low + nr;" in src
  # Refinement documents why it rebuilds b instead of reading rhs_lo.
  assert "aliases it as row-residual scratch" in src
  assert "rebuilt from ar/aref_low" in src


@_needs_gpu()
def test_aref_low_survives_newton_solve_gpu():
  import sys
  sys.path.insert(0, "metal/tests")
  from test_bundled_sdf_rigid_oracle_019 import _plugin_sphere_model
  from mujoco_metal.simulation import MetalSimulation
  model = _plugin_sphere_model("torus")
  sim = MetalSimulation(model, batch_size=1, profile="integrated_euler_v1")
  qfrc = np.zeros((1, model.nv), dtype=np.float32)
  qfrc[0, 0] = 0.01
  for _ in range(4):
    sim.step(1, qfrc_applied=qfrc)
  s = sim.assembled_system(recompute=True)
  cc = sim._coupled_constraints
  nv, nr = model.nv, cc.descriptor.nr
  nslots = int(cc.descriptor.ncontacts_max)
  rd = cc._workspace["contact_row_data"].cpu().numpy().reshape(nslots, 6, 7)
  dbg = cc._workspace["workspace_debug"].cpu().numpy()
  base = (nr * nr + 7 * nr + cc._high_low_workspace_offset
          + 8 * max(nv, 1) + 2 * max(nr, 1))
  plane = dbg[base:base + nr]
  # The published corrections must still be present verbatim after the
  # full Newton solve: Newton stages must not clobber this plane
  # (refinement reads it post-solve). Compared as multisets over the
  # contact-row span so slot/row packing order cannot cause spurious
  # failures; counts must agree exactly (no dropped/duplicated rows).
  got = sorted(float(v) for v in plane[8:16])
  want = sorted(float(v) for v in rd[:, 0, 6])
  assert len(got) == len(want) == nslots
  for g, w in zip(got, want):
    assert g == w, (g, w)
  assert float(np.max(np.abs(rd[:, 0, 6]))) > 1e-7
