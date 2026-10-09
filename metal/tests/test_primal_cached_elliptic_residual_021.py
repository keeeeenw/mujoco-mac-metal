"""Native regression for the cached elliptic-residual double count.

Executes production ``primal_accel_cost_delta_elliptic`` on device with
nonzero effective-Jacobian residuals and acceleration/direction low words,
comparing cached (base pair) vs uncached (rebuilt from J) cost delta, first
and second derivatives at positive step sizes, across cone zones and
elliptic dims 2/3. Zero-low controls must match on any source. Fails on
pre-fix source (cached path adds Jlo*acc a second time); passes post-fix.
"""
import os
from pathlib import Path

import numpy as np
import pytest


def _compile():
  torch = pytest.importorskip("torch")
  if not torch.backends.mps.is_available() or not hasattr(
      torch.mps, "compile_shader"):
    pytest.skip("requires native MPS shader execution")
  root = Path(__file__).resolve().parents[1] / "mujoco_metal" / "shaders"
  override = os.environ.get("MUJOCO_METAL_SHADER_ROOT")
  if override and Path(override).resolve() != root:
    raise AssertionError("native witness must use its own shader sources")
  from mujoco_metal.coupled_constraints import _coupled_shader_source
  return torch, torch.mps.compile_shader(
      _coupled_shader_source(expected_root=root))


def _paired_record(jhi, jlo):
  from mujoco_metal.constraint_jacobian import (
      PackedJacobianLayout, initialize_packed_jacobian_records)
  jhi = np.asarray(jhi, dtype=np.float32)
  nr, nv = jhi.shape
  layout = PackedJacobianLayout.create(nr, nv, mode=0, paired=True)
  rec = np.zeros(layout.stride_words, dtype=np.float32)
  initialize_packed_jacobian_records(rec.view(np.int32), 1, layout)
  rec[layout.values_offset:layout.values_offset + layout.nnz] = jhi.reshape(-1)
  rec[layout.lo_values_offset():layout.lo_values_offset() + layout.nnz] = (
      np.asarray(jlo, dtype=np.float32).reshape(-1))
  return rec


def _run_case(lib, torch, dim, zone, alpha, zero_low):
  nv, nr = 2, 4
  Jhi = np.zeros((nr, nv), dtype=np.float32)
  Jlo = np.zeros((nr, nv), dtype=np.float32)
  # Elliptic rows 1..dim: exact-binary hi, visible lo (2^-8).
  base_hi = np.array([[1.0, 0.5], [0.5, 1.0], [1.0, 1.0]], dtype=np.float32)
  for k in range(dim):
    Jhi[1 + k] = base_hi[k]
    if not zero_low:
      Jlo[1 + k, k % nv] = np.float32(2.0 ** -8)
  acc = np.array([1.0, 1.0], dtype=np.float32)
  acc_low = (np.zeros((nv,), dtype=np.float32) if zero_low else
             np.array([2.0 ** -12, -(2.0 ** -12)], dtype=np.float32))
  direction = np.array([0.5, -0.25], dtype=np.float32)
  direction_low = (np.zeros((nv,), dtype=np.float32) if zero_low else
                   np.array([2.0 ** -20, -(2.0 ** -20)], dtype=np.float32))
  # Target residuals select the cone zone for the elliptic block.
  if zone == "middle":
    r_target = np.array([0.04, -0.5, 0.3], dtype=np.float64)[:dim]
  elif zone == "bottom":
    r_target = np.array([-0.8, 0.1, -0.05], dtype=np.float64)[:dim]
  else:
    raise ValueError(zone)
  # aref (high only; low backing is zero) completes the uncached residual:
  # aref = J_eff @ acc_eff - r_target, rounded once to float32.
  Jeff = Jhi[1:1 + dim].astype(np.float64) + Jlo[1:1 + dim].astype(np.float64)
  aeff = acc.astype(np.float64) + acc_low.astype(np.float64)
  aref = np.zeros((nr,), dtype=np.float32)
  aref[1:1 + dim] = (Jeff @ aeff - r_target).astype(np.float32)
  base = np.zeros((nr,), dtype=np.float32)
  base_low = np.zeros((nr,), dtype=np.float32)
  base[1:1 + dim] = r_target.astype(np.float32)
  mass = np.array([2.0, 0.0, 0.0, 2.0], dtype=np.float32)
  R = np.array([1.0, 0.3, 0.5, 0.7], dtype=np.float32)
  lo = np.zeros((nr,), dtype=np.float32)
  hi = np.zeros((nr,), dtype=np.float32)
  q0 = np.zeros((nv,), dtype=np.float32)
  enabled = np.array([0, 1, 1, 1], dtype=np.int32)
  enabled[1 + dim:] = 0
  member = np.array([0, 1, 1, 1], dtype=np.int32)
  member[1 + dim:] = 0
  estart = np.array([1], dtype=np.int32)
  edim = np.array([dim], dtype=np.int32)
  efric = np.array([0.8, 0.6, 0.3, 0.4, 0.5], dtype=np.float32)
  dims = np.zeros(128, dtype=np.int32)
  dims[0], dims[1], dims[2], dims[3], dims[4] = nv, nr, nr, 0, 1
  debug = np.zeros(128, dtype=np.float32)
  mgrad = np.zeros((nv,), dtype=np.float32)
  proj = np.zeros((nv,), dtype=np.float32)
  alpha_pair = np.array([alpha, 0.0], dtype=np.float32)
  f32 = lambda a: torch.tensor(np.asarray(a).copy(), dtype=torch.float32,
                               device="mps")
  i32 = lambda a: torch.tensor(np.asarray(a).copy(), dtype=torch.int32,
                               device="mps")
  args = [f32(mass), f32(_paired_record(Jhi, Jlo)), f32(R), f32(aref), f32(lo),
          f32(hi), f32(q0), f32(acc), f32(acc_low), f32(direction),
          f32(direction_low), i32(enabled), i32(member), i32(estart),
          i32(edim), f32(efric), f32(base), f32(base_low), i32(dims),
          f32(debug), f32(mgrad), f32(proj)]
  out = torch.zeros((6,), dtype=torch.float32, device="mps")
  got = []
  for flag in (0, 1):
    lib.primal_cached_elliptic_residual_probe(
        *args, out, f32(alpha_pair), i32(np.array([flag], dtype=np.int32)),
        threads=(1,), group_size=(1,))
    torch.mps.synchronize()
    got.append(out.cpu().numpy().copy())
  return got[0], got[1]


@pytest.mark.skipif(os.environ.get("MUJOCO_METAL_RUN_GPU") != "1",
                    reason="native cached-residual witness is opt-in")
@pytest.mark.parametrize("dim", [2, 3])
@pytest.mark.parametrize("zone,alpha", [("middle", 0.375), ("bottom", 0.9)])
@pytest.mark.gpu
@pytest.mark.skipif(os.getenv("MUJOCO_METAL_RUN_GPU") != "1", reason="native opt-in")
def test_native_cached_elliptic_residual_matches_uncached_gpu(
    dim, zone, alpha):
  torch, lib = _compile()
  uncached, cached = _run_case(lib, torch, dim, zone, alpha, False)
  diff = np.max(np.abs((cached - uncached).astype(np.float64)))
  print(f"CACHED_ELLIPTIC dim={dim} zone={zone} alpha={alpha} "
        f"maxabs={diff:.3e} uncached={uncached.tolist()} "
        f"cached={cached.tolist()}", flush=True)
  # Pre-fix double count is Jlo*acc ~ 2^-8 = 3.9e-3 in the residual,
  # visible in cost/derivatives far above pair rounding (~1e-7 here
  # from the single aref float32 rounding). Post-fix must be tight.
  assert diff < 5e-7, f"cached/uncached diverge: {diff:.3e}"
  # Zero-low control: identical setup with exact-zero low planes must
  # also match (holds pre- and post-fix; guards against a test that
  # only passes because both paths are broken equally).
  uncached0, cached0 = _run_case(lib, torch, dim, zone, alpha, True)
  diff0 = np.max(np.abs((cached0 - uncached0).astype(np.float64)))
  assert diff0 < 5e-7, f"zero-low control diverges: {diff0:.3e}"
