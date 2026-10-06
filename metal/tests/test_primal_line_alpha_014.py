"""Source-precision and opt-in MPS witness for primal paired line alpha."""
from pathlib import Path
import os

import numpy as np
import pytest


def _source_alpha_case():
  # All operands are the exact float32 values consumed by the shader. The
  # quotient lies between float32 values; multiplying by a large direction
  # exposes losing alpha.lo in the updated point.
  values = np.asarray([
      -0.1, 1.0e-9, 0.3, 1.0e-8,
      2.0e6, 0.0, 0.0, 0.0,
  ], dtype=np.float32)
  exact_alpha = -(float(values[0]) + float(values[1])) / (
      float(values[2]) + float(values[3]))
  exact_updated = exact_alpha * (float(values[4]) + float(values[5])) + float(values[6]) + float(values[7])
  collapsed_alpha = np.float32(exact_alpha)
  collapsed_updated = float(collapsed_alpha) * float(values[4])
  assert abs(exact_updated - collapsed_updated) > 4.0e-3
  return values, exact_alpha, exact_updated, collapsed_updated


def test_line_alpha_fixture_distinguishes_paired_and_float_transport():
  _, exact_alpha, exact_updated, collapsed_updated = _source_alpha_case()
  assert exact_alpha != float(np.float32(exact_alpha))
  assert abs(exact_updated - collapsed_updated) > 4.0e-3


def test_native_line_alpha_witness_retains_source_precision():
  if os.environ.get("MUJOCO_METAL_RUN_GPU") != "1":
    pytest.skip("native MPS execution is opt-in via MUJOCO_METAL_RUN_GPU=1")
  torch = pytest.importorskip("torch")
  if not torch.backends.mps.is_available() or not hasattr(torch.mps, "compile_shader"):
    pytest.skip("requires native MPS shader execution")

  values, exact_alpha, exact_updated, collapsed_updated = _source_alpha_case()
  shader_root = Path(__file__).parents[1] / "mujoco_metal" / "shaders"
  from mujoco_metal.coupled_constraints import _coupled_shader_source
  source = _coupled_shader_source(expected_root=shader_root)
  library = torch.mps.compile_shader(source)
  device_input = torch.tensor(values, dtype=torch.float32, device="mps")
  output = torch.zeros((7,), dtype=torch.float32, device="mps")
  library.primal_line_alpha_witness(
      device_input, output, threads=(1,), group_size=(1,))
  actual = output.cpu().numpy().astype(np.float64)

  paired_alpha = actual[0] + actual[1]
  paired_updated = actual[3] + actual[4]
  collapsed_updated_native = actual[5] + actual[6]
  print("PRIMAL_LINE_ALPHA native=", actual.tolist(),
        "source=", [exact_alpha, exact_updated, collapsed_updated], flush=True)
  # Two float words should reproduce this represented quotient close to
  # binary64 source arithmetic; the update's looser bound allows its 2e6 scale.
  assert abs(paired_alpha - exact_alpha) <= 2.0e-15
  assert abs(paired_updated - exact_updated) <= 2.0e-9
  assert abs(collapsed_updated_native - collapsed_updated) <= 2.0e-9
  assert abs(paired_updated - collapsed_updated_native) > 4.0e-3
