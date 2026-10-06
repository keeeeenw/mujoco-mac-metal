"""CPU-only source-rounding regression for pinned HField addPrismVert."""
from __future__ import annotations

from decimal import Decimal, localcontext
import importlib.util
from pathlib import Path

import mujoco
import numpy as np

_CANDIDATE = Path(__file__).resolve().parents[1]
_BRIDGE_PATH = _CANDIDATE / "mujoco_metal" / "common_ccd_bridge.py"
_SPEC = importlib.util.spec_from_file_location("candidate_common_ccd_bridge", _BRIDGE_PATH)
_BRIDGE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_BRIDGE)


def _fma53(a, b, c):
  with localcontext() as context:
    context.prec = 120
    return float(Decimal.from_float(float(a)) * Decimal.from_float(float(b))
                 + Decimal.from_float(float(c)))


def _model():
  elevation = " ".join(["0.5"] * 81)
  xml = f'''<mujoco>
    <asset><hfield name="h" nrow="9" ncol="9" size=".4 .4 .05 .02"
      elevation="{elevation}"/></asset>
    <worldbody><geom name="terrain" type="hfield" hfield="h"/></worldbody>
  </mujoco>'''
  return mujoco.MjModel.from_xml_string(xml)


def test_pinned_arm64_grid_expression_contracts_before_source_rounding():
  size = 0.4
  dx = (2.0 * size) / 8
  source = _fma53(dx, 5, -size)
  separately_rounded = float(dx * 5) - size
  assert source == 0.1
  assert separately_rounded == 0.09999999999999998
  assert source != separately_rounded


def test_production_prism_record_preserves_contracted_grid_coordinate():
  model = _model()
  geom = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "terrain")
  _, words, high = _BRIDGE.hfield_prism_support_record(
      model, geom, 2, 5, 1, 0.0, vertex_offset=0)
  residual = words[48:84].reshape(6, 3, 2)
  vertices = high.reshape(6, 3).astype(np.float64)
  vertices += residual[:, :, 0].astype(np.float64)
  vertices += residual[:, :, 1].astype(np.float64)
  expected = _fma53((2.0 * float(model.hfield_size[0, 0])) / 8, 5,
                    -float(model.hfield_size[0, 0]))
  assert vertices[5, 0] == expected
  assert vertices[5, 0] == 0.1


def test_production_shader_uses_one_source_fma_for_grid_vertices():
  shader = (_CANDIDATE / "mujoco_metal" / "shaders" /
            "common_ccd_production.metal").read_text()
  assert "dx,flex_dd(float(cc[k])),flex_dd_neg(size0)" in shader
  assert "dy,flex_dd(float(rr[k])),flex_dd_neg(size1)" in shader
  assert "flex_dd_source_mul(dx,flex_dd(float(cc[k])))" not in shader
  assert "flex_dd_source_mul(dy,flex_dd(float(rr[k])))" not in shader


def test_captured_k4_support_tie_uses_source_vertex_order():
  model = _model()
  geom = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "terrain")
  _, words, high = _BRIDGE.hfield_prism_support_record(
      model, geom, 2, 5, 1, 0.0, vertex_offset=0)
  residual = words[48:84].reshape(6, 3, 2)
  vertices = high.reshape(6, 3).astype(np.float64)
  vertices += residual[:, :, 0].astype(np.float64)
  vertices += residual[:, :, 1].astype(np.float64)
  # Exact HLT support direction recorded for e140 intersection iteration 4.
  direction = np.asarray([
      -0.707106781186547573, 0.707106781186547462,
      1.96261554898367647e-16], dtype=np.float64)
  assert vertices[3].tolist()[:2] == [0.0, -0.2]
  assert vertices[4].tolist()[:2] == [0.1, -0.1]
  assert _BRIDGE.hfield_source_support_index(vertices, direction) == 3
