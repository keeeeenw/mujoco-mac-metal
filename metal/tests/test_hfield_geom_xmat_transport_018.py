"""CPU source oracles for the FK-to-common-CCD matrix consumer delta.

The native full-contact gate remains the original HField lifecycle test. This
file checks the exact pose arguments and the pinned relative-frame operations
independently, without starting MPS during collection.
"""

from pathlib import Path

import mujoco
import numpy as np


_ROOT = Path(__file__).parents[1]
_HOST = (_ROOT / "mujoco_metal" / "coupled_constraints.py").read_text()
_SHADER = (_ROOT / "mujoco_metal" / "shaders" /
           "common_ccd_production.metal").read_text()
_XML = b"""
<mujoco model="hfield-xmat-source">
  <asset>
    <hfield name="terrain" size=".42 .37 .09 .02" nrow="4" ncol="5">
      .01 .03 .02 .04 .01
      .04 .02 .05 .01 .03
      .02 .05 .01 .04 .02
      .03 .01 .04 .02 .05
    </hfield>
  </asset>
  <worldbody>
    <geom name="terrain" type="hfield" hfield="terrain"
          pos=".1234567890123 -.0712345678901 .0176543210987"
          quat=".9238795325112867 .3826834323650898 0 0"/>
    <body name="moving" pos="-.0812345678901 .1434567890123 .2056789012345"
          quat=".8660254037844386 .2886751345948129 .2886751345948129 .2886751345948129">
      <geom name="hull" type="box" pos=".0376543210987 -.0234567890123 .0187654321098"
            quat=".9659258262890683 0 .2588190451025207 0" size=".12 .09 .08"/>
    </body>
  </worldbody>
</mujoco>
"""


def _triple(values):
  source = np.asarray(values, dtype=np.float64)
  high = source.astype(np.float32)
  residual = source - high.astype(np.float64)
  middle = residual.astype(np.float32)
  tail = (residual - middle.astype(np.float64)).astype(np.float32)
  return high, middle, tail


def _reconstruct(words):
  high, middle, tail = words
  return (high.astype(np.float64) + middle.astype(np.float64)
          + tail.astype(np.float64))


def _source_tpose(matrix, vector):
  out = np.zeros(3, dtype=np.float64)
  mujoco.mju_mulMatTVec3(out, np.asarray(matrix, dtype=np.float64).reshape(9),
                         np.asarray(vector, dtype=np.float64).reshape(3))
  return out


def _source_tmat(left, right):
  out = np.zeros((3, 3), dtype=np.float64)
  mujoco.mju_mulMatTMat(out, np.asarray(left, dtype=np.float64).reshape(3, 3),
                        np.asarray(right, dtype=np.float64).reshape(3, 3))
  return out


def test_hfield_consumer_requires_and_passes_all_retained_fk_pose_words():
  for name in ("geom_pos_low", "geom_pos_tail", "geom_xmat",
               "geom_xmat_low", "geom_xmat_tail"):
    assert f'("{name}",' in _HOST
    assert f'poses["{name}"]' in _HOST
    assert f"device const float* {name} [[buffer(" in _SHADER
  for kernel_name in ("_common_ccd_hfield_mesh_kernel",
                      "_common_ccd_rigid_kernel"):
    call_start = _HOST.index(f"if self.{kernel_name} is not None:")
    call = _HOST[call_start:call_start + 1100]
    for name in ("geom_pos_low", "geom_pos_tail", "geom_xmat",
                 "geom_xmat_low", "geom_xmat_tail"):
      assert f'poses["{name}"]' in call
  assert "common_ccd_load_geom_matrix(geom_xmat,geom_xmat_low,geom_xmat_tail" in _SHADER
  assert "object.mat[i]=rotation[i]" in _SHADER
  assert "common_ccd_matrix_transpose_vector(\n      Rh,flex_dd3_source_sub(pm,ph))" in _SHADER
  assert "common_ccd_matrix_transpose_product(Rh,Rm,local_mat)" in _SHADER
  assert "common_ccd_matrix_vector(Rh,flex_dd3(contact.pos))" in _SHADER
  assert "quat_to_mat(qa)" not in _SHADER
  assert "quat_to_mat(qb)" not in _SHADER


def test_hfield_relative_transform_triples_match_pinned_source_operations():
  model = mujoco.MjModel.from_xml_string(_XML.decode())
  data = mujoco.MjData(model)
  mujoco.mj_forward(model, data)
  hfield = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "terrain")
  mesh = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "hull")
  source_pos = np.asarray(data.geom_xpos, dtype=np.float64)
  source_mat = np.asarray(data.geom_xmat, dtype=np.float64)

  pos_words = _triple(source_pos.reshape(-1))
  mat_words = _triple(source_mat.reshape(-1))
  represented_pos = _reconstruct(pos_words)
  represented_mat = _reconstruct(mat_words)
  np.testing.assert_allclose(
      represented_pos, source_pos.reshape(-1), rtol=0, atol=2e-16)
  np.testing.assert_allclose(
      represented_mat, source_mat.reshape(-1), rtol=0, atol=2e-16)

  hbase, mbase = 3 * hfield, 3 * mesh
  hmat, mmat = 9 * hfield, 9 * mesh
  delta = (represented_pos[mbase:mbase + 3]
           - represented_pos[hbase:hbase + 3])
  local_pos = _source_tpose(represented_mat[hmat:hmat + 9], delta)
  local_mat = _source_tmat(represented_mat[hmat:hmat + 9],
                           represented_mat[mmat:mmat + 9])

  source_pos_flat = source_pos.reshape(-1)
  source_mat_flat = source_mat.reshape(-1)
  source_delta = (source_pos_flat[mbase:mbase + 3]
                  - source_pos_flat[hbase:hbase + 3])
  expected_pos = _source_tpose(source_mat_flat[hmat:hmat + 9], source_delta)
  expected_mat = _source_tmat(source_mat_flat[hmat:hmat + 9],
                              source_mat_flat[mmat:mmat + 9])
  np.testing.assert_allclose(local_pos, expected_pos, rtol=0, atol=5e-16)
  np.testing.assert_allclose(local_mat, expected_mat, rtol=0, atol=5e-16)

  # This witness confirms the fixture exercises the residuals that the old
  # quaternion reconstruction discarded. It does not loosen the native
  # contact, position, normal, or trajectory bounds in the original gate.
  high_only_mat = mat_words[0].astype(np.float64)
  assert np.max(np.abs(source_mat_flat - high_only_mat)) > 0.0
  assert np.max(np.abs(local_pos - _source_tpose(
      high_only_mat[hmat:hmat + 9],
      pos_words[0][mbase:mbase + 3] - pos_words[0][hbase:hbase + 3]))) > 0.0
