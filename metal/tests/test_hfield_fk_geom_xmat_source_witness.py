"""Strict FK matrix witness for the compiled HField/mesh source fixture.

The MPS case is opt-in. Its CPU reference promotes the exact public float32
qpos words into pinned MuJoCo's double state ABI before running mj_forward.
"""

import os
from pathlib import Path

import mujoco
import numpy as np
import pytest

from mujoco_metal.metal_kinematics import MetalKinematics
from mujoco_metal.model import load_model


_XML = Path(__file__).parent / "fixtures" / "hfield_mesh_source.xml"


def _source_case():
  model = mujoco.MjModel.from_xml_path(str(_XML))
  qpos_public = np.asarray(model.qpos0, dtype=np.float32).copy()
  data = mujoco.MjData(model)
  data.qpos[:] = qpos_public.astype(np.float64)
  mujoco.mj_forward(model, data)
  mesh_geom = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "mesh_geom")
  return model, qpos_public, data, mesh_geom


def test_fixture_reproduces_compiled_mesh_matrix_residual_witness():
  model, qpos_public, data, mesh_geom = _source_case()
  quat = np.asarray(model.geom_quat[mesh_geom], dtype=np.float64)
  matrix = np.asarray(data.geom_xmat[mesh_geom], dtype=np.float64)
  assert np.array_equal(qpos_public, model.qpos0.astype(np.float32))
  assert quat[0] == quat[1] == -quat[2] == quat[3]
  assert abs(quat[0] - 0.5) > 0.0
  # Pinned quat2Mat's compiled result contains the residual that rounds away
  # when this quaternion is uploaded as a single float32 word.
  assert matrix[1] == -1.00000000000000266
  high_only_quat = quat.astype(np.float32).astype(np.float64)
  high_only_matrix = np.asarray([
      high_only_quat[0]**2 + high_only_quat[1]**2
      - high_only_quat[2]**2 - high_only_quat[3]**2,
      2*(high_only_quat[1]*high_only_quat[2]
         - high_only_quat[0]*high_only_quat[3]),
      2*(high_only_quat[1]*high_only_quat[3]
         + high_only_quat[0]*high_only_quat[2]),
      2*(high_only_quat[1]*high_only_quat[2]
         + high_only_quat[0]*high_only_quat[3]),
      high_only_quat[0]**2 - high_only_quat[1]**2
      + high_only_quat[2]**2 - high_only_quat[3]**2,
      2*(high_only_quat[2]*high_only_quat[3]
         - high_only_quat[0]*high_only_quat[1]),
      2*(high_only_quat[1]*high_only_quat[3]
         - high_only_quat[0]*high_only_quat[2]),
      2*(high_only_quat[2]*high_only_quat[3]
         + high_only_quat[0]*high_only_quat[1]),
      high_only_quat[0]**2 - high_only_quat[1]**2
      - high_only_quat[2]**2 + high_only_quat[3]**2,
  ])
  assert high_only_matrix[1] == -1.0
  assert matrix[1] != high_only_matrix[1]


@pytest.mark.skipif(os.environ.get("MUJOCO_METAL_RUN_GPU") != "1",
                    reason="set MUJOCO_METAL_RUN_GPU=1 for native Metal")
def test_native_fk_geom_xmat_retains_compiled_mesh_source_residual():
  import torch
  if not torch.backends.mps.is_available():
    pytest.skip("requires native Metal")

  model, qpos_public, data, mesh_geom = _source_case()
  stage = MetalKinematics(load_model(model), batch_size=1)
  observed = stage.run_device(
      torch.as_tensor(qpos_public[None, :], dtype=torch.float32, device="mps"))
  represented = sum(
      observed[f"geom_xmat{suffix}"].detach().cpu().numpy().astype(np.float64)
      for suffix in ("", "_low", "_tail"))[0, mesh_geom]
  high = observed["geom_xmat"].detach().cpu().numpy()[0, mesh_geom]
  target = np.asarray(data.geom_xmat[mesh_geom], dtype=np.float64)

  # Check the entries carrying the matrix's +/-1 source witness in ULPs;
  # broad FK tolerances elsewhere do not establish this residual transport.
  selected = np.flatnonzero(np.abs(target) >= 0.5)
  ulp = np.abs(np.spacing(target[selected]))
  assert np.all(ulp > 0)
  error_ulps = np.abs(represented[selected] - target[selected]) / ulp
  assert float(np.max(error_ulps, initial=0.0)) <= 1.0
  assert np.any(represented[selected] != high[selected])
  assert np.max(np.abs(represented - target), initial=0.0) <= 1.0e-15
