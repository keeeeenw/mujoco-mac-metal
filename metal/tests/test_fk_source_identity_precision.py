"""Actual pinned compiled-pose witness for source-visible near-identity residuals."""
import os
import mujoco
import numpy as np
import pytest

_XML = '<mujoco><worldbody><body name="b" quat="1.0000000000000002 0 0 0"><geom name="g" type="sphere" size=".1"/></body></worldbody></mujoco>'


def _source():
  model = mujoco.MjModel.from_xml_string(_XML)
  data = mujoco.MjData(model)
  mujoco.mj_forward(model, data)
  return model, data


def test_compiled_source_keeps_near_identity_residual():
  model, data = _source()
  assert model.body_quat[1, 0].hex() == '0x1.0000000000001p+0'
  assert data.xquat[1, 0].hex() == '0x1.0000000000001p+0'
  assert int(model.geom_sameframe[0]) == 1
  expected = np.eye(3).reshape(1, 9) * (1.0 + 2.0**-51)
  np.testing.assert_array_equal(data.geom_xmat, expected)
  assert not np.array_equal(data.geom_xmat.astype(np.float32).astype(np.float64), expected)


@pytest.mark.gpu
@pytest.mark.skipif(os.getenv('MUJOCO_METAL_RUN_GPU') != '1', reason='opt-in native Metal')
def test_native_fk_preserves_compiled_near_identity_matrix():
  import torch
  from mujoco_metal.model import load_model
  from mujoco_metal.metal_kinematics import MetalKinematics
  model, data = _source()
  fk = MetalKinematics(load_model(model), batch_size=1)
  poses = fk.run_device(torch.zeros((1, model.nq), dtype=torch.float32, device='mps'))
  def represented(name):
    return sum(poses[name + suffix].detach().cpu().numpy().astype(np.float64)
               for suffix in ('', '_low', '_tail'))
  np.testing.assert_array_equal(represented('body_quat')[0], data.xquat)
  np.testing.assert_array_equal(represented('geom_xmat')[0], data.geom_xmat)
