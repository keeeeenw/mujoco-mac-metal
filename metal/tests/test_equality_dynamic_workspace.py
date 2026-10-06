"""Equality assembly uses exact model row spans and dynamic DOF scratch."""

import os
from pathlib import Path

import mujoco
import numpy as np
import pytest

from mujoco_metal.coupled_constraints import lower_coupled_constraints
from mujoco_metal.capacity import CapacityLimits

_DYNAMIC_LIMITS = CapacityLimits(max_nv=64, max_rows=256)


def _large_equality_model():
  body = ""
  for index in reversed(range(33)):
    child = body
    body = (
        f'<body name="link{index}" pos="0 0 .02">'
        f'<joint name="slide{index}" type="slide" axis="1 .1 .03"/>'
        '<geom type="sphere" size=".01" mass=".1" contype="0" conaffinity="0"/>'
        f'{child}</body>')
  repeated = "".join(
      '<connect body1="link0" body2="link32" anchor=".1 .2 .3"/>'
      for _ in range(33))
  xml = ("<mujoco><option gravity='0 0 0' solver='PGS' iterations='0' "
         "jacobian='dense'/><worldbody>" + body + "</worldbody>"
         "<equality>" + repeated + "</equality></mujoco>")
  return mujoco.MjModel.from_xml_string(xml)


def test_dynamic_equality_descriptor_has_no_32_dof_or_96_row_gate():
  model = _large_equality_model()
  descriptor = lower_coupled_constraints(model, limits=_DYNAMIC_LIMITS)
  assert model.nv == 33
  assert descriptor.n_eq_rows == 99
  np.testing.assert_array_equal(descriptor.eq_rownum, np.full(33, 3))
  np.testing.assert_array_equal(descriptor.eq_rowadr, np.arange(33) * 3)
  shader = (Path(__file__).parents[1] / "mujoco_metal" / "shaders"
            / "equality_assembly.metal").read_text()
  assert "nv > 32" not in shader and "nr > 96" not in shader
  assert "thread float Jp" not in shader
  assert "device const int* eq_rownum [[buffer(27)]]" in shader


@pytest.mark.gpu
@pytest.mark.skipif(os.getenv("MUJOCO_METAL_RUN_GPU") != "1",
                    reason="dynamic equality shader parity requires native opt-in")
def test_native_connect_assembly_exceeds_old_dof_and_row_limits():
  pytest.importorskip("torch")
  import torch
  from mujoco_metal import MetalSimulation

  if not torch.backends.mps.is_available():
    pytest.skip("native equality parity requires MPS")
  model = _large_equality_model()
  qpos = np.linspace(-.03, .04, model.nq, dtype=np.float32)
  qvel = np.linspace(.02, -.01, model.nv, dtype=np.float32)
  data = mujoco.MjData(model)
  data.qpos[:] = qpos
  data.qvel[:] = qvel
  mujoco.mj_forward(model, data)
  simulation = MetalSimulation(model, qpos=qpos[None], qvel=qvel[None],
                               profile="integrated_euler_v1",
                               limits=_DYNAMIC_LIMITS)
  _, status, _ = simulation._acceleration(
      torch.tensor(qpos[None], dtype=torch.float32, device="mps"),
      torch.tensor(qvel[None], dtype=torch.float32, device="mps"))
  np.testing.assert_array_equal(status.cpu().numpy(), 0)
  descriptor = simulation._coupled_constraints.descriptor
  assert model.nv == 33 and descriptor.n_eq_rows == 99
  rows = np.flatnonzero(
      (np.asarray(data.efc_type) == int(mujoco.mjtConstraint.mjCNSTR_EQUALITY))
      & (np.asarray(data.efc_id) < 33))
  expected = np.asarray(data.efc_J).reshape(data.nefc, model.nv)[rows]
  actual = (simulation._coupled_constraints._workspace["workspace_J"]
            .reshape(1, descriptor.nr, model.nv)[0, :99].cpu().numpy())
  np.testing.assert_allclose(actual, expected, atol=3e-5, rtol=2e-5)
