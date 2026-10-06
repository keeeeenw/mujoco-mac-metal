# Copyright 2026 The MuJoCo Metal contributors
# Licensed under the Apache License, Version 2.0.
"""Large equality-stage witnesses beyond the former local-array limits."""

import os

import mujoco
import numpy as np
import pytest

from mujoco_metal.capacity import CapacityLimits
from mujoco_metal.capacity import solver_debug_layout
from mujoco_metal.coupled_constraints import lower_coupled_constraints


def _xml(kind):
  if kind == "joint":
    bodies = "".join(
        f'<body name="b{i}" pos="{2 * i} 0 0">'
        f'<joint name="j{i}" type="slide" axis="1 0 0"/>'
        '<inertial pos="0 0 0" mass="1" '
        'diaginertia=".001 .001 .001"/></body>'
        for i in range(40))
    equality = ('<joint joint1="j0" joint2="j1" '
                'polycoef="0 1 0 0 0"/>')
  else:
    bodies = (
        '<body name="b0" pos="0 0 0"><freejoint name="f0"/>'
        '<inertial pos="0 0 0" mass="1" '
        'diaginertia=".01 .01 .01"/></body>'
        '<body name="b1" pos="1 0 0"><freejoint name="f1"/>'
        '<inertial pos="0 0 0" mass="1" '
        'diaginertia=".01 .01 .01"/></body>'
        + "".join(
            f'<body name="b{i + 2}" pos="{2 * (i + 2)} 0 0">'
            f'<joint name="j{i}" type="slide" axis="1 0 0"/>'
            '<inertial pos="0 0 0" mass="1" '
            'diaginertia=".001 .001 .001"/></body>'
            for i in range(28)))
    if kind == "connect":
      equality = '<connect body1="b0" body2="b1" anchor="0.5 0.2 0"/>'
    elif kind == "weld":
      equality = '<weld body1="b0" body2="b1"/>'
    else:
      raise AssertionError(kind)
  return (
      '<mujoco><option gravity="0 0 0" solver="PGS" iterations="8" '
      'tolerance="1e-8"/><worldbody>' + bodies + '</worldbody><equality>'
      + equality + '</equality></mujoco>')


@pytest.mark.parametrize("kind,expected_rows", [
    ("joint", 1), ("connect", 3), ("weld", 6),
])
def test_large_equality_lowering_and_pinned_cpu_oracle(kind, expected_rows):
  """Lowering retains exact row spans above 32 DOFs and 96 solver rows."""
  model = mujoco.MjModel.from_xml_string(_xml(kind))
  limits = CapacityLimits(max_nv=40, max_rows=256, max_batch=2)
  descriptor = lower_coupled_constraints(model, limits=limits)
  assert descriptor.nv == 40
  assert descriptor.nr > 96
  assert descriptor.n_eq_rows == expected_rows
  np.testing.assert_array_equal(descriptor.eq_rowadr, [0])
  np.testing.assert_array_equal(descriptor.eq_rownum, [expected_rows])

  # The dynamic point-J scratch lives in the solver's already-accounted tail;
  # it must fit before the distinct retained-qacc vector even for nr > 96.
  layout = solver_debug_layout(
      descriptor.nv, descriptor.nr, descriptor.solver_type,
      nbody=descriptor.nbody)
  eq_scratch = layout["equality_jacobian_offset"]
  assert eq_scratch >= layout["debug_prefix"]
  assert eq_scratch + 12 * descriptor.nv == layout["qacc_warmstart_offset"]

  data = mujoco.MjData(model)
  data.qvel[:] = 0.0
  mujoco.mj_forward(model, data)
  equalities = data.efc_type == int(mujoco.mjtConstraint.mjCNSTR_EQUALITY)
  assert int(np.count_nonzero(equalities)) == expected_rows
  jacobian = np.asarray(data.efc_J).reshape(data.nefc, model.nv)[equalities]
  assert np.linalg.norm(jacobian) > 0.1
  assert np.all(np.asarray(data.efc_R)[equalities] > 0.0)


@pytest.mark.gpu
@pytest.mark.skipif(os.getenv("MUJOCO_METAL_RUN_GPU") != "1",
                    reason="opt-in native MPS equality-stage gate")
@pytest.mark.parametrize("kind,expected_rows", [
    ("joint", 1), ("connect", 3), ("weld", 6),
])
def test_large_equality_assembly_matches_pinned(kind, expected_rows):
  """The real Metal equality kernel writes complete large-model J/R/ar rows."""
  torch = pytest.importorskip("torch")
  from mujoco_metal.coupled_constraints import MetalCoupledConstraints
  from mujoco_metal.model import load_model
  from mujoco_metal.smooth_metal import MetalSmoothDynamics

  model = mujoco.MjModel.from_xml_string(_xml(kind))
  limits = CapacityLimits(max_nv=40, max_rows=256, max_batch=1)
  descriptor = load_model(model)
  qpos = np.asarray(model.qpos0, dtype=np.float32)[None, :].copy()
  qvel = np.zeros((1, model.nv), dtype=np.float32)
  qpos_device = torch.as_tensor(qpos, dtype=torch.float32, device="mps")
  qvel_device = torch.as_tensor(qvel, dtype=torch.float32, device="mps")
  smooth = MetalSmoothDynamics(descriptor, 1).run_device(qpos_device,
                                                         qvel_device)
  poses = dict(smooth["poses"])
  poses["root_com"] = smooth["root_com"]
  solver = MetalCoupledConstraints(model, 1, limits=limits)
  # Keep this prerequisite test on the long-standing public solver entry
  # point. `assemble_device` is a newer split-stage API and belongs in a
  # separate packet; this test should qualify the equality shader change
  # against the committed run_device ABI.
  mass = torch.eye(model.nv, dtype=torch.float32, device="mps").unsqueeze(0)
  qfrc = torch.zeros((1, model.nv), dtype=torch.float32, device="mps")
  actual = solver.run_device(
      poses, mass, qfrc, qpos_device, qvel_device,
      cvel=smooth["cvel"], cdof=smooth["cdof"],
      cdof_dot=smooth["cdof_dot"])

  data = mujoco.MjData(model)
  data.qpos[:] = qpos[0]
  data.qvel[:] = qvel[0]
  mujoco.mj_forward(model, data)
  equality_mask = data.efc_type == int(mujoco.mjtConstraint.mjCNSTR_EQUALITY)
  rows = np.flatnonzero(equality_mask)
  start = int(solver.descriptor.eq_rowadr[0])
  stop = start + expected_rows
  np.testing.assert_allclose(
      actual["J"][0, start:stop].detach().cpu().numpy(),
      np.asarray(data.efc_J).reshape(data.nefc, model.nv)[rows],
      rtol=3e-5, atol=3e-5)
  np.testing.assert_allclose(
      actual["R"][0, start:stop].detach().cpu().numpy(),
      np.asarray(data.efc_R)[rows], rtol=3e-5, atol=3e-6)
  np.testing.assert_allclose(
      actual["ar"][0, start:stop].detach().cpu().numpy(),
      np.asarray(data.efc_aref)[rows], rtol=2e-4, atol=2e-4)
