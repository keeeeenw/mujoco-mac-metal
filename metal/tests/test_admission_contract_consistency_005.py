# Copyright 2026 The MuJoCo Metal contributors
# Licensed under the Apache License, Version 2.0.
"""Actual CPU profile admission must agree with its advertised feature scope.

These are admission/metadata checks, never evidence of native physics parity.
The independent native force/trajectory tests remain required for qualification.
"""

import mujoco
import numpy as np
import pytest

from mujoco_metal.capacity import CapacityLimits
from mujoco_metal.coupled_constraints import lower_coupled_constraints
from mujoco_metal.registry import REQUIREMENTS, Qualification
from mujoco_metal.stepping import validate_stepping_profile


def _spatial_model(integrator="Euler", user=False):
  actuator = ('<actuator><general joint="j" dyntype="user" '
              'gaintype="user" biastype="user"/></actuator>') if user else ""
  return mujoco.MjModel.from_xml_string(f"""
    <mujoco><option integrator="{integrator}" gravity="0 0 0"/>
      <worldbody><site name="a" pos="0 0 1"/>
        <body pos=".2 0 1"><joint name="j" type="slide" axis="1 0 0"/>
          <geom type="sphere" size=".05" mass="1"/>
          <site name="b"/></body></worldbody>
      <tendon><spatial name="path" limited="true" range=".1 .5"
                       frictionloss=".01">
        <site site="a"/><site site="b"/></spatial></tendon>
      <equality><tendon tendon1="path"/></equality>{actuator}
    </mujoco>""")


@pytest.mark.parametrize(("integrator", "profile"), [
    ("Euler", "integrated_euler_v1"),
    ("implicit", "integrated_implicit_v1"),
    ("implicitfast", "integrated_implicitfast_v1"),
])
@pytest.mark.parametrize("user", [False, True])
def test_admitted_spatial_tendon_and_user_features_are_not_advertised_rejected(
    integrator, profile, user):
  model = _spatial_model(integrator, user)
  descriptor = lower_coupled_constraints(model)
  assert int(model.eq_type[0]) == int(mujoco.mjtEq.mjEQ_TENDON)
  assert int(descriptor.eq_rownum[0]) > 0
  admitted = validate_stepping_profile(model, profile=profile)
  assert any("spatial tendons" in item for item in admitted.supported)
  assert not any("spatial/wrapping tendons" in item for item in admitted.rejected)
  if user:
    assert any("USER actuator source defaults" in item
               for item in admitted.supported)
    assert not any("user-callback actuators" in item
                   for item in admitted.rejected)
  assert not any("warmstart disable flag" == item
                 for item in admitted.irrelevant)


@pytest.mark.parametrize("kind", ["flexvert", "flexstrain"])
def test_flex_vertex_and_strain_admission_retains_compiled_rows(kind):
  if kind == "flexvert":
    world = ('<flexcomp name="cloth" type="grid" count="2 2 1" '
             'spacing=".1 .1 .1" mass="1" radius=".01" dim="2">'
             '<edge equality="strain"/>'
             '<contact contype="0" conaffinity="0"/></flexcomp>')
    equality = '<equality><flexvert flex="cloth"/></equality>'
    eqtype = int(mujoco.mjtEq.mjEQ_FLEXVERT)
  else:
    world = ('<flexcomp name="volume" type="grid" count="3 2 2" '
             'cellcount="2 1 1" spacing=".1 .1 .1" mass="1" dim="3" '
             'dof="trilinear"><edge equality="strain"/>'
             '<contact contype="0" conaffinity="0" selfcollide="none"/>'
             '</flexcomp>')
    equality = ""
    eqtype = int(mujoco.mjtEq.mjEQ_FLEXSTRAIN)
  model = mujoco.MjModel.from_xml_string(
      '<mujoco><option gravity="0 0 0" jacobian="dense"/>'
      f'<worldbody>{world}</worldbody>{equality}</mujoco>')
  limits = CapacityLimits(max_nv=max(int(model.nv), 32), max_rows=256)
  descriptor = lower_coupled_constraints(model, limits=limits)
  data = mujoco.MjData(model)
  data.qpos[:] += np.linspace(-.003, .004, model.nq)
  data.qvel[:] = np.linspace(-.09, .07, model.nv)
  mujoco.mj_forward(model, data)
  eqids = np.flatnonzero(np.asarray(model.eq_type) == eqtype)
  assert eqids.size
  for eid in eqids:
    cpu_rows = np.flatnonzero(
        (data.efc_type == int(mujoco.mjtConstraint.mjCNSTR_EQUALITY))
        & (data.efc_id == eid))
    assert cpu_rows.size > 0
    assert int(descriptor.eq_rownum[eid]) == cpu_rows.size
  # The compiler may also retain a zero-mode auto-strain equality. Its zero
  # row count is valid and must not be confused with rejection of flexvert.
  profile = "integrated_scalable_v1" if model.nv > 32 else "integrated_euler_v1"
  validate_stepping_profile(model, profile=profile, limits=limits)
  row = next(req for req in REQUIREMENTS if req.id == "REQ-EQ-005")
  assert row.qualification == Qualification.UNQUALIFIED


def test_capacity_default_values_are_the_documented_development_budgets():
  from pathlib import Path
  limits = CapacityLimits()
  text = (Path(__file__).parents[1] / "API.md").read_text()
  for name in ("max_nv", "max_pairs", "max_slots", "max_rows"):
    assert f"{name}={getattr(limits, name)}" in text
  assert limits.memory_budget_bytes == 1 << 30


def test_integrated_implicit_claims_keep_qualification_separate_from_admission():
  rows = {req.id: req for req in REQUIREMENTS}
  assert "integrated_implicit_v1" in rows["REQ-INT-002"].admission
  assert rows["REQ-INT-003"].qualification == Qualification.UNQUALIFIED
  assert "test_sparse_implicit_public_019.py" in rows["REQ-INT-003"].tests
