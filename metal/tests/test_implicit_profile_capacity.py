# Copyright 2026 The MuJoCo Metal contributors
# Licensed under the Apache License, Version 2.0.
"""Implicit profile recursion preserves the caller's explicit row budget."""
import mujoco
import pytest

from mujoco_metal.capacity import CapacityLimits, CapacityOverflow
from mujoco_metal.coupled_constraints import lower_coupled_constraints
from mujoco_metal.stepping import validate_stepping_profile


def test_integrated_implicit_preserves_explicit_capacity():
  integrator = "implicit"
  planes = ''.join(f'<geom name="floor{i}" type="plane" size="1 1 .1" '
                   f'pos="0 0 {-i*.001}"/>' for i in range(7))
  model = mujoco.MjModel.from_xml_string(f'''<mujoco>
    <option integrator="{integrator}"/><worldbody>{planes}
      <body pos="0 0 .2"><freejoint/>
        <geom type="box" size=".1 .1 .1" mass="1"/>
      </body>
    </worldbody></mujoco>''')
  profile_name = f"integrated_{integrator}_v1"
  with pytest.raises(CapacityOverflow):
    validate_stepping_profile(
        model, profile=profile_name, limits=CapacityLimits(max_rows=96))
  limits = CapacityLimits(max_rows=128)
  descriptor = lower_coupled_constraints(model, limits=limits)
  assert 96 < descriptor.nr <= limits.max_rows
  profile = validate_stepping_profile(model, profile=profile_name, limits=limits)
  assert profile.name == profile_name
  assert profile.execution_plan.is_stage_enabled("coupled_constraints")
