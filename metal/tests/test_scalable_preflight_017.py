# Copyright 2026 The MuJoCo Metal contributors
# Licensed under the Apache License, Version 2.0.

"""CPU admission checks for the public component-sparse profile."""

import mujoco
import pytest

from mujoco_metal.capacity import CapacityLimits
import mujoco_metal.simulation as simulation_module


_CG_XML = """<mujoco>
  <option timestep=".001" solver="CG"><flag contact="disable"/></option>
  <worldbody>
    <body><joint name="slide" type="slide" axis="1 0 0"/>
      <geom type="sphere" size=".1" mass="1"/></body>
  </worldbody>
</mujoco>"""


def test_sparse_primal_solver_profiles_lower_without_dense_fallback():
  for solver, enum_name in (("PGS", "mjSOL_PGS"),
                            ("CG", "mjSOL_CG"),
                            ("Newton", "mjSOL_NEWTON")):
    model = mujoco.MjModel.from_xml_string(
        _CG_XML.replace('solver="CG"', f'solver="{solver}"'))
    profile = simulation_module.validate_stepping_profile(
        model, profile="integrated_scalable_v1")
    assert profile.name == "integrated_scalable_v1"
    assert profile.execution_plan is not None
    assert int(model.opt.solver) == int(getattr(mujoco.mjtSolver, enum_name))


def test_scalable_profile_uses_explicit_capacity_for_large_nv():
  joints = "".join(
      '<body><joint type="slide" axis="1 0 0"/>'
      '<inertial pos="0 0 0" mass="1" diaginertia="1 1 1"/></body>'
      for _ in range(130))
  model = mujoco.MjModel.from_xml_string(
      '<mujoco><option solver="PGS"><flag contact="disable"/></option>'
      f'<worldbody>{joints}</worldbody></mujoco>')
  assert model.nv == 130
  limits = CapacityLimits(
      max_nv=130, max_pairs=64, max_slots=64, max_rows=512,
      max_batch=2, memory_budget_bytes=1 << 30)
  profile = simulation_module.validate_stepping_profile(
      model, profile="integrated_scalable_v1", limits=limits)
  assert profile.name == "integrated_scalable_v1"
  with pytest.raises(ValueError, match="nv to 32"):
    simulation_module.validate_stepping_profile(
        model, profile="integrated_euler_v1", limits=limits)


def test_device_state_revalidation_receives_callers_capacity_limits(monkeypatch):
  import mujoco_metal.device_state as device_state_module

  model = mujoco.MjModel.from_xml_string(_CG_XML)
  limits = CapacityLimits(
      max_nv=1, max_pairs=64, max_slots=64, max_rows=256,
      max_batch=2, memory_budget_bytes=1 << 30)
  profile = simulation_module.validate_stepping_profile(
      model, profile="integrated_scalable_v1", limits=limits)
  seen = []

  class StopBeforeDeviceAllocation(RuntimeError):
    pass

  def validate(model_arg, timestep, *, profile, limits=None):
    seen.append(limits)
    raise StopBeforeDeviceAllocation()

  monkeypatch.setattr(device_state_module, "validate_stepping_profile", validate)
  with pytest.raises(StopBeforeDeviceAllocation):
    device_state_module.DeviceState(model, profile, 1, limits=limits)
  assert seen == [limits]
