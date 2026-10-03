# Copyright 2026 The MuJoCo Metal contributors
# Licensed under the Apache License, Version 2.0.
"""CPU admission gate for every native demo model (no GPU).

For each `metal/examples/*.xml`, runs stepping-profile validation and
coupled-constraint lowering for `integrated_euler_v1`. Models that fail must
exactly match the documented set of demos targeting other native profiles;
any other failure (new demo rot, capacity breach) fails this test.
"""

import glob
from pathlib import Path

import mujoco
import pytest

from mujoco_metal.coupled_constraints import lower_coupled_constraints
from mujoco_metal.simulation import validate_stepping_profile

_PROFILE = "integrated_euler_v1"

# Demo models that do not target integrated_euler_v1 (each uses a different
# native stepping profile or integrator), hence excluded from admission here.
_NON_EULER_DEMOS = {
    "offcenter_balance": "implicitfast integrator profile",
    "tumbling_toys": "contact_free_rk4_v1 profile",
    "wave_lattice": "implicitfast integrator profile",
    "scanning_rig": "contact_free_sensor_euler_v1 profile",
}


def test_all_demo_models_admit_or_documented():
  examples = Path(__file__).parent.parent / "examples"
  assert examples.is_dir()
  failures = {}
  admitted = {}
  for path in sorted(examples.glob("*.xml")):
    name = path.stem
    try:
      model = mujoco.MjModel.from_xml_path(str(path))
      validate_stepping_profile(model, profile=_PROFILE)
      desc = lower_coupled_constraints(model)
      admitted[name] = (desc.npairs, desc.ncontacts_max)
    except Exception as exc:  # noqa: BLE001 - collected, not hidden
      failures[name] = str(exc)
  assert set(failures) == set(_NON_EULER_DEMOS), (
      f"unexpected admission outcome: new failures "
      f"{sorted(set(failures) - set(_NON_EULER_DEMOS))}, newly passing "
      f"{sorted(set(_NON_EULER_DEMOS) - set(failures))}: {failures}")
  for name, counts in sorted(admitted.items()):
    assert counts[0] <= 16 and counts[1] <= 24, (name, counts)
