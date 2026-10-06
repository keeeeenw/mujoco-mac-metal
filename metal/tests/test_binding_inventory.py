# Copyright 2026 The MuJoCo Metal contributors
# Licensed under the Apache License, Version 2.0.
"""Full pinned header/binding enum drift, including previously omitted physics."""

from types import SimpleNamespace
from pathlib import Path

import mujoco
import pytest

from mujoco_metal.binding_inventory import enum_inventory_drift
from mujoco_metal.binding_inventory import pinned_enum_inventory
from mujoco_metal.binding_inventory import (
    binding_surface_drift, package_entrypoint_drift, pinned_binding_surface)
from mujoco_metal.registry import MILESTONE_IDS


def _bindings_copy():
  return SimpleNamespace(**{
      name: getattr(mujoco, name) for name in dir(mujoco)
      if name == "__version__" or name.startswith("mjt")})


def test_every_pinned_header_enum_matches_python_bindings():
  catalogue = pinned_enum_inventory()
  assert enum_inventory_drift() == ()
  assert catalogue["catalogue_matches_installed_bindings"] is True
  assert (catalogue["group_count"], catalogue["member_count"]) == (70, 529)
  assert catalogue["gpu_qualified"] is False
  for group in catalogue["groups"].values():
    assert group["source"].startswith("include/mujoco/")
    assert group["owner"] in MILESTONE_IDS
    assert group["domain"] and group["members"]
  assert catalogue["groups"]["mjtFlexSelf"]["owner"] == "018"
  assert catalogue["groups"]["mjtSleepPolicy"]["owner"] == "017"
  assert catalogue["groups"]["mjtPluginCapabilityBit"]["owner"] == "019"


def test_new_groups_and_member_value_drift_are_rejected():
  bindings = _bindings_copy()
  bindings.mjtNewPhysics = SimpleNamespace(__members__={"mjNEW": 0})
  assert "Unclassified enum: mjtNewPhysics" in enum_inventory_drift(bindings)
  bindings = _bindings_copy()
  values = dict(mujoco.mjtFlexSelf.__members__)
  values["mjFLEXSELF_NONE"] = 999
  bindings.mjtFlexSelf = SimpleNamespace(__members__=values)
  assert "Changed members or values: mjtFlexSelf" in enum_inventory_drift(bindings)


def test_removed_groups_versions_and_mutable_queries():
  bindings = _bindings_copy()
  del bindings.mjtSleepState
  assert "Missing enum: mjtSleepState" in enum_inventory_drift(bindings)
  bindings.__version__ = "other"
  assert "MuJoCo version differs from the pinned enum catalogue" in enum_inventory_drift(bindings)
  first = pinned_enum_inventory()
  first["groups"]["mjtSleepState"]["owner"] = "corrupted"
  assert pinned_enum_inventory()["groups"]["mjtSleepState"]["owner"] == "017"


def test_preflight_reports_catalogue_without_claiming_native_completion():
  from mujoco_metal.__main__ import preflight
  result = preflight(include_inventory=True)
  assert result["pinned_enums"]["catalogue_matches_installed_bindings"] is True
  assert result["inventory_complete"] is False
  assert result["gpu_qualified"] is False


def _surface_bindings_copy():
  return SimpleNamespace(**{name: getattr(mujoco, name) for name in dir(mujoco)})


def test_pinned_runtime_fields_and_api_surface_match_without_support_claims():
  surface = pinned_binding_surface()
  assert binding_surface_drift() == ()
  assert surface["binding_field_count"] == 809
  assert surface["binding_function_count"] == 317
  assert len(surface["header_functions"]) == 527
  assert "set_mjcb_passive" in surface["binding_only_functions"]
  assert "from_zip" in surface["binding_only_functions"]
  assert len(surface["binding_only_functions"]) == 24
  assert surface["gpu_qualified"] is False
  assert surface["feature_mapping_complete"] is False
  assert surface["python_class_count"] == 69
  assert surface["option_field_count"] == 29
  assert surface["package_entrypoint_count"] == 67
  assert surface["function_mapping_count"] == 317
  assert surface["function_mapping_complete"] is False
  assert package_entrypoint_drift() == ()
  assert surface["package_entrypoint_drift"] == []
  assert surface["python_classes"]["MjSpec"]["owner"] == "REQ-API-001"
  assert surface["python_classes"]["Renderer"]["owner"] == "REQ-API-002"
  assert surface["python_classes"]["MjData"]["owner"] == "REQ-API-003"
  assert surface["option_fields"]["enableflags"]["requirements"] == [
      "REQ-ENBL-001", "REQ-ENBL-002", "REQ-ENBL-003", "REQ-ENBL-004"]
  assert surface["python_function_map"]["mj_step"]["owner"] == "REQ-INT-001"
  assert surface["python_function_map"]["mj_step"]["counterpart"] == \
      "mujoco_metal/simulation.py:MetalSimulation.step"
  assert surface["python_function_map"]["mj_forward"]["counterpart"] == \
      "mujoco_metal/native_api.py:mj_forward"
  for name, group in surface["classes"].items():
    assert set(group["binding_fields"]) - set(group["header_fields"]) == set(group["binding_only_fields"])
    assert set(group["header_fields"]) - set(group["binding_fields"]) == set(group["header_only_fields"])
    assert all(row["source"].startswith("include/mujoco/")
               for row in group["header_fields"].values())
  assert surface["classes"]["MjModel"]["header_fields"]["body_invweight0"]["c_type"] == "mjtNum*"
  assert "mj_step" in surface["functions"]
  assert "mjd_transitionFD" in surface["functions"]


def test_runtime_field_additions_removals_and_function_drift_are_detected():
  bindings = _surface_bindings_copy()
  fields = {name: value for name, value in vars(mujoco.MjOption).items()
            if not name.startswith("_")}
  fields.pop("timestep")
  fields["new_physics_option"] = property(lambda self: 0)
  bindings.MjOption = type("ChangedOption", (), fields)
  bindings.mj_newPhysics = lambda: None
  del bindings.mjd_transitionFD
  errors = binding_surface_drift(bindings)
  assert "Unclassified field: MjOption.new_physics_option" in errors
  assert "Missing field: MjOption.timestep" in errors
  assert "Unclassified function: mj_newPhysics" in errors
  assert "Missing function: mjd_transitionFD" in errors
  del bindings.MjData
  assert "Missing runtime structure: MjData" in binding_surface_drift(bindings)


def test_surface_queries_are_owned_and_preflight_keeps_coverage_incomplete():
  surface = pinned_binding_surface()
  surface["classes"]["MjData"]["binding_fields"].clear()
  surface["functions"].clear()
  surface["python_function_map"].clear()
  assert pinned_binding_surface()["binding_field_count"] == 809
  from mujoco_metal.__main__ import preflight
  result = preflight(include_inventory=True)
  assert result["pinned_binding_surface"]["catalogue_matches_installed_bindings"] is True
  assert result["pinned_binding_surface"]["feature_mapping_complete"] is False
  assert result["inventory_complete"] is False


def test_exported_class_members_and_package_counterparts_have_exact_crosswalks():
  import mujoco_metal.native_api as native_api
  from mujoco_metal.registry import REQUIREMENTS
  surface = pinned_binding_surface()
  requirement_ids = {row.id for row in REQUIREMENTS}
  for field, row in surface["option_fields"].items():
    assert field in surface["classes"]["MjOption"]["binding_fields"]
    assert row["requirements"]
    assert set(row["requirements"]) <= requirement_ids
  for name, row in surface["package_entrypoints"].items():
    assert callable(getattr(native_api, name))
    assert row["path"] == f"mujoco_metal/native_api.py:{name}"
    assert row["requirement"] in requirement_ids
    assert row["semantics"]
  assert set(surface["python_function_map"]) == set(surface["functions"])
  for name, row in surface["python_function_map"].items():
    assert row["owner"] in requirement_ids, name
    assert row["classification"] and row["execution"]
    if row["counterpart_kind"] == "public_wrapper":
      assert row["counterpart"] == surface["package_entrypoints"][name]["path"]
    elif row["counterpart_kind"] == "internal_or_prepared_pipeline":
      module_path = row["counterpart"].split(":", 1)[0]
      assert (Path(__file__).parents[1] / module_path).is_file()
    else:
      assert row["counterpart"] is None
      assert row["gap"], name
  assert surface["python_function_map"]["mj_camlight"]["owner"] == "REQ-API-002"
  assert surface["python_function_map"]["mj_step"]["counterpart_kind"] == \
      "internal_or_prepared_pipeline"
  assert surface["python_function_map"]["mju_muscleDynamics"]["owner"] == \
      "REQ-DYN-002"
  assert surface["python_function_map"]["mju_encodePyramid"]["owner"] == \
      "REQ-CON-001"
  assert surface["python_function_map"]["mj_rayHfield"]["owner"] == \
      "REQ-GEO-004"
  for name, row in surface["python_classes"].items():
    assert row["owner"] in requirement_ids
    assert row["execution"] == "upstream_host"
    assert row["public_members"] == sorted(set(row["public_members"]))


def test_new_or_removed_python_classes_methods_and_package_wrappers_are_rejected():
  bindings = _surface_bindings_copy()
  class NewPublicSurface:
    __module__ = "mujoco._test"
    value = 1
  bindings.MjNewPublicSurface = NewPublicSurface
  errors = binding_surface_drift(bindings)
  assert "Unclassified Python class: MjNewPublicSurface" in errors

  bindings = _surface_bindings_copy()
  del bindings.MjsKey
  assert "Missing Python class: MjsKey" in binding_surface_drift(bindings)

  bindings = _surface_bindings_copy()
  bindings.MjsKey = type("ChangedKey", (mujoco.MjsKey,),
                         {"__module__": "mujoco._specs",
                          "new_public_member": 1})
  assert "Unclassified class member: MjsKey.new_public_member" in \
      binding_surface_drift(bindings)
  bindings.MjsKey = type("IncompleteKey", (),
                         {"__module__": "mujoco._specs"})
  assert "Missing class member: MjsKey.time" in binding_surface_drift(bindings)

  import mujoco_metal.native_api as native_api
  wrappers = SimpleNamespace(**vars(native_api))
  wrappers.mj_newPhysics = lambda: None
  del wrappers.mj_forward
  errors = package_entrypoint_drift(wrappers)
  assert "Unclassified package entrypoint: mj_newPhysics" in errors
  assert "Missing package entrypoint: mj_forward" in errors


def test_source_wrapper_inventory_never_executes_runtime_source(tmp_path):
  source = tmp_path / "native_api.py"
  source.write_text("""
raise RuntimeError('inventory must never execute this source')
def mj_newPhysics():
  pass
class Nested:
  def mj_not_a_public_entrypoint(self):
    pass
def private():
  def mj_nested():
    pass
""")
  errors = package_entrypoint_drift(source_path=source)
  assert "Unclassified package entrypoint: mj_newPhysics" in errors
  assert not any("mj_not_a_public_entrypoint" in error or "mj_nested" in error
                 for error in errors)
  assert "Missing package entrypoint: mj_factorM" in errors


def test_source_wrapper_inventory_agrees_with_explicit_runtime_module():
  import mujoco_metal.native_api as native_api
  assert package_entrypoint_drift() == package_entrypoint_drift(native_api) == ()
  with pytest.raises(ValueError, match="api_module or source_path"):
    package_entrypoint_drift(native_api, source_path="unused.py")
