# Copyright 2026 The MuJoCo Metal contributors
# Licensed under the Apache License, Version 2.0.
"""Pinned public enum catalogue, distinct from native feature qualification."""

from copy import deepcopy
import ast
from functools import lru_cache
import json
import inspect
import re
from pathlib import Path

import mujoco


@lru_cache(maxsize=1)
def _catalogue():
  return json.loads(Path(__file__).with_name("pinned_enums.json").read_text())


def enum_inventory_drift(bindings=mujoco):
  """Report new/removed enums, members and values; do not silently classify them."""
  expected = _catalogue()
  errors = []
  if getattr(bindings, "__version__", None) != expected["target_version"]:
    errors.append("MuJoCo version differs from the pinned enum catalogue")
  actual = {name: getattr(bindings, name) for name in dir(bindings)
            if name.startswith("mjt")
            and hasattr(getattr(bindings, name), "__members__")}
  for name in sorted(set(actual) - set(expected["groups"])):
    errors.append(f"Unclassified enum: {name}")
  for name, group in expected["groups"].items():
    if name not in actual:
      errors.append(f"Missing enum: {name}")
      continue
    members = {member: int(value)
               for member, value in actual[name].__members__.items()}
    if members != group["members"]:
      errors.append(f"Changed members or values: {name}")
  return tuple(errors)


def pinned_enum_inventory():
  """Return owned host-only metadata; completeness never implies physics support."""
  result = deepcopy(_catalogue())
  result["drift"] = list(enum_inventory_drift())
  result["catalogue_matches_installed_bindings"] = not result["drift"]
  result["group_count"] = len(result["groups"])
  result["member_count"] = sum(len(row["members"])
                               for row in result["groups"].values())
  result["gpu_qualified"] = False
  return result


@lru_cache(maxsize=1)
def _surface_catalogue():
  return json.loads(Path(__file__).with_name("pinned_surface.json").read_text())


def binding_surface_drift(bindings=mujoco):
  """Reject added/removed public runtime fields and function bindings.

  The catalogue retains C-only fields/functions separately. A matching surface
  proves pinned classification, never that a native substitute is available.
  """
  expected = _surface_catalogue()
  errors = []
  if getattr(bindings, "__version__", None) != expected["target_version"]:
    errors.append("MuJoCo version differs from the pinned binding surface")
  for name, group in expected["classes"].items():
    cls = getattr(bindings, name, None)
    if cls is None:
      errors.append(f"Missing runtime structure: {name}")
      continue
    actual = {field for field, value in vars(cls).items()
              if not field.startswith("_")
              and (isinstance(value, property) or inspect.isdatadescriptor(value))}
    wanted = set(group["binding_fields"])
    for field in sorted(actual - wanted):
      errors.append(f"Unclassified field: {name}.{field}")
    for field in sorted(wanted - actual):
      errors.append(f"Missing field: {name}.{field}")
  expected_classes = expected.get("python_classes", {})
  actual_classes = {
      name: getattr(bindings, name) for name in dir(bindings)
      if not name.startswith("_")
      and inspect.isclass(getattr(bindings, name))
      and getattr(getattr(bindings, name), "__module__", "").startswith("mujoco")
      and not hasattr(getattr(bindings, name), "__members__")
  }
  for name in sorted(set(actual_classes) - set(expected_classes)):
    errors.append(f"Unclassified Python class: {name}")
  for name in sorted(set(expected_classes) - set(actual_classes)):
    errors.append(f"Missing Python class: {name}")
  for name in sorted(set(actual_classes) & set(expected_classes)):
    actual_members = {member for member in dir(actual_classes[name])
                      if not member.startswith("_")}
    wanted_members = set(expected_classes[name]["public_members"])
    for member in sorted(actual_members - wanted_members):
      errors.append(f"Unclassified class member: {name}.{member}")
    for member in sorted(wanted_members - actual_members):
      errors.append(f"Missing class member: {name}.{member}")
  actual_functions = {name for name in dir(bindings)
                      if not name.startswith("_")
                      and inspect.isroutine(getattr(bindings, name))}
  wanted = set(expected["functions"])
  for name in sorted(actual_functions - wanted):
    errors.append(f"Unclassified function: {name}")
  for name in sorted(wanted - actual_functions):
    errors.append(f"Missing function: {name}")
  function_map = expected.get("python_function_map", {})
  if set(function_map) != wanted:
    errors.append("Python function map does not exactly match pinned function surface")
  for name, row in function_map.items():
    if (not row.get("classification") or not row.get("owner")
        or not row.get("execution")):
      errors.append(f"Incomplete Python function classification: {name}")
    counterpart = row.get("counterpart")
    kind = row.get("counterpart_kind")
    if kind == "public_wrapper":
      entrypoint = expected.get("package_entrypoints", {}).get(name)
      if entrypoint is None or counterpart != entrypoint.get("path"):
        errors.append(f"Invalid package counterpart mapping: {name}")
    elif kind == "internal_or_prepared_pipeline" and not counterpart:
      errors.append(f"Missing internal pipeline counterpart: {name}")
    elif kind is None and counterpart is not None:
      errors.append(f"Unclassified function counterpart kind: {name}")
  actual_options = set(expected.get("option_fields", {}))
  if actual_options != set(expected["classes"]["MjOption"]["binding_fields"]):
    errors.append("MjOption crosswalk does not match its binding fields")
  for field, row in expected.get("option_fields", {}).items():
    if not row.get("requirements") or any(not isinstance(req, str)
                                            for req in row["requirements"]):
      errors.append(f"Incomplete MjOption owner mapping: {field}")
  return tuple(errors)


def package_entrypoint_drift(api_module=None, *, source_path=None):
  """Check wrapper classification without importing device execution code.

  Default preflight inspects top-level public function definitions in the
  installed source. An explicitly supplied module retains runtime inspection
  for tests/tools that already imported it. Neither route proves qualification.
  """
  expected = set(_surface_catalogue().get("package_entrypoints", {}))
  if api_module is None:
    path = (Path(__file__).with_name("native_api.py")
            if source_path is None else Path(source_path))
    syntax = ast.parse(path.read_text(), filename=str(path))
    actual = {node.name for node in syntax.body
              if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
              and node.name.startswith(("mj_", "mju_", "mjd_"))}
  else:
    if source_path is not None:
      raise ValueError("provide api_module or source_path, not both")
    actual = {name for name, value in vars(api_module).items()
              if name.startswith(("mj_", "mju_", "mjd_")) and callable(value)
              and not name.startswith("_")}
  errors = []
  for name in sorted(actual - expected):
    errors.append(f"Unclassified package entrypoint: {name}")
  for name in sorted(expected - actual):
    errors.append(f"Missing package entrypoint: {name}")
  return tuple(errors)


def counterpart_source_drift(surface=None, *, source_root=None):
  """Check declared source modules and named callables without device imports.

  Module-only pipeline mappings retain their deliberately broad scope. A
  declared callable must actually exist at its class/function path; an
  existing module alone must not hide a deleted or renamed counterpart.
  This structural check does not confer numerical qualification.
  """
  surface = _surface_catalogue() if surface is None else surface
  root = (Path(__file__).parent.parent if source_root is None
          else Path(source_root)).resolve()
  errors = []
  syntax_cache = {}
  for name, row in surface.get("python_function_map", {}).items():
    counterpart = row.get("counterpart")
    if counterpart is None:
      continue
    module, _, symbol = counterpart.partition(":")
    path = (root / module).resolve()
    if not path.is_relative_to(root) or not path.is_file():
      errors.append(f"Missing counterpart source: {name}")
      continue
    if not symbol:
      continue
    if path not in syntax_cache:
      syntax_cache[path] = ast.parse(path.read_text(), filename=str(path))
    nodes = syntax_cache[path].body
    for part in symbol.split("."):
      match = next((node for node in nodes
                    if isinstance(node, (ast.FunctionDef,
                                         ast.AsyncFunctionDef, ast.ClassDef))
                    and node.name == part), None)
      if match is None:
        errors.append(f"Missing counterpart callable: {name}")
        break
      nodes = match.body
  return tuple(errors)


def requirement_source_drift(requirements=None, *, source_root=None):
  """Check backend anchors in support requirements, without GPU imports.

  A requirement can cite multiple backend and upstream references. Only
  backend fragments are checked here; upstream references retain their pinned
  semantic audit rather than being checked against this newer source tree.
  Existence of an anchor is structural evidence, not physics qualification.
  """
  if requirements is None:
    from .registry import REQUIREMENTS
    requirements = REQUIREMENTS
  rows = {}
  for requirement in requirements:
    fragments = [part.strip() for part in requirement.source_ref.split(';')
                 if part.strip().startswith('mujoco_metal/')]
    for index, fragment in enumerate(fragments):
      name = requirement.id if len(fragments) == 1 else f'{requirement.id}[{index}]'
      rows[name] = {'counterpart': fragment}
  return counterpart_source_drift({'python_function_map': rows},
                                  source_root=source_root)


def pinned_binding_surface():
  """Return owned specification metadata and drift, without execution claims."""
  result = deepcopy(_surface_catalogue())
  result["drift"] = list(binding_surface_drift())
  result["catalogue_matches_installed_bindings"] = not result["drift"]
  result["binding_field_count"] = sum(len(c["binding_fields"])
                                      for c in result["classes"].values())
  result["binding_function_count"] = len(result["functions"])
  result["python_class_count"] = len(result.get("python_classes", {}))
  result["option_field_count"] = len(result.get("option_fields", {}))
  result["package_entrypoint_count"] = len(result.get("package_entrypoints", {}))
  result["package_entrypoint_drift"] = list(package_entrypoint_drift())
  result["counterpart_source_drift"] = list(counterpart_source_drift())
  result["function_mapping_count"] = len(result.get("python_function_map", {}))
  result["function_mapping_complete"] = False
  result["gpu_qualified"] = False
  result["feature_mapping_complete"] = False
  result["header_drift"] = list(header_inventory_drift())
  return result


def header_inventory_drift(header_root=None):
  """Compare C-only declarations too, against the installed pinned headers.

  Checking Python exposure alone misses C-only additions. Parse exported
  function prototypes (not extern callback variables), and the five runtime
  structures including each member's C type and array extent.
  """
  root = (Path(mujoco.__file__).parent / "include" / "mujoco"
          if header_root is None else Path(header_root))
  surface = _surface_catalogue()
  errors = []
  text = {}
  for name in ("mujoco.h", "mjmodel.h", "mjdata.h"):
    path = root / name
    if not path.is_file():
      errors.append(f"Missing pinned header: {name}")
      continue
    text[name] = re.sub(r"/\*.*?\*/|//[^\n]*", "", path.read_text(), flags=re.S)
  if errors:
    return tuple(errors)
  enum_groups = {}
  for path in root.glob("*.h"):
    source = re.sub(r"/\*.*?\*/|//[^\n]*", "", path.read_text(), flags=re.S)
    for body, group in re.findall(
        r"typedef\s+enum(?:\s+\w+)?\s*\{(.*?)\}\s*(mjt\w+)\s*;", source, flags=re.S):
      enum_groups[group] = {
          re.match(r"\s*(\w+)", part).group(1)
          for part in body.split(",") if part.strip()}
  expected_enums = _catalogue()["groups"]
  for name in sorted(set(enum_groups) - set(expected_enums)):
    errors.append(f"Unclassified C enum: {name}")
  for name in sorted(set(expected_enums) - set(enum_groups)):
    errors.append(f"Missing C enum: {name}")
  for name in set(enum_groups) & set(expected_enums):
    if enum_groups[name] != set(expected_enums[name]["members"]):
      errors.append(f"Changed C enum members: {name}")
  functions = set()
  for statement in re.findall(r"\bMJAPI\s+[^;]+;", text["mujoco.h"]):
    if re.match(r"MJAPI\s+extern\b", statement):
      continue  # Callback variables are not function API declarations.
    match = re.search(r"\b([A-Za-z_]\w*)\s*\(", statement)
    if match:
      functions.add(match.group(1))
  for name in sorted(functions - set(surface["header_functions"])):
    errors.append(f"Unclassified C function: {name}")
  for name in sorted(set(surface["header_functions"]) - functions):
    errors.append(f"Missing C function: {name}")
  for cls, group in surface["classes"].items():
    name = "mjdata.h" if cls in ("MjData", "MjContact") else "mjmodel.h"
    match = re.search(r"struct\s+" + group["c_name"] + r"_\s*\{(.*?)\n\};",
                      text[name], flags=re.S)
    if match is None:
      errors.append(f"Missing C structure: {cls}")
      continue
    fields = {}
    for statement in match.group(1).split(";")[:-1]:
      declaration = re.fullmatch(r"\s*(.*?)\b([A-Za-z_]\w*)\s*((?:\[[^]]*\]\s*)*)", statement, flags=re.S)
      if declaration is None:
        errors.append(f"Unparsed C declaration: {cls}")
        continue
      ctype, field, extent = declaration.groups()
      fields[field] = ("".join(ctype.split()), "".join(extent.split()))
    wanted = group["header_fields"]
    for field in sorted(set(fields) - set(wanted)):
      errors.append(f"Unclassified C field: {cls}.{field}")
    for field in sorted(set(wanted) - set(fields)):
      errors.append(f"Missing C field: {cls}.{field}")
    for field in set(fields) & set(wanted):
      target = ("".join(wanted[field]["c_type"].split()), wanted[field]["extent"])
      if fields[field] != target:
        errors.append(f"Changed C field type/extent: {cls}.{field}")
  return tuple(errors)
