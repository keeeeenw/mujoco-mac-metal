# Copyright 2026 The MuJoCo Metal contributors
# Licensed under the Apache License, Version 2.0.
"""Pinned classification and combination crosswalk, not physics qualification.

This host-only inventory distinguishes specification coverage, model admission,
public wrappers and internal pipelines. A classified row never proves that its
native implementation works. Numerical acceptance belongs to its owning task.
"""

from itertools import combinations_with_replacement, product

from .binding_inventory import (
    binding_surface_drift, counterpart_source_drift, enum_inventory_drift,
    header_inventory_drift, package_entrypoint_drift, pinned_binding_surface, pinned_enum_inventory,
    requirement_source_drift,
)
from .registry import MILESTONE_IDS, REQUIREMENTS, classify_model_field


_DATA_RULES = (
    ("act", "007"), ("ctrl", "007"), ("moment_", "007"),
    ("ten", "008"), ("wrap_", "008"), ("flex", "018"),
    ("plugin", "019"), ("nplugin", "019"), ("userdata", "019"),
    ("sensor", "016"), ("cam_", "016"), ("light_", "out-of-scope"),
    ("energy", "015"), ("flg_energy", "015"), ("history", "015"),
    ("qDeriv", "015"), ("qH", "015"), ("qLU", "015"),
    ("efc_", "014"), ("solver", "014"), ("contact", "010"),
    ("island", "017"), ("map_", "017"), ("tree_", "017"),
    ("body_awake", "017"), ("dof_awake", "017"),
    ("dof_island", "017"), ("parent_awake", "017"),
    ("bvh_", "017"), ("maxuse_", "017"), ("iefc_", "017"),
    ("iacc", "017"), ("ifrc_", "017"),
)
_DATA_EXACT = {
    **dict.fromkeys(("qpos", "qvel", "qacc", "qacc_smooth", "qacc_warmstart",
        "qfrc_applied", "xfrc_applied", "eq_active", "mocap_pos", "mocap_quat",
        "time", "warning", "model"), "006"),
    **dict.fromkeys(("M", "qM", "qLD", "qLDiagInv", "cdof", "cdof_dot",
        "cinert", "crb", "cvel", "xanchor", "xaxis", "ximat", "xipos", "xmat",
        "xpos", "xquat", "geom_xmat", "geom_xpos", "site_xmat", "site_xpos",
        "subtree_com", "qfrc_bias", "qfrc_smooth"), "baseline"),
    **dict.fromkeys(("cacc", "cfrc_ext", "cfrc_int", "subtree_angmom",
        "subtree_linvel", "flg_rnepost", "flg_subtreevel"), "016"),
    **dict.fromkeys(("qfrc_damper", "qfrc_fluid", "qfrc_gravcomp",
        "qfrc_passive", "qfrc_spring"), "015"),
    "qfrc_actuator": "007", "qfrc_constraint": "014", "qfrc_inverse": "019",
    **dict.fromkeys(("nA", "nJ", "nY", "narena", "nbuffer", "ncon", "ne",
        "nefc", "nf", "nidof", "nisland", "nl", "nv_awake", "nbody_awake",
        "nparent_awake", "ntree_awake", "parena", "pbase", "pstack", "arena",
        "buffer", "threadlock", "threadpool"), "017"),
    "signature": "006", "timer": "019",
}


def _data_owner(name):
  if name in _DATA_EXACT:
    return _DATA_EXACT[name]
  matches = [(prefix, owner) for prefix, owner in _DATA_RULES
             if name.startswith(prefix)]
  if not matches:
    raise ValueError(f"unclassified pinned data field: {name}")
  return max(matches, key=lambda row: len(row[0]))[1]


def _header_function_owner(name):
  # C-only entries are host allocation, compilation, resource/extension or UI
  # services. Device physics APIs exposed in Python use the explicit map.
  if name.startswith(("mjr_", "mjv_", "mjui_")):
    return "REQ-API-002", "upstream_host_visualization"
  if name.startswith("mjc_"):
    return "REQ-GEO-005", "upstream_C_SDF_API; native pipeline is separately qualified"
  if name.startswith(("mjp_", "mju_")) or name in ("mj_getPluginConfig", "mj_warning"):
    return "REQ-API-003", "upstream_host_resource_memory_or_extension_service"
  if name.startswith("mjs_") or name.startswith("mj_"):
    return "REQ-API-001", "upstream_host_compilation_or_storage_service"
  raise ValueError(f"unclassified pinned C-only function: {name}")


def _field_rows(surface):
  rows = []
  for cls, group in surface["classes"].items():
    names = set(group["binding_fields"]) | set(group["header_fields"])
    for name in sorted(names):
      if cls == "MjModel":
        rule = classify_model_field(name)
        if rule is None and name == "buffer":
          rule = ("017", "upstream buffer ownership; device layouts differ")
        if rule is None:
          raise ValueError(f"unclassified pinned model field: {name}")
        owner, note = rule
      elif cls == "MjData":
        owner, note = _data_owner(name), "runtime state or derived stage storage"
      elif cls == "MjOption":
        reqs = surface["option_fields"][name]["requirements"]
        owner = next(r.milestone for r in REQUIREMENTS if r.id == reqs[0])
        note = "option consumed by the explicitly listed subsystem requirements"
      elif cls == "MjContact":
        owner, note = "014", "collision/constraint observation; storage ABI differs"
      else:
        owner, note = "014", "compiled solver scaling and host model statistics"
      metadata = group["header_fields"].get(name, {})
      rows.append({"id": f"FIELD:{cls}.{name}", "owner": owner,
          "source": metadata.get("source", f"Python binding {cls}.{name}"),
          "binding_exposed": name in group["binding_fields"],
          "execution": "host_specification; backend use requires owning-stage qualification",
          "qualification": "unqualified", "limitation": note,
          "admission": "compiled pinned layout; public mutation and storage ABI are separate contracts",
          "entry_point": cls, "implementation": "inventory_only",
          "tests": ["test_support_inventory_005.py"]})
  return rows


def _enum_rows(enums):
  by_member = {}
  for req in REQUIREMENTS:
    for member in req.enums:
      by_member.setdefault(member, []).append(req.id)
  rows = []
  for name, group in enums["groups"].items():
    for member, value in group["members"].items():
      rows.append({"id": f"ENUM:{name}.{member}", "value": value,
          "owner": group["owner"], "source": group["source"],
          "domain": group["domain"], "requirements": by_member.get(f"{name}.{member}", []),
          "execution": "specification; see requirement execution for selectable physics",
          "qualification": "unqualified", "implementation": "inventory_only",
          "tests": ["test_binding_inventory.py"]})
  return rows


def _function_rows(surface):
  rows = []
  for name in sorted(set(surface["functions"]) | set(surface["header_functions"])):
    mapping = surface["python_function_map"].get(name)
    if mapping is None:
      owner, classification = _header_function_owner(name)
      mapping = {"owner": owner, "classification": classification,
          "execution": "upstream_host", "counterpart": None,
          "counterpart_kind": None,
          "gap": "C-only entrypoint remains upstream; native device/API coverage is separately tracked"}
    row = dict(mapping)
    req = next(r for r in REQUIREMENTS if r.id == row["owner"])
    row.update(id=f"API:{name}", requirement=row.pop("owner"), owner=req.milestone,
        source=surface["header_functions"].get(name, {}).get("source", f"Python binding {name}"),
        binding_exposed=name in surface["functions"], qualification="unqualified",
        implementation="public_wrapper" if row.get("counterpart_kind") == "public_wrapper"
          else "internal_pipeline" if row.get("counterpart_kind") == "internal_or_prepared_pipeline"
          else "upstream_host_or_owned_gap",
        tests=["test_binding_inventory.py", *req.tests],
        standalone_counterpart=row.get("counterpart_kind") == "public_wrapper")
    if row.get("counterpart_kind") == "internal_or_prepared_pipeline":
      row["gap"] = "Internal/prepared pipeline is not evidence of same-named standalone API equivalence. " + row.get("gap", "")
    rows.append(row)
  return rows


def _combination_rows(enums):
  def members(group, prefix):
    return sorted(name for name in enums["groups"][group]["members"]
                  if name.startswith(prefix) and "UNDEFINED" not in name)
  geoms = members("mjtGeom", "mjGEOM_")
  # Decorations have values above mjNGEOMTYPES; they are renderer-only, not
  # collision combinations. All nine real geometry kinds remain inventoried.
  count = enums["groups"]["mjtGeom"]["members"]["mjNGEOMTYPES"]
  geoms = [name for name in geoms if enums["groups"]["mjtGeom"]["members"][name] < count]
  rows = []
  for left, right in combinations_with_replacement(geoms, 2):
    owner = ("013" if "mjGEOM_SDF" in (left, right) else
             "012" if "mjGEOM_HFIELD" in (left, right) else
             "011" if "mjGEOM_MESH" in (left, right) else
             "010" if set((left, right)) & {"mjGEOM_CYLINDER", "mjGEOM_ELLIPSOID"}
             else "baseline")
    absent = set((left, right)) <= {"mjGEOM_PLANE", "mjGEOM_HFIELD"}
    rows.append({"id": f"PAIR:{left}:{right}", "axes": [left, right],
        "owner": owner, "execution": "none" if absent else "device_candidate",
        "source": "src/engine/engine_collision_driver.c:mjCOLLISIONFUNC",
        "admission": ("reject absent pinned dispatcher" if left == right == "mjGEOM_HFIELD"
                      else "empty absent pinned dispatcher" if absent else
                      "validate compiled pair, profile and capacities; not universal admission"),
        "qualification": "unqualified", "implementation": "inventory_only",
        "entry_point": "lower_coupled_constraints",
        "tests": ["test_combination_admission_005.py"]})
  groups = (("mjtTrn", "mjTRN_"), ("mjtDyn", "mjDYN_"),
            ("mjtGain", "mjGAIN_"), ("mjtBias", "mjBIAS_"))
  for axes in product(*(members(group, prefix) for group, prefix in groups)):
    rows.append({"id": "ACT:" + ":".join(axes), "axes": list(axes),
        "owner": "019" if any(name.endswith("_USER") for name in axes) else "007",
        "source": "src/engine/engine_forward.c:mj_fwdActuation",
        "execution": "device_candidate", "qualification": "unqualified", "implementation": "inventory_only",
        "admission": "CPU compiler then ActuatorModel/profile validation; typed registration for native callbacks",
        "entry_point": "ActuatorModel; validate_stepping_profile",
        "tests": ["test_combination_admission_005.py"]})
  for axes in product(*(members(group, prefix) for group, prefix in (
      ("mjtIntegrator", "mjINT_"), ("mjtSolver", "mjSOL_"),
      ("mjtCone", "mjCONE_"), ("mjtJacobian", "mjJAC_")))):
    rows.append({"id": "STEP:" + ":".join(axes), "axes": list(axes),
        "owner": "015", "dependencies": ["014", "017"],
        "source": "src/engine/engine_forward.c:mj_step",
        "execution": "device_candidate", "qualification": "unqualified", "implementation": "inventory_only",
        "admission": "validate compiled model through selected profile; classification does not admit arbitrary compositions",
        "entry_point": "validate_stepping_profile",
        "tests": ["test_combination_admission_005.py"]})
  return rows


def support_inventory():
  """Return an owned JSON-compatible crosswalk with explicit remaining owners."""
  enums, surface = pinned_enum_inventory(), pinned_binding_surface()
  result = {"target_version": enums["target_version"],
      "source_revision": enums["source_revision"],
      "scope": "complete pinned classification; model admission and physics qualification are separate",
      "enums": _enum_rows(enums), "fields": _field_rows(surface),
      "functions": _function_rows(surface), "combinations": _combination_rows(enums),
      "python_classes": surface["python_classes"],
      "option_fields": surface["option_fields"],
      "package_entrypoints": surface["package_entrypoints"],
      "requirements": [r.id for r in REQUIREMENTS],
      "gpu_qualified": False, "standalone_api_qualified": False}
  result["drift"] = list(support_inventory_drift(result))
  result["classification_complete"] = not result["drift"]
  return result


def support_inventory_drift(inventory=None):
  """Detect missing/extra inventory rows and broken source/test/owner contracts."""
  errors = [*enum_inventory_drift(), *binding_surface_drift(),
            *package_entrypoint_drift(), *counterpart_source_drift(),
            *requirement_source_drift(), *header_inventory_drift()]
  if inventory is None:
    # Build only after checking binding drift; unknown new data must not be
    # silently classified by a prefix or a general host-utility category.
    if errors:
      return tuple(errors)
    return tuple(support_inventory()["drift"])
  enums, surface = pinned_enum_inventory(), pinned_binding_surface()
  expected = {
      "enums": {f"ENUM:{group}.{member}" for group, row in enums["groups"].items() for member in row["members"]},
      "fields": {f"FIELD:{cls}.{name}" for cls, row in surface["classes"].items()
                 for name in set(row["binding_fields"]) | set(row["header_fields"])},
      "functions": {f"API:{name}" for name in set(surface["functions"]) | set(surface["header_functions"])},
      "combinations": {row["id"] for row in _combination_rows(enums)},
  }
  from pathlib import Path
  test_root = Path(__file__).parent.parent / "tests"
  for category, ids in expected.items():
    rows = inventory.get(category, [])
    actual = [row.get("id") for row in rows]
    if set(actual) != ids or len(actual) != len(set(actual)):
      errors.append(f"Incomplete or duplicate {category} crosswalk")
    for row in rows:
      if row.get("owner") not in MILESTONE_IDS:
        errors.append(f"Invalid owner: {row.get('id')}")
      for key in ("source", "execution", "implementation", "qualification", "tests"):
        if not row.get(key):
          errors.append(f"Missing {key}: {row.get('id')}")
      for test in row.get("tests", []):
        # Wheels reference repository witnesses without distributing tests.
        if Path(test).name != test or (test_root.is_dir() and not (test_root / test).is_file()):
          errors.append(f"Missing test reference: {row.get('id')}")
      if row.get("qualification") != "unqualified":
        errors.append(f"Classification promoted to physics qualification: {row.get('id')}")
  for category in ("python_classes", "option_fields", "package_entrypoints"):
    if inventory.get(category) != surface[category]:
      errors.append(f"Changed {category} crosswalk")
  if inventory.get("gpu_qualified") is not False or inventory.get("standalone_api_qualified") is not False:
    errors.append("Classification promoted to complete backend qualification")
  if inventory.get("target_version") != enums["target_version"] or inventory.get("source_revision") != enums["source_revision"]:
    errors.append("Incorrect pinned inventory provenance")
  return tuple(errors)


def inventory_summary_table(inventory=None):
  """Render the coverage explorer's dimension counts without support promotion."""
  inventory = support_inventory() if inventory is None else inventory
  lines = ["| Inventory dimension | Classified entries | Native qualification |",
           "|---|---:|---|"]
  for key in ("enums", "fields", "functions", "python_classes", "option_fields",
              "package_entrypoints", "combinations"):
    lines.append(f"| {key} | {len(inventory[key])} | Separate owning-milestone gates |")
  return "\n".join(lines) + "\n"
