# Copyright 2026 The MuJoCo Metal contributors
# Licensed under the Apache License, Version 2.0.
"""Milestone 005 specification completeness, not native numerical acceptance."""

from copy import deepcopy
import json
from pathlib import Path
import shutil

import mujoco
import pytest

from mujoco_metal.support_inventory import (
    inventory_summary_table, support_inventory, support_inventory_drift,
)
from mujoco_metal.binding_inventory import header_inventory_drift


def test_full_crosswalk_has_every_pinned_dimension_and_explicit_boundaries():
  result = support_inventory()
  assert result["drift"] == []
  assert result["classification_complete"] is True
  assert result["gpu_qualified"] is False
  assert result["standalone_api_qualified"] is False
  assert result["target_version"] == "3.10.0"
  assert result["source_revision"] == "28009f9105cd92784b7b0b30c0605a5e29107a77"
  assert len(result["enums"]) == 529
  assert len(result["fields"]) == 813  # binding union C header, not summing duplicates
  assert len(result["functions"]) == 554
  assert len(result["combinations"]) == 1167  # 45 pairs + 1050 actuators + 72 stepping
  assert len(result["python_classes"]) == 69
  assert len(result["option_fields"]) == 29
  assert len(result["package_entrypoints"]) == 79
  # Public API equivalence must not be inferred from an internal stage.
  rows = {row["id"]: row for row in result["functions"]}
  assert rows["API:mj_forward"]["standalone_counterpart"] is True
  assert rows["API:mj_Euler"]["standalone_counterpart"] is False
  assert "remains open" in rows["API:mj_Euler"]["gap"]
  assert "not evidence" in rows["API:mj_checkAcc"]["gap"]
  assert rows["API:mjs_addBody"]["execution"] == "upstream_host"
  assert rows["API:mjui_render"]["owner"] == "out-of-scope"
  assert rows["API:mjc_distance"]["owner"] == "013"


@pytest.mark.parametrize("category", ["enums", "fields", "functions", "combinations"])
@pytest.mark.parametrize("change", ["missing", "new", "duplicate", "owner", "test"])
def test_crosswalk_drift_cannot_hide_missing_or_unaudited_rows(category, change):
  inventory = support_inventory()
  rows = inventory[category]
  if change == "missing":
    rows.pop()
  elif change == "new":
    new = deepcopy(rows[0]); new["id"] = "unaudited-new-feature"; rows.append(new)
  elif change == "duplicate":
    rows.append(deepcopy(rows[0]))
  elif change == "owner":
    rows[0]["owner"] = "unassigned"
  else:
    rows[0]["tests"] = ["missing_witness.py"]
  assert support_inventory_drift(inventory)


def test_inventory_is_owned_json_data_and_table_stays_in_sync():
  original = support_inventory()
  copy = json.loads(json.dumps(original))
  copy["enums"].clear()
  assert len(support_inventory()["enums"]) == 529
  doc = (Path(__file__).parents[1] / "COVERAGE.md").read_text()
  begin = doc.index("<!-- INVENTORY_TABLE_BEGIN -->\n") + len("<!-- INVENTORY_TABLE_BEGIN -->\n")
  end = doc.index("<!-- INVENTORY_TABLE_END -->", begin)
  assert doc[begin:end] == inventory_summary_table(original)


def test_inventory_cli_reports_classification_without_physics_promotion(capsys):
  from mujoco_metal.__main__ import main
  assert main(["coverage", "--inventory", "--json"]) == 0
  data = json.loads(capsys.readouterr().out)
  assert data["classification_complete"] is True
  assert data["gpu_qualified"] is False
  assert main(["coverage", "--inventory"]) == 0
  assert "combinations" in capsys.readouterr().out


@pytest.mark.parametrize("change,expected", [
    ("function", "Unclassified C function: mj_newPhysics"),
    ("enum", "Unclassified C enum: mjtNewPhysics"),
    ("enum_member", "Changed C enum members: mjtSleepState"),
    ("field", "Unclassified C field: MjStatistic.new_physics"),
    ("extent", "Changed C field type/extent: MjStatistic.center"),
    ("type", "Changed C field type/extent: MjContact.dist"),
])
def test_actual_header_declarations_cannot_drift_behind_unchanged_python_bindings(
    tmp_path, change, expected):
  headers = Path(mujoco.__file__).parent / "include" / "mujoco"
  for path in headers.glob("*.h"):
    shutil.copyfile(path, tmp_path / path.name)
  if change == "function":
    path = tmp_path / "mujoco.h"
    path.write_text(path.read_text() + "\nMJAPI void mj_newPhysics(void);\n")
  elif change == "enum":
    path = tmp_path / "mjmodel.h"
    path.write_text(path.read_text() + "\ntypedef enum { mjNEW_PHYSICS = 0 } mjtNewPhysics;\n")
  else:
    filename = "mjtype.h" if change == "enum_member" else "mjdata.h" if change == "type" else "mjmodel.h"
    path = tmp_path / filename
    text = path.read_text()
    if change == "enum_member":
      text = text.replace("typedef enum mjtSleepState_ {", "typedef enum mjtSleepState_ { mjNEW_SLEEP = 999,")
    elif change == "field":
      text = text.replace("struct mjStatistic_ {", "struct mjStatistic_ {\n  int new_physics;")
    elif change == "extent":
      import re
      text = re.sub(r"(mjtNum\s+center)\[3\]", r"\1[4]", text)
    else:
      import re
      text = re.sub(r"(struct\s+mjContact_\s*\{.*?)mjtNum(\s+dist\s*;)",
                    r"\1int\2", text, count=1, flags=re.S)
    assert text != path.read_text(), "fixture mutation must actually modify the declaration"
    path.write_text(text)
  assert expected in header_inventory_drift(tmp_path)
