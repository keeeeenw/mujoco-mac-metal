# Copyright 2026 The MuJoCo Metal contributors
# Licensed under the Apache License, Version 2.0.
"""Executable support-contract checks: enum/field inventory drift and docs agreement."""

import mujoco
import pytest

from mujoco_metal.registry import (
    FIELD_RULES,
    MILESTONE_IDS,
    REQUIREMENTS,
    Implementation,
    _ENUMS,
    classify_model_field,
    coverage_table,
    requirement_ids,
    unclassified_enum_members,
)

MARKER_BEGIN = "<!-- COVERAGE_TABLE_BEGIN -->\n"
MARKER_END = "<!-- COVERAGE_TABLE_END -->\n"


def test_requirement_ids_unique_and_milestones_valid():
  ids = requirement_ids()
  assert len(ids) == len(set(ids)) > 0
  for req in REQUIREMENTS:
    assert req.milestone in MILESTONE_IDS, req.id
    assert req.id.startswith("REQ-"), req.id


def test_every_pinned_enum_member_classified():
  missing = unclassified_enum_members()
  assert missing == (), f"unclassified pinned enum members: {missing}"
  # The contract must actually cover the full pinned surface, not a subset.
  total = sum(
      len(getattr(mujoco, name).__members__)
      for name in _ENUMS
      if getattr(mujoco, name, None) is not None
  )
  assert total == 164, total


def test_every_model_field_classified_without_dead_rules():
  names = [n for n in dir(mujoco.MjModel) if not n.startswith("_")]
  assert len(names) == 598, len(names)
  unclassified = [n for n in names if classify_model_field(n) is None]
  assert unclassified == [], unclassified
  dead = [
      rule for rule, _, _ in FIELD_RULES
      if not any(n == rule or n.startswith(rule) for n in names)
  ]
  assert dead == [], dead


def test_docs_match_generated_table():
  from pathlib import Path
  doc = Path(__file__).parents[1] / "COVERAGE.md"
  text = doc.read_text()
  begin = text.index(MARKER_BEGIN) + len(MARKER_BEGIN)
  end = text.index(MARKER_END)
  assert text[begin:end] == coverage_table()


def test_omissions_have_future_owners():
  implemented = {"baseline", "004", "005"}
  future = [r for r in REQUIREMENTS if r.milestone not in implemented
            and r.milestone != "out-of-scope"]
  assert future, "contract must assign omissions to later milestones"
  owners = {r.milestone for r in future}
  assert owners >= {"006", "007", "008", "009", "010", "011", "012", "013",
                    "014", "015", "016", "017", "018", "019", "020"}, owners


def test_coverage_cli_lists_contract(capsys):
  from mujoco_metal.__main__ import main
  assert main(["coverage"]) == 0
  out, _ = capsys.readouterr()
  assert "| REQ-MOD-001 |" in out
  assert main(["coverage", "--json"]) == 0
  out, _ = capsys.readouterr()
  assert '"REQ-EQ-001"' in out


def _user_model(kind):
  body = ('<body pos="0 0 1"><joint name="j" type="hinge" axis="0 1 0"/>'
          '<geom type="sphere" size="0.1" mass="1"/></body>')
  if kind == "dyn":
    act = '<general joint="j" dyntype="user"/>'
  elif kind == "gain":
    act = '<general joint="j" gaintype="user"/>'
  else:
    act = '<general joint="j" biastype="user"/>'
  return mujoco.MjModel.from_xml_string(
      f'<mujoco><option timestep="0.002"/><worldbody>{body}</worldbody>'
      f'<actuator>{act}</actuator></mujoco>')


def test_user_callbacks_rejected_with_019_owner():
  # R6: USER variants are explicit 019 rows, rejected at admission.
  from mujoco_metal.stateful_actuation import ActuatorModel
  for kind in ("dyn", "gain", "bias"):
    with pytest.raises(ValueError, match="019"):
      ActuatorModel(_user_model(kind))


def test_supported_builtin_admitted_with_evidence():
  # R6: a declared-supported combination is admitted AND its row carries
  # qualified status with test evidence.
  from mujoco_metal.stateful_actuation import ActuatorModel
  m = mujoco.MjModel.from_xml_string(
      '<mujoco><option timestep="0.002"/><worldbody>'
      '<body pos="0 0 1"><joint name="j" type="hinge" axis="0 1 0"/>'
      '<geom type="sphere" size="0.1" mass="1"/></body>'
      '</worldbody><actuator>'
      '<general joint="j" dyntype="filter" dynprm="0.05 0 0"/>'
      '</actuator></mujoco>')
  meta = ActuatorModel(m)
  assert meta.na == 1
  rows = {r.id: r for r in REQUIREMENTS}
  dyn = rows["REQ-DYN-002"]
  assert "mjtDyn.mjDYN_FILTER" in dyn.enums
  assert "mjtDyn.mjDYN_USER" not in dyn.enums
  assert dyn.tests, "qualified rows must name evidence"
  user = rows["REQ-DYN-003"]
  assert user.milestone == "019"
  assert user.implementation == Implementation.NOT_IMPLEMENTED
