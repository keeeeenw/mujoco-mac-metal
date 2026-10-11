# Copyright 2026 The MuJoCo Metal contributors
# Licensed under the Apache License, Version 2.0.
"""Source counterparts must survive callable drift, independently of GPU tests."""
from copy import deepcopy

import pytest

from mujoco_metal.binding_inventory import (
    counterpart_source_drift, pinned_binding_surface)


def test_all_declared_counterpart_sources_resolve():
  surface = pinned_binding_surface()
  assert counterpart_source_drift() == ()
  assert surface['counterpart_source_drift'] == []
  assert surface['feature_mapping_complete'] is False
  assert surface['gpu_qualified'] is False


@pytest.mark.parametrize('name', ['mj_checkPos', 'mj_checkVel', 'mj_checkAcc'])
def test_implemented_state_check_mapping_names_real_backend_function(name):
  from mujoco_metal import state_checks
  row = pinned_binding_surface()['python_function_map'][name]
  assert row['counterpart_kind'] == 'internal_or_prepared_pipeline'
  assert row['counterpart'] == f'mujoco_metal/state_checks.py:{name}'
  assert callable(getattr(state_checks, name))
  assert 'unimplemented' not in row['execution']


def test_callable_drift_is_rejected_when_the_module_still_exists():
  surface = deepcopy(pinned_binding_surface())
  surface['python_function_map']['mj_checkAcc']['counterpart'] = \
      'mujoco_metal/state_checks.py:renamed_checkAcc'
  assert counterpart_source_drift(surface) == (
      'Missing counterpart callable: mj_checkAcc',)
  surface['python_function_map']['mj_checkAcc']['counterpart'] = \
      'mujoco_metal/renamed_state_checks.py:mj_checkAcc'
  assert counterpart_source_drift(surface) == (
      'Missing counterpart source: mj_checkAcc',)


def test_class_method_and_module_only_pipeline_contracts(tmp_path):
  (tmp_path / 'backend.py').write_text(
      'class Simulator:\n  def step(self):\n    pass\n')
  surface = {'python_function_map': {
      'step': {'counterpart': 'backend.py:Simulator.step'},
      'pipeline': {'counterpart': 'backend.py'},
  }}
  assert counterpart_source_drift(surface, source_root=tmp_path) == ()
  (tmp_path / 'backend.py').write_text(
      'class Simulator:\n  def renamed_step(self):\n    pass\n')
  assert counterpart_source_drift(surface, source_root=tmp_path) == (
      'Missing counterpart callable: step',)


def test_source_mapping_cannot_escape_package_root(tmp_path):
  surface = {'python_function_map': {
      'bad': {'counterpart': '../outside.py:step'}}}
  assert counterpart_source_drift(surface, source_root=tmp_path) == (
      'Missing counterpart source: bad',)


def test_all_requirement_backend_anchors_resolve():
  from mujoco_metal.binding_inventory import requirement_source_drift
  assert requirement_source_drift() == ()


def test_requirement_anchor_drift_cannot_hide_behind_existing_module():
  from dataclasses import replace
  from mujoco_metal.binding_inventory import requirement_source_drift
  from mujoco_metal.registry import REQUIREMENTS
  row = next(row for row in REQUIREMENTS if row.id == 'REQ-FLEX-001')
  stale = replace(row, source_ref='mujoco_metal/flex.py:FlexProgram')
  assert requirement_source_drift([stale]) == (
      'Missing counterpart callable: REQ-FLEX-001',)
  missing = replace(row, source_ref='mujoco_metal/shaders/missing_material.metal')
  assert requirement_source_drift([missing]) == (
      'Missing counterpart source: REQ-FLEX-001',)


def test_all_backend_fragments_checked_but_upstream_refs_not_reinterpreted():
  from dataclasses import replace
  from mujoco_metal.binding_inventory import requirement_source_drift
  from mujoco_metal.registry import REQUIREMENTS
  row = next(row for row in REQUIREMENTS if row.id == 'REQ-FLEX-001')
  combined = replace(row, source_ref='mujoco_metal/flex.py:MetalFlex; '
      'engine/engine_passive.c:mj_flexPassive; '
      'mujoco_metal/flex.py:FlexProgram')
  assert requirement_source_drift([combined]) == (
      'Missing counterpart callable: REQ-FLEX-001[1]',)


def test_active_material_requirement_retains_all_physics_witnesses():
  from mujoco_metal.registry import REQUIREMENTS, Qualification
  row = next(row for row in REQUIREMENTS if row.id == 'REQ-FLEX-001')
  assert row.source_ref == 'mujoco_metal/flex.py:MetalFlex'
  assert row.qualification == Qualification.UNQUALIFIED
  assert set(row.tests) >= {
      'test_flex_material_parity.py', 'test_flex_integrator_lifecycle.py',
      'test_active_material_invariants_018.py', 'test_active_material_energy_018.py'}


def test_every_requirement_test_reference_is_retained():
  from pathlib import Path
  from mujoco_metal.registry import REQUIREMENTS
  root = Path(__file__).parent
  missing = [(row.id, name) for row in REQUIREMENTS for name in row.tests
             if not (root / name).is_file()]
  assert missing == []
