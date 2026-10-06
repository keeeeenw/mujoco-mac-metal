# Copyright 2026 The MuJoCo Metal contributors
# Licensed under the Apache License, Version 2.0.
"""Opt-in capture of the public FPS path before its unchanged identity gate.

This observer is diagnostic only: it records actual public FK/contact/filter
buffers and still asserts the original full-source CPU identity set.
"""
import json
import os
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")


@pytest.mark.gpu
@pytest.mark.skipif(os.getenv("MUJOCO_METAL_RUN_GPU") != "1",
                    reason="native FPS observer requires explicit GPU opt-in")
@pytest.mark.parametrize("midphase", [False, True])
def test_native_public_fps_first_divergence_observer(midphase):
  if not torch.backends.mps.is_available():
    pytest.skip("public FPS observer requires MPS")
  out_root = os.getenv("MUJOCO_METAL_FPS_OBSERVER_DIR")
  if not out_root:
    pytest.skip("set MUJOCO_METAL_FPS_OBSERVER_DIR to capture diagnostics")

  from mujoco_metal.simulation import MetalSimulation
  from test_flex_fps_input_contract_018 import (
      _cpu_collision, _cpu_collision_native_plane_inputs, _fixture,
      _identities, _native_geom_inputs, _native_selector_ids,
      _refresh_source_bounds)

  model, reference = _fixture(midphase)
  full_vertices = np.asarray(reference.flexvert_xpos, np.float64).reshape(-1, 3)
  full_data, expected = _cpu_collision(model, reference, full_vertices)
  raw_vertices = full_vertices.astype(np.float32).astype(np.float64)
  represented_data, _ = _cpu_collision(model, reference, raw_vertices)
  geom_pos_cpu, geom_quat_cpu = _native_geom_inputs(represented_data)
  _, represented_expected = _cpu_collision_native_plane_inputs(
      model, reference, raw_vertices, geom_pos_cpu, geom_quat_cpu)

  qpos = np.asarray(reference.qpos, np.float32)[None].copy()
  qvel = np.asarray(reference.qvel, np.float32)[None].copy()
  sim = MetalSimulation(model, batch_size=1, qpos=qpos, qvel=qvel,
                        profile="integrated_scalable_v1")
  sim.prepare_forward_position(sim._state._qpos)
  program = sim._flex._contact_program
  assert program is not None and program._narrowphase_admitted
  bundle = sim._coupled_constraints._flex_contact_current
  assert bundle is not None
  result = bundle["contact_result"]
  descriptor = sim._coupled_constraints.descriptor.flex_contact_descriptor
  actual = _native_selector_ids(program, descriptor, result)

  def host(value):
    if value is None:
      return None
    if isinstance(value, torch.Tensor):
      return value.detach().cpu().numpy().copy()
    return np.asarray(value).copy()

  arrays = {
      "full_cpu_flexvert_xpos": full_vertices,
      "full_cpu_geom_xpos": np.asarray(full_data.geom_xpos).copy(),
      "full_cpu_geom_xmat": np.asarray(full_data.geom_xmat).copy(),
      "public_flexvert_xpos_high": host(sim._flex.flexvert_xpos),
      "public_flexvert_xpos_low": host(sim._flex._flexvert_xpos_low),
      "public_flexvert_xpos_tail": host(sim._flex._flexvert_xpos_tail),
      "public_flexvert_xpos_device_pair": host(
          program._flexvert_xpos_pair),
      "public_geom_pos_low_tail": host(program._geom_pos_low_tail),
      "public_geom_size_high": host(program._geom_size),
      "public_geom_size_midlow": host(program._geom_size_midlow),
      "public_radius_high": host(program._radius1),
      "public_radius_midlow": host(program._radius1_midlow),
      "public_margin_gap_high": host(program._margin_gap),
      "public_margin_gap_midlow": host(program._margin_gap_midlow),
      "public_ccd_tolerance_high": host(program._ccd_tolerance),
      "public_ccd_tolerance_midlow": host(program._ccd_tolerance_midlow),
      "public_qpos": host(sim._state._qpos),
      "candidate_geom": host(program._geom),
      "candidate_elem1": host(program._elem1),
      "candidate_flex1": host(program._flex1),
      "candidate_group": host(program._filter_group),
      "candidate_filterable": host(program._filterable),
      "candidate_filter_mode": host(program._filter_mode),
      "candidate_raw_active": host(result.get("raw_active")),
      "candidate_active": host(result.get("active")),
      "candidate_distance_high": host(result.get("dist")),
      "candidate_position_high": host(result.get("pos")),
      "candidate_position_low_tail": host(program._contact_pos_low_tail),
      "candidate_selection_position": host(program._selection_pos),
      "candidate_permutation_after_filter": host(program._filter_permutation),
      "candidate_selected_position_after_filter": host(
          program._filter_selected_position),
      "candidate_min_distance_words_after_filter": host(
          program._filter_min_distance_words),
      "candidate_narrowphase_status": host(result.get("narrowphase_status")),
      "candidate_ccd_trace": host(result.get("ccd_trace")),
  }
  fk = sim._smooth._fk
  poses = fk.run_device(
      sim._state._qpos, getattr(sim._state, "_mpos", None),
      getattr(sim._state, "_mquat", None))
  for name in ("body_quat", "geom_quat", "geom_pos_pair"):
    arrays["compiled_model_" + name] = host(fk._arrays[name])
  for name in ("geom_pos", "geom_pos_low", "geom_pos_tail", "geom_quat"):
    arrays["fk_" + name] = host(poses.get(name))
  raw_active = np.asarray(arrays.get("candidate_raw_active", []))
  if raw_active.size:
    raw_mask = raw_active[0].astype(bool)
    raw_slots = np.flatnonzero(raw_mask &
                               (np.asarray(descriptor.geom) >= 0) &
                               (np.asarray(descriptor.vert1) >= 0))
    raw_ids = {(int(descriptor.geom[i]), int(descriptor.vert1[i]))
               for i in raw_slots}
  else:
    raw_ids = set()

  out = Path(out_root)
  out.mkdir(parents=True, exist_ok=True)
  suffix = "midphase-on" if midphase else "midphase-off"
  np.savez_compressed(out / f"fps-{suffix}.npz", **{
      key: value for key, value in arrays.items() if value is not None
  })
  payload = {
      "midphase": bool(midphase),
      "slot_count": int(descriptor.slot_count),
      "actual_ids": sorted([list(x) for x in actual]),
      "raw_candidate_ids": sorted([list(x) for x in raw_ids]),
      "expected_full_source_ids": sorted([list(x) for x in expected]),
      "expected_represented_cpu_ids": sorted([list(x) for x in represented_expected]),
      "actual_count": len(actual),
      "full_source_count": len(expected),
      "represented_cpu_count": len(represented_expected),
      "array_shapes": {key: list(value.shape) for key, value in arrays.items()
                        if value is not None},
      "array_dtypes": {key: str(value.dtype) for key, value in arrays.items()
                        if value is not None},
  }
  (out / f"fps-{suffix}.json").write_text(json.dumps(payload, indent=2) + "\n")

  # Preserve the production acceptance oracle; diagnostic capture does not
  # alter contact identity expectations or thresholds.
  assert len(expected) == (50 if midphase else 100)
  assert actual == expected
