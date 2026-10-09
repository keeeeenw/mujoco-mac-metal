# Copyright 2026 The MuJoCo Metal contributors
# Licensed under the Apache License, Version 2.0.
"""One staged source SDF search shared by simulation and geometry queries."""

from dataclasses import dataclass


@dataclass(frozen=True)
class StagedSdfKernels:
  seed_init: object
  phase_reset: object
  descent_prepare: object
  line_search: object
  publish_normal: object
  publish_contact: object
  finalize: object

  @classmethod
  def from_library(cls, library):
    return cls(library.contact_sdf_seed_init, library.contact_sdf_phase_reset,
               library.contact_sdf_descent_prepare, library.contact_sdf_line_search,
               library.contact_sdf_publish_normal, library.contact_sdf_publish_contact,
               library.contact_finalize_sdf)


def dispatch_staged_sdf(kernels, torch, poses, geometry, pair_dims,
                        logical_map, w, stage_dims, *, batch_size, npairs,
                        seed_count, seed_tile, max_iterations):
  """Publish complete source-ordered minima using bounded per-seed scratch.

  Caller owns all persistent buffers and admitted geometry. The logical map
  and pair header carry GPU world selection; no host readback determines a
  numerical branch. Each dispatch retains the production search order.
  """
  b = batch_size
  seed_capacity = seed_tile
  seed_starts = (range(0, seed_count, seed_tile)
                 if seed_count else (0,))
  for seed_start in seed_starts:
    tile_count = min(seed_tile,
                     max(0, seed_count - seed_start))
    thread_count = b * npairs * max(tile_count, 1)
    def set_sdf_stage(mode, step):
      stage_dims.copy_(torch.tensor(
          [tile_count, seed_start, seed_capacity, mode, step],
          dtype=torch.int32, device=stage_dims.device))
    if tile_count == 0:
      set_sdf_stage(0, 0)
      kernels.finalize(
          geometry["geom_type"], geometry["pair_geoms"],
          pair_dims,
          logical_map,
          w["contact_sdf_seed_records"], w["contact_sdf_seed_valid"],
          w["contact_pair_count"], w["contact_pair_records"],
          stage_dims,
          threads=(b * npairs,), group_size=(1,))
      continue
    set_sdf_stage(2, 0)
    kernels.seed_init(
        poses["geom_pos"], poses["geom_quat"],
        geometry["geom_size"], geometry["geom_type"],
        geometry["pair_geoms"],
        pair_dims,
        logical_map,
        geometry["mesh_hull"], geometry["mesh_hull_info"],
        w["contact_sdf_seed_valid"],
        w["contact_sdf_seed_state"], w["contact_sdf_seed_ctrl"],
        stage_dims,
        poses["geom_pos_low"], poses["geom_pos_tail"],
        poses["geom_quat_low"], poses["geom_quat_tail"],
        poses["geom_xmat"], poses["geom_xmat_low"],
        poses["geom_xmat_tail"],
        threads=(thread_count,), group_size=(1,))
    for step in range(max_iterations):
      set_sdf_stage(2, step)
      kernels.descent_prepare(
          poses["geom_pos"], poses["geom_quat"],
          geometry["geom_size"], geometry["geom_type"],
          geometry["pair_geoms"],
          pair_dims,
          logical_map,
          geometry["mesh_hull"], geometry["mesh_hull_info"],
          w["contact_sdf_seed_valid"], w["contact_sdf_seed_state"],
          w["contact_sdf_seed_ctrl"], stage_dims,
          threads=(thread_count,), group_size=(1,))
      kernels.line_search(
          poses["geom_pos"], poses["geom_quat"],
          geometry["geom_size"], geometry["geom_type"],
          geometry["pair_geoms"],
          pair_dims,
          logical_map,
          geometry["mesh_hull"], geometry["mesh_hull_info"],
          w["contact_sdf_seed_valid"], w["contact_sdf_seed_state"],
          w["contact_sdf_seed_ctrl"], stage_dims,
          threads=(thread_count,), group_size=(1,))
    set_sdf_stage(0, 0)
    kernels.phase_reset(
        poses["geom_pos"], poses["geom_quat"],
        geometry["geom_size"], geometry["geom_type"],
        geometry["pair_geoms"],
        pair_dims,
        logical_map,
        w["contact_sdf_seed_valid"], w["contact_sdf_seed_state"],
        w["contact_sdf_seed_ctrl"], stage_dims,
        threads=(thread_count,), group_size=(1,))
    kernels.descent_prepare(
        poses["geom_pos"], poses["geom_quat"],
        geometry["geom_size"], geometry["geom_type"],
        geometry["pair_geoms"],
        pair_dims,
        logical_map,
        geometry["mesh_hull"], geometry["mesh_hull_info"],
        w["contact_sdf_seed_valid"], w["contact_sdf_seed_state"],
        w["contact_sdf_seed_ctrl"], stage_dims,
        threads=(thread_count,), group_size=(1,))
    kernels.line_search(
        poses["geom_pos"], poses["geom_quat"],
        geometry["geom_size"], geometry["geom_type"],
        geometry["pair_geoms"],
        pair_dims,
        logical_map,
        geometry["mesh_hull"], geometry["mesh_hull_info"],
        w["contact_sdf_seed_valid"], w["contact_sdf_seed_state"],
        w["contact_sdf_seed_ctrl"], stage_dims,
        threads=(thread_count,), group_size=(1,))
    kernels.publish_normal(
        poses["geom_pos"], poses["geom_quat"],
        geometry["geom_size"], geometry["geom_type"],
        geometry["pair_geoms"],
        pair_dims,
        logical_map,
        geometry["mesh_hull"], geometry["mesh_hull_info"],
        w["contact_sdf_seed_valid"], w["contact_sdf_seed_state"],
        stage_dims,
        threads=(thread_count,), group_size=(1,))
    kernels.publish_contact(
        poses["geom_pos"], poses["geom_quat"], geometry["geom_type"],
        geometry["pair_geoms"],
        pair_dims,
        logical_map,
        w["contact_sdf_seed_records"], w["contact_sdf_seed_valid"],
        w["contact_sdf_seed_state"], stage_dims,
        threads=(thread_count,), group_size=(1,))
    kernels.finalize(
        geometry["geom_type"], geometry["pair_geoms"],
        pair_dims,
        logical_map,
        w["contact_sdf_seed_records"], w["contact_sdf_seed_valid"],
        w["contact_pair_count"], w["contact_pair_records"],
        stage_dims,
        threads=(b * npairs,), group_size=(1,))
