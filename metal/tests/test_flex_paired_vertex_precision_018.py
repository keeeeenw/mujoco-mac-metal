# Copyright 2026 The MuJoCo Metal contributors
# Licensed under the Apache License, Version 2.0.

"""CPU/source witnesses for high/mid/tail flex vertex coordinates."""

import mujoco
import numpy as np
import os
import pytest

torch = pytest.importorskip("torch")

from mujoco_metal.flex import MetalFlex
from mujoco_metal.flex_contact import FlexContactProgram
from mujoco_metal.metal_kinematics import _prepare_host_arrays
from mujoco_metal.model import load_model


def _flex_fixture(dof):
  if dof == "direct":
    declaration = ('<flexcomp name="f" type="grid" count="2 2 2" '
                   'pos=".123456789 .023456789 .034567891" '
                   'spacing=".071234567 .083456789 .092345671" '
                   'mass="1" dim="3"><contact selfcollide="none"/>'
                   '</flexcomp>')
  else:
    declaration = (f'<flexcomp name="f" type="grid" count="2 2 2" '
                   f'pos=".123456789 .023456789 .034567891" '
                   f'spacing=".071234567 .083456789 .092345671" '
                   f'mass="1" dim="3" dof="{dof}">'
                   '<contact selfcollide="none"/></flexcomp>')
  model = mujoco.MjModel.from_xml_string(
      '<mujoco><option gravity="0 0 0"/><worldbody>'
      + declaration + '</worldbody></mujoco>')
  data = mujoco.MjData(model)
  mujoco.mj_forward(model, data)
  return model, data


def _paired_body_pose(data):
  body_pos = np.asarray(data.xpos, dtype=np.float64)
  high = body_pos.astype(np.float32)
  low = (body_pos - high.astype(np.float64)).astype(np.float32)
  tail = (body_pos - high.astype(np.float64)
          - low.astype(np.float64)).astype(np.float32)
  return {
      "body_pos": torch.as_tensor(high[None].copy()),
      "body_pos_low": torch.as_tensor(low[None].copy()),
      "body_pos_tail": torch.as_tensor(tail[None].copy()),
      "body_quat": torch.as_tensor(
          np.asarray(data.xquat, dtype=np.float32)[None].copy()),
  }


def test_model_descriptor_preserves_body_position_residual_words():
  model, _ = _flex_fixture("direct")
  descriptor = load_model(model)
  host = _prepare_host_arrays(descriptor)
  high = host["body_pos"].reshape(model.nbody, 3).astype(np.float64)
  low = host["body_pos_low"].reshape(model.nbody, 3).astype(np.float64)
  tail = host["body_pos_tail"].reshape(model.nbody, 3).astype(np.float64)
  actual = high + low + tail
  np.testing.assert_allclose(actual, np.asarray(model.body_pos, np.float64),
                             rtol=0.0, atol=0.0)
  pair = host["body_pos_pair"].reshape(-1)
  np.testing.assert_allclose(
      pair[:model.nbody * 3].reshape(model.nbody, 3).astype(np.float64)
      + pair[model.nbody * 3:2*model.nbody*3].reshape(model.nbody, 3).astype(np.float64)
      + pair[2*model.nbody*3:].reshape(model.nbody, 3).astype(np.float64),
      np.asarray(model.body_pos, np.float64), rtol=0.0, atol=0.0)


@pytest.mark.parametrize("dof", ["direct", "trilinear", "quadratic"])
def test_flex_kinematics_low_word_reconstructs_pinned_vertex_positions(dof):
  model, data = _flex_fixture(dof)
  manager = MetalFlex(model, batch_size=1, device="cpu")
  manager.update_kinematics(_paired_body_pose(data))
  reconstructed = (manager.flexvert_xpos.detach().numpy()[0].astype(np.float64)
                   + manager._flexvert_xpos_low.detach().numpy()[0].astype(np.float64)
                   + manager._flexvert_xpos_tail.detach().numpy()[0].astype(np.float64))
  pinned = np.asarray(data.flexvert_xpos, dtype=np.float64).reshape(-1, 3)
  np.testing.assert_allclose(reconstructed, pinned, rtol=0.0, atol=2e-8)
  represented_error = np.max(np.abs(reconstructed - pinned))
  high_error = np.max(np.abs(manager.flexvert_xpos.detach().numpy()[0]
                             .astype(np.float64) - pinned))
  assert represented_error <= high_error


def test_contact_triple_vertex_argument_is_preallocated_and_world_ordered():
  model, data = _flex_fixture("direct")
  program = FlexContactProgram(model, batch_size=2, device="cpu")
  high = np.stack((data.flexvert_xpos, data.flexvert_xpos + 1.0)).astype(
      np.float32).reshape(2, model.nflexvert, 3)
  low = np.stack((data.flexvert_xpos - high[0],
                  data.flexvert_xpos + 1.0 - high[1])).astype(
      np.float32).reshape(2, model.nflexvert, 3)
  tail = np.zeros_like(low)
  expected_storage = program._flexvert_xpos_pair
  returned = program._pack_paired_flexvert_positions(
      torch.as_tensor(high), torch.as_tensor(low), torch.as_tensor(tail))
  assert returned is expected_storage
  packed = returned.detach().cpu().numpy().reshape(2, 3, model.nflexvert, 3)
  np.testing.assert_array_equal(packed[:, 0], high)
  np.testing.assert_array_equal(packed[:, 1], low)
  np.testing.assert_array_equal(packed[:, 2], tail)
  vertex_words = 2 * model.nflexvert * 3
  geom_words = 2 * model.ngeom * 3
  residual = program._flexvert_xpos_low_tail.cpu().numpy()
  assert residual.shape == (2 * vertex_words + 2 * geom_words,)
  np.testing.assert_array_equal(residual[:model.nflexvert * 3], low[0].reshape(-1))
  np.testing.assert_array_equal(
      residual[model.nflexvert * 3:vertex_words], low[1].reshape(-1))
  np.testing.assert_array_equal(
      residual[vertex_words:vertex_words + model.nflexvert * 3],
      tail[0].reshape(-1))
  np.testing.assert_array_equal(
      residual[vertex_words + model.nflexvert * 3:2 * vertex_words],
      tail[1].reshape(-1))
  # Geometry low/tail occupy disjoint trailing planes in the same ABI slab.
  geom_low = torch.arange(2 * model.ngeom * 3, dtype=torch.float32).reshape(
      2, model.ngeom, 3)
  geom_tail = geom_low + 1000.0
  program._geom_pos_low_tail[0].copy_(geom_low)
  program._geom_pos_low_tail[1].copy_(geom_tail)
  np.testing.assert_array_equal(
      residual[2 * vertex_words:2 * vertex_words + geom_words],
      geom_low.numpy().reshape(-1))
  np.testing.assert_array_equal(
      residual[2 * vertex_words + geom_words:], geom_tail.numpy().reshape(-1))
  assert program._geom_pos_low_tail.untyped_storage().data_ptr() == (
      program._flexvert_xpos_low_tail.untyped_storage().data_ptr())


def test_capacity_counts_triple_vertex_argument_storage():
  from mujoco_metal.capacity import _runtime_buffer_sizes, estimate_capacity
  from mujoco_metal.smooth_metal import (
      position_context_shapes, position_context_workspace_sizes)

  model, _ = _flex_fixture("direct")
  sizes = dict(_runtime_buffer_sizes(model, 2))
  assert sizes["flex_contact.vertex_high_low_pair"] == 2 * model.nflexvert * 9
  assert sizes["flex_contact.vertex_low_tail_dispatch"] == (
      2 * model.nflexvert * 6 + 2 * model.ngeom * 6)
  assert sizes["flex_contact.zero_vertex_residual"] == (
      2 * 2 * model.nflexvert * 3)
  assert sizes["flex_contact.zero_geom_residual"] == 2 * 2 * model.ngeom * 3
  assert sizes["fk.dims"] == 61
  # Pose residuals belong to the output arena; auxiliary owns only mocap,
  # body maps, tree flags, and cache flags. Count each allocation once.
  assert sizes["fk.auxiliary"] == (
      (2 * model.nmocap * 7 if model.nmocap else 1)
      + 2 * model.nbody + 4 * max(model.ntree, 1) + 2)
  output_words = (6 * (max(2 * model.nbody * 3, 1)
                       + max(2 * model.nbody * 4, 1))
                  + 3 * (max(2 * model.ngeom * 3, 1)
                         + max(2 * model.ngeom * 4, 1)
                         + max(2 * model.ngeom * 9, 1))
                  + 3 * (max(2 * model.nsite * 3, 1)
                         + max(2 * model.nsite * 4, 1))
                  + 2 * max(2 * model.njnt * 3, 1))
  assert sizes["fk.pose_output_arena"] == output_words
  estimate = estimate_capacity(model, 2, npairs=3, nslots=5, nr=20)
  owned = dict(estimate.memory_breakdown)
  assert "model.fk.geom_pos_pair" not in owned
  assert owned["fk.pose_output_arena"] == output_words * 4
  assert owned["fk.auxiliary"] == sizes["fk.auxiliary"] * 4
  assert owned["flex_contact.vertex_high_low_pair"] == (
      2 * model.nflexvert * 9 * 4)
  assert owned["flex_contact.vertex_low_tail_dispatch"] == (
      2 * model.nflexvert * 6 + 2 * model.ngeom * 6) * 4
  assert owned["flex_contact.zero_vertex_residual"] == (
      2 * 2 * model.nflexvert * 3 * 4)
  assert owned["flex_contact.zero_geom_residual"] == (
      2 * 2 * model.ngeom * 3 * 4)
  position_sizes = position_context_workspace_sizes(model, 2)
  assert position_sizes["position_cache.poses.body_pos_tail"] == (
      2 * model.nbody * 3)
  assert position_sizes["position_cache.poses.body_pos_low"] == (
      2 * model.nbody * 3)
  assert position_context_shapes(model, 2)["poses.body_pos_tail"] == (
      2, model.nbody, 3)
  assert position_sizes["position_cache.poses.geom_pos_low"] == max(
      2 * model.ngeom * 3, 1)
  assert position_sizes["position_cache.poses.geom_pos_tail"] == max(
      2 * model.ngeom * 3, 1)
  assert position_context_shapes(model, 2)["poses.geom_pos_low"] == (
      2, model.ngeom, 3)
  assert position_context_shapes(model, 2)["poses.geom_pos_tail"] == (
      2, model.ngeom, 3)


def _native_fk_fixture(kind):
  if kind == "direct":
    dof = ""
    count = "2 2 2"
    elasticity = ""
  else:
    shell = kind.endswith("-shell")
    dof = "quadratic" if kind.startswith("quadratic") else "trilinear"
    count = "3 3 3" if dof == "quadratic" else "2 2 2"
    elasticity = ('<elasticity young="100" poisson=".2" thickness=".01" '
                  + ('elastic2d="bend"' if shell else
                     'elastic2d="stretch"') + '/>')
  dof_attr = f' dof="{dof}"' if dof else ""
  model = mujoco.MjModel.from_xml_string(f"""
    <mujoco><option gravity="0 0 0"/><worldbody>
      <flexcomp name="f" type="grid" count="{count}"
                pos=".123456789 .023456789 .034567891"
                spacing=".071234567 .083456789 .092345671"
                mass="1" dim="3"{dof_attr}>
        <contact selfcollide="none"/>
        {elasticity}
      </flexcomp>
    </worldbody></mujoco>
  """)
  oracle = mujoco.MjData(model)
  # Match the actual float32 state sent to MPS while retaining compiled
  # model constants at their pinned mjtNum precision in the CPU oracle.
  qpos = (np.asarray(model.qpos0, dtype=np.float64)
          + np.linspace(-.003, .004, int(model.nq), dtype=np.float64))
  qpos = qpos.astype(np.float32).astype(np.float64)
  oracle.qpos[:] = qpos
  mujoco.mj_forward(model, oracle)
  return model, oracle, qpos


@pytest.mark.gpu
@pytest.mark.parametrize("kind", [
    "direct", "trilinear", "quadratic", "trilinear-shell",
    "quadratic-shell"])
@pytest.mark.skipif(os.getenv("MUJOCO_METAL_RUN_GPU") != "1",
                    reason="paired FK producer qualification requires GPU opt-in")
def test_native_fk_preserves_source_vertex_residuals_through_flex_paths(kind):
  if not torch.backends.mps.is_available():
    pytest.skip("paired FK producer requires MPS")
  from mujoco_metal.flex import MetalFlex
  from mujoco_metal.metal_kinematics import MetalKinematics
  from mujoco_metal.model import load_model

  model, oracle, qpos = _native_fk_fixture(kind)
  descriptor = load_model(model)
  fk = MetalKinematics(descriptor, batch_size=1)
  qpos_device = torch.as_tensor(qpos[None].astype(np.float32).copy(),
                                dtype=torch.float32, device="mps")
  poses = fk.run_device(qpos_device)
  manager = MetalFlex(model, batch_size=1, device="mps")
  manager.update_kinematics(poses)
  high = manager.flexvert_xpos.detach().cpu().numpy()[0].astype(np.float64)
  low = manager._flexvert_xpos_low.detach().cpu().numpy()[0].astype(np.float64)
  tail = manager._flexvert_xpos_tail.detach().cpu().numpy()[0].astype(np.float64)
  represented = high + low + tail
  expected = np.asarray(oracle.flexvert_xpos, dtype=np.float64).reshape(-1, 3)
  np.testing.assert_allclose(represented, expected, rtol=0.0, atol=6e-6)
  represented_error = np.max(np.abs(represented - expected))
  high_error = np.max(np.abs(high - expected))
  assert represented_error <= high_error


@pytest.mark.gpu
@pytest.mark.skipif(os.getenv("MUJOCO_METAL_RUN_GPU") != "1",
                    reason="paired CCD vertex transport requires GPU opt-in")
def test_native_common_ccd_consumes_paired_direct_simplex_vertices_B2():
  if not torch.backends.mps.is_available():
    pytest.skip("paired CCD vertex transport requires MPS")
  from mujoco_metal.flex_contact import FlexContactProgram, lower_flex_contacts
  from test_flex_triangle_pair_common_ccd_018 import (
      _direct_simplex_pair_fixture, _expected_pair_contacts)

  model, data = _direct_simplex_pair_fixture(2, 3)
  descriptor = lower_flex_contacts(model)
  expected = _expected_pair_contacts(model, data, descriptor)
  source = np.asarray(data.flexvert_xpos, dtype=np.float64).reshape(-1, 3)
  high = source.astype(np.float32)
  low = (source - high.astype(np.float64)).astype(np.float32)
  tail = (source - high.astype(np.float64)
          - low.astype(np.float64)).astype(np.float32)
  vertices_high = np.stack((high, high.copy()))
  vertices_low = np.stack((low, low.copy()))
  vertices_tail = np.stack((tail, tail.copy()))
  second = int(model.flex_vertadr[1])
  second_count = int(model.flex_vertnum[1])
  vertices_high[1, second:second + second_count, 2] += np.float32(.3)

  def mps(value):
    return torch.as_tensor(np.asarray(value, np.float32).copy(),
                           dtype=torch.float32, device="mps").contiguous()

  program = FlexContactProgram(model, batch_size=2, device="mps")
  assert program._narrowphase_admitted
  assert program._has_native_flex_pair_candidates
  result = program.run_device(
      mps(vertices_high),
      mps(np.zeros((2, model.ngeom, 3), np.float32)),
      mps(np.tile(np.asarray([1, 0, 0, 0], np.float32),
                  (2, model.ngeom, 1))),
      flexvert_xpos_low=mps(vertices_low),
      flexvert_xpos_tail=mps(vertices_tail))
  active = result["active"].cpu().numpy()
  np.testing.assert_array_equal(np.flatnonzero(active[0]),
                                np.asarray(sorted(expected)))
  assert not np.any(active[1])
  distance = result["dist"].cpu().numpy()
  position = result["pos"].cpu().numpy()
  normal = result["normal"].cpu().numpy()
  for slot, contact in expected.items():
    np.testing.assert_allclose(distance[0, slot], contact.dist,
                               rtol=0.0, atol=6e-6)
    np.testing.assert_allclose(position[0, slot], contact.pos,
                               rtol=0.0, atol=6e-6)
    np.testing.assert_allclose(normal[0, slot], contact.frame[:3],
                               rtol=0.0, atol=6e-5)


def test_flex_position_capacity_exact_limit_and_preallocation_overflow():
  from types import SimpleNamespace
  from mujoco_metal.flex_contact import (
      _INT32_MAX, _flex_position_workspace_elements,
      _validate_contact_program_capacity,
      _validate_flex_position_workspace_capacity)

  exact_nvertex = _INT32_MAX // 9
  exact = _validate_flex_position_workspace_capacity(1, exact_nvertex, 0)
  assert exact["flex_vertex_pair"] == 9 * exact_nvertex
  assert exact["flex_vertex_pair"] <= _INT32_MAX
  # Geometry residual dispatch uses two planes, three coordinates each.
  exact_ngeom = _INT32_MAX // 6
  geom_exact = _flex_position_workspace_elements(1, 0, exact_ngeom)
  assert geom_exact["geom_low_tail"] == 6 * exact_ngeom
  assert geom_exact["geom_low_tail"] <= _INT32_MAX

  # Both views fit signed 32-bit indexing individually, but the actual
  # shared flat dispatch slab does not. This must fail before allocation.
  vertex_near_limit = _INT32_MAX // 9
  geom_near_limit = _INT32_MAX // 6
  parts = _flex_position_workspace_elements(
      1, vertex_near_limit, geom_near_limit)
  assert parts["flex_vertex_pair"] <= _INT32_MAX
  assert parts["flex_vertex_low_tail"] <= _INT32_MAX
  assert parts["geom_low_tail"] <= _INT32_MAX
  assert parts["vertex_geom_low_tail_dispatch"] > _INT32_MAX
  with pytest.raises(ValueError,
                     match="vertex_geom_low_tail_dispatch.*signed 32-bit"):
    _validate_flex_position_workspace_capacity(
        1, vertex_near_limit, geom_near_limit)

  descriptor = SimpleNamespace(
      slot_count=1, nv=1, feature_capacity=1, row_capacity=1,
      link_capacity=1, max_links_per_candidate=1)
  model = SimpleNamespace(
      nflex=1, nflexvert=exact_nvertex + 1, ngeom=0, nbody=1,
      opt=SimpleNamespace(ccd_iterations=1))
  with pytest.raises(ValueError, match="flex_vertex_pair.*signed 32-bit"):
    _validate_contact_program_capacity(model, descriptor, 1)
  model.nflexvert = exact_nvertex
  assert _validate_contact_program_capacity(model, descriptor, 1) == 1
