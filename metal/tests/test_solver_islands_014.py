# Copyright 2026 The MuJoCo Metal contributors
# Licensed under the Apache License, Version 2.0.
"""CPU reference contracts for solver islands and active row maps."""

import os

import numpy as np
import pytest
import mujoco

from mujoco_metal.solver_islands import (
    build_solver_islands,
    lower_solver_row_metadata,
    pack_solver_dimension_metadata,
    pgs_island_sweep_orders,
)


def _source_row_metadata(model, data):
  """Lower actual MuJoCo row shortcuts for the CPU partition oracle."""
  nefc = int(data.nefc)
  jacobian = np.asarray(data.efc_J).reshape(nefc, model.nv).copy()
  links = np.full((nefc, 1, 2), -1, dtype=np.int32)
  groups = np.full(nefc, -1, dtype=np.int32)
  flex_exempt = np.zeros(nefc, dtype=bool)
  max_id = max(int(model.nconmax), int(model.neq), int(model.ntendon),
               int(model.njnt), int(model.nv), 1) + 1
  for row in range(nefc):
    kind, ident = int(data.efc_type[row]), int(data.efc_id[row])
    groups[row] = kind * max_id + ident
    if kind in (int(mujoco.mjtConstraint.mjCNSTR_CONTACT_FRICTIONLESS),
                int(mujoco.mjtConstraint.mjCNSTR_CONTACT_PYRAMIDAL),
                int(mujoco.mjtConstraint.mjCNSTR_CONTACT_ELLIPTIC)):
      contact = data.contact[ident]
      trees = []
      for geom in contact.geom:
        if int(geom) >= 0:
          body = int(model.geom_bodyid[int(geom)])
          trees.append(int(model.body_treeid[body]))
        else:
          trees.append(-1)
      links[row, 0] = trees
    elif kind == int(mujoco.mjtConstraint.mjCNSTR_EQUALITY):
      eq = int(model.eq_type[ident])
      flex_types = (int(mujoco.mjtEq.mjEQ_FLEX),
                    int(mujoco.mjtEq.mjEQ_FLEXVERT),
                    int(mujoco.mjtEq.mjEQ_FLEXSTRAIN))
      flex_exempt[row] = eq in flex_types
      if eq in (int(mujoco.mjtEq.mjEQ_CONNECT), int(mujoco.mjtEq.mjEQ_WELD)):
        b1, b2 = int(model.eq_obj1id[ident]), int(model.eq_obj2id[ident])
        if int(model.eq_objtype[ident]) == int(mujoco.mjtObj.mjOBJ_SITE):
          b1, b2 = int(model.site_bodyid[b1]), int(model.site_bodyid[b2])
        links[row, 0] = [int(model.body_treeid[b1]), int(model.body_treeid[b2])]
  return jacobian, links, groups, flex_exempt


def _assert_partition_matches_mujoco(model, data):
  J, links, groups, exempt = _source_row_metadata(model, data)
  result = build_solver_islands(
      np.asarray(model.dof_treeid, dtype=np.int32), J,
      np.ones(data.nefc, dtype=bool), row_links=links, group_key=groups,
      group_exempt=exempt,
      body_tree=np.asarray(model.body_treeid, dtype=np.int32))
  # Island integer IDs may be numbered by a different stable traversal, but
  # equivalence classes and constrained/unconstrained DOFs must match exactly.
  expected_dof = np.asarray(data.dof_island, dtype=np.int32)
  np.testing.assert_array_equal(result.dof_island >= 0, expected_dof >= 0)
  for a in range(model.nv):
    for b in range(model.nv):
      if expected_dof[a] >= 0 and expected_dof[b] >= 0:
        assert (result.dof_island[a] == result.dof_island[b]) == (
            expected_dof[a] == expected_dof[b])
  expected_row = np.asarray(data.efc_island[:data.nefc], dtype=np.int32)
  np.testing.assert_array_equal(result.row_island >= 0, expected_row >= 0)
  for a in range(data.nefc):
    for b in range(data.nefc):
      assert (result.row_island[a] == result.row_island[b]) == (
          expected_row[a] == expected_row[b])
  return result


def test_structural_zero_contact_rows_join_tree_islands_and_pack_rows():
  dof_tree = np.asarray([0, 0, 1, 1, 2], dtype=np.int32)
  jacobian = np.zeros((5, 5), dtype=np.float32)
  jacobian[0, 0] = 1.0
  jacobian[2, 4] = 1.0
  # The contact row has a zero Jacobian on this frame but still joins the
  # two body trees structurally, as the pinned contact shortcut requires.
  links = np.full((5, 1, 2), -1, dtype=np.int32)
  links[1, 0] = [0, 1]
  links[4, 0, 0] = 2
  active = np.asarray([1, 1, 1, 0, 1], dtype=np.int32)
  body_endpoints = np.full((5, 2), -1, dtype=np.int32)
  body_endpoints[1] = [1, 3]
  tendon_trees = np.full((5, 2), -1, dtype=np.int32)
  tendon_trees[4, 0] = 2
  result = build_solver_islands(
      dof_tree, jacobian, active, row_links=links,
      midpoint_body_ids=body_endpoints, midpoint_tree_ids=tendon_trees,
      body_tree=np.asarray([-1, 0, 0, 1, 2], dtype=np.int32))

  np.testing.assert_array_equal(result.dof_island, [0, 0, 0, 0, 1])
  np.testing.assert_array_equal(result.row_island, [0, 0, 1, -1, 1])
  np.testing.assert_array_equal(result.row_order, [0, 1, 2, 4, -1])
  np.testing.assert_array_equal(result.row_inverse, [0, 1, 2, -1, 3])
  np.testing.assert_array_equal(result.island_offsets, [0, 2, 4])
  np.testing.assert_array_equal(result.island_tree_count, [2, 1])
  np.testing.assert_array_equal(
      result.midpoint_blocked_tree, [False, False, True])
  np.testing.assert_array_equal(
      result.midpoint_blocked_body, [False, True, False, True, False])


def test_numerical_jacobian_connects_trees_without_structural_metadata():
  dof_tree = np.asarray([0, 1, 2], dtype=np.int32)
  jacobian = np.asarray([[1.0, 0.0, -2.0], [0.0, 1.0, 1.0]], dtype=np.float32)
  result = build_solver_islands(dof_tree, jacobian, [True, True])
  np.testing.assert_array_equal(result.dof_island, [0, 0, 0])
  np.testing.assert_array_equal(result.row_island, [0, 0])
  np.testing.assert_array_equal(result.row_order, [0, 1])
  np.testing.assert_array_equal(result.island_tree_count, [3])


def test_same_group_rows_join_but_exempt_family_does_not():
  dof_tree = np.arange(4, dtype=np.int32)
  jacobian = np.eye(4, dtype=np.float32)
  group = np.asarray([7, 7, 8, 8], dtype=np.int32)
  exempt = np.asarray([False, False, False, True])
  result = build_solver_islands(
      dof_tree, jacobian, np.ones(4, dtype=bool),
      group_key=group, group_exempt=exempt)
  # MuJoCo reuses the first same-type/ID row's tree iterator. The second
  # row does not union its otherwise-disconnected numerical J tree.
  np.testing.assert_array_equal(result.dof_island, [0, -1, 1, 2])
  np.testing.assert_array_equal(result.row_island, [0, 0, 1, 2])
  np.testing.assert_array_equal(result.row_order, [0, 1, 2, 3])
  np.testing.assert_array_equal(result.island_tree_count, [1, 1, 1])


def test_pgs_reseeds_pcg32_for_each_island_and_keeps_elliptic_blocks_whole():
  # solPGS_island initializes PCG32 inside each island call. Equal-sized
  # islands therefore receive the same first permutation; the stream is
  # persistent across sweeps within each island.
  order = np.asarray([0, 1, 2, 3, 4, 5, 6, 7], dtype=np.int32)
  offsets = np.asarray([0, 4, 8], dtype=np.int32)
  actual = pgs_island_sweep_orders(
      order, offsets, 2, elliptic_blocks=[(1, 3), (5, 3)])
  assert actual == (
      ((1, 0), (0, 1)),
      ((5, 4), (4, 5)),
  )


def test_lowered_row_metadata_preserves_site_equality_and_contact_identity():
  from mujoco_metal.coupled_constraints import lower_coupled_constraints

  model = mujoco.MjModel.from_xml_string("""<mujoco>
    <option gravity="0 0 -9.81" jacobian="dense"/>
    <worldbody>
      <geom type="plane" size="2 2 .1"/>
      <body name="a" pos="0 0 .09"><freejoint/>
        <geom type="sphere" size=".1" mass="1"/><site name="sa"/>
      </body>
      <body name="b" pos="1 0 .09"><freejoint/>
        <geom type="sphere" size=".1" mass="1"/><site name="sb"/>
      </body>
    </worldbody>
    <equality><connect site1="sa" site2="sb"/></equality>
  </mujoco>""")
  descriptor = lower_coupled_constraints(model)
  meta = lower_solver_row_metadata(model, descriptor)
  assert meta.shape == (descriptor.nr, 10)
  eq_type = int(mujoco.mjtConstraint.mjCNSTR_EQUALITY)
  eq_start = int(descriptor.eq_rowadr[0])
  eq_stop = eq_start + int(descriptor.eq_rownum[0])
  a, b = int(model.body("a").id), int(model.body("b").id)
  assert np.all(meta[eq_start:eq_stop, 0] == eq_type)
  assert np.all(meta[eq_start:eq_stop, 1] == 0)
  assert meta[eq_start, 5] == 1
  np.testing.assert_array_equal(meta[eq_start, 6:8], [a, b])
  np.testing.assert_array_equal(
      meta[eq_start, 2:4],
      [int(model.body_treeid[a]), int(model.body_treeid[b])])
  contact_type = int(mujoco.mjtConstraint.mjCNSTR_CONTACT_PYRAMIDAL)
  contact_rows = np.flatnonzero(meta[:, 0] == contact_type)
  assert contact_rows.size > 0
  ids, counts = np.unique(meta[contact_rows, 1], return_counts=True)
  np.testing.assert_array_equal(counts, [4, 4, 4])
  for ident in ids:
    rows = contact_rows[meta[contact_rows, 1] == ident]
    assert np.all(meta[rows, 2:4] == meta[rows[0], 2:4])


def test_interpolated_flex_contact_uses_compiled_row_support_not_vertex_body():
  """Interpolated flex vertices have no single body-tree owner."""
  from mujoco_metal.capacity import CapacityLimits
  from mujoco_metal.coupled_constraints import lower_coupled_constraints
  from mujoco_metal.flex_contact import _KIND_PLANE_VERTEX

  model = mujoco.MjModel.from_xml_string("""<mujoco>
    <option gravity="0 0 -9.81" jacobian="dense"/>
    <worldbody>
      <geom name="floor" type="plane" size="0 0 .1"
            contype="0" conaffinity="1" condim="1"/>
      <flexcomp name="interp" type="grid" count="3 3 3"
                pos="0 0 -.02" spacing=".1 .1 .1" mass="1" dim="3"
                dof="trilinear">
        <contact contype="1" conaffinity="0" selfcollide="none"
                 condim="1"/>
        <elasticity young="100" poisson=".2"/>
      </flexcomp>
    </worldbody>
  </mujoco>""")
  data = mujoco.MjData(model)
  mujoco.mj_forward(model, data)
  descriptor = lower_coupled_constraints(
      model, limits=CapacityLimits(max_nv=32, max_rows=128))
  metadata = lower_solver_row_metadata(model, descriptor)
  flex = descriptor.flex_contact_descriptor
  assert flex is not None and int(model.flex_interp[0]) != 0
  assert np.all(np.asarray(model.flex_vertbodyid) == -1)
  np.testing.assert_array_equal(np.unique(model.dof_treeid), np.arange(8))

  # Map pinned contact rows to their stable candidate slots, then feed the
  # canonical rows to the CPU island oracle. The row has multiple compiled
  # interpolation-tree contributors although flex_vertbodyid is -1.
  J = np.zeros((descriptor.nr, model.nv), dtype=np.float32)
  pinned_J = np.asarray(data.efc_J, dtype=np.float32).reshape(
      int(data.nefc), int(model.nv))
  active = np.zeros(descriptor.nr, dtype=bool)
  candidates = np.flatnonzero(flex.kind == _KIND_PLANE_VERTEX)
  for contact in data.contact[:data.ncon]:
    if int(contact.flex[1]) != 0:
      continue
    vertex = int(contact.vert[1])
    matches = candidates[flex.vert1[candidates] == vertex]
    assert matches.size == 1
    slot = int(matches[0])
    row = descriptor.flex_contact_base + int(flex.row_start[slot])
    assert metadata[row, 5] == 0, "flex rows must scan their actual Jacobian support"
    active[row] = True
    J[row] = pinned_J[int(contact.efc_address)]

  support_trees = np.unique(model.dof_treeid[np.any(J != 0, axis=0)])
  assert support_trees.size > 2
  result = build_solver_islands(
      np.asarray(model.dof_treeid, dtype=np.int32), J, active,
      body_tree=np.asarray(model.body_treeid, dtype=np.int32))
  assert result.island_count == int(data.nisland) == 1
  assert np.all(result.dof_island[np.unique(model.dof_treeid[J.any(axis=0)])] == 0)
  assert np.all(result.row_island[active] == 0)


def test_solver_dimension_metadata_has_fixed_header_and_exact_offsets():
  scalar = np.arange(22, dtype=np.int32)
  dof_tree = np.asarray([0, 1, 1], dtype=np.int32)
  rows = np.arange(20, dtype=np.int32).reshape(2, 10)
  packed = pack_solver_dimension_metadata(
      scalar, dof_tree, rows, ntree=2, core_scratch=35,
      disable_island_bit=1 << 4, nbody=1)
  assert packed.shape == (22 + 6 + 3 + 20,)
  ntree, dof_offset, row_offset, core, disable, awake_offset = packed[22:28]
  np.testing.assert_array_equal(
      [ntree, dof_offset, row_offset, core, disable, awake_offset],
      [2, 28, 31, 35, 16, 49])
  np.testing.assert_array_equal(packed[dof_offset:dof_offset + 3], dof_tree)
  np.testing.assert_array_equal(packed[row_offset:].reshape(2, 10), rows)
  with pytest.raises(ValueError, match="row_metadata"):
    pack_solver_dimension_metadata(scalar, dof_tree, np.zeros((2, 9), dtype=np.int32),
                                   2, 35, 16)


def test_tendon_midpoint_metadata_uses_first_two_compiled_trees():
  from mujoco_metal.coupled_constraints import lower_coupled_constraints

  model = mujoco.MjModel.from_xml_string("""<mujoco>
    <worldbody>
      <body><joint name="a" type="slide" axis="1 0 0"/>
        <geom type="sphere" size=".1" mass="1"/></body>
      <body pos="1 0 0"><joint name="b" type="slide" axis="1 0 0"/>
        <geom type="sphere" size=".1" mass="1"/></body>
    </worldbody>
    <tendon><fixed name="t" limited="true" range="-.1 .1" frictionloss=".2">
      <joint joint="a" coef="1"/><joint joint="b" coef="-.5"/>
    </fixed></tendon>
  </mujoco>""")
  desc = lower_coupled_constraints(model)
  rows = lower_solver_row_metadata(model, desc)
  expected = np.asarray(model.tendon_treeid[0], dtype=np.int32)
  friction = rows[desc.ten_base]
  limits = rows[desc.ten_base + 1:desc.ten_base + 3]
  np.testing.assert_array_equal(friction[8:10], expected)
  np.testing.assert_array_equal(limits[:, 8:10], np.broadcast_to(expected, (2, 2)))


def test_mujoco_forward_disconnected_contact_islands_match_cpu_oracle():
  model = mujoco.MjModel.from_xml_string("""<mujoco>
    <option gravity="0 0 -9.81" jacobian="dense"/>
    <worldbody>
      <geom type="plane" size="3 3 .1"/>
      <body pos="-1 0 .09"><freejoint/><geom type="sphere" size=".1" mass="1"/></body>
      <body pos="1 0 .09"><freejoint/><geom type="sphere" size=".1" mass="1"/></body>
    </worldbody>
  </mujoco>""")
  data = mujoco.MjData(model)
  mujoco.mj_forward(model, data)
  assert data.nisland == 2 and data.nefc >= 2
  result = _assert_partition_matches_mujoco(model, data)
  assert result.island_count == 2
  assert result.row_island[0] != result.row_island[-1]


def test_mujoco_site_connect_rows_match_structural_tree_shortcut():
  model = mujoco.MjModel.from_xml_string("""<mujoco>
    <option gravity="0 0 0" jacobian="dense"/>
    <worldbody>
      <body pos="0 0 0"><freejoint/><geom type="sphere" size=".1" mass="1"/><site name="a"/></body>
      <body pos="1 0 0"><freejoint/><geom type="sphere" size=".1" mass="1"/><site name="b"/></body>
    </worldbody>
    <equality><connect site1="a" site2="b"/></equality>
  </mujoco>""")
  data = mujoco.MjData(model)
  mujoco.mj_forward(model, data)
  assert data.nefc == 3 and data.nisland == 1
  result = _assert_partition_matches_mujoco(model, data)
  assert np.all(result.row_island == result.row_island[0])


def test_mujoco_cross_tree_contact_shortcut_survives_zero_jacobian_rows():
  model = mujoco.MjModel.from_xml_string("""<mujoco>
    <option gravity="0 0 0" jacobian="dense"/>
    <worldbody>
      <body pos="0 0 0"><freejoint/><geom type="sphere" size=".1" mass="1"/></body>
      <body pos=".15 0 0"><freejoint/><geom type="sphere" size=".1" mass="1"/></body>
    </worldbody>
  </mujoco>""")
  data = mujoco.MjData(model)
  mujoco.mj_forward(model, data)
  assert data.ncon >= 1 and data.nisland == 1
  J, links, groups, exempt = _source_row_metadata(model, data)
  rows = np.asarray([int(kind) in (
      int(mujoco.mjtConstraint.mjCNSTR_CONTACT_FRICTIONLESS),
      int(mujoco.mjtConstraint.mjCNSTR_CONTACT_PYRAMIDAL),
      int(mujoco.mjtConstraint.mjCNSTR_CONTACT_ELLIPTIC))
      for kind in data.efc_type[:data.nefc]])
  assert np.any(rows)
  J[rows] = 0.0
  result = build_solver_islands(
      np.asarray(model.dof_treeid, dtype=np.int32), J,
      np.ones(data.nefc, dtype=bool), row_links=links, group_key=groups,
      group_exempt=exempt,
      body_tree=np.asarray(model.body_treeid, dtype=np.int32))
  expected_dof = np.asarray(data.dof_island, dtype=np.int32)
  assert np.all(result.dof_island[:12] == result.dof_island[0])
  assert np.all(expected_dof[:12] == expected_dof[0])
  assert result.dof_island[0] >= 0
  assert np.all(result.row_island[rows] == result.row_island[rows][0])


def test_mujoco_flex_equality_rows_use_per_row_generic_tree_scan():
  model = mujoco.MjModel.from_xml_string("""<mujoco>
    <option jacobian="dense"/>
    <worldbody><body name="anchor" pos="0 0 1">
      <geom type="sphere" size=".02"/>
      <flexcomp name="cable" type="grid" count="4 1 1"
          spacing=".1 .1 .1" mass=".5" radius=".01" dim="1">
        <edge equality="true"/>
        <contact contype="0" conaffinity="0"/>
      </flexcomp>
    </body></worldbody>
  </mujoco>""")
  data = mujoco.MjData(model)
  mujoco.mj_forward(model, data)
  assert data.nefc >= 2 and data.nisland >= 1
  result = _assert_partition_matches_mujoco(model, data)
  # The generated flex-equality rows can span different node-tree sets; the
  # pinned flex family exemption means each row is scanned independently.
  assert np.all(result.row_island >= 0)


def test_islands_disabled_midpoint_mask_uses_exact_contact_body_id():
  model = mujoco.MjModel.from_xml_string("""<mujoco>
    <option gravity="0 0 -9.81" jacobian="dense"/>
    <worldbody>
      <geom type="plane" size="2 2 .1"/>
      <body name="free" pos="0 0 .09">
        <freejoint/>
        <geom type="sphere" size=".1" mass="1" contype="0" conaffinity="0"/>
        <body name="child" pos="0 0 -.05">
          <geom type="sphere" size=".05" mass="0"/>
        </body>
      </body>
    </worldbody>
  </mujoco>""")
  model.opt.disableflags |= int(mujoco.mjtDisableBit.mjDSBL_ISLAND)
  data = mujoco.MjData(model)
  mujoco.mj_forward(model, data)
  free_body, child_body = model.body("free").id, model.body("child").id
  assert data.ncon == 1 and int(data.contact[0].geom[1]) == 2
  assert model.tree_dofnum[model.dof_treeid[0]] == 6
  assert model.body_subtreemass[free_body] == model.body_mass[free_body]

  J, links, groups, exempt = _source_row_metadata(model, data)
  mid_bodies = np.full((data.nefc, 2), -1, dtype=np.int32)
  for row in range(data.nefc):
    kind = int(data.efc_type[row])
    if kind in (int(mujoco.mjtConstraint.mjCNSTR_CONTACT_FRICTIONLESS),
                int(mujoco.mjtConstraint.mjCNSTR_CONTACT_PYRAMIDAL),
                int(mujoco.mjtConstraint.mjCNSTR_CONTACT_ELLIPTIC)):
      con = data.contact[int(data.efc_id[row])]
      for endpoint, geom in enumerate(con.geom):
        if int(geom) >= 0:
          mid_bodies[row, endpoint] = int(model.geom_bodyid[int(geom)])
  result = build_solver_islands(
      np.asarray(model.dof_treeid, dtype=np.int32), J,
      np.ones(data.nefc, dtype=bool), row_links=links, group_key=groups,
      group_exempt=exempt, midpoint_body_ids=mid_bodies,
      body_tree=np.asarray(model.body_treeid, dtype=np.int32))
  assert result.midpoint_blocked_body[child_body]
  assert not result.midpoint_blocked_body[free_body]
  assert not result.midpoint_blocked_tree[model.body_treeid[free_body]]
  # Pinned midpoint_eligible checks the exact free-joint body ID when islands
  # are disabled, so the zero-mass child's contact does not block its parent.
  assert bool(model.opt.disableflags & int(mujoco.mjtDisableBit.mjDSBL_ISLAND))
  assert not result.midpoint_blocked_body[free_body]


def test_midpoint_consumer_uses_exact_disabled_island_masks_and_island_labels():
  torch = pytest.importorskip("torch")
  from types import SimpleNamespace
  from mujoco_metal.implicit_midpoint import midpoint_eligibility

  model = mujoco.MjModel.from_xml_string("""<mujoco>
    <option gravity="0 0 -9.81" jacobian="dense"/>
    <worldbody><geom type="plane" size="2 2 .1"/>
      <body name="free" pos="0 0 .09"><freejoint/>
        <geom type="sphere" size=".1" mass="1" contype="0" conaffinity="0"/>
        <body name="child" pos="0 0 -.05"><geom type="sphere" size=".05" mass="0"/></body>
      </body>
    </worldbody>
  </mujoco>""")
  model.opt.disableflags |= int(mujoco.mjtDisableBit.mjDSBL_ISLAND)
  data = mujoco.MjData(model)
  mujoco.mj_forward(model, data)
  child = int(model.body("child").id)
  free = int(model.body("free").id)
  assert int(data.contact[0].geom[1]) == 2
  blocked_body = np.zeros((1, model.nbody), dtype=np.int32)
  blocked_body[0, child] = 1
  blocked_tree = np.zeros((1, model.ntree), dtype=np.int32)
  desc = SimpleNamespace(nfree=1, nbody=int(model.nbody), nv=int(model.nv),
                         bodyid=np.asarray([free], dtype=np.int32),
                         dofadr=np.asarray([0], dtype=np.int32))
  actual = midpoint_eligibility(
      desc, 1, torch=torch, device="cpu",
      midpoint_blocked_body=torch.as_tensor(blocked_body),
      midpoint_blocked_tree=torch.as_tensor(blocked_tree),
      dof_treeid=model.dof_treeid, islands_disabled=True)
  # The exact colliding child endpoint does not block the eligible free body.
  assert bool(actual[0, 0])
  blocked_body[0, free] = 1
  actual = midpoint_eligibility(
      desc, 1, torch=torch, device="cpu",
      midpoint_blocked_body=torch.as_tensor(blocked_body),
      midpoint_blocked_tree=torch.as_tensor(blocked_tree),
      dof_treeid=model.dof_treeid, islands_disabled=True)
  assert not bool(actual[0, 0])
  actual = midpoint_eligibility(
      desc, 1, torch=torch, device="cpu",
      dof_island=torch.zeros((1, model.nv), dtype=torch.int32),
      islands_disabled=False)
  assert not bool(actual[0, 0])
  actual = midpoint_eligibility(
      desc, 1, torch=torch, device="cpu",
      dof_island=torch.full((1, model.nv), -1, dtype=torch.int32),
      islands_disabled=False)
  assert bool(actual[0, 0])


@pytest.mark.parametrize("dof_tree,jacobian,active", [
    ([[0]], np.zeros((1, 1)), [True]),
    ([0], np.zeros((2, 1)), [True]),
    ([0], np.zeros((1, 1)), [1.0]),
])
def test_invalid_solver_island_abi_is_rejected(dof_tree, jacobian, active):
  with pytest.raises(ValueError):
    build_solver_islands(dof_tree, jacobian, active)


def _pinned_island_statistics(data):
  """Return only populated source solver records, using the pinned island stride."""
  nslots = len(data.solver.improvement)
  nisland_slots = len(data.solver_niter)
  assert nslots % nisland_slots == 0
  stride = nslots // nisland_slots
  # MuJoCo 3.10 saveStats writes island * mjNSOLVER + iteration.
  assert stride == 200
  rows = []
  for island in range(int(data.nisland)):
    count = int(data.solver_niter[island])
    first = island * stride
    stop = first + min(count, stride)
    rows.append({"island": island, "iterations": count,
                 **{name: getattr(data.solver, name)[first:stop].tolist()
                    for name in ("improvement", "gradient", "lineslope")}})
  return rows


@pytest.mark.gpu
@pytest.mark.skipif(os.getenv("MUJOCO_METAL_RUN_GPU") != "1", reason="opt-in GPU")
def test_metal_solver_island_workspace_matches_cpu_multibatch_oracle():
  import torch
  from mujoco_metal.solver_islands import MetalSolverIslandWorkspace

  dof_tree = np.asarray([0, 0, 1, 2], dtype=np.int32)
  links = np.asarray([[0, 1], [-1, -1], [2, -1]], dtype=np.int32)
  groups = np.asarray([[6, 0], [6, 0], [1, 1]], dtype=np.int32)
  exempt = np.zeros(3, dtype=np.int32)
  jacobian = np.zeros((3, 3, 4), dtype=np.float32)
  jacobian[:, 0, 0] = 1.0
  jacobian[:, 1, 3] = 1.0
  jacobian[:, 2, 3] = 1.0
  active = np.asarray([[1, 1, 1], [0, 1, 0], [1, 0, 1]], dtype=np.int32)
  workspace = MetalSolverIslandWorkspace(
      dof_tree, links, groups, exempt, batch_size=3)
  with pytest.raises(ValueError, match="jacobian"):
    workspace.run_device(torch.zeros((3, 3, 4), dtype=torch.float32),
                         torch.zeros((3, 3), dtype=torch.int32))
  with pytest.raises(ValueError, match="jacobian"):
    workspace.run_device(
        torch.zeros((3, 3, 4), dtype=torch.float64),
        torch.zeros((3, 3), dtype=torch.int32))
  with pytest.raises(ValueError, match="row_activity"):
    workspace.run_device(
        torch.zeros((3, 3, 4), dtype=torch.float32, device="mps"),
        torch.zeros((3, 3), dtype=torch.int64, device="mps"))
  result = workspace.run_device(
      torch.as_tensor(jacobian, dtype=torch.float32, device="mps"),
      torch.as_tensor(active, dtype=torch.float32, device="mps"))
  actual = {
      name: getattr(result, name).detach().cpu().numpy()
      for name in ("dof_island", "row_island", "row_order", "row_inverse",
                   "island_offsets", "island_count")
  }
  for world in range(3):
    cpu = build_solver_islands(
        dof_tree, jacobian[world], active[world], row_links=links[:, None, :],
        group_key=groups[:, 0] * 100 + groups[:, 1],
        group_exempt=exempt)
    np.testing.assert_array_equal(actual["dof_island"][world], cpu.dof_island)
    np.testing.assert_array_equal(actual["row_island"][world], cpu.row_island)
    np.testing.assert_array_equal(actual["row_order"][world], cpu.row_order)
    np.testing.assert_array_equal(actual["row_inverse"][world], cpu.row_inverse)
    np.testing.assert_array_equal(actual["island_count"][world], cpu.island_count)
    n = cpu.island_count
    np.testing.assert_array_equal(
        actual["island_offsets"][world, :n + 1], cpu.island_offsets)


@pytest.mark.parametrize("solver", ["CG", "Newton"])
@pytest.mark.gpu
@pytest.mark.skipif(os.getenv("MUJOCO_METAL_RUN_GPU") != "1",
                    reason="requires explicit GPU qualification slot")
def test_primal_solver_mixed_mass_coupled_and_contact_islands(solver):
  """Primal solve keeps a coupled mass island separate from a cone island."""
  from mujoco_metal.capacity import CapacityLimits
  from mujoco_metal.native_api import mj_forward
  from mujoco_metal.simulation import MetalSimulation

  model = mujoco.MjModel.from_xml_string(f"""<mujoco>
    <option timestep=".0005" gravity="0 0 -9.81" solver="{solver}"
      iterations="3" tolerance="1e-8" cone="pyramidal"/>
    <worldbody>
      <geom name="floor" type="plane" size="4 4 .1"/>
      <body name="chain" pos="-1 0 .4">
        <joint name="chain_slide" type="slide" axis="1 0 0"/>
        <geom type="box" pos=".1 0 0" size=".1 .1 .1" mass="1"/>
        <body name="link" pos=".35 0 0">
          <joint name="chain_hinge" type="hinge" axis="0 0 1"/>
          <geom type="capsule" fromto="0 0 0 .3 0 0" size=".06" mass=".7"/>
        </body>
      </body>
      <body name="coupled" pos="1 0 .3">
        <joint name="coupled_slide" type="slide" axis="1 0 0"/>
        <geom type="sphere" size=".08" mass=".8"/>
      </body>
      <body name="contact" pos="2 0 .09">
        <freejoint/><geom type="sphere" size=".1" mass="1"/>
      </body>
    </worldbody>
    <equality><joint name="cross_tree" joint1="chain_hinge"
      joint2="coupled_slide" polycoef="0 1 0 0 0" solref=".02 1"/></equality>
  </mujoco>""")
  assert model.nv == 9
  qpos = np.tile(model.qpos0, (2, 1)).astype(np.float32)
  qvel = np.zeros((2, model.nv), dtype=np.float32)
  qpos[:, 0] = [.03, -.02]
  qpos[:, 1] = [.18, -.11]
  qpos[:, 2] = [.01, -.015]
  # The free-joint translation is qpos[3:6], with z at index 5.
  qpos[:, 5] = [.09, .095]
  qvel[:, :3] = [[.2, -.3, .1], [-.15, .25, -.08]]
  qvel[:, 3:9] = [[.1, -.04, .03, .02, -.01, .04],
                   [-.08, .06, -.02, -.03, .02, -.01]]

  references = []
  reference_masses = []
  for world in range(2):
    data = mujoco.MjData(model)
    data.qpos[:] = qpos[world]
    data.qvel[:] = qvel[world]
    mujoco.mj_forward(model, data)
    assert data.nisland == 2
    assert np.any(data.efc_island[:data.nefc] == 0)
    assert np.any(data.efc_island[:data.nefc] == 1)
    full_mass = np.empty((model.nv, model.nv), dtype=np.float64)
    mujoco.mj_fullM(model, data, full_mass)
    assert abs(full_mass[0, 1]) > 1e-5
    reference_masses.append(full_mass.copy())
    references.append(data)

  sim = MetalSimulation(
      model, batch_size=2, qpos=qpos, qvel=qvel,
      profile="integrated_scalable_v1",
      limits=CapacityLimits(max_nv=16, max_rows=128, max_pairs=16,
                            max_slots=16, max_batch=2))
  sim._coupled_constraints.set_iteration_trace(True)
  stages = mj_forward(sim)
  actual_qacc = stages["qacc"].detach().cpu().numpy()
  expected_qacc = np.stack([data.qacc.copy() for data in references])
  constraint = stages["constraint"]
  actual_force = constraint["qfrc_constraint"].detach().cpu().numpy()
  actual_J = constraint["J"].detach().cpu().numpy()
  actual_lambda = constraint["lambda"].detach().cpu().numpy()
  force_from_rows = np.einsum("brd,br->bd", actual_J, actual_lambda)
  expected_force = np.stack([data.qfrc_constraint.copy() for data in references])
  actual_smooth = stages["acceleration"]["qacc_smooth"].detach().cpu().numpy()
  force_from_acceleration = np.stack([
      reference_masses[world] @ (actual_qacc[world] - actual_smooth[world])
      for world in range(2)])
  expected_smooth = np.stack([data.qacc_smooth.copy() for data in references])
  reference_stationarity = np.stack([
      reference_masses[world] @ (references[world].qacc - expected_smooth[world])
      - expected_force[world]
      for world in range(2)])
  actual_stationarity = force_from_acceleration - actual_force
  # The later island solve must not erase force from an earlier island.
  # Check both canonical Jᵀλ and the public force before qacc parity can fail.
  np.testing.assert_allclose(force_from_rows, actual_force, rtol=2e-6, atol=2e-6)
  np.testing.assert_allclose(actual_force, expected_force, rtol=5e-4, atol=3e-5)
  # Finite-budget CG can stop with a nonzero residual on the contact island.
  # Compare the residual to the pinned finite-budget result instead of
  # incorrectly demanding a converged KKT point.  The first island has an
  # unconstrained generalized coordinate and remains a useful zero-residual
  # witness.
  np.testing.assert_allclose(actual_stationarity, reference_stationarity,
                             rtol=5e-4, atol=3e-5)
  np.testing.assert_allclose(actual_stationarity[:, 0], 0.0,
                             rtol=0.0, atol=3e-5)
  np.testing.assert_allclose(actual_qacc, expected_qacc,
                             rtol=5e-4, atol=3e-5)
  expected_iterations = np.asarray([
      np.sum(data.solver_niter[:data.nisland]) for data in references])
  actual_iterations = constraint["solver_diagnostics"][:, 1].detach().cpu().numpy()
  # Trace mode replaces history slots [2:10] with per-island primal counts.
  # This fixture has two source islands; unused slots retain the -1 sentinel.
  actual_island_iterations = (
      constraint["solver_history"].detach().cpu().numpy())
  expected_island_iterations = np.full((2, 8), -1.0, dtype=np.float32)
  for world, data in enumerate(references):
    expected_island_iterations[world, :data.nisland] = np.asarray(
        data.solver_niter[:data.nisland], dtype=np.float32)
  print(f"ITER_TRACE solver={solver} per-island native=",
        actual_island_iterations[:, :2].tolist(),
        "source=", expected_island_iterations[:, :2].tolist(), flush=True)
  sim._coupled_constraints.set_primal_detail_trace(True)
  detailed = mj_forward(sim)
  detail_history = (detailed["constraint"]["solver_history"]
                    .detach().cpu().numpy())
  print(f"PRIMAL_DETAIL solver={solver} fields="
        "[scale,tol,cold_cost,warm_delta,initial_scaled_grad,alpha,"
        "scaled_improvement,post_scaled_grad] native=",
        detail_history.tolist(),
        "source_stats=",
        [{"counts": data.solver_niter[:data.nisland].tolist(),
          "qacc_smooth": data.qacc_smooth.copy().tolist(),
          "qacc_warmstart": data.qacc_warmstart.copy().tolist(),
          "qacc": data.qacc.copy().tolist(),
          "island_statistics": _pinned_island_statistics(data)}
         for data in references], flush=True)
  np.testing.assert_array_equal(
      detailed["qacc"].detach().cpu().numpy(), actual_qacc)
  np.testing.assert_array_equal(
      detailed["constraint"]["qfrc_constraint"].detach().cpu().numpy(),
      actual_force)
  # Capture the next accepted point and the paired gradient components on
  # fresh executions of the same deterministic input. The extra executions
  # are observers only; the original public outputs must remain bitwise equal.
  sim._coupled_constraints.set_primal_detail_trace(
      True, accepted_iteration=2)
  second = mj_forward(sim)
  second_detail = (second["constraint"]["solver_history"]
                   .detach().cpu().numpy())
  print(f"PRIMAL_SECOND_POINT solver={solver} fields="
        "[alpha,scaled_improvement,scaled_gradient] native=",
        second_detail.tolist(), flush=True)
  np.testing.assert_array_equal(second["qacc"].detach().cpu().numpy(),
                                actual_qacc)
  np.testing.assert_array_equal(
      second["constraint"]["qfrc_constraint"].detach().cpu().numpy(),
      actual_force)
  sim._coupled_constraints.set_primal_detail_trace(
      True, accepted_iteration=2, gradient_components=True)
  second_components = mj_forward(sim)
  component_history = (second_components["constraint"]["solver_history"]
                       .detach().cpu().numpy())
  # The source gradient is M(qacc-qacc_smooth)-qfrc_constraint. Run the
  # pinned solver for two accepted iterations to compare the same state; the
  # native observer stores high and low components before the norm reduction.
  saved_iterations = int(model.opt.iterations)
  source_gradients = []
  try:
    model.opt.iterations = 2
    for world in range(2):
      source = mujoco.MjData(model)
      source.qpos[:] = qpos[world]
      source.qvel[:] = qvel[world]
      mujoco.mj_forward(model, source)
      source_mass = np.empty((model.nv, model.nv), dtype=np.float64)
      mujoco.mj_fullM(model, source, source_mass)
      source_gradients.append(
          source_mass @ (source.qacc - source.qacc_smooth)
          - source.qfrc_constraint)
  finally:
    model.opt.iterations = saved_iterations
  print(f"PRIMAL_SECOND_GRADIENT solver={solver} native_hi_lo=",
        component_history[:, :6].tolist(), "source=",
        [gradient[:3].tolist() for gradient in source_gradients], flush=True)
  np.testing.assert_array_equal(
      second_components["qacc"].detach().cpu().numpy(), actual_qacc)
  np.testing.assert_array_equal(
      second_components["constraint"]["qfrc_constraint"].detach().cpu().numpy(),
      actual_force)
  sim._coupled_constraints.set_primal_detail_trace(False)
  np.testing.assert_array_equal(actual_iterations, expected_iterations)
  np.testing.assert_array_equal(actual_island_iterations,
                                expected_island_iterations)


def test_iteration_trace_flag_is_idempotent_cpu():
  torch = pytest.importorskip("torch")
  from mujoco_metal.coupled_constraints import MetalCoupledConstraints

  solver = MetalCoupledConstraints.__new__(MetalCoupledConstraints)
  dims = torch.tensor([0, 0, 0, 0, 0, 0, 0, 13], dtype=torch.int32)
  solver._constants = {"solver_dims": dims}
  solver.set_iteration_trace(True)
  assert int(dims[7]) == 45
  solver.set_iteration_trace(True)
  assert int(dims[7]) == 45
  solver.set_iteration_trace(False)
  assert int(dims[7]) == 13
  solver.set_iteration_trace(False)
  assert int(dims[7]) == 13


def test_primal_detail_trace_flag_and_cached_stage_are_cpu_safe():
  import importlib.util
  from pathlib import Path

  source = Path(__file__).parents[1] / "mujoco_metal" / "coupled_constraints.py"
  spec = importlib.util.spec_from_file_location(
      "mujoco_metal._primal_detail_flag_test", source)
  candidate = importlib.util.module_from_spec(spec)
  spec.loader.exec_module(candidate)

  class _View:
    def __init__(self, data):
      self.data = data

    def add_(self, amount):
      self.data[7] += int(amount)
      return self

  class _Dims:
    def __init__(self):
      self.data = [0] * 7 + [1]

    def __getitem__(self, index):
      return _View(self.data) if isinstance(index, slice) else self.data[index]

    def __setitem__(self, index, value):
      self.data[index] = int(value)

  solver = candidate.MetalCoupledConstraints.__new__(
      candidate.MetalCoupledConstraints)
  dims = _Dims()
  solver._constants = {"solver_dims": dims}
  solver.set_primal_detail_trace(True)
  assert dims[7] == 1 | 32 | 64
  with candidate._cached_position_stage(dims, True, True, True):
    assert dims[7] == 1 | 2 | 32 | 64
    with candidate._provided_smooth_acceleration_stage(dims):
      assert dims[7] == 1 | 2 | 4 | 32 | 64
    with candidate._paired_smooth_input_stage(dims):
      assert dims[7] == 1 | 2 | 16 | 32 | 64
    with candidate._preassembled_rows_stage(dims):
      assert dims[7] == 1 | 2 | 8 | 32 | 64
  assert dims[7] == 1 | 32 | 64
  solver.set_primal_detail_trace(True)
  assert dims[7] == 1 | 32 | 64
  solver.set_primal_detail_trace(False)
  assert dims[7] == 1 | 32


def test_primal_detail_trace_nested_flags_cpu():
  import importlib.util
  from pathlib import Path

  source = Path(__file__).parents[1] / "mujoco_metal" / "coupled_constraints.py"
  spec = importlib.util.spec_from_file_location(
      "mujoco_metal._primal_detail_candidate", source)
  candidate = importlib.util.module_from_spec(spec)
  spec.loader.exec_module(candidate)

  class _WordView:
    def __init__(self, words, index):
      self.words = words
      self.index = index

    def add_(self, value):
      self.words[self.index] += int(value)
      return self

  class _IntDims:
    def __init__(self, values):
      self.words = list(values)

    def __getitem__(self, index):
      if isinstance(index, slice):
        assert index.start == 7 and index.stop == 8
        return _WordView(self.words, 7)
      return self.words[index]

    def __setitem__(self, index, value):
      self.words[index] = int(value)

  solver = candidate.MetalCoupledConstraints.__new__(
      candidate.MetalCoupledConstraints)
  dims = _IntDims([0] * 7 + [1] + [0] * 13)
  solver._constants = {"solver_dims": dims}
  solver.set_iteration_trace(True)
  solver.set_primal_detail_trace(True)
  solver.set_primal_detail_trace(True)
  base = 1 | 32 | 64
  assert int(dims[7]) == base
  with candidate._cached_position_stage(dims, True, True, True):
    assert int(dims[7]) == base | 2
    with candidate._provided_smooth_acceleration_stage(dims):
      assert int(dims[7]) == base | 2 | 4
    with candidate._paired_smooth_input_stage(dims):
      assert int(dims[7]) == base | 2 | 16
    with candidate._preassembled_rows_stage(dims):
      assert int(dims[7]) == base | 2 | 8
  assert int(dims[7]) == base
  solver.set_primal_detail_trace(False)
  assert int(dims[7]) == 1 | 32
  assert solver._iteration_trace_enabled is True
  solver.set_primal_detail_trace(True, accepted_iteration=2)
  assert int(dims[7]) == 1 | 32 | 64 | 128
  solver.set_primal_detail_trace(
      True, accepted_iteration=2, gradient_components=True)
  assert int(dims[7]) == 1 | 32 | 64 | 128 | 256
  observer_base = 1 | 32 | 64 | 128 | 256
  extras = solver._primal_detail_extra_flags()
  assert extras == 128 | 256
  with candidate._component_primal_stage(
      dims, True, -1, 40, True, True, True, extras):
    assert int(dims[7]) == observer_base | 2 | 4
    with candidate._preassembled_rows_stage(dims):
      assert int(dims[7]) == observer_base | 2 | 4 | 8
  assert int(dims[7]) == observer_base
  with candidate._cached_position_stage(dims, True, True, True, extras):
    assert int(dims[7]) == observer_base | 2
  assert int(dims[7]) == observer_base
  solver.set_iteration_trace(False)
  assert int(dims[7]) == 1 | 64 | 128 | 256
  solver.set_primal_detail_trace(False)
  assert int(dims[7]) == 1


def test_cached_position_stage_preserves_iteration_trace_and_nested_flags_cpu():
  import importlib.util
  from pathlib import Path

  source = Path(__file__).parents[1] / "mujoco_metal" / "coupled_constraints.py"
  spec = importlib.util.spec_from_file_location(
      "mujoco_metal._cached_flag_candidate", source)
  candidate = importlib.util.module_from_spec(spec)
  spec.loader.exec_module(candidate)
  solver_type = candidate.MetalCoupledConstraints
  cached = candidate._cached_position_stage
  paired = candidate._paired_smooth_input_stage
  preassembled = candidate._preassembled_rows_stage
  provided = candidate._provided_smooth_acceleration_stage

  class _WordView:
    def __init__(self, words, index):
      self.words = words
      self.index = index

    def add_(self, value):
      self.words[self.index] += int(value)
      return self

  class _IntDims:
    def __init__(self, values):
      self.words = list(values)

    def __getitem__(self, index):
      if isinstance(index, slice):
        assert index.start == 7 and index.stop == 8
        return _WordView(self.words, 7)
      return self.words[index]

    def __setitem__(self, index, value):
      self.words[index] = int(value)

  solver = solver_type.__new__(solver_type)
  dims = _IntDims([0] * 7 + [1])
  solver._constants = {"solver_dims": dims}
  solver.set_iteration_trace(True)
  base = 1 | 32
  assert int(dims[7]) == base

  with cached(dims, True, solver._iteration_trace_enabled):
    assert int(dims[7]) == base | 2
    with provided(dims):
      assert int(dims[7]) == base | 2 | 4
    with paired(dims):
      assert int(dims[7]) == base | 2 | 16
    with preassembled(dims):
      assert int(dims[7]) == base | 2 | 8
  assert int(dims[7]) == base
  assert solver._iteration_trace_enabled is True

  with pytest.raises(RuntimeError, match="injected cached-stage failure"):
    with cached(dims, True, solver._iteration_trace_enabled):
      with provided(dims), paired(dims), preassembled(dims):
        raise RuntimeError("injected cached-stage failure")
  assert int(dims[7]) == base
  assert solver._iteration_trace_enabled is True
