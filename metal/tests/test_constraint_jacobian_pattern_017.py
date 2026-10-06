# Copyright 2026 The MuJoCo Metal contributors
# Licensed under the Apache License, Version 2.0.

"""CPU structure and opt-in runtime checks for canonical sparse constraint J."""

import os
import mujoco
import numpy as np
import pytest

from mujoco_metal.capacity import CapacityLimits, estimate_capacity
from mujoco_metal.constraint_jacobian import (
    PACKED_J_DENSE,
    PACKED_J_CSR,
    PACKED_J_MAGIC,
    PACKED_J_VERSION,
    PackedJacobianLayout,
    compile_constraint_jacobian_pattern,
    initialize_packed_jacobian_records,
    materialize_packed_jacobian_torch,
    packed_jacobian_device_storage,
    packed_jacobian_scatter_rows_torch,
    packed_jacobian_read,
    packed_jacobian_write,
)
from mujoco_metal.coupled_constraints import lower_coupled_constraints


def _model(jacobian="dense", cone="pyramidal"):
  return mujoco.MjModel.from_xml_string(f"""
    <mujoco><option gravity="0 0 0" solver="PGS" cone="{cone}" iterations="4"
                    jacobian="{jacobian}"/>
      <worldbody>
        <body name="a" pos="0 0 0"><joint name="ja" type="slide"
              axis="1 0 0"/><geom name="ga" type="sphere" size=".1"
              mass="1" contype="1" conaffinity="2"/>
          <body name="child" pos="0 .2 0"><joint name="jc" type="hinge"
              axis="0 0 1"/><geom name="gc" type="sphere" size=".1"
              mass=".2" contype="1" conaffinity="2"/></body>
        </body>
        <body name="b" pos="1 0 0"><joint name="jb" type="slide"
              axis="0 1 0"/><geom name="gb" type="sphere" size=".1"
              mass="1" contype="2" conaffinity="1"/></body>
      </worldbody>
      <equality><connect body1="child" body2="b" anchor=".5 .1 0"/>
        <joint joint1="ja" joint2="jb" polycoef="0 1 0 0 0"/>
      </equality>
    </mujoco>
  """)


def _sparse_runtime_model(jacobian, cone):
  """Contact/equality/tendon model with four row dimensions and one gap world."""
  return mujoco.MjModel.from_xml_string(f"""
    <mujoco><option timestep=".0005" gravity="0 0 -9.81" solver="PGS"
                    cone="{cone}" iterations="80" jacobian="{jacobian}"/>
      <worldbody>
        <geom name="floor" type="plane" size="0 0 .1"
              contype="0" conaffinity="0"/>
        <body name="a" pos="0 0 .08"><freejoint name="ja"/>
          <geom name="ga" type="sphere" size=".1" mass="1"/>
          <site name="sa"/></body>
        <body name="b" pos=".4 0 .08"><freejoint name="jb"/>
          <geom name="gb" type="sphere" size=".1" mass="1"/>
          <site name="sb"/></body>
        <body name="c" pos=".8 0 .08"><freejoint name="jc"/>
          <geom name="gc" type="sphere" size=".1" mass="1"/>
          <site name="sc"/></body>
        <body name="d" pos="1.2 0 .08"><freejoint name="jd"/>
          <geom name="gd" type="sphere" size=".1" mass="1"/>
          <site name="sd"/></body>
      </worldbody>
      <contact>
        <pair geom1="floor" geom2="ga" condim="1"/>
        <pair geom1="floor" geom2="gb" condim="3"/>
        <pair geom1="floor" geom2="gc" condim="4"/>
        <pair geom1="floor" geom2="gd" condim="6"/>
      </contact>
      <equality>
        <connect body1="a" body2="b" anchor=".2 0 .08"/>
      </equality>
      <tendon><spatial name="t" frictionloss=".1" limited="false">
        <site site="sc"/><site site="sd"/>
      </spatial></tendon>
    </mujoco>
  """)


def test_sparse_constraint_pattern_is_sorted_immutable_and_covers_pinned_rows():
  model = _model()
  descriptor = lower_coupled_constraints(model)
  pattern = compile_constraint_jacobian_pattern(model, descriptor)
  assert pattern.nnz < descriptor.nr * descriptor.nv
  assert pattern.row_offsets.shape == (descriptor.nr + 1,)
  assert pattern.row_offsets[0] == 0
  assert pattern.row_offsets[-1] == pattern.nnz
  assert pattern.columns.dtype == np.int32
  assert not pattern.row_offsets.flags.writeable
  assert not pattern.columns.flags.writeable
  for row in range(descriptor.nr):
    cols = pattern.row_columns(row)
    assert np.all(cols[1:] > cols[:-1])
    assert np.all((0 <= cols) & (cols < model.nv))

  data = mujoco.MjData(model)
  mujoco.mj_forward(model, data)
  # Equality row addresses are the same canonical prefix used by this solver.
  native_j = np.asarray(data.efc_J).reshape(data.nefc, model.nv)
  native_eq = np.flatnonzero(
      data.efc_type == int(mujoco.mjtConstraint.mjCNSTR_EQUALITY))
  assert native_eq.size == descriptor.n_eq_rows
  for canonical, reference in enumerate(native_eq):
    support = set(map(int, pattern.row_columns(canonical)))
    nonzero = set(map(int, np.flatnonzero(native_j[reference])))
    assert nonzero <= support

  # Contact rows include the complete two-body ancestor support, including
  # zero-valued structural columns at this symmetric initial pose.
  contact_base = descriptor.nr_joint
  contact_rows = [row for row in range(contact_base, descriptor.nr)
                  if pattern.row_columns(row).size]
  assert contact_rows
  assert any(set(map(int, pattern.row_columns(row))) >= {0, 1, 2}
             for row in contact_rows)


def test_sparse_pattern_is_owned_by_sparse_descriptor_not_dense_profile():
  model = _model()
  sparse = _model(jacobian="sparse")
  sparse_descriptor = lower_coupled_constraints(sparse)
  assert sparse_descriptor.jacobian_kind == "sparse"
  assert sparse_descriptor.jacobian_pattern is not None
  compiled = compile_constraint_jacobian_pattern(sparse, sparse_descriptor)
  np.testing.assert_array_equal(sparse_descriptor.jacobian_pattern.row_offsets,
                                compiled.row_offsets)
  np.testing.assert_array_equal(sparse_descriptor.jacobian_pattern.columns,
                                compiled.columns)
  dense_descriptor = lower_coupled_constraints(model)
  assert dense_descriptor.jacobian_kind == "dense"
  assert dense_descriptor.jacobian_pattern is None


def test_packed_csr_jacobian_abi_uses_typed_header_and_checked_slots():
  model = _model(jacobian="sparse")
  descriptor = lower_coupled_constraints(model)
  pattern = descriptor.jacobian_pattern
  layout = PackedJacobianLayout.create(
      descriptor.nr, descriptor.nv, pattern=pattern)
  assert layout.mode == PACKED_J_CSR
  assert layout.stride_words == (
      12 + descriptor.nr + 1 + pattern.nnz + pattern.nnz)

  batch = 2
  words = np.full(batch * layout.stride_words, -1, dtype=np.int32)
  initialized = initialize_packed_jacobian_records(
      words, batch, layout, pattern)
  assert initialized == words.size
  # The ABI header is int32 metadata, not float-encoded offsets. Pattern
  # metadata is replicated per world so device indexing never relies on an
  # untyped shared tail or world-zero-only columns.
  f32 = words.view(np.float32)
  for world in range(batch):
    base = world * layout.stride_words
    h = words[base:base + 12]
    assert tuple(map(int, h[:6])) == (
        PACKED_J_MAGIC, PACKED_J_VERSION, PACKED_J_CSR,
        descriptor.nr, descriptor.nv, pattern.nnz)
    assert h[9] == layout.stride_words
    assert h[10] == 0  # per-world unsupported-write status
    np.testing.assert_array_equal(
        words[base + layout.rowptr_offset:
              base + layout.rowptr_offset + descriptor.nr + 1],
        pattern.row_offsets)
    np.testing.assert_array_equal(
        words[base + layout.columns_offset:
              base + layout.columns_offset + pattern.nnz], pattern.columns)

  # Every supported slot round-trips, including the second independent world.
  for world in range(batch):
    base = world * layout.stride_words
    record = f32[base:base + layout.stride_words]
    for row in range(descriptor.nr):
      for col in pattern.row_columns(row):
        assert packed_jacobian_write(record, layout, row, int(col),
                                     0.25 + world)
        assert packed_jacobian_read(record, layout, row, int(col)) == 0.25 + world
      off_pattern = next((col for col in range(descriptor.nv)
                          if col not in set(map(int, pattern.row_columns(row)))),
                         None)
      if off_pattern is not None:
        assert packed_jacobian_write(record, layout, row, off_pattern, 0.0)
        assert not packed_jacobian_write(record, layout, row, off_pattern, 1.0)
        assert packed_jacobian_read(record, layout, row, off_pattern) == 0.0


def test_packed_jacobian_layout_rejects_signed_address_overflow():
  class Pattern:
    nr = 1
    nv = 1
    nnz = np.iinfo(np.int32).max

  with np.testing.assert_raises_regex(ValueError, "signed int32"):
    PackedJacobianLayout.create(1, 1, pattern=Pattern())


def test_sparse_capacity_charges_per_world_csr_records_instead_of_dense_j():
  model = _model(jacobian="sparse")
  descriptor = lower_coupled_constraints(model)
  pattern = descriptor.jacobian_pattern
  layout = PackedJacobianLayout.create(
      descriptor.nr, descriptor.nv, pattern=pattern)
  estimate = estimate_capacity(
      model, 2, descriptor.npairs, descriptor.ncontacts_max, descriptor.nr,
      neq=descriptor.neq, nr_joint=descriptor.nr_joint,
      jacobian_kind="sparse", jacobian_nnz=pattern.nnz)
  sizes = dict(estimate.memory_breakdown)
  assert sizes["workspace_J"] == 4 * 2 * layout.stride_words
  assert sizes["position_cache_J"] == 4 * 2 * layout.stride_words


def test_packed_runtime_scatter_and_explicit_dense_materialization_cpu():
  import torch

  model = _model(jacobian="sparse")
  descriptor = lower_coupled_constraints(model)
  pattern = descriptor.jacobian_pattern
  layout = PackedJacobianLayout.create(
      descriptor.nr, descriptor.nv, pattern=pattern)
  storage, runtime_layout = packed_jacobian_device_storage(
      torch, torch.device("cpu"), 2, pattern)
  assert runtime_layout == layout
  row_start = max(0, descriptor.nr - 2)
  count = descriptor.nr - row_start
  source = torch.zeros((2, count, descriptor.nv), dtype=torch.float32)
  active = torch.tensor([[1, 0], [0, 1]], dtype=torch.int32)[:, :count]
  for local in range(count):
    cols = pattern.row_columns(row_start + local)
    if cols.size:
      source[:, local, torch.as_tensor(cols.copy(), dtype=torch.int64)] = (
          torch.tensor([1.25, -2.5])[:, None])
  packed_jacobian_scatter_rows_torch(
      torch, storage, 2, layout, pattern, source, row_start,
      row_count=count, active=active)
  actual = materialize_packed_jacobian_torch(
      torch, storage, 2, layout, pattern).numpy()
  expected = np.zeros_like(actual)
  for world in range(2):
    for local in range(count):
      if active[world, local]:
        for col in pattern.row_columns(row_start + local):
          expected[world, row_start + local, int(col)] = source[
              world, local, int(col)]
  np.testing.assert_array_equal(actual, expected)


def test_sparse_inverse_constraint_force_consumes_csr_without_dense_J():
  """The inverse reduction accepts the stepping ABI directly."""
  import torch
  from mujoco_metal.inverse_constraints import inverse_constraint_force

  model = _model(jacobian="sparse")
  descriptor = lower_coupled_constraints(model)
  pattern = descriptor.jacobian_pattern
  layout = PackedJacobianLayout.create(
      descriptor.nr, descriptor.nv, pattern=pattern)
  storage, _ = packed_jacobian_device_storage(
      torch, torch.device("cpu"), 2, pattern)
  # Start from the numerical values a producer would have emitted. The rows
  # are zero outside their topology-compiled support by construction.
  source = torch.zeros((2, descriptor.nr, descriptor.nv), dtype=torch.float32)
  for row in range(descriptor.nr):
    columns = pattern.row_columns(row)
    if columns.size:
      col = torch.as_tensor(columns.copy(), dtype=torch.int64)
      source[:, row, col] = torch.arange(
          1, columns.size + 1, dtype=torch.float32)[None, :]
      source[1, row, col] *= -0.25
  packed_jacobian_scatter_rows_torch(
      torch, storage, 2, layout, pattern, source, 0)

  active = torch.ones((2, descriptor.nr), dtype=torch.float32)
  regularizer = torch.ones_like(active)
  aref = torch.linspace(-.2, .3, descriptor.nr).expand(2, -1).contiguous()
  rows = {
      "J_packed": storage,
      "jacobian_layout": layout,
      "jacobian_pattern": pattern,
      "R": regularizer,
      "ar": aref,
      "lo": torch.full_like(active, -torch.inf),
      "hi": torch.full_like(active, torch.inf),
      "active": active,
  }
  qacc = torch.linspace(-.5, .75, 2 * descriptor.nv).reshape(
      2, descriptor.nv).to(torch.float32)
  # Deliberately do not provide rows["J"]. A sparse query must use the packed
  # per-world CSR values and return J.T*f without materializing B*nr*nv.
  actual = inverse_constraint_force(rows, qacc, descriptor)
  dense = materialize_packed_jacobian_torch(
      torch, storage, 2, layout, pattern)
  jar = torch.bmm(dense, qacc.unsqueeze(-1)).squeeze(-1) - aref
  expected = torch.bmm(dense.transpose(1, 2), (-jar).unsqueeze(-1)).squeeze(-1)
  torch.testing.assert_close(actual, expected, rtol=1e-7, atol=2e-6)
  assert "J" not in rows


def test_sparse_elliptic_inverse_keeps_packed_j_after_contact_metadata_load():
  """Elliptic contact handling must not shadow the packed-J query buffer."""
  import torch
  from mujoco_metal.inverse_constraints import inverse_constraint_force

  model = _model(jacobian="sparse", cone="elliptic")
  descriptor = lower_coupled_constraints(model)
  assert descriptor.ncontacts_max > 0
  assert int(descriptor.cone_type) == int(mujoco.mjtCone.mjCONE_ELLIPTIC)
  pattern = descriptor.jacobian_pattern
  layout = PackedJacobianLayout.create(
      descriptor.nr, descriptor.nv, pattern=pattern)
  storage, _ = packed_jacobian_device_storage(
      torch, torch.device("cpu"), 2, pattern)
  dense = torch.zeros((2, descriptor.nr, descriptor.nv), dtype=torch.float32)
  for row in range(descriptor.nr):
    columns = pattern.row_columns(row)
    if columns.size:
      cols = torch.as_tensor(columns.copy(), dtype=torch.int64)
      values = torch.arange(1, columns.size + 1, dtype=torch.float32)
      dense[0, row, cols] = values
      dense[1, row, cols] = -0.25 * values
  packed_jacobian_scatter_rows_torch(
      torch, storage, 2, layout, pattern, dense, 0)

  active = torch.ones((2, descriptor.nr), dtype=torch.float32)
  R = torch.linspace(.5, 1.5, descriptor.nr).expand(2, -1).contiguous()
  rows = {
      "J_packed": storage,
      "jacobian_layout": layout,
      "jacobian_pattern": pattern,
      "R": R,
      "ar": torch.linspace(-.1, .2, descriptor.nr).expand(2, -1).contiguous(),
      "lo": torch.full_like(active, -torch.inf),
      "hi": torch.full_like(active, torch.inf),
      "active": active,
      "contact_friction": torch.as_tensor(
          np.array(descriptor.contact_friction, dtype=np.float32, copy=True)
          .reshape(-1, 5)),
      "contact_mask": torch.ones((2, descriptor.ncontacts_max), dtype=torch.float32),
  }
  qacc = torch.linspace(-.75, .9, 2 * descriptor.nv).reshape(
      2, descriptor.nv).to(torch.float32)
  packed_result = inverse_constraint_force(rows, qacc, descriptor)

  dense_rows = dict(rows)
  dense_rows.pop("J_packed")
  dense_rows.pop("jacobian_layout")
  dense_rows.pop("jacobian_pattern")
  dense_rows["J"] = materialize_packed_jacobian_torch(
      torch, storage, 2, layout, pattern)
  dense_result = inverse_constraint_force(dense_rows, qacc, descriptor)
  torch.testing.assert_close(packed_result, dense_result, rtol=1e-6, atol=2e-6)


def test_sparse_scatter_fails_closed_on_unsupported_nonzero_per_world():
  import torch

  model = _model(jacobian="sparse")
  descriptor = lower_coupled_constraints(model)
  pattern = descriptor.jacobian_pattern
  layout = PackedJacobianLayout.create(
      descriptor.nr, descriptor.nv, pattern=pattern)
  storage, _ = packed_jacobian_device_storage(
      torch, torch.device("cpu"), 2, pattern)
  row = next(r for r in range(descriptor.nr)
             if pattern.row_columns(r).size < descriptor.nv)
  support = set(map(int, pattern.row_columns(row)))
  off_pattern = next(d for d in range(descriptor.nv) if d not in support)
  source = torch.zeros((2, 1, descriptor.nv), dtype=torch.float32)
  source[1, 0, off_pattern] = 2.0
  packed_jacobian_scatter_rows_torch(
      torch, storage, 2, layout, pattern, source, row, row_count=1)
  records = storage.view(torch.int32).reshape(2, layout.stride_words)
  torch.testing.assert_close(records[:, 10], torch.tensor([0, 1], dtype=torch.int32))
  assert packed_jacobian_read(
      storage[:layout.stride_words].numpy(), layout, row, off_pattern) == 0.0


def test_spatial_tendon_rows_use_compiled_wrap_ancestry_not_all_dofs():
  model = mujoco.MjModel.from_xml_string("""
    <mujoco><option gravity="0 0 0" jacobian="dense"/>
      <worldbody>
        <body name="a" pos="0 0 1"><joint type="slide" axis="1 0 0"/>
          <geom type="sphere" size=".1" mass="1"/><site name="s1"/></body>
        <body name="b" pos="1 0 1"><joint type="slide" axis="0 1 0"/>
          <geom type="sphere" size=".1" mass="1"/><site name="s2"/></body>
        <body name="c" pos="0 2 1"><joint type="slide" axis="0 0 1"/>
          <geom type="sphere" size=".1" mass="1"/><site name="s3"/></body>
        <body name="d" pos="1 2 1"><joint type="slide" axis="1 0 0"/>
          <geom type="sphere" size=".1" mass="1"/><site name="s4"/></body>
      </worldbody>
      <tendon>
        <spatial name="t1" frictionloss=".1"><site site="s1"/>
          <site site="s2"/></spatial>
        <spatial name="t2"><site site="s3"/><site site="s4"/></spatial>
      </tendon>
      <equality><tendon tendon1="t1" tendon2="t2"
                        polycoef="0 1 0 0 0"/></equality>
    </mujoco>
  """)
  descriptor = lower_coupled_constraints(model)
  pattern = compile_constraint_jacobian_pattern(model, descriptor)
  # The first spatial tendon friction row depends on its two wrapped body
  # chains, but not on the independent second tendon.
  friction_row = descriptor.ten_base
  np.testing.assert_array_equal(pattern.row_columns(friction_row), [0, 1])
  # The equality couples both tendon lengths, so its support is their union.
  np.testing.assert_array_equal(pattern.row_columns(0), [0, 1, 2, 3])
  data = mujoco.MjData(model)
  mujoco.mj_forward(model, data)
  native_j = np.asarray(data.efc_J).reshape(data.nefc, model.nv)
  tendon_friction = np.flatnonzero(
      data.efc_type == int(mujoco.mjtConstraint.mjCNSTR_FRICTION_TENDON))
  assert tendon_friction.size == 1
  assert set(map(int, np.flatnonzero(native_j[tendon_friction[0]]))) <= set(
      map(int, pattern.row_columns(friction_row)))


def test_sparse_pattern_covers_interpolated_flex_contact_dofs():
  """Q1/Q2 volume and shell rows include non-corner interpolation DOFs."""
  shell_attr = ' thickness=".01" elastic2d="bend"'
  for dof, shell in (("trilinear", False), ("quadratic", False),
                     ("trilinear", True), ("quadratic", True)):
    bend = shell_attr if shell else ""
    model = mujoco.MjModel.from_xml_string(f"""
      <mujoco><option gravity="0 0 0" jacobian="dense"/>
        <worldbody>
          <geom name="floor" type="plane" size="0 0 .1"
                contype="0" conaffinity="1" condim="1"/>
          <flexcomp name="interp" type="grid" count="3 3 3"
                    pos="0 0 -.02" spacing=".1 .1 .1" mass="1" dim="3"
                    dof="{dof}">
            <contact contype="1" conaffinity="0" selfcollide="none"
                     condim="1"/>
            <elasticity young="100" poisson=".2" damping=".1"{bend}/>
          </flexcomp>
        </worldbody>
      </mujoco>
    """)
    descriptor = lower_coupled_constraints(
        model, limits=CapacityLimits(max_nv=81, max_rows=512, max_batch=2))
    pattern = compile_constraint_jacobian_pattern(model, descriptor)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    assert data.ncon > 0
    assert int(model.flex_nodenum[0]) > 0
    flex = descriptor.flex_contact_descriptor
    flex_rows_by_vertex = {
        int(vertex): int(descriptor.flex_contact_base) + int(start)
        for fid, vertex, start in zip(flex.flex1, flex.vert1, flex.row_start)
        if int(fid) == 0 and int(vertex) >= 0
    }
    native_j = np.asarray(data.efc_J).reshape(data.nefc, model.nv)
    all_node_bodies = set(map(
        int, np.asarray(model.flex_nodebodyid, dtype=np.int32)[
            int(model.flex_nodeadr[0]):
            int(model.flex_nodeadr[0] + model.flex_nodenum[0])]))
    node_dofs = {dof for dof, body in enumerate(model.dof_bodyid)
                 if int(body) in all_node_bodies}
    assert len(node_dofs) > 3
    for contact in data.contact[:data.ncon]:
      vertex = int(contact.vert[1])
      row = flex_rows_by_vertex[vertex]
      support = set(map(int, pattern.row_columns(row)))
      reference = set(map(int, np.flatnonzero(native_j[int(contact.efc_address)])))
      assert reference <= support
      # A Q1/Q2 interpolation field can use nodes beyond the contacted corner.
    assert node_dofs <= support


def test_runtime_csr_row_clear_mutates_only_selected_packed_value_spans():
  """Execute the production clearer on real tensor storage, not a fake map."""
  import torch
  from types import SimpleNamespace
  from mujoco_metal.coupled_constraints import MetalCoupledConstraints

  model = _model(jacobian="sparse")
  descriptor = lower_coupled_constraints(model)
  pattern = compile_constraint_jacobian_pattern(model, descriptor)
  storage, layout = packed_jacobian_device_storage(
      torch, torch.device("cpu"), 2, pattern)
  values_begin = layout.values_offset
  values_end = layout.stride_words
  for world in range(2):
    base = world * layout.stride_words
    storage[base + values_begin:base + values_end].fill_(1.0)

  cc = MetalCoupledConstraints.__new__(MetalCoupledConstraints)
  cc._torch = torch
  cc._workspace = {"workspace_J": storage}
  cc.batch_size = 2
  cc.descriptor = SimpleNamespace(nr=descriptor.nr, nv=descriptor.nv)
  cc._jacobian_pattern = pattern
  cc._jacobian_layout = layout
  row_start, row_stop = 1, min(4, descriptor.nr)
  first = int(pattern.row_offsets[row_start])
  last = int(pattern.row_offsets[row_stop])
  before_i32 = storage.view(torch.int32).clone()
  cc._clear_jacobian_rows(row_start, row_stop)

  after_i32 = storage.view(torch.int32)
  for world in range(2):
    base = world * layout.stride_words
    values = storage[base + values_begin:base + values_end]
    assert torch.count_nonzero(values[first:last]).item() == 0
    assert torch.count_nonzero(values[:first]).item() == first
    assert torch.count_nonzero(values[last:]).item() == layout.nnz - last
    metadata_end = base + values_begin
    assert torch.equal(after_i32[base:metadata_end],
                       before_i32[base:metadata_end])


@pytest.mark.gpu
@pytest.mark.parametrize("cone", ("elliptic", "pyramidal"))
@pytest.mark.skipif(os.getenv("MUJOCO_METAL_RUN_GPU") != "1",
                    reason="public sparse constraint stepping requires GPU opt-in")
def test_native_sparse_public_cc_step_uses_packed_rows_without_dense_j(cone):
  """Exercise packed J through public stepping, all cones/row families, B2."""
  pytest.importorskip("torch")
  from mujoco_metal.simulation import MetalSimulation

  dense_model = _sparse_runtime_model("dense", cone)
  sparse_model = _sparse_runtime_model("sparse", cone)
  assert not mujoco.mj_isSparse(dense_model)
  assert mujoco.mj_isSparse(sparse_model)
  qpos = np.tile(np.asarray(sparse_model.qpos0, np.float64), (2, 1))
  qvel = np.tile(np.linspace(-.012, .017, sparse_model.nv), (2, 1))
  # The final condim-6 pair is a deliberate gap in world 1. All other pair
  # slots remain active, so both live and inactive packed row writes execute.
  jd = mujoco.mj_name2id(sparse_model, mujoco.mjtObj.mjOBJ_JOINT, "jd")
  qpos[1, int(sparse_model.jnt_qposadr[jd]) + 2] += 1.25
  dense_sim = MetalSimulation(
      dense_model, batch_size=2, qpos=qpos.astype(np.float32),
      qvel=qvel.astype(np.float32), profile="integrated_euler_v1")
  sparse_sim = MetalSimulation(
      sparse_model, batch_size=2, qpos=qpos.astype(np.float32),
      qvel=qvel.astype(np.float32), profile="integrated_euler_v1")
  cc = sparse_sim._coupled_constraints
  assert cc is not None and cc._jacobian_layout.mode == PACKED_J_CSR
  descriptor = cc.descriptor
  assert descriptor.n_eq_rows > 0
  assert descriptor.ten_friction_rows > 0
  contact_condim = np.asarray(
      descriptor.contact_condim_packed, dtype=np.int32).reshape(-1, 3)[:, 0]
  assert set(contact_condim.tolist()) >= {1, 3, 4, 6}
  # Persistent stepping storage is packed; the explicitly dense query created
  # below is test-only and never feeds a solver step.
  assert cc._workspace["contact_jacobian"].numel() == 1
  assert cc._workspace["position_cache_contact_jacobian"].numel() == 1
  assert cc._workspace["workspace_J"].numel() == 2 * cc._jacobian_layout.stride_words
  assert cc._jacobian_pattern.nnz < descriptor.nr * descriptor.nv

  refs = [mujoco.MjData(sparse_model) for _ in range(2)]
  for env, data in enumerate(refs):
    data.qpos[:] = qpos[env]
    data.qvel[:] = qvel[env]
    mujoco.mj_forward(sparse_model, data)
  floor = mujoco.mj_name2id(
      sparse_model, mujoco.mjtObj.mjOBJ_GEOM, "floor")
  gap_geom = mujoco.mj_name2id(
      sparse_model, mujoco.mjtObj.mjOBJ_GEOM, "gd")
  assert any(int(c.geom[0]) == floor and int(c.geom[1]) == gap_geom
             for c in refs[0].contact[:refs[0].ncon])
  assert not any(int(c.geom[0]) == floor and int(c.geom[1]) == gap_geom
                 for c in refs[1].contact[:refs[1].ncon])

  for _ in range(2):
    dense_status = dense_sim.step()
    sparse_status = sparse_sim.step()
    np.testing.assert_array_equal(dense_status.cpu().numpy(), [0, 0])
    np.testing.assert_array_equal(sparse_status.cpu().numpy(), [0, 0])
    for data in refs:
      mujoco.mj_step(sparse_model, data)
  dense_state = dense_sim.state.snapshot()
  sparse_state = sparse_sim.state.snapshot()
  for env, data in enumerate(refs):
    np.testing.assert_allclose(dense_state.qpos[env], data.qpos,
                               rtol=5e-5, atol=5e-6)
    np.testing.assert_allclose(sparse_state.qpos[env], data.qpos,
                               rtol=5e-5, atol=5e-6)
    np.testing.assert_allclose(dense_state.qvel[env], data.qvel,
                               rtol=2e-4, atol=2e-5)
    np.testing.assert_allclose(sparse_state.qvel[env], data.qvel,
                               rtol=2e-4, atol=2e-5)

  # Ordinary stepping does not promise the private assembly-only activity
  # slice as a query record. Capture the current public CONSTRAINT stage and
  # inspect its canonical row view, which owns activity and contact-mask
  # metadata for the same prepared state.
  from mujoco_metal import native_api as api
  prepared = api.mj_forward(sparse_sim, skipsensor=True)
  rows = prepared["constraint"]["canonical_rows"]
  active = rows["active"].detach().cpu().numpy() > .5
  # Every inactive logical row has a fully zero packed numeric segment.
  packed_dense = cc.materialize_jacobian().detach().cpu().numpy()
  assert np.all(packed_dense[~active] == 0.0)
  contact_mask = rows["contact_mask"].detach().cpu().numpy()
  gap_pair = next(i for i, (a, b) in enumerate(
      zip(descriptor.geom1, descriptor.geom2))
      if int(a) == floor and int(b) == gap_geom)
  gap_slot = int(descriptor.pair_contact_offset[gap_pair])
  assert contact_mask[0, gap_slot] > 0.0
  assert contact_mask[1, gap_slot] == 0.0


@pytest.mark.parametrize("mode", (PACKED_J_DENSE, PACKED_J_CSR))
def test_primal_row_dot_products_match_dense_and_csr_headers(mode):
  """CPU execution oracle for the packed row-dot consumer contract."""
  model = _model(jacobian="sparse")
  descriptor = lower_coupled_constraints(model)
  pattern = descriptor.jacobian_pattern if mode == PACKED_J_CSR else None
  layout = PackedJacobianLayout.create(
      descriptor.nr, descriptor.nv, pattern=pattern, mode=mode)
  batch = 2
  words = np.zeros(batch * layout.stride_words, dtype=np.int32)
  initialize_packed_jacobian_records(words, batch, layout, pattern)
  records = words.view(np.float32)
  rng = np.random.default_rng(24019)
  dense = np.zeros((batch, descriptor.nr, descriptor.nv), np.float32)
  for world in range(batch):
    base = world * layout.stride_words
    record = records[base:base + layout.stride_words]
    for row in range(descriptor.nr):
      cols = (pattern.row_columns(row) if pattern is not None
              else np.arange(descriptor.nv, dtype=np.int32))
      values = rng.normal(size=len(cols)).astype(np.float32)
      for col, value in zip(cols, values):
        assert packed_jacobian_write(record, layout, row, int(col), value)
        dense[world, row, col] = value
    vector = rng.normal(size=descriptor.nv).astype(np.float32)
    for row in range(descriptor.nr):
      # This mirrors the device accessor loop: absent CSR columns contribute
      # exact zero, while dense and packed records share the same row contract.
      actual = sum(float(packed_jacobian_read(record, layout, row, dof))
                   * float(vector[dof]) for dof in range(descriptor.nv))
      expected = float(np.dot(dense[world, row].astype(np.float64),
                              vector.astype(np.float64)))
      assert abs(actual - expected) <= 2e-6 * max(1.0, abs(expected))


def test_primal_coupled_shader_never_treats_packed_record_as_dense_rows():
  shader = (__import__("pathlib").Path(__file__).parents[1]
            / "mujoco_metal" / "shaders" / "coupled_constraints.metal")
  source = shader.read_text()
  # These are the exact primal Hessian helpers and component RHS solve that
  # previously bypassed the canonical header/CSR map.
  hessian_apply = source.split("inline void primal_hessian_apply_elliptic", 1)[1].split(
      "__attribute__((noinline))", 1)[0]
  curvature = source.split("inline float primal_accel_curvature_elliptic", 1)[1].split(
      "struct PrimalLinePoint", 1)[0]
  assert "J + row * nv" not in hessian_apply
  assert "primal_ccj_compensated_dot_pair" in hessian_apply
  assert "J + row * nv" not in curvature
  assert "J + (start + k) * nv" not in curvature
  assert "primal_ccj_compensated_dot_pair" in curvature
  assert "component_row_rhs[dof] = ccj_get(J_world" in source
  assert "J_world + row * nv" not in source
  assert "J_world + b * nv" not in source
