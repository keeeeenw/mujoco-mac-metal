# Copyright 2026 The MuJoCo Metal contributors
# Licensed under the Apache License, Version 2.0.

"""CPU admission tests for profile-selected coupled mass storage."""

import mujoco
import numpy as np
import pytest
from types import SimpleNamespace

torch = pytest.importorskip("torch")

from mujoco_metal import coupled_constraints
from mujoco_metal.capacity import (
    CapacityLimits, CapacityOverflow, estimate_capacity)
from mujoco_metal.coupled_constraints import (
    MetalCoupledConstraints, lower_coupled_constraints)
from mujoco_metal.mass_layout import compile_tree_mass_layout
from mujoco_metal.model import actuator_tendon_inheritance
from mujoco_metal.stepping import validate_stepping_profile


def _disconnected_model(nv=64):
  bodies = "".join(
      f'<body name="b{i}" pos="{i * .01} 0 0">'
      f'<joint name="j{i}" type="slide" axis="0 1 0"/>'
      '<inertial pos="0 0 0" mass=".2" '
      'diaginertia=".001 .001 .001"/></body>'
      for i in range(nv))
  return mujoco.MjModel.from_xml_string(
      '<mujoco><option><flag contact="disable"/></option>'
      f'<worldbody>{bodies}</worldbody></mujoco>')


def _component_layout(model):
  inherited, _, _ = actuator_tendon_inheritance(model)
  return compile_tree_mass_layout(
      model.body_treeid, model.dof_bodyid,
      tendon_treeid=model.tendon_treeid,
      tendon_treenum=model.tendon_treenum,
      tendon_armature=(np.asarray(model.tendon_armature, dtype=np.float64)
                       + inherited),
      tendon_j_rowadr=model.ten_J_rowadr,
      tendon_j_rownnz=model.ten_J_rownnz,
      tendon_j_colind=model.ten_J_colind,
      mass_rowadr=model.M_rowadr,
      mass_rownnz=model.M_rownnz,
      mass_colind=model.M_colind)


def _limits(memory):
  return CapacityLimits(max_nv=64, max_pairs=128, max_slots=128,
                        max_rows=1024, max_batch=2,
                        memory_budget_bytes=memory)


def test_disconnected_profile_estimate_and_admission_use_selected_storage(monkeypatch):
  model = _disconnected_model()
  generous = _limits(1 << 40)
  dense_descriptor = lower_coupled_constraints(
      model, limits=generous, mass_storage="dense")
  sparse_descriptor = lower_coupled_constraints(
      model, limits=generous, mass_storage="block_sparse")
  dense = estimate_capacity(
      model, 1, dense_descriptor.npairs, dense_descriptor.ncontacts_max,
      dense_descriptor.nr, nr_joint=dense_descriptor.nr_joint,
      mass_storage="dense",
      jacobian_kind=dense_descriptor.jacobian_kind,
      jacobian_nnz=(None if dense_descriptor.jacobian_pattern is None else
                    dense_descriptor.jacobian_pattern.nnz))
  sparse = estimate_capacity(
      model, 1, sparse_descriptor.npairs, sparse_descriptor.ncontacts_max,
      sparse_descriptor.nr, nr_joint=sparse_descriptor.nr_joint,
      mass_storage="block_sparse",
      jacobian_kind=sparse_descriptor.jacobian_kind,
      jacobian_nnz=(None if sparse_descriptor.jacobian_pattern is None else
                    sparse_descriptor.jacobian_pattern.nnz))
  assert sparse.memory_bytes < dense.memory_bytes
  # Put the budget strictly between the two complete estimates. Dense must
  # fail in host lowering before MPS setup; the selected sparse profile must
  # pass the same admission and reach the deliberate no-MPS boundary.
  budget = (sparse.memory_bytes + dense.memory_bytes) // 2
  assert sparse.memory_bytes < budget < dense.memory_bytes
  mps_calls = []
  allocations = []
  monkeypatch.setattr(torch.backends.mps, "is_available",
                      lambda: mps_calls.append("mps") or False)
  monkeypatch.setattr(MetalCoupledConstraints, "_tensor",
                      lambda self, value: allocations.append(value))
  with pytest.raises(CapacityOverflow, match="estimated device memory"):
    MetalCoupledConstraints(
        model, limits=_limits(budget), mass_storage="dense")
  assert mps_calls == []
  assert allocations == []

  with pytest.raises(RuntimeError, match="require PyTorch MPS"):
    MetalCoupledConstraints(
        model, limits=_limits(budget), mass_storage="block_sparse",
        component_layout=_component_layout(model))
  assert mps_calls == ["mps"]
  assert allocations == []

  sparse_batch2 = estimate_capacity(
      model, 2, sparse_descriptor.npairs, sparse_descriptor.ncontacts_max,
      sparse_descriptor.nr, nr_joint=sparse_descriptor.nr_joint,
      mass_storage="block_sparse",
      jacobian_kind=sparse_descriptor.jacobian_kind,
      jacobian_nnz=(None if sparse_descriptor.jacobian_pattern is None else
                    sparse_descriptor.jacobian_pattern.nnz))
  budget_batch2 = (sparse.memory_bytes + sparse_batch2.memory_bytes) // 2
  assert sparse.memory_bytes < budget_batch2 < sparse_batch2.memory_bytes
  with pytest.raises(CapacityOverflow, match="estimated device memory"):
    MetalCoupledConstraints(
        model, batch_size=2, limits=_limits(budget_batch2),
        mass_storage="block_sparse", component_layout=_component_layout(model))
  # The batch-one descriptor admission was insufficient; the exact batch-two
  # check rejects before another MPS probe or component-layout upload.
  assert mps_calls == ["mps"]
  assert allocations == []


def test_sparse_constructor_requires_layout_before_lower_or_allocation(monkeypatch):
  model = _disconnected_model(4)
  lower_calls = []
  monkeypatch.setattr(
      coupled_constraints, "lower_coupled_constraints",
      lambda *args, **kwargs: lower_calls.append((args, kwargs)))
  allocations = []
  monkeypatch.setattr(MetalCoupledConstraints, "_tensor",
                      lambda self, value: allocations.append(value))
  with pytest.raises(ValueError, match="require a compiled component_layout"):
    MetalCoupledConstraints(model, mass_storage="block_sparse")
  assert lower_calls == []
  assert allocations == []


def test_component_layout_is_checked_before_native_runtime_probe(monkeypatch):
  model = _disconnected_model(4)
  mps_calls = []
  allocations = []
  monkeypatch.setattr(torch.backends.mps, "is_available",
                      lambda: mps_calls.append("mps") or False)
  monkeypatch.setattr(MetalCoupledConstraints, "_tensor",
                      lambda self, value: allocations.append(value))
  invalid = dict(_component_layout(model))
  invalid["component_dof_ids"] = np.zeros(model.nv, dtype=np.int32)
  with pytest.raises(ValueError, match="must permute"):
    MetalCoupledConstraints(
        model, mass_storage="block_sparse", component_layout=invalid)
  assert mps_calls == []
  assert allocations == []

  # This plan is internally valid but merges four model components into one
  # dense block. It must not use the smaller model-derived sparse estimate.
  mismatched = dict(_component_layout(model))
  mismatched.update(
      ncomponent=1, nnz=model.nv * model.nv,
      component_dof_offsets=np.asarray([0, model.nv], dtype=np.int32),
      component_dof_ids=np.arange(model.nv, dtype=np.int32),
      component_mass_offsets=np.asarray([0], dtype=np.int32),
      dof_component=np.zeros(model.nv, dtype=np.int32),
      dof_local_index=np.arange(model.nv, dtype=np.int32))
  with pytest.raises(ValueError, match="does not match the model"):
    MetalCoupledConstraints(
        model, mass_storage="block_sparse", component_layout=mismatched)
  assert mps_calls == []
  assert allocations == []


def test_scalable_profile_lowering_threads_sparse_mode(monkeypatch):
  model = mujoco.MjModel.from_xml_string(
      '<mujoco><option><flag contact="disable"/></option><worldbody>'
      '<body><joint type="slide" axis="1 0 0"/><geom type="sphere" '
      'size=".1" mass="1"/></body></worldbody></mujoco>')
  observed = []
  original = coupled_constraints.lower_coupled_constraints

  def record_mode(*args, **kwargs):
    observed.append(kwargs.get("mass_storage", "dense"))
    return original(*args, **kwargs)

  monkeypatch.setattr(coupled_constraints, "lower_coupled_constraints", record_mode)
  validate_stepping_profile(
      model, profile="integrated_scalable_v1", limits=_limits(1 << 30))
  assert observed == ["block_sparse"]


def _small_valid_layout():
  return {
      "ncomponent": 2,
      "nnz": 5,
      "component_dof_offsets": np.asarray([0, 2, 3], dtype=np.int32),
      "component_dof_ids": np.asarray([0, 1, 2], dtype=np.int32),
      "component_mass_offsets": np.asarray([0, 4], dtype=np.int32),
      "dof_component": np.asarray([0, 0, 1], dtype=np.int32),
      "dof_local_index": np.asarray([0, 1, 0], dtype=np.int32),
  }


def _fake_component_program(batch, allocations, prepared):
  program = MetalCoupledConstraints.__new__(MetalCoupledConstraints)
  program.descriptor = SimpleNamespace(nv=3)
  program.batch_size = batch
  program._assembly_generation = 0
  program._position_context_valid = False
  program._workspace = None
  program._component_operator_layout_device = None
  program._constants = {}
  program._torch = SimpleNamespace(
      int32="int32",
      as_tensor=lambda value, **kwargs: allocations.append(
          (np.asarray(value).copy(), kwargs)) or np.asarray(value).copy())
  program._device = "fake-device"
  program._solver_dimension_tensor = lambda: "packed-dimensions"
  program.prepare_workspace = lambda actual_batch: prepared.append(
      (actual_batch, program._component_mass_storage,
       program._component_operator_layout_device is not None)) or setattr(
           program, "_workspace", object())
  return program


def test_configure_component_layout_executes_cpu_fake_runtime_before_commit():
  allocations, prepared = [], []
  program = _fake_component_program(2, allocations, prepared)
  program.configure_component_layout(_small_valid_layout())
  assert program._component_mass_storage == "block_sparse"
  assert program._component_operator_ncomponent == 2
  assert program._component_operator_nnz == 5
  assert len(allocations) == 1
  assert allocations[0][0].size == 14
  assert prepared == [(2, "block_sparse", True)]
  assert program._constants["solver_dims"] == "packed-dimensions"


def test_component_tail_overflow_precedes_device_upload_and_workspace():
  allocations, prepared = [], []
  program = _fake_component_program(np.iinfo(np.int32).max,
                                   allocations, prepared)
  with pytest.raises(ValueError, match="tail exceeds Metal int32"):
    program.configure_component_layout(_small_valid_layout())
  assert allocations == []
  assert prepared == []


class _FakeTensor:
  def __init__(self, values, *, dtype="float32", device="fake"):
    self.array = np.asarray(values)
    self.dtype = dtype
    self.device = device

  @property
  def shape(self):
    return self.array.shape

  def numel(self):
    return self.array.size

  def is_contiguous(self):
    return self.array.flags.c_contiguous

  def reshape(self, *shape):
    return _FakeTensor(self.array.reshape(*shape), dtype=self.dtype,
                       device=self.device)

  def copy_(self, source):
    src = source.array if isinstance(source, _FakeTensor) else np.asarray(source)
    self.array[...] = src
    return self

  def fill_(self, value):
    self.array.fill(value)
    return self

  def __getitem__(self, key):
    return _FakeTensor(self.array[key], dtype=self.dtype, device=self.device)

  def __setitem__(self, key, value):
    self.array[key] = value.array if isinstance(value, _FakeTensor) else value


class _StopAfterCommonDispatch(RuntimeError):
  pass


def _candidate_dispatch_fixture():
  """CPU dispatch seam for persistent buffers; not a physics oracle."""

  class FakeLibrary:
    def clear_contact_pair_counts(self, counts, dims, **_):
      npairs, batch = int(dims.array[1]), int(dims.array[3])
      mask_index = 10 + npairs + 1 + int(dims.array[2])
      mask_offset = int(dims.array[mask_index])
      mask = dims.array[mask_offset:mask_offset + batch]
      counts.array.reshape(batch, npairs)[mask != 0] = 0

    def clear_contact_candidate_workspace(self, row_data, frame, dims, **_):
      return None

    def clear_selected_jacobian_values(self, destination, dims, **_):
      # The test stops before the physical contact kernel; this dispatch only
      # models the selected-row clear precondition.
      return None

    def clear_selected_world_int(self, destination, dims, **_):
      b, width, stride, offset = (int(dims.array[i]) for i in (0, 1, 3, 5))
      mask = dims.array[6:6 + b]
      values = destination.array.reshape(b, stride)
      values[mask != 0, offset:offset + width] = 0

  class FakeCompaction:
    def run(self, _mask, *, world_mask):
      assert world_mask.array.tolist() == [1, 0]
      return SimpleNamespace(
          logical_to_packed=_FakeTensor([[0], [0]], dtype="int32"))

  program = MetalCoupledConstraints.__new__(MetalCoupledConstraints)
  program.batch_size = 2
  program._torch = SimpleNamespace(
      Tensor=_FakeTensor, int32="int32", float32="float32")
  program._device = "fake"
  program._jacobian_layout = SimpleNamespace(mode=0)
  program.descriptor = SimpleNamespace(
      npairs=1, ncontacts_max=1, nr=1, nv=1, ngeom=2)
  program._workspace = {
      "contact_row_data": _FakeTensor(np.zeros((2, 36), dtype=np.float32)),
      "contact_frame": _FakeTensor(np.zeros((2, 12), dtype=np.float32)),
      "contact_jacobian": _FakeTensor(np.zeros((2, 6), dtype=np.float32)),
      "pair_mask": _FakeTensor(np.zeros(2, dtype=np.int32), dtype="int32"),
      "out_status": _FakeTensor([7, 9], dtype="int32"),
      "contact_pair_count": _FakeTensor([13, 17], dtype="int32"),
  }
  program._constants = {
      "pair_contact_offsets_dims": _FakeTensor(np.zeros(16, dtype=np.int32),
                                                dtype="int32"),
      "solver_dims": _FakeTensor([1, 0], dtype="int32"),
      "geom_rbound": _FakeTensor([1.0]),
      "pair_geoms": _FakeTensor([0, 1], dtype="int32"),
      "pair_margin_gap": _FakeTensor([0.0, 0.0]),
      "geom_size": _FakeTensor([1.0, 1.0, 1.0]),
      "geom_type": _FakeTensor([1, 7], dtype="int32"),
      "mesh_hull": _FakeTensor(np.zeros(8)),
      "mesh_hull_info": _FakeTensor(np.zeros(8, dtype=np.int32),
                                     dtype="int32"),
  }
  from mujoco_metal.coupled_constraints import _pack_contact_dims_offsets
  packed = _FakeTensor(_pack_contact_dims_offsets(
      [1, 1, 1, 2, 0, 0, 2, 0, 0, 0], [0, 1], [0]), dtype="int32")
  program._constants["pair_contact_offsets_dims"] = packed
  # Like the actual constructor, retain a view of the packed mask suffix.
  program._pair_contact_world_mask = packed[14:16]
  program._pair_contact_jclear_dims = _FakeTensor(
      np.zeros(6, dtype=np.int32), dtype="int32")
  program._position_refresh_copy_dims = _FakeTensor(
      np.zeros(8, dtype=np.int32), dtype="int32")
  program._world_mask_offset = 0
  program._library = FakeLibrary()
  program._clear_contact_pair_counts = program._library.clear_contact_pair_counts
  program._reset_jacobian_write_status = lambda **_: None
  program._clear_jacobian_rows = lambda *_, **__: None
  program._broadphase_kernel = lambda *_, **__: None
  program._pair_compaction = FakeCompaction()
  program._common_ccd_hfield_mesh_kernel = lambda *_, **__: (
      (_ for _ in ()).throw(_StopAfterCommonDispatch()))
  program._common_ccd_rigid_kernel = None
  program._contact_kernel = lambda *_, **__: pytest.fail(
      "dispatch should stop at the common kernel witness")
  poses = {
      "geom_pos": _FakeTensor(np.zeros((2, 2, 3), dtype=np.float32)),
      "geom_quat": _FakeTensor(np.zeros((2, 2, 4), dtype=np.float32)),
      "geom_pos_low": _FakeTensor(np.zeros((2, 2, 3), dtype=np.float32)),
      "geom_pos_tail": _FakeTensor(np.zeros((2, 2, 3), dtype=np.float32)),
      "geom_xmat": _FakeTensor(np.zeros((2, 2, 9), dtype=np.float32)),
      "geom_xmat_low": _FakeTensor(np.zeros((2, 2, 9), dtype=np.float32)),
      "geom_xmat_tail": _FakeTensor(np.zeros((2, 2, 9), dtype=np.float32)),
  }
  mask = _FakeTensor([1, 0], dtype="int32")
  return program, poses, mask


def test_generate_candidates_clears_common_status_only_for_selected_worlds():
  """Execute the production dispatch seam with fake kernels, no MPS physics."""
  program, poses, mask = _candidate_dispatch_fixture()
  with pytest.raises(_StopAfterCommonDispatch):
    program.generate_candidates(poses, _FakeTensor(np.zeros((2, 1))),
                                world_mask=mask)
  np.testing.assert_array_equal(program._workspace["out_status"].array,
                                np.asarray([0, 9], dtype=np.int32))
  np.testing.assert_array_equal(program._pair_contact_world_mask.array,
                                np.asarray([1, 0], dtype=np.int32))
  np.testing.assert_array_equal(program._workspace["contact_pair_count"].array,
                                np.asarray([0, 17], dtype=np.int32))


@pytest.mark.parametrize("field,width", [
    ("geom_pos_low", 3), ("geom_pos_tail", 3), ("geom_xmat", 9),
    ("geom_xmat_low", 9), ("geom_xmat_tail", 9),
])
@pytest.mark.parametrize("invalid", [
    "missing", "not_tensor", "shape", "dtype", "device", "noncontiguous",
])
def test_generate_candidates_rejects_invalid_pose_before_any_mutation(
    field, width, invalid):
  program, poses, mask = _candidate_dispatch_fixture()
  # Nonzero backing makes a premature workspace clear observable.
  for value in program._workspace.values():
    value.array.fill(17)
  before = {name: value.array.copy()
            for name, value in program._workspace.items()}
  mask_before = program._pair_contact_world_mask.array.copy()
  constant_before = {name: value.array.copy()
                     for name, value in program._constants.items()}
  if invalid == "missing":
    del poses[field]
  elif invalid == "not_tensor":
    poses[field] = np.zeros((2, 2, width), dtype=np.float32)
  elif invalid == "shape":
    poses[field] = _FakeTensor(np.zeros((2, 1, width)))
  elif invalid == "dtype":
    poses[field].dtype = "float64"
  elif invalid == "device":
    poses[field].device = "foreign"
  else:
    poses[field] = _FakeTensor(np.zeros((2, 2, 2 * width))[:, :, ::2])
    assert not poses[field].is_contiguous()
  calls = []
  program._reset_jacobian_write_status = lambda **_: calls.append("status")
  program._clear_jacobian_rows = lambda *_, **__: calls.append("rows")
  program._broadphase_kernel = lambda *_, **__: calls.append("broadphase")
  program._common_ccd_hfield_mesh_kernel = lambda *_, **__: calls.append("common")
  with pytest.raises(ValueError, match=field):
    program.generate_candidates(poses, _FakeTensor(np.zeros((2, 1))),
                                world_mask=mask)
  assert calls == []
  np.testing.assert_array_equal(program._pair_contact_world_mask.array,
                                mask_before)
  for name, expected in before.items():
    np.testing.assert_array_equal(program._workspace[name].array, expected)
  for name, expected in constant_before.items():
    np.testing.assert_array_equal(program._constants[name].array, expected)
