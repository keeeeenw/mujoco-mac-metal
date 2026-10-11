# Copyright 2026 The MuJoCo Metal contributors
# Licensed under the Apache License, Version 2.0.
"""Owned compensated RHS accumulation: signed-zero, validation, scratch.

`_rhs_accumulate` must preserve unselected words bitwise (including -0),
validate before any write, and use only admitted owned scratch (no
batch-sized hidden temporaries). Covers PGS/CG/Newton profiles via ordinary
simulation stepping plus direct accumulation probes.
"""

import os

import numpy as np
import pytest

import mujoco

torch = pytest.importorskip("torch")

_NEEDS_GPU = pytest.mark.skipif(
    os.getenv("MUJOCO_METAL_RUN_GPU") != "1", reason="native accumulation")
_GPU = pytest.mark.gpu


def _model(solver="CG"):
  return mujoco.MjModel.from_xml_string(f"""<mujoco>
    <option gravity="0 0 0" solver="{solver}" timestep="0.001"><flag contact="disable"/></option>
    <worldbody><body><joint name="j" type="slide" axis="1 0 0"/>
      <geom type="sphere" size=".1" mass="1"/></body></worldbody>
    <actuator><motor joint="j" ctrllimited="false" forcelimited="false"/></actuator>
  </mujoco>""")


@_GPU
@_NEEDS_GPU
def test_masked_accumulate_preserves_unselected_signed_zero_bitwise():
  """Inactive world low word -0 must survive masked accumulation."""
  import torch
  from mujoco_metal.simulation import MetalSimulation
  model = _model()
  sim = MetalSimulation(model, 2, profile="integrated_euler_v1")
  device = sim._state._device
  hi = torch.zeros((2, model.nv), dtype=torch.float32, device=device)
  lo = torch.zeros((2, model.nv), dtype=torch.float32, device=device)
  # NOTE: MPS index assignment canonicalizes -0 to +0, so set the word via
  # its int32 view (constructor preserves -0 too). This is the exact word
  # the masked path must not disturb.
  lo.view(torch.int32)[1, 0] = -2147483648  # 0x80000000
  assert lo[1, 0].item() == 0.0
  assert np.frombuffer(
      lo[1, 0].detach().cpu().numpy().tobytes(), dtype=np.uint32)[0] == 0x80000000
  err = torch.ones((2, model.nv), dtype=torch.float32, device=device)
  mask = torch.tensor([1, 0], dtype=torch.int32, device=device)
  sim._rhs_accumulate(hi, lo, err, mask=mask)
  # Selected world accumulated; unselected keeps exact -0 word.
  assert hi[0, 0].item() == pytest.approx(1.0)
  got = np.frombuffer(
      lo[1, 0].detach().cpu().numpy().tobytes(), dtype=np.uint32)[0]
  assert got == 0x80000000, hex(int(got))
  got_hi = np.frombuffer(
      hi[1, 0].detach().cpu().numpy().tobytes(), dtype=np.uint32)[0]
  assert got_hi == 0x00000000


@_GPU
@_NEEDS_GPU
def test_accumulate_validates_before_any_write():
  """Bad shapes/dtypes/devices/masks raise and leave hi/lo untouched."""
  import torch
  from mujoco_metal.simulation import MetalSimulation
  model = _model()
  sim = MetalSimulation(model, 2, profile="integrated_euler_v1")
  device = sim._state._device
  hi = torch.ones((2, model.nv), dtype=torch.float32, device=device)
  lo = torch.zeros((2, model.nv), dtype=torch.float32, device=device)
  good = torch.ones((2, model.nv), dtype=torch.float32, device=device)
  hi_before, lo_before = hi.clone(), lo.clone()
  bad_value = torch.ones((2, model.nv + 1), dtype=torch.float32, device=device)
  with pytest.raises(ValueError):
    sim._rhs_accumulate(hi, lo, bad_value)
  bad_mask = torch.tensor([1, 0, 0], dtype=torch.int32, device=device)
  with pytest.raises(ValueError):
    sim._rhs_accumulate(hi, lo, good, mask=bad_mask)
  bad_dtype = torch.ones((2, model.nv), dtype=torch.float64, device="cpu")
  with pytest.raises(ValueError):
    sim._rhs_accumulate(bad_dtype, lo, good)
  assert torch.equal(hi, hi_before) and torch.equal(lo, lo_before)


@_GPU
@_NEEDS_GPU
def test_accumulate_reuses_owned_scratch_without_reallocation():
  """Pre-allocated pool is reused; repeated calls never reallocate."""
  import torch
  from mujoco_metal.simulation import MetalSimulation
  model = _model()
  sim = MetalSimulation(model, 2, profile="integrated_euler_v1")
  device = sim._state._device
  # Pre-allocated once at construction against the admitted budget.
  assert sim._twosum_float is not None
  assert sim._twosum_float.shape == (8, max(2 * model.nv, 1))
  assert sim._twosum_mask is not None
  assert sim._twosum_mask.shape == (2, 1)
  assert sim._twosum_mask.dtype == torch.bool
  first_id = id(sim._twosum_float)
  first_count = sim._twosum_float_count
  hi = torch.zeros((2, model.nv), dtype=torch.float32, device=device)
  lo = torch.zeros((2, model.nv), dtype=torch.float32, device=device)
  val = torch.ones((2, model.nv), dtype=torch.float32, device=device)
  sim._rhs_accumulate(hi, lo, val)
  hi2 = torch.zeros((2, model.nv), dtype=torch.float32, device=device)
  lo2 = torch.zeros((2, model.nv), dtype=torch.float32, device=device)
  sim._rhs_accumulate(hi2, lo2, val)
  assert id(sim._twosum_float) == first_id
  assert sim._twosum_float_count == first_count
  assert sim._twosum_float.numel() == 8 * first_count


@_GPU
@_NEEDS_GPU
def test_native_scratch_bytes_match_admitted_breakdown():
  """Actual device bytes equal the capacity inventory (bool is 1 byte)."""
  from mujoco_metal.capacity import estimate_capacity
  from mujoco_metal.simulation import MetalSimulation
  model = _model()
  batch = 2
  sim = MetalSimulation(model, batch, profile="integrated_euler_v1")
  est = estimate_capacity(model, batch, npairs=0, nslots=0, nr=0)
  table = dict(est.memory_breakdown)
  assert sim._twosum_float.numel() * 4 == table["simulation.twosum_float"]
  assert sim._twosum_mask.numel() * 1 == table["simulation.twosum_mask"]
  assert table["simulation.twosum_float"] == 8 * max(batch * model.nv, 1) * 4


@_GPU
@_NEEDS_GPU
def test_oversized_request_rejected_before_any_write():
  """Requests beyond the admitted scratch fail with hi/lo untouched."""
  import torch
  from mujoco_metal.simulation import MetalSimulation
  model = _model()
  sim = MetalSimulation(model, 2, profile="integrated_euler_v1")
  device = sim._state._device
  nv = int(model.nv)
  hi = torch.ones((3, nv), dtype=torch.float32, device=device)
  lo = torch.zeros((3, nv), dtype=torch.float32, device=device)
  val = torch.ones((3, nv), dtype=torch.float32, device=device)
  hi_before, lo_before = hi.clone(), lo.clone()
  with pytest.raises(ValueError, match="admitted"):
    sim._rhs_accumulate(hi, lo, val)
  assert torch.equal(hi, hi_before) and torch.equal(lo, lo_before)


@_GPU
@_NEEDS_GPU
def test_hi_lo_alias_rejected_value_alias_is_safe():
  """hi/lo sharing storage is rejected; value viewing hi is documented-safe."""
  import torch
  from mujoco_metal.simulation import MetalSimulation
  model = _model()
  sim = MetalSimulation(model, 2, profile="integrated_euler_v1")
  device = sim._state._device
  nv = int(model.nv)
  shared = torch.ones((2, nv), dtype=torch.float32, device=device)
  val = torch.ones((2, nv), dtype=torch.float32, device=device)
  before = shared.clone()
  with pytest.raises(ValueError, match="overlap"):
    sim._rhs_accumulate(shared, shared, val)
  assert torch.equal(shared, before)
  # Value aliasing hi accumulates hi+hi exactly (writes land in scratch first).
  hi = torch.ones((2, nv), dtype=torch.float32, device=device)
  lo = torch.zeros((2, nv), dtype=torch.float32, device=device)
  sim._rhs_accumulate(hi, lo, hi)
  assert torch.equal(hi, torch.full((2, nv), 2.0, device=device))
  assert torch.equal(lo, torch.zeros((2, nv), device=device))


@_GPU
@_NEEDS_GPU
def test_zero_dof_construction_carries_guarded_scratch():
  """nv==0 sims admit minimum backings and reject nothing valid."""
  import torch
  from mujoco_metal.simulation import MetalSimulation
  model = mujoco.MjModel.from_xml_string(
      '<mujoco><worldbody><geom type="sphere" size=".1"/></worldbody></mujoco>')
  assert int(model.nv) == 0
  sim = MetalSimulation(model, 2, profile="integrated_scalable_v1")
  device = sim._state._device
  assert sim._twosum_float.shape == (8, 1)
  assert sim._twosum_mask.shape == (2, 1)
  hi = torch.zeros((2, 0), dtype=torch.float32, device=device)
  lo = torch.zeros((2, 0), dtype=torch.float32, device=device)
  val = torch.zeros((2, 0), dtype=torch.float32, device=device)
  sim._rhs_accumulate(hi, lo, val)  # empty no-op, must not raise


@_GPU
@_NEEDS_GPU
@pytest.mark.parametrize("solver", ["PGS", "CG", "Newton"])
def test_cancellation_survives_two_world_step_and_restore(solver):
  """1 + 2**24 - 2**24 cancellation through ordinary two-world step/restore.

  No production mask is involved here (both worlds step together); masked
  recovery coverage lives in the selected-world tests below.
  """
  import torch
  from mujoco_metal.simulation import MetalSimulation
  model = mujoco.MjModel.from_xml_string(f"""<mujoco>
    <option gravity="0 0 0" solver="{solver}" timestep="0.001"><flag contact="disable"/></option>
    <worldbody><body><joint name="j" type="slide" axis="1 0 0" stiffness="1" ref="0"/>
      <geom type="sphere" size=".1" mass="1"/></body></worldbody>
    <actuator><motor joint="j" ctrllimited="false" forcelimited="false"/></actuator>
  </mujoco>""")
  sim = MetalSimulation(model, 2, profile="integrated_scalable_v1")
  # World 1 sits at the spring reference (qpos 0) so passive is ~0 there;
  # world 0 starts displaced with the cancelling trio.
  sim.reset(qpos=np.array([[-1.0], [0.0]], dtype=np.float32))
  snap = sim.snapshot()
  sim.step(1, qfrc_applied=torch.tensor([[2.0**24], [0.0]], device="mps"),
           ctrl=torch.tensor([[-2.0**24], [0.0]], device="mps"))
  qvel = sim.state.qvel.cpu().numpy()
  # World 0 sees the unit residual through compensation; world 1 stays put.
  assert qvel[0, 0] == pytest.approx(model.opt.timestep * 1.0, rel=1e-5)
  assert qvel[1, 0] == pytest.approx(0.0, abs=1e-7)
  sim.restore(snap)
  sim.step(1, qfrc_applied=torch.tensor([[2.0**24], [0.0]], device="mps"),
           ctrl=torch.tensor([[-2.0**24], [0.0]], device="mps"))
  qvel2 = sim.state.qvel.cpu().numpy()
  np.testing.assert_array_equal(qvel2, qvel)


def _accum_sim(profile="integrated_euler_v1"):
  import torch
  from mujoco_metal.simulation import MetalSimulation
  sim = MetalSimulation(_model(), 2, profile=profile)
  return sim, sim._state._device


@_GPU
@_NEEDS_GPU
def test_partial_overlap_with_different_offsets_rejected():
  """Slices sharing backing with different pointers must still be caught."""
  import torch
  sim, device = _accum_sim()
  base = torch.zeros(4, dtype=torch.float32, device=device)
  hi = base[0:2].reshape(2, 1)
  lo = base[1:3].reshape(2, 1)
  assert hi.data_ptr() != lo.data_ptr()
  val = torch.ones((2, 1), dtype=torch.float32, device=device)
  before = base.detach().cpu().numpy().tobytes()
  with pytest.raises(ValueError, match="overlap"):
    sim._rhs_accumulate(hi, lo, val)
  # Bitwise rollback over the whole affected backing range.
  assert base.detach().cpu().numpy().tobytes() == before


@_GPU
@_NEEDS_GPU
def test_disjoint_slices_of_one_backing_are_accepted():
  """Sharing storage without sharing bytes is a valid configuration."""
  import torch
  sim, device = _accum_sim()
  base = torch.zeros(4, dtype=torch.float32, device=device)
  hi = base[0:2].reshape(2, 1)
  lo = base[2:4].reshape(2, 1)
  val = torch.ones((2, 1), dtype=torch.float32, device=device)
  sim._rhs_accumulate(hi, lo, val)
  assert torch.equal(hi, torch.ones((2, 1), device=device))
  assert torch.equal(base[2:4], torch.zeros(2, device=device))


@_GPU
@_NEEDS_GPU
@pytest.mark.parametrize("sign", [1, -1])
@pytest.mark.parametrize("masked", [False, True])
def test_value_viewing_hi_storage_is_safe(sign, masked):
  """A value that reads `hi` storage is consumed before publication."""
  import torch
  sim, device = _accum_sim()
  base = torch.full((2, 1), 3.0, dtype=torch.float32, device=device)
  lo = torch.zeros((2, 1), dtype=torch.float32, device=device)
  mask = (torch.tensor([1, 0], dtype=torch.int32, device=device)
          if masked else None)
  sim._rhs_accumulate(base, lo, base, sign=sign, mask=mask)
  expect_hi = 6.0 if sign > 0 else 0.0
  if masked:
    # World 0 accumulates base+base; world 1 keeps its exact words.
    assert base[0, 0].item() == pytest.approx(expect_hi)
    assert base[1, 0].item() == pytest.approx(3.0)
  else:
    assert torch.equal(base, torch.full((2, 1), expect_hi, device=device))


@_GPU
@_NEEDS_GPU
def test_scratch_backed_arguments_rejected():
  """Caller tensors must never view owned TwoSum working storage."""
  import torch
  sim, device = _accum_sim()
  nv = int(_model().nv)
  good = lambda *shape: torch.zeros(*shape, dtype=torch.float32, device=device)
  pool_alias = sim._twosum_float[0, :2 * nv].reshape(2, nv)
  with pytest.raises(ValueError, match="scratch"):
    sim._rhs_accumulate(pool_alias, good(2, nv), good(2, nv))
  with pytest.raises(ValueError, match="scratch"):
    sim._rhs_accumulate(good(2, nv), pool_alias, good(2, nv))
  with pytest.raises(ValueError, match="scratch"):
    sim._rhs_accumulate(good(2, nv), good(2, nv), pool_alias)
  with pytest.raises(ValueError, match="scratch"):
    sim._rhs_accumulate(good(2, nv), good(2, nv), good(2, nv),
                        mask=_mask_view_of_pool(sim, device))


def _mask_view_of_pool(sim, device):
  import torch
  return sim._twosum_float.view(torch.int32)[0, :2].reshape(2, 1)


@_GPU
@_NEEDS_GPU
def test_shape_rank_mismatch_rejected_despite_equal_numel():
  """Same element count with different shapes must not reinterpret."""
  import torch
  sim, device = _accum_sim()
  hi = torch.zeros((2, 1), dtype=torch.float32, device=device)
  lo = torch.zeros((2, 1), dtype=torch.float32, device=device)
  val = torch.zeros((1, 2), dtype=torch.float32, device=device)
  before_hi, before_lo = hi.clone(), lo.clone()
  with pytest.raises(ValueError, match="shapes must agree"):
    sim._rhs_accumulate(hi, lo, val)
  assert torch.equal(hi, before_hi) and torch.equal(lo, before_lo)


@_GPU
@_NEEDS_GPU
@pytest.mark.parametrize("bad_sign", [0, 2, 0.5, "1", None])
def test_non_unit_sign_rejected_without_write(bad_sign):
  import torch
  sim, device = _accum_sim()
  hi = torch.ones((2, 1), dtype=torch.float32, device=device)
  lo = torch.zeros((2, 1), dtype=torch.float32, device=device)
  val = torch.ones((2, 1), dtype=torch.float32, device=device)
  with pytest.raises(ValueError, match="sign"):
    sim._rhs_accumulate(hi, lo, val, sign=bad_sign)
  assert torch.equal(hi, torch.ones((2, 1), device=device))
  assert torch.equal(lo, torch.zeros((2, 1), device=device))


@_GPU
@_NEEDS_GPU
def test_nonfinite_inputs_flow_without_raising():
  """No finite constraint: failure-row infinities propagate per float32."""
  import torch
  sim, device = _accum_sim()
  hi = torch.tensor([[1.0], [float("inf")]], dtype=torch.float32, device=device)
  lo = torch.zeros((2, 1), dtype=torch.float32, device=device)
  val = torch.ones((2, 1), dtype=torch.float32, device=device)
  sim._rhs_accumulate(hi, lo, val)  # must not raise
  assert hi[0, 0].item() == pytest.approx(2.0)


@_GPU
@_NEEDS_GPU
@pytest.mark.parametrize("dtype", [torch.bool, torch.int32, torch.float32])
def test_mask_dtypes_share_nonzero_selection_contract(dtype):
  """Bool/int/float masks all select on nonzero with bitwise retention."""
  import torch
  sim, device = _accum_sim()
  hi = torch.zeros((2, 1), dtype=torch.float32, device=device)
  lo = torch.zeros((2, 1), dtype=torch.float32, device=device)
  lo.view(torch.int32)[1, 0] = -2147483648
  val = torch.ones((2, 1), dtype=torch.float32, device=device)
  if dtype == torch.bool:
    mask = torch.tensor([True, False], dtype=torch.bool, device=device)
  else:
    mask = torch.tensor([[2], [0]], dtype=dtype, device=device)
  sim._rhs_accumulate(hi, lo, val, mask=mask)
  assert hi[0, 0].item() == pytest.approx(1.0)
  assert hi[1, 0].item() == pytest.approx(0.0)
  got = int(np.frombuffer(
      lo[1, 0].detach().cpu().numpy().tobytes(), dtype=np.uint32)[0])
  assert got == 0x80000000


@_GPU
@_NEEDS_GPU
def test_repeated_failures_leave_scratch_coherent():
  """Two failed validations change nothing; the next valid call is exact."""
  import torch
  sim, device = _accum_sim()
  pool_before = sim._twosum_float.detach().cpu().numpy().tobytes()
  hi = torch.ones((2, 1), dtype=torch.float32, device=device)
  lo = torch.zeros((2, 1), dtype=torch.float32, device=device)
  with pytest.raises(ValueError):
    sim._rhs_accumulate(hi, lo, torch.ones((2, 2), device=device,
                                           dtype=torch.float32))
  with pytest.raises(ValueError):
    sim._rhs_accumulate(hi, hi.clone(), torch.ones((2, 1), device=device,
                                                   dtype=torch.float32),
                        mask=torch.zeros(3, dtype=torch.int32, device=device))
  assert torch.equal(hi, torch.ones((2, 1), device=device))
  assert sim._twosum_float.detach().cpu().numpy().tobytes() == pool_before
  sim._rhs_accumulate(hi, lo, torch.ones((2, 1), device=device))
  assert torch.equal(hi, torch.full((2, 1), 2.0, device=device))


def _contact_ctor_model():
  return mujoco.MjModel.from_xml_string("""<mujoco>
    <option gravity="0 0 0"><flag contact="disable"/></option>
    <worldbody>
      <geom name="floor" type="plane" size="5 5 .1"/>
      <body pos="0 0 0.05"><freejoint/>
        <geom type="sphere" size=".1" mass="1"/></body>
    </worldbody></mujoco>""")


def _mirrored_estimate(model, batch, profile):
  """Estimate with the constructor's lowered counts and mass storage."""
  from mujoco_metal.coupled_constraints import lower_coupled_constraints
  from mujoco_metal.capacity import estimate_capacity
  storage = ("block_sparse" if profile == "integrated_scalable_v1"
             else "dense")
  lowered = lower_coupled_constraints(model, mass_storage=storage)
  return estimate_capacity(
      model, batch, int(lowered.npairs), int(lowered.ncontacts_max),
      int(lowered.nr), neq=int(lowered.neq),
      nr_joint=int(lowered.nr_joint),
      mass_storage=storage)


@_GPU
@_NEEDS_GPU
@pytest.mark.parametrize("profile", ["integrated_euler_v1",
                                     "integrated_scalable_v1"])
@pytest.mark.parametrize("kind", ["free", "contact"])
def test_constructor_twosum_budget_boundary_with_lowered_counts(profile, kind):
  """Discovered whole-construction budget constructs; one byte under fails.

  The budget is discovered empirically through the real constructor by
  climbing staged CapacityOverflow requests (simulation, coupled, upload,
  cache stages in order), so the boundary covers every stage with actual
  lowered counts — no mirrored arithmetic to drift. The sim-level table
  still proves the twosum inventory bytes.
  """
  import re
  import torch
  from mujoco_metal.capacity import CapacityLimits, CapacityOverflow
  from mujoco_metal.simulation import MetalSimulation
  model = _contact_ctor_model() if kind == "contact" else _model()
  est = _mirrored_estimate(model, 2, profile)
  table = dict(est.memory_breakdown)
  assert table["simulation.twosum_float"] == 8 * max(2 * model.nv, 1) * 4
  assert table["simulation.twosum_mask"] == 2
  assert table["simulation.twosum_damped_qvel"] == max(2 * model.nv, 1) * 4
  budget = 1
  sim = None
  for _ in range(10):
    try:
      sim = MetalSimulation(
          model, 2, profile=profile,
          limits=CapacityLimits(memory_budget_bytes=int(budget)))
      break
    except CapacityOverflow as exc:
      hit = re.search(r"\((\d+) bytes\) exceeds budget", str(exc))
      assert hit is not None, str(exc)[:200]
      budget = int(hit.group(1))
  assert sim is not None
  assert sim._twosum_float.numel() * 4 == table["simulation.twosum_float"]
  assert sim._twosum_mask.numel() * 1 == table["simulation.twosum_mask"]
  assert (sim._damped_qvel.numel() * 4
          <= table["simulation.twosum_damped_qvel"])
  with pytest.raises(CapacityOverflow, match="memory|budget"):
    MetalSimulation(
        model, 2, profile=profile,
        limits=CapacityLimits(memory_budget_bytes=int(budget) - 1))


@_GPU
@_NEEDS_GPU
def test_constructor_rejection_precedes_compile_and_device_alloc():
  """A 1-byte budget fails with no shader compile and no MPS allocation."""
  import torch
  from unittest import mock
  from mujoco_metal.capacity import CapacityLimits, CapacityOverflow
  from mujoco_metal.simulation import MetalSimulation
  model = _model()
  mps_allocs = []
  real_empty, real_zeros = torch.empty, torch.zeros
  real_empty_like, real_zeros_like = torch.empty_like, torch.zeros_like
  real_as_tensor = torch.as_tensor

  def _record(fn, tag):
    def wrap(*args, **kwargs):
      dev = kwargs.get("device", None)
      if dev is not None and getattr(dev, "type", dev) == "mps":
        mps_allocs.append(tag)
      return fn(*args, **kwargs)
    return wrap

  with mock.patch.object(torch.mps, "compile_shader",
                         side_effect=AssertionError("compiled")) as compiled, \
      mock.patch.object(torch, "empty", _record(real_empty, "empty")), \
      mock.patch.object(torch, "zeros", _record(real_zeros, "zeros")), \
      mock.patch.object(torch, "empty_like",
                        _record(real_empty_like, "empty_like")), \
      mock.patch.object(torch, "zeros_like",
                        _record(real_zeros_like, "zeros_like")), \
      mock.patch.object(torch, "as_tensor",
                        _record(real_as_tensor, "as_tensor")):
    with pytest.raises(CapacityOverflow, match="memory"):
      MetalSimulation(model, 2, profile="integrated_euler_v1",
                      limits=CapacityLimits(memory_budget_bytes=1))
  assert compiled.call_count == 0
  assert mps_allocs == []


@_GPU
@_NEEDS_GPU
def test_zero_dof_no_coupled_construction_admits_guarded_scratch():
  """nv==0 builds no coupled stage and carries minimum scratch backings."""
  import torch
  from mujoco_metal.capacity import CapacityLimits
  from mujoco_metal.simulation import MetalSimulation
  model = mujoco.MjModel.from_xml_string(
      '<mujoco><worldbody><geom type="sphere" size=".1"/></worldbody></mujoco>')
  assert int(model.nv) == 0
  est = _mirrored_estimate(model, 2, "integrated_euler_v1")
  table = dict(est.memory_breakdown)
  assert table["simulation.twosum_float"] == 8 * 1 * 4
  assert table["simulation.twosum_mask"] == 2
  sim = MetalSimulation(
      model, 2, profile="integrated_euler_v1",
      limits=CapacityLimits(memory_budget_bytes=int(est.memory_bytes)))
  assert sim._coupled_constraints is None
  assert sim._twosum_float.shape == (8, 1)
  assert sim._twosum_mask.shape == (2, 1)
  assert tuple(sim._damped_qvel.shape) == (2, 0)
