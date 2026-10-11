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
  with pytest.raises(ValueError, match="alias"):
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
