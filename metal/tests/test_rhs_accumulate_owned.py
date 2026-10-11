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
  """Second same-shape accumulate must reuse the admitted pool."""
  import torch
  from mujoco_metal.simulation import MetalSimulation
  model = _model()
  sim = MetalSimulation(model, 2, profile="integrated_euler_v1")
  device = sim._state._device
  hi = torch.zeros((2, model.nv), dtype=torch.float32, device=device)
  lo = torch.zeros((2, model.nv), dtype=torch.float32, device=device)
  val = torch.ones((2, model.nv), dtype=torch.float32, device=device)
  assert sim._twosum_float is None
  sim._rhs_accumulate(hi, lo, val)
  first_id = id(sim._twosum_float)
  first_count = sim._twosum_float_count
  assert first_count >= int(hi.numel())
  assert sim._twosum_float.shape[0] == 8
  hi2 = torch.zeros((2, model.nv), dtype=torch.float32, device=device)
  lo2 = torch.zeros((2, model.nv), dtype=torch.float32, device=device)
  sim._rhs_accumulate(hi2, lo2, val)
  assert id(sim._twosum_float) == first_id
  assert sim._twosum_float_count == first_count
  # Admitted footprint: 8 float words per element + bool selector rows.
  assert sim._twosum_float.numel() == 8 * first_count


@_GPU
@_NEEDS_GPU
@pytest.mark.parametrize("solver", ["PGS", "CG", "Newton"])
def test_cancellation_survives_masked_step_and_restore(solver):
  """1 + 2**24 - 2**24 cancellation with world mask, step and restore."""
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
