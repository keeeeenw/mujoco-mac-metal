# Copyright 2026 The MuJoCo Metal contributors
# Licensed under the Apache License, Version 2.0.
"""R08a repair: batch-aware allocation enforcement (host side).

Failing-first: the capacity estimate must mirror prepare_workspace
buffer-for-buffer at the REAL batch size, and allocation must fail before
partial device allocation. Same scene across batch sizes must agree
per-world; broadphase active counts stay distinguished from allocated.
"""

import mujoco
import numpy as np
import pytest

from mujoco_metal.capacity import (
    CapacityLimits,
    CapacityOverflow,
    check_capacity,
    estimate_capacity,
    estimate_workspace,
)


def _model():
  return mujoco.MjModel.from_xml_string(
      "<mujoco><worldbody><geom name='floor' type='plane' size='5 5 0.1'/>"
      "<body pos='0 0 0.3'><freejoint/><geom type='sphere' size='0.1'/></body>"
      "</worldbody></mujoco>")


def test_workspace_mirror_is_byte_exact_cpu():
  parts, total = estimate_workspace(nv=1, npairs=1, nslots=2, nr=3,
                                    nr_joint=1, neq=0, batch=2)
  by_name = dict(parts)
  assert by_name["contact_row_data"] == 2 * 2 * 36 * 4
  assert by_name["contact_frame"] == 2 * 2 * 12 * 4
  assert by_name["contact_jacobian"] == 2 * 2 * 6 * 1 * 4
  assert by_name["pair_mask"] == 2 * 1 * 4
  assert by_name["workspace_J"] == 2 * 3 * 1 * 4
  assert by_name["workspace_debug"] == 2 * (9 + 21) * 4
  assert by_name["out_force"] == 2 * 1 * 4
  assert by_name["out_acc"] == 2 * 1 * 4
  assert by_name["out_status"] == 2 * 4
  assert by_name["out_diagnostics"] == 2 * 10 * 4
  assert by_name["out_contact_force"] == 2 * 2 * 11 * 4
  assert by_name["out_joint_force"] == 2 * 1 * 4
  assert by_name["eq_active"] == 2 * 1 * 4
  assert total == sum(by_name.values())


def test_estimate_scales_linearly_with_real_batch_cpu():
  m = _model()
  e1 = estimate_capacity(m, 1, npairs=1, nslots=2, nr=4)
  e4 = estimate_capacity(m, 4, npairs=1, nslots=2, nr=4)
  assert e4.batch == 4
  assert e4.memory_bytes == 4 * e1.memory_bytes
  assert e1.memory_bytes == sum(v for _, v in e1.memory_breakdown)


def test_batch_and_memory_overflow_cpu():
  m = _model()
  with pytest.raises(CapacityOverflow, match="batch size"):
    check_capacity(estimate_capacity(m, 17, 1, 1, 1))
  with pytest.raises(CapacityOverflow, match="memory"):
    check_capacity(estimate_capacity(m, 2, 1, 2, 4),
                   CapacityLimits(memory_budget_bytes=8))
  # Historical count texts preserved.
  with pytest.raises(ValueError, match="bounds nv to 32"):
    import dataclasses
    check_capacity(dataclasses.replace(estimate_capacity(m, 1, 0, 0, 0), nv=33))


def _needs_gpu():
  import os
  return pytest.mark.skipif(os.getenv("MUJOCO_METAL_RUN_GPU") != "1", reason="opt-in GPU")


@_needs_gpu()
def test_allocation_gate_rejects_before_allocating_gpu():
  import os
  assert os.getenv("MUJOCO_METAL_RUN_GPU") == "1"
  from mujoco_metal.coupled_constraints import MetalCoupledConstraints
  m = _model()
  with pytest.raises(CapacityOverflow, match="batch size"):
    MetalCoupledConstraints(m, batch_size=17)


@_needs_gpu()
def test_same_scene_across_batch_sizes_gpu():
  import os
  assert os.getenv("MUJOCO_METAL_RUN_GPU") == "1"
  from mujoco_metal.simulation import MetalSimulation
  m = _model()
  sims = []
  for b in (1, 4):
    sim = MetalSimulation(m, batch_size=b, profile="integrated_euler_v1")
    sim.reset(qpos=np.asarray(m.qpos0, dtype=np.float32).reshape(1, -1).repeat(b, 0),
              qvel=np.zeros((b, m.nv), dtype=np.float32))
    sims.append(sim)
  for _ in range(30):
    for sim in sims:
      sim.step(1)
  q1 = sims[0].state.qpos.cpu().numpy()
  q4 = sims[1].state.qpos.cpu().numpy()
  for w in range(4):
    np.testing.assert_array_equal(q4[w], q1[0])
  w1 = sims[0]._coupled_constraints.get_warmstart()
  w4 = sims[1]._coupled_constraints.get_warmstart()
  for w in range(4):
    np.testing.assert_array_equal(w4[w], w1[0])


@_needs_gpu()
def test_active_counts_distinguished_from_allocated_gpu():
  import os
  assert os.getenv("MUJOCO_METAL_RUN_GPU") == "1"
  from mujoco_metal.simulation import MetalSimulation
  # Two candidate pairs (floor/near-ball, floor/far-ball); broadphase
  # sphere overlap keeps only the near pair while slots stay allocated.
  m = mujoco.MjModel.from_xml_string(
      "<mujoco><worldbody><geom name='floor' type='plane' size='5 5 0.1'/>"
      "<body pos='0 0 0.09'><freejoint/><geom name='a' type='sphere' size='0.1'/></body>"
      "<body pos='0 0 8.0'><freejoint/><geom name='b' type='sphere' size='0.1'/></body>"
      "</worldbody></mujoco>")
  sim = MetalSimulation(m, batch_size=1, profile="integrated_euler_v1")
  sim.reset(qpos=np.asarray(m.qpos0, dtype=np.float32).reshape(1, -1),
            qvel=np.zeros((1, m.nv), dtype=np.float32))
  sim.assembled_system(recompute=True)
  mask_active, slot_active = sim._coupled_constraints.broadphase_counts()[0]
  d = sim._coupled_constraints.descriptor
  assert 0 < slot_active <= mask_active <= d.npairs
  assert mask_active < d.npairs  # far pair pruned from work, slots remain
  assert slot_active <= d.ncontacts_max
