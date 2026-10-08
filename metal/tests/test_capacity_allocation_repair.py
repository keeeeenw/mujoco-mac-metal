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
    SIGNED_I32_ELEMENT_LIMIT,
    _check_i32_elements,
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
  assert by_name["contact_row_data"] == 2 * 2 * 42 * 4
  assert by_name["contact_frame"] == 2 * 2 * 12 * 4
  assert by_name["contact_jacobian"] == 2 * 2 * 6 * 1 * 4
  assert by_name["pair_mask"] == 2 * 1 * 4
  # Dense records keep the typed 12-word row-layout header ahead of values.
  # Production records are always paired (high + low planes).
  assert by_name["workspace_J"] == 2 * (12 + 2 * 3 * 1) * 4
  assert by_name["position_cache_J"] == by_name["workspace_J"]
  assert by_name["position_context_zero"] == 2 * 3 * 4
  # PGS core plus qacc input/output, canonical labels, union-find and tree
  # masks, midpoint body eligibility, and the independent warmstart tail.
  # Dense mass factor and vectors are dynamic device scratch here; sparse
  # component mode omits them and is checked against its separate estimate.
  # This is a one-DOF, three-row PGS fixture.  The tail includes the dynamic
  # row/cone metadata (12 words per row), the retained J-transpose solve
  # block (nv*nr), island/equality workspaces, and the dense factor backing.
  # The independent high/low tail adds 8*nv + 3*nr words (17 here),
  # including the retained aref residual row. Keep this expected
  # arithmetic independent of the estimator helper.
  assert by_name["workspace_debug"] == 2 * (9 + 21 + 107 + 17) * 4
  assert by_name["out_force"] == 2 * 1 * 4
  assert by_name["out_acc"] == 2 * 2 * 1 * 4
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
  assert e4.memory_bytes > e1.memory_bytes
  assert e4.memory_breakdown == estimate_capacity(
      m, 4, npairs=1, nslots=2, nr=4).memory_breakdown
  assert dict(e4.memory_breakdown)["smooth.crb"] == 4 * dict(e1.memory_breakdown)["smooth.crb"]
  assert e1.memory_bytes == sum(v for _, v in e1.memory_breakdown)


def test_static_heavy_model_checks_real_spatial_matrix_buffer_before_allocation():
  # Keep fixed bodies through compilation without adding generalized
  # coordinates, isolating body-spatial buffers from nv/mass guards.
  xml = "".join(
      f'<body pos="0 0 {0.01 * i}"><geom type="sphere" size="0.01"/></body>'
      for i in range(400))
  model = mujoco.MjModel.from_xml_string(
      f"<mujoco><compiler fusestatic='false'/><worldbody>{xml}</worldbody></mujoco>")
  assert model.nbody == 401 and model.nv == 0
  # The estimator itself is the exercised pre-allocation path. A nearby
  # legal arithmetic shape remains below int32; the next batch crosses the
  # actual crb[batch,nbody,6,6] backing allocation.
  boundary_batch = (SIGNED_I32_ELEMENT_LIMIT // (model.nbody * 36))
  from mujoco_metal.capacity import _runtime_buffer_sizes
  # The consolidated FK arena is now larger than CRB. Exercise the actual
  # CRB element shape independently, then check the earliest physical backing
  # in the complete preflight rather than pretending CRB is still first.
  below = dict(_runtime_buffer_sizes(model, boundary_batch))
  assert below["smooth.crb"] == boundary_batch * model.nbody * 36
  _check_i32_elements("smooth.crb", below["smooth.crb"])
  above = dict(_runtime_buffer_sizes(model, boundary_batch + 1))
  with pytest.raises(CapacityOverflow, match="smooth.crb.*signed 32-bit"):
    _check_i32_elements("smooth.crb", above["smooth.crb"])
  # Body/inertial (2 * 3 * (3+4)) and geom (3 * (3+4+9))
  # outputs share one arena; absent site/joint fields keep eight sentinel words.
  arena_stride = model.nbody * 42 + model.ngeom * 48
  arena_batch = (SIGNED_I32_ELEMENT_LIMIT - 8) // arena_stride
  complete = estimate_capacity(model, arena_batch, 0, 0, 0)
  assert dict(complete.memory_breakdown)["fk.pose_output_arena"] == (
      arena_batch * arena_stride + 8) * 4
  with pytest.raises(CapacityOverflow, match="fk.pose_output_arena.*signed 32-bit"):
    estimate_capacity(model, arena_batch + 1, 0, 0, 0)
  _check_i32_elements("exact-boundary", SIGNED_I32_ELEMENT_LIMIT)
  with pytest.raises(CapacityOverflow, match="exact-boundary"):
    _check_i32_elements("exact-boundary", SIGNED_I32_ELEMENT_LIMIT + 1)


def test_simulation_constructor_checks_body_spatial_stride_before_device_init():
  from mujoco_metal.simulation import MetalSimulation
  xml = "".join(
      f'<body pos="0 0 {0.01 * i}"><geom type="sphere" size="0.01"/></body>'
      for i in range(400))
  model = mujoco.MjModel.from_xml_string(
      f"<mujoco><compiler fusestatic='false'/><worldbody>{xml}</worldbody></mujoco>")
  boundary_batch = SIGNED_I32_ELEMENT_LIMIT // (model.nbody * 36)
  # This exercises the constructor's early guard. Without it the call would
  # proceed into MPS initialization rather than rejecting the invalid shape.
  with pytest.raises(CapacityOverflow, match="fk.pose_output_arena.*signed 32-bit"):
    MetalSimulation(model, batch_size=boundary_batch + 1,
                    profile="contact_free_euler_v1")


def test_simulation_constructor_rejects_invalid_batch_before_device_init():
  from mujoco_metal.simulation import MetalSimulation
  model = _model()
  for batch in (True, np.bool_(True), 1.5, np.float32(2)):
    with pytest.raises(TypeError):
      MetalSimulation(model, batch_size=batch,
                      profile="contact_free_euler_v1")
  for batch in (0, -1):
    with pytest.raises(ValueError, match="batch_size"):
      MetalSimulation(model, batch_size=batch,
                      profile="contact_free_euler_v1")


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


def _plinko_model(n_friction=16, n_free=9):
  pegs = []
  for i in range(n_friction + n_free):
    x = -0.6 + 0.15 * (i % 7)
    z = 0.9 - 0.12 * (i // 7)
    cd = 3 if i < n_friction else 1
    pegs.append(f'<geom name="peg{i}" type="box" size="0.02 0.02 0.02" '
                f'pos="{x} 0 {z}" condim="{cd}"/>')
  return mujoco.MjModel.from_xml_string(
      '<mujoco><option timestep="0.002" cone="pyramidal"/>'
      '<worldbody><geom name="floor" type="plane" size="5 5 0.1"/>'
      + "".join(pegs) +
      '<body pos="0 0 1.3"><freejoint/>'
      '<geom name="ball" type="sphere" size="0.05" condim="1"/></body>'
      '</worldbody></mujoco>')


def test_raised_ceilings_admit_beyond_old_caps_cpu():
  # R08/F2: 26 pairs / 26 slots exceed the old 16/24 ceilings with rows in
  # budget; beyond the new ceilings still overflows before allocating.
  # (The demo XML sets the floor decorative for 25; here default masks
  # keep the ball/floor pair.)
  from mujoco_metal.coupled_constraints import lower_coupled_constraints
  m = _plinko_model()
  desc = lower_coupled_constraints(m)
  assert desc.npairs == 26, desc.npairs
  assert desc.ncontacts_max == 26, desc.ncontacts_max
  assert desc.nr <= 96, desc.nr
  assert desc.npairs > 16 and desc.ncontacts_max > 24
  m2 = _plinko_model(n_friction=24, n_free=12)
  with pytest.raises(ValueError, match="exceeds capacity"):
    lower_coupled_constraints(m2)


def test_paired_jacobian_dense_bytes_and_shared_stride_cpu():
  # Production coupled records always carry high + low planes. The
  # estimator must agree with the shared model-only stride definition:
  # dense 12 + 2*nr*nv words per world (both workspace_J and cache).
  from mujoco_metal.constraint_jacobian import paired_jacobian_stride_words
  assert paired_jacobian_stride_words(16, 6, mode=0, paired=True) == 12 + 2 * 16 * 6
  assert paired_jacobian_stride_words(16, 6, mode=0, paired=False) == 12 + 16 * 6
  parts, _ = estimate_workspace(nv=6, npairs=1, nslots=8, nr=16,
                                nr_joint=8, neq=0, batch=1)
  by_name = dict(parts)
  assert by_name["workspace_J"] == (12 + 2 * 16 * 6) * 4
  assert by_name["position_cache_J"] == (12 + 2 * 16 * 6) * 4


def test_paired_jacobian_csr_bytes_batch_and_zero_guards_cpu():
  from mujoco_metal.constraint_jacobian import paired_jacobian_stride_words
  assert (paired_jacobian_stride_words(16, 6, mode=1, nnz=48, paired=True)
          == 12 + 16 + 1 + 3 * 48)
  parts, _ = estimate_workspace(nv=6, npairs=1, nslots=8, nr=16,
                                nr_joint=8, neq=0, batch=2,
                                jacobian_kind="sparse", jacobian_nnz=48)
  by_name = dict(parts)
  assert by_name["workspace_J"] == 2 * (12 + 16 + 1 + 3 * 48) * 4
  assert by_name["position_cache_J"] == 2 * (12 + 16 + 1 + 3 * 48) * 4
  empty, _ = estimate_workspace(nv=0, npairs=0, nslots=0, nr=0,
                                nr_joint=0, neq=0, batch=1,
                                jacobian_kind="sparse", jacobian_nnz=0)
  assert dict(empty)["workspace_J"] == (12 + 0 + 1 + 0) * 4
  assert dict(empty)["position_cache_J"] == (12 + 0 + 1 + 0) * 4


def test_paired_stride_rejects_signed_overflow_cpu():
  from mujoco_metal.constraint_jacobian import paired_jacobian_stride_words
  with pytest.raises(CapacityOverflow, match="signed int32"):
    estimate_workspace(nv=1, npairs=1, nslots=1, nr=1, nr_joint=0,
                       neq=0, batch=1, jacobian_kind="sparse",
                       jacobian_nnz=SIGNED_I32_ELEMENT_LIMIT)
  with pytest.raises(ValueError, match="signed int32"):
    paired_jacobian_stride_words(0, 0, mode=1, nnz=SIGNED_I32_ELEMENT_LIMIT,
                                paired=True)


def test_budget_between_old_and_actual_paired_totals_rejects_cpu():
  # Old high-only dense J for nv=6,nr=16 is 384 bytes per J buffer below
  # the actual paired total. A memory budget inside that gap must reject
  # via check_capacity (invoked before any device allocation).
  from mujoco_metal.capacity import CapacityLimits
  old_j = (12 + 16 * 6) * 4
  new_j = (12 + 2 * 16 * 6) * 4
  assert new_j - old_j == 384
  m = _model()
  est = estimate_capacity(m, 1, npairs=1, nslots=2, nr=4)
  full = est.memory_bytes
  assert full > 2 * (new_j - old_j)
  # A budget covering everything except one J plane delta still rejects.
  with pytest.raises(CapacityOverflow, match="memory"):
    check_capacity(est, CapacityLimits(memory_budget_bytes=full - 1))
  check_capacity(est, CapacityLimits(memory_budget_bytes=full))


def test_paired_correction_uses_existing_tail_without_new_words_cpu():
  # Paired W/solver correction introduces no new global words: J low
  # planes are counted above; multiplier/rhs low words reuse the existing
  # 8*nv+3*nr tail (pgs_rhs_low/lambda_low/aref_low for all solver
  # types); accumulation beyond that is thread-local compensated
  # arithmetic. Assert the tail formula still covers exactly three row
  # planes and eight DOF planes.
  parts, _ = estimate_workspace(nv=6, npairs=1, nslots=8, nr=16,
                                nr_joint=8, neq=0, batch=1)
  by_name = dict(parts)
  # Tail length is embedded in workspace_debug; verify the 8*nv+3*nr
  # structure by comparing two nv values with nr fixed (difference is
  # exactly 8 words per DOF, no hidden paired-W plane).
  parts2, _ = estimate_workspace(nv=7, npairs=1, nslots=8, nr=16,
                                 nr_joint=8, neq=0, batch=1)
  d1 = dict(parts2)["workspace_debug"] - by_name["workspace_debug"]
  # nv 6->7 changes dense mass factor (v^2+2v: 48->63 = +15), the v*r
  # Delassus-transpose block (96->112 = +16), equality scratch (12v:
  # 72->84 = +12), qacc pair (2v: +2), and tail 8v (+8): total +53 words
  # = +212 bytes. No hidden paired-W plane (that correction reuses the
  # existing tail + thread-local arithmetic).
  assert d1 == (15 + 16 + 12 + 2 + 8) * 4


@_needs_gpu()
def test_runtime_paired_buffers_match_estimates_gpu():
  # Runtime-owned allocations must equal the pre-allocation estimates on
  # real paired layouts, including batch>1 and the SDF fixture layout.
  import os
  assert os.getenv("MUJOCO_METAL_RUN_GPU") == "1"
  from mujoco_metal.coupled_constraints import MetalCoupledConstraints
  from mujoco_metal.capacity import estimate_capacity
  import sys
  sys.path.insert(0, "metal/tests")
  from test_bundled_sdf_rigid_oracle_019 import _plugin_sphere_model
  for name in ("bowl",):
    model = _plugin_sphere_model(name)
    for batch in (1, 2):
      cc = MetalCoupledConstraints(model, batch_size=batch)
      est = estimate_capacity(
          model, batch, cc.descriptor.npairs, cc.descriptor.ncontacts_max,
          cc.descriptor.nr, neq=cc.descriptor.neq,
          nr_joint=cc.descriptor.nr_joint,
          mass_storage=cc._component_mass_storage,
          jacobian_kind=str(getattr(cc.descriptor, "jacobian_kind", "dense")),
          jacobian_nnz=(int(cc.descriptor.jacobian_pattern.nnz)
                        if getattr(cc.descriptor, "jacobian_pattern", None)
                        is not None else None))
      by_name = dict(est.memory_breakdown)
      w = cc._workspace
      assert w["workspace_J"].numel() * 4 == by_name["workspace_J"]
      assert w["position_cache_J"].numel() * 4 == by_name["position_cache_J"]
      layout = cc._jacobian_layout
      assert layout.paired == 1
      assert layout.stride_words * 4 * batch == by_name["workspace_J"]
  # Zero-row/contact guard: floor-only model keeps one-word sentinels.
  m = _model()
  cc0 = MetalCoupledConstraints(m, batch_size=1)
  assert cc0._workspace["workspace_J"].numel() >= 1
  assert cc0._workspace["position_cache_J"].numel() == (
      cc0._workspace["workspace_J"].numel())
