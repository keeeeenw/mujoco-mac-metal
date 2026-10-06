"""CPU contract tests and opt-in native selectors for staged SDF contacts."""

import os
import inspect

import mujoco
import numpy as np
import pytest


_GPU = pytest.mark.skipif(
    os.getenv("MUJOCO_METAL_RUN_GPU") != "1", reason="opt-in native Metal")
_VBOX = ("-0.05 -0.05 -0.04 0.05 -0.05 -0.04 "
         "0.05 0.05 -0.04 -0.05 0.05 -0.04 "
         "-0.05 -0.05 0.04 0.05 -0.05 0.04 "
         "0.05 0.05 0.04 -0.05 0.05 0.04")
_ASSET = f'<asset><mesh name="box" vertex="{_VBOX}"/></asset>'
_QID = (1.0, 0.0, 0.0, 0.0)


def _analytic_scene(*, initpoints=40, z=0.08, contact=True):
  flags = 'contype="1" conaffinity="1"' if contact else 'contype="0" conaffinity="0"'
  return mujoco.MjModel.from_xml_string(
      f'<mujoco>{_ASSET}<option timestep="0.002" integrator="Euler" '
      f'iterations="40" tolerance="1e-8" gravity="0 0 0" '
      f'sdf_initpoints="{initpoints}"/><worldbody>'
      f'<geom name="sdf" type="sdf" mesh="box" condim="1" {flags}/>'
      f'<body pos="0 0 {z}"><freejoint/>'
      f'<geom name="ball" type="sphere" size="0.05" condim="1" {flags}/>'
      '</body></worldbody></mujoco>')


def _mixed_scene(*, initpoints=40):
  return mujoco.MjModel.from_xml_string(
      f'<mujoco>{_ASSET}<option timestep="0.002" integrator="Euler" '
      f'iterations="40" tolerance="1e-8" gravity="0 0 0" '
      f'sdf_initpoints="{initpoints}"/><worldbody>'
      '<geom name="plane" type="plane" size="2 2 .1" condim="1"/>'
      '<geom name="sdf" type="sdf" mesh="box" condim="1"/>'
      '<body pos="0 0 .08"><freejoint/>'
      '<geom name="ball" type="sphere" size=".05" condim="1"/></body>'
      '<body pos=".1 0 .08"><freejoint/>'
      '<geom name="mesh" type="mesh" mesh="box" condim="1"/></body>'
      '</worldbody></mujoco>')


def test_sdf_seed_budget_and_output_capacity_are_separate():
  from mujoco_metal.coupled_constraints import pair_max_contacts

  sdf = int(mujoco.mjtGeom.mjGEOM_SDF)
  sphere = int(mujoco.mjtGeom.mjGEOM_SPHERE)
  mesh = int(mujoco.mjtGeom.mjGEOM_MESH)
  for budget, output in ((0, 0), (4, 4), (40, 40), (51, 50), (64, 50), (256, 50)):
    assert pair_max_contacts(sdf, sdf, budget) == output
    assert pair_max_contacts(sphere, sdf, budget) == output
  assert pair_max_contacts(mesh, sdf, 0) == 50


def test_cpu_mujoco_accepts_more_seeds_than_contact_capacity_when_few_hit():
  """Pinned C is the oracle: initpoints is not itself the 50-contact cap."""
  model = _analytic_scene(initpoints=64, z=10.0)
  data = mujoco.MjData(model)
  mujoco.mj_forward(model, data)
  assert data.ncon == 0

  from mujoco_metal.coupled_constraints import lower_coupled_constraints
  from mujoco_metal.capacity import CapacityLimits
  descriptor = lower_coupled_constraints(
      model, limits=CapacityLimits(max_slots=50))
  assert descriptor.ncontacts_max == 50
  assert int(descriptor.pair_max_contacts.max()) == 50


@pytest.mark.parametrize("budget, expected", [(4, 4), (40, 34), (51, 44)])
def test_cpu_sdf_seed_budget_contact_counts_match_pinned_source(budget, expected):
  model = _analytic_scene(initpoints=budget, z=0.08)
  data = mujoco.MjData(model)
  mujoco.mj_forward(model, data)
  assert data.ncon == expected


def test_sdf_overflow_status_is_merged_before_solver_and_world_local():
  from pathlib import Path
  from mujoco_metal.coupled_constraints import MetalCoupledConstraints

  root = Path(__file__).resolve().parents[1]
  shader = (root / "mujoco_metal" / "shaders" / "coupled_constraints.metal").read_text()
  assert "merge_sdf_narrowphase_status" in shader
  assert "contact_count[int(world) * npairs + pair_idx] < 0" in shader
  assert "candidate_world_mask[world] == 0" in shader
  normal = inspect.getsource(MetalCoupledConstraints.run_device)
  reset = normal.index('w["out_status"].zero_()')
  producer = normal.index("self.generate_candidates(", reset)
  merge = normal.index("self._merge_sdf_narrowphase_status_kernel(", producer)
  solver = normal.index("def dispatch_solver(", merge)
  assert reset < producer < merge < solver

  cached = inspect.getsource(MetalCoupledConstraints.run_velocity_device)
  reset = cached.index(
      'w["out_status"]', cached.index("self.refresh_position_references"))
  merge = cached.index("self._merge_sdf_narrowphase_status_kernel(", reset)
  solver = cached.index("solve_fn(", merge)
  assert reset < merge < solver


def test_sdf_status_merge_reuses_capacity_accounted_pair_header_cpu():
  from pathlib import Path

  root = Path(__file__).resolve().parents[1]
  host = (root / "mujoco_metal" / "coupled_constraints.py").read_text()
  shader = (root / "mujoco_metal" / "shaders" / "coupled_constraints.metal").read_text()
  assert "_sdf_status_dims" not in host
  assert host.count('["pair_contact_offsets_dims"],') >= 2
  kernel = shader[shader.index("kernel void merge_sdf_narrowphase_status("):]
  assert "int batch = dims[3];" in kernel
  assert "int npairs = dims[1];" in kernel
  assert "int batch = dims[0];" not in kernel


def test_analytic_sdf_zero_seed_budget_lowers_without_output_slots():
  model = _analytic_scene(initpoints=0, z=10.0)
  from mujoco_metal.coupled_constraints import lower_coupled_constraints
  descriptor = lower_coupled_constraints(model)
  assert descriptor.ncontacts_max == 0
  data = mujoco.MjData(model)
  mujoco.mj_forward(model, data)
  assert data.ncon == 0


@_GPU
def test_native_mixed_generic_analytic_and_mesh_sdf_producer_selectors():
  from mujoco_metal.coupled_constraints import MetalCoupledConstraints

  from mujoco_metal.capacity import CapacityLimits
  stage = MetalCoupledConstraints(
      _mixed_scene(initpoints=0), batch_size=2,
      limits=CapacityLimits(max_slots=128, max_rows=256))
  assert stage._has_generic_contact_pairs
  assert stage._has_analytic_sdf_pairs
  assert stage._has_mesh_sdf_pairs
  assert stage._contact_generic_producer is not None
  assert stage._contact_sdf_seed_init is not None
  assert stage._contact_sdf_descent_prepare is not None
  assert stage._contact_sdf_line_search is not None
  assert stage._contact_sdf_publish_normal is not None
  assert stage._contact_sdf_publish_contact is not None
  assert stage._contact_sdf_finalize_producer is not None
  assert stage._contact_mesh_sdf_producer is not None
  assert stage._merge_sdf_narrowphase_status_kernel is not None
  sdf = stage._mjmodel_ref.geom("sdf").id
  mesh = stage._mjmodel_ref.geom("mesh").id
  mesh_pair = next(i for i, (a, b) in enumerate(zip(stage.descriptor.geom1, stage.descriptor.geom2))
                   if {int(a), int(b)} == {sdf, mesh})
  assert int(stage.descriptor.pair_max_contacts[mesh_pair]) == 50
  assert int(stage.descriptor.mesh_hull_info[9 * sdf + 4]) == 1
  assert stage._contact_kernel is not None


@_GPU
def test_native_default_and_large_seed_budgets_use_tiled_workspace():
  from mujoco_metal.coupled_constraints import MetalCoupledConstraints

  for budget in (40, 64, 257):
    from mujoco_metal.capacity import CapacityLimits
    stage = MetalCoupledConstraints(
        _analytic_scene(initpoints=budget), batch_size=2,
        limits=CapacityLimits(max_slots=50))
    assert stage._sdf_seed_count == budget
    assert stage._sdf_seed_tile == 32
    scratch = stage._workspace["contact_sdf_seed_records"]
    assert scratch.numel() == 2 * stage.descriptor.npairs * 32 * 25
    state = stage._workspace["contact_sdf_seed_state"]
    assert state.numel() == 2 * stage.descriptor.npairs * 32 * 159
    assert stage.descriptor.ncontacts_max == min(budget, 50)

  from mujoco_metal.simulation import MetalSimulation
  zero = MetalSimulation(_analytic_scene(initpoints=0, z=10.0), batch_size=1,
                         profile="integrated_euler_v1")
  zero.assembled_system(recompute=True)
  assert int(zero._coupled_constraints._workspace["contact_pair_count"].sum().item()) == 0


@_GPU
def test_native_no_hit_disabled_pruned_worldmask_repeat_and_restore():
  import torch
  from mujoco_metal.simulation import MetalSimulation

  model = _analytic_scene(initpoints=40, z=10.0)
  sim = MetalSimulation(model, batch_size=2, profile="integrated_euler_v1")
  qpos = np.tile(np.asarray(model.qpos0, dtype=np.float32), (2, 1))
  sim.reset(qpos=qpos, qvel=np.zeros((2, model.nv), dtype=np.float32))
  cc = sim._coupled_constraints
  dynamics = sim._smooth.run_device(sim._state._qpos, sim._state._qvel)
  count = cc._workspace["contact_pair_count"]
  npairs = cc.descriptor.npairs

  # No-hit route updates count to zero and repeated production stays stable.
  cc.generate_candidates(dynamics["poses"], sim._state._qvel)
  assert not bool((count[:2 * npairs] != 0).any().item())
  cc.generate_candidates(dynamics["poses"], sim._state._qvel)
  assert not bool((count[:2 * npairs] != 0).any().item())

  # A world mask leaves the masked world's prior count untouched.
  count[npairs:2 * npairs].fill_(-123)
  mask = torch.tensor([1, 0], dtype=torch.int32, device="mps")
  cc.generate_candidates(dynamics["poses"], sim._state._qvel, world_mask=mask)
  np.testing.assert_array_equal(
      count[npairs:2 * npairs].detach().cpu().numpy(), -123)
  # Re-enabling the masked world overwrites its retained prior count.
  cc.generate_candidates(dynamics["poses"], sim._state._qvel)
  assert not bool((count[:2 * npairs] != 0).any().item())

  # Broadphase-pruned pairs keep the established frame sentinel and zero count.
  cc.set_broadphase_pruning(True)
  cc.generate_candidates(dynamics["poses"], sim._state._qvel)
  assert not bool((count[:2 * npairs] != 0).any().item())
  cc.set_broadphase_pruning(False)
  cc.generate_candidates(dynamics["poses"], sim._state._qvel)
  assert not bool((count[:2 * npairs] != 0).any().item())

  disabled = MetalSimulation(
      _analytic_scene(initpoints=4, z=0.08, contact=False), batch_size=1,
      profile="integrated_euler_v1")
  disabled_result = disabled.assembled_system(recompute=True)
  disabled_counts = disabled._coupled_constraints._workspace[
      "contact_pair_count"]
  assert not bool((disabled_counts != 0).any().item())
  if "contact_mask" in disabled_result:
    assert not bool((disabled_result["contact_mask"] != 0).any().item())

  snapshot = sim.state.snapshot()
  sim.state.restore(snapshot)
  cc.generate_candidates(dynamics["poses"], sim._state._qvel)
  assert not bool((count[:2 * npairs] != 0).any().item())


@_GPU
def test_native_finalizer_sets_overflow_sentinel_before_any_consumer():
  import torch
  from mujoco_metal.capacity import CapacityLimits
  from mujoco_metal.coupled_constraints import MetalCoupledConstraints

  stage = MetalCoupledConstraints(
      _analytic_scene(initpoints=64, z=10.0), batch_size=1,
      limits=CapacityLimits(max_slots=50))
  workspace = stage._workspace
  valid = workspace["contact_sdf_seed_valid"]
  records = workspace["contact_sdf_seed_records"]
  pair_count = workspace["contact_pair_count"]
  output = workspace["contact_pair_records"]
  pair_map = torch.zeros(1, dtype=torch.int32, device="mps")
  geom_type = stage._constants["geom_type"]
  pair_geoms = stage._constants["pair_geoms"]
  dims = stage._constants["pair_contact_offsets_dims"]

  # Supply 51 distinct valid seed witnesses directly to the actual finalizer.
  # This exercises output overflow without inventing a source geometry that
  # might not exceed 50 accepted unique contacts.
  for start, count in ((0, 32), (32, 19)):
    stage._sdf_stage_dims.copy_(torch.tensor(
        [count, start, 32, 0, 0], dtype=torch.int32, device="mps"))
    for i in range(count):
      slot = i * 25
      records[slot:slot + 25].zero_()
      records[slot] = (start + i) * 0.01
      valid[i] = 1
    stage._contact_sdf_finalize_producer(
        geom_type, pair_geoms, dims, pair_map, records, valid,
        pair_count, output, stage._sdf_stage_dims,
        threads=(stage.descriptor.npairs,), group_size=(1,))
  assert int(pair_count[0].item()) == -1


@_GPU
def test_native_overflow_public_status_is_world_local():
  from mujoco_metal.simulation import MetalSimulation

  sim = MetalSimulation(
      _analytic_scene(initpoints=40, z=10.0), batch_size=2,
      profile="integrated_euler_v1")
  cc = sim._coupled_constraints
  d = cc.descriptor
  model = sim._mjmodel
  pair = next(i for i, (g1, g2) in enumerate(zip(d.geom1, d.geom2))
              if 8 in (int(model.geom_type[g1]), int(model.geom_type[g2]))
              and 7 not in (int(model.geom_type[g1]), int(model.geom_type[g2])))
  original = cc._contact_sdf_finalize_producer

  def inject_overflow(*args, **kwargs):
    original(*args, **kwargs)
    counts = args[6]
    counts[pair] = -1

  # Inject the same private sentinel the finalizer writes on a 51st unique
  # candidate, then verify the public coupled result rejects only world 0.
  cc._contact_sdf_finalize_producer = inject_overflow
  qpos = np.tile(np.asarray(d.qpos0, dtype=np.float32), (2, 1))
  sim.reset(qpos=qpos, qvel=np.zeros((2, d.nv), dtype=np.float32))
  result = sim.assembled_system(recompute=True)
  status = result["status"].detach().cpu().numpy().reshape(-1)
  assert status[0] != 0
  assert status[1] == 0


@_GPU
def test_native_overflow_status_normal_and_cached_public_paths_restore():
  """Both public constraint entry points preserve fail-closed SDF status."""
  from mujoco_metal.simulation import MetalSimulation

  sim = MetalSimulation(
      _analytic_scene(initpoints=40, z=10.0), batch_size=2,
      profile="integrated_euler_v1")
  cc = sim._coupled_constraints
  d = cc.descriptor
  model = sim._mjmodel
  pair = next(i for i, (g1, g2) in enumerate(zip(d.geom1, d.geom2))
              if 8 in (int(model.geom_type[g1]), int(model.geom_type[g2]))
              and 7 not in (int(model.geom_type[g1]), int(model.geom_type[g2])))
  original = cc._contact_sdf_finalize_producer
  overflow = [True]

  def inject_world_zero(*args, **kwargs):
    original(*args, **kwargs)
    if overflow[0]:
      args[6][pair] = -1

  cc._contact_sdf_finalize_producer = inject_world_zero
  snapshot = sim.snapshot()

  # Ordinary assembly clears status, builds current candidates, and must
  # merge the producer's sentinel before its dense/component solver dispatch.
  normal = sim.assembled_system(recompute=True)
  normal_status = normal["status"].detach().cpu().numpy().reshape(-1)
  assert normal_status[0] != 0
  assert normal_status[1] == 0

  # Re-enabling a world after an unmarked pass must clear its old sentinel.
  overflow[0] = False
  sim.reset(qpos=np.tile(np.asarray(d.qpos0, dtype=np.float32), (2, 1)),
            qvel=np.zeros((2, d.nv), dtype=np.float32))
  healthy = sim.assembled_system(recompute=True)
  np.testing.assert_array_equal(
      healthy["status"].detach().cpu().numpy().reshape(-1), [0, 0])

  # Restore the captured state and exercise the cached POS/VEL -> CONSTRAINT
  # route. Its run_velocity_device status reset must merge the retained SDF
  # count and keep the healthy world isolated.
  sim.restore(snapshot)
  overflow[0] = True
  record = sim.prepare_forward_position()
  sim.prepare_forward_velocity(record)
  sim.prepare_forward_actuation(record)
  sim.prepare_forward_acceleration(record)
  cached = sim.prepare_forward_constraint(record, skipsensor=True)
  cached_status = cached["status"].detach().cpu().numpy().reshape(-1)
  assert cached_status[0] != 0
  assert cached_status[1] == 0

  # Restore/re-enable must not let the prior retained count poison either
  # public path once a new position generation has cleared pair counts.
  sim.restore(snapshot)
  overflow[0] = False
  record = sim.prepare_forward_position()
  sim.prepare_forward_velocity(record)
  sim.prepare_forward_actuation(record)
  sim.prepare_forward_acceleration(record)
  cached_healthy = sim.prepare_forward_constraint(record, skipsensor=True)
  np.testing.assert_array_equal(
      cached_healthy["status"].detach().cpu().numpy().reshape(-1), [0, 0])


@_GPU
def test_native_sdf_overflow_status_respects_mask_and_reenable():
  import torch
  from mujoco_metal.simulation import MetalSimulation

  sim = MetalSimulation(
      _analytic_scene(initpoints=40, z=10.0), batch_size=2,
      profile="integrated_euler_v1")
  cc = sim._coupled_constraints
  model = sim._mjmodel
  pair = next(i for i, (g1, g2) in enumerate(zip(
      cc.descriptor.geom1, cc.descriptor.geom2))
              if 8 in (int(model.geom_type[g1]), int(model.geom_type[g2]))
              and 7 not in (int(model.geom_type[g1]), int(model.geom_type[g2])))
  counts = cc._workspace["contact_pair_count"]
  status = cc._workspace["out_status"]
  counts.zero_()
  counts[cc.descriptor.npairs + pair] = -1
  status.zero_()
  mask = torch.tensor([1, 0], dtype=torch.int32, device="mps")

  # A stale masked-world sentinel is ignored by the actual status producer.
  cc._merge_sdf_narrowphase_status_kernel(
      counts, status, mask, cc._constants["pair_contact_offsets_dims"],
      threads=(2,), group_size=(1,))
  np.testing.assert_array_equal(status.detach().cpu().numpy(), [0, 0])

  # Re-enabling that world makes its still-present invalid count visible.
  mask.fill_(1)
  cc._merge_sdf_narrowphase_status_kernel(
      counts, status, mask, cc._constants["pair_contact_offsets_dims"],
      threads=(2,), group_size=(1,))
  np.testing.assert_array_equal(status.detach().cpu().numpy(), [0, 4])

  # Once the producer clears it for a re-enabled pass, stale failure vanishes.
  counts[cc.descriptor.npairs + pair] = 0
  status.zero_()
  cc._merge_sdf_narrowphase_status_kernel(
      counts, status, mask, cc._constants["pair_contact_offsets_dims"],
      threads=(2,), group_size=(1,))
  np.testing.assert_array_equal(status.detach().cpu().numpy(), [0, 0])
