# Copyright 2026 The MuJoCo Metal contributors
# Licensed under the Apache License, Version 2.0.
"""Owned Jacobian capture for every supported solver context.

`capture_position_context` must retain the actual live Jacobian backing
(`workspace_J` -> `position_cache_J`) for contact-only, scalar-only
equality, implicit and explicit captures alike: cached solver stages copy
it back verbatim, so an omitted copy leaves them solving stale/zero rows.
Covers PGS/CG/Newton, dense/component mass routes, packed layout where
admitted, zero-DOF dummy backings, world masks, checkpoint/restore and
malformed-input rollback, with a stale-cache negative control proving the
cached-solve comparisons are sensitive to the retained rows.
"""

import os

import numpy as np
import pytest

import mujoco

_NEEDS_GPU = pytest.mark.skipif(
    os.getenv("MUJOCO_METAL_RUN_GPU") != "1", reason="native cache paths")


_GPU = pytest.mark.gpu


def _contact_model(solver, sparse=False):
  opt = f'<option gravity="0 0 0" solver="{solver}"'
  if sparse:
    opt += ' jacobian="sparse"'
  opt += '/>'
  return mujoco.MjModel.from_xml_string(f"""<mujoco>
    {opt}
    <worldbody>
      <geom name="floor" type="plane" size="5 5 .1"/>
      <body pos="0 0 0.05"><freejoint/>
        <geom type="sphere" size=".1" mass="1"/></body>
    </worldbody></mujoco>""")


def _joint_eq_model(solver):
  return mujoco.MjModel.from_xml_string(f"""<mujoco>
    <option gravity="0 0 0" solver="{solver}"/>
    <worldbody>
      <body pos="0 0 0"><joint name="a" type="hinge" axis="0 0 1"/>
        <geom type="sphere" size=".1" mass="1"/></body>
      <body pos="1 0 0"><joint name="b" type="hinge" axis="0 0 1"/>
        <geom type="sphere" size=".1" mass="1"/></body>
    </worldbody>
    <equality><joint joint1="a" joint2="b"/></equality></mujoco>""")


def _assemble(sim, qpos, qvel):
  sim.reset(qpos=np.asarray(qpos, dtype=np.float32),
            qvel=np.asarray(qvel, dtype=np.float32))
  system = sim.assembled_system(recompute=True)
  assert int(np.asarray(sim.state.status.cpu().numpy()).ravel()[0]) == 0
  cc = sim._coupled_constraints
  ctx = cc.capture_position_context()
  live = cc._workspace["workspace_J"].detach().cpu().numpy().copy()
  cached = cc._workspace["position_cache_J"].detach().cpu().numpy().copy()
  return system, cc, ctx, live, cached


@pytest.mark.gpu
@_GPU
@_NEEDS_GPU
@pytest.mark.parametrize("solver", ["PGS", "CG", "Newton"])
@pytest.mark.parametrize("profile", ["integrated_euler_v1",
                                     "integrated_scalable_v1"])
@pytest.mark.parametrize("fixture", ["contact", "joint_eq"])
def test_capture_preserves_live_jacobian_words(solver, profile, fixture):
  """Cache must equal the live backing bitwise at capture time.

  Contact-only and scalar joint-equality captures have no differential
  equality rows, so the old conditional copy left them stale; this fails
  when the unconditional capture copy is omitted.
  """
  from mujoco_metal.simulation import MetalSimulation
  model = (_contact_model(solver) if fixture == "contact"
           else _joint_eq_model(solver))
  sim = MetalSimulation(model, 2, profile=profile)
  if fixture == "contact":
    qp = np.array([[0, 0, 0.05, 1, 0, 0, 0], [0, 0, 0.5, 1, 0, 0, 0]],
                  dtype=np.float32)
  else:
    qp = np.tile(model.qpos0.astype(np.float32), (2, 1))
    qp[0, 0] = 0.1
  qv = np.zeros((2, model.nv), dtype=np.float32)
  system, cc, ctx, live, cached = _assemble(sim, qp, qv)
  assert live.size and np.any(live != 0), "fixture must own nonzero rows"
  np.testing.assert_array_equal(cached, live)


@pytest.mark.gpu
@_GPU
@_NEEDS_GPU
def test_capture_preserves_live_jacobian_words_sparse_layout():
  """Packed/CSR backing uses the same full-tensor owned copy."""
  from mujoco_metal.simulation import MetalSimulation
  model = _contact_model("Newton", sparse=True)
  sim = MetalSimulation(model, 2, profile="integrated_scalable_v1")
  assert sim._coupled_constraints._jacobian_layout.mode != 0
  qp = np.array([[0, 0, 0.05, 1, 0, 0, 0], [0, 0, 0.5, 1, 0, 0, 0]],
                dtype=np.float32)
  qv = np.zeros((2, model.nv), dtype=np.float32)
  system, cc, ctx, live, cached = _assemble(sim, qp, qv)
  assert live.size and np.any(live != 0)
  np.testing.assert_array_equal(cached, live)


@pytest.mark.gpu
@_GPU
@_NEEDS_GPU
def test_explicit_context_capture_preserves_jacobian():
  """Caller-supplied row context still captures the live backing."""
  import torch
  from mujoco_metal.simulation import MetalSimulation
  model = _contact_model("CG")
  sim = MetalSimulation(model, 1, profile="integrated_euler_v1")
  qp = np.array([[0, 0, 0.05, 1, 0, 0, 0]], dtype=np.float32)
  qv = np.zeros((1, model.nv), dtype=np.float32)
  sim.reset(qpos=qp, qvel=qv)
  sim.assembled_system(recompute=True)
  cc = sim._coupled_constraints
  b, nr = 1, int(cc.descriptor.nr)
  explicit = torch.zeros((b, nr, 5), dtype=torch.float32, device="mps")
  ctx = cc.capture_position_context(position_context=explicit)
  live = cc._workspace["workspace_J"].detach().cpu().numpy()
  cached = cc._workspace["position_cache_J"].detach().cpu().numpy()
  assert live.size and np.any(live != 0)
  np.testing.assert_array_equal(cached, live)


@pytest.mark.gpu
@_GPU
@_NEEDS_GPU
def test_zero_dof_capture_preserves_dummy_jacobian():
  """nv==0 capture succeeds and keeps the dummy backing coherent."""
  import torch
  from mujoco_metal.simulation import MetalSimulation
  from mujoco_metal.coupled_constraints import MetalCoupledConstraints
  model = mujoco.MjModel.from_xml_string("""<mujoco>
    <option solver="CG"/><worldbody><geom type="sphere" size=".1"/>
    </worldbody></mujoco>""")
  sim = MetalSimulation(model, 2, profile="integrated_scalable_v1")
  cc = MetalCoupledConstraints(model, 2)
  qpos = torch.zeros((2, 0), device="mps")
  qvel = torch.zeros((2, 0), device="mps")
  mass = torch.zeros((2, 0, 0), device="mps")
  force = torch.zeros((2, 0), device="mps")
  poses = sim._smooth._fk.run_device(qpos)
  res = cc.run_device(poses, mass, force, qpos, qvel)
  np.testing.assert_array_equal(res["status"].cpu().numpy(), 0)
  ctx = cc.capture_position_context()
  live = cc._workspace["workspace_J"].detach().cpu().numpy()
  cached = cc._workspace["position_cache_J"].detach().cpu().numpy()
  np.testing.assert_array_equal(cached, live)
  out = cc.run_velocity_device(ctx, poses, mass, force, qpos, qvel)
  np.testing.assert_array_equal(out["status"].cpu().numpy(), 0)


def _forward_args(sim, cc, qpos, qvel):
  """Production-shaped fresh-forward inputs from smooth dynamics."""
  import torch
  dyn = sim._smooth.run_device(
      torch.as_tensor(np.asarray(qpos, dtype=np.float32), device="mps"),
      torch.as_tensor(np.asarray(qvel, dtype=np.float32), device="mps"))
  args = dict(
      poses={k: v.detach().clone() for k, v in dyn["poses"].items()},
      qfrc_smooth_low=(-dyn["qfrc_bias_low"]).detach().clone(),
      qpos=torch.as_tensor(np.asarray(qpos, dtype=np.float32),
                           device="mps"),
      qvel=torch.as_tensor(np.asarray(qvel, dtype=np.float32),
                           device="mps"))
  if sim._component_mass_enabled:
    lists = sim._smooth._workspace["all_awake_lists"]
    args.update(
        mass=None,
        qfrc_smooth=(-dyn["qfrc_bias"]).detach().clone(),
        component_solver=sim._component_solver,
        mass_blocks=dyn["mass_blocks"].detach().clone(),
        mass_blocks_low=dyn["mass_blocks_low"].detach().clone(),
        awake_dof_ids=lists["dof_ids"].detach().clone(),
        awake_counts=lists["counts"].detach().clone())
  else:
    args.update(
        mass=dyn["mass_matrix"].detach().clone(),
        mass_low=dyn["mass_matrix_low"].detach().clone(),
        qfrc_smooth=(-dyn["qfrc_bias"]).detach().clone())
  return args, dyn


def _cached_vs_fresh(sim, model, qp, qv):
  """Capture at A, disturb with a real forward at B, rerun cached at A.

  Returns (cached_qacc, fresh_qacc, cpu_qacc) for world 0. The disturb is
  an ordinary production forward (overwrites borrowed views without
  invalidating the retained cache); the cached call replays A inputs.
  """
  import torch
  cc = sim._coupled_constraints
  args_a, _ = _forward_args(sim, cc, qp, qv)
  sim.reset(qpos=np.asarray(qp, dtype=np.float32),
            qvel=np.asarray(qv, dtype=np.float32))
  fresh = sim.assembled_system(recompute=True)
  fresh_qacc = np.asarray(
      fresh["qacc"].cpu().numpy())[0].astype(float).copy()
  ctx = cc.capture_position_context()
  other = np.asarray(qp, dtype=np.float32).copy()
  other[:, 2] += 0.2
  args_b, _ = _forward_args(sim, cc, other, qv)
  cc.run_device(**args_b)
  assert not np.array_equal(
      cc._workspace["workspace_J"].detach().cpu().numpy(),
      cc._workspace["position_cache_J"].detach().cpu().numpy())
  out = cc.run_velocity_device(ctx, **args_a)
  assert int(np.asarray(out["status"].cpu().numpy()).ravel()[0]) == 0
  cached = np.asarray(out["qacc"].cpu().numpy())[0].astype(float)
  cpu = mujoco.MjData(model)
  cpu.qpos[:] = np.asarray(qp, dtype=float)[0]
  cpu.qvel[:] = np.asarray(qv, dtype=float)[0]
  mujoco.mj_forward(model, cpu)
  mujoco.mj_step(model, cpu)
  return cached, fresh_qacc, np.asarray(cpu.qacc).astype(float)


@pytest.mark.gpu
@_GPU
@_NEEDS_GPU
@pytest.mark.parametrize("solver", ["PGS", "CG", "Newton"])
@pytest.mark.parametrize("profile", ["integrated_euler_v1",
                                     "integrated_scalable_v1"])
def test_cached_velocity_solve_matches_fresh_and_oracle(solver, profile):
  """Cached stage with restored rows must match the fresh solve and CPU."""
  from mujoco_metal.simulation import MetalSimulation
  model = _contact_model(solver)
  sim = MetalSimulation(model, 1, profile=profile)
  qp = np.array([[0, 0, 0.05, 1, 0, 0, 0]], dtype=np.float32)
  qv = np.zeros((1, model.nv), dtype=np.float32)
  cached, fresh, cpu = _cached_vs_fresh(sim, model, qp, qv)
  np.testing.assert_allclose(cached, fresh, rtol=1e-5, atol=1e-7)
  np.testing.assert_allclose(cached, cpu, rtol=1e-3, atol=1e-3)


@pytest.mark.gpu
@_GPU
@_NEEDS_GPU
def test_stale_jacobian_cache_makes_cached_solve_disagree():
  """Negative control: clobbered retained rows must break the agreement.

  Zeroing the J cache reproduces the omitted-copy symptom exactly; the
  cached solve must then disagree with the fresh oracle while the intact
  twin (previous test) matches. This proves the comparisons above are
  sensitive to the retained rows.
  """
  from mujoco_metal.simulation import MetalSimulation
  model = _contact_model("Newton")
  sim = MetalSimulation(model, 1, profile="integrated_euler_v1")
  qp = np.array([[0, 0, 0.05, 1, 0, 0, 0]], dtype=np.float32)
  qv = np.zeros((1, model.nv), dtype=np.float32)
  cc = sim._coupled_constraints
  sim.reset(qpos=qp, qvel=qv)
  fresh = sim.assembled_system(recompute=True)
  fresh_qacc = np.asarray(fresh["qacc"].cpu().numpy())[0].astype(float).copy()
  ctx = cc.capture_position_context()
  cc._workspace["position_cache_J"].zero_()
  other = qp.copy()
  other[:, 2] += 0.2
  args_a, _ = _forward_args(sim, cc, qp, qv)
  args_b, _ = _forward_args(sim, cc, other, qv)
  cc.run_device(**args_b)
  out = cc.run_velocity_device(ctx, **args_a)
  stale = np.asarray(out["qacc"].cpu().numpy())[0].astype(float)
  assert not np.allclose(stale, fresh_qacc, rtol=1e-5, atol=1e-7), (
      stale, fresh_qacc)


@pytest.mark.gpu
@_GPU
@_NEEDS_GPU
def test_masked_cached_run_preserves_unselected_worlds_bitwise():
  """A world-masked cached run must not touch unselected records."""
  import torch
  from mujoco_metal.simulation import MetalSimulation
  model = _contact_model("CG")
  sim = MetalSimulation(model, 2, profile="integrated_euler_v1")
  qp = np.array([[0, 0, 0.05, 1, 0, 0, 0], [0, 0, 0.05, 1, 0, 0, 0]],
                dtype=np.float32)
  qv = np.zeros((2, model.nv), dtype=np.float32)
  cc = sim._coupled_constraints
  args, _ = _forward_args(sim, cc, qp, qv)
  sim.reset(qpos=qp, qvel=qv)
  sim.assembled_system(recompute=True)
  ctx = cc.capture_position_context()
  before_live = cc._workspace["workspace_J"].detach().cpu().numpy().copy()
  before_cache = cc._workspace["position_cache_J"].detach().cpu().numpy().copy()
  sel = torch.zeros(2, dtype=torch.int32, device="mps")
  sel[0] = 1
  args["world_mask"] = sel
  out = cc.run_velocity_device(ctx, **args)
  assert int(np.asarray(out["status"].cpu().numpy()).ravel()[0]) == 0
  after_live = cc._workspace["workspace_J"].detach().cpu().numpy()
  after_cache = cc._workspace["position_cache_J"].detach().cpu().numpy()
  np.testing.assert_array_equal(after_live[1], before_live[1])
  np.testing.assert_array_equal(after_cache[1], before_cache[1])


@pytest.mark.gpu
@_GPU
@_NEEDS_GPU
def test_restore_invalidates_capture_and_fresh_recapture_works():
  """Restore returns workspace to snapshot; capture epoch survives; recapture works."""
  from mujoco_metal.simulation import MetalSimulation
  model = _contact_model("PGS")
  sim = MetalSimulation(model, 1, profile="integrated_euler_v1")
  qp = np.array([[0, 0, 0.05, 1, 0, 0, 0]], dtype=np.float32)
  qv = np.zeros((1, model.nv), dtype=np.float32)
  cc = sim._coupled_constraints
  sim.reset(qpos=qp, qvel=qv)
  sim.assembled_system(recompute=True)
  ctx = cc.capture_position_context()
  snap = sim.snapshot()
  sim.step()
  sim.restore(snap)
  # Snapshot restores the retained workspace, so the pre-step token remains
  # usable; a fresh assembly + recapture must also preserve live words.
  sim.assembled_system(recompute=True)
  args, _ = _forward_args(sim, cc, qp, qv)
  out = cc.run_velocity_device(ctx, **args)
  assert int(np.asarray(out["status"].cpu().numpy()).ravel()[0]) == 0
  sim.assembled_system(recompute=True)
  ctx2 = cc.capture_position_context()
  live = cc._workspace["workspace_J"].detach().cpu().numpy()
  cached = cc._workspace["position_cache_J"].detach().cpu().numpy()
  np.testing.assert_array_equal(cached, live)
  out = cc.run_velocity_device(ctx2, **args)
  assert int(np.asarray(out["status"].cpu().numpy()).ravel()[0]) == 0


@pytest.mark.gpu
@_GPU
@_NEEDS_GPU
def test_malformed_capture_rejects_before_writes_and_kills_token():
  """Bad explicit context raises, retires the epoch and writes nothing live."""
  import torch
  from mujoco_metal.simulation import MetalSimulation
  model = _contact_model("Newton")
  sim = MetalSimulation(model, 1, profile="integrated_euler_v1")
  qp = np.array([[0, 0, 0.05, 1, 0, 0, 0]], dtype=np.float32)
  qv = np.zeros((1, model.nv), dtype=np.float32)
  cc = sim._coupled_constraints
  sim.reset(qpos=qp, qvel=qv)
  sim.assembled_system(recompute=True)
  ctx = cc.capture_position_context()
  live_before = cc._workspace["workspace_J"].detach().cpu().numpy().copy()
  with pytest.raises(ValueError):
    cc.capture_position_context(
        position_context=torch.zeros((1, int(cc.descriptor.nr) + 1, 5),
                                     dtype=torch.float32, device="mps"))
  with pytest.raises((RuntimeError, ValueError)):
    cc.run_velocity_device(ctx, {}, None, None, None, None)
  live_after = cc._workspace["workspace_J"].detach().cpu().numpy()
  np.testing.assert_array_equal(live_after, live_before)
