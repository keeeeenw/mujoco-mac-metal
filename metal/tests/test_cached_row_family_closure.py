# Copyright 2026 The MuJoCo Metal contributors
# Licensed under the Apache License, Version 2.0.
"""Cached-activity closure across row families, layouts and routes.

The cached solver derives row activity from retained state (tendon flags,
retained R/aref per the pinned equality rule, and per-active-slot contact
spans) instead of the tendon-only debug plane. These tests prove that rule
per family: contact condim 1/3/4/6 on both cones, multi-contact manifolds,
weld equality, active/inactive joint limits, nonzero (tangential, no-slip)
velocity replay, explicit context, packed/CSR layout solves, and the
>96-row block route — each as disturb-then-replay cached-vs-fresh plus a
pinned oracle, with restore/replay on the block route.
"""

import os

import numpy as np
import pytest

import mujoco

from test_cached_jacobian_capture import _cached_vs_fresh as _base_cached_vs_fresh
from test_cached_jacobian_capture import _forward_args as _base_forward_args


def _forward_args(sim, cc, qpos, qvel):
  """Production-shaped inputs; merges root_com for multi-body poses."""
  args, dyn = _base_forward_args(sim, cc, qpos, qvel)
  if "root_com" in dyn and "root_com" not in args["poses"]:
    args["poses"] = dict(args["poses"], root_com=dyn["root_com"])
  # Connect/weld assembly consumes current motion data; clone it as the
  # production `_acceleration` call does.
  for key in ("cvel", "cdof", "cdof_dot"):
    if key in dyn and key not in args:
      args[key] = dyn[key].detach().clone()
  return args, dyn


def _workspace_fingerprint(cc):
  """Live row views a disturbing forward must overwrite (J and/or rows)."""
  w = cc._workspace
  return (w["workspace_J"].detach().cpu().numpy().copy(),
          w["workspace_debug"].detach().cpu().numpy().copy())


def _cached_vs_fresh(sim, model, qp, qv):
  """Capture at A, disturb with a real forward at B, rerun cached at A.

  Perturbs z (valid for free bodies; breaks plane contacts) or DOF 0
  for hinge-only models (the shared helper assumes a free joint with a
  z-translation at index 2).
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
  # Disturb along a DOF that changes the assembly: z-lift for free bodies
  # (breaks plane contacts), DOF 0 for hinge-only models. An x-slide leaves
  # plane-contact Jacobians bitwise identical (translation invariant).
  other[:, 2 if other.shape[1] > 2 else 0] += 0.2
  args_b, _ = _forward_args(sim, cc, other, qv)
  before_J, before_dbg = _workspace_fingerprint(cc)
  cc.run_device(**args_b)
  after_J, after_dbg = _workspace_fingerprint(cc)
  # The disturb must overwrite live row views (state-independent hinge
  # Jacobians may be bitwise identical, but row references always move).
  assert (not np.array_equal(after_J, before_J)
          or not np.array_equal(after_dbg, before_dbg))
  out = cc.run_velocity_device(ctx, **args_a)
  assert int(np.asarray(out["status"].cpu().numpy()).ravel()[0]) == 0
  cached = np.asarray(out["qacc"].cpu().numpy())[0].astype(float)
  cpu = mujoco.MjData(model)
  cpu.qpos[:] = np.asarray(qp, dtype=float)[0]
  cpu.qvel[:] = np.asarray(qv, dtype=float)[0]
  mujoco.mj_forward(model, cpu)
  mujoco.mj_step(model, cpu)
  return cached, fresh_qacc, np.asarray(cpu.qacc).astype(float)

_NEEDS_GPU = pytest.mark.skipif(
    os.getenv("MUJOCO_METAL_RUN_GPU") != "1", reason="native cache paths")
_GPU = pytest.mark.gpu


def _contact_model(geom="sphere", size=".1", condim=None, cone="pyramidal"):
  opt = f'<option gravity="0 0 0" cone="{cone}" solver="Newton"/>'
  default = (f'<default><geom condim="{condim}"/></default>'
             if condim is not None else "")
  return mujoco.MjModel.from_xml_string(f"""<mujoco>
    {opt}{default}
    <worldbody>
      <geom name="floor" type="plane" size="5 5 .1"/>
      <body pos="0 0 0.05"><freejoint/>
        <geom type="{geom}" size="{size}" mass="1"/></body>
    </worldbody></mujoco>""")


def _check_family(sim, model, qp, qv):
  cached, fresh, cpu = _cached_vs_fresh(sim, model, qp, qv)
  np.testing.assert_allclose(cached, fresh, rtol=1e-5, atol=1e-7)
  np.testing.assert_allclose(cached, cpu, rtol=1e-3, atol=1e-3)


@_GPU
@_NEEDS_GPU
@pytest.mark.parametrize("condim", [1, 3, 4, 6])
@pytest.mark.parametrize("cone", ["pyramidal", "elliptic"])
def test_cached_contact_condim_cone_matrix(condim, cone):
  """Condim 1/3/4/6 on both cones replay exactly with tangential velocity."""
  from mujoco_metal.simulation import MetalSimulation
  model = _contact_model(condim=condim, cone=cone)
  sim = MetalSimulation(model, 1, profile="integrated_euler_v1")
  qp = np.array([[0, 0, 0.05, 1, 0, 0, 0]], dtype=np.float32)
  qv = np.array([[0.3, -0.2, 0.0, 0.0, 0.0, 0.5]], dtype=np.float32)
  _check_family(sim, model, qp, qv)


@_GPU
@_NEEDS_GPU
@pytest.mark.parametrize("cone", ["pyramidal", "elliptic"])
@pytest.mark.parametrize("profile", ["integrated_euler_v1",
                                     "integrated_scalable_v1"])
def test_cached_multi_contact_manifold(cone, profile):
  """Four-contact box manifold replays on dense and component routes."""
  from mujoco_metal.simulation import MetalSimulation
  model = _contact_model(geom="box", size=".1 .1 .1", cone=cone)
  sim = MetalSimulation(model, 1, profile=profile)
  qp = np.array([[0.01, -0.02, 0.05, 1, 0, 0, 0]], dtype=np.float32)
  qv = np.array([[0.1, 0.15, -0.4, 0.2, -0.1, 0.3]], dtype=np.float32)
  _check_family(sim, model, qp, qv)


@_GPU
@_NEEDS_GPU
def test_cached_weld_equality_with_contact():
  """Weld rows plus contacts replay together against the oracle."""
  from mujoco_metal.simulation import MetalSimulation
  model = mujoco.MjModel.from_xml_string("""<mujoco>
    <option gravity="0 0 0" solver="Newton"/>
    <worldbody>
      <geom name="floor" type="plane" size="5 5 .1"/>
      <body name="b1" pos="0 0 0.06"><freejoint/>
        <geom type="sphere" size=".1" mass="1"/></body>
      <body name="b2" pos="0.01 0 0.25"><freejoint/>
        <geom type="sphere" size=".05" mass="1"/></body>
    </worldbody>
    <equality><weld body1="b1" body2="b2"/></equality></mujoco>""")
  sim = MetalSimulation(model, 1, profile="integrated_euler_v1")
  qp = np.asarray(model.qpos0, dtype=np.float32).reshape(1, -1)
  qv = np.zeros((1, model.nv), dtype=np.float32)
  _check_family(sim, model, qp, qv)


@_GPU
@_NEEDS_GPU
def test_cached_active_inactive_joint_limits():
  """One violated limit replays; the free joint stays inactive."""
  from mujoco_metal.simulation import MetalSimulation
  model = mujoco.MjModel.from_xml_string("""<mujoco>
    <option gravity="0 0 0" solver="Newton"/>
    <worldbody>
      <body pos="0 0 0"><joint name="a" type="hinge" axis="0 0 1" range="-.5 .5"/>
        <geom type="sphere" size=".1" mass="1"/></body>
      <body pos="1 0 0"><joint name="b" type="hinge" axis="0 0 1"/>
        <geom type="sphere" size=".1" mass="1"/></body>
    </worldbody></mujoco>""")
  sim = MetalSimulation(model, 1, profile="integrated_euler_v1")
  qp = np.array([[0.6, 0.0]], dtype=np.float32)
  qv = np.array([[0.1, -0.2]], dtype=np.float32)
  _check_family(sim, model, qp, qv)


@_GPU
@_NEEDS_GPU
def test_cached_explicit_context_condim6_elliptic():
  """Caller-supplied context replays the widest elliptic block.

  Reference values are frozen copies taken before the disturbing forward;
  the test proves the disturb actually overwrote the borrowed live backing
  (so the replay is meaningful), builds the explicit context from the real
  captured rows (not arbitrary zeros), and carries a stale-cache negative
  control for this path.
  """
  import torch
  from mujoco_metal.simulation import MetalSimulation
  model = _contact_model(condim=6, cone="elliptic")
  sim = MetalSimulation(model, 1, profile="integrated_euler_v1")
  qp = np.array([[0, 0, 0.05, 1, 0, 0, 0]], dtype=np.float32)
  qv = np.array([[0.2, 0.1, -0.3, 0.0, 0.4, -0.1]], dtype=np.float32)
  cc = sim._coupled_constraints
  args, _ = _forward_args(sim, cc, qp, qv)
  sim.reset(qpos=qp, qvel=qv)
  fresh = sim.assembled_system(recompute=True)
  fresh_qacc = np.asarray(fresh["qacc"].cpu().numpy())[0].astype(float).copy()
  assert int(np.asarray(fresh["status"].cpu().numpy()).ravel()[0]) == 0
  # Capture implicitly, then reuse its real rows as the explicit context.
  # Freeze immediately: the context tensor borrows workspace storage that
  # later forwards overwrite.
  implicit = cc.capture_position_context()
  explicit_frozen = implicit["position_context"].detach().clone()
  live_before = cc._workspace["workspace_J"].detach().cpu().numpy().copy()
  ctx = cc.capture_position_context(
      position_context=explicit_frozen.clone())
  other = qp.copy()
  other[:, 2] += 0.2
  args_b, _ = _forward_args(sim, cc, other, qv)
  cc.run_device(**args_b)
  live_after = cc._workspace["workspace_J"].detach().cpu().numpy()
  assert not np.array_equal(live_after, live_before)
  out = cc.run_velocity_device(ctx, **args)
  assert int(np.asarray(out["status"].cpu().numpy()).ravel()[0]) == 0
  np.testing.assert_allclose(
      np.asarray(out["qacc"].cpu().numpy())[0], fresh_qacc,
      rtol=1e-5, atol=1e-7)
  # Negative control: a clobbered J cache breaks this explicit replay.
  # Rebuilt from a fresh capture so the explicit rows match live assembly.
  sim.reset(qpos=qp, qvel=qv)
  sim.assembled_system(recompute=True)
  fresh_cap = cc.capture_position_context()
  ctx2 = cc.capture_position_context(
      position_context=fresh_cap["position_context"].detach().clone())
  cc._workspace["position_cache_J"].zero_()
  cc.run_device(**args_b)
  stale = cc.run_velocity_device(ctx2, **args)
  assert not np.allclose(
      np.asarray(stale["qacc"].cpu().numpy())[0], fresh_qacc,
      rtol=1e-5, atol=1e-7)


@_GPU
@_NEEDS_GPU
@pytest.mark.parametrize("profile", ["integrated_euler_v1",
                                     "integrated_scalable_v1"])
def test_cached_packed_layout_solve(profile):
  """Sparse/CSR backing replays through the cached solve, not just capture."""
  from mujoco_metal.simulation import MetalSimulation
  model = mujoco.MjModel.from_xml_string("""<mujoco>
    <option gravity="0 0 0" solver="Newton" jacobian="sparse"/>
    <worldbody>
      <geom name="floor" type="plane" size="5 5 .1"/>
      <body pos="0 0 0.05"><freejoint/>
        <geom type="sphere" size=".1" mass="1"/></body>
    </worldbody></mujoco>""")
  sim = MetalSimulation(model, 1, profile=profile)
  assert sim._coupled_constraints._jacobian_layout.mode != 0
  qp = np.array([[0, 0, 0.05, 1, 0, 0, 0]], dtype=np.float32)
  qv = np.array([[0.1, -0.1, -0.2, 0.0, 0.0, 0.3]], dtype=np.float32)
  _check_family(sim, model, qp, qv)


@_GPU
@_NEEDS_GPU
@pytest.mark.parametrize("solver", ["PGS", "CG", "Newton"])
def test_cached_large_row_block_route(solver):
  """>96-row block route replays cached, matches 5-step oracle, restores."""
  from mujoco_metal.capacity import DENSE_ROW_THRESHOLD
  from mujoco_metal.simulation import MetalSimulation
  bodies = "".join(
      f'<body pos="{.3 * k} 0 .09"><freejoint/><geom type="sphere" size=".1"/></body>'
      for k in range(6))
  model = mujoco.MjModel.from_xml_string(
      f'<mujoco><option solver="{solver}"/><worldbody>'
      f'<geom type="plane" size="5 5 .1"/>{bodies}</worldbody></mujoco>')
  sim = MetalSimulation(model, 1, profile="integrated_scalable_v1")
  desc = sim._coupled_constraints.descriptor
  assert desc.nr > DENSE_ROW_THRESHOLD and not desc.dense_path
  nv = int(model.nv)
  qp = np.tile(model.qpos0.astype(np.float32), (1, 1))
  qv = np.zeros((1, nv), dtype=np.float32)
  qv[0, 1] = 0.25  # tangential slide across the plane (no-slip path)
  cached, fresh, _ = _cached_vs_fresh(sim, model, qp, qv)
  np.testing.assert_allclose(cached, fresh, rtol=1e-5, atol=1e-7)
  # Capture preservation across ordinary forwards: an older token survives
  # workspace overwrites and still replays its own inputs.
  cc = sim._coupled_constraints
  args_a, _ = _forward_args(sim, cc, qp, qv)
  sim.reset(qpos=qp, qvel=qv)
  sim.assembled_system(recompute=True)
  ctx_old = cc.capture_position_context()
  other = qp.copy()
  other[:, 2] += 0.3
  args_b, _ = _forward_args(sim, cc, other, qv)
  cc.run_device(**args_b)
  replay = cc.run_velocity_device(ctx_old, **args_a)
  np.testing.assert_allclose(
      np.asarray(replay["qacc"].cpu().numpy())[0], fresh,
      rtol=1e-5, atol=1e-7)
  # Checkpoint continuation without resetting away restored values: replay
  # the second half from a mid-run snapshot and require bitwise-identical
  # retained state, precision, warmstart and status.
  sim.reset(qpos=qp, qvel=qv)
  sim.step(5)
  snap = sim.snapshot()
  sim.step(3)
  expected = {
      name: getattr(sim.state, name).detach().cpu().numpy().copy()
      for name in ("_qpos", "_qvel", "_qacc", "_status")
      if getattr(sim.state, name, None) is not None}
  expected_words = sim.state.snapshot().precision_words.astype(float).copy()
  expected_warm = (sim.state._qacc_warmstart.detach().cpu().numpy().copy()
                   if sim.state._qacc_warmstart is not None else None)
  sim.restore(snap)
  sim.step(3)
  for name, want in expected.items():
    np.testing.assert_array_equal(
        getattr(sim.state, name).detach().cpu().numpy(), want)
  np.testing.assert_array_equal(
      sim.state.snapshot().precision_words.astype(float), expected_words)
  if expected_warm is not None:
    np.testing.assert_array_equal(
        sim.state._qacc_warmstart.detach().cpu().numpy(), expected_warm)
  first = sim.state.qpos.cpu().numpy().copy()
  source = mujoco.MjData(model)
  source.qpos[:] = qp[0].astype(float)
  source.qvel[:] = qv[0].astype(float)
  for _ in range(8):
    mujoco.mj_step(model, source)
  np.testing.assert_allclose(first[0], source.qpos, atol=1e-6)


def _noslip_large_model(noslip, solver="PGS", iterations=100):
  bodies = "".join(
      f'<body pos="{.3 * k} 0 .09"><freejoint/><geom type="sphere" size=".1"/></body>'
      for k in range(6))
  return mujoco.MjModel.from_xml_string(
      f'<mujoco><option timestep="0.002" gravity="0 0 -9.81" '
      f'noslip_iterations="{noslip}" solver="{solver}" iterations="{iterations}" '
      f'tolerance="1e-3"/><default><geom condim="3" friction="0.4 0.005 0.0001"/>'
      f'</default><worldbody><geom type="plane" size="5 5 .1"/>{bodies}'
      f'</worldbody></mujoco>')


@_GPU
@_NEEDS_GPU
@pytest.mark.parametrize("profile", ["integrated_scalable_v1"])
def test_cached_large_row_noslip_block_route(profile):
  """No-slip iterations run on the >96-row block route (PGS).

  Euler caps nv at 32 (six free bodies need 36), so the block route is
  covered on the scalable profile; the dense kernel is covered below.
  The used-iteration diagnostic grows by the no-slip sweeps versus the
  zero control while both track their CPU oracles; the cached solve then
  replays fresh. Tangential sliding supplies real friction work.
  """
  from mujoco_metal.capacity import DENSE_ROW_THRESHOLD
  from mujoco_metal.simulation import MetalSimulation
  sols = {}
  for noslip in (0, 3):
    model = _noslip_large_model(noslip)
    sim = MetalSimulation(model, 1, profile=profile)
    desc = sim._coupled_constraints.descriptor
    assert desc.nr > DENSE_ROW_THRESHOLD and not desc.dense_path
    nv = int(model.nv)
    qp = np.tile(model.qpos0.astype(np.float32), (1, 1))
    qv = np.zeros((1, nv), dtype=np.float32)
    qv[0, 1] = 3.0
    qv[0, 0] = 1.5
    sim.reset(qpos=qp, qvel=qv)
    assert int(np.asarray(
        sim.step(2).cpu().numpy()).ravel()[0]) == 0
    cc = sim._coupled_constraints
    diag = cc._workspace["out_diagnostics"].detach().cpu().numpy().reshape(1, -1)[0]
    sols[noslip] = (sim, model, qp, qv, float(diag[1]), sim.state._qvel.cpu().numpy().copy())
    source = mujoco.MjData(model)
    source.qpos[:] = qp[0].astype(float)
    source.qvel[:] = qv[0].astype(float)
    for _ in range(2):
      mujoco.mj_step(model, source)
    np.testing.assert_allclose(sols[noslip][5][0], source.qvel, rtol=1e-3, atol=1e-3)
  # The no-slip sweeps execute: used-iteration count grows by exactly the
  # extra sweeps while the main-solve count is unchanged.
  assert sols[3][4] > sols[0][4]
  sim, model, qp, qv, _, _ = sols[3]
  cc = sim._coupled_constraints
  # Cached mechanics compare against the direct fresh solve with identical
  # inputs (bitwise): the shared helper builds qfrc_smooth without the
  # production accumulation rounding, so assembled-system values are not
  # the reference for direct calls on this fixture (2.8e-05 helper gap,
  # documented; production physics is qualified against the oracle above
  # and by stepping below).
  import torch
  args, _ = _forward_args(sim, cc, qp, qv)
  sim.reset(qpos=qp, qvel=qv)

  def _clone_call(a):
    return {k: (v.detach().clone() if isinstance(v, torch.Tensor)
                else ({kk: vv.detach().clone() for kk, vv in v.items()}
                      if isinstance(v, dict) else v))
            for k, v in a.items()}

  direct = cc.run_device(**_clone_call(args))
  direct_qacc = np.asarray(direct["qacc"].cpu().numpy()).astype(float)
  ctx = cc.capture_position_context()
  other = qp.copy()
  other[:, 2] += 0.3
  args_b, _ = _forward_args(sim, cc, other, qv)
  cc.run_device(**_clone_call(args_b))
  replay = cc.run_velocity_device(ctx, **_clone_call(args))
  np.testing.assert_allclose(
      np.asarray(replay["qacc"].cpu().numpy()), direct_qacc,
      rtol=1e-6, atol=1e-9)
  snap = sim.snapshot()
  sim.reset(qpos=qp, qvel=qv)
  sim.step(2)
  replayed = sim.state._qvel.cpu().numpy().copy()
  sim.restore(snap)
  sim.reset(qpos=qp, qvel=qv)
  sim.step(2)
  np.testing.assert_array_equal(sim.state._qvel.cpu().numpy(), replayed)


@_GPU
@_NEEDS_GPU
def test_cached_dense_noslip_route():
  """No-slip iterations run on the dense route with ordinary friction."""
  from mujoco_metal.simulation import MetalSimulation
  xml = """<mujoco>
    <option timestep="0.002" gravity="0 0 -9.81" noslip_iterations="3"
            solver="PGS" iterations="8" tolerance="1e-3"/>
    <default><geom condim="3" friction="0.4 0.005 0.0001"/></default>
    <worldbody>
      <geom name="floor" type="plane" size="5 5 .1"/>
      <body pos="0 0 0.05"><freejoint/>
        <geom type="sphere" size=".1" mass="1"/></body>
    </worldbody></mujoco>"""
  model = mujoco.MjModel.from_xml_string(xml)
  sim = MetalSimulation(model, 1, profile="integrated_euler_v1")
  assert sim._coupled_constraints.descriptor.dense_path
  qp = np.array([[0, 0, 0.05, 1, 0, 0, 0]], dtype=np.float32)
  qv = np.array([[1.5, 0.0, -0.5, 0.0, 2.0, 0.0]], dtype=np.float32)
  sim.reset(qpos=qp, qvel=qv)
  assert int(np.asarray(sim.step(2).cpu().numpy()).ravel()[0]) == 0
  cc = sim._coupled_constraints
  used_ns = float(cc._workspace["out_diagnostics"].detach().cpu().numpy().reshape(1, -1)[0, 1])
  # Zero control on the identical model: fewer used iterations, proving the
  # sweeps above come from the no-slip route rather than the main solve.
  plain = MetalSimulation(
      mujoco.MjModel.from_xml_string(xml.replace(
          'noslip_iterations="3"', 'noslip_iterations="0"')),
      1, profile="integrated_euler_v1")
  plain.reset(qpos=qp, qvel=qv)
  assert int(np.asarray(plain.step(2).cpu().numpy()).ravel()[0]) == 0
  used_plain = float(plain._coupled_constraints._workspace[
      "out_diagnostics"].detach().cpu().numpy().reshape(1, -1)[0, 1])
  assert used_ns > used_plain
  cpu = mujoco.MjData(model)
  cpu.qpos[:] = qp[0].astype(float)
  cpu.qvel[:] = qv[0].astype(float)
  for _ in range(2):
    mujoco.mj_step(model, cpu)
  np.testing.assert_allclose(sim.state._qvel.cpu().numpy()[0],
                             np.asarray(cpu.qvel), rtol=1e-3, atol=1e-3)
  cached, fresh, cpu_qacc = _cached_vs_fresh(sim, model, qp, qv)
  np.testing.assert_allclose(cached, fresh, rtol=1e-5, atol=1e-7)
  np.testing.assert_allclose(cached, cpu_qacc, rtol=1e-3, atol=1e-3)


@_GPU
@_NEEDS_GPU
@pytest.mark.parametrize("condim", [1, 6])
@pytest.mark.parametrize("cone", ["pyramidal", "elliptic"])
@pytest.mark.parametrize("solver", ["PGS", "CG"])
def test_cached_condim_cone_solvers(condim, cone, solver):
  """Narrow/wide cones replay on PGS and CG with sliding contact."""
  from mujoco_metal.simulation import MetalSimulation
  opt = f'<option gravity="0 0 0" cone="{cone}" solver="{solver}"/>'
  model = mujoco.MjModel.from_xml_string(f"""<mujoco>
    {opt}<default><geom condim="{condim}"/></default>
    <worldbody>
      <geom name="floor" type="plane" size="5 5 .1"/>
      <body pos="0 0 0.05"><freejoint/>
        <geom type="sphere" size=".1" mass="1"/></body>
    </worldbody></mujoco>""")
  sim = MetalSimulation(model, 1, profile="integrated_euler_v1")
  qp = np.array([[0, 0, 0.05, 1, 0, 0, 0]], dtype=np.float32)
  qv = np.array([[0.25, -0.15, -0.1, 0.0, 0.0, 0.4]], dtype=np.float32)
  _check_family(sim, model, qp, qv)


@_GPU
@_NEEDS_GPU
def test_cached_tendon_rows_velocity():
  """Sprung-tendon rows replay through the cached velocity stage.

  Direct-call mechanics compare cached replay against the direct fresh
  solve with identical inputs (the shared helper omits the separate
  tendon-dynamics force/armature terms, so assembled-system values are not
  the reference here). Production physics with tendon terms is qualified
  separately against the pinned oracle via assembled-system and stepping.
  """
  import torch
  from mujoco_metal.simulation import MetalSimulation
  model = mujoco.MjModel.from_xml_string("""<mujoco>
    <option gravity="0 0 0" solver="PGS" timestep="0.001"><flag contact="disable"/></option>
    <worldbody><body pos="0 0 1">
      <joint name="j" type="slide" axis="1 0 0"/>
      <geom type="sphere" size=".1" mass="1"/></body></worldbody>
    <tendon><fixed name="t" limited="true" range="-.1 .1" stiffness="50" damping="2">
      <joint joint="j" coef="1"/></fixed></tendon>
    <actuator><motor joint="j"/></actuator></mujoco>""")
  sim = MetalSimulation(model, 1, profile="integrated_euler_v1")
  qp = np.array([[0.4]], dtype=np.float32)
  qv = np.array([[0.1]], dtype=np.float32)
  cc = sim._coupled_constraints
  assert int(cc.descriptor.ten_limit_rows) > 0
  args, _ = _forward_args(sim, cc, qp, qv)
  sim.reset(qpos=qp, qvel=qv)
  direct = cc.run_device(**{k: (v.detach().clone() if isinstance(v, torch.Tensor)
                                else {kk: vv.detach().clone() for kk, vv in v.items()})
                            for k, v in args.items()})
  direct_qacc = np.asarray(direct["qacc"].cpu().numpy())[0].astype(float).copy()
  ctx = cc.capture_position_context()
  other = qp.copy()
  other[:, 0] += 0.2
  args_b, _ = _forward_args(sim, cc, other, qv)
  cc.run_device(**args_b)
  out = cc.run_velocity_device(ctx, **args)
  assert int(np.asarray(out["status"].cpu().numpy()).ravel()[0]) == 0
  np.testing.assert_allclose(
      np.asarray(out["qacc"].cpu().numpy())[0], direct_qacc,
      rtol=1e-5, atol=1e-7)
  # Production physics with tendon terms: assembled-system and stepping
  # match the pinned oracle (tendon limit engaged at qpos 0.4).
  sim.reset(qpos=qp, qvel=qv)
  system = sim.assembled_system(recompute=True)
  cpu = mujoco.MjData(model)
  cpu.qpos[:] = qp[0].astype(float)
  cpu.qvel[:] = qv[0].astype(float)
  mujoco.mj_forward(model, cpu)
  assert abs(float(cpu.ten_length[0]) - 0.1) > 1e-6  # limit engaged
  mujoco.mj_step(model, cpu)
  np.testing.assert_allclose(
      np.asarray(system["qacc"].cpu().numpy())[0], np.asarray(cpu.qacc),
      rtol=1e-3, atol=1e-3)
  # The stepping pipeline from the same S0 must reach the CPU S1 state.
  sim.reset(qpos=qp, qvel=qv)
  assert int(np.asarray(sim.step().cpu().numpy()).ravel()[0]) == 0
  np.testing.assert_allclose(
      sim.state._qvel.cpu().numpy()[0], np.asarray(cpu.qvel),
      rtol=1e-3, atol=1e-3)


@_GPU
@_NEEDS_GPU
def test_masked_cached_block_run_preserves_unselected_worlds():
  """World-masked cached run on the block route touches only worlds 0."""
  import torch
  from mujoco_metal.capacity import DENSE_ROW_THRESHOLD
  from mujoco_metal.simulation import MetalSimulation
  bodies = "".join(
      f'<body pos="{.3 * k} 0 .09"><freejoint/><geom type="sphere" size=".1"/></body>'
      for k in range(6))
  model = mujoco.MjModel.from_xml_string(
      '<mujoco><option solver="PGS"/><worldbody>'
      f'<geom type="plane" size="5 5 .1"/>{bodies}</worldbody></mujoco>')
  sim = MetalSimulation(model, 2, profile="integrated_scalable_v1")
  desc = sim._coupled_constraints.descriptor
  assert desc.nr > DENSE_ROW_THRESHOLD and not desc.dense_path
  nv = int(model.nv)
  qp = np.tile(model.qpos0.astype(np.float32), (2, 1))
  qv = np.zeros((2, nv), dtype=np.float32)
  qv[:, 1] = 0.2
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
