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
  """Caller-supplied context replays the widest elliptic block."""
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
  nr = int(cc.descriptor.nr)
  explicit = torch.zeros((1, nr, 5), dtype=torch.float32, device="mps")
  ctx = cc.capture_position_context(position_context=explicit)
  other = qp.copy()
  other[:, 2] += 0.2
  args_b, _ = _forward_args(sim, cc, other, qv)
  cc.run_device(**args_b)
  out = cc.run_velocity_device(ctx, **args)
  np.testing.assert_allclose(
      np.asarray(out["qacc"].cpu().numpy())[0],
      np.asarray(fresh["qacc"].cpu().numpy())[0], rtol=1e-5, atol=1e-7)


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
  snap = sim.snapshot()
  sim.reset(qpos=qp, qvel=qv)
  sim.step(5)
  first = sim.state.qpos.cpu().numpy().copy()
  source = mujoco.MjData(model)
  source.qpos[:] = qp[0].astype(float)
  source.qvel[:] = qv[0].astype(float)
  for _ in range(5):
    mujoco.mj_step(model, source)
  np.testing.assert_allclose(first[0], source.qpos, atol=1e-6)
  sim.restore(snap)
  sim.reset(qpos=qp, qvel=qv)
  sim.step(5)
  np.testing.assert_array_equal(sim.state.qpos.cpu().numpy(), first)
