# Copyright 2026 The MuJoCo Metal contributors
# Licensed under the Apache License, Version 2.0.
"""Complete native implicit flex steps against the pinned CPU engine."""
import os

import mujoco
import numpy as np
import pytest
from types import SimpleNamespace

from mujoco_metal.flex_implicit import requires_flex_implicit_correction
from mujoco_metal.implicit import lower_implicit, lower_implicitfast
from mujoco_metal.simulation import MetalSimulation
from mujoco_metal.sparse_flex_implicit import sparse_flex_correction_workspace_sizes
from mujoco_metal.stepping import validate_stepping_profile
from mujoco_metal.flex import _lower_flexedge_jacobian_csr


def test_compiled_flexedge_jacobian_csr_lowering_is_exact_and_validated():
  model = SimpleNamespace(
      nflexedge=3, nv=5,
      flexedge_J_rowadr=np.array([0, 2, 2], dtype=np.int32),
      flexedge_J_rownnz=np.array([2, 0, 1], dtype=np.int32),
      flexedge_J_colind=np.array([0, 4, 3], dtype=np.int32))
  rowadr, rownnz, colind = _lower_flexedge_jacobian_csr(model)
  np.testing.assert_array_equal(rowadr, [0, 2, 2])
  np.testing.assert_array_equal(rownnz, [2, 0, 1])
  np.testing.assert_array_equal(colind, [0, 4, 3])
  assert not rowadr.flags.writeable
  assert not rownnz.flags.writeable
  assert not colind.flags.writeable

  malformed = SimpleNamespace(
      nflexedge=1, nv=3,
      flexedge_J_rowadr=np.array([0]),
      flexedge_J_rownnz=np.array([2]),
      flexedge_J_colind=np.array([2, 1]))
  with pytest.raises(ValueError, match="sorted unique"):
    _lower_flexedge_jacobian_csr(malformed)


def test_pinned_empty_flexedge_rows_emit_no_damper_force():
  model = mujoco.MjModel.from_xml_string('''<mujoco>
    <option gravity="0 0 0" integrator="Euler" timestep=".002">
      <flag contact="disable"/>
    </option><worldbody>
      <flexcomp name="volume" type="grid" count="2 2 2"
                spacing=".1 .1 .1" mass="1" dim="3" dof="trilinear">
        <contact contype="0" conaffinity="0" selfcollide="none"/>
        <edge stiffness="0" damping=".07"/>
      </flexcomp>
    </worldbody></mujoco>''')
  assert model.nflexedge > 0
  assert not np.any(model.flexedge_J_rownnz)
  assert len(model.flexedge_J_colind) == 0
  data = mujoco.MjData(model)
  data.qvel[:] = np.linspace(-.2, .3, model.nv)
  mujoco.mj_forward(model, data)
  # No compiled generalized edge-J entries means the pinned sparse force
  # scatter emits no edge contribution, regardless of geometric edge motion.
  np.testing.assert_array_equal(data.qfrc_spring, np.zeros(model.nv))
  np.testing.assert_array_equal(data.qfrc_damper, np.zeros(model.nv))
  np.testing.assert_array_equal(data.flexedge_velocity, np.zeros(model.nflexedge))


@pytest.mark.parametrize("disabled", [0, 1, 2, 3])
def test_nonempty_compiled_edge_passive_force_matches_pinned(disabled):
  """Check compiled CSR scatter with spring/damper toggles and moving edges."""
  model = mujoco.MjModel.from_xml_string('''<mujoco>
    <option gravity="0 0 0" integrator="Euler" timestep=".002">
      <flag contact="disable"/>
    </option><worldbody>
      <flexcomp name="cable" type="grid" count="3 1 1"
                spacing=".1 .1 .1" mass="1" dim="1">
        <contact contype="0" conaffinity="0" selfcollide="none"/>
        <edge stiffness="100" damping=".4"/>
      </flexcomp>
    </worldbody></mujoco>''')
  spring = int(mujoco.mjtDisableBit.mjDSBL_SPRING)
  damper = int(mujoco.mjtDisableBit.mjDSBL_DAMPER)
  if disabled & 1:
    model.opt.disableflags |= spring
  if disabled & 2:
    model.opt.disableflags |= damper
  assert np.all(model.flexedge_J_rownnz > 0)
  qpos = np.asarray(model.qpos0, dtype=np.float64).copy()
  qpos[3] += .02
  qpos[7] -= .01
  qvel = np.linspace(-.08, .11, model.nv, dtype=np.float64)
  data = mujoco.MjData(model)
  data.qpos[:] = qpos
  data.qvel[:] = qvel
  mujoco.mj_forward(model, data)
  assert np.max(np.abs(data.flexedge_velocity)) > .02

  # Reconstruct engine_passive.c's compiled sparse scatter independently from
  # the public pinned CPU outputs. This part also runs in the Torch-free CPU
  # test environment.
  expected_spring = np.zeros(model.nv, dtype=np.float64)
  expected_damper = np.zeros(model.nv, dtype=np.float64)
  for flex_id in range(model.nflex):
    start = int(model.flex_edgeadr[flex_id])
    stop = start + int(model.flex_edgenum[flex_id])
    if bool(model.flex_rigid[flex_id]):
      continue
    stiffness = (0. if model.opt.disableflags & spring
                 else float(model.flex_edgestiffness[flex_id]))
    damping = (0. if model.opt.disableflags & damper
               else float(model.flex_edgedamping[flex_id]))
    for edge in range(start, stop):
      if bool(model.flexedge_rigid[edge]):
        continue
      adr = int(model.flexedge_J_rowadr[edge])
      count = int(model.flexedge_J_rownnz[edge])
      cols = model.flexedge_J_colind[adr:adr + count]
      jac = data.flexedge_J[adr:adr + count]
      expected_spring[cols] += jac * stiffness * (
          model.flexedge_length0[edge] - data.flexedge_length[edge])
      expected_damper[cols] += jac * (-damping * data.flexedge_velocity[edge])
  np.testing.assert_allclose(data.qfrc_spring, expected_spring,
                             rtol=1e-12, atol=1e-12)
  np.testing.assert_allclose(data.qfrc_damper, expected_damper,
                             rtol=1e-12, atol=1e-12)

  try:
    import torch
  except ImportError:
    return
  from mujoco_metal.flex import MetalFlex

  flex = MetalFlex(model, device="cpu")
  root_com = np.zeros((model.nbody, 3), dtype=np.float64)
  for body in range(1, model.nbody):
    root = body
    while int(model.body_parentid[root]) > 0:
      root = int(model.body_parentid[root])
    root_com[body] = data.subtree_com[root]
  poses = {
      "body_pos": torch.tensor(data.xpos[None], dtype=torch.float32),
      "body_quat": torch.tensor(data.xquat[None], dtype=torch.float32),
      "joint_anchor": torch.tensor(data.xanchor[None], dtype=torch.float32),
      "joint_axis": torch.tensor(data.xaxis[None], dtype=torch.float32),
      "root_com": torch.tensor(root_com[None], dtype=torch.float32),
  }
  force, _, _ = flex.run_device(
      torch.tensor(qpos[None], dtype=torch.float32),
      torch.tensor(qvel[None], dtype=torch.float32), poses,
      torch.tensor(data.cvel[None], dtype=torch.float32))
  np.testing.assert_allclose(
      force.detach().numpy()[0], data.qfrc_passive,
      rtol=1e-4, atol=2e-6)
  expected_jacobian = np.zeros((model.nflexedge, model.nv), dtype=np.float64)
  for edge in range(model.nflexedge):
    adr = int(model.flexedge_J_rowadr[edge])
    count = int(model.flexedge_J_rownnz[edge])
    cols = model.flexedge_J_colind[adr:adr + count]
    expected_jacobian[edge, cols] = data.flexedge_J[adr:adr + count]
  np.testing.assert_allclose(
      flex._flexedge_J.detach().numpy()[0], expected_jacobian,
      rtol=1e-6, atol=1e-7)


@pytest.mark.gpu
@pytest.mark.skipif(os.getenv("MUJOCO_METAL_RUN_GPU") != "1",
                    reason="native compiled flex-edge CSR qualification")
def test_native_empty_flexedge_csr_matches_pinned_passive_force():
  model = mujoco.MjModel.from_xml_string('''<mujoco>
    <option gravity="0 0 0" integrator="Euler" timestep=".002">
      <flag contact="disable"/>
    </option><worldbody>
      <flexcomp name="volume" type="grid" count="2 2 2"
                spacing=".1 .1 .1" mass="1" dim="3" dof="trilinear">
        <contact contype="0" conaffinity="0" selfcollide="none"/>
        <edge stiffness="0" damping=".07"/>
      </flexcomp>
    </worldbody></mujoco>''')
  assert not np.any(model.flexedge_J_rownnz)
  qvel = np.linspace(-.2, .3, model.nv).astype(np.float32)
  sim = MetalSimulation(model, 1, model.qpos0[None].astype(np.float32),
                        qvel[None], profile="integrated_euler_v1")
  ref = mujoco.MjData(model)
  ref.qvel[:] = qvel
  for step in range(3):
    sim.step()
    mujoco.mj_step(model, ref)
    state = sim.state.snapshot()
    np.testing.assert_array_equal(state.status, 0, err_msg=f"step {step}")
    edge_jac = sim._flex._flexedge_J.detach().cpu().numpy()
    edge_velocity = sim._flex._flexedge_velocity.detach().cpu().numpy()
    np.testing.assert_array_equal(edge_jac, np.zeros_like(edge_jac))
    np.testing.assert_array_equal(edge_velocity, np.zeros_like(edge_velocity))
    np.testing.assert_allclose(
        sim._flex._qfrc_passive.detach().cpu().numpy()[0],
        ref.qfrc_spring + ref.qfrc_damper, atol=2e-6, rtol=2e-5,
        err_msg=f"step {step} compiled empty edge rows")
    np.testing.assert_allclose(state.qpos[0], ref.qpos, atol=5e-6, rtol=5e-5)
    np.testing.assert_allclose(state.qvel[0], ref.qvel, atol=2e-5, rtol=2e-4)


@pytest.mark.gpu
@pytest.mark.skipif(os.getenv("MUJOCO_METAL_RUN_GPU") != "1",
                    reason="native nonempty compiled flex-edge qualification")
@pytest.mark.parametrize("disabled", [0, 1, 2, 3])
def test_native_nonempty_compiled_edge_flags_match_pinned(disabled):
  """Exercise nonzero CSR rows, edge velocity, and passive disable flags."""
  model = mujoco.MjModel.from_xml_string('''<mujoco>
    <option gravity="0 0 0" integrator="Euler" timestep=".002"/>
    <worldbody>
      <flexcomp name="cable" type="grid" count="3 1 1"
                spacing=".1 .1 .1" mass="1" dim="1">
        <contact contype="0" conaffinity="0" selfcollide="none"/>
        <edge stiffness="100" damping=".4"/>
      </flexcomp>
    </worldbody></mujoco>''')
  spring = int(mujoco.mjtDisableBit.mjDSBL_SPRING)
  damper = int(mujoco.mjtDisableBit.mjDSBL_DAMPER)
  if disabled & 1:
    model.opt.disableflags |= spring
  if disabled & 2:
    model.opt.disableflags |= damper
  assert np.all(model.flexedge_J_rownnz > 0)
  qpos = np.asarray(model.qpos0, dtype=np.float32).copy()
  qpos[3] += .02
  qpos[7] -= .01
  qvel = np.linspace(-.08, .11, model.nv, dtype=np.float32)
  sim = MetalSimulation(model, 1, qpos[None], qvel[None],
                        profile="integrated_euler_v1")
  ref = mujoco.MjData(model)
  ref.qpos[:] = qpos
  ref.qvel[:] = qvel
  for step in range(3):
    sim.step()
    mujoco.mj_step(model, ref)
    state = sim.state.snapshot()
    np.testing.assert_array_equal(state.status, 0, err_msg=f"step {step}")
    assert np.max(np.abs(sim._flex._flexedge_velocity.detach().cpu().numpy())) > 1e-3
    np.testing.assert_allclose(
        sim._flex._qfrc_passive.detach().cpu().numpy()[0],
        ref.qfrc_spring + ref.qfrc_damper, atol=2e-6, rtol=2e-5,
        err_msg=f"step {step} disabled flags {disabled}")
    np.testing.assert_allclose(state.qpos[0], ref.qpos, atol=5e-6, rtol=5e-5)
    np.testing.assert_allclose(state.qvel[0], ref.qvel, atol=2e-5, rtol=2e-4)


def _model(integrator, disabled=False, edge_damping=0):
  model = mujoco.MjModel.from_xml_string(f'''<mujoco>
    <option gravity="0 0 0" integrator="{integrator}" timestep=".002">
      <flag contact="disable"/>
    </option><worldbody>
      <flexcomp name="volume" type="grid" count="2 2 2" spacing=".1 .1 .1"
                mass="1" dim="3" dof="trilinear">
        <contact contype="0" conaffinity="0" selfcollide="none"/>
        <edge stiffness="0" damping="{edge_damping}"/>
        <elasticity young="1000" poisson=".2" damping=".1"/>
      </flexcomp>
    </worldbody></mujoco>''')
  if disabled:
    model.opt.disableflags |= int(mujoco.mjtDisableBit.mjDSBL_SPRING)
    model.opt.disableflags |= int(mujoco.mjtDisableBit.mjDSBL_DAMPER)
  return model


def test_sparse_flex_correction_workspace_is_linear_in_nv():
  sizes = sparse_flex_correction_workspace_sizes(3, 17, edge_count=29)
  assert sizes["sparse_flex.rhs"] == 3 * 17
  assert sizes["sparse_flex.preconditioned"] == 3 * 17
  assert sizes["sparse_flex.status"] == 3
  assert sizes["sparse_flex.active"] == 3
  assert sizes["sparse_flex.full_derivative_values"] == 3 * 29
  assert sizes["sparse_flex.preconditioner_derivative_values"] == 3 * 29
  assert sum(sizes.values()) == 7 * 3 * 17 + 10 * 3 + 2 * 3 * 29
  for args in ((0, 2), (1, -1), (True, 1), (1, 2, -1)):
    with pytest.raises(ValueError):
      sparse_flex_correction_workspace_sizes(*args)


def test_sparse_flex_constructor_checks_capacity_before_torch_import():
  from unittest import mock
  from mujoco_metal.sparse_flex_implicit import SparseFlexImplicitCorrection

  # An invalid compiled dimension must be rejected before even importing the
  # device runtime (the test blocks that import as a hard ordering witness).
  with mock.patch.dict("sys.modules", {"torch": None}):
    with pytest.raises(ValueError, match="nv exceeds the Metal int32"):
      SparseFlexImplicitCorrection(1, 1 << 31, edge_count=1)


def test_sparse_flex_pcg_direction_mask_clears_inactive_nonfinite_rows():
  """Exercise the exact out-buffer update used by every PCG iteration."""
  torch = pytest.importorskip("torch")
  from mujoco_metal.sparse_flex_implicit import SparseFlexImplicitCorrection

  direction = torch.empty((2, 3), dtype=torch.float32)
  candidate = torch.tensor([[1., -2., 3.], [float("nan"), float("inf"), -4.]])
  active = torch.tensor([True, False])
  SparseFlexImplicitCorrection._mask_direction_(direction, candidate, active)
  torch.testing.assert_close(direction[0], candidate[0])
  torch.testing.assert_close(direction[1], torch.zeros(3))
  assert torch.isfinite(direction).all()


@pytest.mark.gpu
@pytest.mark.skipif(os.getenv("MUJOCO_METAL_RUN_GPU") != "1",
                    reason="native sparse flex buffer ownership qualification")
def test_native_sparse_flex_snapshots_full_and_symmetric_derivatives():
  import torch
  from mujoco_metal.sparse_flex_implicit import SparseFlexImplicitCorrection

  correction = SparseFlexImplicitCorrection(2, 2, edge_count=4)
  source = torch.tensor([[1., 2., 3., 4.], [-2., 5., 7., 11.]],
                        dtype=torch.float32, device="mps")
  lower = torch.tensor([[1., 2., 3., 4.], [-2., 5., 7., 11.]],
                       dtype=torch.float32, device="mps")
  correction.capture_full_derivative_values(source)
  correction._preconditioner_derivative_values.copy_(lower)
  source.zero_()
  lower.fill_(99.)
  torch.testing.assert_close(
      correction.full_derivative_values,
      torch.tensor([[1., 2., 3., 4.], [-2., 5., 7., 11.]],
                   dtype=torch.float32, device="mps"))
  torch.testing.assert_close(
      correction._preconditioner_derivative_values,
      torch.tensor([[1., 2., 3., 4.], [-2., 5., 7., 11.]],
                   dtype=torch.float32, device="mps"))


@pytest.mark.gpu
@pytest.mark.skipif(os.getenv("MUJOCO_METAL_RUN_GPU") != "1",
                    reason="native sparse flex operator ownership qualification")
def test_native_sparse_flex_full_and_mirrored_operators_do_not_alias():
  import torch
  from mujoco_metal.sparse_flex_implicit import SparseFlexImplicitCorrection

  class ZeroFlex:
    @staticmethod
    def apply_material_operator_context_device(_context, vector):
      return {name: torch.zeros_like(vector) for name in (
          "interp_stiffness", "interp_damped_stiffness", "bend_stiffness",
          "bend_damped_stiffness")}

  class ReusingEffectiveSolver:
    def __init__(self):
      self.workspace = torch.empty((1, 4), dtype=torch.float32, device="mps")
      self.seen = []

    def apply_effective_operator_device(self, mass, vector, values, dt,
                                        **_kwargs):
      self.workspace.copy_(values)
      self.seen.append(self.workspace.clone())
      out = mass * vector
      out = out.clone()
      for slot, (row, col) in enumerate(((0, 0), (0, 1), (1, 0), (1, 1))):
        out[:, row] -= dt * self.workspace[:, slot] * vector[:, col]
      self.workspace.fill_(77.)
      return out, torch.zeros((1,), dtype=torch.int32, device="mps")

    def solve_device(self, _mass, rhs, values, _dt, **_kwargs):
      self.workspace.copy_(values)
      self.seen.append(self.workspace.clone())
      self.workspace.fill_(-91.)
      return (rhs.clone(), torch.zeros((1,), dtype=torch.int32, device="mps"),
              torch.zeros((1,), dtype=torch.float32, device="mps"),
              torch.ones((1,), dtype=torch.int32, device="mps"))

  correction = SparseFlexImplicitCorrection(1, 2, edge_count=4)
  source = torch.tensor([[.1, .4, -.2, .3]], dtype=torch.float32,
                        device="mps")
  mirrored = torch.tensor([[.1, -.2, -.2, .3]], dtype=torch.float32,
                          device="mps")
  correction.capture_full_derivative_values(source)
  correction._preconditioner_derivative_values.copy_(mirrored)
  source.zero_()
  mirrored.fill_(99.)
  full_before = correction.full_derivative_values.clone()
  lower_before = correction._preconditioner_derivative_values.clone()
  effective = ReusingEffectiveSolver()
  vector = torch.tensor([[.5, -1.]], dtype=torch.float32, device="mps")
  mass = torch.ones_like(vector)
  # engine_forward.c flexInterp_cgsolve uses full d->qDeriv for A*x,
  # independent of integrator; implicitfast's preconditioner uses qH, which
  # is gathered from the lower symmetric projection of qDeriv.
  full, _ = correction._apply(
      vector, flex=ZeroFlex(), context=None, effective_solver=effective,
      mass_blocks=mass, edge_values=correction.full_derivative_values,
      timestep=.2, dof_ids=None, counts=None, armature_blocks=None)
  expected_full = torch.tensor([[.57, -.92]], dtype=torch.float32,
                               device="mps")
  torch.testing.assert_close(full, expected_full)
  lower_product, _ = correction._apply(
      vector, flex=ZeroFlex(), context=None, effective_solver=effective,
      mass_blocks=mass, edge_values=correction._preconditioner_derivative_values,
      timestep=.2, dof_ids=None, counts=None, armature_blocks=None)
  expected_qh = torch.tensor([[.45, -.92]], dtype=torch.float32,
                             device="mps")
  torch.testing.assert_close(lower_product, expected_qh)
  correction._precondition(
      vector, effective_solver=effective, mass_blocks=mass,
      edge_values=correction._preconditioner_derivative_values,
      timestep=.2, dof_ids=None, counts=None, armature_blocks=None,
      active=torch.ones((1,), dtype=torch.bool, device="mps"))
  full_again, _ = correction._apply(
      vector, flex=ZeroFlex(), context=None, effective_solver=effective,
      mass_blocks=mass, edge_values=correction.full_derivative_values,
      timestep=.2, dof_ids=None, counts=None, armature_blocks=None)
  torch.testing.assert_close(full_again, expected_full)
  torch.testing.assert_close(effective.seen[0], full_before)
  torch.testing.assert_close(effective.seen[1], lower_before)
  torch.testing.assert_close(correction.full_derivative_values, full_before)
  torch.testing.assert_close(correction._preconditioner_derivative_values,
                             lower_before)


@pytest.mark.parametrize("integrator,lower", [("implicit", lower_implicit),
                                            ("implicitfast", lower_implicitfast)])
def test_implicit_flex_profile_has_native_material_correction(integrator, lower):
  model = _model(integrator, disabled=True)
  assert lower(model).auto_derivative
  assert requires_flex_implicit_correction(model)
  profile = validate_stepping_profile(model, profile=f"integrated_{integrator}_v1")
  assert profile.execution_plan.is_stage_enabled("implicit_velocity")
  # The compiled stiffness correction is independent of passive disable bits.
  data = mujoco.MjData(model)
  data.qvel[:] = np.linspace(-.2, .3, model.nv)
  initial = data.qvel.copy()
  mujoco.mj_forward(model, data)
  np.testing.assert_array_equal(data.qfrc_passive, 0)
  mujoco.mj_step(model, data)
  assert np.max(np.abs(data.qvel-initial)) > 1e-5


@pytest.mark.gpu
@pytest.mark.skipif(os.getenv("MUJOCO_METAL_RUN_GPU") != "1", reason="native opt-in")
@pytest.mark.parametrize("integrator", ["implicit", "implicitfast"])
@pytest.mark.parametrize("disabled,edge_damping", [(False, 0), (True, .07),
                                                 (False, .07)])
def test_native_implicit_flex_trajectory_and_restore(integrator, disabled, edge_damping):
  model = _model(integrator, disabled, edge_damping)
  qpos = np.tile(model.qpos0, (2, 1)).astype(np.float32)
  qpos[0, 3] += .003
  qpos[1, -3] -= .002
  qvel = np.tile(np.linspace(-.2, .3, model.nv), (2, 1)).astype(np.float32)
  qvel[1] *= -.7
  sim = MetalSimulation(model, 2, qpos, qvel,
                        profile=f"integrated_{integrator}_v1")
  if sim._component_mass_enabled:
    assert sim._sparse_flex_implicit is not None
    assert sim._flex_implicit is None
    assert sim._smooth._workspace.get("mass") is None
    assert "mass_matrix" not in sim._smooth._workspace
  else:
    assert sim._flex_implicit is not None
  refs = [mujoco.MjData(model) for _ in range(2)]
  for row, data in enumerate(refs):
    data.qpos[:] = qpos[row]
    data.qvel[:] = qvel[row]
  for step in range(25):
    sim.step()
    for data in refs:
      mujoco.mj_step(model, data)
    native = sim.state.snapshot()
    np.testing.assert_array_equal(native.status, 0)
    for row, data in enumerate(refs):
      np.testing.assert_allclose(native.qpos[row], data.qpos, atol=5e-6, rtol=5e-5)
      np.testing.assert_allclose(native.qvel[row], data.qvel, atol=2e-5, rtol=2e-4)
      # Flex CG corrects the stack integration vector. MuJoCo retains the
      # ordinary forward-stage qacc in mjData (and copies that value to
      # qacc_warmstart) unless midpoint explicitly overwrites its eligible
      # free-joint entries after advance.
      np.testing.assert_allclose(native.qacc[row], data.qacc, atol=5e-4, rtol=3e-3)
      np.testing.assert_allclose(
          sim._state._qacc_warmstart[row].detach().cpu().numpy(),
          data.qacc_warmstart, atol=5e-4, rtol=3e-3)
  snapshot = sim.snapshot()
  sim.step(3)
  expected = sim.state.snapshot()
  sim.restore(snapshot)
  sim.step(3)
  actual = sim.state.snapshot()
  np.testing.assert_array_equal(actual.qpos, expected.qpos)
  np.testing.assert_array_equal(actual.qvel, expected.qvel)
  np.testing.assert_array_equal(actual.qacc, expected.qacc)


@pytest.mark.gpu
@pytest.mark.skipif(os.getenv("MUJOCO_METAL_RUN_GPU") != "1", reason="native opt-in")
def test_native_euler_keeps_flex_material_damping_out_of_joint_preconditioner():
  # Pinned mj_EulerSkip adds only joint/actuator DOF damping to qH. Flex
  # material and edge damping belong to the passive force, not its diagonal.
  model = _model("Euler", edge_damping=.07)
  qpos = np.asarray(model.qpos0, dtype=np.float32).copy()
  qpos[3] += .002
  qvel = np.linspace(-.2, .3, model.nv).astype(np.float32)
  sim = MetalSimulation(model, 1, qpos[None], qvel[None],
                        profile="integrated_euler_v1")
  data = mujoco.MjData(model)
  data.qpos[:], data.qvel[:] = qpos, qvel
  for step in range(10):
    sim.step()
    mujoco.mj_step(model, data)
    actual = sim.state.snapshot()
    np.testing.assert_array_equal(actual.status, 0)
    np.testing.assert_allclose(actual.qvel[0], data.qvel, atol=2e-5, rtol=2e-4)
    np.testing.assert_allclose(actual.qpos[0], data.qpos, atol=5e-6, rtol=5e-5)


def test_pinned_euler_flex_damping_uses_forward_acceleration():
  model = _model("Euler", edge_damping=.07)
  data = mujoco.MjData(model)
  data.qpos[3] += .002
  data.qvel[:] = np.linspace(-.2, .3, model.nv)
  assert not np.any(model.dof_damping)
  mujoco.mj_forward(model, data)
  assert np.linalg.norm(data.qfrc_passive) > .01
  qvel = data.qvel.copy()
  qacc = data.qacc.copy()
  mujoco.mj_step(model, data)
  np.testing.assert_allclose(data.qvel, qvel+model.opt.timestep*qacc,
                             atol=1e-14, rtol=1e-14)
