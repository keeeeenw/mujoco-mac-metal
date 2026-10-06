# Copyright 2026 The MuJoCo Metal contributors
# Licensed under the Apache License, Version 2.0.

"""Pinned passive-material/equality composition checks for flex models.

These cases exercise source combinations rather than testing the material and
equality producers in isolation.  The runtime oracle is pinned ``mj_forward``
at the same generalized state, with nonzero elastic/damping witnesses.
"""

import mujoco
import numpy as np
import pytest

try:
  import torch
except ImportError:  # The Torch-free preflight still collects this module.
  torch = None


def _composition_model(kind, jacobian=None):
  if kind == "edge_equality":
    model = mujoco.MjModel.from_xml_string("""
      <mujoco><option gravity="0 0 0" timestep=".001"/>
        <worldbody><flexcomp name="cable" type="grid" count="4 1 1"
            spacing=".1 .1 .1" mass="1" dim="1">
          <contact contype="0" conaffinity="0"/>
          <edge stiffness="100" damping=".3" equality="true"
                solref=".02 1" solimp=".8 .9 .01 .5 2"/>
        </flexcomp></worldbody>
      </mujoco>
    """)
  elif kind == "vertex_equality_shell":
    model = mujoco.MjModel.from_xml_string("""
      <mujoco><option gravity="0 0 0" timestep=".001"/>
        <worldbody><flexcomp name="shell" type="grid" count="2 2 1"
            spacing=".1 .1 .1" mass="1" dim="2">
          <contact contype="0" conaffinity="0"/>
          <elasticity young="900" poisson=".23" damping=".17"
                      thickness=".012" elastic2d="bend"/>
        </flexcomp></worldbody>
        <equality><flexvert flex="shell" solref=".03 1.1"
                            solimp=".75 .92 .02 .4 2"/></equality>
      </mujoco>
    """)
  elif kind in ("strain_q1", "strain_q2"):
    dof, count = ("trilinear", "2 2 2") if kind == "strain_q1" else ("quadratic", "3 3 3")
    model = mujoco.MjModel.from_xml_string(f"""
      <mujoco><option gravity="0 0 0" timestep=".001"/>
        <worldbody><flexcomp name="volume" type="grid" count="{count}"
            spacing=".1 .1 .1" mass="1" dim="3" dof="{dof}">
          <contact contype="0" conaffinity="0" selfcollide="none"/>
          <elasticity young="700" poisson=".21" damping=".13"/>
        </flexcomp></worldbody>
        <equality><flexstrain flex="volume" cell="0 0 0"
                              solref=".025 1.2"
                              solimp=".8 .94 .015 .5 2"/></equality>
      </mujoco>
    """)
  else:
    raise AssertionError(kind)
  if jacobian is not None:
    model.opt.jacobian = {
        "dense": mujoco.mjtJacobian.mjJAC_DENSE,
        "sparse": mujoco.mjtJacobian.mjJAC_SPARSE,
    }[jacobian]
  return model


def _composed_state(model, kind):
  qpos = np.asarray(model.qpos0, dtype=np.float64).copy()
  if kind == "edge_equality":
    qpos[3] += .013
    qpos[6] -= .008
  elif kind == "vertex_equality_shell":
    qpos[2] += .01
    qpos[8] -= .006
  elif kind == "strain_q1":
    qpos[7] += .012
    qpos[10] -= .007
  else:
    qpos[13] += .009
    qpos[16] -= .005
  qvel = np.linspace(-.07, .09, int(model.nv), dtype=np.float64)
  return qpos, qvel


def _evaluate_composed(kind):
  if torch is None:
    pytest.skip("flex material/equality composition requires Torch")
  from mujoco_metal.flex import MetalFlex

  model = _composition_model(kind)
  qpos, qvel = _composed_state(model, kind)
  data = mujoco.MjData(model)
  data.qpos[:] = qpos
  data.qvel[:] = qvel
  mujoco.mj_forward(model, data)

  # Keep the test independent of the private program helper: provide the
  # source smooth pose and velocity arrays explicitly.
  poses = {
      "body_pos": torch.as_tensor(data.xpos[None], dtype=torch.float32),
      "body_quat": torch.as_tensor(data.xquat[None], dtype=torch.float32),
      "joint_anchor": torch.as_tensor(data.xanchor[None], dtype=torch.float32),
      "joint_axis": torch.as_tensor(data.xaxis[None], dtype=torch.float32),
      "root_com": torch.zeros((1, int(model.nbody), 3), dtype=torch.float32),
  }
  flex = MetalFlex(model, device="cpu")
  force, damping, stiffness = flex.run_device(
      torch.as_tensor(qpos[None], dtype=torch.float32),
      torch.as_tensor(qvel[None], dtype=torch.float32), poses,
      torch.as_tensor(data.cvel[None], dtype=torch.float32))
  np.testing.assert_allclose(force.detach().numpy()[0], data.qfrc_passive,
                             rtol=7e-4, atol=5e-5)
  if kind.startswith("strain_"):
    # Explicit FLEXSTRAIN compiles edgeequality==3. Pinned
    # mj_flexPassiveInterp therefore skips this interpolation material block.
    np.testing.assert_array_equal(data.qfrc_passive, np.zeros(model.nv))
    assert np.count_nonzero(damping.detach().numpy()) == 0
    assert np.count_nonzero(stiffness.detach().numpy()) == 0
  else:
    assert np.linalg.norm(data.qfrc_passive) > 1e-3
    assert np.linalg.norm(damping.detach().numpy()) > 1e-5
    assert np.linalg.norm(stiffness.detach().numpy()) > 1e-3

  # This verifies the same model actually compiles the declared equality
  # family and generates rows at the disturbed state.
  eq_types = np.asarray(model.eq_type)
  expected = {
      "edge_equality": int(mujoco.mjtEq.mjEQ_FLEX),
      "vertex_equality_shell": int(mujoco.mjtEq.mjEQ_FLEXVERT),
      "strain_q1": int(mujoco.mjtEq.mjEQ_FLEXSTRAIN),
      "strain_q2": int(mujoco.mjtEq.mjEQ_FLEXSTRAIN),
  }[kind]
  eqids = np.flatnonzero(eq_types == expected)
  assert eqids.size
  equality_rows = np.flatnonzero(
      np.asarray(data.efc_type) == int(mujoco.mjtConstraint.mjCNSTR_EQUALITY))
  assert np.intersect1d(equality_rows, np.flatnonzero(
      np.isin(data.efc_id, eqids))).size


@pytest.mark.parametrize("kind", [
    "edge_equality", "vertex_equality_shell", "strain_q1", "strain_q2",
])
def test_flex_material_and_compiled_equality_share_pinned_passive_state(kind):
  _evaluate_composed(kind)


@pytest.mark.parametrize("kind", [
    "edge_equality", "vertex_equality_shell", "strain_q1", "strain_q2",
])
@pytest.mark.parametrize("jacobian", ["dense", "sparse"])
def test_composition_fixtures_admit_with_model_derived_scalable_limits(
    kind, jacobian):
  """Preflight every public equality fixture at its compiled dimensions."""
  from mujoco_metal.simulation import (
      _model_component_capacity_limits, validate_stepping_profile)

  model = _composition_model(kind, jacobian=jacobian)
  if kind == "strain_q2":
    assert int(model.nv) > 32
  else:
    assert int(model.nv) <= 32
  limits = _model_component_capacity_limits(model, batch_size=2)
  profile = validate_stepping_profile(
      model, profile="integrated_scalable_v1", limits=limits)
  assert profile.name == "integrated_scalable_v1"
  assert limits.max_nv >= int(model.nv)
  assert limits.max_rows >= int(model.nv)


@pytest.mark.parametrize("disableflags", [
    0,
    int(mujoco.mjtDisableBit.mjDSBL_SPRING),
    int(mujoco.mjtDisableBit.mjDSBL_DAMPER),
    int(mujoco.mjtDisableBit.mjDSBL_SPRING)
    | int(mujoco.mjtDisableBit.mjDSBL_DAMPER),
])
def test_direct_shell_stretch_preserves_pinned_passive_disable_semantics(
    disableflags):
  """Direct `mj_flexPassiveStretch` applies its source squared-length law.

  In MuJoCo 3.10 this routine receives enable booleans but does not branch on
  them; only `mj_passive`'s both-disabled early return suppresses the result.
  This deliberately differs from interpolated stretch and shell bending.
  """
  if torch is None:
    pytest.skip("flex material flags check requires Torch")
  from mujoco_metal.flex import MetalFlex

  model = mujoco.MjModel.from_xml_string("""
    <mujoco><option gravity="0 0 0" timestep=".001"/>
      <worldbody><flexcomp name="shell" type="grid" count="2 2 1"
          spacing=".1 .1 .1" mass="1" dim="2">
        <contact contype="0" conaffinity="0"/>
        <edge stiffness="0" damping="0"/>
        <elasticity young="1000" poisson=".2" damping=".4"
                    thickness=".01" elastic2d="stretch"/>
      </flexcomp></worldbody>
    </mujoco>
  """)
  model.opt.disableflags = int(disableflags)
  qpos = np.asarray(model.qpos0, dtype=np.float64).copy()
  qpos[2] += .01
  qvel = np.linspace(-.1, .1, int(model.nv), dtype=np.float64)
  data = mujoco.MjData(model)
  data.qpos[:] = qpos
  data.qvel[:] = qvel
  mujoco.mj_forward(model, data)
  poses = {
      "body_pos": torch.as_tensor(data.xpos[None], dtype=torch.float32),
      "body_quat": torch.as_tensor(data.xquat[None], dtype=torch.float32),
      "joint_anchor": torch.as_tensor(data.xanchor[None], dtype=torch.float32),
      "joint_axis": torch.as_tensor(data.xaxis[None], dtype=torch.float32),
      "root_com": torch.zeros((1, int(model.nbody), 3), dtype=torch.float32),
  }
  flex = MetalFlex(model, device="cpu")
  force = flex.run_device(
      torch.as_tensor(qpos[None], dtype=torch.float32),
      torch.as_tensor(qvel[None], dtype=torch.float32), poses,
      torch.as_tensor(data.cvel[None], dtype=torch.float32))[0]
  np.testing.assert_allclose(force.detach().numpy()[0], data.qfrc_passive,
                             rtol=7e-4, atol=5e-5)
  if disableflags == (
      int(mujoco.mjtDisableBit.mjDSBL_SPRING)
      | int(mujoco.mjtDisableBit.mjDSBL_DAMPER)):
    np.testing.assert_array_equal(data.qfrc_passive, np.zeros(model.nv))
  else:
    assert np.linalg.norm(data.qfrc_passive) > 1e-3


def _dense_efc_jacobian(model, data):
  values = np.asarray(data.efc_J, dtype=np.float64)
  if not mujoco.mj_isSparse(model):
    return values.reshape(int(data.nefc), int(model.nv)).copy()
  dense = np.zeros((int(data.nefc), int(model.nv)), dtype=np.float64)
  for row in range(int(data.nefc)):
    start, count = int(data.efc_J_rowadr[row]), int(data.efc_J_rownnz[row])
    cols = np.asarray(data.efc_J_colind[start:start + count], dtype=np.int64)
    dense[row, cols] = values[start:start + count]
  return dense


def _evaluate_public_composition(kind, jacobian):
  from mujoco_metal.simulation import (
      MetalSimulation, _model_component_capacity_limits,
      validate_stepping_profile)

  model = _composition_model(kind, jacobian=jacobian)
  qpos, qvel = _composed_state(model, kind)
  qpos_batch = np.stack((qpos, np.asarray(model.qpos0, dtype=np.float64)))
  qvel_batch = np.stack((qvel, np.zeros_like(qvel)))
  refs = []
  for env in range(2):
    data = mujoco.MjData(model)
    data.qpos[:] = qpos_batch[env]
    data.qvel[:] = qvel_batch[env]
    mujoco.mj_forward(model, data)
    refs.append(data)

  limits = _model_component_capacity_limits(model, batch_size=2)
  # Admission is checked with a host budget derived from this exact compiled
  # model.  In particular, Q2's 81 DOFs and reserved contact/equality rows
  # must not inherit the fixed-v1 profile's small defaults.
  profile = validate_stepping_profile(
      model, profile="integrated_scalable_v1", limits=limits)
  assert profile.name == "integrated_scalable_v1"
  sim = MetalSimulation(
      model, batch_size=2, qpos=qpos_batch.astype(np.float32),
      qvel=qvel_batch.astype(np.float32), profile=profile.name, limits=limits)
  system = sim.assembled_system(recompute=True)
  cc = sim._coupled_constraints
  native_j = cc.materialize_jacobian().detach().cpu().numpy().copy()
  native_R = system["R"].detach().cpu().numpy().copy()
  native_ar = system["ar"].detach().cpu().numpy().copy()
  flex_force = sim._flex._qfrc_passive.detach().cpu().numpy().copy()
  eq_type = {
      "edge_equality": int(mujoco.mjtEq.mjEQ_FLEX),
      "vertex_equality_shell": int(mujoco.mjtEq.mjEQ_FLEXVERT),
      "strain_q1": int(mujoco.mjtEq.mjEQ_FLEXSTRAIN),
      "strain_q2": int(mujoco.mjtEq.mjEQ_FLEXSTRAIN),
  }[kind]
  eqids = np.flatnonzero(np.asarray(model.eq_type) == eq_type)
  assert eqids.size
  for env, data in enumerate(refs):
    np.testing.assert_allclose(flex_force[env], data.qfrc_passive,
                               rtol=8e-4, atol=6e-5)
    cpu_j = _dense_efc_jacobian(model, data)
    for eqid in eqids:
      pinned_rows = np.flatnonzero(
          (np.asarray(data.efc_type)
           == int(mujoco.mjtConstraint.mjCNSTR_EQUALITY))
          & (np.asarray(data.efc_id) == int(eqid)))
      row = int(cc.descriptor.eq_rowadr[int(eqid)])
      span = int(cc.descriptor.eq_rownum[int(eqid)])
      assert span == pinned_rows.size and span > 0
      np.testing.assert_allclose(native_j[env, row:row + span], cpu_j[pinned_rows],
                                 rtol=6e-5, atol=6e-6)
      np.testing.assert_allclose(native_R[env, row:row + span],
                                 data.efc_R[pinned_rows], rtol=5e-4, atol=5e-6)
      np.testing.assert_allclose(native_ar[env, row:row + span],
                                 data.efc_aref[pinned_rows], rtol=8e-4, atol=6e-5)

  initial = sim.snapshot()
  status = sim.step().detach().cpu().numpy()
  np.testing.assert_array_equal(status, np.zeros(2, dtype=status.dtype))
  endpoint = sim.state.snapshot()
  cpu_endpoint = []
  for data in refs:
    mujoco.mj_step(model, data)
    cpu_endpoint.append((np.asarray(data.qpos).copy(), np.asarray(data.qvel).copy()))
  for env, (expected_qpos, expected_qvel) in enumerate(cpu_endpoint):
    np.testing.assert_allclose(endpoint.qpos[env], expected_qpos,
                               rtol=2e-3, atol=3e-5)
    np.testing.assert_allclose(endpoint.qvel[env], expected_qvel,
                               rtol=3e-3, atol=5e-5)
  checkpoint = sim.snapshot()
  sim.step()
  replay = sim.state.snapshot()
  sim.restore(checkpoint)
  sim.step()
  replay2 = sim.state.snapshot()
  np.testing.assert_array_equal(replay.qpos, replay2.qpos)
  np.testing.assert_array_equal(replay.qvel, replay2.qvel)
  sim.restore(initial)
  sim.reset(qpos=qpos_batch.astype(np.float32), qvel=qvel_batch.astype(np.float32))
  reset = sim.state.snapshot()
  np.testing.assert_array_equal(reset.qpos, qpos_batch.astype(np.float32))
  np.testing.assert_array_equal(reset.qvel, qvel_batch.astype(np.float32))


@pytest.mark.gpu
@pytest.mark.parametrize("kind", [
    "edge_equality", "vertex_equality_shell", "strain_q1", "strain_q2",
])
@pytest.mark.parametrize("jacobian", ["dense", "sparse"])
@pytest.mark.skipif(__import__("os").getenv("MUJOCO_METAL_RUN_GPU") != "1",
                    reason="public flex material/equality lifecycle is opt-in")
def test_native_public_flex_material_equality_dense_sparse_lifecycle(kind, jacobian):
  pytest.importorskip("torch")
  if not torch.backends.mps.is_available():
    pytest.skip("native Metal flex composition requires MPS")
  _evaluate_public_composition(kind, jacobian)
