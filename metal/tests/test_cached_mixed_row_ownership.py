# Copyright 2026 The MuJoCo Metal contributors
# Licensed under the Apache License, Version 2.0.

"""Mixed canonical row-owner lifecycle: equality, tendon, and flex contact."""

import os

import mujoco
import numpy as np
import pytest

from mujoco_metal.coupled_constraints import (
    _row_velocity_scales,
    lower_coupled_constraints,
)
from mujoco_metal.solver_islands import lower_solver_row_metadata


def _mixed_model():
  return mujoco.MjModel.from_xml_string("""<mujoco>
    <option gravity="0 0 0" timestep=".0002" integrator="Euler"
            solver="PGS" iterations="80" tolerance="1e-6">
      <flag multiccd="disable"/>
    </option>
    <worldbody>
      <geom name="floor" type="plane" pos="0 0 .04" size="0 0 .1"
            contype="0" conaffinity="1" condim="1" margin=".02"/>
      <body name="sphere_body" pos=".033 .041 .045">
        <freejoint/>
        <geom name="sphere" type="sphere" size=".07"
              contype="0" conaffinity="1" condim="1"/>
      </body>
      <flexcomp name="sheet" type="grid" count="2 2 1" pos="0 0 .05"
                spacing=".1 .1 .1" mass="1" dim="2" radius=".005">
        <contact contype="1" conaffinity="0" selfcollide="none"
                 condim="1" margin=".005"/>
        <edge stiffness="0" damping="0"/>
        <elasticity young="100" poisson=".2" thickness=".01"
                    elastic2d="stretch"/>
      </flexcomp>
      <body name="link_a" pos="2 0 1"><freejoint/>
        <geom size=".05" mass="1" contype="0" conaffinity="0"/>
      </body>
      <body name="link_b" pos="2.5 0 1"><freejoint/>
        <geom size=".05" mass="1" contype="0" conaffinity="0"/>
      </body>
      <body name="tendon_a" pos="0 2 1">
        <joint name="ta" type="slide" axis="1 0 0"/>
        <geom size=".04" mass="1" contype="0" conaffinity="0"/>
      </body>
      <body name="tendon_b" pos="0 3 1">
        <joint name="tb" type="slide" axis="1 0 0"/>
        <geom size=".04" mass="1" contype="0" conaffinity="0"/>
      </body>
    </worldbody>
    <tendon><fixed name="limit_cable" limited="true" range="-.1 .1"
                   frictionloss=".2">
      <joint joint="ta" coef="1"/><joint joint="tb" coef="-.5"/>
    </fixed></tendon>
    <equality>
      <connect body1="link_a" body2="link_b" anchor="2.25 0 1"/>
    </equality>
  </mujoco>""")


def _mixed_states(model, batch=2):
  qpos = np.tile(np.asarray(model.qpos0, dtype=np.float64), (batch, 1))
  qvel = np.zeros((batch, int(model.nv)), dtype=np.float32)
  for joint, value in (("ta", .3), ("tb", -.1)):
    jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint)
    qpos[:, int(model.jnt_qposadr[jid])] = value
  for world in range(batch):
    qvel[world] = np.linspace(-.025, .03, int(model.nv), dtype=np.float32)
    if world:
      qvel[world] *= -.63
    mujoco.mj_integratePos(model, qpos[world], qvel[world].astype(np.float64),
                           .0002 * (world + 1))
  return qpos.astype(np.float32), qvel


def test_mixed_row_ownership_fixture_has_all_three_producer_families():
  model = _mixed_model()
  qpos, qvel = _mixed_states(model)
  data = mujoco.MjData(model)
  data.qpos[:] = qpos[0]
  data.qvel[:] = qvel[0]
  mujoco.mj_forward(model, data)
  assert data.nefc > 0
  ctype = np.asarray(data.efc_type[:data.nefc])
  assert np.any(ctype == int(mujoco.mjtConstraint.mjCNSTR_EQUALITY))
  assert np.any(ctype == int(mujoco.mjtConstraint.mjCNSTR_LIMIT_TENDON))
  assert np.any(np.isin(ctype, [
      int(mujoco.mjtConstraint.mjCNSTR_CONTACT_FRICTIONLESS),
      int(mujoco.mjtConstraint.mjCNSTR_CONTACT_PYRAMIDAL),
      int(mujoco.mjtConstraint.mjCNSTR_CONTACT_ELLIPTIC)]))
  assert any(int(c.geom[0]) == 1 and int(c.elem[1]) >= 0
             for c in data.contact[:data.ncon])


def test_cached_tendon_velocity_scale_uses_solimp_dmax_for_friction_and_limit():
  """Cached tendon B follows the pinned solimp dmax field, not width."""
  model = _mixed_model()
  desc = lower_coupled_constraints(model)
  metadata = lower_solver_row_metadata(model, desc)
  scales = _row_velocity_scales(model, desc, metadata)
  types = metadata[:, 0]
  friction_type = int(mujoco.mjtConstraint.mjCNSTR_FRICTION_TENDON)
  limit_type = int(mujoco.mjtConstraint.mjCNSTR_LIMIT_TENDON)
  checks = 0
  for row in np.flatnonzero(np.isin(types, [friction_type, limit_type])):
    ctype, tendon_id = map(int, metadata[row, :2])
    params = (desc.ten_solref_fri[tendon_id] if ctype == friction_type
              else desc.ten_solref_lim[tendon_id])
    solimp = (desc.ten_solimp_fri[tendon_id] if ctype == friction_type
              else desc.ten_solimp_lim[tendon_id])
    dmax = max(1e-15, float(solimp[1]))
    r0, r1 = map(float, params[:2])
    if desc.refsafe and r0 > 0:
      r0 = max(r0, 2 * float(desc.timestep))
    expected = (2.0 / max(1e-15, dmax * r0)
                if r0 > 0 and r1 > 0 and dmax * r0 > 1e-15
                else -r1 / dmax)
    width = max(1e-15, float(solimp[2]))
    width_result = (2.0 / max(1e-15, width * r0)
                    if r0 > 0 and r1 > 0 and width * r0 > 1e-15
                    else -r1 / width)
    assert scales[row] == pytest.approx(expected, rel=1e-7, abs=1e-7)
    assert abs(expected - width_result) > 1e-3
    checks += 1
  assert checks >= 2  # both tendon friction and limit rows are represented


def test_pinned_cached_tendon_reference_uses_dmax_for_friction_and_limit():
  """Pinned POS-to-VEL aref delta independently witnesses tendon B."""
  model = _mixed_model()
  qpos, _ = _mixed_states(model, batch=1)
  qvel0 = np.zeros(int(model.nv), dtype=np.float64)
  qvel1 = qvel0.copy()
  for name, value in (("ta", .23), ("tb", -.17)):
    jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
    qvel1[int(model.jnt_dofadr[jid])] = value
  initial = mujoco.MjData(model)
  initial.qpos[:] = qpos[0]
  initial.qvel[:] = qvel0
  mujoco.mj_forward(model, initial)
  cached = mujoco.MjData(model)
  cached.qpos[:] = qpos[0]
  cached.qvel[:] = qvel0
  mujoco.mj_forward(model, cached)
  cached.qvel[:] = qvel1
  mujoco.mj_forwardSkip(model, cached, mujoco.mjtStage.mjSTAGE_POS, 1)
  outputs = (initial, cached)

  tendon_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_TENDON,
                                "limit_cable")
  types0 = np.asarray(outputs[0].efc_type[:outputs[0].nefc])
  ids0 = np.asarray(outputs[0].efc_id[:outputs[0].nefc])
  types1 = np.asarray(outputs[1].efc_type[:outputs[1].nefc])
  ids1 = np.asarray(outputs[1].efc_id[:outputs[1].nefc])
  J0 = np.asarray(outputs[0].efc_J).reshape(outputs[0].nefc, model.nv)
  J1 = np.asarray(outputs[1].efc_J).reshape(outputs[1].nefc, model.nv)
  checks = 0
  for ctype, params, solimp in (
      (int(mujoco.mjtConstraint.mjCNSTR_FRICTION_TENDON),
       model.tendon_solref_fri[tendon_id], model.tendon_solimp_fri[tendon_id]),
      (int(mujoco.mjtConstraint.mjCNSTR_LIMIT_TENDON),
       model.tendon_solref_lim[tendon_id], model.tendon_solimp_lim[tendon_id]),
  ):
    row0 = np.flatnonzero((types0 == ctype) & (ids0 == tendon_id))
    row1 = np.flatnonzero((types1 == ctype) & (ids1 == tendon_id))
    assert row0.size == row1.size == 1
    row0, row1 = int(row0[0]), int(row1[0])
    velocity_delta = float(J0[row0] @ (qvel1 - qvel0))
    reference_delta = float(outputs[1].efc_aref[row1]
                             - outputs[0].efc_aref[row0])
    assert abs(velocity_delta) > 1e-5
    inferred = -reference_delta / velocity_delta
    dmax = max(1e-15, float(solimp[1]))
    r0, r1 = map(float, params[:2])
    expected = 2.0 / (dmax * r0) if r0 > 0 and r1 > 0 else -r1 / dmax
    assert inferred == pytest.approx(expected, rel=2e-5, abs=2e-4)
    checks += 1
  assert checks == 2


@pytest.mark.gpu
@pytest.mark.skipif(os.getenv("MUJOCO_METAL_RUN_GPU") != "1",
                    reason="mixed row-owner native lifecycle requires GPU opt-in")
def test_native_mixed_tendon_connect_flex_cache_reset_restore_B2():
  pytest.importorskip("torch")
  import torch
  from mujoco_metal.simulation import MetalSimulation

  model = _mixed_model()
  qpos, qvel0 = _mixed_states(model)
  qvel1 = qvel0.copy()
  qvel1[:, :6] *= -.37
  qvel1[:, -2:] += np.asarray([[.25, -.2], [-.15, .3]], dtype=np.float32)
  simulation = MetalSimulation(model, batch_size=2, qpos=qpos, qvel=qvel0,
                               profile="integrated_euler_v1")
  simulation.reset(qpos=qpos, qvel=qvel0,
                   eq_active=np.ones((2, int(model.neq)), dtype=np.int32))
  qpos_t = torch.as_tensor(qpos, dtype=torch.float32, device="mps")
  qvel0_t = torch.as_tensor(qvel0, dtype=torch.float32, device="mps")
  qvel1_t = torch.as_tensor(qvel1, dtype=torch.float32, device="mps")
  _, status0, dynamics = simulation._acceleration(qpos_t, qvel0_t)
  np.testing.assert_array_equal(status0.cpu().numpy(), [0, 0])
  context = simulation._capture_forward_position(qpos_t, dynamics)
  cc = simulation._coupled_constraints
  desc = cc.descriptor
  before = cc._assembly_views(include_optimizer_outputs=False)
  nr = int(desc.nr)
  active = before["active"].cpu().numpy().copy()
  assert np.any(active[:, int(desc.eq_rowadr[0]):
                       int(desc.eq_rowadr[0] + desc.eq_rownum[0])] > .5)
  tendon_rows = int(desc.ten_base)
  assert np.any(active[:, tendon_rows:
                       tendon_rows + int(desc.ten_friction_rows + desc.ten_limit_rows)] > .5)
  flex_start = int(desc.flex_contact_base)
  flex_stop = flex_start + int(desc.n_flex_contact_rows)
  assert flex_stop <= nr and np.any(active[:, flex_start:flex_stop] > .5)
  retained = {name: before[name].clone() for name in ("J_packed", "R", "active")}
  snapshot = simulation.snapshot()

  actual_qacc, status1, _ = simulation._acceleration(
      qpos_t, qvel1_t, skip_sleep_prepare=True, position_context=context)
  np.testing.assert_array_equal(status1.cpu().numpy(), [0, 0])
  after = cc._assembly_views(include_optimizer_outputs=False)
  for name in retained:
    torch.testing.assert_close(after[name], retained[name], rtol=0, atol=0)
  expected_qacc = []
  for world in range(2):
    data = mujoco.MjData(model)
    data.qpos[:] = qpos[world]
    data.qvel[:] = qvel0[world]
    data.eq_active[:] = 1
    mujoco.mj_forward(model, data)
    data.qvel[:] = qvel1[world]
    mujoco.mj_forwardSkip(model, data, mujoco.mjtStage.mjSTAGE_POS, 1)
    expected_qacc.append(data.qacc.copy())
  np.testing.assert_allclose(actual_qacc.cpu().numpy(), expected_qacc,
                             rtol=3e-3, atol=3e-3)

  # Turning the bilateral equality off must not leave its old active bit or
  # bounds in the next full assembly; tendon and flex owners remain live.
  inactive = np.zeros((2, int(model.neq)), dtype=np.int32)
  simulation.reset(qpos=qpos, qvel=qvel1, eq_active=inactive)
  _, status2, _ = simulation._acceleration(qpos_t, qvel1_t)
  np.testing.assert_array_equal(status2.cpu().numpy(), [0, 0])
  off = cc._assembly_views(include_optimizer_outputs=False)
  eq_start = int(desc.eq_rowadr[0])
  eq_stop = eq_start + int(desc.eq_rownum[0])
  assert not np.any(off["active"][:, eq_start:eq_stop].cpu().numpy() > .5)
  assert np.all(off["lo"][:, eq_start:eq_stop].cpu().numpy() == 0)
  assert np.all(off["hi"][:, eq_start:eq_stop].cpu().numpy() == 0)
  assert np.any(off["active"][:, tendon_rows:
                              tendon_rows + int(desc.ten_friction_rows + desc.ten_limit_rows)].cpu().numpy() > .5)
  assert np.any(off["active"][:, flex_start:flex_stop].cpu().numpy() > .5)

  simulation.restore(snapshot)
  restored_pos = simulation.state.qpos.cpu().numpy()
  restored_vel = simulation.state.qvel.cpu().numpy()
  pos_t = torch.as_tensor(restored_pos, dtype=torch.float32, device="mps")
  vel_t = torch.as_tensor(restored_vel, dtype=torch.float32, device="mps")
  _, status3, _ = simulation._acceleration(pos_t, vel_t)
  np.testing.assert_array_equal(status3.cpu().numpy(), [0, 0])
  restored = cc._assembly_views(include_optimizer_outputs=False)
  assert np.any(restored["active"][:, eq_start:eq_stop].cpu().numpy() > .5)
  assert np.any(restored["active"][:, tendon_rows:
                                   tendon_rows + int(desc.ten_friction_rows + desc.ten_limit_rows)].cpu().numpy() > .5)
  assert np.any(restored["active"][:, flex_start:flex_stop].cpu().numpy() > .5)
