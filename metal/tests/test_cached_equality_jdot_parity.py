# Copyright 2026 The MuJoCo Metal contributors
# Licensed under the Apache License, Version 2.0.
"""Real articulated equality rows, including cached-POS velocity replay."""
import os

import mujoco
import numpy as np
import pytest


def _fixture(kind, endpoints):
  equality = (f'<{kind} site1="tip_a" site2="tip_b"/>' if endpoints == "site" else
              f'<{kind} body1="end_a" body2="end_b"' +
              (' anchor=".8 .1 .4"/>' if kind == "connect" else '/>'))
  model = mujoco.MjModel.from_xml_string(f'''<mujoco>
    <option gravity="0 0 0" solver="PGS" iterations="0"><flag contact="disable"/></option>
    <worldbody>
      <body pos=".1 .2 .3" quat=".9659258263 0 .2588190451 0"><joint axis="0 0 1"/>
        <geom type="box" pos=".15 0 .1" size=".05 .1 .08" mass="2"/>
        <body name="end_a" pos=".4 .1 .2"><joint axis="0 1 0"/>
          <geom pos=".1 .2 .15" size=".1" mass="1"/>
          <site name="tip_a" pos=".2 .15 .1" quat=".9238795325 .3826834324 0 0"/>
        </body>
      </body>
      <body pos="1 -.2 .5" quat=".9238795325 .3826834324 0 0"><joint axis="1 0 0"/>
        <geom type="box" pos=".1 -.15 .2" size=".07 .1 .08" mass="1.2"/>
        <body name="end_b" pos=".3 .2 .1"><joint axis="0 0 1"/>
          <geom pos=".2 .1 -.1" size=".08" mass=".7"/>
          <site name="tip_b" pos="-.1 .2 .15" quat=".9659258263 0 0 .2588190451"/>
        </body>
      </body>
    </worldbody><equality>{equality}</equality></mujoco>''')
  q3 = np.array([.35, -.4, .2, .45], np.float32)
  v3 = np.array([.7, -.8, .45, -.6], np.float32)
  q0 = np.array([-.2, .25, -.3, .1], np.float32)
  v0 = np.array([-.4, .65, -.55, .3], np.float32)
  data = mujoco.MjData(model)
  data.qpos[:], data.qvel[:] = q3, v3
  mujoco.mj_forward(model, data)
  return model, data, q3, v3, q0, v0


def _rows(data, nv):
  base = (-data.efc_KBIP[:, 1] * data.efc_vel
          - data.efc_KBIP[:, 0] * data.efc_KBIP[:, 2]
          * (data.efc_pos - data.efc_margin))
  return {"J": data.efc_J.reshape(data.nefc, nv).copy(),
          "R": data.efc_R.copy(), "aref": data.efc_aref.copy(),
          "extra": data.efc_aref - base,
          "pos": data.efc_pos.copy()}


@pytest.mark.parametrize("kind", ["connect", "weld"])
@pytest.mark.parametrize("endpoints", ["body", "site"])
def test_pinned_cached_equality_keeps_position_rows_and_refreshes_jdot(kind, endpoints):
  model, data, _q3, _v3, q0, v0 = _fixture(kind, endpoints)
  previous = _rows(data, model.nv)
  assert np.max(np.abs(previous["extra"])) > .1
  data.qpos[:], data.qvel[:] = q0, v0
  mujoco.mj_forwardSkip(model, data, mujoco.mjtStage.mjSTAGE_POS, 1)
  refreshed = _rows(data, model.nv)
  np.testing.assert_array_equal(refreshed["J"], previous["J"])
  np.testing.assert_array_equal(refreshed["R"], previous["R"])
  np.testing.assert_array_equal(refreshed["pos"], previous["pos"])
  assert np.max(np.abs(refreshed["extra"] - previous["extra"])) > .03
  assert np.max(np.abs(refreshed["aref"] - previous["aref"])) > 1
  mujoco.mj_forward(model, data)
  recomputed = _rows(data, model.nv)
  assert np.max(np.abs(recomputed["J"] - refreshed["J"])) > .03
  assert np.max(np.abs(recomputed["aref"] - refreshed["aref"])) > 10


@pytest.mark.gpu
@pytest.mark.skipif(os.getenv("MUJOCO_METAL_RUN_GPU") != "1", reason="native opt-in")
@pytest.mark.parametrize("kind", ["connect", "weld"])
@pytest.mark.parametrize("endpoints", ["body", "site"])
def test_native_articulated_equality_rows_survive_overwritten_pose_workspaces(
    kind, endpoints, monkeypatch):
  import torch
  from mujoco_metal import MetalSimulation

  model, data, q3, v3, q0, v0 = _fixture(kind, endpoints)
  initial = _rows(data, model.nv)
  simulation = MetalSimulation(model, qpos=q3[None], qvel=v3[None],
                                profile="integrated_euler_v1")
  def tensor(value):
    return torch.tensor(value[None], dtype=torch.float32, device="mps")
  q3d, v3d, q0d, v0d = (tensor(x) for x in (q3, v3, q0, v0))
  _, status, dynamics = simulation._acceleration(q3d, v3d)
  np.testing.assert_array_equal(status.cpu().numpy(), 0)
  cc = simulation._coupled_constraints
  nr, ne = cc.descriptor.nr, int(data.ne)
  def current_rows():
    debug = cc._workspace["workspace_debug"].reshape(1, cc._debug_stride)
    return {"J": cc._workspace["workspace_J"].reshape(1, nr, model.nv)[0, :ne].cpu().numpy().copy(),
            "R": debug[0, nr*nr:nr*nr+ne].cpu().numpy().copy(),
            "aref": debug[0, nr*nr+nr:nr*nr+nr+ne].cpu().numpy().copy()}
  for name, value in current_rows().items():
    np.testing.assert_allclose(value, initial[name], atol=2e-3, rtol=3e-5, err_msg=name)
  context = simulation._capture_forward_position(q3d, dynamics)
  # A valid context retains its own POS values across unrelated full forwards.
  simulation._acceleration(q0d, v0d)
  data.qpos[:], data.qvel[:] = q0, v0
  mujoco.mj_forwardSkip(model, data, mujoco.mjtStage.mjSTAGE_POS, 1)
  reference = _rows(data, model.nv)
  def forbid(*_args, **_kwargs):
    raise AssertionError("cached velocity replay rebuilt equality/POS")
  monkeypatch.setattr(cc, "_equality_kernel", forbid)
  monkeypatch.setattr(simulation._smooth, "run_device", forbid)
  _, status, _ = simulation._acceleration(
      q0d, v0d, skip_sleep_prepare=True, position_context=context)
  np.testing.assert_array_equal(status.cpu().numpy(), 0)
  for name, value in current_rows().items():
    np.testing.assert_allclose(value, reference[name], atol=2e-3, rtol=3e-5, err_msg=name)


def _cached_low_batch_fixture():
  return mujoco.MjModel.from_xml_string('''<mujoco>
    <option gravity="0 0 0" solver="Newton" iterations="80" tolerance="1e-8">
      <flag contact="disable"/>
    </option>
    <worldbody>
      <body name="a"><joint name="ja" type="hinge" axis="0 0 1"/>
        <geom type="capsule" fromto="0 0 0 0 0 1" size=".1" mass="1"
          contype="0" conaffinity="0"/>
      </body>
      <body name="b" pos="1 0 0"><joint name="jb" type="hinge" axis="0 1 0"/>
        <geom type="capsule" fromto="0 0 0 0 0 1" size=".1" mass="1"
          contype="0" conaffinity="0"/>
      </body>
    </worldbody>
    <equality><joint joint1="ja" joint2="jb" solref=".013 .7"
      solimp=".8 .9 .003 .5 2"/></equality>
  </mujoco>''')


@pytest.mark.parametrize("batch", [2, 3])
@pytest.mark.gpu
@pytest.mark.skipif(os.getenv("MUJOCO_METAL_RUN_GPU") != "1",
                    reason="native opt-in")
def test_native_cached_equality_velocity_low_scale_is_world_safe(batch):
  """Cached equality VEL refresh stays within static row-scale storage."""
  import torch
  from mujoco_metal import MetalSimulation

  model = _cached_low_batch_fixture()
  qpos0 = np.asarray([[.2 + .1 * w, -.1 + .05 * w]
                      for w in range(batch)], dtype=np.float32)
  qvel0 = np.asarray([[.7 - .1 * w, -.4 + .2 * w]
                      for w in range(batch)], dtype=np.float32)
  qvel1 = np.asarray([[-.3 + .15 * w, .6 - .1 * w]
                      for w in range(batch)], dtype=np.float32)
  simulation = MetalSimulation(model, qpos=qpos0, qvel=qvel0,
                               batch_size=batch,
                               profile="integrated_euler_v1")
  qpos_device = torch.tensor(qpos0, dtype=torch.float32, device="mps")
  qvel0_device = torch.tensor(qvel0, dtype=torch.float32, device="mps")
  qvel1_device = torch.tensor(qvel1, dtype=torch.float32, device="mps")
  _, status0, dynamics = simulation._acceleration(qpos_device, qvel0_device)
  np.testing.assert_array_equal(status0.cpu().numpy(), np.zeros(batch))
  context = simulation._capture_forward_position(qpos_device, dynamics)

  cc = simulation._coupled_constraints
  # Equality rows occupy the canonical leading [0:n_eq_rows) block.
  eq_row = int(cc.descriptor.eq_rowadr[0])
  assert abs(float(cc._row_velocity_scales_low_cpu[eq_row])) > 1e-8
  expected_qacc, expected_force, expected_aref = [], [], []
  for world in range(batch):
    data = mujoco.MjData(model)
    data.qpos[:] = qpos0[world]
    data.qvel[:] = qvel0[world]
    mujoco.mj_forward(model, data)
    data.qvel[:] = qvel1[world]
    mujoco.mj_forwardSkip(model, data, mujoco.mjtStage.mjSTAGE_POS, 1)
    expected_qacc.append(data.qacc.copy())
    expected_force.append(data.qfrc_constraint.copy())
    eq = np.flatnonzero(data.efc_type == int(mujoco.mjtConstraint.mjCNSTR_EQUALITY))
    assert eq.size == 1
    expected_aref.append(float(data.efc_aref[eq[0]]))

  actual_qacc, status1, _ = simulation._acceleration(
      qpos_device, qvel1_device, skip_sleep_prepare=True,
      position_context=context)
  np.testing.assert_array_equal(status1.cpu().numpy(), np.zeros(batch))
  np.testing.assert_allclose(actual_qacc.cpu().numpy(), expected_qacc,
                             rtol=4e-5, atol=3e-4)
  actual_force = simulation._last_coupled["qfrc_constraint"].cpu().numpy()
  np.testing.assert_allclose(actual_force, expected_force, rtol=4e-5, atol=3e-4)

  nr = int(cc.descriptor.nr)
  debug = cc._workspace["workspace_debug"].reshape(batch, cc._debug_stride)
  high_offset = nr * nr + nr + eq_row
  low = cc._workspace["position_current_aref_low"].reshape(batch, nr)
  high = debug[:, high_offset].cpu().numpy().astype(np.float64)
  low_word = low[:, eq_row].cpu().numpy().astype(np.float64)
  reconstructed = high + low_word
  np.testing.assert_allclose(reconstructed, expected_aref,
                             rtol=2e-6, atol=2e-5)
  np.testing.assert_allclose(low_word,
                             np.asarray(expected_aref) - high,
                             rtol=0.0, atol=2e-5)
  assert np.all(np.isfinite(low[:, eq_row].cpu().numpy()))
