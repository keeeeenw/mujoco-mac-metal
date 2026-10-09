# Copyright 2026 The MuJoCo Metal contributors
# Licensed under the Apache License, Version 2.0.
"""The NATIVECCD disable flag selects pinned libccd MPR, not GJK/EPA."""
import os

import mujoco
import numpy as np
import pytest

from test_common_ccd_rigid_native import _CONVEX_PAIRS, _pair_xml, _sorted_contacts


def _model(pair, multiccd):
  model = mujoco.MjModel.from_xml_string(_pair_xml(*pair, multiccd=multiccd))
  model.opt.disableflags |= int(mujoco.mjtDisableBit.mjDSBL_NATIVECCD)
  return model


def test_pinned_libccd_flag_changes_actual_cylinder_box_witnesses():
  model = _model((5, 6), True)
  data = mujoco.MjData(model)
  data.qpos[:] = np.asarray(model.qpos0, np.float32)
  mujoco.mj_forward(model, data)
  alternate = np.asarray([c.pos.copy() for c in data.contact[:data.ncon]])
  assert data.ncon > 0
  model.opt.disableflags &= ~int(mujoco.mjtDisableBit.mjDSBL_NATIVECCD)
  mujoco.mj_forward(model, data)
  native = np.asarray([c.pos.copy() for c in data.contact[:data.ncon]])
  assert alternate.shape != native.shape or not np.allclose(alternate, native, atol=1e-5)


def test_pinned_libccd_pair_matrix_fixture_has_actual_contacts():
  for pair in _CONVEX_PAIRS:
    for reverse in (False, True):
      for multiccd in (False, True):
        model = _model(pair[::-1] if reverse else pair, multiccd)
        data = mujoco.MjData(model)
        data.qpos[:] = np.asarray(model.qpos0, np.float32)
        data.qpos[7:10] = np.asarray([.04, .013, .217], np.float32)
        mujoco.mj_forward(model, data)
        assert data.ncon > 0, (pair, reverse, multiccd)
        assert all(c.dist < 0 for c in data.contact[:data.ncon])


@pytest.mark.gpu
@pytest.mark.skipif(os.getenv("MUJOCO_METAL_RUN_GPU") != "1", reason="native opt-in")
@pytest.mark.parametrize("pair", _CONVEX_PAIRS)
@pytest.mark.parametrize("multiccd", [False, True])
@pytest.mark.parametrize("reverse", [False, True])
def test_native_libccd_dispatch_matches_pinned_contacts_and_forces(pair, multiccd, reverse):
  from mujoco_metal.simulation import MetalSimulation

  model = _model(pair[::-1] if reverse else pair, multiccd)
  qpos = np.repeat(np.asarray(model.qpos0, np.float32)[None], 2, axis=0)
  # Avoid symmetric enclosed configurations where pinned MPR itself returns
  # no witness. The CPU gate above verifies every tested pair is nonvacuous.
  qpos[0, 7:10] = [.04, .013, .217]
  # An independent separated world verifies that the alternate producer does
  # not retain contacts from its touching peer or write to shared row storage.
  qpos[1, 7] = 2
  qvel = np.zeros((2, model.nv), np.float32)
  qvel[0] = np.linspace(-.12, .17, model.nv)
  simulation = MetalSimulation(model, batch_size=2, qpos=qpos, qvel=qvel,
                               profile="integrated_euler_v1")
  actual = simulation.assembled_system(recompute=True)
  np.testing.assert_array_equal(actual["status"].cpu().numpy(), 0)
  for world in range(2):
    data = mujoco.MjData(model)
    data.qpos[:], data.qvel[:] = qpos[world], qvel[world]
    mujoco.mj_forward(model, data)
    mask = actual["contact_mask"][world].cpu().numpy() > .5
    assert int(mask.sum()) == data.ncon
    assert (data.ncon > 0) == (world == 0)
    if not data.ncon:
      np.testing.assert_array_equal(actual["qfrc_constraint"][world].cpu().numpy(), 0)
      continue
    position, distance, normal = _sorted_contacts(
        actual["contact_position"][world, mask].cpu().numpy(),
        actual["contact_distance"][world, mask].cpu().numpy(),
        actual["contact_normal"][world, mask].cpu().numpy())
    expected_position, expected_distance, expected_normal = _sorted_contacts(
        np.asarray([c.pos for c in data.contact[:data.ncon]]),
        np.asarray([c.dist for c in data.contact[:data.ncon]]),
        np.asarray([c.frame[:3] for c in data.contact[:data.ncon]]))
    np.testing.assert_allclose(position, expected_position, rtol=0, atol=2e-4)
    np.testing.assert_allclose(distance, expected_distance, rtol=0, atol=2e-5)
    np.testing.assert_allclose(normal, expected_normal, rtol=0, atol=2e-4)
    np.testing.assert_allclose(actual["qfrc_constraint"][world].cpu().numpy(),
                               data.qfrc_constraint, rtol=2e-3, atol=2e-3)
    np.testing.assert_allclose(actual["qacc"][world].cpu().numpy(),
                               data.qacc, rtol=2e-3, atol=2e-3)


@pytest.mark.gpu
@pytest.mark.skipif(os.getenv("MUJOCO_METAL_RUN_GPU") != "1", reason="native opt-in")
@pytest.mark.parametrize("bound", [16, 32])
@pytest.mark.parametrize("condim", [1, 3, 4, 6])
def test_libccd_compiled_mesh_graph_trajectory_reset_restore(bound, condim):
  from test_compiled_collision_hull import _simplified_model
  from mujoco_metal.simulation import MetalSimulation

  model = _simplified_model(bound)
  model.opt.disableflags |= int(mujoco.mjtDisableBit.mjDSBL_NATIVECCD)
  model.geom_condim[:] = condim
  model.geom_friction[:] = [.7, .04, .02]
  qpos = np.repeat(model.qpos0.astype(np.float32)[None], 2, axis=0)
  qpos[1, 0] = .6
  qvel = np.repeat(np.asarray([[.02, -.03, .01, .2, -.1, .3]], np.float32),
                   2, axis=0)
  refs = [mujoco.MjData(model) for _ in range(2)]
  for world, data in enumerate(refs):
    data.qpos[:], data.qvel[:] = qpos[world], qvel[world]
    mujoco.mj_forward(model, data)
  assert refs[0].ncon == 1 and refs[1].ncon == 0
  assert np.linalg.norm(refs[0].qfrc_constraint) > 100
  simulation = MetalSimulation(model, batch_size=2, qpos=qpos, qvel=qvel,
                               profile="integrated_scalable_v1")
  host = lambda value: value.detach().cpu().numpy().copy()
  initial = simulation.assembled_system()
  np.testing.assert_array_equal(host(initial["status"]), 0)
  masks = host(initial["contact_mask"]) > .5
  assert int(masks[0].sum()) == 1 and not masks[1].any()
  slot = np.flatnonzero(masks[0])[0]
  for field, expected, tolerance in (
      ("contact_distance", refs[0].contact[0].dist, 2e-5),
      ("contact_position", refs[0].contact[0].pos, 2e-4),
      ("contact_normal", refs[0].contact[0].frame[:3], 2e-4)):
    np.testing.assert_allclose(host(initial[field])[0, slot], expected,
                               rtol=0, atol=tolerance)
  np.testing.assert_allclose(host(initial["qfrc_constraint"]),
                             [data.qfrc_constraint for data in refs],
                             rtol=8e-3, atol=8e-3)
  for _ in range(8):
    simulation.step()
    for data in refs:
      mujoco.mj_step(model, data)
    np.testing.assert_array_equal(host(simulation.state.status), 0)
    for field in ("qpos", "qvel", "qacc"):
      np.testing.assert_allclose(host(getattr(simulation.state, "_" + field)),
                                 [getattr(data, field) for data in refs],
                                 rtol=8e-3, atol=8e-3)
  checkpoint = simulation.snapshot()
  simulation.step()
  expected = [host(getattr(simulation.state, "_" + field))
              for field in ("qpos", "qvel", "qacc")]
  simulation.restore(checkpoint)
  simulation.step()
  for field, wanted in zip(("qpos", "qvel", "qacc"), expected):
    np.testing.assert_array_equal(host(getattr(simulation.state, "_" + field)), wanted)
  simulation.reset(qpos=qpos, qvel=qvel)
  reset = simulation.assembled_system()
  for field in ("contact_position", "contact_distance", "contact_normal",
                "contact_mask", "qfrc_constraint", "qacc", "status"):
    np.testing.assert_array_equal(host(reset[field]), host(initial[field]))
