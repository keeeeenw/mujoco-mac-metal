# Copyright 2026 The MuJoCo Metal contributors
# Licensed under the Apache License, Version 2.0.
"""Pinned alternate MPR must also generate every heightfield prism witness."""
import os

import mujoco
import numpy as np
import pytest

from test_mesh_hfield_contact_012 import _model, _XML, _ELEVATION, _GRID, _LIMITS


@pytest.mark.gpu
@pytest.mark.skipif(os.getenv("MUJOCO_METAL_RUN_GPU") != "1", reason="native opt-in")
@pytest.mark.parametrize("terrain", ["flat", "slope", "ridge"])
@pytest.mark.parametrize("margin", [0., .005])
def test_alternate_mpr_heightfield_mesh_contacts_forces_and_replay(terrain, margin):
  from mujoco_metal.simulation import MetalSimulation

  xml = _XML
  if terrain == "slope":
    xml = xml.replace(_ELEVATION, " ".join(str(j / (_GRID - 1))
                                          for i in range(_GRID) for j in range(_GRID)))
  elif terrain == "ridge":
    xml = xml.replace(_ELEVATION, " ".join(str(1. if abs(j - 4) <= 1 else .1)
                                          for i in range(_GRID) for j in range(_GRID)))
  model = _model(xml=xml)
  model.opt.disableflags |= int(mujoco.mjtDisableBit.mjDSBL_NATIVECCD)
  model.geom_margin[:] = margin
  qpos = np.repeat(model.qpos0.astype(np.float32)[None], 2, axis=0)
  qpos[0, :2] = [.011, .017]
  qpos[1, 2] = 1
  qvel = np.repeat(np.asarray([[.02, -.01, .03, .04, -.02, .01]], np.float32),
                   2, axis=0)
  refs = [mujoco.MjData(model) for _ in range(2)]
  for world, data in enumerate(refs):
    data.qpos[:], data.qvel[:] = qpos[world], qvel[world]
    mujoco.mj_forward(model, data)
  assert refs[0].ncon >= 4 and refs[1].ncon == 0
  assert np.linalg.norm(refs[0].qfrc_constraint) > 1
  simulation = MetalSimulation(model, batch_size=2, qpos=qpos, qvel=qvel,
                               profile="integrated_scalable_v1", limits=_LIMITS)
  host = lambda value: value.detach().cpu().numpy().copy()
  actual = simulation.assembled_system(recompute=True)
  np.testing.assert_array_equal(host(actual["status"]), 0)
  mask = host(actual["contact_mask"]) > .5
  assert int(mask[0].sum()) == refs[0].ncon
  assert not mask[1].any()
  # Both routes walk the same source row/column/triangle order. Comparing by
  # index also checks complete prism enumeration rather than nearest matches.
  for field, expected, tolerance in (
      ("contact_distance", [c.dist for c in refs[0].contact[:refs[0].ncon]], 2e-5),
      ("contact_position", [c.pos for c in refs[0].contact[:refs[0].ncon]], 2e-4),
      ("contact_normal", [c.frame[:3] for c in refs[0].contact[:refs[0].ncon]], 2e-4)):
    np.testing.assert_allclose(host(actual[field])[0, mask[0]], expected,
                               rtol=0, atol=tolerance)
  np.testing.assert_allclose(host(actual["qfrc_constraint"]),
                             [data.qfrc_constraint for data in refs],
                             rtol=2e-3, atol=2e-3)
  for _ in range(4):
    simulation.step()
    for data in refs:
      mujoco.mj_step(model, data)
    np.testing.assert_array_equal(host(simulation.state.status), 0)
    for field in ("qpos", "qvel", "qacc"):
      np.testing.assert_allclose(host(getattr(simulation.state, "_" + field)),
                                 [getattr(data, field) for data in refs],
                                 rtol=2e-3, atol=2e-3)
  snapshot = simulation.snapshot()
  simulation.step()
  expected = [host(getattr(simulation.state, "_" + field))
              for field in ("qpos", "qvel", "qacc")]
  simulation.restore(snapshot)
  simulation.step()
  for field, wanted in zip(("qpos", "qvel", "qacc"), expected):
    np.testing.assert_array_equal(host(getattr(simulation.state, "_" + field)), wanted)
