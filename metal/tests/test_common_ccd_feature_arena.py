# Copyright 2026 The MuJoCo Metal contributors
# Licensed under the Apache License, Version 2.0.
"""Actual EPA support-ID writes must stay in each query's int2 arena."""
import os

import mujoco
import numpy as np
import pytest

from test_common_ccd_rigid_native import _XML, _XML_REVERSED, _sorted_contacts


@pytest.mark.gpu
@pytest.mark.skipif(os.getenv("MUJOCO_METAL_RUN_GPU") != "1", reason="native opt-in")
@pytest.mark.parametrize("xml", [_XML, _XML_REVERSED])
def test_concurrent_rigid_queries_write_disjoint_feature_pairs_and_manifolds(xml):
  from mujoco_metal.simulation import MetalSimulation

  model = mujoco.MjModel.from_xml_string(xml)
  qpos = np.repeat(model.qpos0.astype(np.float32)[None], 3, axis=0)
  for world in range(3):
    qpos[world, :3] += np.asarray([.03 * world, -.02 * world, .01 * world], np.float32)
    qpos[world, 7:10] += np.asarray([.03 * world, -.02 * world, .01 * world], np.float32)
  refs = [mujoco.MjData(model) for _ in range(3)]
  for world, data in enumerate(refs):
    data.qpos[:] = qpos[world]
    mujoco.mj_forward(model, data)
    assert data.ncon == 4
  simulation = MetalSimulation(model, batch_size=3, qpos=qpos,
                               profile="integrated_euler_v1")
  program = simulation._coupled_constraints
  layout = program._common_ccd_layout
  base = int(layout["vertex_id_base"])
  capacity = int(layout["vertex_capacity"])
  sidecar = program._constants["mesh_hull_info"]
  # Initialize only the allocated runtime feature arena, leaving compiled
  # descriptors untouched. The actual candidate producer must write a pair
  # in every active world's range, not merely produce plausible contacts.
  sidecar[base:base + int(layout["vertex_id_words"])].fill_(-123)
  actual = simulation.assembled_system(recompute=True)
  host = lambda value: value.detach().cpu().numpy().copy()
  ids = host(sidecar)[base:base + int(layout["vertex_id_words"])].reshape(3, capacity, 2)
  assert np.all(ids[:, 0] >= 0), ids[:, 0]
  np.testing.assert_array_equal(host(actual["status"]), 0)
  masks = host(actual["contact_mask"]) > .5
  for world, data in enumerate(refs):
    assert int(masks[world].sum()) == data.ncon
    pos, dist, normal = _sorted_contacts(
        host(actual["contact_position"])[world, masks[world]],
        host(actual["contact_distance"])[world, masks[world]],
        host(actual["contact_normal"])[world, masks[world]])
    cpu_pos, cpu_dist, cpu_normal = _sorted_contacts(
        np.asarray([c.pos for c in data.contact[:data.ncon]]),
        np.asarray([c.dist for c in data.contact[:data.ncon]]),
        np.asarray([c.frame[:3] for c in data.contact[:data.ncon]]))
    np.testing.assert_allclose(pos, cpu_pos, rtol=0, atol=2e-4)
    np.testing.assert_allclose(dist, cpu_dist, rtol=0, atol=2e-5)
    np.testing.assert_allclose(normal, cpu_normal, rtol=0, atol=2e-4)
    np.testing.assert_allclose(host(actual["qfrc_constraint"])[world],
                               data.qfrc_constraint, rtol=2e-3, atol=2e-3)
