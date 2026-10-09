# Copyright 2026 The MuJoCo Metal contributors
# Licensed under the Apache License, Version 2.0.
"""Actual compiled octree assets and native ray hits against the pinned engine."""
import os

import mujoco
import numpy as np
import pytest

from mujoco_metal.ray_queries import lower_ray_scene, _ray_shader_source
from test_ray_queries_019 import _matrix_quat


def _model():
  vertices = "-.3 -.3 -.2 .3 -.3 -.2 .3 .3 -.2 -.3 .3 -.2 " \
             "-.3 -.3 .2 .3 -.3 .2 .3 .3 .2 -.3 .3 .2"
  return mujoco.MjModel.from_xml_string(f'''<mujoco><asset>
    <mesh name="cube" vertex="{vertices}"/>
    </asset><worldbody>
    <geom type="sdf" mesh="cube" pos="0 0 0"/>
    <geom type="sdf" mesh="cube" pos="2 0 .1" euler="10 20 5"/>
    </worldbody></mujoco>''')


def test_actual_ray_octree_lowering_retains_three_planes_and_shared_asset():
  model = _model()
  for field in (model.oct_aabb, model.oct_coeff):
    nonzero = field != 0
    field[nonzero] = np.nextafter(field[nonzero], np.inf)
  scene = lower_ray_scene(model)
  np.testing.assert_array_equal(scene.octree_info[0], scene.octree_info[1])
  offset, count = scene.octree_info[0]
  assert offset == 0 and scene.octree.size == 50*count
  adr = int(model.mesh_octadr[0])
  np.testing.assert_array_equal(scene.octree[:8*count], model.oct_child[adr:adr+count].reshape(-1))
  for name, width, starts in (("oct_aabb", 6, (8, 22, 36)),
                              ("oct_coeff", 8, (14, 28, 42))):
    planes = [scene.octree[start*count:(start+width)*count].astype(np.float64)
              for start in starts]
    assert np.any(planes[1] != 0)
    np.testing.assert_array_equal((planes[0]+planes[1]+planes[2]).view(np.uint64),
        getattr(model, name)[adr:adr+count].reshape(-1).view(np.uint64))
  assert scene.nbytes == sum(v.nbytes for v in vars(scene).values()
                             if isinstance(v, np.ndarray))
  assert not scene.octree.flags.writeable


def test_ray_octree_rejects_nonfinite_constants_before_device_runtime():
  model = _model()
  model.oct_coeff[0, 0] = np.nan
  with pytest.raises(ValueError, match="finite"):
    lower_ray_scene(model)


def test_actual_ray_library_exposes_helpers_without_unrelated_sensor_entries():
  source = _ray_shader_source()
  assert source.index("struct SdfWide") < source.index("struct SdfMatrixWide")
  assert source.startswith("#define MUJOCO_METAL_SENSOR_ENTRY 0")
  assert "#define MUJOCO_METAL_SDF_HELPERS_ONLY" in source
  assert source.index("struct SdfWide") < source.index("inline float qr_sdf(")


@pytest.mark.skipif(os.environ.get("MUJOCO_METAL_RUN_GPU") != "1",
                   reason="explicit serialized native ray qualification")
def test_actual_native_plugin_free_octree_rays_match_pinned_hits_misses_and_normals():
  import torch
  from mujoco_metal.ray_queries import MetalRayQueries
  model = _model()
  data = mujoco.MjData(model)
  mujoco.mj_forward(model, data)
  origins = np.array([[.04, .07, 1], [2.04, .07, 1], [3, 3, 1],
                      [.4, 0, .01], [.04, .07, .01]], np.float32)
  vectors = np.array([[0, 0, -2], [.01, -.01, -1], [0, 0, -1],
                      [-1, 0, 0], [0, 0, 1]], np.float32)
  origins, vectors = np.stack([origins]*2), np.stack([vectors]*2)
  origins[1, :, 0] += np.float32(.02)
  tensor = lambda value: torch.tensor(np.asarray(value).copy(), device="mps")
  poses = {"geom_pos": tensor(np.stack([data.geom_xpos]*2).astype(np.float32)),
           "geom_quat": tensor(np.array([[_matrix_quat(mat) for mat in data.geom_xmat]]*2,
                                         np.float32))}
  query = MetalRayQueries(model, 2)
  output = query.run_device(poses, tensor(origins), tensor(vectors))
  expected_distance = np.empty((2, len(origins[0])))
  expected_id = np.empty_like(expected_distance, dtype=np.int32)
  expected_normal = np.zeros((*expected_distance.shape, 3))
  for world in range(2):
    for ray, (origin, direction) in enumerate(zip(origins[world], vectors[world])):
      gid = np.array([-1], np.int32)
      expected_distance[world, ray] = mujoco.mj_ray(model, data, origin.astype(float),
          direction.astype(float), None, True, -1, gid, expected_normal[world, ray])
      expected_id[world, ray] = gid[0]
  assert np.count_nonzero(expected_distance >= 0) >= 6
  assert np.count_nonzero(expected_distance < 0) >= 2
  np.testing.assert_array_equal(output["status"].cpu(), 0)
  np.testing.assert_array_equal(output["geomid"].cpu(), expected_id)
  np.testing.assert_allclose(output["distance"].cpu(), expected_distance, rtol=2e-5, atol=5e-6)
  np.testing.assert_allclose(output["normal"].cpu(), expected_normal, rtol=2e-4, atol=3e-5)
  retained = {name: value.clone() for name, value in output.items()}
  query.run_device(poses, tensor(origins+np.float32(10)), tensor(vectors))
  for name in output:
    torch.testing.assert_close(output[name], retained[name], rtol=0, atol=0)
