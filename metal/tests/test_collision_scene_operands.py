# Copyright 2026 The MuJoCo Metal contributors
# Licensed under the Apache License, Version 2.0.
"""Collision masks cannot hide assets required by geometry queries."""
import mujoco
import numpy as np
import pytest

from mujoco_metal.coupled_constraints import (
    lower_collision_geometry, lower_coupled_constraints)
from test_dynamic_hfield_capacity import _model


def test_query_and_physics_pack_identical_complete_terrain_operands():
  model = _model(count=1)
  physical = lower_coupled_constraints(model)
  pairs = tuple(zip(physical.geom1.tolist(), physical.geom2.tolist()))
  query = lower_collision_geometry(model, pairs)
  np.testing.assert_array_equal(query.mesh_hull, physical.mesh_hull)
  np.testing.assert_array_equal(query.mesh_hull_info, physical.mesh_hull_info)
  np.testing.assert_array_equal(query.geom_size, physical.geom_size)
  assert not query.mesh_hull.flags.writeable
  assert query.byte_count == query.mesh_hull.nbytes + query.mesh_hull_info.nbytes


def test_queries_pack_static_masked_assets_without_changing_model():
  model = _model(count=9)
  model.geom_contype[:] = 0
  model.geom_conaffinity[:] = 0
  before = model.geom_contype.copy(), model.geom_conaffinity.copy()
  physical = lower_coupled_constraints(model)
  assert physical.npairs == 0
  sphere = model.geom('sphere').id
  query = lower_collision_geometry(model, [(g, sphere) for g in range(9)])
  for geom in range(9):
    offset, rows, cols = query.mesh_hull_info[9*geom+5:9*geom+8]
    assert (rows, cols) == (80, 96)
    np.testing.assert_array_equal(query.mesh_hull[offset:offset+rows*cols],
                                 model.hfield_data)
  np.testing.assert_array_equal(model.geom_contype, before[0])
  np.testing.assert_array_equal(model.geom_conaffinity, before[1])


@pytest.mark.parametrize('pairs', [[(-1, 0)], [(0, 100)], [(False, 0)], [(0., 1)]])
def test_invalid_scene_pair_ids_fail_before_packing(pairs):
  with pytest.raises(ValueError, match='geom IDs'):
    lower_collision_geometry(_model(), pairs)


def test_model_sized_mesh_payload_retains_every_compiled_vertex_and_face():
  from test_compiled_collision_hull import _simplified_model
  from mujoco_metal.mesh_hull import collision_mesh_hull
  model = _simplified_model(-1)
  scene = lower_collision_geometry(model, [(0, 1)])
  vertices, faces = collision_mesh_hull(model, 0)
  info = scene.mesh_hull_info[:9]
  assert len(vertices) == 100 and len(faces) > 64
  assert scene.mesh_hull.size == 3 * (len(vertices) + 2*len(faces))
  np.testing.assert_array_equal(scene.mesh_hull[:3*len(vertices)].reshape(-1,3),
                                vertices)
  np.testing.assert_array_equal(scene.mesh_hull[3*int(info[3]):].reshape(-1,3),
                                faces)


def test_model_sized_assets_still_fail_before_upload_on_explicit_budget():
  from test_compiled_collision_hull import _simplified_model
  from mujoco_metal.capacity import CapacityLimits, CapacityOverflow
  with pytest.raises(CapacityOverflow, match='memory|budget'):
    lower_collision_geometry(_simplified_model(-1), [(0, 1)],
                             limits=CapacityLimits(memory_budget_bytes=1024))


def _with_distance_sensors(model, geom_a, geom_b, tmp_path, *, body_a=False):
  source = tmp_path / 'query.xml'
  mujoco.mj_saveLastXML(str(source), model)
  spec = mujoco.MjSpec.from_file(str(source))
  for name in ((geom_b,) if body_a else (geom_a, geom_b)):
    spec.geoms[model.geom(name).id].name = name
  for kind in (mujoco.mjtSensor.mjSENS_GEOMDIST,
               mujoco.mjtSensor.mjSENS_GEOMNORMAL,
               mujoco.mjtSensor.mjSENS_GEOMFROMTO):
    spec.add_sensor(type=kind,objtype=(mujoco.mjtObj.mjOBJ_BODY if body_a else mujoco.mjtObj.mjOBJ_GEOM),objname=geom_a,
                    reftype=mujoco.mjtObj.mjOBJ_GEOM,refname=geom_b,cutoff=1.)
  return spec.compile()



@pytest.mark.gpu
@pytest.mark.skipif(__import__('os').getenv('MUJOCO_METAL_RUN_GPU') != '1',
                    reason='native geometry-query opt-in')
@pytest.mark.parametrize('fixture', ['mesh100', 'terrain9'])
def test_native_queries_use_complete_model_sized_masked_assets(fixture,tmp_path):
  import torch
  from mujoco_metal.model import load_model
  from mujoco_metal.sensors import SensorProgram
  if fixture == 'mesh100':
    from test_compiled_collision_hull import _simplified_model
    model = _simplified_model(-1)
    # Names survive XML serialization and are independent of candidate masks.
    xml = tmp_path / 'original.xml'
    mujoco.mj_saveLastXML(str(xml), model)
    spec = mujoco.MjSpec.from_string(xml.read_text())
    spec.geoms[0].name, spec.geoms[1].name = 'hgeom', 'sphere'
    model = _with_distance_sensors(spec.compile(), 'hgeom', 'sphere', tmp_path)
  else:
    model = _with_distance_sensors(_model(count=9), 'world', 'sphere', tmp_path, body_a=True)
  model.geom_contype[:] = 0
  model.geom_conaffinity[:] = 0
  descriptor = load_model(model)
  positions, quats, expected = [], [], []
  for world in range(2):
    data = mujoco.MjData(model)
    data.qpos[:] = model.qpos0.astype(np.float32)
    data.qpos[0 if fixture == 'mesh100' else 2] += world*.3
    mujoco.mj_forward(model,data)
    poses = descriptor.forward_kinematics(data.qpos)
    positions.append(poses['geom_pos']); quats.append(poses['geom_quat'])
    expected.append(data.sensordata.copy())
  program = SensorProgram(model,batch_size=2)
  tensor=lambda a:torch.tensor(np.asarray(a),dtype=torch.float32,device='mps')
  poses={'geom_pos':tensor(positions),'geom_quat':tensor(quats)}
  result=program.run_geomdist_device(poses,program._sp_hull,program._sp_hull_info)
  np.testing.assert_allclose(result.cpu().numpy(),expected,rtol=3e-5,atol=3e-5)
  saved=result.clone()
  poses['geom_pos'][0,:,0].add_(.17)
  result=program.run_geomdist_device(poses,program._sp_hull,program._sp_hull_info,
                                     world_mask=torch.tensor([False,True],device='mps'))
  torch.testing.assert_close(result,saved,rtol=0,atol=0)


def test_query_workspace_admission_counts_all_owned_tensor_payloads(tmp_path):
  from mujoco_metal.geometry_queries import lower_geometry_queries
  from mujoco_metal.capacity import CapacityLimits,CapacityOverflow
  from test_compiled_collision_hull import _simplified_model
  original=_simplified_model(-1)
  xml=tmp_path/'plain.xml';mujoco.mj_saveLastXML(str(xml),original)
  spec=mujoco.MjSpec.from_file(str(xml));spec.geoms[0].name='h';spec.geoms[1].name='s'
  model=_with_distance_sensors(spec.compile(),'h','s',tmp_path)
  desc=lower_geometry_queries(model,3)
  count=len(desc['pairs'])
  expected=(desc['common_bytes']+3*count*10*4+count*16+desc['spans'].nbytes+
            model.ngeom*4+6*4+3*model.ngeom*9*4)
  assert desc['bytes_total']==expected
  with pytest.raises(CapacityOverflow,match='workspace.*budget'):
    lower_geometry_queries(model,3,CapacityLimits(memory_budget_bytes=expected-1))
  assert lower_geometry_queries(model,3,CapacityLimits(memory_budget_bytes=expected))['bytes_total']==expected
