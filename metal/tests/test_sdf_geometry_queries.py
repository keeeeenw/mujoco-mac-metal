# Copyright 2026 The MuJoCo Metal contributors
# Licensed under the Apache License, Version 2.0.
"""Actual distance minima and source witnesses over SDF scene operands."""
import os
import mujoco
import numpy as np
import pytest
from test_sdf_contact_013 import VBOX,KINDS
from test_bundled_sdf_rigid_oracle_019 import _plugin_sphere_model,_plugin_mesh_sdf_model


def _sensors(model,a,b,tmp_path):
  path=tmp_path/'scene.xml'
  mujoco.mj_saveLastXML(str(path),model)
  spec=mujoco.MjSpec.from_file(str(path))
  # sensor references use exact compiled IDs; names must survive recompilation.
  refs={}
  for g in (a,b):
    # Preserve existing names referenced by explicit collision pairs/plugins.
    if not spec.geoms[g].name: spec.geoms[g].name=f'g{g}'
    if int(model.geom_type[g])==1:
      body=int(model.geom_bodyid[g])
      if not spec.bodies[body].name: spec.bodies[body].name=f'b{body}'
      refs[g]=(mujoco.mjtObj.mjOBJ_BODY,spec.bodies[body].name)
    else:refs[g]=(mujoco.mjtObj.mjOBJ_GEOM,spec.geoms[g].name)
  for left,right in ((a,b),(b,a)):
    for kind in (mujoco.mjtSensor.mjSENS_GEOMDIST,mujoco.mjtSensor.mjSENS_GEOMNORMAL,
                 mujoco.mjtSensor.mjSENS_GEOMFROMTO):
      spec.add_sensor(type=kind,objtype=refs[left][0],objname=refs[left][1],
          reftype=refs[right][0],refname=refs[right][1],cutoff=1.)
  return spec.compile()


def _model(kind,seed_count=8):
  shapes=dict(KINDS)
  shapes.update(mesh='<geom name="o" type="mesh" mesh="b"/>',
                plane='<geom name="o" type="plane" size="1 1 .1"/>',
                hfield='<geom name="o" type="hfield" hfield="hf"/>')
  asset=f'<asset><mesh name="b" vertex="{VBOX}"/><hfield name="hf" nrow="2" ncol="2" size="1 1 .1 .1" elevation="0 0 0 0"/></asset>'
  return mujoco.MjModel.from_xml_string(f'<mujoco>{asset}<option gravity="0 0 0" sdf_initpoints="{seed_count}"/>'
      '<worldbody><geom name="sdf" type="sdf" mesh="b"/>'
      f'<body pos=".015625 .015625 .0625">{shapes[kind]}</body></worldbody></mujoco>')


@pytest.mark.parametrize('kind',tuple(KINDS)+('mesh','plane','hfield'))
def test_sdf_sensor_descriptors_admit_pinned_dispatches(kind,tmp_path):
  from mujoco_metal.sensors import lower_sensors
  model=_sensors(_model(kind),0,1,tmp_path)
  assert lower_sensors(model).nsensor==6


@pytest.mark.gpu
@pytest.mark.skipif(os.getenv('MUJOCO_METAL_RUN_GPU')!='1',reason='native SDF query opt-in')
@pytest.mark.parametrize('fixture',tuple(KINDS)+('mesh','plane','hfield','bolt','bowl','gear','nut','torus','plugin_mesh'))
def test_native_sdf_source_minima_actual_witnesses_and_masked_reuse(fixture,tmp_path):
  import torch
  from mujoco_metal.sensors import SensorProgram
  if fixture=='plugin_mesh':model=_plugin_mesh_sdf_model()
  elif fixture in ('bolt','bowl','gear','nut','torus'):model=_plugin_sphere_model(fixture)
  else:model=_model(fixture)
  model=_sensors(model,0,1,tmp_path)
  model.geom_contype[:]=0;model.geom_conaffinity[:]=0
  poses={};expected=[];data=mujoco.MjData(model);samples={name:[] for name in ('geom_pos','geom_xmat','geom_quat')}
  for world in range(2):
    data.qpos[:]=model.qpos0
    if model.nq:data.qpos[0]+=.01*world
    else:model.body_pos[1,0]+=.015625*world
    mujoco.mj_forward(model,data);expected.append(data.sensordata.copy())
    samples['geom_pos'].append(data.geom_xpos.copy())
    samples['geom_xmat'].append(data.geom_xmat.copy().reshape(model.ngeom,9))
    quats=np.empty((model.ngeom,4))
    for g in range(model.ngeom):mujoco.mju_mat2Quat(quats[g],data.geom_xmat[g])
    samples['geom_quat'].append(quats)
  tensor=lambda x:torch.tensor(np.asarray(x),dtype=torch.float32,device='mps')
  for name,value in samples.items():
    value=np.asarray(value);hi=value.astype(np.float32);low=(value-hi).astype(np.float32)
    tail=(value-hi-low).astype(np.float32)
    poses.update({name:tensor(hi),name+'_low':tensor(low),name+'_tail':tensor(tail)})
  program=SensorProgram(model,batch_size=2)
  actual=program.run_geomdist_device(poses,program._sp_hull,program._sp_hull_info)
  np.testing.assert_allclose(actual.cpu().numpy(),expected,atol=3e-5,rtol=3e-5)
  saved=actual.clone();poses['geom_pos'].add_(.125)
  program.run_geomdist_device(poses,program._sp_hull,program._sp_hull_info,
      world_mask=torch.zeros(2,dtype=torch.bool,device='mps'))
  torch.testing.assert_close(actual,saved,rtol=0,atol=0)
