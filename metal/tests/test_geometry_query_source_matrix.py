# Copyright 2026 The MuJoCo Metal contributors
# Licensed under the Apache License, Version 2.0.
"""Pinned distance/witness queries are independent of physical pair filters."""
import os
import itertools
import mujoco
import numpy as np
import pytest
from test_common_ccd_rigid_native import _TYPE_XML


def _fixture(a,b,libccd):
  shapes=dict(_TYPE_XML);shapes[0]='<geom name="g" type="plane" size="2 2 .1"/>'
  flag='<flag nativeccd="disable"/>' if libccd else ''
  asset='<asset><mesh name="tet" vertex="-.15 -.15 -.15 .15 -.15 -.15 -.15 .15 -.15 -.15 -.15 .15"/></asset>'
  ga=shapes[a].replace('name="g"','name="a"');gb=shapes[b].replace('name="g"','name="b"')
  sensors=''.join(f'<{kind} geom1="{left}" geom2="{right}" cutoff="1"/>'
      for left,right in (('a','b'),('b','a')) for kind in ('distance','normal','fromto'))
  return mujoco.MjModel.from_xml_string(f'<mujoco><option gravity="0 0 0" ccd_iterations="100" ccd_tolerance="1e-6">{flag}</option>'
      '<default><geom contype="0" conaffinity="0"/></default>'+asset+
      f'<worldbody><body pos="0 0 0">{ga}</body><body name="moving" pos=".1 .03125 .0625">{gb}</body></worldbody>'
      f'<sensor>{sensors}</sensor></mujoco>')


@pytest.mark.parametrize('a,b',tuple(itertools.combinations_with_replacement((0,2,3,4,5,6,7),2)))
def test_every_query_shape_pair_compiles_independently_of_physical_masks(a,b):
  from mujoco_metal.sensors import lower_sensors
  model=_fixture(a,b,False)
  descriptor=lower_sensors(model)
  assert descriptor.nsensor==6
  assert np.all(model.geom_contype==0) and np.all(model.geom_conaffinity==0)


@pytest.mark.gpu
@pytest.mark.skipif(os.getenv('MUJOCO_METAL_RUN_GPU')!='1',reason='native query qualification')
@pytest.mark.parametrize('libccd',[False,True])
@pytest.mark.parametrize('a,b',tuple(itertools.combinations_with_replacement((0,2,3,4,5,6,7),2)))
def test_native_query_distance_and_actual_witness_matrix(a,b,libccd):
  import torch
  from mujoco_metal.sensors import SensorProgram
  model=_fixture(a,b,libccd)
  # Each query evaluates immutable model geometry at externally supplied poses;
  # the CPU oracle never feeds numerical results into the production kernels.
  pos=[];quat=[];mat=[];expected=[]
  data=mujoco.MjData(model)
  for displacement in (.0625,.3125,.75,2.):
    model.body_pos[model.body('moving').id,0]=displacement
    mujoco.mj_forward(model,data)
    pos.append(data.geom_xpos.copy());mat.append(data.geom_xmat.copy().reshape(model.ngeom,9))
    quaternions=np.empty((model.ngeom,4))
    for geom in range(model.ngeom):mujoco.mju_mat2Quat(quaternions[geom],data.geom_xmat[geom])
    quat.append(quaternions);expected.append(data.sensordata.copy())
  model.body_pos[model.body('moving').id,0]=.1
  program=SensorProgram(model,batch_size=4)
  tensor=lambda value:torch.tensor(np.asarray(value),dtype=torch.float32,device='mps')
  poses={'geom_quat':tensor(quat)}
  for name,values in (('geom_pos',pos),('geom_xmat',mat)):
    values=np.asarray(values,np.float64);hi=values.astype(np.float32)
    low=(values-hi.astype(np.float64)).astype(np.float32)
    tail=(values-hi.astype(np.float64)-low.astype(np.float64)).astype(np.float32)
    poses.update({name:tensor(hi),name+'_low':tensor(low),name+'_tail':tensor(tail)})
  actual=program.run_geomdist_device(poses,program._sp_hull,program._sp_hull_info)
  np.testing.assert_allclose(actual.cpu().numpy(),expected,atol=3e-5,rtol=3e-5)
  before=actual.clone()
  program.run_geomdist_device(poses,program._sp_hull,program._sp_hull_info,
      world_mask=torch.zeros(4,dtype=torch.bool,device='mps'))
  torch.testing.assert_close(actual,before,rtol=0,atol=0)
