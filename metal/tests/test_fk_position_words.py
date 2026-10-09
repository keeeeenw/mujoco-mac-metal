# Copyright 2026 The MuJoCo Metal contributors
# Licensed under the Apache License, Version 2.0.
"""Source words and admission atomicity of public native forward kinematics."""
import math
import os
import mujoco
import numpy as np
import pytest

native=pytest.mark.skipif(os.getenv('MUJOCO_METAL_RUN_GPU')!='1',reason='native position qualification')

@pytest.mark.gpu
@native
def test_source_word_trigonometry_matches_independent_libm_across_float_domain():
  import torch
  from mujoco_metal.metal_kinematics import _kinematics_shader_source
  rng=np.random.default_rng(1573)
  values=np.r_[0.,-0.,np.nextafter(math.pi/2,0.),math.pi/2,np.nextafter(math.pi/2,math.inf),
      rng.uniform(-20,20,64),np.ldexp(rng.uniform(.5,1.,128),rng.integers(-80,128,128))]
  values=np.r_[values,-values]
  hi=values.astype(np.float32);lo=(values-hi.astype(float)).astype(np.float32)
  tail=(values-hi.astype(float)-lo.astype(float)).astype(np.float32)
  inp=torch.tensor(np.column_stack((hi,lo,tail)),device='mps')
  out=torch.empty((len(values),6),device='mps')
  src=_kinematics_shader_source()+'''
  kernel void evaluate_source_trig(device const float* a [[buffer(0)]],device float* out [[buffer(1)]],uint i [[thread_position_in_grid]]) {
    KinDD x=KinDD{a[3*i],a[3*i+1],a[3*i+2]};
    KinDD s=kin_sincos(x,true),c=kin_sincos(x,false);
    out[6*i]=s.hi;out[6*i+1]=s.lo;out[6*i+2]=s.tail;
    out[6*i+3]=c.hi;out[6*i+4]=c.lo;out[6*i+5]=c.tail;
  }'''
  lib=torch.mps.compile_shader(src)
  lib.evaluate_source_trig(inp.reshape(-1),out.reshape(-1),threads=(len(values),))
  data=out.cpu().numpy().astype(float)
  actual=np.c_[data[:,:3].sum(axis=1),data[:,3:].sum(axis=1)]
  expected=np.array([[math.sin(x),math.cos(x)] for x in values])
  np.testing.assert_allclose(actual,expected,rtol=0,atol=7e-16)


@pytest.mark.gpu
@native
def test_host_fk_preserves_source_words_without_renormalizing_high_projection():
  from mujoco_metal.metal_kinematics import MetalKinematics
  from mujoco_metal.model import load_model
  from test_fk_source_rounding import _CHAIN_XML
  m=mujoco.MjModel.from_xml_string(_CHAIN_XML)
  q=np.tile(m.qpos0,(2,1));q[:,0]=[.123456789012345,-.234567890123456]
  q[:,3:7]=[[.812345678901234,.223456789012345,-.334567890123456,.445678901234567],
             [-.712345678901234,.123456789012345,.234567890123456,-.545678901234567]]
  q[:,7:11]=q[:,3:7]*1.23456789012345;q[:,11]=.234567890123456
  refs=[]
  for row in q:
    d=mujoco.MjData(m);d.qpos[:]=row;mujoco.mj_kinematics(m,d);refs.append(d)
  fk=MetalKinematics(load_model(m),batch_size=2);out=fk.run(q)
  for name,field in [('geom_pos','geom_xpos'),('geom_xmat','geom_xmat'),('body_quat','xquat')]:
    represented=sum(out[name+suffix].cpu().numpy().astype(float) for suffix in ('','_low','_tail'))
    np.testing.assert_array_equal(represented,np.array([getattr(d,field) for d in refs]))


@pytest.mark.gpu
@native
def test_invalid_fk_metadata_preserves_owned_residual_input_and_published_poses():
  import torch
  from mujoco_metal.metal_kinematics import MetalKinematics
  from mujoco_metal.model import load_model
  m=mujoco.MjModel.from_xml_string('<mujoco><worldbody><body><joint/><geom type="sphere" size=".1"/></body></worldbody></mujoco>')
  fk=MetalKinematics(load_model(m),batch_size=2)
  q=np.full((2,1),.123456789012345);fk.run(q)
  owned=fk._workspace['outputs'];before={name:v.clone() for name,v in owned.items() if isinstance(v,torch.Tensor)}
  high=torch.tensor(q,dtype=torch.float32,device='mps')
  for kwargs in ({'world_mask':torch.zeros(1,dtype=torch.int32,device='mps')},
                 {'tree_awake':torch.zeros((2,3),dtype=torch.int32,device='mps')}):
    with pytest.raises(ValueError):fk.run_device(high,**kwargs)
    for name,v in before.items():assert torch.equal(owned[name],v),name

