# Copyright 2026 The MuJoCo Metal contributors
# Licensed under the Apache License, Version 2.0.
"""Position words are exercised through native integration, FK and lifecycle."""
import math
import os
from pathlib import Path
import mujoco
import numpy as np
import pytest
from test_integration import _MIXED_XML
from mujoco_metal.retained_state import pack_residuals

native=pytest.mark.skipif(os.getenv('MUJOCO_METAL_RUN_GPU')!='1',reason='native position qualification')




@pytest.mark.gpu
@native
def test_native_mixed_joint_integration_retains_position_words_for_32_steps():
  import torch
  from mujoco_metal.device_state import DeviceState
  from mujoco_metal.integration import MetalEulerIntegration
  from mujoco_metal.model import load_model
  from mujoco_metal.stepping import validate_stepping_profile
  from mujoco_metal.metal_kinematics import MetalKinematics
  model=mujoco.MjModel.from_xml_string(_MIXED_XML)
  model.opt.disableflags|=int(mujoco.mjtDisableBit.mjDSBL_CONTACT)
  descriptor=load_model(model);batch=2;dt=.001234567890123
  q=np.tile(model.qpos0,(batch,1));v=np.random.default_rng(184).normal(0,.31,(batch,model.nv))
  q[:,0]=[.123456789012345,-.234567890123456]
  for row in range(batch):mujoco.mj_integratePos(model,q[row],v[row],.172345678901234)
  acc=np.random.default_rng(179).normal(0,.2,(batch,model.nv)).astype(np.float32)
  state=DeviceState(model,validate_stepping_profile(model),batch,qpos=q,qvel=v)
  integrator=MetalEulerIntegration(descriptor,batch,dt)
  fk=MetalKinematics(descriptor,batch_size=batch)
  agpu=torch.tensor(acc,device='mps')
  source_q=q.copy();source_v=v.copy()
  for step in range(32):
    precision=state._precision.synchronize(state._qpos,state._qvel)
    nq=descriptor.nq;nv=descriptor.nv
    qp,vp,t,st=integrator.run_device(state._qpos,state._qvel,agpu,state._time,state._status,
                                   precision_words=precision)
    assert st.cpu().tolist()==[0,0]
    state._qpos=qp.clone();state._qvel=vp.clone();state._time=t.clone()
    state._precision.commit(state._qpos,state._qvel,integrator._output_words,st==0)
    for row in range(batch):
      # Actual source mju_addToScl contractions, followed by mj_integratePos.
      mujoco.mju_addToScl(source_v[row],acc[row].astype(float),dt)
      mujoco.mj_integratePos(model,source_q[row],source_v[row],dt)
    snap=state.snapshot();words=snap.precision_words.astype(float)
    represented=snap.qpos.astype(float)+words[:,:nq]+words[:,nq:2*nq]
    np.testing.assert_allclose(represented,source_q,rtol=0,atol=3e-13)
    assert np.any(words[:,:2*nq])
  poses=fk.run_device(state._qpos,qpos_words=state._precision.words[:,:2*descriptor.nq])
  actual=poses['geom_pos'].cpu().numpy().astype(float)+poses['geom_pos_low'].cpu().numpy()+poses['geom_pos_tail'].cpu().numpy()
  expected=[]
  for row in range(batch):
    d=mujoco.MjData(model);d.qpos[:]=source_q[row];mujoco.mj_kinematics(model,d);expected.append(d.geom_xpos.copy())
  np.testing.assert_allclose(actual,np.array(expected),rtol=0,atol=6e-13)


@pytest.mark.gpu
@native
def test_actual_mps_state_writes_lifecycle_and_exact_owned_storage():
  import torch
  from mujoco_metal.device_state import DeviceState
  from mujoco_metal.stepping import validate_stepping_profile
  from mujoco_metal.integration import MetalEulerIntegration,integration_workspace_sizes
  from mujoco_metal.model import load_model
  from mujoco_metal.capacity import _runtime_buffer_sizes
  m=mujoco.MjModel.from_xml_string('<mujoco><option><flag contact="disable"/></option><worldbody><body><joint type="hinge"/><geom type="sphere" size=".1"/></body></worldbody></mujoco>')
  q=np.array([[.123456789012345],[.234567890123456]]);v=-q*.2718281828459
  s=DeviceState(m,validate_stepping_profile(m),2,qpos=q,qvel=v)
  before=s.snapshot();version=s._qpos._version
  lib=torch.mps.compile_shader('#include <metal_stdlib>\nusing namespace metal;\nkernel void change_state(device float* x [[buffer(0)]],uint i [[thread_position_in_grid]]){if(i==1)x[i]+=.25f;}')
  lib.change_state(s._qpos.reshape(-1),threads=(2,))
  assert s._qpos._version==version
  s._precision.synchronize(s._qpos,s._qvel)
  np.testing.assert_array_equal(s._precision.words.cpu().numpy()[0],before.precision_words[0])
  np.testing.assert_array_equal(s._precision.words.cpu().numpy()[1,:2],0)
  s.restore(before);s.reset([1],qpos=[.987654321098765],qvel=[-.123456789012345])
  np.testing.assert_array_equal(s.snapshot().precision_words[0],before.precision_words[0])
  s.restore(before,[1]);np.testing.assert_array_equal(s.snapshot().precision_words,before.precision_words)
  s.copy_environment(0,1);np.testing.assert_array_equal(s.snapshot().precision_words[1],before.precision_words[0])
  s._qpos=s._qpos.clone();s._precision.synchronize(s._qpos,s._qvel)
  np.testing.assert_array_equal(s._precision.words.cpu().numpy()[:,:2],0)
  s.restore(before);s.reset_mask(torch.tensor([False,True],device='mps'))
  np.testing.assert_array_equal(s.snapshot().precision_words[0],before.precision_words[0]);np.testing.assert_array_equal(s.snapshot().precision_words[1],0)
  s.restore(before);s.validate_snapshot(before);np.testing.assert_array_equal(s.snapshot().precision_words,before.precision_words)
  
  program=MetalEulerIntegration(load_model(m),2,.001234567890123)
  seen=set();words=0
  for name,value in vars(program).items():
   for t in value if isinstance(value,tuple) else (value,):
    if isinstance(t,torch.Tensor):
     ptr=t.untyped_storage().data_ptr()
     if ptr not in seen:words+=t.untyped_storage().nbytes()//4;seen.add(ptr)
  expected=sum(integration_workspace_sizes(m.nq,m.nv,m.njnt,m.nbody,2).values())
  assert words==expected,(words,expected)
  parts=dict(_runtime_buffer_sizes(m,2))
  assert parts['state.precision.words']==s._precision.words.numel()
  assert parts['state.precision.qpos_shadow']==s._precision.qpos_shadow.numel()
  assert parts['state.precision.qvel_shadow']==s._precision.qvel_shadow.numel()
  


@pytest.mark.gpu
@native
@pytest.mark.parametrize('sleep_filter',[False,True])
def test_failed_integration_preserves_every_high_and_residual_word(sleep_filter):
  import torch
  from mujoco_metal.integration import MetalEulerIntegration
  from mujoco_metal.model import load_model
  m=mujoco.MjModel.from_xml_string('<mujoco><worldbody><body><joint/><geom type="sphere" size=".1"/></body></worldbody></mujoco>')
  batch=4;dt=.001234567890123
  q=np.full((batch,1),.123456789012345);v=np.full((batch,1),.234567890123456)
  words=pack_residuals(q,v)
  def device(a,dtype=torch.float32):return torch.tensor(a,dtype=dtype,device='mps')
  qgpu=device(q);vgpu=device(v);wgpu=device(words);a=device(np.ones((batch,1)))
  low=device(np.zeros((batch,1)));low[1,0]=float('nan')
  incoming=device([0,0,7,0],torch.int32)
  # A nonfinite time supplies a separate integration failure. Every failure
  # retains input qvel even when asleep output would ordinarily be zero.
  time=device([0,0,0,float('nan')]);stage=MetalEulerIntegration(load_model(m),batch,dt)
  awake=None
  if sleep_filter:
    awake={'body_ids':device(np.zeros((batch,m.nbody)),torch.int32),
           'dof_ids':device(np.zeros((batch,m.nv)),torch.int32),
           'counts':device(np.zeros((batch,3)),torch.int32)}
  out=stage.run_device(qgpu,vgpu,a,time,incoming,precision_words=wgpu,qacc_low=low,awake_lists=awake)
  assert out[3].cpu().tolist()==[0,10,7,10]
  for actual,original in ((out[0],qgpu),(out[1],vgpu),(out[2],time),(stage._output_words,wgpu)):
    np.testing.assert_array_equal(actual.cpu().numpy()[1:],original.cpu().numpy()[1:])
  if sleep_filter:
    np.testing.assert_array_equal(out[1].cpu().numpy()[0],0)
    np.testing.assert_array_equal(stage._output_words.cpu().numpy()[0,2:],0)
  else:assert np.any(stage._output_words.cpu().numpy()[0]!=words[0])


@pytest.mark.gpu
@native
def test_rk4_zero_dof_and_exact_owned_precision_backings():
  import torch
  from mujoco_metal.runge_kutta import MetalRungeKutta
  from mujoco_metal.integration import integration_workspace_sizes
  from mujoco_metal.model import load_model
  from mujoco_metal.capacity import _runtime_buffer_sizes
  for xml in ('<mujoco><option integrator="RK4"/><worldbody/></mujoco>',
              '<mujoco><option integrator="RK4"/><worldbody><body><joint/><geom type="sphere" size=".1"/></body></worldbody></mujoco>'):
    m=mujoco.MjModel.from_xml_string(xml);d=load_model(m);b=2
    rk=MetalRungeKutta(d,b,.001234567890123)
    q=torch.tensor(np.tile(m.qpos0,(b,1)),dtype=torch.float32,device='mps')
    v=torch.zeros((b,m.nv),device='mps');t=torch.tensor([1.,2.],device='mps')
    status=torch.tensor([0,7],dtype=torch.int32,device='mps')
    def acceleration(q,v,act):return torch.ones_like(v),None,torch.zeros_like(status)
    out=rk.run_device(q,v,None,t,status,acceleration)
    assert out[-1].cpu().tolist()==[0,7]
    np.testing.assert_array_equal(out[0].cpu().numpy()[1],q.cpu().numpy()[1])
    np.testing.assert_array_equal(out[1].cpu().numpy()[1],v.cpu().numpy()[1])
    assert out[-2].cpu().numpy()[1]==2.
    seen=set();actual=0
    for value in vars(rk).values():
      for x in value if isinstance(value,tuple) else (value,):
        if isinstance(x,torch.Tensor):
          ptr=x.untyped_storage().data_ptr()
          if ptr not in seen:actual+=x.untyped_storage().nbytes()//4;seen.add(ptr)
    parts=dict(_runtime_buffer_sizes(m,b))
    expected=sum(count for name,count in parts.items() if name.startswith('rk4.') and not name.startswith('rk4.integration.'))
    assert actual==expected,(actual,expected)




