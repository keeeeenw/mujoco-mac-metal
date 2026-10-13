# Copyright 2026 The MuJoCo Metal contributors
# Licensed under the Apache License, Version 2.0.
"""Independent native CRBA/RNE precision and retained POS/VEL ownership."""
import os
from pathlib import Path
import mujoco
import numpy as np
import pytest
from mujoco_metal.model import load_model
from mujoco_metal.source_rigid import _source_constants, source_rigid_position_words

XML='''<mujoco><option gravity=".4 -.7 -9.81"/>
<worldbody><body name="carrier" mocap="true" pos=".1 .2 1.2" quat=".96 .1 -.12 .05">
 <body pos=".02 -.03 -.07"><joint type="ball" pos=".01 .03 -.04" armature=".013"/>
  <inertial pos=".08 -.09 -.17" quat=".86 .13 -.23 .32" mass=".53" diaginertia=".017 .022 .006"/>
  <body pos=".2 .07 .13" quat=".9 .12 .24 -.17">
   <joint axis=".2 .7 -.3" pos=".03 -.05 .02" armature=".017"/>
   <inertial pos=".07 .04 .02" quat=".8 .31 .17 -.22" mass=".29" diaginertia=".008 .01 .013"/>
   <body pos=".13 -.09 .11"><joint type="slide" axis=".3 -.4 .8" armature=".019"/>
    <inertial pos=".04 -.03 .08" quat=".9 -.07 .24 .15" mass=".21" diaginertia=".004 .005 .007"/>
   </body>
  </body>
 </body>
</body><body pos="1 .4 .6"><freejoint/><geom type="box" size=".07 .13 .11" mass=".43" contype="0" conaffinity="0"/></body>
</worldbody></mujoco>'''


def test_compiled_source_constants_reconstruct_the_actual_model_fields():
  m=load_model(mujoco.MjModel.from_xml_string(XML))
  meta,packed=_source_constants(m,m.dof_armature)
  actual=packed.reshape(-1,3).astype(float).sum(axis=1)
  expected=np.concatenate([m.body_mass,m.body_subtreemass,m.body_inertia.ravel(),m.dof_armature,m.dof_M0,m.gravity])
  np.testing.assert_array_equal(actual,expected)
  assert meta.size==6*m.nbody+3*m.nv+2*m.njnt
  assert source_rigid_position_words(m)==3*(23*m.nbody+6*m.nv)


def test_source_rigid_runtime_and_cached_capacity_are_charged():
  from mujoco_metal.capacity import _runtime_buffer_sizes
  m=mujoco.MjModel.from_xml_string(XML);sizes=dict(_runtime_buffer_sizes(m,3))
  assert sizes['smooth.source_rigid.position']==3*source_rigid_position_words(m)
  assert sizes['smooth.source_rigid.velocity']==9*(18*m.nbody+6*m.nv)
  assert sizes['smooth.position_cache.source_rigid']==3*source_rigid_position_words(m)


def _oracles(m,q,v,mp,mq):
  matrices=[];bias=[]
  for qi,vi,pi,ri in zip(q,v,mp,mq):
    d=mujoco.MjData(m);d.qpos[:]=qi;d.qvel[:]=vi;d.mocap_pos[:]=pi;d.mocap_quat[:]=ri
    mujoco.mj_forward(m,d);M=np.zeros((m.nv,m.nv));mujoco.mj_fullM(m,d,M)
    matrices.append(M);bias.append(d.qfrc_bias.copy())
  return np.asarray(matrices),np.asarray(bias)


@pytest.mark.gpu
@pytest.mark.skipif(os.getenv('MUJOCO_METAL_RUN_GPU')!='1',reason='opt-in GPU')
@pytest.mark.parametrize('storage',['dense','block_sparse'])
@pytest.mark.parametrize('suspension',[False,True])
def test_source_mass_bias_cache_mask_and_real_mutation_gpu(storage,suspension,monkeypatch):
  import torch
  from mujoco_metal.smooth_metal import MetalSmoothDynamics
  m=(mujoco.MjModel.from_xml_path(str(Path(__file__).parents[1]/'examples/suspension_platform.xml'))
     if suspension else mujoco.MjModel.from_xml_string(XML))
  batch=3;s=MetalSmoothDynamics(load_model(m),batch_size=batch,mass_storage=storage)
  # Retained sleeping-child force imports must start from owned zero state.
  assert torch.count_nonzero(s._workspace['body_force'])==0
  assert torch.count_nonzero(s._workspace['cacc'])==0
  rng=np.random.default_rng(914)
  q=np.tile(m.qpos0,(batch,1));v=rng.uniform(-1.4,1.1,(batch,m.nv))
  for j,typ in enumerate(m.jnt_type):
    a=m.jnt_qposadr[j]
    if int(typ) in (0,1):
      qa=a+(3 if int(typ)==0 else 0)
      for w in range(batch):
        quat=rng.normal(size=4);quat/=np.linalg.norm(quat);q[w,qa:qa+4]=quat
    else:q[:,a]+=[.13,-.08,.24]
  q=q.astype(np.float32);v=v.astype(np.float32)
  mp=np.tile(np.asarray(m.body_pos)[np.asarray(m.body_mocapid)>=0],(batch,1,1)).astype(np.float32)
  mq=np.tile(np.array([.5,.5,.5,.5],np.float32),(batch,m.nmocap,1))
  referenceM,referenceB=_oracles(m,q,v,mp,mq)
  td=lambda a:torch.as_tensor(a,device='mps').contiguous()
  qt,vt,pt,rt=map(td,(q,v,mp,mq))
  # The runtime implementation must remain native, with no source oracle calls.
  with monkeypatch.context() as guard:
    def forbidden(*args,**kwargs):raise AssertionError('CPU physics in native smooth stage')
    guard.setattr(mujoco,'mj_forward',forbidden);guard.setattr(mujoco,'mj_crb',forbidden);guard.setattr(mujoco,'mj_rne',forbidden)
    d=s.run_device(qt,vt,pt,rt)
    if storage=='dense':M=d['mass_matrix'].cpu().numpy().astype(float)+d['mass_matrix_low'].cpu().numpy().astype(float)
    else:
      M=np.zeros((batch,m.nv,m.nv));blocks=d['mass_blocks'].cpu().numpy().astype(float)+d['mass_blocks_low'].cpu().numpy().astype(float)
      layout=s.mass_block_layout
      for c,count in enumerate(layout['component_dofnum']):
        start=layout['component_dof_offsets'][c];ids=layout['component_dof_ids'][start:start+count];off=layout['component_mass_offsets'][c]
        for w in range(batch):M[w][np.ix_(ids,ids)]=blocks[w,off:off+count*count].reshape(count,count)
    B=d['qfrc_bias'].cpu().numpy().astype(float)+d['qfrc_bias_low'].cpu().numpy().astype(float)
    np.testing.assert_allclose(M,referenceM,rtol=3e-13,atol=3e-14)
    np.testing.assert_allclose(B,referenceB,rtol=3e-12,atol=3e-13)
    # The explicit host query must use the same operator and mocap frame,
    # without replacing the persistent stepping or retained-position packet.
    persistent=d['source_rigid'].clone()
    public=s.run(q,v,mp[0],mq[0])
    np.testing.assert_allclose(public['mass_matrix'].cpu().numpy().astype(float)+public['mass_matrix_low'].cpu().numpy().astype(float),referenceM,rtol=3e-13,atol=3e-14)
    np.testing.assert_allclose(public['qfrc_bias'].cpu().numpy().astype(float)+public['qfrc_bias_low'].cpu().numpy().astype(float),referenceB,rtol=3e-12,atol=3e-13)
    torch.testing.assert_close(s._source_rigid.position_packet,persistent,rtol=0,atol=0)
    context=s.position_context(d)
    expected=d['source_rigid'].clone()
    # Actual live scratch replacement must not change the owned POS packet.
    s._source_rigid.position_packet.fill_(.125)
    torch.testing.assert_close(context['source_rigid'],expected,rtol=0,atol=0)
    fresh=s.run_velocity_device(context,vt)
    np.testing.assert_allclose(fresh['qfrc_bias'].cpu().numpy().astype(float)+fresh['qfrc_bias_low'].cpu().numpy().astype(float),referenceB,rtol=3e-12,atol=3e-13)
    s.run_device(qt,vt,pt,rt)
    old=s._source_rigid.position_packet.clone();beforeM=(s._workspace['mass'] if storage=='dense' else s._workspace['mass_blocks']).clone();beforeB=s._workspace['bias'].clone()
    vt[0].add_(.07);qt[0,0].add_(.003)
    mask=td(np.array([1,0,0],np.int32));s.run_device(qt,vt,pt,rt,world_mask=mask)
    torch.testing.assert_close(s._source_rigid.position_packet[1:],old[1:],rtol=0,atol=0)
    outM=s._workspace['mass'] if storage=='dense' else s._workspace['mass_blocks']
    torch.testing.assert_close(outM.reshape(batch,-1)[1:],beforeM.reshape(batch,-1)[1:],rtol=0,atol=0)
    torch.testing.assert_close(s._workspace['bias'].reshape(batch,-1)[1:],beforeB.reshape(batch,-1)[1:],rtol=0,atol=0)
    assert not torch.equal(s._source_rigid.position_packet[0],old[0])
