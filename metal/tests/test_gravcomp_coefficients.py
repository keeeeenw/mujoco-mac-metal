"""Source-scaled standalone compensation, disabling and borrowed-row history."""
import os
import mujoco
import numpy as np
import pytest


def fixture(mode):
  bodies = ''.join(f'''<body pos="{i*2} 0 0" gravcomp="{c}">
    <joint type="hinge" axis="0 1 0" damping=".4" stiffness="2"/>
    <inertial pos=".5 0 0" mass="1" diaginertia=".1 .1 .1"/>
    <geom type="sphere" size=".1" pos=".5 0 0"/></body>'''
    for i,c in enumerate((0,.35,1,1.4)))
  m=mujoco.MjModel.from_xml_string(f'<mujoco><option><flag contact="disable"/></option><worldbody>{bodies}</worldbody></mujoco>')
  if mode=='gravity':m.opt.disableflags |= int(mujoco.mjtDisableBit.mjDSBL_GRAVITY)
  elif mode=='passive':m.opt.disableflags |= int(mujoco.mjtDisableBit.mjDSBL_SPRING)|int(mujoco.mjtDisableBit.mjDSBL_DAMPER)
  qp=np.array([[.7,.7,.7,.7],[-.3,-.3,-.3,-.3]],np.float32)
  qv=np.array([[1.2]*4,[-.4]*4],np.float32)
  return m,qp,qv


def oracle(m,qp,qv):
  gc=[]; passive=[]
  for q,v in zip(qp,qv):
    d=mujoco.MjData(m);d.qpos[:]=q;d.qvel[:]=v;mujoco.mj_forward(m,d)
    gc.append(d.qfrc_gravcomp.copy());passive.append(d.qfrc_passive.copy())
  return np.stack(gc),np.stack(passive)


@pytest.mark.parametrize('mode',['normal','gravity','passive'])
def test_source_compensation_uses_coefficients_and_disable_semantics(mode):
  m,q,v=fixture(mode);gc,p=oracle(m,q,v)
  if mode=='normal':
    assert np.all(np.abs(gc[:,2])>3)
    np.testing.assert_array_equal(gc[:,0],0)
    np.testing.assert_allclose(gc[:,1],gc[:,2]*.35,atol=1e-12)
    np.testing.assert_allclose(gc[:,3],gc[:,2]*1.4,atol=1e-12)
  else:np.testing.assert_array_equal(gc,0)
  if mode=='passive':np.testing.assert_array_equal(p,0)


@pytest.mark.gpu
@pytest.mark.skipif(os.environ.get('MUJOCO_METAL_RUN_GPU')!='1',reason='native opt-in')
@pytest.mark.parametrize('mode',['normal','gravity','passive'])
@pytest.mark.parametrize('masked',[False,True])
def test_native_compensation_coefficient_and_passive_interleave(mode,masked):
  import torch
  from mujoco_metal.passive import MetalPassiveForces
  m,q,v=fixture(mode);stage=MetalPassiveForces(m,batch_size=2)
  qt=torch.tensor(q,device='mps');vt=torch.tensor(v,device='mps')
  seed=stage.gravcomp_device(qt).clone()
  stage.run_device(qt,vt)
  qt.add_(.1);vt.mul_(-.5)
  gc,pa=oracle(m,qt.cpu().numpy(),vt.cpu().numpy())
  mask=torch.tensor([1,0],device='mps',dtype=torch.int32) if masked else None
  if masked:qt[1]=float('nan');vt[1]=float('nan')
  actual=stage.gravcomp_device(qt,world_mask=mask).clone()
  passive=stage.run_device(qt,vt,world_mask=mask).clone()
  repeated=stage.gravcomp_device(qt,world_mask=mask).clone()
  active=slice(0,1) if masked else slice(None)
  for t in (actual,repeated):
    np.testing.assert_allclose(t.cpu().numpy()[active],gc[active],atol=3e-5,rtol=2e-5)
    if masked:np.testing.assert_array_equal(t.cpu().numpy()[1],seed.cpu().numpy()[1])
  np.testing.assert_allclose(passive.cpu().numpy()[active],pa[active],atol=3e-5,rtol=2e-5)
