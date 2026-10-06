"""Joint passive accumulation and selected-world standalone projection."""
import os
import mujoco
import numpy as np
import pytest
from mujoco_metal.passive import PassiveForceModel

XML='''<mujoco><option gravity="0 0 -9.81"><flag contact="disable"/></option>
<worldbody><body gravcomp="1"><joint type="hinge" axis="0 1 0"
 stiffness="2" springref=".2" damping=".4"/>
 <inertial pos=".5 0 0" mass="1" diaginertia=".1 .1 .1"/>
 <geom pos=".5 0 0" type="sphere" size=".1"/></body></worldbody></mujoco>'''
def oracle(model,qp,qv,wrench):
  passive=[]; gravcomp=[]
  for q,v,w in zip(qp,qv,wrench):
    d=mujoco.MjData(model); d.qpos[:]=q; d.qvel[:]=v
    mujoco.mj_forward(model,d)
    f=d.qfrc_passive.copy()
    for body in range(1,model.nbody):
      mujoco.mj_applyFT(model,d,w[body,:3],w[body,3:],d.xipos[body],body,f)
    passive.append(f); gravcomp.append(d.qfrc_gravcomp.copy())
  return np.stack(passive),np.stack(gravcomp)
def fixture(disabled=False):
  m=mujoco.MjModel.from_xml_string(XML)
  if disabled: m.opt.disableflags |= int(mujoco.mjtDisableBit.mjDSBL_GRAVITY)
  qp=np.array([[.7],[-.3]],np.float32);qv=np.array([[1.2],[-.4]],np.float32)
  w=np.zeros((2,m.nbody,6),np.float32);w[:,1]=[.3,-.2,.4,.1,.7,-.3]
  return m,qp,qv,w
@pytest.mark.parametrize('disabled',[False,True])
def test_pinned_joint_and_projection_terms_are_nonzero(disabled):
  m,qp,qv,w=fixture(disabled); f,g=oracle(m,qp,qv,w)
  joint=PassiveForceModel(m).force(qp,qv)
  assert np.all(np.abs(joint)> .1)
  assert np.all(np.abs(f-joint)> .1)
  if disabled: np.testing.assert_array_equal(g,0)
  else: assert np.all(np.abs(g)> 1)
@pytest.mark.gpu
@pytest.mark.skipif(os.environ.get('MUJOCO_METAL_RUN_GPU')!='1',reason='native opt-in')
@pytest.mark.parametrize('disabled',[False,True])
@pytest.mark.parametrize('masked',[False,True])
def test_native_projection_modes_preserve_joint_force_and_unselected_world(disabled,masked):
  import torch
  from mujoco_metal.passive import MetalPassiveForces
  m,qp,qv,w=fixture(disabled); expected,gexpected=oracle(m,qp,qv,w)
  stage=MetalPassiveForces(m,batch_size=2)
  q=torch.tensor(qp,device='mps');v=torch.tensor(qv,device='mps');wt=torch.tensor(w,device='mps')
  # Seed real forces and capture borrowed history before selected-world reuse.
  seed=stage.run_device(q,v,wt).clone()
  gs=stage.gravcomp_device(q).clone()
  q.add_(.1);v.mul_(-.5)
  expected,gexpected=oracle(m,q.cpu().numpy(),v.cpu().numpy(),w)
  mask=torch.tensor([1,0],dtype=torch.int32,device='mps') if masked else None
  if masked:
    q[1]=float('nan');v[1]=float('nan');wt[1]=float('nan')
  actual=stage.run_device(q,v,wt,world_mask=mask).clone()
  # Interleave standalone and passive users of the same projection entrypoint.
  ga=stage.gravcomp_device(q,world_mask=mask).clone()
  again=stage.run_device(q,v,wt,world_mask=mask).clone()
  active=slice(0,1) if masked else slice(None)
  for value in (actual,again):
    np.testing.assert_allclose(value.cpu().numpy()[active],expected[active],atol=3e-5,rtol=2e-5)
  np.testing.assert_allclose(ga.cpu().numpy()[active],gexpected[active],atol=3e-5,rtol=2e-5)
  if masked:
    np.testing.assert_array_equal(actual.cpu().numpy()[1],seed.cpu().numpy()[1])
    np.testing.assert_array_equal(again.cpu().numpy()[1],seed.cpu().numpy()[1])
    np.testing.assert_array_equal(ga.cpu().numpy()[1],gs.cpu().numpy()[1])
