# Copyright 2026 The MuJoCo Metal contributors
# Licensed under the Apache License, Version 2.0.
"""Source and independent tangent finite-difference derivative qualification."""
import os

import mujoco
import numpy as np
import pytest

from mujoco_metal.quaternion_derivatives import MetalQuaternionDerivatives, validate_scale

_GPU = pytest.mark.skipif(os.getenv('MUJOCO_METAL_RUN_GPU') != '1',
                         reason='serialized native quaternion derivative gate')


def test_quaternion_derivative_budgets_and_indexing_fail_before_allocation():
  with pytest.raises(ValueError,match='before device allocation'):
    MetalQuaternionDerivatives(8,memory_budget_bytes=1)
  for batch in (0,-1,True,1.5,2**31):
    with pytest.raises(ValueError):
      MetalQuaternionDerivatives(batch)


def test_public_quaternion_derivative_admission_precedes_device_setup(monkeypatch):
  torch=pytest.importorskip('torch')
  from mujoco_metal.native_api import mjd_subQuat,mjd_quatIntegrate
  import mujoco_metal.quaternion_derivatives as module
  def forbidden(*args,**kwargs):
    raise AssertionError('invalid input reached device setup')
  monkeypatch.setattr(module,'MetalQuaternionDerivatives',forbidden)
  for value in (None,np.zeros((2,4)),torch.zeros((2,4)),
                torch.zeros(4),torch.zeros((0,4)),torch.zeros((2,3))):
    with pytest.raises(ValueError):
      mjd_subQuat(value,value)
  with pytest.raises(ValueError):
    mjd_quatIntegrate(torch.zeros((2,3)),1.)
  for scale in (float('nan'),float('inf'),1e-60,True,'1'):
    with pytest.raises(ValueError):
      validate_scale(scale)
  assert validate_scale(0.)==0.
  assert validate_scale(-.5)==-.5


def _quats():
  base=np.array([.87,.22,-.3,.1],dtype=float)
  base/=np.linalg.norm(base)
  qb=np.tile(base,(6,1))
  qa=qb.copy()
  velocities=np.array([[0,0,0],[1e-7,0,0],[.2,-.1,.3],
                       [-.9,.4,.2],[3.1,0,0],[-3.1,0,0]],float)
  for w,vel in enumerate(velocities):
    mujoco.mju_quatIntegrate(qa[w],vel,1)
  qa[2]*=-1
  qb[3]*=-1
  return qa.astype(np.float32),qb.astype(np.float32)


@_GPU
def test_native_sub_quat_derivatives_match_pinned_and_independent_tangent_fd():
  torch=pytest.importorskip('torch')
  qa,qb=_quats()
  program=MetalQuaternionDerivatives(len(qa))
  tensor=lambda a:torch.as_tensor(a.copy(),device='mps')
  result=program.sub_quat(tensor(qa),tensor(qb))
  np.testing.assert_array_equal(result['status'].cpu(),0)
  da,db=result['Da'].cpu().numpy(),result['Db'].cpu().numpy()
  for w in range(len(qa)):
    expected_a,expected_b=np.empty((3,3)),np.empty((3,3))
    mujoco.mjd_subQuat(qa[w].astype(float),qb[w].astype(float),expected_a,expected_b)
    np.testing.assert_allclose(da[w],expected_a,rtol=4e-6,atol=6e-7)
    np.testing.assert_allclose(db[w],expected_b,rtol=4e-6,atol=6e-7)
    for column in range(3):
      axis=np.eye(3)[column]
      step=1e-5
      for which,expected in ((0,da[w]),(1,db[w])):
        values=[]
        for sign in (-1,1):
          a,b=qa[w].astype(float),qb[w].astype(float)
          mujoco.mju_quatIntegrate(a if which==0 else b,axis,sign*step)
          diff=np.empty(3)
          mujoco.mju_subQuat(diff,a,b)
          values.append(diff)
        measured=(values[1]-values[0])/(2*step)
        np.testing.assert_allclose(expected[:,column],measured,rtol=5e-6,atol=1e-6)


@_GPU
@pytest.mark.parametrize('scale',[0.,-.01,.2,2.])
def test_native_quat_integrate_derivatives_match_pinned_and_independent_fd(scale):
  torch=pytest.importorskip('torch')
  velocities=np.array([[0,0,0],[1e-8,-2e-8,0],[.02,.01,-.01],
      [.2,-.1,.3],[1.,.4,-.7],[-1.,-.4,.7]],np.float32)
  result=MetalQuaternionDerivatives(len(velocities)).quat_integrate(
      torch.as_tensor(velocities,device='mps'),scale)
  np.testing.assert_array_equal(result['status'].cpu(),0)
  for w,v32 in enumerate(velocities):
    vel=v32.astype(float)
    dq,dv,ds=np.empty((3,3)),np.empty((3,3)),np.empty(3)
    mujoco.mjd_quatIntegrate(vel,scale,dq.reshape(-1),dv.reshape(-1),ds)
    for name,expected in (('Dquat',dq),('Dvel',dv),('Dscale',ds)):
      np.testing.assert_allclose(result[name][w].cpu(),expected,rtol=4e-6,atol=6e-7)
    base=np.array([1.,0,0,0])
    mujoco.mju_quatIntegrate(base,vel,scale)
    eps=1e-5
    for column in range(3):
      axis=np.eye(3)[column]
      measurements=[]
      for sign in (-1,1):
        q=np.array([1.,0,0,0])
        mujoco.mju_quatIntegrate(q,vel+sign*eps*axis,scale)
        diff=np.empty(3)
        mujoco.mju_subQuat(diff,q,base)
        measurements.append(diff)
      measured=(measurements[1]-measurements[0])/(2*eps)
      # Pinned Dvel is explicitly with respect to scaled velocity.
      np.testing.assert_allclose(result['Dvel'][w,:,column].cpu()*scale,
          measured,rtol=5e-6,atol=1e-6)
      rotated=[]
      for sign in (-1,1):
        q=np.array([1.,0,0,0])
        mujoco.mju_quatIntegrate(q,axis,sign*eps)
        mujoco.mju_quatIntegrate(q,vel,scale)
        diff=np.empty(3)
        mujoco.mju_subQuat(diff,q,base)
        rotated.append(diff)
      np.testing.assert_allclose(result['Dquat'][w,:,column].cpu(),
          (rotated[1]-rotated[0])/(2*eps),rtol=5e-6,atol=1e-6)
    scaled=[]
    for sign in (-1,1):
      q=np.array([1.,0,0,0])
      mujoco.mju_quatIntegrate(q,vel,scale+sign*eps)
      diff=np.empty(3)
      mujoco.mju_subQuat(diff,q,base)
      scaled.append(diff)
    np.testing.assert_allclose(result['Dscale'][w].cpu(),
        (scaled[1]-scaled[0])/(2*eps),rtol=5e-6,atol=1e-6)


@_GPU
def test_quaternion_derivative_public_outputs_ownership_and_invalid_worlds():
  torch=pytest.importorskip('torch')
  from mujoco_metal.native_api import mjd_subQuat,mjd_quatIntegrate
  qa,qb=_quats()
  a,b=torch.as_tensor(qa,device='mps'),torch.as_tensor(qb,device='mps')
  out=mjd_subQuat(a,b)
  saved={name:value.clone() for name,value in out.items()}
  velocities=torch.zeros((6,3),device='mps')
  mjd_quatIntegrate(velocities,-.2)
  for name,value in saved.items():
    torch.testing.assert_close(out[name],value,rtol=0,atol=0)
  a[1,0]=float('nan')
  invalid=mjd_subQuat(a,b)
  np.testing.assert_array_equal(invalid['status'].cpu(),[0,1,0,0,0,0])
  np.testing.assert_array_equal(invalid['Da'][1].cpu(),0)
  velocities[1,0]=1e30
  overflow=mjd_quatIntegrate(velocities,1.)
  np.testing.assert_array_equal(overflow['status'].cpu(),[0,2,0,0,0,0])
  with pytest.raises(ValueError):
    mjd_subQuat(a.cpu(),b)
  for scale in (float('nan'),float('inf'),1e-60,True):
    with pytest.raises(ValueError):
      mjd_quatIntegrate(velocities,scale)
