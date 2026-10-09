# Copyright 2026 The MuJoCo Metal contributors
# Licensed under the Apache License, Version 2.0.
"""Pinned public history API semantics through complete native simulation state."""
import os
from dataclasses import replace
import mujoco
import numpy as np
import pytest
from mujoco_metal.history_api import history_entry


def _model(interp=1):
  s=mujoco.MjSpec.from_string('''<mujoco><option timestep=".002" gravity="0 0 0"/>
    <worldbody><body pos=".2 .1 0"><joint name="j" type="slide" axis="1 0 0"/>
      <geom type="sphere" size=".1" mass="1" contype="0" conaffinity="0"/></body></worldbody>
    <actuator><motor name="delayed" joint="j"/><motor name="held" joint="j"/></actuator>
    <sensor><framepos name="vector" objtype="body" objname=""/><clock name="stored"/></sensor></mujoco>'''.replace('objname=""','objname="body"').replace('<body pos=','<body name="body" pos='))
  a=s.actuator('delayed');v=s.sensor('vector')
  a.nsample=v.nsample=8;a.interp=v.interp=interp;a.delay=v.delay=.003
  return s.compile()

@pytest.mark.parametrize('kind',[0,1])
@pytest.mark.parametrize('identifier',[-1,2,True,.5])
def test_history_ids_rejected_without_device_allocation(kind,identifier):
  with pytest.raises(ValueError,match='compiled'):history_entry(_model(),kind,identifier)

@pytest.mark.gpu
@pytest.mark.skipif(os.getenv('MUJOCO_METAL_RUN_GPU')!='1',reason='native public history')
@pytest.mark.parametrize('interp',[0,1,2])
def test_public_history_initialization_owned_reads_and_all_interpolation(interp):
  import torch
  from mujoco_metal import MetalSimulation
  from mujoco_metal.native_api import mj_initCtrlHistory,mj_initSensorHistory,mj_readCtrl,mj_readSensor,mj_setState
  m=_model(interp);sim=MetalSimulation(m,2,profile='integrated_euler_v1')
  times=np.array([-.016,-.014,-.011,-.007,-.004,-.001,.003,.008],np.float32)
  vals=np.array([.2,-.7,.4,1.3,-.2,.9,-.4,.1],np.float32)
  values=np.stack((vals,2*vals,vals+.7),axis=1)
  perctrl=np.stack((vals[:,None],-vals[:,None]))
  pervalues=np.stack((values,-values))
  # Host broadcast and per-world device input cover both boundary routes.
  mj_initCtrlHistory(sim,0,times,torch.tensor(perctrl,device='mps'))
  mj_initSensorHistory(sim,0,times,pervalues,phase=np.array([-.002,.001],np.float32))
  refs=[mujoco.MjData(m) for _ in range(2)]
  for w,d in enumerate(refs):
    mujoco.mj_initCtrlHistory(m,d,0,times.astype(float),perctrl[w].astype(float))
    mujoco.mj_initSensorHistory(m,d,0,times.astype(float),pervalues[w].astype(float),[-.002,.001][w])
  np.testing.assert_allclose(sim.state._history.cpu(),[d.history for d in refs],rtol=2e-6,atol=2e-6)
  before=sim.snapshot();words=sim.state._precision.words.clone();generation=sim.state.generation
  saved=[]
  for override in (-1,0,1,2):
    for query in (-.03,-.011,-.0085,-.003,.002,.007,.03):
      stamp=np.array([query,query+.0007],np.float32)
      actualctrl=mj_readCtrl(sim,0,stamp,override)
      actualsensor=mj_readSensor(sim,0,torch.tensor(stamp,device='mps'),override)
      expectctrl=[];expectsensor=[]
      for w,d in enumerate(refs):
        expectctrl.append(mujoco.mj_readCtrl(m,d,0,float(stamp[w]),override))
        result=np.zeros(3);ptr=mujoco.mj_readSensor(m,d,0,float(stamp[w]),result,override)
        expectsensor.append(ptr if ptr is not None else result)
      np.testing.assert_allclose(actualctrl.cpu(),expectctrl,rtol=5e-6,atol=5e-6)
      np.testing.assert_allclose(actualsensor.cpu(),expectsensor,rtol=5e-6,atol=5e-6)
      saved.append((actualsensor,actualsensor.clone()))
  for value,copy in saved:torch.testing.assert_close(value,copy,rtol=0,atol=0)
  assert sim.state.generation==generation
  torch.testing.assert_close(words,sim.state._precision.words,rtol=0,atol=0)
  np.testing.assert_array_equal(sim.state._history.cpu(),before['native_state']['history'])
  # No-history reads return held control and stored samples, never recompute.
  mj_setState(sim,{'ctrl':np.array([[.3,.7],[-.2,.9]],np.float32)})
  np.testing.assert_array_equal(mj_readCtrl(sim,1,.8).cpu(),np.array([.7,.9],np.float32))
  sim.step(3);stored=sim.step_sensordata().copy()
  mj_setState(sim,{'qpos':np.array([[.9],[-.4]],np.float32)})
  np.testing.assert_array_equal(mj_readSensor(sim,1,-99).cpu(),stored[:,3:4])
  # NULL values preserve physical samples; NULL times preserves physical stamps.
  hist=sim.state._history.cpu().numpy().copy()
  # Reinitialize with sorted stamps before testing source-preserved physical arrays.
  mj_initCtrlHistory(sim,0,times,None,env_ids=[1]);partial=sim.state._history.cpu().numpy().copy()
  np.testing.assert_array_equal(partial[0],hist[0])
  adr=int(m.actuator_historyadr[0]);n=8
  np.testing.assert_array_equal(partial[1,adr+2+n:adr+2+2*n],hist[1,adr+2+n:adr+2+2*n])
  mj_initCtrlHistory(sim,0,None,None,env_ids=[1])
  np.testing.assert_array_equal(sim.state._history.cpu(),partial)

@pytest.mark.gpu
@pytest.mark.skipif(os.getenv('MUJOCO_METAL_RUN_GPU')!='1',reason='native history atomicity')
def test_public_history_atomic_failure_restore_and_rotated_physical_times():
  import torch
  from mujoco_metal import MetalSimulation
  from mujoco_metal.capacity import CapacityLimits,CapacityOverflow
  from mujoco_metal.native_api import mj_initCtrlHistory,mj_initSensorHistory,mj_readSensor
  m=_model();sim=MetalSimulation(m,2,profile='integrated_euler_v1')
  times=np.arange(8,dtype=np.float32)*.002-.016
  values=np.arange(24,dtype=np.float32).reshape(8,3)*.1
  mj_initSensorHistory(sim,0,times,values,phase=-.001)
  checkpoint=sim.snapshot();h=sim.state._history.cpu().numpy().copy();gen=sim.state.generation
  for action in (lambda:mj_initSensorHistory(sim,0,times[::-1].copy(),values),
                 lambda:mj_initSensorHistory(sim,0,times,np.full((8,3),np.nan)),
                 lambda:mj_initSensorHistory(sim,0,times,values,phase=np.inf),
                 lambda:mj_initCtrlHistory(sim,1,times,np.ones((8,1))),
                 lambda:mj_initSensorHistory(sim,0,times,values,env_ids=[0,0]),
                 lambda:mj_readSensor(sim,0,.2,3)):
    with pytest.raises((ValueError,TypeError)):action()
    np.testing.assert_array_equal(sim.state._history.cpu(),h)
    assert sim.state.generation==gen
  sim.step(10,ctrl=np.array([[.5,0],[-.7,0]],np.float32))
  rotated=sim.state._history.cpu().numpy().copy();gen=sim.state.generation
  with pytest.raises(ValueError,match='physical'):mj_initCtrlHistory(sim,0,None,None)
  np.testing.assert_array_equal(sim.state._history.cpu(),rotated)
  assert sim.state.generation==gen
  sim.restore(checkpoint)
  sim.step(2,ctrl=np.array([[.5,0],[-.7,0]],np.float32));expected=sim.state._history.cpu().numpy().copy()
  sim.restore(checkpoint);sim.step(2,ctrl=np.array([[.5,0],[-.7,0]],np.float32))
  np.testing.assert_array_equal(sim.state._history.cpu(),expected)
  from mujoco_metal.native_api import _owned_native_backing_bytes
  sim.limits=replace(sim.limits or CapacityLimits(),memory_budget_bytes=_owned_native_backing_bytes(sim))
  before=sim.state._history.cpu().numpy().copy();gen=sim.state.generation
  with pytest.raises(CapacityOverflow,match='history read'):mj_readSensor(sim,0,.001)
  with pytest.raises(CapacityOverflow,match='history initialization'):mj_initSensorHistory(sim,0,times,values)
  np.testing.assert_array_equal(sim.state._history.cpu(),before)
  assert sim.state.generation==gen
