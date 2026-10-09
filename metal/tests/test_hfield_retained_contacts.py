# Copyright 2026 The MuJoCo Metal contributors
# Licensed under the Apache License, Version 2.0.
"""The alternate terrain producer preserves its actual source-wide witnesses.

The source oracle evaluates represented native poses for geometry publication.
This does not replace the separate, unchanged continuous trajectory oracle.
"""
import os
import mujoco
import numpy as np
import pytest
from test_primitive_hfield_full_manifold import _model, _LIMITS

@pytest.mark.gpu
@pytest.mark.skipif(os.getenv('MUJOCO_METAL_RUN_GPU')!='1',reason='native retained terrain contacts')
@pytest.mark.parametrize('shape,size',[('cylinder','.25 .15'),('sphere','.3')])
@pytest.mark.parametrize('reverse',[False,True])
def test_actual_source_contact_distance_position_normal_words(shape,size,reverse):
  from mujoco_metal import MetalSimulation
  m=_model(shape,size,True,reverse)
  q=np.tile(m.qpos0.astype(np.float32),(2,1));q[1,2]=.8
  v=np.tile(np.array([.02,-.01,.03,.04,-.02,.01],np.float32),(2,1))
  sim=MetalSimulation(m,2,qpos=q,qvel=v,profile='integrated_scalable_v1',limits=_LIMITS)
  host=lambda value:value.cpu().numpy().copy()
  for _ in range(3):
    snap=sim.state.snapshot();words=snap.precision_words.astype(float);nq,nv=m.nq,m.nv
    cpu=mujoco.MjData(m)
    cpu.qpos[:]=snap.qpos[0].astype(float)+words[0,:nq]+words[0,nq:2*nq]
    cpu.qvel[:]=snap.qvel[0].astype(float)+words[0,2*nq:2*nq+nv]+words[0,2*nq+nv:]
    mujoco.mj_forward(m,cpu)
    result=sim.assembled_system(recompute=True)
    np.testing.assert_array_equal(host(result['status']),0)
    mask=host(result['contact_mask'])>.5
    assert mask.sum(axis=1).tolist()==[cpu.ncon,0]
    records=host(sim._coupled_constraints._workspace['contact_pair_records']).reshape(2,50,37)[0,mask[0]].astype(float)
    assert np.any(records[:,9:11]!=0)
    assert np.any(records[:,25:]!=0)
    sign=1 if int(sim._coupled_constraints.descriptor.geom1[0])==int(cpu.contact[0].geom[0]) else -1
    np.testing.assert_allclose(records[:,12]+records[:,9]+records[:,10],
                               [c.dist for c in cpu.contact[:cpu.ncon]],rtol=2e-12,atol=2e-12)
    np.testing.assert_allclose(records[:,13:16]+records[:,25:28]+records[:,31:34],
                               [c.pos for c in cpu.contact[:cpu.ncon]],rtol=2e-12,atol=2e-12)
    np.testing.assert_allclose(records[:,16:19]+records[:,28:31]+records[:,34:37],
                               sign*np.array([c.frame[:3] for c in cpu.contact[:cpu.ncon]]),rtol=2e-12,atol=2e-12)
    np.testing.assert_array_equal(host(sim.step()),0)
