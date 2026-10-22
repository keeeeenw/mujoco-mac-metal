# Copyright 2026 The MuJoCo Metal contributors
# Licensed under the Apache License, Version 2.0.
"""Integration reads a source scalar, even when paired solver words overlap."""
import os
import mujoco
import numpy as np
import pytest


@pytest.mark.gpu
@pytest.mark.skipif(os.getenv('MUJOCO_METAL_RUN_GPU') != '1', reason='native velocity input precision')
@pytest.mark.parametrize('timestep', [.0005,.001,.002])
def test_velocity_update_matches_source_for_noncanonical_acceleration_words(timestep):
  import torch
  from mujoco_metal.integration import MetalEulerIntegration
  from mujoco_metal.model import load_model
  descriptor=load_model('<mujoco><worldbody><body><freejoint/><geom type="sphere" size=".1"/></body></worldbody></mujoco>')
  rng=np.random.default_rng(51673)
  velocity=rng.uniform(-.1,.1,(32,6)).astype(np.float32)
  acceleration=rng.uniform(-300,300,(32,6)).astype(np.float32)
  low=np.ldexp(rng.uniform(-1,1,(32,6)),rng.integers(-33,-21,(32,6))).astype(np.float32)
  # Literal actual emitted solver words at the saved original ridge step.
  velocity[0,1]=np.float32(-.009999999776482582)
  acceleration[0,1]=np.float32(-68.12631225585938)
  low[0,1]=np.float32(-5.968778538090191e-08)
  expected=velocity.astype(float)
  source_acceleration=acceleration.astype(float)+low.astype(float)
  for world in range(32):mujoco.mju_addToScl(expected[world],source_acceleration[world],timestep)
  stage=MetalEulerIntegration(descriptor,32,timestep)
  q=np.tile(np.array([0,0,0,1,0,0,0],np.float32),(32,1))
  q=torch.tensor(q,device='mps');v=torch.tensor(velocity,device='mps')
  a=torch.tensor(acceleration,device='mps');correction=torch.tensor(low,device='mps')
  protected=[t.clone()for t in (q,v,a,correction)]
  _,actual,_,status=stage.run_device(q,v,a,torch.zeros(32,device='mps'),
      torch.zeros(32,device='mps',dtype=torch.int32),qacc_low=correction)
  np.testing.assert_array_equal(status.cpu().numpy(),0)
  words=stage._output_words.cpu().double().numpy()
  represented=actual.cpu().double().numpy()+words[:,2*descriptor.nq:2*descriptor.nq+descriptor.nv]+words[:,2*descriptor.nq+descriptor.nv:]
  np.testing.assert_array_equal(represented,expected)
  for actual_input,before in zip((q,v,a,correction),protected):
    torch.testing.assert_close(actual_input.view(torch.int32),before.view(torch.int32),rtol=0,atol=0)
