# Copyright 2026 The MuJoCo Metal contributors
# Licensed under the Apache License, Version 2.0.
"""PlaneConvex physical manifolds use the pinned support and neighbor order."""
import os
import mujoco
import numpy as np
import pytest
from mujoco_metal.coupled_constraints import lower_coupled_constraints

_VERTICES='-.1 -.1 -.1 .1 -.1 -.1 .1 .1 -.1 -.1 .1 -.1 -.1 -.1 .1 .1 -.1 .1 .1 .1 .1 -.1 .1 .1'

def _model(shape,reverse,libccd,integrator='Euler'):
  floor='<geom name="floor" type="plane" size="1 1 .1"/>'
  geom='<geom name="solid" type="mesh" mesh="solid"/>' if shape=='mesh' else '<geom name="solid" type="ellipsoid" size=".1 .07 .1"/>'
  solid='<body name="solid" pos="0 0 .08"><freejoint/>'+geom+'</body>'
  flag='<flag nativeccd="disable"/>' if libccd else ''
  return mujoco.MjModel.from_xml_string(f'<mujoco><option timestep=".001" integrator="{integrator}" gravity="0 0 -9.81" iterations="100" tolerance="1e-8">{flag}</option><asset><mesh name="solid" vertex="{_VERTICES}"/></asset><worldbody>{solid+floor if reverse else floor+solid}</worldbody><contact><pair geom1="solid" geom2="floor" condim="3" friction=".7 .4 .01 .01 .01"/></contact></mujoco>')

@pytest.mark.parametrize('shape',['mesh','ellipsoid'])
@pytest.mark.parametrize('reverse',[False,True])
@pytest.mark.parametrize('libccd',[False,True])
def test_plane_convex_contact_capacity_matches_pinned_callback(shape,reverse,libccd):
  model=_model(shape,reverse,libccd);data=mujoco.MjData(model);mujoco.mj_forward(model,data)
  expected=3 if shape=='mesh' else 1
  desc=lower_coupled_constraints(model)
  assert data.ncon==desc.ncontacts_max==expected
  assert desc.pair_max_contacts.tolist()==[expected]
  assert np.linalg.norm(data.qfrc_constraint)>1

@pytest.mark.gpu
@pytest.mark.skipif(os.getenv('MUJOCO_METAL_RUN_GPU')!='1',reason='native PlaneConvex qualification')
@pytest.mark.parametrize('shape',['mesh','ellipsoid'])
@pytest.mark.parametrize('reverse',[False,True])
@pytest.mark.parametrize('libccd',[False,True])
@pytest.mark.parametrize('integrator',['Euler','RK4','implicitfast'])
def test_plane_convex_contacts_forces_motion_and_lifecycle(shape,reverse,libccd,integrator):
  from mujoco_metal.simulation import MetalSimulation
  model=_model(shape,reverse,libccd,integrator)
  q=np.tile(model.qpos0.astype(np.float32),(2,1));q[1,2]=.4
  v=np.zeros((2,model.nv),np.float32);v[0,0]=.03;v[0,4]=.01
  profile={'Euler':'integrated_euler_v1','RK4':'integrated_rk4_v1','implicitfast':'integrated_implicitfast_v1'}[integrator]
  sim=MetalSimulation(model,2,qpos=q,qvel=v,profile=profile)
  refs=[mujoco.MjData(model) for _ in range(2)]
  for world,ref in enumerate(refs):
    ref.qpos[:]=q[world];ref.qvel[:]=v[world];mujoco.mj_forward(model,ref)
  host=lambda value:value.cpu().numpy().copy()
  actual=sim.assembled_system(recompute=True);mask=host(actual['contact_mask'])>.5
  assert mask.sum(axis=1).tolist()==[refs[0].ncon,0]
  assert refs[0].ncon==(3 if shape=='mesh' else 1)
  for field,expected in (
      ('contact_distance',[c.dist for c in refs[0].contact[:refs[0].ncon]]),
      ('contact_position',[c.pos for c in refs[0].contact[:refs[0].ncon]]),
      ('contact_normal',[c.frame[:3] for c in refs[0].contact[:refs[0].ncon]]),
      ('contact_tangent1',[c.frame[3:6] for c in refs[0].contact[:refs[0].ncon]]),
      ('contact_tangent2',[c.frame[6:9] for c in refs[0].contact[:refs[0].ncon]])):
    np.testing.assert_allclose(host(actual[field])[0,mask[0]],expected,rtol=2e-6,atol=2e-7,err_msg=field)
  for field in ('qacc','qfrc_constraint'):
    np.testing.assert_allclose(host(actual[field]),[getattr(ref,field) for ref in refs],rtol=2e-4,atol=2e-4,err_msg=field)
  initial=sim.snapshot();trajectory=[]
  for _ in range(6):
    np.testing.assert_array_equal(host(sim.step()),0)
    for ref in refs:mujoco.mj_step(model,ref)
    for field in ('qpos','qvel'):
      np.testing.assert_allclose(host(getattr(sim.state,'_'+field)),[getattr(ref,field) for ref in refs],rtol=3e-4,atol=3e-4,err_msg=field)
    trajectory.append({field:host(getattr(sim.state,'_'+field)) for field in ('qpos','qvel','qacc')})
  sim.restore(initial)
  for expected in trajectory:
    np.testing.assert_array_equal(host(sim.step()),0)
    for field,wanted in expected.items():np.testing.assert_array_equal(host(getattr(sim.state,'_'+field)),wanted)
  sim.reset(qpos=q,qvel=v)
  np.testing.assert_array_equal(host(sim.assembled_system(recompute=True)['contact_mask'])>.5,mask)
