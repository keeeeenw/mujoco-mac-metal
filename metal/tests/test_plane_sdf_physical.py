# Copyright 2026 The MuJoCo Metal contributors
# Licensed under the Apache License, Version 2.0.
"""Plane/SDF is the pinned PlaneConvex callback, not an empty SDF search."""
import os
import mujoco
import numpy as np
import pytest
from mujoco_metal.coupled_constraints import lower_coupled_constraints


def _model(reverse=False,libccd=False):
  vertices='-.1 -.1 -.1 .1 -.1 -.1 .1 .1 -.1 -.1 .1 -.1 -.1 -.1 .1 .1 -.1 .1 .1 .1 .1 -.1 .1 .1'
  floor='<geom name="floor" type="plane" size="1 1 .1"/>'
  solid='<body name="solid" pos="0 0 .08"><freejoint/><geom name="solid" type="sdf" mesh="solid"/></body>'
  m=mujoco.MjModel.from_xml_string(f'<mujoco><option timestep=".001" gravity="0 0 -9.81" iterations="100" tolerance="1e-8" sdf_initpoints="8"/><asset><mesh name="solid" vertex="{vertices}"/></asset><worldbody>{solid+floor if reverse else floor+solid}</worldbody><contact><pair geom1="solid" geom2="floor" condim="3" friction=".7 .4 .01 .01 .01"/></contact></mujoco>')
  if libccd:m.opt.disableflags|=int(mujoco.mjtDisableBit.mjDSBL_NATIVECCD)
  return m


@pytest.mark.parametrize('reverse',[False,True])
@pytest.mark.parametrize('libccd',[False,True])
def test_plane_sdf_reserves_the_actual_source_manifold_without_seeds(reverse,libccd):
  m=_model(reverse,libccd);d=mujoco.MjData(m);mujoco.mj_forward(m,d)
  # Source PlaneConvex does not consume Halton seeds. Its pinned driver
  # unsafely allocates zero contacts at initpoints=0, so only the native
  # lowering is evaluated with that option; the oracle used eight slots.
  m.opt.sdf_initpoints=0
  desc=lower_coupled_constraints(m)
  assert d.ncon==3 and desc.npairs==1
  assert desc.ncontacts_max==3 and desc.pair_max_contacts.tolist()==[3]


@pytest.mark.parametrize('reverse',[False,True])
@pytest.mark.parametrize('libccd',[False,True])
@pytest.mark.gpu
@pytest.mark.skipif(os.getenv('MUJOCO_METAL_RUN_GPU')!='1',reason='native plane/SDF physical qualification')
def test_plane_sdf_full_contacts_forces_nonzero_motion_and_replay(reverse,libccd):
  from mujoco_metal.simulation import MetalSimulation
  m=_model(reverse,libccd);sim=MetalSimulation(m,2,profile='integrated_euler_v1')
  q=np.tile(m.qpos0,(2,1)).astype(np.float32);q[1,2]=.4
  v=np.zeros((2,m.nv),np.float32);v[0,0]=.03;v[0,4]=.01
  sim.reset(qpos=q,qvel=v)
  cpu=[]
  for world in range(2):
    d=mujoco.MjData(m);d.qpos[:]=q[world];d.qvel[:]=v[world];mujoco.mj_forward(m,d);cpu.append(d)
  system=sim.assembled_system(recompute=True)
  initial_mask=system['contact_mask'].cpu().numpy().copy()
  counts=np.count_nonzero(initial_mask>.5,axis=1)
  assert counts.tolist()==[3,0]
  # Source order is part of this callback: no post-hoc contact sorting.
  np.testing.assert_allclose(system['contact_position'][0].cpu().numpy(),
      [c.pos for c in cpu[0].contact[:cpu[0].ncon]],rtol=2e-6,atol=2e-7)
  np.testing.assert_allclose(system['contact_distance'][0].cpu().numpy(),
      [c.dist for c in cpu[0].contact[:cpu[0].ncon]],rtol=2e-6,atol=2e-7)
  np.testing.assert_allclose(system['contact_normal'][0].cpu().numpy(),
      [c.frame[:3] for c in cpu[0].contact[:cpu[0].ncon]],rtol=2e-6,atol=2e-7)
  np.testing.assert_allclose(system['qacc'][0].cpu().numpy(),cpu[0].qacc,
      rtol=2e-4,atol=2e-4)
  np.testing.assert_allclose(system['qfrc_constraint'][0].cpu().numpy(),cpu[0].qfrc_constraint,
      rtol=2e-4,atol=2e-4)
  snapshot=sim.snapshot();first=[]
  for _ in range(6):
    for d in cpu:mujoco.mj_step(m,d)
    assert sim.step(1).cpu().tolist()==[0,0]
    np.testing.assert_allclose(sim.state._qpos.cpu().numpy(),[d.qpos for d in cpu],rtol=3e-4,atol=3e-5)
    np.testing.assert_allclose(sim.state._qvel.cpu().numpy(),[d.qvel for d in cpu],rtol=3e-4,atol=3e-4)
    first.append((sim.state._qpos.clone(),sim.state._qvel.clone()))
  sim.restore(snapshot)
  for qp,qv in first:
    assert sim.step(1).cpu().tolist()==[0,0]
    assert sim.state._qpos.equal(qp) and sim.state._qvel.equal(qv)
  sim.reset(qpos=q,qvel=v)
  np.testing.assert_array_equal(sim.assembled_system(recompute=True)['contact_mask'].cpu().numpy(),
      initial_mask)


@pytest.mark.parametrize('reverse',[False,True])
@pytest.mark.parametrize('libccd',[False,True])
@pytest.mark.gpu
@pytest.mark.skipif(os.getenv('MUJOCO_METAL_RUN_GPU')!='1',reason='native zero-seed plane/SDF qualification')
def test_plane_sdf_zero_seeds_still_executes_the_source_callback(reverse,libccd):
  from mujoco_metal.simulation import MetalSimulation
  oracle=_model(reverse,libccd)
  cpu=mujoco.MjData(oracle);mujoco.mj_forward(oracle,cpu)
  # The source callback has no seed input. Keep its driver allocation safe
  # while independently testing native zero-seed allocation and execution.
  native=_model(reverse,libccd);native.opt.sdf_initpoints=0
  sim=MetalSimulation(native,1,profile='integrated_euler_v1')
  system=sim.assembled_system(recompute=True)
  assert int(system['contact_mask'][0].sum().cpu())==cpu.ncon==3
  np.testing.assert_allclose(system['contact_position'][0].cpu().numpy(),
      [c.pos for c in cpu.contact[:cpu.ncon]],rtol=2e-6,atol=2e-7)
  np.testing.assert_allclose(system['qacc'][0].cpu().numpy(),cpu.qacc,rtol=2e-4,atol=2e-4)
  for _ in range(3):
    mujoco.mj_step(oracle,cpu)
    assert sim.step(1).cpu().tolist()==[0]
    np.testing.assert_allclose(sim.state._qvel[0].cpu().numpy(),cpu.qvel,rtol=3e-4,atol=3e-4)
