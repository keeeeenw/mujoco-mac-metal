# Copyright 2026 The MuJoCo Metal contributors
# Licensed under the Apache License, Version 2.0.
"""The accepted force and acceleration must describe one effective operator."""
import os
import mujoco
import numpy as np
import pytest
from test_bundled_sdf_rigid_oracle_019 import _plugin_sphere_model


@pytest.mark.gpu
@pytest.mark.skipif(os.getenv('MUJOCO_METAL_RUN_GPU')!='1',reason='native retained operator qualification')
@pytest.mark.parametrize('solver',[mujoco.mjtSolver.mjSOL_NEWTON,mujoco.mjtSolver.mjSOL_CG])
def test_source_torus_retains_mass_inverse_and_acceleration_low_words(solver):
  from mujoco_metal.simulation import MetalSimulation
  m=_plugin_sphere_model('torus');m.opt.solver=solver
  sim=MetalSimulation(m,2,profile='integrated_euler_v1')
  q=np.tile(m.qpos0,(2,1)).astype(np.float32);q[1,0]=1.5
  sim.reset(qpos=q,qvel=np.zeros((2,m.nv),np.float32))
  result=sim.assembled_system(recompute=True)
  assert result['status'].cpu().tolist()==[0,0]
  cc=sim._coupled_constraints;layout=cc._jacobian_layout;nr=cc.descriptor.nr;nv=m.nv
  record=cc._workspace['workspace_J'].cpu().numpy().reshape(2,-1)[0]
  jhi=record[layout.values_offset:layout.values_offset+layout.nnz].reshape(nr,nv).astype(float)
  jlo=record[layout.lo_values_offset():layout.lo_values_offset()+layout.nnz].reshape(nr,nv).astype(float)
  live=np.flatnonzero(result['contact_mask'][0].cpu().numpy()>.5)
  packed=np.asarray(cc.descriptor.contact_condim_packed).reshape(-1,3)
  rows=cc.descriptor.nr_joint+packed[live,1]
  assert len(rows)==8 and np.any(jlo[rows]!=0)
  j=(jhi+jlo)[rows]
  mass=result['mass_matrix'][0].cpu().numpy().astype(float)
  r=result['R'][0].cpu().numpy()[rows].astype(float)
  aref=(result['ar'][0].cpu().numpy()[rows].astype(float)
        +result['ar_low'][0].cpu().numpy()[rows].astype(float))
  # Independently solve the active quadratic at the actual prepared operator.
  # The CPU engine identifies the active set, and all inactive-row inequalities
  # are verified below; neither oracle supplies any production input.
  cpu=mujoco.MjData(m);cpu.qpos[:]=q[0];mujoco.mj_forward(m,cpu)
  source_positions=np.array([c.pos for c in cpu.contact[:cpu.ncon]])
  positions=result['contact_position'][0].cpu().numpy()[live]
  source_order=np.argmin(np.linalg.norm(positions[:,None]-source_positions[None],axis=2),axis=1)
  assert len(set(source_order))==len(live)
  active=cpu.efc_force[source_order]>0
  z=np.linalg.solve(mass,j.T);operator=j@z+np.diag(r)
  force=np.zeros(len(rows));force[active]=np.linalg.solve(operator[np.ix_(active,active)],aref[active])
  assert np.all(force[active]>0)
  gradient=operator@force-aref
  assert np.max(abs(gradient[active]))<1e-9 and np.all(gradient[~active]>=-1e-9)
  expected=z@force
  actual=(result['qacc'][0].cpu().numpy().astype(float)
          +result['qacc_low'][0].cpu().numpy().astype(float))
  np.testing.assert_allclose(actual,expected,rtol=2e-7,atol=5e-7)
  np.testing.assert_allclose(actual,cpu.qacc,rtol=2e-6,atol=2e-5)
  assert np.any(result['qacc_low'][0].cpu().numpy()!=0)
  np.testing.assert_array_equal(result['qacc'][1].cpu().numpy(),np.zeros(nv))
  np.testing.assert_array_equal(result['qacc_low'][1].cpu().numpy(),np.zeros(nv))
