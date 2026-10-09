# Copyright 2026 The MuJoCo Metal contributors
# Licensed under the Apache License, Version 2.0.
"""Actual terrain rows retain source precision on every mass/solver route."""
import os
import mujoco
import numpy as np
import pytest
from test_primitive_hfield_full_manifold import _model, _LIMITS


def _effective_rows(cc, world=0):
  from mujoco_metal.constraint_jacobian import PACKED_J_CSR
  layout=cc._jacobian_layout; nr=cc.descriptor.nr; nv=cc.descriptor.nv
  raw=cc._workspace['workspace_J'].cpu().numpy().reshape(cc.batch_size,-1)[world]
  high=raw[layout.values_offset:layout.values_offset+layout.nnz].astype(float)
  low=raw[layout.lo_values_offset():layout.lo_values_offset()+layout.nnz].astype(float)
  if layout.mode != PACKED_J_CSR:
    return (high+low).reshape(nr,nv)
  indices=raw.view(np.int32)
  ptr=indices[layout.rowptr_offset:layout.rowptr_offset+nr+1]
  cols=indices[layout.columns_offset:layout.columns_offset+layout.nnz]
  result=np.zeros((nr,nv))
  for row in range(nr):
    span=slice(ptr[row],ptr[row+1]);result[row,cols[span]]=high[span]+low[span]
  return result


@pytest.mark.gpu
@pytest.mark.skipif(os.getenv('MUJOCO_METAL_RUN_GPU')!='1',reason='native row precision')
@pytest.mark.parametrize('profile',['integrated_euler_v1','integrated_scalable_v1'])
@pytest.mark.parametrize('solver',[mujoco.mjtSolver.mjSOL_PGS,
                                  mujoco.mjtSolver.mjSOL_CG,
                                  mujoco.mjtSolver.mjSOL_NEWTON])
def test_actual_contact_jacobian_reference_and_inverse_mass_rows(profile,solver):
  from mujoco_metal.simulation import MetalSimulation
  m=_model('cylinder','.25 .15',True);m.opt.solver=solver
  q=np.tile(m.qpos0.astype(np.float32),(2,1));q[1,2]=.8
  v=np.tile(np.array([.02,-.01,.03,.04,-.02,.01],np.float32),(2,1))
  sim=MetalSimulation(m,2,qpos=q,qvel=v,profile=profile,limits=_LIMITS)
  for _ in range(3):
    snap=sim.state.snapshot();words=snap.precision_words.astype(float);nq,nv=m.nq,m.nv
    cpu=mujoco.MjData(m)
    cpu.qpos[:]=snap.qpos[0].astype(float)+words[0,:nq]+words[0,nq:2*nq]
    cpu.qvel[:]=snap.qvel[0].astype(float)+words[0,2*nq:2*nq+nv]+words[0,2*nq+nv:]
    mujoco.mj_forward(m,cpu)
    result=sim.assembled_system(recompute=True);cc=sim._coupled_constraints
    np.testing.assert_array_equal(result['status'].cpu().numpy(),0)
    mask=result['contact_mask'].cpu().numpy()>.5
    assert mask.sum(axis=1).tolist()==[cpu.ncon,0]
    rows=[];source_rows=[]
    offsets=np.asarray(cc.descriptor.contact_condim_packed).reshape(-1,3)
    for slot,contact in zip(np.flatnonzero(mask[0]),cpu.contact[:cpu.ncon]):
      base=int(cc.descriptor.nr_joint)+int(offsets[slot,1])
      rows.extend(range(base,base+3));source_rows.extend(range(contact.efc_address,contact.efc_address+3))
    effective=_effective_rows(cc)
    np.testing.assert_allclose(effective[rows],np.asarray(cpu.efc_J).reshape(cpu.nefc,nv)[source_rows],
                               rtol=2e-12,atol=2e-12)
    ar=result['ar'][0].cpu().numpy().astype(float)+result['ar_low'][0].cpu().numpy()
    assert np.any(result['ar_low'][0].cpu().numpy()[rows]!=0)
    np.testing.assert_allclose(ar[rows],cpu.efc_aref[source_rows],rtol=2e-11,atol=2e-10)
    np.testing.assert_array_equal(_effective_rows(cc,1),0)
    nr=cc.descriptor.nr;debug=cc._workspace['workspace_debug'].cpu().numpy().reshape(2,-1)[0].astype(float)
    regularizer_low=debug[cc._regularizer_low_debug_offset:cc._regularizer_low_debug_offset+nr]
    regularizer=result['R'][0].cpu().numpy().astype(float)+regularizer_low
    np.testing.assert_allclose(regularizer[rows],cpu.efc_R[source_rows],rtol=2e-12,atol=2e-12)
    np.testing.assert_array_equal(cc._workspace['workspace_debug'].cpu().numpy().reshape(2,-1)[1,cc._regularizer_low_debug_offset:cc._regularizer_low_debug_offset+nr],0)
    prefix=nr*nr+7*nr
    zbase=prefix+cc._z_workspace_offset
    zlowbase=prefix+cc._high_low_workspace_offset+8*max(nv,1)+3*max(nr,1)
    z=debug[zbase:zbase+nr*nv].reshape(nr,nv)+debug[zlowbase:zlowbase+nr*nv].reshape(nr,nv)
    if result['mass_matrix'] is not None:
      mass=result['mass_matrix'][0].cpu().numpy().astype(float)
    else:
      # This fixture owns a single six-DOF component, stored row-major.
      assert result['mass_blocks'][0].numel()==nv*nv
      mass=result['mass_blocks'][0].cpu().numpy().astype(float).reshape(nv,nv)
    np.testing.assert_allclose(z[rows],np.linalg.solve(mass,effective[rows].T).T,
                               rtol=2e-11,atol=2e-11)
    np.testing.assert_array_equal(sim.step().cpu().numpy(),0)
