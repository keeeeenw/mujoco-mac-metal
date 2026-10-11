# Copyright 2026 The MuJoCo Metal contributors
# Licensed under the Apache License, Version 2.0.
"""Native matrix residuals survive fused, retained and matvec routes."""
import os

import mujoco
import numpy as np
import pytest

from mujoco_metal.component_solve import component_solver_workspace_sizes


def test_retained_matrix_admission_includes_both_planes_and_empty_sentinels():
  for nv, nnz in ((0,0),(3,5)):
    parts=component_solver_workspace_sizes(batch=2,nv=nv,ncomponent=int(nv>0),
        nnz=nnz,rhs_capacity=2,layout_size=8)
    assert parts['component_mass_retained_matrix']==2*max(nnz,1)
    assert parts['component_mass_retained_matrix_low']==2*max(nnz,1)


_GPU=pytest.mark.skipif(os.getenv('MUJOCO_METAL_RUN_GPU')!='1',reason='native matrix-word execution')


def _fixture(indefinite=False):
  import torch
  from mujoco_metal.component_solve import MetalComponentMassSolver
  from test_component_mass_solver import _layout
  source=mujoco.MjModel.from_xml_string('''<mujoco><worldbody>
    <body><joint/><geom size=".1"/></body>
    <body><joint/><geom size=".1"/></body>
    <body><joint/><geom size=".1"/></body>
    </worldbody></mujoco>''')
  solver=MetalComponentMassSolver(source,_layout(),batch_size=2,rhs_capacity=3)
  high=np.array([[2,.125,.125,3,1.75],[2.25,-.25,-.25,4,2]],np.float32)
  if indefinite: high[:,3]*=-1
  low=np.array([[1e-8,-2e-8,-2e-8,-1.7e-7,7e-8],
                [3e-8,4e-8,4e-8,2e-8,-6e-8]],np.float32)
  arm=np.array([[.25,.0625,.0625,.375,.125], [.125,-.03125,-.03125,.5,.25]],np.float32)
  arm_low=np.array([[2e-8,1e-8,1e-8,-2e-8,3e-9],[-1e-8,2e-8,2e-8,1e-8,-3e-9]],np.float32)
  rhs=np.linspace(-.7,1.3,18,dtype=np.float32).reshape(2,3,3)
  rhs_low=np.linspace(1e-8,-3e-8,18,dtype=np.float32).reshape(2,3,3)
  tensors=[torch.tensor(a,device='mps') for a in (high,low,arm,arm_low,rhs,rhs_low)]
  full=high.astype(float)+low+arm.astype(float)+arm_low
  matrix=np.zeros((2,3,3))
  matrix[:,0,0]=full[:,0];matrix[:,0,2]=full[:,1]
  matrix[:,2,0]=full[:,2];matrix[:,2,2]=full[:,3];matrix[:,1,1]=full[:,4]
  return solver,tensors,matrix,rhs.astype(float)+rhs_low


@pytest.mark.gpu
@_GPU
@pytest.mark.parametrize('indefinite',[False,True])
def test_fused_and_retained_matrix_pairs_match_full_operator(indefinite):
  solver,(mass,low,arm,arm_low,rhs,rhs_low),matrix,full_rhs=_fixture(indefinite)
  expected=np.stack([np.linalg.solve(matrix[w],full_rhs[w].T).T for w in range(2)])
  solver.run_device(mass,rhs,mass_blocks_low=low,tendon_armature_blocks=arm,
                    tendon_armature_blocks_low=arm_low,rhs_low=rhs_low,
                    allow_indefinite=indefinite)
  actual=solver.paired_output.cpu().numpy().astype(float).sum(axis=0)
  np.testing.assert_array_equal(solver._workspace['status'].cpu().numpy(),0)
  np.testing.assert_allclose(actual,expected,rtol=3e-12,atol=3e-12)
  solver.factorize_device(mass,mass_blocks_low=low,tendon_armature_blocks=arm,
                          tendon_armature_blocks_low=arm_low,allow_indefinite=indefinite)
  # Cached solves own the complete original operator, not borrowed caller words.
  mass.fill_(float('nan'));low.fill_(99);arm.zero_();arm_low.fill_(float('nan'))
  solver.solve_factored_device(rhs,rhs_low=rhs_low)
  actual=solver.paired_output.cpu().numpy().astype(float).sum(axis=0)
  np.testing.assert_array_equal(solver._workspace['status'].cpu().numpy(),0)
  np.testing.assert_allclose(actual,expected,rtol=3e-12,atol=3e-12)
  solver.solve_factored_vector_device(rhs[:,0].contiguous(),rhs_low=rhs_low[:,0].contiguous())
  actual=solver._workspace['single_output_pair'].cpu().numpy().astype(float).sum(axis=0)
  np.testing.assert_allclose(actual,expected[:,0],rtol=3e-12,atol=3e-12)


@pytest.mark.gpu
@_GPU
def test_matrix_and_vector_words_are_retained_in_both_matvec_routes():
  solver,(mass,low,arm,arm_low,rhs,rhs_low),matrix,full_rhs=_fixture()
  hi,lo=solver.run_mass_matvec_pair_device(mass,rhs[:,0].contiguous(),rhs_low[:,0].contiguous(),
      mass_blocks_low=low,tendon_armature_blocks=arm,tendon_armature_blocks_low=arm_low)
  actual=hi.cpu().numpy().astype(float)+lo.cpu().numpy()
  expected=np.einsum('bij,bj->bi',matrix,full_rhs[:,0])
  np.testing.assert_allclose(actual,expected,rtol=3e-14,atol=3e-14)
  result=solver.run_mass_matvec_device(mass,rhs[:,0].contiguous(),mass_blocks_low=low,
      tendon_armature_blocks=arm,tendon_armature_blocks_low=arm_low)
  np.testing.assert_array_equal(result.cpu().numpy(),
      np.einsum('bij,bj->bi',matrix,rhs[:,0].cpu().numpy().astype(float)).astype(np.float32))


@pytest.mark.gpu
@_GPU
def test_low_matrix_failure_is_isolated_and_masked_peer_retains_outputs():
  import torch
  solver,(mass,low,arm,arm_low,rhs,rhs_low),matrix,full_rhs=_fixture()
  solver.run_device(mass,rhs,mass_blocks_low=low,tendon_armature_blocks=arm,
                    tendon_armature_blocks_low=arm_low,rhs_low=rhs_low)
  before=solver.paired_output.clone();status_before=solver._workspace['status'].clone()
  low[0,0]=float('nan');low[1].fill_(float('nan'))
  solver.run_device(mass,rhs,mass_blocks_low=low,tendon_armature_blocks=arm,
      tendon_armature_blocks_low=arm_low,rhs_low=rhs_low,
      world_mask=torch.tensor([1,0],dtype=torch.int32,device='mps'))
  np.testing.assert_array_equal(solver._workspace['status'][0].cpu().numpy(),[1,0])
  np.testing.assert_array_equal(solver._workspace['status'][1].cpu().numpy(),status_before[1].cpu().numpy())
  np.testing.assert_array_equal(solver.paired_output[:,1].cpu().numpy(),before[:,1].cpu().numpy())
  # A fresh healthy call can recover the failed component.
  low.zero_()
  solver.run_device(mass,rhs,mass_blocks_low=low,tendon_armature_blocks=arm,
                    tendon_armature_blocks_low=arm_low,rhs_low=rhs_low)
  np.testing.assert_array_equal(solver._workspace['status'].cpu().numpy(),0)


@pytest.mark.gpu
@_GPU
def test_invalid_residual_metadata_does_not_destroy_retained_operator():
  import torch
  solver,(mass,low,arm,arm_low,rhs,rhs_low),matrix,full_rhs=_fixture()
  solver.factorize_device(mass,mass_blocks_low=low,tendon_armature_blocks=arm,
                          tendon_armature_blocks_low=arm_low)
  old=solver._workspace['retained_matrix_low'].clone()
  for method in (solver.run_device,solver.factorize_device,solver.run_mass_matvec_device,
                 solver.run_mass_matvec_pair_device):
    args=(mass,) if method==solver.factorize_device else ((mass,rhs) if method==solver.run_device
          else (mass,rhs[:,0].contiguous()) if method==solver.run_mass_matvec_device
          else (mass,rhs[:,0].contiguous(),rhs_low[:,0].contiguous()))
    with pytest.raises(ValueError,match='mass_blocks_low'):
      method(*args,mass_blocks_low=torch.zeros((2,5),dtype=torch.float64,device='cpu'))
    assert solver._factorized
    np.testing.assert_array_equal(solver._workspace['retained_matrix_low'].cpu().numpy(),old.cpu().numpy())
  solver.solve_factored_device(rhs,rhs_low=rhs_low)
  expected=np.stack([np.linalg.solve(matrix[w],full_rhs[w].T).T for w in range(2)])
  np.testing.assert_allclose(solver.paired_output.cpu().numpy().astype(float).sum(axis=0),
                             expected,rtol=3e-12,atol=3e-12)


@pytest.mark.gpu
@_GPU
def test_cached_diagonal_is_owned_across_mutation_and_scratch_reuse():
  import torch
  solver,(mass,low,arm,arm_low,rhs,rhs_low),matrix,full_rhs=_fixture()
  diagonal=torch.tensor([[.3,.2,.1],[.4,.1,.2]],device='mps')
  full_matrix=matrix.copy()
  full_matrix[:,np.arange(3),np.arange(3)]+=diagonal.cpu().numpy()
  solver.factorize_device(mass,mass_blocks_low=low,tendon_armature_blocks=arm,
      tendon_armature_blocks_low=arm_low,diagonal_add=diagonal)
  diagonal.fill_(99)
  solver.run_mass_matvec_pair_device(mass,rhs[:,0].contiguous(),rhs_low[:,0].contiguous())
  solver.solve_factored_device(rhs,rhs_low=rhs_low)
  expected=np.stack([np.linalg.solve(full_matrix[w],full_rhs[w].T).T for w in range(2)])
  np.testing.assert_allclose(solver.paired_output.cpu().numpy().astype(float).sum(axis=0),
                             expected,rtol=3e-12,atol=3e-12)


@pytest.mark.gpu
@_GPU
def test_actual_source_inertia_residual_corrects_quantized_mass_solve():
  import torch
  from mujoco_metal.component_solve import MetalComponentMassSolver
  from mujoco_metal.mass_layout import compile_tree_mass_layout
  from mujoco_metal.model import load_model
  source=mujoco.MjModel.from_xml_string('''<mujoco><option gravity="0 0 0"/>
    <worldbody><body><freejoint/><inertial pos="0 0 0" mass="1.3"
      diaginertia=".023125 .024125 .03125"/></body></worldbody></mujoco>''')
  descriptor=load_model(source)
  layout=compile_tree_mass_layout(descriptor.body_treeid,descriptor.dof_bodyid)
  solver=MetalComponentMassSolver(source,layout,batch_size=2,rhs_capacity=2)
  data=mujoco.MjData(source);mujoco.mj_forward(source,data)
  matrix=np.empty((6,6));mujoco.mj_fullM(source,data,matrix)
  high=matrix.astype(np.float32)
  low=(matrix-high).astype(np.float32)
  rhs=np.linspace(-.4,.5,24,dtype=np.float32).reshape(2,2,6)
  mass=torch.tensor(np.tile(high.reshape(1,-1),(2,1)),device='mps')
  residual=torch.tensor(np.tile(low.reshape(1,-1),(2,1)),device='mps')
  expected=np.stack([np.linalg.solve(matrix,row.T).T for row in rhs])
  solver.run_device(mass,torch.tensor(rhs,device='mps'))
  original=solver.paired_output.cpu().numpy().astype(float).sum(axis=0)
  assert np.max(np.abs(original-expected))>1e-8
  solver.run_device(mass,torch.tensor(rhs,device='mps'),mass_blocks_low=residual)
  corrected=solver.paired_output.cpu().numpy().astype(float).sum(axis=0)
  np.testing.assert_allclose(corrected,expected,rtol=3e-13,atol=3e-13)
