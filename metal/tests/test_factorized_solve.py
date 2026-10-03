# Copyright 2026 The MuJoCo Metal contributors
# Licensed under the Apache License, Version 2.0.
"""Reusable factors, changing RHS, pivoting and independent failure worlds."""
import os
import sys

import numpy as np
import pytest

from mujoco_metal.smooth_solve import (
    MetalFactorizedSolve, factored_workspace_elements)


def test_factor_workspace_preflight_without_torch(monkeypatch):
  monkeypatch.setitem(sys.modules,"torch",None)
  assert factored_workspace_elements(72,3,2) == {
      "factor":15552,"pivots":216,"solution":432,"status":3}
  for args in ((-1,1,1),(3,0,1),(3,1,0),(65536,1,1),(72,1 << 30,1)):
    with pytest.raises(ValueError):
      MetalFactorizedSolve(*args)
  for args in ((True,1,1),(3,1.5,1),(3,1,False)):
    with pytest.raises(TypeError):
      MetalFactorizedSolve(*args)


def _oracle(matrices, rhs):
  # Explicit per-world solves avoid NumPy's matrix-RHS broadcasting ambiguity
  # for [B,n] vectors when the batch size happens to equal n.
  return np.stack([np.linalg.solve(matrix,value) for matrix,value in zip(matrices,rhs)])


native = pytest.mark.skipif(os.getenv("MUJOCO_METAL_RUN_GPU") != "1",
                            reason="native opt-in")


@pytest.mark.gpu
@native
@pytest.mark.parametrize("general",[False,True])
@pytest.mark.parametrize("nrhs",[1,3])
def test_native_factor_reuse_pivots_and_rhs_failures(general,nrhs):
  import torch
  n,b = 3,3
  matrix = (np.array([[0.,2.,1.],[1.,-1.,1.],[3.,1.,4.]]) if general
            else np.array([[2.,.2,.1],[.2,1.,.05],[.1,.05,.7]]))
  matrices = np.stack([matrix,2*matrix,.6*matrix])
  program = MetalFactorizedSolve(n,b,nrhs,general=general)
  def tensor(value):
    return torch.tensor(value,dtype=torch.float32,device="mps").contiguous()
  original = tensor(matrices)
  shape = (b,n) if nrhs==1 else (b,n,nrhs)
  rhs = (np.arange(np.prod(shape)).reshape(shape)+1)/7
  with pytest.raises(RuntimeError,match="factor_device"):
    program.solve_factored_device(tensor(rhs))
  np.testing.assert_array_equal(program.factor_device(original).cpu().numpy(),0)
  factor_copy = program._factor.cpu().numpy().copy()
  factor_pointer = program._factor.data_ptr()
  # The factor is a submitted snapshot, not an alias of the original matrix.
  original.zero_()
  with pytest.raises(ValueError,match="shape"):
    program.run_device(tensor(3*matrices),tensor(np.zeros((b,n+1))))
  np.testing.assert_array_equal(program._factor.cpu().numpy(),factor_copy)
  solution_pointer = None
  for scale in (1.,-.3,2.):
    value,status = program.solve_factored_device(tensor(scale*rhs))
    expected = _oracle(matrices,scale*rhs)
    np.testing.assert_array_equal(status.cpu().numpy(),0)
    np.testing.assert_allclose(value.cpu().numpy(),expected,atol=3e-6,rtol=3e-6)
    np.testing.assert_array_equal(program._factor.cpu().numpy(),factor_copy)
    assert program._factor.data_ptr()==factor_pointer
    if solution_pointer is not None: assert value.data_ptr()==solution_pointer
    solution_pointer=value.data_ptr()
  bad_rhs=rhs.copy(); bad_rhs[1].flat[0]=np.nan
  value,status=program.solve_factored_device(tensor(bad_rhs))
  np.testing.assert_array_equal(status.cpu().numpy(),[0,1,0])
  np.testing.assert_array_equal(value.cpu().numpy()[1],0)
  # A bad RHS cannot poison a valid factor or a neighboring world.
  value,status=program.solve_factored_device(tensor(rhs))
  np.testing.assert_array_equal(status.cpu().numpy(),0)
  np.testing.assert_allclose(value.cpu().numpy(),_oracle(matrices,rhs),atol=3e-6,rtol=3e-6)
  changed=matrices.copy(); changed[:,np.arange(n),np.arange(n)]+=1.5
  value,status=program.run_device(tensor(changed),tensor(rhs))
  np.testing.assert_array_equal(status.cpu().numpy(),0)
  np.testing.assert_allclose(value.cpu().numpy(),_oracle(changed,rhs),atol=3e-6,rtol=3e-6)
  singular=changed.copy(); singular[1].fill(0)
  np.testing.assert_array_equal(program.factor_device(tensor(singular)).cpu().numpy(),[0,3,0])
  for scale in (1.,2.):
    value,status=program.solve_factored_device(tensor(scale*rhs))
    np.testing.assert_array_equal(status.cpu().numpy(),[0,3,0])
    np.testing.assert_array_equal(value.cpu().numpy()[1],0)
    np.testing.assert_allclose(value.cpu().numpy()[[0,2]],
        _oracle(changed[[0,2]],scale*rhs[[0,2]]),atol=3e-6,rtol=3e-6)
  program.factor_device(tensor(changed))
  np.testing.assert_array_equal(program.solve_factored_device(tensor(rhs))[1].cpu().numpy(),0)


@pytest.mark.gpu
@native
@pytest.mark.parametrize("general",[False,True])
def test_native_factor_empty_system(general):
  import torch
  program=MetalFactorizedSolve(0,2,general=general)
  value,status=program.run_device(torch.empty((2,0,0),device="mps"),
                                 torch.empty((2,0),device="mps"))
  assert value.shape==(2,0)
  np.testing.assert_array_equal(status.cpu().numpy(),0)


@pytest.mark.gpu
@native
@pytest.mark.parametrize("general",[False,True])
def test_native_factor_compiled_72_dof_workspace(general):
  import torch
  rng=np.random.default_rng(901)
  n,b,nrhs=72,2,2
  raw=rng.normal(size=(b,n,n))
  matrices=(raw*.025+np.eye(n)*3 if general
            else np.einsum("bji,bjk->bik",raw,raw)/n+np.eye(n))
  if general:
    # Require real pivot recording, not a triangular/no-pivot shortcut.
    matrices=matrices[:,np.roll(np.arange(n),1),:]
  rhs=rng.normal(size=(b,n,nrhs))
  program=MetalFactorizedSolve(n,b,nrhs,general=general)
  def tensor(value): return torch.tensor(value,dtype=torch.float32,device="mps").contiguous()
  matrix=tensor(matrices)
  np.testing.assert_array_equal(program.factor_device(matrix).cpu().numpy(),0)
  factor_pointer=program._factor.data_ptr()
  for scale in (1.,-.5):
    value,status=program.solve_factored_device(tensor(scale*rhs))
    np.testing.assert_array_equal(status.cpu().numpy(),0)
    np.testing.assert_allclose(value.cpu().numpy(),_oracle(matrices,scale*rhs),
                               atol=4e-6,rtol=2e-5)
    assert program._factor.data_ptr()==factor_pointer
