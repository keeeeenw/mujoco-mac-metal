# Copyright 2026 The MuJoCo Metal contributors
# Licensed under the Apache License, Version 2.0.
"""Native implicit solve gates above former DOF and body caps."""
import os

import mujoco
import numpy as np
import pytest

from mujoco_metal.implicit import (ImplicitFastProgram, ImplicitProgram,
    implicit_workspace_elements, lower_implicit, lower_implicitfast)


def _model(integrator):
  # Single connected chain gives a nontrivial compiled derivative sparsity.
  body = '<body pos=".03 .01 0"><joint axis="0 1 0"/>'
  geom = '<geom type="sphere" size=".01" mass=".1"/>'
  return mujoco.MjModel.from_xml_string(f'''<mujoco>
    <option integrator="{integrator}"><flag contact="disable"/></option>
    <worldbody>{(body+geom)*72}{'</body>'*72}</worldbody></mujoco>''')


@pytest.mark.parametrize("integrator,lower",[("implicitfast",lower_implicitfast),
                                            ("implicit",lower_implicit)])
def test_implicit_lowering_uses_compiled_dimensions(integrator,lower):
  model = _model(integrator)
  assert lower(model).nv == 72 and model.nbody == 73


def test_implicit_workspace_signed_offsets_and_preallocation_validation():
  assert implicit_workspace_elements(72,2) == 2*72*72
  assert implicit_workspace_elements(0,2) == 2
  for args in ((72,True),(72,1.5),(-1,2),(72,0)):
    with pytest.raises(ValueError):
      implicit_workspace_elements(*args)
  with pytest.raises(ValueError,match="signed int32"):
    implicit_workspace_elements(32768,2)
  # Both constructors reject before needing Torch or allocating a matrix.
  for integrator,program in (("implicitfast",ImplicitFastProgram),
                            ("implicit",ImplicitProgram)):
    with pytest.raises(ValueError,match="indexing capacity"):
      program(_model(integrator),1 << 30)


@pytest.mark.gpu
@pytest.mark.skipif(os.getenv("MUJOCO_METAL_RUN_GPU") != "1",reason="native opt-in")
@pytest.mark.parametrize("integrator,program",[("implicitfast",ImplicitFastProgram),
                                              ("implicit",ImplicitProgram)])
def test_native_implicit_large_dense_solve_parity_and_reuse(integrator,program):
  import torch
  model = _model(integrator)
  native = program(model,2,external_derivative=True)
  rng = np.random.default_rng(71)
  a = rng.normal(size=(2,72,72))*.03
  mass = np.eye(72)[None]+a@a.transpose(0,2,1)
  derivative = rng.normal(size=(2,72,72))*.03-np.eye(72)[None]*.4
  # Use the independently compiled derivative mask, not the native program's.
  pattern = np.zeros((72,72))
  for row in range(72):
    adr,count = int(model.D_rowadr[row]),int(model.D_rownnz[row])
    pattern[row,model.D_colind[adr:adr+count]] = 1
  derivative *= pattern
  full_effective = mass-model.opt.timestep*derivative
  input_derivative = derivative.copy()
  if integrator == "implicitfast":
    derivative = np.tril(derivative)+np.tril(derivative,-1).transpose(0,2,1)
  effective = mass-model.opt.timestep*derivative
  pointer = None
  tensor = lambda x: torch.tensor(x,dtype=torch.float32,device="mps")
  m,d = tensor(mass),tensor(input_derivative)
  full_pointer = None
  for replay in range(2):
    force = rng.normal(size=(2,72))
    result = native.run_device(m,tensor(force),d)
    np.testing.assert_array_equal(result["status"].cpu().numpy(),0)
    np.testing.assert_allclose(result["effective_mass"].cpu().numpy(),effective,
                               atol=2e-7,rtol=2e-6)
    np.testing.assert_allclose(result["full_effective_mass"].cpu().numpy(),
                               full_effective,atol=2e-7,rtol=2e-6)
    if integrator == "implicitfast":
      assert np.max(np.abs(full_effective-effective)) > 1e-5
    if full_pointer is not None:
      assert result["full_effective_mass"].data_ptr() == full_pointer
    full_pointer = result["full_effective_mass"].data_ptr()
    expected = np.stack([np.linalg.solve(effective[b],force[b]) for b in range(2)])
    np.testing.assert_allclose(result["qacc"].cpu().numpy(),expected,atol=2e-6,rtol=1e-5)
    if pointer is not None:
      assert result["qacc"].data_ptr() == pointer
    pointer = result["qacc"].data_ptr()
