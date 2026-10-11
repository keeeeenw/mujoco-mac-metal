# Copyright 2026 The MuJoCo Metal contributors
# Licensed under the Apache License, Version 2.0.
"""Component midpoint ownership, actual allocation and early admission."""
import os
from dataclasses import replace
from types import SimpleNamespace
import mujoco
import numpy as np
import pytest

from mujoco_metal import MetalSimulation
from mujoco_metal.capacity import (
    CapacityOverflow, estimate_capacity, model_component_capacity_limits)
from mujoco_metal.implicit_midpoint import lower_free_body_midpoints
from test_implicit_midpoint import XML


def test_packed_solver_labels_remain_authoritative_without_a_dense_jacobian():
  import torch
  model = mujoco.MjModel.from_xml_string(XML)
  descriptor = lower_free_body_midpoints(model)
  sim = MetalSimulation.__new__(MetalSimulation)
  sim.batch_size = 3
  sim._state = SimpleNamespace(_torch=torch, _device=torch.device('cpu'), generation=7)
  sim._mjmodel = model
  sim._midpoint = SimpleNamespace(descriptor=descriptor)
  labels = torch.full((3, model.nv), -1, dtype=torch.int32)
  labels[0, int(descriptor.dofadr[0])] = 0
  labels[1, int(descriptor.dofadr[1])] = 1
  sim._last_coupled = dict(J_packed=torch.empty(0), dof_island=labels)
  sim._last_coupled_generation = 7
  assert sim._midpoint_eligibility().tolist() == [[False, True], [True, False], [True, True]]
  # Old assembly labels must not be revived when the state generation changes.
  sim._state.generation += 1
  assert sim._midpoint_eligibility().tolist() == [[True, True]]*3


@pytest.mark.parametrize('disabled', ['inverse_discrete', 'fluid'])
def test_source_ineligible_models_do_not_reserve_a_midpoint_program(disabled):
  model = mujoco.MjModel.from_xml_string(XML)
  if disabled == 'inverse_discrete':
    model.opt.enableflags |= int(mujoco.mjtEnableBit.mjENBL_INVDISCRETE)
  else:
    model.opt.viscosity = .1
  assert lower_free_body_midpoints(model).nfree == 0
  report = estimate_capacity(model, 2, 0, 0, 0, mass_storage='block_sparse')
  assert not any(name.startswith('midpoint.') for name, _ in report.memory_breakdown)


def test_component_midpoint_budget_rejects_before_shader_initialization(monkeypatch):
  import torch
  model = mujoco.MjModel.from_xml_string(XML)
  limits = replace(model_component_capacity_limits(model, 2), memory_budget_bytes=1)
  def forbidden(*args, **kwargs):
    raise AssertionError('insufficient budget reached device initialization')
  monkeypatch.setattr(torch.mps, 'compile_shader', forbidden)
  with pytest.raises(CapacityOverflow):
    MetalSimulation(model, 2, profile='integrated_implicitfast_v1', limits=limits)


@pytest.mark.gpu
@pytest.mark.skipif(os.getenv('MUJOCO_METAL_RUN_GPU') != '1', reason='actual midpoint allocation')
@pytest.mark.parametrize('profile,storage', [
    ('contact_free_implicitfast_v1', 'dense'), ('integrated_implicitfast_v1', 'block_sparse')])
def test_admission_covers_every_actual_midpoint_tensor_and_reuses_selection_storage(profile, storage):
  import torch
  model = mujoco.MjModel.from_xml_string(XML)
  sim = MetalSimulation(model, 2, profile=profile)
  program = sim._midpoint
  assert program is not None
  parts = dict(estimate_capacity(model, 2, 0, 0, 0, mass_storage=storage).memory_breakdown)
  bindings = dict(program._arrays, **program._outputs,
                  gravity=program._gravity, dims=program._dims,
                  timestep=program._timestep, empty=program._empty,
                  eligibility=program._eligibility)
  for name, value in bindings.items():
    assert parts['midpoint.'+name] >= value.numel()*value.element_size()
  assert sum(size for name, size in parts.items() if name.startswith('midpoint.')) >= sum(
      value.numel()*value.element_size() for value in bindings.values())
  original = program._eligibility
  # Execute the actual stage, including repeated mask conversion into the same
  # owned buffer. Numerical correctness is independently covered by trajectories.
  zero = torch.zeros((2, model.nv), device=sim.state._device)
  quat = torch.as_tensor(np.tile(model.body_quat, (2, 1, 1)),
                         dtype=torch.float32, device=sim.state._device)
  for enabled in (False, True, False):
    mask = torch.full((2, program.descriptor.nfree), enabled, device=sim.state._device)
    result = program.run_device(zero, zero, zero, zero, quat, eligible_mask=mask)
    assert program._eligibility is original
    np.testing.assert_array_equal(program._eligibility.cpu().numpy(), int(enabled))
    np.testing.assert_array_equal(result['status'].cpu().numpy(), 0)
