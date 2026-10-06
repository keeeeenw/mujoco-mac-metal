# Copyright 2026 The MuJoCo Metal contributors
# Licensed under the Apache License, Version 2.0.
"""Native inverse queries preserve component implicit solver storage."""

import os

import mujoco
import numpy as np
import pytest

from tests.test_sparse_implicit_public_019 import _fixture


@pytest.mark.gpu
@pytest.mark.skipif(os.getenv("MUJOCO_METAL_RUN_GPU") != "1", reason="opt-in GPU")
@pytest.mark.parametrize("profile", ["integrated_implicit_v1", "integrated_implicitfast_v1"])
@pytest.mark.parametrize("fail", [False, True])
def test_native_sparse_inverse_query_restores_operator_storage_and_next_step(profile, fail,
                                                                          monkeypatch):
  import torch
  from mujoco_metal import MetalSimulation
  from mujoco_metal.forward_stages import ForwardStage
  from mujoco_metal.native_api import mj_forward, mj_inverseSkip

  model = _fixture(profile)
  model.opt.enableflags |= int(mujoco.mjtEnableBit.mjENBL_INVDISCRETE)
  qpos = np.asarray([[.12, -.21], [-.17, .24]], dtype=np.float32)
  qvel = np.asarray([[.7, -.45], [-.3, .62]], dtype=np.float32)
  control = np.asarray([[.18], [-.11]], dtype=np.float32)
  sim, twin = [MetalSimulation(model, batch_size=2, qpos=qpos, qvel=qvel,
                              profile=profile) for _ in range(2)]
  for candidate in (sim, twin):
    assert candidate.step(ctrl=control).cpu().tolist() == [0, 0]
    mj_forward(candidate)
  record = sim._forward_stages._record
  writer = sim._velocity_derivative_values
  owner = writer.values
  before_values = owner.clone()
  workspace = sim._effective_implicit._workspace
  before = {name: (value, value.clone()) for name, value in workspace.items()
            if isinstance(value, torch.Tensor)}
  original = sim._inverse_discrete_acceleration

  def injected(*args, **kwargs):
    original(*args, **kwargs)
    raise RuntimeError("injected post-operator inverse failure")

  requested = torch.tensor([[.9, -.35], [-.6, .42]], device="mps")
  if fail:
    with monkeypatch.context() as context:
      context.setattr(sim, "_inverse_discrete_acceleration", injected)
      with pytest.raises(RuntimeError, match="post-operator inverse failure"):
        mj_inverseSkip(sim, qacc=requested, return_details=True)
  else:
    result = mj_inverseSkip(sim, qacc=requested, return_details=True)
    assert result["status"].cpu().tolist() == [0, 0]
    assert torch.isfinite(result["qfrc_inverse"]).all()
  assert writer.values is owner
  torch.testing.assert_close(owner, before_values, rtol=0, atol=0)
  assert sim._effective_implicit._workspace is workspace
  assert workspace.keys() == before.keys()
  for name, (tensor, saved) in before.items():
    assert workspace[name] is tensor
    torch.testing.assert_close(tensor, saved, rtol=0, atol=0)
  assert sim._forward_stages._record is record
  sim.validate_forward_stage_record(record, ForwardStage.CONSTRAINT)
  assert sim.step(ctrl=control).cpu().tolist() == [0, 0]
  assert twin.step(ctrl=control).cpu().tolist() == [0, 0]
  torch.testing.assert_close(sim.state.qpos, twin.state.qpos, rtol=0, atol=0)
  torch.testing.assert_close(sim.state.qvel, twin.state.qvel, rtol=0, atol=0)
  torch.testing.assert_close(sim.state.qacc, twin.state.qacc, rtol=0, atol=0)
