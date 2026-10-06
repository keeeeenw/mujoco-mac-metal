# Copyright 2026 The MuJoCo Metal contributors
# Licensed under the Apache License, Version 2.0.
"""Analytical held-position RNE derivative and compiled workspace gates."""
import os

import mujoco
import numpy as np
import pytest

from mujoco_metal.smooth_metal import bias_derivative_workspace_sizes


def _model(kind):
  count = 72 if kind == "scalar" else 7
  joints = ["<freejoint/>" if kind == "mixed" else
            '<joint type="hinge" axis="1 .2 .1"/>']
  joints += ['<joint type="ball"/>' if kind == "mixed" and i % 2 == 0 else
             '<joint type="hinge" axis=".2 1 .3"/>' for i in range(count - 1)]
  nested = ''.join(f'<body pos=".08 .02 .01">{joint}'
      '<geom type="box" pos=".04 .01 .02" size=".03 .02 .01" mass=".1"/>'
      for joint in joints) + '</body>'*count
  return mujoco.MjModel.from_xml_string(f'''<mujoco>
    <option gravity=".2 -.1 -9.81"><flag contact="disable"/></option>
    <worldbody>{nested}</worldbody></mujoco>''')


def _reference(model, qpos, qvel):
  data = mujoco.MjData(model)
  data.qpos[:] = qpos
  derivative = np.empty((model.nv, model.nv))
  step = 1e-6
  for col in range(model.nv):
    data.qvel[:] = qvel
    data.qvel[col] += step
    mujoco.mj_forward(model, data)
    plus = data.qfrc_bias.copy()
    data.qvel[col] -= 2*step
    mujoco.mj_forward(model, data)
    derivative[:, col] = -(plus-data.qfrc_bias)/(2*step)
  return derivative


def test_derivative_workspace_sizes_overflow_and_dimensions():
  assert bias_derivative_workspace_sizes(73,72,2) == {
      "bias_derivative_scratch": 2*72*6*(3*73+72),
      "bias_derivative": 2*72*72}
  with pytest.raises(ValueError, match="capacity|limit"):
    bias_derivative_workspace_sizes(1 << 28,72,2)
  for name, args in (("nbody",(-1,1,1)),("nv",(1,True,1)),
                     ("batch_size",(1,1,False))):
    with pytest.raises(ValueError, match=name):
      bias_derivative_workspace_sizes(*args)


@pytest.mark.parametrize("kind", ["scalar", "mixed"])
def test_derivative_fixture_has_nontrivial_coupling(kind):
  model = _model(kind)
  qpos = model.qpos0.copy()
  mujoco.mj_integratePos(model,qpos,np.linspace(-.04,.07,model.nv),1)
  derivative = _reference(model,qpos,np.linspace(.1,.7,model.nv))
  assert np.max(np.abs(derivative)) > .01
  assert np.count_nonzero(np.abs(derivative) > 1e-6) > model.nv


@pytest.mark.gpu
@pytest.mark.skipif(os.getenv("MUJOCO_METAL_RUN_GPU") != "1", reason="native opt-in")
@pytest.mark.parametrize("kind", ["scalar", "mixed"])
def test_native_analytical_bias_derivative_parity_and_workspace_reuse(kind):
  import torch
  from mujoco_metal.model import load_model
  from mujoco_metal.smooth_metal import MetalSmoothDynamics
  model = _model(kind)
  program = MetalSmoothDynamics(load_model(model),batch_size=2)
  pointer = None
  for replay in range(2):
    qpos, qvel, expected = [], [], []
    for world in range(2):
      q = model.qpos0.copy()
      mujoco.mj_integratePos(model,q,np.linspace(-.04,.07,model.nv),1)
      v = np.linspace(.1,.7,model.nv)*(world+1)*(1-.4*replay)
      qpos.append(q); qvel.append(v); expected.append(_reference(model,q,v))
    tensor = lambda x: torch.tensor(np.asarray(x),dtype=torch.float32,device="mps")
    qp,qv = tensor(qpos),tensor(qvel)
    dynamics = program.run_device(qp,qv)
    result = program.bias_derivative_device(qp,qv,dynamics)
    np.testing.assert_allclose(result.cpu().numpy(),expected,atol=5e-4,rtol=3e-4)
    if pointer is not None:
      assert result.data_ptr() == pointer
    pointer = result.data_ptr()


@pytest.mark.gpu
@pytest.mark.skipif(os.getenv("MUJOCO_METAL_RUN_GPU") != "1", reason="native opt-in")
@pytest.mark.parametrize("kind", ["scalar", "mixed"])
def test_native_analytical_bias_derivative_writes_compiled_coo(kind):
  import torch
  from mujoco_metal.model import load_model
  from mujoco_metal.smooth_metal import MetalSmoothDynamics
  from mujoco_metal.velocity_derivative import (
      MetalVelocityDerivativeValues, compile_velocity_derivative_layout)

  model = _model(kind)
  layout = compile_velocity_derivative_layout(model)
  program = MetalSmoothDynamics(
      load_model(model), batch_size=2, velocity_derivative_layout=layout)
  qpos, qvel, expected = [], [], []
  for world in range(2):
    q = model.qpos0.copy()
    mujoco.mj_integratePos(model, q, np.linspace(-.04, .07, model.nv), 1)
    v = np.linspace(.1, .7, model.nv) * (world + 1)
    qpos.append(q)
    qvel.append(v)
    expected.append(_reference(model, q, v))
  tensor = lambda x: torch.tensor(np.asarray(x), dtype=torch.float32,
                                 device="mps")
  qp, qv = tensor(qpos), tensor(qvel)
  dynamics = program.run_device(qp, qv)
  values = torch.zeros((2, max(layout.edge_count, 1)), dtype=torch.float32,
                       device="mps")
  writer = MetalVelocityDerivativeValues(layout, 2, values)
  result = program.bias_derivative_device(qp, qv, dynamics,
                                          edge_values=writer.values)
  torch.mps.synchronize()
  expected_coo = np.zeros((2, max(layout.edge_count, 1)), dtype=np.float32)
  expected_coo[:, :layout.edge_count] = np.asarray(expected)[:,
      layout.edge_rows, layout.edge_cols]
  np.testing.assert_allclose(result.cpu().numpy(), expected_coo,
                             atol=5e-4, rtol=3e-4)
  assert "bias_derivative" not in program._workspace
