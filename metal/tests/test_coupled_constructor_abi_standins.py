# Copyright 2026 The MuJoCo Metal contributors
# Licensed under the Apache License, Version 2.0.

"""Host stand-in regression for the complete coupled constructor ordering."""

import sys
from types import SimpleNamespace

import mujoco
import numpy as np
import pytest


class _Tensor:
  def __init__(self, value):
    self.value = np.asarray(value)

  def numel(self):
    return self.value.size

  def __getitem__(self, item):
    return _Tensor(self.value[item])


class _Library:
  def __getattr__(self, name):
    return object()


class _TorchStandin:
  float32 = "float32"
  int32 = "int32"

  def __init__(self):
    self.backends = SimpleNamespace(
        mps=SimpleNamespace(is_available=lambda: True))
    self.mps = SimpleNamespace(compile_shader=lambda source: _Library())

  @staticmethod
  def device(name):
    return SimpleNamespace(type=name)

  @staticmethod
  def as_tensor(value, **kwargs):
    return _Tensor(value)

  @staticmethod
  def tensor(value, **kwargs):
    return _Tensor(value)

  @staticmethod
  def empty(shape, **kwargs):
    return _Tensor(np.empty(shape, dtype=np.int32))

  @staticmethod
  def zeros(shape, **kwargs):
    dtype = np.int32 if kwargs.get("dtype") == "int32" else np.float32
    return _Tensor(np.zeros(shape, dtype=dtype))


@pytest.mark.parametrize(("jacobian", "expected_mode"),
                         (("dense", 0), ("sparse", 1)))
def test_real_constructor_packs_dense_and_sparse_layout_before_constants(
    monkeypatch, jacobian, expected_mode):
  from mujoco_metal.coupled_constraints import MetalCoupledConstraints

  model = mujoco.MjModel.from_xml_string(
      f'<mujoco><option jacobian="{jacobian}"/><worldbody><body><joint type="slide"/>'
      '<geom type="sphere" size=".1" mass="1"/></body></worldbody></mujoco>')
  torch = _TorchStandin()
  monkeypatch.setitem(sys.modules, "torch", torch)
  monkeypatch.setattr(MetalCoupledConstraints, "prepare_workspace",
                      lambda self, batch_size: None)
  solver = MetalCoupledConstraints(model)
  assert solver._component_operator_layout_device is None
  assert solver._component_mass_storage == "dense"
  assert isinstance(solver._constants["solver_dims"], _Tensor)
  assert isinstance(solver._constants["body_dims"], _Tensor)
  body_dims = solver._constants["body_dims"].value
  # The recovery selector is a trailing [B] suffix; the Jacobian mode stays
  # in its original fixed header slot immediately before that suffix.
  assert int(body_dims[5]) == expected_mode
  np.testing.assert_array_equal(body_dims[6:], np.ones(solver.batch_size, np.int32))
  assert solver._jacobian_layout.mode == expected_mode
  # The real host packer must have completed all dense/default fields.
  dims = solver._constants["solver_dims"].value
  assert dims.dtype == np.int32
  assert int(dims[21]) > solver.descriptor.nr * solver.descriptor.nr
