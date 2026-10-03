# Copyright 2026 The MuJoCo Metal contributors
# Licensed under the Apache License, Version 2.0.
"""Immutable registrations and simulation-local native extension ownership."""

import os

import mujoco
import numpy as np
import pytest

from mujoco_metal.extensions import ExtensionRegistry, NativePlugin, PluginType, default_registry


class ConfiguredForce(NativePlugin):
  def __init__(self, name="owned_force", value=2):
    super().__init__(name, PluginType.FORCE)
    self.config = np.array([value], dtype=np.float32)
    self.calls = 0

  def run_device(self, state, **kwargs):
    import torch
    self.calls += 1
    return torch.full((self.batch_size, self.model.nv), float(self.config[0]),
                      dtype=torch.float32, device=self.device)


def _model():
  return mujoco.MjModel.from_xml_string('''<mujoco><option gravity="0 0 0"/>
    <worldbody><body><joint type="slide" axis="1 0 0"/>
    <geom type="sphere" size=".1" mass="1" contype="0" conaffinity="0"/>
    </body></worldbody></mujoco>''')


def test_registration_snapshots_configuration_and_binds_distinct_instances():
  registry = ExtensionRegistry()
  prototype = ConfiguredForce()
  registry.register(prototype)
  prototype.config[0] = 9
  m1, m2 = _model(), _model()
  first = registry.instantiate(m1, 1, "cpu")[0]
  second = registry.instantiate(m2, 2, "cpu")[0]
  assert first is not second and first is not prototype and second is not prototype
  assert first.model is m1 and second.model is m2
  assert first.batch_size == 1 and second.batch_size == 2
  assert first.config[0] == second.config[0] == 2
  first.config[0] = 5
  assert second.config[0] == 2
  assert prototype.model is None and prototype.device is None
  registry.clear()
  assert registry.instantiate(m1, 3, "cpu") == ()
  assert first.model is m1  # Registry changes don't rebind live consumers.


def test_factories_validate_role_binding_and_reused_instances():
  registry = ExtensionRegistry()
  shared = ConfiguredForce()
  registry.register_factory(shared.name, shared.plugin_type, lambda: shared)
  registry.instantiate(_model(), 1, "cpu")
  with pytest.raises(ValueError, match="fresh unbound"):
    registry.instantiate(_model(), 2, "cpu")
  bad = ExtensionRegistry()
  bad.register_factory("wrong_name", PluginType.FORCE, ConfiguredForce)
  with pytest.raises(ValueError, match="registered name and role"):
    bad.instantiate(_model(), 1, "cpu")
  with pytest.raises(TypeError, match="callable"):
    bad.register_factory("invalid", PluginType.SENSOR, None)


def test_bound_configuration_cannot_be_registered_as_a_prototype():
  plugin = ConfiguredForce()
  plugin.init(_model(), 1, "cpu")
  with pytest.raises(ValueError, match="unbound plugin"):
    ExtensionRegistry().register(plugin)


@pytest.mark.gpu
@pytest.mark.skipif(os.getenv("MUJOCO_METAL_RUN_GPU") != "1", reason="opt-in GPU")
def test_two_simulations_interleave_different_batches_without_plugin_rebinding():
  from mujoco_metal import MetalSimulation
  prototype = ConfiguredForce()
  default_registry.register(prototype)
  try:
    first = MetalSimulation(_model(), batch_size=1, profile="integrated_euler_v1")
    second = MetalSimulation(_model(), batch_size=2, profile="integrated_euler_v1")
    for _ in range(3):
      first.step()
      second.step()
    np.testing.assert_allclose(first.state.qacc.cpu().numpy(), [[2]], atol=1e-6)
    np.testing.assert_allclose(second.state.qacc.cpu().numpy(), [[2], [2]], atol=1e-6)
    assert first._native_plugins[0] is not second._native_plugins[0]
    assert prototype.calls == 0 and prototype.model is None
    default_registry.unregister(prototype.name)
    first.step()  # Existing instance remains valid after unregister.
    np.testing.assert_allclose(first.state.qacc.cpu().numpy(), [[2]], atol=1e-6)
  finally:
    default_registry.unregister(prototype.name)
