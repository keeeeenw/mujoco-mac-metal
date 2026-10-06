# Copyright 2026 The MuJoCo Metal contributors
# Licensed under the Apache License, Version 2.0.
"""Contracts for the guarded Magnetic and SiteFeedback FORCE kernels."""

from pathlib import Path
from types import SimpleNamespace

import mujoco
import pytest


ROOT = Path(__file__).parents[1]
XML = """<mujoco><worldbody>
  <body name="arm"><joint name="hinge" type="hinge" axis="0 1 0"/>
    <geom type="capsule" size=".04 .2" mass="1"/>
    <site name="tip" pos="0 0 .2"/>
  </body>
</worldbody></mujoco>"""


def test_builtin_force_kernel_returns_before_stage_reads_or_output_writes():
  source = (ROOT / "mujoco_metal" / "shaders" /
            "spatial_plugin_force.metal").read_text()
  for name in ("masked_magnetic_force", "masked_site_feedback_force"):
    body = source.split(f"kernel void {name}(", 1)[1].split("\n}", 1)[0]
    guard = body.index("if (world >= batch || world_mask[world] == 0) return;")
    after_guard = body[guard + len(
        "if (world >= batch || world_mask[world] == 0) return;"):]
    assert "point_velocity(cvel, root_com" in after_guard
    assert "qfrc[world * nv + dof]" in after_guard


def test_magnetic_masked_dispatch_is_delegated_to_guarded_spatial_query():
  torch = pytest.importorskip("torch")
  from mujoco_metal.extensions import CustomMagneticForcePlugin

  model = mujoco.MjModel.from_xml_string(XML)
  plugin = CustomMagneticForcePlugin(charge=.75, b_field=(.2, -.3, .8))
  plugin.init(model, batch_size=2, device="cpu")
  mask = torch.tensor([True, False])
  expected = torch.tensor([[1.25], [9.5]], dtype=torch.float32)

  class GuardedAdapter:
    def masked_magnetic_force(self, dynamics, **kwargs):
      assert dynamics == {"stage": "current"}
      assert kwargs["compute_mask"] is mask
      assert kwargs["body_ids"] is plugin._body_ids_device
      return expected

  plugin._queries = GuardedAdapter()
  plugin.device = torch.device("mps")
  assert plugin.run_device(SimpleNamespace(), dynamics={"stage": "current"},
                           compute_mask=mask) is expected


def test_site_feedback_masked_dispatch_is_delegated_to_guarded_spatial_query():
  torch = pytest.importorskip("torch")
  from mujoco_metal.site_feedback import SiteFeedbackPlugin

  model = mujoco.MjModel.from_xml_string(XML)
  plugin = SiteFeedbackPlugin(site="tip")
  plugin.init(model, batch_size=2, device="cpu")
  mask = torch.tensor([True, False])
  time = torch.tensor([.1, .2], dtype=torch.float32)
  expected = torch.tensor([[.4], [8.]], dtype=torch.float32)

  class GuardedAdapter:
    def masked_site_feedback_force(self, dynamics, **kwargs):
      assert dynamics == {"stage": "current"}
      assert kwargs["compute_mask"] is mask
      assert kwargs["time"] is time
      assert kwargs["site_id"] == plugin.site_id
      assert kwargs["body_id"] == plugin.body_id
      return expected

  plugin._queries = GuardedAdapter()
  plugin.device = torch.device("mps")
  state = SimpleNamespace(_time=time)
  assert plugin.run_device(state, dynamics={"stage": "current"},
                           compute_mask=mask) is expected


def test_spatial_force_capacity_follows_registered_builtin_instance_counts():
  from mujoco_metal.capacity import estimate_capacity
  from mujoco_metal.extensions import (
      CustomMagneticForcePlugin, ExtensionRegistry, PluginType)
  from mujoco_metal.site_feedback import SiteFeedbackPlugin

  registry = ExtensionRegistry()
  registry.register(CustomMagneticForcePlugin(name="mag_a"))
  registry.register(CustomMagneticForcePlugin(name="mag_b", body="arm"))
  registry.register_factory(
      "feedback_a", PluginType.FORCE,
      lambda: SiteFeedbackPlugin(name="feedback_a"),
      spatial_force_kind="site_feedback")
  counts = registry.spatial_force_workspace_counts()
  assert counts == {"magnetic": 2, "site_feedback": 1}
  model = mujoco.MjModel.from_xml_string(XML)
  estimate = estimate_capacity(
      model, 2, npairs=0, nslots=0, nr=0,
      magnetic_force_plugins=counts["magnetic"],
      site_feedback_plugins=counts["site_feedback"])
  runtime = dict(estimate.memory_breakdown)
  instances = 3
  assert runtime["spatial_plugin_force.chain"] == (
      instances * model.nbody * model.nv * 4)
  assert runtime["spatial_plugin_force.output"] == (
      instances * 2 * max(model.nv, 1) * 4)
  assert runtime["spatial_plugin_force.magnetic_body_ids"] == (
      2 * max(model.nbody - 1, 1) * 4)
  assert runtime["spatial_plugin_force.query_parameters"] == 3 * 3 * 4
  assert runtime["spatial_plugin_force.magnetic_field"] == 2 * 3 * 4
  assert runtime["spatial_plugin_force.site_feedback_parameters"] == 14 * 4


def test_spatial_jacobian_masked_refresh_keeps_healthy_rows_and_cache_identity():
  import torch
  from mujoco_metal.simulation import MetalSimulation

  model = mujoco.MjModel.from_xml_string(XML)
  qvel = torch.tensor([[1.0], [2.0]], dtype=torch.float32)
  body_pos = torch.zeros((2, model.nbody, 3), dtype=torch.float32)
  poses = {"body_pos": body_pos}
  old = torch.tensor([[[3.0]], [[4.0]]], dtype=torch.float32)

  class Spatial:
    def __init__(self):
      self.calls = []

    def run_kinematics(self, velocities, pose_values, *, world_mask=None):
      self.calls.append(world_mask)
      if world_mask is None:
        old[:, 0, 0].copy_(velocities[:, 0])
      else:
        active = world_mask.to(torch.bool)
        old[active, 0, 0] = velocities[active, 0]
      return {"jacobian": old}

  spatial = Spatial()
  simulation = SimpleNamespace(
      _spatial_tendons=spatial,
      _state=SimpleNamespace(generation=7,
                             _reset_mask_i32=torch.tensor([1, 0], dtype=torch.int32)),
      _spatial_cache_key=(7, id(qvel), id(body_pos)),
      _spatial_kin={"jacobian": old})
  mask = torch.tensor([True, False])
  actual = MetalSimulation._spatial_jacobian(simulation, qvel, poses, mask)
  assert actual is old
  assert spatial.calls == [simulation._state._reset_mask_i32]
  assert old[:, 0, 0].tolist() == [1.0, 4.0]
  assert simulation._spatial_cache_key == (7, id(qvel), id(body_pos))

  # A cache miss followed by an all-healthy mask must not mark an incomplete
  # tensor valid; the next unmasked stage computes every row.
  simulation._spatial_cache_key = None
  old[:, 0, 0].fill_(9.0)
  simulation._state._reset_mask_i32.zero_()
  MetalSimulation._spatial_jacobian(
      simulation, qvel, poses, torch.tensor([False, False]))
  assert simulation._spatial_cache_key is None
  assert old[:, 0, 0].tolist() == [9.0, 9.0]
  MetalSimulation._spatial_jacobian(simulation, qvel, poses)
  assert old[:, 0, 0].tolist() == [1.0, 2.0]
  assert simulation._spatial_cache_key == (7, id(qvel), id(body_pos))
