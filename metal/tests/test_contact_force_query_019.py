# Copyright 2026 The MuJoCo Metal contributors
# Licensed under the Apache License, Version 2.0.
"""Contact-frame force query ownership and independent native solver oracles."""
import os
from types import SimpleNamespace

import mujoco
import numpy as np
import pytest


def _prepared(result, batch=2, contact=None):
  torch = pytest.importorskip("torch")
  from mujoco_metal.forward_stages import ForwardStage, ForwardStageCoordinator
  sim = SimpleNamespace(state=SimpleNamespace(generation=1, _device="cpu"),
                        batch_size=batch, _contact=contact)
  sim._forward_stages = ForwardStageCoordinator(sim)
  record = sim._forward_stages.begin(
      generation=1, qpos=torch.zeros((batch, 1)), position={"poses": {}})
  for stage in (ForwardStage.VEL, ForwardStage.ACT, ForwardStage.ACC):
    sim._forward_stages.publish(record, stage, {})
  sim._forward_stages.publish(record, ForwardStage.CONSTRAINT, result)
  sim.validate_forward_stage_record = lambda record, minimum: (
      sim._forward_stages.validate(record, generation=sim.state.generation,
                                  minimum=minimum))
  return sim, record


def test_contact_force_query_masks_owns_and_validates_prepared_storage():
  torch = pytest.importorskip("torch")
  from mujoco_metal.native_api import mj_contactForce
  wrench = torch.arange(24, dtype=torch.float32).reshape(2, 2, 6)
  mask = torch.tensor([[1., 0.], [1., 1.]])
  status = torch.tensor([0, 2], dtype=torch.int32)
  sim, record = _prepared({"contact_wrench": wrench, "contact_mask": mask,
                           "status": status})
  result = mj_contactForce(sim, 0, record=record)
  torch.testing.assert_close(result[0], wrench[0, 0], rtol=0, atol=0)
  assert not result[1].any()
  assert result.data_ptr() != wrench.data_ptr()
  saved = result.clone()
  wrench.add_(100)
  torch.testing.assert_close(result, saved, rtol=0, atol=0)
  for slot in (-1, 1, 2, 100000):
    assert not mj_contactForce(sim, slot).any()
  for slot in (True, np.bool_(False), 1.5, "0"):
    with pytest.raises(TypeError, match="integer"):
      mj_contactForce(sim, slot)
  record.qpos.add_(1)
  with pytest.raises(ValueError, match="mutated"):
    mj_contactForce(sim, 0, record=record)


@pytest.mark.parametrize("condim", [1, 3])
def test_legacy_contact_rows_decode_matches_pinned_source(condim):
  torch = pytest.importorskip("torch")
  from mujoco_metal.native_api import mj_contactForce
  model = mujoco.MjModel.from_xml_string(f'''<mujoco>
    <option cone="pyramidal"/><worldbody><geom type="plane" size="1 1 .1" condim="{condim}"/>
    <body pos="0 0 .095"><freejoint/><geom type="sphere" size=".1"
      condim="{condim}" friction=".7 .1 .05"/></body></worldbody></mujoco>''')
  data = mujoco.MjData(model)
  data.qvel[:] = [.1, -.2, -.05, .3, .1, -.2]
  mujoco.mj_forward(model, data)
  assert data.ncon == 1
  con = data.contact[0]
  assert con.efc_address >= 0
  rows = np.zeros((2, 1, 5), dtype=np.float32)
  if condim == 1:
    rows[:, 0, 0] = data.efc_force[con.efc_address]
  else:
    rows[:, 0, 1:] = data.efc_force[con.efc_address:con.efc_address + 4]
  contact = SimpleNamespace(_constants={
      "friction": torch.tensor(con.friction[:2].copy(), dtype=torch.float32),
      "condim": torch.tensor([condim], dtype=torch.int32)})
  result = {"contact_result": {"force_rows": torch.tensor(rows),
                              "mask": torch.ones((2, 1))},
            "status": torch.zeros(2, dtype=torch.int32)}
  sim, _ = _prepared(result, contact=contact)
  expected = np.zeros(6)
  mujoco.mj_contactForce(model, data, 0, expected)
  np.testing.assert_allclose(mj_contactForce(sim, 0).numpy(),
                             np.repeat(expected[None], 2, axis=0),
                             rtol=2e-6, atol=2e-6)


def test_contact_free_force_and_missing_legacy_result_contract():
  torch = pytest.importorskip("torch")
  from mujoco_metal.native_api import mj_contactForce
  sim, _ = _prepared({"status": torch.zeros(2, dtype=torch.int32)})
  assert not mj_contactForce(sim, 0).any()
  sim._contact = object()
  with pytest.raises(RuntimeError, match="publish"):
    mj_contactForce(sim, 0)


@pytest.mark.gpu
@pytest.mark.skipif(os.getenv("MUJOCO_METAL_RUN_GPU") != "1", reason="native opt-in")
@pytest.mark.parametrize("cone,condim,profile", [
    (cone, condim, "integrated_euler_v1")
    for cone in ("pyramidal", "elliptic") for condim in (1, 3, 4, 6)
] + [("pyramidal", 1, "normal_contact_euler_v1"),
     ("pyramidal", 3, "friction_contact_euler_v1")])
def test_native_contact_force_public_forward_matches_cpu(cone, condim, profile, monkeypatch):
  import torch
  from mujoco_metal.simulation import MetalSimulation
  from mujoco_metal.native_api import mj_contactForce, mj_forward
  model = mujoco.MjModel.from_xml_string(f'''<mujoco>
    <option cone="{cone}" solver="Newton" iterations="100" tolerance="1e-8"/>
    <worldbody><geom name="floor" type="plane" size="2 2 .1" condim="{condim}"/>
      <body pos="0 0 .095"><freejoint/>
        <geom name="ball" type="sphere" size=".1" mass="1" condim="{condim}"
          friction=".7 .1 .05"/></body></worldbody></mujoco>''')
  velocities = np.asarray([[.1, -.2, -.05, .3, .1, -.2],
                           [-.15, .1, -.03, -.1, -.3, .2]], dtype=np.float32)
  expected = []
  for velocity in velocities:
    data = mujoco.MjData(model)
    data.qpos[:] = np.asarray(model.qpos0, dtype=np.float32)
    data.qvel[:] = velocity
    mujoco.mj_forward(model, data)
    assert data.ncon == 1
    value = np.zeros(6)
    mujoco.mj_contactForce(model, data, 0, value)
    expected.append(value)
  sim = MetalSimulation(model, batch_size=2, profile=profile)
  sim.reset(qvel=velocities)
  def no_cpu_physics(*args, **kwargs):
    raise AssertionError("native query executed CPU physics")
  monkeypatch.setattr(mujoco, "mj_forward", no_cpu_physics)
  monkeypatch.setattr(mujoco, "mj_step", no_cpu_physics)
  mj_forward(sim, skipsensor=True)
  record = sim._forward_stages._record
  from mujoco_metal.forward_stages import ForwardStage
  system = record.values[ForwardStage.CONSTRAINT]
  assert not system["status"].cpu().numpy().any()
  count = (system["contact_wrench"].shape[1] if "contact_wrench" in system else
           system["contact_result"]["force_rows"].shape[1])
  values = [mj_contactForce(sim, slot) for slot in range(count)]
  actual = torch.stack(values, dim=1).sum(dim=1)
  np.testing.assert_allclose(actual.cpu().numpy(), np.asarray(expected),
                             rtol=2e-3, atol=2e-3)
  assert not mj_contactForce(sim, -1).any()
  assert not mj_contactForce(sim, count).any()
  before = values[0].clone()
  sim.reset(qvel=velocities * 2)
  torch.testing.assert_close(values[0], before, rtol=0, atol=0)
  with pytest.raises(ValueError, match="stale"):
    mj_contactForce(sim, 0, record=record)
