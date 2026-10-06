# Copyright 2026 The MuJoCo Metal contributors
# Licensed under the Apache License, Version 2.0.
"""Inverse equality forces require the complete articulated velocity stage.

CPU tests execute stateless inverse assembly with pinned MuJoCo stage
outputs. They qualify tensor plumbing and the row-gradient reduction only;
the separately marked native tests qualify actual Metal physics.
"""
import os
from types import SimpleNamespace

import mujoco
import numpy as np
import pytest


def _fixture(kind, endpoints):
  eq = (f'<{kind} site1="s1" site2="s2"/>' if endpoints == "site" else
        f'<{kind} body1="b1" body2="b2"' +
        (' anchor=".5 .1 .4"/>' if kind == "connect" else '/>'))
  model = mujoco.MjModel.from_xml_string(f'''<mujoco>
    <option gravity="0 0 0" solver="PGS" iterations="0"><flag contact="disable"/></option>
    <worldbody>
      <body quat=".9659258263 0 .2588190451 0"><joint axis="0 0 1"/>
        <geom size=".1" mass="1" pos=".2 .1 0"/>
        <body name="b1" pos=".4 .1 .2"><joint axis="0 1 0"/>
          <geom size=".08" mass=".7" pos=".1 .2 .15"/>
          <site name="s1" pos=".2 .15 .1" quat=".9238795325 .3826834324 0 0"/>
        </body>
      </body>
      <body pos="1 -.2 .5" quat=".9238795325 .3826834324 0 0"><joint axis="1 0 0"/>
        <geom size=".1" mass="1.2" pos=".1 -.15 .2"/>
        <body name="b2" pos=".3 .2 .1"><joint axis="0 0 1"/>
          <geom size=".08" mass=".6" pos=".2 .1 -.1"/>
          <site name="s2" pos="-.1 .2 .15" quat=".9659258263 0 0 .2588190451"/>
        </body>
      </body>
    </worldbody><equality>{eq}</equality></mujoco>''')
  data = mujoco.MjData(model)
  data.qpos[:] = [.35, -.4, .2, .45]
  data.qvel[:] = [.7, -.8, .45, -.6]
  data.qacc[:] = [.4, -.7, .3, -.2]
  mujoco.mj_inverse(model, data)
  base = (-data.efc_KBIP[:, 1] * data.efc_vel - data.efc_KBIP[:, 0]
          * data.efc_KBIP[:, 2] * (data.efc_pos - data.efc_margin))
  assert np.max(np.abs(data.efc_aref - base)) > .05
  assert np.linalg.norm(data.qfrc_constraint) > 1
  return model, data


@pytest.mark.parametrize("enabled", [False, True])
@pytest.mark.parametrize("kind", ["connect", "weld"])
@pytest.mark.parametrize("endpoints", ["body", "site"])
def test_stateless_inverse_assembly_passes_complete_equality_motion(kind, endpoints, enabled):
  torch = pytest.importorskip("torch")
  from mujoco_metal.native_api import _inverse_impl, _inverse_query_workspaces

  model, data = _fixture(kind, endpoints)
  def tensor(array):
    return torch.tensor(np.asarray(array).copy()[None], dtype=torch.float32)
  mass = np.empty((model.nv, model.nv))
  mujoco.mj_fullM(model, data, mass)
  dynamics = {"mass_matrix": tensor(mass), "qfrc_bias": tensor(data.qfrc_bias),
              "root_com": tensor(data.subtree_com), "cvel": tensor(data.cvel),
              "cdof": tensor(data.cdof), "cdof_dot": tensor(data.cdof_dot),
              "poses": {"body_pos": tensor(data.xpos), "body_quat": tensor(data.xquat)}}
  nr, nv = data.nefc, model.nv
  rows = torch.zeros((1, nr*nr + 7*nr), dtype=torch.float32)
  rows[0, nr*nr:nr*nr+nr] = tensor(data.efc_R)[0]
  rows[0, nr*nr+nr:nr*nr+2*nr] = tensor(data.efc_aref)[0]
  # The input workspace is deliberately unlike the temporary oracle assembly.
  original = {"workspace_J": torch.zeros((1, nr*nv)),
              "workspace_debug": torch.zeros_like(rows)}
  saved = {key: value.clone() for key, value in original.items()}
  seen = []
  def assemble(poses, _qp, _qv, **kwargs):
    assert poses["root_com"] is dynamics["root_com"]
    for name in ("cvel", "cdof", "cdof_dot"):
      assert kwargs[name] is dynamics[name]
    original["workspace_J"].copy_(tensor(data.efc_J.reshape(nr, nv)).reshape(1, -1))
    original["workspace_debug"].copy_(rows)
    cc._position_current_valid = True
    cc._assembly_generation += 1
    seen.append(True)
    return {"active": torch.full((1, nr), int(enabled), dtype=torch.float32),
            "J": original["workspace_J"].view(1, nr, nv),
            "R": rows[:, nr*nr:nr*nr+nr], "ar": rows[:, nr*nr+nr:nr*nr+2*nr],
            "lo": rows[:, nr*nr+4*nr:nr*nr+5*nr], "hi": rows[:, nr*nr+5*nr:nr*nr+6*nr]}
  cc = SimpleNamespace(_workspace=original, assemble_device=assemble,
      _position_current_valid=False, _assembly_generation=7,
      descriptor=SimpleNamespace(nr=nr, nv=nv, n_eq_rows=nr, ncontacts_max=0, cone_type=0))
  state = SimpleNamespace(_qpos=tensor(data.qpos), _qvel=tensor(data.qvel),
      _qacc=tensor(data.qacc), _eq_active=torch.ones((1, model.neq), dtype=torch.int32))
  sim = SimpleNamespace(state=state, batch_size=1, model=model, _mjmodel=model,
      _smooth=SimpleNamespace(run_device=lambda *_args: dynamics), _passive=None,
      _coupled_constraints=cc)
  with _inverse_query_workspaces(sim):
    result = _inverse_impl(sim, None, None, None, None, None)
  assert seen == [True]
  if not enabled:
    # The native reserved-row layout keeps rows which the CPU omits entirely.
    # Positive saved R/J/aref values must not create a force for inactive rows.
    data.eq_active[:] = 0
    mujoco.mj_inverse(model, data)
  np.testing.assert_allclose(result.numpy()[0], data.qfrc_inverse, atol=2e-2, rtol=2e-4)
  for name in saved:
    torch.testing.assert_close(original[name], saved[name], rtol=0, atol=0)
  assert cc._position_current_valid is False
  assert cc._assembly_generation == 7


@pytest.mark.parametrize("fail", [False, True])
def test_inverse_query_preserves_nested_borrowed_stages_on_success_or_failure(fail):
  torch = pytest.importorskip("torch")
  from mujoco_metal.native_api import _inverse_query_workspaces

  class Stage:
    pass
  # Use the same subordinate-program discovery contract as real FK stages.
  Stage.__module__ = "mujoco_metal.test_query_stage"
  fk, smooth = Stage(), Stage()
  fk._workspace = {"outputs": {"body_pos": torch.tensor([[1., 2., 3.]])}}
  smooth._fk = fk
  smooth._workspace = {"mass": torch.tensor([[[2., .3], [.3, 1.]]])}
  smooth._cache_epoch = 4
  fk_pos = fk._workspace["outputs"]["body_pos"]
  mass = smooth._workspace["mass"]
  saved_pos, saved_mass = fk_pos.clone(), mass.clone()
  sim = SimpleNamespace(_smooth=smooth, _spatial_kin=None, _spatial_cache_key=(1, 2))
  def query():
    with _inverse_query_workspaces(sim):
      mass.mul_(7)
      fk_pos.add_(3)
      smooth._workspace["mass"] = torch.zeros_like(mass)
      smooth._workspace["new_view"] = fk_pos
      smooth._cache_epoch = 9
      fk._workspace = {"outputs": {"body_pos": torch.ones_like(fk_pos)}}
      sim._spatial_kin, sim._spatial_cache_key = {}, None
      if fail:
        raise RuntimeError("temporary forward failed")
  if fail:
    with pytest.raises(RuntimeError, match="temporary forward failed"):
      query()
  else:
    query()
  assert smooth._workspace["mass"] is mass
  assert fk._workspace["outputs"]["body_pos"] is fk_pos
  assert set(smooth._workspace) == {"mass"}
  assert smooth._cache_epoch == 4
  torch.testing.assert_close(mass, saved_mass, rtol=0, atol=0)
  torch.testing.assert_close(fk_pos, saved_pos, rtol=0, atol=0)
  assert sim._spatial_kin is None
  assert sim._spatial_cache_key == (1, 2)


@pytest.mark.parametrize("fail", [False, True])
def test_inverse_query_preserves_prepared_forward_record_and_shared_force_storage(fail):
  torch = pytest.importorskip("torch")
  from mujoco_metal.forward_stages import ForwardStage, ForwardStageCoordinator
  from mujoco_metal.native_api import _inverse_query_workspaces
  sim = SimpleNamespace(_forward_stage_passive_force=torch.tensor([[.5, -.7]]),
      _forward_stage_actuator_force=torch.tensor([[1., 2.]]))
  stages = sim._forward_stages = ForwardStageCoordinator(sim)
  record = stages.begin(generation=4, qpos=torch.tensor([[.2, .3]]), position={})
  stages.publish(record, ForwardStage.VEL, {"qfrc_passive": sim._forward_stage_passive_force})
  stages.publish(record, ForwardStage.ACT, {"qfrc_actuator": sim._forward_stage_actuator_force})
  passive, actuator = sim._forward_stage_passive_force, sim._forward_stage_actuator_force
  epoch = stages.epoch
  def query():
    with _inverse_query_workspaces(sim):
      passive.add_(9)
      actuator.mul_(7)
      sim._forward_stage_passive_force = torch.zeros_like(passive)
      record.values[ForwardStage.VEL]["qfrc_passive"] = torch.zeros_like(passive)
      stages.begin(generation=4, qpos=torch.zeros_like(record.qpos), position={})
      if fail:
        raise RuntimeError("temporary query failed")
  if fail:
    with pytest.raises(RuntimeError, match="temporary query failed"):
      query()
  else:
    query()
  assert stages.current(generation=4, minimum=ForwardStage.ACT) is record
  assert stages.epoch == epoch
  assert record.values[ForwardStage.VEL]["qfrc_passive"] is passive
  assert record.values[ForwardStage.ACT]["qfrc_actuator"] is actuator
  assert sim._forward_stage_passive_force is passive
  assert sim._forward_stage_actuator_force is actuator
  torch.testing.assert_close(passive, torch.tensor([[.5, -.7]]), rtol=0, atol=0)
  torch.testing.assert_close(actuator, torch.tensor([[1., 2.]]), rtol=0, atol=0)


@pytest.mark.gpu
@pytest.mark.skipif(os.getenv("MUJOCO_METAL_RUN_GPU") != "1", reason="native opt-in")
@pytest.mark.parametrize("kind", ["connect", "weld"])
@pytest.mark.parametrize("endpoints", ["body", "site"])
def test_native_inverse_articulated_equality_matches_pinned_force(kind, endpoints):
  from mujoco_metal import MetalSimulation
  from mujoco_metal.native_api import mj_inverse
  model, data = _fixture(kind, endpoints)
  sim = MetalSimulation(model, qpos=data.qpos[None].astype(np.float32),
      qvel=data.qvel[None].astype(np.float32), profile="integrated_euler_v1")
  result = mj_inverse(sim, qacc=data.qacc[None].astype(np.float32))
  np.testing.assert_allclose(result.cpu().numpy()[0], data.qfrc_inverse,
                             atol=2e-2, rtol=2e-4)


@pytest.mark.gpu
@pytest.mark.skipif(os.getenv("MUJOCO_METAL_RUN_GPU") != "1", reason="native opt-in")
@pytest.mark.parametrize("fail", [False, True])
def test_native_inverse_preserves_borrowed_mass_and_cached_equality_token(fail, monkeypatch):
  import torch
  from mujoco_metal import MetalSimulation
  from mujoco_metal.native_api import mj_inverse
  model, data = _fixture("weld", "body")
  sim = MetalSimulation(model, qpos=data.qpos[None].astype(np.float32),
      qvel=data.qvel[None].astype(np.float32), profile="integrated_euler_v1")
  _, status, dynamics = sim._acceleration(sim.state._qpos, sim.state._qvel)
  np.testing.assert_array_equal(status.cpu().numpy(), 0)
  context = sim._capture_forward_position(sim.state._qpos, dynamics)
  cc = sim._coupled_constraints
  mass = dynamics["mass_matrix"]
  body_pos = dynamics["poses"]["body_pos"]
  saved_mass, saved_pos = mass.clone(), body_pos.clone()
  saved_epoch = cc._position_context_epoch
  saved_generation = cc._assembly_generation
  query_qpos = (data.qpos + np.array([-.2, .3, -.1, -.2]))[None].astype(np.float32)
  if fail:
    # The paired inverse reducer no longer uses torch.bmm. Inject at its
    # actual dense/CSR transpose operation so rollback is exercised after rows
    # have been assembled.
    import mujoco_metal.inverse_constraints as inverse_constraints
    calls = []
    def fail_after_assembly(*args, **kwargs):
      calls.append(True)
      raise RuntimeError("inverse paired gradient failed")
    with monkeypatch.context() as patch:
      patch.setattr(inverse_constraints,
                    "_dense_jacobian_transpose_pair_matvec",
                    fail_after_assembly)
      patch.setattr(inverse_constraints,
                    "_packed_jacobian_transpose_pair_matvec",
                    fail_after_assembly)
      with pytest.raises(RuntimeError, match="inverse paired gradient failed"):
        mj_inverse(sim, qpos=query_qpos)
    assert calls
  else:
    mj_inverse(sim, qpos=query_qpos)
  torch.testing.assert_close(mass, saved_mass, rtol=0, atol=0)
  torch.testing.assert_close(body_pos, saved_pos, rtol=0, atol=0)
  assert cc._position_context_epoch == saved_epoch
  assert cc._assembly_generation == saved_generation
  velocity = np.array([-.4, .65, -.55, .3], np.float32)
  # _fixture prepared inverse position, which omits forward island/projected
  # constraint storage. Initialize that source-owned position prefix before
  # asking mj_forwardSkip to reuse it; skipping an unprepared forward stage
  # is invalid and can dereference an absent solver arena allocation.
  mujoco.mj_fwdPosition(model, data)
  data.qvel[:] = velocity
  mujoco.mj_forwardSkip(model, data, mujoco.mjtStage.mjSTAGE_POS, 1)
  _, status, _ = sim._acceleration(sim.state._qpos,
      torch.tensor(velocity[None], device="mps"),
      skip_sleep_prepare=True, position_context=context)
  np.testing.assert_array_equal(status.cpu().numpy(), 0)
  nr, ne = cc.descriptor.nr, data.ne
  debug = cc._workspace["workspace_debug"].reshape(1, cc._debug_stride)
  np.testing.assert_allclose(debug[0, nr*nr+nr:nr*nr+nr+ne].cpu().numpy(),
                             data.efc_aref, atol=2e-3, rtol=3e-5)


def test_pinned_forward_skip_oracle_prepares_forward_position_after_inverse():
  model, reused = _fixture("weld", "body")
  original_qpos = reused.qpos.copy()
  velocity = np.asarray([-.4, .65, -.55, .3])
  mujoco.mj_fwdPosition(model, reused)
  reused.qvel[:] = velocity
  mujoco.mj_forwardSkip(model, reused, mujoco.mjtStage.mjSTAGE_POS, 1)
  fresh = mujoco.MjData(model)
  fresh.qpos[:] = original_qpos
  fresh.qvel[:] = velocity
  mujoco.mj_forward(model, fresh)
  assert fresh.nefc == reused.nefc == 6
  assert np.linalg.norm(fresh.efc_aref) > .1
  np.testing.assert_array_equal(reused.qpos, original_qpos)
  np.testing.assert_allclose(reused.efc_aref, fresh.efc_aref, rtol=0, atol=1e-12)
  np.testing.assert_allclose(reused.qfrc_bias, fresh.qfrc_bias, rtol=0, atol=1e-12)
  np.testing.assert_allclose(reused.qacc, fresh.qacc, rtol=0, atol=1e-12)
