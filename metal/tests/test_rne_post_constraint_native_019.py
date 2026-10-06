# Copyright 2026 The MuJoCo Metal contributors
# Licensed under the Apache License, Version 2.0.
"""Pinned CPU oracle and opt-in native API coverage for RNE post-constraint."""

import os
import ast
import re
from types import SimpleNamespace
from pathlib import Path

import mujoco
import numpy as np
import pytest


def _free_sphere_model():
  return mujoco.MjModel.from_xml_string('''<mujoco>
    <option timestep=".002" gravity="0 0 -9.81" solver="PGS" iterations="40"/>
    <worldbody>
      <geom name="floor" type="plane" size="2 2 .1" condim="3"/>
      <body name="ball" pos="0 0 .08"><freejoint/>
        <geom type="sphere" size=".1" mass="1" condim="3"/>
        <site name="imu" pos=".02 0 .01"/>
      </body>
    </worldbody>
    <sensor><accelerometer site="imu"/></sensor>
  </mujoco>''')


def _rich_rne_model(cone, condim):
  return mujoco.MjModel.from_xml_string(f'''<mujoco>
    <option timestep=".002" gravity="0 0 -9.81" cone="{cone}"
            solver="PGS" iterations="60"/>
    <worldbody>
      <geom name="floor" type="plane" size="2 2 .1" condim="{condim}"/>
      <body name="ball" pos="0 0 .095"><freejoint/>
        <geom name="ball_geom" type="sphere" size=".1" mass="1"
              friction="1 .7 .02"/>
        <site name="ball_site" pos=".02 .01 .03"/>
      </body>
      <body name="arm" pos=".22 0 .36">
        <joint name="hinge" type="hinge" axis="0 1 0"/>
        <geom type="capsule" size=".035 .18" mass=".3"/>
        <site name="arm_site" pos=".03 0 .16"/>
      </body>
    </worldbody>
    <contact><pair geom1="floor" geom2="ball_geom" condim="{condim}"/>
    </contact>
    <equality><connect body1="ball" body2="arm" anchor=".11 0 .35"/>
    </equality>
  </mujoco>''')


def _legacy_contact_model(condim):
  return mujoco.MjModel.from_xml_string(f'''<mujoco>
    <option timestep=".002" gravity="0 0 -9.81"/>
    <worldbody>
      <geom name="floor" type="plane" size="2 2 .1" condim="{condim}"/>
      <body pos="0 0 .095"><freejoint/>
        <geom name="ball" type="sphere" size=".1" mass="1"
              condim="{condim}" friction="1 .7 .02"/>
      </body>
    </worldbody>
  </mujoco>''')


def _sleeping_rne_model():
  return mujoco.MjModel.from_xml_string('''<mujoco>
    <option gravity="0 0 -9.81"><flag sleep="enable" contact="disable"/></option>
    <worldbody><body sleep="init">
      <joint type="slide" axis="0 0 1" damping=".2"/>
      <geom type="sphere" size=".1" mass="1" contype="0" conaffinity="0"/>
      <site pos=".02 0 0"/>
    </body></worldbody>
  </mujoco>''')


def test_pinned_rne_post_constraint_cpu_oracle_spatial_layout():
  """The CPU reference confirms output shapes and rot:lin world convention."""
  model = _free_sphere_model()
  data = mujoco.MjData(model)
  data.qpos[:] = model.qpos0
  data.qvel[:] = np.linspace(-.2, .3, model.nv)
  data.xfrc_applied[1] = np.asarray([.4, -.3, .2, 2., -1., 3.])
  mujoco.mj_forward(model, data)
  # Exercise the supplied-acceleration part of the public post stage instead
  # of the unconstrained forward solution, which cancels external wrench.
  data.qacc[:] = 0
  mujoco.mj_rnePostConstraint(model, data)
  assert data.flg_rnepost == 1
  assert data.cacc.shape == (model.nbody, 6)
  assert data.cfrc_ext.shape == (model.nbody, 6)
  assert data.cfrc_int.shape == (model.nbody, 6)
  np.testing.assert_allclose(data.cacc[0, 3:], -model.opt.gravity,
                             rtol=0, atol=1e-14)
  assert np.linalg.norm(data.cfrc_ext[1]) > 0
  assert np.linalg.norm(data.cfrc_int[1]) > 0


def test_rne_post_query_allocation_preflight_checks_products():
  from mujoco_metal.rne_post_constraint import (
      _check_query_budget, _legacy_contact_workspace_bytes,
      _query_memory_bytes, _rne_allocation_bytes, _rne_result_bytes)

  tiny = SimpleNamespace(nbody=2, nv=6, njnt=1, ngeom=2, nsite=1,
                         neq=0, nu=0)
  assert _rne_allocation_bytes(tiny, 3) > 0
  assert _rne_result_bytes(tiny, 0) > 0
  huge = SimpleNamespace(nbody=(1 << 30), nv=1, njnt=1, ngeom=0,
                         nsite=0, neq=0, nu=0)
  with pytest.raises(ValueError, match="int32"):
    _rne_allocation_bytes(huge, 2)
  legacy = SimpleNamespace(batch_size=3,
                           _contact=SimpleNamespace(
                               descriptor=SimpleNamespace(pair_count=7)))
  assert _legacy_contact_workspace_bytes(legacy) == 4 * (
      3 * 7 * 11 + 7 * 3 + 7 * 5 + 14 + 8)
  class CpuStandIn:
    pass
  sim = CpuStandIn()
  sim._scratch = np.zeros(17, dtype=np.float32)
  sim._native_plugins = ()
  required = _query_memory_bytes(sim, tiny, 3, None)
  assert required == 17 * 4 + _rne_allocation_bytes(tiny, 3)
  _check_query_budget(required, required)
  with pytest.raises(ValueError, match="needs"):
    _check_query_budget(required, required - 1)


def test_rne_post_velocity_stage_cvel_is_smooth_output_not_pose_member():
  from mujoco_metal.rne_post_constraint import _velocity_cvel

  cvel = np.arange(18, dtype=np.float32).reshape(1, 3, 6)
  dynamics = {"poses": {"body_pos": np.zeros((1, 3, 3))}, "cvel": cvel}
  assert _velocity_cvel(dynamics) is cvel
  with pytest.raises(ValueError, match="missing cvel"):
    _velocity_cvel({"poses": {"cvel": cvel}})


def test_rne_post_output_finalization_supports_empty_world_and_dof_axes():
  torch = pytest.importorskip("torch")
  from mujoco_metal.rne_post_constraint import _finish_rne_outputs

  empty = torch.empty((0, 0), dtype=torch.float32)
  raw_empty = {"cacc": torch.empty((0, 1, 6), dtype=torch.float32),
               "cfrc_int": torch.empty((0, 1, 6), dtype=torch.float32)}
  status_empty = torch.empty((0,), dtype=torch.int32)
  output = _finish_rne_outputs(raw_empty, empty, empty, status_empty)
  assert tuple(output["cacc"].shape) == (0, 1, 6)
  assert tuple(output["qacc"].shape) == (0, 0)
  assert tuple(output["status"].shape) == (0,)
  assert output["cacc"] is not raw_empty["cacc"]

  qvel = torch.empty((2, 0), dtype=torch.float32)
  qacc = torch.empty((2, 0), dtype=torch.float32)
  spatial = torch.ones((2, 1, 6), dtype=torch.float32)
  result = _finish_rne_outputs(
      {"cacc": spatial}, qvel, qacc,
      torch.tensor([0, 2], dtype=torch.int32))
  assert result["status"].tolist() == [0, 2]
  assert torch.equal(result["cacc"][0], spatial[0])
  assert torch.count_nonzero(result["cacc"][1]).item() == 0


def test_rne_post_query_rejects_mixed_stage_bundle_before_kernel_dispatch():
  from mujoco_metal.forward_stages import ForwardStage
  from mujoco_metal.rne_post_constraint import _validate_stage_result

  qacc = object()
  pos, vel = {}, {}
  constraint = {"qacc": qacc}
  record = SimpleNamespace(values={
      ForwardStage.POS: pos,
      ForwardStage.VEL: vel,
      ForwardStage.CONSTRAINT: constraint,
  })

  class Coordinator:
    def validate(self, candidate, stage):
      assert candidate is record
      assert stage == ForwardStage.CONSTRAINT
      return candidate

  class Sim:
    _forward_stages = Coordinator()

    @staticmethod
    def validate_forward_stage_record(candidate, stage):
      return Sim._forward_stages.validate(candidate, stage)

  bundle = {"record": record, "position": pos, "velocity": vel,
            "constraint": constraint, "qacc": qacc}
  assert _validate_stage_result(Sim(), bundle) == (
      record, pos, vel, constraint)
  foreign_qacc = object()
  with pytest.raises(ValueError, match="published stages"):
    _validate_stage_result(Sim(), dict(bundle, qacc=foreign_qacc))


@pytest.mark.parametrize("cone", ["pyramidal", "elliptic"])
@pytest.mark.parametrize("condim", [3, 4, 6])
def test_rne_post_rich_cpu_oracle_fixture_has_contact_equality_and_sites(
    cone, condim):
  model = _rich_rne_model(cone, condim)
  assert model.neq == 1 and model.nsite == 2
  data = mujoco.MjData(model)
  mujoco.mj_forward(model, data)
  assert data.ncon > 0
  assert data.ne > 0
  mujoco.mj_rnePostConstraint(model, data)
  assert np.isfinite(data.cacc).all()
  assert np.isfinite(data.cfrc_ext).all()
  assert np.isfinite(data.cfrc_int).all()


def test_rne_post_sleeping_cpu_fixture_matches_pinned_source():
  model = _sleeping_rne_model()
  data = mujoco.MjData(model)
  assert int(data.tree_asleep[0]) >= 0
  mujoco.mj_forward(model, data)
  mujoco.mj_rnePostConstraint(model, data)
  assert np.isfinite(data.cacc).all()
  assert np.isfinite(data.cfrc_ext).all()
  assert np.isfinite(data.cfrc_int).all()


def test_rne_post_world_only_cpu_oracle_zero_dofs():
  model = mujoco.MjModel.from_xml_string("<mujoco><worldbody/></mujoco>")
  assert model.nv == 0
  data = mujoco.MjData(model)
  mujoco.mj_forward(model, data)
  mujoco.mj_rnePostConstraint(model, data)
  assert data.cacc.shape == (1, 6)
  np.testing.assert_array_equal(
      data.cacc, np.concatenate((np.zeros((1, 3)), -model.opt.gravity[None]),
                                axis=1))
  assert data.cfrc_ext.shape == (1, 6)
  assert data.cfrc_int.shape == (1, 6)


def test_rne_post_query_dispatch_matches_native_kernel_abi():
  package = Path(__file__).parents[1] / "mujoco_metal"
  source = (package / "rne_post_constraint.py").read_text()
  tree = ast.parse(source)
  shader = (package / "shaders" / "sensors_rne.metal").read_text()
  for attribute, kernel in (("_rne_assemble", "assemble_cfrc_ext"),
                            ("_rne_post", "rne_post")):
    calls = [node for node in ast.walk(tree)
             if isinstance(node, ast.Call)
             and isinstance(node.func, ast.Attribute)
             and node.func.attr == attribute]
    declaration = re.search(
        r"^kernel void %s\((.*?)uint \w+ \[\[thread_position_in_grid\]\]"
        % re.escape(kernel), shader, re.S | re.M)
    assert declaration is not None, kernel
    slots = [int(value) for value in re.findall(
        r"\[\[buffer\((\d+)\)\]\]", declaration.group(1))]
    assert slots == list(range(len(slots))), (kernel, slots)
    assert len(calls) == 1 and len(calls[0].args) == len(slots), (
        kernel, len(calls), [len(call.args) for call in calls], slots)


@pytest.mark.gpu
@pytest.mark.skipif(os.getenv("MUJOCO_METAL_RUN_GPU") != "1",
                    reason="opt-in GPU")
def test_rne_post_constraint_native_matches_cpu_and_restores_query_state():
  import torch
  from mujoco_metal.forward_stages import ForwardStage
  from mujoco_metal.native_api import mj_rnePostConstraint
  from mujoco_metal.simulation import MetalSimulation

  model = _free_sphere_model()
  qpos = np.asarray(model.qpos0, dtype=np.float32).reshape(1, -1)
  qvel = np.linspace(-.2, .3, model.nv, dtype=np.float32).reshape(1, -1)
  wrench = np.zeros((1, model.nbody, 6), dtype=np.float32)
  wrench[0, 1] = [.4, -.3, .2, 2., -1., 3.]
  sim = MetalSimulation(model, batch_size=1, qpos=qpos, qvel=qvel,
                        profile="integrated_scalable_v1")
  sim._prepare_wrench(wrench)

  # Save persistent inputs and the published-stage graph. The query computes
  # native force stages internally and must leave both exactly coherent.
  state_before = {name: getattr(sim.state, name).clone() for name in
                  ("_qpos", "_qvel", "_qacc", "_time", "_qacc_warmstart")}
  old_record = sim._forward_stages._record
  result = mj_rnePostConstraint(sim)
  assert set(result) == {"cacc", "cfrc_ext", "cfrc_int", "subtree_com",
                         "qacc", "status"}
  for name, shape in (("cacc", (1, model.nbody, 6)),
                      ("cfrc_ext", (1, model.nbody, 6)),
                      ("cfrc_int", (1, model.nbody, 6)),
                      ("subtree_com", (1, model.nbody, 3)),
                      ("qacc", (1, model.nv)), ("status", (1,))):
    assert tuple(result[name].shape) == shape
    assert result[name].device.type == "mps"
  assert all(result[name].is_contiguous() for name in result)
  assert result["status"].item() == 0
  assert sim._forward_stages._record is old_record
  for name, before in state_before.items():
    assert torch.equal(getattr(sim.state, name), before), name

  cpu = mujoco.MjData(model)
  cpu.qpos[:] = qpos[0]
  cpu.qvel[:] = qvel[0]
  cpu.xfrc_applied[:] = wrench[0]
  mujoco.mj_forward(model, cpu)
  mujoco.mj_rnePostConstraint(model, cpu)
  for name in ("cacc", "cfrc_ext", "cfrc_int", "subtree_com"):
    np.testing.assert_allclose(result[name][0].cpu().numpy(),
                               getattr(cpu, name), rtol=2e-4, atol=3e-4,
                               err_msg=name)

  # A caller-owned coherent forward bundle is accepted without creating a
  # different stage state; supplied qacc is honored and output remains owned.
  from mujoco_metal.native_api import mj_forward
  forward = mj_forward(sim, skipsensor=True)
  custom_acc = forward["qacc"].clone()
  custom_acc.add_(.125)
  override = mj_rnePostConstraint(sim, forward_result=forward, qacc=custom_acc)
  torch.testing.assert_close(override["qacc"], custom_acc)
  assert override["qacc"].data_ptr() != custom_acc.data_ptr()
  assert forward["record"].values[ForwardStage.CONSTRAINT]["qacc"] is forward["qacc"]
  override_cpu = mujoco.MjData(model)
  override_cpu.qpos[:] = qpos[0]
  override_cpu.qvel[:] = qvel[0]
  override_cpu.xfrc_applied[:] = wrench[0]
  mujoco.mj_forward(model, override_cpu)
  override_cpu.qacc[:] = custom_acc.cpu().numpy()[0]
  mujoco.mj_rnePostConstraint(model, override_cpu)
  for name in ("cacc", "cfrc_ext", "cfrc_int", "subtree_com"):
    np.testing.assert_allclose(override[name][0].cpu().numpy(),
                               getattr(override_cpu, name),
                               rtol=2e-4, atol=3e-4,
                               err_msg=f"supplied qacc {name}")


@pytest.mark.gpu
@pytest.mark.skipif(os.getenv("MUJOCO_METAL_RUN_GPU") != "1",
                    reason="opt-in GPU")
def test_rne_post_constraint_sensor_free_lazy_workspace_batch_two():
  """The no-sensor path allocates only its bounded RNE workspace lazily."""
  import torch
  from mujoco_metal.native_api import mj_rnePostConstraint
  from mujoco_metal.simulation import MetalSimulation

  model = mujoco.MjModel.from_xml_string('''<mujoco>
    <option timestep=".002" gravity="0 0 -9.81">
      <flag contact="disable"/></option>
    <worldbody><body name="ball" pos="0 0 .4"><freejoint/>
      <geom type="sphere" size=".1" mass="1"/>
    </body></worldbody>
  </mujoco>''')
  assert model.nsensor == 0
  qpos = np.repeat(np.asarray(model.qpos0, dtype=np.float32)[None], 2, axis=0)
  qpos[1, 2] += .17
  qvel = np.asarray([[.1, -.2, .3, .4, -.5, .6],
                     [-.3, .2, -.1, .6, .2, -.4]], dtype=np.float32)
  sim = MetalSimulation(model, batch_size=2, qpos=qpos, qvel=qvel,
                        profile="integrated_euler_v1")
  assert sim._sensors is None
  result = mj_rnePostConstraint(sim)
  assert sim._sensors is None
  assert result["status"].tolist() == [0, 0]

  for w in range(2):
    cpu = mujoco.MjData(model)
    cpu.qpos[:] = qpos[w]
    cpu.qvel[:] = qvel[w]
    mujoco.mj_forward(model, cpu)
    mujoco.mj_rnePostConstraint(model, cpu)
    for name in ("cacc", "cfrc_ext", "cfrc_int", "subtree_com"):
      np.testing.assert_allclose(result[name][w].cpu().numpy(),
                                 getattr(cpu, name), rtol=2e-4, atol=3e-4,
                                 err_msg=f"{name}, world {w}")

  # The returned tensors are owned; later calls may reuse query-local scratch.
  saved = result["cacc"].clone()
  sim.reset(qvel=qvel * 0.5)
  mj_rnePostConstraint(sim)
  torch.testing.assert_close(result["cacc"], saved)


@pytest.mark.gpu
@pytest.mark.skipif(os.getenv("MUJOCO_METAL_RUN_GPU") != "1",
                    reason="opt-in GPU")
@pytest.mark.parametrize("condim,profile", [
    (1, "normal_contact_euler_v1"),
    (3, "friction_contact_euler_v1"),
])
def test_rne_post_constraint_legacy_contact_workspace(condim, profile):
  from mujoco_metal.native_api import mj_rnePostConstraint
  from mujoco_metal.simulation import MetalSimulation

  model = _legacy_contact_model(condim)
  qpos = np.asarray(model.qpos0, dtype=np.float32)[None]
  qvel = np.asarray([[.03, -.02, .01, .12, -.08, .04]], dtype=np.float32)
  sim = MetalSimulation(model, batch_size=1, qpos=qpos, qvel=qvel,
                        profile=profile)
  assert sim._contact is not None and sim._coupled_constraints is None
  result = mj_rnePostConstraint(sim)
  assert result["status"].cpu().tolist() == [0]

  cpu = mujoco.MjData(model)
  cpu.qpos[:] = qpos[0]
  cpu.qvel[:] = qvel[0]
  mujoco.mj_forward(model, cpu)
  assert cpu.ncon > 0
  mujoco.mj_rnePostConstraint(model, cpu)
  for name in ("cacc", "cfrc_ext", "cfrc_int", "subtree_com"):
    np.testing.assert_allclose(result[name][0].cpu().numpy(),
                               getattr(cpu, name), rtol=4e-4, atol=5e-4,
                               err_msg=f"{condim} {name}")


@pytest.mark.gpu
@pytest.mark.skipif(os.getenv("MUJOCO_METAL_RUN_GPU") != "1",
                    reason="opt-in GPU")
def test_rne_post_constraint_initially_sleeping_tree_matches_source():
  from mujoco_metal.native_api import mj_rnePostConstraint
  from mujoco_metal.simulation import MetalSimulation

  model = _sleeping_rne_model()
  cpu = mujoco.MjData(model)
  assert int(cpu.tree_asleep[0]) >= 0
  mujoco.mj_forward(model, cpu)
  mujoco.mj_rnePostConstraint(model, cpu)

  qpos = np.asarray(model.qpos0, dtype=np.float32)[None]
  qvel = np.zeros((1, model.nv), dtype=np.float32)
  sim = MetalSimulation(model, batch_size=1, qpos=qpos, qvel=qvel,
                        profile="integrated_scalable_v1")
  assert int(sim._sleep_schedule.tree_state[0, 0].item()) >= 0
  result = mj_rnePostConstraint(sim)
  assert result["status"].cpu().tolist() == [0]
  for name in ("cacc", "cfrc_ext", "cfrc_int", "subtree_com"):
    np.testing.assert_allclose(result[name][0].cpu().numpy(),
                               getattr(cpu, name), rtol=4e-4, atol=5e-4,
                               err_msg=name)


@pytest.mark.gpu
@pytest.mark.skipif(os.getenv("MUJOCO_METAL_RUN_GPU") != "1",
                    reason="opt-in GPU")
def test_rne_post_constraint_world_only_zero_dofs_native():
  from mujoco_metal.native_api import mj_rnePostConstraint
  from mujoco_metal.simulation import MetalSimulation

  model = mujoco.MjModel.from_xml_string("<mujoco><worldbody/></mujoco>")
  sim = MetalSimulation(model, batch_size=2, profile="integrated_euler_v1")
  result = mj_rnePostConstraint(sim)
  assert result["status"].cpu().tolist() == [0, 0]
  np.testing.assert_array_equal(result["qacc"].cpu().numpy(), np.zeros((2, 0)))
  cpu = mujoco.MjData(model)
  mujoco.mj_forward(model, cpu)
  mujoco.mj_rnePostConstraint(model, cpu)
  # The native output is float32; convert the pinned CPU reference to that
  # representable value before requiring exact equality (e.g. 9.81).
  expected_cacc = np.broadcast_to(
      np.asarray(cpu.cacc, dtype=np.float32), (2, 1, 6))
  np.testing.assert_array_equal(result["cacc"].cpu().numpy(), expected_cacc)
  np.testing.assert_allclose(result["cfrc_int"].cpu().numpy(),
                             np.zeros((2, 1, 6)), rtol=0, atol=0)


@pytest.mark.gpu
@pytest.mark.skipif(os.getenv("MUJOCO_METAL_RUN_GPU") != "1",
                    reason="opt-in GPU")
def test_rne_post_constraint_failed_override_world_is_zero_and_isolated():
  import torch
  from mujoco_metal.native_api import mj_forward
  from mujoco_metal.native_api import mj_rnePostConstraint
  from mujoco_metal.simulation import MetalSimulation

  model = mujoco.MjModel.from_xml_string('''<mujoco>
    <option><flag contact="disable"/></option>
    <worldbody><body pos="0 0 .3"><freejoint/>
      <geom type="sphere" size=".1" mass="1"/>
    </body></worldbody>
  </mujoco>''')
  sim = MetalSimulation(model, batch_size=2, profile="integrated_euler_v1")
  forward = mj_forward(sim, skipsensor=True)
  qacc = forward["qacc"].clone()
  qacc[1, 2] = float("nan")
  result = mj_rnePostConstraint(sim, forward_result=forward, qacc=qacc)
  assert result["status"].tolist() == [0, 3]
  for name in ("cacc", "cfrc_ext", "cfrc_int", "subtree_com", "qacc"):
    assert torch.count_nonzero(result[name][1]).item() == 0, name
    assert torch.isfinite(result[name][0]).all().item(), name


@pytest.mark.gpu
@pytest.mark.skipif(os.getenv("MUJOCO_METAL_RUN_GPU") != "1",
                    reason="opt-in GPU")
@pytest.mark.parametrize("cone", ["pyramidal", "elliptic"])
@pytest.mark.parametrize("condim", [3, 4, 6])
def test_rne_post_constraint_mixed_contact_equality_batch_two(cone, condim):
  """The post stage composes active contact/equality forces and moving sites."""
  from mujoco_metal.native_api import mj_rnePostConstraint
  from mujoco_metal.simulation import MetalSimulation

  model = _rich_rne_model(cone, condim)
  qpos = np.repeat(np.asarray(model.qpos0, dtype=np.float32)[None], 2, axis=0)
  qpos[1, 2] += .001
  qpos[1, -1] += .04
  qvel = np.asarray([
      [.05, -.08, .1, .2, -.15, .1, .16],
      [-.07, .04, -.03, -.1, .12, .18, -.11]], dtype=np.float32)
  sim = MetalSimulation(model, batch_size=2, qpos=qpos, qvel=qvel,
                        profile="integrated_euler_v1")
  result = mj_rnePostConstraint(sim)
  assert result["status"].cpu().tolist() == [0, 0]

  for world in range(2):
    cpu = mujoco.MjData(model)
    cpu.qpos[:] = qpos[world]
    cpu.qvel[:] = qvel[world]
    mujoco.mj_forward(model, cpu)
    assert cpu.ncon > 0 and cpu.ne > 0
    mujoco.mj_rnePostConstraint(model, cpu)
    for name in ("cacc", "cfrc_ext", "cfrc_int", "subtree_com"):
      np.testing.assert_allclose(result[name][world].cpu().numpy(),
                                 getattr(cpu, name), rtol=4e-4, atol=5e-4,
                                 err_msg=f"{cone}/{condim} {name}, world {world}")


@pytest.mark.gpu
@pytest.mark.skipif(os.getenv("MUJOCO_METAL_RUN_GPU") != "1",
                    reason="opt-in GPU")
@pytest.mark.parametrize("cone", ["pyramidal", "elliptic"])
@pytest.mark.parametrize("condim", [3, 4, 6])
def test_rne_post_constraint_mixed_contact_shared_qacc(cone, condim):
  """Compare post-stage math independently of native qacc solve error."""
  from mujoco_metal.native_api import mj_forward
  from mujoco_metal.native_api import mj_rnePostConstraint
  from mujoco_metal.simulation import MetalSimulation

  model = _rich_rne_model(cone, condim)
  qpos = np.repeat(np.asarray(model.qpos0, dtype=np.float32)[None], 2, axis=0)
  qpos[1, 2] += .001
  qpos[1, -1] += .04
  qvel = np.asarray([
      [.05, -.08, .1, .2, -.15, .1, .16],
      [-.07, .04, -.03, -.1, .12, .18, -.11]], dtype=np.float32)
  sim = MetalSimulation(model, batch_size=2, qpos=qpos, qvel=qvel,
                        profile="integrated_euler_v1")
  forward = mj_forward(sim, skipsensor=True)
  assert forward["status"].cpu().tolist() == [0, 0]
  shared_acc = forward["qacc"].clone()
  native = mj_rnePostConstraint(sim, forward_result=forward, qacc=shared_acc)
  assert native["status"].cpu().tolist() == [0, 0]

  for world in range(2):
    cpu = mujoco.MjData(model)
    cpu.qpos[:] = qpos[world]
    cpu.qvel[:] = qvel[world]
    mujoco.mj_forward(model, cpu)
    assert cpu.ncon > 0 and cpu.ne > 0
    cpu.qacc[:] = shared_acc[world].cpu().numpy()
    mujoco.mj_rnePostConstraint(model, cpu)
    for name in ("cacc", "cfrc_ext", "cfrc_int", "subtree_com"):
      np.testing.assert_allclose(native[name][world].cpu().numpy(),
                                 getattr(cpu, name), rtol=4e-4, atol=5e-4,
                                 err_msg=(f"{cone}/{condim} {name}, "
                                          f"shared qacc, world {world}"))


@pytest.mark.gpu
@pytest.mark.skipif(os.getenv("MUJOCO_METAL_RUN_GPU") != "1",
                    reason="opt-in GPU")
def test_rne_post_constraint_foreign_record_rejected_and_exception_rolls_back():
  import torch
  from mujoco_metal.native_api import mj_forward
  from mujoco_metal.native_api import mj_rnePostConstraint
  from mujoco_metal.simulation import MetalSimulation

  model = _free_sphere_model()
  qpos = np.asarray(model.qpos0, dtype=np.float32)[None]
  qvel = np.zeros((1, model.nv), dtype=np.float32)
  first = MetalSimulation(model, batch_size=1, qpos=qpos, qvel=qvel,
                          profile="integrated_euler_v1")
  second = MetalSimulation(model, batch_size=1, qpos=qpos, qvel=qvel,
                           profile="integrated_euler_v1")
  foreign = mj_forward(first, skipsensor=True)
  second_record = second._forward_stages._record
  with pytest.raises(ValueError, match="stale or belongs"):
    mj_rnePostConstraint(second, forward_result=foreign)
  assert second._forward_stages._record is second_record

  forward = mj_forward(second, skipsensor=True)
  record = forward["record"]
  before = {name: getattr(second._sensors, name).clone()
            for name in ("_rne_ext", "_rne_cacc_pair", "_rne_cfrc",
                         "_rne_scom", "_rne_com_scratch")}
  original = second._sensors._rne_post

  def fail_after_kernel(*args, **kwargs):
    original(*args, **kwargs)
    raise RuntimeError("injected post-constraint RNE failure")

  second._sensors._rne_post = fail_after_kernel
  try:
    with pytest.raises(RuntimeError, match="injected post-constraint"):
      mj_rnePostConstraint(second, forward_result=forward)
  finally:
    second._sensors._rne_post = original
  assert second._forward_stages._record is record
  for name, value in before.items():
    torch.testing.assert_close(getattr(second._sensors, name), value)
