# Copyright 2026 The MuJoCo Metal contributors
# Licensed under the Apache License, Version 2.0.
"""Standalone inverse stages against actual pinned-engine intermediate outputs."""
import os

import mujoco
import numpy as np
import pytest

from mujoco_metal import MetalSimulation
from mujoco_metal import native_api as api
from mujoco_metal.forward_stages import ForwardStage


def _fixture(cone="elliptic", profile="integrated_euler_v1"):
  if profile == "normal_contact_euler_v1":
    xml = '''<mujoco><worldbody><geom type="plane" size="2 2 .1" condim="1"/>
      <body pos="0 0 .09"><freejoint/><geom type="sphere" size=".1" mass="1"
      condim="1"/></body></worldbody></mujoco>'''
  elif profile == "joint_constraints_euler_v1":
    xml = '''<mujoco><option gravity="0 0 0"><flag contact="disable"/></option>
      <worldbody><body><joint name="j" type="slide" axis="1 0 0" limited="true"
      range="-.1 .1" margin=".01" frictionloss=".3" damping=".2"/>
      <geom type="sphere" size=".1" mass="1"/></body></worldbody></mujoco>'''
  else:
    xml = f'''<mujoco><option cone="{cone}" solver="PGS" impratio="2"/>
      <worldbody><geom name="floor" type="plane" size="2 2 .1"/>
        <body pos="0 0 .09"><freejoint/>
          <geom name="ball" type="sphere" size=".1" mass="1"/>
        </body>
        <body pos="1 0 .6"><joint name="a" type="slide" axis="1 0 0"
          frictionloss=".3" damping=".2" stiffness="1"/>
          <geom type="sphere" size=".08" mass=".7" contype="0" conaffinity="0"/>
        </body>
        <body pos="2 0 .6"><joint name="b" type="slide" axis="1 0 0"
          damping=".1"/><geom type="sphere" size=".08" mass="1.2"
          contype="0" conaffinity="0"/></body>
      </worldbody>
      <equality><joint joint1="a" joint2="b" polycoef="0 1 0 0 0"/></equality>
      <contact><pair geom1="floor" geom2="ball" condim="6"
        friction="1 .7 .2 .15 .1"/></contact>
      <sensor><jointpos joint="a"/><jointvel joint="a"/></sensor>
      </mujoco>'''
  model = mujoco.MjModel.from_xml_string(xml)
  qp = np.tile(model.qpos0, (2, 1)).astype(np.float32)
  if model.nq == 1:
    qp[:, 0] = [.12, -.12]
  else:
    qp[:, 2] = [.09, .095]
    if model.nq > 7:
      qp[:, -2:] = [[.02, -.01], [-.01, .03]]
  qv = np.stack((np.linspace(-.3, .4, model.nv),
                 np.linspace(.2, -.35, model.nv))).astype(np.float32)
  qa = np.stack((np.linspace(.15, -.6, model.nv),
                 np.linspace(-.3, .8, model.nv))).astype(np.float32)
  return model, qp, qv, qa


def _pinned_stages(model, qp, qv, qa):
  result = []
  for qpos, qvel, qacc in zip(qp, qv, qa):
    data = mujoco.MjData(model)
    data.qpos[:] = qpos
    data.qvel[:] = qvel
    data.qacc[:] = qacc
    mujoco.mj_invPosition(model, data)
    mass = np.empty((model.nv, model.nv))
    mujoco.mj_fullM(model, data, mass)
    pos = data.xpos.copy()
    mujoco.mj_invVelocity(model, data)
    bias, passive = data.qfrc_bias.copy(), data.qfrc_passive.copy()
    mujoco.mj_invConstraint(model, data)
    result.append({"xpos": pos, "mass": mass, "bias": bias, "passive": passive,
                   "force": data.qfrc_constraint.copy(), "rows": data.nefc,
                   "qacc": data.qacc.copy()})
  return result


@pytest.mark.parametrize("profile", ["integrated_euler_v1",
    "normal_contact_euler_v1", "joint_constraints_euler_v1"])
def test_pinned_inverse_split_fixture_has_nonzero_constraints_and_no_acceleration_conversion(profile):
  model, qp, qv, qa = _fixture(profile=profile)
  model.opt.enableflags |= int(mujoco.mjtEnableBit.mjENBL_INVDISCRETE)
  stages = _pinned_stages(model, qp, qv, qa)
  for stage, requested in zip(stages, qa):
    assert stage["rows"] > 0
    assert np.linalg.norm(stage["force"]) > .1
    np.testing.assert_array_equal(stage["qacc"], requested)


def _host(value):
  return value.detach().cpu().numpy().copy()


def _fail(*args, **kwargs):
  raise AssertionError("later-stage producer or optimizer ran")


@pytest.mark.parametrize("profile,cone", [
    ("integrated_euler_v1", "pyramidal"),
    ("integrated_euler_v1", "elliptic"),
    ("integrated_scalable_v1", "pyramidal"),
    ("integrated_scalable_v1", "elliptic"),
    ("normal_contact_euler_v1", "pyramidal"),
    ("joint_constraints_euler_v1", "pyramidal"),
])
@pytest.mark.gpu
@pytest.mark.skipif(os.getenv("MUJOCO_METAL_RUN_GPU") != "1", reason="opt-in native qualification")
def test_native_inverse_split_matches_stages_and_preserves_constraint_query(
    monkeypatch, cone, profile):
  import torch
  model, qp, qv, qa = _fixture(cone, profile)
  model.opt.enableflags |= int(mujoco.mjtEnableBit.mjENBL_INVDISCRETE)
  expected = _pinned_stages(model, qp, qv, qa)
  sim = MetalSimulation(model, 2, qpos=qp, qvel=qv, profile=profile)
  # Standalone inverse stages must not silently evaluate later forward work.
  monkeypatch.setattr(sim, "prepare_forward_actuation", _fail)
  monkeypatch.setattr(sim, "prepare_forward_acceleration", _fail)
  monkeypatch.setattr(sim, "prepare_forward_constraint", _fail)
  monkeypatch.setattr(sim, "inverse_skip", _fail)
  state_before = {name: _host(getattr(sim.state, "_"+name))
                  for name in ("qpos", "qvel", "qacc", "time", "status")}
  sensors_before = None if sim._sensordata is None else _host(sim._sensordata)
  record = api.mj_invPosition(sim, return_record=True)
  assert record.stage == ForwardStage.POS
  pos = record.values[ForwardStage.POS]
  dynamics = pos["dynamics"]
  for world, oracle in enumerate(expected):
    np.testing.assert_allclose(_host(pos["poses"]["body_pos"]).reshape(2, model.nbody, 3)[world], oracle["xpos"],
                               atol=2e-6, rtol=2e-6)
  mass = api.mj_fullM(sim, dynamics=dynamics)
  for world, oracle in enumerate(expected):
    np.testing.assert_allclose(_host(mass)[world], oracle["mass"], atol=2e-5, rtol=2e-5)
  # Reuse the exact POS; a second FK would fail this gate.
  monkeypatch.setattr(sim._smooth._fk, "run_device", _fail)
  dynamics = api.mj_invVelocity(sim, record=record)
  assert record.stage == ForwardStage.VEL
  velocity = record.values[ForwardStage.VEL]
  for world, oracle in enumerate(expected):
    np.testing.assert_allclose(_host(velocity["qfrc_bias"])[world], oracle["bias"],
                               atol=2e-5, rtol=2e-5)
    np.testing.assert_allclose(_host(velocity["qfrc_passive"])[world], oracle["passive"],
                               atol=2e-5, rtol=2e-5)
  # The constraint primitive consumes cached rows, not the full inverse pipeline.
  pointers = {name: value.data_ptr() for name, value in dynamics.items()
              if isinstance(value, torch.Tensor)}
  saved = {name: _host(value) for name, value in dynamics.items()
           if isinstance(value, torch.Tensor)}
  borrowed_acc = sim.state._qacc
  borrowed_acc.copy_(torch.as_tensor(qa, device="mps"))
  result = api.mj_invConstraint(sim, record=record, return_details=True)
  force = result["qfrc_constraint_inverse"]
  np.testing.assert_array_equal(_host(result["status"]), [0, 0])
  for world, oracle in enumerate(expected):
    np.testing.assert_allclose(_host(force)[world], oracle["force"], atol=3e-2, rtol=3e-4)
  np.testing.assert_array_equal(_host(borrowed_acc), qa)
  force.zero_()
  replay = api.mj_invConstraint(sim, qacc=qa, record=record)
  assert replay.data_ptr() != force.data_ptr()
  for world, oracle in enumerate(expected):
    np.testing.assert_allclose(_host(replay)[world], oracle["force"], atol=3e-2, rtol=3e-4)
  sim.validate_forward_stage_record(record, "VEL")
  for name in saved:
    assert dynamics[name].data_ptr() == pointers[name]
    np.testing.assert_array_equal(_host(dynamics[name]), saved[name])
  for name, before in state_before.items():
    if name != "qacc":
      np.testing.assert_array_equal(_host(getattr(sim.state, "_"+name)), before)
  if sensors_before is not None:
    np.testing.assert_array_equal(_host(sim._sensordata), sensors_before)


@pytest.mark.gpu
@pytest.mark.skipif(os.getenv("MUJOCO_METAL_RUN_GPU") != "1", reason="opt-in native qualification")
def test_native_inverse_constraint_rejects_stale_prefix_and_restores_after_failure(monkeypatch):
  import torch
  model, qp, qv, qa = _fixture()
  sim = MetalSimulation(model, 2, qpos=qp, qvel=qv, profile="integrated_euler_v1")
  record = api.mj_invPosition(sim, return_record=True)
  api.mj_invVelocity(sim, record=record)
  cc = sim._coupled_constraints
  rows = cc._assembly_views(include_optimizer_outputs=False)
  before = {name: (value.data_ptr(), _host(value)) for name, value in rows.items()
            if isinstance(value, torch.Tensor)}
  import mujoco_metal.inverse_constraints as inverse_constraints
  injected = {"hit": False}
  def fail_paired_reduction(*_args, **_kwargs):
    injected["hit"] = True
    raise RuntimeError("inverse constraint reduction failed")
  with monkeypatch.context() as patch:
    patch.setattr(inverse_constraints,
                  "_dense_jacobian_transpose_pair_matvec",
                  fail_paired_reduction)
    patch.setattr(inverse_constraints,
                  "_packed_jacobian_transpose_pair_matvec",
                  fail_paired_reduction)
    with pytest.raises(RuntimeError, match="constraint reduction failed"):
      api.mj_invConstraint(sim, qa, record=record)
  assert injected["hit"], "inverse query did not reach the paired row reducer"
  sim.validate_forward_stage_record(record, "VEL")
  for name, (pointer, values) in before.items():
    assert rows[name].data_ptr() == pointer
    np.testing.assert_array_equal(_host(rows[name]), values)
  api.mj_invConstraint(sim, qa, record=record)
  sim.state._qpos.add_(.001)
  with pytest.raises(ValueError, match="mutat|stale|changed"):
    api.mj_invConstraint(sim, qa, record=record)


@pytest.mark.parametrize("moving", [False, True])
@pytest.mark.gpu
@pytest.mark.skipif(os.getenv("MUJOCO_METAL_RUN_GPU") != "1", reason="opt-in native qualification")
def test_native_inverse_split_without_constraints_accepts_empty_device_inputs(moving):
  import torch
  joint = '<joint type="slide"/>' if moving else ''
  model = mujoco.MjModel.from_xml_string(f'''<mujoco>
    <option><flag contact="disable"/></option><worldbody><body>{joint}
    <geom type="sphere" size=".1" mass="1"/></body></worldbody></mujoco>''')
  sim = MetalSimulation(model, 2, profile="contact_free_euler_v1")
  qpos = torch.zeros((2, model.nq), device="mps")
  qvel = torch.zeros((2, model.nv), device="mps")
  qacc = torch.ones((2, model.nv), device="mps")
  record = api.mj_invPosition(sim, qpos=qpos, return_record=True)
  api.mj_invVelocity(sim, qvel=qvel, record=record)
  result = api.mj_invConstraint(sim, qacc=qacc, record=record, return_details=True)
  assert tuple(result["qfrc_constraint_inverse"].shape) == (2, model.nv)
  np.testing.assert_array_equal(_host(result["qfrc_constraint_inverse"]), np.zeros((2, model.nv)))
  np.testing.assert_array_equal(_host(result["status"]), [0, 0])
  sim.validate_forward_stage_record(record, "VEL")
