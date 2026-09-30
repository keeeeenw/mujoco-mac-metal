# Copyright 2026 The MuJoCo Metal contributors
# Licensed under the Apache License, Version 2.0.
"""Native mocap inputs, keyframes and model-change ownership (CPU + GPU)."""

import os

import mujoco
import numpy as np
import pytest

XML_HOOK = """<mujoco><option timestep="0.002" gravity="0 0 -9.81"/>
<worldbody>
<body name="hook" pos="0 0 1" mocap="true"><geom name="hg" type="sphere" size="0.05" contype="1" conaffinity="0"/></body>
<body name="cargo" pos="0.3 0 0.8"><freejoint/><geom name="cg" type="sphere" size="0.08" mass="0.5" contype="1" conaffinity="1"/></body>
</worldbody>
<contact><pair geom1="hg" geom2="cg"/></contact>
<equality><weld name="grab" body1="hook" body2="cargo"/></equality>
</mujoco>"""

XML_KEYS = """<mujoco><option timestep="0.002" gravity="0 0 -9.81"/>
<worldbody>
<body pos="0 0 1" mocap="true"><geom type="sphere" size="0.05"/></body>
<body pos="0 0 0.8"><freejoint/><geom type="sphere" size="0.08" mass="0.5"/></body>
</worldbody>
<keyframe><key name="k0" qpos="0 0 0.5 1 0 0 0" mpos="0.2 0 1.1" mquat="1 0 0 0" time="0.5"/></keyframe>
</mujoco>"""


def _profile(name="integrated_euler_v1"):
  from mujoco_metal.stepping import validate_stepping_profile
  m = mujoco.MjModel.from_xml_string(XML_HOOK)
  return m, validate_stepping_profile(m, profile=name)


def test_mocap_fk_matches_cpu_oracle():
  from mujoco_metal.model import load_model
  m = mujoco.MjModel.from_xml_string(XML_HOOK)
  desc = load_model(m)
  assert desc.nmocap == 1
  qpos = np.asarray(desc.qpos0)
  fk = desc.forward_kinematics(qpos)
  d = mujoco.MjData(m)
  mujoco.mj_forward(m, d)
  np.testing.assert_allclose(fk["body_pos"][1], np.asarray(d.xpos)[1], atol=1e-6)
  moved_pos = np.array([[0.5, 0.2, 1.1]])
  moved_quat = np.array([[0.7071068, 0.7071068, 0.0, 0.0]])
  fk2 = desc.forward_kinematics(qpos, mocap_pos=moved_pos, mocap_quat=moved_quat)
  d.mocap_pos[0] = [0.5, 0.2, 1.1]
  d.mocap_quat[0] = [0.7071068, 0.7071068, 0.0, 0.0]
  mujoco.mj_forward(m, d)
  np.testing.assert_allclose(fk2["body_pos"][1], np.asarray(d.xpos)[1], atol=1e-6)
  np.testing.assert_allclose(
      np.abs(fk2["body_quat"][1]), np.abs(np.asarray(d.xquat)[1]), atol=1e-6
  )
  with pytest.raises(ValueError):
    desc.forward_kinematics(qpos, mocap_pos=np.zeros((2, 3)), mocap_quat=np.zeros((2, 4)))
  with pytest.raises(ValueError):
    desc.forward_kinematics(qpos, mocap_pos=moved_pos, mocap_quat=np.array([[0.0, 0, 0, 0]]))


def test_mocap_state_defaults_and_atomic_validation_cpu():
  pytest.importorskip("torch")
  from mujoco_metal.device_state import DeviceState
  m, profile = _profile()
  st = DeviceState(m, profile, 2, device="cpu")
  assert st.nmocap == 1
  np.testing.assert_allclose(st.mocap_pos.numpy()[0, 0], [0, 0, 1], atol=1e-6)
  np.testing.assert_allclose(st.mocap_quat.numpy()[0, 0], [1, 0, 0, 0], atol=1e-6)
  before_pos = st.mocap_pos.numpy().copy()
  with pytest.raises(ValueError):
    st.set_mocap(np.zeros((1, 2)), np.zeros((1, 4)))
  with pytest.raises(ValueError):
    st.set_mocap(np.array([[[0, 0, 1]]]), np.array([[[0, 0, 0, 0]]]))
  with pytest.raises(ValueError):
    st.set_mocap(np.array([[[0, 0, 1]]]), np.array([[[1, 0, 0, 0]]]), env_ids=[5])
  np.testing.assert_array_equal(st.mocap_pos.numpy(), before_pos)
  # Broadcast single pose to selected worlds; unselected untouched.
  st.set_mocap(np.array([[0.4, 0, 1.0]]), np.array([[1., 0, 0, 0]]), env_ids=[1])
  np.testing.assert_allclose(st.mocap_pos.numpy()[1, 0], [0.4, 0, 1.0], atol=1e-6)
  np.testing.assert_allclose(st.mocap_pos.numpy()[0, 0], [0, 0, 1], atol=1e-6)
  # Posquat form.
  st.set_mocap(np.array([[[0.1, 0, 0.9, 1, 0, 0, 0]]]), env_ids=[0])
  np.testing.assert_allclose(st.mocap_pos.numpy()[0, 0], [0.1, 0, 0.9], atol=1e-6)


def test_mocap_reset_snapshot_restore_cpu():
  pytest.importorskip("torch")
  from mujoco_metal.device_state import DeviceState
  m, profile = _profile()
  st = DeviceState(m, profile, 2, device="cpu")
  st.set_mocap(np.array([[0.4, 0, 1.0]]), np.array([[1., 0, 0, 0]]), env_ids=[0])
  snap = st.snapshot()
  assert snap.schema_version == 3 and snap.nmocap == 1
  st.set_mocap(np.array([[0.9, 0, 1.2]]), np.array([[1., 0, 0, 0]]))
  st.restore(snap)
  np.testing.assert_allclose(st.mocap_pos.numpy()[0, 0], [0.4, 0, 1.0], atol=1e-6)
  np.testing.assert_allclose(st.mocap_pos.numpy()[1, 0], [0, 0, 1], atol=1e-6)
  # Reset restores compiled reference frames for selected worlds only.
  st.reset(env_ids=[0])
  np.testing.assert_allclose(st.mocap_pos.numpy()[0, 0], [0, 0, 1], atol=1e-6)
  # Schema-2 snapshot into a mocap model is rejected, never silently dropped.
  from mujoco_metal.device_state import StateSnapshot
  v2 = StateSnapshot(
      model_fingerprint=snap.model_fingerprint, profile_fingerprint=snap.profile_fingerprint,
      timestep=snap.timestep, nq=snap.nq, nv=snap.nv, batch_size=2,
      qpos=snap.qpos, qvel=snap.qvel, qacc=snap.qacc, time=snap.time,
      status=snap.status, schema_version=2, neq=snap.neq, eq_active=snap.eq_active,
  )
  with pytest.raises(ValueError, match="mocap"):
    st.restore(v2)
  # Copy preserves per-env poses.
  st.set_mocap(np.array([[0.7, 0, 1.0]]), np.array([[1., 0, 0, 0]]), env_ids=[0])
  st.copy_environment(0, 1)
  np.testing.assert_allclose(st.mocap_pos.numpy()[1, 0], [0.7, 0, 1.0], atol=1e-6)
  with pytest.raises(ValueError):
    st.copy_environment(0, 5)


def test_geom_contact_update_lifecycle_cpu():
  from mujoco_metal.lifecycle import ModelLifecycle
  m = mujoco.MjModel.from_xml_string(XML_HOOK)
  life = ModelLifecycle(m)
  assert life.generation == 0
  gid = int(mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "cg"))
  friction = np.asarray(m.geom_friction).copy()
  friction[gid] = [0.9, 0.1, 0.05]
  assert life.update_geom_contact([gid], friction=friction[gid : gid + 1]) is True
  assert life.generation == 1
  assert life.update_geom_contact([gid], friction=friction[gid : gid + 1]) is False
  before = life.generation
  with pytest.raises(ValueError):
    life.update_geom_contact([gid], friction=np.array([[0.5, 0.1]]))
  with pytest.raises(ValueError):
    life.update_geom_contact([999], friction=friction[gid : gid + 1])
  assert life.generation == before
  # Body masses still recompute through the same lifecycle.
  assert life.recompute_body_masses([2], [0.75]) is True


@pytest.mark.gpu
@pytest.mark.skipif(os.getenv("MUJOCO_METAL_RUN_GPU") != "1", reason="opt-in GPU")
def test_mocap_fk_native_parity_gpu():
  import torch
  from mujoco_metal.metal_kinematics import MetalKinematics
  from mujoco_metal.model import load_model
  desc = load_model(mujoco.MjModel.from_xml_string(XML_HOOK))
  fk = MetalKinematics(desc, batch_size=2)
  q = torch.as_tensor(
      np.tile(np.asarray(desc.qpos0, dtype=np.float32), (2, 1)), device="mps"
  )
  mp = torch.as_tensor(
      np.array([[[0, 0, 1]], [[0.5, 0.2, 1.1]]], dtype=np.float32), device="mps"
  )
  mq = torch.as_tensor(
      np.array([[[1, 0, 0, 0]], [[1, 0, 0, 0]]], dtype=np.float32), device="mps"
  )
  poses = fk.run_device(q, mp, mq)
  hook = poses["body_pos"][:, 1, :].cpu().numpy()
  np.testing.assert_allclose(hook[0], [0, 0, 1], atol=1e-5)
  np.testing.assert_allclose(hook[1], [0.5, 0.2, 1.1], atol=1e-5)


@pytest.mark.gpu
@pytest.mark.skipif(os.getenv("MUJOCO_METAL_RUN_GPU") != "1", reason="opt-in GPU")
def test_mocap_stepping_parity_and_coupling_gpu():
  from mujoco_metal import MetalSimulation
  m = mujoco.MjModel.from_xml_string(XML_HOOK)
  sim = MetalSimulation(m, batch_size=2, profile="integrated_euler_v1")
  cpu = [mujoco.MjData(m), mujoco.MjData(m)]
  # Env 0 holds; env 1 lifts the hook steadily (deterministic mocap schedule).
  max_err = 0.0
  for step in range(60):
    lift = [0.0, 0.0, 1.0 + 0.002 * step] if step >= 10 else [0.0, 0.0, 1.0]
    lift1 = [0.0, 0.0, 1.0 + 0.004 * step] if step >= 10 else [0.0, 0.0, 1.0]
    sim.set_mocap(np.array([[lift], [lift1]]), np.array([[[1, 0, 0, 0]], [[1, 0, 0, 0]]]))
    for k, target in enumerate((lift, lift1)):
      cpu[k].mocap_pos[0] = target
    sim.step(1)
    for k in range(2):
      mujoco.mj_step(m, cpu[k])
    gq = sim.state.qpos.cpu().numpy()
    for k in range(2):
      max_err = max(max_err, float(np.max(np.abs(gq[k] - cpu[k].qpos))))
  assert max_err < 5e-4, max_err
  # Weld genuinely couples: cargo follows the lifted hook per env.
  gq = sim.state.qpos.cpu().numpy()
  assert gq[1, 2] > gq[0, 2] + 0.02
  asm = sim.assembled_system(recompute=True)
  jf = asm["joint_force"][0].cpu().numpy()
  assert float(np.linalg.norm(jf)) > 0.5


@pytest.mark.gpu
@pytest.mark.skipif(os.getenv("MUJOCO_METAL_RUN_GPU") != "1", reason="opt-in GPU")
def test_mocap_keyframe_copy_snapshot_gpu():
  from mujoco_metal import MetalSimulation
  m = mujoco.MjModel.from_xml_string(XML_KEYS)
  sim = MetalSimulation(m, batch_size=2, profile="integrated_euler_v1")
  sim.step(3)
  sim.reset_to_keyframe(0, env_ids=[0])
  q = sim.state.qpos.cpu().numpy()
  np.testing.assert_allclose(q[0, :3], [0, 0, 0.5], atol=1e-6)
  mp = sim.state.mocap_pos.cpu().numpy()
  np.testing.assert_allclose(mp[0, 0], [0.2, 0, 1.1], atol=1e-6)
  assert abs(float(sim.state.time.cpu().numpy()[0]) - 0.5) < 1e-6
  # Unselected world untouched by the keyframe.
  assert abs(float(sim.state.time.cpu().numpy()[1]) - 0.006) < 1e-3
  snap = sim.state.snapshot()
  assert snap.schema_version == 3
  sim.set_mocap(np.array([[0.9, 0, 1.3]]), np.array([[1., 0, 0, 0]]), env_ids=[0])
  sim.state.restore(snap)
  np.testing.assert_allclose(sim.state.mocap_pos.cpu().numpy()[0, 0], [0.2, 0, 1.1], atol=1e-6)
  sim.copy_environment(0, 1)
  np.testing.assert_allclose(sim.state.mocap_pos.cpu().numpy()[1, 0], [0.2, 0, 1.1], atol=1e-6)
  with pytest.raises(ValueError):
    sim.reset_to_keyframe(7)
  with pytest.raises(ValueError):
    sim.set_mocap(np.array([[0, 0, 1]]), np.array([[0, 0, 0, 0]]))
