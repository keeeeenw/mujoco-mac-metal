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

XML_SLEEP_MUTATIONS = """<mujoco>
<option timestep="0.002" gravity="0 0 0"><flag sleep="enable"/></option>
<worldbody>
  <body name="mover" mocap="true"><geom name="mover_geom" size="0.05" contype="0" conaffinity="0"/></body>
  <body name="a" pos="1 0 0" sleep="allowed"><freejoint/><geom name="ga" size="0.08" mass="0.5" contype="0" conaffinity="0"/></body>
  <body name="b" pos="3 0 0" sleep="allowed"><freejoint/><geom name="gb" size="0.08" mass="0.5" contype="0" conaffinity="0"/></body>
</worldbody>
<equality><weld name="ab" body1="a" body2="b" active="false"/></equality>
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
  assert snap.schema_version == 6 and snap.nmocap == 1
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


def test_device_state_mutation_hooks_preserve_selected_environment_identity():
  pytest.importorskip("torch")
  from mujoco_metal.device_state import DeviceState
  m, profile = _profile()
  # Exercise sparse selected-row notifications with a batch much larger than
  # the two-world native lifecycle fixture.  This is a CPU ownership test; the
  # separate opt-in MPS test below qualifies the scheduler/FK effects.
  batch = 257
  st = DeviceState(m, profile, batch, device="cpu")
  seen = []
  st._on_mocap_change = lambda env_ids=None: seen.append(("mocap", tuple(env_ids)))
  st._on_eq_active_change = lambda env_ids=None: seen.append(("eq", tuple(env_ids)))
  st._on_environment_copy = lambda src, dst: seen.append(("copy", src, dst))
  st._on_reset = lambda env_ids=None: seen.append(("reset", tuple(env_ids)))
  st._on_restore = lambda snap, env_ids=None: seen.append(("restore", None if env_ids is None else tuple(env_ids)))

  snap = st.snapshot()
  st.set_mocap(np.array([[0.2, 0, 1]]), np.array([[1., 0, 0, 0]]), env_ids=[256])
  st.set_equality_active(np.array([0], dtype=np.int32), env_ids=[128])
  st.copy_environment(128, 256)
  st.reset(env_ids=[127, 256])
  st.restore(snap, env_ids=[0, 256])

  assert seen == [
      ("mocap", (256,)),
      ("eq", (128,)),
      ("copy", 128, 256),
      ("reset", (127, 256)),
      ("restore", (0, 256)),
  ]


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
  assert snap.schema_version == 6
  sim.set_mocap(np.array([[0.9, 0, 1.3]]), np.array([[1., 0, 0, 0]]), env_ids=[0])
  sim.state.restore(snap)
  np.testing.assert_allclose(sim.state.mocap_pos.cpu().numpy()[0, 0], [0.2, 0, 1.1], atol=1e-6)
  sim.copy_environment(0, 1)
  np.testing.assert_allclose(sim.state.mocap_pos.cpu().numpy()[1, 0], [0.2, 0, 1.1], atol=1e-6)
  with pytest.raises(ValueError):
    sim.reset_to_keyframe(7)
  with pytest.raises(ValueError):
    sim.set_mocap(np.array([[0, 0, 1]]), np.array([[0, 0, 0, 0]]))


@pytest.mark.gpu
@pytest.mark.skipif(os.getenv("MUJOCO_METAL_RUN_GPU") != "1", reason="opt-in GPU")
def test_sleep_schedule_state_mutation_routes_are_selected_and_replayable_gpu():
  import torch
  from mujoco_metal import MetalSimulation
  from mujoco_metal.lifecycle import ModelLifecycle
  m = mujoco.MjModel.from_xml_string(XML_SLEEP_MUTATIONS)
  sim = MetalSimulation(m, batch_size=2, profile="integrated_scalable_v1")
  scheduler = sim._sleep_schedule
  assert scheduler is not None
  scheduler.tree_state.copy_(torch.tensor([[-5, 1], [-7, 1]], dtype=torch.int32, device="mps"))
  scheduler.tree_awake.copy_((scheduler.tree_state < 0).to(dtype=torch.int32))
  fk = sim._smooth._fk
  fk.run_device(sim.state._qpos, sim.state._mpos, sim.state._mquat,
                tree_awake=torch.ones_like(scheduler.tree_awake))
  cache_valid = fk._workspace["outputs"]["cache_valid"]
  np.testing.assert_array_equal(cache_valid.cpu().numpy(), [1, 1])

  # A moving mocap body is a static-tree participant in pinned wake logic:
  # invalidate selected FK only, leaving the dynamic sleep cycles unchanged.
  sim.set_mocap(np.array([[0.5, 0, 1.0]], dtype=np.float32),
                np.array([[1, 0, 0, 0]], dtype=np.float32), env_ids=[1])
  np.testing.assert_array_equal(cache_valid.cpu().numpy(), [1, 0])
  np.testing.assert_array_equal(scheduler.tree_state.cpu().numpy(), [[-5, 1], [-7, 1]])

  # Equality activation is consumed by the next same-position wake sweep;
  # only world 1's first sleeping equality tree wakes fully.
  sim.state.set_equality_active(np.array([1], dtype=np.int32), env_ids=[1])
  sim._prepare_sleep_schedule(sim.state._qpos, sim.state._qvel)
  np.testing.assert_array_equal(scheduler.tree_state.cpu().numpy(), [[-5, 1], [-7, -11]])

  saved = sim.snapshot()
  scheduler.tree_state.fill_(-11)
  sim.restore(saved)
  np.testing.assert_array_equal(scheduler.tree_state.cpu().numpy(), [[-5, 1], [-7, -11]])
  np.testing.assert_array_equal(cache_valid.cpu().numpy(), [1, 1])

  # Restore followed by a selected setter invalidates only that row and does
  # not wake its dynamic trees; a direct DeviceState row copy copies sleep.
  sim.state.set_mocap(np.array([[0.2, 0, 1.0]], dtype=np.float32),
                      np.array([[1, 0, 0, 0]], dtype=np.float32), env_ids=[0])
  np.testing.assert_array_equal(cache_valid.cpu().numpy(), [0, 1])
  np.testing.assert_array_equal(scheduler.tree_state.cpu().numpy(), [[-5, 1], [-7, -11]])
  sim.state.copy_environment(1, 0)
  np.testing.assert_array_equal(scheduler.tree_state.cpu().numpy(), [[-7, -11], [-7, -11]])
  sim.state.reset(env_ids=[1])
  np.testing.assert_array_equal(scheduler.tree_state.cpu().numpy()[0], [-7, -11])
  np.testing.assert_array_equal(scheduler.tree_awake.cpu().numpy()[1], [1, 1])

  # Rebuilt model constants have a new contact/equality graph, so lifecycle
  # adoption starts with all trees awake for the first current-position pass.
  life = ModelLifecycle(m)
  geom = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "ga")
  life.update_geom_contact([geom], friction=np.array([[0.6, 0.1, 0.01]]))
  sim.apply_lifecycle(life)
  np.testing.assert_array_equal(sim._sleep_schedule.tree_awake.cpu().numpy(), [[1, 1], [1, 1]])


XML_KEYCTRL = """<mujoco><option timestep="0.002" gravity="0 0 -9.81"/>
<worldbody>
<body pos="0 0 1" mocap="true"><geom type="sphere" size="0.05"/></body>
<body pos="0 0 0.8"><joint name="j" type="hinge" axis="0 1 0"/><geom type="sphere" size="0.08" mass="0.5"/></body>
</worldbody>
<actuator><general joint="j" dyntype="filter" dynprm="0.05 0 0" gainprm="2 0 0"/></actuator>
<keyframe><key name="k0" qpos="0.3" qvel="0.1" act="0.2" mpos="0.1 0 1.05" mquat="1 0 0 0" ctrl="0.7" time="1.5"/></keyframe>
</mujoco>"""


@pytest.mark.gpu
@pytest.mark.skipif(os.getenv("MUJOCO_METAL_RUN_GPU") != "1", reason="opt-in GPU")
def test_keyframe_reset_atomic_on_invalid_ctrl_gpu():
  # R4: an invalid keyframe payload must fail before ANY mutation: state,
  # time, activation, mocap, held controls, generation and caches.
  import torch
  from mujoco_metal import MetalSimulation
  m = mujoco.MjModel.from_xml_string(XML_KEYCTRL)
  sim = MetalSimulation(m, batch_size=2, profile="integrated_euler_v1")
  sim.reset(qpos=np.array([[0.2], [0.25]], dtype=np.float32),
            act=np.array([[0.0], [0.0]], dtype=np.float32))
  sim.step(2, ctrl=np.array([[3.0], [4.0]], dtype=np.float32))
  before = {
      "qpos": sim.state.qpos.cpu().numpy().copy(),
      "time": sim.state.time.cpu().numpy().copy(),
      "act": sim.state.act.cpu().numpy().copy(),
      "mpos": sim.state.mocap_pos.cpu().numpy().copy(),
      "ctrl": sim._control.cpu().numpy().copy(),
      "gen": sim.state.generation,
  }
  m.key_ctrl[0, 0] = float("nan")
  with pytest.raises(ValueError, match="keyframe ctrl must be finite"):
    sim.reset_to_keyframe(0)
  np.testing.assert_array_equal(sim.state.qpos.cpu().numpy(), before["qpos"])
  np.testing.assert_array_equal(sim.state.time.cpu().numpy(), before["time"])
  np.testing.assert_array_equal(sim.state.act.cpu().numpy(), before["act"])
  np.testing.assert_array_equal(sim.state.mocap_pos.cpu().numpy(), before["mpos"])
  np.testing.assert_array_equal(sim._control.cpu().numpy(), before["ctrl"])
  assert sim.state.generation == before["gen"]
  # Valid recovery after failure applies the keyframe to all worlds.
  m.key_ctrl[0, 0] = 0.7
  sim.reset_to_keyframe(0)
  np.testing.assert_allclose(sim.state.qpos.cpu().numpy()[:, 0], [0.3, 0.3], atol=1e-6)
  np.testing.assert_allclose(sim.state.act.cpu().numpy()[:, 0], [0.2, 0.2], atol=1e-6)
  np.testing.assert_allclose(sim._control.cpu().numpy()[:, 0], [0.7, 0.7], atol=1e-6)
  assert float(sim.state.time.cpu().numpy()[0]) == 1.5


@pytest.mark.gpu
@pytest.mark.skipif(os.getenv("MUJOCO_METAL_RUN_GPU") != "1", reason="opt-in GPU")
def test_keyframe_reset_partial_env_and_time_overflow_gpu():
  # R4: partial selection isolates unselected worlds; unrepresentable key
  # time fails before mutation.
  from mujoco_metal import MetalSimulation
  m = mujoco.MjModel.from_xml_string(XML_KEYCTRL)
  sim = MetalSimulation(m, batch_size=2, profile="integrated_euler_v1")
  sim.step(2, ctrl=np.array([[1.0], [2.0]], dtype=np.float32))
  sim.reset_to_keyframe(0, env_ids=[0])
  np.testing.assert_allclose(sim.state.qpos.cpu().numpy()[0], [0.3], atol=1e-6)
  assert float(sim.state.time.cpu().numpy()[0]) == 1.5
  t1_before = float(sim.state.time.cpu().numpy()[1])
  q1_before = sim.state.qpos.cpu().numpy()[1].copy()
  m.key_time[0] = 1e300
  with pytest.raises(ValueError, match="float32-representable"):
    sim.reset_to_keyframe(0, env_ids=[0])
  assert float(sim.state.time.cpu().numpy()[0]) == 1.5
  assert float(sim.state.time.cpu().numpy()[1]) == t1_before
  np.testing.assert_array_equal(sim.state.qpos.cpu().numpy()[1], q1_before)
  m.key_time[0] = 1.5
  sim.reset_to_keyframe(0, env_ids=[0])  # valid recovery
  assert float(sim.state.time.cpu().numpy()[0]) == 1.5


SLIDE_XML = """<mujoco><option timestep="0.002" gravity="0 0 -9.81"/>
<worldbody>
<geom name="floor" type="plane" size="5 5 0.1" contype="1" conaffinity="1"/>
<body pos="0 0 0.05"><joint name="s" type="slide" axis="1 0 0"/>
<geom name="box" type="box" size="0.05 0.05 0.05" mass="0.5" friction="0.2 0.05 0.02" contype="1" conaffinity="1"/></body>
</worldbody>
</mujoco>"""


OSC_XML = """<mujoco><option timestep="0.002" gravity="0 0 0"/>
<worldbody>
<body pos="0 0 0.1"><joint name="s" type="slide" axis="1 0 0"/>
<geom name="box" type="sphere" size="0.05" mass="0.5" contype="0" conaffinity="0"/></body>
</worldbody>
<tendon><fixed name="spring" stiffness="100" damping="0" springlength="0 0"><joint joint="s" coef="1.0"/></fixed></tendon>
</mujoco>"""


@pytest.mark.gpu
@pytest.mark.skipif(os.getenv("MUJOCO_METAL_RUN_GPU") != "1", reason="opt-in GPU")
def test_apply_lifecycle_contact_update_gpu():
  # R7: a contact-friction update through a lifecycle changes native physics
  # (steady push slips before, holds after); invalid topology fails
  # atomically; restore-then-update keeps stepping.
  from mujoco_metal import MetalSimulation
  from mujoco_metal.lifecycle import ModelLifecycle
  m = mujoco.MjModel.from_xml_string(SLIDE_XML)
  sim = MetalSimulation(m, batch_size=1, profile="integrated_euler_v1")
  sim.reset(qpos=np.array([[0.0]], dtype=np.float32),
            qvel=np.array([[3.0]], dtype=np.float32))
  for _ in range(50):
    sim.step(1)
  free_x = float(sim.state.qpos.cpu().numpy()[0, 0])
  assert abs(free_x - 0.3) < 1e-3  # no contact yet: flies freely

  life = ModelLifecycle(SLIDE_XML)
  floor = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "floor")
  assert life.update_geom_contact(
      [floor], margin=np.array([0.02]),
      friction=np.array([[2.0, 0.05, 0.02]]))
  gen_before = sim.state.generation
  sim.apply_lifecycle(life)
  assert sim.state.generation > gen_before
  sim.reset(qpos=np.array([[0.0]], dtype=np.float32),
            qvel=np.array([[3.0]], dtype=np.float32))
  for _ in range(50):
    sim.step(1)
  gripped_x = float(sim.state.qpos.cpu().numpy()[0, 0])
  assert gripped_x < free_x - 0.05  # contact + friction engage
  # CPU oracle with the identical mutation agrees.
  cpu = mujoco.MjData(life._model)
  cpu.qpos[0] = 0.0
  cpu.qvel[0] = 3.0
  mujoco.mj_forward(life._model, cpu)
  for _ in range(50):
    mujoco.mj_step(life._model, cpu)
  np.testing.assert_allclose(gripped_x, float(cpu.qpos[0]), rtol=1e-4, atol=1e-4)

  # Incompatible topology fails without touching the simulation.
  bad = ModelLifecycle(SLIDE_XML.replace(
      "</worldbody>",
      '<body pos="1 0 0.1"><geom type="sphere" size="0.05"/></body></worldbody>'))
  q_before = sim.state.qpos.cpu().numpy().copy()
  gen_bad = sim.state.generation
  with pytest.raises(ValueError, match="structural count"):
    sim.apply_lifecycle(bad)
  np.testing.assert_array_equal(sim.state.qpos.cpu().numpy(), q_before)
  assert sim.state.generation == gen_bad
  sim.step(1)  # keeps stepping after a failed update

  # Restore-then-update round trip.
  snap = sim.state.snapshot()
  sim.state.restore(snap)
  life2 = ModelLifecycle(SLIDE_XML)
  assert life2.update_geom_contact([floor], friction=np.array([[0.2, 0.05, 0.02]]))
  sim.apply_lifecycle(life2)
  sim.step(1)
  assert np.all(np.isfinite(sim.state.qpos.cpu().numpy()))


@pytest.mark.gpu
@pytest.mark.skipif(os.getenv("MUJOCO_METAL_RUN_GPU") != "1", reason="opt-in GPU")
def test_apply_lifecycle_mass_update_gpu():
  # R7: a body-mass update changes native spring-mass timing (quarter period
  # doubles when mass quadruples), verified against the CPU mutated oracle.
  from mujoco_metal import MetalSimulation
  from mujoco_metal.lifecycle import ModelLifecycle
  m = mujoco.MjModel.from_xml_string(OSC_XML)
  sim = MetalSimulation(m, batch_size=1, profile="integrated_euler_v1")
  sim.reset(qpos=np.array([[0.2]], dtype=np.float32))
  t0, prev = None, 0.2
  for step in range(400):
    sim.step(1)
    x = float(sim.state.qpos.cpu().numpy()[0, 0])
    if prev > 0 and x <= 0:
      t0 = step
      break
    prev = x
  assert t0 is not None and 40 < t0 < 70  # ~T/4 = 55 steps
  life = ModelLifecycle(OSC_XML)
  assert life.recompute_body_masses([1], [2.0])
  sim.apply_lifecycle(life)
  sim.reset(qpos=np.array([[0.2]], dtype=np.float32))
  t1, prev = None, 0.2
  for step in range(600):
    sim.step(1)
    x = float(sim.state.qpos.cpu().numpy()[0, 0])
    if prev > 0 and x <= 0:
      t1 = step
      break
    prev = x
  assert t1 is not None and 90 < t1 < 130  # ~T/4 = 111 steps
  cpu = mujoco.MjData(life._model)
  cpu.qpos[0] = 0.2
  mujoco.mj_forward(life._model, cpu)
  tc, prev = None, 0.2
  for step in range(600):
    mujoco.mj_step(life._model, cpu)
    if prev > 0 and float(cpu.qpos[0]) <= 0:
      tc = step
      break
    prev = float(cpu.qpos[0])
  assert tc is not None and abs(t1 - tc) <= 2
