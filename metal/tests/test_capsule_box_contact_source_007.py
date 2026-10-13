# Copyright 2026 The MuJoCo Metal contributors
# Licensed under the Apache License, Version 2.0.
"""Capsule/box manifold selection at retained source poses, with real solves."""

from dataclasses import replace
import json
import os
from pathlib import Path

import mujoco
import numpy as np
import pytest


def _fixture():
  root = Path(__file__).resolve().parents[1]
  model = mujoco.MjModel.from_xml_path(str(root / "examples/muscle_gripper.xml"))
  data = json.loads((Path(__file__).parent / "data/gripper_capsule_contact_007.json").read_text())
  return model, data


def _oracle(model, data):
  cpu = mujoco.MjData(model)
  for name in ("qpos", "qvel", "act", "ctrl", "mocap_pos", "mocap_quat", "qacc_warmstart"):
    getattr(cpu, name)[:] = np.asarray(data[name]).reshape(getattr(cpu, name).shape)
  mujoco.mj_forward(model, cpu)
  return cpu


def test_captured_capsule_pose_has_three_source_contacts_and_nonzero_coupling():
  model, data = _fixture()
  cpu = _oracle(model, data)
  capsule = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "ball2_geom")
  contacts = [c for c in cpu.contact if capsule in c.geom]
  assert len(contacts) == data["expected_capsule_contacts"] == 3
  assert max(-c.dist for c in contacts) > .001
  assert np.linalg.norm(cpu.qfrc_constraint[8:]) > .1
  assert np.linalg.norm(cpu.qacc[11:]) > 100


@pytest.mark.gpu
@pytest.mark.skipif(os.getenv("MUJOCO_METAL_RUN_GPU") != "1", reason="native capsule manifold")
def test_native_capsule_manifold_and_coupled_acceleration_match_captured_source_pose(monkeypatch):
  from mujoco_metal import MetalSimulation
  from mujoco_metal.capacity import CapacityLimits
  model, data = _fixture()
  cpu = _oracle(model, data)
  sim = MetalSimulation(model, 1, profile="integrated_euler_v1",
                        limits=CapacityLimits(max_rows=256))
  q = np.asarray(data["qpos"], np.float64)
  v = np.asarray(data["qvel"], np.float64)
  qh, vh = q.astype(np.float32), v.astype(np.float32)
  ql, vl = (q-qh).astype(np.float32), (v-vh).astype(np.float32)
  qt = (q-qh.astype(float)-ql.astype(float)).astype(np.float32)
  vt = (v-vh.astype(float)-vl.astype(float)).astype(np.float32)
  sim.reset(qpos=qh, qvel=vh, act=np.asarray(data["act"], np.float32))
  checkpoint = sim.snapshot()
  checkpoint["device"] = replace(checkpoint["device"],
      precision_words=np.concatenate((ql, qt, vl, vt), axis=1))
  checkpoint["native_state"]["qacc_warmstart"] = np.asarray(data["qacc_warmstart"], np.float32)
  sim.restore(checkpoint)
  sim.set_mocap(np.asarray(data["mocap_pos"], np.float32),
                np.asarray(data["mocap_quat"], np.float32))
  sim._prepare_control(np.asarray(data["ctrl"], np.float32).reshape(1, model.nu))
  with monkeypatch.context() as context:
    def forbidden(*_args, **_kwargs):
      raise AssertionError("native contact solve fell back to CPU physics")
    for name in ("mj_forward", "mj_step", "mj_collision", "mj_kinematics",
                 "mj_fwdPosition", "mj_fwdVelocity", "mj_fwdConstraint"):
      context.setattr(mujoco, name, forbidden)
    acceleration, status, _ = sim._acceleration(sim.state._qpos, sim.state._qvel)
  assert status.cpu().numpy().tolist() == [0]
  result = sim._last_coupled
  desc = sim._coupled_constraints.descriptor
  capsule = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "ball2_geom")
  pairs = [(int(desc.geom1[i]), int(desc.geom2[i])) for i in range(desc.npairs)
           for _ in range(int(desc.pair_contact_offset[i+1]-desc.pair_contact_offset[i]))]
  slots = [i for i in np.flatnonzero(result["contact_mask"].cpu().numpy()[0] > .5)
           if capsule in pairs[i]]
  contacts = [c for c in cpu.contact if capsule in c.geom]
  assert len(slots) == len(contacts) == 3
  for slot, contact in zip(slots, contacts):
    assert pairs[slot] == tuple(contact.geom)
    np.testing.assert_allclose(result["contact_distance"].cpu().numpy()[0, slot], contact.dist,
                               rtol=1e-6, atol=1e-8)
    np.testing.assert_allclose(result["contact_position"].cpu().numpy()[0, slot], contact.pos,
                               rtol=1e-6, atol=3e-8)
    np.testing.assert_allclose(result["contact_normal"].cpu().numpy()[0, slot], contact.frame[:3],
                               rtol=1e-6, atol=1e-7)
  # The missing contact formerly suppressed the capsule's angular acceleration
  # by hundreds of rad/s^2. Compare every capsule DOF and both finger force
  # contributions. Small differences in the cancelled finger accelerations at
  # finite PGS termination belong to the separate solver-iterate qualification;
  # this regression does not claim that all solver families are qualified.
  np.testing.assert_allclose(acceleration.cpu().numpy()[0, 8:14], cpu.qacc[8:14],
                             rtol=3e-5, atol=1e-5)
  indices = np.r_[0:2, 8:14]
  np.testing.assert_allclose(result["qfrc_constraint"].cpu().numpy()[0, indices],
                             cpu.qfrc_constraint[indices], rtol=3e-5, atol=1e-7)


def _pair_model(reverse):
  box = '<geom name="box" type="box" size=".10 .06 .045"/>'
  capsule = '''<body pos="0 0 .08"><freejoint/>
    <geom name="capsule" type="capsule" size=".035 .09" mass=".2"
          quat=".7071067811865476 .7071067811865476 0 0"/></body>'''
  world = capsule+box if reverse else box+capsule
  return mujoco.MjModel.from_xml_string(f'''<mujoco><option gravity="0 0 0"/>
    <worldbody>{world}</worldbody><contact>
    <pair geom1="box" geom2="capsule" margin=".000100000012345"/>
    </contact></mujoco>''')


def test_capsule_source_constants_preserve_margin_words_and_budget():
  from mujoco_metal.coupled_constraints import lower_coupled_constraints, _source_float_words
  from mujoco_metal.capacity import _runtime_buffer_sizes
  model = _pair_model(False)
  descriptor = lower_coupled_constraints(model)
  words = descriptor.pair_margin_source_words.reshape(3, descriptor.npairs)
  np.testing.assert_array_equal(words.astype(float).sum(axis=0), model.pair_margin)
  assert not descriptor.pair_margin_source_words.flags.writeable
  values = np.array([.035, .045, .000100000012345, np.pi, 1e-15])
  np.testing.assert_array_equal(_source_float_words(values).reshape(3, -1).astype(float).sum(axis=0), values)
  sizes = dict(_runtime_buffer_sizes(model, 1, npairs=1))
  assert sizes["model.coupled.geom_size_source"] == 9 * model.ngeom
  assert sizes["model.coupled.pair_margin_source"] == 3


@pytest.mark.gpu
@pytest.mark.skipif(os.getenv("MUJOCO_METAL_RUN_GPU") != "1", reason="native capsule source poses")
@pytest.mark.parametrize("reverse", [False, True])
def test_native_source_capsule_face_edge_inside_and_empty_worlds(reverse):
  from mujoco_metal import MetalSimulation
  model = _pair_model(reverse)
  # Exact horizontal and near-horizontal ties, oblique edge contacts,
  # endpoint contacts, center-inside and contact-free worlds share one batch.
  angles = np.array([0, 1e-8, -1e-8, .14, -.37, 1.1, .44, .22])
  q = np.tile(model.qpos0, (len(angles), 1))
  q[:, :3] = [[0, 0, .0799], [0, 0, .0799], [0, 0, .0799],
              [.10, .04, .085], [.13, .05, .07], [.05, .04, .03],
              [.01, .01, .001], [.8, .1, .4]]
  q[:, 3] = np.cos(angles/2)
  q[:, 4] = np.sin(angles/2)
  q[:, 5:] = 0
  v = np.tile([.03, -.04, .01, .02, -.03, .04], (len(angles), 1))
  refs = [mujoco.MjData(model) for _ in angles]
  for i, data in enumerate(refs):
    data.qpos[:], data.qvel[:] = q[i], v[i]
    mujoco.mj_forward(model, data)
  sim = MetalSimulation(model, len(angles), profile="integrated_euler_v1")
  qh, vh = q.astype(np.float32), v.astype(np.float32)
  ql, vl = (q-qh).astype(np.float32), (v-vh).astype(np.float32)
  qt = (q-qh.astype(float)-ql.astype(float)).astype(np.float32)
  vt = (v-vh.astype(float)-vl.astype(float)).astype(np.float32)
  sim.reset(qpos=qh, qvel=vh)
  checkpoint = sim.snapshot()
  checkpoint["device"] = replace(checkpoint["device"],
      precision_words=np.concatenate((ql, qt, vl, vt), axis=1))
  sim.restore(checkpoint)
  _, status, _ = sim._acceleration(sim.state._qpos, sim.state._qvel)
  assert status.cpu().numpy().tolist() == [0]*len(angles)
  result = sim._last_coupled
  masks = result["contact_mask"].cpu().numpy()
  for i, data in enumerate(refs):
    slots = np.flatnonzero(masks[i] > .5)
    assert len(slots) == data.ncon, (i, q[i], slots, data.ncon)
    for slot, contact in zip(slots, data.contact):
      for key, expected in (("contact_distance", contact.dist),
                            ("contact_position", contact.pos),
                            ("contact_normal", contact.frame[:3])):
        np.testing.assert_allclose(result[key].cpu().numpy()[i, slot], expected,
                                   rtol=1e-6, atol=5e-8, err_msg=f"world {i}: {key}")
  assert refs[-1].ncon == 0 and not masks[-1].any()
  # A contact-free row keeps zero force while adjacent worlds are colliding.
  np.testing.assert_array_equal(result["qfrc_constraint"].cpu().numpy()[-1], 0)
  # Validation precedes all contact-buffer clearing, so a bad borrowed FK
  # plane cannot destroy the preceding assembly in healthy worlds.
  stage = sim._coupled_constraints
  poses = sim._smooth.run_device(sim.state._qpos, sim.state._qvel)["poses"]
  count_before = stage._workspace["contact_pair_count"].cpu().numpy().copy()
  records_before = stage._workspace["contact_pair_records"].cpu().numpy().copy()
  bad = dict(poses, geom_xmat_low=poses["geom_xmat_low"][:, :, :3].contiguous())
  with pytest.raises(ValueError, match="geom_xmat_low.*shape"):
    stage.generate_candidates(bad, sim.state._qvel)
  np.testing.assert_array_equal(stage._workspace["contact_pair_count"].cpu().numpy(), count_before)
  np.testing.assert_array_equal(stage._workspace["contact_pair_records"].cpu().numpy(), records_before)
  checkpoint = sim.snapshot()
  assert sim.step().cpu().numpy().tolist() == [0]*len(angles)
  expected = sim.snapshot()
  sim.restore(checkpoint)
  assert sim.step().cpu().numpy().tolist() == [0]*len(angles)
  repeated = sim.snapshot()
  for name in ("qpos", "qvel", "qacc", "precision_words", "time"):
    assert getattr(repeated["device"], name).tobytes() == getattr(expected["device"], name).tobytes()
  assert repeated["warmstart"].tobytes() == expected["warmstart"].tobytes()


@pytest.mark.gpu
@pytest.mark.skipif(os.getenv("MUJOCO_METAL_RUN_GPU") != "1", reason="native box/plane source poses")
@pytest.mark.parametrize("reverse", [False, True])
def test_box_plane_source_depth_corner_order_retained_rotation_and_empty_worlds(reverse):
  from mujoco_metal import MetalSimulation
  plane = '<body><geom name="plane" type="plane" size="2 2 .1"/></body>'
  box = '<body pos="0 0 .0499"><freejoint/><geom name="box" type="box" size=".05 .05 .05" mass=".5"/></body>'
  model = mujoco.MjModel.from_xml_string(f'''<mujoco><option gravity="0 0 0"/>
    <worldbody>{box+plane if reverse else plane+box}</worldbody>
    <contact><pair geom1="box" geom2="plane" margin=".000100000012345"/></contact></mujoco>''')
  angles = np.asarray([0, 1e-8, -1e-8, .13, -.27, 0])
  q = np.tile(model.qpos0, (len(angles), 1))
  q[:, 3], q[:, 5] = np.cos(angles/2), np.sin(angles/2)
  q[-1, 2] = .3
  v = np.tile([.001, 0, -.001, 0, .003, 0], (len(angles), 1))
  refs = [mujoco.MjData(model) for _ in angles]
  for world, data in enumerate(refs):
    data.qpos[:], data.qvel[:] = q[world], v[world]
    mujoco.mj_forward(model, data)
  sim = MetalSimulation(model, len(angles), profile="integrated_euler_v1")
  qh, vh = q.astype(np.float32), v.astype(np.float32)
  ql, vl = (q-qh).astype(np.float32), (v-vh).astype(np.float32)
  qt, vt = (q-qh.astype(float)-ql).astype(np.float32), (v-vh.astype(float)-vl).astype(np.float32)
  sim.reset(qpos=qh, qvel=vh)
  before = sim.snapshot()
  before["device"] = replace(before["device"], precision_words=np.concatenate((ql, qt, vl, vt), axis=1))
  sim.restore(before)
  _, status, _ = sim._acceleration(sim.state._qpos, sim.state._qvel)
  np.testing.assert_array_equal(status.cpu().numpy(), 0)
  result = sim._last_coupled
  count = sim._coupled_constraints._workspace["contact_pair_count"].cpu().numpy().reshape(len(angles), -1)
  records = sim._coupled_constraints._workspace["contact_pair_records"].cpu().numpy().reshape(len(angles), -1, 37)
  for world, data in enumerate(refs):
    assert count[world, 0] == data.ncon
    for slot, contact in enumerate(data.contact):
      # Inspect retained distances as well as published float32 contact fields.
      stored = records[world, slot]
      np.testing.assert_allclose(float(stored[12])+float(stored[9])+float(stored[10]),
                                 contact.dist, rtol=0, atol=2e-14)
      np.testing.assert_allclose(result["contact_position"].cpu().numpy()[world, slot],
                                 contact.pos, rtol=2e-6, atol=3e-8)
  assert count[-1, 0] == 0
