# Copyright 2026 The MuJoCo Metal contributors
# Licensed under the Apache License, Version 2.0 (the "License");

"""CPU-only contact lowering guards and pinned MuJoCo 3.10.0 oracle checks."""

import os

import mujoco
import numpy as np
import pytest

from mujoco_metal.contact import _get_impedance
from mujoco_metal.contact import lower_contacts
from mujoco_metal.contact import MetalContact
from mujoco_metal.metal_kinematics import MetalKinematics
from mujoco_metal.model import load_model
from mujoco_metal.smooth_metal import MetalSmoothDynamics


def _plane_sphere_xml(sphere_z=0.15, radius=0.2, margin=0.01, gap=0.0):
  return f"""<mujoco><option timestep="0.002"/><worldbody>
    <geom name="ground" type="plane" size="2 2 .1" margin="{margin}" gap="{gap}" condim="1"/>
    <body name="ball" pos="0 0 {sphere_z}"><freejoint/>
      <geom name="ballgeom" type="sphere" size="{radius}" condim="1"/>
    </body>
  </worldbody></mujoco>"""


def test_plane_sphere_lowering_includes_world_root_pair_and_model_parameters():
  source = mujoco.MjModel.from_xml_string(_plane_sphere_xml())
  desc = lower_contacts(source)
  assert desc.pair_count == 1
  assert desc.geom1.tolist() == [0]
  assert desc.geom2.tolist() == [1]
  assert desc.radius1.tolist() == [-1.0]
  assert desc.radius2.tolist() == pytest.approx([0.2])
  assert desc.margin.tolist() == pytest.approx([0.01])
  assert desc.gap.tolist() == pytest.approx([0.0])
  assert desc.solref[0].tolist() == pytest.approx(source.geom_solref[0])
  with pytest.raises(ValueError):
    desc.geom1[0] = 8


def test_plane_sphere_signed_distance_and_surface_point_match_cpu_oracle():
  model = mujoco.MjModel.from_xml_string(_plane_sphere_xml())
  data = mujoco.MjData(model)
  mujoco.mj_forward(model, data)
  assert data.ncon == 1
  contact = data.contact[0]
  assert contact.geom1 == 0 and contact.geom2 == 1
  center_z = float(data.geom_xpos[1, 2])
  expected_dist = center_z - model.geom_size[1, 0]
  assert contact.dist == pytest.approx(expected_dist, abs=1e-12)
  desc = lower_contacts(model)
  assert desc.pair_count == 1
  assert _get_impedance(contact.solimp, contact.dist, contact.includemargin)[0] == pytest.approx(
      model.geom_solimp[0, 1], abs=1e-10
  )


def test_sphere_sphere_pair_is_model_derived_and_collision_matches_oracle():
  xml = """<mujoco><worldbody>
    <body pos="0 0 0"><freejoint/><geom type="sphere" size=".2" condim="1"/></body>
    <body pos="0.35 0 0"><freejoint/><geom type="sphere" size=".2" condim="1"/></body>
  </worldbody></mujoco>"""
  model = mujoco.MjModel.from_xml_string(xml)
  data = mujoco.MjData(model)
  mujoco.mj_forward(model, data)
  desc = lower_contacts(model)
  assert desc.pair_count == 1
  assert data.ncon == 1
  np.testing.assert_array_equal(data.contact[0].geom, [0, 1])
  assert data.contact[0].dist == pytest.approx(-0.05, abs=1e-12)


def test_contact_guards_reject_unsupported_families_and_parameters():
  box = mujoco.MjModel.from_xml_string(
      '<mujoco><worldbody><geom type="plane" size="2 2 .1"/>'
      '<body pos="0 0 .1"><freejoint/><geom type="box" size=".1 .1 .1"/>'
      '</body></worldbody></mujoco>'
  )
  with pytest.raises(ValueError, match="does not support collidable geom pair"):
    lower_contacts(box)

  frictional = mujoco.MjModel.from_xml_string(
      _plane_sphere_xml().replace('condim="1"', 'condim="3"')
  )
  with pytest.raises(ValueError, match="condim"):
    lower_contacts(mujoco.MjModel.from_xml_string(
        _plane_sphere_xml().replace('condim="1"', 'condim="4"')
    ))
  assert lower_contacts(frictional).condim.tolist() == [3]

  explicit_pair = mujoco.MjModel.from_xml_string(
      '<mujoco><worldbody><geom name="floor" type="plane" size="2 2 .1"/>'
      '<body pos="0 0 .1"><freejoint/><geom name="ball" type="sphere" size=".1"/>'
      '</body></worldbody><contact><pair geom1="floor" geom2="ball"/></contact></mujoco>'
  )
  with pytest.raises(ValueError, match="explicit pairs"):
    lower_contacts(explicit_pair)


def test_masking_removes_noncolliding_candidates():
  model = mujoco.MjModel.from_xml_string(
      '<mujoco><worldbody><geom type="plane" size="2 2 .1" contype="0" conaffinity="0"/>'
      '<body pos="0 0 .1"><freejoint/><geom type="sphere" size=".1"/>'
      '</body></worldbody></mujoco>'
  )
  assert lower_contacts(model).pair_count == 0


def test_priority_and_direct_solref_mixing_match_contact_records():
  xml = """<mujoco><worldbody>
    <geom name="floor" type="plane" size="2 2 .1" condim="1"
      priority="2" solref="-800 -30" solimp=".82 .93 .006 .4 2"/>
    <body pos="0 0 .18"><freejoint/><geom name="ball" type="sphere"
      size=".2" condim="1" priority="1" solref=".04 2"/>
    </body>
  </worldbody></mujoco>"""
  model = mujoco.MjModel.from_xml_string(xml)
  data = mujoco.MjData(model)
  mujoco.mj_forward(model, data)
  desc = lower_contacts(model)
  np.testing.assert_allclose(desc.solref[0], data.contact[0].solref, atol=1e-7)
  np.testing.assert_allclose(desc.solimp[0], data.contact[0].solimp, atol=1e-7)
  assert desc.margin[0] == pytest.approx(data.contact[0].includemargin)


def test_pyramidal_condim3_friction_mixing_matches_contact_record():
  xml = """<mujoco><option cone="pyramidal"/><worldbody>
    <geom name="floor" type="plane" size="2 2 .1" condim="3"
      friction=".4 .02 .01" priority="2"/>
    <body pos="0 0 .18"><freejoint/><geom name="ball" type="sphere"
      size=".2" condim="3" friction=".8 .03 .02" priority="1"/>
    </body></worldbody></mujoco>"""
  model = mujoco.MjModel.from_xml_string(xml)
  data = mujoco.MjData(model)
  mujoco.mj_forward(model, data)
  assert data.ncon == 1
  assert data.contact[0].dim == 3
  desc = lower_contacts(model)
  assert desc.condim.tolist() == [3]
  # The higher-priority geom supplies all contact parameters.
  assert desc.friction[0].tolist() == pytest.approx([0.4, 0.4])
  np.testing.assert_allclose(desc.solref[0], data.contact[0].solref, atol=1e-7)
  np.testing.assert_allclose(desc.solimp[0], data.contact[0].solimp, atol=1e-7)


def _mps(array):
  import torch

  return torch.as_tensor(array, dtype=torch.float32, device="mps")


def _gpu_contact_acceleration(model, qpos, qvel):
  import torch

  descriptor = load_model(model)
  qpos_t, qvel_t = _mps(qpos), _mps(qvel)
  fk = MetalKinematics(descriptor, batch_size=len(qpos)).run_device(qpos_t)
  smooth = MetalSmoothDynamics(descriptor, batch_size=len(qpos)).run_device(
      qpos_t, qvel_t
  )
  free_acceleration = torch.linalg.solve(
      smooth["mass_matrix"], -smooth["qfrc_bias"].unsqueeze(-1)
  ).squeeze(-1)
  return MetalContact(model, batch_size=len(qpos)).run_device(
      fk, smooth["mass_matrix"], free_acceleration, qvel_t
  )


@pytest.mark.gpu
@pytest.mark.skipif(
    os.environ.get("MUJOCO_METAL_RUN_GPU") != "1",
    reason="requires explicit MUJOCO_METAL_RUN_GPU=1 and idle GPU",
)
def test_native_plane_sphere_contact_acceleration_matches_cpu_oracle():
  import torch

  xml = _plane_sphere_xml(sphere_z=0.15, radius=0.2, margin=0.01)
  model = mujoco.MjModel.from_xml_string(xml)
  model.opt.gravity[:] = 0
  descriptor = load_model(model)
  qpos = np.tile(model.qpos0, (2, 1))
  qpos[1, 2] = 0.5
  qvel = np.zeros((2, model.nv))
  fk_stage = MetalKinematics(descriptor, batch_size=2)
  smooth_stage = MetalSmoothDynamics(descriptor, batch_size=2)
  contact_stage = MetalContact(model, batch_size=2)
  device = torch.device("mps")
  qpos_t = torch.as_tensor(qpos, dtype=torch.float32, device=device)
  qvel_t = torch.as_tensor(qvel, dtype=torch.float32, device=device)
  fk = fk_stage.run_device(qpos_t)
  smooth = smooth_stage.run_device(qpos_t, qvel_t)
  actual = contact_stage.run_device(
      fk, smooth["mass_matrix"], torch.zeros_like(qvel_t), qvel_t
  )
  assert torch.all(actual["status"] == 0)
  expected = []
  for row in range(len(qpos)):
    data = mujoco.MjData(model)
    data.qpos[:] = qpos[row]
    data.qvel[:] = qvel[row]
    mujoco.mj_forward(model, data)
    expected.append(data.qacc.copy())
  np.testing.assert_allclose(
      actual["qacc"].cpu().numpy(), expected, rtol=3e-3, atol=3e-3
  )
  assert actual["mask"].cpu().numpy()[:, 0].tolist() == [1.0, 0.0]


@pytest.mark.gpu
@pytest.mark.skipif(
    os.environ.get("MUJOCO_METAL_RUN_GPU") != "1",
    reason="requires explicit MUJOCO_METAL_RUN_GPU=1 and idle GPU",
)
def test_native_sphere_sphere_contact_acceleration_matches_cpu_oracle():
  import torch

  xml = """<mujoco><option timestep=".002" gravity="0 0 0"/><worldbody>
    <body pos="0 0 0"><freejoint/><geom type="sphere" size=".2" condim="1"/></body>
    <body pos=".35 0 0"><freejoint/><geom type="sphere" size=".2" condim="1"/></body>
  </worldbody></mujoco>"""
  model = mujoco.MjModel.from_xml_string(xml)
  descriptor = load_model(model)
  qpos = np.tile(model.qpos0, (1, 1))
  qvel = np.zeros((1, model.nv))
  device = torch.device("mps")
  qpos_t = torch.as_tensor(qpos, dtype=torch.float32, device=device)
  qvel_t = torch.as_tensor(qvel, dtype=torch.float32, device=device)
  fk = MetalKinematics(descriptor).run_device(qpos_t)
  smooth = MetalSmoothDynamics(descriptor).run_device(qpos_t, qvel_t)
  actual = MetalContact(model).run_device(
      fk, smooth["mass_matrix"], torch.zeros_like(qvel_t), qvel_t
  )
  assert torch.all(actual["status"] == 0)
  data = mujoco.MjData(model)
  data.qpos[:] = qpos[0]
  mujoco.mj_forward(model, data)
  np.testing.assert_allclose(
      actual["qacc"][0].cpu().numpy(), data.qacc, rtol=3e-3, atol=3e-3
  )


@pytest.mark.gpu
@pytest.mark.skipif(
    os.environ.get("MUJOCO_METAL_RUN_GPU") != "1",
    reason="requires explicit MUJOCO_METAL_RUN_GPU=1 and idle GPU",
)
def test_native_coupled_sphere_contacts_with_nonzero_velocity_match_cpu():
  xml = """<mujoco><option timestep=".002" gravity="0 0 0"/><worldbody>
    <body pos="0 0 0"><freejoint/><geom type="sphere" size=".2" condim="1"/></body>
    <body pos=".35 0 0"><freejoint/><geom type="sphere" size=".2" condim="1"/></body>
    <body pos=".7 0 0"><freejoint/><geom type="sphere" size=".2" condim="1"/></body>
  </worldbody></mujoco>"""
  model = mujoco.MjModel.from_xml_string(xml)
  qpos = model.qpos0[None, :].copy()
  qvel = np.zeros((1, model.nv))
  qvel[0, 0] = 0.3
  qvel[0, 6] = -0.2
  qvel[0, 12] = 0.1
  actual = _gpu_contact_acceleration(model, qpos, qvel)
  assert actual["status"].cpu().numpy().tolist() == [0]
  assert actual["mask"].cpu().numpy().tolist() == [[1.0, 0.0, 1.0]]
  data = mujoco.MjData(model)
  data.qpos[:] = qpos[0]
  data.qvel[:] = qvel[0]
  mujoco.mj_forward(model, data)
  np.testing.assert_allclose(
      actual["qacc"][0].cpu().numpy(), data.qacc, rtol=4e-3, atol=4e-3
  )


@pytest.mark.gpu
@pytest.mark.skipif(
    os.environ.get("MUJOCO_METAL_RUN_GPU") != "1",
    reason="requires explicit MUJOCO_METAL_RUN_GPU=1 and idle GPU",
)
def test_native_pyramidal_friction_contact_matches_cpu_oracle():
  xml = """<mujoco><compiler angle="radian"/><option timestep=".002"
  gravity="0 0 0" cone="pyramidal"/><worldbody>
    <geom type="plane" size="2 2 .1" euler="0 .28 0" condim="3"
      friction=".7 .01 .01"/>
    <body pos=".02 -.01 .18"><freejoint/><geom type="sphere"
      pos=".045 .02 0" size=".2"
      condim="3" friction=".7 .01 .01"/></body>
  </worldbody></mujoco>"""
  model = mujoco.MjModel.from_xml_string(xml)
  qpos = model.qpos0[None, :].copy()
  qvel = np.zeros((1, model.nv))
  qvel[0, 0] = 1.0
  qvel[0, 3:6] = [0.2, -0.3, 0.1]
  actual = _gpu_contact_acceleration(model, qpos, qvel)
  assert actual["status"].cpu().numpy().tolist() == [0]
  data = mujoco.MjData(model)
  data.qpos[:] = qpos[0]
  data.qvel[:] = qvel[0]
  mujoco.mj_forward(model, data)
  assert data.ncon == 1 and data.contact[0].dim == 3
  # Four pyramidal edges carry force; slot zero is the retained normal row.
  assert actual["force_rows"][0, 0, 0].item() == 0.0
  np.testing.assert_allclose(
      actual["qacc"][0].cpu().numpy(), data.qacc, rtol=8e-3, atol=8e-3
  )
  np.testing.assert_allclose(
      actual["qfrc_contact"][0].cpu().numpy(), data.qfrc_constraint,
      rtol=1e-2, atol=1e-2,
  )
