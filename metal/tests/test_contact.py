# Copyright 2026 The MuJoCo Metal contributors
# Licensed under the Apache License, Version 2.0 (the "License");

"""CPU-only contact lowering guards and pinned MuJoCo 3.10.0 oracle checks."""

import os
from pathlib import Path

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
  assert _get_impedance(contact.solimp, contact.dist, contact.includemargin)[
      0
  ] == pytest.approx(model.geom_solimp[0, 1], abs=1e-10)


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
      "</body></worldbody></mujoco>"
  )
  with pytest.raises(ValueError, match="does not support collidable geom pair"):
    lower_contacts(box)

  frictional = mujoco.MjModel.from_xml_string(
      _plane_sphere_xml().replace('condim="1"', 'condim="3"')
  )
  with pytest.raises(ValueError, match="condim"):
    lower_contacts(
        mujoco.MjModel.from_xml_string(
            _plane_sphere_xml().replace('condim="1"', 'condim="4"')
        )
    )
  assert lower_contacts(frictional).condim.tolist() == [3]
  elliptic = mujoco.MjModel.from_xml_string(
      _plane_sphere_xml()
      .replace('condim="1"', 'condim="3"')
      .replace("<mujoco>", '<mujoco><option cone="elliptic"/>')
  )
  with pytest.raises(ValueError, match="pyramidal cone"):
    lower_contacts(elliptic)

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
      "</body></worldbody></mujoco>"
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


def _saved_friction_transition():
  model = mujoco.MjModel.from_xml_path(
      str(Path(__file__).parents[1] / "examples" / "friction_laboratory.xml")
  )
  qpos = np.array(
      [
          [
              -0.31668678,
              0.0,
              0.11998787,
              0.99870896,
              0.0,
              0.05079811,
              0.0,
              0.3495785,
              -6.8982237e-10,
              0.11990365,
              0.81985062,
              -5.9871010e-9,
              0.57257748,
              -3.9654973e-9,
              1.0161743,
              6.6931674e-9,
              0.11963283,
              0.3988457,
              5.5450663e-8,
              0.91701806,
              1.9204172e-9,
          ]
      ],
      dtype=np.float32,
  )
  qvel = np.array(
      [
          [
              1.2693158,
              0.0,
              -0.011822759,
              0.0,
              0.63867962,
              0.0,
              0.93312311,
              -1.2226725e-7,
              -1.7899617e-4,
              -9.7863233e-7,
              7.6359115,
              -2.3501532e-6,
              0.92674041,
              1.2366859e-8,
              -5.4390370e-7,
              6.9035923e-7,
              7.7346683,
              -1.7903335e-7,
          ]
      ],
      dtype=np.float32,
  )
  return model, qpos, qvel


def test_saved_friction_laboratory_transition_matches_cpu_oracle():
  model, qpos, qvel = _saved_friction_transition()
  data = mujoco.MjData(model)
  data.qpos[:], data.qvel[:] = qpos[0], qvel[0]
  mujoco.mj_forward(model, data)
  assert data.ncon == 3 and data.nefc == 12
  np.testing.assert_allclose(
      data.efc_force[: data.nefc],
      [
          0,
          0,
          3.70068223,
          0,
          0.7546075,
          0.75457632,
          1.3809145,
          0.12826932,
          0.73575032,
          0.73575061,
          0.73575122,
          0.73574971,
      ],
      rtol=1e-5,
      atol=3e-4,
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
  diagnostic = actual["solver_diagnostics"].cpu().numpy()[0]
  assert diagnostic[0] <= 1e-6 and diagnostic[1] <= 320, diagnostic
  data = mujoco.MjData(model)
  data.qpos[:] = qpos[0]
  data.qvel[:] = qvel[0]
  mujoco.mj_forward(model, data)
  assert data.ncon == 1 and data.contact[0].dim == 3
  # Four pyramidal edges carry force; slot zero is the retained normal row.
  assert actual["force_rows"][0, 0, 0].item() == 0.0
  np.testing.assert_allclose(
      actual["qacc"][0].cpu().numpy(), data.qacc, rtol=2e-5, atol=1e-4
  )
  np.testing.assert_allclose(
      actual["qfrc_contact"][0].cpu().numpy(),
      data.qfrc_constraint,
      rtol=2e-5,
      atol=4e-4,
  )


@pytest.mark.gpu
@pytest.mark.skipif(
    os.environ.get("MUJOCO_METAL_RUN_GPU") != "1",
    reason="requires explicit MUJOCO_METAL_RUN_GPU=1 and idle GPU",
)
def test_native_saved_friction_transition_matches_cpu_oracle():
  model, qpos, qvel = _saved_friction_transition()
  actual = _gpu_contact_acceleration(model, qpos, qvel)
  assert actual["status"].cpu().numpy().tolist() == [0]
  diagnostic = actual["solver_diagnostics"].cpu().numpy()[0]
  assert diagnostic[0] <= 1e-6 and diagnostic[1] <= 320, diagnostic
  data = mujoco.MjData(model)
  data.qpos[:], data.qvel[:] = qpos[0], qvel[0]
  mujoco.mj_forward(model, data)
  assert data.ncon == 3
  np.testing.assert_allclose(
      actual["qacc"][0].cpu().numpy(), data.qacc, rtol=2e-5, atol=1e-4
  )
  np.testing.assert_allclose(
      actual["qfrc_contact"][0].cpu().numpy(),
      data.qfrc_constraint,
      rtol=2e-5,
      atol=4e-4,
  )


@pytest.mark.gpu
@pytest.mark.skipif(
    os.environ.get("MUJOCO_METAL_RUN_GPU") != "1",
    reason="requires explicit MUJOCO_METAL_RUN_GPU=1 and idle GPU",
)
def test_friction_profile_400_step_sliding_and_separating_batch():
  from mujoco_metal.simulation import MetalSimulation

  xml = """<mujoco><option timestep=".002" gravity="0 0 -9.81"
      cone="pyramidal" iterations="100" tolerance="1e-10"/>
    <worldbody><geom type="plane" size="3 3 .1" condim="3"
        friction=".6 .01 .01"/>
      <body pos="0 0 .098"><freejoint/><geom type="sphere" size=".1"
          mass=".3" condim="3" friction=".6 .01 .01"/></body>
    </worldbody></mujoco>"""
  model = mujoco.MjModel.from_xml_string(xml)
  qpos = np.tile(model.qpos0, (3, 1)).astype(np.float32)
  qvel = np.zeros((3, model.nv), dtype=np.float32)
  qvel[0, 0] = 0.8
  qvel[1, 0] = 0.8
  qvel[1, 2] = 2.0  # rises, then returns to impact within the rollout
  qvel[2, 0] = 0.8
  qvel[2, 2] = 5.0  # remains separated for all 400 steps
  simulation = MetalSimulation(
      model, 3, qpos=qpos, qvel=qvel, profile="friction_contact_euler_v1"
  )
  references = [mujoco.MjData(model) for _ in range(3)]
  for i, data in enumerate(references):
    data.qpos[:] = qpos[i]
    data.qvel[:] = qvel[i]
  errors = np.zeros(2)
  contact_steps = np.zeros(3, dtype=np.int32)
  for _ in range(400):
    simulation.step()
    for data in references:
      mujoco.mj_step(model, data)
    state = simulation.state.snapshot()
    if np.any(state.status):
      diagnostics = simulation._contact._workspace[
          "solver_diagnostics"
      ].reshape(3, 2)
      force_rows = simulation._contact._workspace["force"].reshape(3, -1, 5)
      raise AssertionError(
          f"step status={state.status.tolist()} diagnostics="
          f"{diagnostics.cpu().numpy().tolist()} force_rows="
          f"{force_rows[1].cpu().numpy().tolist()} qvel="
          f"{state.qvel[1].tolist()}"
      )
    for i, data in enumerate(references):
      errors = np.maximum(
          errors,
          [
              np.max(np.abs(state.qpos[i] - data.qpos)),
              np.max(np.abs(state.qvel[i] - data.qvel)),
          ],
      )
      contact_steps[i] += data.ncon > 0
  assert contact_steps[0] > 300
  assert contact_steps[1] > 20
  assert contact_steps[2] < 20
  assert errors[0] < 0.01 and errors[1] < 0.1, errors
