# Copyright 2026 The MuJoCo Metal contributors
# Licensed under the Apache License, Version 2.0.

"""Unit and numerical qualification tests for coupled constraints."""

import os
import mujoco
import numpy as np
import pytest

from mujoco_metal.coupled_constraints import (
    CoupledConstraintDescriptor,
    CoupledSolverSettings,
    coupled_constraint_oracle,
    lower_coupled_constraints,
    MetalCoupledConstraints,
)
from mujoco_metal.model import load_model
from mujoco_metal.metal_kinematics import MetalKinematics
from mujoco_metal.smooth_metal import MetalSmoothDynamics

COUPLED_XML = """<mujoco model="coupled_test">
  <compiler angle="radian"/>
  <option timestep="0.002" integrator="Euler" iterations="1000" tolerance="1e-6">
    <flag contact="enable" equality="enable" limit="enable" frictionloss="enable"/>
  </option>
  <worldbody>
    <geom name="floor" type="plane" size="5 5 0.1" pos="0 0 0"/>
    <body name="lever1" pos="0 0 1">
      <joint name="j1" type="hinge" axis="0 1 0" range="-0.5 0.5" limited="true" frictionloss="0.1" damping="0.05"/>
      <geom name="g_lever1" type="sphere" size="0.2" pos="0.5 0 0" mass="1"/>
    </body>
    <body name="lever2" pos="0 1 1">
      <joint name="j2" type="hinge" axis="0 1 0" damping="0.05"/>
      <geom name="g_lever2" type="sphere" size="0.2" pos="0.5 0 0" mass="1"/>
    </body>
    <body name="ball" pos="0.5 0 1.5">
      <joint name="ball_z" type="slide" axis="0 0 1"/>
      <geom name="g_ball" type="sphere" size="0.1" mass="0.5"/>
    </body>
  </worldbody>
  <equality>
    <joint joint1="j2" joint2="j1" polycoef="0 1.5 0 0 0"/>
  </equality>
  <actuator>
    <motor joint="j1" ctrlrange="-10 10"/>
  </actuator>
  <sensor>
    <jointpos joint="j1"/>
    <jointvel joint="j1"/>
  </sensor>
</mujoco>"""


def test_coupled_lowering_and_immutability():
  model = mujoco.MjModel.from_xml_string(COUPLED_XML)
  desc = lower_coupled_constraints(model)
  assert isinstance(desc, CoupledConstraintDescriptor)
  assert desc.nv == 3
  assert desc.neq == 1
  assert desc.nc >= 1
  assert desc.nr <= 96
  assert desc.qpos0.flags.writeable is False
  assert desc.joint_sol_params.flags.writeable is False
  with pytest.raises(ValueError):
    desc.qpos0[0] = 1.0


def test_coupled_lowering_unsupported_geoms_rejected():
  xml = COUPLED_XML.replace('type="sphere" size="0.2"', 'type="cylinder" size="0.1 0.1"')
  m = mujoco.MjModel.from_xml_string(xml)
  with pytest.raises(ValueError, match="only plane, sphere, capsule, and box"):
    lower_coupled_constraints(m)



def test_coupled_lowering_condim4_and6_supported():
  xml = COUPLED_XML.replace('<compiler angle="radian"/>', '<compiler angle="radian"/><default><geom condim="4"/></default>')
  m = mujoco.MjModel.from_xml_string(xml)
  assert lower_coupled_constraints(m).contact_condim.max() == 4
  xml = COUPLED_XML.replace('<compiler angle="radian"/>', '<compiler angle="radian"/><default><geom condim="6"/></default>')
  m = mujoco.MjModel.from_xml_string(xml)
  assert lower_coupled_constraints(m).contact_condim.max() == 6


def test_coupled_lowering_elliptic_cone_is_recorded():
  xml = COUPLED_XML.replace('tolerance="1e-6">', 'tolerance="1e-6" cone="elliptic">')
  m = mujoco.MjModel.from_xml_string(xml)
  desc = lower_coupled_constraints(m)
  assert desc.cone_type == int(mujoco.mjtCone.mjCONE_ELLIPTIC)


def test_coupled_lowering_rejects_nondefault_noslip_iterations():
  xml = '''<mujoco><option cone="elliptic" noslip_iterations="5"/>
    <worldbody><geom type="plane" size="1 1 .1"/>
      <body pos="0 0 .09"><freejoint/><geom type="sphere" size=".1"
        condim="6"/></body></worldbody></mujoco>'''
  model = mujoco.MjModel.from_xml_string(xml)
  assert model.opt.noslip_iterations == 5
  with pytest.raises(ValueError, match="noslip_iterations is unsupported"):
    lower_coupled_constraints(model)


def test_coupled_lowering_rejects_unimplemented_solver_selection():
  xml = '''<mujoco><option solver="CG"/>
    <worldbody><geom type="plane" size="1 1 .1"/>
      <body pos="0 0 .09"><freejoint/><geom type="sphere" size=".1"/>
      </body></worldbody></mujoco>'''
  model = mujoco.MjModel.from_xml_string(xml)
  assert model.opt.solver == mujoco.mjtSolver.mjSOL_CG
  with pytest.raises(ValueError, match="CG solver selection is unsupported"):
    lower_coupled_constraints(model)


@pytest.mark.parametrize("cone", ["pyramidal", "elliptic"])
@pytest.mark.parametrize("condim", [1, 3, 4, 6])
def test_coupled_lowering_allocates_exact_cone_dimensions(cone, condim):
  xml = f'''<mujoco><option cone="{cone}"/>
    <worldbody><geom type="plane" size="2 2 .1" condim="{condim}"
      friction=".8 .6 .07"/>
      <body pos="0 0 .15"><freejoint/><geom type="sphere" size=".2"
        condim="{condim}" friction=".8 .6 .07"/></body>
    </worldbody></mujoco>'''
  desc = lower_coupled_constraints(mujoco.MjModel.from_xml_string(xml))
  assert desc.ncontacts_max == 1
  assert desc.contact_condim.tolist() == [condim]
  assert desc.contact_friction[0].tolist() == pytest.approx([.8, .8, .6, .07, .07])
  expected_rows = (
      1 if condim == 1 else
      2 * (condim - 1) if cone == "pyramidal" else
      condim
  )
  assert desc.nr == desc.nr_joint + expected_rows
  assert desc.contact_condim_packed.tolist() == [condim, 0, int(desc.cone_type)]


def test_coupled_explicit_pair_preserves_five_friction_and_solreffriction():
  xml = '''<mujoco><option cone="elliptic"/>
    <worldbody><geom name="floor" type="plane" size="2 2 .1"/>
      <body pos="0 0 .15"><freejoint/><geom name="ball" type="sphere"
        size=".2"/></body></worldbody>
    <contact><pair geom1="floor" geom2="ball" condim="6"
      friction=".8 .6 .07 .04 .03" solreffriction=".03 1"/></contact>
  </mujoco>'''
  desc = lower_coupled_constraints(mujoco.MjModel.from_xml_string(xml))
  np.testing.assert_allclose(desc.friction[0], [.8, .6, .07, .04, .03])
  np.testing.assert_allclose(desc.contact_friction[0], [.8, .6, .07, .04, .03])
  np.testing.assert_allclose(desc.solreffriction[0], [.03, 1.0])
  np.testing.assert_allclose(desc.contact_solreffriction[0], [.03, 1.0])


def test_contact_friction_priority_and_equal_priority_mixing():
  xml = '''<mujoco><option cone="elliptic"/><worldbody>
    <geom name="floor" type="plane" size="2 2 .1" condim="6"
      priority="2" friction=".2 .03 .004"/>
    <body pos="0 0 .15"><freejoint/><geom name="ball" type="sphere"
      size=".2" condim="4" priority="1" friction=".8 .6 .07"/></body>
  </worldbody></mujoco>'''
  prioritized = lower_coupled_constraints(mujoco.MjModel.from_xml_string(xml))
  assert prioritized.contact_condim.tolist() == [6]
  np.testing.assert_allclose(prioritized.contact_friction[0], [.2, .2, .03, .004, .004])

  xml = xml.replace('priority="2"', 'priority="0"').replace('priority="1"', 'priority="0"')
  mixed = lower_coupled_constraints(mujoco.MjModel.from_xml_string(xml))
  assert mixed.contact_condim.tolist() == [6]
  np.testing.assert_allclose(mixed.contact_friction[0], [.8, .8, .6, .07, .07])


def test_contact_friction_lowering_preserves_zero_and_near_zero_coefficients():
  xml = '''<mujoco><option cone="elliptic"/><worldbody>
    <geom type="plane" size="2 2 .1" condim="6" friction="0 .0000001 0"/>
    <body pos="0 0 .15"><freejoint/><geom type="sphere" size=".2"
      condim="6" friction=".0000001 0 .0000002"/></body>
  </worldbody></mujoco>'''
  desc = lower_coupled_constraints(mujoco.MjModel.from_xml_string(xml))
  np.testing.assert_allclose(desc.contact_friction[0], [1e-7, 1e-7, 1e-7, 2e-7, 2e-7])


def _row_capacity_model(cone, contact_count):
  condim = 6
  root = [
      f'<mujoco><option cone="{cone}"/><worldbody>',
      '<geom name="floor" type="plane" size="10 10 .1" condim="6" '
      'friction=".8 .6 .07" contype="1" conaffinity="1"/>',
  ]
  for index in range(contact_count):
    root.append(
        f'<body pos="{index * .4} 0 .095"><joint type="slide" axis="0 0 1"/>'
        '<geom type="sphere" size=".1" condim="6" friction=".8 .6 .07" '
        'contype="1" conaffinity="0"/></body>'
    )
  root.append("</worldbody></mujoco>")
  return mujoco.MjModel.from_xml_string("".join(root))


@pytest.mark.parametrize(("cone", "max_contacts", "overflow_contacts"), [
    ("elliptic", 10, 11),
    ("pyramidal", 7, 8),
])
def test_contact_row_capacity_boundary_rejects_overflow(cone, max_contacts, overflow_contacts):
  desc = lower_coupled_constraints(_row_capacity_model(cone, max_contacts))
  rows_per_contact = 6 if cone == "elliptic" else 10
  assert desc.nr <= 96
  assert desc.contact_condim_packed[-2] + rows_per_contact == (max_contacts - 1) * rows_per_contact + rows_per_contact
  with pytest.raises(ValueError, match="total candidate constraint rows"):
    lower_coupled_constraints(_row_capacity_model(cone, overflow_contacts))


def _elliptic_project(x):
  """Euclidean projection onto the standard second-order cone."""
  x = np.asarray(x, dtype=np.float64)
  tail = np.linalg.norm(x[1:])
  if tail <= x[0]:
    return x.copy()
  if tail <= -x[0]:
    return np.zeros_like(x)
  projected = np.zeros_like(x)
  projected[0] = 0.5 * (tail + x[0])
  projected[1:] = projected[0] * x[1:] / tail
  return projected


@pytest.mark.gpu
@pytest.mark.skipif(os.getenv("MUJOCO_METAL_RUN_GPU") != "1", reason="opt-in GPU")
@pytest.mark.parametrize("cone", ["pyramidal", "elliptic"])
@pytest.mark.parametrize("condim", [1, 3, 4, 6])
def test_native_contact_cone_dimension_matrix(cone, condim):
  """Check actual cone engagement, physical wrench, and an independent KKT residual."""
  import torch

  xml = f'''<mujoco><option timestep=".002" gravity="0 0 0" cone="{cone}"
      iterations="1000" tolerance="1e-10"/>
    <worldbody><geom type="plane" size="2 2 .1" euler="0 .28 .11"
        condim="{condim}" friction=".8 .6 .07"/>
      <body pos=".02 -.01 .18" euler=".2 -.1 .3"><freejoint/>
        <geom type="sphere" pos=".045 .02 0" size=".2" condim="{condim}"
            friction=".8 .6 .07"/>
      </body></worldbody></mujoco>'''
  model = mujoco.MjModel.from_xml_string(xml)
  qpos = model.qpos0[None, :].astype(np.float32)
  qvel = np.zeros((1, model.nv), dtype=np.float32)
  qvel[0, :3] = [.3, -.2, .1]
  qvel[0, 3:6] = [1.3, -2.1, 3.4]
  descriptor = lower_coupled_constraints(model)
  device = torch.device("mps")
  qpos_mps = torch.as_tensor(qpos, device=device)
  qvel_mps = torch.as_tensor(qvel, device=device)
  model_descriptor = load_model(model)
  poses = MetalKinematics(model_descriptor, 1).run_device(qpos_mps)
  dynamics = MetalSmoothDynamics(model_descriptor, 1).run_device(qpos_mps, qvel_mps)
  result = MetalCoupledConstraints(model, 1).run_device(
      poses, dynamics["mass_matrix"], -dynamics["qfrc_bias"], qpos_mps, qvel_mps
  )

  reference = mujoco.MjData(model)
  reference.qpos[:], reference.qvel[:] = qpos[0], qvel[0]
  mujoco.mj_forward(model, reference)
  assert reference.ncon == 1 and int(result["status"][0]) == 0

  # Compare the full active contact block and its Delassus operator against
  # MuJoCo's rows, including cone-specific row count, R, reference and rhs.
  cpu_contact_base = next(
      r for r in range(reference.nefc) if int(reference.efc_type[r]) in (5, 6, 7)
  )
  row_count = (
      1 if condim == 1 else
      2 * (condim - 1) if cone == "pyramidal" else
      condim
  )
  gpu_row_base = descriptor.nr_joint + int(descriptor.contact_condim_packed[1])
  gpu_rows = np.arange(gpu_row_base, gpu_row_base + row_count)
  cpu_rows = np.arange(cpu_contact_base, cpu_contact_base + row_count)
  cpu_J = reference.efc_J.reshape(reference.nefc, model.nv)
  np.testing.assert_allclose(result["J"][0, gpu_rows].cpu().numpy(), cpu_J[cpu_rows], atol=3e-6)
  np.testing.assert_allclose(result["R"][0, gpu_rows].cpu().numpy(), reference.efc_R[cpu_rows], rtol=2e-5, atol=1e-5)
  np.testing.assert_allclose(result["ar"][0, gpu_rows].cpu().numpy(), reference.efc_aref[cpu_rows], rtol=2e-4, atol=2e-3)
  np.testing.assert_allclose(-result["rhs"][0, gpu_rows].cpu().numpy(), reference.efc_b[cpu_rows], rtol=2e-4, atol=2e-3)
  cpu_mass = np.zeros((model.nv, model.nv))
  mujoco.mj_fullM(model, reference, cpu_mass)
  np.testing.assert_allclose(dynamics["mass_matrix"][0].cpu().numpy(), cpu_mass, rtol=2e-5, atol=3e-6)
  cpu_W = cpu_J[cpu_rows] @ np.linalg.solve(cpu_mass, cpu_J[cpu_rows].T)
  np.testing.assert_allclose(
      result["W"][0][np.ix_(gpu_rows, gpu_rows)].cpu().numpy(),
      cpu_W,
      rtol=2e-5,
      atol=3e-6,
  )

  point_jac = np.zeros((3, model.nv))
  rotation_jac = np.zeros((3, model.nv))
  moving_body = int(model.geom_bodyid[reference.contact[0].geom[1]])
  mujoco.mj_jac(
      model, reference, point_jac, rotation_jac,
      reference.contact[0].pos, moving_body,
  )
  frame_axes = np.asarray(reference.contact[0].frame).reshape(3, 3)
  expected_jacobian = np.vstack([
      frame_axes @ point_jac,
      frame_axes @ rotation_jac,
  ])
  np.testing.assert_allclose(
      result["contact_jacobian"][0, 0].cpu().numpy(),
      expected_jacobian,
      rtol=2e-5,
      atol=3e-6,
  )
  np.testing.assert_allclose(
      result["qacc"][0].cpu().numpy(), reference.qacc,
      rtol=5e-4, atol=2e-2,
  )
  np.testing.assert_allclose(
      result["qfrc_constraint"][0].cpu().numpy(), reference.qfrc_constraint,
      rtol=5e-4, atol=8e-2,
  )
  np.testing.assert_array_equal(result["contact_mask"][0].cpu().numpy(), [1.0])

  native_wrench = result["contact_wrench"][0, 0].cpu().numpy()
  cpu_wrench = np.zeros(6)
  mujoco.mj_contactForce(model, reference, 0, cpu_wrench)
  np.testing.assert_allclose(native_wrench, cpu_wrench, rtol=5e-4, atol=8e-2)
  if condim == 4:
    assert abs(native_wrench[3]) > 1.0  # torsional friction is engaged

  lam = result["lambda"][0].cpu().numpy().astype(np.float64)
  Wreg = result["W_regularized"][0].cpu().numpy().astype(np.float64)
  rhs = result["rhs"][0].cpu().numpy().astype(np.float64)
  row_start = descriptor.nr_joint + int(descriptor.contact_condim_packed[1])
  if cone == "elliptic" and condim > 1:
    rows = np.arange(row_start, row_start + condim)
    friction = descriptor.contact_friction[0, : condim - 1].astype(np.float64)
    scale = np.concatenate([[1.0], friction])
    y = lam[rows] / scale
    block = Wreg[np.ix_(rows, rows)]
    hessian = scale[:, None] * block * scale[None, :]
    gradient = Wreg[np.ix_(rows, np.arange(len(lam)))] @ lam - rhs[rows]
    linear = scale * (gradient - block @ lam[rows])
    assert y[0] >= 0.0
    assert np.linalg.norm(y[1:]) <= y[0] + 1e-4
    lipschitz = np.linalg.norm(hessian, ord=np.inf)
    projected = _elliptic_project(y - (hessian @ y + linear) / lipschitz)
    residual = np.linalg.norm(y - projected) * lipschitz / max(
        1.0, np.linalg.norm(linear) + np.linalg.norm(hessian @ y)
    )
    assert residual <= 2e-5, (cone, condim, residual)
  elif condim > 1:
    edge_count = 2 * (condim - 1)
    rows = np.arange(row_start, row_start + edge_count)
    assert np.all(lam[rows] >= -2e-5)
    gradient = Wreg[np.ix_(rows, np.arange(len(lam)))] @ lam - rhs[rows]
    diagonal = np.maximum(np.diag(Wreg)[rows], 1e-12)
    projected = np.maximum(0.0, lam[rows] - gradient / diagonal)
    scale = np.maximum(
        1.0,
        np.abs(rhs[rows])
        + np.abs(Wreg[np.ix_(rows, np.arange(len(lam)))] @ lam),
    )
    residual = np.max(np.abs(projected - lam[rows]) * diagonal / scale)
    assert residual <= 2e-5, (cone, condim, residual)


@pytest.mark.gpu
@pytest.mark.skipif(os.getenv("MUJOCO_METAL_RUN_GPU") != "1", reason="opt-in GPU")
@pytest.mark.parametrize(("shape_a", "shape_b", "offset", "cone", "condim"), [
    ("sphere", "sphere", .35, "pyramidal", 4),
    ("sphere", "capsule", .18, "elliptic", 4),
    ("sphere", "box", .22, "pyramidal", 6),
    ("capsule", "capsule", .15, "elliptic", 6),
    ("capsule", "box", .12, "pyramidal", 4),
    ("box", "box", .25, "elliptic", 6),
])
def test_high_dimensional_nonplane_pairs_compare_both_moving_bodies(
    shape_a, shape_b, offset, cone, condim
):
  """Qualify friction blocks on every non-plane pair with two dynamic bodies."""
  import torch

  geoms = {
      "sphere": '<geom type="sphere" size=".2" mass=".3" contype="0" conaffinity="0"/>',
      "capsule": '<geom type="capsule" size=".1 .25" mass=".3" contype="0" conaffinity="0"/>',
      "box": '<geom type="box" size=".15 .15 .15" mass=".3" contype="0" conaffinity="0"/>',
  }
  geom_a = geoms[shape_a].replace("<geom type=", '<geom name="geom-a" type=')
  geom_b = geoms[shape_b].replace("<geom type=", '<geom name="geom-b" type=')
  xml = f'''<mujoco><option timestep=".002" gravity="0 0 0" cone="{cone}"
      iterations="1000" tolerance="1e-6"/>
    <worldbody>
      <body name="body-a" pos="0 0 0" euler=".1 .2 .3">
        <freejoint name="free-a"/>{geom_a}
      </body>
      <body name="body-b" pos="{offset} 0 0" euler="-.1 .1 -.2">
        <freejoint name="free-b"/>{geom_b}
      </body>
    </worldbody>
    <contact><pair geom1="geom-a" geom2="geom-b" condim="{condim}"
        friction=".8 .6 .07 .04 .03"/></contact>
  </mujoco>'''
  model = mujoco.MjModel.from_xml_string(xml)
  qpos = model.qpos0[None, :].astype(np.float32)
  qvel = np.array([[.2, -.1, .15, 1.0, -.6, .8,
                    -.15, .24, -.05, -.5, .7, -1.2]], dtype=np.float32)
  descriptor = lower_coupled_constraints(model)
  qpos_mps = torch.as_tensor(qpos, device="mps")
  qvel_mps = torch.as_tensor(qvel, device="mps")
  model_descriptor = load_model(model)
  poses = MetalKinematics(model_descriptor, 1).run_device(qpos_mps)
  dynamics = MetalSmoothDynamics(model_descriptor, 1).run_device(qpos_mps, qvel_mps)
  result = MetalCoupledConstraints(model, 1).run_device(
      poses, dynamics["mass_matrix"], -dynamics["qfrc_bias"], qpos_mps, qvel_mps
  )

  reference = mujoco.MjData(model)
  reference.qpos[:], reference.qvel[:] = qpos[0], qvel[0]
  mujoco.mj_forward(model, reference)
  assert reference.ncon > 0 and int(result["status"][0]) == 0
  rows_per_contact = condim if cone == "elliptic" else 2 * (condim - 1)
  assert reference.nefc == reference.ncon * rows_per_contact
  active_slots = np.flatnonzero(result["contact_mask"][0].cpu().numpy() > .5)
  offset_slot = int(descriptor.pair_contact_offset[0])
  candidate_slots = range(offset_slot, offset_slot + int(descriptor.pair_max_contacts[0]))
  active_slots = np.asarray([slot for slot in candidate_slots if slot in set(active_slots)])
  assert len(active_slots) == reference.ncon

  # Pair contact order can vary between the CPU and Metal manifold builders;
  # pair contacts by world position before comparing either body's Jacobian.
  gpu_pos = result["contact_position"][0].cpu().numpy()
  gpu_wrench = result["contact_wrench"][0].cpu().numpy()
  gpu_contact_jac = result["contact_jacobian"][0].cpu().numpy()
  unused_slots = set(map(int, active_slots))
  cpu_to_gpu_slot = []
  for cpu_contact in range(reference.ncon):
    position = np.asarray(reference.contact[cpu_contact].pos)
    distances = [(np.linalg.norm(gpu_pos[slot] - position), slot) for slot in unused_slots]
    distance, slot = min(distances)
    assert distance < 2e-5, (shape_a, shape_b, distance)
    unused_slots.remove(slot)
    cpu_to_gpu_slot.append(slot)

    geom0, geom1 = map(int, reference.contact[cpu_contact].geom)
    body0, body1 = int(model.geom_bodyid[geom0]), int(model.geom_bodyid[geom1])
    point_jac0 = np.zeros((3, model.nv)); rot_jac0 = np.zeros((3, model.nv))
    point_jac1 = np.zeros((3, model.nv)); rot_jac1 = np.zeros((3, model.nv))
    mujoco.mj_jac(model, reference, point_jac0, rot_jac0,
                  reference.contact[cpu_contact].pos, body0)
    mujoco.mj_jac(model, reference, point_jac1, rot_jac1,
                  reference.contact[cpu_contact].pos, body1)
    axes = np.asarray(reference.contact[cpu_contact].frame).reshape(3, 3)
    expected_jac = np.vstack([
        axes @ (point_jac1 - point_jac0),
        axes @ (rot_jac1 - rot_jac0),
    ])
    np.testing.assert_allclose(gpu_contact_jac[slot], expected_jac, rtol=3e-5, atol=1e-5)
    assert np.linalg.norm(expected_jac[3:, 3:6]) > .1
    assert np.linalg.norm(expected_jac[3:, 9:12]) > .1
    assert np.linalg.norm(gpu_contact_jac[slot, :, :6]) > .1
    assert np.linalg.norm(gpu_contact_jac[slot, :, 6:]) > .1

    cpu_wrench = np.zeros(6)
    mujoco.mj_contactForce(model, reference, cpu_contact, cpu_wrench)
    np.testing.assert_allclose(gpu_wrench[slot], cpu_wrench, rtol=8e-4, atol=9e-2)
    assert np.linalg.norm(gpu_wrench[slot, 3:]) > .05
    if condim == 6:
      assert np.linalg.norm(gpu_wrench[slot, 4:]) > .02

  # Compare each mapped contact block and the full coupled Delassus matrix.
  cpu_J = reference.efc_J.reshape(reference.nefc, model.nv)
  cpu_R = reference.efc_R
  cpu_ar = reference.efc_aref
  cpu_rhs = -reference.efc_b
  gpu_rows = []
  for cpu_contact, slot in enumerate(cpu_to_gpu_slot):
    packed = np.asarray(descriptor.contact_condim_packed).reshape(-1, 3)
    row_start = descriptor.nr_joint + int(packed[slot, 1])
    cpu_rows = np.arange(cpu_contact * rows_per_contact,
                         (cpu_contact + 1) * rows_per_contact)
    slot_rows = np.arange(row_start, row_start + rows_per_contact)
    gpu_rows.extend(slot_rows.tolist())
    np.testing.assert_allclose(result["J"][0, slot_rows].cpu().numpy(), cpu_J[cpu_rows],
                               rtol=3e-5, atol=1e-5)
    np.testing.assert_allclose(result["R"][0, slot_rows].cpu().numpy(), cpu_R[cpu_rows],
                               rtol=3e-5, atol=3e-6)
    np.testing.assert_allclose(result["ar"][0, slot_rows].cpu().numpy(), cpu_ar[cpu_rows],
                               rtol=3e-4, atol=3e-3)
    np.testing.assert_allclose(result["rhs"][0, slot_rows].cpu().numpy(), cpu_rhs[cpu_rows],
                               rtol=3e-4, atol=3e-3)
  gpu_rows = np.asarray(gpu_rows, dtype=np.int64)
  cpu_mass = np.zeros((model.nv, model.nv), dtype=np.float64)
  mujoco.mj_fullM(model, reference, cpu_mass)
  np.testing.assert_allclose(dynamics["mass_matrix"][0].cpu().numpy(), cpu_mass,
                             rtol=2e-5, atol=4e-6)
  cpu_W = cpu_J @ np.linalg.solve(cpu_mass, cpu_J.T) + np.diag(cpu_R)
  native_W = result["W_regularized"][0].cpu().numpy()[np.ix_(gpu_rows, gpu_rows)]
  # All six pair fixtures use this same tolerance. Box-box is most sensitive:
  # its observed max W difference was 2.60e-3, while other observed errors were
  # smaller; those smaller results do not have tighter assertion bounds.
  np.testing.assert_allclose(native_W, cpu_W, rtol=5e-3, atol=3e-3)
  np.testing.assert_allclose(result["qacc"][0].cpu().numpy(), reference.qacc,
                             rtol=8e-4, atol=2e-2)
  np.testing.assert_allclose(result["qfrc_constraint"][0].cpu().numpy(),
                             reference.qfrc_constraint, rtol=8e-4, atol=8e-2)


@pytest.mark.gpu
@pytest.mark.skipif(os.getenv("MUJOCO_METAL_RUN_GPU") != "1", reason="opt-in GPU")
@pytest.mark.parametrize("condim", [3, 4, 6])
@pytest.mark.parametrize("yaw", [0.0, 0.37])
def test_native_elliptic_multicontact_residual_uses_returned_global_lambda(condim, yaw):
  """A four-point box manifold must converge at the returned coupled solution."""
  import torch

  xml = f'''<mujoco><option timestep=".002" gravity="0 0 -9.81" cone="elliptic"
      iterations="1000" tolerance="1e-8"/>
    <worldbody><geom type="plane" size="2 2 .1" condim="{condim}"
        friction=".8 .6 .07"/>
      <body pos="0 0 .095" euler="0 0 {yaw}"><freejoint/>
        <geom type="box" size=".2 .15 .1" mass="1" condim="{condim}"
            friction=".8 .6 .07"/>
      </body></worldbody></mujoco>'''
  model = mujoco.MjModel.from_xml_string(xml)
  qpos = model.qpos0[None, :].astype(np.float32)
  qvel = np.array([[.3, -.2, .1, 1.3, -2.1, 3.4]], dtype=np.float32)
  desc = lower_coupled_constraints(model)
  device = torch.device("mps")
  qpos_mps = torch.as_tensor(qpos, device=device)
  qvel_mps = torch.as_tensor(qvel, device=device)
  model_desc = load_model(model)
  poses = MetalKinematics(model_desc, 1).run_device(qpos_mps)
  dynamics = MetalSmoothDynamics(model_desc, 1).run_device(qpos_mps, qvel_mps)
  result = MetalCoupledConstraints(model, 1).run_device(
      poses, dynamics["mass_matrix"], -dynamics["qfrc_bias"], qpos_mps, qvel_mps
  )

  reference = mujoco.MjData(model)
  reference.qpos[:], reference.qvel[:] = qpos[0], qvel[0]
  mujoco.mj_forward(model, reference)
  assert reference.ncon == 4
  assert int(result["status"][0]) == 0, result["solver_diagnostics"].cpu().numpy()

  lam = result["lambda"][0].cpu().numpy().astype(np.float64)
  Wreg = result["W_regularized"][0].cpu().numpy().astype(np.float64)
  rhs = result["rhs"][0].cpu().numpy().astype(np.float64)
  active_slots = np.flatnonzero(result["contact_mask"][0].cpu().numpy() > 0.5)
  assert len(active_slots) == 4
  block_residuals = []
  for slot in active_slots:
    dim, offset, cone = desc.contact_condim_packed[3 * slot : 3 * slot + 3]
    assert dim == condim and cone == int(mujoco.mjtCone.mjCONE_ELLIPTIC)
    rows = np.arange(desc.nr_joint + offset, desc.nr_joint + offset + condim)
    scale = np.concatenate([[1.0], desc.contact_friction[slot, : condim - 1].astype(np.float64)])
    y = lam[rows] / scale
    assert y[0] >= 0.0
    assert np.linalg.norm(y[1:]) <= y[0] + 1e-4
    block = Wreg[np.ix_(rows, rows)]
    hessian = scale[:, None] * block * scale[None, :]
    gradient = Wreg[np.ix_(rows, np.arange(len(lam)))] @ lam - rhs[rows]
    linear = scale * (gradient - block @ lam[rows])
    lipschitz = np.linalg.norm(hessian, ord=np.inf)
    projected = _elliptic_project(y - (hessian @ y + linear) / lipschitz)
    residual = np.linalg.norm(y - projected) * lipschitz / max(
        1.0, np.linalg.norm(linear) + np.linalg.norm(hessian @ y)
    )
    block_residuals.append(float(residual))
  assert max(block_residuals) <= 2e-5, block_residuals
  np.testing.assert_allclose(result["qacc"][0].cpu().numpy(), reference.qacc,
                             rtol=5e-4, atol=2e-2)
  np.testing.assert_allclose(result["qfrc_constraint"][0].cpu().numpy(),
                             reference.qfrc_constraint, rtol=5e-4, atol=8e-2)


@pytest.mark.gpu
@pytest.mark.skipif(os.getenv("MUJOCO_METAL_RUN_GPU") != "1", reason="opt-in GPU")
def test_native_elliptic_multicontact_iteration_exhaustion_is_reported():
  """A one-sweep coupled solve must report the residual at returned lambda."""
  import torch

  xml = '''<mujoco><option timestep=".002" gravity="0 0 -9.81" cone="elliptic"
      iterations="1" tolerance="1e-8"/>
    <worldbody><geom type="plane" size="2 2 .1" condim="3" friction=".8 .6 .07"/>
      <body pos="0 0 .095"><freejoint/><geom type="box" size=".2 .15 .1"
          mass="1" condim="3" friction=".8 .6 .07"/></body>
    </worldbody></mujoco>'''
  model = mujoco.MjModel.from_xml_string(xml)
  qpos = model.qpos0[None, :].astype(np.float32)
  qvel = np.array([[.3, -.2, .1, 1.3, -2.1, 3.4]], dtype=np.float32)
  qpos_mps = torch.as_tensor(qpos, device="mps")
  qvel_mps = torch.as_tensor(qvel, device="mps")
  model_desc = load_model(model)
  poses = MetalKinematics(model_desc, 1).run_device(qpos_mps)
  dynamics = MetalSmoothDynamics(model_desc, 1).run_device(qpos_mps, qvel_mps)
  result = MetalCoupledConstraints(model, 1).run_device(
      poses, dynamics["mass_matrix"], -dynamics["qfrc_bias"], qpos_mps, qvel_mps
  )
  diagnostic = result["solver_diagnostics"][0].cpu().numpy()
  assert int(result["status"][0]) == 3, diagnostic
  assert diagnostic[1] == 1
  assert diagnostic[0] > 1e-6, diagnostic
@pytest.mark.gpu
@pytest.mark.skipif(os.getenv("MUJOCO_METAL_RUN_GPU") != "1", reason="opt-in GPU")
@pytest.mark.parametrize("cone", ["pyramidal", "elliptic"])
@pytest.mark.parametrize("condim", [4, 6])
@pytest.mark.parametrize("shape", ["capsule", "box"])
def test_native_friction_contact_capsule_and_box(cone, condim, shape):
  """Qualify rotational contact blocks on the remaining admitted geom families."""
  import torch

  if shape == "capsule":
    geom = '<geom type="capsule" size=".08 .12" condim="{}" friction=".8 .6 .07"/>'.format(condim)
    position = 'pos=".02 -.01 .19" euler=".1 .2 .3"'
  else:
    geom = '<geom type="box" size=".1 .12 .08" condim="{}" friction=".8 .6 .07"/>'.format(condim)
    position = 'pos=".02 -.01 .07" euler=".1 .2 .3"'
  xml = f'''<mujoco><option timestep=".002" gravity="0 0 0" cone="{cone}"
      iterations="1000" tolerance="1e-10"/>
    <worldbody><geom type="plane" size="2 2 .1" condim="{condim}"
        friction=".8 .6 .07"/>
      <body {position}><freejoint/>{geom}</body>
    </worldbody></mujoco>'''
  model = mujoco.MjModel.from_xml_string(xml)
  qpos = model.qpos0[None, :].astype(np.float32)
  qvel = np.array([[.12, -.08, .03, .4, -.7, 1.1]], dtype=np.float32)
  device = torch.device("mps")
  qpos_mps = torch.as_tensor(qpos, device=device)
  qvel_mps = torch.as_tensor(qvel, device=device)
  model_descriptor = load_model(model)
  poses = MetalKinematics(model_descriptor, 1).run_device(qpos_mps)
  dynamics = MetalSmoothDynamics(model_descriptor, 1).run_device(qpos_mps, qvel_mps)
  result = MetalCoupledConstraints(model, 1).run_device(
      poses, dynamics["mass_matrix"], -dynamics["qfrc_bias"], qpos_mps, qvel_mps
  )
  reference = mujoco.MjData(model)
  reference.qpos[:], reference.qvel[:] = qpos[0], qvel[0]
  mujoco.mj_forward(model, reference)
  assert reference.ncon > 0, shape
  assert int(result["status"][0]) == 0, result["solver_diagnostics"].cpu().numpy()
  assert int((result["contact_mask"][0] > 0).sum()) == reference.ncon
  np.testing.assert_allclose(
      result["qacc"][0].cpu().numpy(), reference.qacc, rtol=1e-3, atol=5e-2
  )
  np.testing.assert_allclose(
      result["qfrc_constraint"][0].cpu().numpy(),
      reference.qfrc_constraint,
      rtol=1e-3,
      atol=1e-1,
  )


@pytest.mark.gpu
@pytest.mark.skipif(os.getenv("MUJOCO_METAL_RUN_GPU") != "1", reason="opt-in GPU")
@pytest.mark.parametrize("cone", ["pyramidal", "elliptic"])
def test_native_condim6_contact_engages_rolling_friction(cone):
  """Use an independent contact-force oracle and require a real roll moment."""
  import torch

  xml = f'''<mujoco><option timestep=".002" gravity="0 0 0" cone="{cone}"
      iterations="1000" tolerance="1e-10"/>
    <worldbody><geom type="plane" size="2 2 .1" condim="6"
        friction=".8 .6 .07"/>
      <body pos="0 0 .15"><freejoint/>
        <geom type="sphere" size=".2" condim="6" friction=".8 .6 .07"/>
      </body></worldbody></mujoco>'''
  model = mujoco.MjModel.from_xml_string(xml)
  qpos = model.qpos0[None, :].astype(np.float32)
  qvel = np.array([[.3, -.2, .1, 1.3, -2.1, 3.4]], dtype=np.float32)
  device = torch.device("mps")
  qpos_mps = torch.as_tensor(qpos, device=device)
  qvel_mps = torch.as_tensor(qvel, device=device)
  model_descriptor = load_model(model)
  poses = MetalKinematics(model_descriptor, 1).run_device(qpos_mps)
  dynamics = MetalSmoothDynamics(model_descriptor, 1).run_device(qpos_mps, qvel_mps)
  result = MetalCoupledConstraints(model, 1).run_device(
      poses, dynamics["mass_matrix"], -dynamics["qfrc_bias"], qpos_mps, qvel_mps
  )
  reference = mujoco.MjData(model)
  reference.qpos[:], reference.qvel[:] = qpos[0], qvel[0]
  mujoco.mj_forward(model, reference)
  cpu_wrench = np.zeros(6)
  mujoco.mj_contactForce(model, reference, 0, cpu_wrench)
  native_wrench = result["contact_wrench"][0, 0].cpu().numpy()
  assert np.linalg.norm(cpu_wrench[4:6]) > 1.0, cpu_wrench
  assert np.linalg.norm(native_wrench[4:6]) > 1.0, native_wrench
  np.testing.assert_allclose(native_wrench, cpu_wrench, rtol=5e-4, atol=8e-2)


@pytest.mark.gpu
@pytest.mark.skipif(os.getenv("MUJOCO_METAL_RUN_GPU") != "1", reason="opt-in GPU")
@pytest.mark.parametrize("solreffriction", [".03 1", "-.03 -1"])
def test_native_explicit_elliptic_pair_preserves_anisotropic_friction(solreffriction):
  """Exercise pair-specific five-axis friction and standard/direct friction refs."""
  import torch

  xml = f'''<mujoco><option timestep=".002" gravity="0 0 0" cone="elliptic"
      iterations="1000" tolerance="1e-10"/>
    <worldbody><geom name="floor" type="plane" size="2 2 .1"
        euler="0 .28 .11"/>
      <body name="ball" pos=".02 -.01 .18" euler=".2 -.1 .3"><freejoint/>
        <geom name="sphere" type="sphere" pos=".045 .02 0" size=".2"/>
      </body></worldbody>
    <contact><pair geom1="floor" geom2="sphere" condim="6"
        friction=".8 .6 .07 .04 .03" solreffriction="{solreffriction}"/>
    </contact></mujoco>'''
  model = mujoco.MjModel.from_xml_string(xml)
  descriptor = lower_coupled_constraints(model)
  np.testing.assert_allclose(descriptor.contact_friction[0], [.8, .6, .07, .04, .03])
  qpos = model.qpos0[None, :].astype(np.float32)
  qvel = np.array([[.3, -.2, .1, 1.3, -2.1, 3.4]], dtype=np.float32)
  device = torch.device("mps")
  qpos_mps = torch.as_tensor(qpos, device=device)
  qvel_mps = torch.as_tensor(qvel, device=device)
  model_descriptor = load_model(model)
  poses = MetalKinematics(model_descriptor, 1).run_device(qpos_mps)
  dynamics = MetalSmoothDynamics(model_descriptor, 1).run_device(qpos_mps, qvel_mps)
  result = MetalCoupledConstraints(model, 1).run_device(
      poses, dynamics["mass_matrix"], -dynamics["qfrc_bias"], qpos_mps, qvel_mps
  )
  reference = mujoco.MjData(model)
  reference.qpos[:], reference.qvel[:] = qpos[0], qvel[0]
  mujoco.mj_forward(model, reference)
  assert reference.ncon == 1 and int(result["status"][0]) == 0
  np.testing.assert_allclose(
      result["qacc"][0].cpu().numpy(), reference.qacc, rtol=5e-4, atol=2e-2
  )
  native_wrench = result["contact_wrench"][0, 0].cpu().numpy()
  cpu_wrench = np.zeros(6)
  mujoco.mj_contactForce(model, reference, 0, cpu_wrench)
  np.testing.assert_allclose(native_wrench, cpu_wrench, rtol=5e-4, atol=8e-2)


@pytest.mark.gpu
@pytest.mark.skipif(os.getenv("MUJOCO_METAL_RUN_GPU") != "1", reason="opt-in GPU")
def test_native_elliptic_contact_handles_zero_and_near_zero_friction():
  import torch

  xml = '''<mujoco><option timestep=".002" gravity="0 0 0" cone="elliptic"
      iterations="1000" tolerance="1e-10"/>
    <worldbody><geom type="plane" size="2 2 .1" condim="6"
        friction="0 .0000001 0"/>
      <body pos="0 0 .15"><freejoint/>
        <geom type="sphere" size=".2" condim="6"
            friction=".0000001 0 .0000002"/>
      </body></worldbody></mujoco>'''
  model = mujoco.MjModel.from_xml_string(xml)
  qpos = model.qpos0[None, :].astype(np.float32)
  qvel = np.array([[.3, -.2, .1, 1.3, -2.1, 3.4]], dtype=np.float32)
  device = torch.device("mps")
  qpos_mps = torch.as_tensor(qpos, device=device)
  qvel_mps = torch.as_tensor(qvel, device=device)
  model_descriptor = load_model(model)
  poses = MetalKinematics(model_descriptor, 1).run_device(qpos_mps)
  dynamics = MetalSmoothDynamics(model_descriptor, 1).run_device(qpos_mps, qvel_mps)
  result = MetalCoupledConstraints(model, 1).run_device(
      poses, dynamics["mass_matrix"], -dynamics["qfrc_bias"], qpos_mps, qvel_mps
  )
  reference = mujoco.MjData(model)
  reference.qpos[:], reference.qvel[:] = qpos[0], qvel[0]
  mujoco.mj_forward(model, reference)
  assert reference.ncon == 1 and int(result["status"][0]) == 0
  assert torch.isfinite(result["qacc"]).all()
  assert torch.isfinite(result["contact_wrench"]).all()
  native = result["contact_wrench"][0, 0].cpu().numpy()
  cpu = np.zeros(6)
  mujoco.mj_contactForce(model, reference, 0, cpu)
  assert native[0] > 0 and cpu[0] > 0
  assert np.linalg.norm(native[1:]) < 1e-3 * native[0]
  np.testing.assert_allclose(native, cpu, rtol=2e-3, atol=5e-2)


@pytest.mark.gpu
@pytest.mark.skipif(os.getenv("MUJOCO_METAL_RUN_GPU") != "1", reason="opt-in GPU")
@pytest.mark.parametrize(("cone", "contact_count"), [("elliptic", 10), ("pyramidal", 7)])
def test_native_contact_final_valid_row_capacity(cone, contact_count):
  """Fill the last admitted slot in one world beside a contact-free world."""
  import torch

  model = _row_capacity_model(cone, contact_count)
  descriptor = lower_coupled_constraints(model)
  assert descriptor.nv == contact_count
  assert descriptor.nr <= 96
  qpos = np.tile(model.qpos0, (2, 1)).astype(np.float32)
  for j in range(model.njnt):
    qpos[1, model.jnt_qposadr[j]] = 0.2  # Keep the second world's spheres above the floor.
  qvel = np.zeros((2, model.nv), dtype=np.float32)
  device = torch.device("mps")
  qpos_mps = torch.as_tensor(qpos, device=device)
  qvel_mps = torch.as_tensor(qvel, device=device)
  model_descriptor = load_model(model)
  poses = MetalKinematics(model_descriptor, 2).run_device(qpos_mps)
  dynamics = MetalSmoothDynamics(model_descriptor, 2).run_device(qpos_mps, qvel_mps)
  result = MetalCoupledConstraints(model, 2).run_device(
      poses, dynamics["mass_matrix"], -dynamics["qfrc_bias"], qpos_mps, qvel_mps
  )
  references = []
  for world in range(2):
    reference = mujoco.MjData(model)
    reference.qpos[:], reference.qvel[:] = qpos[world], qvel[world]
    mujoco.mj_forward(model, reference)
    references.append(reference)
  assert [ref.ncon for ref in references] == [contact_count, 0]
  assert torch.all(result["status"] == 0)
  np.testing.assert_array_equal(
      result["contact_mask"][0].cpu().numpy(), np.ones(contact_count)
  )
  np.testing.assert_array_equal(
      result["contact_mask"][1].cpu().numpy(), np.zeros(contact_count)
  )
  np.testing.assert_allclose(
      result["qacc"][0].cpu().numpy(), references[0].qacc, rtol=5e-4, atol=2e-2
  )
  np.testing.assert_allclose(
      result["qacc"][1].cpu().numpy(), references[1].qacc, rtol=5e-4, atol=2e-2
  )


def test_coupled_oracle_matches_mujoco():
  model = mujoco.MjModel.from_xml_string(COUPLED_XML)
  qpos = np.array([[0.0, 0.0, -0.25], [0.1, 0.15, -0.2]], dtype=np.float64)
  qvel = np.array([[0.1, -0.1, 0.05], [-0.05, 0.1, -0.1]], dtype=np.float64)
  res = coupled_constraint_oracle(model, qpos, qvel)
  assert np.all(res["status"] == 0)
  for b in range(2):
    d = mujoco.MjData(model)
    d.qpos[:] = qpos[b]
    d.qvel[:] = qvel[b]
    mujoco.mj_forward(model, d)
    np.testing.assert_allclose(res["qacc"][b], d.qacc, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(res["qfrc_constraint"][b], d.qfrc_constraint, rtol=1e-5, atol=1e-5)


def test_decoupled_vs_coupled_divergence():
  """Demonstrates that solving decoupled produces substantial error vs coupled solve."""
  model = mujoco.MjModel.from_xml_string(COUPLED_XML)
  d_cpu = mujoco.MjData(model)
  d_cpu.qpos[model.jnt_qposadr[model.joint("ball_z").id]] = -0.25
  mujoco.mj_forward(model, d_cpu)

  # Coupled reference acceleration
  coupled_acc = d_cpu.qacc.copy()

  # Decoupled sequence:
  # Step 1: Solve contacts independently (disable equality, limits, frictionloss)
  m_contact = mujoco.MjModel.from_xml_string(COUPLED_XML)
  m_contact.opt.disableflags |= (
      mujoco.mjtDisableBit.mjDSBL_EQUALITY
      | mujoco.mjtDisableBit.mjDSBL_LIMIT
      | mujoco.mjtDisableBit.mjDSBL_FRICTIONLOSS
  )
  d_contact = mujoco.MjData(m_contact)
  d_contact.qpos[:] = d_cpu.qpos
  d_contact.qvel[:] = d_cpu.qvel
  mujoco.mj_forward(m_contact, d_contact)
  f_contact = d_contact.qfrc_constraint.copy()

  # Step 2: Joint constraints solve with f_contact injected as external force
  m_joint = mujoco.MjModel.from_xml_string(COUPLED_XML)
  m_joint.opt.disableflags |= mujoco.mjtDisableBit.mjDSBL_CONTACT
  d_joint = mujoco.MjData(m_joint)
  d_joint.qpos[:] = d_cpu.qpos
  d_joint.qvel[:] = d_cpu.qvel
  d_joint.qfrc_applied[:] = f_contact
  mujoco.mj_forward(m_joint, d_joint)
  decoupled_acc = d_joint.qacc.copy()

  # Verify Delassus cross-coupling block norm is substantial
  M = np.zeros((model.nv, model.nv), dtype=np.float64)
  mujoco.mj_fullM(model, d_cpu, M)
  Minv = np.linalg.inv(M)
  J = d_cpu.efc_J.reshape(d_cpu.nefc, model.nv)
  c_mask = (d_cpu.efc_type == mujoco.mjtConstraint.mjCNSTR_CONTACT_PYRAMIDAL) | (
      d_cpu.efc_type == mujoco.mjtConstraint.mjCNSTR_CONTACT_FRICTIONLESS
  )
  J_c = J[c_mask]
  J_j = J[~c_mask]
  W_cj = J_c @ Minv @ J_j.T
  cross_norm = float(np.linalg.norm(W_cj))
  assert cross_norm > 5.0, f"Expected strong Delassus cross-term coupling, got {cross_norm}"

  # Calculate relative acceleration difference
  rel_error = float(np.linalg.norm(decoupled_acc - coupled_acc) / np.linalg.norm(coupled_acc))
  assert rel_error > 0.10, f"Expected decoupled relative error > 10%, got {rel_error * 100:.2f}%"
  assert np.isclose(rel_error, 0.15688, atol=1e-3), f"Expected ~15.69% relative error, got {rel_error}"


@pytest.mark.gpu
@pytest.mark.skipif(
    os.getenv("MUJOCO_METAL_RUN_GPU") != "1", reason="opt-in GPU"
)
def test_metal_coupled_constraints_matches_cpu():
  import torch

  model = mujoco.MjModel.from_xml_string(COUPLED_XML)
  batch = 2
  qpos = np.array([[0.0, 0.0, -0.25], [0.05, 0.075, -0.23]], dtype=np.float32)
  qvel = np.array([[0.0, 0.0, 0.0], [0.1, -0.05, 0.02]], dtype=np.float32)

  coupled_stage = MetalCoupledConstraints(model, batch_size=batch)
  desc = load_model(model)
  smooth = MetalSmoothDynamics(desc, batch_size=batch)

  qpos_dev = torch.tensor(qpos, device="mps")
  qvel_dev = torch.tensor(qvel, device="mps")

  dyn = smooth.run_device(qpos_dev, qvel_dev)
  res = coupled_stage.run_device(
      dyn["poses"], dyn["mass_matrix"], -dyn["qfrc_bias"], qpos_dev, qvel_dev
  )

  assert torch.all(res["status"] == 0)

  # Compare with CPU oracle
  oracle = coupled_constraint_oracle(model, qpos, qvel)
  np.testing.assert_allclose(res["qacc"].cpu().numpy(), oracle["qacc"], rtol=2e-3, atol=1e-2)
  np.testing.assert_allclose(res["qfrc_constraint"].cpu().numpy(), oracle["qfrc_constraint"], rtol=2e-3, atol=1e-2)


@pytest.mark.gpu
@pytest.mark.skipif(
    os.getenv("MUJOCO_METAL_RUN_GPU") != "1", reason="opt-in GPU"
)
def test_metal_coupled_constraints_zero_contact():
  import torch

  xml = COUPLED_XML.replace('type="sphere" size="0.1"', 'type="sphere" size="0.01"').replace('pos="0.5 0 1.5"', 'pos="10 10 10"')
  model = mujoco.MjModel.from_xml_string(xml)
  coupled = MetalCoupledConstraints(model, batch_size=1)
  desc = load_model(model)
  smooth = MetalSmoothDynamics(desc, batch_size=1)
  qpos = torch.zeros((1, model.nq), device="mps")
  qvel = torch.zeros((1, model.nv), device="mps")
  dyn = smooth.run_device(qpos, qvel)
  res = coupled.run_device(dyn["poses"], dyn["mass_matrix"], -dyn["qfrc_bias"], qpos, qvel)
  assert res["status"].item() == 0


@pytest.mark.gpu
@pytest.mark.skipif(
    os.getenv("MUJOCO_METAL_RUN_GPU") != "1", reason="opt-in GPU"
)
def test_hot_loop_device_resident_invariants():
  """Verifies that running coupled constraints in a hot loop preserves device resident state."""
  import torch

  model = mujoco.MjModel.from_xml_string(COUPLED_XML)
  batch = 2
  qpos = torch.zeros((batch, model.nq), device="mps", dtype=torch.float32)
  qvel = torch.zeros((batch, model.nv), device="mps", dtype=torch.float32)

  stage = MetalCoupledConstraints(model, batch_size=batch)
  desc = load_model(model)
  smooth = MetalSmoothDynamics(desc, batch_size=batch)

  # Check preallocated default equality state on device
  assert hasattr(stage, "_eq_active_default")
  assert stage._eq_active_default.device.type == "mps"
  assert stage._eq_active_default.dtype == torch.int32
  assert stage._eq_active_default.shape == (batch, max(1, stage.descriptor.neq))

  dyn = smooth.run_device(qpos, qvel)

  for _ in range(5):
    res = stage.run_device(
        dyn["poses"], dyn["mass_matrix"], -dyn["qfrc_bias"], qpos, qvel
    )
    assert res["status"].device.type == "mps"
    assert res["qacc"].device.type == "mps"


def test_coupled_solver_settings_range_contracts():
  model = mujoco.MjModel.from_xml_string(COUPLED_XML)
  desc = lower_coupled_constraints(model)
  settings = desc.solver_settings
  assert isinstance(settings, CoupledSolverSettings)
  assert settings.requested_iterations == 1000
  assert settings.effective_iterations == 1000
  assert settings.requested_tolerance == 1e-6
  assert settings.effective_tolerance == 1e-6
  assert settings.max_refinement_sweeps == 256
  assert settings.metric == "max_normalized_projected_gradient"

  # Iterations contract: [1, 2048]
  # Value 100 is NOT silently overridden to 1024
  m100 = mujoco.MjModel.from_xml_string(COUPLED_XML)
  m100.opt.iterations = 100
  d100 = lower_coupled_constraints(m100)
  assert d100.solver_settings.requested_iterations == 100
  assert d100.solver_settings.effective_iterations == 100

  # Bound 2048 is accepted
  m2048 = mujoco.MjModel.from_xml_string(COUPLED_XML)
  m2048.opt.iterations = 2048
  d2048 = lower_coupled_constraints(m2048)
  assert d2048.solver_settings.effective_iterations == 2048

  # <= 0 or > 2048 rejected
  for bad_iter in [0, -1, 2049]:
    mbad = mujoco.MjModel.from_xml_string(COUPLED_XML)
    mbad.opt.iterations = bad_iter
    with pytest.raises(ValueError, match="bounds iterations to \\[1, 2048\\]"):
      lower_coupled_constraints(mbad)

  # Tolerance contract: finite and positive, floored at float32 floor 1e-6
  m_tol4 = mujoco.MjModel.from_xml_string(COUPLED_XML)
  m_tol4.opt.tolerance = 1e-4
  d_tol4 = lower_coupled_constraints(m_tol4)
  assert d_tol4.solver_settings.requested_tolerance == 1e-4
  assert d_tol4.solver_settings.effective_tolerance == 1e-4

  m_tol8 = mujoco.MjModel.from_xml_string(COUPLED_XML)
  m_tol8.opt.tolerance = 1e-8
  d_tol8 = lower_coupled_constraints(m_tol8)
  assert d_tol8.solver_settings.requested_tolerance == 1e-8
  assert d_tol8.solver_settings.effective_tolerance == 1e-6

  for bad_tol in [0.0, -1e-6, float("nan"), float("inf")]:
    mtol_bad = mujoco.MjModel.from_xml_string(COUPLED_XML)
    mtol_bad.opt.tolerance = bad_tol
    with pytest.raises(ValueError, match="model.opt.tolerance must be finite and positive"):
      lower_coupled_constraints(mtol_bad)
