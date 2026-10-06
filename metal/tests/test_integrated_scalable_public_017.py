# Copyright 2026 The MuJoCo Metal contributors
# Licensed under the Apache License, Version 2.0.

"""Opt-in public-path gates for the component-sparse integrated profile."""

import os

import mujoco
import numpy as np
import pytest

from mujoco_metal.simulation import (MetalSimulation,
                                     _model_component_capacity_limits)
from mujoco_metal.capacity import CapacityLimits, validate_runtime_buffers
from mujoco_metal.coupled_constraints import lower_coupled_constraints
from mujoco_metal.stepping import validate_stepping_profile


_GPU_ONLY = pytest.mark.skipif(
    os.getenv("MUJOCO_METAL_RUN_GPU") != "1",
    reason="requires explicit GPU qualification slot")


_XML = """<mujoco>
  <option timestep=".001" gravity="0 0 0" solver="PGS" iterations="20">
    <flag contact="disable"/>
  </option>
  <worldbody>
    <body name="left" pos="-1 0 0"><joint name="jl" type="slide" axis="1 0 0"/>
      <geom type="sphere" size=".08" mass="1"/>
    </body>
    <body name="right" pos="1 0 0"><joint name="jr" type="slide" axis="1 0 0"/>
      <geom type="sphere" size=".08" mass="1"/>
    </body>
  </worldbody>
  <tendon><fixed name="coupling_armature" armature=".2" limited="false">
    <joint joint="jl" coef="1"/><joint joint="jr" coef="-.5"/>
  </fixed></tendon>
  <equality><joint name="equal_position" joint1="jl" joint2="jr"
    polycoef="0 1 0 0 0" solref=".02 1"/></equality>
  <sensor><e_kinetic/></sensor>
</mujoco>"""


_SPATIAL_XML = """<mujoco>
  <option timestep=".001" gravity="0 0 0" solver="PGS" iterations="20">
    <flag contact="disable"/>
  </option>
  <worldbody>
    <site name="anchor" pos="0 0 0"/>
    <body name="moving" pos="1 0 0">
      <joint name="slide" type="slide" axis="1 0 0"/>
      <geom type="sphere" size=".1" mass="1"/>
      <site name="tip" pos=".3 0 0"/>
    </body>
  </worldbody>
  <tendon><spatial name="spatial_armature" armature=".4">
    <site site="anchor"/><site site="tip"/>
  </spatial></tendon>
  <sensor><e_kinetic/></sensor>
</mujoco>"""


@pytest.mark.parametrize("solver", ["PGS", "CG", "Newton"])
def test_scalable_large_solver_host_lowering_has_no_legacy_dimension_cap(solver):
  """Host ABI admits dimensions above the former thread-array limits."""
  joint_count = 130
  bodies = "".join(
      f'<body name="b{i}" pos="{i * .01} 0 0">'
      f'<joint name="j{i}" type="slide" axis="0 1 0" '
      'limited="true" range="0 1" frictionloss=".025" '
      'solreflimit=".02 1"/>'
      '<inertial pos="0 0 0" mass=".2" '
      'diaginertia=".001 .001 .001"/>'
      '</body>'
      for i in range(joint_count))
  model = mujoco.MjModel.from_xml_string(
      f'<mujoco><option timestep=".0005" gravity="0 0 0" solver="{solver}" '
      'iterations="3"><flag contact="disable"/></option>'
      f'<worldbody>{bodies}</worldbody></mujoco>')
  limits = CapacityLimits(max_nv=joint_count, max_pairs=64,
                          max_slots=64, max_rows=512)
  descriptor = lower_coupled_constraints(model, limits=limits)
  assert descriptor.nv == joint_count
  assert descriptor.nr > 256
  automatic = _model_component_capacity_limits(model, batch_size=2)
  automatic_descriptor = lower_coupled_constraints(model, limits=automatic)
  assert automatic.max_rows >= automatic_descriptor.nr
  assert automatic.max_pairs >= automatic_descriptor.npairs
  assert automatic.max_slots >= automatic_descriptor.ncontacts_max
  validate_runtime_buffers(
      model, 2, rhs_capacity=descriptor.nr + 1,
      mass_storage="block_sparse")
  # A CPU oracle is captured here even though device execution is opt-in.
  data = mujoco.MjData(model)
  data.qpos[:] = model.qpos0
  mujoco.mj_forward(model, data)
  assert np.all(np.isfinite(data.qacc))


def test_integrated_rk4_component_profile_host_lowers_model_derived_limits():
  """RK4 component mode keeps compiled rows rather than inheriting small caps."""
  model = mujoco.MjModel.from_xml_string(
      _XML.replace('<option timestep=',
                   '<option integrator="RK4" timestep='))
  limits = _model_component_capacity_limits(model, batch_size=2)
  rows = lower_coupled_constraints(model, limits=limits)
  assert limits.max_nv >= model.nv
  assert limits.max_pairs >= rows.npairs
  assert limits.max_slots >= rows.ncontacts_max
  assert limits.max_rows >= rows.nr
  profile = validate_stepping_profile(
      model, profile="integrated_rk4_v1", limits=limits)
  assert profile.name == "integrated_rk4_v1"
  assert profile.nv == model.nv
  validate_runtime_buffers(
      model, 2, rhs_capacity=max(rows.nr + 1, 1),
      mass_storage="block_sparse")
  model.opt.enableflags |= int(mujoco.mjtEnableBit.mjENBL_SLEEP)
  sleep_limits = _model_component_capacity_limits(model, batch_size=2)
  sleep_profile = validate_stepping_profile(
      model, profile="integrated_rk4_v1", limits=sleep_limits)
  assert sleep_profile.name == "integrated_rk4_v1"
  assert "sleep mode" not in sleep_profile.rejected


def _reconstruct_component_mass(blocks, layout):
  dense = np.zeros((int(layout["nv"]), int(layout["nv"])), dtype=np.float64)
  offsets = layout["component_dof_offsets"]
  dofs = layout["component_dof_ids"]
  mass_offsets = layout["component_mass_offsets"]
  for component in range(int(layout["ncomponent"])):
    begin, end = int(offsets[component]), int(offsets[component + 1])
    width = end - begin
    ids = np.asarray(dofs[begin:end], dtype=np.int64)
    start = int(mass_offsets[component])
    dense[np.ix_(ids, ids)] = np.asarray(blocks[start:start + width * width]).reshape(
        width, width)
  return dense


@pytest.mark.gpu
@_GPU_ONLY
def test_public_scalable_assembly_preserves_pinned_tendon_sparsity_and_steps():
  model = mujoco.MjModel.from_xml_string(_XML)
  assert model.nv == 2 and model.ntendon == 1 and model.neq == 1
  qpos = np.asarray([[.03, -.01], [-.02, .015]], dtype=np.float32)
  qvel = np.asarray([[.2, -.1], [-.05, .17]], dtype=np.float32)
  sim = MetalSimulation(model, batch_size=2, qpos=qpos, qvel=qvel,
                        profile="integrated_scalable_v1")

  system = sim.assembled_system(recompute=True)
  assert system["mass_matrix"] is None
  assert "mass_blocks" in system and "mass_block_layout" in system
  layout = system["mass_block_layout"]
  assert layout["nv"] == model.nv
  assert layout["ntree"] == model.ntree
  assert layout["ncomponent"] == sim._smooth.mass_block_layout["ncomponent"]
  assert "component_dof_ids" in layout and "component_mass_offsets" in layout
  assert system["tendon_armature_blocks"] is not None
  blocks = system["mass_blocks"].detach().cpu().numpy()
  armature = system["tendon_armature_blocks"].detach().cpu().numpy()
  assert np.any(np.abs(armature) > 1e-6)
  # MuJoCo 3.10's compressed M drops this possible cross-tree tendon entry;
  # the sparse layout must preserve that exact compiled sparsity.
  assert int(layout["ncomponent"]) == 2
  for world in range(2):
    data = mujoco.MjData(model)
    data.qpos[:] = qpos[world]
    data.qvel[:] = qvel[world]
    mujoco.mj_forward(model, data)
    full = np.zeros((model.nv, model.nv), dtype=np.float64)
    mujoco.mj_fullM(model, data, full)
    np.testing.assert_allclose(
        _reconstruct_component_mass(blocks[world] + armature[world], layout), full,
        rtol=7e-5, atol=7e-5)

  expected_energy = []
  for world in range(2):
    data = mujoco.MjData(model)
    data.qpos[:] = qpos[world]
    data.qvel[:] = qvel[world]
    mujoco.mj_forward(model, data)
    expected_energy.append(data.sensordata.copy())
  energy = sim.sensor_values().detach().cpu().numpy()
  np.testing.assert_allclose(energy, np.asarray(expected_energy),
                             rtol=7e-5, atol=7e-6)

  refs = []
  for world in range(2):
    data = mujoco.MjData(model)
    data.qpos[:] = qpos[world]
    data.qvel[:] = qvel[world]
    mujoco.mj_step(model, data)
    refs.append(data)
  sim.step()
  got_qpos = sim.state.qpos.detach().cpu().numpy()
  got_qvel = sim.state.qvel.detach().cpu().numpy()
  for world, data in enumerate(refs):
    np.testing.assert_allclose(got_qpos[world], data.qpos, rtol=2e-4, atol=2e-6)
    np.testing.assert_allclose(got_qvel[world], data.qvel, rtol=2e-4, atol=2e-5)
  assert sim._smooth._workspace["mass"] is None


@pytest.mark.gpu
@_GPU_ONLY
def test_public_integrated_rk4_uses_component_mass_for_armature_and_equality():
  """Four real RK stages solve through sparse M with mixed-world rows."""
  model = mujoco.MjModel.from_xml_string(
      _XML.replace('<option timestep=',
                   '<option integrator="RK4" timestep='))
  qpos = np.asarray([[.03, -.01], [-.02, .015]], dtype=np.float32)
  qvel = np.asarray([[.2, -.1], [-.05, .17]], dtype=np.float32)
  sim = MetalSimulation(model, batch_size=2, qpos=qpos, qvel=qvel,
                        profile="integrated_rk4_v1")
  assert sim._component_mass_enabled
  assert sim._smooth._workspace["mass"] is None
  assert sim._component_solver is not None
  refs = []
  for world in range(2):
    data = mujoco.MjData(model)
    data.qpos[:] = qpos[world]
    data.qvel[:] = qvel[world]
    mujoco.mj_step(model, data)
    refs.append(data)
  sim.step()
  got_qpos = sim.state.qpos.detach().cpu().numpy()
  got_qvel = sim.state.qvel.detach().cpu().numpy()
  for world, data in enumerate(refs):
    np.testing.assert_allclose(got_qpos[world], data.qpos,
                               rtol=3e-4, atol=3e-6)
    np.testing.assert_allclose(got_qvel[world], data.qvel,
                               rtol=3e-4, atol=3e-5)


@pytest.mark.gpu
@_GPU_ONLY
def test_public_integrated_rk4_component_sleep_mixed_worlds_match_pinned():
  """RK4's deferred X3/X0 sleep path retains one sleeping tree per world."""
  xml = _XML.replace('<option timestep=',
                     '<option integrator="RK4" timestep=')
  xml = xml.replace('contact="disable"',
                    'contact="disable" sleep="enable"')
  xml = xml.replace('<body name="left"',
                    '<body name="left" sleep="init"')
  xml = xml.replace('<body name="right"',
                    '<body name="right" sleep="init"')
  model = mujoco.MjModel.from_xml_string(xml)
  # Keep the two original moving/coupled worlds, and add an untouched
  # qpos0/qvel0 world whose INIT tree remains asleep. This separates actual
  # sleep-state retention from worlds that wake through their equality cycle.
  qpos = np.asarray([[.03, -.01], [-.02, .015], model.qpos0],
                    dtype=np.float32)
  qvel = np.asarray([[0., 0.], [.2, -.1], [0., 0.]], dtype=np.float32)
  sim = MetalSimulation(model, batch_size=3, qpos=qpos, qvel=qvel,
                        profile="integrated_rk4_v1")
  assert sim._sleep_schedule is not None
  refs = []
  for world in range(3):
    data = mujoco.MjData(model)
    data.qpos[:] = qpos[world]
    data.qvel[:] = qvel[world]
    refs.append(data)
  witnessed_sleep = False
  initial_sleep = refs[2].tree_asleep.copy()
  for _ in range(4):
    for data in refs:
      mujoco.mj_step(model, data)
    sim.step()
    np.testing.assert_allclose(sim.state.qpos.detach().cpu().numpy(),
                               np.stack([d.qpos for d in refs]),
                               rtol=4e-4, atol=5e-6)
    np.testing.assert_allclose(sim.state.qvel.detach().cpu().numpy(),
                               np.stack([d.qvel for d in refs]),
                               rtol=5e-4, atol=5e-5)
    np.testing.assert_array_equal(
        sim._sleep_schedule.tree_state.detach().cpu().numpy(),
        np.stack([d.tree_asleep for d in refs]))
    witnessed_sleep |= np.any(refs[2].tree_asleep >= 0)
    np.testing.assert_array_equal(refs[2].tree_asleep, initial_sleep)
  assert witnessed_sleep


def test_rk4_mixed_sleep_fixture_has_a_pinned_untouched_asleep_world():
  """The added qpos0/qvel0 world really remains INIT-asleep on CPU."""
  xml = _XML.replace('<option timestep=',
                     '<option integrator="RK4" timestep=')
  xml = xml.replace('contact="disable"',
                    'contact="disable" sleep="enable"')
  xml = xml.replace('<body name="left"',
                    '<body name="left" sleep="init"')
  xml = xml.replace('<body name="right"',
                    '<body name="right" sleep="init"')
  model = mujoco.MjModel.from_xml_string(xml)
  data = mujoco.MjData(model)
  initial = np.array(data.tree_asleep, copy=True)
  data.qpos[:] = model.qpos0
  data.qvel[:] = 0
  for _ in range(4):
    mujoco.mj_step(model, data)
  np.testing.assert_array_equal(data.qpos, model.qpos0)
  np.testing.assert_array_equal(data.qvel, 0)
  np.testing.assert_array_equal(data.tree_asleep, initial)
  assert np.any(data.tree_asleep >= 0)


@pytest.mark.gpu
@_GPU_ONLY
def test_public_scalable_spatial_tendon_armature_uses_component_blocks():
  model = mujoco.MjModel.from_xml_string(_SPATIAL_XML)
  assert model.nv == 1 and model.ntendon == 1
  qpos = np.asarray([[.02], [-.015]], dtype=np.float32)
  qvel = np.asarray([[.3], [-.2]], dtype=np.float32)
  sim = MetalSimulation(
      model, batch_size=2, qpos=qpos, qvel=qvel,
      profile="integrated_scalable_v1",
      limits=CapacityLimits(max_nv=64, max_pairs=64, max_slots=64,
                            max_rows=512))

  system = sim.assembled_system(recompute=True)
  assert system["mass_matrix"] is None
  assert system["tendon_armature_blocks"] is not None
  layout = system["mass_block_layout"]
  assert layout["nv"] == model.nv and layout["ntree"] == model.ntree
  assert "component_dof_ids" in layout and "component_mass_offsets" in layout
  assert int(layout["ncomponent"]) == 1
  blocks = system["mass_blocks"].detach().cpu().numpy()
  armature = system["tendon_armature_blocks"].detach().cpu().numpy()
  assert np.all(np.abs(armature) > 1e-6)
  for world in range(2):
    data = mujoco.MjData(model)
    data.qpos[:] = qpos[world]
    data.qvel[:] = qvel[world]
    mujoco.mj_forward(model, data)
    full = np.zeros((model.nv, model.nv), dtype=np.float64)
    mujoco.mj_fullM(model, data, full)
    np.testing.assert_allclose(
        _reconstruct_component_mass(blocks[world] + armature[world], layout), full,
        rtol=7e-5, atol=7e-5)

  expected_energy = []
  refs = []
  for world in range(2):
    data = mujoco.MjData(model)
    data.qpos[:] = qpos[world]
    data.qvel[:] = qvel[world]
    mujoco.mj_forward(model, data)
    expected_energy.append(data.sensordata.copy())
    data = mujoco.MjData(model)
    data.qpos[:] = qpos[world]
    data.qvel[:] = qvel[world]
    mujoco.mj_step(model, data)
    refs.append(data)
  np.testing.assert_allclose(
      sim.sensor_values().detach().cpu().numpy(), np.asarray(expected_energy),
      rtol=7e-5, atol=7e-6)
  sim.step()
  got_qpos = sim.state.qpos.detach().cpu().numpy()
  got_qvel = sim.state.qvel.detach().cpu().numpy()
  for world, data in enumerate(refs):
    np.testing.assert_allclose(got_qpos[world], data.qpos,
                               rtol=2e-4, atol=2e-6)
    np.testing.assert_allclose(got_qvel[world], data.qvel,
                               rtol=2e-4, atol=2e-5)


@pytest.mark.parametrize("solver", ["PGS", "CG", "Newton"])
@pytest.mark.gpu
@_GPU_ONLY
def test_public_scalable_solver_exceeds_legacy_nv_and_row_shapes(solver):
  """Opt-in end-to-end oracle for the dynamically sized block kernel."""
  joint_count = 130
  bodies = "".join(
      f'<body name="b{i}" pos="{i * .01} 0 0">'
      f'<joint name="j{i}" type="slide" axis="0 1 0" '
      'limited="true" range="0 1" frictionloss=".025" '
      'solreflimit=".02 1"/>'
      '<inertial pos="0 0 0" mass=".2" '
      'diaginertia=".001 .001 .001"/>'
      '</body>'
      for i in range(joint_count))
  model = mujoco.MjModel.from_xml_string(
      f'<mujoco><option timestep=".0005" gravity="0 0 0" solver="{solver}" '
      'iterations="3"><flag contact="disable"/></option>'
      f'<worldbody>{bodies}</worldbody></mujoco>')
  assert model.nv == joint_count
  qpos = np.stack((np.full(joint_count, .002),
                   np.full(joint_count, .004))).astype(np.float32)
  qvel = np.stack((np.linspace(-.02, .02, joint_count),
                   np.linspace(.015, -.015, joint_count))).astype(np.float32)
  refs = []
  for world in range(2):
    data = mujoco.MjData(model)
    data.qpos[:] = qpos[world]
    data.qvel[:] = qvel[world]
    mujoco.mj_forward(model, data)
    refs.append(data.qacc.copy())

  sim = MetalSimulation(model, batch_size=2, qpos=qpos, qvel=qvel,
                        profile="integrated_scalable_v1",
                        limits=CapacityLimits(max_nv=joint_count,
                                              max_pairs=64, max_slots=64,
                                              max_rows=512))
  descriptor = sim._coupled_constraints.descriptor
  assert descriptor.nv > 64 and descriptor.nr > 256
  from mujoco_metal.native_api import mj_forward
  stages = mj_forward(sim)
  got = stages["qacc"].detach().cpu().numpy()
  np.testing.assert_allclose(got, np.asarray(refs), rtol=5e-4, atol=3e-5)
