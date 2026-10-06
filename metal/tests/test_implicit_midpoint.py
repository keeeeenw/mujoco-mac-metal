"""CPU oracle regressions for MuJoCo 3.10 implicitfast free-body midpoint."""
import mujoco
import numpy as np
import pytest

from mujoco_metal.implicit import implicitfast_oracle
from mujoco_metal.implicit_midpoint import (
    FreeBodyMidpointProgram, free_midpoint_oracle, lower_free_body_midpoints,
)

XML = '''<mujoco><option integrator="implicitfast" timestep=".002" gravity="0 0 -9.81"><flag contact="disable"/></option>
<worldbody>
 <body name="aligned" pos="0 0 1"><freejoint/><inertial pos="0 0 0" quat=".9238795 0 0 .3826834" mass="1.2" diaginertia=".12 .18 .23"/></body>
 <body name="offset" pos="1 0 1"><freejoint/><inertial pos=".18 -.07 .11" quat=".9659258 0 .258819 0" mass=".8" diaginertia=".09 .14 .19"/></body>
 <body name="articulated" pos="-1 0 1"><freejoint/><geom type="box" size=".1 .2 .3" mass="1"/><body pos=".4 0 0"><joint type="hinge" axis="0 1 0"/><geom type="sphere" size=".1" mass=".2"/></body></body>
</worldbody></mujoco>'''


def test_runtime_midpoint_eligibility_uses_each_worlds_active_rows_and_awake_trees():
  torch = pytest.importorskip("torch")
  from mujoco_metal.implicit_midpoint import midpoint_eligibility
  model = mujoco.MjModel.from_xml_string(XML)
  desc = lower_free_body_midpoints(model)
  rows = torch.zeros((3, 4, model.nv))
  rows[0, 2, int(desc.dofadr[0])+3] = 1
  rows[1, 1, int(desc.dofadr[1])] = -1
  awake = torch.ones((3, model.ntree), dtype=torch.int32)
  awake[2, int(model.dof_treeid[int(desc.dofadr[0])])] = 0
  result = midpoint_eligibility(desc, 3, torch=torch, device="cpu",
      constraint_jacobian=rows, tree_awake=awake, dof_treeid=model.dof_treeid)
  assert result.tolist() == [[False, True], [True, False], [False, True]]
  # Removing current rows changes eligibility immediately; no candidate or
  # previous-step constraint cache can keep a free tree ineligible.
  rows.zero_()
  result = midpoint_eligibility(desc, 3, torch=torch, device="cpu",
      constraint_jacobian=rows, tree_awake=awake, dof_treeid=model.dof_treeid)
  assert result.tolist() == [[True, True], [True, True], [False, True]]


def _forward(model, batch=1, gravity=False):
  data = mujoco.MjData(model)
  data.qpos[:] = np.linspace(-.2, .3, model.nq)
  # Set valid unit quaternions for each free joint.
  for j in range(model.njnt):
    if model.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE:
      a = int(model.jnt_qposadr[j])
      data.qpos[a+3:a+7] = [0.94, .1, -.2, .25]
      data.qpos[a+3:a+7] /= np.linalg.norm(data.qpos[a+3:a+7])
  data.qvel[:] = np.linspace(-.8, .9, model.nv)
  data.qfrc_applied[:] = np.linspace(.7, -.6, model.nv)
  if gravity:
    model.opt.disableflags |= int(mujoco.mjtDisableBit.mjDSBL_GRAVITY)
  mujoco.mj_forward(model, data)
  return data


def _expected(model, data):
  mass = np.empty((model.nv, model.nv), dtype=np.float64)
  mujoco.mj_fullM(model, data, mass)
  effective = implicitfast_oracle(model, mass, data.qfrc_smooth)
  return free_midpoint_oracle(
      model, data.qvel, effective["qacc"], data.qacc,
      data.qfrc_smooth + data.qfrc_bias, data.xquat,
  )


@pytest.mark.parametrize("gravity_disabled", [False, True])
def test_cpu_oracle_matches_real_mj_step_with_aligned_offset_and_ineligible_bodies(gravity_disabled):
  model = mujoco.MjModel.from_xml_string(XML)
  model.dof_armature[:] = np.linspace(0, .03, model.nv)
  if gravity_disabled:
    model.opt.disableflags |= int(mujoco.mjtDisableBit.mjDSBL_GRAVITY)
  data = _forward(model)
  program = lower_free_body_midpoints(model)
  assert program.nfree == 2  # articulated subtree is ineligible
  expected = _expected(model, data)
  assert np.all(expected["status"] == 0)
  qpos = data.qpos.copy()
  mujoco.mj_integratePos(model, qpos, expected["position_velocity"][0], model.opt.timestep)
  mujoco.mj_step(model, data)
  np.testing.assert_allclose(data.qvel, expected["qvel_next"][0], rtol=3e-5, atol=2e-6)
  np.testing.assert_allclose(data.qpos, qpos, rtol=3e-5, atol=2e-6)
  np.testing.assert_allclose(data.qacc, expected["qacc"][0], rtol=3e-5, atol=2e-6)


def test_inverse_discrete_disables_midpoint_and_uses_ordinary_fallback():
  model = mujoco.MjModel.from_xml_string(XML)
  model.opt.enableflags |= int(mujoco.mjtEnableBit.mjENBL_INVDISCRETE)
  data = _forward(model)
  descriptor = lower_free_body_midpoints(model)
  assert descriptor.nfree == 0
  result = _expected(model, data)
  np.testing.assert_allclose(
      result["qvel_next"][0], data.qvel + model.opt.timestep*result["qacc"][0],
      rtol=1e-12, atol=1e-12,
  )


def test_midpoint_oracle_rejects_nonfinite_inputs_and_supports_iteration_limit():
  model = mujoco.MjModel.from_xml_string(XML)
  data = _forward(model)
  args = (data.qvel, data.qacc, data.qacc,
          data.qfrc_smooth+data.qfrc_bias, data.xquat)
  with pytest.raises(ValueError, match="finite"):
    free_midpoint_oracle(model, np.full(model.nv, np.nan), *args[1:])
  failed = free_midpoint_oracle(model, *args, max_iterations=0)
  assert np.all(failed["status"] == 20)
  np.testing.assert_array_equal(failed["qvel_next"][0], data.qvel)


def test_empty_world_midpoint_metadata_and_status():
  model = mujoco.MjModel.from_xml_string(
      '<mujoco><option integrator="implicitfast"><flag contact="disable"/></option><worldbody/></mujoco>'
  )
  descriptor = lower_free_body_midpoints(model)
  assert descriptor.nfree == 0 and descriptor.nv == 0
  result = free_midpoint_oracle(
      model, np.empty(0), np.empty(0), np.empty(0), np.empty(0),
      np.ones((model.nbody, 4)),
  )
  assert result["qvel_next"].shape == (1, 0)
  assert result["status"].tolist() == [0]


def test_midpoint_descriptor_constants_are_immutable_and_shader_abi_dense():
  import re
  from pathlib import Path
  model = mujoco.MjModel.from_xml_string(XML)
  descriptor = lower_free_body_midpoints(model)
  for array in (descriptor.gravity, descriptor.dofadr, descriptor.bodyid,
                descriptor.mass, descriptor.inertia, descriptor.ipos,
                descriptor.iquat, descriptor.aligned):
    assert not array.flags.writeable
  shader = Path(__file__).parents[1] / "mujoco_metal" / "shaders" / "implicit_midpoint.metal"
  indices = [int(value) for value in re.findall(r"\[\[buffer\((\d+)\)\]\]", shader.read_text())]
  assert indices == list(range(20))


def test_batch_index_capacity_is_checked_before_device_initialization():
  model = mujoco.MjModel.from_xml_string(XML)
  with pytest.raises(ValueError, match="indexing capacity"):
    FreeBodyMidpointProgram(model, batch_size=2**31)


@pytest.mark.parametrize("field", ["density", "viscosity"])
def test_fluid_disables_midpoint_per_pinned_global_eligibility(field):
  model = mujoco.MjModel.from_xml_string(XML)
  setattr(model.opt, field, .1)
  assert lower_free_body_midpoints(model).nfree == 0


@pytest.mark.gpu
@pytest.mark.skipif(__import__("os").getenv("MUJOCO_METAL_RUN_GPU") != "1", reason="native GPU opt-in")
def test_device_midpoint_mask_preserves_ordinary_implicit_velocity_for_constrained_world():
  import torch
  model = mujoco.MjModel.from_xml_string(XML)
  data = _forward(model)
  mass = np.empty((model.nv, model.nv))
  mujoco.mj_fullM(model, data, mass)
  effective = implicitfast_oracle(model, mass, data.qfrc_smooth)["qacc"][0]
  tensor = lambda x: torch.as_tensor(np.tile(np.asarray(x, dtype=np.float32)[None],
                                             (2,)+(1,)*np.asarray(x).ndim), device="mps")
  program = FreeBodyMidpointProgram(model, 2)
  args = (tensor(data.qvel), tensor(effective), tensor(data.qacc),
          tensor(data.qfrc_smooth+data.qfrc_bias), tensor(data.xquat))
  result = program.run_device(*args, eligible_mask=torch.tensor([[True, True], [False, False]], device="mps"))
  ordinary = data.qvel+model.opt.timestep*effective
  np.testing.assert_allclose(result["qvel_next"].cpu().numpy()[1], ordinary, atol=2e-7, rtol=2e-6)
  np.testing.assert_array_equal(result["qacc"].cpu().numpy()[1], data.qacc.astype(np.float32))
  np.testing.assert_allclose(result["qvel_next"].cpu().numpy()[0], _expected(model,data)["qvel_next"][0],
                             rtol=3e-5, atol=2e-6)
  assert np.max(np.abs(result["qvel_next"].cpu().numpy()[0]-ordinary)) > 1e-6


SLEEP_XML = """<mujoco><option integrator="implicitfast" timestep=".002"
  gravity="0 0 0"><flag sleep="enable" contact="disable"/></option>
  <worldbody>
    <body name="moving" sleep="allowed"><freejoint/>
      <inertial pos="0 0 0" mass="1" diaginertia=".12 .18 .23"/>
      <geom type="box" size=".2 .15 .1"/></body>
    <body name="quiet" sleep="allowed" pos="2 0 0"><freejoint/>
      <inertial pos=".1 -.04 .02" mass="2" diaginertia=".09 .14 .19"/>
      <geom type="box" size=".15 .1 .2"/></body>
  </worldbody></mujoco>"""


def test_sleep_enabled_midpoint_fixture_admits_awake_bodies_and_sleeps_quiet_tree():
  model = mujoco.MjModel.from_xml_string(SLEEP_XML)
  descriptor = lower_free_body_midpoints(model)
  assert descriptor.nfree == 2
  from mujoco_metal.stepping import validate_stepping_profile
  profile = validate_stepping_profile(model, profile="integrated_implicitfast_v1")
  assert "sleep mode" not in profile.rejected
  data = mujoco.MjData(model)
  data.qvel[3:6] = [.2, -.3, .4]
  for _ in range(18):
    mujoco.mj_step(model, data)
  assert data.tree_awake.tolist() == [1, 0]
  assert np.linalg.norm(data.qvel[3:6]) > .4
  np.testing.assert_array_equal(data.qvel[6:], 0)


@pytest.mark.gpu
@pytest.mark.skipif(__import__("os").getenv("MUJOCO_METAL_RUN_GPU") != "1",
                    reason="opt-in GPU")
def test_native_implicitfast_midpoint_with_sleep_matches_pinned_trajectories():
  from mujoco_metal import MetalSimulation
  model = mujoco.MjModel.from_xml_string(SLEEP_XML)
  references = [mujoco.MjData(model) for _ in range(2)]
  references[0].qvel[3:6] = [.2, -.3, .4]
  references[1].qvel[3:6] = [-.35, .25, -.15]
  initial = np.stack([data.qvel for data in references]).astype(np.float32)
  sim = MetalSimulation(model, 2, qvel=initial, profile="integrated_implicitfast_v1")
  assert sim._midpoint is not None
  for step in range(18):
    sim.step()
    for world, data in enumerate(references):
      mujoco.mj_step(model, data)
      for field in ("qpos", "qvel", "qacc"):
        np.testing.assert_allclose(
            getattr(sim.state, "_" + field)[world].cpu().numpy(),
            getattr(data, field), rtol=4e-5, atol=2e-5,
            err_msg=f"{field}, step {step}, world {world}")
      np.testing.assert_array_equal(
          sim._sleep_schedule.tree_state[world].cpu().numpy(), data.tree_asleep)
  assert all(data.tree_awake.tolist() == [1, 0] for data in references)
  checkpoint = sim.snapshot()
  sim.step(3)
  expected = {name: getattr(sim.state, "_" + name).cpu().numpy().copy()
              for name in ("qpos", "qvel", "qacc", "time", "status")}
  sim.restore(checkpoint)
  sim.step(3)
  for name, value in expected.items():
    np.testing.assert_array_equal(getattr(sim.state, "_" + name).cpu().numpy(), value)
