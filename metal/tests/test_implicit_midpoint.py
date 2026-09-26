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
  assert indices == list(range(19))


def test_batch_index_capacity_is_checked_before_device_initialization():
  model = mujoco.MjModel.from_xml_string(XML)
  with pytest.raises(ValueError, match="indexing capacity"):
    FreeBodyMidpointProgram(model, batch_size=2**31)
