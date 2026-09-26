"""Opt-in MPS joint-stage comparisons against actual MuJoCo forward data."""
import os

import mujoco
import numpy as np
import pytest
import torch

from mujoco_metal.joint_constraints import JointConstraintProgram

pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(os.getenv("MUJOCO_METAL_RUN_GPU") != "1", reason="opt-in GPU"),
    pytest.mark.skipif(not torch.backends.mps.is_available(), reason="Apple MPS unavailable"),
]

XML = '''<mujoco><option timestep="0.002" gravity="0 0 0"><flag contact="disable"/></option>
<worldbody><body><joint name="a" type="hinge" axis="0 0 1" range="-0.2 0.2" limited="true" margin="0.03" frictionloss="0.3"/>
<geom type="capsule" size=".1 .2" mass="1"/><body pos="0 0 .4"><joint name="b" type="hinge" axis="0 1 0" range="-0.1 0.1" limited="true" margin="0.02" frictionloss="0.2"/>
<geom type="capsule" size=".1 .2" mass="1"/></body></body></worldbody>
<equality><joint name="link" joint1="a" joint2="b" polycoef="0 1 .2 0 0" solref="0.02 1"/></equality></mujoco>'''


def _gpu_case(model, states, eq_active=None):
  batch = len(states)
  program = JointConstraintProgram(model, batch)
  masses, smooth, positions, velocities, expected_force, expected_acc = [], [], [], [], [], []
  for world, (pos, vel) in enumerate(states):
    data = mujoco.MjData(model)
    data.qpos[:] = pos
    data.qvel[:] = vel
    if eq_active is not None:
      data.eq_active[:] = np.asarray(eq_active[world], dtype=np.uint8)
    mujoco.mj_forward(model, data)
    mass = np.empty((model.nv, model.nv), dtype=np.float64)
    mujoco.mj_fullM(model, data, mass)
    masses.append(mass.astype(np.float32))
    smooth.append(data.qfrc_smooth.astype(np.float32))
    positions.append(pos)
    velocities.append(vel)
    expected_force.append(data.qfrc_constraint.copy())
    expected_acc.append(data.qacc.copy())
  kwargs = {}
  if eq_active is not None:
    kwargs["eq_active"] = torch.tensor(eq_active, dtype=torch.int32, device="mps").contiguous()
  output = program.run_device(
      torch.tensor(np.asarray(masses), dtype=torch.float32, device="mps").contiguous(),
      torch.tensor(np.asarray(smooth), dtype=torch.float32, device="mps").contiguous(),
      torch.tensor(np.asarray(positions), dtype=torch.float32, device="mps").contiguous(),
      torch.tensor(np.asarray(velocities), dtype=torch.float32, device="mps").contiguous(),
      **kwargs,
  )
  status = output["status"].cpu().numpy()
  assert np.all(status == 0), (status, output["residual"].cpu().numpy())
  np.testing.assert_allclose(output["qfrc_constraint"].cpu().numpy(), expected_force, rtol=4e-4, atol=4e-4)
  np.testing.assert_allclose(output["qacc"].cpu().numpy(), expected_acc, rtol=5e-4, atol=2e-2)


def test_mps_coupled_equalities_friction_limits_multiple_worlds():
  model = mujoco.MjModel.from_xml_string(XML)
  _gpu_case(model, [([.24, .16], [1., -.4]), ([-.24, -.16], [-.7, .5]),
                    ([.04, -.02], [.1, -.2])])


def test_mps_equality_inactive_and_model_equality_disabled():
  model = mujoco.MjModel.from_xml_string(XML)
  _gpu_case(model, [([.24, .16], [1., -.4]), ([.24, .16], [1., -.4])],
            eq_active=[[1], [0]])
  model.opt.disableflags |= int(mujoco.mjtDisableBit.mjDSBL_EQUALITY)
  _gpu_case(model, [([.24, .16], [1., -.4]), ([-.24, -.16], [-.7, .5])])


def test_mps_empty_world_returns_empty_acceleration_and_success():
  model = mujoco.MjModel.from_xml_string('<mujoco><worldbody/></mujoco>')
  program = JointConstraintProgram(model, batch_size=2)
  result = program.run_device(
      torch.empty((2, 0, 0), dtype=torch.float32, device="mps"),
      torch.empty((2, 0), dtype=torch.float32, device="mps"),
      torch.empty((2, 0), dtype=torch.float32, device="mps"),
      torch.empty((2, 0), dtype=torch.float32, device="mps"),
  )
  assert result["qacc"].shape == (2, 0)
  assert np.all(result["status"].cpu().numpy() == 0)
