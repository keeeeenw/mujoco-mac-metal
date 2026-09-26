"""Opt-in GPU tests for the bounded implicitfast modified-mass solve."""
import os

import mujoco
import numpy as np
import pytest
torch = pytest.importorskip("torch")

from mujoco_metal.implicit import ImplicitFastProgram

pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(os.getenv("MUJOCO_METAL_RUN_GPU") != "1", reason="opt-in GPU"),
    pytest.mark.skipif(not torch.backends.mps.is_available(), reason="Apple MPS unavailable"),
]

XML = '''<mujoco><option integrator="implicitfast" timestep=".004" gravity="0 0 0"><flag contact="disable"/></option>
<worldbody><body><joint name="h" type="hinge" axis="0 1 0" damping=".3" stiffness="2" springref=".1"/>
<geom type="box" size=".1 .2 .3" mass="1"/><body pos=".3 0 0"><joint type="ball" damping=".12"/>
<geom type="box" size=".08 .09 .1" mass=".3"/></body><body pos="0 0 .7"><joint type="slide" axis="0 0 1" damping=".2"/>
<geom type="capsule" size=".07 .2" mass=".4"/></body></body></worldbody>
<actuator><motor joint="h" gear=".6"/></actuator></mujoco>'''


def test_mps_implicitfast_matches_cpu_step_velocity_and_position():
  model = mujoco.MjModel.from_xml_string(XML)
  batch = 3
  qpos = np.tile(model.qpos0, (batch, 1))
  qvel = np.tile(np.linspace(-.4, .5, model.nv), (batch, 1))
  qpos[0, 0] = .25
  qpos[1, 0] = -.18
  qpos[2, 0] = .06
  qvel[1] *= -1
  data = []
  masses, forces, expected_qvel, expected_qpos = [], [], [], []
  for row in range(batch):
    item = mujoco.MjData(model)
    item.qpos[:] = qpos[row]
    item.qvel[:] = qvel[row]
    item.ctrl[:] = [.15 * (-1 if row == 1 else 1)]
    mujoco.mj_forward(model, item)
    mass = np.empty((model.nv, model.nv), dtype=np.float64)
    mujoco.mj_fullM(model, item, mass)
    masses.append(mass.astype(np.float32))
    forces.append(item.qfrc_smooth.astype(np.float32))
    mujoco.mj_step(model, item)
    data.append(item)
    expected_qvel.append(item.qvel.copy())
    expected_qpos.append(item.qpos.copy())
  # mj_step uses the same current ctrl and current qfrc_smooth; qfrc_smooth
  # was captured before advancing above.
  program = ImplicitFastProgram(model, batch)
  result = program.run_device(
      torch.tensor(np.asarray(masses), dtype=torch.float32, device="mps").contiguous(),
      torch.tensor(np.asarray(forces), dtype=torch.float32, device="mps").contiguous(),
  )
  assert np.all(result["status"].cpu().numpy() == 0)
  acceleration = result["qacc"].cpu().numpy()
  predicted = qvel + model.opt.timestep*acceleration
  np.testing.assert_allclose(predicted, expected_qvel, atol=2e-5, rtol=2e-5)
  for row in range(batch):
    position = qpos[row].copy()
    mujoco.mj_integratePos(model, position, predicted[row], model.opt.timestep)
    np.testing.assert_allclose(position, expected_qpos[row], atol=2e-5, rtol=2e-5)


def test_mps_empty_world():
  model = mujoco.MjModel.from_xml_string(
      '<mujoco><option integrator="implicitfast"><flag contact="disable"/></option><worldbody/></mujoco>'
  )
  program = ImplicitFastProgram(model, batch_size=2)
  output = program.run_device(
      torch.empty((2, 0, 0), dtype=torch.float32, device="mps"),
      torch.empty((2, 0), dtype=torch.float32, device="mps"),
  )
  assert output["qacc"].shape == (2, 0)
  assert np.all(output["status"].cpu().numpy() == 0)
