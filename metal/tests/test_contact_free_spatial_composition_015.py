"""Public contact-free profiles retain mixed tendon inertia and passive forces."""
import os

import mujoco
import numpy as np
import pytest

from mujoco_metal.stepping import validate_stepping_profile
from test_cached_position_015 import ARMATURE_STAGE_XML


def _model(integrator):
  xml = ARMATURE_STAGE_XML.replace(
      '<option gravity="0 0 0"/>',
      f'<option gravity="0 0 0" timestep=".001" integrator="{integrator}"/>')
  xml = xml.replace('<fixed name="fixed" armature=".17">',
                    '<fixed name="fixed" armature=".17" stiffness="1.7" '
                    'damping=".02" springlength=".1">')
  xml = xml.replace('<spatial name="spatial" armature=".23">',
                    '<spatial name="spatial" armature=".23" stiffness="2" '
                    'damping=".04" springlength=".3">')
  xml = xml.replace('</tendon>', '</tendon><actuator>'
                    '<motor tendon="fixed" gear="1.3"/></actuator>')
  model = mujoco.MjModel.from_xml_string(xml)
  model.opt.disableflags |= int(mujoco.mjtDisableBit.mjDSBL_CONTACT)
  return model


@pytest.mark.parametrize("integrator", ["Euler", "RK4"])
def test_contact_free_mixed_spatial_profile_admission_cpu(integrator):
  profile = f"contact_free_transmission_{integrator.lower()}_v1"
  model = _model(integrator)
  assert validate_stepping_profile(model, profile=profile).name == profile
  # Prove the mixed fixture transmits nonzero velocity, passive force and
  # armature bias before any opt-in native construction.
  data = mujoco.MjData(model)
  data.qpos[:] = np.float32(.41)
  data.qvel[:] = np.float32(-.64)
  data.ctrl[:] = np.float32(.21)
  mujoco.mj_forward(model, data)
  assert abs(float(data.ten_velocity[1])) > .01
  assert abs(float(data.qfrc_passive[0])) > .01
  assert abs(float(data.qfrc_bias[0])) > 1e-5
  # A contact-free profile still has no coupled stage to handle tendon limits.
  model.tendon_limited[1] = 1
  model.tendon_range[1] = [.1, .8]
  with pytest.raises(ValueError, match="tendon limits"):
    validate_stepping_profile(model, profile=profile)


def test_actual_spatial_constructor_allocates_force_flags_in_dimension_abi_cpu(monkeypatch):
  import torch
  from mujoco_metal.spatial_tendons import MetalSpatialTendonDynamics

  original_device = torch.device
  monkeypatch.setattr(torch, "device", lambda value:
                      original_device("cpu") if value == "mps"
                      else original_device(value))

  class NoDispatch:
    def __getattr__(self, name):
      def unexpected(*args, **kwargs):
        raise AssertionError(f"CPU constructor check dispatched {name}")
      return unexpected

  monkeypatch.setattr(torch.mps, "compile_shader", lambda source: NoDispatch())
  model = _model("Euler")
  program = MetalSpatialTendonDynamics(model, batch_size=3)
  assert program._dims.dtype == torch.int32
  assert program._dims.shape == (14,)
  np.testing.assert_array_equal(program._dims[8:11].numpy(), [1, 1, 1])
  program._dims[11:14] = torch.tensor([1, 0, 1], dtype=torch.int32)
  np.testing.assert_array_equal(program._dims[11:14].numpy(), [1, 0, 1])


@pytest.mark.skipif(os.getenv("MUJOCO_METAL_RUN_GPU") != "1",
                    reason="opt-in native trajectory qualification")
@pytest.mark.parametrize("integrator", ["Euler", "RK4"])
def test_contact_free_mixed_spatial_force_mass_trajectory_and_replay_gpu(integrator):
  from mujoco_metal import MetalSimulation

  model = _model(integrator)
  profile = f"contact_free_transmission_{integrator.lower()}_v1"
  qpos = np.asarray([[.41]], dtype=np.float32)
  qvel = np.asarray([[-.64]], dtype=np.float32)
  ctrl = np.asarray([[.21]], dtype=np.float32)
  cpu = mujoco.MjData(model)
  cpu.qpos[:] = qpos[0]
  cpu.qvel[:] = qvel[0]
  cpu.ctrl[:] = ctrl[0]
  mujoco.mj_forward(model, cpu)
  assert abs(float(cpu.ten_velocity[1])) > .01
  assert abs(float(cpu.qfrc_passive[0])) > .01
  assert abs(float(cpu.qfrc_bias[0])) > 1e-5
  mass = np.zeros((model.nv, model.nv), dtype=np.float64)
  mujoco.mj_fullM(model, cpu, mass)
  sim = MetalSimulation(model, 1, qpos=qpos, qvel=qvel, profile=profile)
  assert sim._spatial_tendons is not None
  position = sim.prepare_forward_position()
  from mujoco_metal.forward_stages import ForwardStage
  native_mass = position.values[ForwardStage.POS]["dynamics"]["mass_matrix"]
  np.testing.assert_allclose(native_mass[0].cpu().numpy(), mass,
                             rtol=6e-5, atol=6e-6)
  initial = sim.snapshot()
  for _ in range(8):
    sim.step(1, ctrl=ctrl)
    mujoco.mj_step(model, cpu)
  assert int(sim.state.status[0].cpu()) == 0
  expected = {name: getattr(sim.state, name).cpu().numpy().copy()
              for name in ("qpos", "qvel", "qacc")}
  np.testing.assert_allclose(expected["qpos"][0], cpu.qpos, rtol=2e-4, atol=2e-5)
  np.testing.assert_allclose(expected["qvel"][0], cpu.qvel, rtol=2e-4, atol=2e-5)
  np.testing.assert_allclose(expected["qacc"][0], cpu.qacc, rtol=3e-4, atol=3e-4)
  sim.restore(initial)
  sim.step(8, ctrl=ctrl)
  for name, value in expected.items():
    np.testing.assert_array_equal(getattr(sim.state, name).cpu().numpy(), value)
