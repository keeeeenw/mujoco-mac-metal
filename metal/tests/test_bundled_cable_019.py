"""Pinned MuJoCo oracles for the native bundled cable passive force."""

import os

import pytest

mujoco = pytest.importorskip("mujoco")
np = pytest.importorskip("numpy")

from mujoco_metal.bundled_plugins import lower_bundled_plugins
from mujoco_metal.bundled_cable import (
    cable_force_reference,
    cable_workspace_sizes,
    lower_cable,
)


@pytest.fixture(scope="module", autouse=True)
def _load_plugins():
  assert mujoco.__version__ == "3.10.0"
  mujoco.mj_loadAllPluginLibraries(mujoco.PLUGINS_DIR)


def _model(*, root_geom="capsule", child_geom="capsule", fixed_root=False,
           integrator="Euler", flat="false", child_quat="1 0 0 0"):
  root_joint = "" if fixed_root else "<freejoint/>"
  if root_geom == "sphere":
    root_geom_xml = '<geom type="sphere" size=".025" mass=".2"/>'
  else:
    root_geom_xml = f'<geom type="{root_geom}" size=".025 .2" mass=".2"/>'
  if child_geom == "sphere":
    child_geom_xml = '<geom type="sphere" size=".025" mass=".2"/>'
  else:
    child_geom_xml = f'<geom type="{child_geom}" size=".025 .2" mass=".2"/>'
  return mujoco.MjModel.from_xml_string(f"""
    <mujoco>
      <option timestep="0.001" gravity="0 0 0" integrator="{integrator}"/>
      <extension><plugin plugin="mujoco.elasticity.cable">
        <instance name="rod">
          <config key="twist" value="800"/>
          <config key="bend" value="1200"/>
          <config key="flat" value="{flat}"/>
        </instance>
      </plugin></extension>
      <worldbody>
        <body name="segment0" pos="0 0 0">
          {root_joint}{root_geom_xml}
          <plugin instance="rod"/>
          <body name="segment1" pos="0 0 .4" quat="{child_quat}">
            <joint name="joint1" type="ball"/>
            {child_geom_xml}
            <plugin instance="rod"/>
          </body>
        </body>
      </worldbody>
    </mujoco>
  """)


def test_cable_compiled_constants_and_force_match_pinned_callback():
  model = _model()
  bundled = lower_bundled_plugins(model)
  cable = lower_cable(model, bundled)
  assert cable["body_ids"].shape == (2,)
  assert cable["previous"].tolist() == [-1, int(cable["body_ids"][0])]
  assert cable["following"].tolist() == [int(cable["body_ids"][1]), -1]
  np.testing.assert_allclose(cable["stiffness"][:, 3], [0, .4], atol=1e-7)

  data = mujoco.MjData(model)
  data.qpos[:] = model.qpos0
  # Exercise both twist and bend stress without changing the compiled model.
  address = int(model.jnt_qposadr[1])
  quaternion = np.asarray([1.0, .16, -.09, 0.0])
  data.qpos[address:address + 4] = quaternion / np.linalg.norm(quaternion)
  mujoco.mj_forward(model, data)
  reference = cable_force_reference(
      model, data.qpos, data.xquat, data.cdof, cable)
  np.testing.assert_allclose(reference, data.qfrc_passive,
                             rtol=8e-6, atol=2e-6)
  assert np.linalg.norm(reference) > 1e-4


def test_fixed_root_and_zero_stiffness_geom_follow_pinned_compute_skip():
  model = _model(root_geom="sphere", fixed_root=True)
  cable = lower_cable(model, lower_bundled_plugins(model))
  assert int(model.body_dofnum[int(cable["body_ids"][0])]) == 0
  np.testing.assert_array_equal(cable["stiffness"][0, :3], 0.0)
  data = mujoco.MjData(model)
  address = int(model.jnt_qposadr[0])
  quaternion = np.asarray([1.0, .16, -.09, 0.0])
  data.qpos[address:address + 4] = quaternion / np.linalg.norm(quaternion)
  mujoco.mj_forward(model, data)
  reference = cable_force_reference(model, data.qpos, data.xquat, data.cdof, cable)
  np.testing.assert_allclose(reference, data.qfrc_passive,
                             rtol=8e-6, atol=2e-6)


def test_unsupported_cross_section_keeps_zero_stiffness_and_skips_segment():
  model = _model(child_geom="sphere")
  cable = lower_cable(model, lower_bundled_plugins(model))
  np.testing.assert_array_equal(cable["stiffness"][1, :3], 0.0)
  data = mujoco.MjData(model)
  address = int(model.jnt_qposadr[1])
  quaternion = np.asarray([1.0, .16, -.09, 0.0])
  data.qpos[address:address + 4] = quaternion / np.linalg.norm(quaternion)
  mujoco.mj_forward(model, data)
  reference = cable_force_reference(model, data.qpos, data.xquat, data.cdof, cable)
  np.testing.assert_allclose(reference, data.qfrc_passive,
                             rtol=8e-6, atol=2e-6)


def test_cable_flat_flag_matches_pinned_rest_curvature_branch():
  child_quat = "0.9238795325 0 0.3826834324 0"
  lowered = []
  for flat in ("false", "true"):
    model = _model(flat=flat, child_quat=child_quat)
    cable = lower_cable(model, lower_bundled_plugins(model))
    data = mujoco.MjData(model)
    address = int(model.jnt_qposadr[1])
    quaternion = np.asarray([1.0, .16, -.09, 0.0])
    data.qpos[address:address + 4] = quaternion / np.linalg.norm(quaternion)
    mujoco.mj_forward(model, data)
    reference = cable_force_reference(
        model, data.qpos, data.xquat, data.cdof, cable)
    np.testing.assert_allclose(reference, data.qfrc_passive,
                               rtol=8e-6, atol=2e-6)
    lowered.append(cable["omega0"].copy())
  assert np.linalg.norm(lowered[0][1]) > .1
  np.testing.assert_array_equal(lowered[1][1], 0.0)


def test_cable_runtime_capacity_is_exact_and_profile_admits_plugin():
  from mujoco_metal.capacity import estimate_capacity
  from mujoco_metal.coupled_constraints import lower_coupled_constraints
  from mujoco_metal.stepping import validate_stepping_profile

  model = _model()
  bundled = lower_bundled_plugins(model)
  sizes = cable_workspace_sizes(model, bundled)
  assert sizes["body_ids"] == 2
  assert sizes["stiffness"] == 8
  assert sizes["omega0"] == 6
  assert sizes["local_quat"] == 8
  assert sizes["dims"] == 5 and sizes["flags"] == 1
  rows = lower_coupled_constraints(model)
  estimate = estimate_capacity(
      model, 2, rows.npairs, rows.ncontacts_max, rows.nr,
      bundled_cable_segments=2)
  sizes_by_name = dict(estimate.memory_breakdown)
  assert sizes_by_name["plugin.cable.stiffness"] == 8 * 4
  assert sizes_by_name["plugin.cable.body_parent"] == int(model.nbody) * 4
  profile = validate_stepping_profile(model, profile="integrated_euler_v1")
  assert "bundled elasticity.cable" in " ".join(profile.supported)


@pytest.mark.parametrize("integrator,profile", [
    ("Euler", "integrated_euler_v1"),
    ("RK4", "integrated_rk4_v1"),
    ("implicit", "integrated_implicit_v1"),
    ("implicitfast", "integrated_implicitfast_v1"),
])
def test_cable_public_admission_and_lowering_cover_integrator_matrix(
    integrator, profile):
  from mujoco_metal.stepping import validate_stepping_profile

  model = _model(integrator=integrator)
  bundled = lower_bundled_plugins(model)
  cable = lower_cable(model, bundled)
  assert cable["body_ids"].shape == (2,)
  decision = validate_stepping_profile(model, profile=profile)
  assert decision.name == profile
  assert "bundled elasticity.cable" in " ".join(decision.supported)


@pytest.mark.skipif(os.getenv("MUJOCO_METAL_RUN_GPU") != "1",
                    reason="opt-in MPS bundled cable trajectory")
def test_native_bundled_cable_trajectory_matches_pinned_steps():
  from mujoco_metal import MetalSimulation

  model = _model()
  initial = mujoco.MjData(model)
  address = int(model.jnt_qposadr[1])
  quaternion = np.asarray([1.0, .16, -.09, 0.0])
  initial.qpos[address:address + 4] = quaternion / np.linalg.norm(quaternion)
  sim = MetalSimulation(
      model, batch_size=1, qpos=initial.qpos[None].astype(np.float32),
      qvel=initial.qvel[None].astype(np.float32),
      profile="integrated_euler_v1")
  reference = mujoco.MjData(model)
  reference.qpos[:] = initial.qpos
  reference.qvel[:] = initial.qvel
  for step in range(12):
    mujoco.mj_step(model, reference)
    status = sim.step()
    assert np.all(status.detach().cpu().numpy() == 0), f"step {step}"
    np.testing.assert_allclose(sim.state.qpos[0].detach().cpu().numpy(),
                               reference.qpos, rtol=2e-4, atol=3e-6,
                               err_msg=f"qpos at step {step}")
    np.testing.assert_allclose(sim.state.qvel[0].detach().cpu().numpy(),
                               reference.qvel, rtol=3e-4, atol=4e-5,
                               err_msg=f"qvel at step {step}")


@pytest.mark.skipif(os.getenv("MUJOCO_METAL_RUN_GPU") != "1",
                    reason="opt-in native cable query/checkpoint composition")
@pytest.mark.parametrize("flat", ["false", "true"])
@pytest.mark.parametrize("integrator,profile", [
    ("Euler", "integrated_euler_v1"),
    ("RK4", "integrated_rk4_v1"),
    ("implicit", "integrated_implicit_v1"),
    ("implicitfast", "integrated_implicitfast_v1"),
])
def test_native_cable_query_preserves_program_state_and_replays(
    integrator, profile, flat):
  from mujoco_metal import MetalSimulation
  from mujoco_metal.forward_stages import ForwardStage
  from mujoco_metal.native_api import mj_inverseSkip

  model = _model(integrator=integrator, flat=flat,
                 child_quat="0.9238795325 0 0.3826834324 0")
  reference = mujoco.MjData(model)
  address = int(model.jnt_qposadr[1])
  quaternion = np.asarray([1.0, .16, -.09, 0.0])
  reference.qpos[address:address + 4] = quaternion / np.linalg.norm(quaternion)
  mujoco.mj_forward(model, reference)
  sim = MetalSimulation(
      model, batch_size=1,
      qpos=np.asarray(reference.qpos, dtype=np.float32)[None].copy(),
      qvel=np.asarray(reference.qvel, dtype=np.float32)[None].copy(),
      profile=profile)
  for step in range(3):
    mujoco.mj_step(model, reference)
    status = sim.step()
    assert np.all(status.detach().cpu().numpy() == 0), (integrator, step)
    np.testing.assert_allclose(
        sim.state.qpos[0].detach().cpu().numpy(), reference.qpos,
        rtol=4e-4, atol=3e-5, err_msg=f"{integrator} qpos step {step}")
    np.testing.assert_allclose(
        sim.state.qvel[0].detach().cpu().numpy(), reference.qvel,
        rtol=4e-4, atol=4e-5, err_msg=f"{integrator} qvel step {step}")

  checkpoint = sim.snapshot()
  cpu_checkpoint = (reference.qpos.copy(), reference.qvel.copy(),
                    reference.act.copy(), float(reference.time))
  for step in range(2):
    mujoco.mj_step(model, reference)
    status = sim.step()
    assert np.all(status.detach().cpu().numpy() == 0), (integrator, step + 3)
  continued_qpos = sim.state.qpos.detach().cpu().numpy().copy()
  continued_qvel = sim.state.qvel.detach().cpu().numpy().copy()
  sim.restore(checkpoint)
  reference.qpos[:] = cpu_checkpoint[0]
  reference.qvel[:] = cpu_checkpoint[1]
  reference.act[:] = cpu_checkpoint[2]
  reference.time = cpu_checkpoint[3]
  mujoco.mj_forward(model, reference)
  for step in range(2):
    mujoco.mj_step(model, reference)
    status = sim.step()
    assert np.all(status.detach().cpu().numpy() == 0), (integrator, step + 3)
  np.testing.assert_array_equal(sim.state.qpos.detach().cpu().numpy(), continued_qpos)
  np.testing.assert_array_equal(sim.state.qvel.detach().cpu().numpy(), continued_qvel)

  # The query evaluates the current cable passive force. Its transient run flag
  # and all prepared/state owners must be restored afterward.
  position = sim.prepare_forward_position()
  assert position.stage == ForwardStage.POS
  sim._cable._flags.fill_(1)
  flags_before = sim._cable._flags.clone()
  qpos_before = sim.state.qpos.detach().clone()
  qvel_before = sim.state.qvel.detach().clone()
  accepted_before = sim.accepted_step
  result = mj_inverseSkip(
      sim, skipstage=mujoco.mjtStage.mjSTAGE_POS, record=position,
      qacc=sim.state._qacc.clone().contiguous(), return_details=True)
  assert np.all(result["status"].detach().cpu().numpy() == 0)
  torch = sim.state._torch
  torch.testing.assert_close(sim._cable._flags, flags_before, rtol=0, atol=0)
  torch.testing.assert_close(sim.state.qpos, qpos_before, rtol=0, atol=0)
  torch.testing.assert_close(sim.state.qvel, qvel_before, rtol=0, atol=0)
  assert sim._forward_stages._record is position
  accepted_after = sim.accepted_step
  assert accepted_after["generation"] == accepted_before["generation"]
  torch.testing.assert_close(accepted_after["input_time"],
                             accepted_before["input_time"], rtol=0, atol=0)
