"""Pinned MuJoCo 3.10 compiled-plugin descriptor lowering tests."""

import ctypes
import os

import pytest

mujoco = pytest.importorskip("mujoco")
np = pytest.importorskip("numpy")

from mujoco_metal.bundled_plugins import lower_bundled_plugins
from mujoco_metal.bundled_pid import lower_pid_actuators
from mujoco_metal.plugin_sdf import sdf_distance_local, sdf_gradient_local

_KIND_BY_NAME = {name: kind for kind, name in enumerate(
    ("", "bolt", "bowl", "gear", "nut", "torus")) if name}


@pytest.fixture(scope="module", autouse=True)
def _load_bundled_plugin_libraries():
  assert mujoco.__version__ == "3.10.0"
  mujoco.mj_loadAllPluginLibraries(mujoco.PLUGINS_DIR)


def _mixed_plugin_model(integrator):
  """A single model composing cable, PID, and touch-grid plugin producers."""
  return mujoco.MjModel.from_xml_string(f"""
    <mujoco>
      <option timestep="0.001" gravity="0 0 -9.81"
              integrator="{integrator}" cone="elliptic"/>
      <extension>
        <plugin plugin="mujoco.elasticity.cable">
          <instance name="rod">
            <config key="twist" value="800"/>
            <config key="bend" value="1200"/>
            <config key="flat" value="false"/>
          </instance>
        </plugin>
        <plugin plugin="mujoco.pid">
          <instance name="pid">
            <config key="kp" value="8"/>
            <config key="ki" value="2"/>
            <config key="kd" value=".5"/>
            <config key="imax" value=".2"/>
            <config key="slewmax" value="20"/>
          </instance>
        </plugin>
        <plugin plugin="mujoco.sensor.touch_grid"/>
      </extension>
      <worldbody>
        <geom name="floor" type="plane" size="1 1 .1"/>
        <body name="pad" pos="0 0 .08">
          <freejoint name="pad_free"/>
          <geom name="pad_geom" type="box" size=".12 .12 .04" mass="1"/>
          <site name="touch" size=".1 .1 .1"/>
        </body>
        <body name="rod0" pos=".4 0 .5">
          <freejoint name="rod0_free"/><geom type="capsule" size=".025 .2" mass=".2"
              contype="0" conaffinity="0"/>
          <plugin instance="rod"/>
          <body name="rod1" pos="0 0 .4">
            <joint name="rod_joint" type="ball"/>
            <geom type="capsule" size=".025 .2" mass=".2"
                contype="0" conaffinity="0"/>
            <plugin instance="rod"/>
          </body>
        </body>
        <body name="pid_body" pos="-.4 0 .5">
          <joint name="pid_joint" type="slide" axis="0 0 1"/>
          <geom type="box" size=".04 .04 .04" mass=".2"
              contype="0" conaffinity="0"/>
        </body>
      </worldbody>
      <actuator>
        <plugin joint="pid_joint" plugin="mujoco.pid" instance="pid"
                actdim="2"/>
      </actuator>
      <sensor>
        <plugin name="grid" plugin="mujoco.sensor.touch_grid"
                objtype="site" objname="touch">
          <config key="nchannel" value="3"/>
          <config key="size" value="4 3"/>
          <config key="fov" value="150 80"/>
          <config key="gamma" value=".2"/>
        </plugin>
      </sensor>
    </mujoco>
  """)


def test_pid_plugin_compiled_attributes_and_actuator_map_lower_exactly():
  model = mujoco.MjModel.from_xml_string("""
    <mujoco>
      <extension>
        <plugin plugin="mujoco.pid">
          <instance name="pid">
            <config key="kp" value="40"/>
            <config key="ki" value="40"/>
            <config key="kd" value="4"/>
            <config key="imax" value="1"/>
            <config key="slewmax" value="3"/>
          </instance>
        </plugin>
      </extension>
      <worldbody>
        <body>
          <joint name="j" type="slide" axis="0 0 1"/>
          <geom size=".01" mass="1"/>
        </body>
      </worldbody>
      <actuator>
        <plugin joint="j" plugin="mujoco.pid" instance="pid" actdim="2"/>
      </actuator>
    </mujoco>
  """)
  lowered = lower_bundled_plugins(model)
  assert len(lowered.instances) == 1
  plugin = lowered.instance(0)
  assert plugin.name == "mujoco.pid"
  assert plugin.attribute_names == ("kp", "ki", "kd", "imax", "slewmax")
  assert plugin.config == {
      "kp": "40", "ki": "40", "kd": "4", "imax": "1", "slewmax": "3",
  }
  np.testing.assert_array_equal(lowered.actuator_plugin, [0])
  np.testing.assert_array_equal(lowered.instance_kind, [0])
  assert lowered.geom_sdf_aabb.shape == (int(model.ngeom), 6)
  assert not lowered.geom_sdf_aabb.flags.writeable
  assert not lowered.actuator_plugin.flags.writeable
  pid_params, pid_flags, pid_mask = lower_pid_actuators(model, lowered)
  np.testing.assert_array_equal(pid_mask, [1])
  np.testing.assert_array_equal(pid_flags, [[1, 1]])
  np.testing.assert_allclose(pid_params[0], [40, 40, 4, 1 / 40, 3],
                             rtol=0, atol=1e-7)
  from mujoco_metal.stateful_actuation import ActuatorModel
  actuator = ActuatorModel(model, allow_inherited=True,
                           bundled_plugins=lowered)
  np.testing.assert_array_equal(actuator.plugin_actuator_mask, [1])
  np.testing.assert_array_equal(actuator.actnum, [2])
  np.testing.assert_array_equal(actuator.gainprm[0], np.zeros(10, np.float32))


@pytest.mark.parametrize("integrator", ["Euler", "RK4", "implicit", "implicitfast"])
def test_cable_pid_touch_grid_compose_in_public_integrator_admission(integrator):
  from mujoco_metal.bundled_pid import validate_pid_profile_plugins
  from mujoco_metal.stepping import validate_stepping_profile

  profile = {
      "Euler": "integrated_euler_v1",
      "RK4": "integrated_rk4_v1",
      "implicit": "integrated_implicit_v1",
      "implicitfast": "integrated_implicitfast_v1",
  }[integrator]
  model = _mixed_plugin_model(integrator)
  plugins = lower_bundled_plugins(model)
  names = {instance.name for instance in plugins.instances}
  assert names == {"mujoco.elasticity.cable", "mujoco.pid",
                   "mujoco.sensor.touch_grid"}
  assert validate_pid_profile_plugins(model, plugins) is plugins
  result = validate_stepping_profile(model, profile=profile)
  assert result.name == profile
  supported = " ".join(result.supported)
  assert "bundled elasticity.cable" in supported
  assert "bundled PID" in supported
  assert "touch_grid ACC sensors" in supported
  # Exercise the independent upstream callbacks for all three producers on
  # the same compiled scene, so an admission-only fixture cannot pass while
  # one of its model/plugin bindings is inert.
  from mujoco_metal.bundled_cable import cable_force_reference, lower_cable
  from mujoco_metal.bundled_touch_grid import lower_touch_grid_sensors
  data = mujoco.MjData(model)
  pad_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "pad_free")
  rod_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "rod_joint")
  data.qpos[int(model.jnt_qposadr[pad_id]) + 2] = .03
  rod_addr = int(model.jnt_qposadr[rod_id])
  perturb = np.asarray([1., .12, -.07, .03])
  data.qpos[rod_addr:rod_addr + 4] = perturb / np.linalg.norm(perturb)
  data.ctrl[:] = .2
  mujoco.mj_forward(model, data)
  assert data.ncon > 0
  grid = lower_touch_grid_sensors(
      model, plugins)[0]
  assert np.any(np.abs(data.sensordata[grid.address:grid.address + grid.dimension]) > 0)
  assert np.linalg.norm(data.actuator_force) > 0
  cable = lower_cable(model, plugins)
  expected_cable = cable_force_reference(
      model, data.qpos, data.xquat, data.cdof, cable)
  np.testing.assert_allclose(data.qfrc_passive, expected_cable,
                             rtol=8e-6, atol=2e-6)


@pytest.mark.skipif(os.getenv("MUJOCO_METAL_RUN_GPU") != "1",
                    reason="opt-in composed cable/PID/touch-grid physics")
@pytest.mark.parametrize("integrator,profile", [
    ("Euler", "integrated_euler_v1"),
    ("RK4", "integrated_rk4_v1"),
    ("implicit", "integrated_implicit_v1"),
    ("implicitfast", "integrated_implicitfast_v1"),
])
def test_native_cable_pid_touch_grid_composition_matches_pinned_steps_and_replay(
    integrator, profile):
  from mujoco_metal import MetalSimulation
  from mujoco_metal.bundled_touch_grid import lower_touch_grid_sensors

  model = _mixed_plugin_model(integrator)
  refs = [mujoco.MjData(model), mujoco.MjData(model)]
  pad_joint = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "pad_free")
  rod_joint = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "rod_joint")
  pad_qadr = int(model.jnt_qposadr[pad_joint])
  rod_qadr = int(model.jnt_qposadr[rod_joint])
  perturb = np.asarray([1.0, .12, -.07, .03])
  perturb /= np.linalg.norm(perturb)
  for world, ref in enumerate(refs):
    ref.qpos[pad_qadr + 2] = .03 if world == 0 else .35
    ref.qpos[rod_qadr:rod_qadr + 4] = perturb
    ref.ctrl[:] = .2
    mujoco.mj_forward(model, ref)
  qpos0 = np.stack([ref.qpos for ref in refs]).astype(np.float32)
  qvel0 = np.stack([ref.qvel for ref in refs]).astype(np.float32)
  sim = MetalSimulation(model, batch_size=2, qpos=qpos0, qvel=qvel0,
                        profile=profile)
  sensor = lower_touch_grid_sensors(
      model, lower_bundled_plugins(model))[0]
  contact_seen = False
  for step in range(4):
    for ref in refs:
      ref.ctrl[:] = .2
      mujoco.mj_step(model, ref)
    status = sim.step(ctrl=np.full((2, model.nu), .2, dtype=np.float32))
    assert np.all(status.detach().cpu().numpy() == 0), (integrator, step)
    for world, ref in enumerate(refs):
      np.testing.assert_allclose(
          sim.state.qpos[world].detach().cpu().numpy(), ref.qpos,
          rtol=5e-4, atol=3e-5,
          err_msg=f"{integrator} composed qpos world {world} step {step}")
      np.testing.assert_allclose(
          sim.state.qvel[world].detach().cpu().numpy(), ref.qvel,
          rtol=8e-4, atol=6e-5,
          err_msg=f"{integrator} composed qvel world {world} step {step}")
      np.testing.assert_allclose(
          sim.state._act[world].detach().cpu().numpy(), ref.act,
          rtol=5e-4, atol=3e-5,
          err_msg=f"{integrator} PID activation world {world} step {step}")
      sensor_actual = sim.step_sensordata()[world,
          sensor.address:sensor.address + sensor.dimension]
      np.testing.assert_allclose(
          sensor_actual,
          ref.sensordata[sensor.address:sensor.address + sensor.dimension],
          rtol=4e-4, atol=4e-5,
          err_msg=f"{integrator} touch-grid world {world} step {step}")
      if world == 0:
        contact_seen |= bool(np.any(np.abs(sensor_actual) > 0))
  assert contact_seen, "composed model never produced a touch-grid contact sample"

  checkpoint = sim.snapshot()
  cpu_saved = [(ref.qpos.copy(), ref.qvel.copy(), ref.act.copy(),
                ref.ctrl.copy(), float(ref.time)) for ref in refs]
  for ref in refs:
    ref.ctrl[:] = .2
    mujoco.mj_step(model, ref)
  assert np.all(sim.step(ctrl=np.full((2, model.nu), .2, dtype=np.float32))
                .detach().cpu().numpy() == 0)
  replay_values = tuple(
      getattr(sim.state, field).detach().cpu().numpy().copy()
      for field in ("_qpos", "_qvel", "_act"))
  replay_sensor = sim.step_sensordata().copy()
  sim.restore(checkpoint)
  for ref, saved in zip(refs, cpu_saved):
    ref.qpos[:], ref.qvel[:], ref.act[:], ref.ctrl[:], ref.time = saved
    mujoco.mj_forward(model, ref)
    ref.ctrl[:] = .2
    mujoco.mj_step(model, ref)
  assert np.all(sim.step(ctrl=np.full((2, model.nu), .2, dtype=np.float32))
                .detach().cpu().numpy() == 0)
  for field, expected in zip(("_qpos", "_qvel", "_act"), replay_values):
    np.testing.assert_array_equal(
        getattr(sim.state, field).detach().cpu().numpy(), expected)
  np.testing.assert_array_equal(
      sim.step_sensordata(), replay_sensor)


def test_pid_plugin_cpu_force_and_slot_derivatives_match_pinned_callback():
  model = mujoco.MjModel.from_xml_string("""
    <mujoco><option timestep=".002"/>
      <extension><plugin plugin="mujoco.pid"><instance name="pid">
        <config key="kp" value="4"/><config key="ki" value="2"/>
        <config key="kd" value=".5"/><config key="imax" value="1"/>
        <config key="slewmax" value="3"/>
      </instance></plugin></extension>
      <worldbody><body><joint name="j" type="slide" axis="0 0 1"/>
        <geom type="box" size=".1 .1 .1" mass="1"/>
      </body></worldbody>
      <actuator><plugin joint="j" plugin="mujoco.pid" instance="pid"
        actdim="2"/></actuator>
    </mujoco>
  """)
  data = mujoco.MjData(model)
  data.qpos[0] = .1
  data.qvel[0] = .2
  data.ctrl[0] = .5
  data.act[:] = [.3, .45]
  mujoco.mj_forward(model, data)
  error = float(data.ctrl[0] - data.actuator_length[0])
  integral = float(np.clip(data.act[0] + error * model.opt.timestep,
                           -1.0 / 2.0, 1.0 / 2.0))
  expected_dot = np.asarray([
      (integral - data.act[0]) / model.opt.timestep,
      (data.ctrl[0] - data.act[1]) / model.opt.timestep,
  ])
  expected_force = 4 * error + .5 * -data.actuator_velocity[0] + 2 * integral
  np.testing.assert_allclose(data.act_dot, expected_dot, rtol=0, atol=1e-12)
  np.testing.assert_allclose(data.actuator_force, [expected_force],
                             rtol=0, atol=1e-12)


def test_pid_plugin_stateless_pd_lowering_matches_pinned_cpu():
  model = mujoco.MjModel.from_xml_string("""
    <mujoco><extension><plugin plugin="mujoco.pid"><instance name="pid">
      <config key="kp" value="3"/><config key="kd" value=".4"/>
    </instance></plugin></extension>
    <worldbody><body><joint name="j" type="slide" axis="0 0 1"/>
      <geom type="box" size=".1 .1 .1" mass="1"/></body></worldbody>
    <actuator><plugin joint="j" plugin="mujoco.pid" instance="pid"/></actuator>
  </mujoco>
  """)
  lowered = lower_bundled_plugins(model)
  params, flags, mask = lower_pid_actuators(model, lowered)
  np.testing.assert_array_equal(model.actuator_actnum, [0])
  np.testing.assert_array_equal(model.actuator_actadr, [-1])
  np.testing.assert_array_equal(mask, [1])
  np.testing.assert_array_equal(flags, [[0, 0]])
  np.testing.assert_allclose(params[0], [3, 0, .4, -1, -1], rtol=0, atol=1e-7)
  data = mujoco.MjData(model)
  data.qpos[0] = .2
  data.qvel[0] = -.5
  data.ctrl[0] = .7
  mujoco.mj_forward(model, data)
  expected = 3 * (data.ctrl[0] - data.actuator_length[0]) - .4 * data.actuator_velocity[0]
  np.testing.assert_allclose(data.actuator_force, [expected], rtol=0, atol=1e-12)


def _mixed_pid_model(integrator, disable_case):
  """One stateless instance plus a shared stateful instance on three rows."""
  xml = """
    <mujoco><option timestep="0.002" gravity="0 0 -9.81"/>
      <extension><plugin plugin="mujoco.pid">
        <instance name="pd">
          <config key="kp" value="3"/><config key="kd" value=".4"/>
        </instance>
        <instance name="pi">
          <config key="kp" value="8"/><config key="ki" value="1"/>
          <config key="kd" value=".3"/><config key="imax" value=".2"/>
          <config key="slewmax" value="3"/>
        </instance>
      </plugin></extension>
      <worldbody><body name="b0">
        <joint name="j0" type="slide" axis="1 0 0"/>
        <geom type="box" size=".1 .1 .1" mass="1"/>
        <body name="b1">
          <joint name="j1" type="slide" axis="0 1 0"/>
          <geom type="box" size=".1 .1 .1" mass="1"/>
        </body>
      </body></worldbody>
      <tendon><fixed name="t" limited="true" range="-.03 .03">
        <joint joint="j0" coef="1"/><joint joint="j1" coef="1"/>
      </fixed></tendon>
      <actuator>
        <plugin name="pd_joint" joint="j0" plugin="mujoco.pid"
          instance="pd" group="0" ctrllimited="true" ctrlrange="-.5 .5"
          forcelimited="true" forcerange="-.2 .2"/>
        <plugin name="pi_joint_a" joint="j1" plugin="mujoco.pid"
          instance="pi" group="1" actdim="2" ctrllimited="true"
          ctrlrange="-.5 .5" forcelimited="true" forcerange="-.2 .2"/>
        <plugin name="pi_joint_b" joint="j0" plugin="mujoco.pid"
          instance="pi" group="1" actdim="2" ctrllimited="true"
          ctrlrange="-.5 .5" forcelimited="true" forcerange="-.2 .2"/>
        <plugin name="pd_tendon" tendon="t" plugin="mujoco.pid"
          instance="pd" group="1" ctrllimited="true" ctrlrange="-.5 .5"
          forcelimited="true" forcerange="-.2 .2"/>
      </actuator>
      <sensor><actuatorfrc actuator="pd_joint"/>
        <actuatorfrc actuator="pi_joint_a"/>
        <actuatorfrc actuator="pi_joint_b"/>
        <actuatorfrc actuator="pd_tendon"/>
      </sensor>
    </mujoco>
  """
  model = mujoco.MjModel.from_xml_string(xml)
  model.opt.integrator = getattr(mujoco.mjtIntegrator, integrator)
  if disable_case == "group1":
    model.opt.disableactuator = 1 << 1
  elif disable_case == "clampctrl":
    model.opt.disableflags |= int(mujoco.mjtDisableBit.mjDSBL_CLAMPCTRL)
  elif disable_case == "actuation":
    model.opt.disableflags |= int(mujoco.mjtDisableBit.mjDSBL_ACTUATION)
  elif disable_case != "none":
    raise AssertionError(disable_case)
  return model


def test_pid_mixed_stateless_and_shared_stateful_layout_matches_pinned_cpu():
  model = _mixed_pid_model("mjINT_EULER", "none")
  lowered = lower_bundled_plugins(model)
  _, flags, mask = lower_pid_actuators(model, lowered)
  np.testing.assert_array_equal(mask, [1, 1, 1, 1])
  np.testing.assert_array_equal(model.actuator_actnum, [0, 2, 2, 0])
  np.testing.assert_array_equal(model.actuator_plugin, [0, 1, 1, 0])
  np.testing.assert_array_equal(flags, [[0, 0], [1, 1], [1, 1], [0, 0]])
  # The zero-state P/D rows are legal alongside two rows sharing one I/slew
  # instance; the activation cursor remains dense and covers only four slots.
  np.testing.assert_array_equal(model.actuator_actadr, [-1, 0, 2, -1])


@pytest.mark.parametrize("disable_case", [
    "none", "group1", "clampctrl", "actuation"])
def test_pid_mixed_rows_execute_pinned_plugin_callback_on_cpu(disable_case):
  model = _mixed_pid_model("mjINT_EULER", disable_case)
  data = mujoco.MjData(model)
  data.qpos[:] = [.08, -.035]
  data.qvel[:] = [.12, -.08]
  data.ctrl[:] = [1.1, .95, -.85, 1.2]
  mujoco.mj_forward(model, data)
  if disable_case == "actuation":
    np.testing.assert_array_equal(data.actuator_force, np.zeros(model.nu))
    np.testing.assert_array_equal(data.act_dot, np.zeros(model.na))
  elif disable_case == "group1":
    assert data.actuator_force[0] != 0
    # Pid::Compute runs after the generic group-filtered actuator loop, so
    # bundled PID force rows are still computed for a disabled group.
    active_model = _mixed_pid_model("mjINT_EULER", "none")
    active = mujoco.MjData(active_model)
    active.qpos[:] = data.qpos
    active.qvel[:] = data.qvel
    active.ctrl[:] = data.ctrl
    mujoco.mj_forward(active_model, active)
    np.testing.assert_allclose(data.actuator_force, active.actuator_force,
                               rtol=0, atol=1e-12)
    np.testing.assert_allclose(data.qfrc_actuator, active.qfrc_actuator,
                               rtol=0, atol=1e-12)
  else:
    # Both global clampctrl settings still respect the PID callback's own
    # GetCtrl clipping; group/tendon/actuator force limits apply afterward.
    clipped_ctrl = .5
    expected_pd_joint = np.clip(
        3 * (clipped_ctrl - data.actuator_length[0])
        - .4 * data.actuator_velocity[0], -.2, .2)
    np.testing.assert_allclose(data.actuator_force[0], expected_pd_joint,
                               rtol=0, atol=1e-12)
    assert np.all(np.isfinite(data.actuator_force))
  mujoco.mj_step(model, data)
  assert np.all(np.isfinite(data.qpos))
  assert np.all(np.isfinite(data.qvel))


@pytest.mark.parametrize("integrator_profile", [
    ("mjINT_EULER", "integrated_euler_v1"),
    ("mjINT_RK4", "integrated_rk4_v1"),
    ("mjINT_IMPLICIT", "integrated_implicit_v1"),
    ("mjINT_IMPLICITFAST", "integrated_implicitfast_v1"),
])
def test_pid_integrator_profiles_validate_with_pinned_derivative_semantics(
    integrator_profile):
  from mujoco_metal.stepping import validate_stepping_profile
  integrator, profile_name = integrator_profile
  model = _mixed_pid_model(integrator, "none")
  descriptor = validate_stepping_profile(model, profile=profile_name)
  assert "MuJoCo 3.10 bundled PID actuator force and activation callbacks" in descriptor.supported


@pytest.mark.parametrize("integrator_profile", [
    ("mjINT_EULER", "integrated_euler_v1"),
    ("mjINT_RK4", "integrated_rk4_v1"),
    ("mjINT_IMPLICIT", "integrated_implicit_v1"),
    ("mjINT_IMPLICITFAST", "integrated_implicitfast_v1"),
])
@pytest.mark.parametrize("disable_case", [
    "none", "group1", "clampctrl", "actuation"])
@pytest.mark.skipif(os.environ.get("MUJOCO_METAL_RUN_GPU") != "1",
                    reason="serialized mixed bundled-PID MPS trajectories")
def test_native_pid_mixed_rows_limits_and_integrators_match_pinned(
    integrator_profile, disable_case):
  torch = pytest.importorskip("torch")
  from mujoco_metal.simulation import MetalSimulation

  integrator, profile = integrator_profile
  model = _mixed_pid_model(integrator, disable_case)
  qpos = np.asarray([[.08, -.035], [-.06, .025]], dtype=np.float32)
  qvel = np.asarray([[.12, -.08], [-.09, .11]], dtype=np.float32)
  sim = MetalSimulation(model, batch_size=2, qpos=qpos, qvel=qvel,
                        profile=profile)
  refs = [mujoco.MjData(model) for _ in range(2)]
  for i, ref in enumerate(refs):
    ref.qpos[:] = qpos[i]
    ref.qvel[:] = qvel[i]
    mujoco.mj_forward(model, ref)

  checkpoint = None
  cpu_checkpoint = None
  replay_controls = []
  for step in range(10):
    ctrl = np.asarray([
        [1.1, .95 - .025 * step, -.85 + .02 * step, 1.2],
        [-1.3, -.9 + .03 * step, .8 - .015 * step, -.7],
    ], dtype=np.float32)
    status = sim.step(ctrl=ctrl)
    np.testing.assert_array_equal(status.detach().cpu().numpy(), [0, 0])
    for i, ref in enumerate(refs):
      ref.ctrl[:] = ctrl[i]
      mujoco.mj_step(model, ref)
    if step >= 5:
      replay_controls.append((step, ctrl.copy()))
    for label, got, expected, atol, rtol in (
        ("qpos", sim.state._qpos, [ref.qpos for ref in refs], 2e-5, 2e-4),
        ("qvel", sim.state._qvel, [ref.qvel for ref in refs], 3e-5, 3e-4),
        ("qacc", sim.state._qacc, [ref.qacc for ref in refs], 5e-4, 3e-3),
        ("act", sim.state._act, [ref.act for ref in refs], 2e-5, 2e-4),
        ("actuator sensors", sim._sensordata,
         [ref.sensordata for ref in refs], 2e-5, 2e-4),
    ):
      np.testing.assert_allclose(
          got.detach().cpu().numpy(), np.stack(expected).astype(np.float32),
          atol=atol, rtol=rtol, err_msg=f"{integrator}/{disable_case}/{label}")
    if step == 4:
      checkpoint = sim.snapshot()
      cpu_checkpoint = [{
          name: np.asarray(getattr(ref, name)).copy()
          for name in ("qpos", "qvel", "act", "ctrl", "qacc_warmstart")
      } | {"time": float(ref.time)} for ref in refs]
    if step == 6:
      # A public forward query reuses actuator/plugin workspaces. It must leave
      # the accepted state untouched so the following PID step remains exact.
      from mujoco_metal.native_api import mj_forward
      before = tuple(value.detach().cpu().clone() for value in (
          sim.state._qpos, sim.state._qvel, sim.state._qacc,
          sim.state._act, sim.state._time))
      mj_forward(sim, ctrl=ctrl)
      after = (sim.state._qpos, sim.state._qvel, sim.state._qacc,
               sim.state._act, sim.state._time)
      for old, current in zip(before, after):
        torch.testing.assert_close(current.detach().cpu(), old, rtol=0, atol=0)
      for i, ref in enumerate(refs):
        ref.ctrl[:] = ctrl[i]
        mujoco.mj_forward(model, ref)
      np.testing.assert_allclose(
          sim._sensordata.detach().cpu().numpy(),
          np.stack([ref.sensordata for ref in refs]).astype(np.float32),
          rtol=2e-4, atol=2e-5, err_msg="forward-query sensors")

  assert checkpoint is not None and cpu_checkpoint is not None
  final_native = tuple(value.detach().cpu().clone() for value in (
      sim.state._qpos, sim.state._qvel, sim.state._qacc,
      sim.state._act, sim.state._time, sim._sensordata))
  sim.restore(checkpoint)
  for ref, saved in zip(refs, cpu_checkpoint):
    for name in ("qpos", "qvel", "act", "ctrl", "qacc_warmstart"):
      getattr(ref, name)[:] = saved[name]
    ref.time = saved["time"]
  for step, ctrl in replay_controls:
    status = sim.step(ctrl=ctrl)
    np.testing.assert_array_equal(status.detach().cpu().numpy(), [0, 0])
    for i, ref in enumerate(refs):
      ref.ctrl[:] = ctrl[i]
      mujoco.mj_step(model, ref)
    if step == 6:
      from mujoco_metal.native_api import mj_forward
      mj_forward(sim, ctrl=ctrl)
      for i, ref in enumerate(refs):
        ref.ctrl[:] = ctrl[i]
        mujoco.mj_forward(model, ref)
  final_replayed = (sim.state._qpos, sim.state._qvel, sim.state._qacc,
                    sim.state._act, sim.state._time, sim._sensordata)
  for name, expected, actual in zip(
      ("qpos", "qvel", "qacc", "act", "time", "sensordata"),
      final_native, final_replayed):
    torch.testing.assert_close(
        actual.detach().cpu(), expected, rtol=0, atol=0,
        msg=lambda error, name=name: f"implicit replay differs in {name}: {error}")


@pytest.mark.skipif(os.environ.get("MUJOCO_METAL_RUN_GPU") != "1",
                    reason="serialized public bundled-PID MPS trajectory")
@pytest.mark.parametrize("integrator", ["euler", "rk4", "implicit", "implicitfast"])
def test_native_bundled_pid_step_and_checkpoint_replay_matches_pinned(integrator):
  torch = pytest.importorskip("torch")
  from mujoco_metal.simulation import MetalSimulation

  xml_integrator = {"euler": "Euler", "rk4": "RK4",
                    "implicit": "implicit", "implicitfast": "implicitfast"}[integrator]
  model = mujoco.MjModel.from_xml_string(f"""
    <mujoco><option timestep="0.002" gravity="0 0 -9.81"
      integrator="{xml_integrator}"/>
      <extension><plugin plugin="mujoco.pid"><instance name="pid">
        <config key="kp" value="4"/><config key="ki" value="2"/>
        <config key="kd" value=".5"/><config key="imax" value="1"/>
        <config key="slewmax" value="3"/>
      </instance></plugin></extension>
      <worldbody><body><joint name="j" type="slide" axis="0 0 1"/>
        <geom type="box" size=".1 .1 .1" mass="1"/>
      </body></worldbody>
      <actuator><plugin name="pid_act" joint="j" plugin="mujoco.pid"
        instance="pid" actdim="2"/></actuator>
      <sensor><jointpos joint="j"/><actuatorfrc actuator="pid_act"/></sensor>
    </mujoco>
  """)
  batch = 2
  qpos = np.asarray([[.1], [-.2]], dtype=np.float32)
  qvel = np.asarray([[.3], [-.1]], dtype=np.float32)
  sim = MetalSimulation(model, batch_size=batch, qpos=qpos, qvel=qvel,
                        profile=f"integrated_{integrator}_v1")
  refs = [mujoco.MjData(model) for _ in range(batch)]
  for world, data in enumerate(refs):
    data.qpos[:] = qpos[world]
    data.qvel[:] = qvel[world]
    mujoco.mj_forward(model, data)

  def compare():
    for name, native_value, cpu_values, atol, rtol in (
        ("qpos", sim.state._qpos, [d.qpos for d in refs], 2e-5, 2e-4),
        ("qvel", sim.state._qvel, [d.qvel for d in refs], 3e-5, 3e-4),
        ("qacc", sim.state._qacc, [d.qacc for d in refs], 5e-4, 3e-3),
        ("act", sim.state._act, [d.act for d in refs], 2e-5, 2e-4),
        ("sensordata", sim._sensordata, [d.sensordata for d in refs], 2e-5, 2e-4),
    ):
      expected = np.stack(cpu_values).astype(np.float32)
      np.testing.assert_allclose(native_value.detach().cpu().numpy(), expected,
                                 atol=atol, rtol=rtol, err_msg=name)

  checkpoint = None
  cpu_checkpoint = None
  for step in range(12):
    control = np.asarray([[.4 + .01 * step], [-.25 + .015 * step]], dtype=np.float32)
    status = sim.step(ctrl=control)
    np.testing.assert_array_equal(status.detach().cpu().numpy(), [0, 0])
    for world, data in enumerate(refs):
      data.ctrl[:] = control[world]
      mujoco.mj_step(model, data)
    compare()
    if step == 5:
      checkpoint = sim.snapshot()
      cpu_checkpoint = [{
          name: np.asarray(getattr(data, name)).copy()
          for name in ("qpos", "qvel", "act", "ctrl", "qacc_warmstart")
      } | {"time": float(data.time)} for data in refs]
  assert checkpoint is not None and cpu_checkpoint is not None
  sim.step(ctrl=np.asarray([[.61], [-.08]], dtype=np.float32))
  for world, data in enumerate(refs):
    data.ctrl[:] = [.61, -.08][world]
    mujoco.mj_step(model, data)
  compare()
  after = tuple(t.detach().cpu().clone() for t in
                (sim.state._qpos, sim.state._qvel, sim.state._qacc,
                 sim.state._act, sim._sensordata))
  sim.restore(checkpoint)
  refs = [mujoco.MjData(model) for _ in range(batch)]
  # Replay from the exact checkpoint state; controls/history are supplied in
  # the same subsequent call, while plugin integral/slew slots are restored.
  for data, saved in zip(refs, cpu_checkpoint):
    for name in ("qpos", "qvel", "act", "ctrl", "qacc_warmstart"):
      getattr(data, name)[:] = saved[name]
    data.time = saved["time"]
  for k in range(6, 12):
    control = np.asarray([[.4 + .01 * k], [-.25 + .015 * k]], dtype=np.float32)
    status = sim.step(ctrl=control)
    np.testing.assert_array_equal(status.detach().cpu().numpy(), [0, 0])
    for world, data in enumerate(refs):
      data.ctrl[:] = control[world]
      mujoco.mj_step(model, data)
  sim.step(ctrl=np.asarray([[.61], [-.08]], dtype=np.float32))
  for world, data in enumerate(refs):
    data.ctrl[:] = [.61, -.08][world]
    mujoco.mj_step(model, data)
  compare()
  for value, current in zip(after, (sim.state._qpos, sim.state._qvel,
                                   sim.state._qacc, sim.state._act,
                                   sim._sensordata)):
    torch.testing.assert_close(current.detach().cpu(), value, rtol=0, atol=0)


@pytest.mark.parametrize("dyntype", ["filter", "filterexact"])
@pytest.mark.skipif(os.environ.get("MUJOCO_METAL_RUN_GPU") != "1",
                    reason="serialized actearly/filter PID MPS trajectory")
def test_native_bundled_pid_actearly_filter_modes_match_pinned(dyntype):
  from mujoco_metal.simulation import MetalSimulation
  model = mujoco.MjModel.from_xml_string(f"""
    <mujoco><option timestep="0.002" gravity="0 0 -9.81"/>
      <extension><plugin plugin="mujoco.pid"><instance name="pid">
        <config key="kp" value="3"/><config key="ki" value="2"/>
        <config key="kd" value=".4"/><config key="imax" value=".2"/>
        <config key="slewmax" value="2"/>
      </instance></plugin></extension>
      <worldbody><body><joint name="j" type="slide" axis="0 0 1"/>
        <geom type="box" size=".1 .1 .1" mass="1"/></body></worldbody>
      <actuator><plugin joint="j" plugin="mujoco.pid" instance="pid"
        dyntype="{dyntype}" dynprm=".03" actdim="3" actearly="true"/></actuator>
    </mujoco>
  """)
  qpos = np.asarray([[.1], [-.15]], dtype=np.float32)
  qvel = np.asarray([[.2], [-.1]], dtype=np.float32)
  sim = MetalSimulation(model, batch_size=2, qpos=qpos, qvel=qvel,
                        profile="integrated_euler_v1")
  refs = [mujoco.MjData(model) for _ in range(2)]
  for i, ref in enumerate(refs):
    ref.qpos[:] = qpos[i]
    ref.qvel[:] = qvel[i]
    mujoco.mj_forward(model, ref)
  for step in range(8):
    ctrl = np.asarray([[.3 + .02 * step], [-.2 + .01 * step]], dtype=np.float32)
    status = sim.step(ctrl=ctrl)
    np.testing.assert_array_equal(status.detach().cpu().numpy(), [0, 0])
    for i, ref in enumerate(refs):
      ref.ctrl[:] = ctrl[i]
      mujoco.mj_step(model, ref)
    for label, got, expected, atol, rtol in (
        ("qpos", sim.state._qpos, [ref.qpos for ref in refs], 2e-5, 2e-4),
        ("qvel", sim.state._qvel, [ref.qvel for ref in refs], 3e-5, 3e-4),
        ("qacc", sim.state._qacc, [ref.qacc for ref in refs], 5e-4, 3e-3),
        ("act", sim.state._act, [ref.act for ref in refs], 2e-5, 2e-4),
    ):
      np.testing.assert_allclose(got.detach().cpu().numpy(),
                                 np.stack(expected).astype(np.float32),
                                 atol=atol, rtol=rtol, err_msg=label)


def test_torus_default_and_sdf_geom_mapping_are_lowered_from_plugin_instance():
  model = mujoco.MjModel.from_xml_string("""
    <mujoco>
      <extension>
        <plugin plugin="mujoco.sdf.torus">
          <instance name="torus"/>
        </plugin>
      </extension>
      <asset>
        <mesh name="torus_mesh"><plugin instance="torus"/></mesh>
      </asset>
      <worldbody>
        <geom name="torus_geom" type="sdf" mesh="torus_mesh">
          <plugin instance="torus"/>
        </geom>
      </worldbody>
    </mujoco>
  """)
  lowered = lower_bundled_plugins(model)
  assert lowered.instance(0).name == "mujoco.sdf.torus"
  assert lowered.instance(0).config == {"radius1": "0.35", "radius2": "0.15"}
  np.testing.assert_array_equal(lowered.instance_kind, [5])
  np.testing.assert_array_equal(lowered.geom_plugin_instance, [0])
  np.testing.assert_allclose(lowered.plugin_attributes, [[.35, .15, 0, 0, 0]],
                             rtol=0, atol=1e-7)
  np.testing.assert_allclose(lowered.geom_sdf_aabb,
                             [[0, 0, 0, .5, .5, .15]], rtol=0, atol=1e-7)
  np.testing.assert_array_equal(lowered.sdf_geom_id, [0])
  assert not lowered.plugin_attributes.flags.writeable
  assert lowered.plugin_attributes_low.shape == lowered.plugin_attributes.shape


def test_plugin_free_model_has_int32_sentinels_and_no_sdf_rows():
  model = mujoco.MjModel.from_xml_string("""
    <mujoco><worldbody><geom type="plane" size="1 1 .1"/></worldbody></mujoco>
  """)
  lowered = lower_bundled_plugins(model)
  assert lowered.instances == ()
  assert lowered.instance_kind.shape == (0,)
  assert lowered.plugin_attributes.shape == (0, 5)
  np.testing.assert_array_equal(lowered.geom_plugin_instance, [-1])
  assert lowered.geom_sdf_aabb.shape == (1, 6)
  assert lowered.sdf_geom_attributes.shape == (0, 5)


@pytest.mark.parametrize(("kind", "point", "attrs", "expected"), [
    (2, [0, 0, 1], [.4, 1, .02, 0, 0], 1.0754451150103321),
    (3, [0, 0, 0], [0, 2.8, 25, .2, -1], .8736),
    (5, [.5, 0, 0], [.35, .15, 0, 0, 0], 0.0),
])
def test_sdf_cpu_values_match_pinned_source_formula(kind, point, attrs, expected):
  assert sdf_distance_local(kind, point, attrs) == pytest.approx(expected, abs=2e-7)
  gradient = sdf_gradient_local(kind, point, attrs)
  assert gradient.shape == (3,)
  assert np.all(np.isfinite(gradient))
  if kind == 5:
    np.testing.assert_allclose(gradient, [1, 0, 0], rtol=0, atol=1e-7)


@pytest.mark.parametrize(("kind", "point", "attrs"), [
    (1, [.37, -.21, .18], [.26, 0, 0, 0, 0]),
    (2, [.47, .16, .52], [.4, 1, .02, 0, 0]),
    (3, [.31, -.29, .11], [0, 2.8, 25, .2, -1]),
    (4, [.27, .14, -.33], [.26, 0, 0, 0, 0]),
])
def test_source_scale_forward_sdf_normals_are_finite(kind, point, attrs):
  value = sdf_distance_local(kind, point, attrs)
  gradient = sdf_gradient_local(kind, point, attrs)
  assert np.isfinite(value)
  assert gradient.shape == (3,)
  assert np.all(np.isfinite(gradient))
  assert np.linalg.norm(gradient) > .1


def _plugin_descriptor(slot):
  """Read callback pointers from MuJoCo's registered public plugin table."""
  fields = [
      ("name", ctypes.c_char_p), ("nattribute", ctypes.c_int),
      ("attributes", ctypes.POINTER(ctypes.c_char_p)),
      ("capabilityflags", ctypes.c_int), ("needstage", ctypes.c_int),
  ]
  fields.extend((name, ctypes.c_void_p) for name in (
      "nstate", "nsensordata", "init", "destroy", "copy", "reset",
      "compute", "advance", "visualize", "actuator_act_dot",
      "sdf_distance", "sdf_gradient", "sdf_staticdistance",
      "sdf_attribute", "sdf_aabb"))
  descriptor_type = type("_RegisteredPlugin", (ctypes.Structure,),
                         {"_fields_": fields})
  library = ctypes.CDLL(mujoco._structs.__file__)
  getter = library.mjp_getPluginAtSlot
  getter.argtypes = [ctypes.c_int]
  getter.restype = ctypes.POINTER(descriptor_type)
  descriptor = getter(int(slot))
  assert descriptor
  return descriptor.contents


def _sdf_model(name, config=None):
  config_xml = "" if config is None else "".join(
      f'<config key="{key}" value="{value}"/>'
      for key, value in config.items())
  return mujoco.MjModel.from_xml_string(f"""
    <mujoco>
      <extension><plugin plugin="mujoco.sdf.{name}">
        <instance name="sdf">{config_xml}</instance>
      </plugin></extension>
      <asset><mesh name="sdfmesh"><plugin instance="sdf"/></mesh></asset>
      <worldbody><geom type="sdf" mesh="sdfmesh">
        <plugin instance="sdf"/>
      </geom></worldbody>
    </mujoco>
  """)


def _sdf_models(configs):
  names = tuple(configs)
  extension = "".join(
      f'<plugin plugin="mujoco.sdf.{name}"><instance name="p{i}">'
      + "".join(f'<config key="{key}" value="{value}"/>'
                for key, value in configs[name].items())
      + "</instance></plugin>"
      for i, name in enumerate(names))
  assets = "".join(
      f'<mesh name="m{i}"><plugin instance="p{i}"/></mesh>'
      for i in range(len(names)))
  geoms = "".join(
      f'<geom type="sdf" mesh="m{i}"><plugin instance="p{i}"/></geom>'
      for i in range(len(names)))
  return mujoco.MjModel.from_xml_string(
      f"<mujoco><extension>{extension}</extension><asset>{assets}</asset>"
      f"<worldbody>{geoms}</worldbody></mujoco>")


@pytest.mark.parametrize(("name", "kind", "points"), [
    ("bolt", 1, [[.37, -.21, .18], [.50, 0, 0], [.26, 0, -.25]]),
    ("bowl", 2, [[.47, .16, .52], [0, .4, 0], [-1, -.25, .25]]),
    ("gear", 3, [[.31, -.29, .11], [1.4, 0, .1], [.7, .7, 0]]),
    ("nut", 4, [[.27, .14, -.33], [.26, 0, -.25], [.3, 0, .125]]),
    ("torus", 5, [[.61, .07, .13], [.5, 0, 0], [0, 0, .15]]),
])
def test_sdf_distance_and_gradient_match_registered_pinned_callbacks(
    name, kind, points):
  """Compare the Python lowering against actual callbacks in pinned 3.10."""
  from mujoco_metal.bundled_plugins import lower_bundled_plugins

  model = _sdf_model(name)
  data = mujoco.MjData(model)
  mujoco.mj_forward(model, data)
  lowered = lower_bundled_plugins(model)
  plugin = _plugin_descriptor(model.plugin[0])
  distance_fn = ctypes.CFUNCTYPE(
      ctypes.c_double, ctypes.POINTER(ctypes.c_double),
      ctypes.POINTER(ctypes.c_double))(plugin.sdf_staticdistance)
  gradient_fn = ctypes.CFUNCTYPE(
      None, ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double),
      ctypes.c_void_p, ctypes.c_int)(plugin.sdf_gradient)
  attr_count = len(plugin.attributes[:plugin.nattribute])
  # SDF attributes are already numeric in the plugin instance. The public
  # model exposes source config strings; defaults are supplied by lowering.
  attrs = np.asarray(lowered.plugin_attributes[0, :attr_count], dtype=np.float64)
  c_attrs = (ctypes.c_double * attr_count)(*attrs.tolist())
  c_data = ctypes.c_void_p(data._address)
  for point in points:
    c_point = (ctypes.c_double * 3)(*point)
    c_gradient = (ctypes.c_double * 3)()
    source_distance = distance_fn(c_point, c_attrs)
    gradient_fn(c_gradient, c_point, c_data, 0)
    np.testing.assert_allclose(
        sdf_distance_local(kind, point, lowered.plugin_attributes[0]),
        source_distance, rtol=0, atol=2e-7)
    local_gradient = sdf_gradient_local(kind, point, lowered.plugin_attributes[0])
    source_gradient = np.asarray(c_gradient)
    if kind == 5 and np.hypot(*point[:2]) == 0:
      assert np.any(~np.isfinite(local_gradient))
      assert np.any(~np.isfinite(source_gradient))
    else:
      np.testing.assert_allclose(local_gradient, source_gradient, rtol=0, atol=3e-6)


@pytest.mark.parametrize(("name", "config", "points"), [
    ("bolt", {"radius": ".33"},
     [[.38, 0, .083333333333], [.38, 0, .083333334],
      [.38*np.cos(np.pi/6), .38*np.sin(np.pi/6), .12],
      [.38*np.cos(-np.pi/6), .38*np.sin(-np.pi/6), .12]]),
    ("bowl", {"height": ".32", "radius": ".9", "thickness": ".05"},
     [[-.9, -.2, .28], [-.4, .2, -.1], [.9, 0, .32]]),
    ("gear", {"alpha": ".2", "diameter": "1.9", "teeth": "17",
              "thickness": ".28", "innerdiameter": ".3"},
     [[.61, .13, .1], [.7, 0, 0], [.2, .5, .14]]),
    ("nut", {"radius": ".31"},
     [[.31, 0, .083333333333], [.31, 0, .083333334],
      [.31*np.cos(np.pi/6), .31*np.sin(np.pi/6), -.25],
      [.31*np.cos(-np.pi/6), .31*np.sin(-np.pi/6), -.25]]),
    ("torus", {"radius1": ".28", "radius2": ".11"},
     [[.28, 0, 0], [.39, 0, .05], [0, .28, .11]]),
])
def test_sdf_custom_attributes_match_registered_pinned_callbacks(name, config,
                                                                  points):
  model = _sdf_model(name, config)
  data = mujoco.MjData(model)
  mujoco.mj_forward(model, data)
  lowered = lower_bundled_plugins(model)
  plugin = _plugin_descriptor(model.plugin[0])
  distance_fn = ctypes.CFUNCTYPE(
      ctypes.c_double, ctypes.POINTER(ctypes.c_double), ctypes.c_void_p,
      ctypes.c_int)(plugin.sdf_distance)
  gradient_fn = ctypes.CFUNCTYPE(
      None, ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double),
      ctypes.c_void_p, ctypes.c_int)(plugin.sdf_gradient)
  attr_names = tuple(plugin.attributes[i].decode("utf-8")
                     for i in range(plugin.nattribute))
  attrs = np.zeros((5,), dtype=np.float64)
  attrs[:len(attr_names)] = [float(lowered.instances[0].config[key])
                             for key in attr_names]
  for point in points:
    p = np.asarray(point, dtype=np.float64)
    c_point = (ctypes.c_double * 3)(*p.tolist())
    source_distance = distance_fn(c_point, ctypes.c_void_p(data._address), 0)
    c_gradient = (ctypes.c_double * 3)()
    gradient_fn(c_gradient, c_point, ctypes.c_void_p(data._address), 0)
    np.testing.assert_allclose(sdf_distance_local(
        _KIND_BY_NAME[name], p, attrs), source_distance, rtol=0, atol=3e-6)
    source_gradient = np.asarray(c_gradient)
    if np.all(np.isfinite(source_gradient)):
        np.testing.assert_allclose(sdf_gradient_local(
            _KIND_BY_NAME[name], p, attrs), source_gradient, rtol=0, atol=3e-6)


def test_combined_custom_sdf_instances_lower_independently():
  configs = {
      "bolt": {"radius": ".33"},
      "bowl": {"height": ".32", "radius": ".9", "thickness": ".05"},
      "gear": {"alpha": ".2", "diameter": "1.9", "teeth": "17",
               "thickness": ".28", "innerdiameter": ".3"},
      "nut": {"radius": ".31"},
      "torus": {"radius1": ".28", "radius2": ".11"},
  }
  model = _sdf_models(configs)
  lowered = lower_bundled_plugins(model)
  assert tuple(instance.name for instance in lowered.instances) == tuple(
      f"mujoco.sdf.{name}" for name in configs)
  np.testing.assert_array_equal(lowered.sdf_geom_instance, np.arange(5))
  for i, (name, config) in enumerate(configs.items()):
    for key, expected in config.items():
      assert lowered.instances[i].config[key] == expected
  assert np.all(np.isfinite(lowered.sdf_geom_attributes))


def test_sdf_attribute_high_low_pair_preserves_source_decimal_values():
  configs = {"torus": {"radius1": ".28", "radius2": ".11"}}
  model = _sdf_models(configs)
  lowered = lower_bundled_plugins(model)
  descriptor = lowered.instances[0]
  exact = np.asarray([float(descriptor.config[key])
                      for key in descriptor.attribute_names], dtype=np.float64)
  pair = (lowered.plugin_attributes[0, :len(exact)].astype(np.float64) +
          lowered.plugin_attributes_low[0, :len(exact)].astype(np.float64))
  np.testing.assert_allclose(pair, exact, rtol=0, atol=2e-17)
  np.testing.assert_array_equal(lowered.sdf_geom_attributes[0],
                                lowered.plugin_attributes[0])
  np.testing.assert_array_equal(lowered.sdf_geom_attributes_low[0],
                                lowered.plugin_attributes_low[0])


def test_rigid_sdf_high_low_attributes_match_all_pinned_plugin_configs():
  configs = {
      "bolt": {"radius": ".3333333333333333"},
      "bowl": {"height": ".3212345678901234", "radius": ".9123456789012345",
               "thickness": ".0543210987654321"},
      "gear": {"alpha": ".2012345678901234", "diameter": "1.9123456789012345",
               "teeth": "17", "thickness": ".2812345678901234",
               "innerdiameter": ".3123456789012345"},
      "nut": {"radius": ".3123456789012345"},
      "torus": {"radius1": ".2812345678901234",
                "radius2": ".1123456789012345"},
  }
  model = _sdf_models(configs)
  lowered = lower_bundled_plugins(model)
  for row, (name, config) in enumerate(configs.items()):
    descriptor = lowered.instances[row]
    exact = np.zeros((5,), dtype=np.float64)
    for j, key in enumerate(descriptor.attribute_names):
      if key in config:
        exact[j] = float(config[key])
    packed = (lowered.plugin_attributes[row].astype(np.float64) +
              lowered.plugin_attributes_low[row].astype(np.float64))
    # The residual is itself binary32, so the paired representation has a
    # small final-rounding error at roughly one binary64 ulp.
    np.testing.assert_allclose(packed, exact, rtol=0, atol=3e-15,
                               err_msg=name)
    assert np.any(lowered.plugin_attributes_low[row] != 0), name


@pytest.mark.skipif(os.getenv("MUJOCO_METAL_RUN_GPU") != "1",
                    reason="explicit serialized MPS qualification")
def test_native_sdf_query_matches_pinned_callbacks_for_all_plugin_classes():
  torch = pytest.importorskip("torch")
  from mujoco_metal.bundled_plugins import lower_bundled_plugins
  from mujoco_metal.plugin_sdf import MetalPluginSDFQuery

  names = ("bolt", "bowl", "gear", "nut", "torus")
  # Each SDF sees smooth and branch-boundary points.  The bowl corner and
  # bolt/nut thread/head creases cover the callback mismatches found by the
  # first native oracle.  The final slot is a neutral no-plugin candidate.
  points = np.asarray([
      [.37, -.21, .18], [.50, 0.0, 0.0], [.001, 0.0, .1], [.26, 0, -.25],
      [.47, .16, .52], [.50, 0.0, 0.0], [.001, 0.0, .1], [-1, -.25, .25],
      [.31, -.29, .11], [.50, 0.0, 0.0], [.001, 0.0, .1], [1.4, 0, .1],
      [.27, .14, -.33], [.50, 0.0, 0.0], [.001, 0.0, .1], [.26, 0, -.25],
      [.61, .07, .13], [.50, 0.0, 0.0], [.001, 0.0, .1], [0, 0, .15],
      [.1, .2, .3],
  ], dtype=np.float32)
  extension = "".join(
      f'<plugin plugin="mujoco.sdf.{name}"><instance name="p{i}"/></plugin>'
      for i, name in enumerate(names))
  assets = "".join(
      f'<mesh name="m{i}"><plugin instance="p{i}"/></mesh>'
      for i in range(len(names)))
  geoms = "".join(
      f'<geom type="sdf" mesh="m{i}"><plugin instance="p{i}"/></geom>'
      for i in range(len(names)))
  model = mujoco.MjModel.from_xml_string(
      f"<mujoco><extension>{extension}</extension><asset>{assets}</asset>"
      f"<worldbody>{geoms}</worldbody></mujoco>")
  descriptor = lower_bundled_plugins(model)
  data = mujoco.MjData(model)
  mujoco.mj_forward(model, data)
  query = MetalPluginSDFQuery(descriptor, batch_size=1, candidate_capacity=21)
  device = torch.device("mps")
  device_points = torch.as_tensor(points[None], dtype=torch.float32, device=device)
  device_instances = torch.as_tensor(
      [[0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2,
        3, 3, 3, 3, 4, 4, 4, 4, -1]],
      dtype=torch.int32, device=device)
  result = query.run_device(device_points, device_instances)
  assert result["distance"].device.type == "mps"
  distances = result["distance"].cpu().numpy()[0]
  gradients = result["gradient"].cpu().numpy()[0]
  status = result["status"].cpu().numpy()[0]
  expected_distance = []
  expected_gradient = []
  for instance in range(5):
    plugin = _plugin_descriptor(model.plugin[instance])
    distance_fn = ctypes.CFUNCTYPE(
        ctypes.c_double, ctypes.POINTER(ctypes.c_double), ctypes.c_void_p,
        ctypes.c_int)(plugin.sdf_distance)
    gradient_fn = ctypes.CFUNCTYPE(
        None, ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double),
        ctypes.c_void_p, ctypes.c_int)(plugin.sdf_gradient)
    for candidate in range(4):
      point = points[4 * instance + candidate]
      c_point = (ctypes.c_double * 3)(*(float(x) for x in point))
      expected_distance.append(distance_fn(c_point, ctypes.c_void_p(data._address),
                                           instance))
      c_gradient = (ctypes.c_double * 3)()
      gradient_fn(c_gradient, c_point, ctypes.c_void_p(data._address), instance)
      expected_gradient.append(tuple(c_gradient))
  expected_distance = np.asarray(expected_distance, dtype=np.float32)
  expected_gradient = np.asarray(expected_gradient, dtype=np.float32)
  expected_status = np.zeros((21,), dtype=np.int32)
  # The pinned torus callback divides by hypot(x,y); its axis gradient is
  # undefined even though distance is finite. The device contract reports
  # that nonfinite normal with status=1 and zeroed outputs.
  expected_status[19] = 1
  np.testing.assert_array_equal(status, expected_status)
  valid_gradient_slots = [i for i in range(20) if i != 19]
  assert np.any(~np.isfinite(expected_gradient[19]))
  np.testing.assert_allclose(distances[valid_gradient_slots],
                             expected_distance[valid_gradient_slots],
                             rtol=0, atol=3e-6)
  np.testing.assert_allclose(gradients[valid_gradient_slots],
                             expected_gradient[valid_gradient_slots],
                             rtol=0, atol=3e-5)
  np.testing.assert_array_equal(distances[19], 0)
  np.testing.assert_array_equal(gradients[19], [0, 0, 0])
  np.testing.assert_array_equal(distances[20:], [0])
  np.testing.assert_array_equal(gradients[20:], [[0, 0, 0]])


@pytest.mark.skipif(os.getenv("MUJOCO_METAL_RUN_GPU") != "1",
                    reason="explicit serialized MPS qualification")
def test_native_sdf_custom_attributes_and_periodic_creases_match_callbacks():
  torch = pytest.importorskip("torch")
  from mujoco_metal.bundled_plugins import lower_bundled_plugins
  from mujoco_metal.plugin_sdf import MetalPluginSDFQuery

  configs = {
      "bolt": {"radius": ".33"},
      "bowl": {"height": ".32", "radius": ".9", "thickness": ".05"},
      "gear": {"alpha": ".2", "diameter": "1.9", "teeth": "17",
               "thickness": ".28", "innerdiameter": ".3"},
      "nut": {"radius": ".31"},
      "torus": {"radius1": ".28", "radius2": ".11"},
  }
  names = tuple(configs)
  points = np.asarray([
      [.38, 0, .083333333333], [.38, 0, .083333334],
      [.38*np.cos(np.pi/6), .38*np.sin(np.pi/6), .12],
      [.38*np.cos(-np.pi/6), .38*np.sin(-np.pi/6), .12],
      [-.9, -.2, .28], [-.4, .2, -.1], [.9, 0, .32], [-1, -.25, .25],
      [.61, .13, .1], [.7, 0, 0], [.2, .5, .14], [1.4, 0, .1],
      [.31, 0, .083333333333], [.31, 0, .083333334],
      [.31*np.cos(np.pi/6), .31*np.sin(np.pi/6), -.25],
      [.31*np.cos(-np.pi/6), .31*np.sin(-np.pi/6), -.25],
      [.28, 0, 0], [.39, 0, .05], [0, .28, .11], [0, .28, .11],
      [.1, .2, .3],
  ], dtype=np.float32)
  points = np.stack((points, points.copy()))
  # Only world one has the torus-axis singularity; its other candidates must
  # still produce valid results and match the same pinned CPU rows.
  points[1, 19] = [0, 0, .11]
  instances = np.asarray([[i//4 for i in range(20)] + [-1]]*2,
                         dtype=np.int32)
  model = _sdf_models(configs)
  data = mujoco.MjData(model)
  mujoco.mj_forward(model, data)
  descriptor = lower_bundled_plugins(model)
  query = MetalPluginSDFQuery(descriptor, batch_size=2,
                              candidate_capacity=21)
  result = query.run_device(
      torch.as_tensor(points, dtype=torch.float32, device="mps"),
      torch.as_tensor(instances, dtype=torch.int32, device="mps"))
  native_distance = result["distance"].cpu().numpy()
  native_gradient = result["gradient"].cpu().numpy()
  native_status = result["status"].cpu().numpy()
  expected_distance = np.zeros((2, 21), dtype=np.float32)
  expected_gradient = np.zeros((2, 21, 3), dtype=np.float32)
  expected_status = np.zeros((2, 21), dtype=np.int32)
  for world in range(2):
    for slot in range(20):
      instance = int(instances[world, slot])
      plugin = _plugin_descriptor(model.plugin[instance])
      point = points[world, slot]
      c_point = (ctypes.c_double * 3)(*(float(x) for x in point))
      distance_fn = ctypes.CFUNCTYPE(
          ctypes.c_double, ctypes.POINTER(ctypes.c_double), ctypes.c_void_p,
          ctypes.c_int)(plugin.sdf_distance)
      gradient_fn = ctypes.CFUNCTYPE(
          None, ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double),
          ctypes.c_void_p, ctypes.c_int)(plugin.sdf_gradient)
      expected_distance[world, slot] = distance_fn(
          c_point, ctypes.c_void_p(data._address), instance)
      grad = (ctypes.c_double * 3)()
      gradient_fn(grad, c_point, ctypes.c_void_p(data._address), instance)
      expected_gradient[world, slot] = np.asarray(grad, dtype=np.float32)
      if not np.all(np.isfinite(grad)):
        expected_status[world, slot] = 1
        expected_distance[world, slot] = 0
        expected_gradient[world, slot] = 0
  np.testing.assert_array_equal(native_status, expected_status)
  valid = expected_status == 0
  np.testing.assert_allclose(native_distance[valid], expected_distance[valid],
                             rtol=0, atol=3e-6)
  np.testing.assert_allclose(native_gradient[valid], expected_gradient[valid],
                             rtol=0, atol=3e-5)
  np.testing.assert_array_equal(native_distance[~valid], 0)
  np.testing.assert_array_equal(native_gradient[~valid], 0)


@pytest.mark.skipif(os.getenv("MUJOCO_METAL_RUN_GPU") != "1",
                    reason="explicit serialized MPS qualification")
def test_native_sdf_signed_zero_axes_match_pinned_callbacks():
  """Keep atan2 signed-zero quadrants identical to the registered callbacks."""
  torch = pytest.importorskip("torch")
  from mujoco_metal.plugin_sdf import MetalPluginSDFQuery

  configs = {
      "bolt": {"radius": ".33"},
      "nut": {"radius": ".31"},
      "gear": {"alpha": ".2", "diameter": "1.9", "teeth": "17",
               "thickness": ".28", "innerdiameter": ".3"},
  }
  names = tuple(configs)
  signed_zeros = ((0.0, 0.0), (0.0, -0.0), (-0.0, 0.0), (-0.0, -0.0))
  points = []
  for _ in names:
    # The first four points distinguish atan2(+/-0, +/-x) quadrants. The
    # remaining four distinguish signed-zero behavior at the origin itself.
    points.extend((x, y, .13) for x, y in
                  ((.31, 0.0), (.31, -0.0), (-.31, 0.0), (-.31, -0.0)))
    points.extend((x, y, z) for x, y in signed_zeros for z in (0.0,))
  points = np.asarray(points, dtype=np.float32)
  instances = np.repeat(np.arange(len(names), dtype=np.int32), 8)[None, :]

  model = _sdf_models(configs)
  data = mujoco.MjData(model)
  mujoco.mj_forward(model, data)
  descriptor = lower_bundled_plugins(model)
  query = MetalPluginSDFQuery(descriptor, batch_size=1,
                              candidate_capacity=len(points))
  result = query.run_device(
      torch.as_tensor(points[None], dtype=torch.float32, device="mps"),
      torch.as_tensor(instances, dtype=torch.int32, device="mps"))
  actual_distance = result["distance"].cpu().numpy()[0]
  actual_gradient = result["gradient"].cpu().numpy()[0]
  actual_status = result["status"].cpu().numpy()[0]
  expected_distance = np.zeros((len(points),), dtype=np.float32)
  expected_gradient = np.zeros((len(points), 3), dtype=np.float32)
  expected_status = np.zeros((len(points),), dtype=np.int32)

  for slot, point in enumerate(points):
    instance = int(instances[0, slot])
    plugin = _plugin_descriptor(model.plugin[instance])
    c_point = (ctypes.c_double * 3)(*(float(x) for x in point))
    distance_fn = ctypes.CFUNCTYPE(
        ctypes.c_double, ctypes.POINTER(ctypes.c_double), ctypes.c_void_p,
        ctypes.c_int)(plugin.sdf_distance)
    gradient_fn = ctypes.CFUNCTYPE(
        None, ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double),
        ctypes.c_void_p, ctypes.c_int)(plugin.sdf_gradient)
    expected_distance[slot] = distance_fn(
        c_point, ctypes.c_void_p(data._address), instance)
    gradient = (ctypes.c_double * 3)()
    gradient_fn(gradient, c_point, ctypes.c_void_p(data._address), instance)
    expected_gradient[slot] = np.asarray(gradient, dtype=np.float32)
    if not np.all(np.isfinite(gradient)):
      expected_status[slot] = 1
      expected_distance[slot] = 0
      expected_gradient[slot] = 0

  np.testing.assert_array_equal(actual_status, expected_status)
  valid = expected_status == 0
  np.testing.assert_allclose(actual_distance[valid], expected_distance[valid],
                             rtol=0, atol=3e-6)
  np.testing.assert_allclose(actual_gradient[valid], expected_gradient[valid],
                             rtol=0, atol=3e-5)
  np.testing.assert_array_equal(actual_distance[~valid], 0)
  np.testing.assert_array_equal(actual_gradient[~valid], 0)
