# Copyright 2026 The MuJoCo Metal contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0

"""CPU oracle and opt-in device checks for the bounded sensor stage."""

import os
from pathlib import Path
import re

import mujoco
import numpy as np
import pytest

from mujoco_metal.model import load_model
from mujoco_metal.sensors import SensorProgram
from mujoco_metal.sensors import lower_sensors
from mujoco_metal.sensors import sensor_oracle
from mujoco_metal.sensors import _KERNEL_FIXED_BUFFER_COUNT
from mujoco_metal.sensors import _KERNEL_METADATA

_XML = """<mujoco><worldbody>
  <body name="hinge" pos=".2 .1 .3"><joint name="h" type="hinge" axis="0 0 1"/>
    <geom name="g" type="sphere" size=".1" mass="1"/><site name="s" pos=".1 0 0"/>
  </body>
  <body name="slide" pos="-.3 .2 .1"><joint name="sl" type="slide" axis="1 0 0"/>
    <geom type="sphere" size=".1" mass="1"/><site name="r" pos="0 .2 0" quat=".9238795 0 0 .3826834"/>
  </body>
  <body name="ball" pos="0 -.2 .2"><joint name="b" type="ball"/>
    <geom type="sphere" size=".1" mass="1"/>
  </body>
</worldbody><sensor>
  <jointpos joint="h"/><jointvel joint="h"/><ballquat joint="b"/><ballangvel joint="b"/>
  <framepos objtype="body" objname="hinge"/>
  <framequat objtype="site" objname="s" reftype="site" refname="r"/>
  <framexaxis objtype="site" objname="s" reftype="site" refname="r"/>
  <frameyaxis objtype="site" objname="s"/>
  <framezaxis objtype="geom" objname="g"/>
  <framelinvel objtype="site" objname="s" reftype="site" refname="r"/>
  <frameangvel objtype="site" objname="s"/>
  <gyro site="s"/><velocimeter site="s"/>
  <clock/>
</sensor></mujoco>"""


def _input_batch(model, descriptor, batch=3):
  qpos = np.tile(model.qpos0, (batch, 1))
  qvel = np.zeros((batch, model.nv))
  for w in range(batch):
    qpos[w, descriptor.jnt_qposadr[0]] += .17 * (w + 1)
    qpos[w, descriptor.jnt_qposadr[1]] -= .08 * w
    qa = int(descriptor.jnt_qposadr[2])
    angle = .23 * (w + 1)
    qpos[w, qa:qa+4] = [np.cos(angle/2), 0, np.sin(angle/2), 0]
    qvel[w] = np.linspace(-.3, .4, model.nv) + .1 * w
  return qpos, qvel, .4 + np.arange(batch) * .85


def test_sensor_cpu_oracle_matches_mujoco_310_for_pose_and_velocity_families():
  model = mujoco.MjModel.from_xml_string(_XML)
  descriptor = load_model(model)
  qpos, qvel, times = _input_batch(model, descriptor)
  poses = {key: [] for key in ("body_pos", "body_quat", "geom_pos", "geom_quat", "site_pos", "site_quat", "inertial_pos", "inertial_quat", "cvel", "root_com")}
  expected = []
  for world in range(len(qpos)):
    data = mujoco.MjData(model)
    data.qpos[:] = qpos[world]
    data.qvel[:] = qvel[world]
    data.time = times[world]
    mujoco.mj_forward(model, data)
    expected.append(data.sensordata.copy())
    fk = descriptor.forward_kinematics(qpos[world])
    for key in ("body_pos", "body_quat", "geom_pos", "geom_quat", "site_pos", "site_quat", "inertial_pos", "inertial_quat"):
      poses[key].append(fk[key])
    poses["cvel"].append(data.cvel.copy())
    poses["root_com"].append(data.subtree_com.copy())
  poses = {key: np.asarray(value) for key, value in poses.items()}
  actual = sensor_oracle(model, qpos, qvel, times, poses)
  # Descriptor constants intentionally match the backend's float32 device ABI.
  np.testing.assert_allclose(actual, expected, rtol=3e-8, atol=2e-8)


def test_sensor_stage_filter_preserves_prior_other_stage_and_disable_is_noop():
  model = mujoco.MjModel.from_xml_string(_XML)
  descriptor = load_model(model)
  qpos, qvel, times = _input_batch(model, descriptor, batch=1)
  times = times[:1]
  poses = {key: np.asarray([value]) for key, value in descriptor.forward_kinematics(qpos[0]).items() if key in ("body_pos", "body_quat", "geom_pos", "geom_quat", "site_pos", "site_quat", "inertial_pos", "inertial_quat")}
  poses["cvel"] = np.zeros((1, model.nbody, 6))
  poses["root_com"] = np.zeros((1, model.nbody, 3))
  initial = np.full((1, model.nsensordata), 17.0)
  pos = sensor_oracle(model, qpos, qvel, times, poses, initial, stages=(int(mujoco.mjtStage.mjSTAGE_POS),))
  vel_cols = np.concatenate([np.arange(a, a+d) for a, d, stage in zip(model.sensor_adr, model.sensor_dim, model.sensor_needstage) if stage == int(mujoco.mjtStage.mjSTAGE_VEL)])
  np.testing.assert_array_equal(pos[0, vel_cols], initial[0, vel_cols])
  model.opt.disableflags |= int(mujoco.mjtDisableBit.mjDSBL_SENSOR)
  unchanged = sensor_oracle(model, qpos, qvel, times, poses, initial)
  np.testing.assert_array_equal(unchanged, initial)


def test_sensor_lowering_accepts_acceleration_and_valid_compiled_delay():
  accel = mujoco.MjModel.from_xml_string("""<mujoco><worldbody><body><joint/><geom type="sphere" size=".1" mass="1"/><site name="s"/></body></worldbody><sensor><accelerometer site="s"/></sensor></mujoco>""")
  lower_sensors(accel)
  delayed = mujoco.MjModel.from_xml_string("""<mujoco><worldbody><body><joint name="j"/><geom type="sphere" size=".1" mass="1"/></body></worldbody><sensor><jointpos joint="j" delay=".01" nsample="2"/></sensor></mujoco>""")
  lower_sensors(delayed)
  delayed.sensor_history[0, 0] = 100
  with pytest.raises(ValueError, match="recompiling"):
    lower_sensors(delayed)


def test_sensor_shader_buffer_abi_is_contiguous_and_matches_host_arguments():
  import re as _re
  base = Path(__file__).parents[1].joinpath("mujoco_metal", "shaders")
  source = base.joinpath("sensors.metal").read_text()
  kernels = _re.split(r"(?=^kernel void )", source, flags=_re.M)
  kernels = [k for k in kernels if "[[buffer(" in k]
  assert len(kernels) == 3
  for block in kernels:
    indices = [int(x) for x in _re.findall(r"\[\[buffer\((\d+)\)\]\]", block)]
    assert indices == list(range(len(indices)))
    assert len(indices) <= 31
  assert len(_re.findall(r"\[\[buffer\((\d+)\)\]\]", kernels[0])) == _KERNEL_FIXED_BUFFER_COUNT + len(_KERNEL_METADATA)
  for name in ("sensors_rne.metal", "sensors_spatial.metal"):
    src = base.joinpath(name).read_text()
    blocks = [k for k in _re.split(r"(?=^kernel void )", src, flags=_re.M) if "[[buffer(" in k]
    assert blocks, name
    for block in blocks:
      indices = [int(x) for x in _re.findall(r"\[\[buffer\((\d+)\)\]\]", block)]
      assert indices == list(range(len(indices))), (name, indices)
      assert len(indices) <= 31, (name, len(indices))


def test_sensor_oracle_accepts_empty_world_without_sensor_buffers():
  model = mujoco.MjModel.from_xml_string("<mujoco><worldbody/></mujoco>")
  actual = sensor_oracle(
      model, np.zeros((1, 0)), np.zeros((1, 0)), np.zeros(1), {}
  )
  assert actual.shape == (1, 0)


@pytest.mark.gpu
@pytest.mark.skipif(
    os.environ.get("MUJOCO_METAL_RUN_GPU") != "1",
    reason="requires explicit MUJOCO_METAL_RUN_GPU=1 and idle GPU",
)
def test_device_sensor_output_matches_mujoco_forward():
  import torch
  from mujoco_metal.metal_kinematics import MetalKinematics

  model = mujoco.MjModel.from_xml_string(_XML)
  descriptor = load_model(model)
  batch = 2
  qpos, qvel, times = _input_batch(model, descriptor, batch=batch)
  fk = MetalKinematics(descriptor, batch_size=batch)
  poses = fk.run_device(torch.as_tensor(qpos, dtype=torch.float32, device="mps"))
  sensor = SensorProgram(model, batch_size=batch)
  qpos_t = torch.as_tensor(qpos, dtype=torch.float32, device="mps")
  qvel_t = torch.as_tensor(qvel, dtype=torch.float32, device="mps")
  time_t = torch.as_tensor(times, dtype=torch.float32, device="mps")
  expected = []
  cvel = []
  root_com = []
  for row in range(batch):
    data = mujoco.MjData(model)
    data.qpos[:] = qpos[row]
    data.qvel[:] = qvel[row]
    data.time = times[row]
    mujoco.mj_forward(model, data)
    expected.append(data.sensordata.copy())
    cvel.append(data.cvel.copy())
    root_com.append(data.subtree_com.copy())
  poses["cvel"] = torch.as_tensor(np.asarray(cvel), dtype=torch.float32, device="mps")
  poses["root_com"] = torch.as_tensor(np.asarray(root_com), dtype=torch.float32, device="mps")
  output = sensor.run_device(qpos_t, qvel_t, time_t, poses)
  np.testing.assert_allclose(output.cpu().numpy(), expected, rtol=3e-5, atol=3e-6)
