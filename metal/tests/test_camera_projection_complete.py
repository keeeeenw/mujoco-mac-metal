# Copyright 2026 The MuJoCo Metal contributors
# Licensed under the Apache License, Version 2.0.
"""Multiple camera projection records, world rotations and all frame modes."""
import os

import mujoco
import numpy as np
import pytest

from test_sensor_spatial_016 import _feed


def _model():
  modes = ("fixed", "track", "trackcom", "targetbody", "targetbodycom")
  cameras, sensors = [], []
  for mode in modes:
    target = 'target="target"' if mode.startswith("target") else ""
    for lens, params in (("fovy", 'fovy="58"'),
                         ("intrinsic", 'sensorsize=".036 .024" focal=".023 .021" principal=".01 .008"'),
                         ("ortho", 'projection="orthographic" fovy="1.8"')):
      name = mode+"_"+lens
      cameras.append(f'<camera name="{name}" pos=".03 -.01 .07" euler="10 -8 12" '
                     f'resolution="640 480" mode="{mode}" {target} {params}/>')
      sensors.append(f'<camprojection site="point" camera="{name}"/>')
  return mujoco.MjModel.from_xml_string(f'''<mujoco><compiler angle="degree"/>
    <option gravity="0 0 0"><flag contact="disable"/></option>
    <worldbody>
      <body name="mount" pos=".2 -.15 2" euler="8 16 27"><freejoint/>
        <geom type="box" size=".03 .04 .05" mass="1"/>{''.join(cameras)}
        <body pos=".12 .06 .04"><joint type="hinge"/>
          <geom type="capsule" size=".03 .1" pos=".12 0 .03" mass=".8"/>
        </body>
      </body>
      <body name="target" pos=".3 .2 .3"><freejoint/>
        <geom type="sphere" size=".05" mass=".7"/>
        <site name="point" pos=".09 -.06 .04"/>
        <body pos=".1 .07 .09"><geom type="box" size=".03 .07 .04" mass=".6"/></body>
      </body>
    </worldbody><sensor>{''.join(sensors)}</sensor></mujoco>''')


def _states(model):
  qp = np.stack([model.qpos0]*2).astype(np.float32)
  qp[1, :3] += [.04, -.03, .05]
  rotation = np.array([.97, .1, -.2, .1])
  qp[1, 3:7] = rotation/np.linalg.norm(rotation)
  qp[1, 7] = .23
  qp[1, 8:11] += [-.02, .06, .04]
  return qp, np.zeros((2, model.nv), np.float32), np.zeros(2)


def test_host_projection_uses_pinned_global_camera_frames_for_every_mode():
  from mujoco_metal.sensors import sensor_oracle
  model = _model()
  qp, qv, times = _states(model)
  poses, extra, expected = _feed(model, qp, qv, times)
  actual = sensor_oracle(model, qp, qv, times, poses, extra=extra)
  np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-6)
  assert np.max(np.abs(expected[0]-expected[1])) > 1
  assert np.max(np.abs(expected[:, :6]-expected[:, 6:12])) > 1


@pytest.mark.skipif(os.getenv("MUJOCO_METAL_RUN_GPU") != "1", reason="native opt-in")
def test_native_projection_all_camera_records_modes_rotations_and_world_masks():
  import torch
  from mujoco_metal.sensors import SensorProgram
  from mujoco_metal.capacity import _runtime_buffer_sizes
  model = _model()
  qp, qv, times = _states(model)
  poses, extra, expected = _feed(model, qp, qv, times)
  tensor = lambda value: torch.tensor(np.asarray(value).copy(), dtype=torch.float32, device="mps")
  program = SensorProgram(model, batch_size=2)
  gpu_poses = {name: tensor(value) for name, value in poses.items()}
  actual = program.run_state_device(tensor(qp), tensor(qv), gpu_poses)
  np.testing.assert_allclose(actual.cpu(), expected, rtol=3e-5, atol=2e-4)
  assert program._s_econst.numel() == 11+model.nq+3*model.njnt+34*model.ncam
  assert dict(_runtime_buffer_sizes(model, 2))["model.sensor.econst"] == program._s_econst.numel()
  previous = actual.clone()
  gpu_poses["site_pos"][0].add_(.1)
  selected = program.run_state_device(tensor(qp), tensor(qv), gpu_poses,
      world_mask=torch.tensor([False, True], device="mps"))
  torch.testing.assert_close(selected, previous, rtol=0, atol=0)
  changed = program.run_state_device(tensor(qp), tensor(qv), gpu_poses)
  assert float(torch.max(torch.abs(changed[0]-previous[0])).cpu()) > 1


@pytest.mark.skipif(os.getenv("MUJOCO_METAL_RUN_GPU") != "1", reason="native opt-in")
def test_public_camera_projection_stored_steps_and_checkpoint_replay():
  from mujoco_metal import MetalSimulation
  model = _model()
  qp, qv, times = _states(model)
  qv[:, 0] = .03
  qv[:, 7] = -.02
  sim = MetalSimulation(model, batch_size=2, qpos=qp, qvel=qv,
                        profile="integrated_euler_v1")
  refs = [mujoco.MjData(model) for _ in range(2)]
  for data, p, v in zip(refs, qp, qv):
    data.qpos[:] = p
    data.qvel[:] = v
  before = sim.snapshot()
  results = []
  for _ in range(8):
    assert sim.step().cpu().tolist() == [0, 0]
    for data in refs:
      mujoco.mj_step(model, data)
    results.append(sim.step_sensordata().copy())
    np.testing.assert_allclose(results[-1], [d.sensordata for d in refs],
                                rtol=3e-5, atol=2e-4)
  sim.restore(before)
  for expected in results:
    assert sim.step().cpu().tolist() == [0, 0]
    np.testing.assert_array_equal(sim.step_sensordata(), expected)
