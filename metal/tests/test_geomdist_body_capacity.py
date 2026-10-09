# Copyright 2026 The MuJoCo Metal contributors
# Licensed under the Apache License, Version 2.0.
"""The closest geometry may occur beyond either body's former local cap."""
import os

import mujoco
import numpy as np
import pytest


def _fixture():
  geoms = ''.join(f'<geom type="sphere" size=".1" pos="{10+i} 0 0"/>'
                   for i in range(39))
  return mujoco.MjModel.from_xml_string(f'''<mujoco>
    <default><geom contype="0" conaffinity="0"/></default>
    <worldbody><body name="many"><freejoint/>{geoms}
      <geom name="closest" type="sphere" size=".1" pos=".5 0 0"/>
    </body><geom name="anchor" type="sphere" size=".1"/></worldbody>
    <sensor>
      <distance body1="many" geom2="anchor" cutoff="100"/>
      <distance geom1="anchor" body2="many" cutoff="100"/>
      <normal body1="many" geom2="anchor" cutoff="100"/>
      <fromto geom1="anchor" body2="many" cutoff="100"/>
    </sensor></mujoco>''')


def test_cpu_body_distance_witness_is_after_former_capacity():
  model = _fixture()
  data = mujoco.MjData(model)
  mujoco.mj_forward(model, data)
  body = model.body("many").id
  assert int(model.geom("closest").id) - int(model.body_geomadr[body]) == 39
  np.testing.assert_allclose(data.sensordata[:2], .3, atol=1e-14)


def test_native_body_distance_uses_complete_ranges_in_both_orders():
  if os.environ.get("MUJOCO_METAL_RUN_GPU") != "1":
    pytest.skip("explicit native qualification required")
  import torch
  from mujoco_metal.sensors import SensorProgram
  model = _fixture()
  program = SensorProgram(model, batch_size=2)
  positions, expected = [], []
  for displacement in (0.0, .125):
    data = mujoco.MjData(model)
    data.qpos[0] = displacement
    mujoco.mj_forward(model, data)
    positions.append(data.geom_xpos.copy())
    expected.append(data.sensordata.copy())
  tensor = lambda x: torch.tensor(np.asarray(x), dtype=torch.float32, device="mps")
  poses = {"geom_pos": tensor(positions),
           "geom_quat": tensor(np.tile([1., 0., 0., 0.], (2, model.ngeom, 1)))}
  actual = program.run_geomdist_device(poses, program._sp_hull, program._sp_hull_info)
  np.testing.assert_allclose(actual.cpu().numpy(), expected, rtol=2e-6, atol=2e-6)
  # An explicitly unselected world retains its prior output even when poses
  # change; selection never substitutes an abbreviated geometry list.
  previous = actual.clone()
  poses["geom_pos"][0, model.geom("closest").id, 0].add_(.25)
  actual = program.run_geomdist_device(poses, program._sp_hull, program._sp_hull_info,
                                       world_mask=torch.tensor([False, True], device="mps"))
  torch.testing.assert_close(actual, previous, rtol=0, atol=0)
  unmasked = program.run_geomdist_device(poses, program._sp_hull, program._sp_hull_info)
  assert abs(float(unmasked[0, 0]) - float(previous[0, 0])) > .2
