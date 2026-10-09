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


@pytest.mark.gpu
@pytest.mark.skipif(os.getenv("MUJOCO_METAL_RUN_GPU") != "1", reason="native opt-in")
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


@pytest.mark.parametrize("native", [False, True])
@pytest.mark.gpu
@pytest.mark.skipif(os.getenv("MUJOCO_METAL_RUN_GPU") != "1", reason="native opt-in")
def test_fromto_search_cutoff_does_not_clip_world_coordinates(native):
  if native and os.environ.get("MUJOCO_METAL_RUN_GPU") != "1":
    pytest.skip("explicit native qualification required")
  from mujoco_metal.sensors import SensorProgram, sensor_oracle
  from test_sensor_spatial_016 import _feed
  model = mujoco.MjModel.from_xml_string('''<mujoco>
    <worldbody>
      <geom name="a" type="sphere" pos="100 0 0" size=".1"/>
      <geom name="b" type="sphere" pos="100.5 0 0" size=".1"/>
    </worldbody>
    <sensor><fromto geom1="a" geom2="b" cutoff="1"/>
      <fromto geom1="b" geom2="a" cutoff="1"/></sensor>
  </mujoco>''')
  poses, extra, expected = _feed(model, np.empty((2, 0)),
                                np.empty((2, 0)), np.zeros(2))
  assert np.min(expected[:, [0, 3, 6, 9]]) > 100
  if native:
    import torch
    program = SensorProgram(model, batch_size=2)
    gpu_poses = {name: torch.tensor(np.asarray(value).copy(), dtype=torch.float32,
                                   device="mps") for name, value in poses.items()}
    actual = program.run_geomdist_device(gpu_poses, program._sp_hull,
                                         program._sp_hull_info).cpu().numpy()
    np.testing.assert_allclose(actual, expected, rtol=2e-6, atol=2e-6)
  else:
    actual = sensor_oracle(model, np.empty((2, 0)), np.empty((2, 0)),
                           np.zeros(2), poses, extra=extra)
    np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)



def _overlap_fixture():
  return mujoco.MjModel.from_xml_string("""<mujoco>
    <default><geom contype="0" conaffinity="0"/></default>
    <worldbody><body name="b"><freejoint/>
      <geom name="a" type="sphere" size=".1"/>
      <geom name="other" type="sphere" size=".1" pos=".35 0 0"/>
    </body></worldbody><sensor>
      <distance body1="b" geom2="a" cutoff="1"/>
      <normal body1="b" geom2="a" cutoff="1"/>
      <fromto body1="b" geom2="a" cutoff="1"/>
      <distance geom1="a" body2="b" cutoff="1"/>
      <normal geom1="a" body2="b" cutoff="1"/>
      <fromto geom1="a" body2="b" cutoff="1"/>
    </sensor></mujoco>""")


def test_body_geom_overlap_keeps_every_pinned_query_pair():
  from mujoco_metal.sensors import lower_sensors
  from mujoco_metal.geometry_queries import lower_geometry_queries
  model=_overlap_fixture();d=mujoco.MjData(model);mujoco.mj_forward(model,d)
  assert lower_sensors(model).nsensor==6
  desc=lower_geometry_queries(model,2)
  assert desc['pairs'].shape==(12,3)
  assert np.count_nonzero(desc['pairs'][:,1]==desc['pairs'][:,2])==6
  np.testing.assert_allclose(d.sensordata[[0,10]],-.2,atol=1e-15)


@pytest.mark.gpu
@pytest.mark.skipif(os.getenv("MUJOCO_METAL_RUN_GPU") != "1",reason="native overlap qualification")
def test_native_body_geom_overlap_uses_source_self_witnesses_and_replay():
  import torch
  from mujoco_metal.sensors import SensorProgram
  model=_overlap_fixture();d=mujoco.MjData(model);p=SensorProgram(model,2)
  positions=[];expected=[]
  for x in (0.,.125):
    d.qpos[:]=model.qpos0;d.qpos[0]=x;mujoco.mj_forward(model,d)
    positions.append(d.geom_xpos.copy());expected.append(d.sensordata.copy())
  poses={'geom_pos':torch.tensor(np.asarray(positions),dtype=torch.float32,device='mps'),
         'geom_quat':torch.tensor(np.tile([1.,0.,0.,0.],(2,model.ngeom,1)),dtype=torch.float32,device='mps')}
  actual=p.run_geomdist_device(poses,p._sp_hull,p._sp_hull_info)
  np.testing.assert_allclose(actual.cpu().numpy(),expected,atol=2e-6,rtol=2e-6)
  saved=actual.clone()
  poses['geom_pos'][0].add_(.5)
  p.run_geomdist_device(poses,p._sp_hull,p._sp_hull_info,
      world_mask=torch.tensor([False,True],device='mps'))
  torch.testing.assert_close(actual,saved,rtol=0,atol=0)
  poses['geom_pos'][0].sub_(.5)
  replay=p.run_geomdist_device(poses,p._sp_hull,p._sp_hull_info)
  torch.testing.assert_close(replay,saved,rtol=0,atol=0)
