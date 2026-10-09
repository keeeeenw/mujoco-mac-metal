# Copyright 2026 The MuJoCo Metal contributors
# Licensed under the Apache License, Version 2.0.
"""Stored rangefinder queries use the complete native scene-ray route."""
import os

import mujoco
import numpy as np
import pytest

from mujoco_metal.ray_queries import site_ray_workspace_sizes
from test_ray_queries_019 import _matrix_quat


def _scene():
  vertices, faces = [], []
  for i in range(16):
    u=2*np.pi*i/16
    for j in range(8):
      v=2*np.pi*j/8
      vertices.append(((.7+.2*np.cos(v))*np.cos(u),
                       (.7+.2*np.cos(v))*np.sin(u),.2*np.sin(v)))
      a=8*i+j;b=8*((i+1)%16)+j;c=8*((i+1)%16)+(j+1)%8;d=8*i+(j+1)%8
      faces.extend(((a,b,c),(a,c,d)))
  text=lambda rows:' '.join(str(x) for row in rows for x in row)
  plugins=('bolt','bowl','gear','nut','torus')
  extension=''.join(f'<plugin plugin="mujoco.sdf.{name}"><instance name="p{i}"/></plugin>'
                    for i,name in enumerate(plugins))
  assets=''.join(f'<mesh name="m{i}"><plugin instance="p{i}"/></mesh>' for i in range(5))
  geoms=''.join(f'<geom type="sdf" mesh="m{i}" pos="{4*i} 0 0"><plugin instance="p{i}"/></geom>'
                for i in range(5))
  # Pick visible material on each provider rather than rays through the gear,
  # nut and torus holes. Miss-only plugin gates cannot qualify a sensor path.
  points=[(4*i+x,.07,2) for i,x in enumerate((.19,.19,1.,.3,.3))]
  points += [(21.04,.07,2),(24,0,2),(24.67,.08,2),(28,.04,2),(40,0,2)]
  sites=''.join(f'<site name="s{i}" pos="{x} {y} {z}" quat="0 1 0 0"/>'
                for i,(x,y,z) in enumerate(points))
  sensors=''.join(f'<rangefinder site="s{i}" data="dist dir origin point normal depth"/>'
                   for i in range(len(points)))
  model=mujoco.MjModel.from_xml_string(f'''<mujoco><extension>{extension}</extension>
    <asset>{assets}<mesh name="ring" vertex="{text(vertices)}" face="{text(faces)}"/>
    <mesh name="cube" vertex="-.3 -.3 -.2 .3 -.3 -.2 .3 .3 -.2 -.3 .3 -.2
      -.3 -.3 .2 .3 -.3 .2 .3 .3 .2 -.3 .3 .2"/>
    <hfield name="hill" nrow="67" ncol="3" size="1 1 .2 .1"/>
    </asset><worldbody>{geoms}
    <geom type="sdf" mesh="cube" pos="21 0 0" contype="0" conaffinity="0"/>
    <geom type="mesh" mesh="ring" pos="24 0 0" contype="0" conaffinity="0"/>
    <geom type="hfield" hfield="hill" pos="28 0 0" contype="0" conaffinity="0"/>
    <body>{sites}</body></worldbody><sensor>{sensors}</sensor></mujoco>''')
  model.hfield_data[:] = np.linspace(.1, .8, model.hfield_data.size)
  return model


def test_site_ray_inventory_has_complete_mesh_terrain_and_octree_assets():
  model=_scene()
  sizes=site_ray_workspace_sizes(model)
  assert sizes["hull"] > 3*128 + 3*256 + 67*3
  assert sizes["octree"] == 50*int(model.mesh_octnum[model.mesh("cube").id])
  assert sizes["dimensions"] == 6
  empty=mujoco.MjModel.from_xml_string('<mujoco/>')
  assert site_ray_workspace_sizes(empty) == {}


@pytest.mark.skipif(os.environ.get("MUJOCO_METAL_RUN_GPU") != "1",
                   reason="explicit serialized native rangefinder gate")
def test_actual_native_rangefinder_all_fields_full_scene_mask_and_buffer_inventory():
  import torch
  from mujoco_metal.sensors import SensorProgram
  from mujoco_metal.capacity import _runtime_buffer_sizes
  model=_scene()
  data=mujoco.MjData(model);mujoco.mj_forward(model,data)
  expected=np.stack([data.sensordata]*2)
  values=expected[0].reshape(-1,14)
  assert np.count_nonzero(values[:,0]>=0)==8
  assert np.all(values[:5,0]>=0)
  assert values[6,0]==values[9,0]==-1  # Mesh hole and empty scene ray.
  tensor=lambda value:torch.tensor(np.asarray(value).copy(),device="mps")
  quat=lambda mats:np.array([[_matrix_quat(m) for m in mats]]*2,np.float32)
  poses={"site_pos":tensor(np.stack([data.site_xpos]*2).astype(np.float32)),
         "site_quat":tensor(quat(data.site_xmat)),
         "geom_pos":tensor(np.stack([data.geom_xpos]*2).astype(np.float32)),
         "geom_quat":tensor(quat(data.geom_xmat))}
  program=SensorProgram(model,batch_size=2)
  actual=program.run_rays_device(poses,program._sp_hull,program._sp_hull_info)
  np.testing.assert_allclose(actual.cpu(),expected,rtol=2e-4,atol=3e-5)
  constants=program._site_ray_program._constants
  sizes=site_ray_workspace_sizes(model)
  inventory=dict(_runtime_buffer_sizes(model,2))
  for name,count in sizes.items():
    actual_buffer=program._site_ray_dims if name=="dimensions" else constants[name]
    assert actual_buffer.numel()==count==inventory[f"sensor.site_rays.{name}"]
  retained=actual.clone()
  poses["site_pos"][:,:,2] += .03
  selected=program.run_rays_device(poses,program._sp_hull,program._sp_hull_info,
      world_mask=torch.tensor([False,True],device="mps"))
  torch.testing.assert_close(selected[0],retained[0],rtol=0,atol=0)
  assert float(torch.max(torch.abs(selected[1]-retained[1])).cpu()) > .02
  unmasked=program.run_rays_device(poses,program._sp_hull,program._sp_hull_info)
  torch.testing.assert_close(unmasked[0],unmasked[1],rtol=0,atol=0)
