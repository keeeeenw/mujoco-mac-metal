# Copyright 2026 The MuJoCo Metal contributors
# Licensed under the Apache License, Version 2.0.
"""Pinned camera depth-image rays, every camera mode and both projections."""
import os
import ctypes
from pathlib import Path
import subprocess

import mujoco
import numpy as np
import pytest

from mujoco_metal.ray_queries import lower_rangefinder_pixels, site_ray_workspace_sizes
from mujoco_metal.sensors import lower_sensors
from test_ray_queries_019 import _matrix_quat
from test_common_ccd_epa_witness_source import _function


def _model(mode="fixed", lens="perspective", history=""):
  projection={"perspective":'fovy="52"',
              "intrinsic":'sensorsize=".036 .024" focal=".023 .021" principal=".015 .01"',
              "orthographic":'projection="orthographic" fovy="1.8"'}[lens]
  target='target="target"' if mode in ("targetbody","targetbodycom") else ''
  return mujoco.MjModel.from_xml_string(f'''<mujoco>
    <option gravity="0 0 0"><flag contact="disable"/></option>
    <worldbody><geom type="plane" size="5 5 .1"/>
      <geom type="box" pos=".3 0 .2" size=".2 .25 .2"/>
      <geom type="sphere" pos="-.5 .2 .3" size=".3"/>
      <body name="mount" pos=".4 -.3 2.1" euler="8 12 -5"><freejoint/>
        <geom type="box" size=".04 .05 .06" mass="1"/>
        <camera name="eye" pos=".03 .01 .02" euler="-10 -9 8" resolution="6 4"
                mode="{mode}" {target} {projection}/>
        <site name="site_ray" quat="0 1 0 0"/>
        <body pos=".1 .02 .03"><joint type="hinge" axis="0 1 0"/>
          <geom type="capsule" size=".03 .12" mass=".7" pos=".1 0 0"/>
        </body>
      </body>
      <body name="target" pos=".2 .1 .4"><freejoint/>
        <geom type="sphere" size=".08" mass=".8"/>
        <body pos=".1 .1 .12"><geom type="box" size=".06 .03 .05" mass=".5"/></body>
      </body>
    </worldbody><sensor>
      <rangefinder camera="eye" data="dist dir origin point normal depth" {history}/>
      <rangefinder site="site_ray" data="dist normal depth"/>
    </sensor></mujoco>''')


def _states(model):
  qpos=np.stack([model.qpos0]*2).astype(np.float32)
  qpos[1,0]+=.03;qpos[1,2]+=.04;qpos[1,7]=.16;qpos[1,8]-=.07
  return qpos


def _feed(model, qpos):
  refs=[]
  for qp in qpos:
    d=mujoco.MjData(model);d.qpos[:]=qp.astype(float)
    mujoco.mj_forward(model,d);refs.append(d)
  tensor_poses={}
  for prefix,pos,mat in (("site","site_xpos","site_xmat"),("geom","geom_xpos","geom_xmat")):
    tensor_poses[prefix+"_pos"]=np.stack([getattr(d,pos) for d in refs]).astype(np.float32)
    tensor_poses[prefix+"_quat"]=np.array([[_matrix_quat(m) for m in getattr(d,mat)] for d in refs],np.float32)
  tensor_poses.update(body_pos=np.stack([d.xpos for d in refs]).astype(np.float32),
      body_quat=np.stack([d.xquat for d in refs]).astype(np.float32),
      inertial_pos=np.stack([d.xipos for d in refs]).astype(np.float32))
  return refs,tensor_poses,np.stack([d.sensordata for d in refs])


@pytest.mark.parametrize("lens",["perspective","intrinsic","orthographic"])
def test_actual_camera_rangefinder_lowering_preserves_pixels_and_complete_dimensions(lens):
  model=_model(lens=lens)
  desc=lower_sensors(model)
  assert desc.sensor_dim.tolist()==[6*4*14,5]
  pixels=lower_rangefinder_pixels(model)
  expected=[(0,col,row) for row in range(4) for col in range(6)]+[(1,0,0)]
  np.testing.assert_array_equal(pixels["ray_entries"],expected)
  assert pixels["camera_meta"][0,3] == (lens=="orthographic")
  assert all(not value.flags.writeable for value in pixels.values())


def _camera_source_oracle(tmp_path):
  root=os.getenv("MUJOCO_METAL_TRACE_ROOT")
  if not root:
    pytest.skip("pinned source required for internal camera helpers not exported by the wheel")
  root=Path(root)/".upstream/mujoco-3.10.0/src/engine"
  support=(root/"engine_support.c").read_text()
  misc=(root/"engine_util_misc.c").read_text()
  code=support[:support.index("#include")]+"#include <math.h>\n#include <mujoco/mujoco.h>\n"
  # Compile the unchanged target functions, retaining their original license.
  code+=_function(support,r"void mju_camIntrinsics\(")+"\n"
  code+=_function(misc,r"void mju_camPixelRay\(")+"\n"
  source=tmp_path/"pinned_camera.c";source.write_text(code)
  package=Path(mujoco.__file__).parent
  binary=package/"libmujoco.3.10.0.dylib"
  assert binary.is_file(),"camera helper/model ABI is pinned to MuJoCo 3.10.0"
  library=tmp_path/"pinned_camera.dylib"
  subprocess.run(["/usr/bin/clang","-O3","-std=c11","-shared",
      "-I",str(package/"include"),str(source),str(binary),
      "-Wl,-rpath,"+str(package),"-o",str(library)],check=True,
      capture_output=True,text=True,timeout=30)
  return ctypes.CDLL(str(library))


@pytest.mark.parametrize("lens",["perspective","intrinsic","orthographic"])
def test_camera_lens_constants_and_host_sensor_oracle_match_actual_pinned_functions(tmp_path,lens):
  from mujoco_metal.sensors import sensor_oracle
  model=_model("targetbodycom",lens);qp=_states(model)
  refs,poses,expected=_feed(model,qp)
  library=_camera_source_oracle(tmp_path);pd=ctypes.POINTER(ctypes.c_double)
  intrinsics=library.mju_camIntrinsics
  intrinsics.argtypes=[ctypes.c_void_p,ctypes.c_int]+[pd]*5
  intrinsics.restype=None
  lens_values=[ctypes.c_double() for _ in range(5)]
  intrinsics(model._address,0,*[ctypes.byref(v) for v in lens_values])
  params=[v.value for v in lens_values]
  np.testing.assert_array_equal(lower_rangefinder_pixels(model)["camera_values"][0,22:],
                                 np.asarray(params,np.float32))
  pixel_ray=library.mju_camPixelRay
  pixel_ray.argtypes=[pd,pd,pd,pd,ctypes.c_int,ctypes.c_int]+[ctypes.c_double]*4+[ctypes.c_int,ctypes.c_double]
  pixel_ray.restype=None
  camera={name:[] for name in ("distance","origin","direction","normal","camera_pos","camera_z")}
  site_ray=np.zeros((2,2,8))
  for world,ref in enumerate(refs):
    origins=np.empty((24,3));directions=np.empty_like(origins)
    cp=np.ascontiguousarray(ref.cam_xpos[0]);cm=np.ascontiguousarray(ref.cam_xmat[0])
    for row in range(4):
      for col in range(6):
        origin,direction=origins[row*6+col],directions[row*6+col]
        pixel_ray(origin.ctypes.data_as(pd),direction.ctypes.data_as(pd),cp.ctypes.data_as(pd),
            cm.ctypes.data_as(pd),col,row,*params[:4],int(model.cam_projection[0]),params[4])
    distances=np.empty(24);normals=np.empty_like(origins);gids=np.empty(24,np.int32)
    if lens!="orthographic":
      mujoco.mj_multiRay(model,ref,cp,directions.reshape(-1),None,True,
          int(model.cam_bodyid[0]),gids,distances,normals.reshape(-1),24,1e10)
    else:
      for pixel in range(24):
        distances[pixel]=mujoco.mj_ray(model,ref,origins[pixel],directions[pixel],None,
            True,int(model.cam_bodyid[0]),gids[pixel:pixel+1],normals[pixel])
    for name,value in (("distance",distances),("origin",origins),("direction",directions),
                       ("normal",normals),("camera_pos",cp),("camera_z",cm.reshape(3,3)[:,2])):
      camera[name].append(value.copy())
    origin=ref.site_xpos[0];direction=ref.site_xmat[0].reshape(3,3)[:,2].copy()
    normal=np.zeros(3);gid=np.array([-1],np.int32)
    distance=mujoco.mj_ray(model,ref,origin,direction,None,True,int(model.site_bodyid[0]),gid,normal)
    site_ray[world,1]=np.concatenate(([distance],direction,normal,gid))
  extra={"camera_ray":{0:{name:np.asarray(value) for name,value in camera.items()}},"ray":site_ray}
  # Actual camera/site engine geometry feeds, with independently assembled
  # data fields. Neither the production native kernel nor a saved expected
  # sensordata vector supplies this host oracle result.
  actual=sensor_oracle(model,qp,np.zeros((2,model.nv)),np.zeros(2),poses,extra=extra)
  np.testing.assert_allclose(actual,expected,rtol=3e-7,atol=3e-7)
  bad={**extra,"camera_ray":{0:{**extra["camera_ray"][0],"distance":np.zeros((2,23))}}}
  with pytest.raises(ValueError,match="compiled pixel"):
    sensor_oracle(model,qp,np.zeros((2,model.nv)),np.zeros(2),poses,extra=bad)


gpu=pytest.mark.skipif(os.environ.get("MUJOCO_METAL_RUN_GPU") != "1",
                      reason="explicit serialized native camera qualification")


@gpu
@pytest.mark.parametrize("mode",["fixed","track","trackcom","targetbody","targetbodycom"])
@pytest.mark.parametrize("lens",["perspective","intrinsic","orthographic"])
def test_native_camera_rangefinder_all_modes_lenses_fields_and_masked_worlds(mode,lens):
  import torch
  from mujoco_metal.sensors import SensorProgram
  from mujoco_metal.capacity import _runtime_buffer_sizes
  model=_model(mode,lens);qp=_states(model)
  refs,poses,expected=_feed(model,qp)
  pixels=expected[:,:336].reshape(2,24,14)
  assert np.count_nonzero(pixels[:,:,0]>=0)>12
  if lens!="orthographic":
    assert np.max(abs(pixels[:,:,0]-pixels[:,:,13]))>.01
  tensor=lambda a:torch.tensor(np.asarray(a).copy(),device="mps")
  native_poses={key:tensor(a) for key,a in poses.items()}
  program=SensorProgram(model,batch_size=2)
  result=program.run_rays_device(native_poses,program._sp_hull,program._sp_hull_info)
  np.testing.assert_allclose(result.cpu(),expected,rtol=3e-5,atol=3e-5)
  constants=program._site_ray_program._constants
  inventory=dict(_runtime_buffer_sizes(model,2))
  for name,count in site_ray_workspace_sizes(model).items():
    buffer=(program._site_ray_dims if name=="dimensions" else
        program._site_ray_pixels[name] if name in program._site_ray_pixels else constants[name])
    assert buffer.numel()==count==inventory[f"sensor.site_rays.{name}"]
  retained=result.clone()
  changed=qp.copy();changed[1,0]+=.08;changed[1,2]+=.06;changed[1,7]+=.3
  _,next_poses,next_expected=_feed(model,changed)
  selected=program.run_rays_device({key:tensor(a) for key,a in next_poses.items()},
      program._sp_hull,program._sp_hull_info,
      world_mask=torch.tensor([False,True],device="mps"))
  torch.testing.assert_close(selected[0],retained[0],rtol=0,atol=0)
  np.testing.assert_allclose(selected[1].cpu(),next_expected[1],rtol=3e-5,atol=3e-5)
  assert float(torch.max(torch.abs(selected[1]-retained[1])).cpu())>.01


@gpu
@pytest.mark.parametrize("lens",["perspective","orthographic"])
def test_native_camera_and_site_fields_obey_pinned_compiled_cutoff(lens):
  import torch
  from mujoco_metal.sensors import SensorProgram
  model=_model(lens=lens)
  model.sensor_cutoff[:] = .2
  refs,poses,expected=_feed(model,_states(model))
  assert np.max(expected)<=.2+1e-7
  assert np.min(expected)>=-.2-1e-7
  program=SensorProgram(model,batch_size=2)
  tensor=lambda a:torch.tensor(np.asarray(a).copy(),device="mps")
  actual=program.run_rays_device({name:tensor(value) for name,value in poses.items()},
      program._sp_hull,program._sp_hull_info)
  np.testing.assert_allclose(actual.cpu(),expected,rtol=3e-5,atol=3e-5)


@gpu
@pytest.mark.parametrize("lens",["perspective","orthographic"])
@pytest.mark.parametrize("history",["",'nsample="6" interval=".004"',
                                    'nsample="6" delay=".004" interval=".004" interp="linear"'])
def test_public_camera_rangefinder_stored_step_restore_and_query_order(lens,history):
  import torch
  from mujoco_metal import MetalSimulation
  model=_model("targetbodycom",lens,history);qp=_states(model)
  qv=np.zeros((2,model.nv),np.float32);qv[:,0]=.02;qv[:,6]=.05
  sim=MetalSimulation(model,batch_size=2,qpos=qp,qvel=qv,profile="integrated_euler_v1")
  refs=[mujoco.MjData(model) for _ in range(2)]
  for ref,p,v in zip(refs,qp,qv):ref.qpos[:]=p;ref.qvel[:]=v
  saved=sim.snapshot()
  before=sim.sensor_values().clone()
  torch.testing.assert_close(sim.sensor_values(),before,rtol=0,atol=0)
  results=[]
  for _ in range(8):
    assert sim.step().cpu().tolist()==[0,0]
    for ref in refs:mujoco.mj_step(model,ref)
    results.append(sim.step_sensordata().copy())
    np.testing.assert_allclose(results[-1],np.stack([r.sensordata for r in refs]),rtol=4e-5,atol=4e-5)
  assert np.max(np.abs(results[-1][:,:336])) > .1
  if 'delay=' in history:
    assert np.count_nonzero(results[0][:,:336]) == 0
  if 'interval=' in history and 'delay=' not in history:
    assert any(np.array_equal(a[:,:336],b[:,:336]) for a,b in zip(results,results[1:]))
  sim.restore(saved)
  for expected in results:
    assert sim.step().cpu().tolist()==[0,0]
    np.testing.assert_array_equal(sim.step_sensordata(),expected)
