# Copyright 2026 The MuJoCo Metal contributors
# Licensed under the Apache License, Version 2.0.
"""Admission must cover real owned buffers, rather than just helper products."""
import builtins
import os
from dataclasses import replace
import mujoco
import numpy as np
import pytest
from mujoco_metal.capacity import CapacityLimits, CapacityOverflow, estimate_capacity
from mujoco_metal.sensor_capacity import sensor_owned_workspace_sizes


def _model(kind):
  asset=''
  shape='<geom name="b" type="sphere" size=".1"/>'
  if kind=='sdf':
    asset='<asset><mesh name="box" vertex="-.1 -.1 -.1 .1 -.1 -.1 .1 .1 -.1 -.1 .1 -.1 -.1 -.1 .1 .1 -.1 .1 .1 .1 .1 -.1 .1 .1"/></asset>'
    shape='<geom name="b" type="sdf" mesh="box"/>'
  sensors='<clock/><jointpos joint="j"/>'
  if kind in ('query','sdf'):sensors+='<distance geom1="a" geom2="b" cutoff="1"/><fromto geom1="b" geom2="a" cutoff="1"/>'
  if kind=='ray':sensors+='<rangefinder site="s"/>'
  return mujoco.MjModel.from_xml_string(f'<mujoco>{asset}<option sdf_initpoints="3"/><worldbody><geom name="a" type="sphere" size=".1"/><body><joint name="j"/><site name="s"/>{shape}</body></worldbody><sensor>{sensors}</sensor></mujoco>')


@pytest.mark.parametrize('kind',['empty','state','query','sdf','ray'])
def test_sensor_inventory_is_part_of_aggregate_preflight(kind):
  model=(_model(kind) if kind!='empty' else
      mujoco.MjModel.from_xml_string('<mujoco/>'))
  owned=sensor_owned_workspace_sizes(model,2)
  total=dict(estimate_capacity(model,2,0,0,0).memory_breakdown)
  for name,count in owned.items():assert total['sensor.'+name]==count*4
  assert owned['state_output_pair']==2*max(2*model.nsensordata,1)
  if kind in ('query','sdf'):
    from mujoco_metal.geometry_queries import lower_geometry_queries
    desc=lower_geometry_queries(model,2)
    assert desc['bytes_total']==4*sum(desc['workspace_sizes'].values())
    assert 'sp_hull_info' not in owned
    assert desc['workspace_sizes']['hull']+desc['workspace_sizes']['info']==desc['common_bytes']//4


@pytest.mark.parametrize('kind',['state','query','sdf','ray'])
def test_sensor_budget_rejection_precedes_torch_import(kind,monkeypatch):
  from mujoco_metal.sensors import SensorProgram
  model=_model(kind)
  requested=4*sum(sensor_owned_workspace_sizes(model,2).values())
  original=builtins.__import__
  def guarded(name,*args,**kwargs):
    if name=='torch':raise AssertionError('device boundary reached before admission')
    return original(name,*args,**kwargs)
  monkeypatch.setattr(builtins,'__import__',guarded)
  with pytest.raises(CapacityOverflow,match='workspace.*budget'):
    SensorProgram(model,2,limits=replace(CapacityLimits(),memory_budget_bytes=requested-1))


@pytest.mark.parametrize('kind',['empty','state','query','sdf','ray'])
@pytest.mark.gpu
@pytest.mark.skipif(os.getenv('MUJOCO_METAL_RUN_GPU')!='1',reason='owned MPS allocation qualification')
def test_actual_sensor_backings_match_host_admission_at_exact_budget(kind):
  import torch
  from mujoco_metal.sensors import SensorProgram
  model=(_model(kind) if kind!='empty' else mujoco.MjModel.from_xml_string('<mujoco/>'))
  sizes=sensor_owned_workspace_sizes(model,2)
  expected=4*sum(sizes.values())
  program=SensorProgram(model,2,limits=replace(CapacityLimits(),memory_budget_bytes=expected))
  storage={};seen=set()
  def visit(value):
    if isinstance(value,torch.Tensor):
      allocation=value.untyped_storage()
      storage[allocation.data_ptr()]=allocation.nbytes()
    elif isinstance(value,dict):
      for v in value.values():visit(v)
    elif isinstance(value,(tuple,list)):
      for v in value:visit(v)
    elif value is program or type(value).__module__ in (
        'mujoco_metal.geometry_queries','mujoco_metal.ray_queries'):
      if id(value) in seen:return
      seen.add(id(value))
      for v in vars(value).values():visit(v)
  visit(program)
  assert sum(storage.values())==expected,(kind,sum(storage.values()),expected)
  if kind in ('query','sdf'):
    assert program._sp_hull is program._geometry_queries.hull
    assert program._sp_hull_info is program._geometry_queries.info
