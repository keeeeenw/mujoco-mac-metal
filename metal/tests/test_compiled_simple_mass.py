# Copyright 2026 The MuJoCo Metal contributors
# Licensed under the Apache License, Version 2.0.
"""Execute mj_crb's compiled simple-DOF route, not a diagonal-model heuristic."""
from dataclasses import replace
import os

import mujoco
import numpy as np
import pytest

from mujoco_metal.model import load_model, snapshot_descriptor


_XML = '''<mujoco><option gravity="0 0 0"><flag contact="disable"/></option>
  <worldbody>
    <body><freejoint/><inertial pos="0 0 0" mass="1.3"
      diaginertia=".023125 .024125 .03125"/></body>
    <body pos="2 0 0"><joint name="h" axis="0 1 0" armature=".037"/>
      <inertial pos="0 0 0" mass="1.1" diaginertia=".2 .3 .4"/></body>
    <body pos="4 0 0"><joint type="ball"/>
      <inertial pos="0 0 0" mass="1.2" diaginertia=".3 .4 .5"/></body>
    <body pos="6 0 0"><joint type="slide" axis="1 0 0"/>
      <joint type="slide" axis="0 1 0"/>
      <inertial pos="0 0 0" mass="1.7" diaginertia=".4 .5 .6"/></body>
    <body pos="8 0 0"><joint pos=".2 0 0" axis="0 0 1"/>
      <inertial pos=".1 .2 .3" mass="1.4" diaginertia=".2 .3 .4"/></body>
  </worldbody><actuator><motor joint="h" gear="-3" armature=".013"/></actuator>
</mujoco>'''


def test_simple_constants_owned_and_invalidation_is_stable():
  source = mujoco.MjModel.from_xml_string(_XML)
  desc = load_model(source)
  assert np.any(source.dof_simplenum > 0) and np.any(source.dof_simplenum == 0)
  for name in ("dof_simplenum", "dof_M0"):
    np.testing.assert_array_equal(getattr(desc, name), getattr(source, name))
    np.testing.assert_array_equal(getattr(snapshot_descriptor(desc), name), getattr(source, name))
    with pytest.raises(ValueError):
      getattr(desc, name).setflags(write=True)
  edited = replace(desc, body_mass=np.asarray(desc.body_mass) * 2)
  detached = snapshot_descriptor(edited)
  np.testing.assert_array_equal(detached.dof_simplenum, 0)
  np.testing.assert_array_equal(snapshot_descriptor(detached).dof_simplenum, 0)
  # A compiled source recomputation restores the shortcut and fresh constants.
  source.body_mass[:] *= 2
  mujoco.mj_setConst(source, mujoco.MjData(source))
  fresh = snapshot_descriptor(load_model(source))
  np.testing.assert_array_equal(fresh.dof_simplenum, source.dof_simplenum)
  np.testing.assert_array_equal(fresh.dof_M0, source.dof_M0)
  assert not np.array_equal(fresh.dof_M0, desc.dof_M0)


def test_lifecycle_updates_keep_source_simple_constants_coherent():
  from mujoco_metal.lifecycle import ModelLifecycle
  life = ModelLifecycle(_XML)
  old = life.descriptor.dof_M0.copy()
  assert life.recompute_body_inertias([1], [[.04, .05, .06]])
  assert life.recompute_body_masses([1], [2.7])
  current = snapshot_descriptor(life.descriptor)
  np.testing.assert_array_equal(current.dof_simplenum, life._model.dof_simplenum)
  np.testing.assert_array_equal(current.dof_M0, life._model.dof_M0)
  assert not np.array_equal(current.dof_M0, old)


def test_compiled_simple_constants_accounted_in_admission():
  # Admission owns both new physical device allocations, including empty models.
  from mujoco_metal.capacity import _runtime_buffer_sizes
  for xml in (_XML, '<mujoco/>'):
    source = mujoco.MjModel.from_xml_string(xml)
    parts = dict(_runtime_buffer_sizes(source, 2))
    assert parts["model.smooth.dof_simplenum"] == max(source.nv, 1)
    assert parts["model.smooth.dof_M0"] == max(source.nv, 1)


_GPU = pytest.mark.skipif(os.getenv("MUJOCO_METAL_RUN_GPU") != "1",
                          reason="native compiled inertia execution")


def _dense(result, smooth):
  if "mass_matrix" in result:
    return result["mass_matrix"].cpu().numpy()
  blocks = result["mass_blocks"].cpu().numpy()
  layout = smooth.mass_block_layout
  out = np.zeros((blocks.shape[0], smooth.model.nv, smooth.model.nv), np.float32)
  for c in range(layout["ncomponent"]):
    start, end = layout["component_dof_offsets"][c:c+2]
    ids = layout["component_dof_ids"][start:end]
    offset = layout["component_mass_offsets"][c]
    out[:, ids[:, None], ids] = blocks[:, offset:offset+len(ids)**2].reshape(-1,len(ids),len(ids))
  return out


@pytest.mark.gpu
@_GPU
@pytest.mark.parametrize("route", ["query", "dense", "block_sparse"])
def test_rotated_simple_and_non_simple_mass_execute_source_rule(route):
  import torch
  from mujoco_metal.smooth_metal import MetalSmoothDynamics
  source = mujoco.MjModel.from_xml_string(_XML)
  smooth = MetalSmoothDynamics(load_model(source), 2,
                               mass_storage="dense" if route == "query" else route)
  simple = np.flatnonzero(source.dof_simplenum)
  for seed in range(4):
    rng = np.random.default_rng(seed + 400)
    q = np.tile(source.qpos0.astype(np.float32), (2,1))
    for typ, adr in zip(source.jnt_type, source.jnt_qposadr):
      if int(typ) in (0, 1):
        start = int(adr) + (3 if int(typ) == 0 else 0)
        quat = rng.normal(size=(2,4)); quat /= np.linalg.norm(quat,axis=1)[:,None]
        q[:,start:start+4] = quat
      else:
        q[:,adr] = rng.normal(size=2)
    if route == "query":
      mass = smooth.mass_matrix(q).cpu().numpy()
    else:
      mass = _dense(smooth.run_device(torch.tensor(q,device="mps"),
          torch.zeros((2,source.nv),device="mps")), smooth)
    for world in range(2):
      d = mujoco.MjData(source); d.qpos[:] = q[world]; mujoco.mj_forward(source,d)
      expected = np.empty((source.nv,source.nv)); mujoco.mj_fullM(source,d,expected)
      # The compiled diagonal includes armature exactly once. All off-diagonal
      # words are exact zero even after arbitrary free/ball rotations.
      np.testing.assert_array_equal(mass[world,simple], expected[simple].astype(np.float32))
      np.testing.assert_allclose(mass[world],expected,rtol=3e-5,atol=3e-5)
  if route != "query":
    # Both a completely unselected world and sleeping simple rows retain their
    # caller-owned backing, not a fresh diagonal/zero from this new dispatch.
    backing = smooth._workspace["mass" if route == "dense" else "mass_blocks"]
    backing.fill_(123.25)
    lists = smooth._all_awake_lists(2)
    for col, name in enumerate(("body_count", "parent_count", "dof_count")):
      lists[name] = lists["counts"][:,col]
    lists["counts"][1].zero_()
    smooth.run_device(torch.tensor(q,device="mps"),torch.zeros((2,source.nv),device="mps"),
                      awake_lists=lists, world_mask=torch.tensor([0,1],dtype=torch.int32,device="mps"))
    np.testing.assert_array_equal(backing.cpu().numpy(),123.25)


@pytest.mark.gpu
@_GPU
def test_direct_descriptor_edit_does_not_use_stale_M0():
  import torch
  from mujoco_metal.smooth_metal import MetalSmoothDynamics
  source = mujoco.MjModel.from_xml_string(_XML)
  desc = load_model(source)
  edited = replace(desc,body_mass=np.asarray(desc.body_mass)*2)
  smooth = MetalSmoothDynamics(edited)
  assert not np.any(smooth.model.dof_simplenum)
  source.body_mass[:] *= 2; mujoco.mj_setConst(source,mujoco.MjData(source))
  data = mujoco.MjData(source); mujoco.mj_forward(source,data)
  expected = np.empty((source.nv,source.nv)); mujoco.mj_fullM(source,data,expected)
  actual = smooth.mass_matrix(source.qpos0).cpu().numpy()[0]
  np.testing.assert_allclose(actual,expected,rtol=3e-5,atol=3e-5)
