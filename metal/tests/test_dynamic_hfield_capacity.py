# Copyright 2026 The MuJoCo Metal contributors
# Licensed under the Apache License, Version 2.0.
"""Compiled terrain operands exceed former fixed allocation ceilings intact."""
import os

import mujoco
import numpy as np
import pytest

from mujoco_metal.capacity import CapacityLimits, CapacityOverflow
from mujoco_metal.coupled_constraints import lower_coupled_constraints


def _model(count=1, rows=80, cols=96, alternate=False):
  values = np.asarray([((r + 2*c) % 17) / 16
                       for r in range(rows) for c in range(cols)])
  asset = (f'<hfield name="grid" nrow="{rows}" ncol="{cols}" '
           f'size="1 1 .03 .1" elevation="{" ".join(map(str, values))}"/>')
  terrains = ''.join(f'<geom name="terrain{i}" type="hfield" hfield="grid" '
                     f'pos="{3*i} 0 0"/>' for i in range(count))
  model = mujoco.MjModel.from_xml_string(f'''<mujoco>
      <option gravity="0 0 0" timestep=".0002" solver="Newton"
              cone="elliptic" iterations="100" tolerance="1e-8"/>
      <default><geom condim="1"/></default>
      <asset>{asset}</asset><worldbody>{terrains}
        <body pos=".013 .017 .035"><freejoint/>
          <geom name="sphere" type="sphere" size=".025" mass="1"/>
        </body></worldbody></mujoco>''')
  if alternate:
    model.opt.disableflags |= int(mujoco.mjtDisableBit.mjDSBL_NATIVECCD)
  return model


@pytest.mark.parametrize('count', [1, 9])
def test_every_compiled_sample_and_terrain_descriptor_survives_old_limits(count):
  model = _model(count)
  desc = lower_coupled_constraints(model,
      limits=CapacityLimits(max_slots=50*count, max_rows=50*count+8))
  assert desc.npairs == count
  assert desc.ncontacts_max == 50*count
  intervals = []
  for index in range(count):
    geom = int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, f'terrain{index}'))
    record = desc.mesh_hull_info[9*geom:9*geom+9]
    start, rows, cols, size = map(int, record[5:9])
    assert (rows, cols) == (80, 96)
    intervals.append((start, start+rows*cols))
    np.testing.assert_array_equal(desc.mesh_hull[start:start+rows*cols], model.hfield_data)
    np.testing.assert_array_equal(desc.mesh_hull[size:size+4],
                                 model.hfield_size[0].astype(np.float32))
  assert all(a[1] <= b[0] for a,b in zip(intervals, intervals[1:]))
  assert intervals[-1][1] <= desc.mesh_hull.size


def test_dynamic_terrain_still_rejects_an_insufficient_explicit_memory_budget():
  model = _model()
  with pytest.raises(CapacityOverflow, match='memory|budget'):
    lower_coupled_constraints(model,
        limits=CapacityLimits(memory_budget_bytes=1024))


@pytest.mark.gpu
@pytest.mark.skipif(os.getenv('MUJOCO_METAL_RUN_GPU') != '1', reason='native opt-in')
@pytest.mark.parametrize('alternate', [False, True])
def test_large_rectangular_terrain_native_forces_steps_and_exact_restore(alternate):
  from mujoco_metal.simulation import MetalSimulation
  model = _model(alternate=alternate)
  qpos = np.tile(model.qpos0.astype(np.float32), (2,1)); qpos[1,2] = .8
  qvel = np.tile(np.asarray([.02,-.01,.03,.04,-.02,.01],np.float32),(2,1))
  refs = [mujoco.MjData(model) for _ in range(2)]
  for world, data in enumerate(refs):
    data.qpos[:], data.qvel[:] = qpos[world], qvel[world]
    mujoco.mj_forward(model,data)
  assert refs[0].ncon > 0 and refs[1].ncon == 0
  assert np.linalg.norm(refs[0].qfrc_constraint) > 1
  sim = MetalSimulation(model,batch_size=2,qpos=qpos,qvel=qvel,
                        profile='integrated_scalable_v1')
  host = lambda value: value.detach().cpu().numpy().copy()
  system = sim.assembled_system(recompute=True)
  np.testing.assert_array_equal(host(system['status']),0)
  active = host(system['contact_mask']) > .5
  assert active[0].sum() == refs[0].ncon and not active[1].any()
  for field,expected,bound in (
      ('contact_distance',[c.dist for c in refs[0].contact[:refs[0].ncon]],2e-5),
      ('contact_position',[c.pos for c in refs[0].contact[:refs[0].ncon]],2e-4),
      ('contact_normal',[c.frame[:3] for c in refs[0].contact[:refs[0].ncon]],2e-4)):
    np.testing.assert_allclose(host(system[field])[0,active[0]],expected,rtol=0,atol=bound)
  np.testing.assert_allclose(host(system['qfrc_constraint']),
                             [d.qfrc_constraint for d in refs],rtol=2e-3,atol=2e-3)
  for _ in range(4):
    np.testing.assert_array_equal(host(sim.step()),0)
    for data in refs: mujoco.mj_step(model,data)
    for field in ('qpos','qvel','qacc'):
      np.testing.assert_allclose(host(getattr(sim.state,'_'+field)),
                                 [getattr(d,field) for d in refs],rtol=2e-3,atol=2e-3)
  checkpoint=sim.snapshot(); np.testing.assert_array_equal(host(sim.step()),0)
  expected={f:host(getattr(sim.state,'_'+f)) for f in ('qpos','qvel','qacc')}
  sim.restore(checkpoint); np.testing.assert_array_equal(host(sim.step()),0)
  for field,wanted in expected.items():
    np.testing.assert_array_equal(host(getattr(sim.state,'_'+field)),wanted)
