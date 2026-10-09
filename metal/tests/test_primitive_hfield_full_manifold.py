# Copyright 2026 The MuJoCo Metal contributors
# Licensed under the Apache License, Version 2.0.
"""Every pinned convex terrain witness reaches the public coupled solve."""
import os

import mujoco
import numpy as np
import pytest

from mujoco_metal.capacity import CapacityLimits
from mujoco_metal.coupled_constraints import lower_coupled_constraints

_SHAPES = [('sphere', '.3'), ('capsule', '.2 .15'),
           ('ellipsoid', '.3 .22 .2'), ('cylinder', '.25 .15'),
           ('box', '.3 .22 .2')]
_LIMITS = CapacityLimits(max_slots=50, max_rows=600)


def _model(shape, size, alternate, reversed_order=False, margin=0.):
  terrain = '<geom name="terrain" type="hfield" hfield="terrain"/>'
  moving = (f'<body pos=".017 .023 .1"><freejoint/>'
            f'<geom name="moving" type="{shape}" size="{size}" mass="1"/>'
            '</body>')
  bodies = moving + terrain if reversed_order else terrain + moving
  model = mujoco.MjModel.from_xml_string(f'''<mujoco>
    <option timestep=".0005" gravity="0 0 0" cone="elliptic"
            solver="Newton" iterations="100" tolerance="1e-8"/>
    <default><geom condim="3" friction=".7 .01 .001" margin="{margin}"/></default>
    <asset><hfield name="terrain" nrow="17" ncol="17"
        size="1 1 .1 .1" elevation="{' '.join(['0'] * 289)}"/></asset>
    <worldbody>{bodies}</worldbody></mujoco>''')
  if alternate:
    model.opt.disableflags |= int(mujoco.mjtDisableBit.mjDSBL_NATIVECCD)
  return model


@pytest.mark.parametrize('shape,size', _SHAPES)
@pytest.mark.parametrize('alternate', [False, True])
def test_primitive_terrain_reserves_the_complete_pinned_contact_bound(shape, size, alternate):
  model = _model(shape, size, alternate)
  data = mujoco.MjData(model)
  mujoco.mj_forward(model, data)
  old_bound = 16 if shape == 'box' else 8
  assert old_bound < data.ncon <= 50
  assert np.linalg.norm(data.qfrc_constraint) > 1
  desc = lower_coupled_constraints(model, limits=_LIMITS)
  assert tuple(desc.pair_max_contacts) == (50,)
  assert desc.ncontacts_max == 50


@pytest.mark.gpu
@pytest.mark.skipif(os.getenv('MUJOCO_METAL_RUN_GPU') != '1', reason='native opt-in')
@pytest.mark.parametrize('shape,size', _SHAPES)
@pytest.mark.parametrize('alternate', [False, True])
@pytest.mark.parametrize('reversed_order', [False, True])
@pytest.mark.parametrize('margin', [0., .005])
def test_full_primitive_terrain_contacts_forces_trajectory_and_restore(
    shape, size, alternate, reversed_order, margin):
  from mujoco_metal.simulation import MetalSimulation

  model = _model(shape, size, alternate, reversed_order, margin)
  qpos = np.tile(model.qpos0.astype(np.float32), (2, 1))
  qpos[1, 2] = .8
  qvel = np.tile(np.asarray([.02, -.01, .03, .04, -.02, .01], np.float32), (2, 1))
  refs = [mujoco.MjData(model) for _ in range(2)]
  for world, data in enumerate(refs):
    data.qpos[:], data.qvel[:] = qpos[world], qvel[world]
    mujoco.mj_forward(model, data)
  assert refs[0].ncon > (16 if shape == 'box' else 8)
  assert refs[1].ncon == 0
  sim = MetalSimulation(model, batch_size=2, qpos=qpos, qvel=qvel,
                        profile='integrated_scalable_v1', limits=_LIMITS)
  host = lambda tensor: tensor.detach().cpu().numpy().copy()
  system = sim.assembled_system(recompute=True)
  np.testing.assert_array_equal(host(system['status']), 0)
  active = host(system['contact_mask']) > .5
  assert active[0].sum() == refs[0].ncon and not active[1].any()
  desc = sim._coupled_constraints.descriptor
  first_geom = int(desc.geom1[0])
  contacts = list(refs[0].contact[:refs[0].ncon])
  for field, expected, tolerance in (
      ('contact_distance', [c.dist for c in contacts], 2e-5),
      ('contact_position', [c.pos for c in contacts], 2e-4),
      ('contact_normal', [c.frame[:3] * (1 if int(c.geom[0]) == first_geom else -1)
                          for c in contacts], 2e-4)):
    np.testing.assert_allclose(host(system[field])[0, active[0]], expected,
                               rtol=0, atol=tolerance, err_msg=field)
  normals = host(system['contact_normal'])[0, active[0]]
  tangent1 = host(system['contact_tangent1'])[0, active[0]]
  tangent2 = host(system['contact_tangent2'])[0, active[0]]
  np.testing.assert_allclose(np.sum(normals * tangent1, axis=1), 0, atol=2e-6)
  np.testing.assert_allclose(np.cross(normals, tangent1), tangent2, atol=2e-6)
  for field, axis in (('contact_tangent1', slice(3, 6)),
                      ('contact_tangent2', slice(6, 9))):
    np.testing.assert_allclose(host(system[field])[0, active[0]],
                               [c.frame[axis] for c in contacts],
                               rtol=0, atol=2e-4, err_msg=field)
  np.testing.assert_allclose(host(system['qfrc_constraint']),
                             [d.qfrc_constraint for d in refs],
                             rtol=2e-3, atol=2e-3)
  for _ in range(4):
    np.testing.assert_array_equal(host(sim.step()), 0)
    for data in refs:
      mujoco.mj_step(model, data)
    for field in ('qpos', 'qvel', 'qacc'):
      np.testing.assert_allclose(host(getattr(sim.state, '_' + field)),
                                 [getattr(d, field) for d in refs],
                                 rtol=2e-3, atol=2e-3, err_msg=field)
  checkpoint = sim.snapshot()
  np.testing.assert_array_equal(host(sim.step()), 0)
  expected = {f: host(getattr(sim.state, '_' + f)) for f in ('qpos', 'qvel', 'qacc')}
  sim.restore(checkpoint)
  np.testing.assert_array_equal(host(sim.step()), 0)
  for field, wanted in expected.items():
    np.testing.assert_array_equal(host(getattr(sim.state, '_' + field)), wanted)
