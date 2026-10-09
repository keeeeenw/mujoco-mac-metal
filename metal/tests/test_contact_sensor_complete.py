# Copyright 2026 The MuJoCo Metal contributors
# Licensed under the Apache License, Version 2.0.
"""Actual pinned contact queries: complete slots, fields, reductions and masks."""
import os

import mujoco
import numpy as np
import pytest

from test_sensor_spatial_016 import _feed
from mujoco_metal.sensors import sensor_oracle


def _fixture(cone, condim, inactive):
  geoms = ''.join(f'<geom name="g{i}" type="sphere" size=".05" '
                  f'pos="{(i-15)*.03} 0 {i*.0001}" mass=".1"/>' for i in range(30))
  queries = ''.join(f'<contact num="4" data="found force torque dist pos normal tangent" '
                    f'reduce="{reduction}" {objects}/>'
                    for reduction in ("none", "mindist", "maxforce", "netforce")
                    for objects in ('', 'geom1="g29"', 'body2="many"', 'subtree1="many"',
                                    'site="zone"'))
  pairs = ('<contact>' + ''.join(f'<pair geom1="floor" geom2="g{i}" '
                               f'condim="{condim}" margin="0" gap=".06"/>' for i in range(30))
           + '</contact>') if inactive else ''
  return mujoco.MjModel.from_xml_string(f'''<mujoco>
    <option cone="{cone}" solver="PGS" iterations="100" gravity="0 0 -9.81"/>
    <default><geom condim="{condim}" friction=".7 .01 .001"/></default>
    <worldbody><geom name="floor" type="plane" size="2 2 .1"/>
      <site name="zone" type="box" pos="0 0 .01" size="2 2 .04"/>
      <body name="many" pos="0 0 {'.08' if inactive else '.04'}"><freejoint/>{geoms}</body>
    </worldbody>{pairs}<sensor>{queries}</sensor></mujoco>''')


def _states(model):
  qpos = np.tile(model.qpos0, (2, 1))
  qpos[1, 2] += .001
  qvel = np.tile([.08, -.03, -.02, .13, .04, -.05], (2, 1))
  qvel[1] *= -.7
  return qpos, qvel, np.zeros(2)


@pytest.mark.parametrize("inactive", [False, True])
def test_contact_host_oracle_all_fields_slots_and_detected_inactive_contacts(inactive):
  model = _fixture("elliptic", 6, inactive)
  qp, qv, times = _states(model)
  poses, extra, expected = _feed(model, qp, qv, times)
  assert np.all(extra["ncon"] == 30)
  assert np.all(expected[:, 0] == 30)
  if inactive:
    assert np.all(extra["contact_efc"] < 0)
    assert np.all(extra["contact_force"] == 0)
  else:
    assert np.max(np.abs(extra["contact_force"])) > 1
  actual = sensor_oracle(model, qp, qv, times, poses, extra=extra,
                         stages=(mujoco.mjtStage.mjSTAGE_ACC,))
  np.testing.assert_allclose(actual, expected, rtol=3e-8, atol=3e-8)


@pytest.mark.parametrize("cone,condim", [("pyramidal", 1), ("pyramidal", 3),
                                       ("pyramidal", 6), ("elliptic", 6)])
@pytest.mark.parametrize("inactive", [False, True])
def test_actual_native_contact_query_all_matches_slots_fields_and_reductions(cone, condim, inactive):
  if os.environ.get("MUJOCO_METAL_RUN_GPU") != "1":
    pytest.skip("explicit native qualification required")
  import torch
  from mujoco_metal.sensors import SensorProgram
  model = _fixture(cone, condim, inactive)
  qp, qv, times = _states(model)
  poses, extra, expected = _feed(model, qp, qv, times)
  count = int(extra["ncon"][0])
  assert count == 30 and np.all(extra["ncon"] == count)
  records = np.zeros((2, count, 42))
  frames = np.zeros((2, count, 12))
  force = np.zeros((2, count, 11))
  for world in range(2):
    data = mujoco.MjData(model)
    data.qpos[:] = qp[world];data.qvel[:] = qv[world]
    mujoco.mj_forward(model, data)
    for slot, contact in enumerate(data.contact):
      records[world, slot, 0] = contact.efc_address >= 0
      records[world, slot, 1] = contact.dist
      frames[world, slot, :9] = contact.frame
      frames[world, slot, 9:] = contact.pos
      if contact.efc_address >= 0:
        address = int(contact.efc_address)
        if cone == "pyramidal" and condim > 1:
          ne = 2 * (condim-1)
          force[world, slot, 1:ne+1] = data.efc_force[address:address+ne]
          force[world, slot, 0] = np.sum(force[world, slot, 1:ne+1])
        else:
          force[world, slot, :condim] = data.efc_force[address:address+condim]
  tensor = lambda value, dtype=torch.float32: torch.tensor(
      np.asarray(value).copy(), dtype=dtype, device="mps")
  program = SensorProgram(model, batch_size=2)
  contact = {"frame": tensor(frames), "force": tensor(force), "records": tensor(records),
             "row": tensor(records[..., 0]),
             "packed": tensor(np.tile([condim, 0, int(model.opt.cone)], (count, 1)), torch.int32),
             "mu": tensor(np.tile([.7, .7, .01, .001, .001], (count, 1))),
             "slot_pair": tensor(np.arange(count), torch.int32),
             "pair_live": tensor(np.ones(count), torch.int32),
             "pair_geoms": tensor(extra["contact_geom"][0], torch.int32)}
  gpu_poses = {name: tensor(value) for name, value in poses.items()}
  actual = program.run_contact_device(gpu_poses, contact)
  np.testing.assert_allclose(actual.cpu().numpy(), expected, rtol=3e-5, atol=3e-5)
  retained = actual.clone()
  contact["force"][0].zero_()
  contact["frame"][0].zero_()
  selected = program.run_contact_device(gpu_poses, contact,
      world_mask=torch.tensor([False, True], device="mps"))
  torch.testing.assert_close(selected, retained, rtol=0, atol=0)
  unmasked = program.run_contact_device(gpu_poses, contact)
  assert float(unmasked[0, 0]) == 0
  assert float(retained[0, 0]) == 30
