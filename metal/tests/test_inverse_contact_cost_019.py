# Copyright 2026 The MuJoCo Metal contributors
# Licensed under the Apache License, Version 2.0.
"""Independent pinned contact-cost gradients through stateless inverse assembly.

CPU smooth/row inputs isolate the inverse force reduction. These tests do not
qualify native contact detection or native row assembly.
"""
from types import SimpleNamespace

import mujoco
import numpy as np
import pytest


@pytest.mark.parametrize('condim', [3, 4, 6])
@pytest.mark.parametrize('cone', ['pyramidal', 'elliptic'])
@pytest.mark.parametrize('region', ['top', 'bottom', 'middle'])
def test_inverse_contact_cost_matches_pinned_force(condim, cone, region):
  torch = pytest.importorskip('torch')
  from mujoco_metal.native_api import _inverse_impl, _inverse_query_workspaces
  model = mujoco.MjModel.from_xml_string(f'''<mujoco>
    <option gravity="0 0 0" cone="{cone}" impratio="3"/>
    <default><geom condim="{condim}" friction=".7 .03 .02"/></default>
    <worldbody><geom type="plane" size="2 2 .1"/>
      <body pos="0 0 .09"><freejoint/><geom size=".1" mass="1"/></body>
    </worldbody></mujoco>''')
  data = mujoco.MjData(model)
  data.qvel[:] = [.03, -.02, -.3, .5, -.1, .2]
  mujoco.mj_inverse(model, data)
  assert data.ncon == 1 and data.nefc > 0
  nr, nv = data.nefc, model.nv
  J = data.efc_J.reshape(nr, nv).copy()
  # Set generalized acceleration to reach all three elliptic-cone regions;
  # this solves independent CPU row geometry, not the native cost helper.
  jar = np.zeros(nr)
  if region == 'top':
    jar[:] = 1000
    if cone == 'elliptic':
      jar[1:] = 0
  elif region == 'bottom':
    jar[:] = -1000
    if cone == 'elliptic':
      jar[1:] = 0
  else:
    jar[:] = np.linspace(-10, 30, nr)
    if cone == 'elliptic':
      jar[0] = 0
    else:
      jar[:] = -5 + np.tile([-20., 20.], nr//2)
  data.qacc[:] = np.linalg.lstsq(J, jar+data.efc_aref, rcond=None)[0]
  mujoco.mj_inverse(model, data)
  if region == 'top':
    assert np.linalg.norm(data.qfrc_constraint) < 1e-8
  else:
    assert np.linalg.norm(data.qfrc_constraint) > 1
  tensor = lambda value: torch.tensor(np.asarray(value).copy()[None], dtype=torch.float32)
  mass = np.empty((nv, nv))
  mujoco.mj_fullM(model, data, mass)
  dynamics = dict(mass_matrix=tensor(mass), qfrc_bias=tensor(data.qfrc_bias),
      root_com=tensor(data.subtree_com), cvel=tensor(data.cvel),
      cdof=tensor(data.cdof), cdof_dot=tensor(data.cdof_dot), poses={})
  debug = torch.zeros((1, nr*nr+7*nr), dtype=torch.float32)
  debug[:, nr*nr:nr*nr+nr] = tensor(data.efc_R)
  debug[:, nr*nr+nr:nr*nr+2*nr] = tensor(data.efc_aref)
  debug[:, nr*nr+4*nr:nr*nr+5*nr] = 0 if cone == 'pyramidal' else -float('inf')
  debug[:, nr*nr+5*nr:nr*nr+6*nr] = float('inf')
  workspace = {'workspace_J': tensor(J).reshape(1, -1), 'workspace_debug': debug}
  cone_id = int(model.opt.cone)
  cc = SimpleNamespace(_workspace=workspace,
      descriptor=SimpleNamespace(nr=nr, nv=nv, n_eq_rows=0, ncontacts_max=1,
          cone_type=cone_id, nr_joint=0,
          contact_condim_packed=np.array([[condim, 0, cone_id]], dtype=np.int32)),
      assemble_device=lambda *_args, **_kwargs: dict(active=torch.ones((1, nr)),
          J=tensor(J), R=tensor(data.efc_R), ar=tensor(data.efc_aref),
          lo=debug[:, nr*nr+4*nr:nr*nr+5*nr], hi=debug[:, nr*nr+5*nr:nr*nr+6*nr],
          contact_mask=torch.ones((1, 1)), contact_friction=tensor(data.contact[0].friction)))
  sim = SimpleNamespace(model=model, _mjmodel=model, batch_size=1, _passive=None,
      _smooth=SimpleNamespace(run_device=lambda *_args: dynamics), _coupled_constraints=cc,
      state=SimpleNamespace(_qpos=tensor(data.qpos), _qvel=tensor(data.qvel), _qacc=tensor(data.qacc)))
  with _inverse_query_workspaces(sim):
    result = _inverse_impl(sim, None, None, None, None, None)
  np.testing.assert_allclose(result.numpy()[0], data.qfrc_inverse, atol=4e-3, rtol=3e-5)
