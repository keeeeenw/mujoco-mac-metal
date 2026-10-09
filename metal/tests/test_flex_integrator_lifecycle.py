# Copyright 2026 The MuJoCo Metal contributors
# Licensed under the Apache License, Version 2.0.
"""Full-size interpolated materials across all public integrated methods."""
import os

import mujoco
import numpy as np
import pytest

_METHODS = [('Euler', 'integrated_scalable_v1'), ('RK4', 'integrated_rk4_v1'),
            ('implicit', 'integrated_implicit_v1'),
            ('implicitfast', 'integrated_implicitfast_v1')]


def _fixture(integrator, order, shell, *, timestep=.0002, young=None):
  count, dof = ('2 2 2', 'trilinear') if order == 1 else ('3 3 3', 'quadratic')
  young = (1_000_000 if shell else 1000) if young is None else young
  bending = ' thickness=".01" elastic2d="bend"' if shell else ''
  model = mujoco.MjModel.from_xml_string(f'''<mujoco>
    <option integrator="{integrator}" timestep="{timestep}" gravity="0 0 0"/>
    <worldbody><flexcomp name="material" type="grid" count="{count}"
        spacing=".1 .1 .1" mass="1" dim="3" dof="{dof}">
      <contact contype="0" conaffinity="0" selfcollide="none"/>
      <elasticity young="{young}" poisson=".2" damping=".1"{bending}/>
    </flexcomp></worldbody></mujoco>''')
  qpos = np.tile(model.qpos0.astype(np.float32), (2, 1))
  qpos[0, 3] += .007
  qpos[0, -1] -= .004
  qpos[1, 3] -= .005
  qpos[1, -1] += .003
  qvel = np.stack([np.linspace(-.01, .02, model.nv),
                   np.linspace(.02, -.015, model.nv)]).astype(np.float32)
  return model, qpos, qvel


@pytest.mark.parametrize('timestep', [.00002, .0002])
@pytest.mark.parametrize('integrator,profile', _METHODS)
@pytest.mark.parametrize('order', [1, 2])
@pytest.mark.parametrize('shell', [False, True])
def test_material_integrator_source_fixture_has_nonzero_force(integrator, profile, order, shell, timestep):
  from mujoco_metal.stepping import validate_stepping_profile
  model, qpos, qvel = _fixture(integrator, order, shell, timestep=timestep)
  # The pinned quadratic shell removes the three unused interior-node DOFs.
  assert model.nv == (24 if order == 1 else (78 if shell else 81))
  descriptor = validate_stepping_profile(model, profile=profile)
  assert descriptor.nv == model.nv
  for world in range(2):
    data = mujoco.MjData(model)
    data.qpos[:], data.qvel[:] = qpos[world], qvel[world]
    mujoco.mj_forward(model, data)
    assert np.linalg.norm(data.qfrc_passive) > .01


@pytest.mark.gpu
@pytest.mark.skipif(os.getenv('MUJOCO_METAL_RUN_GPU') != '1', reason='native opt-in')
@pytest.mark.parametrize('timestep', [.00002, .0002])
@pytest.mark.parametrize('integrator,profile', _METHODS)
@pytest.mark.parametrize('order', [1, 2])
@pytest.mark.parametrize('shell', [False, True])
def test_material_nonzero_force_trajectory_and_lifecycle(integrator, profile, order, shell, timestep):
  from mujoco_metal import MetalSimulation
  model, qpos, qvel = _fixture(integrator, order, shell, timestep=timestep)
  refs = [mujoco.MjData(model) for _ in range(2)]
  for world, data in enumerate(refs):
    data.qpos[:], data.qvel[:] = qpos[world], qvel[world]
    mujoco.mj_forward(model, data)
    assert np.linalg.norm(data.qfrc_passive) > .01
  sim = MetalSimulation(model, batch_size=2, qpos=qpos, qvel=qvel, profile=profile)
  assert sim._component_mass_enabled
  host = lambda tensor: tensor.detach().cpu().numpy().copy()
  system = sim.assembled_system(recompute=True)
  np.testing.assert_array_equal(host(system['status']), 0)
  np.testing.assert_allclose(host(sim._flex._qfrc_passive),
                             [data.qfrc_passive for data in refs],
                             rtol=7e-4, atol=5e-5)
  middle = None
  for step in range(6):
    np.testing.assert_array_equal(host(sim.step()), 0)
    for data in refs:
      mujoco.mj_step(model, data)
    np.testing.assert_allclose(host(sim.state._qpos), [data.qpos for data in refs],
                               rtol=(2e-5 if timestep < .0001 else 3e-3),
                               atol=(1e-7 if timestep < .0001 else 3e-5))
    np.testing.assert_allclose(host(sim.state._qvel), [data.qvel for data in refs],
                               rtol=(3e-4 if timestep < .0001 else 5e-3),
                               atol=(3e-6 if timestep < .0001 else 5e-4))
    if step == 3:
      middle = sim.snapshot()
  final = sim.snapshot()
  # Stiff Q2 RK4 at the larger step intentionally exercises pinned BADQACC
  # reset semantics. It must not be counted as stable material integration.
  np.testing.assert_array_equal(final['device'].warning_number,
                                 [data.warning.number for data in refs])
  if timestep < .0001:
    assert not np.any(final['device'].warning_number)
    assert np.linalg.norm(final['device'].qvel-qvel) > 5e-6
  sim.restore(middle)
  for _ in range(2):
    np.testing.assert_array_equal(host(sim.step()), 0)
  for name in ['qpos', 'qvel', 'qacc']:
    np.testing.assert_array_equal(host(getattr(sim.state, '_' + name)),
                                 getattr(final['device'], name))
  peer = host(sim.state._qpos)[1]
  sim.reset(env_ids=[0], qpos=qpos[:1], qvel=qvel[:1])
  np.testing.assert_array_equal(host(sim.state._qpos)[1], peer)
  reset_ref = mujoco.MjData(model)
  reset_ref.qpos[:], reset_ref.qvel[:] = qpos[0], qvel[0]
  mujoco.mj_step(model, reset_ref)
  np.testing.assert_array_equal(host(sim.step()), 0)
  np.testing.assert_allclose(host(sim.state._qvel)[0], reset_ref.qvel,
                             rtol=5e-3, atol=5e-4)
