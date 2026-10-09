# Copyright 2026 The MuJoCo Metal contributors
# Licensed under the Apache License, Version 2.0.
"""Non-Euler integrated profiles use actual compiled component capacities."""
import os

import mujoco
import numpy as np
import pytest

from mujoco_metal.capacity import CapacityLimits, CapacityOverflow
from mujoco_metal.stepping import validate_stepping_profile

_PROFILES = [('RK4', 'integrated_rk4_v1'),
             ('implicit', 'integrated_implicit_v1'),
             ('implicitfast', 'integrated_implicitfast_v1')]


def _model(integrator, count=40):
  bodies = ''.join(
      f'<body pos="{i*.3} 0 0"><joint name="j{i}" type="slide" '
      'axis="0 1 0" damping=".15" stiffness="2" frictionloss=".02" '
      'limited="true" range="-.1 .1"/>'
      '<geom type="sphere" size=".04" mass=".2" contype="0" '
      'conaffinity="0"/></body>' for i in range(count))
  return mujoco.MjModel.from_xml_string(f'''<mujoco>
    <option integrator="{integrator}" timestep=".0005" gravity="0 0 0"
            solver="Newton" iterations="30" tolerance="1e-8"/>
    <worldbody>{bodies}</worldbody></mujoco>''')


@pytest.mark.parametrize('integrator,profile', _PROFILES)
@pytest.mark.parametrize('explicit', [False, True])
def test_profile_above_legacy_dof_cap_preserves_actual_budget(integrator, profile, explicit):
  model = _model(integrator, 130)
  limits = CapacityLimits(max_nv=130, max_rows=512) if explicit else None
  result = validate_stepping_profile(model, profile=profile, limits=limits)
  assert result.nv == 130 and result.name == profile
  assert result.execution_plan.profile_name == profile
  assert 'semi-implicit Euler integration' not in result.supported
  assert result.execution_plan.is_stage_enabled('coupled_constraints')
  with pytest.raises(CapacityOverflow):
    validate_stepping_profile(model, profile=profile,
                             limits=CapacityLimits(max_nv=129, max_rows=512))
  with pytest.raises(CapacityOverflow):
    validate_stepping_profile(model, profile=profile,
                             limits=CapacityLimits(max_nv=130, max_rows=1))


@pytest.mark.gpu
@pytest.mark.skipif(os.getenv('MUJOCO_METAL_RUN_GPU') != '1', reason='native opt-in')
@pytest.mark.parametrize('integrator,profile', _PROFILES)
def test_nonzero_batched_component_integration_and_replay(integrator, profile):
  from mujoco_metal import MetalSimulation

  model = _model(integrator)
  qpos = np.stack([np.linspace(-.12, .12, model.nq),
                   np.linspace(.03, -.03, model.nq)]).astype(np.float32)
  qvel = np.stack([np.linspace(-.04, .06, model.nv),
                   np.linspace(.05, -.03, model.nv)]).astype(np.float32)
  references = [mujoco.MjData(model) for _ in range(2)]
  for world, data in enumerate(references):
    data.qpos[:], data.qvel[:] = qpos[world], qvel[world]
    mujoco.mj_forward(model, data)
  assert np.linalg.norm(references[0].qfrc_constraint) > 1
  assert np.linalg.norm(references[1].qfrc_passive) > .1
  sim = MetalSimulation(model, batch_size=2, qpos=qpos, qvel=qvel, profile=profile)
  assert sim._component_mass_enabled
  host = lambda tensor: tensor.detach().cpu().numpy().copy()
  checkpoint = None
  for step in range(6):
    np.testing.assert_array_equal(host(sim.step()), 0)
    for data in references:
      mujoco.mj_step(model, data)
    for name, atol in [('qpos', 3e-6), ('qvel', 3e-5), ('qacc', 3e-4)]:
      np.testing.assert_allclose(host(getattr(sim.state, '_' + name)),
                                 [getattr(data, name) for data in references],
                                 rtol=5e-4, atol=atol, err_msg=name)
    if step == 3:
      checkpoint = sim.snapshot()
  final = sim.snapshot()
  sim.restore(checkpoint)
  for _ in range(2):
    np.testing.assert_array_equal(host(sim.step()), 0)
  for name in ['qpos', 'qvel', 'qacc']:
    np.testing.assert_array_equal(host(getattr(sim.state, '_' + name)),
                                 getattr(final['device'], name))
  peer = host(sim.state._qpos)[1]
  sim.reset(env_ids=[0], qpos=qpos[:1], qvel=qvel[:1])
  np.testing.assert_array_equal(host(sim.state._qpos)[1], peer)
  reset_reference = mujoco.MjData(model)
  reset_reference.qpos[:], reset_reference.qvel[:] = qpos[0], qvel[0]
  mujoco.mj_step(model, reset_reference)
  np.testing.assert_array_equal(host(sim.step()), 0)
  np.testing.assert_allclose(host(sim.state._qvel)[0], reset_reference.qvel,
                             rtol=5e-4, atol=3e-5)
