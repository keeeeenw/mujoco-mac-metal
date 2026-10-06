"""CPU oracle and rollback tests for model inertia lifecycle updates."""

import numpy as np
import mujoco
import pytest

from mujoco_metal.lifecycle import ModelLifecycle


_XML = """<mujoco>
  <worldbody><body name="free">
    <freejoint/><inertial pos=".1 0 0" mass="2" diaginertia=".2 .3 .4"/>
    <geom name="inertia-aligned" type="sphere" size=".05" pos=".1 0 0"/>
    <site name="inertia-site" pos=".1 0 0" size=".01"/>
  </body></worldbody>
</mujoco>"""


def _mass_matrix(model):
  data = mujoco.MjData(model)
  mujoco.mj_forward(model, data)
  dense = np.empty((model.nv, model.nv), dtype=np.float64)
  mujoco.mj_fullM(model, data, dense)
  return dense


def test_inertial_update_recomputes_derived_constants_and_changes_cpu_dynamics():
  source = mujoco.MjModel.from_xml_string(_XML)
  life = ModelLifecycle(source)
  before_model = life._model
  before_generation = life.generation
  before_mass = _mass_matrix(before_model)

  theta = .37
  quat = np.array([[np.cos(theta / 2), 0., 0., np.sin(theta / 2)]])
  updated = np.array([[.8, .5, .35]])
  assert life.recompute_body_inertias([1], updated, inertial_quats=quat)
  assert life._model is not before_model
  assert life.generation == before_generation + 1
  np.testing.assert_allclose(life._model.body_inertia[1], updated[0], rtol=0, atol=2e-8)
  np.testing.assert_allclose(life._model.body_iquat[1], quat[0], rtol=0, atol=1e-7)
  np.testing.assert_allclose(life.descriptor.body_inertia[1], updated[0], rtol=0, atol=2e-8)

  # Compile the equivalent updated source independently. mj_setConst alone
  # does not refresh compiler classifications when the inertial frame changes.
  expected_xml = f"""<mujoco><worldbody><body name="free">
    <freejoint/><inertial pos=".1 0 0" mass="2" diaginertia=".8 .5 .35"
      quat="{quat[0, 0]} {quat[0, 1]} {quat[0, 2]} {quat[0, 3]}"/>
    <geom name="inertia-aligned" type="sphere" size=".05" pos=".1 0 0"/>
    <site name="inertia-site" pos=".1 0 0" size=".01"/>
  </body></worldbody></mujoco>"""
  expected_model = mujoco.MjModel.from_xml_string(expected_xml)
  for name in ("body_sameframe", "body_simple", "geom_sameframe", "site_sameframe"):
    np.testing.assert_array_equal(getattr(life._model, name),
                                  getattr(expected_model, name))
  after_mass = _mass_matrix(life._model)
  assert not np.allclose(before_mass, after_mass, rtol=0, atol=1e-8)
  np.testing.assert_allclose(after_mass, _mass_matrix(expected_model),
                             rtol=0, atol=2e-7)
  represented_qvel = np.array([.12, -.08, .04, .2, -.15, .09], dtype=np.float64)
  represented_force = np.array([.3, -.2, .1, .7, -.4, .25], dtype=np.float64)
  updated_data = mujoco.MjData(life._model)
  expected_data = mujoco.MjData(expected_model)
  for data in (updated_data, expected_data):
    data.qvel[:] = represented_qvel
    data.qfrc_applied[:] = represented_force
    mujoco.mj_forward(life._model if data is updated_data else expected_model, data)
  np.testing.assert_allclose(updated_data.qacc, expected_data.qacc,
                             rtol=0, atol=2e-7)

  assert not life.recompute_body_inertias([1], updated, inertial_quats=quat)
  assert life.generation == before_generation + 1


def test_inertial_update_rejects_invalid_inputs_and_rolls_back_setconst_failure(monkeypatch):
  model = mujoco.MjModel.from_xml_string(_XML)
  life = ModelLifecycle(model)
  original_model, original_descriptor = life._model, life.descriptor
  original_generation = life.generation

  for ids, values, kwargs, match in (
      ([0], [[1., 1., 1.]], {}, "non-world"),
      ([1], [[1., -1., 1.]], {}, "nonnegative"),
      ([1], [[np.nan, 1., 1.]], {}, "finite"),
      ([1, 1], [[1., 1., 1.], [2., 2., 2.]], {}, "unique"),
      ([1], [[1., 1.]], {}, "shapes"),
      ([1], [[1., 1., 1.]], {"inertial_quats": [[0., 0., 0., 0.]]}, "nonzero"),
  ):
    with pytest.raises(ValueError, match=match):
      life.recompute_body_inertias(ids, values, **kwargs)
    assert life._model is original_model
    assert life.descriptor is original_descriptor
    assert life.generation == original_generation

  def fail_setconst(*_args, **_kwargs):
    raise RuntimeError("injected setConst failure")

  monkeypatch.setattr(mujoco, "mj_setConst", fail_setconst)
  with pytest.raises(RuntimeError, match="injected"):
    life.recompute_body_inertias([1], [[.9, .6, .4]])
  assert life._model is original_model
  assert life.descriptor is original_descriptor
  assert life.generation == original_generation


def test_mass_update_rejects_nonrepresentable_native_constants_atomically():
  life = ModelLifecycle(mujoco.MjModel.from_xml_string(_XML))
  original_model, original_descriptor = life._model, life.descriptor
  original_generation = life.generation
  # Both are finite, positive float64 inputs accepted by MuJoCo's model
  # storage, but cannot represent a positive finite native float32 mass.
  for mass in (1e100, 1e-100):
    with pytest.raises(ValueError, match="float32"):
      life.recompute_body_masses([1], [mass])
    assert life._model is original_model
    assert life.descriptor is original_descriptor
    assert life.generation == original_generation


def test_sparse_inertia_layout_transition_is_rejected_atomically_against_source():
  base_xml = """<mujoco><worldbody><body name="free">
    <freejoint/><inertial pos="0 0 0" mass="2" diaginertia=".2 .3 .4"/>
  </body></worldbody></mujoco>"""
  theta = .37
  quat = np.array([np.cos(theta / 2), 0., 0., np.sin(theta / 2)])
  rotated_xml = f"""<mujoco><worldbody><body name="free">
    <freejoint/><inertial pos="0 0 0" mass="2" diaginertia=".8 .5 .35"
      quat="{' '.join(map(str, quat))}"/>
  </body></worldbody></mujoco>"""
  base = mujoco.MjModel.from_xml_string(base_xml)
  rotated = mujoco.MjModel.from_xml_string(rotated_xml)
  assert int(base.body_simple[1]) != int(rotated.body_simple[1])
  assert not np.array_equal(base.dof_simplenum, rotated.dof_simplenum)
  assert int(base.nC) != int(rotated.nC)

  # These are real source-factorized mass solves, not only layout metadata.
  rhs = (np.arange(base.nv, dtype=np.float64) + 1.0)[None, :]
  solutions = []
  for model in (base, rotated):
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    solution = np.empty_like(rhs)
    mujoco.mj_solveM(model, data, solution, rhs)
    dense = np.empty((model.nv, model.nv), dtype=np.float64)
    mujoco.mj_fullM(model, data, dense)
    np.testing.assert_allclose(dense @ solution[0], rhs[0], rtol=0, atol=1e-12)
    solutions.append(solution.copy())
  assert not np.allclose(solutions[0], solutions[1], rtol=0, atol=1e-8)

  life = ModelLifecycle(base)
  original_model, original_descriptor = life._model, life.descriptor
  original_generation = life.generation
  with pytest.raises(ValueError, match="compiled sparse inertia layout"):
    life.recompute_body_inertias([1], [[.8, .5, .35]], inertial_quats=[quat])
  assert life._model is original_model
  assert life.descriptor is original_descriptor
  assert life.generation == original_generation
  np.testing.assert_array_equal(life._model.dof_simplenum, base.dof_simplenum)


def test_tendon_armature_demotion_is_preserved_during_reference_updates():
  xml = """<mujoco><worldbody><body name="slider">
    <joint type="slide" axis="1 0 0"/>
    <inertial pos="0 0 0" mass="1" diaginertia=".1 .1 .1"/>
    <site name="s1" pos=".2 0 0" size=".01"/>
    <site name="s2" pos=".3 0 0" size=".01"/>
  </body></worldbody><tendon><spatial name="t" width=".003" armature=".1">
    <site site="s1"/><site site="s2"/>
  </spatial></tendon></mujoco>"""
  source = mujoco.MjModel.from_xml_string(xml)
  life = ModelLifecycle(source)
  assert int(source.body_simple[1]) == 0
  np.testing.assert_array_equal(source.dof_simplenum, [0])
  assert life.update_reference_frames(site_ids=[0], site_pos=[[.25, .02, 0.]])
  np.testing.assert_array_equal(life._model.body_simple, source.body_simple)
  np.testing.assert_array_equal(life._model.dof_simplenum, source.dof_simplenum)
  assert int(life._model.nC) == int(source.nC)


@pytest.mark.gpu
@pytest.mark.skipif(__import__("os").getenv("MUJOCO_METAL_RUN_GPU") != "1",
                    reason="opt-in GPU")
def test_public_apply_lifecycle_inertia_quat_force_and_restore_replay_gpu():
  """Public lifecycle swaps mass caches, then snapshot replay stays exact."""
  from mujoco_metal import MetalSimulation

  source = mujoco.MjModel.from_xml_string(_XML)
  sim = MetalSimulation(source, batch_size=2, profile="integrated_euler_v1")
  qpos = np.tile(np.array([0., 0., 0., 1., 0., 0., 0.], np.float32), (2, 1))
  qvel = np.array([[.1, -.2, .05, .2, .1, -.3],
                   [-.05, .15, -.1, -.1, .25, .12]], np.float32)
  force = np.array([[.3, -.1, .2, .8, -.4, .5],
                    [-.2, .25, .1, -.5, .7, -.3]], np.float32)
  sim.reset(qpos=qpos, qvel=qvel)

  life = ModelLifecycle(source)
  theta = .37
  quat = np.array([[np.cos(theta / 2), 0., 0., np.sin(theta / 2)]])
  updated = np.array([[.8, .5, .35]])
  assert life.recompute_body_inertias([1], updated, inertial_quats=quat)
  old_generation = sim.state.generation
  sim.apply_lifecycle(life)
  assert sim.state.generation > old_generation
  np.testing.assert_array_equal(sim.state.qpos.cpu().numpy(), qpos)
  np.testing.assert_array_equal(sim.state.qvel.cpu().numpy(), qvel)

  cpu = [mujoco.MjData(life._model) for _ in range(2)]
  for world, data in enumerate(cpu):
    data.qpos[:] = qpos[world]
    data.qvel[:] = qvel[world]
    data.qfrc_applied[:] = force[world]
    mujoco.mj_forward(life._model, data)
  for _ in range(1):
    sim.step(1, qfrc_applied=force)
    for data in cpu:
      mujoco.mj_step(life._model, data)
  np.testing.assert_allclose(sim.state.qpos.cpu().numpy(),
                             np.stack([d.qpos for d in cpu]),
                             rtol=3e-4, atol=3e-4)
  np.testing.assert_allclose(sim.state.qvel.cpu().numpy(),
                             np.stack([d.qvel for d in cpu]),
                             rtol=3e-4, atol=3e-4)
  np.testing.assert_allclose(sim.state.qacc.cpu().numpy(),
                             np.stack([d.qacc for d in cpu]),
                             rtol=5e-4, atol=2e-3)

  checkpoint = sim.snapshot()
  sim.step(1, qfrc_applied=force)
  expected = {name: getattr(sim.state, name).cpu().numpy().copy()
              for name in ("qpos", "qvel", "qacc", "time")}
  sim.restore(checkpoint)
  sim.step(1, qfrc_applied=force)
  for name, value in expected.items():
    np.testing.assert_array_equal(getattr(sim.state, name).cpu().numpy(), value)
