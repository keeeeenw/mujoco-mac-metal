"""Source-oracle tests for transactional compiled reference-frame updates."""

import numpy as np
import mujoco
import pytest

from mujoco_metal.lifecycle import ModelLifecycle


_XML = """<mujoco><worldbody>
  <body name="anchor" pos=".2 -.1 .3">
    <joint name="hinge" type="hinge" axis="0 0 1"/>
    <geom name="g" type="box" size=".1 .2 .3" pos=".1 .05 0"/>
    <site name="s" pos="-.1 .15 .05" size=".01"/>
    <body name="child" pos=".1 0 0">
      <joint type="slide" axis="1 0 0"/>
      <geom type="sphere" size=".03"/>
    </body>
  </body>
</worldbody></mujoco>"""


def _compare_fk(lifecycle, qpos):
  model = lifecycle._model
  data = mujoco.MjData(model)
  data.qpos[:] = qpos
  mujoco.mj_forward(model, data)
  result = lifecycle.descriptor.forward_kinematics(qpos)
  np.testing.assert_allclose(result["body_pos"], data.xpos, rtol=0, atol=2e-6)
  np.testing.assert_allclose(result["body_quat"], data.xquat, rtol=0, atol=2e-6)
  np.testing.assert_allclose(result["geom_pos"], data.geom_xpos, rtol=0, atol=2e-6)
  np.testing.assert_allclose(result["site_pos"], data.site_xpos, rtol=0, atol=2e-6)


def _compiled_fk(model, qpos):
  data = mujoco.MjData(model)
  data.qpos[:] = qpos
  mujoco.mj_forward(model, data)
  return data


def _with_explicit_inertias(xml, model, body_ids):
  """Make an independent XML source while pinning pre-edit mass properties."""
  for body in body_ids:
    name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body)
    if name is None:
      raise AssertionError(f"fixture body {body} needs a stable name")
    pos = " ".join(f"{x:.17g}" for x in model.body_ipos[body])
    quat = " ".join(f"{x:.17g}" for x in model.body_iquat[body])
    inertia = " ".join(f"{x:.17g}" for x in model.body_inertia[body])
    mass = f"{model.body_mass[body]:.17g}"
    open_tag = f'<body name="{name}"'
    tag_end = xml.index(">", xml.index(open_tag)) + 1
    xml = (xml[:tag_end] + f'<inertial pos="{pos}" quat="{quat}" '
           f'mass="{mass}" diaginertia="{inertia}"/>' + xml[tag_end:])
  return xml


def test_reference_frame_updates_relower_and_match_pinned_fk():
  lifecycle = ModelLifecycle(mujoco.MjModel.from_xml_string(_XML))
  source_model = lifecycle._model
  old_model = lifecycle._model
  old_generation = lifecycle.generation
  qpos = np.array([.37, -.08], dtype=np.float64)
  _compare_fk(lifecycle, qpos)

  angle = .29
  body_quat = np.array([[np.cos(angle / 2), 0., np.sin(angle / 2), 0.]])
  geom_quat = np.array([[np.cos(angle / 2), 0., 0., np.sin(angle / 2)]])
  site_quat = np.array([[np.cos(angle / 2), np.sin(angle / 2), 0., 0.]])
  assert lifecycle.update_reference_frames(
      body_ids=[1], body_pos=[[.4, -.2, .1]], body_quat=body_quat,
      geom_ids=[0], geom_pos=[[.2, .03, -.04]], geom_quat=geom_quat,
      site_ids=[0], site_pos=[[-.2, .1, .08]], site_quat=site_quat)
  assert lifecycle._model is not old_model
  assert lifecycle.generation == old_generation + 1
  _compare_fk(lifecycle, qpos)

  # Compare compiler-derived frame classifications against a fresh compile
  # of the same represented model. mj_setConst does not update these flags.
  expected_xml = _with_explicit_inertias(_XML, source_model, [1, 2]).replace(
      'pos=".2 -.1 .3"', 'pos=".4 -.2 .1"').replace(
      'pos=".1 .05 0"', 'pos=".2 .03 -.04"').replace(
      'pos="-.1 .15 .05"', 'pos="-.2 .1 .08"')
  expected_xml = expected_xml.replace(
      '<geom name="g" type="box" size=".1 .2 .3" pos=".2 .03 -.04"/>',
      '<geom name="g" type="box" size=".1 .2 .3" pos=".2 .03 -.04" '
      f'quat="{geom_quat[0,0]} {geom_quat[0,1]} {geom_quat[0,2]} '
      f'{geom_quat[0,3]}"/>')
  expected_xml = expected_xml.replace(
      '<site name="s" pos="-.2 .1 .08" size=".01"/>',
      '<site name="s" pos="-.2 .1 .08" size=".01" '
      f'quat="{site_quat[0,0]} {site_quat[0,1]} {site_quat[0,2]} '
      f'{site_quat[0,3]}"/>')
  expected_xml = expected_xml.replace(
      '<body name="anchor" pos=".4 -.2 .1">',
      '<body name="anchor" pos=".4 -.2 .1" '
      f'quat="{body_quat[0,0]} {body_quat[0,1]} {body_quat[0,2]} '
      f'{body_quat[0,3]}">')
  expected_model = mujoco.MjModel.from_xml_string(expected_xml)
  for name in ("body_sameframe", "body_simple", "geom_sameframe", "site_sameframe"):
    np.testing.assert_array_equal(getattr(lifecycle._model, name),
                                  getattr(expected_model, name))
  updated_data = _compiled_fk(lifecycle._model, qpos)
  expected_data = _compiled_fk(expected_model, qpos)
  for name in ("xpos", "xquat", "xmat", "geom_xpos", "geom_xmat",
               "site_xpos", "site_xmat"):
    np.testing.assert_allclose(getattr(updated_data, name),
                               getattr(expected_data, name), rtol=0, atol=2e-7)

  # A represented no-op does not replace the compiled model or invalidate caches.
  generation = lifecycle.generation
  current_body_pos = lifecycle._model.body_pos[[1]].copy()
  assert not lifecycle.update_reference_frames(body_ids=[1], body_pos=current_body_pos)
  assert lifecycle.generation == generation


def test_reference_frame_validation_and_setconst_failure_are_atomic(monkeypatch):
  lifecycle = ModelLifecycle(mujoco.MjModel.from_xml_string(_XML))
  before_model, before_descriptor = lifecycle._model, lifecycle.descriptor
  before_generation = lifecycle.generation
  for kwargs, match in (
      ({"body_ids": [0], "body_pos": [[1., 0., 0.]]}, "world body"),
      ({"geom_ids": [0.5], "geom_pos": [[1., 0., 0.]]}, "integers"),
      ({"site_ids": [0], "site_pos": [[1., 2.]]}, "shape"),
      ({"body_ids": [1], "body_quat": [[0., 0., 0., 0.]]}, "nonzero"),
      ({"geom_ids": [0], "geom_pos": [[np.inf, 0., 0.]]}, "finite"),
      ({"body_ids": [1], "body_pos": [[1e100, 0., 0.]]}, "float32"),
  ):
    with pytest.raises(ValueError, match=match):
      lifecycle.update_reference_frames(**kwargs)
    assert lifecycle._model is before_model
    assert lifecycle.descriptor is before_descriptor
    assert lifecycle.generation == before_generation

  def fail_setconst(*_args, **_kwargs):
    raise RuntimeError("injected const failure")

  monkeypatch.setattr(mujoco, "mj_setConst", fail_setconst)
  with pytest.raises(RuntimeError, match="injected"):
    lifecycle.update_reference_frames(body_ids=[1], body_pos=[[.3, 0., 0.]])
  assert lifecycle._model is before_model
  assert lifecycle.descriptor is before_descriptor
  assert lifecycle.generation == before_generation


@pytest.mark.gpu
@pytest.mark.skipif(__import__("os").getenv("MUJOCO_METAL_RUN_GPU") != "1",
                    reason="opt-in GPU")
def test_public_reference_frame_update_refreshes_contact_cache_and_replays_gpu():
  """Reference-frame updates rebuild public contact/J/force state atomically."""
  from mujoco_metal import MetalSimulation

  xml = """<mujoco><option timestep=".002" gravity="0 0 -9.81"
      solver="PGS" iterations="120"/><worldbody>
    <geom name="floor" type="plane" size="2 2 .1"/>
    <body name="slider" pos="0 0 .09">
      <joint name="slide" type="slide" axis="0 0 1"/>
      <geom name="ball" type="sphere" size=".1" mass="1"/>
      <site name="marker" pos=".02 0 .01" size=".01"/>
    </body></worldbody></mujoco>"""
  source = mujoco.MjModel.from_xml_string(xml)
  body = mujoco.mj_name2id(source, mujoco.mjtObj.mjOBJ_BODY, "slider")
  geom = mujoco.mj_name2id(source, mujoco.mjtObj.mjOBJ_GEOM, "ball")
  site = mujoco.mj_name2id(source, mujoco.mjtObj.mjOBJ_SITE, "marker")
  sim = MetalSimulation(source, batch_size=2, profile="integrated_euler_v1")
  qpos = np.array([[0.], [.001]], np.float32)
  qvel = np.array([[-.02], [.03]], np.float32)
  force = np.array([[.1], [-.2]], np.float32)
  sim.reset(qpos=qpos, qvel=qvel)

  life = ModelLifecycle(source)
  assert life.update_reference_frames(
      body_ids=[body], body_pos=[[0., 0., .085]],
      geom_ids=[geom], geom_pos=[[.015, 0., .005]],
      site_ids=[site], site_pos=[[.03, .01, .02]])
  generation = sim.state.generation
  sim.apply_lifecycle(life)
  assert sim.state.generation > generation
  np.testing.assert_array_equal(sim.state.qpos.cpu().numpy(), qpos)
  np.testing.assert_array_equal(sim.state.qvel.cpu().numpy(), qvel)

  cpu = [mujoco.MjData(life._model) for _ in range(2)]
  for world, data in enumerate(cpu):
    data.qpos[:] = qpos[world]
    data.qvel[:] = qvel[world]
    data.qfrc_applied[:] = force[world]
    mujoco.mj_forward(life._model, data)
  assembled = sim.assembled_system(qfrc_applied=force, recompute=True)
  assert np.all(assembled["status"].detach().cpu().numpy() == 0)
  assert all(int(data.nefc) > 0 for data in cpu)
  np.testing.assert_allclose(
      assembled["qacc"].detach().cpu().numpy(),
      np.stack([data.qacc for data in cpu]), rtol=5e-4, atol=3e-3)
  np.testing.assert_allclose(
      assembled["qfrc_constraint"].detach().cpu().numpy(),
      np.stack([data.qfrc_constraint for data in cpu]),
      rtol=5e-4, atol=3e-3)

  # Step, restore a checkpoint under the rebuilt model, and replay the same
  # held force. This exercises the refreshed model descriptor and caches via
  # the public lifecycle and snapshot APIs rather than the lowering helper.
  checkpoint = sim.snapshot()
  sim.step(1, qfrc_applied=force)
  expected = {name: getattr(sim.state, name).cpu().numpy().copy()
              for name in ("qpos", "qvel", "qacc", "time")}
  sim.restore(checkpoint)
  sim.step(1, qfrc_applied=force)
  for name, value in expected.items():
    np.testing.assert_array_equal(getattr(sim.state, name).cpu().numpy(), value)
