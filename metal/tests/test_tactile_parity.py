# Copyright 2026 The MuJoCo Metal contributors
# Licensed under the Apache License, Version 2.0.
"""Taxel layout and contact-stage parity against pinned MuJoCo 3.10.

Tactile channels are penetration depth and absolute tangential velocities,
not contact force. The tolerance is 2e-5 in metres/metres per second for
these small, well-conditioned float32 fixtures, separately from dynamics.
"""

import os

import mujoco
import numpy as np
import pytest

VERTICES = "-.1 -.1 -.1 .1 -.1 -.1 .1 .1 -.1 -.1 .1 -.1 " \
           "-.1 -.1 .1 .1 -.1 .1 .1 .1 .1 -.1 .1 .1"


def _model(shape="plane", framed=True, cutoff=0, disabled=False,
           moving=False, mesh_octree=False, duplicate=False, inactive=False):
  normals = 'normal="' + " ".join(
      "0 0 1 1 0 0 0 1 0" for _ in range(8)) + '"' if framed else ""
  shapes = {
      "plane": 'type="plane" size="2 2 .1"',
      "sphere": 'type="sphere" size=".3"',
      "capsule": 'type="capsule" size=".25 .2"',
      "ellipsoid": 'type="ellipsoid" size=".35 .3 .2"',
      "cylinder": 'type="cylinder" size=".3 .2"',
      "box": 'type="box" size=".3 .3 .2"',
      "mesh": 'type="mesh" mesh="opponent_mesh"',
  }
  asset = (f'<mesh name="opponent_mesh" vertex="{VERTICES}" scale="3 3 2"/>'
           if shape == "mesh" else "")
  # Compiling an SDF geom requests an octree for the shared mesh. A mesh-only
  # asset does not; these are the two pinned tactile branches.
  octree_request = ('<geom type="sdf" mesh="opponent_mesh" pos="10 0 0" '
                    'contype="0" conaffinity="0"/>' if shape == "mesh" and mesh_octree else "")
  opponent = f'<geom name="opponent" {shapes[shape]} mass="1"/>'
  if duplicate:
    # Two sensor-body geoms generate several contacts with the same opponent.
    second = '<geom name="pad2" type="box" size=".08 .08 .05" pos=".01 0 0"/>'
  else:
    second = ""
  if moving:
    opponent = f'<body name="other"><freejoint/>{opponent}</body>'
  model = mujoco.MjModel.from_xml_string(f'''<mujoco>
    <option gravity="0 0 0" timestep=".002" solver="PGS" iterations="100">
      <flag sensor="{'disable' if disabled else 'enable'}"/>
    </option><asset><mesh name="taxels" vertex="{VERTICES}" {normals}/>{asset}</asset>
    <worldbody>{opponent}{octree_request}<body name="sensor_body" pos="0 0 {'.08' if inactive else '.04'}"><freejoint/>
      <geom name="pad" type="box" size=".1 .1 .05" mass="1" condim="3"
            {'margin=".1" gap=".15"' if inactive else ''}/>{second}
    </body></worldbody>
    {'<contact><pair geom1="pad" geom2="opponent" margin="0" gap=".15"/></contact>' if inactive else ''}
    <sensor><tactile name="t" geom="pad" mesh="taxels"/></sensor>
  </mujoco>''')
  # The pinned XML schema omits tactile cutoff, but the compiled field is
  # writable and apply_cutoff processes it for every sensor.
  model.sensor_cutoff[0] = cutoff
  return model


def _initial(model, separated=False, moving=False):
  data = mujoco.MjData(model)
  sensor = model.body("sensor_body").id
  joint = int(model.body_jntadr[sensor])
  qa, va = int(model.jnt_qposadr[joint]), int(model.jnt_dofadr[joint])
  if separated:
    data.qpos[qa+2] = 1.0
  data.qvel[va:va+6] = [.13, -.08, .03, .2, -.1, .4]
  if moving:
    data.qvel[:6] = [-.06, .04, -.02, -.3, .15, -.1]
  return data


def test_cpu_framed_taxels_have_penetration_and_masked_velocities():
  model = _model()
  data = _initial(model)
  mujoco.mj_forward(model, data)
  channels = data.sensordata.reshape(3, 8)
  assert data.ncon > 1  # Deduplication matters for this planar manifold.
  assert np.count_nonzero(channels[0]) == 4
  assert np.all(channels[1:, channels[0] == 0] == 0)
  assert np.max(channels[1:]) > .05


gpu = pytest.mark.skipif(os.getenv("MUJOCO_METAL_RUN_GPU") != "1",
                         reason="opt-in GPU")


@pytest.mark.gpu
@gpu
@pytest.mark.parametrize("shape", ["plane", "sphere", "capsule", "ellipsoid", "cylinder", "box"])
@pytest.mark.parametrize("framed", [False, True])
def test_tactile_analytic_distance_and_frame_channels(shape, framed):
  from mujoco_metal import MetalSimulation
  model = _model(shape, framed)
  data = _initial(model)
  sim = MetalSimulation(model, qpos=data.qpos[None].astype(np.float32),
                        qvel=data.qvel[None].astype(np.float32),
                        profile="integrated_euler_v1")
  sim.step()
  mujoco.mj_step(model, data)
  assert data.ncon > 0
  assert np.max(data.sensordata) > 0
  np.testing.assert_allclose(sim.step_sensordata()[0], data.sensordata,
                             rtol=2e-5, atol=2e-5)


@pytest.mark.gpu
@gpu
@pytest.mark.parametrize("moving,duplicate,cutoff", [(True, False, 0), (False, True, 0), (True, True, .025)])
def test_tactile_other_body_velocity_deduplication_and_cutoff(moving, duplicate, cutoff):
  from mujoco_metal import MetalSimulation
  model = _model("box", cutoff=cutoff, moving=moving, duplicate=duplicate)
  data = _initial(model, moving=moving)
  sim = MetalSimulation(model, qpos=data.qpos[None].astype(np.float32),
                        qvel=data.qvel[None].astype(np.float32),
                        profile="integrated_euler_v1")
  sim.step()
  mujoco.mj_step(model, data)
  assert np.max(data.sensordata[8:]) > 0
  np.testing.assert_allclose(sim.step_sensordata()[0], data.sensordata,
                             rtol=2e-5, atol=2e-5)


@pytest.mark.gpu
@gpu
@pytest.mark.parametrize("octree", [False, True])
def test_tactile_mesh_uses_compiled_distance_octree_or_pinned_exclusion(octree):
  from mujoco_metal import MetalSimulation
  model = _model("mesh", mesh_octree=octree)
  data = _initial(model)
  sim = MetalSimulation(model, qpos=data.qpos[None].astype(np.float32),
                        qvel=data.qvel[None].astype(np.float32),
                        profile="integrated_euler_v1")
  sim.step()
  mujoco.mj_step(model, data)
  assert data.ncon > 0
  assert (np.max(data.sensordata) > 0) == octree
  np.testing.assert_allclose(sim.step_sensordata()[0], data.sensordata,
                             rtol=2e-5, atol=2e-5)


@pytest.mark.gpu
@gpu
def test_tactile_batch_separation_reset_and_restore():
  from mujoco_metal import MetalSimulation
  model = _model()
  active, separated = _initial(model), _initial(model, separated=True)
  qpos = np.array([active.qpos, separated.qpos], dtype=np.float32)
  qvel = np.array([active.qvel, separated.qvel], dtype=np.float32)
  sim = MetalSimulation(model, batch_size=2, qpos=qpos, qvel=qvel,
                        profile="integrated_euler_v1")
  checkpoint = sim.snapshot()
  sim.step()
  first = sim.step_sensordata().copy()
  assert np.max(first[0]) > 0
  assert np.all(first[1] == 0)
  sim.restore(checkpoint)
  sim.step()
  np.testing.assert_array_equal(sim.step_sensordata(), first)
  sim.reset(qpos=qpos[::-1].copy(), qvel=qvel[::-1].copy())
  sim.step()
  np.testing.assert_allclose(sim.step_sensordata(), first[::-1], atol=2e-5)


@pytest.mark.gpu
@gpu
def test_tactile_disable_sensor_flag():
  from mujoco_metal import MetalSimulation
  model = _model(disabled=True)
  sim = MetalSimulation(model, profile="integrated_euler_v1")
  sim.step()
  assert np.all(sim.step_sensordata() == 0)


@pytest.mark.gpu
@gpu
def test_tactile_detected_contacts_without_active_solver_rows():
  from mujoco_metal import MetalSimulation
  model = _model(inactive=True)
  data = _initial(model)
  sim = MetalSimulation(model, qvel=data.qvel[None].astype(np.float32),
                        profile="integrated_euler_v1")
  sim.step()
  mujoco.mj_step(model, data)
  assert data.ncon > 0
  assert all(data.contact[c].efc_address < 0 for c in range(data.ncon))
  assert np.max(data.sensordata) > 0
  np.testing.assert_allclose(sim.step_sensordata()[0], data.sensordata,
                             atol=2e-5, rtol=2e-5)
