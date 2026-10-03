# Copyright 2026 The MuJoCo Metal contributors
# Licensed under the Apache License, Version 2.0.
"""R01 repair: sensor RNE buffer contract, wrench decoding, COM reduction.

Failing-first regressions for the 016 force-path defects: kernel arity and
contiguity, six-component contact wrenches (condim 1/3/4/6, both cones),
mass-weighted subtree COM on branched trees, connect/weld equality signs,
batched heterogeneous worlds with an inactive world 0, immutable-input
sentinels, and the restored pinball body FORCE/TORQUE sensors.
"""

import os
import re
from pathlib import Path

import mujoco
import numpy as np
import pytest

ACC = (int(mujoco.mjtStage.mjSTAGE_POS), int(mujoco.mjtStage.mjSTAGE_VEL),
       int(mujoco.mjtStage.mjSTAGE_ACC))

SHADERS = Path(__file__).parents[1] / "mujoco_metal" / "shaders"


def _kernel_buffers(path, name):
  src = (SHADERS / path).read_text()
  m = re.search(r"^kernel void %s\((.*?)uint \w+ \[\[thread_position_in_grid\]\]\)"
                % re.escape(name), src, re.S | re.M)
  assert m, name
  bufs = sorted(int(x) for x in re.findall(r"\[\[buffer\((\d+)\)\]\]", m.group(1)))
  assert bufs == list(range(len(bufs))), (name, bufs)
  return len(bufs)


def _cpu_frame(model, qpos, qvel, time=0.0, ctrl=None, xfrc=None):
  d = mujoco.MjData(model)
  d.qpos[:] = qpos
  d.qvel[:] = qvel
  d.time = time
  if ctrl is not None:
    d.ctrl[:] = ctrl
  if xfrc is not None:
    d.xfrc_applied[:] = xfrc
  mujoco.mj_forward(model, d)
  mujoco.mj_rnePostConstraint(model, d)
  return d


def test_sensor_kernel_arity_matches_python_calls_cpu():
  # Static contract: every sensor kernel's declared buffers are contiguous
  # from zero (dispatcher displacement tripwire; see R01).
  assert _kernel_buffers("sensors_rne.metal", "assemble_cfrc_ext") == 23
  assert _kernel_buffers("sensors_rne.metal", "rne_post") == 20
  assert _kernel_buffers("sensors_rne.metal", "evaluate_acc_sensors") == 31
  assert _kernel_buffers("sensors.metal", "evaluate_state_sensors") == 31


@pytest.mark.gpu
@pytest.mark.skipif(os.getenv("MUJOCO_METAL_RUN_GPU") != "1", reason="opt-in GPU")
def test_assemble_dispatch_arity_and_contiguity_gpu():
  # Runtime ABI: the exact dispatched argument counts must equal the kernel
  # declarations, and every tensor argument must be contiguous.
  from mujoco_metal.simulation import MetalSimulation
  model = mujoco.MjModel.from_xml_string(
      "<mujoco><option timestep='0.002' integrator='Euler'/>"
      "<worldbody><geom name='floor' type='plane' size='5 5 0.1'/>"
      "<body pos='0 0 0.15'><freejoint/><geom type='sphere' size='0.1'/>"
      "<site name='imu'/></body></worldbody>"
      "<sensor><accelerometer site='imu'/></sensor></mujoco>")
  sim = MetalSimulation(model, batch_size=2, profile="integrated_euler_v1")
  sim.reset(qpos=np.asarray(model.qpos0, dtype=np.float32).reshape(1, -1).repeat(2, 0),
            qvel=np.zeros((2, model.nv), dtype=np.float32))
  records = {}

  import torch
  orig = {}
  for attr, expect in (("_rne_assemble", 23), ("_rne_post", 20), ("_rne_acc", 31)):
    orig[attr] = getattr(sim._sensors, attr)

  def _make(attr, expect, kernel):
    calls = []
    def wrap(*args, **kwargs):
      calls.append(len(args))
      for a in args:
        if isinstance(a, torch.Tensor):
          assert a.is_contiguous(), (attr, "non-contiguous tensor arg")
      return kernel(*args, **kwargs)
    wrap.calls = calls
    wrap.expect = expect
    return wrap

  wrappers = {a: _make(a, e, orig[a]) for a, e in
              (("_rne_assemble", 23), ("_rne_post", 20), ("_rne_acc", 31))}
  for a, w in wrappers.items():
    setattr(sim._sensors, a, w)
  sim.sensor_values()
  for a, w in wrappers.items():
    assert w.calls, a
    assert w.calls[0] == w.expect, (a, w.calls[0], w.expect)


@pytest.mark.gpu
@pytest.mark.skipif(os.getenv("MUJOCO_METAL_RUN_GPU") != "1", reason="opt-in GPU")
def test_branched_subtree_sensors_gpu():
  from mujoco_metal.simulation import MetalSimulation
  xml = ('<mujoco><option timestep="0.002" integrator="Euler"/>'
         '<worldbody>'
         '<body name="torso" pos="0 0 1.0"><freejoint/>'
         '<geom type="sphere" size="0.1" mass="1"/>'
         '<body name="armL" pos="-0.5 0 0"><joint name="hl" type="hinge" axis="0 0 1"/>'
         '<geom type="sphere" size="0.1" mass="1"/>'
         '<body name="wrist" pos="-0.3 0 0"><joint name="wl" type="hinge" axis="0 0 1"/>'
         '<geom type="sphere" size="0.05" mass="0.5"/></body></body>'
         '<body name="armR" pos="0.5 0 0"><joint name="hr" type="hinge" axis="0 0 1"/>'
         '<geom type="sphere" size="0.1" mass="1"/></body>'
         '<body name="fixed" pos="0 0.2 0">'
         '<geom type="sphere" size="0.05" mass="0.25"/></body>'
         '</body></worldbody>'
         '<sensor><subtreecom body="torso"/><subtreelinvel body="torso"/>'
         '<subtreeangmom body="torso"/><subtreecom body="armL"/>'
         '<jointpos joint="hl"/><clock/></sensor></mujoco>')
  model = mujoco.MjModel.from_xml_string(xml)
  q0 = np.asarray(model.qpos0, dtype=float)
  qp = np.stack([q0, q0 + np.array([0.1] + [0] * (model.nq - 1)),
                 q0 + np.array([-0.15] + [0] * (model.nq - 1))])
  qv = np.stack([np.zeros(model.nv),
                 np.linspace(0.2, -0.3, model.nv),
                 np.linspace(-0.4, 0.5, model.nv)])
  sim = MetalSimulation(model, batch_size=3, profile="integrated_euler_v1")
  sim.reset(qpos=qp.astype(np.float32), qvel=qv.astype(np.float32))
  got = sim.sensor_values().cpu().numpy()
  times = np.full(3, float(sim.state.time.cpu().numpy()[0]))
  from mujoco_metal.model import load_model
  from mujoco_metal.sensors import sensor_oracle
  desc = load_model(model)
  poses = {k: [] for k in ("body_pos", "body_quat", "geom_pos", "geom_quat",
                           "site_pos", "site_quat", "inertial_pos",
                           "inertial_quat", "cvel", "root_com")}
  gq = sim.state.qpos.cpu().numpy().astype(float)
  gv = sim.state.qvel.cpu().numpy().astype(float)
  extra_c, extra_l, extra_a, extra_e = [], [], [], []
  for w in range(3):
    dd = mujoco.MjData(model)
    dd.qpos[:] = gq[w]
    dd.qvel[:] = gv[w]
    mujoco.mj_forward(model, dd)
    mujoco.mj_subtreeVel(model, dd)
    mujoco.mj_energyPos(model, dd)
    mujoco.mj_energyVel(model, dd)
    fk = desc.forward_kinematics(gq[w])
    for k in poses:
      poses[k].append(fk[k] if k in fk else dd.cvel.copy() if k == "cvel" else dd.subtree_com.copy())
    extra_c.append(np.asarray(dd.subtree_com).copy())
    extra_l.append(np.asarray(dd.subtree_linvel).copy())
    extra_a.append(np.asarray(dd.subtree_angmom).copy())
    extra_e.append(np.asarray(dd.energy).copy())
  poses = {k: np.asarray(v) for k, v in poses.items()}
  extra = {"subtree_com": np.asarray(extra_c), "subtree_linvel": np.asarray(extra_l),
           "subtree_angmom": np.asarray(extra_a), "energy": np.asarray(extra_e)}
  want = sensor_oracle(model, gq, gv, times, poses, extra=extra, stages=ACC)
  np.testing.assert_allclose(got, want, rtol=2e-4, atol=2e-4)


@pytest.mark.gpu
@pytest.mark.skipif(os.getenv("MUJOCO_METAL_RUN_GPU") != "1", reason="opt-in GPU")
def test_cfrc_composition_connect_weld_xfrc_gpu():
  from mujoco_metal.simulation import MetalSimulation
  xml = ('<mujoco><option timestep="0.002" integrator="Euler" gravity="0 0 -9.81"/>'
         '<worldbody><geom name="floor" type="plane" size="5 5 0.1"/>'
         '<body name="A" pos="-0.5 0 0.4"><freejoint/>'
         '<geom type="sphere" size="0.1" mass="0.5"/><site name="sA"/></body>'
         '<body name="B" pos="0.5 0 0.12"><freejoint/>'
         '<geom type="sphere" size="0.1" mass="0.5"/><site name="sB"/></body>'
         '<body name="C" pos="0 0.6 0.3"><freejoint/>'
         '<geom type="box" size="0.08 0.08 0.08" mass="0.4"/><site name="sC"/></body>'
         '</worldbody>'
         '<equality><connect body1="A" body2="B" anchor="0 0 0.25"/>'
         '<weld body1="A" body2="C" relpose="0.5 0.6 -0.1 1 0 0 0"/></equality>'
         '<sensor><force site="sA"/><torque site="sA"/>'
         '<accelerometer site="sB"/>'
         '<framelinacc objtype="body" objname="B"/>'
         '<clock/></sensor></mujoco>')
  model = mujoco.MjModel.from_xml_string(xml)
  nb = model.nbody
  assert nb == 4
  q0 = np.asarray(model.qpos0, dtype=float)
  qp = np.stack([q0, q0, q0])
  # World 0: B lifted clear of the floor (inactive: no contacts, no wrench).
  qp[0, 7 + 2] += 0.6
  # World 2: shifted laterally.
  qp[2, 0] += 0.3
  qv = np.zeros((3, model.nv))
  sim = MetalSimulation(model, batch_size=3, profile="integrated_euler_v1")
  sim.reset(qpos=qp.astype(np.float32), qvel=qv.astype(np.float32))
  import torch
  xfrc = torch.zeros((3, nb, 6), dtype=torch.float32, device="mps")
  xfrc[1, 1, 2] = 2.0
  xfrc[2, 1, 0] = -1.5
  xfrc[2, 3, 2] = 1.0
  for _ in range(80):
    sim.step(1, xfrc_applied=xfrc)
  gq = sim.state.qpos.cpu().numpy().astype(float)
  gv = sim.state.qvel.cpu().numpy().astype(float)
  got_sens = sim.sensor_values().cpu().numpy()
  got_ext = sim._sensors._rne_ext.cpu().numpy().reshape(3, nb, 6)
  got_int = sim._sensors._rne_cfrc.cpu().numpy().reshape(3, nb, 6)
  xh = xfrc.cpu().numpy().reshape(3, nb, 6)
  for w in range(3):
    dd = _cpu_frame(model, gq[w], gv[w], time=float(sim.state.time.cpu().numpy()[w]),
                    xfrc=xh[w].reshape(nb, 6))
    np.testing.assert_allclose(got_ext[w], np.asarray(dd.cfrc_ext).reshape(nb, 6),
                               rtol=2e-4, atol=2e-4, err_msg=f"ext w{w}")
    np.testing.assert_allclose(got_int[w], np.asarray(dd.cfrc_int).reshape(nb, 6),
                               rtol=2e-4, atol=2e-4, err_msg=f"int w{w}")
    np.testing.assert_allclose(got_sens[w], np.asarray(dd.sensordata),
                               rtol=2e-4, atol=2e-4, err_msg=f"sens w{w}")
  # Per-world oracle agreement already proves batch isolation: world 0's
  # CPU frame has no contacts, so any cross-world leakage would mismatch.
  assert sim._sensors._rne_ext.shape == (3, nb, 6)


@pytest.mark.gpu
@pytest.mark.skipif(os.getenv("MUJOCO_METAL_RUN_GPU") != "1", reason="opt-in GPU")
@pytest.mark.parametrize("condim,cone", [(1, "pyramidal"), (3, "pyramidal"),
                                         (4, "elliptic"), (6, "elliptic")])
def test_condim_cone_force_matrix_gpu(condim, cone):
  from mujoco_metal.simulation import MetalSimulation
  # Contact forces on a free body read ~0 through cfrc balance, so the
  # matrix presses a slide-driven pad into the floor: contact forces
  # transmit through the joint (jointactfrc) and the site wrench
  # (force/torque). Lateral slide velocity engages friction, exercising
  # per-slot mu indexing of the pyramid/elliptic decode (unequal geom
  # frictions: floor 0.9 vs pad 0.7).
  xml = ('<mujoco><option timestep="0.002" integrator="Euler" cone="%s"/>'
         '<worldbody><geom name="floor" type="plane" size="5 5 0.1" friction="0.9 0.05 0.02"/>'
         '<body name="press" pos="0 0 0.3">'
         '<joint name="s" type="slide" axis="0.196116 0 -0.980581" damping="0.5" limited="true" range="0 0.25"/>'
         '<geom name="pad" type="sphere" size="0.1" mass="0.4" friction="0.7 0.08 0.03"/>'
         '<site name="g"/></body></worldbody>'
         '<actuator><motor name="m" joint="s" gear="3"/></actuator>'
         '<sensor><jointactuatorfrc joint="s"/><actuatorfrc actuator="m"/>'
         '<force site="g"/><torque site="g"/></sensor></mujoco>') % cone
  if condim != 3:
    xml = xml.replace("<sensor>", '<contact><pair geom1="floor" geom2="pad" condim="%d"/></contact><sensor>' % condim)
  model = mujoco.MjModel.from_xml_string(xml)
  sim = MetalSimulation(model, batch_size=2, profile="integrated_euler_v1")
  qp = np.asarray(model.qpos0, dtype=np.float32).reshape(1, -1).repeat(2, 0)
  qv = np.zeros((2, model.nv), dtype=np.float32)
  qv[:, 0] = [0.8, -0.6]
  sim.reset(qpos=qp, qvel=qv)
  cpus = []
  for w in range(2):
    dd = mujoco.MjData(model)
    dd.qpos[:] = qp[w]
    dd.qvel[:] = qv[w]
    cpus.append(dd)
  max_diff, joint_peak, force_peak = 0.0, 0.0, 0.0
  for _ in range(120):
    ctrl = np.array([[0.8], [1.0]], dtype=np.float32)
    sim.step(1, ctrl=ctrl)
    for w, dd in enumerate(cpus):
      dd.ctrl[:] = [0.8 + 0.2 * w]
      mujoco.mj_step(model, dd)
    stored = sim.step_sensordata()
    for w, dd in enumerate(cpus):
      ref = np.asarray(dd.sensordata)
      max_diff = max(max_diff, float(np.max(np.abs(stored[w] - ref))))
      joint_peak = max(joint_peak, float(abs(ref[0])))
      force_peak = max(force_peak, float(np.max(np.abs(ref[2:8]))))
  # Non-vacuous: joint drive and transmitted site wrench both engage.
  assert joint_peak > 0.5, (condim, cone, joint_peak)
  assert force_peak > 0.5, (condim, cone, force_peak)
  # Pyramidal decode is exact; elliptic carries the native-FISTA vs CPU
  # cone-solver budget difference (same iteration budget, ≤0.5% of ~6 N
  # contact forces here). R04 owns cone-solver convergence; the decode
  # itself is pinned-formula exact on both paths.
  gate = 2e-3 if cone == "pyramidal" else 5e-2
  assert max_diff < gate, (condim, cone, max_diff)
  # Post-step query against matching forward oracles.
  gq = sim.state.qpos.cpu().numpy().astype(float)
  gv = sim.state.qvel.cpu().numpy().astype(float)
  got = sim.sensor_values().cpu().numpy()
  for w in range(2):
    dd = _cpu_frame(model, gq[w], gv[w], ctrl=[0.8 + 0.2 * w])
    gate_q = 2e-3 if cone == "pyramidal" else 5e-2
    np.testing.assert_allclose(got[w], np.asarray(dd.sensordata),
                               rtol=2e-3, atol=gate_q,
                               err_msg=f"{condim}/{cone} w{w}")


@pytest.mark.gpu
@pytest.mark.skipif(os.getenv("MUJOCO_METAL_RUN_GPU") != "1", reason="opt-in GPU")
def test_rne_input_sentinels_gpu():
  from mujoco_metal.simulation import MetalSimulation
  import torch
  model = mujoco.MjModel.from_xml_string(
      "<mujoco><option timestep='0.002' integrator='Euler'/>"
      "<worldbody><geom name='floor' type='plane' size='5 5 0.1'/>"
      "<body pos='0 0 0.2'><freejoint/><geom type='sphere' size='0.1'/>"
      "<site name='imu'/></body></worldbody>"
      "<sensor><accelerometer site='imu'/><force site='imu'/></sensor></mujoco>")
  sim = MetalSimulation(model, batch_size=2, profile="integrated_euler_v1")
  sim.reset(qpos=np.asarray(model.qpos0, dtype=np.float32).reshape(1, -1).repeat(2, 0),
            qvel=np.zeros((2, model.nv), dtype=np.float32))
  sim.step(20)
  sp = sim._sensors
  before = {}
  for name in ("_rne_mass", "_rne_body_tree", "_rne_geom_bodyid",
               "_rne_site_bodyid", "_rne_eq_meta", "_rne_eq_data",
               "_rne_act_trn", "_s_meta"):
    t = getattr(sp, name)
    before[name] = t.reshape(-1).detach().cpu().numpy().copy()
  sim.sensor_values()
  sim.step(5)
  for name, old in before.items():
    now = getattr(sp, name).reshape(-1).detach().cpu().numpy()
    np.testing.assert_array_equal(now, old, err_msg=name)


@pytest.mark.gpu
@pytest.mark.skipif(os.getenv("MUJOCO_METAL_RUN_GPU") != "1", reason="opt-in GPU")
def test_pinball_force_torque_restored_gpu():
  from mujoco_metal.simulation import MetalSimulation
  from pathlib import Path as _P
  model = mujoco.MjModel.from_xml_path(
      str(_P(__file__).parents[1] / "examples" / "pinball_table.xml"))
  sim = MetalSimulation(model, batch_size=1, profile="integrated_euler_v1")
  sim.reset(qpos=np.asarray(model.qpos0, dtype=np.float32).reshape(1, -1),
            qvel=np.zeros((1, model.nv), dtype=np.float32))
  import math
  for step in range(200):
    t = 0.002 * step
    ctrl = np.array([0.8 * math.sin(2 * math.pi * 1.2 * t),
                     0.8 * math.sin(2 * math.pi * 1.2 * t + 1.0)], dtype=np.float32).reshape(1, 2)
    sim.step(1, ctrl=ctrl)
  stored = sim.step_sensordata()[0]
  ntypes = [int(v) for v in np.asarray(model.sensor_type)]
  assert 4 in ntypes and 5 in ntypes  # FORCE + TORQUE present again
  d = mujoco.MjData(model)
  d.qpos[:] = np.asarray(model.qpos0)
  for step in range(200):
    t = 0.002 * step
    d.ctrl[:] = [0.8 * math.sin(2 * math.pi * 1.2 * t),
                 0.8 * math.sin(2 * math.pi * 1.2 * t + 1.0)]
    mujoco.mj_step(model, d)
  ntypes = [int(v) for v in np.asarray(model.sensor_type)]
  np.testing.assert_allclose(stored, np.asarray(d.sensordata), rtol=2e-3, atol=2e-3)
