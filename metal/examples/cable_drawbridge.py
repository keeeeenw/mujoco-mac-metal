# Copyright 2026 The MuJoCo Metal contributors
# Licensed under the Apache License, Version 2.0.
"""Cable-driven drawbridge demo with spatial tendon wrapping.

A counterweight winch hauls a cable routed over a tower pulley, around a
bollard cylinder wrap, to a hinged deck carrying a crate. The tendon uses a
pulley divisor, a cylinder wrap with side-site selection, spring/damper
passive forces with a rest range (genuine slack/taut transitions), and joint
limits on the winch travel. A filter actuator drives the winch open-loop:
haul (tension, raise deck), release (slack, lower deck), re-haul
(re-tension, raise again). A paired always-slack run is not physically
meaningful for force-driven schedules, so the counterfactual instead holds a
constant light force that never raises the deck.
"""

import argparse
import json
from pathlib import Path

import mujoco
import numpy as np

PROFILE = "integrated_euler_v1"

HAUL = [-0.8]
HOLD = [-0.1]
SLACK = [0.5]

# (step, ctrl)
WPS = [
    (0, HOLD),
    (100, HAUL),
    (500, HAUL),
    (800, SLACK),
    (1100, HAUL),
]


def _load_model():
  model_path = Path(__file__).with_suffix(".xml")
  model = mujoco.MjModel.from_xml_path(str(model_path))
  if model.ntendon != 1 or model.nu != 1 or model.na != 1:
    raise RuntimeError(
        f"Expected drawbridge with 1 tendon, 1 actuator, 1 slot, "
        f"got ntendon={model.ntendon} nu={model.nu} na={model.na}")
  return model


def _sched(step):
  for k in range(len(WPS) - 1):
    if step < WPS[k + 1][0]:
      return WPS[k][1]
  return WPS[-1][1]


def run(steps=1400, mode="metal", check=False, record=None):
  if mode not in ("metal", "cpu"):
    raise ValueError(f"Unknown mode {mode!r}")
  model = _load_model()
  qpos0 = model.qpos0.copy().astype(np.float32)
  qvel0 = np.zeros(model.nv, dtype=np.float32)

  native = None
  if mode == "metal":
    from mujoco_metal import MetalSimulation
    native = MetalSimulation(model, batch_size=1, qpos=qpos0[None, :], qvel=qvel0[None, :], profile=PROFILE)
    native.reset_to_keyframe(0)

  cpu_haul = mujoco.MjData(model)
  cpu_hold = mujoco.MjData(model)
  for d in (cpu_haul, cpu_hold):
    mujoco.mj_resetDataKeyframe(model, d, 0)
    mujoco.mj_forward(model, d)
  rest = float(np.asarray(model.tendon_lengthspring)[0, 1])
  deck_idx, crate_x, crate_z = 1, 2, 4

  recorder = None
  if record:
    from demo_recording import ComparisonRecorder
    recorder = ComparisonRecorder(model, record, "Cable Drawbridge | Wrap, Haul and Slack",
                                  [0.2, 0, 0.5], 3.0, azimuth=135, elevation=-15)

  max_qpos_err = 0.0
  max_act_err = 0.0
  max_pos_err = 0.0
  max_quat_err = 0.0
  max_geodesic_err = 0.0
  end_pos_err = 0.0
  end_quat_err = 0.0
  end_geodesic_err = 0.0
  deck_min = float("inf")
  deck_slack = None
  crate_peak = 0.0
  slack_steps = 0
  taut_steps = 0
  nat_slack_steps = 0
  nat_taut_steps = 0
  nat_deck_min = float("inf")
  nat_crate_peak = 0.0

  for step in range(steps):
    ctrl = _sched(step)
    if native is not None:
      native.step(1, ctrl=np.asarray(ctrl, dtype=np.float32).reshape(1, 1))
    cpu_haul.ctrl[:] = ctrl
    mujoco.mj_step(model, cpu_haul)
    cpu_hold.ctrl[:] = SLACK
    mujoco.mj_step(model, cpu_hold)
    length = float(np.asarray(cpu_haul.ten_length)[0])
    stretch = length - rest
    if stretch <= 0:
      slack_steps += 1
    else:
      taut_steps += 1
    # Native feature activity comes from native state/kinematics, never the
    # CPU oracle (review gap). CPU counters above remain the oracle baseline.
    if native is not None:
      nat_kin = native._spatial_kin
      nat_len = float(nat_kin["length"].cpu().numpy()[0, 0]) if nat_kin is not None else length
      nat_stretch = nat_len - rest
      if nat_stretch <= 0:
        nat_slack_steps += 1
      else:
        nat_taut_steps += 1
      nat_q = native.state.qpos[0].cpu().numpy()
      nat_deck_min = min(nat_deck_min, float(nat_q[deck_idx]))
      nat_crate_peak = max(nat_crate_peak, float(nat_q[crate_z]))
    deck_min = min(deck_min, float(cpu_haul.qpos[deck_idx]))
    if 800 <= step < 1100:
      if deck_slack is None:
        deck_slack = float(cpu_haul.qpos[deck_idx])
      else:
        deck_slack = max(deck_slack, float(cpu_haul.qpos[deck_idx]))
    crate_peak = max(crate_peak, float(cpu_haul.qpos[crate_z]))
    if native is not None:
      gq = native.state.qpos[0].cpu().numpy()
      ga = native.state.act.cpu().numpy()
      # Translations (winch/hinge/crate xyz) vs crate orientation compared
      # separately: contact transients hit rotation hardest. qpos layout is
      # [winch, hinge, crate(7: x,y,z,qw,qx,qy,qz)].
      epos = float(np.max(np.abs(gq[[0, 1, 2, 3, 4]] - cpu_haul.qpos[[0, 1, 2, 3, 4]])))
      equat = float(np.max(np.abs(gq[[5, 6, 7, 8]] - cpu_haul.qpos[[5, 6, 7, 8]])))
      # Geodesic orientation error: angle between unit quats, robust to sign.
      dot = float(np.clip(abs(np.dot(gq[[5, 6, 7, 8]], cpu_haul.qpos[[5, 6, 7, 8]])), -1.0, 1.0))
      geo = float(2.0 * np.arccos(dot))
      max_pos_err = max(max_pos_err, epos)
      max_quat_err = max(max_quat_err, equat)
      max_geodesic_err = max(max_geodesic_err, geo)
      max_qpos_err = max(max_qpos_err, float(np.max(np.abs(gq - cpu_haul.qpos))))
      max_act_err = max(max_act_err, float(np.max(np.abs(ga - cpu_haul.act))))
      if step == steps - 1:
        end_pos_err, end_quat_err, end_geodesic_err = epos, equat, geo
    if recorder and native is not None:
      actual = mujoco.MjData(model)
      actual.qpos[:] = native.state.qpos[0].cpu().numpy()
      actual.qvel[:] = native.state.qvel[0].cpu().numpy()
      actual.time = float(native.state.time[0].cpu().numpy())
      extra = f"{'HAUL' if ctrl == HAUL else ('SLACK' if ctrl == SLACK else 'HOLD')} | deck={float(actual.qpos[deck_idx]):+.2f}"
      recorder.frame(step, actual, cpu_haul, extra=extra)

  if recorder:
    recorder.close()

  if mode == "metal" and native is not None:
    rel = native.state.qpos[0].cpu().numpy()
    released = (float(rel[deck_idx]), float(rel[crate_z]))
    from mujoco_metal import MetalSimulation as MS
    lat_sim = MS(model, batch_size=1, qpos=qpos0[None, :], qvel=qvel0[None, :], profile=PROFILE)
    lat_sim.reset_to_keyframe(0)
    for step in range(steps):
      lat_sim.step(1, ctrl=np.asarray(SLACK, dtype=np.float32).reshape(1, 1))
    lat = lat_sim.state.qpos[0].cpu().numpy()
    latched = (float(lat[deck_idx]), float(lat[crate_z]))
  else:
    released = (float(cpu_haul.qpos[deck_idx]), float(cpu_haul.qpos[crate_z]))
    latched = (float(cpu_hold.qpos[deck_idx]), float(cpu_hold.qpos[crate_z]))

  result = {
      "max_qpos_err": max_qpos_err,
      "max_act_err": max_act_err,
      "max_pos_err": max_pos_err,
      "max_quat_err": max_quat_err,
      "max_geodesic_err": max_geodesic_err,
      "end_pos_err": end_pos_err,
      "end_quat_err": end_quat_err,
      "end_geodesic_err": end_geodesic_err,
      "deck_min": deck_min,
      "deck_slack": deck_slack,
      "crate_peak": crate_peak,
      "slack_steps": slack_steps,
      "taut_steps": taut_steps,
      "nat_slack_steps": nat_slack_steps,
      "nat_taut_steps": nat_taut_steps,
      "nat_deck_min": nat_deck_min,
      "nat_crate_peak": nat_crate_peak,
      "released": released,
      "latched": latched,
      "steps": steps,
  }
  if check and mode == "metal":
    assert max_pos_err < 3.0e-2, max_pos_err
    assert max_quat_err < 0.1, max_quat_err
    assert max_geodesic_err < 0.2, max_geodesic_err
    assert end_pos_err < 2.0e-2 and end_quat_err < 5e-3, (end_pos_err, end_quat_err)
    assert end_geodesic_err < 5e-3, end_geodesic_err
    assert max_act_err < 5e-4, max_act_err
    assert deck_min < -0.3, deck_min  # hauled up
    assert deck_slack is not None and deck_slack > 0.05, deck_slack  # slack while lowered
    assert slack_steps > 20 and taut_steps > 100, (slack_steps, taut_steps)
    # Native-side feature counters must agree the demo exercised slack/taut
    # and lifted the payload (review gap: previously CPU-oracle only).
    assert nat_slack_steps > 20 and nat_taut_steps > 100, (nat_slack_steps, nat_taut_steps)
    assert nat_deck_min < -0.3, nat_deck_min
    assert nat_crate_peak > 0.35, nat_crate_peak
    assert crate_peak > 0.35, crate_peak  # payload lifted with the deck
    assert released[0] < -0.3, released  # ends raised after re-haul
    assert latched[0] > -0.05, latched  # hold-only run never raises
  return result


def run_viewer(steps=1400, seconds=0):
  """Interactive native viewer (requires a display; headless uses --headless).

  Steps the native simulation with the open-loop haul schedule and shows
  native state live, with the CPU oracle tracked alongside for the report.
  With seconds > 0 the viewer auto-closes after that wall time (used for
  explicit visual checks); 0 runs the full finite rollout (steps), then closes.
  """
  import time
  from mujoco import viewer as mj_viewer
  from mujoco_metal import MetalSimulation
  model = _load_model()
  qpos0 = model.qpos0.copy().astype(np.float32)
  qvel0 = np.zeros(model.nv, dtype=np.float32)
  native = MetalSimulation(model, batch_size=1, qpos=qpos0[None, :], qvel=qvel0[None, :],
                           profile=PROFILE)
  native.reset_to_keyframe(0)
  cpu = mujoco.MjData(model)
  mujoco.mj_resetDataKeyframe(model, cpu, 0)
  mujoco.mj_forward(model, cpu)
  actual = mujoco.MjData(model)
  mujoco.mj_resetDataKeyframe(model, actual, 0)
  mujoco.mj_forward(model, actual)
  max_err = 0.0
  with mj_viewer.launch_passive(model, actual) as viewer:
    viewer.cam.lookat[:] = [0.2, 0, 0.5]
    viewer.cam.distance = 3.0
    viewer.cam.azimuth = 135
    viewer.cam.elevation = -15
    deadline = time.monotonic() + seconds
    for step in range(steps):
      if not viewer.is_running():
        break
      if seconds > 0 and time.monotonic() >= deadline:
        break
      ctrl = _sched(step)
      native.step(1, ctrl=np.asarray(ctrl, dtype=np.float32).reshape(1, 1))
      cpu.ctrl[:] = ctrl
      mujoco.mj_step(model, cpu)
      gq = native.state.qpos[0].cpu().numpy()
      max_err = max(max_err, float(np.max(np.abs(gq - cpu.qpos))))
      with viewer.lock():
        actual.qpos[:] = gq
        actual.qvel[:] = native.state.qvel[0].cpu().numpy()
        actual.time = float(native.state.time[0].cpu().numpy())
      viewer.sync()
  return {"max_qpos_err": max_err, "steps": steps}


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument("--steps", type=int, default=1400)
  parser.add_argument("--mode", choices=["metal", "cpu"], default="metal")
  parser.add_argument("--headless", action="store_true")
  parser.add_argument("--check", action="store_true", help="check native/CPU rollout")
  parser.add_argument("--record", help="record native/CPU renders to a GIF")
  parser.add_argument("--json", help="write numerical report to JSON")
  parser.add_argument("--viewer-seconds", type=float, default=0,
                      help="interactive native viewer; auto-close after this wall time (0 runs the finite rollout, requires a display)")
  args = parser.parse_args()
  if args.steps <= 0 or (args.check and not args.headless):
    raise SystemExit("use --headless --check for verification")
  if not args.headless and args.record is None and args.mode == "metal":
    out = run_viewer(steps=args.steps, seconds=args.viewer_seconds)
    print(json.dumps(out, indent=2))
    if args.json:
      Path(args.json).write_text(json.dumps(out, indent=2))
    return
  out = run(steps=args.steps, mode=args.mode, check=args.check, record=args.record)
  print(json.dumps(out, indent=2))
  if args.json:
    Path(args.json).write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
  main()
