# Copyright 2026 The MuJoCo Metal contributors
# Licensed under the Apache License, Version 2.0.
"""Full built-in actuator lowering for MuJoCo 3.10.0 (milestone 007).

Admits every pinned actuator dynamics/gain/bias family except USER callbacks
(owned by milestone 019) and every rigid transmission type except spatial
tendons (owned by milestone 008): INTEGRATOR/FILTER/FILTEREXACT/MUSCLE/DCMOTOR
dynamics, FIXED/AFFINE/MUSCLE/DCMOTOR gains and NONE/AFFINE/MUSCLE/DCMOTOR
biases, JOINT/JOINTINPARENT/SLIDERCRANK/TENDON(fixed)/SITE/BODY transmissions
including ball/free non-scalar gears. Covers actearly, activation/control/force
limits, group disabling, aggregated joint/tendon force limits and
gravity-compensation routing per the pinned engine. Actuator delay/history is
rejected (control-stage timing, owned by 015); actuator armature/damping stay
rejected (passive-stage properties, owned by 015); plugins rejected (019).

Pinned sources: engine/engine_forward.c (mj_fwdActuation, mj_nextActivation,
dcmotorVoltage), engine/engine_support.c (mj_nextActivation exact forms),
engine/engine_core_smooth.c (mj_transmission), engine/engine_util_misc.c
(muscle/DC-slot utilities), engine/engine_core_util.c (armature/damping folds,
kept as rejection rationale).
"""

import mujoco
import numpy as np

_MJMINVAL = 1e-15
_INT32_MAX = (1 << 31) - 1
_UINT32_MAX = (1 << 32) - 1
_NU_CAP = 32


def _frozen(values, dtype):
  array = np.array(values, dtype=dtype, order="C", copy=True)
  if array.dtype.kind == "f" and not np.all(np.isfinite(array)):
    raise ValueError("actuator constants must be finite and float32-representable")
  return np.frombuffer(array.tobytes(), dtype=array.dtype).reshape(array.shape)


def _fixed_maps(model, nu, nq, nv):
  """Constant length/moment maps for scalar joint and fixed-tendon rows.

  Mirrors TransmissionModel math exactly; rows for state-dependent
  transmissions stay zero and are assembled by the general kernel.
  """
  hinge = int(mujoco.mjtJoint.mjJNT_HINGE)
  slide = int(mujoco.mjtJoint.mjJNT_SLIDE)
  wrap_joint = int(mujoco.mjtWrap.mjWRAP_JOINT)
  trn_joint = int(mujoco.mjtTrn.mjTRN_JOINT)
  trn_parent = int(mujoco.mjtTrn.mjTRN_JOINTINPARENT)
  trn_tendon = int(mujoco.mjtTrn.mjTRN_TENDON)
  length_map = np.zeros((nu, nq), dtype=np.float64)
  moment_map = np.zeros((nu, nv), dtype=np.float64)
  trntype = np.asarray(model.actuator_trntype)
  trnid = np.asarray(model.actuator_trnid)
  gear = np.asarray(model.actuator_gear, dtype=np.float64)
  for a in range(nu):
    t = int(trntype[a])
    t0 = int(trnid[a, 0])
    if t in (trn_joint, trn_parent) and int(model.jnt_type[t0]) in (hinge, slide):
      qa, da = int(model.jnt_qposadr[t0]), int(model.jnt_dofadr[t0])
      length_map[a, qa] = gear[a, 0]
      moment_map[a, da] = gear[a, 0]
    elif t == trn_tendon:
      start, count = int(model.tendon_adr[t0]), int(model.tendon_num[t0])
      if count <= 0 or int(model.wrap_type[start]) != wrap_joint:
        continue  # spatial tendon: zero rows here, overlay owns them (R1)
      for wrap in range(start, start + count):
        if int(model.wrap_type[wrap]) != wrap_joint:
          break
        joint = int(model.wrap_objid[wrap])
        coefficient = gear[a, 0] * float(model.wrap_prm[wrap])
        length_map[a, int(model.jnt_qposadr[joint])] += coefficient
        moment_map[a, int(model.jnt_dofadr[joint])] += coefficient
  return length_map, moment_map


def dcmotor_slots(dynprm, gainprm):
  """Port of pinned mj_dcmotorSlots: ordered optional state slot indices."""
  slew = integral = temperature = bristle = current = -1
  n = 0
  if dynprm[7] > 0:
    slew = n; n += 1
  if gainprm[5] > 0:
    integral = n; n += 1
  if dynprm[2] > 0:
    temperature = n; n += 1
  if dynprm[5] > 0:
    bristle = n; n += 1
  if dynprm[0] > 0:
    current = n; n += 1
  return {"slew": slew, "integral": integral, "temperature": temperature,
          "bristle": bristle, "current": current, "num_slots": n}


def _clip(x, lo, hi):
  return min(max(x, lo), hi)


def muscle_gain_length(length, lmin, lmax):
  """Port of pinned mju_muscleGainLength."""
  if lmin <= length <= lmax:
    a = 0.5 * (lmin + 1.0)
    b = 0.5 * (1.0 + lmax)
    if length <= a:
      x = (length - lmin) / max(_MJMINVAL, a - lmin)
      return 0.5 * x * x
    if length <= 1.0:
      x = (1.0 - length) / max(_MJMINVAL, 1.0 - a)
      return 1.0 - 0.5 * x * x
    if length <= b:
      x = (length - 1.0) / max(_MJMINVAL, b - 1.0)
      return 1.0 - 0.5 * x * x
    x = (lmax - length) / max(_MJMINVAL, lmax - b)
    return 0.5 * x * x
  return 0.0


def muscle_gain(length, velocity, lengthrange, acc0, prm):
  """Port of pinned mju_muscleGain (active force-length-velocity)."""
  rng = [prm[0], prm[1]]
  force, scale = prm[2], prm[3]
  lmin, lmax, vmax, fvmax = prm[4], prm[5], prm[6], prm[8]
  if force < 0:
    force = scale / max(_MJMINVAL, acc0)
  l0 = (lengthrange[1] - lengthrange[0]) / max(_MJMINVAL, rng[1] - rng[0])
  ln = rng[0] + (length - lengthrange[0]) / max(_MJMINVAL, l0)
  vn = velocity / max(_MJMINVAL, l0 * vmax)
  fl = muscle_gain_length(ln, lmin, lmax)
  y = fvmax - 1.0
  if vn <= -1.0:
    fv = 0.0
  elif vn <= 0.0:
    fv = (vn + 1.0) * (vn + 1.0)
  elif vn <= y:
    fv = fvmax - (y - vn) * (y - vn) / max(_MJMINVAL, y)
  else:
    fv = fvmax
  return -force * fl * fv


def muscle_bias(length, lengthrange, acc0, prm):
  """Port of pinned mju_muscleBias (passive force)."""
  rng = [prm[0], prm[1]]
  force, scale = prm[2], prm[3]
  lmax, fpmax = prm[5], prm[7]
  if force < 0:
    force = scale / max(_MJMINVAL, acc0)
  l0 = (lengthrange[1] - lengthrange[0]) / max(_MJMINVAL, rng[1] - rng[0])
  ln = rng[0] + (length - lengthrange[0]) / max(_MJMINVAL, l0)
  b = 0.5 * (1.0 + lmax)
  if ln <= 1.0:
    return 0.0
  if ln <= b:
    x = (ln - 1.0) / max(_MJMINVAL, b - 1.0)
    return -force * fpmax * 0.5 * x * x
  x = (ln - b) / max(_MJMINVAL, b - 1.0)
  return -force * fpmax * (0.5 + x)


def _sigmoid(x):
  return 1.0 / (1.0 + np.exp(-x))


def muscle_dynamics(ctrl, act, prm):
  """Port of pinned mju_muscleDynamics."""
  ctrl_c = _clip(ctrl, 0.0, 1.0)
  act_c = _clip(act, 0.0, 1.0)
  tau_act = prm[0] * (0.5 + 1.5 * act_c)
  tau_deact = prm[1] / (0.5 + 1.5 * act_c)
  width = prm[2]
  dctrl = ctrl_c - act
  if width < _MJMINVAL:
    tau = tau_act if dctrl > 0 else tau_deact
  else:
    tau = tau_deact + (tau_act - tau_deact) * float(_sigmoid(dctrl / width + 0.5))
  return dctrl / max(_MJMINVAL, tau)


def dcmotor_voltage(ctrl, length, velocity, x_i, gainprm):
  """Port of pinned dcmotorVoltage."""
  mode = int(gainprm[8])
  vmax = gainprm[7]
  if mode > 0:
    kp, ki, kd = gainprm[4], gainprm[5], gainprm[6]
    if mode == 1:
      voltage = kp * (ctrl - length) + ki * x_i - kd * velocity
    else:
      voltage = kp * (ctrl - velocity) + ki * (x_i - length)
  else:
    voltage = ctrl
  if vmax > 0:
    voltage = _clip(voltage, -vmax, vmax)
  return voltage


def lugre_stribeck(velocity, f_c, f_s, v_s):
  """Port of pinned mj_lugreStribeck."""
  ratio = velocity / max(_MJMINVAL, v_s)
  return f_c + (f_s - f_c) * np.exp(-ratio * ratio)


class ActuatorModel:
  """Immutable full-family actuator lowering for MuJoCo 3.10.0.

  Owns per-actuator transmission/dynamics/gain/bias constants, limit and group
  data, and admission for the integrated native pipeline. Stateless scalar
  models report ``needs_general_path == False`` so the pre-qualified scalar
  stage can be reused byte-identically.
  """

  def __init__(self, model):
    if mujoco.__version__ != "3.10.0":
      raise RuntimeError(f"actuator lowering requires MuJoCo 3.10.0; found {mujoco.__version__}")
    if not isinstance(model, mujoco.MjModel):
      raise TypeError("model must be a compiled mujoco.MjModel")
    nu, nv, nq = int(model.nu), int(model.nv), int(model.nq)
    na = int(model.na)
    if any(v < 0 or v > _INT32_MAX for v in (nq, nv, nu, na)):
      raise ValueError("actuator dimensions exceed int32")
    if nu > _NU_CAP:
      raise ValueError(f"actuator count {nu} exceeds the native cap {_NU_CAP}")
    if int(model.nplugin):
      raise ValueError("actuator plugins are unsupported (owned by milestone 019)")
    if np.any(np.asarray(model.actuator_plugin) >= 0):
      raise ValueError("actuator plugins are unsupported (owned by milestone 019)")
    if np.any(np.asarray(model.actuator_armature) != 0):
      raise ValueError("actuator armature is unsupported (owned by milestone 015)")
    if np.any(np.asarray(model.actuator_damping) != 0) or np.any(
        np.asarray(model.actuator_dampingpoly) != 0):
      raise ValueError("actuator damping is unsupported (owned by milestone 015)")
    if np.any(np.asarray(model.actuator_delay) != 0):
      raise ValueError("actuator delay is unsupported (owned by milestone 015)")
    if np.any(np.asarray(model.actuator_history) != 0):
      raise ValueError("actuator history buffers are unsupported (owned by milestone 015)")

    dyn_none = int(mujoco.mjtDyn.mjDYN_NONE)
    dyn_int = int(mujoco.mjtDyn.mjDYN_INTEGRATOR)
    dyn_filter = int(mujoco.mjtDyn.mjDYN_FILTER)
    dyn_filter_exact = int(mujoco.mjtDyn.mjDYN_FILTEREXACT)
    dyn_muscle = int(mujoco.mjtDyn.mjDYN_MUSCLE)
    dyn_dcmotor = int(mujoco.mjtDyn.mjDYN_DCMOTOR)
    dyn_user = int(mujoco.mjtDyn.mjDYN_USER)
    gain_fixed = int(mujoco.mjtGain.mjGAIN_FIXED)
    gain_affine = int(mujoco.mjtGain.mjGAIN_AFFINE)
    gain_muscle = int(mujoco.mjtGain.mjGAIN_MUSCLE)
    gain_dcmotor = int(mujoco.mjtGain.mjGAIN_DCMOTOR)
    gain_user = int(mujoco.mjtGain.mjGAIN_USER)
    bias_none = int(mujoco.mjtBias.mjBIAS_NONE)
    bias_affine = int(mujoco.mjtBias.mjBIAS_AFFINE)
    bias_muscle = int(mujoco.mjtBias.mjBIAS_MUSCLE)
    bias_dcmotor = int(mujoco.mjtBias.mjBIAS_DCMOTOR)
    bias_user = int(mujoco.mjtBias.mjBIAS_USER)
    trn_joint = int(mujoco.mjtTrn.mjTRN_JOINT)
    trn_parent = int(mujoco.mjtTrn.mjTRN_JOINTINPARENT)
    trn_crank = int(mujoco.mjtTrn.mjTRN_SLIDERCRANK)
    trn_tendon = int(mujoco.mjtTrn.mjTRN_TENDON)
    trn_site = int(mujoco.mjtTrn.mjTRN_SITE)
    trn_body = int(mujoco.mjtTrn.mjTRN_BODY)
    hinge = int(mujoco.mjtJoint.mjJNT_HINGE)
    slide = int(mujoco.mjtJoint.mjJNT_SLIDE)
    ball = int(mujoco.mjtJoint.mjJNT_BALL)
    free = int(mujoco.mjtJoint.mjJNT_FREE)
    wrap_joint = int(mujoco.mjtWrap.mjWRAP_JOINT)

    dyntype = np.asarray(model.actuator_dyntype)
    gaintype = np.asarray(model.actuator_gaintype)
    biastype = np.asarray(model.actuator_biastype)
    trntype = np.asarray(model.actuator_trntype)
    for i in range(nu):
      if int(dyntype[i]) == dyn_user or int(gaintype[i]) == gain_user or int(biastype[i]) == bias_user:
        raise ValueError(f"actuator {i}: user callbacks are unsupported (owned by milestone 019)")
      if int(dyntype[i]) not in (dyn_none, dyn_int, dyn_filter, dyn_filter_exact, dyn_muscle, dyn_dcmotor):
        raise ValueError(f"actuator {i}: unknown dynamics type")
      if int(gaintype[i]) not in (gain_fixed, gain_affine, gain_muscle, gain_dcmotor):
        raise ValueError(f"actuator {i}: unknown gain type")
      if int(biastype[i]) not in (bias_none, bias_affine, bias_muscle, bias_dcmotor):
        raise ValueError(f"actuator {i}: unknown bias type")

    actnum = np.asarray(model.actuator_actnum)
    actadr = np.asarray(model.actuator_actadr)
    dynprm = np.asarray(model.actuator_dynprm, dtype=np.float64)
    gainprm = np.asarray(model.actuator_gainprm, dtype=np.float64)
    biasprm = np.asarray(model.actuator_biasprm, dtype=np.float64)
    # Dynamics/state shape validation (mirrors the pinned SHOULD-NOT-OCCUR checks).
    cursor = 0
    for i in range(nu):
      dt = int(dyntype[i])
      if dt == dyn_none:
        if int(actnum[i]) != 0:
          raise ValueError(f"actuator {i}: stateless dynamics must have actnum 0")
      elif dt in (dyn_int, dyn_filter, dyn_filter_exact, dyn_muscle):
        if int(actnum[i]) != 1:
          raise ValueError(f"actuator {i}: scalar dynamics must have actnum 1")
      else:  # dcmotor
        slots = dcmotor_slots(dynprm[i], gainprm[i])
        if slots["num_slots"] != int(actnum[i]):
          raise ValueError(f"actuator {i}: DC-motor slots do not match actnum")
      if int(actadr[i]) != cursor and int(actnum[i]) != 0:
        raise ValueError(f"actuator {i}: non-dense activation addressing")
      if int(actnum[i]) == 0 and int(actadr[i]) not in (-1, cursor):
        raise ValueError(f"actuator {i}: invalid stateless activation address")
      cursor += int(actnum[i])
    if cursor != na:
      raise ValueError("activation slots must densely cover [0, na)")

    gear = np.asarray(model.actuator_gear, dtype=np.float64)
    trnid = np.asarray(model.actuator_trnid)
    crank = np.asarray(model.actuator_cranklength, dtype=np.float64)
    for i in range(nu):
      if not np.all(np.isfinite(gear[i])):
        raise ValueError(f"actuator {i}: gear must be finite")
      t = int(trntype[i])
      t0, t1 = int(trnid[i, 0]), int(trnid[i, 1])
      if t in (trn_joint, trn_parent):
        if t0 < 0 or t0 >= int(model.njnt):
          raise ValueError(f"actuator {i}: joint target out of range")
        jt = int(model.jnt_type[t0])
        if jt in (hinge, slide):
          if np.any(gear[i, 1:] != 0):
            raise ValueError(f"actuator {i}: scalar gear must use only its first component")
        elif jt == ball:
          if np.any(gear[i, 3:] != 0):
            raise ValueError(f"actuator {i}: ball gear must use only its first three components")
        elif jt == free:
          pass
        else:
          raise ValueError(f"actuator {i}: unknown joint type")
      elif t == trn_crank:
        if t0 < 0 or t0 >= int(model.nsite) or t1 < 0 or t1 >= int(model.nsite):
          raise ValueError(f"actuator {i}: slider-crank sites out of range")
        if not np.isfinite(crank[i]) or crank[i] <= 0:
          raise ValueError(f"actuator {i}: crank length must be finite and positive")
        if np.any(gear[i, 1:] != 0):
          raise ValueError(f"actuator {i}: scalar gear must use only its first component")
      elif t == trn_tendon:
        if t0 < 0 or t0 >= int(model.ntendon):
          raise ValueError(f"actuator {i}: tendon target out of range")
        start, count = int(model.tendon_adr[t0]), int(model.tendon_num[t0])
        if count <= 0:
          raise ValueError(f"actuator {i}: tendon has an empty wrap path")
        if int(model.wrap_type[start]) == wrap_joint:
          for wrap in range(start, start + count):
            if int(model.wrap_type[wrap]) != wrap_joint:
              raise ValueError(f"actuator {i}: mixed joint/spatial wraps are unsupported")
            joint = int(model.wrap_objid[wrap])
            if joint < 0 or joint >= int(model.njnt) or int(model.jnt_type[joint]) not in (hinge, slide):
              raise ValueError(f"actuator {i}: fixed tendon wraps must target hinge or slide joints")
        else:
          # Spatial tendon target (R1): full path validation lives in
          # SpatialTendonModel; the general actuator path consumes its
          # length/velocity/moment at runtime (see apply_spatial_tendon_state).
          from mujoco_metal.spatial_tendons import SpatialTendonModel as _Spatial
          _Spatial(model)
        if np.any(gear[i, 1:] != 0):
          raise ValueError(f"actuator {i}: scalar gear must use only its first component")
      elif t == trn_site:
        if t0 < 0 or t0 >= int(model.nsite):
          raise ValueError(f"actuator {i}: site target out of range")
        if t1 != -1 and (t1 < 0 or t1 >= int(model.nsite)):
          raise ValueError(f"actuator {i}: reference site out of range")
      elif t == trn_body:
        if t0 < 0 or t0 >= int(model.nbody):
          raise ValueError(f"actuator {i}: body target out of range")
      else:
        raise ValueError(f"actuator {i}: unknown transmission type")

    group = np.asarray(model.actuator_group)
    if np.any(group < 0) or np.any(group >= 31):
      raise ValueError("actuator groups must be in [0, 30]")
    for limited, ranges, label in (
        (model.actuator_ctrllimited, model.actuator_ctrlrange, "control"),
        (model.actuator_forcelimited, model.actuator_forcerange, "force"),
        (model.actuator_actlimited, model.actuator_actrange, "activation"),
    ):
      limited = np.asarray(limited)
      ranges = np.asarray(ranges, dtype=np.float64)
      if np.any(limited & (ranges[:, 0] > ranges[:, 1])):
        raise ValueError(f"limited actuator {label} ranges must be ordered")
      if not np.all(np.isfinite(ranges)):
        raise ValueError(f"actuator {label} ranges must be finite")
    for name in ("actuator_dynprm", "actuator_gainprm", "actuator_biasprm",
                 "actuator_lengthrange"):
      if not np.all(np.isfinite(np.asarray(getattr(model, name), dtype=np.float64))):
        raise ValueError(f"{name} must be finite")
    if not np.all(np.isfinite(np.asarray(model.actuator_acc0, dtype=np.float64))):
      raise ValueError("actuator_acc0 must be finite")

    self.nq, self.nv, self.nu, self.na = nq, nv, nu, na
    self.nbody, self.njnt, self.nsite, self.ntendon = (
        int(model.nbody), int(model.njnt), int(model.nsite), int(model.ntendon))
    self.trntype = _frozen(trntype, np.int32)
    self.trnid = _frozen(trnid, np.int32)
    self.gear = _frozen(gear, np.float32)
    self.cranklength = _frozen(crank, np.float32)
    self.dyntype = _frozen(dyntype, np.int32)
    self.dynprm = _frozen(dynprm, np.float32)
    self.gaintype = _frozen(gaintype, np.int32)
    self.gainprm = _frozen(gainprm, np.float32)
    self.biastype = _frozen(biastype, np.int32)
    self.biasprm = _frozen(biasprm, np.float32)
    self.actadr = _frozen(actadr, np.int32)
    self.actnum = _frozen(actnum, np.int32)
    self.actearly = _frozen(model.actuator_actearly, np.uint8)
    self.ctrllimited = _frozen(model.actuator_ctrllimited, np.uint8)
    self.ctrlrange = _frozen(model.actuator_ctrlrange, np.float32)
    self.forcelimited = _frozen(model.actuator_forcelimited, np.uint8)
    self.forcerange = _frozen(model.actuator_forcerange, np.float32)
    self.actlimited = _frozen(model.actuator_actlimited, np.uint8)
    self.actrange = _frozen(model.actuator_actrange, np.float32)
    self.group = _frozen(group, np.int32)
    self.lengthrange = _frozen(model.actuator_lengthrange, np.float32)
    self.acc0 = _frozen(model.actuator_acc0, np.float32)
    self.tendon_actfrclimited = _frozen(model.tendon_actfrclimited, np.uint8)
    self.tendon_actfrcrange = _frozen(model.tendon_actfrcrange, np.float32)
    self.jnt_actfrclimited = _frozen(model.jnt_actfrclimited, np.uint8)
    self.jnt_actfrcrange = _frozen(model.jnt_actfrcrange, np.float32)
    self.jnt_actgravcomp = _frozen(model.jnt_actgravcomp, np.uint8)
    self.jnt_type = _frozen(model.jnt_type, np.int32)
    self.jnt_qposadr = _frozen(model.jnt_qposadr, np.int32)
    self.jnt_dofadr = _frozen(model.jnt_dofadr, np.int32)
    self.body_weldid = _frozen(model.body_weldid, np.int32)
    self.body_dofadr = _frozen(model.body_dofadr, np.int32)
    self.body_dofnum = _frozen(model.body_dofnum, np.int32)
    self.body_parentid = _frozen(model.body_parentid, np.int32)
    self.body_jntadr = _frozen(model.body_jntadr, np.int32)
    self.body_jntnum = _frozen(model.body_jntnum, np.int32)
    self.dof_parentid = _frozen(model.dof_parentid, np.int32)
    self.site_bodyid = _frozen(model.site_bodyid, np.int32)
    self.disableflags = int(model.opt.disableflags)
    self.disableactuator = int(model.opt.disableactuator)
    self.timestep = float(model.opt.timestep)
    fixed_lmap, fixed_mmap = _fixed_maps(model, nu, nq, nv)
    self.fixed_length_map = _frozen(fixed_lmap, np.float32)
    self.fixed_moment_map = _frozen(fixed_mmap, np.float32)

    # Fast-path check: stateless scalar fixed/affine models keep the
    # pre-qualified scalar stage byte-identical.
    scalar_ok = True
    if na != 0:
      scalar_ok = False
    else:
      for i in range(nu):
        t = int(trntype[i])
        if t not in (trn_joint, trn_parent, trn_tendon):
          scalar_ok = False
          break
        if t in (trn_joint, trn_parent):
          if int(model.jnt_type[int(trnid[i, 0])]) not in (hinge, slide):
            scalar_ok = False
            break
        if t == trn_tendon:
          # Spatial tendon targets need the general path (R1); the scalar
          # fast path only holds fixed-tendon maps.
          tid = int(trnid[i, 0])
          a0 = int(model.tendon_adr[tid])
          if int(model.wrap_type[a0]) != int(mujoco.mjtWrap.mjWRAP_JOINT):
            scalar_ok = False
            break
        if int(gaintype[i]) not in (
            int(mujoco.mjtGain.mjGAIN_FIXED), int(mujoco.mjtGain.mjGAIN_AFFINE)):
          scalar_ok = False
          break
        if int(biastype[i]) not in (
            int(mujoco.mjtBias.mjBIAS_NONE), int(mujoco.mjtBias.mjBIAS_AFFINE)):
          scalar_ok = False
          break
      if (np.any(np.asarray(model.tendon_actfrclimited))
          or np.any(np.asarray(model.jnt_actfrclimited))
          or np.any(np.asarray(model.jnt_actgravcomp))):
        scalar_ok = False
    self.needs_general_path = not scalar_ok
    self.has_body_transmission = bool(np.any(trntype == trn_body)) if nu else False
    self.has_dcmotor_mechanical = bool(np.any(
        (np.asarray(model.actuator_biastype) == int(mujoco.mjtBias.mjBIAS_DCMOTOR)))) if nu else False


def act_dot_reference(meta, ctrl, act, length, velocity):
  """Independent numpy port of the pinned act_dot switch (one environment)."""
  dyn_none = 0  # mjDYN_NONE == 0 in the pinned enum order
  out = np.zeros(meta.na, dtype=np.float64)
  for i in range(meta.nu):
    first = int(meta.actadr[i])
    n = int(meta.actnum[i])
    if n == 0:
      continue
    dt = int(meta.dyntype[i])
    dp = np.asarray(meta.dynprm[i], dtype=np.float64)
    gp = np.asarray(meta.gainprm[i], dtype=np.float64)
    last = first + n - 1
    if dt == 1:  # INTEGRATOR
      out[last] = ctrl[i]
    elif dt in (2, 3):  # FILTER / FILTEREXACT
      tau = max(_MJMINVAL, float(dp[0]))
      out[last] = (ctrl[i] - act[last]) / tau
    elif dt == 4:  # MUSCLE
      out[last] = muscle_dynamics(ctrl[i], act[last], dp[:3])
    elif dt == 5:  # DCMOTOR slot machine
      slots = dcmotor_slots(dp, gp)
      assert slots["num_slots"] == n
      adr = first
      vel = float(velocity[i])
      length_i = float(length[i])
      r, k, ki, te = gp[0], gp[1], gp[5], dp[0]
      u = ctrl[i]
      if slots["slew"] >= 0:
        u_prev = act[adr]
        slew = dp[7] * meta.timestep
        u_eff = _clip(u, u_prev - slew, u_prev + slew)
        out[adr] = (u_eff - u_prev) / meta.timestep
        u = u_eff
        adr += 1
      x_i = 0.0
      if slots["integral"] >= 0:
        x_i = act[adr]
        mode = int(gp[8])
        imax = dp[8]
        a_dot = u
        if mode == 1:
          a_dot = u - length_i
        if imax > 0:
          if x_i >= imax:
            a_dot = min(a_dot, 0.0)
          elif x_i <= -imax:
            a_dot = max(a_dot, 0.0)
        out[adr] = a_dot
        adr += 1
      v = dcmotor_voltage(u, length_i, vel, x_i, gp)
      if slots["temperature"] >= 0:
        c, ta, alpha, t0 = dp[3], dp[4], gp[2], gp[3]
        temp = act[adr]
        rr = r * (1.0 + alpha * (temp + ta - t0))
        current = act[last] if te > 0 else (v - k * vel) / rr
        out[adr] = (rr * current * current - temp / dp[2]) / c
        adr += 1
      if slots["bristle"] >= 0:
        z = act[adr]
        g = lugre_stribeck(vel, meta_bias_f_c(meta, i), meta_bias_f_s(meta, i), meta_bias_v_s(meta, i))
        a = -dp[5] * abs(vel) / max(_MJMINVAL, g)
        out[adr] = a * z + vel
        adr += 1
      if slots["current"] >= 0:
        i_dot = (v / r - k / r * vel - act[last]) / te
        if dp[1] > 0:
          i_dot = _clip(i_dot, -dp[1], dp[1])
        out[last] = i_dot
  return out


def force_reference(meta, ctrl, act, length, velocity, moment, gravcomp=None,
                    disableflags=0, disableactuator=0):
  """Independent numpy port of the pinned force path (one environment).

  Returns dict with actuator `force`, per-type `gain`/`bias`, clipped `ctrl`,
  generalized `qfrc` and the tendon-aggregate/joint-clamp stages applied in
  pinned order. `moment` is the dense [nu, nv] transmission moment.
  """
  import mujoco as _mj
  nu, nv = meta.nu, meta.nv
  trn_tendon = int(_mj.mjtTrn.mjTRN_TENDON)
  ctrl = np.asarray(ctrl, dtype=np.float64).copy()
  act = np.asarray(act, dtype=np.float64)
  if disableflags & int(_mj.mjtDisableBit.mjDSBL_ACTUATION):
    return {"force": np.zeros(nu), "gain": np.zeros(nu), "bias": np.zeros(nu),
            "ctrl": ctrl, "qfrc": np.zeros(nv)}
  if not (disableflags & int(_mj.mjtDisableBit.mjDSBL_CLAMPCTRL)):
    for i in range(nu):
      if bool(np.asarray(meta.ctrllimited)[i]):
        lo, hi = np.asarray(meta.ctrlrange[i], dtype=np.float64)
        ctrl[i] = _clip(ctrl[i], lo, hi)
  disabled = [bool(disableactuator & (1 << int(np.asarray(meta.group)[i]))) for i in range(nu)]
  force = np.zeros(nu)
  gain = np.zeros(nu)
  bias = np.zeros(nu)
  for i in range(nu):
    if disabled[i]:
      continue
    dp = np.asarray(meta.dynprm[i], dtype=np.float64)
    gp = np.asarray(meta.gainprm[i], dtype=np.float64)
    bp = np.asarray(meta.biasprm[i], dtype=np.float64)
    n = int(meta.actnum[i])
    first = int(meta.actadr[i])
    gt = int(meta.gaintype[i])
    u = float(ctrl[i])
    # DCMOTOR slew modifies ctrl shared with the force path (pinned).
    if gt == 3 and int(meta.dyntype[i]) == 5:
      slots = dcmotor_slots(dp, gp)
      if slots["slew"] >= 0 and n > 0:
        u_prev = act[first + slots["slew"]]
        slew = dp[7] * meta.timestep
        u = _clip(u, u_prev - slew, u_prev + slew)
        ctrl[i] = u
    if gt == 0:
      g = gp[0]
    elif gt == 1:
      g = gp[0] + gp[1] * length[i] + gp[2] * velocity[i]
    elif gt == 2:
      g = muscle_gain(length[i], velocity[i],
                      np.asarray(meta.lengthrange[i], dtype=np.float64),
                      float(np.asarray(meta.acc0)[i]), gp)
    else:  # DCMOTOR
      r, k = gp[0], gp[1]
      slots = dcmotor_slots(dp, gp)
      if slots["temperature"] >= 0 and n > 0:
        temp = act[first + slots["temperature"]]
        r = r * (1.0 + gp[2] * (temp + dp[4] - gp[3]))
      g = k if dp[0] > 0 else k / max(_MJMINVAL, r)
      if int(gp[8]) > 0:
        x_i = act[first + slots["integral"]] if (slots["integral"] >= 0 and n > 0) else 0.0
        u = dcmotor_voltage(u, length[i], velocity[i], x_i, gp)
        ctrl[i] = u
    gain[i] = g
    dcmotor_no_current = (gt == 3 and dp[0] <= 0)
    if n == 0 or dcmotor_no_current:
      f = g * u
    else:
      adr = first + n - 1
      if bool(np.asarray(meta.actearly)[i]):
        a = advance_activation_reference(
            meta, act, act_dot_reference(meta, ctrl, act, length, velocity),
            velocity, [False] * nu)[adr]
      else:
        a = act[adr]
      f = g * a
    bt = int(meta.biastype[i])
    if bt == 0:
      b = 0.0
    elif bt == 1:
      b = bp[0] + bp[1] * length[i] + bp[2] * velocity[i]
    elif bt == 2:
      b = muscle_bias(length[i], np.asarray(meta.lengthrange[i], dtype=np.float64),
                      float(np.asarray(meta.acc0)[i]), bp)
    else:  # DCMOTOR back-EMF, stateless only
      b = 0.0
      if dp[0] <= 0:
        b -= g * gp[1] * velocity[i]
    bias[i] = b
    force[i] = f + b
  # Tendon aggregated clamp (pinned order: before force-range clamp).
  force = _tendon_clamp(meta, force, disabled)
  for i in range(nu):
    if disabled[i]:
      continue
    if bool(np.asarray(meta.forcelimited)[i]):
      lo, hi = np.asarray(meta.forcerange[i], dtype=np.float64)
      force[i] = _clip(force[i], lo, hi)
  # DC mechanical forces post-clamp: cogging + LuGre.
  for i in range(nu):
    if disabled[i] or int(meta.biastype[i]) != 3:
      continue
    dp = np.asarray(meta.dynprm[i], dtype=np.float64)
    gp = np.asarray(meta.gainprm[i], dtype=np.float64)
    bp = np.asarray(meta.biasprm[i], dtype=np.float64)
    if bp[0] != 0:
      force[i] += bp[0] * np.sin(bp[1] * length[i] + bp[2])
    if dp[5] > 0 and meta.na > 0:
      slots = dcmotor_slots(dp, gp)
      if slots["bristle"] >= 0:
        adr = int(meta.actadr[i]) + slots["bristle"]
        z = act[adr]
        g = lugre_stribeck(velocity[i], bp[3], bp[4], bp[5])
        force[i] -= dp[5] * z + dp[6] * (act_dot_reference(
            meta, ctrl, act, length, velocity)[adr])
  qfrc = np.asarray(moment, dtype=np.float64).T @ force
  if gravcomp is not None:
    flags = np.asarray(meta.jnt_actgravcomp)
    for j in range(meta.njnt):
      if flags[j]:
        da = int(meta.jnt_dofadr[j])
        nd = {0: 6, 1: 3, 2: 1, 3: 1}[int(meta.jnt_type[j])]
        qfrc[da:da + nd] += np.asarray(gravcomp, dtype=np.float64)[da:da + nd]
  jflags = np.asarray(meta.jnt_actfrclimited)
  if np.any(jflags):
    import mujoco as _mj2
    for j in range(len(jflags)):
      if jflags[j]:
        da = int(meta.jnt_dofadr[j])
        nd = {0: 6, 1: 3, 2: 1, 3: 1}[int(meta.jnt_type[j])]
        lo, hi = np.asarray(meta.jnt_actfrcrange[j], dtype=np.float64)
        qfrc[da:da + nd] = np.clip(qfrc[da:da + nd], lo, hi)
  return {"force": force, "gain": gain, "bias": bias, "ctrl": ctrl, "qfrc": qfrc}


def _tendon_clamp(meta, force, disabled):
  """Pinned tendon-total-force scaling."""
  import mujoco as _mj
  trn_tendon = int(_mj.mjtTrn.mjTRN_TENDON)
  force = np.asarray(force, dtype=np.float64).copy()
  lim = np.asarray(meta.tendon_actfrclimited)
  if not np.any(lim):
    return force
  totals = {}
  for i in range(meta.nu):
    if int(meta.trntype[i]) != trn_tendon:
      continue
    tid = int(meta.trnid[i, 0])
    if lim[tid]:
      totals[tid] = totals.get(tid, 0.0) + force[i]
  for i in range(meta.nu):
    if int(meta.trntype[i]) != trn_tendon or disabled[i]:
      continue
    tid = int(meta.trnid[i, 0])
    if lim[tid] and totals.get(tid, 0.0):
      lo, hi = np.asarray(meta.tendon_actfrcrange[tid], dtype=np.float64)
      t = totals[tid]
      if t < lo:
        force[i] *= lo / t
      elif t > hi:
        force[i] *= hi / t
  return force


def advance_activation_reference(meta, act, act_dot, velocity=None, disabled_mask=None):
  """Independent numpy port of pinned mj_nextActivation + advance loop.

  `velocity` (per-actuator actuator_velocity) is required when any DCMOTOR
  bristle slot exists; otherwise it may be None.
  """
  out = np.asarray(act, dtype=np.float64).copy()
  h = meta.timestep
  for i in range(meta.nu):
    first = int(meta.actadr[i])
    n = int(meta.actnum[i])
    dt = int(meta.dyntype[i])
    dp = np.asarray(meta.dynprm[i], dtype=np.float64)
    gp = np.asarray(meta.gainprm[i], dtype=np.float64)
    bp = np.asarray(meta.biasprm[i], dtype=np.float64)
    vel = float(velocity[i]) if velocity is not None else 0.0
    for j in range(first, first + n):
      dot = 0.0 if (disabled_mask is not None and disabled_mask[i]) else float(act_dot[j])
      if dt == 3:  # FILTEREXACT
        tau = max(_MJMINVAL, float(dp[0]))
        out[j] = out[j] + dot * tau * (1.0 - np.exp(-h / tau))
      elif dt == 5:  # DCMOTOR slot-specific
        slots = dcmotor_slots(dp, gp)
        off = j - first
        if off == slots["current"]:
          te = max(_MJMINVAL, float(dp[0]))
          out[j] = out[j] + dot * te * (1.0 - np.exp(-h / te))
        elif off == slots["bristle"]:
          out[j] = advance_bristle_exact(
              act[j], dot, vel, bp[3], bp[4], bp[5], dp[5], h)
        elif off == slots["integral"]:
          out[j] = out[j] + dot * h
          if dp[8] > 0:
            out[j] = _clip(out[j], -dp[8], dp[8])
        else:
          out[j] = out[j] + dot * h
      else:
        out[j] = out[j] + dot * h
    # Clamp to actrange unless DC motor (pinned never clamps DCMOTOR).
    if dt != 5 and bool(np.asarray(meta.actlimited)[i]):
      lo, hi = np.asarray(meta.actrange[i], dtype=np.float64)
      out[first:first + n] = np.clip(out[first:first + n], lo, hi)
  return out


def advance_bristle_exact(act_z, act_dot_z, velocity, f_c, f_s, v_s, sigma0, h):
  """Exact ZOH bristle advance shared by the reference and tests."""
  g = lugre_stribeck(velocity, f_c, f_s, v_s)
  a = -sigma0 * abs(velocity) / max(_MJMINVAL, g)
  exp_ah = np.exp(a * h)
  int_h = (exp_ah - 1.0) / a if abs(a) > _MJMINVAL else h
  return exp_ah * act_z + int_h * velocity


class MetalActuators:
  """Native MPS full-family actuator kinematics + dynamics stage."""

  def __init__(self, model, batch_size):
    from pathlib import Path as _Path
    import torch as _torch
    self._torch = _torch
    self._meta = ActuatorModel(model)
    self._device = _torch.device("mps")
    tlib = _torch.mps.compile_shader(
        (_Path(__file__).parent / "shaders" / "transmissions.metal").read_text())
    alib = _torch.mps.compile_shader(
        (_Path(__file__).parent / "shaders" / "actuation.metal").read_text())
    self._kin_kernel = tlib.general_actuator_kinematics
    self._body_kernel = tlib.body_adhesion_moment
    self._dot_kernel = alib.actuator_act_dot
    self._force_kernel = alib.actuator_force
    self._qfrc_kernel = alib.actuator_assemble_qfrc
    self._adv_kernel = alib.advance_activations
    meta = self._meta
    b = int(batch_size)
    if b <= 0:
      raise ValueError("batch_size must be positive")

    def tensor(values, dtype=_torch.float32):
      # Metal kernels declare integer buffers as int (4 bytes); uint8 device
      # buffers would overread (see milestone 007 report). Force int32.
      if dtype == _torch.uint8:
        raise ValueError("device flag buffers must be int32, not uint8")
      npdtype = np.int32 if dtype == _torch.int32 else np.float32
      arr = np.array(values, dtype=npdtype, order="C", copy=True)
      if arr.dtype == np.float32 and not np.all(np.isfinite(arr)):
        raise ValueError("actuator constants must be finite float32")
      if arr.size == 0:
        arr = np.zeros(1, dtype=npdtype)
      return _torch.as_tensor(arr, dtype=dtype, device=self._device)

    self.batch_size = b
    self._trntype = tensor(meta.trntype, _torch.int32)
    self._trnid = tensor(meta.trnid.reshape(-1), _torch.int32)
    self._gear = tensor(meta.gear.reshape(-1))
    self._crank = tensor(meta.cranklength.reshape(-1) if meta.nu else np.zeros(1, dtype=np.float32))
    self._flmap = tensor(meta.fixed_length_map.reshape(-1))
    self._fmmap = tensor(meta.fixed_moment_map.reshape(-1))
    self._jnt_type = tensor(meta.jnt_type if meta.njnt else np.zeros(1, dtype=np.int32), _torch.int32)
    self._jnt_qposadr = tensor(meta.jnt_qposadr if meta.njnt else np.zeros(1, dtype=np.int32), _torch.int32)
    self._jnt_dofadr = tensor(meta.jnt_dofadr if meta.njnt else np.zeros(1, dtype=np.int32), _torch.int32)
    self._body_parentid = tensor(meta.body_parentid, _torch.int32)
    self._body_jntadr = tensor(meta.body_jntadr, _torch.int32)
    self._body_jntnum = tensor(meta.body_jntnum, _torch.int32)
    self._body_weldid = tensor(meta.body_weldid, _torch.int32)
    self._body_dofadr = tensor(meta.body_dofadr, _torch.int32)
    self._body_dofnum = tensor(meta.body_dofnum, _torch.int32)
    self._dof_parentid = tensor(meta.dof_parentid if meta.nv else np.zeros(1, dtype=np.int32), _torch.int32)
    self._site_bodyid = tensor(meta.site_bodyid if meta.nsite else np.zeros(1, dtype=np.int32), _torch.int32)
    self._kin_dims = tensor([meta.nq, meta.nv, meta.nu, meta.nbody,
                             max(meta.njnt, 0), meta.nsite, meta.ntendon, b], _torch.int32)
    self._dyntype = tensor(meta.dyntype, _torch.int32)
    self._dynprm = tensor(meta.dynprm.reshape(-1))
    self._gaintype = tensor(meta.gaintype, _torch.int32)
    self._gainprm = tensor(meta.gainprm.reshape(-1))
    self._biastype = tensor(meta.biastype, _torch.int32)
    self._biasprm = tensor(meta.biasprm.reshape(-1))
    self._actadr = tensor(meta.actadr if meta.nu else np.zeros(1, dtype=np.int32), _torch.int32)
    self._actnum = tensor(meta.actnum if meta.nu else np.zeros(1, dtype=np.int32), _torch.int32)
    self._actearly = tensor(meta.actearly if meta.nu else np.zeros(1, dtype=np.uint8), _torch.int32)
    self._ctrllimited = tensor(meta.ctrllimited, _torch.int32)
    self._ctrlrange = tensor(meta.ctrlrange.reshape(-1))
    self._forcelimited = tensor(meta.forcelimited, _torch.int32)
    self._forcerange = tensor(meta.forcerange.reshape(-1))
    self._tendon_limited = tensor(meta.tendon_actfrclimited if meta.ntendon else np.zeros(1, dtype=np.uint8), _torch.int32)
    self._tendon_range = tensor(meta.tendon_actfrcrange.reshape(-1) if meta.ntendon else np.zeros(2, dtype=np.float32))
    nj = max(meta.njnt, 1)
    self._jnt_limited = tensor(meta.jnt_actfrclimited if meta.njnt else np.zeros(1, dtype=np.uint8), _torch.int32)
    self._jnt_range = tensor(meta.jnt_actfrcrange.reshape(-1) if meta.njnt else np.zeros(2, dtype=np.float32))
    self._jnt_gravcomp = tensor(meta.jnt_actgravcomp if meta.njnt else np.zeros(1, dtype=np.uint8), _torch.int32)
    self._group = tensor(meta.group, _torch.int32)
    self._lengthrange = tensor(meta.lengthrange.reshape(-1))
    self._acc0 = tensor(meta.acc0.reshape(-1) if meta.nu else np.zeros(1, dtype=np.float32))
    self._actlimited = tensor(meta.actlimited if meta.nu else np.zeros(1, dtype=np.uint8), _torch.int32)
    self._actrange = tensor(meta.actrange.reshape(-1) if meta.nu else np.zeros(2, dtype=np.float32))
    self._dyn_dims = tensor([meta.nv, meta.nu, meta.na, b,
                             int(bool(meta.disableflags & int(mujoco.mjtDisableBit.mjDSBL_ACTUATION))),
                             int(bool(meta.disableflags & int(mujoco.mjtDisableBit.mjDSBL_CLAMPCTRL))),
                             meta.disableactuator, meta.ntendon, nj if meta.njnt else 0],
                            _torch.int32)
    self._dot_dims = tensor([meta.nu, meta.na, b,
                             int(bool(meta.disableflags & int(mujoco.mjtDisableBit.mjDSBL_ACTUATION))),
                             int(bool(meta.disableflags & int(mujoco.mjtDisableBit.mjDSBL_CLAMPCTRL)))],
                            _torch.int32)
    self._force_dims = tensor([meta.nu, meta.na, b,
                               int(bool(meta.disableflags & int(mujoco.mjtDisableBit.mjDSBL_ACTUATION))),
                               meta.disableactuator, meta.ntendon],
                              _torch.int32)
    self._qfrc_dims = tensor([meta.nv, meta.nu, b, meta.njnt], _torch.int32)
    self._adv_dims = tensor([meta.nu, meta.na, b, meta.disableactuator], _torch.int32)
    self._dt = tensor(np.array([meta.timestep], dtype=np.float32))
    self._dummy = _torch.zeros(1, dtype=_torch.float32, device=self._device)
    nv, nu, na = max(meta.nv, 1), max(meta.nu, 1), max(meta.na, 1)
    self._ws = {
        "length": _torch.zeros(b * nu, dtype=_torch.float32, device=self._device),
        "velocity": _torch.zeros(b * nu, dtype=_torch.float32, device=self._device),
        "moment": _torch.zeros(b * nu * nv, dtype=_torch.float32, device=self._device),
        "act_dot": _torch.zeros(b * na, dtype=_torch.float32, device=self._device),
        "force": _torch.zeros(b * nu, dtype=_torch.float32, device=self._device),
        "ctrl_used": _torch.zeros(b * nu, dtype=_torch.float32, device=self._device),
        "qfrc": _torch.zeros(b * nv, dtype=_torch.float32, device=self._device),
        "act_next": _torch.zeros(b * na, dtype=_torch.float32, device=self._device),
    }

  @property
  def meta(self):
    return self._meta

  def _check(self, value, name, shape):
    torch = self._torch
    if not isinstance(value, torch.Tensor) or value.ndim != len(shape):
      raise TypeError(f"{name} must be a rank-{len(shape)} torch.Tensor")
    if tuple(value.shape) != tuple(shape):
      raise ValueError(f"{name} must have shape {shape}")
    if value.dtype != torch.float32 or value.device.type != "mps" or not value.is_contiguous():
      raise ValueError(f"{name} must be contiguous float32 MPS")

  def run_kinematics(self, qpos, qvel, poses, contacts=None):
    """Compute length/velocity/dense-moment from borrowed device state."""
    torch = self._torch
    meta = self._meta
    b, nq, nv, nu = self.batch_size, meta.nq, meta.nv, meta.nu
    if nq:
      self._check(qpos, "qpos", (b, nq))
    self._check(qvel, "qvel", (b, max(nv, 1)))
    w = self._ws
    qpos_flat = qpos.reshape(-1) if nq else self._dummy
    qvel_flat = qvel.reshape(-1) if nv else self._dummy
    args = [
        qpos_flat, qvel_flat,
        poses["body_pos"].reshape(-1), poses["body_quat"].reshape(-1),
        poses["site_pos"].reshape(-1) if meta.nsite else self._dummy,
        poses["site_quat"].reshape(-1) if meta.nsite else self._dummy,
        poses["joint_anchor"].reshape(-1), poses["joint_axis"].reshape(-1),
        self._trntype, self._trnid, self._gear, self._crank,
        self._flmap, self._fmmap,
        self._jnt_type, self._jnt_qposadr, self._jnt_dofadr,
        self._body_parentid, self._body_jntadr, self._body_jntnum,
        self._body_weldid, self._body_dofadr, self._body_dofnum,
        self._dof_parentid, self._site_bodyid,
        self._kin_dims,
        w["length"], w["velocity"], w["moment"],
    ]
    self._kin_kernel(*args, threads=(b,), group_size=(1,))
    if meta.has_body_transmission:
      if contacts is None:
        raise ValueError("BODY transmissions require same-step candidate contacts")
      self._body_kernel(
          w["moment"],
          contacts["frame"], contacts["jacobian"],
          contacts["pair_geoms"], contacts["geom_bodyid"], contacts["pair_offset"],
          self._trntype, self._trnid,
          contacts["dims"],
          threads=(b,), group_size=(1,),
      )
      # Velocity follows the completed moment rows (pinned: velocity = moment*qvel).
      # Recompute host-side via a second kernel is wasteful; fold here on CPU? No:
      # recompute in the force stage instead. Mark velocity stale for BODY rows.
      # Simplest correct: recompute velocity rows for BODY actuators on device
      # with a tiny inline loop is unavailable; instead the force kernel takes
      # velocity as input, so refresh it here with torch ops (device, no readback).
      with torch.no_grad():
        mom = w["moment"].reshape(b, nu, max(nv, 1))
        vel = w["velocity"].reshape(b, nu)
        qv = qvel.reshape(b, max(nv, 1)) if nv else torch.zeros((b, 1), device=self._device)
        vel.copy_((mom * qv.unsqueeze(1)).sum(-1))
    return {"length": w["length"].reshape(b, nu),
            "velocity": w["velocity"].reshape(b, nu),
            "moment": w["moment"].reshape(b, nu, max(nv, 1))}

  def run_forces(self, ctrl, act, kin, gravcomp=None):
    """Compute act_dot/force/qfrc from held inputs and kinematics outputs."""
    torch = self._torch
    meta = self._meta
    b, nv, nu, na = self.batch_size, meta.nv, meta.nu, meta.na
    self._check(ctrl, "ctrl", (b, nu))
    if na:
      self._check(act, "act", (b, na))
      act_flat = act.reshape(-1)
    else:
      act_flat = self._dummy
    w = self._ws
    if gravcomp is None:
      grav = torch.zeros((b * max(nv, 1),), dtype=torch.float32, device=self._device)
    else:
      self._check(gravcomp, "gravcomp", (b, max(nv, 1)))
      grav = gravcomp.reshape(-1)
    kin_len = kin["length"].reshape(-1)
    kin_vel = kin["velocity"].reshape(-1)
    kin_mom = kin["moment"].reshape(-1)
    self._dot_kernel(
        ctrl.reshape(-1), act_flat, kin_len, kin_vel,
        self._dyntype, self._dynprm, self._gainprm, self._biasprm,
        self._actadr, self._actnum, self._ctrllimited, self._ctrlrange,
        self._dot_dims, self._dt,
        w["act_dot"], w["ctrl_used"],
        threads=(b,), group_size=(1,),
    )
    self._force_kernel(
        w["ctrl_used"], act_flat, w["act_dot"], kin_len, kin_vel,
        self._gaintype, self._gainprm, self._biastype, self._biasprm,
        self._actadr, self._actnum, self._actearly,
        self._dyntype, self._dynprm,
        self._forcelimited, self._forcerange,
        self._trntype, self._trnid, self._tendon_limited, self._tendon_range,
        self._group, self._lengthrange, self._acc0,
        self._force_dims, self._dt,
        w["force"], w["ctrl_used"],
        threads=(b,), group_size=(1,),
    )
    self._qfrc_kernel(
        w["force"], kin_mom, grav,
        self._jnt_limited, self._jnt_range, self._jnt_dofadr, self._jnt_type,
        self._jnt_gravcomp, self._qfrc_dims, w["qfrc"],
        threads=(b,), group_size=(1,),
    )
    return {"act_dot": w["act_dot"].reshape(b, max(na, 1)),
            "force": w["force"].reshape(b, nu),
            "ctrl": w["ctrl_used"].reshape(b, nu),
            "qfrc": w["qfrc"].reshape(b, max(nv, 1))}

  def advance(self, act, act_dot, velocity):
    """Integrate activations one Euler step with exact slot forms; returns borrowed view."""
    torch = self._torch
    meta = self._meta
    b, nu, na = self.batch_size, meta.nu, meta.na
    if na == 0:
      raise ValueError("model has no activation state")
    self._check(act, "act", (b, na))
    self._check(act_dot, "act_dot", (b, na))
    self._check(velocity, "velocity", (b, nu))
    w = self._ws
    self._adv_kernel(
        act.reshape(-1), act_dot.reshape(-1), velocity.reshape(-1),
        self._dyntype, self._dynprm, self._gainprm, self._biasprm,
        self._actadr, self._actnum, self._actlimited,
        self._actrange,
        self._group, self._adv_dims, self._dt,
        w["act_next"],
        threads=(b,), group_size=(1,),
    )
    return w["act_next"].reshape(b, na)


def advance_bristle_exact(act_z, act_dot_z, velocity, f_c, f_s, v_s, sigma0, h):
  """Exact ZOH bristle advance shared by the reference and tests."""
  g = lugre_stribeck(velocity, f_c, f_s, v_s)
  a = -sigma0 * abs(velocity) / max(_MJMINVAL, g)
  exp_ah = np.exp(a * h)
  int_h = (exp_ah - 1.0) / a if abs(a) > _MJMINVAL else h
  return exp_ah * act_z + int_h * velocity
