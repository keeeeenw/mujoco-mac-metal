# Copyright 2026 The MuJoCo Metal contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""CPU-only model-constant and batched kinematics state lifecycle."""

from copy import copy
from dataclasses import dataclass
from dataclasses import fields
import hashlib
from pathlib import Path

import mujoco
import numpy as np

from mujoco_metal.model import load_model
from mujoco_metal.model import ModelDescriptor
from mujoco_metal.model import snapshot_descriptor


def _fingerprint(model):
  digest = hashlib.sha256(b"mujoco-metal-lifecycle-v1\0")
  if isinstance(model, mujoco.MjModel):
    buffer = np.empty(mujoco.mj_sizeModel(model), dtype=np.uint8)
    mujoco.mj_saveModel(model, buffer=buffer)
    digest.update(buffer.tobytes())
    return digest.hexdigest()
  for name in (
      "nq",
      "nv",
      "nu",
      "nmocap",
      "nbody",
      "njnt",
      "ngeom",
      "nsite",
      "ntendon",
      "disableflags",
  ):
    digest.update(int(getattr(model, name)).to_bytes(8, "little"))
  for item in fields(ModelDescriptor):
    name = item.name
    value = getattr(model, name)
    if not isinstance(value, np.ndarray):
      continue
    value = np.ascontiguousarray(getattr(model, name))
    digest.update(name.encode())
    digest.update(value.dtype.str.encode())
    digest.update(value.tobytes())
  return digest.hexdigest()


def _perturb_quaternions(base, rotation_vector):
  result = np.empty_like(base)
  for index, (quat, vector) in enumerate(zip(base, rotation_vector)):
    angle = np.linalg.norm(vector)
    if angle == 0:
      result[index] = quat / np.linalg.norm(quat)
      continue
    delta = np.r_[np.cos(angle / 2), vector * (np.sin(angle / 2) / angle)]
    w, x, y, z = quat / np.linalg.norm(quat)
    a, b, c, d = delta
    result[index] = [
        w * a - x * b - y * c - z * d,
        w * b + x * a + y * d - z * c,
        w * c - x * d + y * a + z * b,
        w * d + x * c - y * b + z * a,
    ]
  return result


def _clone_model(model):
  """Copy an MjModel with MuJoCo's supported Python copy semantics."""
  return copy(model)


_FRAME_EPS = 1e-6  # MuJoCo 3.10.0 src/user/user_model.cc:kFrameEps


def _is_null_pose(pos, quat):
  return (_is_same_vector(pos, np.zeros(3)) and
          _is_same_quaternion(quat, np.array([1., 0., 0., 0.])))


def _is_same_vector(a, b):
  return bool(np.all(np.abs(np.asarray(a) - np.asarray(b)) < _FRAME_EPS))


def _is_same_quaternion(a, b):
  a, b = np.asarray(a), np.asarray(b)
  return bool(np.all(np.abs(a - b) < _FRAME_EPS) or
              np.all(np.abs(a + b) < _FRAME_EPS))


def _is_same_pose(pos_a, pos_b, quat_a, quat_b):
  return (_is_same_vector(pos_a, pos_b) and
          _is_same_quaternion(quat_a, quat_b))


def _refresh_compiler_frame_flags(model):
  """Refresh compile-time frame flags invalidated by frame/inertia edits.

  ``mj_setConst`` recomputes dynamics constants, but it does not rerun the
  compiler's ``body_sameframe``, ``body_simple``, ``geom_sameframe`` or
  ``site_sameframe`` classification.  Those flags select optimized kinematics
  paths, so stale classifications can make updated models use old transforms.
  This follows MuJoCo 3.10.0 ``user_model.cc``'s sameframe classification.
  """
  body_code = mujoco.mjtSameFrame.mjSAMEFRAME_BODY.value
  inertia_code = mujoco.mjtSameFrame.mjSAMEFRAME_INERTIA.value
  bodyrot_code = mujoco.mjtSameFrame.mjSAMEFRAME_BODYROT.value
  inertiarot_code = mujoco.mjtSameFrame.mjSAMEFRAME_INERTIAROT.value
  none_code = mujoco.mjtSameFrame.mjSAMEFRAME_NONE.value

  def classify_local(pos, quat, ipos, iquat):
    if _is_null_pose(pos, quat):
      return body_code
    if _is_null_pose(np.zeros(3), quat):
      return bodyrot_code
    if _is_same_pose(pos, ipos, quat, iquat):
      return inertia_code
    if _is_same_quaternion(quat, iquat):
      return inertiarot_code
    return none_code

  body_sameframe = np.empty(model.nbody, dtype=model.body_sameframe.dtype)
  for body in range(model.nbody):
    ipos = np.asarray(model.body_ipos[body])
    iquat = np.asarray(model.body_iquat[body])
    if _is_null_pose(ipos, iquat):
      body_sameframe[body] = body_code
    elif _is_null_pose(np.zeros(3), iquat):
      body_sameframe[body] = bodyrot_code
    else:
      body_sameframe[body] = none_code
  model.body_sameframe[:] = body_sameframe

  # Reproduce user_model.cc's simple-body classification using compiled
  # topology/joint arrays, including parent demotion and slide-only level 2.
  simple = np.zeros(model.nbody, dtype=model.body_simple.dtype)
  for body in range(model.nbody):
    parent = int(model.body_parentid[body])
    if body_sameframe[body] == body_code and (
        int(model.body_rootid[body]) == body or
        (int(model.body_parentid[parent]) == 0 and
         int(model.body_dofnum[parent]) == 0)):
      simple[body] = 1
      first = int(model.body_jntadr[body])
      count = int(model.body_jntnum[body])
      rotfound = False
      for joint in range(first, first + count):
        jtype = model.jnt_type[joint]
        jpos = model.jnt_pos[joint]
        axis = model.jnt_axis[joint]
        # MuJoCo 3.10.0 src/user/user_util.h:mjEPS.
        axis_aligned = int(np.count_nonzero(np.abs(axis) > 1e-14)) == 1
        if (rotfound or not _is_same_vector(jpos, np.zeros(3)) or
            (jtype in (mujoco.mjtJoint.mjJNT_HINGE,
                       mujoco.mjtJoint.mjJNT_SLIDE) and not axis_aligned)):
          simple[body] = 0
        if jtype in (mujoco.mjtJoint.mjJNT_BALL,
                     mujoco.mjtJoint.mjJNT_HINGE):
          rotfound = True
      if simple[body] and int(model.body_dofnum[body]):
        first = int(model.body_jntadr[body])
        count = int(model.body_jntnum[body])
        simple[body] = 2
        if any(model.jnt_type[j] != mujoco.mjtJoint.mjJNT_SLIDE
               for j in range(first, first + count)):
          simple[body] = 1
  for body in range(1, model.nbody):
    parent = int(model.body_parentid[body])
    if parent > 0:
      simple[parent] = 0

  # The compiler performs a final tendon-armature demotion after its initial
  # body classification (user_model.cc:FinalizeSimple). Keep that rule in
  # sync before considering sparse DOF structure.
  wrap_site = mujoco.mjtWrap.mjWRAP_SITE.value
  wrap_cylinder = mujoco.mjtWrap.mjWRAP_CYLINDER.value
  wrap_sphere = mujoco.mjtWrap.mjWRAP_SPHERE.value
  for tendon in range(model.ntendon):
    if model.tendon_armature[tendon] == 0:
      continue
    begin = int(model.tendon_adr[tendon])
    end = begin + int(model.tendon_num[tendon])
    for item in range(begin, end):
      kind = int(model.wrap_type[item])
      object_id = int(model.wrap_objid[item])
      if kind == wrap_site:
        simple[int(model.site_bodyid[object_id])] = 0
      elif kind in (wrap_cylinder, wrap_sphere):
        simple[int(model.geom_bodyid[object_id])] = 0

  # mjModel sparse buffers have compile-time allocation sizes. FinalizeSimple
  # derives dof_simplenum after tendon demotion and checks nC against the
  # earlier ComputeSparseSizes allocation. A lifecycle update cannot resize
  # those arrays, so fail atomically whenever the required sparse pattern
  # changes instead of leaving stale indices behind.
  simplenum = np.zeros(model.nv, dtype=model.dof_simplenum.dtype)
  count = 0
  for dof in range(model.nv - 1, -1, -1):
    body = int(model.dof_bodyid[dof])
    if simple[body]:
      count += 1
    else:
      count = 0
    simplenum[dof] = count
  n_offdiag = 0
  for dof in range(model.nv):
    if simplenum[dof] == 0:
      ancestor = dof
      while ancestor >= 0:
        if ancestor != dof:
          n_offdiag += 1
        ancestor = int(model.dof_parentid[ancestor])
  required_nC = n_offdiag + int(model.nv)
  if required_nC != int(model.nC) or not np.array_equal(
      simplenum, np.asarray(model.dof_simplenum)):
    raise ValueError(
        "model update changes the compiled sparse inertia layout; "
        "recompile the model before applying this inertia/reference update")

  model.body_simple[:] = simple
  model.dof_simplenum[:] = simplenum

  for geom in range(model.ngeom):
    body = int(model.geom_bodyid[geom])
    model.geom_sameframe[geom] = classify_local(
        model.geom_pos[geom], model.geom_quat[geom],
        model.body_ipos[body], model.body_iquat[body])
  for site in range(model.nsite):
    body = int(model.site_bodyid[site])
    model.site_sameframe[site] = classify_local(
        model.site_pos[site], model.site_quat[site],
        model.body_ipos[body], model.body_iquat[body])


def _compile_source(source):
  if isinstance(source, mujoco.MjModel):
    return _clone_model(source)
  if isinstance(source, bytes):
    return mujoco.MjModel.from_xml_string(source.decode("utf-8"))
  if isinstance(source, Path) or (
      isinstance(source, str) and "<" not in source
  ):
    return mujoco.MjModel.from_xml_path(str(source))
  return mujoco.MjModel.from_xml_string(str(source))


class ModelLifecycle:
  """Own constants and recomputes derived fields transactionally on CPU.

  This is a host preparation utility: it compiles and edits models and
  validates parameter updates, but it does not itself advance physics. To run
  the edited model natively, adopt it into a live simulation with
  ``MetalSimulation.apply_lifecycle`` (which rebuilds device stages
  atomically); holding only a lifecycle never changes simulation behavior.
  """

  def __init__(self, source):
    if mujoco.__version__ != "3.10.0":
      raise RuntimeError(f"requires MuJoCo 3.10.0; found {mujoco.__version__}")
    self._model = _compile_source(source)
    if self._model.nmocap:
      mocapid = np.asarray(self._model.body_mocapid)
      for bid in range(int(self._model.nbody)):
        if int(mocapid[bid]) >= 0 and int(self._model.body_parentid[bid]) != 0:
          raise ValueError(
              f"mocap body {bid} must be a direct child of world"
          )
    self.descriptor = load_model(self._model)
    self.generation = 0

  def update_geom_contact(self, geom_ids, friction=None, solref=None,
                          solimp=None, margin=None, gap=None):
    """Update geom contact parameters transactionally; returns True if changed.

    Only contact-frame parameters with no derived constants are mutable here:
    `friction` (...,3), `solref` (...,2), `solimp` (...,5), `margin`/`gap`
    scalars. Geometry shape/type/affinity and reference poses are
    recompile-required and rejected here (build a new lifecycle/simulation).
    Validation is atomic: bad input leaves the model unchanged.
    """
    raw_ids = np.asarray(geom_ids)
    if raw_ids.dtype.kind not in "iu":
      raise ValueError("geom_ids must contain integers")
    ids = raw_ids.astype(np.int64, copy=False)
    if ids.ndim != 1 or ids.size == 0:
      raise ValueError("geom_ids must be a nonempty vector")
    if np.unique(ids).size != ids.size:
      raise ValueError("geom_ids must be unique")
    if np.any(ids < 0) or np.any(ids >= self.descriptor.ngeom):
      raise ValueError("geom_ids out of range")
    n = ids.size
    updates = {}
    if friction is not None:
      arr = np.asarray(friction, dtype=np.float64)
      if arr.shape != (n, 3):
        raise ValueError(f"friction must have shape ({n}, 3)")
      if not np.all(np.isfinite(arr)) or np.any(arr[:, 0] < 0) or np.any(arr[:, 1:] < 0):
        raise ValueError("friction must be finite and nonnegative")
      with np.errstate(over="ignore", under="ignore"):
        f32 = np.asarray(arr, dtype=np.float32)
      if not np.all(np.isfinite(f32)):
        raise ValueError("friction must be representable as finite float32")
      updates["geom_friction"] = f32
    if solref is not None:
      arr = np.asarray(solref, dtype=np.float64)
      if arr.shape != (n, 2):
        raise ValueError(f"solref must have shape ({n}, 2)")
      if not np.all(np.isfinite(arr)):
        raise ValueError("solref must be finite")
      with np.errstate(over="ignore", under="ignore"):
        f32 = np.asarray(arr, dtype=np.float32)
      if not np.all(np.isfinite(f32)):
        raise ValueError("solref must be representable as finite float32")
      updates["geom_solref"] = f32
    if solimp is not None:
      arr = np.asarray(solimp, dtype=np.float64)
      if arr.shape != (n, 5):
        raise ValueError(f"solimp must have shape ({n}, 5)")
      if not np.all(np.isfinite(arr)):
        raise ValueError("solimp must be finite")
      with np.errstate(over="ignore", under="ignore"):
        f32 = np.asarray(arr, dtype=np.float32)
      if not np.all(np.isfinite(f32)):
        raise ValueError("solimp must be representable as finite float32")
      updates["geom_solimp"] = f32
    if margin is not None:
      arr = np.asarray(margin, dtype=np.float64).reshape(n)
      if arr.shape != (n,):
        raise ValueError(f"margin must have shape ({n},)")
      if not np.all(np.isfinite(arr)):
        raise ValueError("margin must be finite")
      with np.errstate(over="ignore", under="ignore"):
        f32 = np.asarray(arr, dtype=np.float32)
      if not np.all(np.isfinite(f32)):
        raise ValueError("margin must be representable as finite float32")
      updates["geom_margin"] = f32
    if gap is not None:
      arr = np.asarray(gap, dtype=np.float64).reshape(n)
      if arr.shape != (n,):
        raise ValueError(f"gap must have shape ({n},)")
      if not np.all(np.isfinite(arr)):
        raise ValueError("gap must be finite")
      with np.errstate(over="ignore", under="ignore"):
        f32 = np.asarray(arr, dtype=np.float32)
      if not np.all(np.isfinite(f32)):
        raise ValueError("gap must be representable as finite float32")
      updates["geom_gap"] = f32
    if not updates:
      raise ValueError("no contact parameters given")
    # Detect no-op before cloning.
    changed = False
    for name, value in updates.items():
      current = np.asarray(getattr(self._model, name))[ids]
      if not np.array_equal(np.asarray(current, dtype=np.float32), value):
        changed = True
        break
    if not changed:
      return False
    candidate = _clone_model(self._model)
    for name, value in updates.items():
      getattr(candidate, name)[ids] = value
    # Contact parameters have no mj_setConst derived state; lowering validates.
    descriptor = load_model(candidate)
    self._model = candidate
    self.descriptor = descriptor
    self.generation += 1
    return True

  def update_reference_frames(
      self, *, body_ids=None, body_pos=None, body_quat=None,
      geom_ids=None, geom_pos=None, geom_quat=None,
      site_ids=None, site_pos=None, site_quat=None):
    """Update selected compiled body/geom/site reference frames atomically.

    These are shared model parameters, not per-world state. Body ids name
    non-world bodies; geom/site ids may include world-attached records. Each
    supplied pose field must have its matching id vector and exact ``(n, 3)``
    or ``(n, 4)`` shape. Quaternions are normalized in float32 before commit.
    The candidate is passed through ``mj_setConst`` and model lowering before
    replacing this lifecycle, so simulations can adopt it through
    ``apply_lifecycle`` without stale transforms or derived constants. Updates
    that alter the compiled ``dof_simplenum`` sparse pattern are rejected
    atomically and require recompiling the model because ``mjModel`` sparse
    buffers have fixed compile-time sizes.
    """
    plans = (
        ("body", body_ids, body_pos, body_quat, self.descriptor.nbody, True),
        ("geom", geom_ids, geom_pos, geom_quat, self.descriptor.ngeom, False),
        ("site", site_ids, site_pos, site_quat, self.descriptor.nsite, False),
    )
    updates = []
    for label, raw_ids, pos, quat, limit, body_records in plans:
      if raw_ids is None:
        if pos is not None or quat is not None:
          raise ValueError(f"{label}_ids are required for pose updates")
        continue
      ids0 = np.asarray(raw_ids)
      if ids0.dtype.kind not in "iu":
        raise ValueError(f"{label}_ids must contain integers")
      ids = ids0.astype(np.int64, copy=False)
      if ids.ndim != 1 or ids.size == 0 or np.unique(ids).size != ids.size:
        raise ValueError(f"{label}_ids must be a nonempty vector of unique indices")
      if body_records and np.any(ids == 0):
        raise ValueError("body reference updates cannot move the world body")
      if np.any(ids < 0) or np.any(ids >= limit):
        raise ValueError(f"{label}_ids out of range")
      if pos is None and quat is None:
        raise ValueError(f"{label} pose update must supply position or quaternion")
      if pos is not None:
        value = np.asarray(pos, dtype=np.float64)
        if value.shape != (ids.size, 3) or not np.all(np.isfinite(value)):
          raise ValueError(f"{label}_pos must be finite with shape ({ids.size}, 3)")
        with np.errstate(over="ignore", under="ignore", invalid="ignore"):
          value32 = np.asarray(value, dtype=np.float32)
        if not np.all(np.isfinite(value32)):
          raise ValueError(f"{label}_pos must be representable as finite float32")
        updates.append((label, ids, "pos", value32))
      if quat is not None:
        value = np.asarray(quat, dtype=np.float64)
        if value.shape != (ids.size, 4) or not np.all(np.isfinite(value)):
          raise ValueError(f"{label}_quat must be finite with shape ({ids.size}, 4)")
        norms = np.linalg.norm(value, axis=1)
        if np.any(~np.isfinite(norms)) or np.any(norms <= np.finfo(np.float64).tiny):
          raise ValueError(f"{label}_quat must be nonzero")
        with np.errstate(over="ignore", under="ignore", invalid="ignore"):
          value32 = np.asarray(value / norms[:, None], dtype=np.float32)
        norms32 = np.linalg.norm(value32.astype(np.float64), axis=1)
        if (not np.all(np.isfinite(value32)) or np.any(norms32 <= 0)
            or np.any(~np.isfinite(norms32))):
          raise ValueError(f"{label}_quat must remain finite and nonzero in float32")
        value32 = np.asarray(value32 / norms32[:, None], dtype=np.float32)
        updates.append((label, ids, "quat", value32))
    if not updates:
      raise ValueError("no reference frame updates given")

    changed = False
    for label, ids, field, value in updates:
      current = getattr(self._model, f"{label}_{field}")[ids]
      if not np.array_equal(np.asarray(current, dtype=np.float32), value):
        changed = True
        break
    if not changed:
      return False
    candidate = _clone_model(self._model)
    for label, ids, field, value in updates:
      getattr(candidate, f"{label}_{field}")[ids] = value
    _refresh_compiler_frame_flags(candidate)
    mujoco.mj_setConst(candidate, mujoco.MjData(candidate))
    descriptor = load_model(candidate)
    self._model = candidate
    self.descriptor = descriptor
    self.generation += 1
    return True

  def recompute_body_inertias(self, body_ids, inertias, inertial_quats=None):
    """Set principal inertias/optional inertial frames with `mj_setConst`.

    Updates are shared by every environment using this model. Body IDs must
    name non-world bodies; massless bodies must remain zero-inertia, while
    positive-mass bodies require three finite positive principal moments.
    An optional quaternion uses MuJoCo's ``wxyz`` convention and is
    normalized before commit. Candidate mutation, constant recomputation and
    descriptor lowering all finish before the lifecycle is replaced. An update
    that changes the compiled sparse inertia layout is rejected atomically;
    ``mj_setConst`` cannot resize the compiled mass-matrix buffers.
    """
    raw_ids = np.asarray(body_ids)
    if raw_ids.dtype.kind not in "iu":
      raise ValueError("body_ids must contain integers")
    ids = raw_ids.astype(np.int64, copy=False)
    values = np.asarray(inertias, dtype=np.float64)
    if ids.ndim != 1 or values.shape != (ids.size, 3) or ids.size == 0:
      raise ValueError("body_ids and inertias must have shapes (n,) and (n, 3)")
    if np.unique(ids).size != ids.size:
      raise ValueError("body_ids must be unique")
    if np.any(ids <= 0) or np.any(ids >= self.descriptor.nbody):
      raise ValueError("body_ids must refer to non-world bodies")
    masses = np.asarray(self._model.body_mass)[ids]
    if not np.all(np.isfinite(values)) or np.any(values < 0):
      raise ValueError("inertias must be finite and nonnegative")
    if np.any((masses > 0)[:, None] & (values <= 0)):
      raise ValueError("positive-mass bodies require positive principal inertias")
    if np.any((masses == 0)[:, None] & (values != 0)):
      raise ValueError("massless bodies must retain zero principal inertias")
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
      inertia32 = np.asarray(values, dtype=np.float32)
    if not np.all(np.isfinite(inertia32)):
      raise ValueError("inertias must be representable as finite float32")
    if np.any((masses > 0)[:, None] & (inertia32 <= 0)):
      raise ValueError("positive principal inertias must remain positive in float32")

    quat32 = None
    if inertial_quats is not None:
      quats = np.asarray(inertial_quats, dtype=np.float64)
      if quats.shape != (ids.size, 4) or not np.all(np.isfinite(quats)):
        raise ValueError("inertial_quats must be finite with shape (n, 4)")
      norms = np.linalg.norm(quats, axis=1)
      if np.any(norms <= np.finfo(np.float64).tiny):
        raise ValueError("inertial_quats must be nonzero")
      with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        quat32 = np.asarray(quats / norms[:, None], dtype=np.float32)
      if not np.all(np.isfinite(quat32)):
        raise ValueError("inertial_quats must be representable as finite float32")
      quat_norms = np.linalg.norm(quat32.astype(np.float64), axis=1)
      if np.any(quat_norms <= 0):
        raise ValueError("inertial_quats must remain nonzero in float32")
      quat32 = np.asarray(quat32 / quat_norms[:, None], dtype=np.float32)

    same_inertia = np.array_equal(
        np.asarray(self._model.body_inertia)[ids], inertia32)
    same_quat = (quat32 is None or np.array_equal(
        np.asarray(self._model.body_iquat)[ids], quat32))
    if same_inertia and same_quat:
      return False

    candidate = _clone_model(self._model)
    candidate.body_inertia[ids] = inertia32
    if quat32 is not None:
      candidate.body_iquat[ids] = quat32
    _refresh_compiler_frame_flags(candidate)
    mujoco.mj_setConst(candidate, mujoco.MjData(candidate))
    descriptor = load_model(candidate)
    self._model = candidate
    self.descriptor = descriptor
    self.generation += 1
    return True

  def recompute_body_masses(self, body_ids, masses):
    """Set body masses, run `mj_setConst`, and commit only a valid candidate."""
    raw_ids = np.asarray(body_ids)
    if raw_ids.dtype.kind not in "iu":
      raise ValueError("body_ids must contain integers")
    ids = raw_ids.astype(np.int64, copy=False)
    values = np.asarray(masses, dtype=np.float64)
    if ids.ndim != 1 or values.shape != ids.shape or ids.size == 0:
      raise ValueError("body_ids and masses must be matching nonempty vectors")
    if np.unique(ids).size != ids.size:
      raise ValueError("body_ids must be unique")
    if np.any(ids <= 0) or np.any(ids >= self.descriptor.nbody):
      raise ValueError("body_ids must refer to non-world bodies")
    base_mass = self.descriptor.body_mass[ids]
    if (
        not np.all(np.isfinite(values))
        or np.any((base_mass > 0) & (values <= 0))
        or np.any((base_mass == 0) & (values != 0))
    ):
      raise ValueError(
          "masses must preserve zero-mass fixed bodies and keep other masses positive"
      )
    # Runtime passive/mass kernels consume float32 constants. Validate the
    # representation before compiling or replacing the lifecycle so finite
    # float64 values that overflow or underflow to zero cannot be admitted as
    # a usable native model.
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
      mass32 = np.asarray(values, dtype=np.float32)
    if (not np.all(np.isfinite(mass32))
        or np.any((base_mass > 0) & (mass32 <= 0))):
      raise ValueError("masses must remain positive and finite in float32")
    if np.array_equal(self.descriptor.body_mass[ids], values):
      return False

    # Changes stay private until recomputation and lowering both succeed.
    candidate = _clone_model(self._model)
    candidate.body_mass[ids] = values
    mujoco.mj_setConst(candidate, mujoco.MjData(candidate))
    descriptor = load_model(candidate)
    self._model = candidate
    self.descriptor = descriptor
    self.generation += 1
    return True


@dataclass(frozen=True)
class BatchSnapshot:
  """Immutable qpos checkpoint for a kinematics batch."""

  nq: int
  batch_size: int
  model_fingerprint: str
  qpos: np.ndarray
  schema_version: int = 1

  def __post_init__(self):
    value = np.asarray(self.qpos)
    if value.dtype.kind not in "fiu":
      raise ValueError("snapshot qpos must be numeric")
    frozen = np.frombuffer(
        np.ascontiguousarray(value).tobytes(), dtype=value.dtype
    ).reshape(value.shape)
    object.__setattr__(self, "qpos", frozen)


class KinematicsBatchState:
  """Batched qpos with per-row generation and FK cache invalidation."""

  def __init__(self, model: ModelDescriptor, batch_size: int):
    if (
        isinstance(batch_size, (bool, np.bool_))
        or not isinstance(batch_size, (int, np.integer))
        or batch_size <= 0
    ):
      raise ValueError("batch_size must be a positive integer")
    self.model = snapshot_descriptor(model)
    self._fingerprint = _fingerprint(self.model)
    self._qpos = np.broadcast_to(
        self.model.qpos0, (batch_size, self.model.nq)
    ).copy()
    self._row_generation = np.zeros(batch_size, dtype=np.int64)
    self._cache = [None] * batch_size
    self._cache_generation = np.full(batch_size, -1, dtype=np.int64)

  @property
  def qpos(self):
    """Read-only detached state view."""
    result = self._qpos.copy()
    result.setflags(write=False)
    return result

  @property
  def row_generation(self):
    """Read-only detached per-row cache generations."""
    result = self._row_generation.copy()
    result.setflags(write=False)
    return result

  def _ids(self, env_ids):
    raw = np.asarray(env_ids)
    if raw.dtype.kind not in "iu":
      raise ValueError("env_ids must contain integers")
    ids = raw.astype(np.int64, copy=False)
    if ids.ndim != 1 or np.unique(ids).size != ids.size:
      raise ValueError("env_ids must be a vector of unique environment indices")
    if np.any(ids < 0) or np.any(ids >= self._qpos.shape[0]):
      raise ValueError("env_ids are out of range")
    return ids

  def set_qpos(self, env_ids, values):
    """Update selected rows atomically; unchanged rows keep cached poses."""
    ids = self._ids(env_ids)
    values = np.asarray(values, dtype=np.float64)
    if values.shape != (ids.size, self.model.nq) or not np.all(
        np.isfinite(values)
    ):
      raise ValueError("qpos values must be finite and match selected rows")
    changed = []
    for index, env_id in enumerate(ids):
      if np.array_equal(values[index], self._qpos[env_id]):
        continue
      self.model.forward_kinematics(values[index])
      changed.append((int(env_id), values[index].copy()))
    for env_id, value in changed:
      self._qpos[env_id] = value
      self._row_generation[env_id] += 1
      self._cache[env_id] = None
      self._cache_generation[env_id] = -1
    return tuple(env_id for env_id, _ in changed)

  def poses(self, env_id):
    """Return cached CPU FK poses for one row, copying arrays for ownership."""
    ids = self._ids([env_id])
    index = int(ids[0])
    if self._cache_generation[index] != self._row_generation[index]:
      self._cache[index] = self.model.forward_kinematics(self._qpos[index])
      self._cache_generation[index] = self._row_generation[index]
    return {name: value.copy() for name, value in self._cache[index].items()}

  def snapshot(self):
    """Capture a detached immutable checkpoint."""
    raw = self.qpos.copy()
    frozen = np.frombuffer(raw.tobytes(), dtype=raw.dtype).reshape(raw.shape)
    return BatchSnapshot(
        self.model.nq, self._qpos.shape[0], self._fingerprint, frozen
    )

  def restore(self, snapshot):
    """Restore all rows after validating them; invalidate every cached pose."""
    if (
        snapshot.schema_version != 1
        or snapshot.nq != self.model.nq
        or snapshot.batch_size != self._qpos.shape[0]
        or snapshot.model_fingerprint != self._fingerprint
    ):
      raise ValueError(
          "snapshot schema, dimensions, or model fingerprint do not match"
      )
    values = np.asarray(snapshot.qpos, dtype=np.float64)
    if values.shape != self._qpos.shape:
      raise ValueError("snapshot qpos shape is invalid")
    for value in values:
      self.model.forward_kinematics(value)
    self._qpos[:] = values
    self._row_generation += 1
    self._cache = [None] * self._qpos.shape[0]
    self._cache_generation[:] = -1

  def randomize(self, env_ids, seed, scale=0.1):
    """Randomize valid joint coordinates in selected rows deterministically."""
    if not np.isfinite(scale) or scale < 0:
      raise ValueError("scale must be finite and nonnegative")
    ids = self._ids(env_ids)
    rng = np.random.default_rng(seed)
    values = np.broadcast_to(self.model.qpos0, (ids.size, self.model.nq)).copy()
    for joint, typ in enumerate(self.model.jnt_type):
      qa = int(self.model.jnt_qposadr[joint])
      typ = int(typ)
      if typ == int(mujoco.mjtJoint.mjJNT_FREE):
        values[:, qa : qa + 3] += rng.normal(0, scale, (ids.size, 3))
        values[:, qa + 3 : qa + 7] = _perturb_quaternions(
            values[:, qa + 3 : qa + 7], rng.normal(size=(ids.size, 3)) * scale
        )
      elif typ == int(mujoco.mjtJoint.mjJNT_BALL):
        values[:, qa : qa + 4] = _perturb_quaternions(
            values[:, qa : qa + 4], rng.normal(size=(ids.size, 3)) * scale
        )
      else:
        values[:, qa] += rng.normal(0, scale, ids.size)
    return self.set_qpos(ids, values)


@dataclass(frozen=True)
class ConstantsSnapshot:
  """Immutable batched model-parameter checkpoint."""

  batch_size: int
  nbody: int
  model_fingerprint: str
  body_mass: np.ndarray
  schema_version: int = 1

  def __post_init__(self):
    value = np.asarray(self.body_mass)
    if value.dtype.kind not in "fiu":
      raise ValueError("snapshot body_mass must be numeric")
    frozen = np.frombuffer(
        np.ascontiguousarray(value).tobytes(), dtype=value.dtype
    ).reshape(value.shape)
    object.__setattr__(self, "body_mass", frozen)


class BatchedConstants:
  """Per-environment model constants with transactional CPU recomputation."""

  def __init__(self, source, batch_size):
    if mujoco.__version__ != "3.10.0":
      raise RuntimeError(f"requires MuJoCo 3.10.0; found {mujoco.__version__}")
    if (
        isinstance(batch_size, (bool, np.bool_))
        or not isinstance(batch_size, (int, np.integer))
        or batch_size <= 0
    ):
      raise ValueError("batch_size must be a positive integer")
    self._base_model = _compile_source(source)
    if self._base_model.nmocap:
      raise ValueError(
          "mocap inputs are unsupported by the current kinematics stage"
      )
    self.descriptor = load_model(self._base_model)
    self._fingerprint = _fingerprint(self._base_model)
    self._batch_size = int(batch_size)
    self._body_mass = np.broadcast_to(
        self.descriptor.body_mass, (self._batch_size, self.descriptor.nbody)
    ).copy()
    self._body_invweight0 = np.empty(
        (self._batch_size,) + self._base_model.body_invweight0.shape
    )
    self._row_generation = np.zeros(self._batch_size, dtype=np.int64)
    self.generation = 0
    for env_id in range(self._batch_size):
      self._body_invweight0[env_id] = self._derive(self._body_mass[env_id])

  @property
  def batch_size(self):
    return self._batch_size

  @property
  def body_mass(self):
    result = self._body_mass.copy()
    result.setflags(write=False)
    return result

  @property
  def body_invweight0(self):
    result = self._body_invweight0.copy()
    result.setflags(write=False)
    return result

  @property
  def row_generation(self):
    result = self._row_generation.copy()
    result.setflags(write=False)
    return result

  def _env_ids(self, env_ids):
    raw = np.asarray(env_ids)
    if raw.dtype.kind not in "iu":
      raise ValueError("env_ids must contain integers")
    ids = raw.astype(np.int64, copy=False)
    if ids.ndim != 1 or ids.size == 0 or np.unique(ids).size != ids.size:
      raise ValueError("env_ids must be a nonempty vector of unique indices")
    if np.any(ids < 0) or np.any(ids >= self._batch_size):
      raise ValueError("env_ids are out of range")
    return ids

  def _derive(self, masses):
    candidate = _clone_model(self._base_model)
    candidate.body_mass[:] = masses
    mujoco.mj_setConst(candidate, mujoco.MjData(candidate))
    result = np.array(candidate.body_invweight0, copy=True)
    if not np.all(np.isfinite(result)):
      raise ValueError("mj_setConst produced nonfinite body_invweight0")
    return result

  def _commit(self, env_ids, candidates, force=False):
    changed = [
        int(env_id)
        for env_id in env_ids
        if force
        or not np.array_equal(candidates[env_id], self._body_mass[env_id])
    ]
    if not changed:
      return ()
    recomputed = {}
    # Compute every selected derived row before mutating published state.
    for env_id in changed:
      recomputed[env_id] = self._derive(candidates[env_id])
    for env_id in changed:
      self._body_mass[env_id] = candidates[env_id]
      self._body_invweight0[env_id] = recomputed[env_id]
      self._row_generation[env_id] += 1
    self.generation += 1
    return tuple(changed)

  def recompute_constants(self, parameters, env_ids):
    """Recompute rows from complete per-world `body_mass` parameter vectors."""
    ids = self._env_ids(env_ids)
    if not isinstance(parameters, dict) or set(parameters) != {"body_mass"}:
      raise ValueError("parameters must contain only a body_mass array")
    values = np.asarray(parameters["body_mass"], dtype=np.float64)
    shape = (ids.size, self.descriptor.nbody)
    if values.shape != shape or not np.all(np.isfinite(values)):
      raise ValueError(f"body_mass must be finite with shape {shape}")
    if not np.allclose(
        values[:, 0], self._base_model.body_mass[0], rtol=0, atol=0
    ):
      raise ValueError("world body mass is immutable")
    base_mass = self._base_model.body_mass[None, 1:]
    if np.any((base_mass > 0) & (values[:, 1:] <= 0)) or np.any(
        (base_mass == 0) & (values[:, 1:] != 0)
    ):
      raise ValueError("mass parameters must preserve massless fixed bodies")
    candidate = self._body_mass.copy()
    candidate[ids] = values
    return self._commit(ids, candidate)

  def set_body_masses(self, env_ids, body_ids, masses):
    """Update selected bodies in selected worlds, preserving other rows."""
    ids = self._env_ids(env_ids)
    raw_body_ids = np.asarray(body_ids)
    if raw_body_ids.dtype.kind not in "iu":
      raise ValueError("body_ids must contain integers")
    body_ids = raw_body_ids.astype(np.int64, copy=False)
    if body_ids.ndim != 1 or body_ids.size == 0:
      raise ValueError("body_ids must be a nonempty vector")
    if np.unique(body_ids).size != body_ids.size:
      raise ValueError("body_ids must be unique")
    if np.any(body_ids <= 0) or np.any(body_ids >= self.descriptor.nbody):
      raise ValueError("body_ids must refer to non-world bodies")
    values = np.asarray(masses, dtype=np.float64)
    if values.shape != (ids.size, body_ids.size) or not np.all(
        np.isfinite(values)
    ):
      raise ValueError(
          "masses must be finite and match selected worlds and bodies"
      )
    base_mass = self._base_model.body_mass[body_ids]
    if np.any((base_mass[None, :] > 0) & (values <= 0)) or np.any(
        (base_mass[None, :] == 0) & (values != 0)
    ):
      raise ValueError("mass values must preserve massless fixed bodies")
    candidate = self._body_mass.copy()
    candidate[np.ix_(ids, body_ids)] = values
    return self._commit(ids, candidate)

  def snapshot(self):
    raw = self._body_mass.copy()
    frozen = np.frombuffer(raw.tobytes(), dtype=raw.dtype).reshape(raw.shape)
    return ConstantsSnapshot(
        self._batch_size, self.descriptor.nbody, self._fingerprint, frozen, 1
    )

  def restore(self, snapshot):
    if (
        snapshot.schema_version != 1
        or snapshot.batch_size != self._batch_size
        or snapshot.nbody != self.descriptor.nbody
        or snapshot.model_fingerprint != self._fingerprint
    ):
      raise ValueError("snapshot schema or model fingerprint does not match")
    values = np.asarray(snapshot.body_mass, dtype=np.float64)
    if values.shape != self._body_mass.shape or not np.all(np.isfinite(values)):
      raise ValueError("snapshot body_mass is invalid")
    if not np.allclose(
        values[:, 0], self._base_model.body_mass[0], rtol=0, atol=0
    ):
      raise ValueError("snapshot changes world body mass")
    base_mass = self._base_model.body_mass[None, 1:]
    if np.any((base_mass > 0) & (values[:, 1:] <= 0)) or np.any(
        (base_mass == 0) & (values[:, 1:] != 0)
    ):
      raise ValueError("snapshot changes massless fixed bodies")
    # Force recomputation even for equal parameters so derived caches are fresh.
    candidates = values.copy()
    self._commit(np.arange(self._batch_size), candidates, force=True)

  def randomize(self, env_ids, seed, scale=0.1):
    """Apply seeded log-normal mass factors; scale zero is a true no-op."""
    if not np.isfinite(scale) or scale < 0:
      raise ValueError("scale must be finite and nonnegative")
    ids = self._env_ids(env_ids)
    body_ids = np.flatnonzero(self._base_model.body_mass[1:] > 0) + 1
    if not body_ids.size or scale == 0:
      return ()
    rng = np.random.default_rng(seed)
    values = self._body_mass[np.ix_(ids, body_ids)] * np.exp(
        rng.normal(0, scale, (ids.size, body_ids.size))
    )
    return self.set_body_masses(ids, body_ids, values)
