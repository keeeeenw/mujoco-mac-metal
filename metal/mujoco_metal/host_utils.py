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

"""Milestone 019: explicitly labeled host-only pinned utilities.

These are thin validated wrappers around pinned CPU APIs (inverse
dynamics, mass/Jacobian/object-velocity queries). They run on the HOST,
never on the device, and must never be cited as native GPU evidence: a
host compatibility result is recorded under host compatibility. Native
counterparts and their independent qualification are documented in API.md;
the existence of these host wrappers neither proves nor rules out native support.
"""

import mujoco
import numpy as np


def _model(model):
  if not isinstance(model, mujoco.MjModel):
    raise TypeError("model must be a compiled mujoco.MjModel")
  if mujoco.__version__ != "3.10.0":
    raise RuntimeError(f"host utilities require MuJoCo 3.10.0; found {mujoco.__version__}")
  return model


def _state(model, qpos, qvel):
  qpos = np.asarray(qpos, dtype=np.float64)
  qvel = np.asarray(qvel, dtype=np.float64)
  if qpos.shape != (model.nq,) or not np.all(np.isfinite(qpos)):
    raise ValueError(f"qpos must be finite with shape ({model.nq},)")
  if qvel.shape != (model.nv,) or not np.all(np.isfinite(qvel)):
    raise ValueError(f"qvel must be finite with shape ({model.nv},)")
  return qpos, qvel


def inverse_dynamics(model, qpos, qvel, qacc):
  """HOST: generalized forces for the given acceleration (pinned mj_inverse)."""
  model = _model(model)
  qpos, qvel = _state(model, qpos, qvel)
  qacc = np.asarray(qacc, dtype=np.float64)
  if qacc.shape != (model.nv,) or not np.all(np.isfinite(qacc)):
    raise ValueError(f"qacc must be finite with shape ({model.nv},)")
  data = mujoco.MjData(model)
  data.qpos[:] = qpos
  data.qvel[:] = qvel
  data.qacc[:] = qacc
  mujoco.mj_inverse(model, data)
  return np.asarray(data.qfrc_inverse).copy()


def mass_matrix_host(model, qpos):
  """HOST: dense joint-space inertia via pinned mj_fullM."""
  model = _model(model)
  qpos = np.asarray(qpos, dtype=np.float64)
  if qpos.shape != (model.nq,) or not np.all(np.isfinite(qpos)):
    raise ValueError(f"qpos must be finite with shape ({model.nq},)")
  data = mujoco.MjData(model)
  data.qpos[:] = qpos
  mujoco.mj_forward(model, data)
  out = np.zeros((model.nv, model.nv))
  mujoco.mj_fullM(model, data, out)
  return out


def body_jacobian_host(model, qpos, body_id):
  """HOST: world-frame translational/rotational Jacobian of a body."""
  model = _model(model)
  qpos = np.asarray(qpos, dtype=np.float64)
  if qpos.shape != (model.nq,) or not np.all(np.isfinite(qpos)):
    raise ValueError(f"qpos must be finite with shape ({model.nq},)")
  if not 0 <= int(body_id) < model.nbody:
    raise ValueError(f"body id {body_id} out of range")
  data = mujoco.MjData(model)
  data.qpos[:] = qpos
  mujoco.mj_forward(model, data)
  jacp = np.zeros((3, model.nv))
  jacr = np.zeros((3, model.nv))
  mujoco.mj_jacBody(model, data, jacp, jacr, int(body_id))
  return jacp, jacr


def object_velocity_host(model, qpos, qvel, objtype, objid):
  """HOST: 6D world velocity of a body/geom/site (pinned mj_objectVelocity)."""
  model = _model(model)
  qpos, qvel = _state(model, qpos, qvel)
  data = mujoco.MjData(model)
  data.qpos[:] = qpos
  data.qvel[:] = qvel
  mujoco.mj_forward(model, data)
  out = np.zeros(6)
  mujoco.mj_objectVelocity(model, data, int(objtype), int(objid), out, 1)
  return out


def forward_inverse_consistency(model, qpos, qvel, qacc):
  """HOST: round-trip check ``forward(inverse(qacc)) == qacc`` residuals.

  Returns ``(qfrc_inverse, qacc_roundtrip, residual)``. The forward leg runs
  pinned Euler-consistent dynamics with contacts disabled by the caller if
  desired; constraint rows are out of scope for this host check.
  """
  model = _model(model)
  forces = inverse_dynamics(model, qpos, qvel, qacc)
  data = mujoco.MjData(model)
  data.qpos[:] = np.asarray(qpos, dtype=np.float64)
  data.qvel[:] = np.asarray(qvel, dtype=np.float64)
  data.qfrc_applied[:] = forces
  mujoco.mj_forward(model, data)
  roundtrip = np.asarray(data.qacc).copy()
  return forces, roundtrip, roundtrip - np.asarray(qacc, dtype=np.float64)
