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

"""Native MPS semi-implicit Euler position, velocity, and time update."""

from pathlib import Path
import math
import numbers
import struct

import numpy as np

from mujoco_metal.model import ModelDescriptor
from mujoco_metal.model import snapshot_descriptor

_SHADER = Path(__file__).parent / "shaders" / "integration.metal"
_JOINT_FREE = 0
_JOINT_BALL = 1
_JOINT_SLIDE = 2
_JOINT_HINGE = 3


def integration_workspace_sizes(nq,nv,njnt,nbody,batch):
  """Actual owned tensor backings; borrowed output views add no storage."""
  return {
      'joint_type':max(njnt,1),'joint_qposadr':max(njnt,1),'joint_dofadr':max(njnt,1),
      'body_jntadr':max(nbody,1),'body_jntnum':max(nbody,1),'dims':5,'timestep':3,
      'velocity_modes':4,'sleep_filter':1,'dummy_int':1,'empty_input':1,
      'candidate_qpos':max(batch*nq,1),'candidate_qvel':max(batch*nv,1),'candidate_time':batch,
      'output_qpos':max(batch*nq,1),'output_qvel':max(batch*nv,1),'output_time':batch,'output_status':batch,
      'zero_words':batch*(2*nq+2*nv),'candidate_words':batch*(2*nq+2*nv),
      'output_words':batch*(2*nq+2*nv),'zero_acc_low':batch*nv,'zero_override_words':batch*4*nv,
  }


class MetalEulerIntegration:
  """Reusable native MPS semi-implicit Euler stage for a fixed model/batch.

  ``run_device(qpos, qvel, qacc, time, solve_status)`` consumes contiguous
  float32 MPS states shaped ``[B,nq]``, ``[B,nv]``, ``[B,nv]``, ``[B]`` and an
  int32 status vector. It returns borrowed ``(qpos, qvel, time, status)`` MPS
  views backed by reusable output storage; each returned view is overwritten
  by the next call. A nonzero solve status is propagated unchanged and that
  row is left untouched. Integration failures use 10 for nonfinite input, 11
  for a zero quaternion, and 12 for nonfinite arithmetic. Failed rows preserve
  their input state. Rows commit independently after the full candidate state
  passes validation.

  ``precision_words`` optionally supplies packed per-world position low/tail
  and velocity low/tail planes. ``qacc_low`` retains the accepted solver
  correction. ``override_words`` supplies low/tail planes for explicit next
  and position velocities. The corresponding borrowed output is
  ``_output_words``; a persistent state owner must copy it on a successful
  commit before reusing this stage. Failure rows retain all input words.

  The velocity update is ``qvel += dt*qacc``. Position integration follows
  MuJoCo 3.10 for hinge, slide, ball and free joints, using the updated
  velocity, or an explicit ``position_velocity`` for midpoint integration.
  An optional ``next_velocity`` supplies the candidate velocity directly. An
  optional ``awake_lists`` applies MuJoCo's pre-integration sleep ownership:
  only listed DOFs and joints advance, while asleep qpos is retained and asleep
  qvel is zero. Both velocity overrides receive the same finite-value and
  per-row commit checks.
  Construction validates and snapshots model constants, initializes
  MPS, and compiles the shader. It does not read state values back to the host.
  """

  def __init__(self, model: ModelDescriptor, batch_size: int, timestep: float):
    if not isinstance(model, ModelDescriptor):
      raise TypeError("model must be a ModelDescriptor")
    if isinstance(batch_size, bool) or not isinstance(batch_size, int):
      raise TypeError("batch_size must be an integer")
    if batch_size <= 0:
      raise ValueError("batch_size must be positive")
    if isinstance(timestep, bool) or not isinstance(timestep, numbers.Real):
      raise TypeError("timestep must be a real number")
    dt = float(timestep)
    source_dt = dt
    if not math.isfinite(dt) or dt <= 0:
      raise ValueError("timestep must be finite and positive")
    try:
      dt = struct.unpack("f", struct.pack("f", dt))[0]
    except (OverflowError, struct.error) as error:
      raise ValueError(
          "timestep must be representable as finite float32"
      ) from error
    if not math.isfinite(dt) or dt <= 0:
      raise ValueError(
          "timestep must be representable as positive finite float32"
      )

    from mujoco_metal.model import require_triple_word_domain
    require_triple_word_domain([source_dt],what='timestep')
    self.model = snapshot_descriptor(model)
    if np.any(
        ~np.isin(
            self.model.jnt_type,
            (_JOINT_FREE, _JOINT_BALL, _JOINT_SLIDE, _JOINT_HINGE),
        )
    ):
      raise ValueError(
          "integration supports only free, ball, slide, and hinge joints"
      )
    self.batch_size = batch_size
    self.dt = source_dt
    self.nq = self.model.nq
    self.nv = self.model.nv
    self.njnt = self.model.njnt
    self.nbody = self.model.nbody
    max_index = (1 << 32) - 1
    if any(
        value > (1 << 31) - 1
        for value in (self.nq, self.nv, self.njnt, self.nbody, batch_size)
    ):
      raise ValueError("dimensions must fit the shader's signed 32-bit ABI")
    if batch_size * max(self.nq, self.nv) > max_index:
      raise ValueError("state storage dimensions exceed shader indexing range")

    from mujoco_metal.capacity import _check_i32_elements
    for name,count in integration_workspace_sizes(self.nq,self.nv,self.njnt,self.nbody,batch_size).items():
      _check_i32_elements('integration.'+name,count)
    import torch

    if not torch.backends.mps.is_available():
      raise RuntimeError("PyTorch MPS is unavailable")
    if not hasattr(torch.mps, "compile_shader"):
      raise RuntimeError("PyTorch does not provide torch.mps.compile_shader")
    self._torch = torch
    self._device = torch.device("mps")
    from mujoco_metal.metal_kinematics import _SOURCE_MATH_SHADER
    source=('#define MUJOCO_METAL_SOURCE_MATH_ONLY\n'+_SOURCE_MATH_SHADER.read_text()
            +'\n#undef MUJOCO_METAL_SOURCE_MATH_ONLY\n'
            +_SHADER.with_name('position_math.metal').read_text()+_SHADER.read_text())
    self._library = torch.mps.compile_shader(source)
    self._kernel = self._library.semi_implicit_euler
    self._joint_type = self._device_array(self.model.jnt_type)
    self._joint_qposadr = self._device_array(self.model.jnt_qposadr)
    self._joint_dofadr = self._device_array(self.model.jnt_dofadr)
    self._body_jntadr = self._device_array(self.model.body_jntadr)
    self._body_jntnum = self._device_array(self.model.body_jntnum)
    self._dims = torch.tensor(
        [self.nq, self.nv, self.njnt, batch_size, self.nbody],
        dtype=torch.int32,
        device=self._device,
    )
    dt_low=np.float32(source_dt-np.float32(source_dt))
    dt_tail=np.float32(source_dt-float(np.float32(source_dt))-float(dt_low))
    self._dt = torch.tensor([dt,float(dt_low),float(dt_tail)], dtype=torch.float32, device=self._device)
    self._precision_width=2*self.nq+2*self.nv
    self._zero_words=torch.zeros((batch_size,self._precision_width),dtype=torch.float32,device=self._device)
    self._candidate_words=torch.empty_like(self._zero_words)
    self._output_words=torch.empty_like(self._zero_words)
    self._zero_override_words=torch.zeros((batch_size,4*self.nv),dtype=torch.float32,device=self._device)
    self._zero_acc_low=torch.zeros((batch_size,self.nv),dtype=torch.float32,device=self._device)
    self._velocity_modes = tuple(
        torch.tensor([mode], dtype=torch.int32, device=self._device)
        for mode in range(4)
    )
    self._sleep_filter = torch.zeros(1, dtype=torch.int32, device=self._device)
    self._dummy_int = torch.zeros(1, dtype=torch.int32, device=self._device)
    self._empty_input = torch.zeros(1, dtype=torch.float32, device=self._device)
    self._candidate_qpos = torch.empty(
        max(batch_size * self.nq, 1), dtype=torch.float32, device=self._device
    )
    self._candidate_qvel = torch.empty(
        max(batch_size * self.nv, 1), dtype=torch.float32, device=self._device
    )
    self._candidate_time = torch.empty(
        batch_size, dtype=torch.float32, device=self._device
    )
    self._output_qpos = torch.empty_like(self._candidate_qpos)
    self._output_qvel = torch.empty_like(self._candidate_qvel)
    self._output_time = torch.empty_like(self._candidate_time)
    self._output_status = torch.empty(
        batch_size, dtype=torch.int32, device=self._device
    )

  def _device_array(self, value):
    torch = self._torch
    if value.size == 0:
      return torch.zeros(1, dtype=torch.int32, device=self._device)
    return torch.from_numpy(np.array(value, dtype=np.int32, copy=True)).to(
        self._device
    )

  def _validate_tensor(self, tensor, name, shape, dtype):
    torch = self._torch
    if not isinstance(tensor, torch.Tensor):
      raise TypeError(f"{name} must be a torch.Tensor")
    if tensor.dtype != dtype:
      raise TypeError(f"{name} must have dtype {dtype}")
    if tuple(tensor.shape) != shape:
      raise ValueError(
          f"{name} must have shape {shape}, got {tuple(tensor.shape)}"
      )
    if tensor.device.type != "mps":
      raise ValueError(f"{name} must be on {self._device}")
    if not tensor.is_contiguous():
      raise ValueError(f"{name} must be contiguous")

  def run_device(
      self,
      qpos,
      qvel,
      qacc,
      time,
      solve_status,
      *,
      next_velocity=None,
      position_velocity=None,
      awake_lists=None,
      precision_words=None, qacc_low=None, override_words=None,
  ):
    """Return candidate states, optionally restricted to current awake lists.

    For a pinned sleep step, update the scheduler from pre-integration qvel
    first, zero qacc entries for trees newly put to sleep, and pass its current
    awake lists here. A row that fails validation preserves its input state.
    """
    batch = self.batch_size
    self._validate_tensor(qpos, "qpos", (batch, self.nq), self._torch.float32)
    self._validate_tensor(qvel, "qvel", (batch, self.nv), self._torch.float32)
    self._validate_tensor(qacc, "qacc", (batch, self.nv), self._torch.float32)
    self._validate_tensor(time, "time", (batch,), self._torch.float32)
    self._validate_tensor(
        solve_status, "solve_status", (batch,), self._torch.int32
    )

    if precision_words is None: precision_words=self._zero_words
    self._validate_tensor(precision_words,'precision_words',(batch,self._precision_width),self._torch.float32)
    if qacc_low is None:qacc_low=self._zero_acc_low
    self._validate_tensor(qacc_low,'qacc_low',(batch,self.nv),self._torch.float32)
    if override_words is None:override_words=self._zero_override_words
    self._validate_tensor(override_words,'override_words',(batch,4*self.nv),self._torch.float32)
    for name, value in (
        ("next_velocity", next_velocity),
        ("position_velocity", position_velocity),
    ):
      if value is not None:
        self._validate_tensor(
            value, name, (batch, self.nv), self._torch.float32
        )
    mode = int(next_velocity is not None) + 2 * int(
        position_velocity is not None
    )
    sleep_filter = awake_lists is not None
    if sleep_filter:
      if not isinstance(awake_lists, dict):
        raise TypeError("awake_lists must be a dictionary")
      required = ("body_ids", "dof_ids", "counts")
      if any(name not in awake_lists for name in required):
        raise ValueError("awake_lists must contain body_ids, dof_ids, counts")
      expected_shapes = {
          "body_ids": (batch, max(self.nbody, 1)),
          "dof_ids": (batch, max(self.nv, 1)),
          "counts": (batch, 3),
      }
      for name, shape in expected_shapes.items():
        value = awake_lists[name]
        if (not isinstance(value, self._torch.Tensor)
            or tuple(value.shape) != shape
            or value.dtype != self._torch.int32
            or value.device.type != "mps"
            or not value.is_contiguous()):
          raise ValueError(
              f"awake_lists.{name} must be contiguous MPS int32 {shape}")
      body_ids = awake_lists["body_ids"]
      dof_ids = awake_lists["dof_ids"]
      counts = awake_lists["counts"]
    else:
      body_ids = dof_ids = counts = self._dummy_int
    self._sleep_filter.fill_(int(sleep_filter))
    next_buffer = (
        next_velocity.reshape(-1)
        if next_velocity is not None and self.nv
        else self._empty_input
    )
    position_buffer = (
        position_velocity.reshape(-1)
        if position_velocity is not None and self.nv
        else self._empty_input
    )

    qpos_buffer = qpos.reshape(-1) if self.nq else self._empty_input
    qvel_buffer = qvel.reshape(-1) if self.nv else self._empty_input
    qacc_buffer = qacc.reshape(-1) if self.nv else self._empty_input
    self._kernel(
        qpos_buffer,
        qvel_buffer,
        qacc_buffer,
        time,
        solve_status,
        self._joint_type,
        self._joint_qposadr,
        self._joint_dofadr,
        self._candidate_qpos,
        self._candidate_qvel,
        self._candidate_time,
        self._output_qpos,
        self._output_qvel,
        self._output_time,
        self._output_status,
        self._dims,
        self._dt,
        next_buffer,
        position_buffer,
        self._velocity_modes[mode],
        self._body_jntadr,
        self._body_jntnum,
        body_ids.reshape(-1) if sleep_filter else self._dummy_int,
        dof_ids.reshape(-1) if sleep_filter else self._dummy_int,
        counts,
        self._sleep_filter,
        precision_words.reshape(-1) if self._precision_width else self._empty_input,
        self._candidate_words.reshape(-1) if self._precision_width else self._empty_input,
        self._output_words.reshape(-1) if self._precision_width else self._empty_input,
        qacc_low.reshape(-1) if self.nv else self._empty_input,
        override_words.reshape(-1) if self.nv else self._empty_input,
        threads=(batch,),
        group_size=(1,),
    )
    qpos_output = self._output_qpos[: batch * self.nq].reshape(batch, self.nq)
    qvel_output = self._output_qvel[: batch * self.nv].reshape(batch, self.nv)
    return qpos_output, qvel_output, self._output_time, self._output_status
