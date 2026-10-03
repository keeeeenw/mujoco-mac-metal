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

"""Prepared native MPS dense SPD, signed symmetric and general solves."""

from pathlib import Path

_SHADER = Path(__file__).parent / "shaders" / "smooth_solve.metal"


class MetalDenseSolve:
  """Reusable dense SPD factorization and solve stage on the MPS device.

  Construct with fixed ``nv``, ``batch_size`` and ``nrhs`` capacities. For one
  right-hand side, ``rhs`` and the solution have shape ``[B,nv]``. For multiple
  right-hand sides they have shape ``[B,nv,nrhs]``. ``run_device`` returns
  ``(solution, status)`` where status is an int32 MPS vector, zero on success.
  Status codes are 1 for nonfinite input, 2 for asymmetric mass, 3 for a
  non-positive Cholesky pivot, and 4 for nonfinite intermediate arithmetic.
  Failed rows return a zero solution; rows are independent. Output and factor
  storage are reused, so returned views are valid only until the next call.

  This primitive validates but never modifies its mass input. It does not add
  diagonal jitter, perform a host solve, or copy values back to the CPU.
  Construction explicitly initializes MPS and compiles its shader.
  """

  def __init__(self, nv: int, batch_size: int, nrhs: int = 1):
    for name, value in (("nv", nv), ("batch_size", batch_size), ("nrhs", nrhs)):
      if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    if nv < 0:
      raise ValueError("nv must be nonnegative")
    if batch_size <= 0:
      raise ValueError("batch_size must be positive")
    if nrhs <= 0:
      raise ValueError("nrhs must be positive")
    max_index = (1 << 32) - 1
    if any(value > (1 << 31) - 1 for value in (nv, batch_size, nrhs)):
      raise ValueError("dimensions must fit the shader's signed 32-bit ABI")
    if nv * nv > max_index or batch_size * nv * nv > max_index:
      raise ValueError("mass storage dimensions exceed shader indexing range")
    if batch_size * nv * nrhs > max_index:
      raise ValueError("RHS storage dimensions exceed shader indexing range")

    import torch

    if not torch.backends.mps.is_available():
      raise RuntimeError("PyTorch MPS is unavailable")
    if not hasattr(torch.mps, "compile_shader"):
      raise RuntimeError("PyTorch does not provide torch.mps.compile_shader")
    self._torch = torch
    self._device = torch.device("mps")
    self.nv = nv
    self.batch_size = batch_size
    self.nrhs = nrhs
    self._library = torch.mps.compile_shader(_SHADER.read_text())
    self._kernel = self._library.dense_spd_solve
    self._awake_kernel = self._library.dense_awake_solve
    self._awake_work = torch.empty(
        max(batch_size * nv * nrhs, 1), dtype=torch.float32, device=self._device
    )
    self._awake_dims = torch.tensor(
        [nv, batch_size, nrhs], dtype=torch.int32, device=self._device)
    self._awake_flags = torch.zeros((2,), dtype=torch.int32, device=self._device)
    self._factor = torch.empty(
        max(batch_size * nv * nv, 1), dtype=torch.float32, device=self._device
    )
    self._solution = torch.empty(
        max(batch_size * nv * nrhs, 1), dtype=torch.float32, device=self._device
    )
    self._status = torch.empty(
        batch_size, dtype=torch.int32, device=self._device
    )
    self._empty_input = torch.zeros(1, dtype=torch.float32, device=self._device)
    self._dims = torch.tensor(
        [nv, batch_size, nrhs], dtype=torch.int32, device=self._device
    )

  def _validate_tensor(self, tensor, name, shape):
    torch = self._torch
    if not isinstance(tensor, torch.Tensor):
      raise TypeError(f"{name} must be a torch.Tensor")
    if tensor.dtype != torch.float32:
      raise TypeError(f"{name} must have dtype torch.float32")
    if tuple(tensor.shape) != shape:
      raise ValueError(
          f"{name} must have shape {shape}, got {tuple(tensor.shape)}"
      )
    if tensor.device.type != "mps":
      raise ValueError(f"{name} must be on {self._device}")
    if not tensor.is_contiguous():
      raise ValueError(f"{name} must be contiguous")

  def run_device(self, mass, rhs, *, awake_lists=None, retained=None):
    """Solve ``mass @ solution = rhs`` for all fixed-capacity batch rows.

    ``mass`` is ``[B,nv,nv]``. ``rhs`` is ``[B,nv]`` when ``nrhs==1`` and
    ``[B,nv,nrhs]`` otherwise. Input values are checked in the shader so
    failures are reported per row without a synchronization or host readback.
    """
    self._validate_tensor(mass, "mass", (self.batch_size, self.nv, self.nv))
    rhs_shape = (
        (self.batch_size, self.nv)
        if self.nrhs == 1
        else (self.batch_size, self.nv, self.nrhs)
    )
    self._validate_tensor(rhs, "rhs", rhs_shape)

    if awake_lists is not None:
      if not isinstance(awake_lists, dict):
        raise TypeError("awake_lists must be a mapping")
      ids = awake_lists.get("dof_ids")
      counts = awake_lists.get("counts")
      if (not isinstance(ids, self._torch.Tensor)
          or tuple(ids.shape) != (self.batch_size, max(self.nv, 1))
          or ids.dtype != self._torch.int32 or ids.device.type != "mps"
          or not ids.is_contiguous()):
        raise ValueError("awake_lists.dof_ids must be contiguous MPS int32")
      if (not isinstance(counts, self._torch.Tensor)
          or tuple(counts.shape) != (self.batch_size, 3)
          or counts.dtype != self._torch.int32 or counts.device.type != "mps"
          or not counts.is_contiguous()):
        raise ValueError("awake_lists.counts must be contiguous MPS int32 [batch,3]")
      if retained is not None:
        if self.nrhs != 1:
          raise ValueError("retained output is supported only for one RHS")
        self._validate_tensor(retained, "retained", (self.batch_size, self.nv))
      flags = self._awake_flags
      flags[1].fill_(int(retained is not None))
      self._awake_kernel(
          mass.reshape(-1) if self.nv else self._empty_input,
          rhs.reshape(-1) if self.nv else self._empty_input, ids, counts,
          retained.reshape(-1) if retained is not None and self.nv else self._empty_input,
          self._factor, self._awake_work, self._solution, self._status,
          self._awake_dims, flags,
          threads=(self.batch_size,), group_size=(1,))
      shape = ((self.batch_size, self.nv) if self.nrhs == 1
               else (self.batch_size, self.nv, self.nrhs))
      count = self.batch_size * self.nv * self.nrhs
      return self._solution[:count].reshape(shape), self._status

    mass_buffer = mass.reshape(-1)
    rhs_buffer = rhs.reshape(-1)
    if self.nv == 0:
      mass_buffer = self._empty_input
      rhs_buffer = self._empty_input
    self._kernel(
        mass_buffer,
        rhs_buffer,
        self._factor,
        self._solution,
        self._status,
        self._dims,
        threads=(self.batch_size,),
        group_size=(1,),
    )
    if self.nrhs == 1:
      solution = self._solution[: self.batch_size * self.nv].reshape(
          self.batch_size, self.nv
      )
    else:
      solution = self._solution[
          : self.batch_size * self.nv * self.nrhs
      ].reshape(self.batch_size, self.nv, self.nrhs)
    return solution, self._status


class MetalGeneralDenseSolve(MetalDenseSolve):
  """Reusable dense nonsymmetric LU factorization and solve stage on MPS."""

  def __init__(self, nv: int, batch_size: int, nrhs: int = 1):
    super().__init__(nv, batch_size, nrhs)
    self._kernel = self._library.dense_general_solve
    self._awake_flags[0] = 1


def symmetric_ldl_workspace_elements(nv, batch_size, nrhs=1):
  """Validate all signed Euler solve backings before importing Torch."""
  counts = factored_workspace_elements(nv, batch_size, nrhs)
  counts.update({"default_dof_ids": int(batch_size) * max(int(nv), 1),
                 "default_counts": int(batch_size) * 3})
  if max(counts.values()) > (1 << 32) - 1:
    raise ValueError("signed solve workspace exceeds shader indexing range")
  return counts


class MetalSymmetricLDLSolve(MetalDenseSolve):
  """Solve signed Euler matrices with MuJoCo's descending ``L' D L`` order.

  Finite negative pivots are valid; a zero pivot is status 3. This is the
  Euler effective-matrix path, not the physical SPD mass solver. It uses the
  same prepared awake principal-system gather/scatter for full and sleeping
  worlds, supports multiple RHS columns and performs no host numerical solve.
  """

  def __init__(self, nv: int, batch_size: int, nrhs: int = 1):
    symmetric_ldl_workspace_elements(nv, batch_size, nrhs)
    super().__init__(nv, batch_size, nrhs)
    torch = self._torch
    self._awake_flags[0] = 2
    self._default_dof_ids = torch.arange(
        max(nv, 1), dtype=torch.int32, device=self._device
    ).expand(batch_size, -1).contiguous()
    self._default_counts = torch.tensor(
        [0, 0, nv], dtype=torch.int32, device=self._device
    ).expand(batch_size, -1).contiguous()
    self._default_awake = {"dof_ids": self._default_dof_ids,
                           "counts": self._default_counts}

  def run_device(self, mass, rhs, *, awake_lists=None, retained=None):
    return super().run_device(
        mass, rhs, awake_lists=(self._default_awake if awake_lists is None
                               else awake_lists), retained=retained)


def factored_workspace_elements(nv, batch_size, nrhs=1):
  """Validate compiled dimensions before importing Torch or allocating MPS."""
  import numbers
  for name, value in (("nv", nv), ("batch_size", batch_size), ("nrhs", nrhs)):
    if isinstance(value, bool) or not isinstance(value, numbers.Integral):
      raise TypeError(f"{name} must be an integer")
    if value < (0 if name == "nv" else 1):
      raise ValueError(f"{name} is outside its valid range")
    if value > (1 << 31) - 1:
      raise ValueError("dimensions must fit signed 32-bit shader arguments")
  nv, batch_size, nrhs = int(nv), int(batch_size), int(nrhs)
  counts = {"factor": batch_size*nv*nv, "pivots": batch_size*nv,
            "solution": batch_size*nv*nrhs, "status": batch_size}
  if max(counts.values()) > (1 << 32) - 1:
    raise ValueError("factor workspace dimensions exceed shader indexing range")
  return counts


class MetalFactorizedSolve(MetalDenseSolve):
  """Factor a matrix once and apply it to subsequent right-hand sides.

  ``general=False`` uses Cholesky; ``general=True`` uses LU with recorded row
  pivots. Matrix and RHS validation retain the ordinary dense solver status
  codes. A bad RHS cannot invalidate a previously valid factor. An invalid
  factor affects only its own world and is replaced by the next factor call.
  Factors and outputs stay in reusable device buffers. No inverse is formed.
  """
  def __init__(self, nv, batch_size, nrhs=1, *, general=False):
    counts = factored_workspace_elements(nv, batch_size, nrhs)
    if not isinstance(general, bool):
      raise TypeError("general must be a bool")
    import torch
    if not torch.backends.mps.is_available():
      raise RuntimeError("PyTorch MPS is unavailable")
    if not hasattr(torch.mps, "compile_shader"):
      raise RuntimeError("PyTorch does not provide torch.mps.compile_shader")
    self._torch, self._device = torch, torch.device("mps")
    self.nv, self.batch_size, self.nrhs = int(nv), int(batch_size), int(nrhs)
    self._library = torch.mps.compile_shader(
        _SHADER.with_name("smooth_factor_solve.metal").read_text())
    self._factor = torch.empty(max(counts["factor"],1), dtype=torch.float32, device=self._device)
    self._pivots = torch.empty(max(counts["pivots"],1), dtype=torch.int32, device=self._device)
    self._solution = torch.empty(max(counts["solution"],1), dtype=torch.float32, device=self._device)
    self._factor_status = torch.empty(self.batch_size, dtype=torch.int32, device=self._device)
    self._status = torch.empty_like(self._factor_status)
    self._empty_input = torch.zeros(1, dtype=torch.float32, device=self._device)
    self._dims = torch.tensor([self.nv,self.batch_size,self.nrhs,int(general)],
                              dtype=torch.int32, device=self._device)
    self._factor_ready = False

  def factor_device(self, matrix):
    """Replace the stored factor, returning its borrowed per-world status."""
    self._validate_tensor(matrix,"matrix",(self.batch_size,self.nv,self.nv))
    value = matrix.reshape(-1) if self.nv else self._empty_input
    self._library.dense_factor(value,self._factor,self._pivots,
        self._factor_status,self._dims,threads=(self.batch_size,),group_size=(1,))
    self._factor_ready = True
    return self._factor_status

  def solve_factored_device(self, rhs):
    """Solve with the last submitted factor, without reading input values."""
    if not self._factor_ready:
      raise RuntimeError("factor_device must be called before solving")
    shape = ((self.batch_size,self.nv) if self.nrhs==1
             else (self.batch_size,self.nv,self.nrhs))
    self._validate_tensor(rhs,"rhs",shape)
    value = rhs.reshape(-1) if self.nv else self._empty_input
    self._library.dense_solve_factored(self._factor,self._pivots,
        self._factor_status,value,self._solution,self._status,self._dims,
        threads=(self.batch_size,),group_size=(1,))
    count = self.batch_size*self.nv*self.nrhs
    return self._solution[:count].reshape(shape),self._status

  def run_device(self, matrix, rhs):
    # Reject bad metadata before replacing a previously valid factor.
    shape = ((self.batch_size,self.nv) if self.nrhs==1
             else (self.batch_size,self.nv,self.nrhs))
    self._validate_tensor(matrix,"matrix",(self.batch_size,self.nv,self.nv))
    self._validate_tensor(rhs,"rhs",shape)
    self.factor_device(matrix)
    return self.solve_factored_device(rhs)
