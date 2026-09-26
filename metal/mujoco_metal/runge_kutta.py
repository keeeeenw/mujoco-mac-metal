# Copyright 2026 The MuJoCo Metal contributors
# Licensed under the Apache License, Version 2.0.
"""Native four-stage RK4 orchestration using MuJoCo's tangent-space update.

Mirrors MuJoCo 3.10 engine_forward.c mj_RungeKutta: every stage integrates
from the original quaternion, not from an intermediate orientation. All
numeric operations remain on MPS. The Python callback dispatches device
force/solve kernels; it must return borrowed acceleration and status tensors.
"""

from mujoco_metal.integration import MetalEulerIntegration


class MetalRungeKutta:
  """Bounded stateless-force RK4; borrowed outputs, atomic per-world commit."""

  def __init__(self, model, batch_size, timestep):
    self._position = MetalEulerIntegration(model, batch_size, timestep)
    self._torch = self._position._torch
    self.dt = self._position.dt
    self._zero = self._torch.zeros((batch_size, model.nv), device="mps")

  def run_device(self, qpos, qvel, time, status, acceleration):
    torch = self._torch
    status = status.clone()
    velocities, accelerations = [], []
    q, v = qpos, qvel
    for stage in range(4):
      a, solve_status = acceleration(q, v)
      status = torch.where(status == 0, solve_status, status)
      velocities.append(v.clone())
      accelerations.append(a.clone())
      if stage < 3:
        fraction = (0.5, 0.5, 1.0)[stage]
        q, _, _, position_status = self._position.run_device(
            qpos,
            v * fraction,
            self._zero,
            time,
            status,
        )
        q = q.clone()
        status = torch.where(status == 0, position_status, status)
        v = qvel + (self.dt * fraction) * a
        status = torch.where(
            (status == 0) & ~torch.isfinite(v).all(dim=1),
            torch.full_like(status, 12),
            status,
        )
    rate = (
        velocities[0] + 2 * velocities[1] + 2 * velocities[2] + velocities[3]
    ) / 6
    weighted_acc = (
        accelerations[0]
        + 2 * accelerations[1]
        + 2 * accelerations[2]
        + accelerations[3]
    ) / 6
    next_v = qvel + self.dt * weighted_acc
    status = torch.where(
        (status == 0) & ~torch.isfinite(next_v).all(dim=1),
        torch.full_like(status, 12),
        status,
    )
    q, _, next_t, status = self._position.run_device(
        qpos, rate, self._zero, time, status
    )
    success = status == 0
    # MuJoCo leaves qacc from the final forward evaluation, not RK's weighted sum.
    return (
        torch.where(success[:, None], q, qpos),
        torch.where(success[:, None], next_v, qvel),
        accelerations[3],
        torch.where(success, next_t, time),
        status,
    )
