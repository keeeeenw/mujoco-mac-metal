# Copyright 2026 The MuJoCo Metal contributors
# Licensed under the Apache License, Version 2.0.
"""Native four-stage RK4 orchestration using MuJoCo's tangent-space update.

Mirrors MuJoCo 3.10 engine_forward.c mj_RungeKutta: every stage integrates
from the original quaternion, not from an intermediate orientation. All
numeric operations remain on MPS. The Python callback dispatches device
force/solve kernels; it must return borrowed acceleration and status tensors.
"""

from mujoco_metal.integration import MetalEulerIntegration
import inspect


class MetalRungeKutta:
  """Bounded RK4 with optional activation-state integration; borrowed outputs, atomic per-world commit."""

  def __init__(self, model, batch_size, timestep):
    self._position = MetalEulerIntegration(model, batch_size, timestep)
    self._torch = self._position._torch
    self.dt = self._position.dt
    self._zero = self._torch.zeros((batch_size, model.nv), device="mps")
    # Callbacks may reuse/mask the caller's velocity scratch while producing
    # later stages. Keep the canonical X0 velocity in owned storage so every
    # RK rate is formed from the same immutable pre-step value.
    self._initial_qvel = self._torch.empty(
        (batch_size, model.nv), dtype=self._torch.float32, device="mps")
    # ModelDescriptor snapshots omit actuator counts; raw models carry na.
    self._na = int(getattr(model, "na", 0))

  def run_device(self, qpos, qvel, act, time, status, acceleration, *,
                 initial_stage=None, defer_integration=False):
    """Four RK4 stages with pinned stage structure (R06/D3).

    ``act`` is ``(batch, na)`` or None when the model has no activation
    state. ``acceleration(q, v, act)`` returns ``(a, act_dot, status)``
    (``act_dot`` None without activation). Returns ``(qpos, qvel, acc,
    act_dot_weighted, time, status)``: the weighted act_dot feeds the
    standard activation advance at the caller, mirroring pinned
    mj_RungeKutta (stages integrate act; mj_advance applies
    mj_nextActivation to the weighted derivative).
    """
    torch = self._torch
    if (not isinstance(qvel, torch.Tensor)
        or tuple(qvel.shape) != tuple(qpos.shape[:1]) + (self._initial_qvel.shape[1],)
        or qvel.dtype != torch.float32
        or qvel.device.type != self._initial_qvel.device.type
        or (self._initial_qvel.device.index is not None
            and qvel.device.index != self._initial_qvel.device.index)
        or not qvel.is_contiguous()):
      raise ValueError("qvel must be contiguous float32 [batch,nv]")
    self._initial_qvel.copy_(qvel)
    qvel = self._initial_qvel
    # Select the supported callback arity once. Catching TypeError from an
    # executed callback would repeat side effects and hide its actual error.
    signature = inspect.signature(acceleration)
    try:
      signature.bind(qpos, qvel, act, time)
      timed_callback = True
    except TypeError:
      signature.bind(qpos, qvel, act)
      timed_callback = False
    status = status.clone()
    na = self._na
    if na > 0:
      if act is None or tuple(act.shape) != (qpos.shape[0], na):
        raise ValueError(f"act must have shape ({qpos.shape[0]}, {na})")
      act = act.clone()
    velocities, accelerations, act_dots = [], [], []
    final_context = None
    q, v, cur_act = qpos, qvel, act
    stage_times = (time, time + 0.5 * self.dt, time + 0.5 * self.dt, time + self.dt)
    for stage in range(4):
      stage_time = stage_times[stage]
      if stage == 0 and initial_stage is not None:
        result = initial_stage
      elif timed_callback:
        result = acceleration(q, v, cur_act, stage_time)
      else:
        result = acceleration(q, v, cur_act)
      if not isinstance(result, (tuple, list)) or len(result) not in (3, 4):
        raise ValueError(
            "acceleration callback must return (acc, act_dot, status) "
            "and may append a fourth stage context")
      a, adot, solve_status = result[:3]
      if stage == 3 and len(result) == 4:
        final_context = result[3]
      status = torch.where(status == 0, solve_status, status)
      velocities.append(v.clone())
      accelerations.append(a.clone())
      if na > 0:
        if adot is None or tuple(adot.shape) != (qpos.shape[0], na):
          raise ValueError("acceleration callback must return act_dot with activation state")
        act_dots.append(adot.clone())
      else:
        act_dots.append(None)
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
        # The empty-DOF finite predicate is vacuously true. Avoid an empty
        # MPS reduction, whose result need not implement that identity.
        if v.numel():
          status = torch.where(
              (status == 0) & ~torch.isfinite(v).all(dim=1),
              torch.full_like(status, 12),
              status,
          )
        if na > 0:
          cur_act = act + (self.dt * fraction) * adot
          status = torch.where(
              (status == 0) & ~torch.isfinite(cur_act).all(dim=1),
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
    if next_v.numel():
      status = torch.where(
          (status == 0) & ~torch.isfinite(next_v).all(dim=1),
          torch.full_like(status, 12),
          status,
      )
    if na > 0:
      weighted_dot = (
          act_dots[0]
          + 2 * act_dots[1]
          + 2 * act_dots[2]
          + act_dots[3]
      ) / 6
      status = torch.where(
          (status == 0) & ~torch.isfinite(weighted_dot).all(dim=1),
          torch.full_like(status, 12),
          status,
      )
    else:
      weighted_dot = None
    if defer_integration:
      # RK4 sleep advancement happens between this result and the final state
      # update. MuJoCo calls mj_sleep on the original qvel and weighted qacc,
      # then integrates weighted position rate/qacc through the new awake
      # lists. The caller owns that ordered transition.
      next_time = time + self.dt
      status = torch.where(
          (status == 0) & ~torch.isfinite(next_time),
          torch.full_like(status, 12), status)
      return {
          "position_velocity": rate,
          "weighted_acceleration": weighted_acc,
          "next_velocity": next_v,
          "final_acceleration": accelerations[3],
          "act_dot": weighted_dot,
          "time": next_time,
          "status": status,
          "final_context": final_context,
      }
    q, _, next_t, status = self._position.run_device(
        qpos, rate, self._zero, time, status
    )
    success = status == 0
    # MuJoCo leaves qacc from the final forward evaluation, not RK's weighted sum.
    return (
        torch.where(success[:, None], q, qpos),
        torch.where(success[:, None], next_v, qvel),
        accelerations[3],
        weighted_dot,
        torch.where(success, next_t, time),
        status,
    )
