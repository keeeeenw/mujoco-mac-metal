# Copyright 2026 The MuJoCo Metal contributors
# Licensed under the Apache License, Version 2.0.
"""Bounded witness for the original 50-degree initial Newton-H failure.

The source oracle is pinned MuJoCo 3.10.0 at 28009f9105cd92784b7b0b30c0605a5e29107a77:
``src/engine/engine_solver.c`` (``PrimalUpdateGradient`` and
``mj_solPrimal``). The 50-degree scene is copied from
``test_solver_completion_014.py::test_noslip_stick_slip_gpu[50-False]``.
This observer diagnoses the failed initial H solve; it does not certify a
physics correction or weaken the original completion gate.
"""

import os
from fractions import Fraction
from pathlib import Path

import numpy as np
import pytest


_HFAIL_SITES = {
    111: "input RHS is nonfinite",
    112: "input RHS contains a subnormal word",
    113: "RHS component scale is nonfinite or subnormal",
    114: "RHS normalization loses a word or creates a subnormal",
    115: "normalized represented RHS is nonfinite or subnormal",
    116: "normalized RHS norm is nonpositive or nonfinite",
    201: "initial paired mass preconditioner failed",
    202: "initial preconditioned pair is nonfinite or subnormal",
    203: "initial precondition scale is invalid",
    204: "initial normalized search seed is invalid",
    301: "initial PCG rho is nonpositive or nonfinite",
    351: "outer or relative residual limit is nonfinite",
    352: "normalized component residual limit is invalid",
    353: "pair-roundoff or underflow bound is nonfinite",
    401: "Hessian denominator is nonpositive or nonfinite",
    402: "Newton-PCG alpha is nonfinite",
    501: "recursive residual norm is nonfinite",
    502: "true residual norm is nonfinite",
    503: "true residual failed the certificate at work limit",
    205: "restarted paired mass preconditioner failed",
    206: "restarted normalized preconditioned pair is invalid",
    302: "restarted PCG rho is nonpositive or nonfinite",
    207: "ordinary recursive paired preconditioner failed",
    208: "ordinary normalized preconditioned pair is invalid",
    303: "next PCG rho is nonpositive or nonfinite",
    504: "bounded PCG work exhausted without a certificate",
}
_HFAIL_CODES = set(_HFAIL_SITES)
_RESCALE_CODE_BASES = {
    60000: "strict-certificate rescale",
    62000: "roundoff-bound rescale",
}


def _rescale_code(code, nv):
  """Decode the opt-in rescale site, DOF and reason from its exact-float code."""
  code = int(code)
  for base, site in _RESCALE_CODE_BASES.items():
    offset = code - base
    if 1 <= offset <= 2 * nv:
      return site, (offset - 1) // 2, (offset - 1) % 2 + 1
  return None


def _fraction32(value):
  return Fraction.from_float(float(np.float32(value)))


def _round_fraction_to_f32_bits(value):
  """Exact binary32 round-to-nearest/ties-even, including subnormals."""
  value = Fraction(value)
  sign = 0x80000000 if value < 0 else 0
  value = abs(value)
  if value == 0:
    return sign

  numerator, denominator = value.numerator, value.denominator
  exponent = numerator.bit_length() - denominator.bit_length()
  if (exponent >= 0 and numerator < (denominator << exponent)) or (
      exponent < 0 and (numerator << -exponent) < denominator):
    exponent -= 1

  def rounded_integer(num, den):
    whole, remainder = divmod(num, den)
    twice = 2 * remainder
    if twice > den or (twice == den and whole & 1):
      whole += 1
    return whole

  if exponent >= -126:
    shift = 23 - exponent
    if shift >= 0:
      significand = rounded_integer(numerator << shift, denominator)
    else:
      significand = rounded_integer(numerator, denominator << -shift)
    if significand >= (1 << 24):
      significand >>= 1
      exponent += 1
    if exponent > 127:
      return sign | 0x7f800000
    return sign | ((exponent + 127) << 23) | (significand - (1 << 23))

  subnormal = rounded_integer(numerator << 149, denominator)
  if subnormal == 0:
    return sign
  if subnormal >= (1 << 23):
    return sign | 0x00800000
  return sign | subnormal


def _f32_word(bits):
  return np.asarray([bits], dtype=np.uint32).view(np.float32)[0]


def test_rescale_cpu_oracle_rounds_exact_zero_and_subnormal_ties_to_even():
  quantum = Fraction(1, 1 << 149)
  cases = (
      (Fraction(0), 0x00000000),
      (quantum / 2, 0x00000000),
      (quantum * Fraction(3, 2), 0x00000002),
      (quantum * Fraction(5, 2), 0x00000002),
      (-quantum / 2, 0x80000000),
      (quantum * ((1 << 23) - 1), 0x007fffff),
      (quantum * (1 << 23), 0x00800000),
  )
  for exact, expected in cases:
    assert _round_fraction_to_f32_bits(exact) == expected
  assert float(_f32_word(_round_fraction_to_f32_bits(quantum * 3))) == 3 * 2.0**-149


def _shader():
  return (Path(__file__).parents[1] / "mujoco_metal" / "shaders"
          / "coupled_constraints.metal").read_text()


def test_initial_hsolve_observer_is_bounded_and_source_gated():
  shader = _shader()
  primal = shader[shader.index("inline int solve_primal_accel_scalar"):]
  context_init = primal.index("initial_grad_norm_sq")
  initial_hsolve = primal.index("if (solver_type == 2) {")
  assert context_init < initial_hsolve
  assert "residual_out[4] = cold_start_cost;" in primal[:initial_hsolve]
  assert "residual_out[5] = warm_cost_delta;" in primal[:initial_hsolve]
  assert "residual_out[6] = scale * sqrt(max(0.0f, initial_grad_norm_sq));" in primal[:initial_hsolve]
  trace = shader[shader.index("inline void primal_hessian_trace_failure"):
                 shader.index("inline bool primal_hessian_solve_elliptic_legacy")]
  assert "if (trace_out == nullptr) return;" in trace
  assert "if (trace_out[9] != 0.0f) return;" in trace
  assert "isfinite(value) ? value : 0.0f" in trace
  assert "isfinite(limit) ? limit : 0.0f" in trace
  rescale = shader[shader.index("inline void primal_rescale_trace_failure"):
                   shader.index("inline float primal_pair_value")]
  assert "trace_out[9] = float(code_base + 2 * dof + subcase);" in rescale
  assert "inline bool primal_rescale_pair_vector_traced" in rescale
  assert "trace_out, 60000" in shader
  assert "trace_out, 62000" in shader
  assert "trace_primal_detail && island == 0 ? residual_out : nullptr" in shader
  assert "residual_out[7] = residual_out[8] = residual_out[9] = 0.0f;" in shader
  # The initial call alone receives the trace sink. The later per-iteration
  # H solve passes nullptr so it cannot be confused with this initial failure.
  assert "trace_primal_detail && island == 0 ? residual_out : nullptr" in shader
  assert "dof_island, island, partitioned, nv, n, tolerance, scale,\n          nullptr))" in shader
  observer = Path(__file__).read_text()
  assert 'current["dispatches"].append({' in observer
  assert 'sim.step(1)' in observer
  assert "bad_index = next((i for i, call in enumerate(dispatches)" in observer
  assert "run_steps(False, step_count)" in observer
  assert '"constraint_force", "warmstart"' in observer
  # Codes remain exact float32 integers; rescale codes additionally encode
  # their active DOF and numeric failure subcase.
  assert len(_HFAIL_CODES) == len(set(_HFAIL_CODES))
  assert all(np.float32(code).is_integer() for code in _HFAIL_CODES)
  assert {code // 100 for code in _HFAIL_CODES} >= {1, 2, 3, 4, 5}
  assert _rescale_code(60001, 6) == (_RESCALE_CODE_BASES[60000], 0, 1)
  assert _rescale_code(62012, 6) == (_RESCALE_CODE_BASES[62000], 5, 2)


@pytest.mark.gpu
@pytest.mark.skipif(os.getenv("MUJOCO_METAL_RUN_GPU") != "1",
                    reason="opt-in serialized native MPS diagnostic")
def test_original_50_degree_initial_newton_hsolve_failure_observer_gpu():
  import mujoco
  from mujoco_metal.simulation import MetalSimulation

  theta = float(np.deg2rad(50))
  c, s = float(np.cos(theta)), float(np.sin(theta))
  xml = (f'<mujoco><option timestep="0.002" integrator="Euler" '
         f'iterations="100" tolerance="1e-8" gravity="0 0 -9.81"/>'
         f'<worldbody><geom name="slope" type="plane" size="5 5 0.1" '
         f'quat="0 {s / 2:.6f} 0 {c / 2:.6f}" friction="0.8 0.05 0.02"/>'
         '<body pos="0 0 0.3"><freejoint/>'
         '<geom name="box" type="box" size="0.06 0.06 0.04"/>'
         '</body></worldbody></mujoco>')
  model = mujoco.MjModel.from_xml_string(xml)
  np.testing.assert_array_equal(
      model.geom_friction[model.geom("slope").id], [.8, .05, .02])
  np.testing.assert_array_equal(
      model.geom_friction[model.geom("box").id], [1., .005, .0001])
  qpos = np.asarray(model.qpos0, dtype=np.float32).reshape(1, -1)
  qvel = np.zeros((1, model.nv), dtype=np.float32)

  def run_steps(trace_enabled, step_limit):
    sim = MetalSimulation(model, batch_size=1, profile="integrated_euler_v1")
    sim.reset(qpos=qpos, qvel=qvel)
    cc = sim._coupled_constraints
    cc.set_primal_detail_trace(trace_enabled)
    original_run_device = cc.run_device
    current = {"dispatches": []}

    def capture_dispatches(*args, **kwargs):
      result = original_run_device(*args, **kwargs)
      mask = kwargs.get("world_mask")
      selected = (mask is None
                  or bool(mask.detach().bool().any().item()))
      if selected:
        # Each result borrows reusable workspace. Copy every selected return
        # immediately, before another primary or masked recovery can overwrite it.
        current["dispatches"].append({
            "status": result["status"].detach().cpu().numpy().copy(),
            "history": np.concatenate((
                result["solver_diagnostics"].detach().cpu().numpy().copy(),
                result["solver_history"].detach().cpu().numpy().copy()), axis=1),
            "qacc": result["qacc"].detach().cpu().numpy().copy(),
            "constraint_force": result["qfrc_constraint"].detach().cpu().numpy().copy(),
            "warmstart": cc.get_warmstart(),
            # Copy caller-owned solver scratch before another selected call
            # can reuse it. The default/off path takes no additional copy.
            "debug": (cc._workspace["workspace_debug"].detach().cpu()
                      .numpy().copy() if trace_enabled else None),
        })
      return result

    cc.run_device = capture_dispatches
    steps = []
    first_failure = None
    for step_index in range(1, step_limit + 1):
      before = {
          name: getattr(sim.state, name).detach().cpu().numpy().copy()
          for name in ("qpos", "qvel", "qacc", "status")}
      before["qacc_warmstart"] = (
          sim.state._qacc_warmstart.detach().cpu().numpy().copy())
      current["dispatches"] = []
      sim.step(1)
      dispatches = current["dispatches"]
      after = {
          name: getattr(sim.state, name).detach().cpu().numpy().copy()
          for name in ("qpos", "qvel", "qacc", "status")}
      after["qacc_warmstart"] = (
          sim.state._qacc_warmstart.detach().cpu().numpy().copy())
      steps.append({"before": before, "after": after,
                    "dispatches": dispatches})
      # Select the first nonzero status from this step's selected dispatches.
      # Calls whose world mask is entirely false never enter this list.
      bad_index = next((i for i, call in enumerate(dispatches)
                        if np.any(call["status"] != 0)), None)
      if bad_index is not None:
        first_failure = (step_index, bad_index, dispatches[bad_index])
        break
    return sim, steps, first_failure

  traced, traced_steps, failure = run_steps(True, 300)
  step_count = len(traced_steps)
  assert step_count > 0
  if failure is not None:
    failed_step, failed_index, failed_call = failure
    fail_status = int(failed_call["status"][0])
    fail_history = failed_call["history"][0]
    fail_code_value = float(fail_history[9])
    assert fail_status == 2, (failed_step, failed_index, fail_status)
    assert np.isfinite(fail_history[7:10]).all()
    assert fail_code_value.is_integer()
    fail_code = int(fail_code_value)
    rescale_failure = _rescale_code(fail_code, model.nv)
    assert fail_code in _HFAIL_CODES or rescale_failure is not None, (
        failed_step, failed_index, fail_history[7:10].tolist())
    if rescale_failure is not None:
      rescale_site, failed_dof, rescale_subcase = rescale_failure
      assert 0 <= failed_dof < model.nv
      assert rescale_subcase in (1, 2)
      assert failed_call["debug"] is not None
      debug = failed_call["debug"].reshape(1, -1)[0]
      cc = traced._coupled_constraints
      debug_prefix = int(cc._debug_prefix)
      primal_offset = debug_prefix + int(cc._primal_workspace_offset)
      pair_offset = debug_prefix + int(cc._high_low_workspace_offset)
      nv = int(model.nv)
      # At a rescale return, the failing DOF has not yet been assigned the
      # rejected physical-unit output, so these are the normalized input pair.
      normalized_input = np.asarray((
          debug[primal_offset + 4 * nv + failed_dof],
          debug[pair_offset + 4 * nv + failed_dof]), dtype=np.float32)
      assert np.isfinite(normalized_input).all()
      assert np.any(normalized_input.view(np.uint32) & 0x7fffffff)
      rhs_hi = debug[primal_offset + nv:primal_offset + 2 * nv]
      rhs_lo = debug[pair_offset + nv:pair_offset + 2 * nv]
      assert np.isfinite(rhs_hi).all() and np.isfinite(rhs_lo).all()
      physical_scale = float(max(np.max(np.abs(rhs_hi)),
                                 np.max(np.abs(rhs_lo))))
      assert physical_scale > 0.0 and np.isfinite(physical_scale)
      # Cross-check that this is the original step-14 primary H RHS, rather
      # than scratch retained from a later inner dispatch in run_device.
      np.testing.assert_allclose(physical_scale, 7.573064690121713e-29,
                                 rtol=2e-6, atol=0.0)
      device_output = np.asarray(fail_history[7:9], dtype=np.float32)
      exact_product = (_fraction32(normalized_input[0])
                       + _fraction32(normalized_input[1])) * _fraction32(
                           physical_scale)
      rounded_exact_bits = _round_fraction_to_f32_bits(exact_product)
      output_bits = device_output.copy().view(np.uint32).tolist()
      if rescale_subcase == 2:
        assert output_bits[0] & 0x7fffffff == 0
        assert output_bits[1] & 0x7fffffff == 0
      print("PRIMAL_INITIAL_HSOLVE_RESCALE_WITNESS", {
          "step": failed_step, "dispatch": failed_index,
          "site": rescale_site, "dof": failed_dof,
          "subcase": ("nonfinite-output" if rescale_subcase == 1
                      else "nonzero-input-rounded-to-two-zeros"),
          "normalized_input_hi_lo": normalized_input.tolist(),
          "physical_rhs_scale": physical_scale,
          "device_output_hi_lo": device_output.tolist(),
          "device_output_bits": output_bits,
          "exact_pair_product_rne_bits": rounded_exact_bits,
          "exact_pair_product_rne": float(_f32_word(rounded_exact_bits)),
      }, flush=True)
    pre_state = traced_steps[-1]["before"]
    post_state = traced_steps[-1]["after"]
    print("PRIMAL_INITIAL_HSOLVE_FAILURE_STATE", "before=", pre_state,
          "after=", post_state, flush=True)
    tail_meaning = ("rescale failure: [rejected_output_hi,rejected_output_lo,"
                    "site+dof+subcase code]" if rescale_failure is not None
                    else "[initial_H_decisive_value,initial_H_limit,return_site_code]")
  else:
    # On a future successful correction, all 300 original steps must be
    # healthy. Accepted-point history [7:10] is not interpreted as a code.
    assert step_count == 300
    assert all(int(step["after"]["status"][0]) == 0
               for step in traced_steps)
    failed_step = failed_index = None
    fail_history = np.zeros(10, dtype=np.float32)
    tail_meaning = "success through 300 steps; accepted history is not a code"

  print("PRIMAL_INITIAL_HSOLVE_50_FALSE", "first_failure_step=", failed_step,
        "dispatch_index=", failed_index,
        "status=", (None if failure is None else
                    int(failure[2]["status"][0])),
        "[scale,tolerance,cold_cost,warm_cost_delta,initial_scaled_gradient]=",
        fail_history[2:7].tolist(), tail_meaning,
        fail_history[7:10].tolist(), "code_sites=", _HFAIL_SITES,
        "dispatches_per_step=", [len(step["dispatches"])
                                  for step in traced_steps], flush=True)
  print("PRIMAL_INITIAL_HSOLVE_ALL_SELECTED_DISPATCHES",
        [{"step": i,
          "returns": [{"status": call["status"].tolist(),
                       "history_2_9": call["history"][0, 2:10].tolist()}
                      for call in step["dispatches"]]}
         for i, step in enumerate(traced_steps, start=1)], flush=True)

  # Repeat exactly N individually advanced steps with tracing disabled. Compare
  # every selected dispatch and the full retained state at every step; only the
  # explicitly requested diagnostic history words may differ.
  control, control_steps, control_failure = run_steps(False, step_count)
  assert len(control_steps) == step_count
  assert (failure is None) == (control_failure is None)
  if failure is not None:
    assert control_failure is not None
    assert control_failure[:2] == failure[:2]
  for step_index, (trace_step, plain_step) in enumerate(
      zip(traced_steps, control_steps), start=1):
    for field in ("qpos", "qvel", "qacc", "status", "qacc_warmstart"):
      np.testing.assert_array_equal(
          trace_step["before"][field], plain_step["before"][field],
          err_msg=f"detail-trace/pre-{field}/step-{step_index}")
      np.testing.assert_array_equal(
          trace_step["after"][field], plain_step["after"][field],
          err_msg=f"detail-trace/post-{field}/step-{step_index}")
    assert len(trace_step["dispatches"]) == len(plain_step["dispatches"])
    for dispatch_index, (trace_call, plain_call) in enumerate(zip(
        trace_step["dispatches"], plain_step["dispatches"])):
      for field in ("status", "qacc", "constraint_force", "warmstart"):
        np.testing.assert_array_equal(
            trace_call[field], plain_call[field],
            err_msg=(f"detail-trace/{field}/step-{step_index}/"
                     f"dispatch-{dispatch_index}"))
