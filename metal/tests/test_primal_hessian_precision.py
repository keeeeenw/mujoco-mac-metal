"""Float32 replay for the bounded elliptic-Newton Hessian PCG."""

from pathlib import Path

import numpy as np


# Rounded native inputs from the 6-DOF condim=6 elliptic failure capture.  The
# matrix is the actual matrix-free Hessian applied to basis vectors; M and rhs
# are the corresponding captured mass operator and final inner RHS.
_MASS = np.array([
    [33.5103226, 0.0, 0.0, -0.00116972905, 0.00263211853, -0.678091824],
    [0.0, 33.5103226, 0.0, -0.00233967858, 0.00526377279, 1.50442982],
    [0.0, 0.0, 33.5103226, 0.670201242, -1.50795329, 0.00406797044],
    [-0.00116972905, -0.00233967858, 0.670201242, 0.549569249,
     -0.0301592909, -1.02931965e-8],
    [0.00263211853, 0.00526377279, -1.50795329, -0.0301592909,
     0.604023516, -4.81339280e-9],
    [-0.678091824, 1.50442982, 0.00406797044, -1.02931965e-8,
     -4.81339280e-9, 0.617427528],
], dtype=np.float32)

_HESSIAN = np.array([
    [233.07880, 8.5243092, -243.57823, 1.0391183, -26.848560, 17.701355],
    [8.5243092, 160.65108, -27.977736, 24.127806, -0.44688225, 9.4290171],
    [-243.57823, -27.977736, 424.12479, -11.346469, 26.406168, -68.419782],
    [1.0391181, 24.127806, -11.346469, 5.9403510, 0.28903195, 1.8616148],
    [-26.848562, -0.44688225, 26.406166, 0.28903177, 5.4316239, -0.13941115],
    [17.701353, 9.4290180, -68.419792, 1.8616148, -0.13941069, 77.781387],
], dtype=np.float32)

_RHS = np.array([
    -6.103515625e-5, -4.577636719e-5, 0.0, -7.62939453125e-6,
    7.62939453125e-6, -3.0517578125e-5,
], dtype=np.float32)


def _fma32(a, b, c):
  """Correctly rounded float32 FMA emulation for float32 operands."""
  return np.float32(float(a) * float(b) + float(c))


def _pair_add(a, b):
  """Two-float expansion add with the shader's error-free renormalization."""
  ahi, alo = map(np.float32, a)
  bhi, blo = map(np.float32, b)
  total = np.float32(ahi + bhi)
  virtual = np.float32(total - ahi)
  error = np.float32((ahi - (total - virtual)) + (bhi - virtual))
  error = np.float32(error + alo + blo)
  high = np.float32(total + error)
  virtual_low = np.float32(high - total)
  low = np.float32((total - (high - virtual_low)) + (error - virtual_low))
  return high, low


def _pair_fma(alpha, value, accumulator):
  """Replay the shader's two-word multiply-add using float32 hi/lo words."""
  alpha = np.float32(alpha)
  high, low = map(np.float32, value)
  product = np.float32(alpha * high)
  error = np.float32(float(alpha) * float(high) - float(product)
                     + float(alpha) * float(low))
  return _pair_add((product, error), accumulator)


def _middle_elliptic_force_hessian(residual, row_R, friction):
  """Double oracle for the pinned active middle-zone cone equations."""
  residual = np.asarray(residual, dtype=np.float64)
  row_R = np.asarray(row_R, dtype=np.float64)
  friction = np.asarray(friction, dtype=np.float64)
  mu = friction[0] * np.sqrt(row_R[1] / row_R[0])
  tangent_vec = friction[:len(residual) - 1] * residual[1:]
  tangent = np.linalg.norm(tangent_vec)
  normal = residual[0]
  assert normal < tangent and mu * normal + tangent > 0.0
  scale = (1.0 / row_R[0]) / (1.0 + mu * mu)
  gap = normal - tangent
  a = np.zeros(len(residual), dtype=np.float64)
  force = np.zeros(len(residual), dtype=np.float64)
  force[0] = -scale * gap
  for j in range(1, len(residual)):
    f = friction[j - 1]
    a[j] = f * f * residual[j]
    force[j] = scale * gap * a[j] / tangent
  hessian = np.zeros((len(residual), len(residual)), dtype=np.float64)
  hessian[0, 0] = scale
  for j in range(1, len(residual)):
    hessian[j, 0] = hessian[0, j] = -scale * a[j] / tangent
    fj = friction[j - 1]
    for k in range(1, len(residual)):
      second_tangent = ((fj * fj if j == k else 0.0) / tangent
                        - a[j] * a[k] / tangent**3)
      hessian[j, k] = (scale * a[j] * a[k] / tangent**2
                       - scale * gap * second_tangent)
  return force, hessian


def _dense_ccj_record(matrix):
  """Pack and decode a dense matrix through the production CCJ ABI."""
  from mujoco_metal.constraint_jacobian import (
      PackedJacobianLayout, initialize_packed_jacobian_records)

  matrix = np.asarray(matrix, dtype=np.float32)
  if matrix.ndim != 2:
    raise ValueError("canonical dense Jacobian must be rank two")
  nrow, nv = matrix.shape
  layout = PackedJacobianLayout.create(nrow, nv, mode=0)
  record = np.zeros(layout.stride_words, dtype=np.float32)
  initialize_packed_jacobian_records(record.view(np.int32), 1, layout)
  record[layout.values_offset:layout.values_offset + layout.nnz] = matrix.reshape(-1)
  # Check the exact float values at the same offsets the shader's ccj_get reads.
  decoded = record[layout.values_offset:layout.values_offset + layout.nnz]
  np.testing.assert_array_equal(decoded, matrix.reshape(-1))
  return record


def _compensated_product_sum(a, b):
  """Match the shader's Neumaier sum with exact product-error recovery."""
  total = np.float32(0.0)
  correction = np.float32(0.0)
  for left, right in zip(np.asarray(a).flat, np.asarray(b).flat):
    product = np.float32(left * right)
    product_error = np.float32(float(left) * float(right) - float(product))
    next_total = np.float32(total + product)
    if abs(float(total)) >= abs(float(product)):
      addition_error = np.float32(float(total) - float(next_total)
                                  + float(product))
    else:
      addition_error = np.float32(float(product) - float(next_total)
                                  + float(total))
    correction = np.float32(correction + addition_error + product_error)
    total = next_total
  return np.float32(total + correction)


def _compensated_matvec(matrix, vector):
  return np.array([_compensated_product_sum(row, vector) for row in matrix],
                  dtype=np.float32)


def _compensated_product3_add(a, b, c, total, correction):
  product = np.float32(np.float32(a) * np.float32(b))
  product_error = np.float32(float(a) * float(b) - float(product))
  value = np.float32(product * np.float32(c))
  error = np.float32(float(product) * float(c) - float(value)
                     + float(product_error) * float(c))
  next_total = np.float32(total + value)
  if abs(float(total)) >= abs(float(value)):
    addition_error = np.float32(float(total) - float(next_total) + float(value))
  else:
    addition_error = np.float32(float(value) - float(next_total) + float(total))
  correction = np.float32(correction + addition_error + error)
  return next_total, correction


def _compensated_operator_apply(mass, scalar_rows, elliptic_blocks, vector):
  """Replay the new per-DOF M + scalar-row + cone-Hessian accumulation."""
  mass = np.asarray(mass, dtype=np.float32)
  vector = np.asarray(vector, dtype=np.float32)
  output = np.zeros_like(vector)
  for dof in range(len(vector)):
    total = correction = np.float32(0.0)
    for other in range(len(vector)):
      p = np.float32(mass[dof, other] * vector[other])
      pe = np.float32(float(mass[dof, other]) * float(vector[other]) - float(p))
      next_total = np.float32(total + p)
      err = np.float32(float(total) - float(next_total) + float(p))
      correction = np.float32(correction + err + pe)
      total = next_total
    for jacobian, curvature in scalar_rows:
      jd = _compensated_product_sum(jacobian, vector)
      total, correction = _compensated_product3_add(
          curvature, jd, jacobian[dof], total, correction)
    for jacobian, local_hessian in elliptic_blocks:
      local_direction = np.asarray(
          [_compensated_product_sum(row, vector) for row in jacobian],
          dtype=np.float32)
      for row in range(len(local_direction)):
        hv = _compensated_product_sum(local_hessian[row], local_direction)
        p = np.float32(jacobian[row, dof] * hv)
        pe = np.float32(float(jacobian[row, dof]) * float(hv) - float(p))
        next_total = np.float32(total + p)
        err = np.float32(float(total) - float(next_total) + float(p))
        correction = np.float32(correction + err + pe)
        total = next_total
    output[dof] = np.float32(total + correction)
  return output


def _mass_solve(cholesky, rhs):
  """Replay the shader's triangular solve, including fused updates."""
  n = len(rhs)
  forward = np.zeros(n, dtype=np.float32)
  solution = np.zeros(n, dtype=np.float32)
  for i in range(n):
    value = rhs[i]
    for k in range(i):
      value = _fma32(-cholesky[i, k], forward[k], value)
    forward[i] = np.float32(value / cholesky[i, i])
  for i in range(n - 1, -1, -1):
    value = forward[i]
    for k in range(i + 1, n):
      value = _fma32(-cholesky[k, i], solution[k], value)
    solution[i] = np.float32(value / cholesky[i, i])
  return solution


def _pcg_replay(max_iterations, mass=_MASS, hessian=_HESSIAN, rhs=_RHS):
  """Replay bounded PCG with true residual checks at convergence and budget."""
  n = len(rhs)
  factor = np.linalg.cholesky(np.asarray(mass, dtype=np.float64)).astype(np.float32)
  rhs = np.asarray(rhs, dtype=np.float32)
  hessian = np.asarray(hessian, dtype=np.float32)
  solution = np.zeros(n, dtype=np.float32)
  residual = rhs.copy()
  rhs_norm_sq = _compensated_product_sum(residual, residual)
  preconditioned = _mass_solve(factor, residual)
  direction = preconditioned.copy()
  rho = _compensated_product_sum(residual, preconditioned)
  for iteration in range(max_iterations):
    product = _compensated_matvec(hessian, direction)
    denominator = _compensated_product_sum(direction, product)
    if not np.isfinite(denominator) or denominator <= 0:
      return {"status": "nonpositive_curvature", "iterations": iteration,
              "relative_residual": np.inf, "solution": solution}

    alpha = np.float32(rho / denominator)
    for i in range(n):
      solution[i] = _fma32(alpha, direction[i], solution[i])
      residual[i] = _fma32(-alpha, product[i], residual[i])
    recursive_sq = _compensated_product_sum(residual, residual)

    if recursive_sq <= np.float32(1e-12) * rhs_norm_sq or iteration + 1 == max_iterations:
      residual = np.asarray(rhs - _compensated_matvec(hessian, solution),
                            dtype=np.float32)
      true_sq = _compensated_product_sum(residual, residual)
      relative_residual = float(np.sqrt(true_sq / rhs_norm_sq))
      if true_sq <= np.float32(1e-12) * rhs_norm_sq:
        return {"status": "converged", "iterations": iteration + 1,
                "relative_residual": relative_residual, "solution": solution}
      if iteration + 1 == max_iterations:
        return {"status": "iteration_limit", "iterations": iteration + 1,
                "relative_residual": relative_residual, "solution": solution}
      preconditioned = _mass_solve(factor, residual)
      direction = preconditioned.copy()
      rho = _compensated_product_sum(residual, preconditioned)
      continue

    preconditioned = _mass_solve(factor, residual)
    next_rho = _compensated_product_sum(residual, preconditioned)
    beta = np.float32(next_rho / rho)
    for i in range(n):
      direction[i] = _fma32(beta, direction[i], preconditioned[i])
    rho = next_rho

  raise AssertionError("bounded PCG replay did not classify its termination")


def test_captured_elliptic_hessian_replay_recovers_true_residual_without_relaxing_it():
  """The captured SPD system converges under compensated float32 arithmetic."""
  assert np.all(np.linalg.eigvalsh((_HESSIAN + _HESSIAN.T) * 0.5) > 0)
  assert np.all(np.linalg.eigvalsh((_MASS + _MASS.T) * 0.5) > 0)
  result = _pcg_replay(4 * len(_RHS))
  assert result["status"] == "converged"
  assert result["iterations"] <= 4 * len(_RHS)
  assert result["relative_residual"] <= 1e-6
  oracle = np.linalg.solve(_HESSIAN.astype(np.float64), _RHS.astype(np.float64))
  np.testing.assert_allclose(result["solution"], oracle, rtol=2e-5, atol=5e-12)

  # A shorter fixed budget must report failure, with no stale success status
  # or tolerance widening.
  limited = _pcg_replay(len(_RHS))
  assert limited["status"] == "iteration_limit"
  assert limited["relative_residual"] > 1e-6


def test_captured_multicontact_hessian_keeps_cg_conjugacy_to_true_residual():
  """An actual failed inner solve converges if it does not restart every n."""
  # Rounded matrix, mass, and RHS from the final failed native inner attempt.
  # The values are copied into this fixture so this regression stays hermetic.
  mass = np.diag(np.array([1, 1, 1, 0.0108333332464,
                          0.0166666675359, 0.0208333339542], np.float32))
  hessian = np.array([
      [17.2558975, -6.6003108, -24.896164, -0.29069293, -1.2574031, 8.0187807],
      [-6.6003108, 10.926298, 15.343413, 1.1441346, 1.600579, -4.9203568],
      [-24.896164, 15.343413, 47.341454, 1.4959829, 2.427376, -17.761707],
      [-0.29069293, 1.1441346, 1.4959829, 1.1822605, 0.12165487, -0.12509191],
      [-1.2574031, 1.600579, 2.427376, 0.12165487, 1.9609857, -1.1762021],
      [8.0187807, -4.9203568, -17.761707, -0.12509191, -1.1762021, 8.9869699],
  ], dtype=np.float32)
  rhs = np.array([10.8788414, -3.7990246, -45.6651344, 0.06450689,
                  -0.11712134, 18.6711102], dtype=np.float32)
  result = _pcg_replay(4 * len(rhs), mass=mass, hessian=hessian, rhs=rhs)
  assert result["status"] == "converged"
  assert result["iterations"] <= 4 * len(rhs)
  assert result["relative_residual"] <= 1e-6
  oracle = np.linalg.solve(hessian.astype(np.float64), rhs.astype(np.float64))
  np.testing.assert_allclose(result["solution"], oracle, rtol=2e-5, atol=2e-6)


def test_captured_contact_hessians_converge_within_original_pcg_budget():
  """Replay the other native failures without dimension-periodic restarts."""
  systems = [
      (
          "multicontact-condim6",
          [1, 1, 1, .01083333138, .01666666567, .02083333395],
          [[17.650404, -6.875268, -25.234566, -.35877824, -1.26303, 7.7973742],
           [-6.875268, 11.230134, 15.829136, 1.192434, 1.558355, -4.906529],
           [-25.234566, 15.829136, 47.341454, 1.5889013, 2.3864665, -17.18121],
           [-.35877824, 1.192434, 1.588901, 1.2302271, .12002066, -.12065089],
           [-1.2630298, 1.5583547, 2.3864665, .12002057, 2.0001438, -1.1641568],
           [7.7973747, -4.9065294, -17.18121, -.12065107, -1.1641567, 8.7039013]],
          [14.378584, -6.84696, -50.505867, -.61260056, -.81177056, 19.852045],
      ),
      (
          "capsule-elliptic-condim6",
          [6.9701471, 6.9701471, 6.9701471, .082698107, .082698129, .020931883],
          [[61.399498, .415844, -31.813591, -.090213, -10.699964, -.02646716],
           [.41584486, 62.082867, 15.035286, 10.80108, -.09676497, .03041012],
           [-31.813591, 15.035286, 87.721825, -.95852298, 3.593766, -1.2187231],
           [-.09021246, 10.801081, -.95852292, 2.6084254, .003181141, -.10531363],
           [-10.699963, -.09676469, 3.5937665, .0031811304, 2.6013515, .05595654],
           [-.026467154, .03041012, -1.2187231, -.10531363, .05595654, 31.130625]],
          [3.8146973e-6, -1.9073486e-6, -1.5258789e-5,
           3.5762787e-7, -4.7683716e-7, -1.4305115e-6],
      ),
      (
          "box-pyramidal-condim4",
          [7.6799998, 7.6799998, 7.6799998, .053248, .041984, .06246399],
          [[185.63118, 0, -222.43896, -.1012582, -13.408074, -.00155227],
           [0, 185.63118, 222.43896, 13.377576, -.00801537, -.003105383],
           [-222.43896, 222.43896, 702.80176, 17.158562, 2.739357, -83.391403],
           [-.10125857, 13.377577, 17.158564, 11.041438, -.01752784, 3.0228333],
           [-13.408075, -.008016097, 2.739357, -.017527803, 8.009538, 6.1636486],
           [-.0015524714, -.0031053834, -83.391403, 3.0228348, 6.1636481, 54.453209]],
          [-3.3447418, -1.8776627, -18.76381, -2.2641432, -.9067378, 7.258286],
      ),
  ]
  for label, mass_diagonal, hessian, rhs in systems:
    mass = np.diag(np.asarray(mass_diagonal, dtype=np.float32))
    hessian = np.asarray(hessian, dtype=np.float32)
    rhs = np.asarray(rhs, dtype=np.float32)
    result = _pcg_replay(4 * len(rhs), mass=mass, hessian=hessian, rhs=rhs)
    assert result["status"] == "converged", label
    assert result["relative_residual"] <= 1e-6, label
    oracle = np.linalg.solve(hessian.astype(np.float64), rhs.astype(np.float64))
    np.testing.assert_allclose(result["solution"], oracle, rtol=3e-5, atol=1e-6,
                               err_msg=label)


def test_primal_hessian_shader_uses_compensated_products_and_keeps_true_tolerance():
  shader_path = (Path(__file__).parents[1] / "mujoco_metal" / "shaders"
                 / "coupled_constraints.metal")
  shader = shader_path.read_text()
  assert "inline void primal_compensated_product_add" in shader
  assert "float product_error = fma(a, b, -product);" in shader
  assert "primal_compensated_dot(direction, product, nv)" in shader
  assert "primal_compensated_dot(residual, residual, nv)" in shader
  solver = shader.split(
      "inline bool primal_hessian_solve_elliptic_scaled", 1)[1].split(
      "// Solve the mass-preconditioner system on one independent solver island.",
      1)[0]
  # Normalize the equation and the arbitrary common scale of preconditioned
  # directions without changing the requested outer criterion or work cap.
  assert "float rhs_norm = primal_pair_norm(residual, residual_low, nv);" in solver
  assert "float precondition_scale = 0.0f;" in solver
  assert "{preconditioned[dof], product_low[dof]}, {precondition_scale, 0.0f}" in solver
  assert "preconditioned[dof] = z.hi;" in solver
  assert "outer_absolute_residual / rhs_scale" in solver
  assert "float relative_residual_limit = 1e-6f * rhs_norm;" in solver
  assert "float residual_tolerance_sq" not in solver
  assert "rhs_norm_sq <= 1e-30f" not in solver
  assert "max(1, 4 * active_dof_count)" in solver
  assert "strict_component_residual" in solver
  assert "pair_underflow_floor <= component_residual_limit;" in solver
  assert "component_residual_limit = outer_scaled_limit" in solver
  assert "component_bound = component_residual_limit" in solver
  assert "inner_residual[dof];" in solver
  assert "if (!isfinite(operator_magnitude) || !isfinite(component_bound)" in solver
  assert "pair_underflow_floor" in solver
  assert "true_residual_norm <= outer_scaled_limit" in solver
  assert "primal_rescale_pair_vector_traced(solution, solution_low" in solver
  rescale = shader.split(
      "inline bool primal_rescale_pair_vector(device float* hi", 1)[1].split(
      "inline float primal_pair_value", 1)[0]
  assert "primal_rescale_pair_vector_impl(hi, lo, awake_tree, dims" in rescale
  assert "scale, nv, nullptr, 0);" in rescale
  assert "an exactly zero represented residual" in solver
  assert "device float* product_scale" in shader
  assert "absolute_scale += abs(mij)" in shader
  assert "absolute_scale += abs(row_hessian[row])" in shader
  assert "absolute_scale += abs(ccj_get(J, start + a, dof, nv)) * hv_scale;" in shader
  assert "PrimalFloatPair contribution = primal_pair_product(coefficient, jd);" in shader
  assert "complete operator result per DOF" in shader
  # Constraint rows may be dense or header-packed CSR. The Hessian operator
  # must obtain both direction words through the canonical accessor and carry
  # the paired row projection into the DOF accumulation.
  assert "primal_ccj_compensated_dot_pair(\n          J, row, vector, nv)" in shader
  assert "primal_ccj_compensated_dot_pair(\n          J, row, vector_low, nv)" in shader


def _masked_float32_pcg(matrix, rhs, active, mass):
  """Float32 replay using the independent mass matrix as PCG preconditioner."""
  f32 = np.float32
  matrix = np.asarray(matrix, dtype=np.float32)
  rhs = np.asarray(rhs, dtype=np.float32)
  mass = np.asarray(mass, dtype=np.float32)
  active = np.asarray(active, dtype=bool)
  ids = np.flatnonzero(active)
  out = np.zeros_like(rhs)
  if not len(ids):
    return out
  A = matrix[np.ix_(ids, ids)]
  M = mass[np.ix_(ids, ids)]
  b = rhs[ids]
  max_rhs = max(float(np.max(np.abs(b))), 1.0)
  scale = f32(1.0 / max_rhs)
  r = b.copy()
  z = np.linalg.solve(M.astype(np.float64), r.astype(np.float64)).astype(np.float32)
  z = (z * scale).astype(np.float32)
  d = z.copy()
  rho = f32(np.dot(r, z))
  x = np.zeros_like(b)
  for _ in range(4 * len(ids)):
    Hd = (A @ d).astype(np.float32)
    denominator = f32(np.dot(d, Hd))
    if not np.isfinite(denominator) or denominator <= 0:
      raise ArithmeticError("scaled PCG denominator is not finite positive")
    alpha = f32(rho / denominator)
    x = (x + alpha * d).astype(np.float32)
    r = (r - alpha * Hd).astype(np.float32)
    z = np.linalg.solve(M.astype(np.float64), r.astype(np.float64)).astype(np.float32)
    z = (z * scale).astype(np.float32)
    next_rho = f32(np.dot(r, z))
    if np.linalg.norm(r.astype(np.float64)) <= 1e-6:
      break
    beta = f32(next_rho / rho)
    d = (z + beta * d).astype(np.float32)
    rho = next_rho
  out[ids] = x
  return out


def test_scaled_pcg_stiff_non_diagonal_hessian_keeps_true_residual_representable():
  # A rank-one stiff constraint contribution produces a non-diagonal 2x2
  # Hessian with a large finite RHS. The mass preconditioner is M=I, independent
  # of H as in the production algorithm. Scaling only the search direction
  # keeps rho and d^T H d finite; one pair correction then verifies the true
  # operator residual rather than treating the recursive residual as proof.
  f32 = np.float32
  hessian = np.array([[f32(2.5e15), f32(1.5e15)],
                      [f32(1.5e15), f32(2.5e15)]], dtype=np.float32)
  rhs = f32(2.3684217e18)
  b = np.array([rhs, rhs], dtype=np.float32)
  # The omitted third DOF is outside this active principal island; its large
  # poisoned RHS must not participate in scaling or either dot product.
  full_rhs = np.array([rhs, rhs, f32(3.0e30)], dtype=np.float32)
  active = np.array([True, True, False])
  direction = b.copy()  # M^-1 rhs for active M=I.
  with np.errstate(over="ignore", invalid="ignore"):
    unscaled_hdirection = (hessian @ direction).astype(np.float32)
    unscaled_denominator = f32(np.dot(direction, unscaled_hdirection))
  assert np.isinf(unscaled_denominator)
  rhs_component_scale = max(abs(float(full_rhs[i])) for i in range(3) if active[i])
  scale = f32(1.0 / max(1.0, rhs_component_scale))
  scaled_direction = (direction * scale).astype(np.float32)
  rho = f32(np.dot(b, scaled_direction))
  product = (hessian @ scaled_direction).astype(np.float32)
  denominator = f32(np.dot(scaled_direction, product))
  alpha = f32(rho / denominator)
  solution_hi = (alpha * scaled_direction).astype(np.float32)
  true_residual = b.astype(np.float64) - hessian.astype(np.float64) @ solution_hi.astype(np.float64)
  solution_low = np.linalg.solve(hessian.astype(np.float64), true_residual).astype(np.float32)
  accepted_residual = b.astype(np.float64) - hessian.astype(np.float64) @ (
      solution_hi.astype(np.float64) + solution_low.astype(np.float64))
  assert np.isinf(unscaled_denominator)
  assert all(np.isfinite(x).all() if isinstance(x, np.ndarray) else np.isfinite(x)
             for x in (scaled_direction, rho, product, denominator, alpha,
                       solution_hi, solution_low))
  assert np.linalg.norm(true_residual) > 1e-6
  assert np.linalg.norm(accepted_residual) <= 1e-6


def test_scaled_pcg_preserves_active_principal_operator_and_zero_mask():
  matrix = np.array([
      [2.0e5, 0.0, 0.0, 0.0],
      [0.0, 3.0, 0.25, 0.0],
      [0.0, 0.25, 2.0, 0.0],
      [0.0, 0.0, 0.0, 7.0],
  ], dtype=np.float32)
  mass = np.array([
      [2.0, .1, 0.0, 0.0],
      [.1, 1.5, .05, 0.0],
      [0.0, .05, 1.2, 0.0],
      [0.0, 0.0, 0.0, 1.0],
  ], dtype=np.float32)
  rhs = np.array([2.5e5, 1.25, -0.5, 9.0], dtype=np.float32)
  active = np.array([True, True, True, False])
  actual = _masked_float32_pcg(matrix, rhs, active, mass)
  ids = np.flatnonzero(active)
  expected_active = np.linalg.solve(matrix[np.ix_(ids, ids)].astype(np.float64),
                                    rhs[ids].astype(np.float64))
  np.testing.assert_allclose(actual[ids], expected_active, rtol=2e-5, atol=1e-6)
  assert actual[3] == 0.0
  np.testing.assert_array_equal(
      _masked_float32_pcg(matrix, rhs, np.zeros(4, dtype=bool), mass),
      np.zeros(4, dtype=np.float32))


def test_pair_residual_bound_is_componentwise_for_heterogeneous_active_rows():
  eps2 = 1.4210854715202004e-14
  active_count = 2
  operation_count = 2 * 4 + 2 * (2 * 4 + 16) + (12 * 4 + 80)
  pair_roundoff = 32 * operation_count * eps2
  requested_outer = 1e-8
  outer_component = requested_outer / np.sqrt(active_count)
  residual = np.array([1000.0, 1e-2])
  rhs_magnitude = np.array([2e18, 1.0])
  product_terms_magnitude = np.array([2e18, 1.0])
  per_row_bound = outer_component + pair_roundoff * (
      rhs_magnitude + product_terms_magnitude)
  # A first-row residual at the two-word rounding scale is admissible, but a
  # wrong small second row is rejected even though it is invisible to a
  # single global RHS-norm backward-error floor.
  assert residual[0] <= per_row_bound[0]
  assert residual[1] > per_row_bound[1]
  assert not _componentwise_residual_accepts(
      residual, rhs_magnitude, product_terms_magnitude, outer_component,
      pair_roundoff, 32 * operation_count * 1.17549435e-38)
  # A non-finite sum of absolute operator terms disables the fallback instead
  # of turning the bound into +inf and accepting any finite residual.
  assert not _componentwise_residual_accepts(
      np.array([1.0]), np.array([1.0]), np.array([np.inf]), 1e-8,
      pair_roundoff, 32 * operation_count * 1.17549435e-38)


def _componentwise_residual_accepts(residual, rhs_magnitude,
                                    product_terms_magnitude,
                                    outer_component, pair_roundoff,
                                    underflow_floor=0.0):
  residual = np.asarray(residual, dtype=np.float64)
  rhs_magnitude = np.asarray(rhs_magnitude, dtype=np.float64)
  product_terms_magnitude = np.asarray(product_terms_magnitude, dtype=np.float64)
  operator_magnitude = rhs_magnitude + product_terms_magnitude
  component_bound = (outer_component + pair_roundoff * operator_magnitude
                     + underflow_floor)
  if not np.isfinite(operator_magnitude).all() or not np.isfinite(component_bound).all():
    return False
  # Mirror production ordering: strict requested residual first, then the
  # arithmetic-error fallback only if its underflow charge fits the requested
  # component allowance.
  if np.all(np.abs(residual) <= outer_component):
    return True
  if underflow_floor > outer_component:
    return False
  return bool(np.all(np.abs(residual) <= component_bound))


def test_pair_residual_bound_retains_strict_tiny_rhs_inner_threshold():
  rhs_norm_sq = 1e-20
  outer_absolute_residual = 1e-8
  residual_tolerance_sq = min(1e-12 * rhs_norm_sq,
                              outer_absolute_residual ** 2)
  component_limit = np.sqrt(residual_tolerance_sq)
  eps2 = 1.4210854715202004e-14
  term_count = 2 + (2 + 16)
  pair_roundoff = 32 * term_count * eps2
  true_residual = 1e-10
  operator_magnitude = 2e-10
  component_bound = component_limit + pair_roundoff * operator_magnitude
  assert component_limit < outer_absolute_residual
  assert true_residual > component_bound
  assert not _componentwise_residual_accepts(
      np.array([true_residual]), np.array([1e-10]), np.array([1e-10]),
      component_limit, pair_roundoff, 32 * term_count * 1.17549435e-38)


def test_pair_residual_does_not_accept_nonzero_tiny_rhs_after_norm_underflow():
  # Float32 squared norms for values around 1e-30 underflow. A zero squared
  # residual must not certify a nonzero component whose pair-error floor is
  # larger than the requested component allowance.
  residual = np.array([1e-30], dtype=np.float64)
  assert np.float32(residual[0] * residual[0]) == 0.0
  assert not _componentwise_residual_accepts(
      residual, np.array([1e-30]), np.array([1e-30]), 0.0, 1e-12, 1e-35)
  assert _componentwise_residual_accepts(
      np.array([0.0]), np.array([1e-30]), np.array([1e-30]), 0.0, 1e-12,
      1e-35)


def test_composed_mass_scalar_and_elliptic_hessian_apply_matches_double_operator():
  """Each matrix family contributes before the single compensated row write."""
  mass = np.array([[7.0, -1.75, 0.0], [-1.75, 2.5, .25], [0.0, .25, 1.3]],
                  dtype=np.float32)
  scalar_rows = [
      (np.array([1.0, -2.0, .5], dtype=np.float32), np.float32(.7)),
      (np.array([-.3, .8, 1.1], dtype=np.float32), np.float32(1.25)),
  ]
  elliptic_j = np.array([[.4, -1.2, .7], [1.1, .2, -.5]], dtype=np.float32)
  elliptic_h = np.array([[2.0, -.35], [-.35, 1.4]], dtype=np.float32)
  vector = np.array([.125, -.375, .625], dtype=np.float32)
  result = _compensated_operator_apply(
      mass, scalar_rows, [(elliptic_j, elliptic_h)], vector)
  exact = mass.astype(np.float64) @ vector.astype(np.float64)
  for jacobian, curvature in scalar_rows:
    jacobian64 = jacobian.astype(np.float64)
    exact += float(curvature) * jacobian64 * (jacobian64 @ vector.astype(np.float64))
  jacobian64 = elliptic_j.astype(np.float64)
  exact += jacobian64.T @ elliptic_h.astype(np.float64) @ jacobian64 @ vector.astype(np.float64)
  np.testing.assert_allclose(result, exact, rtol=2e-6, atol=2e-7)


def test_pair_state_preserves_sub_ulp_acceleration_and_row_residual():
  """A line update below qacc's float32 ulp survives through J*qacc-aref."""
  q0 = np.float32(1.0)
  qacc_hi, qacc_lo = _pair_fma(
      np.float32(0.5), (np.float32(-1e-8), np.float32(0.0)),
      (q0, np.float32(0.0)))
  assert qacc_hi == q0
  assert qacc_lo < 0.0

  jacobian = np.float32(1.0)
  aref = np.float32(1.0)
  row_hi, row_lo = _pair_add(
      _pair_add((np.float32(jacobian * qacc_hi), np.float32(0.0)),
                (np.float32(jacobian * qacc_lo), np.float32(0.0))),
      (-aref, np.float32(0.0)))
  row_value = float(row_hi) + float(row_lo)
  assert row_value < 0.0
  np.testing.assert_allclose(row_value, -5e-9, rtol=0.0, atol=2e-16)

  force_hi, force_lo = _pair_fma(-1.0, (row_hi, row_lo), (0.0, 0.0))
  mass_delta_hi, mass_delta_lo = qacc_hi - q0, qacc_lo
  gradient_hi, gradient_lo = _pair_add(
      (mass_delta_hi, mass_delta_lo), (-force_hi, -force_lo))
  assert abs(float(gradient_hi) + float(gradient_lo)) > 0.0


def test_refreshed_row_residual_subtracts_both_aref_words():
  """A low aref word is subtracted, not added, in J*qacc-aref."""
  # Values are from the mixed-island CG fixture's first accepted update.  The
  # equality reference is large enough that its low word materially changes
  # the tiny residual used to refresh the preconditioned gradient.
  aref_hi = np.float32(-405.26318)
  aref_lo = np.float32(7.6999804e-6)
  jacobian = np.float32(1.0)
  acc_hi = np.float32(-373.61697)
  acc_lo = np.float32(-4.6753325e-6)

  expected = (float(jacobian) * (float(acc_hi) + float(acc_lo))
              - (float(aref_hi) + float(aref_lo)))
  correct = _pair_add((-aref_hi, -aref_lo), (jacobian * acc_hi,
                                               jacobian * acc_lo))
  incorrect = _pair_add((-aref_hi, aref_lo), (jacobian * acc_hi,
                                              jacobian * acc_lo))
  np.testing.assert_allclose(float(correct[0]) + float(correct[1]),
                             expected, rtol=0.0, atol=2e-5)
  assert abs((float(incorrect[0]) + float(incorrect[1])) - expected) \
      > 1.5 * float(aref_lo)

  shader = (Path(__file__).parents[1] / "mujoco_metal" / "shaders"
            / "coupled_constraints.metal").read_text()
  assert "float sum = -aref[row], correction = -aref_low[row];" in shader


def test_elliptic_pair_low_force_uses_negative_objective_hessian():
  """The cone force is -gradient, so its low-order delta is -H*r_low."""
  residual = np.array([0.3, 0.5], dtype=np.float64)
  row_R = np.array([0.9, 1.1], dtype=np.float64)
  friction = np.array([0.8, 0.8], dtype=np.float64)
  perturbation = np.array([2e-7, -3e-7], dtype=np.float64)
  force, hessian = _middle_elliptic_force_hessian(
      residual, row_R, friction)
  shifted_force, _ = _middle_elliptic_force_hessian(
      residual + perturbation, row_R, friction)
  finite_difference = shifted_force - force
  predicted = -hessian @ perturbation
  np.testing.assert_allclose(finite_difference, predicted, rtol=2e-6, atol=1e-13)
  assert np.linalg.norm(finite_difference - hessian @ perturbation) > 1e-7


def test_native_hessian_witness_dense_j_uses_packed_record_abi():
  """The production row accessor requires the initialized 12-word header."""
  matrix = np.array([[0.75, -1.25], [.5, 2.]], dtype=np.float32)
  record = _dense_ccj_record(matrix)
  header = record[:12].view(np.int32)
  assert tuple(header[:6]) == (0x4d4a4353, 1, 0, 2, 2, 4)
  assert tuple(header[6:10]) == (0, 0, 12, 16)
  np.testing.assert_array_equal(record[header[8]:header[8] + header[5]],
                                matrix.reshape(-1))
  # Exercise the single-row shape used by primal_precision_witness as well:
  # the mathematical oracle holds a vector, but the record has (nr, nv).
  vector = np.array([0.75, -1.25], dtype=np.float32)
  single = _dense_ccj_record(vector.reshape(1, -1))
  single_header = single[:12].view(np.int32)
  assert tuple(single_header[:6]) == (0x4d4a4353, 1, 0, 1, 2, 2)
  np.testing.assert_array_equal(single[single_header[8]:], vector)


def test_primal_pair_native_witness_executes_production_hessian_and_line_helpers():
  """Run the production MSL H*v and line-polynomial helpers on an exact quad."""
  import pytest
  torch = pytest.importorskip("torch")
  if not torch.backends.mps.is_available() or not hasattr(torch.mps, "compile_shader"):
    pytest.skip("requires native MPS shader execution")

  root = Path(__file__).parents[1]
  shader_root = root / "mujoco_metal" / "shaders"
  from mujoco_metal.coupled_constraints import _coupled_shader_source
  source = _coupled_shader_source(expected_root=shader_root)
  library = torch.mps.compile_shader(source)
  f32 = lambda values: torch.tensor(values, dtype=torch.float32, device="mps")
  i32 = lambda values: torch.tensor(values, dtype=torch.int32, device="mps")
  mass_np = np.array([[4.0, 1.0], [1.0, 3.0]], dtype=np.float64)
  jac_np = np.array([0.75, -1.25], dtype=np.float64)
  row_r = np.array([0.5], dtype=np.float64)
  q0 = np.array([0.25, -0.5], dtype=np.float64)
  acc = np.array([0.75, 0.125], dtype=np.float32)
  acc_low = np.array([2e-8, -1e-8], dtype=np.float32)
  direction = np.array([0.5, -0.25], dtype=np.float32)
  direction_low = np.array([1e-8, 3e-9], dtype=np.float32)
  mass_delta = mass_np @ (acc.astype(np.float64) - q0)
  mass_delta_low = np.array([2e-8, -3e-8], dtype=np.float32)
  residual = np.array([0.2], dtype=np.float32)
  residual_low = np.array([2e-8], dtype=np.float32)
  dimensions = np.zeros(32, dtype=np.int32)
  dimensions[22] = 1
  dimensions[23] = 30
  dimensions[30:32] = -1  # Unknown ancestry deliberately means awake.
  output = torch.zeros((10,), dtype=torch.float32, device="mps")
  scratch = torch.zeros((4,), dtype=torch.float32, device="mps")
  library.primal_precision_witness(
      f32(mass_np.astype(np.float32).reshape(-1)),
      f32(_dense_ccj_record(jac_np.reshape(1, -1))),
      f32(row_r.astype(np.float32)), f32([0.1]), f32(q0.astype(np.float32)),
      f32(acc), f32(acc_low), f32(direction), f32(direction_low), f32([2.0]),
      i32([1]), i32([0]), i32([-1]), i32([0]), f32([0.0] * 5),
      f32(mass_delta.astype(np.float32)), f32(mass_delta_low), f32(residual),
      f32(residual_low), i32(dimensions), output, scratch,
      threads=(1,), group_size=(1,))
  actual = output.cpu().numpy()

  # The line objective is 1/2 d'Md + 1/2 (r + alpha*Jd)^2/R,
  # expanded around the accepted acceleration. Inputs are exactly the
  # float32 high/low pairs sent to the production kernel.
  d = direction.astype(np.float64) + direction_low.astype(np.float64)
  mdelta = mass_delta.astype(np.float64) + mass_delta_low.astype(np.float64)
  r0 = float(residual[0]) + float(residual_low[0])
  jd = float(jac_np @ d)
  alpha = 0.375
  cost = alpha * (mdelta @ d + r0 * jd / row_r[0]) + 0.5 * alpha**2 * (
      d @ mass_np @ d + jd * jd / row_r[0])
  slope = mdelta @ d + r0 * jd / row_r[0] + alpha * (
      d @ mass_np @ d + jd * jd / row_r[0])
  curvature = d @ mass_np @ d + jd * jd / row_r[0]
  hessian = mass_np + np.outer(jac_np, jac_np) * 2.0
  hv = hessian @ d
  expected = np.array([cost, slope, curvature, hv[0], hv[1]])
  np.testing.assert_allclose(
      actual[[0, 2, 4, 6, 7]] + actual[[1, 3, 5, 8, 9]], expected,
      rtol=2e-6, atol=2e-7)


def test_native_hager_zhang_keeps_outer_cg_history_and_direction_paired():
  """The production HZ helper retains low words and excludes other islands."""
  import pytest
  torch = pytest.importorskip("torch")
  if not torch.backends.mps.is_available() or not hasattr(torch.mps, "compile_shader"):
    pytest.skip("requires native MPS shader execution")

  root = Path(__file__).parents[1]
  shader_root = root / "mujoco_metal" / "shaders"
  from mujoco_metal.coupled_constraints import _coupled_shader_source
  source = _coupled_shader_source(expected_root=shader_root)
  library = torch.mps.compile_shader(source)
  f32 = lambda values: torch.tensor(values, dtype=torch.float32, device="mps")
  i32 = lambda values: torch.tensor(values, dtype=torch.int32, device="mps")
  direction = np.array([.5, -.25, 9., 1.2, .1], dtype=np.float32)
  direction_low = np.array([2e-8, -1e-8, .4, 1e-7, 3e-9], dtype=np.float32)
  gradient = np.array([.1, -.2, 7., .05, .4], dtype=np.float32)
  gradient_low = np.array([2e-8, -3e-8, .2, 4e-8, 1e-8], dtype=np.float32)
  previous_gradient = np.array([.09, -.19, -2., .04, .39], dtype=np.float32)
  previous_gradient_low = np.array([-1e-8, 2e-8, .3, -2e-8, 4e-9], dtype=np.float32)
  mgradient = np.array([.025, -.07, 5., .08, .2], dtype=np.float32)
  mgradient_low = np.array([-2e-9, 4e-9, .7, -1e-8, 3e-9], dtype=np.float32)
  old_mgradient = np.array([.02, -.06, -3., .07, .19], dtype=np.float32)
  old_mgradient_low = np.array([1e-9, -4e-9, .1, 3e-9, -2e-9], dtype=np.float32)
  islands = np.array([0, 0, 1, 0, 0], dtype=np.int32)
  output = torch.zeros((12,), dtype=torch.float32, device="mps")
  library.primal_hager_zhang_witness(
      f32(direction), f32(direction_low), f32(gradient), f32(gradient_low),
      f32(previous_gradient), f32(previous_gradient_low), f32(mgradient),
      f32(mgradient_low), f32(old_mgradient), f32(old_mgradient_low),
      i32(islands), output, threads=(1,), group_size=(1,))
  actual = output.cpu().numpy()

  d = direction.astype(np.float64) + direction_low.astype(np.float64)
  g = gradient.astype(np.float64) + gradient_low.astype(np.float64)
  pg = (previous_gradient.astype(np.float64)
        + previous_gradient_low.astype(np.float64))
  mg = mgradient.astype(np.float64) + mgradient_low.astype(np.float64)
  oldmg = old_mgradient.astype(np.float64) + old_mgradient_low.astype(np.float64)
  mask = islands == 0
  y, my = g[mask] - pg[mask], mg[mask] - oldmg[mask]
  dy = float(d[mask] @ y)
  assert dy > 1e-15
  beta_hz = (float(y @ mg[mask])
             - 2.0 * float(y @ my) / dy * float(d[mask] @ g[mask])) / dy
  eta = -1.0 / max(1e-15, np.linalg.norm(d[mask])
                   * min(.01, np.linalg.norm(g[mask])))
  beta = max(eta, beta_hz)
  expected_direction = np.zeros(5, dtype=np.float64)
  expected_direction[mask] = -mg[mask] + beta * d[mask]
  np.testing.assert_allclose(actual[0] + actual[1], beta, rtol=2e-6, atol=2e-8)
  np.testing.assert_allclose(actual[2:7] + actual[7:12], expected_direction,
                             rtol=2e-6, atol=2e-8)


def test_native_paired_hessian_operator_matches_basis_reconstruction():
  """Paired H*v must match applying the operator captured column-by-column."""
  import pytest
  torch = pytest.importorskip("torch")
  if not torch.backends.mps.is_available() or not hasattr(torch.mps, "compile_shader"):
    pytest.skip("requires native MPS shader execution")

  root = Path(__file__).parents[1]
  shader_root = root / "mujoco_metal" / "shaders"
  from mujoco_metal.coupled_constraints import _coupled_shader_source
  source = _coupled_shader_source(expected_root=shader_root)
  library = torch.mps.compile_shader(source)
  f32 = lambda values: torch.tensor(values, dtype=torch.float32, device="mps")
  i32 = lambda values: torch.tensor(values, dtype=torch.int32, device="mps")

  nv, n = 6, 5
  raw = np.array([
      [1.1, -.3, .5, .2, -.8, .4],
      [.2, 1.3, -.4, .7, .1, -.6],
      [-.3, .1, .9, -.2, .5, .8],
      [.6, -.7, .2, 1.0, .3, -.4],
      [.5, .4, -.9, .1, 1.2, .2],
      [-.2, .8, .3, -.5, .7, 1.1],
  ], dtype=np.float64)
  mass = raw.T @ raw + 2.0 * np.eye(nv)
  jacobian = np.array([
      [1.7, -2.1, .3, .7, -1.2, .9],
      [.4, .8, -.3, 1.1, .2, -.6],
      [-.7, .5, 1.2, -.4, .9, .3],
      [.2, -1.0, .6, .8, -.5, 1.3],
      [.9, .1, -.8, .3, .7, -.2],
  ], dtype=np.float64)
  row_r = np.array([.7, .2, .4, .6, .8], dtype=np.float32)
  row_hessian = np.array([.65, 0, 0, 0, 0], dtype=np.float32)
  residual = np.array([0, -.1, .4, -.2, .3], dtype=np.float32)
  residual_low = np.array([0, 2e-8, -3e-8, 1e-8, -2e-8], dtype=np.float32)
  friction = np.array([.8, .6, .04, .1, .2], dtype=np.float32)
  vector = np.array([90., -90., 1., -1., .123, -.123], dtype=np.float32)
  vector_low = np.array([2e-6, -2e-6, -3e-7, 3e-7, 2e-8, -2e-8], dtype=np.float32)
  dimensions = np.zeros(48, dtype=np.int32)
  dimensions[0] = nv
  dimensions[1] = n
  dimensions[22] = 1
  dimensions[23] = 32
  dimensions[32:32 + nv] = 0
  dims = i32(dimensions)
  output = torch.zeros((2 * nv * nv + 2 * nv,), dtype=torch.float32, device="mps")
  scratch = torch.zeros((4 * nv,), dtype=torch.float32, device="mps")
  library.primal_hessian_operator_pair_witness(
      f32(mass.astype(np.float32).reshape(-1)),
      f32(_dense_ccj_record(jacobian)), f32(row_r),
      f32(np.zeros(n, dtype=np.float32)), f32(row_hessian),
      i32(np.ones(n, dtype=np.int32)),
      i32(np.array([0, 1, 1, 1, 1], dtype=np.int32)),
      f32(np.zeros(nv, dtype=np.float32)), f32(residual), f32(residual_low),
      i32([1]), i32([4]), f32(friction), i32([1]), f32(vector),
      f32(vector_low), i32([1]), i32(np.zeros(nv, dtype=np.int32)), dims,
      output, scratch, threads=(1,), group_size=(1,))
  actual = output.cpu().numpy().astype(np.float64)
  basis_hi = actual[:nv * nv].reshape(nv, nv)
  basis_low = actual[nv * nv:2 * nv * nv].reshape(nv, nv)
  operator_hi = actual[2 * nv * nv:2 * nv * nv + nv]
  operator_low = actual[2 * nv * nv + nv:]
  basis = basis_hi + basis_low
  operator = operator_hi + operator_low

  _, cone_hessian = _middle_elliptic_force_hessian(
      residual[1:].astype(np.float64) + residual_low[1:].astype(np.float64),
      row_r[1:].astype(np.float64), friction.astype(np.float64))
  expected = mass + row_hessian[0] * np.outer(jacobian[0], jacobian[0])
  expected += jacobian[1:].T @ cone_hessian @ jacobian[1:]
  direction = vector.astype(np.float64) + vector_low.astype(np.float64)
  np.testing.assert_allclose(basis, expected, rtol=3e-5, atol=3e-5)
  np.testing.assert_allclose(operator, basis @ direction, rtol=2e-6, atol=2e-5)
  np.testing.assert_allclose(operator, expected @ direction, rtol=3e-5, atol=4e-5)


def _scaled_scalar_pcg_oracle(mass, rhs, outer_tolerance, outer_scale):
  """Independent float64 oracle for scale choices and fail-closed outputs."""
  mass = float(mass)
  rhs = float(np.float32(rhs))
  assert np.isfinite(mass) and mass > 0.0 and np.isfinite(rhs)
  if rhs == 0.0:
    return 0.0, True
  if abs(rhs) < float(np.finfo(np.float32).tiny):
    return float("nan"), False
  rhs_scale = abs(rhs)
  normalized_rhs = rhs / rhs_scale
  normalized_solution = normalized_rhs / mass
  solution = normalized_solution * rhs_scale
  if (not np.isfinite(solution)
      or abs(solution) > float(np.finfo(np.float32).max)):
    return float("nan"), False
  # Validate the same strict original residual condition in normalized units.
  normalized_residual = normalized_rhs - mass * normalized_solution
  allowed = min(1e-6, (outer_tolerance / max(outer_scale, 1e-30)) / rhs_scale)
  return solution, abs(normalized_residual) <= allowed


def test_scaled_scalar_pcg_cpu_oracle_covers_tiny_huge_and_overflow_scales():
  tiny = np.float32(2.0 ** -80)
  solution, accepted = _scaled_scalar_pcg_oracle(2.0, tiny, 1e-8, 1e20)
  assert accepted
  assert solution == float(tiny) / 2.0
  # The old unscaled rhs^2 cutoff took this as an exact zero RHS.
  assert np.float32(tiny * tiny) == 0.0
  canonical_tiny = np.float32(1.0e-23)
  assert canonical_tiny != 0.0
  assert np.float32(canonical_tiny * canonical_tiny) == 0.0
  tiny_solution, tiny_accepted = _scaled_scalar_pcg_oracle(
      2.0, canonical_tiny, 1e-8, 1.0)
  assert tiny_accepted and tiny_solution == float(canonical_tiny) / 2.0
  # If normalization makes the requested bound exactly zero, only an exact
  # represented residual may pass. The normalized scalar solve is exact.
  solution, accepted = _scaled_scalar_pcg_oracle(
      2.0, tiny, float(np.finfo(np.float32).tiny), 1e20)
  assert accepted and solution == float(tiny) / 2.0
  # Norm scaling must use the represented pair sum. Scaling from the uncancelled
  # high word would square this 2^-24 residual down near float32 precision.
  cancel_low = np.nextafter(np.float32(-1.0), np.float32(0.0))
  represented = float(np.float32(1.0)) + float(cancel_low)
  assert represented == 2.0 ** -24
  assert represented / 2.0 == 2.0 ** -25

  huge = np.float32(2.0 ** 100)
  solution, accepted = _scaled_scalar_pcg_oracle(2.0, huge, 1e30, 1.0)
  assert accepted and solution == float(huge) / 2.0

  # The preconditioned direction itself can be tiny even for a normal RHS.
  solution, accepted = _scaled_scalar_pcg_oracle(2.0 ** 100, 1.0, 1e-8, 1.0)
  assert accepted and solution == 2.0 ** -100

  mixed = np.asarray([2.0 ** 40, 2.0 ** -20], dtype=np.float32)
  mixed_scale = np.max(np.abs(mixed))
  normalized = mixed / mixed_scale
  assert normalized[1] == np.float32(2.0 ** -60)
  assert normalized[1] != 0.0
  expected = np.asarray([2.0 ** 39, 2.0 ** -22], dtype=np.float64)
  np.testing.assert_array_equal(
      np.asarray(mixed, dtype=np.float64)
      / np.asarray([2.0, 4.0], dtype=np.float64), expected)

  # A finite input whose exact solution is outside float32 must fail closed.
  _, accepted = _scaled_scalar_pcg_oracle(2.0 ** -100, huge, 1e30, 1.0)
  assert not accepted
  _, accepted = _scaled_scalar_pcg_oracle(
      1.0, np.nextafter(np.float32(0.0), np.float32(1.0)), 1e-8, 1.0)
  assert not accepted


def test_componentwise_inner_certificate_rejects_one_bad_active_dof():
  # The global norm is under the requested bound, while one DOF exceeds its
  # equal-share component bound. The shader must retain both checks.
  residual = np.asarray([7e-7, 0.0, 0.0, 0.0], dtype=np.float64)
  norm_limit = 1e-6
  component_limit = norm_limit / np.sqrt(residual.size)
  assert np.linalg.norm(residual) <= norm_limit
  assert np.any(np.abs(residual) > component_limit)
  assert not np.all(np.abs(residual) <= component_limit)


def test_cancellation_pair_is_canonicalized_before_mass_preconditioning():
  # Mirror the exact two-sum renormalization used before the source's
  # high-word-first triangular solve. Without this step, solving 1 and then
  # correcting by -0.99999994 loses the represented 2^-24 RHS.
  hi = np.float32(1.0)
  lo = np.nextafter(np.float32(-1.0), np.float32(0.0))
  total = np.float32(hi + lo)
  virtual_lo = np.float32(total - hi)
  error = np.float32(np.float32(hi - np.float32(total - virtual_lo))
                     + np.float32(lo - virtual_lo))
  assert total == np.float32(2.0 ** -24)
  assert error == np.float32(0.0)
  solution = np.float64(total) / np.float64(2.0)
  assert solution == 2.0 ** -25

  source = (Path(__file__).parents[1] / "mujoco_metal" / "shaders"
            / "coupled_constraints.metal").read_text()
  entry = source.index("inline bool primal_hessian_solve_elliptic(")
  end = source.index("// Native regression entrypoint for the production bounded PCG solve.",
                     entry)
  dispatch = source[entry:end]
  assert "primal_hessian_solve_elliptic_legacy(" in dispatch
  assert "primal_hessian_solve_elliptic_scaled(" in dispatch
  assert "if (legacy_safe)" in dispatch
  assert "rhs_all_active_zero" in dispatch


def test_native_production_pcg_scale_witness():
  """Run the production PCG routine at tiny, huge and ill-scaled magnitudes."""
  import pytest
  torch = pytest.importorskip("torch")
  if not torch.backends.mps.is_available() or not hasattr(torch.mps, "compile_shader"):
    pytest.skip("requires native MPS shader execution")

  root = Path(__file__).parents[1]
  shader_root = root / "mujoco_metal" / "shaders"
  from mujoco_metal.coupled_constraints import _coupled_shader_source
  source = _coupled_shader_source(expected_root=shader_root)
  library = torch.mps.compile_shader(source)
  f32 = lambda values: torch.tensor(values, dtype=torch.float32, device="mps")
  i32 = lambda values: torch.tensor(values, dtype=torch.int32, device="mps")
  cases = [
      # The canonical nonzero RHS has a square that flushes to zero. It must
      # bypass the legacy zero-norm shortcut and retain the scaled solution.
      ([2.0], [np.sqrt(2.0)], [1.0e-23], [0.0],
       1e-8, 1.0, True, [5.0e-24]),
      # Old rhs_norm_sq <= 1e-30 returned the zero solution here.
      ([2.0], [np.sqrt(2.0)], [2.0 ** -80], [0.0],
       1e-8, 1e20, True, [2.0 ** -81]),
      # The normalized outer bound underflows to zero; exact residual still
      # satisfies the unrelaxed requested bound.
      ([2.0], [np.sqrt(2.0)], [2.0 ** -80], [0.0],
       float(np.finfo(np.float32).tiny), 1e20, True, [2.0 ** -81]),
      # A small represented RHS can be carried by cancellation between its
      # high and low words; norm scaling must retain that difference.
      ([2.0], [np.sqrt(2.0)], [1.0],
       [float(np.nextafter(np.float32(-1.0), np.float32(0.0)))],
       1e-8, 1.0, True, [2.0 ** -25]),
      # Squaring this RHS overflowed; normalized PCG remains well-scaled.
      ([2.0], [np.sqrt(2.0)], [2.0 ** 100], [0.0],
       1e30, 1.0, True, [2.0 ** 99]),
      # Large mass makes the unscaled preconditioned seed tiny.
      ([2.0 ** 100], [2.0 ** 50], [1.0], [0.0],
       1e-8, 1.0, True, [2.0 ** -100]),
      # The large and small active components both survive normalization.
      ([2.0, 0.0, 0.0, 4.0], [np.sqrt(2.0), 0.0, 0.0, 2.0],
       [2.0 ** 40, 2.0 ** -20], [0.0, 0.0],
       1e30, 1.0, True, [2.0 ** 39, 2.0 ** -22]),
      # Exact output is outside float32; the solve must report failure.
      ([2.0 ** -100], [2.0 ** -50], [2.0 ** 100], [0.0],
       1e30, 1.0, False, [0.0]),
      # Nonzero high/low words can represent an exact zero right-hand side.
      ([2.0], [np.sqrt(2.0)], [1.0], [-1.0],
       1e-8, 1.0, True, [0.0]),
  ]
  failures = []
  for mass_values, factor_values, rhs_hi, rhs_lo, tolerance, outer_scale, expected_ok, expected_x in cases:
    nv = len(rhs_hi)
    dims_np = np.zeros(32 + nv, dtype=np.int32)
    dims_np[0], dims_np[22], dims_np[23] = nv, 1, 32
    dims_np[32:32 + nv] = np.arange(nv, dtype=np.int32)
    dims = i32(dims_np)
    mass = f32(mass_values)
    factor = f32(factor_values)
    rhs = f32(rhs_hi + rhs_lo)
    tol_scale = f32([tolerance, outer_scale])
    awake = i32([1] * nv)
    island = i32([0] * nv)
    dummy_i, dummy_f = i32([0]), f32([0.0])
    factor_scratch = f32(np.zeros(4 * nv, dtype=np.float32))
    pair_tail = f32(np.zeros(7 * nv, dtype=np.float32))
    temp = f32(np.zeros(nv, dtype=np.float32))
    inner_residual = f32(np.zeros(nv, dtype=np.float32))
    inner_preconditioned = f32(np.zeros(nv, dtype=np.float32))
    inner_direction = f32(np.zeros(nv, dtype=np.float32))
    output = f32(np.zeros(2 * nv + 1, dtype=np.float32))
    library.primal_hessian_scale_witness(
        mass, factor, rhs, tol_scale, dims, awake, island, dummy_i, dummy_f,
        factor_scratch, pair_tail, temp, inner_residual,
        inner_preconditioned, inner_direction, output,
        threads=(1,), group_size=(1,))
    actual = output.cpu().numpy()
    case = {"mass": mass_values, "rhs_hi": rhs_hi, "rhs_lo": rhs_lo,
            "tolerance": tolerance, "scale": outer_scale,
            "expected_ok": expected_ok, "actual": actual.tolist()}
    print("PCG_SCALE_CASE", case)
    if actual[2 * nv] != (1.0 if expected_ok else 0.0):
      failures.append(case)
      continue
    if expected_ok:
      np.testing.assert_allclose(actual[:nv].astype(np.float64)
                                 + actual[nv:2 * nv].astype(np.float64), expected_x,
                                 rtol=2e-6, atol=0.0)
  assert not failures, failures
