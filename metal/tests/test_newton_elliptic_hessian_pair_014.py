"""CPU oracle and opt-in MPS witness for paired elliptic-cone curvature."""
import os
from pathlib import Path

import numpy as np
import pytest


def _fixture(dim=3, zone="middle"):
  rhi = np.zeros(6, dtype=np.float32)
  rlo = np.zeros(6, dtype=np.float32)
  if zone == "middle":
    rhi[:dim] = [0.04, -0.5, 0.3, 0.17, -0.23, 0.31][:dim]
    rlo[:dim] = [1.0e-8, -2.0e-8, 3.0e-8, -4.0e-8, 5.0e-8, -6.0e-8][:dim]
  elif zone == "top":
    rhi[0] = 2.0
    rhi[1:dim] = 0.002
  elif zone == "bottom":
    rhi[0] = -2.0
    rhi[1:dim] = 0.002
  elif zone == "zero_tangent":
    rhi[0] = -0.2
  else:
    raise ValueError(zone)
  R = np.asarray([2.0, 0.3, 0.5, 1.0, 1.2, 0.7], dtype=np.float32)
  friction = np.asarray([0.8, 0.6, 0.3, 0.4, 0.5], dtype=np.float32)
  return np.concatenate((rhi, rlo, R, friction,
                         np.asarray([dim], dtype=np.float32)))


def _pinned_source_case(dim, zone):
  """Run pinned MuJoCo 3.10 mj_constraintUpdate on one real elliptic contact."""
  mujoco = pytest.importorskip("mujoco")
  xml = f'''<mujoco><option cone="elliptic" impratio="1"/>
    <worldbody>
      <geom name="floor" type="plane" size="2 2 .1" condim="{dim}"
            friction=".5 .25 .125"/>
      <body pos="0 0 .09"><freejoint/>
        <geom name="ball" type="sphere" size=".1" condim="{dim}"
              friction=".5 .25 .125"/>
      </body>
    </worldbody></mujoco>'''
  model = mujoco.MjModel.from_xml_string(xml)
  data = mujoco.MjData(model)
  mujoco.mj_forward(model, data)
  assert data.ncon == 1 and data.nefc == dim
  assert data.contact[0].dim == dim
  hi = np.zeros(6, dtype=np.float32)
  lo = np.zeros(6, dtype=np.float32)
  if zone == "middle":
    hi[:dim] = [0.04, -0.5, 0.3, 0.17, -0.23, 0.31][:dim]
    lo[:dim] = [1e-8, -2e-8, 3e-8, -4e-8, 5e-8, -6e-8][:dim]
  elif zone == "top":
    hi[0] = 2.0
    hi[1:dim] = 0.002
  elif zone == "bottom":
    hi[0] = -2.0
    hi[1:dim] = 0.002
  elif zone == "zero_tangent":
    hi[0] = -0.2
  else:
    raise ValueError(zone)
  jar = hi[:dim].astype(np.float64) + lo[:dim].astype(np.float64)
  cost = np.zeros(1, dtype=np.float64)
  mujoco.mj_constraintUpdate(model, data, jar, cost, 1)
  source_force = data.efc_force[:dim].copy()
  source_hessian = data.contact[0].H[:dim * dim].reshape(dim, dim).copy()
  R = np.zeros(6, dtype=np.float64)
  friction = np.zeros(5, dtype=np.float64)
  R[:dim] = data.efc_R[:dim]
  friction[:dim - 1] = data.contact[0].friction[:dim - 1]
  values = np.concatenate((hi.astype(np.float64), lo.astype(np.float64), R,
                           friction, np.asarray([dim], dtype=np.float64)))
  return (values, source_force, source_hessian, data.efc_state[:dim].copy(),
          data.efc_D[:dim].copy())


def _source_hessian(values):
  x = values.astype(np.float64)
  r = x[:6] + x[6:12]
  R = x[12:18]
  friction = x[18:23]
  dim = int(x[23])
  mu = max(float(friction[0]) * np.sqrt(max(float(R[1]), 0.0)
                                         / max(float(R[0]), 1e-15)), 0.0)
  U = np.zeros(dim)
  U[0] = mu * r[0]
  for j in range(1, dim):
    U[j] = friction[j - 1] * r[j]
  N = U[0]
  T = float(np.linalg.norm(U[1:]))
  H = np.zeros((dim, dim), dtype=np.float64)
  if T <= 0.0:
    if N < 0.0:
      H[np.arange(dim), np.arange(dim)] = 1.0 / R[:dim]
    return H
  if mu * N + T <= 0.0:
    H[np.arange(dim), np.arange(dim)] = 1.0 / R[:dim]
    return H
  if N >= mu * T:
    return H
  # Pinned engine_core_constraint.c HessianCone construction, followed by
  # Dm and diagonal friction pre/post scaling.
  Dm = 1.0 / (R[0] * mu * mu * (1.0 + mu * mu))
  raw = np.zeros((dim, dim), dtype=np.float64)
  raw[0, 0] = 1.0
  raw[0, 1:] = (-mu / T) * U[1:]
  s = mu * N / (T * T * T)
  for j in range(1, dim):
    for k in range(j, dim):
      raw[j, k] = s * U[j] * U[k]
  s = mu * mu - mu * N / T
  raw[np.arange(1, dim), np.arange(1, dim)] += s
  scales = np.asarray([mu, *friction[:dim - 1]])
  H = Dm * raw * scales[:, None] * scales[None, :]
  H = np.triu(H) + np.triu(H, 1).T
  return H


def _source_force(values):
  x = values.astype(np.float64)
  r = x[:6] + x[6:12]
  R = x[12:18]
  friction = x[18:23]
  dim = int(x[23])
  mu = max(float(friction[0]) * np.sqrt(max(float(R[1]), 0.0)
                                         / max(float(R[0]), 1e-15)), 0.0)
  U = np.zeros(dim)
  U[0] = mu * r[0]
  for j in range(1, dim):
    U[j] = friction[j - 1] * r[j]
  N = U[0]
  T = float(np.linalg.norm(U[1:]))
  force = np.zeros(dim, dtype=np.float64)
  if T <= 0.0:
    if N < 0.0:
      force[:] = -r[:dim] / np.maximum(R[:dim], 1e-15)
    return force
  if mu * N + T <= 0.0:
    force[:] = -r[:dim] / np.maximum(R[:dim], 1e-15)
    return force
  if N >= mu * T:
    return force
  Dm = 1.0 / (R[0] * mu * mu * (1.0 + mu * mu))
  force[0] = -Dm * (N - mu * T) * mu
  for j in range(1, dim):
    force[j] = -force[0] / T * U[j] * friction[j - 1]
  return force


def _shader_root():
  local = Path(__file__).resolve().parents[1] / "mujoco_metal" / "shaders"
  override = os.environ.get("MUJOCO_METAL_SHADER_ROOT")
  if override and Path(override).resolve() != local:
    raise AssertionError("native witness must use its own revision's shader sources")
  assert local.is_dir(), local
  return local


@pytest.mark.parametrize("dim", [3, 4, 6])
@pytest.mark.parametrize("zone", ["top", "middle", "bottom", "zero_tangent"])
def test_cpu_source_hessian_and_force_fixtures_cover_cone_zones(dim, zone):
  mujoco = pytest.importorskip("mujoco")
  values, source_force, source_hessian, state, source_D = _pinned_source_case(dim, zone)
  H = _source_hessian(values)
  force = _source_force(values)
  assert H.shape == (dim, dim) and force.shape == (dim,)
  assert np.all(np.isfinite(H)) and np.all(np.isfinite(force))
  assert np.max(np.abs(H - H.T)) < 1e-13
  if zone == "middle":
    assert np.linalg.norm(H) > 0.1
    assert np.linalg.norm(force) > 0.01
  np.testing.assert_allclose(force, source_force, rtol=0.0, atol=1e-12)
  if np.all(state == int(mujoco.mjtConstraintState.mjCNSTRSTATE_CONE)):
    source_operator_hessian = source_hessian
  elif np.all(state == int(mujoco.mjtConstraintState.mjCNSTRSTATE_QUADRATIC)):
    # mj_constraintUpdate only stores contact.H in the middle cone. In the
    # quadratic bottom branch its force is exactly -D*jar, so the source
    # objective Hessian is diag(D), although contact.H remains untouched.
    source_operator_hessian = np.diag(source_D)
  else:
    # The top zone is satisfied and has zero force derivative/Hessian.
    source_operator_hessian = np.zeros((dim, dim), dtype=np.float64)
  np.testing.assert_allclose(H, source_operator_hessian, rtol=0.0, atol=1e-12)
  if zone == "middle":
    assert np.all(state == int(mujoco.mjtConstraintState.mjCNSTRSTATE_CONE))


@pytest.mark.parametrize("dim", [3, 4, 6])
@pytest.mark.parametrize("zone", ["top", "middle", "bottom", "zero_tangent"])
def test_native_paired_hessian_matches_source_formula(dim, zone):
  if os.environ.get("MUJOCO_METAL_RUN_GPU") != "1":
    pytest.skip("native MPS execution is opt-in via MUJOCO_METAL_RUN_GPU=1")
  torch = pytest.importorskip("torch")
  if not torch.backends.mps.is_available() or not hasattr(torch.mps, "compile_shader"):
    pytest.skip("requires native MPS shader execution")
  values, source_force, source_hessian, _, _ = _pinned_source_case(dim, zone)
  shader_root = _shader_root()
  from mujoco_metal.coupled_constraints import _coupled_shader_source
  source = _coupled_shader_source(expected_root=shader_root)
  library = torch.mps.compile_shader(source)
  output = torch.zeros((72,), dtype=torch.float32, device="mps")
  library.primal_elliptic_hessian_pair_witness(
      torch.tensor(values, dtype=torch.float32, device="mps"), output,
      threads=(1,), group_size=(1,))
  raw = output.cpu().numpy().astype(np.float64)
  actual = (raw[:36] + raw[36:]).reshape(6, 6)[:dim, :dim]
  high = raw[:36].reshape(6, 6)[:dim, :dim]
  expected = _source_hessian(values.astype(np.float32).astype(np.float64))
  print("ELLIPTIC_HESSIAN_PAIR", dim, zone, "native=", actual.tolist(),
        "high=", high.tolist(), "source=", expected.tolist(), flush=True)
  assert np.max(np.abs(actual - expected)) <= 1e-13
  high_error = np.linalg.norm(high - expected)
  if high_error > 0.0:
    assert np.linalg.norm(actual - expected) < high_error
  else:
    assert np.array_equal(actual, high)
  # The callback is the pinned C source oracle above; its inputs are binary64
  # model values rounded to the float32 values consumed by this shader.
  print("PINNED_C_HESSIAN", dim, zone, "source=", source_hessian.tolist(),
        "force=", source_force.tolist(), flush=True)


@pytest.mark.parametrize("dim", [3, 4, 6])
@pytest.mark.parametrize("zone", ["top", "middle", "bottom", "zero_tangent"])
def test_native_paired_force_matches_source_formula(dim, zone):
  if os.environ.get("MUJOCO_METAL_RUN_GPU") != "1":
    pytest.skip("native MPS execution is opt-in via MUJOCO_METAL_RUN_GPU=1")
  torch = pytest.importorskip("torch")
  if not torch.backends.mps.is_available() or not hasattr(torch.mps, "compile_shader"):
    pytest.skip("requires native MPS shader execution")
  values, source_force, source_hessian, _, _ = _pinned_source_case(dim, zone)
  shader_root = _shader_root()
  from mujoco_metal.coupled_constraints import _coupled_shader_source
  source = _coupled_shader_source(expected_root=shader_root)
  library = torch.mps.compile_shader(source)
  output = torch.zeros((12,), dtype=torch.float32, device="mps")
  library.primal_elliptic_force_pair_witness(
      torch.tensor(values, dtype=torch.float32, device="mps"), output,
      threads=(1,), group_size=(1,))
  raw = output.cpu().numpy().astype(np.float64)
  actual = raw[:6] + raw[6:]
  expected = _source_force(values.astype(np.float32).astype(np.float64))
  print("ELLIPTIC_FORCE_PAIR", dim, zone, "native=", actual.tolist(),
        "source=", expected.tolist(), flush=True)
  assert np.max(np.abs(actual[:dim] - expected)) <= 1e-13
  high_error = np.linalg.norm(raw[:dim] - expected)
  if high_error > 0.0:
    assert np.linalg.norm(actual[:dim] - expected) < high_error
  else:
    assert np.array_equal(actual[:dim], raw[:dim])
  print("PINNED_C_FORCE", dim, zone, "source=", source_force.tolist(),
        "hessian_diag=", np.diag(source_hessian).tolist(), flush=True)
