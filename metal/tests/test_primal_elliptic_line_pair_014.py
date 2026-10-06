"""Source-order CPU oracle and opt-in witness for the elliptic PrimalEval line."""
import os
from pathlib import Path

import numpy as np
import pytest


def _case(dim, zone, alpha):
  r = np.zeros(6, dtype=np.float32)
  rlo = np.zeros(6, dtype=np.float32)
  d = np.zeros(6, dtype=np.float32)
  dlo = np.zeros(6, dtype=np.float32)
  if zone == "middle":
    r[:dim] = [0.04, -0.5, 0.3, 0.17, -0.23, 0.31][:dim]
    rlo[:dim] = [1e-8, -2e-8, 3e-8, -4e-8, 5e-8, -6e-8][:dim]
    d[:dim] = [-0.13, 0.22, -0.31, 0.07, 0.11, -0.19][:dim]
    dlo[:dim] = [2e-8, -3e-8, 1e-8, 4e-8, -5e-8, 6e-8][:dim]
  elif zone == "top_to_middle":
    r[0] = 2.; r[1:dim] = .002
    d[0] = -2.5; d[1:dim] = [.4, -.25, .1, .3, -.2][:dim-1]
  elif zone == "middle_to_bottom":
    r[0] = .05; r[1:dim] = [.3, -.2, .15, .1, -.08][:dim-1]
    d[0] = -.9; d[1:dim] = [.02, -.01, .03, -.02, .01][:dim-1]
  elif zone == "bottom":
    r[0] = -.8; r[1:dim] = [.1, -.1, .03, .05, -.02][:dim-1]
    d[:dim] = [.2, -.03, .02, .01, -.02, .04][:dim]
  elif zone == "zero_tangent":
    r[0] = -.2; d[:dim] = [.1, 0., 0., 0., 0., 0.][:dim]
  else:
    raise ValueError(zone)
  R = np.asarray([2., .3, .5, 1., 1.2, .7], np.float32)
  friction = np.asarray([.8, .6, .3, .4, .5], np.float32)
  return r, rlo, d, dlo, R, friction, np.asarray([alpha, 2e-8], np.float32)


def _source_cost(alpha, residual, direction, R, friction, dim):
  """engine_solver.c ellipticCost, evaluated on the represented pair values."""
  x = residual.astype(np.float64) + np.arange(6) * 0.0
  v = direction.astype(np.float64)
  a = float(alpha)
  mu = max(float(friction[0]) * np.sqrt(max(float(R[1]), 0.) /
                                         max(float(R[0]), 1e-15)), 0.)
  N0 = mu * x[0]
  Nd = mu * v[0]
  Ts0 = sum((float(friction[k-1]) * x[k])**2 for k in range(1, dim))
  Tcross = sum((float(friction[k-1]) * x[k]) *
               (float(friction[k-1]) * v[k]) for k in range(1, dim))
  Tsdir = sum((float(friction[k-1]) * v[k])**2 for k in range(1, dim))
  T2 = Ts0 + a * (2*Tcross + a*Tsdir)
  N = N0 + a*Nd
  bottom0 = .5 * sum(x[k]**2 / max(float(R[k]), 1e-15)
                     for k in range(dim))
  bottom_linear = sum(x[k]*v[k] / max(float(R[k]), 1e-15)
                      for k in range(dim))
  bottom_quadratic = .5 * sum(v[k]**2 / max(float(R[k]), 1e-15)
                              for k in range(dim))
  Dm = 1. / (max(float(R[0]), 1e-15)*mu*mu*(1+mu*mu))
  def at(a0):
    n = N0 + a0*Nd
    ts2 = Ts0 + a0*(2*Tcross + a0*Tsdir)
    if ts2 <= 0:
      return (a0*a0*bottom_quadratic + a0*bottom_linear + bottom0
              if n < 0 else 0.)
    t = np.sqrt(ts2)
    if n >= mu*t:
      return 0.
    if mu*n+t <= 0:
      return a0*a0*bottom_quadratic + a0*bottom_linear + bottom0
    return .5*Dm*(n-mu*t)**2
  return at(a)-at(0.)


def _source_derivatives(alpha, residual, direction, R, friction, dim):
  a = float(alpha)
  x = residual.astype(np.float64); v = direction.astype(np.float64)
  mu = max(float(friction[0])*np.sqrt(max(float(R[1]),0.) /
                                       max(float(R[0]),1e-15)),0.)
  N0, Nd = mu*x[0], mu*v[0]
  t0sq = sum((float(friction[k-1])*x[k])**2 for k in range(1,dim))
  cross = sum((float(friction[k-1])*x[k])*(float(friction[k-1])*v[k])
              for k in range(1,dim))
  dsq = sum((float(friction[k-1])*v[k])**2 for k in range(1,dim))
  n=N0+a*Nd; tsq=t0sq+a*(2*cross+a*dsq)
  if tsq <= 0:
    if n < 0:
      return (sum(x[k]*v[k]/R[k] for k in range(dim))+
              2*a*.5*sum(v[k]**2/R[k] for k in range(dim)),
              sum(v[k]**2/R[k] for k in range(dim)))
    return 0.,0.
  t=np.sqrt(tsq)
  if n >= mu*t: return 0.,0.
  if mu*n+t <= 0:
    return (sum(x[k]*v[k]/R[k] for k in range(dim))+
            2*a*.5*sum(v[k]**2/R[k] for k in range(dim)),
            sum(v[k]**2/R[k] for k in range(dim)))
  Dm=1./(R[0]*mu*mu*(1+mu*mu))
  tf=(cross+a*dsq)/t
  t2=dsq/t-(cross+a*dsq)*tf/(t*t)
  gap=n-mu*t; gf=Nd-mu*tf; g2=-mu*t2
  return Dm*gap*gf, Dm*(gf*gf+gap*g2)


@pytest.mark.parametrize("dim", [3, 4, 6])
@pytest.mark.parametrize("zone,alpha", [("middle", .375),
    ("top_to_middle", .7), ("middle_to_bottom", .8), ("bottom", .3),
    ("zero_tangent", .5)])
def test_cpu_source_line_formula_is_finite_and_zone_sensitive(dim, zone, alpha):
  r, rl, d, dl, R, friction, _ = _case(dim, zone, alpha)
  residual = r.astype(np.float64) + rl.astype(np.float64)
  direction = d.astype(np.float64) + dl.astype(np.float64)
  delta = _source_cost(alpha, residual, direction, R, friction, dim)
  deriv, second = _source_derivatives(alpha, residual, direction, R,
                                      friction, dim)
  h = 2e-5
  fd1 = (_source_cost(alpha+h,residual,direction,R,friction,dim)-
         _source_cost(alpha-h,residual,direction,R,friction,dim))/(2*h)
  assert np.isfinite(delta) and np.isfinite(deriv) and np.isfinite(second)
  if zone in ("middle", "top_to_middle", "middle_to_bottom"):
    assert abs(deriv-fd1) < 2e-6


@pytest.mark.parametrize("dim", [3,4,6])
@pytest.mark.parametrize("zone,alpha", [("middle",.375),
    ("top_to_middle",.7), ("middle_to_bottom",.8), ("bottom",.3),
    ("zero_tangent",.5)])
def test_native_pair_line_matches_source_cost_and_derivatives(dim,zone,alpha):
  if os.environ.get("MUJOCO_METAL_RUN_GPU") != "1":
    pytest.skip("native MPS execution is opt-in via MUJOCO_METAL_RUN_GPU=1")
  torch=pytest.importorskip("torch")
  if not torch.backends.mps.is_available() or not hasattr(torch.mps,"compile_shader"):
    pytest.skip("requires native MPS shader execution")
  r,rl,d,dl,R,friction,alpha_pair=_case(dim,zone,alpha)
  inp=np.concatenate((r,rl,d,dl,R,friction,alpha_pair,
                      np.asarray([dim],np.float32)))
  root=Path(__file__).resolve().parents[1]/"mujoco_metal"/"shaders"
  override=os.environ.get("MUJOCO_METAL_SHADER_ROOT")
  if override and Path(override).resolve()!=root:
    raise AssertionError("native witness must use its own revision's shader sources")
  assert root.is_dir(),root
  from mujoco_metal.coupled_constraints import _coupled_shader_source
  source = _coupled_shader_source(expected_root=root)
  lib=torch.mps.compile_shader(source)
  out=torch.zeros((6,),dtype=torch.float32,device="mps")
  lib.primal_elliptic_line_pair_witness(
      torch.tensor(inp,dtype=torch.float32,device="mps"),out,
      threads=(1,),group_size=(1,))
  got=out.cpu().numpy().astype(np.float64)
  residual=r.astype(np.float64)+rl.astype(np.float64)
  direction=d.astype(np.float64)+dl.astype(np.float64)
  alpha_value=float(alpha_pair[0])+float(alpha_pair[1])
  cost=_source_cost(alpha_value,residual,direction,R,friction,dim)
  derivative,second=_source_derivatives(alpha_value,residual,direction,R,
                                        friction,dim)
  expected=np.asarray([cost,0.,derivative,0.,second,0.])
  print("PRIMAL_ELLIPTIC_LINE_PAIR",dim,zone,"actual=",got.tolist(),
        "source=",expected.tolist(),flush=True)
  assert np.max(np.abs((got[::2]+got[1::2])-expected[::2])) <= 2e-12


def _pinned_constraint_cost(dim, jar):
  """Evaluate the pinned 3.10 callback rather than a Python reconstruction."""
  mujoco=pytest.importorskip("mujoco")
  xml=f'''<mujoco><option cone="elliptic" impratio="1"/><worldbody>
    <geom name="floor" type="plane" size="2 2 .1" condim="{dim}"
          friction=".5 .25 .125"/>
    <body pos="0 0 .09"><freejoint/><geom name="ball" type="sphere"
          size=".1" condim="{dim}" friction=".5 .25 .125"/></body>
    </worldbody></mujoco>'''
  model=mujoco.MjModel.from_xml_string(xml)
  data=mujoco.MjData(model)
  mujoco.mj_forward(model,data)
  assert data.ncon == 1 and data.nefc == dim
  cost=np.zeros(1,dtype=np.float64)
  mujoco.mj_constraintUpdate(model,data,np.asarray(jar[:dim],np.float64),cost,1)
  R=np.zeros(6); R[:dim]=data.efc_R[:dim]
  friction=np.zeros(5); friction[:dim-1]=data.contact[0].friction[:dim-1]
  return float(cost[0]),R,friction


@pytest.mark.parametrize("dim",[3,4,6])
@pytest.mark.parametrize("zone,alpha",[("middle",.375),
    ("top_to_middle",.7),("middle_to_bottom",.8),("bottom",.3),
    ("zero_tangent",.5)])
def test_line_cost_matches_pinned_mj_constraint_update(dim,zone,alpha):
  """The independent formula agrees with the actual pinned C callback."""
  mujoco=pytest.importorskip("mujoco")
  r,rl,d,dl,_,_,_=_case(dim,zone,alpha)
  start=r.astype(np.float64)+rl.astype(np.float64)
  direction=d.astype(np.float64)+dl.astype(np.float64)
  c0,R,friction=_pinned_constraint_cost(dim,start)
  c1,_,_=_pinned_constraint_cost(dim,start+float(alpha)*direction)
  formula=_source_cost(alpha,start,direction,R,friction,dim)
  assert abs((c1-c0)-formula) <= 2e-12
