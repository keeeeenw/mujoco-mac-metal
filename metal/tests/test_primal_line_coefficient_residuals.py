# Copyright 2026 The MuJoCo Metal contributors
# Licensed under the Apache License, Version 2.0.
"""Retained primal-line coefficients against pinned constraint-update oracles."""
import os

import mujoco
import numpy as np
import pytest

_GPU = pytest.mark.skipif(os.getenv("MUJOCO_METAL_RUN_GPU") != "1",
                          reason="native line coefficient qualification")

_WITNESSES = r'''
kernel void line_coefficient_cone_probe(
    device const float* input [[buffer(0)]],
    device float* output [[buffer(1)]], uint tid [[thread_position_in_grid]]) {
  if (tid != 0) return;
  thread PrimalFloatPair residual[6], direction[6], derivative, second;
  for (int i=0; i<6; ++i) {
    residual[i]={input[i],input[6+i]};
    direction[i]={input[12+i],input[18+i]};
  }
  PrimalFloatPair delta=primal_elliptic_line_eval_pair(
      {input[35],input[36]},residual,direction,input+24,input+30,
      int(input[37]),derivative,second,input+38,input+44);
  output[0]=delta.hi; output[1]=delta.lo;
  output[2]=derivative.hi; output[3]=derivative.lo;
  output[4]=second.hi; output[5]=second.lo;
}
kernel void line_coefficient_scalar_probe(
    device const float* input [[buffer(0)]],
    device const float* J [[buffer(1)]], constant int* dims [[buffer(2)]],
    device const float* debug [[buffer(3)]],
    device const int* flags [[buffer(4)]], device float* output [[buffer(5)]],
    device float* scratch [[buffer(6)]], uint tid [[thread_position_in_grid]]) {
  if(tid!=0) return;
  thread PrimalFloatPair derivative,second;
  PrimalFloatPair delta=primal_accel_cost_delta_elliptic(
      {input[0],input[1]},input+11,false,debug,dims,J,input+6,input+14,
      input+12,input+13,input+14,input+14,input+14,input+4,input+5,
      flags,flags+1,flags+2,flags+3,input+14,0,scratch,scratch+1,
      1,1,1,int(input[9]),derivative,second,nullptr,nullptr,
      input+2,input+3,input+7,nullptr);
  output[0]=delta.hi; output[1]=delta.lo;
  output[2]=derivative.hi; output[3]=derivative.lo;
  output[4]=second.hi; output[5]=second.lo;
}
'''


@pytest.fixture(scope="module")
def native_library():
  if os.getenv("MUJOCO_METAL_RUN_GPU") != "1":
    pytest.skip("native opt-in")
  import torch
  from mujoco_metal.coupled_constraints import _coupled_shader_source
  return torch.mps.compile_shader(_coupled_shader_source() + _WITNESSES)


def _split(value):
  high = np.asarray(value, dtype=np.float32)
  return high, (np.asarray(value, dtype=float) - high).astype(np.float32)


def _scalar_source(kind):
  loss = float(np.float32(.41))
  joint = 'frictionloss="%s"' % loss if kind == "friction" else ''
  if kind == "unilateral":
    joint = 'limited="true" range="-1 1"'
  equality = '<equality><joint joint1="slide"/></equality>' if kind == "equality" else ''
  model = mujoco.MjModel.from_xml_string(f'''<mujoco>
    <option gravity="0 0 0"/><default><joint solimpfriction=".83 .91 .001"/></default>
    <worldbody><body><joint name="slide" type="slide" {joint}/>
      <geom type="sphere" size=".1" mass="1.3"/></body></worldbody>
    {equality}</mujoco>''')
  data = mujoco.MjData(model)
  if kind == "unilateral":
    data.qpos[0] = 1.1
  mujoco.mj_forward(model, data)
  assert data.nefc == 1
  return model, data, loss


def _scalar_cost(model, data, residual):
  cost = np.zeros(1)
  mujoco.mj_constraintUpdate(model, data, np.asarray([residual]), cost, 1)
  curvature = (float(data.efc_D[0])
               if data.efc_state[0] == int(mujoco.mjtConstraintState.mjCNSTRSTATE_QUADRATIC)
               else 0.0)
  return float(cost[0]), -float(data.efc_force[0]), curvature


@pytest.mark.parametrize("kind", ["equality", "friction", "unilateral"])
def test_scalar_source_fixture_has_nonzero_retained_regularizer(kind):
  model, data, _ = _scalar_source(kind)
  high, low = _split(data.efc_R[0])
  assert low != 0
  np.testing.assert_allclose(float(high)+float(low), data.efc_R[0], rtol=0, atol=2e-16)
  cost, slope, curvature = _scalar_cost(model, data, -.01)
  assert np.isfinite([cost, slope, curvature]).all()


@pytest.mark.gpu
@_GPU
@pytest.mark.parametrize("kind,start,direction", [
    ("equality", .2, .7), ("equality", -.2, -.7),
    ("unilateral", -.2, .1), ("unilateral", -.2, 1.),
    ("unilateral", .2, -.1), ("unilateral", .2, -1.),
    ("friction", -.2, -.1), ("friction", -.2, 1.),
    ("friction", 0., .001), ("friction", 0., 1.),
    ("friction", .2, .1), ("friction", .2, -1.),
])
def test_native_scalar_line_uses_source_regularizer_pair(kind, start, direction, native_library):
  import torch
  from test_primal_hessian_precision import _dense_ccj_record
  model, data, loss = _scalar_source(kind)
  R = float(data.efc_R[0])
  high, low = _split(R)
  alpha = np.asarray([.375, 2e-8], np.float32)
  r, rlow = _split(start)
  d, dlow = _split(direction)
  values = np.zeros(22, np.float32)
  values[:10] = [alpha[0],alpha[1],r,rlow,d,dlow,high,low,loss,kind=="equality"]
  values[11] = 1.
  values[12:14] = [-loss,loss] if kind == "friction" else [0.,np.inf]
  dims = np.zeros(64, np.int32); dims[24] = 32
  f32 = lambda value: torch.tensor(value,dtype=torch.float32,device="mps")
  output = torch.zeros(6,device="mps")
  native_library.line_coefficient_scalar_probe(
      f32(values),f32(_dense_ccj_record(np.ones((1,1)))),
      torch.tensor(dims,device="mps"),torch.zeros(64,device="mps"),
      torch.tensor([1,0,-1,0],dtype=torch.int32,device="mps"),
      output,torch.zeros(2,device="mps"),threads=(1,),group_size=(1,))
  a = float(alpha[0])+float(alpha[1])
  residual, delta = float(r)+float(rlow), float(d)+float(dlow)
  before, _, _ = _scalar_cost(model,data,residual)
  after, slope, curvature = _scalar_cost(model,data,residual+a*delta)
  expected = [after-before+.5*a*a*delta*delta,
              slope*delta+a*delta*delta,(curvature+1)*delta*delta]
  result = output.cpu().numpy().astype(float).reshape(3,2).sum(axis=1)
  np.testing.assert_allclose(result,expected,rtol=2e-12,atol=2e-12)


@pytest.mark.gpu
@_GPU
@pytest.mark.parametrize("dim", [3,4,6])
@pytest.mark.parametrize("zone,alpha", [("middle",.375),("top_to_middle",.7),
    ("middle_to_bottom",.8),("bottom",.3),("zero_tangent",.5)])
def test_native_cone_line_uses_source_coefficient_pairs(dim,zone,alpha,native_library):
  import torch
  from test_primal_elliptic_line_pair_014 import _case, _source_cost, _source_derivatives
  model = mujoco.MjModel.from_xml_string(f'''<mujoco>
    <option cone="elliptic" impratio="1.3"/>
    <default><geom condim="{dim}" friction=".73 .41 .17"/></default>
    <worldbody><geom type="plane" size="2 2 .1"/>
    <body pos="0 0 .09"><freejoint/><geom type="sphere" size=".1" mass="1.3"/>
    </body></worldbody></mujoco>''')
  data = mujoco.MjData(model); mujoco.mj_forward(model,data)
  assert data.ncon == 1 and data.nefc == dim
  R = np.ones(6); R[:dim] = data.efc_R
  friction = np.ones(5); friction[:dim-1] = data.contact[0].friction[:dim-1]
  Rh,Rl = _split(R); fh,fl = _split(friction)
  assert np.any(Rl != 0) and np.any(fl != 0)
  r,rl,d,dl,_,_,ap = _case(dim,zone,alpha)
  values = np.concatenate((r,rl,d,dl,Rh,fh,ap,[dim],Rl,fl)).astype(np.float32)
  output = torch.zeros(6,device="mps")
  native_library.line_coefficient_cone_probe(
      torch.tensor(values,device="mps"),output,threads=(1,),group_size=(1,))
  start, direction = r.astype(float)+rl, d.astype(float)+dl
  a = float(ap[0])+float(ap[1])
  sourceR, sourceF = Rh.astype(float)+Rl, fh.astype(float)+fl
  expected = [_source_cost(a,start,direction,sourceR,sourceF,dim),
              *_source_derivatives(a,start,direction,sourceR,sourceF,dim)]
  # Verify the cost against the pinned C callback, independently of the formula.
  costs = []
  for jar in (start,start+a*direction):
    cost = np.zeros(1)
    mujoco.mj_constraintUpdate(model,data,jar[:dim],cost,1)
    costs.append(float(cost[0]))
  np.testing.assert_allclose(expected[0],costs[1]-costs[0],rtol=2e-12,atol=2e-12)
  actual = output.cpu().numpy().astype(float).reshape(3,2).sum(axis=1)
  np.testing.assert_allclose(actual,expected,rtol=2e-12,atol=2e-12)
