# Copyright 2026 The MuJoCo Metal contributors
# Licensed under the Apache License, Version 2.0.
"""Near-axis source precision on the actual libccd cylinder support paths."""
import ctypes
import math
import os

import mujoco
import numpy as np
import pytest

from test_common_ccd_matrix_source import _words
from test_common_ccd_support_source import _oracle
from test_primitive_hfield_full_manifold import _model, _LIMITS

_QPOS = np.array([
    .017026007176190613, .02299647140048444, .10004666362003982,
    .9999999996310607, 2.633486949705959e-5,
    3.250161770676974e-6, 5.812879924282872e-6], np.float64)
_DIRECTION = np.array([
    -6.500629702741622e-6, 5.2669701189594364e-5,
    -.9999999985918223], np.float64)

pytestmark = [pytest.mark.gpu, pytest.mark.skipif(
    os.getenv('MUJOCO_METAL_RUN_GPU') != '1', reason='native opt-in')]


def test_source_radial_norm_preserves_tiny_squared_residuals():
  import torch
  from mujoco_metal.coupled_constraints import _coupled_shader_source
  system = ctypes.CDLL(None)
  system.fma.argtypes = [ctypes.c_double] * 3
  system.fma.restype = ctypes.c_double
  rng = np.random.default_rng(92344)
  pairs = rng.uniform(-1, 1, (144, 2)) * np.exp2(
      np.linspace(-60, 60, 144))[:, None]
  pairs = np.concatenate((pairs, [[0, 0], [0, 1e-15], [1e-15, 0],
      [-9.17474690531563e-16, 5.073360445755819e-16],
      [math.nextafter(1e-15, 0), 0], [math.nextafter(1e-15, math.inf), 0]]))
  expected = np.array([math.sqrt(system.fma(x, x, y*y)) for x, y in pairs])
  shader = _coupled_shader_source(libccd=True) + r"""
  kernel void radial_precision(device const float* input [[buffer(0)]],
      device float* output [[buffer(1)]],uint tid [[thread_position_in_grid]]) {
    int base=6*int(tid);FlexDD x={input[base],input[base+1],input[base+2]};
    FlexDD y={input[base+3],input[base+4],input[base+5]};
    FlexDD norm=flex_dd_source_radial_norm(x,y);
    output[3*tid]=norm.hi;output[3*tid+1]=norm.lo;output[3*tid+2]=norm.tail;
  }
  """
  library = torch.mps.compile_shader(shader)
  inputs = torch.tensor(np.stack([_words(pair) for pair in pairs]), device='mps')
  before = inputs.clone(); output = torch.zeros((len(pairs), 3), device='mps')
  library.radial_precision(inputs, output, threads=len(pairs))
  np.testing.assert_array_equal(output.cpu().double().numpy().sum(axis=-1), expected)
  torch.testing.assert_close(inputs.view(torch.int32), before.view(torch.int32), rtol=0, atol=0)


def test_both_libccd_support_adapters_match_installed_near_axis_callback(tmp_path):
  import torch
  from mujoco_metal.coupled_constraints import _coupled_shader_source
  model = _model('cylinder', '.25 .15', True)
  data = mujoco.MjData(model); data.qpos[:] = _QPOS; mujoco.mj_forward(model, data)
  directions = [_DIRECTION]
  for epsilon in [0., math.nextafter(1e-15, 0.), 1e-15,
                  math.nextafter(1e-15, math.inf), 2e-15, 1e-12, 1e-8]:
    for sign in [-1., 1.]:
      out = np.zeros(3)
      mujoco.mju_mulMatVec3(out, data.geom_xmat[1], np.array([epsilon, -epsilon/2, sign]))
      directions.append(out)
  directions = np.asarray(directions)
  expected = np.zeros_like(directions)
  _oracle(tmp_path, libccd=True).query(model._address, data._address, 1,
                                     directions, len(directions), expected)
  object_words = np.concatenate((_words(np.concatenate((data.geom_xpos[1],
      data.geom_xmat[1], model.geom_size[1], [0]))), np.zeros(36, np.float32)))
  shader = _coupled_shader_source(libccd=True) + r"""
  kernel void near_axis_support(device const int* meta [[buffer(0)]],
      device const float* object [[buffer(1)]],device const float* dirs [[buffer(2)]],
      device float* output [[buffer(3)]],uint tid [[thread_position_in_grid]]) {
    CommonCCDSupportObject o=common_ccd_load_object(meta,object);
    FlexDD3 direction=common_ccd_load_dd3(dirs,9*int(tid));
    common_ccd_store_dd3(output,18*int(tid),
        common_ccd_object_support(o,direction,object,meta).point);
    common_ccd_store_dd3(output,18*int(tid)+9,
        mpr_object_support(o,direction,object,meta).point);
  }
  """
  library = torch.mps.compile_shader(shader)
  meta = torch.tensor([7,1,5,-1,0,-1,-1,0,0], dtype=torch.int32, device='mps')
  inputs = [meta, torch.tensor(object_words, device='mps'),
            torch.tensor(np.stack([_words(row) for row in directions]), device='mps')]
  before = [value.clone() for value in inputs]
  output = torch.zeros((len(directions), 18), device='mps')
  library.near_axis_support(*inputs, output, threads=len(directions))
  actual = output.cpu().double().numpy().reshape(-1, 2, 3, 3).sum(axis=-1)
  for adapter in range(2): np.testing.assert_array_equal(actual[:, adapter], expected)
  for value, saved in zip(inputs, before):
    torch.testing.assert_close(value.view(torch.int32), saved.view(torch.int32), rtol=0, atol=0)


def test_frozen_terrain_witnesses_match_source_after_near_axis_support():
  import torch
  from mujoco_metal.simulation import MetalSimulation
  model = _model('cylinder', '.25 .15', True)
  data = mujoco.MjData(model); data.qpos[:] = _QPOS; mujoco.mj_forward(model, data)
  simulation = MetalSimulation(model, 2, profile='integrated_euler_v1', limits=_LIMITS)
  packed = _words(_QPOS).reshape(model.nq, 3)
  qpos = np.tile(packed[:, 0], (2, 1)); qpos[1, 2] = .8
  words = np.tile(np.concatenate((packed[:, 1], packed[:, 2])), (2, 1)); words[1] = 0
  poses = simulation._smooth._fk.run_device(torch.tensor(qpos, device='mps'),
      qpos_words=torch.tensor(words, device='mps'))
  cc = simulation._coupled_constraints
  cc.generate_candidates(poses, torch.zeros((2, model.nv), device='mps'))
  count = cc._workspace['contact_pair_count'].cpu().numpy()
  assert count.tolist() == [data.ncon, 0]
  assert data.ncon == 36
  records = cc._workspace['contact_pair_records'].cpu().double().numpy().reshape(
      2, cc.descriptor.ncontacts_max, 37)[0, :data.ncon]
  positions = records[:, 13:16]+records[:, 25:28]+records[:, 31:34]
  normals = records[:, 16:19]+records[:, 28:31]+records[:, 34:37]
  np.testing.assert_array_equal(positions, [contact.pos for contact in data.contact[:data.ncon]])
  np.testing.assert_allclose(normals, [contact.frame[:3] for contact in data.contact[:data.ncon]],
                             rtol=0, atol=3e-15)
