// Copyright 2026 The MuJoCo Metal contributors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <metal_stdlib>
using namespace metal;

kernel void scalar_motor_force(
    device const float* ctrl [[buffer(0)]],
    device const int* dof [[buffer(1)]],
    device const float* gear [[buffer(2)]],
    device const float* gain [[buffer(3)]],
    device const int* ctrl_limited [[buffer(4)]],
    device const float* ctrl_range [[buffer(5)]],
    device const int* force_limited [[buffer(6)]],
    device const float* force_range [[buffer(7)]],
    device const int* actuator_group [[buffer(8)]],
    constant int* dims [[buffer(9)]],
    device float* qfrc [[buffer(10)]],
    uint world [[thread_position_in_grid]]) {
  int nu = dims[0];
  int nv = dims[1];
  int batch = dims[2];
  int actuation_disabled = dims[3];
  int clampctrl_disabled = dims[4];
  int disableactuator = dims[5];
  if (world >= uint(batch)) return;

  uint force_offset = world * uint(nv);
  for (int v = 0; v < nv; ++v) qfrc[force_offset + uint(v)] = 0.0f;
  if (actuation_disabled != 0) return;

  for (int actuator = 0; actuator < nu; ++actuator) {
    if (!isfinite(ctrl[world * uint(nu) + uint(actuator)])) {
      float invalid = as_type<float>(0x7fc00000u);
      for (int v = 0; v < nv; ++v) qfrc[force_offset + uint(v)] = invalid;
      return;
    }
  }

  for (int actuator = 0; actuator < nu; ++actuator) {
    int group = actuator_group[actuator];
    if ((disableactuator & (1 << group)) != 0) continue;
    float control = ctrl[world * uint(nu) + uint(actuator)];
    if (clampctrl_disabled == 0 && ctrl_limited[actuator] != 0) {
      float lower = ctrl_range[2 * uint(actuator)];
      float upper = ctrl_range[2 * uint(actuator) + 1];
      control = clamp(control, lower, upper);
    }
    float force = gain[actuator] * control;
    if (force_limited[actuator] != 0) {
      float lower = force_range[2 * uint(actuator)];
      float upper = force_range[2 * uint(actuator) + 1];
      force = clamp(force, lower, upper);
    }
    qfrc[force_offset + uint(dof[actuator])] += gear[actuator] * force;
  }
}
