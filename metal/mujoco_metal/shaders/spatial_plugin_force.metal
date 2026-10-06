#include <metal_stdlib>
using namespace metal;

inline float3 load3(device const float* x, uint i) {
  return float3(x[3 * i], x[3 * i + 1], x[3 * i + 2]);
}

inline float3 load_body_motion(device const float* cvel, uint base) {
  return float3(cvel[base + 3], cvel[base + 4], cvel[base + 5]);
}

inline float3 load_body_angular(device const float* cvel, uint base) {
  return float3(cvel[base], cvel[base + 1], cvel[base + 2]);
}

inline float3 load_cdof_part(device const float* cdof, uint base) {
  return float3(cdof[base], cdof[base + 1], cdof[base + 2]);
}

inline float3 load_cdof_linear(device const float* cdof, uint base) {
  return float3(cdof[base + 3], cdof[base + 4], cdof[base + 5]);
}

inline float3 point_velocity(device const float* cvel,
                             device const float* root_com,
                             uint batch, uint nbody, uint body, uint root,
                             float3 point) {
  uint body_base = (batch * nbody + body) * 6;
  float3 angular = load_body_angular(cvel, body_base);
  float3 offset = point - load3(root_com, batch * nbody + root);
  return load_body_motion(cvel, body_base) + cross(angular, offset);
}

inline float generalized_force(device const float* chain,
                               device const float* cdof,
                               uint batch, uint nv, uint nbody,
                               uint body, float3 point, float3 force,
                               device const float* root_com,
                               device const int* roots, uint dof) {
  if (chain[body * nv + dof] == 0.0f) return 0.0f;
  uint index = (batch * nv + dof) * 6;
  float3 angular = load_cdof_part(cdof, index);
  float3 linear = load_cdof_linear(cdof, index);
  float3 offset = point - load3(root_com, batch * nbody + roots[body]);
  return dot(linear + cross(angular, offset), force);
}

kernel void masked_magnetic_force(
    device const float* cvel [[buffer(0)]],
    device const float* root_com [[buffer(1)]],
    device const float* cdof [[buffer(2)]],
    device const float* inertial_pos [[buffer(3)]],
    device const float* chain [[buffer(4)]],
    device const int* roots [[buffer(5)]],
    device const int* bodies [[buffer(6)]],
    device const float* field [[buffer(7)]],
    device float* qfrc [[buffer(8)]],
    device const int* dims [[buffer(9)]],
    device const int* world_mask [[buffer(10)]],
    device const float* charge [[buffer(11)]],
    uint world [[thread_position_in_grid]]) {
  uint batch = uint(dims[0]);
  if (world >= batch || world_mask[world] == 0) return;
  uint nv = uint(dims[1]);
  uint nbody = uint(dims[2]);
  uint body_count = uint(dims[3]);
  for (uint dof = 0; dof < nv; ++dof) qfrc[world * nv + dof] = 0.0f;
  float3 magnetic = float3(field[0], field[1], field[2]);
  for (uint k = 0; k < body_count; ++k) {
    uint body = uint(bodies[k]);
    float3 point = load3(inertial_pos, world * nbody + body);
    float3 velocity = point_velocity(cvel, root_com, world, nbody, body,
                                     uint(roots[body]), point);
    float3 force = charge[0] * cross(velocity, magnetic);
    for (uint dof = 0; dof < nv; ++dof) {
      qfrc[world * nv + dof] += generalized_force(
          chain, cdof, world, nv, nbody, body, point, force, root_com, roots,
          dof);
    }
  }
}

kernel void masked_site_feedback_force(
    device const float* cvel [[buffer(0)]],
    device const float* root_com [[buffer(1)]],
    device const float* cdof [[buffer(2)]],
    device const float* site_pos [[buffer(3)]],
    device const float* chain [[buffer(4)]],
    device const int* roots [[buffer(5)]],
    device const float* time [[buffer(6)]],
    device const float* center [[buffer(7)]],
    device const float* amplitude [[buffer(8)]],
    device const float* frequency [[buffer(9)]],
    device const float* phase [[buffer(10)]],
    device float* qfrc [[buffer(11)]],
    device const int* dims [[buffer(12)]],
    device const int* world_mask [[buffer(13)]],
    device const float* gains [[buffer(14)]],
    uint world [[thread_position_in_grid]]) {
  uint batch = uint(dims[0]);
  if (world >= batch || world_mask[world] == 0) return;
  uint nv = uint(dims[1]);
  uint nbody = uint(dims[2]);
  uint nsite = uint(dims[3]);
  uint site = uint(dims[4]);
  uint body = uint(dims[5]);
  uint root = uint(roots[body]);
  float3 point = load3(site_pos, world * nsite + site);
  float3 velocity = point_velocity(cvel, root_com, world, nbody, body,
                                   root, point);
  float3 target;
  float3 target_velocity;
  for (uint axis = 0; axis < 3; ++axis) {
    float angle = time[world] * frequency[axis] + phase[axis];
    target[axis] = center[axis] + amplitude[axis] * sin(angle);
    target_velocity[axis] = amplitude[axis] * frequency[axis] * cos(angle);
  }
  float3 force = gains[0] * (target - point)
      + gains[1] * (target_velocity - velocity);
  for (uint dof = 0; dof < nv; ++dof) {
    qfrc[world * nv + dof] = generalized_force(
        chain, cdof, world, nv, nbody, body, point, force, root_com, roots,
        dof);
  }
}
