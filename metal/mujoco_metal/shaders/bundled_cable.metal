// MuJoCo 3.10.0 mujoco.elasticity.cable passive generalized force.
#include <metal_stdlib>
using namespace metal;

inline float4 cable_mul(float4 a, float4 b) {
  return float4(a.x*b.x-a.y*b.y-a.z*b.z-a.w*b.w,
                a.x*b.y+a.y*b.x+a.z*b.w-a.w*b.z,
                a.x*b.z-a.y*b.w+a.z*b.x+a.w*b.y,
                a.x*b.w+a.y*b.z-a.z*b.y+a.w*b.x);
}

inline float3 cable_rotate(float4 q, float3 v) {
  float3 qv = q.yzw;
  float3 t = 2.0f * cross(qv, v);
  return v + q.x * t + cross(qv, t);
}

inline float3 cable_quat_velocity(float4 q) {
  float3 axis = q.yzw;
  float sine = length(axis);
  if (sine > 0.0f) axis /= sine;
  float speed = 2.0f * atan2(sine, q.x);
  if (speed > M_PI_F) speed -= 2.0f * M_PI_F;
  return axis * speed;
}

inline float3 cable_local_stress(
    int segment, bool pullback,
    device const float* qpos,
    device const float* stiffness,
    device const float* omega0,
    device const float* local_quat,
    device const int* qadr) {
  uint s = uint(segment);
  float4 qlocal = float4(local_quat[4*s], local_quat[4*s+1],
                         local_quat[4*s+2], local_quat[4*s+3]);
  int adr = qadr[s];
  float4 qjoint = float4(qpos[adr], qpos[adr+1], qpos[adr+2], qpos[adr+3]);
  float4 qrel = cable_mul(qlocal, qjoint);
  float3 omega = cable_quat_velocity(qrel);
  float length0 = stiffness[4*s+3];
  if (!(length0 > 0.0f)) return float3(0.0f);
  float3 reference = float3(omega0[3*s], omega0[3*s+1], omega0[3*s+2]);
  float3 k = float3(stiffness[4*s], stiffness[4*s+1], stiffness[4*s+2]);
  float3 stress = -k * (omega - reference) / length0;
  if (pullback) {
    qrel.yzw = -qrel.yzw;
    stress = cable_rotate(qrel, stress);
  }
  return stress;
}

kernel void bundled_cable_force(
    device const float* qpos [[buffer(0)]],
    device const float* world_quat [[buffer(1)]],
    device const float* cdof [[buffer(2)]],
    device const int* body_ids [[buffer(3)]],
    device const int* previous [[buffer(4)]],
    device const int* following [[buffer(5)]],
    device const int* qadr [[buffer(6)]],
    device const float* stiffness [[buffer(7)]],
    device const float* omega0 [[buffer(8)]],
    device const float* local_quat [[buffer(9)]],
    device const int* body_parent [[buffer(10)]],
    device const int* dof_body [[buffer(11)]],
    device float* force [[buffer(12)]],
    device float* capture_force [[buffer(13)]],
    constant int* dims [[buffer(14)]],
    constant int* flags [[buffer(15)]],
    uint index [[thread_position_in_grid]]) {
  int batch = dims[0], nv = dims[1], nbody = dims[2];
  int nsegment = dims[3], nq = dims[4];
  uint total = uint(batch) * uint(nv);
  if (index >= total) return;
  int world = int(index / uint(nv));
  if (dims[5 + world] == 0) return;
  int dof = int(index - uint(world) * uint(nv));
  uint qbase = uint(world) * uint(nbody) * 4u;
  uint cbase = uint(world) * uint(nv) * 6u;
  float value = 0.0f;
  for (int s = 0; s < nsegment; ++s) {
    if (stiffness[4*uint(s)] == 0.0f &&
        stiffness[4*uint(s)+1] == 0.0f &&
        stiffness[4*uint(s)+2] == 0.0f) continue;
    int body = body_ids[s];
    if (previous[s] < 0 && following[s] < 0) continue;
    float3 local_force = float3(0.0f);
    if (previous[s] >= 0)
      local_force += cable_local_stress(s, true, qpos + uint(world)*uint(nq),
                                        stiffness, omega0, local_quat, qadr);
    if (following[s] >= 0)
      local_force -= cable_local_stress(s + 1, false,
                                        qpos + uint(world)*uint(nq),
                                        stiffness, omega0, local_quat, qadr);
    float4 body_rotation = float4(world_quat[qbase + uint(body)*4u],
        world_quat[qbase + uint(body)*4u+1], world_quat[qbase + uint(body)*4u+2],
        world_quat[qbase + uint(body)*4u+3]);
    float3 torque = cable_rotate(body_rotation, local_force);
    int ancestor = body;
    bool attached = false;
    while (ancestor > 0) {
      if (dof_body[dof] == ancestor) { attached = true; break; }
      ancestor = body_parent[ancestor];
    }
    if (attached) {
      uint addr = cbase + uint(dof)*6u;
      value += cdof[addr] * torque.x + cdof[addr+1] * torque.y
             + cdof[addr+2] * torque.z;
    }
  }
  force[index] += value;
  if (flags[0]) capture_force[index] += value;
}
