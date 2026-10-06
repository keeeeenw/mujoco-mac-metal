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

inline float3 cross_bias(float3 a, float3 b) { return cross(a, b); }
inline float2 bias_dd_add(float2 acc, float value) {
  float sum = acc.x + value;
  float value_part = sum - acc.x;
  float error = (acc.x - (sum - value_part)) + (value - value_part);
  float tail = acc.y + error;
  float hi = sum + tail;
  float lo = tail - (hi - sum);
  return float2(hi, lo);
}
inline float2 bias_dd_add_product(float2 acc, float a, float b) {
  float product = a * b;
  float product_error = fma(a, b, -product);
  return bias_dd_add(bias_dd_add(acc, product), product_error);
}
inline float2 bias_dd_add_words(float2 left, float2 right) {
  float sum = left.x + right.x;
  float right_part = sum - left.x;
  float error = (left.x - (sum - right_part)) + (right.x - right_part);
  float tail = error + left.y + right.y;
  float hi = sum + tail;
  return float2(hi, tail - (hi - sum));
}
inline float2 bias_dd_add_word_product(float2 acc, float2 a, float2 b) {
  float product = a.x * b.x;
  float error = fma(a.x, b.x, -product);
  error += a.x * b.y + a.y * b.x + a.y * b.y;
  return bias_dd_add_words(acc, float2(product, error));
}
inline float bias_dd_residual(float2 value, float legacy) {
  float2 diff = bias_dd_add_words(value, float2(-legacy, 0.0f));
  return diff.x + diff.y;
}
inline float2 bias_dd_cross_component(float2 ax, float2 ay,
                                      float2 bx, float2 by) {
  float2 result = bias_dd_add_word_product(float2(0.0f), ax, by);
  return bias_dd_add_word_product(result, float2(-ay.x, -ay.y), bx);
}
inline void bias_dd_cross3(thread const float2* a, thread const float2* b,
                           thread float2* result) {
  result[0] = bias_dd_cross_component(a[1], a[2], b[1], b[2]);
  result[1] = bias_dd_cross_component(a[2], a[0], b[2], b[0]);
  result[2] = bias_dd_cross_component(a[0], a[1], b[0], b[1]);
}
inline void bias_dd_motion_cross(thread const float2* a, device const float* b,
                                 thread float2* result) {
  float2 aw[3], av[3], bw[3], bv[3];
  for (uint k=0; k<3; ++k) {
    aw[k]=a[k]; av[k]=a[k+3];
    bw[k]=float2(b[k], 0.0f); bv[k]=float2(b[k+3], 0.0f);
  }
  float2 angular[3], linear_a[3], linear_b[3];
  bias_dd_cross3(aw, bw, angular);
  bias_dd_cross3(aw, bv, linear_a);
  bias_dd_cross3(av, bw, linear_b);
  for (uint k=0; k<3; ++k) {
    result[k]=angular[k];
    result[k+3]=bias_dd_add_words(linear_a[k], linear_b[k]);
  }
}
inline void bias_dd_multiply_spatial(device const float* matrix, uint offset,
                                     thread const float2* vector,
                                     thread float2* result) {
  for (uint row=0; row<6; ++row) {
    float2 value=float2(0.0f);
    for (uint col=0; col<6; ++col)
      value=bias_dd_add_word_product(
          value, float2(matrix[offset+row*6+col], 0.0f), vector[col]);
    result[row]=value;
  }
}
inline void bias_dd_force_cross(thread const float2* motion,
                                thread const float2* force,
                                thread float2* result) {
  float2 w[3], v[3], tau[3], f[3];
  for (uint k=0; k<3; ++k) {
    w[k]=motion[k]; v[k]=motion[k+3];
    tau[k]=force[k]; f[k]=force[k+3];
  }
  float2 torque_w[3], torque_v[3], force_out[3];
  bias_dd_cross3(w, tau, torque_w);
  bias_dd_cross3(v, f, torque_v);
  bias_dd_cross3(w, f, force_out);
  for (uint k=0; k<3; ++k) {
    result[k]=bias_dd_add_words(torque_w[k], torque_v[k]);
    result[k+3]=force_out[k];
  }
}
inline void multiply_spatial(device const float* matrix, uint offset,
                            device const float* vector, thread float* result) {
  for (uint row=0; row<6; ++row) {
    result[row] = 0.0f;
    for (uint col=0; col<6; ++col)
      result[row] += matrix[offset+row*6+col]*vector[col];
  }
}
inline void motion_cross(thread const float* a, device const float* b,
                         thread float* result) {
  float3 aw = float3(a[0],a[1],a[2]);
  float3 av = float3(a[3],a[4],a[5]);
  float3 bw = float3(b[0],b[1],b[2]);
  float3 bv = float3(b[3],b[4],b[5]);
  float3 angular = cross_bias(aw,bw);
  float3 linear = cross_bias(aw,bv)+cross_bias(av,bw);
  for (uint k=0;k<3;++k) { result[k]=angular[k]; result[k+3]=linear[k]; }
}
inline void force_cross(device const float* motion, thread const float* force,
                        thread float* result) {
  float3 w = float3(motion[0],motion[1],motion[2]);
  float3 v = float3(motion[3],motion[4],motion[5]);
  float3 tau = float3(force[0],force[1],force[2]);
  float3 f = float3(force[3],force[4],force[5]);
  float3 torque = cross_bias(w,tau)+cross_bias(v,f);
  float3 out_force = cross_bias(w,f);
  for (uint k=0;k<3;++k) { result[k]=torque[k]; result[k+3]=out_force[k]; }
}

// Newton-Euler bias pass over the same generic model mappings and local
// inertias produced by the dense mass kernel. This is inertial/gravity bias;
// passive, actuator, contact, and constraint forces remain unsupported.
kernel void smooth_bias(
    device const int* parent [[buffer(0)]],
    device const int* body_dofadr [[buffer(1)]],
    device const int* body_dofnum [[buffer(2)]],
    device const int* body_jntadr [[buffer(3)]],
    device const int* body_jntnum [[buffer(4)]],
    device const int* dof_bodyid [[buffer(5)]],
    device const int* jnt_type [[buffer(6)]],
    device const int* jnt_dofadr [[buffer(7)]],
    device const float* gravity [[buffer(8)]],
    device const float* cdof [[buffer(9)]],
    device const float* local_inertia [[buffer(10)]],
    device const int* disableflags [[buffer(11)]],
    device const float* qvel [[buffer(12)]],
    device float* cvel [[buffer(13)]],
    device float* cdof_dot [[buffer(14)]],
    device float* cacc [[buffer(15)]],
    device float* body_force [[buffer(16)]],
    device float* qfrc_bias [[buffer(17)]],
    device float* qfrc_bias_low [[buffer(18)]],
    device float* cvel_low [[buffer(19)]],
    device float* cdof_dot_low [[buffer(20)]],
    device float* cacc_low [[buffer(21)]],
    device float* body_force_low [[buffer(22)]],
    constant uint* dims [[buffer(23)]],
    device const int* body_ids [[buffer(24)]],
    device const int* parent_ids [[buffer(25)]],
    device const int* dof_ids [[buffer(26)]],
    device const int* body_awake_mask [[buffer(27)]],
    device const int* awake_counts [[buffer(28)]],
    constant int* awake_dims [[buffer(29)]],
    device const int* world_mask [[buffer(30)]],
    uint world [[thread_position_in_grid]]) {
  uint nbody=dims[0], nv=dims[2], batch=dims[3];
  if (world>=batch || world_mask[world] == 0) return;
  uint vb=world*nv;
  uint body6=world*nbody*6;
  uint dof6=world*nv*6;
  uint inert=world*nbody*36;
  uint body_stride=uint(max(awake_dims[0], 1));
  uint dof_stride=uint(max(awake_dims[1], 1));
  uint body_list_base=world*body_stride;
  uint dof_list_base=world*dof_stride;
  for (uint i=0; i<nv; ++i) qfrc_bias_low[vb+i]=0.0f;
  uint nbody_awake=uint(awake_counts[world*3]);
  uint nparent_awake=uint(awake_counts[world*3+1]);
  uint nv_awake=uint(awake_counts[world*3+2]);

  for (uint k=0;k<6;++k) {
    cvel[body6+k]=0.0f;
    cvel_low[body6+k]=0.0f;
  }

  // MuJoCo comVel grouping: free translation first, then the free/ball
  // angular triplet; scalar joints compute their derivative before update.
  for (uint bi=0;bi<nbody_awake;++bi) {
    uint b=uint(body_ids[body_list_base+bi]);
    if (b==0) continue;
    int p=parent[b];
    float velocity[6];
    float2 velocity_dd[6];
    for (uint k=0;k<6;++k) {
      velocity[k]=cvel[body6+uint(p)*6+k];
      velocity_dd[k]=float2(velocity[k], cvel_low[body6+uint(p)*6+k]);
    }
    int first=body_jntadr[b];
    int count=body_jntnum[b];
    uint body_dof_first=uint(body_dofadr[b]);
    for (int di=0;di<body_dofnum[b];++di)
      for (uint k=0;k<6;++k) {
        cdof_dot[dof6+(body_dof_first+uint(di))*6+k]=0.0f;
        cdof_dot_low[dof6+(body_dof_first+uint(di))*6+k]=0.0f;
      }
    for (int jj=0;jj<count;++jj) {
      uint j=uint(first+jj);
      uint d=uint(jnt_dofadr[j]);
      uint typ=uint(jnt_type[j]);
      if (typ==0 || typ==1) {
        uint skip=typ==0 ? 3 : 0;
        if (typ==0) {
          for (uint k=0;k<3;++k) {
            for (uint r=0;r<6;++r)
              velocity[r]+=cdof[dof6+(d+k)*6+r]*qvel[vb+d+k];
            for (uint r=0;r<6;++r)
              velocity_dd[r]=bias_dd_add_product(
                  velocity_dd[r], cdof[dof6+(d+k)*6+r], qvel[vb+d+k]);
          }
        }
        float derivatives[18];
        for (uint k=0;k<3;++k)
          motion_cross(velocity,cdof+dof6+(d+skip+k)*6,derivatives+k*6);
        for (uint k=0;k<3;++k) {
          float2 derivative_dd[6];
          bias_dd_motion_cross(velocity_dd, cdof+dof6+(d+skip+k)*6,
                               derivative_dd);
          for (uint r=0;r<6;++r)
            cdof_dot[dof6+(d+skip+k)*6+r]=derivatives[k*6+r];
          for (uint r=0;r<6;++r)
            cdof_dot_low[dof6+(d+skip+k)*6+r]=bias_dd_residual(
                derivative_dd[r], derivatives[k*6+r]);
        }
        for (uint k=0;k<3;++k) {
          for (uint r=0;r<6;++r)
            velocity[r]+=cdof[dof6+(d+skip+k)*6+r]*qvel[vb+d+skip+k];
          for (uint r=0;r<6;++r)
            velocity_dd[r]=bias_dd_add_product(
                velocity_dd[r], cdof[dof6+(d+skip+k)*6+r],
                qvel[vb+d+skip+k]);
        }
      } else {
        float derivative[6];
        motion_cross(velocity,cdof+dof6+d*6,derivative);
        float2 derivative_dd[6];
        bias_dd_motion_cross(velocity_dd, cdof+dof6+d*6, derivative_dd);
        for (uint r=0;r<6;++r) {
          cdof_dot[dof6+d*6+r]=derivative[r];
          cdof_dot_low[dof6+d*6+r]=bias_dd_residual(
              derivative_dd[r], derivative[r]);
          velocity[r]+=cdof[dof6+d*6+r]*qvel[vb+d];
          velocity_dd[r]=bias_dd_add_product(
              velocity_dd[r], cdof[dof6+d*6+r], qvel[vb+d]);
        }
      }
    }
    for (uint k=0;k<6;++k) {
      cvel[body6+b*6+k]=velocity[k];
      cvel_low[body6+b*6+k]=bias_dd_residual(velocity_dd[k], velocity[k]);
    }
  }

  // RNE at zero generalized acceleration. World acceleration is -gravity.
  for (uint k=0;k<6;++k) {
    cacc[body6+k]=0.0f;
    cacc_low[body6+k]=0.0f;
    body_force[body6+k]=0.0f;
    body_force_low[body6+k]=0.0f;
  }
  if ((disableflags[0] & 128)==0) {
    cacc[body6+3]=-gravity[0];
    cacc[body6+4]=-gravity[1];
    cacc[body6+5]=-gravity[2];
  }
  for (uint bi=0;bi<nbody_awake;++bi) {
    uint b=uint(body_ids[body_list_base+bi]);
    if (b==0) continue;
    uint p=uint(parent[b]);
    uint first=uint(body_dofadr[b]);
    uint count=uint(body_dofnum[b]);
    for (uint k=0;k<6;++k) {
      float acceleration=cacc[body6+p*6+k];
      float2 acceleration_dd=float2(
          acceleration, cacc_low[body6+p*6+k]);
      for (uint d=0;d<count;++d) {
        acceleration+=cdof_dot[dof6+(first+d)*6+k]*qvel[vb+first+d];
        acceleration_dd=bias_dd_add_word_product(
            acceleration_dd,
            float2(cdof_dot[dof6+(first+d)*6+k],
                   cdof_dot_low[dof6+(first+d)*6+k]),
            float2(qvel[vb+first+d], 0.0f));
      }
      cacc[body6+b*6+k]=acceleration;
      cacc_low[body6+b*6+k]=bias_dd_residual(acceleration_dd, acceleration);
    }
    float inertial_acc[6], inertial_vel[6], coriolis[6];
    multiply_spatial(local_inertia,inert+b*36,cacc+body6+b*6,inertial_acc);
    multiply_spatial(local_inertia,inert+b*36,cvel+body6+b*6,inertial_vel);
    force_cross(cvel+body6+b*6,inertial_vel,coriolis);
    float2 acceleration_words[6], velocity_words[6], inertial_acc_words[6];
    float2 inertial_vel_words[6], coriolis_words[6], force_words[6];
    for (uint k=0;k<6;++k) {
      acceleration_words[k]=float2(cacc[body6+b*6+k],
                                    cacc_low[body6+b*6+k]);
      velocity_words[k]=float2(cvel[body6+b*6+k],
                               cvel_low[body6+b*6+k]);
    }
    bias_dd_multiply_spatial(local_inertia, inert+b*36,
                             acceleration_words, inertial_acc_words);
    bias_dd_multiply_spatial(local_inertia, inert+b*36,
                             velocity_words, inertial_vel_words);
    bias_dd_force_cross(velocity_words, inertial_vel_words, coriolis_words);
    for (uint k=0;k<6;++k) {
      body_force[body6+b*6+k]=inertial_acc[k]+coriolis[k];
      force_words[k]=bias_dd_add_words(inertial_acc_words[k],
                                       coriolis_words[k]);
      body_force_low[body6+b*6+k]=bias_dd_residual(
          force_words[k], body_force[body6+b*6+k]);
    }
  }
  // Match pinned mj_rne: traverse parent_awake_ind verbatim. This list may
  // include a sleeping child when its parent is awake; do not add an
  // independent body-awake mask to this reverse accumulation.
  for (int pi=int(nparent_awake)-1;pi>=0;--pi) {
    int b=parent_ids[body_list_base+uint(pi)];
    int p=parent[b];
    if (p>0)
      for (uint k=0;k<6;++k) {
        float parent_high=body_force[body6+uint(p)*6+k];
        float parent_low=body_force_low[body6+uint(p)*6+k];
        body_force[body6+uint(p)*6+k]=
            parent_high+body_force[body6+uint(b)*6+k];
        float2 parent_force=float2(parent_high,
                                   parent_low);
        float2 child_force=float2(body_force[body6+uint(b)*6+k],
                                  body_force_low[body6+uint(b)*6+k]);
        body_force_low[body6+uint(p)*6+k]=bias_dd_residual(
            bias_dd_add_words(parent_force, child_force),
            body_force[body6+uint(p)*6+k]);
      }
  }
  for (uint di=0;di<nv_awake;++di) {
    uint d=uint(dof_ids[dof_list_base+di]);
    uint body=uint(dof_bodyid[d]);
    float value=0.0f;
    float2 compensated=float2(0.0f,0.0f);
    for (uint k=0;k<6;++k) {
      float axis=cdof[dof6+d*6+k];
      float force=body_force[body6+body*6+k];
      value+=axis*force;
      compensated=bias_dd_add_product(compensated,axis,force);
      compensated=bias_dd_add_product(
          compensated, axis, body_force_low[body6+body*6+k]);
    }
    qfrc_bias[vb+d]=value;
    // Store the low word relative to the legacy high word so ordinary
    // callers observe the exact same qfrc_bias tensor as before.
    qfrc_bias_low[vb+d]=(compensated.x-value)+compensated.y;
  }
}

// Analytical RNE velocity derivative. Each (world,column) owns one reusable
// device scratch slice; only spatial six-vectors remain in thread-local memory.
// Position, inertial data and cdof are held fixed, as in mjd_rne_vel.
kernel void smooth_bias_derivative(
    device const int* parent [[buffer(0)]],
    device const int* body_dofadr [[buffer(1)]],
    device const int* body_dofnum [[buffer(2)]],
    device const int* body_jntadr [[buffer(3)]],
    device const int* body_jntnum [[buffer(4)]],
    device const int* dof_bodyid [[buffer(5)]],
    device const int* jnt_type [[buffer(6)]],
    device const int* jnt_dofadr [[buffer(7)]],
    device const float* gravity [[buffer(8)]],
    device const float* cdof [[buffer(9)]],
    device const float* local_inertia [[buffer(10)]],
    device const int* disableflags [[buffer(11)]],
    device const float* qvel [[buffer(12)]],
    device float* qderiv_bias [[buffer(13)]],
    constant uint* dims [[buffer(14)]],
    device const float* primal_cvel [[buffer(15)]],
    device const float* primal_cdof_dot [[buffer(16)]],
    device float* scratch [[buffer(17)]],
    device const int* column_offsets [[buffer(18)]],
    device const int* edge_rows [[buffer(19)]],
    device const int* edge_slots [[buffer(20)]],
    uint2 tid [[thread_position_in_grid]]) {
  uint nbody=dims[0], nv=dims[2], batch=dims[3];
  uint edge_count=dims[4];
  bool sparse_output=dims[5] != 0;
  uint col=tid.x, world=tid.y;
  if (world>=batch || col>=nv) return;
  uint vb=world*nv, dof6=world*nv*6, body6=world*nbody*6;
  uint inert=world*nbody*36;
  uint stride=(3*nbody+nv)*6;
  device float* dv=scratch+(world*nv+col)*stride;
  device float* ddot=dv+nbody*6;
  device float* da=ddot+nv*6;
  device float* df=da+nbody*6;
  for (uint k=0;k<stride;++k) dv[k]=0.0f;

  // Differentiate comVel's joint grouping before each velocity increment.
  for (uint b=1;b<nbody;++b) {
    uint p=uint(parent[b]);
    float velocity[6];
    for (uint k=0;k<6;++k) velocity[k]=dv[p*6+k];
    int first=body_jntadr[b], count=body_jntnum[b];
    for (int jj=0;jj<count;++jj) {
      uint j=uint(first+jj), d=uint(jnt_dofadr[j]), typ=uint(jnt_type[j]);
      if (typ==0 || typ==1) {
        uint skip=typ==0 ? 3 : 0;
        if (typ==0 && col>=d && col<d+3)
          for (uint r=0;r<6;++r) velocity[r]+=cdof[dof6+col*6+r];
        // Ball/free angular triplet shares its pre-increment velocity.
        for (uint k=0;k<3;++k) {
          float derivative[6];
          motion_cross(velocity,cdof+dof6+(d+skip+k)*6,derivative);
          for (uint r=0;r<6;++r) ddot[(d+skip+k)*6+r]=derivative[r];
        }
        if (col>=d+skip && col<d+skip+3)
          for (uint r=0;r<6;++r) velocity[r]+=cdof[dof6+col*6+r];
      } else {
        float derivative[6];
        motion_cross(velocity,cdof+dof6+d*6,derivative);
        for (uint r=0;r<6;++r) {
          ddot[d*6+r]=derivative[r];
          if (col==d) velocity[r]+=cdof[dof6+d*6+r];
        }
      }
    }
    for (uint k=0;k<6;++k) dv[b*6+k]=velocity[k];
  }

  // D(cacc) = parent's derivative + D(cdof_dot)*v + cdof_dot*D(v).
  // D(I*a + v x* I*v) follows the bilinear spatial cross-force rule.
  for (uint b=1;b<nbody;++b) {
    uint p=uint(parent[b]), first=uint(body_dofadr[b]), count=uint(body_dofnum[b]);
    for (uint k=0;k<6;++k) {
      float value=da[p*6+k];
      for (uint d=first;d<first+count;++d) {
        value+=ddot[d*6+k]*qvel[vb+d];
        if (d==col) value+=primal_cdof_dot[dof6+d*6+k];
      }
      da[b*6+k]=value;
    }
    float ia[6], iv[6], idv[6], cross1[6], cross2[6];
    multiply_spatial(local_inertia,inert+b*36,da+b*6,ia);
    multiply_spatial(local_inertia,inert+b*36,primal_cvel+body6+b*6,iv);
    multiply_spatial(local_inertia,inert+b*36,dv+b*6,idv);
    force_cross(dv+b*6,iv,cross1);
    force_cross(primal_cvel+body6+b*6,idv,cross2);
    for (uint k=0;k<6;++k) df[b*6+k]=ia[k]+cross1[k]+cross2[k];
  }
  for (int b=int(nbody)-1;b>0;--b) {
    int p=parent[b];
    if (p>0) for (uint k=0;k<6;++k) df[uint(p)*6+k]+=df[uint(b)*6+k];
  }
  if (sparse_output) {
    for (int edge=column_offsets[col]; edge<column_offsets[col+1]; ++edge) {
      if (edge<0 || uint(edge)>=edge_count) continue;
      uint row=uint(edge_rows[edge]);
      if (row>=nv) continue;
      uint body=uint(dof_bodyid[row]);
      float value=0.0f;
      for (uint k=0;k<6;++k) value+=cdof[dof6+row*6+k]*df[body*6+k];
      qderiv_bias[world*edge_count+uint(edge_slots[edge])]+=-value;
    }
  } else {
    for (uint row=0;row<nv;++row) {
      uint body=uint(dof_bodyid[row]);
      float value=0.0f;
      for (uint k=0;k<6;++k) value+=cdof[dof6+row*6+k]*df[body*6+k];
      qderiv_bias[world*nv*nv+row*nv+col]=-value;
    }
  }
}
