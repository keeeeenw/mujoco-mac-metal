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

// Inspect exponent bits because shader compilation may enable fast-math.
inline bool finite_float(float value) {
  return (as_type<uint>(value) & 0x7f800000u) != 0x7f800000u;
}

inline float4 quat_multiply(float4 a, float4 b) {
  return float4(
      a.x * b.x - dot(a.yzw, b.yzw),
      a.x * b.yzw + b.x * a.yzw + cross(a.yzw, b.yzw));
}

inline float4 quat_normalize_stable(float4 q) {
  float scale = max(max(abs(q.x), abs(q.y)), max(abs(q.z), abs(q.w)));
  float4 scaled = q / scale;
  return scaled * rsqrt(dot(scaled, scaled));
}

inline float4 quat_increment(float3 omega, float dt, thread bool& ok) {
  float scale = max(max(abs(omega.x), abs(omega.y)), abs(omega.z));
  if (scale == 0.0f) return float4(1.0f, 0.0f, 0.0f, 0.0f);

  float3 scaled = omega / scale;
  float scaled_norm = sqrt(dot(scaled, scaled));
  float dt_scale = dt * scale;
  float angle = dt_scale * scaled_norm;
  if (!finite_float(dt_scale) || !finite_float(angle)) {
    ok = false;
    return float4(1.0f, 0.0f, 0.0f, 0.0f);
  }

  float4 increment;
  if (abs(angle) < 0.0001f) {
    // Taylor series avoids losing the vector part for very small rotations.
    float angle2 = angle * angle;
    float angle4 = angle2 * angle2;
    float scalar = 1.0f - angle2 * 0.125f + angle4 * 0.002604166666666667f;
    float vector_scale = 0.5f - angle2 * 0.020833333333333333f +
                         angle4 * 0.000260416666666667f;
    increment = float4(scalar, scaled * (dt_scale * vector_scale));
  } else {
    float half_angle = 0.5f * angle;
    float axis_scale = sin(half_angle) / scaled_norm;
    increment = float4(cos(half_angle), scaled * axis_scale);
  }
  for (uint i = 0; i < 4; ++i) {
    if (!finite_float(increment[i])) ok = false;
  }
  return increment;
}

kernel void semi_implicit_euler(
    device const float* qpos [[buffer(0)]],
    device const float* qvel [[buffer(1)]],
    device const float* qacc [[buffer(2)]],
    device const float* time [[buffer(3)]],
    device const int* solve_status [[buffer(4)]],
    device const int* joint_type [[buffer(5)]],
    device const int* joint_qposadr [[buffer(6)]],
    device const int* joint_dofadr [[buffer(7)]],
    device float* candidate_qpos [[buffer(8)]],
    device float* candidate_qvel [[buffer(9)]],
    device float* candidate_time [[buffer(10)]],
    device float* output_qpos [[buffer(11)]],
    device float* output_qvel [[buffer(12)]],
    device float* output_time [[buffer(13)]],
    device int* output_status [[buffer(14)]],
    constant int* dims [[buffer(15)]],
    constant float* timestep [[buffer(16)]],
    device const float* next_velocity [[buffer(17)]],
    device const float* position_velocity [[buffer(18)]],
    constant int* velocity_mode [[buffer(19)]],
    device const int* body_jntadr [[buffer(20)]],
    device const int* body_jntnum [[buffer(21)]],
    device const int* awake_body_ids [[buffer(22)]],
    device const int* awake_dof_ids [[buffer(23)]],
    device const int* awake_counts [[buffer(24)]],
    constant int* sleep_filter [[buffer(25)]],
    device const float* input_words [[buffer(26)]],
    device float* candidate_words [[buffer(27)]],
    device float* output_words [[buffer(28)]],
    device const float* qacc_low [[buffer(29)]],
    device const float* override_words [[buffer(30)]],
    uint world [[thread_position_in_grid]]) {
  uint nq = uint(dims[0]);
  uint nv = uint(dims[1]);
  uint njnt = uint(dims[2]);
  uint batch = uint(dims[3]);
  uint nbody = uint(dims[4]);
  if (world >= batch) return;

  uint qb = world * nq;
  uint vb = world * nv;
  uint width=2*nq+2*nv,pb=world*width;
  for(uint i=0;i<width;i++)output_words[pb+i]=input_words[pb+i];
  for (uint i = 0; i < nq; ++i) output_qpos[qb + i] = qpos[qb + i];
  bool filtering = sleep_filter[0] != 0;
  for (uint i = 0; i < nv; ++i)
    output_qvel[vb + i] = qvel[vb + i];
  output_time[world] = time[world];

  int prior_status = solve_status[world];
  if (prior_status != 0) {
    output_status[world] = prior_status;
    return;
  }

  // Reject any nonfinite starting value before writing a candidate.
  for (uint i = 0; i < nq; ++i) {
    if (!finite_float(qpos[qb + i]) || !finite_float(input_words[pb+i]) || !finite_float(input_words[pb+nq+i])) {
      output_status[world] = 10;
      return;
    }
  }
  for (uint i = 0; i < nv; ++i) {
    if (!finite_float(qvel[vb + i]) || !finite_float(qacc[vb + i]) ||
        !finite_float(qacc_low[vb+i]) || !finite_float(input_words[pb+2*nq+i]) ||
        !finite_float(input_words[pb+2*nq+nv+i]) ||
        ((velocity_mode[0] & 1) && (!finite_float(next_velocity[vb + i]) ||
          !finite_float(override_words[world*4*nv+i]) || !finite_float(override_words[world*4*nv+nv+i]))) ||
        ((velocity_mode[0] & 2) && (!finite_float(position_velocity[vb + i]) ||
          !finite_float(override_words[world*4*nv+2*nv+i]) || !finite_float(override_words[world*4*nv+3*nv+i])))) {
      output_status[world] = 10;
      return;
    }
  }
  if (!finite_float(time[world])) {
    output_status[world] = 10;
    return;
  }

  KinDD dt_words=KinDD{timestep[0],timestep[1],timestep[2]};
  float dt = timestep[0];
  for(uint i=0;i<width;i++)candidate_words[pb+i]=input_words[pb+i];
  if(filtering)for(uint i=2*nq;i<width;i++)candidate_words[pb+i]=0.0f;
  for (uint i = 0; i < nq; ++i) candidate_qpos[qb + i] = qpos[qb + i];
  for (uint i = 0; i < nv; ++i)
    candidate_qvel[vb + i] = filtering ? 0.0f : qvel[vb + i];
  bool valid = true;
  int awake_dof_count = filtering ? awake_counts[world*3+2] : int(nv);
  for (int k = 0; k < awake_dof_count; ++k) {
    int dof = filtering
        ? awake_dof_ids[int(world)*max(int(nv),1)+k] : k;
    if (dof < 0 || dof >= int(nv)) continue;
    uint i = uint(dof);
    KinDD previous=kin_state_load(qvel,input_words,vb+i,pb+2*nq+i,pb+2*nq+nv+i);
    KinDD velocity=(velocity_mode[0]&1)?KinDD{next_velocity[vb+i],
        override_words[world*4*nv+i],override_words[world*4*nv+nv+i]}:
        kin_fma(dt_words,KinDD{qacc[vb+i],qacc_low[vb+i],0.0f},previous);
    if (!finite_float(velocity.hi) || !finite_float(velocity.lo) || !finite_float(velocity.tail)) valid=false;
    kin_state_store(candidate_qvel,candidate_words,vb+i,pb+2*nq+i,pb+2*nq+nv+i,velocity);
  }
  float next_time = time[world] + dt;
  if (!finite_float(next_time)) valid = false;
  candidate_time[world] = next_time;
  if (!valid) {
    output_status[world] = 12;
    return;
  }

  // Position velocity may differ from the committed next velocity (midpoint).
  device const float* pos_velocity = (velocity_mode[0] & 2) ?
                                    position_velocity : candidate_qvel;
  // Apply position velocity using MuJoCo's joint address conventions.
  int awake_body_count = filtering ? awake_counts[world*3] : int(nbody-1);
  for (int bi = 0; bi < awake_body_count; ++bi) {
    int body = filtering
        ? awake_body_ids[int(world)*max(int(nbody),1)+bi] : bi+1;
    if (body <= 0 || body >= int(nbody)) continue;
    int first_joint = body_jntadr[body];
    int body_joint_count = body_jntnum[body];
    for (int jj = 0; jj < body_joint_count; ++jj) {
      int j = first_joint + jj;
      if (j < 0 || j >= int(njnt)) continue;
      int type = joint_type[j];
      uint pa = uint(joint_qposadr[j]);
      uint da = uint(joint_dofadr[j]);
      if (type == 0) {
        for (uint k=0;k<3;k++) {
          KinDD position=kin_state_load(candidate_qpos,candidate_words,qb+pa+k,pb+pa+k,pb+nq+pa+k);
          KinDD velocity=(velocity_mode[0]&2)?KinDD{pos_velocity[vb+da+k],
              override_words[world*4*nv+2*nv+da+k],override_words[world*4*nv+3*nv+da+k]}:
              kin_state_load(candidate_qvel,candidate_words,vb+da+k,pb+2*nq+da+k,pb+2*nq+nv+da+k);
          KinDD value=kin_fma(dt_words,velocity,position);
          if(!finite_float(value.hi)||!finite_float(value.lo)||!finite_float(value.tail))valid=false;
          kin_state_store(candidate_qpos,candidate_words,qb+pa+k,pb+pa+k,pb+nq+pa+k,value);
        }
        pa+=3;da+=3;
      }
      if(type==0 || type==1) {
        KinDD component[4];
        for(uint k=0;k<4;k++)component[k]=kin_state_load(candidate_qpos,candidate_words,qb+pa+k,pb+pa+k,pb+nq+pa+k);
        KinQ quaternion=KinQ{component[0],component[1],component[2],component[3]};
        if(kin_sign(quaternion.w)==0 && kin_sign(quaternion.x)==0 &&
           kin_sign(quaternion.y)==0 && kin_sign(quaternion.z)==0) {
          output_status[world]=11;return;
        }
        KinDD velocity[3];
        for(uint k=0;k<3;k++)velocity[k]=(velocity_mode[0]&2)?KinDD{pos_velocity[vb+da+k],
              override_words[world*4*nv+2*nv+da+k],override_words[world*4*nv+3*nv+da+k]}:
            kin_state_load(candidate_qvel,candidate_words,vb+da+k,pb+2*nq+da+k,pb+2*nq+nv+da+k);
        KinDD length=kin_sqrt(kin_fma(velocity[2],velocity[2],
            kin_fma(velocity[0],velocity[0],kin_mul(velocity[1],velocity[1]))));
        KinDD minval=KinDD{1.0000000036274937e-15f,-3.627493647833322e-24f,0};
        if(kin_compare(length,minval)<0) {velocity[0]=kin_dd(1);velocity[1]=velocity[2]=kin_dd(0);}
        else {
          KinDD inv=kin_div(kin_dd(1),length);
          for(uint k=0;k<3;k++)velocity[k]=kin_mul(velocity[k],inv);
        }
        KinDD half_angle=kin_scale(kin_mul(dt_words,length),0.5f);
        KinDD sine=kin_sincos(half_angle,true),cosine=kin_sincos(half_angle,false);
        KinQ increment=KinQ{cosine,kin_mul(velocity[0],sine),kin_mul(velocity[1],sine),kin_mul(velocity[2],sine)};
        // MuJoCo normalizes BEFORE multiplication; a second normalization
        // changes the represented position and must not be added afterwards.
        KinQ next=kinq_mul(kinq_norm(quaternion),increment);
        KinDD c[4]={next.w,next.x,next.y,next.z};
        for(uint k=0;k<4;k++) {
          if(!finite_float(c[k].hi)||!finite_float(c[k].lo)||!finite_float(c[k].tail))valid=false;
          kin_state_store(candidate_qpos,candidate_words,qb+pa+k,pb+pa+k,pb+nq+pa+k,c[k]);
        }
      } else if(type==2 || type==3) {
        KinDD position=kin_state_load(candidate_qpos,candidate_words,qb+pa,pb+pa,pb+nq+pa);
        KinDD velocity=(velocity_mode[0]&2)?KinDD{pos_velocity[vb+da],
            override_words[world*4*nv+2*nv+da],override_words[world*4*nv+3*nv+da]}:
            kin_state_load(candidate_qvel,candidate_words,vb+da,pb+2*nq+da,pb+2*nq+nv+da);
        KinDD value=kin_fma(dt_words,velocity,position);
        if(!finite_float(value.hi)||!finite_float(value.lo)||!finite_float(value.tail))valid=false;
        kin_state_store(candidate_qpos,candidate_words,qb+pa,pb+pa,pb+nq+pa,value);
      }
      if (!valid) {
        output_status[world] = 12;
        return;
      }
    }
  }

  // Commit only after every coordinate, quaternion, and time value is valid.
  for (uint i = 0; i < nq; ++i) output_qpos[qb + i] = candidate_qpos[qb + i];
  for (uint i = 0; i < nv; ++i) output_qvel[vb + i] = candidate_qvel[vb + i];
  for(uint i=0;i<width;i++)output_words[pb+i]=candidate_words[pb+i];
  output_time[world] = candidate_time[world];
  output_status[world] = 0;
}

// Retain the source RK tableau sum order, including its zero coefficients.
// Rates and accelerations are immutable captured stages, never live callbacks.
kernel void rk4_combine_rates(
    device const float* velocities [[buffer(0)]],device const float* velocity_words [[buffer(1)]],
    device const float* accelerations [[buffer(2)]],device const float* acceleration_low [[buffer(3)]],
    device const float* initial_velocity [[buffer(4)]],device const float* initial_words [[buffer(5)]],
    device float* rate [[buffer(6)]],device float* weighted_acc [[buffer(7)]],
    device float* weighted_acc_low [[buffer(8)]],device float* next_velocity [[buffer(9)]],
    device float* override_words [[buffer(10)]],constant int* dims [[buffer(11)]],
    constant float* timestep [[buffer(12)]],constant float* weights [[buffer(13)]],
    uint index [[thread_position_in_grid]]) {
  uint nq=dims[0],nv=dims[1],batch=dims[2];int stage=dims[3];
  if(index>=batch*nv)return;
  uint world=index/nv,dof=index%nv;
  KinDD v=kin_dd(0),a=kin_dd(0),dt=KinDD{timestep[0],timestep[1],timestep[2]};
  int count=stage<0?4:stage+1;
  for(int j=0;j<count;j++) {
    KinDD weight=stage<0 ? KinDD{weights[(j==0||j==3)?0:3],
                                 weights[(j==0||j==3)?1:4],weights[(j==0||j==3)?2:5]} :
        kin_dd(j==stage ? (stage<2?0.5f:1.0f) : 0.0f);
    uint hi=uint(j)*batch*nv+index,lo=(uint(j)*batch+world)*2*nv+dof;
    v=kin_fma(weight,KinDD{velocities[hi],velocity_words[lo],velocity_words[lo+nv]},v);
    a=kin_fma(weight,KinDD{accelerations[hi],acceleration_low[hi],0.0f},a);
  }
  uint input_offset=world*(2*nq+2*nv)+2*nq+dof;
  KinDD initial=KinDD{initial_velocity[index],initial_words[input_offset],initial_words[input_offset+nv]};
  KinDD next=kin_fma(dt,a,initial);
  kin_state_store(next_velocity,override_words,index,world*4*nv+dof,world*4*nv+nv+dof,next);
  kin_state_store(rate,override_words,index,world*4*nv+2*nv+dof,world*4*nv+3*nv+dof,v);
  weighted_acc[index]=kin_value(a);
  weighted_acc_low[index]=kin_value(kin_sub(a,kin_dd(weighted_acc[index])));
}
