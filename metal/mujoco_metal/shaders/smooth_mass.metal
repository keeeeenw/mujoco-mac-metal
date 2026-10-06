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

// Device-only finite status for internally generated or explicitly trusted
// pose packets. The host validates the external pose boundary separately.
kernel void pose_finite_status(
    device const float* body_pos [[buffer(0)]],
    device const float* body_quat [[buffer(1)]],
    device const float* geom_pos [[buffer(2)]],
    device const float* geom_quat [[buffer(3)]],
    device const float* site_pos [[buffer(4)]],
    device const float* site_quat [[buffer(5)]],
    device const float* inertial_pos [[buffer(6)]],
    device const float* inertial_quat [[buffer(7)]],
    device const float* joint_anchor [[buffer(8)]],
    device const float* joint_axis [[buffer(9)]],
    device const float* geom_xmat [[buffer(10)]],
    device const float* geom_xmat_low [[buffer(11)]],
    device const float* geom_xmat_tail [[buffer(12)]],
    device const float* geom_quat_low [[buffer(13)]],
    device const float* geom_quat_tail [[buffer(14)]],
    constant int* dims [[buffer(15)]],
    device int* status [[buffer(16)]],
    device const int* world_mask [[buffer(17)]],
    uint world [[thread_position_in_grid]]) {
  int nbody=dims[0], ngeom=dims[1], nsite=dims[2], njnt=dims[3];
  int batch=dims[4];
  if (int(world) >= batch || world_mask[world] == 0) return;
  int b3=int(world)*nbody*3, b4=int(world)*nbody*4;
  int g3=int(world)*ngeom*3, g4=int(world)*ngeom*4;
  int s3=int(world)*nsite*3, s4=int(world)*nsite*4;
  int j3=int(world)*njnt*3;
  bool valid=true;
  for (int i=0; i<nbody*3; ++i) valid=valid && isfinite(body_pos[b3+i]);
  for (int i=0; i<nbody*4; ++i) valid=valid && isfinite(body_quat[b4+i]);
  for (int i=0; i<ngeom*3; ++i) valid=valid && isfinite(geom_pos[g3+i]);
  for (int i=0; i<ngeom*4; ++i) valid=valid && isfinite(geom_quat[g4+i]);
  for (int i=0; i<nsite*3; ++i) valid=valid && isfinite(site_pos[s3+i]);
  for (int i=0; i<nsite*4; ++i) valid=valid && isfinite(site_quat[s4+i]);
  for (int i=0; i<nbody*3; ++i) valid=valid && isfinite(inertial_pos[b3+i]);
  for (int i=0; i<nbody*4; ++i) valid=valid && isfinite(inertial_quat[b4+i]);
  for (int i=0; i<njnt*3; ++i) valid=valid && isfinite(joint_anchor[j3+i]);
  for (int i=0; i<njnt*3; ++i) valid=valid && isfinite(joint_axis[j3+i]);
  int g9=int(world)*ngeom*9;
  for (int i=0; i<ngeom*9; ++i) {
    valid=valid && isfinite(geom_xmat[g9+i]);
    valid=valid && isfinite(geom_xmat_low[g9+i]);
    valid=valid && isfinite(geom_xmat_tail[g9+i]);
  }
  for (int i=0; i<ngeom*4; ++i) {
    valid=valid && isfinite(geom_quat_low[g4+i]);
    valid=valid && isfinite(geom_quat_tail[g4+i]);
  }
  status[world]=valid ? 0 : 1;
}

// The primary pose checker is close to the Metal per-kernel buffer binding
// limit. Validate the remaining retained residual words in a second pass.
// The active-world primary pass initializes status; this kernel only ORs in
// failures so that masked worlds and already-failed worlds retain their state.
kernel void pose_finite_residual_status(
    device const float* body_pos_low [[buffer(0)]],
    device const float* body_pos_tail [[buffer(1)]],
    device const float* body_quat_low [[buffer(2)]],
    device const float* body_quat_tail [[buffer(3)]],
    device const float* geom_pos_low [[buffer(4)]],
    device const float* geom_pos_tail [[buffer(5)]],
    device const float* geom_quat_low [[buffer(6)]],
    device const float* geom_quat_tail [[buffer(7)]],
    device const float* site_pos_low [[buffer(8)]],
    device const float* site_pos_tail [[buffer(9)]],
    device const float* site_quat_low [[buffer(10)]],
    device const float* site_quat_tail [[buffer(11)]],
    device const float* inertial_pos_low [[buffer(12)]],
    device const float* inertial_pos_tail [[buffer(13)]],
    device const float* inertial_quat_low [[buffer(14)]],
    device const float* inertial_quat_tail [[buffer(15)]],
    constant int* dims [[buffer(16)]],
    device int* status [[buffer(17)]],
    device const int* world_mask [[buffer(18)]],
    uint world [[thread_position_in_grid]]) {
  int nbody=dims[0], ngeom=dims[1], nsite=dims[2], batch=dims[4];
  if (int(world) >= batch || world_mask[world] == 0) return;
  int b3=int(world)*nbody*3, b4=int(world)*nbody*4;
  int g3=int(world)*ngeom*3, g4=int(world)*ngeom*4;
  int s3=int(world)*nsite*3, s4=int(world)*nsite*4;
  bool valid=true;
  for (int i=0; i<nbody*3; ++i) {
    valid=valid && isfinite(body_pos_low[b3+i]);
    valid=valid && isfinite(body_pos_tail[b3+i]);
  }
  for (int i=0; i<nbody*4; ++i) {
    valid=valid && isfinite(body_quat_low[b4+i]);
    valid=valid && isfinite(body_quat_tail[b4+i]);
    valid=valid && isfinite(inertial_quat_low[b4+i]);
    valid=valid && isfinite(inertial_quat_tail[b4+i]);
  }
  for (int i=0; i<nbody*3; ++i) {
    valid=valid && isfinite(inertial_pos_low[b3+i]);
    valid=valid && isfinite(inertial_pos_tail[b3+i]);
  }
  for (int i=0; i<ngeom*3; ++i) {
    valid=valid && isfinite(geom_pos_low[g3+i]);
    valid=valid && isfinite(geom_pos_tail[g3+i]);
  }
  for (int i=0; i<ngeom*4; ++i) {
    valid=valid && isfinite(geom_quat_low[g4+i]);
    valid=valid && isfinite(geom_quat_tail[g4+i]);
  }
  for (int i=0; i<nsite*3; ++i) {
    valid=valid && isfinite(site_pos_low[s3+i]);
    valid=valid && isfinite(site_pos_tail[s3+i]);
  }
  for (int i=0; i<nsite*4; ++i) {
    valid=valid && isfinite(site_quat_low[s4+i]);
    valid=valid && isfinite(site_quat_tail[s4+i]);
  }
  if (!valid) status[world] = status[world] | 1;
}

inline float3 qrot_mass(float4 q, float3 v) {
  return v + 2.0f*cross(q.yzw, cross(q.yzw, v) + q.x*v);
}
inline float3 cross_mass(float3 a, float3 b) { return cross(a, b); }
inline void store_cdof(device float* cdof, uint base,
                       uint dof, float3 angular, float3 linear) {
  uint address = base + dof*6;
  cdof[address] = angular.x; cdof[address+1] = angular.y;
  cdof[address+2] = angular.z; cdof[address+3] = linear.x;
  cdof[address+4] = linear.y; cdof[address+5] = linear.z;
}

// Dense, dimension-derived CRBA reference kernel. One thread owns one batch
// row and uses global scratch sized by the host from nbody and nv.
kernel void dense_mass_matrix(
    device const int* parent [[buffer(0)]],
    device const int* rootid [[buffer(1)]],
    device const int* body_treeid [[buffer(2)]],
    device const int* body_jntadr [[buffer(3)]],
    device const int* body_jntnum [[buffer(4)]],
    device const int* dof_parentid [[buffer(5)]],
    device const int* dof_bodyid [[buffer(6)]],
    device const int* jnt_type [[buffer(7)]],
    device const int* jnt_dofadr [[buffer(8)]],
    device const float* body_mass [[buffer(9)]],
    device const float* body_inertia [[buffer(10)]],
    device const float* dof_armature [[buffer(11)]],
    device const float* body_xquat [[buffer(12)]],
    device const float* inertial_xpos [[buffer(13)]],
    device const float* inertial_xquat [[buffer(14)]],
    device const float* joint_anchor [[buffer(15)]],
    device const float* joint_axis [[buffer(16)]],
    device float* output_M [[buffer(17)]],
    device float* root_com [[buffer(18)]],
    device float* cdof [[buffer(19)]],
    device float* crb [[buffer(20)]],
    constant uint* dims [[buffer(21)]],
    device float* local_inertia [[buffer(22)]],
    device const int* body_ids [[buffer(23)]],
    device const int* parent_ids [[buffer(24)]],
    device const int* dof_ids [[buffer(25)]],
    device const int* awake_counts [[buffer(26)]],
    device const int* body_awake_mask [[buffer(27)]],
    constant int* awake_dims [[buffer(28)]],
    device const int* tree_layout [[buffer(29)]],
    constant int* mass_layout_dims [[buffer(30)]],
    uint world [[thread_position_in_grid]]) {
  uint nbody = dims[0], njnt = dims[1], nv = dims[2], batch = dims[3];
  if (world >= batch || mass_layout_dims[5+world] == 0) return;
  uint body_stride = uint(max(awake_dims[0], 1));
  uint dof_stride = uint(max(awake_dims[1], 1));
  uint body_base = world*body_stride;
  uint dof_list_base = world*dof_stride;
  uint nbody_awake = uint(awake_counts[world*3]);
  uint nparent_awake = uint(awake_counts[world*3+1]);
  uint nv_awake = uint(awake_counts[world*3+2]);
  uint com_base = world*nbody*3;
  uint dof_base = world*nv*6;
  uint inertia_base = world*nbody*36;
  uint mass_stride = uint(mass_layout_dims[0]);
  bool block_sparse = mass_layout_dims[1] != 0;
  uint ncomponent = uint(mass_layout_dims[2]);
  uint nv_layout = uint(mass_layout_dims[3]);
  uint component_dof_base = ncomponent + 1;
  uint dof_component_base = component_dof_base + max(nv_layout, 1u);
  uint dof_local_base = dof_component_base + max(nv_layout, 1u);
  uint mass_offset_base = dof_local_base + max(nv_layout, 1u);
  uint mass_base = world*mass_stride;

  // Center of mass for each independent articulated root.
  for (uint root=0; root<nbody; ++root) {
    float total = 0.0f;
    float3 moment = float3(0.0f);
    bool has_body = false;
    for (uint bi=0; bi<nbody_awake; ++bi) {
      uint b = uint(body_ids[body_base+bi]);
      if (uint(rootid[b]) == root) {
        has_body = true;
        float mass = body_mass[b];
        total += mass;
        moment += mass*float3(inertial_xpos[(world*nbody+b)*3],
                              inertial_xpos[(world*nbody+b)*3+1],
                              inertial_xpos[(world*nbody+b)*3+2]);
      }
    }
    if (!has_body) continue;
    float3 center = total > 1e-15f ? moment/total :
        float3(inertial_xpos[(world*nbody+root)*3],
               inertial_xpos[(world*nbody+root)*3+1],
               inertial_xpos[(world*nbody+root)*3+2]);
    root_com[com_base+root*3] = center.x;
    root_com[com_base+root*3+1] = center.y;
    root_com[com_base+root*3+2] = center.z;
  }

  // Initialize body spatial inertias about their articulated-root COM.
  for (uint bi=0; bi<nbody_awake; ++bi) {
    uint b = uint(body_ids[body_base+bi]);
    float mass = body_mass[b];
    float3 inertia = float3(body_inertia[b*3], body_inertia[b*3+1],
                            body_inertia[b*3+2]);
    float4 quat = float4(inertial_xquat[(world*nbody+b)*4],
                         inertial_xquat[(world*nbody+b)*4+1],
                         inertial_xquat[(world*nbody+b)*4+2],
                         inertial_xquat[(world*nbody+b)*4+3]);
    float3 axis[3] = {qrot_mass(quat, float3(1,0,0)),
                      qrot_mass(quat, float3(0,1,0)),
                      qrot_mass(quat, float3(0,0,1))};
    float3 center = float3(root_com[com_base+uint(rootid[b])*3],
                           root_com[com_base+uint(rootid[b])*3+1],
                           root_com[com_base+uint(rootid[b])*3+2]);
    float3 position = float3(inertial_xpos[(world*nbody+b)*3],
                             inertial_xpos[(world*nbody+b)*3+1],
                             inertial_xpos[(world*nbody+b)*3+2]);
    float3 r = position-center;
    float S[9] = {0,-r.z,r.y, r.z,0,-r.x, -r.y,r.x,0};
    float3x3 Irot = float3x3(0.0f);
    for (uint k=0; k<3; ++k) {
      Irot[0][0] += inertia[k]*axis[k].x*axis[k].x;
      Irot[0][1] += inertia[k]*axis[k].x*axis[k].y;
      Irot[0][2] += inertia[k]*axis[k].x*axis[k].z;
      Irot[1][0] += inertia[k]*axis[k].y*axis[k].x;
      Irot[1][1] += inertia[k]*axis[k].y*axis[k].y;
      Irot[1][2] += inertia[k]*axis[k].y*axis[k].z;
      Irot[2][0] += inertia[k]*axis[k].z*axis[k].x;
      Irot[2][1] += inertia[k]*axis[k].z*axis[k].y;
      Irot[2][2] += inertia[k]*axis[k].z*axis[k].z;
    }
    uint ib = inertia_base+b*36;
    for (uint row=0; row<6; ++row) {
      for (uint col=0; col<6; ++col) {
        crb[ib+row*6+col] = 0.0f;
      }
    }
    for (uint row=0; row<3; ++row) {
      for (uint col=0; col<3; ++col) {
        float ss = 0.0f;
        for (uint k=0; k<3; ++k) ss += S[row*3+k]*S[k*3+col];
        crb[ib+row*6+col] = Irot[row][col]-mass*ss;
        crb[ib+row*6+col+3] = mass*S[row*3+col];
        crb[ib+(row+3)*6+col] = -mass*S[row*3+col];
        crb[ib+(row+3)*6+col+3] = row == col ? mass : 0.0f;
      }
    }
    uint local_ib = (world*nbody+b)*36;
    for (uint k=0; k<36; ++k) local_inertia[local_ib+k] = crb[ib+k];
  }

  // World-oriented generalized motion axes centered on each root COM.
  for (uint di=0; di<nv_awake; ++di) {
    uint d = uint(dof_ids[dof_list_base+di]);
    for (uint k=0; k<6; ++k) cdof[dof_base+d*6+k] = 0.0f;
  }
  for (uint bi=0; bi<nbody_awake; ++bi) {
    uint b = uint(body_ids[body_base+bi]);
    if (b == 0) continue;
    int count = body_jntnum[b];
    int first = body_jntadr[b];
    if (count <= 0) continue;
    float4 bodyq = float4(body_xquat[(world*nbody+b)*4],
                          body_xquat[(world*nbody+b)*4+1],
                          body_xquat[(world*nbody+b)*4+2],
                          body_xquat[(world*nbody+b)*4+3]);
    float3 center = float3(root_com[com_base+uint(rootid[b])*3],
                           root_com[com_base+uint(rootid[b])*3+1],
                           root_com[com_base+uint(rootid[b])*3+2]);
    for (int jj=0; jj<count; ++jj) {
      uint j = uint(first+jj);
      uint dadr = uint(jnt_dofadr[j]);
      uint typ = uint(jnt_type[j]);
      float3 anchor = float3(joint_anchor[(world*njnt+j)*3],
                             joint_anchor[(world*njnt+j)*3+1],
                             joint_anchor[(world*njnt+j)*3+2]);
      float3 offset = center-anchor;
      if (typ == 0 || typ == 1) {
        uint skip = typ == 0 ? 3 : 0;
        if (typ == 0) {
          store_cdof(cdof,dof_base,dadr,float3(0),float3(1,0,0));
          store_cdof(cdof,dof_base,dadr+1,float3(0),float3(0,1,0));
          store_cdof(cdof,dof_base,dadr+2,float3(0),float3(0,0,1));
        }
        for (uint k=0; k<3; ++k) {
          float3 basis = k == 0 ? float3(1,0,0) :
                         (k == 1 ? float3(0,1,0) : float3(0,0,1));
          float3 angular = qrot_mass(bodyq,basis);
          store_cdof(cdof,dof_base,dadr+skip+k,angular,
                     cross_mass(angular,offset));
        }
      } else {
        float3 axis = float3(joint_axis[(world*njnt+j)*3],
                             joint_axis[(world*njnt+j)*3+1],
                             joint_axis[(world*njnt+j)*3+2]);
        if (typ == 3) store_cdof(cdof,dof_base,dadr,axis,cross_mass(axis,offset));
        else store_cdof(cdof,dof_base,dadr,float3(0),axis);
      }
    }
  }

  // Composite spatial inertia accumulation, child to parent.
  // Match pinned mj_crb: parent_awake_ind is authoritative here, including
  // an asleep child whose parent is awake/static. Its retained CRB entry is
  // accumulated exactly as the CPU path does.
  for (int pi=int(nparent_awake)-1; pi>=0; --pi) {
    int b = parent_ids[body_base+uint(pi)];
    int p = parent[b];
    if (p > 0) {
      uint child = inertia_base+uint(b)*36;
      uint par = inertia_base+uint(p)*36;
      for (uint k=0; k<36; ++k) crb[par+k] += crb[child+k];
    }
  }

  for (uint ii=0; ii<nv_awake; ++ii) {
    uint i = uint(dof_ids[dof_list_base+ii]);
    int component = block_sparse ? tree_layout[dof_component_base+i] : -1;
    int component_begin = (block_sparse && component >= 0)
        ? tree_layout[uint(component)] : 0;
    int component_end = (block_sparse && component >= 0)
        ? tree_layout[uint(component)+1] : int(nv);
    int component_width = component_end - component_begin;
    uint component_offset = (block_sparse && component >= 0)
        ? uint(tree_layout[mass_offset_base+uint(component)]) : 0;
    uint local_row = block_sparse
        ? uint(tree_layout[dof_local_base+i]) : i;
    for (int col=0; col<component_width; ++col) {
      uint index = block_sparse
          ? mass_base+component_offset+local_row*uint(component_width)+uint(col)
          : mass_base+i*nv+uint(col);
      output_M[index] = 0.0f;
    }
    uint body = uint(dof_bodyid[i]);
    uint ib = inertia_base+body*36;
    float projected[6];
    for (uint r=0; r<6; ++r) {
      projected[r] = 0.0f;
      for (uint c=0; c<6; ++c)
        projected[r] += crb[ib+r*6+c]*cdof[dof_base+i*6+c];
    }
    int ancestor = int(i);
    while (ancestor >= 0) {
      float value = 0.0f;
      for (uint k=0; k<6; ++k)
        value += cdof[dof_base+uint(ancestor)*6+k]*projected[k];
      if (block_sparse) {
        int ancestor_component = tree_layout[dof_component_base+uint(ancestor)];
        int col = tree_layout[dof_local_base+uint(ancestor)];
        if (component >= 0 && ancestor_component == component
            && col >= 0 && col < component_width) {
          output_M[mass_base+component_offset+local_row*uint(component_width)+uint(col)] = value;
          output_M[mass_base+component_offset+uint(col)*uint(component_width)+local_row] = value;
        }
      } else {
        output_M[mass_base+i*nv+uint(ancestor)] = value;
        output_M[mass_base+uint(ancestor)*nv+i] = value;
      }
      ancestor = dof_parentid[ancestor];
    }
    if (block_sparse) {
      if (component >= 0)
        output_M[mass_base+component_offset+local_row*uint(component_width)+local_row] += dof_armature[i];
    } else {
      output_M[mass_base+i*nv+i] += dof_armature[i];
    }
  }
}

// One batch thread applies the compiled block-CSR inertia. MuJoCo kinematic
// components store one dense block at its statically compiled offset. A
// component merges kinematic trees coupled by nonzero tendon armature.
kernel void tree_block_mass_matvec(
    device const float* mass_blocks [[buffer(0)]],
    device const float* vector [[buffer(1)]],
    device const float* tendon_armature_blocks [[buffer(2)]],
    device const int* tree_layout [[buffer(3)]],
    device float* output [[buffer(4)]],
    constant int* dims [[buffer(5)]],
    uint world [[thread_position_in_grid]]) {
  int nv=dims[0], ncomponent=dims[1], batch=dims[2];
  int component_dof_base=ncomponent+1;
  int dof_component_base=component_dof_base+max(nv,1);
  int dof_local_base=dof_component_base+max(nv,1);
  int mass_offset_base=dof_local_base+max(nv,1);
  if (int(world)>=batch || dims[4+int(world)]==0) return;
  int vbase=int(world)*nv;
  int mbase=int(world)*dims[3];
  for (int d=0; d<nv; ++d) output[vbase+d]=0.0f;
  for (int component=0; component<ncomponent; ++component) {
    int begin=tree_layout[component], end=tree_layout[component+1];
    int width=end-begin;
    int block=mbase+tree_layout[mass_offset_base+component];
    for (int row=0; row<width; ++row) {
      float value=0.0f;
      for (int col=0; col<width; ++col) {
        value += (mass_blocks[block+row*width+col]
            + tendon_armature_blocks[block+row*width+col])
            * vector[vbase+tree_layout[component_dof_base+begin+col]];
      }
      output[vbase+tree_layout[component_dof_base+begin+row]]=value;
    }
  }
}

// Reduce kinetic energy per selected world. Dense profiles read M directly;
// component profiles pass the already computed M*qvel product.
kernel void masked_energy_velocity(
    device const float* mass [[buffer(0)]],
    device const float* qvel [[buffer(1)]],
    constant int* dims [[buffer(2)]],
    device float* energy [[buffer(3)]],
    uint world [[thread_position_in_grid]]) {
  int nv=dims[0], batch=dims[1], has_product=dims[2];
  if (int(world)>=batch || dims[3+int(world)]==0) return;
  float value=0.0f;
  for (int row=0; row<nv; ++row) {
    float mv=has_product!=0 ? mass[int(world)*nv+row] : 0.0f;
    if (has_product==0) {
      for (int col=0; col<nv; ++col)
        mv += mass[(int(world)*nv+row)*nv+col]
            * qvel[int(world)*nv+col];
    }
    value += 0.5f*mv*qvel[int(world)*nv+row];
  }
  energy[int(world)*2+1]=value;
}

inline float energy_poly(float stiffness, float p0, float p1, float x) {
  return 0.5f*stiffness*x*x + (p0/3.0f)*x*x*x
      + (p1/4.0f)*x*x*x*x;
}

inline float energy_quat_angle(device const float* qpos, int qadr,
                               device const float* qspring) {
  float4 q=float4(qpos[qadr],qpos[qadr+1],qpos[qadr+2],qpos[qadr+3]);
  float4 r=float4(qspring[qadr],qspring[qadr+1],
                  qspring[qadr+2],qspring[qadr+3]);
  q /= max(length(q),1e-30f);
  r /= max(length(r),1e-30f);
  float w=q.x*r.x+q.y*r.y+q.z*r.z+q.w*r.w;
  float x=-q.x*r.y+q.y*r.x-q.z*r.w+q.w*r.z;
  float y=-q.x*r.z+q.y*r.w+q.z*r.x-q.w*r.y;
  float z=-q.x*r.w-q.y*r.z+q.z*r.y+q.w*r.x;
  float vn=sqrt(x*x+y*y+z*z);
  return 2.0f*atan2(vn,fabs(w));
}

// Refresh mj_energyPos for selected worlds without evaluating any input or
// mutating retained rows for mask=0. All topology is immutable device data;
// the kernel receives the already produced FK, tendon and flex state.
kernel void masked_energy_position(
    device const float* qpos [[buffer(0)]],
    device const float* inertial_pos [[buffer(1)]],
    device const float* body_mass [[buffer(2)]],
    device const float* gravity [[buffer(3)]],
    device const float* qspring [[buffer(4)]],
    device const int* joint_int [[buffer(5)]],
    device const float* joint_float [[buffer(6)]],
    device const float* fixed_length [[buffer(7)]],
    device const float* spatial_length [[buffer(8)]],
    device const int* tendon_int [[buffer(9)]],
    device const float* tendon_float [[buffer(10)]],
    device const float* flex_length [[buffer(11)]],
    device const int* flex_edge [[buffer(12)]],
    device const float* flex_float [[buffer(13)]],
    device const int* body_awake [[buffer(14)]],
    device const int* body_count [[buffer(15)]],
    device const int* tree_awake [[buffer(16)]],
    constant int* dims [[buffer(17)]],
    device float* energy [[buffer(18)]],
    uint world [[thread_position_in_grid]]) {
  int nq=dims[0], nbody=dims[2], njnt=dims[3];
  int ntendon=dims[4], nflexedge=dims[5], nflexrows=dims[6];
  int ntree=dims[7], spring_off=dims[8], gravity_off=dims[9];
  int sleep_on=dims[10], batch=dims[11];
  if (int(world)>=batch) return;
  if (dims[12+int(world)]==0) return;
  float value=0.0f;
  if (!gravity_off) {
    int pbase=int(world)*nbody*3;
    for (int body=1; body<nbody; ++body) {
      float dotp=0.0f;
      for (int axis=0; axis<3; ++axis)
        dotp += inertial_pos[pbase+body*3+axis]*gravity[axis];
      value -= body_mass[body-1]*dotp;
    }
  }
  bool sleep_filter=sleep_on!=0 && body_count[int(world)]<nbody;
  if (!spring_off) {
    for (int jid=0; jid<njnt; ++jid) {
      int type=joint_int[jid*3], adr=joint_int[jid*3+1];
      int body=joint_int[jid*3+2];
      float k=joint_float[jid*3], p0=joint_float[jid*3+1];
      float p1=joint_float[jid*3+2], x=0.0f, part=0.0f;
      if (type==2 || type==3) {
        x=qpos[int(world)*nq+adr]-qspring[adr];
        part=energy_poly(k,p0,p1,x);
      } else if (type==1) {
        x=energy_quat_angle(qpos+int(world)*nq,adr,qspring);
        part=energy_poly(k,p0,p1,x);
      } else if (type==0) {
        float3 delta=float3(qpos[int(world)*nq+adr]-qspring[adr],
                            qpos[int(world)*nq+adr+1]-qspring[adr+1],
                            qpos[int(world)*nq+adr+2]-qspring[adr+2]);
        float linear=length(delta);
        float angular=energy_quat_angle(qpos+int(world)*nq,adr+3,qspring);
        part=energy_poly(k,p0,p1,linear)+energy_poly(k,p0,p1,angular);
      }
      if (sleep_filter && body_awake[int(world)*nbody+body]==0) part=0.0f;
      value += part;
    }
    for (int tid=0; tid<ntendon; ++tid) {
      float len=(tendon_int[tid*4+3]!=0)
          ? spatial_length[int(world)*ntendon+tid]
          : fixed_length[int(world)*ntendon+tid];
      float lo=tendon_float[tid*5+3], hi=tendon_float[tid*5+4];
      float x=len>hi ? len-hi : (len<lo ? len-lo : 0.0f);
      float part=energy_poly(tendon_float[tid*5],tendon_float[tid*5+1],
                             tendon_float[tid*5+2],x);
      if (sleep_filter) {
        int count=tendon_int[tid*4];
        bool sleeping=count==0;
        if (count==1) {
          int tree=tendon_int[tid*4+1];
          sleeping=tree_awake[int(world)*ntree+tree]==0;
        } else if (count==2) {
          int t0=tendon_int[tid*4+1], t1=tendon_int[tid*4+2];
          sleeping=tree_awake[int(world)*ntree+t0]==0
              && tree_awake[int(world)*ntree+t1]==0;
        } else if (count>2) {
          sleeping=false;
        }
        if (sleeping) part=0.0f;
      }
      value += part;
    }
    for (int row=0; row<nflexrows; ++row) {
      int edge=flex_edge[row];
      float dx=flex_float[row*2]-flex_length[int(world)*nflexedge+edge];
      value += 0.5f*flex_float[row*2+1]*dx*dx;
    }
  }
  energy[int(world)*2]=value;
}

// Assemble the rank-one tendon armature terms in a distinct fixed buffer.
// This keeps cached asleep rigid-body entries untouched while current awake
// tendon rows are refreshed exactly once per call.
kernel void add_tendon_armature(
    device const float* tendon_J [[buffer(0)]],
    device const float* armature [[buffer(1)]],
    device const int* tendon_treeid [[buffer(2)]],
    device const int* tendon_treenum [[buffer(3)]],
    device const int* tendon_j_rowadr [[buffer(4)]],
    device const int* tendon_j_rownnz [[buffer(5)]],
    device const int* tendon_j_colind [[buffer(6)]],
    device const int* tree_awake [[buffer(7)]],
    device const int* component_layout [[buffer(8)]],
    device const int* mass_rowadr [[buffer(9)]],
    device const int* mass_rownnz [[buffer(10)]],
    device const int* mass_colind [[buffer(11)]],
    device const int* dof_treeid [[buffer(12)]],
    device float* output [[buffer(13)]],
    constant int* dims [[buffer(14)]],
    device const int* world_mask [[buffer(15)]],
    uint world [[thread_position_in_grid]]) {
  int nv=dims[0], ntendon=dims[1], batch=dims[2], ntree=dims[3];
  int stride=dims[4];
  bool block_sparse=dims[5] != 0;
  if (int(world) >= batch || world_mask[world] == 0) return;
  int output_base=int(world)*stride;
  int ncomponent=0;
  // The first element after component offsets is not a count; derive the
  // number of components from the mass-layout ABI carried in dims.
  // component_layout is sized from model metadata, and component count is
  // encoded at dims[6] when sparse storage is active.
  ncomponent=dims[6];
  int dof_component_base=ncomponent+1+max(nv,1);
  int dof_local_base=dof_component_base+max(nv,1);
  int mass_offset_base=dof_local_base+max(nv,1);
  int tree_awake_base=int(world)*max(ntree,1);
  // Refresh only entries involving an awake tree. Fully asleep entries retain
  // their previous armature contribution, matching the cached smooth-mass
  // ownership used by the other awake stages. Workspace preparation zeros
  // the initial buffer, so an initially asleep entry has a defined zero.
  if (block_sparse) {
    int component_offset_base=mass_offset_base;
    for (int component=0; component<ncomponent; ++component) {
      int begin=component_layout[component];
      int end=component_layout[component+1];
      int width=end-begin;
      int packed_dof_base=ncomponent+1;
      int block_offset=component_layout[component_offset_base+component];
      for (int row=0; row<width; ++row) {
        int dof_i=component_layout[packed_dof_base+begin+row];
        int tree_i=dof_treeid[dof_i];
        bool awake_i=tree_i >= 0 && tree_i < ntree
            && tree_awake[tree_awake_base+tree_i] != 0;
        for (int col=0; col<width; ++col) {
          int dof_j=component_layout[packed_dof_base+begin+col];
          int tree_j=dof_treeid[dof_j];
          bool awake_j=tree_j >= 0 && tree_j < ntree
              && tree_awake[tree_awake_base+tree_j] != 0;
          if (awake_i || awake_j)
            output[output_base+block_offset+row*width+col]=0.0f;
        }
      }
    }
  } else {
    for (int i=0; i<nv; ++i) {
      int tree_i=dof_treeid[i];
      bool awake_i=tree_i >= 0 && tree_i < ntree
          && tree_awake[tree_awake_base+tree_i] != 0;
      for (int j=0; j<nv; ++j) {
        int tree_j=dof_treeid[j];
        bool awake_j=tree_j >= 0 && tree_j < ntree
            && tree_awake[tree_awake_base+tree_j] != 0;
        if (awake_i || awake_j) output[output_base+i*nv+j]=0.0f;
      }
    }
  }
  for (int t=0; t<ntendon; ++t) {
    float a=armature[t];
    if (a == 0.0f) continue;
    // Match engine_sleep.c: tendons spanning more than two trees are always
    // kept awake because their tree association cannot be represented by the
    // two-entry tendon_treeid table.
    int tree_count=tendon_treenum[t];
    bool is_awake=true;
    if (tree_count == 1) {
      int tree=tendon_treeid[t*2];
      is_awake=tree >= 0 && tree < ntree
          && tree_awake[tree_awake_base+tree] != 0;
    } else if (tree_count == 2) {
      int tree1=tendon_treeid[t*2], tree2=tendon_treeid[t*2+1];
      bool awake1=tree1 >= 0 && tree1 < ntree
          && tree_awake[tree_awake_base+tree1] != 0;
      bool awake2=tree2 >= 0 && tree2 < ntree
          && tree_awake[tree_awake_base+tree2] != 0;
      is_awake=awake1 || awake2;
    }
    if (!is_awake) continue;
    int rowadr=tendon_j_rowadr[t];
    int rownnz=tendon_j_rownnz[t];
    for (int ri=0; ri<rownnz; ++ri) {
      int i=tendon_j_colind[rowadr+ri];
      if (i < 0 || i >= nv) continue;
      float Ji=tendon_J[(world*ntendon+t)*nv+i];
      if (Ji == 0.0f) continue;
      int ci=block_sparse ? component_layout[dof_component_base+i] : -1;
      int local_i=block_sparse ? component_layout[dof_local_base+i] : i;
      int comp_width=block_sparse
          ? component_layout[ci+1]-component_layout[ci] : nv;
      int component_offset=block_sparse
          ? component_layout[mass_offset_base+ci] : 0;
      int mass_adr=mass_rowadr[i];
      int mass_nnz=mass_rownnz[i];
      for (int rj=0; rj<mass_nnz; ++rj) {
        int j=mass_colind[mass_adr+rj];
        if (j < 0 || j >= nv) continue;
        float Jj=tendon_J[(world*ntendon+t)*nv+j];
        if (Jj == 0.0f) continue;
        if (block_sparse && component_layout[dof_component_base+j] != ci) continue;
        int local_j=block_sparse ? component_layout[dof_local_base+j] : j;
        int address=block_sparse
            ? output_base+component_offset+local_i*comp_width+local_j
            : output_base+i*nv+j;
        output[address] += a*Ji*Jj;
        int mirror=block_sparse
            ? output_base+component_offset+local_j*comp_width+local_i
            : output_base+j*nv+i;
        if (mirror != address) output[mirror] += a*Ji*Jj;
      }
    }
  }
}
