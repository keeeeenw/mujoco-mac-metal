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

inline float4 qmul(float4 a, float4 b) {
  return float4(a.x*b.x-dot(a.yzw,b.yzw),
      a.x*b.yzw+b.x*a.yzw+cross(a.yzw,b.yzw));
}
inline float4 qconj(float4 q) { return float4(q.x, -q.yzw); }
inline float4 qunit(float4 q) { return q * rsqrt(dot(q,q)); }
inline float3 qrot(float4 q, float3 v) {
  return v + 2.0f*cross(q.yzw, cross(q.yzw,v)+q.x*v);
}
inline float3 read3(device const float* a, uint i) {
  return float3(a[i], a[i+1], a[i+2]);
}
inline float4 read4(device const float* a, uint i) {
  return float4(a[i], a[i+1], a[i+2], a[i+3]);
}
inline void write3(device float* a, uint i, float3 v) {
  a[i]=v.x; a[i+1]=v.y; a[i+2]=v.z;
}
inline void write4(device float* a, uint i, float4 v) {
  a[i]=v.x; a[i+1]=v.y; a[i+2]=v.z; a[i+3]=v.w;
}

inline void frame_pose(int type, int id, uint world, constant int* dims,
    device const float* body_pos, device const float* body_quat,
    device const float* inertial_pos, device const float* inertial_quat,
    device const float* geom_world_pos, device const float* geom_world_quat,
    device const float* site_world_pos, device const float* site_world_quat,
    thread float3& pos, thread float4& quat) {
  if (type == 1) {
    pos = read3(inertial_pos, (world*uint(dims[5]) + uint(id))*3);
    quat = read4(inertial_quat, (world*uint(dims[5]) + uint(id))*4);
  } else if (type == 2) {
    pos = read3(body_pos, (world*uint(dims[5]) + uint(id))*3);
    quat = read4(body_quat, (world*uint(dims[5]) + uint(id))*4);
  } else if (type == 5) {
    pos = read3(geom_world_pos, (world*uint(dims[6]) + uint(id))*3);
    quat = read4(geom_world_quat, (world*uint(dims[6]) + uint(id))*4);
  } else {
    pos = read3(site_world_pos, (world*uint(dims[7]) + uint(id))*3);
    quat = read4(site_world_quat, (world*uint(dims[7]) + uint(id))*4);
  }
}

inline int object_body(int type, int id,
    device const int* geom_bodyid, device const int* site_bodyid) {
  if (type == 1 || type == 2) return id;
  if (type == 5) return geom_bodyid[id];
  return site_bodyid[id];
}

inline void object_velocity(int type, int id, uint world, float3 pos,
    constant int* dims, device const float* cvel,
    device const float* root_com, device const int* body_rootid,
    device const int* geom_bodyid, device const int* site_bodyid,
    thread float3& angular, thread float3& linear) {
  int body = object_body(type, id, geom_bodyid, site_bodyid);
  int root = body_rootid[body];
  uint nb = uint(dims[5]);
  uint cv = (world*nb+uint(body))*6;
  uint rc = (world*nb+uint(root))*3;
  angular = read3(cvel, cv);
  linear = read3(cvel, cv+3) + cross(angular, pos-read3(root_com, rc));
}

kernel void evaluate_sensors(
    device const float* qpos [[buffer(0)]],
    device const float* qvel [[buffer(1)]],
    device const float* time [[buffer(2)]],
    device const float* body_pos [[buffer(3)]],
    device const float* body_quat [[buffer(4)]],
    device const float* inertial_pos [[buffer(5)]],
    device const float* inertial_quat [[buffer(6)]],
    device const float* geom_world_pos [[buffer(7)]],
    device const float* geom_world_quat [[buffer(8)]],
    device const float* site_world_pos [[buffer(9)]],
    device const float* site_world_quat [[buffer(10)]],
    device const float* cvel [[buffer(11)]],
    device const float* root_com [[buffer(12)]],
    device const int* sensor_type [[buffer(13)]],
    device const int* sensor_datatype [[buffer(14)]],
    device const int* sensor_needstage [[buffer(15)]],
    device const int* sensor_objtype [[buffer(16)]],
    device const int* sensor_objid [[buffer(17)]],
    device const int* sensor_reftype [[buffer(18)]],
    device const int* sensor_refid [[buffer(19)]],
    device const int* sensor_dim [[buffer(20)]],
    device const int* sensor_adr [[buffer(21)]],
    device const float* sensor_cutoff [[buffer(22)]],
    device const int* jnt_qposadr [[buffer(23)]],
    device const int* jnt_dofadr [[buffer(24)]],
    device const int* body_rootid [[buffer(25)]],
    device const int* geom_bodyid [[buffer(26)]],
    device const int* site_bodyid [[buffer(27)]],
    constant int* dims [[buffer(28)]],
    device float* output [[buffer(29)]],
    constant int* stage_mask [[buffer(30)]],
    uint index [[thread_position_in_grid]]) {
  uint nsensor=uint(dims[0]), ndata=uint(dims[1]), nq=uint(dims[2]);
  uint nv=uint(dims[3]), batch=uint(dims[8]);
  if (index >= batch*nsensor || (dims[9] & 8192)) return;
  uint world=index/nsensor, i=index-world*nsensor;
  uint stage=uint(sensor_needstage[i]);
  if ((uint(stage_mask[0]) & (1u<<stage)) == 0) return;
  uint adr=uint(sensor_adr[i]), dim=uint(sensor_dim[i]);
  int typ=sensor_type[i], jid=sensor_objid[i];
  float value[4] = {0.0f,0.0f,0.0f,0.0f};
  uint qa=0, da=0;
  if (typ==9 || typ==18) qa=uint(jnt_qposadr[jid]);
  if (typ==10 || typ==19) da=uint(jnt_dofadr[jid]);
  if (typ==9) value[0]=qpos[world*nq+qa];
  else if (typ==10) value[0]=qvel[world*nv+da];
  else if (typ==18) {
    float4 q=qunit(read4(qpos,world*nq+qa));
    value[0]=q.x; value[1]=q.y; value[2]=q.z; value[3]=q.w;
  } else if (typ==19) {
    float3 v=read3(qvel,world*nv+da);
    value[0]=v.x; value[1]=v.y; value[2]=v.z;
  } else if (typ==2 || typ==3 || typ==31 || typ==32) {
    float3 pos, ref_pos; float4 quat, ref_quat;
    frame_pose(sensor_objtype[i],sensor_objid[i],world,dims,body_pos,body_quat,
        inertial_pos,inertial_quat,geom_world_pos,geom_world_quat,
        site_world_pos,site_world_quat,pos,quat);
    float3 angular, linear;
    object_velocity(sensor_objtype[i],sensor_objid[i],world,pos,dims,cvel,
        root_com,body_rootid,geom_bodyid,site_bodyid,angular,linear);
    if (typ==2 || typ==3) {
      angular=qrot(qconj(quat),angular);
      linear=qrot(qconj(quat),linear);
      float3 v=typ==3 ? angular : linear;
      value[0]=v.x; value[1]=v.y; value[2]=v.z;
    } else {
      int refid=sensor_refid[i];
      if (refid>=0) {
        frame_pose(sensor_reftype[i],refid,world,dims,body_pos,body_quat,
            inertial_pos,inertial_quat,geom_world_pos,geom_world_quat,
            site_world_pos,site_world_quat,ref_pos,ref_quat);
        float3 ref_ang, ref_lin;
        object_velocity(sensor_reftype[i],refid,world,ref_pos,dims,cvel,
            root_com,body_rootid,geom_bodyid,site_bodyid,ref_ang,ref_lin);
        linear=linear-ref_lin+cross(pos-ref_pos,ref_ang);
        angular=angular-ref_ang;
        angular=qrot(qconj(ref_quat),angular);
        linear=qrot(qconj(ref_quat),linear);
      }
      float3 v=typ==31 ? linear : angular;
      value[0]=v.x; value[1]=v.y; value[2]=v.z;
    }
  } else if (typ==45) value[0]=time[world];
  else {
    float3 pos, rpos; float4 quat, rquat;
    frame_pose(sensor_objtype[i],sensor_objid[i],world,dims,body_pos,body_quat,
        inertial_pos,inertial_quat,geom_world_pos,geom_world_quat,
        site_world_pos,site_world_quat,pos,quat);
    if (sensor_refid[i] < 0) {
      rpos=float3(0); rquat=float4(1,0,0,0);
    } else {
      frame_pose(sensor_reftype[i],sensor_refid[i],world,dims,body_pos,body_quat,
          inertial_pos,inertial_quat,geom_world_pos,geom_world_quat,
          site_world_pos,site_world_quat,rpos,rquat);
    }
    if (typ==26) {
      float3 v=qrot(qconj(rquat),pos-rpos);
      value[0]=v.x; value[1]=v.y; value[2]=v.z;
    } else if (typ==27) {
      float4 q=qunit(qmul(qconj(rquat),quat));
      value[0]=q.x; value[1]=q.y; value[2]=q.z; value[3]=q.w;
    } else {
      uint axis=uint(typ-28);
      float3 basis=axis==0 ? float3(1,0,0) : (axis==1 ? float3(0,1,0) : float3(0,0,1));
      float3 v=qrot(qconj(rquat),qrot(quat,basis));
      value[0]=v.x; value[1]=v.y; value[2]=v.z;
    }
  }
  float cutoff=sensor_cutoff[i];
  if (cutoff>0 && sensor_datatype[i]==0) {
    for (uint j=0;j<dim;j++) value[j]=clamp(value[j],-cutoff,cutoff);
  } else if (cutoff>0 && sensor_datatype[i]==1) {
    for (uint j=0;j<dim;j++) value[j]=min(value[j],cutoff);
  }
  uint base=world*ndata+adr;
  for (uint j=0;j<dim;j++) output[base+j]=value[j];
}
