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
inline float3x3 qmat(float4 q) {
  float x=q.y, y=q.z, z=q.w, w=q.x;
  return float3x3(
    1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w),
    2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w),
    2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y));
}
inline float3 sub_quat(float4 qa, float4 qb) {
  qa=qunit(qa); qb=qunit(qb);
  float4 d=qmul(float4(qb.x,-qb.yzw),qa);
  float s=length(d.yzw);
  if (s<1e-12f) return float3(0.0f);
  float speed=2.0f*atan2(s,d.x);
  if (speed>M_PI_F) speed-=2.0f*M_PI_F;
  return d.yzw*(speed/s);
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
  int typ0=sensor_type[i];
  // This kernel serves the 14 baseline families only; milestone-016 state,
  // force and spatial families are evaluated by dedicated kernels.
  bool legacy = typ0==9||typ0==10||typ0==18||typ0==19||typ0==26||typ0==27
      ||typ0==28||typ0==29||typ0==30||typ0==45||typ0==31||typ0==32
      ||typ0==2||typ0==3;
  if (!legacy) return;
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

// Milestone 016 state families: tendon/actuator/limit/subtree/insidesite/
// energy/magnetometer. Tendon length/velocity = fixed linear maps plus the
// spatial kinematics workspace (zero rows for fixed tendons); actuator
// length/velocity = stateless linear maps, or the general kinematics
// workspace when act_kind==1. Joint/tendon limit distances reuse the
// coupled-assembly formulas (lower row first).
// Aligned per-world scratch: one reusable record per compiled body.
struct SensorSubtree { float3 xipos, vcom, scom, slin, sang, smom, slv; float smass; };
static_assert(sizeof(SensorSubtree)==128, "sensor subtree workspace ABI");
kernel void build_sensor_subtrees(
    device const float* inertial_pos [[buffer(0)]],
    device const float* inertial_quat [[buffer(1)]],
    device const float* cvel [[buffer(2)]],
    device const float* root_com [[buffer(3)]],
    device const float* massub [[buffer(4)]],
    device const int* body_tree [[buffer(5)]],
    constant int* dims [[buffer(6)]],
    device float* runtime [[buffer(7)]],
    uint world [[thread_position_in_grid]]) {
  uint nbody=uint(dims[5]);
  if (world>=uint(dims[8])) return;
  device SensorSubtree* sub = reinterpret_cast<device SensorSubtree*>(runtime+4)+world*nbody;
    // Shared subtree pass (pinned mj_subtreeVel): per-body COM velocity
    // from com-based cvel, mass-weighted backward accumulation.

    for (uint k=0;k<nbody;++k) {
      sub[k].xipos=read3(inertial_pos,(world*nbody+k)*3);
      float3 w=read3(cvel,(world*nbody+k)*6);
      float3 vr=read3(cvel,(world*nbody+k)*6+3);
      uint root=uint(body_tree[k*2+1]);
      sub[k].vcom=vr+cross(w,sub[k].xipos-read3(root_com,(world*nbody+root)*3));
      sub[k].scom=sub[k].xipos;
      sub[k].slin=massub[k*5+0]*sub[k].vcom;
      float4 iq=qunit(read4(inertial_quat,(world*nbody+k)*4));
      float3 dv=qrot(qconj(iq),w);
      dv*=float3(massub[k*5+2],massub[k*5+3],massub[k*5+4]);
      sub[k].sang=qrot(iq,dv);
    }
    for (int k=int(nbody)-1;k>0;--k) {
      uint p=uint(body_tree[uint(k)*2+0]);
      sub[p].slin+=sub[uint(k)].slin;
    }

    for (uint k=0;k<nbody;++k) {
      sub[k].smass=massub[k*5+0];
      sub[k].smom=massub[k*5+0]*sub[k].xipos;
    }
    for (int k=int(nbody)-1;k>0;--k) {
      uint p=uint(body_tree[uint(k)*2+0]);
      sub[p].smass+=sub[uint(k)].smass;
      sub[p].smom+=sub[uint(k)].smom;
    }
    for (uint k=0;k<nbody;++k) {
      sub[k].scom=sub[k].smass>1e-15f ? sub[k].smom/sub[k].smass : sub[k].xipos;
    }

    for (uint k=0;k<nbody;++k) {
      float sm=sub[k].smass;
      sub[k].slv=sm>1e-30f ? sub[k].slin/sm : float3(0.0f);
    }
      for (int k=int(nbody)-1;k>0;--k) {
        uint kk=uint(k), pp=uint(body_tree[kk*2+0]);
        float3 dx=sub[kk].xipos-sub[kk].scom;
        float3 dvv=sub[kk].vcom-sub[kk].slv;
        sub[kk].sang+=cross(dx,massub[kk*5+0]*dvv);
        sub[pp].sang+=sub[kk].sang;
        float3 dx2=sub[kk].scom-sub[pp].scom;
        float3 dv2=sub[kk].slv-sub[pp].slv;
        sub[pp].sang+=cross(dx2,sub[kk].smass*dv2);
      }
}

kernel void evaluate_state_sensors(
    device const float* qpos [[buffer(0)]],
    device const float* qvel [[buffer(1)]],
    device const float* body_pos [[buffer(2)]],
    device const float* inertial_pos [[buffer(3)]],
    device const float* inertial_quat [[buffer(4)]],
    device const float* body_quat [[buffer(5)]],
    device const float* site_pos [[buffer(6)]],
    device const float* site_quat [[buffer(7)]],
    device const float* cvel [[buffer(8)]],
    device const float* root_com [[buffer(9)]],
    device const float* mass_matrix [[buffer(10)]],
    device const float* ten_fix_lmap [[buffer(11)]],
    device const float* ten_fix_mmap [[buffer(12)]],
    device const float* ten_spa_len [[buffer(13)]],
    device const float* ten_spa_vel [[buffer(14)]],
    device const float* act_stat_lmap [[buffer(15)]],
    device const float* act_stat_mmap [[buffer(16)]],
    device const float* act_dyn_len [[buffer(17)]],
    device const float* act_dyn_vel [[buffer(18)]],
    device const int* meta [[buffer(19)]],
    device const float* geom_pos [[buffer(20)]],
    device const int* jnt_meta [[buffer(21)]],
    device const float* jnt_lim [[buffer(22)]],
    device const float* ten_lim [[buffer(23)]],
    device const float* massub [[buffer(24)]],
    device const int* body_tree [[buffer(25)]],
    device const float* site_geom [[buffer(26)]],
    device const float* econst [[buffer(27)]],
    constant int* dims [[buffer(28)]],
    device float* output [[buffer(29)]],
    device const float* runtime [[buffer(30)]],
    uint index [[thread_position_in_grid]]) {
  // meta per sensor (10 ints): type, datatype, needstage, objtype, objid,
  //   dim, adr, cutoff_bits, reftype, refid.
  // jnt_meta per joint (4 ints): type, qposadr, dofadr, limited.
  // jnt_lim per joint (3 floats): range0, range1, margin.
  // ten_lim per tendon (9 floats): range0, range1, margin, ls0, ls1, k, p0,
  //   p1, limited.
  // massub per body (5 floats): mass, subtreemass, ix, iy, iz.
  // body_tree per body (2 ints): parent, rootid.
  // site_geom per site (4 floats): size0, size1, size2, type.
  // econst: gravity(3), magnetic(3), dis_spring, dis_damper, dis_gravity,
  //   nq, qpos_spring(nq), jnt_k(njnt), jnt_p0(njnt), jnt_p1(njnt).
  // dims (14 ints): nsensor, ndata, nq, nv, njnt, nbody, ngeom, nsite, batch,
  //   disable, ntendon, nu, act_kind, has_mass.
  uint nsensor=uint(dims[0]), ndata=uint(dims[1]), nq=uint(dims[2]);
  uint nv=uint(dims[3]), njnt=uint(dims[4]), nbody=uint(dims[5]);
  uint nsite=uint(dims[7]), batch=uint(dims[8]);
  uint nt=uint(dims[10]), nu=uint(dims[11]);
  uint act_kind=uint(dims[12]), has_mass=uint(dims[13]);
  if (index >= batch*nsensor || (dims[9] & 8192)) return;
  uint world=index/nsensor, i=index-world*nsensor;
  uint stage=uint(meta[i*10+2]);
  if ((uint(runtime[0]) & (1u<<stage)) == 0) return;
  int typ=meta[i*10+0], objid=meta[i*10+4];
  // This kernel serves POS/VEL state families only; force, contact and
  // spatial families have dedicated kernels.
  bool handled = typ==11||typ==12||typ==13||typ==14||typ==20||typ==21
      ||typ==23||typ==24||typ==6||typ==35||typ==36||typ==37||typ==38
      ||typ==43||typ==44||typ==8;
  if (!handled) return;
  uint dim=uint(meta[i*10+5]), adr=uint(meta[i*10+6]);
  float value[6] = {0.0f,0.0f,0.0f,0.0f,0.0f,0.0f};
  uint qb=world*nq, vb=world*nv;
  float ten_len=0.0f, ten_vel=0.0f;
  bool need_ten=(typ==11||typ==12||typ==23||typ==24);
  if (need_ten && objid>=0 && uint(objid)<nt) {
    uint t=uint(objid);
    for (uint q=0;q<nq;++q) ten_len+=ten_fix_lmap[t*nq+q]*qpos[qb+q];
    for (uint d=0;d<nv;++d) ten_vel+=ten_fix_mmap[t*nv+d]*qvel[vb+d];
    ten_len+=ten_spa_len[world*nt+t];
    ten_vel+=ten_spa_vel[world*nt+t];
  }
  float act_len=0.0f, act_vel=0.0f;
  bool need_act=(typ==13||typ==14);
  if (need_act && objid>=0 && uint(objid)<nu) {
    uint a=uint(objid);
    if (act_kind==1) {
      act_len=act_dyn_len[world*nu+a];
      act_vel=act_dyn_vel[world*nu+a];
    } else {
      for (uint q=0;q<nq;++q) act_len+=act_stat_lmap[a*nq+q]*qpos[qb+q];
      for (uint d=0;d<nv;++d) act_vel+=act_stat_mmap[a*nv+d]*qvel[vb+d];
    }
  }
  if (typ==11) value[0]=ten_len;
  else if (typ==12) value[0]=ten_vel;
  else if (typ==13) value[0]=act_len;
  else if (typ==14) value[0]=act_vel;
  else if (typ==20 || typ==21) {
    if (objid>=0 && uint(objid)<njnt && jnt_meta[uint(objid)*4+3]!=0) {
      uint j=uint(objid);
      int jt=jnt_meta[j*4+0];
      float r0=jnt_lim[j*3+0], r1=jnt_lim[j*3+1], margin=jnt_lim[j*3+2];
      if (jt==2 || jt==3) {
        int q=jnt_meta[j*4+1], d=jnt_meta[j*4+2];
        bool has = d>=0 && uint(d)<nv;
        float d0 = has ? qpos[qb+uint(q)]-r0 : 1e30f;
        float d1 = has ? r1-qpos[qb+uint(q)] : 1e30f;
        if (typ==20) {
          if (d0<margin) value[0]=d0-margin;
          else if (d1<margin) value[0]=d1-margin;
        } else {
          float v = has ? qvel[vb+uint(d)] : 0.0f;
          if (d0<margin) value[0]=v;
          else if (d1<margin) value[0]=-v;
        }
      } else if (jt==1) {
        int q=jnt_meta[j*4+1], d=jnt_meta[j*4+2];
        float4 quat=float4(qpos[qb+uint(q)],qpos[qb+uint(q)+1],qpos[qb+uint(q)+2],qpos[qb+uint(q)+3]);
        float nq4=length(quat);
        quat=nq4>1e-30f ? quat/nq4 : float4(1,0,0,0);
        float3 vv=quat.yzw;
        float s=length(vv);
        float speed=2.0f*atan2(s,quat.x);
        if (speed>3.14159265358979f) speed-=2.0f*3.14159265358979f;
        float rmax=max(r0,r1);
        float dist=rmax-speed;
        if (typ==20) value[0]=dist<margin ? dist-margin : 0.0f;
        else {
          float vel=0.0f;
          float3 naxis=s>1e-30f ? vv/max(s,1e-30f) : float3(0.0f);
          for (int k=0;k<3;++k) {
            int dof=d+k;
            if (dof>=0 && uint(dof)<nv) vel+=-naxis[k]*qvel[vb+uint(dof)];
          }
          value[0]=dist<margin ? vel : 0.0f;
        }
      }
    }
  }
  else if (typ==23 || typ==24) {
    if (objid>=0 && uint(objid)<nt && ten_lim[uint(objid)*9+8]>0.5f) {
      uint t=uint(objid);
      float r0=ten_lim[t*9+0], r1=ten_lim[t*9+1], margin=ten_lim[t*9+2];
      float d0=-(r0-ten_len), d1=(r1-ten_len);
      if (typ==23) {
        if (d0<margin) value[0]=d0-margin;
        else if (d1<margin) value[0]=d1-margin;
      } else {
        // Row Jacobian is -side*Jrow: lower (side -1) reads +ten_vel.
        if (d0<margin) value[0]=ten_vel;
        else if (d1<margin) value[0]=-ten_vel;
      }
    }
  }
  else if (typ==6) {
    float4 q=qunit(read4(site_quat,(world*nsite+uint(objid))*4));
    float3 v=qrot(qconj(q),float3(econst[3],econst[4],econst[5]));
    value[0]=v.x; value[1]=v.y; value[2]=v.z;
  }
  else if (typ>=35 && typ<=38) {
    device const SensorSubtree* sub = reinterpret_cast<device const SensorSubtree*>(runtime+4)+world*nbody;
    if (typ==35) {
      value[0]=sub[uint(objid)].scom.x; value[1]=sub[uint(objid)].scom.y; value[2]=sub[uint(objid)].scom.z;
    } else if (typ==36) {
      float3 lv=sub[uint(objid)].slv;
      value[0]=lv.x; value[1]=lv.y; value[2]=lv.z;
    } else if (typ==37) {
      value[0]=sub[uint(objid)].sang.x; value[1]=sub[uint(objid)].sang.y; value[2]=sub[uint(objid)].sang.z;
    } else {
      // Pinned INSIDESITE (massless-body rule uses subtree com).
      // Site zone types share mjtGeom values: 2 sphere, 3 capsule,
      // 4 ellipsoid, 5 cylinder, 6 box, 0 plane.
      float3 pp;
      int otype=meta[i*10+3];
      if (otype==1) {
        pp=read3(inertial_pos,(world*nbody+uint(objid))*3);
        if (objid>0 && massub[uint(objid)*5+0]<1e-15f
            && massub[uint(objid)*5+1]>=1e-15f) pp=sub[uint(objid)].scom;
      }
      else if (otype==2) pp=read3(body_pos,(world*nbody+uint(objid))*3);
      else if (otype==5) pp=read3(geom_pos,(world*uint(dims[6])+uint(objid))*3);
      else pp=read3(site_pos,(world*nsite+uint(objid))*3);
      uint s=uint(meta[i*10+9]);
      float3 sp=read3(site_pos,(world*nsite+s)*3);
      float4 sq=qunit(read4(site_quat,(world*nsite+s)*4));
      float3 vec=pp-sp;
      float3 pl=qrot(qconj(sq),vec);
      float st=site_geom[s*4+3];
      float sx=site_geom[s*4+0], sy=site_geom[s*4+1], sz=site_geom[s*4+2];
      bool inside=false;
      if (st==2.0f) inside=dot(vec,vec)<sx*sx;
      else if (st==3.0f) {
        float zc=clamp(pl.z,-sy,sy);
        inside=pl.x*pl.x+pl.y*pl.y+(pl.z-zc)*(pl.z-zc)<sx*sx;
      }
      else if (st==4.0f) inside=(pl.x*pl.x/(sx*sx)+pl.y*pl.y/(sy*sy)+pl.z*pl.z/(sz*sz))<1.0f;
      else if (st==5.0f) inside=(abs(pl.z)<sy && pl.x*pl.x+pl.y*pl.y<sx*sx);
      else if (st==6.0f) inside=(abs(pl.x)<sx && abs(pl.y)<sy && abs(pl.z)<sz);
      else if (st==0.0f) inside=(pl.z<0.0f);
      value[0]=inside ? 1.0f : 0.0f;
    }
  }
  else if (typ==43 || typ==44) {
    // Pinned mj_energyPos / mj_energyVel (no flex, no sleep).
    if (typ==43) {
      float e=0.0f;
      float dis_spring=econst[6], dis_gravity=econst[8];
      if (dis_gravity==0.0f) {
        float3 g=float3(econst[0],econst[1],econst[2]);
        for (uint b=1;b<nbody;++b)
          e-=massub[b*5+0]*dot(g,read3(inertial_pos,(world*nbody+b)*3));
      }
      if (dis_spring==0.0f) {
        uint nq2=uint(econst[9]);
        for (uint j=0;j<njnt;++j) {
          int jt=jnt_meta[j*4+0];
          float k=econst[10+nq2+j*3+0], p0=econst[10+nq2+j*3+1], p1=econst[10+nq2+j*3+2];
          if (k==0.0f && p0==0.0f && p1==0.0f) continue;
          if (jt==2 || jt==3) {
            uint q=uint(jnt_meta[j*4+1]);
            float x=qpos[qb+q]-econst[10+q];
            e+=0.5f*k*x*x+p0/3.0f*x*x*x+p1/4.0f*x*x*x*x;
          } else if (jt==0) {
            uint q=uint(jnt_meta[j*4+1]);
            float3 dif=read3(qpos,qb+q)-float3(econst[10+q],econst[10+q+1],econst[10+q+2]);
            float x=length(dif);
            e+=0.5f*k*x*x+p0/3.0f*x*x*x+p1/4.0f*x*x*x*x;
            float4 qc=qunit(read4(qpos,qb+q+3)), qr=qunit(float4(econst[10+q+3],econst[10+q+4],econst[10+q+5],econst[10+q+6]));
            float3 dq=sub_quat(qc,qr);
            float xr=length(dq);
            e+=0.5f*k*xr*xr+p0/3.0f*xr*xr*xr+p1/4.0f*xr*xr*xr*xr;
          } else if (jt==1) {
            uint q=uint(jnt_meta[j*4+1]);
            float4 qc=qunit(read4(qpos,qb+q));
            float4 qr=qunit(float4(econst[10+q],econst[10+q+1],econst[10+q+2],econst[10+q+3]));
            float xr=length(sub_quat(qc,qr));
            e+=0.5f*k*xr*xr+p0/3.0f*xr*xr*xr+p1/4.0f*xr*xr*xr*xr;
          }
        }
        for (uint t=0;t<nt;++t) {
          float k=ten_lim[t*9+5], pp0=ten_lim[t*9+6], pp1=ten_lim[t*9+7];
          if (k==0.0f && pp0==0.0f && pp1==0.0f) continue;
          float L=0.0f;
          for (uint q=0;q<nq;++q) L+=ten_fix_lmap[t*nq+q]*qpos[qb+q];
          L+=ten_spa_len[world*nt+t];
          float lo=ten_lim[t*9+3], hi=ten_lim[t*9+4];
          float x=L>hi ? L-hi : (L<lo ? L-lo : 0.0f);
          e+=0.5f*k*x*x+pp0/3.0f*x*x*x+pp1/4.0f*x*x*x*x;
        }
      }
      value[0]=e;
    } else {
      float e=0.0f;
      if (has_mass!=0) {
        for (uint r=0;r<nv;++r) {
          float s=0.0f;
          for (uint c=0;c<nv;++c) s+=mass_matrix[(world*nv+r)*nv+c]*qvel[vb+c];
          e+=0.5f*s*qvel[vb+r];
        }
      }
      value[0]=e;
    }
  }
  else if (typ==8) {
    // Pinned cam_project (R07a, no rendering): site target through the
    // body-mounted camera frame. Camera constants ride the econst tail at
    // 10+nq+3*njnt: [ncam] then per camera [bodyid, lpos(3), lquat(4),
    // res(2), fovy, intrinsic(4), sensorsize(2)].
    float3 sp=read3(site_pos,(world*nsite+uint(objid))*3);
    uint cam=uint(meta[i*10+9]);
    uint cb=uint(10+nq+3*njnt);
    uint base=cb+1+cam*13;
    uint cbody=uint(econst[base+0]);
    if (cbody>=nbody) cbody=0;
    float3 bpos=read3(body_pos,(world*nbody+cbody)*3);
    float4 bq=qunit(read4(body_quat,(world*nbody+cbody)*4));
    float4 lq=qunit(float4(econst[base+4],econst[base+5],econst[base+6],econst[base+7]));
    float3 campos=bpos+qrot(bq,float3(econst[base+1],econst[base+2],econst[base+3]));
    float4 wq=qunit(qmul(bq,lq));
    float x=wq.y, y=wq.z, z=wq.w, w=wq.x;
    // xmat rows (pinned rotation[i][j] = xmat[j*3+i] = R[i][j]).
    float3 r0=float3(1-2*(y*y+z*z),2*(x*y-z*w),2*(x*z+y*w));
    float3 r1=float3(2*(x*y+z*w),1-2*(x*x+z*z),2*(y*z-x*w));
    float3 r2=float3(2*(x*z-y*w),2*(y*z+x*w),1-2*(x*x+y*y));
    float3 p=sp-campos;
    float3 q=float3(dot(r0,p),dot(r1,p),dot(r2,p));
    float res0=econst[base+8], res1=econst[base+9], fovy=econst[base+10];
    float ss0=econst[base+15], ss1=econst[base+16];
    float fx, fy;
    if (ss0!=0.0f && ss1!=0.0f) {
      fx=econst[base+11]/ss0*res0;
      fy=econst[base+12]/ss1*res1;
    } else {
      fx=fy=0.5f/tan(fovy*3.14159265f/360.0f)*res1;
    }
    float3 px=float3(-fx*q.x,fy*q.y,q.z);
    float3 img=float3(px.x+res0*0.5f*px.z,px.y+res1*0.5f*px.z,px.z);
    float denom=img.z;
    if (abs(denom)<1e-15f) denom=denom<0.0f ? min(denom,-1e-15f) : max(denom,1e-15f);
    value[0]=img.x/denom; value[1]=img.y/denom;
  }
  float cutoff=as_type<float>(meta[i*10+7]);
  if (cutoff>0 && meta[i*10+1]==0) {
    for (uint j=0;j<dim;j++) value[j]=clamp(value[j],-cutoff,cutoff);
  } else if (cutoff>0 && meta[i*10+1]==1) {
    for (uint j=0;j<dim;j++) value[j]=min(value[j],cutoff);
  }
  uint base=world*ndata+adr;
  for (uint j=0;j<dim && j<6;j++) output[base+j]=value[j];
}
