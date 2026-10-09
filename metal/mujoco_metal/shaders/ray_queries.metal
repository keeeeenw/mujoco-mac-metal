// Copyright 2026 The MuJoCo Metal contributors
// Licensed under the Apache License, Version 2.0.
// Compiled after shared spatial/SDF helpers. One thread owns one ray.

inline float qr_mesh(float3 pos, float3x3 mat, float3 size,
    float3 origin, float3 direction, device const float* hull,
    device const int* info, int geom, thread float3& normal) {
  float3 ignored;
  if (rg_analytic(6, pos, mat, size, origin, direction, ignored) < 0.0f)
    return -1.0f;
  // ray_triangle projects into a plane normal to the ray, not the triangle.
  float3 seed = float3(1.0f);
  int axis = abs(direction.x) >= abs(direction.y)
      && abs(direction.x) >= abs(direction.z) ? 0
      : (abs(direction.y) >= abs(direction.z) ? 1 : 2);
  seed[axis] = 0.0f;
  float3 b1 = normalize(seed - direction*dot(seed,direction)/dot(direction,direction));
  float3 b0 = normalize(cross(b1,direction));
  int voff = info[9*geom], ioff = 3*info[9*geom+3];
  float distance = -1.0f;
  for (int face = 0; face < info[9*geom+4]; ++face) {
    float3 v[3];
    for (int k=0;k<3;++k) {
      int vertex_index = int(hull[ioff+3*face+k]);
      v[k] = pos + mat*sp_r3(hull,3*(voff+vertex_index));
    }
    float3 n;
    float candidate = rg_triangle(v[0],v[1],v[2],origin,direction,b0,b1,n);
    if (candidate >= 0.0f && (distance < 0.0f || candidate < distance)) {
      distance = candidate;
      normal = n;
    }
  }
  return distance;
}

inline float qr_sdf(float3 pos, float3x3 mat, float3 size,
    float3 origin, float3 direction, int geom,
    device const int* kind, device const float* attributes,
    device const float* attributes_low, device const float* octree,
    device const int* octinfo, thread float3& normal, thread int& status) {
  float3 ignored;
  if (rg_analytic(6,pos,mat,size,origin,direction,ignored) < 0.0f) return -1.0f;
  float3 start = transpose(mat)*(origin-pos);
  float3 unit = normalize(transpose(mat)*direction);
  float a[5], low[5];
  for (int k=0;k<5;++k) {
    a[k] = attributes[5*geom+k];
    low[k] = attributes_low[5*geom+k];
  }
  float total = 0.0f;
  for (int iteration=0;iteration<40;++iteration) {
    float3 p = start + unit*total;
    float distance = kind[geom] > 0
        ? sjvalue(sjdistance(kind[geom],p,a,low))
        : sdf_oct_dist(octree,octinfo[2*geom],octinfo[2*geom+1],p);
    if (!isfinite(distance)) { status=2; return -1.0f; }
    distance=abs(distance);
    total+=distance;
    if (distance < 1e-7f) {
      p=start+unit*total;
      float3 g = kind[geom] > 0
          ? plugin_sdf_gradient(kind[geom],p,a,low)
          : sdf_oct_grad(octree,octinfo[2*geom],octinfo[2*geom+1],p);
      if (!all(isfinite(g))) { status=2; return -1.0f; }
      float length_g=length(g);
      normal=mat*(length_g < 1e-15f ? float3(1,0,0) : g/length_g);
      // Pinned mj_raySdf normalizes its local direction and returns physical
      // distance, including for non-unit global vectors.
      return total;
    }
    if (distance > 1e6f) break;
  }
  return -1.0f;
}

// Both public scene rays and stored site sensors consume these exact scene
// filters, triangle/heightfield intersections and SDF march.
inline float qr_scene_hit(float3 origin,float3 direction,int world,int ng,
    device const float* geom_pos,device const float* geom_quat,
    device const int* metadata,device const float* sizes,
    device const float* hull,device const int* hull_info,
    device const float* octree,device const int* octinfo,
    device const int* sdf_kind,device const float* sdf_attributes,
    device const float* sdf_attributes_low,device const float* rbound,
    device const int* groups,int flg_static,int bodyexclude,int geomid,
    bool has_cutoff,float cutoff,bool multiray,
    thread int& nearest,thread float3& best_normal,thread int& status) {
  float best=-1.0f;
  nearest=-1;best_normal=float3(0.0f);status=0;
  // The upstream single and multiple ray APIs intentionally differ here.
  if (!all(isfinite(origin)) || !all(isfinite(direction))
      || (multiray ? dot(direction,direction)<1e-15f : length(direction)<1e-15f))
    status=1;
  for (int g=0;g<ng && status==0;++g) {
    int type=metadata[6*g],body=metadata[6*g+1];
    if (geomid>=0) { if (g!=geomid) continue; }
    else if (body==bodyexclude || !metadata[6*g+4]
             || (!flg_static && metadata[6*g+2])
             || (groups && !groups[metadata[6*g+3]])) continue;
    float3 pos=sp_r3(geom_pos,3*(world*ng+g));
    float4 quat=sp_r4(geom_quat,4*(world*ng+g));
    if (!all(isfinite(pos)) || !all(isfinite(quat)) || length(quat)<1e-15f) {
      status=1;break;
    }
    if (has_cutoff && metadata[6*g+5] && length(pos-origin)>cutoff+rbound[g]) continue;
    float3x3 mat=float3x3(sp_qrot(quat,float3(1,0,0)),
                         sp_qrot(quat,float3(0,1,0)),sp_qrot(quat,float3(0,0,1)));
    float3 size=sp_r3(sizes,3*g),n=float3(0.0f);
    float candidate;
    if (type==7) candidate=qr_mesh(pos,mat,size,origin,direction,hull,hull_info,g,n);
    else if (type==1) candidate=rg_hfield(pos,mat,hull,hull_info,g,origin,direction,n);
    else if (type==8) candidate=qr_sdf(pos,mat,size,origin,direction,g,sdf_kind,
        sdf_attributes,sdf_attributes_low,octree,octinfo,n,status);
    else candidate=rg_analytic(type,pos,mat,size,origin,direction,n);
    if (candidate>=0.0f && (best<0.0f || candidate<best)) {
      best=candidate;nearest=g;best_normal=n;
    }
  }
  return status==0 ? best : -1.0f;
}

kernel void query_scene_rays(
    device const float* geom_pos [[buffer(0)]],
    device const float* geom_quat [[buffer(1)]],
    device const int* metadata [[buffer(2)]],
    device const float* sizes [[buffer(3)]],
    device const float* hull [[buffer(4)]],
    device const int* hull_info [[buffer(5)]],
    device const float* octree [[buffer(6)]],
    device const int* octinfo [[buffer(7)]],
    device const int* sdf_kind [[buffer(8)]],
    device const float* sdf_attributes [[buffer(9)]],
    device const float* sdf_attributes_low [[buffer(10)]],
    device const float* rbound [[buffer(11)]],
    device const float* origins [[buffer(12)]],
    device const float* vectors [[buffer(13)]],
    device const int* groups [[buffer(14)]],
    device const float* cutoff [[buffer(15)]],
    constant int* dims [[buffer(16)]],
    device float* distance [[buffer(17)]],
    device int* hit [[buffer(18)]],
    device float* normal [[buffer(19)]],
    device int* out_status [[buffer(20)]],
    uint index [[thread_position_in_grid]]) {
  int batch=dims[0], nray=dims[1], ng=dims[2];
  if (index>=uint(batch*nray)) return;
  int world=int(index)/nray;
  float3 origin=sp_r3(origins,3*index), direction=sp_r3(vectors,3*index);
  int status=0, nearest=-1;
  float3 best_normal=float3(0.0f);
  float best=qr_scene_hit(origin,direction,world,ng,geom_pos,geom_quat,
      metadata,sizes,hull,hull_info,octree,octinfo,sdf_kind,sdf_attributes,
      sdf_attributes_low,rbound,groups,dims[3],dims[4],dims[5],bool(dims[6]),
      cutoff[0],bool(dims[7]),nearest,best_normal,status);
  distance[index]=status==0 ? best : -1.0f;
  hit[index]=status==0 ? nearest : -1;
  normal[3*index]=status==0 ? best_normal.x : 0.0f;
  normal[3*index+1]=status==0 ? best_normal.y : 0.0f;
  normal[3*index+2]=status==0 ? best_normal.z : 0.0f;
  out_status[index]=status;
}

inline float3 qr_camera_unit(float3 v) {
  float size=length(v);
  return size<1e-15f ? float3(1,0,0) : v/size;
}

inline float4 qr_camera_qmul(float4 a,float4 b) {
  return float4(a.x*b.x-dot(a.yzw,b.yzw),
                a.x*b.yzw+b.x*a.yzw+cross(a.yzw,b.yzw));
}

inline float3 qr_camera_subtree(int root,int world,int nb,
    device const float* inertial_pos,device const int* body_tree,
    device const float* body_mass) {
  float mass=0.0f;
  float3 weighted=float3(0.0f);
  for (int body=root;body<nb;++body) {
    int parent=body;
    while (parent>root) parent=body_tree[2*parent];
    if (parent!=root) continue;
    mass+=body_mass[body];
    weighted+=body_mass[body]*sp_r3(inertial_pos,3*(world*nb+body));
  }
  return mass>1e-15f ? weighted/mass : sp_r3(inertial_pos,3*(world*nb+root));
}

inline void qr_camera_frame(int cam,int world,int nb,
    device const int* meta,device const float* values,
    device const float* body_pos,device const float* body_quat,
    device const float* inertial_pos,device const int* body_tree,
    device const float* body_mass,thread float3& pos,thread float3x3& mat) {
  int body=meta[6*cam],mode=meta[6*cam+1],target=meta[6*cam+2];
  float4 bq=sp_qunit(sp_r4(body_quat,4*(world*nb+body)));
  pos=sp_r3(body_pos,3*(world*nb+body))+sp_qrot(bq,sp_r3(values,27*cam));
  float4 q=sp_qunit(qr_camera_qmul(bq,sp_r4(values,27*cam+3)));
  mat=float3x3(sp_qrot(q,float3(1,0,0)),sp_qrot(q,float3(0,1,0)),sp_qrot(q,float3(0,0,1)));
  if (mode==1 || mode==2) {
    mat=transpose(float3x3(sp_r3(values,27*cam+13),
                          sp_r3(values,27*cam+16),sp_r3(values,27*cam+19)));
    pos=(mode==1 ? sp_r3(body_pos,3*(world*nb+body)) :
          qr_camera_subtree(body,world,nb,inertial_pos,body_tree,body_mass))
        +sp_r3(values,27*cam+(mode==1 ? 7 : 10));
  } else if ((mode==3 || mode==4) && target>=0) {
    float3 look=mode==3 ? sp_r3(body_pos,3*(world*nb+target)) :
        qr_camera_subtree(target,world,nb,inertial_pos,body_tree,body_mass);
    float3 z=qr_camera_unit(pos-look);
    float3 x=qr_camera_unit(cross(float3(0,0,1),z));
    float3 y=qr_camera_unit(cross(z,x));
    mat=float3x3(x,y,z);
  }
}

kernel void evaluate_site_rays(
    device const float* geom_pos [[buffer(0)]],
    device const float* geom_quat [[buffer(1)]],
    device const int* metadata [[buffer(2)]],
    device const float* sizes [[buffer(3)]],
    device const float* hull [[buffer(4)]],
    device const int* hull_info [[buffer(5)]],
    device const float* octree [[buffer(6)]],
    device const int* octinfo [[buffer(7)]],
    device const int* sdf_kind [[buffer(8)]],
    device const float* sdf_attributes [[buffer(9)]],
    device const float* sdf_attributes_low [[buffer(10)]],
    device const float* rbound [[buffer(11)]],
    device const float* site_pos [[buffer(12)]],
    device const float* site_quat [[buffer(13)]],
    device const int* site_bodyid [[buffer(14)]],
    device const int* sensors [[buffer(15)]],
    device const int* intprm [[buffer(16)]],
    device float* output [[buffer(17)]],
    constant int* stage_mask [[buffer(18)]],
    constant int* dims [[buffer(19)]],
    device const int* ray_entries [[buffer(20)]],
    device const int* camera_meta [[buffer(21)]],
    device const float* camera_values [[buffer(22)]],
    device const float* body_pos [[buffer(23)]],
    device const float* body_quat [[buffer(24)]],
    device const float* inertial_pos [[buffer(25)]],
    device const int* body_tree [[buffer(26)]],
    device const float* body_mass [[buffer(27)]],
    uint index [[thread_position_in_grid]]) {
  uint batch=uint(dims[0]),nsensor=uint(dims[1]),ndata=uint(dims[2]);
  uint nray=uint(dims[6]);
  if (index>=batch*nray || !(stage_mask[0] & (1<<1)) || (dims[5]&8192)) return;
  uint world=index/nray,entry=index%nray,i=uint(ray_entries[3*entry]);
  if (!stage_mask[1+batch*max(nsensor,1u)+world] || !stage_mask[1+world*nsensor+i]) return;
  if (sensors[10*i]!=7 || sensors[10*i+2]!=1) return;
  uint site=uint(sensors[10*i+4]),nsite=uint(dims[4]);
  uint base=world*ndata+uint(sensors[10*i+6]);
  bool camera=sensors[10*i+3]==7; // mjOBJ_CAMERA; sites are mjOBJ_SITE.
  float3 origin,direction,cam_pos,cam_z;
  int excluded;
  if (camera) {
    int cam=int(site),col=ray_entries[3*entry+1],row=ray_entries[3*entry+2];
    float3x3 mat;
    qr_camera_frame(cam,int(world),dims[7],camera_meta,camera_values,
        body_pos,body_quat,inertial_pos,body_tree,body_mass,cam_pos,mat);
    cam_z=mat[2];
    float fx=camera_values[27*cam+22],fy=camera_values[27*cam+23];
    float px=float(col)+0.5f-camera_values[27*cam+24];
    float py=float(row)+0.5f-camera_values[27*cam+25];
    if (camera_meta[6*cam+3]==0) {
      origin=cam_pos;
      direction=qr_camera_unit(mat*float3(px/fx,-py/fy,-1.0f));
    } else {
      float half_extent=camera_values[27*cam+26]/2.0f;
      origin=cam_pos+mat*float3(px/fx*half_extent,-py/fy*half_extent,0.0f);
      direction=-mat[2];
    }
    excluded=camera_meta[6*cam];
    int spec=intprm[3*i],size=0;
    for (int bit=0;bit<6;++bit) if (spec&(1<<bit)) size+=(bit==0 || bit==5) ? 1 : 3;
    base+=uint((row*camera_meta[6*cam+4]+col)*size);
  } else {
    origin=sp_r3(site_pos,3*(world*nsite+site));
    float4 quat=sp_qunit(sp_r4(site_quat,4*(world*nsite+site)));
    direction=sp_qrot(quat,float3(0,0,1));
    excluded=site_bodyid[site];
  }
  int gid,status;
  float3 normal;
  float distance=qr_scene_hit(origin,direction,int(world),dims[3],geom_pos,geom_quat,
      metadata,sizes,hull,hull_info,octree,octinfo,sdf_kind,sdf_attributes,
      sdf_attributes_low,rbound,nullptr,1,excluded,-1,false,0.0f,
      camera && camera_meta[6*int(site)+3]==0,
      gid,normal,status);
  bool hit=distance>=0.0f && status==0;
  float3 point=hit ? origin+direction*distance : float3(0.0f);
  int spec=intprm[3*i],offset=0;
  if (spec&1) output[base+offset++]=distance;
  if (spec&2) {
    float3 v=hit ? direction : float3(0.0f);
    for (int k=0;k<3;++k) output[base+offset++]=v[k];
  }
  if (spec&4) for (int k=0;k<3;++k) output[base+offset++]=origin[k];
  if (spec&8) for (int k=0;k<3;++k) output[base+offset++]=point[k];
  if (spec&16) for (int k=0;k<3;++k) output[base+offset++]=hit ? normal[k] : 0.0f;
  if (spec&32) output[base+offset++]=!hit ? -1.0f :
      camera ? -dot(point-cam_pos,cam_z) : distance;
}

kernel void query_analytic_rays(
    device const float* pos [[buffer(0)]],
    device const float* mat [[buffer(1)]],
    device const float* sizes [[buffer(2)]],
    device const float* origins [[buffer(3)]],
    device const float* vectors [[buffer(4)]],
    constant int* dims [[buffer(5)]],
    device float* distance [[buffer(6)]],
    device float* normals [[buffer(7)]],
    device int* out_status [[buffer(8)]],
    uint world [[thread_position_in_grid]]) {
  if (world>=uint(dims[0])) return;
  float3 origin=sp_r3(origins,3*world), direction=sp_r3(vectors,3*world);
  float3 position=sp_r3(pos,3*world), size=sp_r3(sizes,3*world);
  float3 r0=sp_r3(mat,9*world), r1=sp_r3(mat,9*world+3), r2=sp_r3(mat,9*world+6);
  int status=0;
  float hit=-1.0f;
  float3 normal=float3(0.0f);
  if (!all(isfinite(origin)) || !all(isfinite(direction))
      || !all(isfinite(position)) || !all(isfinite(size))
      || !all(isfinite(r0)) || !all(isfinite(r1)) || !all(isfinite(r2))
      || length(direction)<1e-15f) status=1;
  else {
    float3x3 rotation=transpose(float3x3(r0,r1,r2));
    hit=rg_analytic(dims[1],position,rotation,size,origin,direction,normal);
    if (!isfinite(hit) || !all(isfinite(normal))) status=2;
  }
  distance[world]=status==0 ? hit : -1.0f;
  normals[3*world]=status==0 ? normal.x : 0.0f;
  normals[3*world+1]=status==0 ? normal.y : 0.0f;
  normals[3*world+2]=status==0 ? normal.z : 0.0f;
  out_status[world]=status;
}

inline float3 qr_unit(float3 v) {
  float n=length(v);
  return n<1e-15f ? float3(1,0,0) : v/n;
}

inline void qr_ray_basis(float3 direction, thread float3& b0, thread float3& b1) {
  b0=float3(1.0f);
  if (abs(direction.x)>=abs(direction.y) && abs(direction.x)>=abs(direction.z)) b0.x=0;
  else if (abs(direction.y)>=abs(direction.z)) b0.y=0;
  else b0.z=0;
  b1=qr_unit(b0-direction*(dot(direction,b0)/dot(direction,direction)));
  b0=qr_unit(cross(b1,direction));
}

kernel void query_surface_rays(
    device const float* vertices [[buffer(0)]],
    device const int* edges [[buffer(1)]],
    device const int* faces [[buffer(2)]],
    device const int* layers [[buffer(3)]],
    device const float* radius [[buffer(4)]],
    device const float* origins [[buffer(5)]],
    device const float* vectors [[buffer(6)]],
    constant int* dims [[buffer(7)]],
    device float* distances [[buffer(8)]],
    device int* vertex_ids [[buffer(9)]],
    device float* normals [[buffer(10)]],
    device int* statuses [[buffer(11)]],
    uint world [[thread_position_in_grid]]) {
  if (world>=uint(dims[0])) return;
  int nv=dims[1], ne=dims[2], nf=dims[3], dim=dims[4];
  bool draw_vert=dims[5], draw_edge=dims[6], draw_face=dims[7], draw_skin=dims[8];
  float3 p=sp_r3(origins,3*world), d=sp_r3(vectors,3*world);
  float best=-1.0f;
  int nearest=-1, status=0;
  float3 normal=float3(0.0f), low=float3(0.0f), high=float3(0.0f);
  if (!all(isfinite(p)) || !all(isfinite(d)) || length(d)<1e-15f) status=1;
  for (int v=0;v<nv && status==0;++v) {
    float3 x=sp_r3(vertices,3*(world*nv+v));
    if (!all(isfinite(x))) { status=1; break; }
    low=v==0 ? x : min(low,x);
    high=v==0 ? x : max(high,x);
  }
  float3 b0,b1,n;
  float3x3 identity=float3x3(float3(1,0,0),float3(0,1,0),float3(0,0,1));
  bool in_bounds=false;
  if (status==0 && nv>0) {
    low-=radius[0]; high+=radius[0];
    in_bounds=rg_analytic(6,.5f*(low+high),identity,.5f*(high-low),p,d,n)>=0;
    qr_ray_basis(d,b0,b1);
  }
  if (in_bounds && (draw_edge || (dim>1 && draw_skin))) {
    for (int e=0;e<ne;++e) {
      int i=edges[2*e], j=edges[2*e+1];
      float3 v1=sp_r3(vertices,3*(world*nv+i)), v2=sp_r3(vertices,3*(world*nv+j));
      float3 z=qr_unit(v2-v1), x,y;
      qr_ray_basis(z,x,y);
      float3x3 rotation=float3x3(x,y,z);
      float hit=rg_analytic(3,.5f*(v1+v2),rotation,
          float3(radius[0],.5f*length(v2-v1),0),p,d,n);
      if (hit>=0 && (best<0 || hit<best)) {
        best=hit; normal=n;
        float3 point=p+d*hit;
        // Pinned capsule endpoint ties select the second vertex.
        nearest=length(v1-point)<length(v2-point) ? i : j;
      }
    }
  } else if (in_bounds && draw_vert && !(dim>1 && draw_skin)) {
    for (int v=0;v<nv;++v) {
      float hit=rg_analytic(2,sp_r3(vertices,3*(world*nv+v)),identity,
          float3(radius[0],0,0),p,d,n);
      if (hit>=0 && (best<0 || hit<best)) {
        best=hit; nearest=v; normal=n;
      }
    }
  }
  if (in_bounds && dim>1 && (draw_face || draw_skin)) {
    for (int f=0;f<nf;++f) {
      if (dim==3 && (draw_skin ? layers[f]>0 : layers[f]!=dims[9])) continue;
      int i=faces[3*f], j=faces[3*f+1], k=faces[3*f+2];
      float3 v0=sp_r3(vertices,3*(world*nv+i));
      float3 v1=sp_r3(vertices,3*(world*nv+j));
      float3 v2=sp_r3(vertices,3*(world*nv+k));
      float hit=rg_triangle(v0,v1,v2,p,d,b0,b1,n);
      if (hit>=0 && (best<0 || hit<best)) {
        best=hit; normal=n;
        float3 point=p+d*hit;
        float a=length(v0-point), b=length(v1-point), c=length(v2-point);
        nearest=a<=b && a<=c ? i : (b<=c ? j : k);
      }
    }
  }
  if (!isfinite(best) || !all(isfinite(normal))) status=2;
  distances[world]=status==0 ? best : -1.0f;
  vertex_ids[world]=status==0 ? nearest : -1;
  normals[3*world]=status==0 ? normal.x : 0.0f;
  normals[3*world+1]=status==0 ? normal.y : 0.0f;
  normals[3*world+2]=status==0 ? normal.z : 0.0f;
  statuses[world]=status;
}
