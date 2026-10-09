// Copyright 2026 The MuJoCo Metal contributors
// Licensed under the Apache License, Version 2.0.
// Shared mj_camlight frame construction for geometric camera consumers.
inline float3 mm_cam_r3(device const float* a,uint i) {
  return float3(a[i],a[i+1],a[i+2]);
}
inline float4 mm_cam_r4(device const float* a,uint i) {
  return float4(a[i],a[i+1],a[i+2],a[i+3]);
}
inline float3 mm_cam_unit(float3 v) {
  float n=length(v);
  return n<1e-15f ? float3(1,0,0) : v/n;
}
inline float4 mm_cam_qunit(float4 q) { return q/length(q); }
inline float4 mm_cam_qmul(float4 a,float4 b) {
  return float4(a.x*b.x-dot(a.yzw,b.yzw),a.x*b.yzw+b.x*a.yzw+cross(a.yzw,b.yzw));
}
inline float3 mm_cam_qrot(float4 q,float3 v) {
  return v+2.0f*cross(q.yzw,cross(q.yzw,v)+q.x*v);
}
inline float3 mm_cam_subtree(int root,int world,int nb,
    device const float* inertial_pos,device const int* body_tree,
    device const float* body_mass,int mass_stride) {
  float mass=0.0f;
  float3 weighted=float3(0.0f);
  for (int body=root;body<nb;++body) {
    int parent=body;
    while (parent>root) parent=body_tree[2*parent];
    if (parent!=root) continue;
    float m=body_mass[mass_stride*body];
    mass+=m;
    weighted+=m*mm_cam_r3(inertial_pos,3*(world*nb+body));
  }
  return mass>1e-15f ? weighted/mass : mm_cam_r3(inertial_pos,3*(world*nb+root));
}
inline void mm_camera_frame(int body,int mode,int target,int world,int nb,
    float3 local_pos,float4 local_quat,float3 pos0,float3 poscom0,float3x3 mat0,
    device const float* body_pos,device const float* body_quat,
    device const float* inertial_pos,device const int* body_tree,
    device const float* body_mass,int mass_stride,
    thread float3& pos,thread float3x3& mat) {
  float4 bq=mm_cam_qunit(mm_cam_r4(body_quat,4*(world*nb+body)));
  pos=mm_cam_r3(body_pos,3*(world*nb+body))+mm_cam_qrot(bq,local_pos);
  float4 q=mm_cam_qunit(mm_cam_qmul(bq,local_quat));
  mat=float3x3(mm_cam_qrot(q,float3(1,0,0)),mm_cam_qrot(q,float3(0,1,0)),
              mm_cam_qrot(q,float3(0,0,1)));
  if (mode==1 || mode==2) {
    mat=mat0;
    pos=(mode==1 ? mm_cam_r3(body_pos,3*(world*nb+body)) :
         mm_cam_subtree(body,world,nb,inertial_pos,body_tree,body_mass,mass_stride))
         +(mode==1 ? pos0 : poscom0);
  } else if ((mode==3 || mode==4) && target>=0) {
    float3 look=mode==3 ? mm_cam_r3(body_pos,3*(world*nb+target)) :
        mm_cam_subtree(target,world,nb,inertial_pos,body_tree,body_mass,mass_stride);
    float3 z=mm_cam_unit(pos-look);
    float3 x=mm_cam_unit(cross(float3(0,0,1),z));
    float3 y=mm_cam_unit(cross(z,x));
    mat=float3x3(x,y,z);
  }
}
