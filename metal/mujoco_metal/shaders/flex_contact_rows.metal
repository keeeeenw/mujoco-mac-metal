#include <metal_stdlib>
using namespace metal;

constant int FLEX_CCJ_MAGIC = 0x4d4a4353;

inline void flex_ccj_set(device float* record, int row, int dof,
                         float value, int nv) {
  device int* h = reinterpret_cast<device int*>(record);
  if (h[0] != FLEX_CCJ_MAGIC || h[1] != 1 || h[2] != 1 || h[3] < 0
      || h[4] != nv || h[5] < 0 || h[6] < 12 || h[7] < h[6] + h[3] + 1
      || h[8] < h[7] + h[5] || h[8] + h[5] > h[9] || h[9] < 12) {
    atomic_store_explicit(reinterpret_cast<device atomic_int*>(&h[10]), 1,
                          memory_order_relaxed);
    return;
  }
  if (row < 0 || row >= h[3] || dof < 0 || dof >= nv) {
    if (value != 0.0f)
      atomic_store_explicit(reinterpret_cast<device atomic_int*>(&h[10]), 1,
                            memory_order_relaxed);
    return;
  }
  device const int* rowptr = reinterpret_cast<device const int*>(record) + h[6];
  device const int* columns = reinterpret_cast<device const int*>(record) + h[7];
  int lo = rowptr[row], hi = rowptr[row + 1];
  if (lo < 0 || hi < lo || hi > h[5]) {
    atomic_store_explicit(reinterpret_cast<device atomic_int*>(&h[10]), 1,
                          memory_order_relaxed);
    return;
  }
  while (lo < hi) {
    int mid = (lo + hi) >> 1;
    if (columns[mid] < 0 || columns[mid] >= nv) {
      atomic_store_explicit(reinterpret_cast<device atomic_int*>(&h[10]), 1,
                            memory_order_relaxed);
      return;
    }
    if (columns[mid] < dof) lo = mid + 1;
    else hi = mid;
  }
  if (lo >= rowptr[row + 1] || columns[lo] != dof) {
    if (value != 0.0f)
      atomic_store_explicit(reinterpret_cast<device atomic_int*>(&h[10]), 1,
                            memory_order_relaxed);
    return;
  }
  record[h[8] + lo] = value;
}

inline float flex_ccj_get(device const float* record, int row, int dof,
                          int nv) {
  device const int* h = reinterpret_cast<device const int*>(record);
  if (h[0] != FLEX_CCJ_MAGIC || h[1] != 1 || h[2] != 1 || h[3] < 0
      || h[4] != nv || h[5] < 0 || h[6] < 12 || h[7] < h[6] + h[3] + 1
      || h[8] < h[7] + h[5] || h[8] + h[5] > h[9] || h[9] < 12
      || row < 0 || row >= h[3] || dof < 0 || dof >= nv) return 0.0f;
  device const int* rowptr = reinterpret_cast<device const int*>(record) + h[6];
  device const int* columns = reinterpret_cast<device const int*>(record) + h[7];
  int lo = rowptr[row], hi = rowptr[row + 1];
  if (lo < 0 || hi < lo || hi > h[5]) return 0.0f;
  while (lo < hi) {
    int mid = (lo + hi) >> 1;
    if (columns[mid] < 0 || columns[mid] >= nv) return 0.0f;
    if (columns[mid] < dof) lo = mid + 1;
    else hi = mid;
  }
  return (lo < rowptr[row + 1] && columns[lo] == dof)
      ? record[h[8] + lo] : 0.0f;
}

// Assemble the canonical MuJoCo constraint rows for fixed flex contact slots.
// One thread owns a slot, so each row span is written once and never compacted.
inline float flex_contact_impedance(float distance, float margin,
                                   thread const float* imp) {
  float d0=clamp(imp[0],1.0e-4f,0.9999f);
  float d1=clamp(imp[1],1.0e-4f,0.9999f);
  float width=max(imp[2],0.0f);
  float mid=clamp(imp[3],1.0e-4f,0.9999f);
  float power=max(imp[4],1.0f);
  if (d0==d1 || width<=1.0e-15f) return 0.5f*(d0+d1);
  float x=abs((distance-margin)/width);
  if (x>=1.0f) return d1;
  if (x<=0.0f) return d0;
  float y;
  if (power==1.0f) y=x;
  else if (x<=mid) y=pow(x,power)/pow(mid,power-1.0f);
  else y=1.0f-pow(1.0f-x,power)/pow(1.0f-mid,power-1.0f);
  return clamp(d0+y*(d1-d0),1.0e-4f,0.9999f);
}

inline void flex_contact_kb(float2 ref, float impedance_max,
                            bool friction_ellipse, float2 ref_friction,
                            bool refsafe, float timestep,
                            thread float& K, thread float& B) {
  float2 used=ref;
  if (friction_ellipse && (ref_friction.x!=0.0f || ref_friction.y!=0.0f))
    used=ref_friction;
  if (refsafe && used.x>0.0f) used.x=max(used.x,2.0f*timestep);
  if (friction_ellipse) K=0.0f;
  else if (used.x>0.0f)
    K=1.0f/max(1.0e-15f,impedance_max*impedance_max*used.x*used.x*used.y*used.y);
  else K=-used.x/max(1.0e-15f,impedance_max*impedance_max);
  if (used.y>0.0f) B=2.0f/max(1.0e-15f,impedance_max*used.x);
  else B=-used.y/max(1.0e-15f,impedance_max);
}

kernel void flex_contact_assemble_rows(
    device const float* spatial_J [[buffer(0)]],
    device const float* frame [[buffer(1)]],
    device const float* distance [[buffer(2)]],
    device const int* contact_active [[buffer(3)]],
    device const int* row_start [[buffer(4)]],
    device const int* row_span [[buffer(5)]],
    device const int* condim [[buffer(6)]],
    device const int* cone [[buffer(7)]],
    device const float* friction [[buffer(8)]],
    device const float* solref [[buffer(9)]],
    device const float* solreffriction [[buffer(10)]],
    device const float* solimp [[buffer(11)]],
    device const float* margin [[buffer(12)]],
    device const float* diagA [[buffer(13)]],
    device const float* qvel [[buffer(14)]],
    constant int* dims [[buffer(15)]],
    constant float* params [[buffer(16)]],
    device float* workspace_J [[buffer(17)]],
    device float* R [[buffer(18)]],
    device float* aref [[buffer(19)]],
    device float* lo [[buffer(20)]],
    device float* hi [[buffer(21)]],
    device int* active [[buffer(22)]],
    device float* position_context [[buffer(23)]],
    device int* row_owner [[buffer(24)]],
    device int* row_local [[buffer(25)]],
    device int* row_cone [[buffer(26)]],
    device float* row_friction [[buffer(27)]],
    device const float* relative_surface_velocity [[buffer(28)]],
    device float* row_surface_velocity [[buffer(29)]],
    uint index [[thread_position_in_grid]]) {
  int batch=dims[0], nslot=dims[1], nv=dims[2], nrow=dims[3];
  bool packed_mode=dims[4]!=0;
  int canonical_row_offset=dims[5];
  bool prewritten_packed=packed_mode && dims[6]!=0;
  if (index>=uint(batch*nslot)) return;
  int env=int(index)/nslot, slot=int(index)%nslot;
  // Recovery-masked worlds preserve their existing row/J/cache state.  Test
  // the per-world suffix before dereferencing any contact or solver buffers.
  if (dims[7+env]==0) return;
  device float* world_j=workspace_J;
  if (packed_mode) {
    device const int* first=reinterpret_cast<device const int*>(workspace_J);
    int stride=first[9];
    if (stride<12 || stride>0x7fffffff/max(env+1,1)) {
      atomic_store_explicit(reinterpret_cast<device atomic_int*>(
          const_cast<device int*>(first)+10),1,memory_order_relaxed);
      return;
    }
    world_j=workspace_J+env*stride;
  }
  int bslot=env*nslot+slot, first=row_start[slot];
  int span=row_span[slot], dim=condim[slot], cn=cone[slot];
  float dist=distance[bslot], incl=margin[slot];
  bool live=contact_active[bslot]!=0 && dist<incl;
  float impvals[5];
  for (int k=0;k<5;k++) impvals[k]=solimp[5*slot+k];
  float impedance=flex_contact_impedance(dist,incl,impvals);
  float2 ref=float2(solref[2*slot],solref[2*slot+1]);
  float2 reff=float2(solreffriction[2*bslot],solreffriction[2*bslot+1]);
  float impedance_max=clamp(impvals[1],1.0e-4f,0.9999f);
  float timestep=params[0], impratio=max(params[1],1.0e-15f);
  bool refsafe=params[2]>0.5f;
  float mu0=friction[5*slot];
  float mu_eff=mu0;
  float Rnormal=0.0f;
  if (span>0) {
    float da0=max(diagA[env*nrow+first],1.0e-15f);
    Rnormal=max(1.0e-15f,(1.0f-impedance)*da0/impedance);
    if (dim>1) mu_eff=mu0/sqrt(impratio);
  }
  for (int local=0; local<span; ++local) {
    int row=first+local;
    if (row<0 || row>=nrow) continue;
    if (!packed_mode)
      for (int k=0;k<nv;k++) workspace_J[(env*nrow+row)*nv+k]=0.0f;
    float rowpos=0.0f, rowmargin=0.0f, K=0.0f, B=0.0f;
    float rvalue=0.0f, lower=-INFINITY, upper=INFINITY;
    if (dim==1) {
      rowpos=dist; rowmargin=incl; lower=0.0f;
      flex_contact_kb(ref,impedance_max,false,reff,refsafe,timestep,K,B);
      float velocity=0.0f;
      for (int k=0;k<nv;k++) {
        float value=prewritten_packed
            ? flex_ccj_get(world_j,canonical_row_offset+row,k,nv) : 0.0f;
        if (!prewritten_packed) {
          for (int axis=0;axis<3;axis++)
            value+=frame[bslot*9+axis]*spatial_J[((bslot*6+axis)*nv)+k];
          if (packed_mode) flex_ccj_set(world_j,canonical_row_offset+row,k,value,nv);
          else workspace_J[(env*nrow+row)*nv+k]=value;
        }
        velocity+=value*qvel[env*nv+k];
      }
      rvalue=Rnormal;
    } else if (cn==1) {  // mjCONE_ELLIPTIC
      int axis=local;
      bool angular=axis>=3;
      int frame_axis=angular ? axis-3 : axis;
      if (axis==0) { rowpos=dist; rowmargin=incl; lower=0.0f; }
      flex_contact_kb(ref,impedance_max,axis>0,reff,refsafe,timestep,K,B);
      for (int k=0;k<nv;k++) {
        float value=prewritten_packed
            ? flex_ccj_get(world_j,canonical_row_offset+row,k,nv) : 0.0f;
        if (!prewritten_packed) {
          for (int c=0;c<3;c++) {
            int spatial_axis=(angular ? 3 : 0)+c;
            value+=frame[bslot*9+frame_axis*3+c]
                *spatial_J[((bslot*6+spatial_axis)*nv)+k];
          }
          if (packed_mode)
            flex_ccj_set(world_j,canonical_row_offset+row,k,value,nv);
          else
            workspace_J[(env*nrow+row)*nv+k]=value;
        }
      }
      rvalue=local==0 ? Rnormal
          : max(1.0e-15f,(1.0f-impedance)*diagA[env*nrow+row]/impedance);
      if (dim>1 && local==1) rvalue=Rnormal/impratio;
      if (dim>2 && local>=2) {
        float mu_axis=friction[5*slot+local-1];
        rvalue=max(1.0e-15f,(Rnormal/impratio)*mu0*mu0
                    /(mu_axis*mu_axis));
      }
    } else {  // mjCONE_PYRAMIDAL
      int pair=local/2, sign=(local%2)==0 ? 1 : -1;
      int axis=pair+1;
      bool angular=axis>=3;
      int frame_axis=angular ? axis-3 : axis;
      float mu=friction[5*slot+pair];
      rowpos=dist; rowmargin=incl; lower=0.0f;
      flex_contact_kb(ref,impedance_max,false,reff,refsafe,timestep,K,B);
      for (int k=0;k<nv;k++) {
        if (!prewritten_packed) {
          float normal_value=0.0f, tangent_value=0.0f;
          for (int c=0;c<3;c++) {
            normal_value+=frame[bslot*9+c]
                *spatial_J[((bslot*6+c)*nv)+k];
            int spatial_axis=(angular ? 3 : 0)+c;
            tangent_value+=frame[bslot*9+frame_axis*3+c]
                *spatial_J[((bslot*6+spatial_axis)*nv)+k];
          }
          float value=normal_value+float(sign)*mu*tangent_value;
          if (packed_mode)
            flex_ccj_set(world_j,canonical_row_offset+row,k,value,nv);
          else
            workspace_J[(env*nrow+row)*nv+k]=value;
        }
      }
      rvalue=max(1.0e-15f,2.0f*mu_eff*mu_eff*Rnormal);
    }
    float velocity=0.0f, surface_velocity=0.0f;
    for (int k=0;k<nv;k++) {
      float value=packed_mode
          ? flex_ccj_get(world_j,canonical_row_offset+row,k,nv)
          : workspace_J[(env*nrow+row)*nv+k];
      velocity+=value*qvel[env*nv+k];
    }
    if (dim==1) {
      for (int c=0;c<3;c++)
        surface_velocity+=frame[bslot*9+c]
            *relative_surface_velocity[bslot*6+c];
    } else if (cn==1) {
      int frame_axis=local<3 ? local : local-3;
      int spatial_base=local<3 ? 0 : 3;
      for (int c=0;c<3;c++)
        surface_velocity+=frame[bslot*9+frame_axis*3+c]
            *relative_surface_velocity[bslot*6+spatial_base+c];
    } else {
      int pair=local/2, axis=pair+1;
      int spatial_base=axis<3 ? 0 : 3;
      int frame_axis=axis<3 ? axis : axis-3;
      float sign=(local%2)==0 ? 1.0f : -1.0f;
      for (int c=0;c<3;c++) {
        surface_velocity+=frame[bslot*9+c]
            *relative_surface_velocity[bslot*6+c];
        surface_velocity+=sign*friction[5*slot+pair]
            *frame[bslot*9+frame_axis*3+c]
            *relative_surface_velocity[bslot*6+spatial_base+c];
      }
    }
    position_context[(env*nrow+row)*5+0]=K;
    position_context[(env*nrow+row)*5+1]=B;
    // Elliptic tangential rows are created by mj_addConstraint with pos=0
    // and margin=0; only the normal row uses contact distance/includemargin.
    // Their impedance is therefore the d0 value, even when the normal row is
    // penetrating and has moved along the solimp curve.
    float row_impedance=(cn==1 && dim>1 && local>0)
        ? flex_contact_impedance(0.0f,0.0f,impvals) : impedance;
    position_context[(env*nrow+row)*5+2]=row_impedance;
    position_context[(env*nrow+row)*5+3]=rowpos;
    position_context[(env*nrow+row)*5+4]=rowmargin;
    row_surface_velocity[env*nrow+row]=surface_velocity;
    R[env*nrow+row]=rvalue;
    aref[env*nrow+row]=live
        ? -B*(velocity+surface_velocity)
            -K*row_impedance*(rowpos-rowmargin)
        : 0.0f;
    lo[env*nrow+row]=lower;
    hi[env*nrow+row]=upper;
    active[env*nrow+row]=live ? 1 : 0;
    row_owner[env*nrow+row]=slot;
    row_local[env*nrow+row]=local;
    row_cone[env*nrow+row]=cn;
    for (int k=0;k<5;k++) row_friction[5*(env*nrow+row)+k]=friction[5*slot+k];
  }
}

// Refresh aref after qvel changes while contact position/J remain cached.
kernel void flex_contact_refresh_aref(
    device const float* workspace_J [[buffer(0)]],
    device const float* qvel [[buffer(1)]],
    device const float* position_context [[buffer(2)]],
    device const int* active [[buffer(3)]],
    constant int* dims [[buffer(4)]],
    device const float* row_surface_velocity [[buffer(5)]],
    device float* aref [[buffer(6)]],
    uint index [[thread_position_in_grid]]) {
  int batch=dims[0], nrow=dims[1], nv=dims[2];
  if (index>=uint(batch*nrow)) return;
  int env=int(index)/nrow, row=int(index)%nrow;
  if (dims[5+env]==0) return;
  bool packed_mode=dims[3]!=0;
  int canonical_row_offset=dims[4];
  device const float* world_j=workspace_J;
  if (packed_mode) {
    device const int* first=reinterpret_cast<device const int*>(workspace_J);
    int stride=first[9];
    if (stride<12 || stride>0x7fffffff/max(env+1,1)) return;
    world_j=workspace_J+env*stride;
  }
  float velocity=0.0f;
  for (int k=0;k<nv;k++) {
    float value=packed_mode
        ? flex_ccj_get(world_j,canonical_row_offset+row,k,nv)
        : workspace_J[(env*nrow+row)*nv+k];
    velocity+=value*qvel[env*nv+k];
  }
  int base=(env*nrow+row)*5;
  aref[index]=active[index]
      ? -position_context[base+1]
          *(velocity+row_surface_velocity[index])
        -position_context[base+0]*position_context[base+2]
          *(position_context[base+3]-position_context[base+4])
      : 0.0f;
}
