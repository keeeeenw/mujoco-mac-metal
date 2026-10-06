// Copyright 2026 The MuJoCo Metal contributors
// Licensed under the Apache License, Version 2.0.
// Device-side selected-row mj_resetData writes. Direct MSL writes preserve
// PyTorch's host mutation counter on tensors borrowed by forward-stage records.
#include <metal_stdlib>
using namespace metal;

kernel void reset_state_rows(
    device const int* reset_mask [[buffer(0)]],
    device float* qpos [[buffer(1)]], device const float* qpos0 [[buffer(2)]],
    device float* qvel [[buffer(3)]], device float* qacc [[buffer(4)]],
    device float* time [[buffer(5)]], device int* status [[buffer(6)]],
    device int* warning_number [[buffer(7)]],
    device int* warning_lastinfo [[buffer(8)]],
    device int* eq_active [[buffer(9)]], device const int* eq_active0 [[buffer(10)]],
    device float* mocap_pos [[buffer(11)]], device const float* mocap_pos0 [[buffer(12)]],
    device float* mocap_quat [[buffer(13)]], device const float* mocap_quat0 [[buffer(14)]],
    device float* act [[buffer(15)]], device float* history [[buffer(16)]],
    device const float* history0 [[buffer(17)]],
    device float* qacc_warmstart [[buffer(18)]],
    device float* userdata [[buffer(19)]], device const float* userdata0 [[buffer(20)]],
    device float* plugin_state [[buffer(21)]], device const float* plugin_state0 [[buffer(22)]],
    device int* row_epoch [[buffer(23)]], constant int* dims [[buffer(24)]],
    uint world [[thread_position_in_grid]]) {
  int batch=dims[0];
  if (world>=uint(batch) || reset_mask[world]==0) return;
  int nq=dims[1],nv=dims[2],neq=dims[3],nmocap=dims[4],na=dims[5];
  int nhistory=dims[6],nuserdata=dims[7],npluginstate=dims[8],nwarning=dims[9];
  uint qb=world*uint(nq),vb=world*uint(nv),wb=world*uint(nwarning);
  for(int i=0;i<nq;++i) qpos[qb+i]=qpos0[i];
  for(int i=0;i<nv;++i) { qvel[vb+i]=0.0f; qacc[vb+i]=0.0f; qacc_warmstart[vb+i]=0.0f; }
  time[world]=0.0f; status[world]=0;
  for(int i=0;i<nwarning;++i) { warning_number[wb+i]=0; warning_lastinfo[wb+i]=0; }
  for(int i=0;i<neq;++i) eq_active[world*uint(neq)+uint(i)]=eq_active0[i];
  for(int i=0;i<3*nmocap;++i) mocap_pos[world*uint(3*nmocap)+uint(i)]=mocap_pos0[i];
  for(int i=0;i<4*nmocap;++i) mocap_quat[world*uint(4*nmocap)+uint(i)]=mocap_quat0[i];
  for(int i=0;i<na;++i) act[world*uint(na)+uint(i)]=0.0f;
  for(int i=0;i<nhistory;++i) history[world*uint(nhistory)+uint(i)]=history0[i];
  for(int i=0;i<nuserdata;++i) userdata[world*uint(nuserdata)+uint(i)]=userdata0[i];
  for(int i=0;i<npluginstate;++i) plugin_state[world*uint(npluginstate)+uint(i)]=plugin_state0[i];
  row_epoch[world]+=1;
}

// Generic row clear for recovery-owned auxiliary state. It avoids Python
// tensor ops on healthy rows (and therefore preserves their version counters
// and borrowed-record provenance) while using the same device reset mask.
kernel void clear_float_rows(
    device const int* reset_mask [[buffer(0)]],
    device float* values [[buffer(1)]], constant int* dims [[buffer(2)]],
    uint index [[thread_position_in_grid]]) {
  int batch=dims[0], width=dims[1];
  if (width<=0) return;
  uint world=index/uint(width), column=index%uint(width);
  if (world>=uint(batch)) return;
  bool selected=reset_mask[world]!=0;
  if (dims[2]!=0) selected=!selected;
  if (!selected) return;
  values[world*uint(width)+column]=0.0f;
}

kernel void clear_int_rows(
    device const int* reset_mask [[buffer(0)]],
    device int* values [[buffer(1)]], constant int* dims [[buffer(2)]],
    uint index [[thread_position_in_grid]]) {
  int batch=dims[0], width=dims[1];
  if (width<=0) return;
  uint world=index/uint(width), column=index%uint(width);
  if (world>=uint(batch)) return;
  bool selected=reset_mask[world]!=0;
  if (dims[2]!=0) selected=!selected;
  if (!selected) return;
  values[world*uint(width)+column]=0;
}

// Selected-row force assembly. op=0 adds source into target; op=1 replaces
// target with sign*source. All unselected rows return before data access.
kernel void update_float_rows(
    device const int* reset_mask [[buffer(0)]],
    device float* target [[buffer(1)]], device const float* source [[buffer(2)]],
    constant int* dims [[buffer(3)]], uint index [[thread_position_in_grid]]) {
  int batch=dims[0], width=dims[1], op=dims[2], sign=dims[3];
  if (width<=0) return;
  uint world=index/uint(width), column=index%uint(width);
  if (world>=uint(batch) || reset_mask[world]==0) return;
  uint offset=world*uint(width)+column;
  if (op==0) target[offset]+=float(sign)*source[offset];
  else target[offset]=float(sign)*source[offset];
}

// Selected-row diagonal damping update used by prepared ACC assembly. The
// predicate precedes both source reads, so invalid cached rows never enter the
// arithmetic even when a batch shares one dispatch.
kernel void add_scaled_float_rows(
    device const int* reset_mask [[buffer(0)]],
    device float* target [[buffer(1)]],
    device const float* source [[buffer(2)]],
    device const float* scale [[buffer(3)]],
    constant int* dims [[buffer(4)]],
    uint index [[thread_position_in_grid]]) {
  int batch=dims[0], width=dims[1], sign=dims[2];
  if (width<=0) return;
  uint world=index/uint(width), column=index%uint(width);
  if (world>=uint(batch) || reset_mask[world]==0) return;
  uint offset=world*uint(width)+column;
  target[offset]+=float(sign)*source[offset]*scale[column];
}

kernel void copy_int_rows(
    device const int* reset_mask [[buffer(0)]],
    device int* target [[buffer(1)]], device const int* source [[buffer(2)]],
    constant int* dims [[buffer(3)]], uint index [[thread_position_in_grid]]) {
  int batch=dims[0], width=dims[1];
  if (width<=0) return;
  uint world=index/uint(width), column=index%uint(width);
  if (world>=uint(batch) || reset_mask[world]==0) return;
  uint offset=world*uint(width)+column;
  target[offset]=source[offset];
}

// Source-compatible selected-world mjWARN_BADQ* bookkeeping. The caller
// clears the selected warning row first when AUTORESET is enabled; otherwise
// each pinned warning increment is retained here as a +2 update.
kernel void record_warning_rows(
    device const int* selected [[buffer(0)]],
    device const int* first_bad [[buffer(1)]],
    device int* warning_number [[buffer(2)]],
    device int* warning_lastinfo [[buffer(3)]],
    constant int* dims [[buffer(4)]],
    uint world [[thread_position_in_grid]]) {
  int batch=dims[0], nwarning=dims[1], warning=dims[2], autoreset=dims[3];
  if (world>=uint(batch) || selected[world]==0) return;
  uint offset=world*uint(nwarning)+uint(warning);
  if (autoreset) warning_number[offset]=1;
  else warning_number[offset]+=2;
  warning_lastinfo[offset]=first_bad[world];
}

kernel void invalidate_bool_rows(
    device const int* selected [[buffer(0)]],
    device uchar* values [[buffer(1)]],
    constant int* dims [[buffer(2)]],
    uint index [[thread_position_in_grid]]) {
  int batch=dims[0];
  if (index>=uint(batch) || selected[index]==0) return;
  values[index]=0;
}

kernel void copy_strided_float_rows(
    device const int* selected [[buffer(0)]],
    device float* target [[buffer(1)]],
    device const float* source [[buffer(2)]],
    constant int* dims [[buffer(3)]],
    uint index [[thread_position_in_grid]]) {
  int batch=dims[0], target_width=dims[1], width=dims[2], offset=dims[3];
  if (width<=0) return;
  uint world=index/uint(width), column=index%uint(width);
  if (world>=uint(batch) || selected[world]==0) return;
  target[world*uint(target_width)+uint(offset)+column]=
      source[world*uint(width)+column];
}

// Copy a contiguous packed row family into a selected batch's larger packed
// row owner. The row selector returns before touching either source or target.
kernel void copy_masked_packed_rows(
    device const int* selected [[buffer(0)]],
    device float* target [[buffer(1)]],
    device const float* source [[buffer(2)]],
    constant int* dims [[buffer(3)]],
    uint index [[thread_position_in_grid]]) {
  int batch=dims[0], rows=dims[1], width=dims[2];
  int source_rows=dims[3], target_rows=dims[4], target_row=dims[5];
  if (rows<=0 || width<=0) return;
  uint row_width=uint(rows*width);
  uint world=index/row_width, local=index%row_width;
  if (world>=uint(batch) || selected[world]==0) return;
  uint source_offset=world*uint(source_rows*width)+local;
  uint target_offset=world*uint(target_rows*width)
      +uint(target_row*width)+local;
  target[target_offset]=source[source_offset];
}
