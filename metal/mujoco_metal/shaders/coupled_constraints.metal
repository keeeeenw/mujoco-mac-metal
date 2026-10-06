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

// Per-call recovery mask packed into the typed int32 suffix of solver_dims.
// The header stores the absolute tensor index of the [batch] mask at word
// header+20. CC kernels with this dimensions ABI guard the world before
// touching its output or scratch.
inline bool cc_solver_world_enabled(constant int* dims, int world) {
  int batch = dims[5];
  if (world < 0 || world >= batch) return false;
  int header = dims[24] + dims[9] * 10;
  int mask_offset = dims[header + 20];
  return mask_offset >= 0 && dims[mask_offset + world] != 0;
}

inline int pair_contact_world_mask_offset(constant int* dims) {
  return 10 + dims[1] + 1 + dims[2];
}

inline bool pair_contact_world_enabled(constant int* dims, int world) {
  int mask_offset=dims[pair_contact_world_mask_offset(dims)];
  return world>=0 && world<dims[3] && mask_offset>=0
      && dims[mask_offset+world]!=0;
}

inline bool pair_contact_world_enabled(device const int* dims, int world) {
  int mask_offset=dims[10+dims[1]+1+dims[2]];
  return world>=0 && world<dims[3] && mask_offset>=0
      && dims[mask_offset+world]!=0;
}


// A common-CCD candidate failure invalidates every contact candidate in that
// world. Fold the device status into the reusable compaction mask so later
// pair/slot/link consumers cannot publish stale or partially generated rows.
kernel void mask_failed_candidate_worlds(
    device const int* world_status [[buffer(0)]],
    device int* world_mask [[buffer(1)]],
    constant int* dims [[buffer(2)]],
    uint world [[thread_position_in_grid]]) {
  if (int(world) >= dims[3]) return;
  world_mask[int(world)] =
      (world_mask[int(world)] != 0 && world_status[int(world)] == 0) ? 1 : 0;
}

kernel void clear_contact_candidate_workspace(
    device float* row_data [[buffer(0)]],
    device float* frame [[buffer(1)]],
    constant int* dims [[buffer(2)]],
    uint tid [[thread_position_in_grid]]) {
  int ncontacts=dims[2], batch=dims[3];
  int width=max(ncontacts,1)*36;
  int world=int(tid)/width, lane=int(tid)%width;
  if(world>=batch || !pair_contact_world_enabled(dims,world)) return;
  if(lane<ncontacts*36) row_data[world*ncontacts*36+lane]=0.0f;
  if(lane<ncontacts*12) frame[world*ncontacts*12+lane]=0.0f;
}

kernel void reset_selected_jacobian_status(
    device int* records [[buffer(0)]],
    constant int* dims [[buffer(1)]],
    uint world [[thread_position_in_grid]]) {
  int batch=dims[0], stride=dims[1];
  if(int(world)>=batch || dims[2+int(world)]==0) return;
  records[int(world)*stride+10]=0;
}

kernel void clear_selected_jacobian_values(
    device float* values [[buffer(0)]],
    constant int* dims [[buffer(1)]],
    uint tid [[thread_position_in_grid]]) {
  int batch=dims[0], width=dims[1], stride=dims[2], offset=dims[3];
  int world=int(tid)/width, lane=int(tid)%width;
  if(world>=batch || dims[4+world]==0) return;
  values[world*stride+offset+lane]=0.0f;
}

// ComponentMassSolver's borrowed mass argument is a contiguous pair
// `[high(B,R,nv), low(B,R,nv)]`. `R=nr+1` reserves the final smooth RHS.
inline int component_pair_index(int batch, int world, int row, int dof,
                                int nr, int nv, bool low) {
  int capacity = nr + 1;
  int plane = batch * capacity * nv;
  return (low ? plane : 0) + world * capacity * nv + row * nv + dof;
}

inline int solver_tree_root(device int* parent, int tree, int ntree) {
  if (tree < 0 || tree >= ntree) return -1;
  int root = tree;
  for (int step = 0; step < ntree; ++step) {
    int next = parent[root];
    if (next == root) break;
    root = next;
  }
  int node = tree;
  for (int step = 0; step < ntree; ++step) {
    int next = parent[node];
    if (next == node) break;
    parent[node] = root;
    node = next;
  }
  return root;
}

inline void solver_tree_union(device int* parent, device int* used,
                              int tree0, int tree1, int ntree) {
  if (tree0 < 0 && tree1 < 0) return;
  if (tree0 < 0) tree0 = tree1;
  if (tree1 < 0) tree1 = tree0;
  if (tree0 < 0 || tree1 < 0 || tree0 >= ntree || tree1 >= ntree) return;
  used[tree0] = 1;
  used[tree1] = 1;
  int root0 = solver_tree_root(parent, tree0, ntree);
  int root1 = solver_tree_root(parent, tree1, ntree);
  if (root0 != root1) parent[max(root0, root1)] = min(root0, root1);
}

inline bool solver_dof_awake(constant int* dims,
    device const int* awake_tree, int dof) {
  int tree_offset = dims[23];
  int tree_count = dims[22];
  int tree = dims[tree_offset + dof];
  // Unknown tree metadata must not silently suppress a solver DOF.
  if (tree < 0 || tree >= tree_count) return true;
  return awake_tree[tree] != 0;
}

// Read a mass entry from either the ordinary dense matrix or the immutable
// packed component operator. Sparse mode is selected only by stage -3. The
// component payload is carried in workspace_debug's per-world tail, so the
// compiled solver stays within Metal's 31-buffer ABI.
inline float solver_mass_entry(device const float* mass,
    device const float* debug, constant int* dims,
    device const int* awake_tree, int world, int nv, int i, int j) {
  if (dims[20] != -3) {
    return mass[i * nv + j];
  }
  bool awake_i = solver_dof_awake(dims, awake_tree, i);
  bool awake_j = solver_dof_awake(dims, awake_tree, j);
  if (!awake_i || !awake_j) return (!awake_i && i == j) ? 1.0f : 0.0f;
  int nr = dims[9];
  int header = dims[24] + nr * 10;
  int ncomponent = dims[header];
  int nnz = dims[header + 1];
  int armature_offset = dims[header + 2];
  int layout_offset = dims[header + 3];
  device const int* layout = reinterpret_cast<device const int*>(debug)
      + layout_offset;
  int dof_component_base = 2 * ncomponent + 1 + nv;
  int dof_local_base = dof_component_base + nv;
  int component_i = layout[dof_component_base + i];
  int component_j = layout[dof_component_base + j];
  if (component_i != component_j) return 0.0f;
  if (component_i < 0 || component_i >= ncomponent) return 0.0f;
  int local_i = layout[dof_local_base + i];
  int local_j = layout[dof_local_base + j];
  int width = layout[component_i + 1] - layout[component_i];
  if (local_i < 0 || local_i >= width || local_j < 0 || local_j >= width)
    return 0.0f;
  int mass_offset_base = ncomponent + 1 + nv;
  int block_offset = layout[mass_offset_base + component_i];
  int block_index = block_offset + local_i * width + local_j;
  return mass[world * nnz + block_index]
      + debug[armature_offset + block_index];
}

// Factor each compiled component block in its packed scratch region.  The
// armature view is temporary input staging; its lower triangle becomes the
// Cholesky factor, so this never materializes a global nv-by-nv mass matrix.
inline bool solver_factor_component_mass(device const float* mass,
    device float* debug, constant int* dims, device const int* awake_tree,
    int world, int nv) {
  int header = dims[24] + dims[9] * 10;
  int ncomponent = dims[header];
  int nnz = dims[header + 1];
  int armature_offset = dims[header + 2];
  int layout_offset = dims[header + 3];
  device const int* layout = reinterpret_cast<device const int*>(debug)
      + layout_offset;
  device float* factor = debug + armature_offset;
  int mass_base = world * nnz;
  int mass_offset_base = ncomponent + 1 + nv;
  for (int component = 0; component < ncomponent; ++component) {
    int begin = layout[component];
    int end = layout[component + 1];
    int width = end - begin;
    int block = layout[mass_offset_base + component];
    for (int i = 0; i < width; ++i) {
      int dof_i = layout[ncomponent + 1 + begin + i];
      for (int j = 0; j <= i; ++j) {
        int dof_j = layout[ncomponent + 1 + begin + j];
        int index = block + i * width + j;
        if (!solver_dof_awake(dims, awake_tree, dof_i)) {
          factor[index] = i == j ? 1.0f : 0.0f;
          continue;
        }
        if (!solver_dof_awake(dims, awake_tree, dof_j)) {
          factor[index] = 0.0f;
          continue;
        }
        float value = mass[mass_base + index] + factor[index];
        for (int k = 0; k < j; ++k)
          value -= factor[block + i * width + k]
              * factor[block + j * width + k];
        if (i == j) {
          if (!(value > 0.0f) || !isfinite(value)) return false;
          factor[index] = sqrt(value);
        } else {
          float pivot = factor[block + j * width + j];
          if (!(pivot > 0.0f) || !isfinite(pivot)) return false;
          factor[index] = value / pivot;
        }
      }
      for (int j = i + 1; j < width; ++j)
        factor[block + i * width + j] = 0.0f;
    }
  }
  return true;
}

inline float solver_component_factor_entry(device const float* debug,
    constant int* dims, int nv, int dof_i, int dof_j) {
  int header = dims[24] + dims[9] * 10;
  int ncomponent = dims[header];
  int armature_offset = dims[header + 2];
  int layout_offset = dims[header + 3];
  device const int* layout = reinterpret_cast<device const int*>(debug)
      + layout_offset;
  int component_i = layout[2 * ncomponent + 1 + nv + dof_i];
  int component_j = layout[2 * ncomponent + 1 + nv + dof_j];
  if (component_i != component_j || component_i < 0
      || component_i >= ncomponent) return 0.0f;
  int local_i = layout[2 * ncomponent + 1 + 2 * nv + dof_i];
  int local_j = layout[2 * ncomponent + 1 + 2 * nv + dof_j];
  int begin = layout[component_i];
  int width = layout[component_i + 1] - begin;
  if (local_i < 0 || local_j < 0 || local_i >= width || local_j >= width)
    return 0.0f;
  if (local_i < local_j) return 0.0f;
  int mass_offset = layout[ncomponent + 1 + nv + component_i];
  device const float* factor = debug + armature_offset + mass_offset;
  return factor[local_i * width + local_j];
}

inline float solver_component_mass_entry(device const float* debug,
    constant int* dims, int nv, int dof_i, int dof_j) {
  int header = dims[24] + dims[9] * 10;
  int ncomponent = dims[header];
  int layout_offset = dims[header + 3];
  device const int* layout = reinterpret_cast<device const int*>(debug)
      + layout_offset;
  int map_base = 2 * ncomponent + 1 + nv;
  int ci = layout[map_base + dof_i];
  int cj = layout[map_base + dof_j];
  if (ci != cj || ci < 0 || ci >= ncomponent) return 0.0f;
  int local_base = map_base + nv;
  int li = layout[local_base + dof_i];
  int lj = layout[local_base + dof_j];
  int begin = layout[ci];
  int width = layout[ci + 1] - begin;
  if (li < 0 || lj < 0 || li >= width || lj >= width) return 0.0f;
  int dof_base = ncomponent + 1 + begin;
  int last = min(li, lj);
  float value = 0.0f;
  for (int k = 0; k <= last; ++k)
    value += solver_component_factor_entry(debug, dims, nv,
        layout[dof_base + li], layout[dof_base + k])
        * solver_component_factor_entry(debug, dims, nv,
            layout[dof_base + lj], layout[dof_base + k]);
  return value;
}

inline void solver_component_mass_apply_difference(device const float* debug,
    constant int* dims, int nv, device const float* value,
    device const float* reference, device float* projected,
    device float* output) {
  int header = dims[24] + dims[9] * 10;
  int ncomponent = dims[header];
  int layout_offset = dims[header + 3];
  device const int* layout = reinterpret_cast<device const int*>(debug)
      + layout_offset;
  for (int component = 0; component < ncomponent; ++component) {
    int begin = layout[component];
    int width = layout[component + 1] - begin;
    int dof_base = ncomponent + 1 + begin;
    for (int j = 0; j < width; ++j) {
      float product = 0.0f;
      for (int i = j; i < width; ++i) {
        int dof = layout[dof_base + i];
        product += solver_component_factor_entry(debug, dims, nv, dof,
            layout[dof_base + j]) * (value[dof] - reference[dof]);
      }
      projected[layout[dof_base + j]] = product;
    }
    for (int i = 0; i < width; ++i) {
      float product = 0.0f;
      for (int j = 0; j <= i; ++j)
        product += solver_component_factor_entry(debug, dims, nv,
            layout[dof_base + i], layout[dof_base + j])
            * projected[layout[dof_base + j]];
      output[layout[dof_base + i]] = product;
    }
  }
}

inline bool solver_component_mass_solve(device const float* debug,
    constant int* dims, device const int* awake_tree, int nv,
    device const float* rhs,
    device float* solution, device float* scratch) {
  int header = dims[24] + dims[9] * 10;
  int ncomponent = dims[header];
  int layout_offset = dims[header + 3];
  device const int* layout = reinterpret_cast<device const int*>(debug)
      + layout_offset;
  for (int dof = 0; dof < nv; ++dof) solution[dof] = 0.0f;
  for (int component = 0; component < ncomponent; ++component) {
    int begin = layout[component];
    int width = layout[component + 1] - begin;
    int dof_base = ncomponent + 1 + begin;
    int block = layout[ncomponent + 1 + nv + component];
    device const float* factor = debug + dims[header + 2] + block;
    for (int i = 0; i < width; ++i) {
      int dof = layout[dof_base + i];
      if (!solver_dof_awake(dims, awake_tree, dof)) {
        scratch[dof] = 0.0f;
        continue;
      }
      float value = rhs[dof];
      for (int j = 0; j < i; ++j)
        value -= factor[i * width + j] * scratch[layout[dof_base + j]];
      float pivot = factor[i * width + i];
      if (!(pivot > 0.0f) || !isfinite(pivot)) return false;
      scratch[layout[dof_base + i]] = value / pivot;
    }
    for (int i = width - 1; i >= 0; --i) {
      int dof = layout[dof_base + i];
      if (!solver_dof_awake(dims, awake_tree, dof)) {
        solution[dof] = 0.0f;
        continue;
      }
      float value = scratch[dof];
      for (int j = i + 1; j < width; ++j)
        value -= factor[j * width + i] * solution[layout[dof_base + j]];
      solution[layout[dof_base + i]] = value / factor[i * width + i];
    }
  }
  return true;
}

// Restore the smooth acceleration after a primal solve when the pinned
// no-slip post-pass changes the final multipliers.  The primal iterate is no
// longer consistent with those multipliers, so dual-finish must start from
// the original smooth acceleration before applying the final constraint
// force.  Component mass storage is solved from its compiled factors; it must
// not fall through to the dense L workspace, which is not allocated on that
// route.
inline bool solver_restore_smooth_acceleration(device float* out_acc,
    device const float* qfrc, device const float* mass,
    device const float* debug, device float* L, device float* work_a,
    device float* work_b, constant int* dims, device const int* awake_tree,
    int batch, int world, int nv, int nr, int qb,
    bool provided_qacc_smooth, bool paired_qfrc_input) {
  if (dims[20] == -2) {
    for (int i = 0; i < nv; ++i) {
      int hi = component_pair_index(batch, world, nr, i, nr, nv, false);
      int lo = component_pair_index(batch, world, nr, i, nr, nv, true);
      out_acc[qb + i] = mass[hi];
      out_acc[batch * nv + qb + i] = mass[lo];
    }
    return true;
  }
  if (provided_qacc_smooth) {
    for (int i = 0; i < nv; ++i) {
      out_acc[qb + i] = qfrc[qb + i];
      out_acc[batch * nv + qb + i] = paired_qfrc_input
          ? qfrc[batch * nv + qb + i] : 0.0f;
    }
    return true;
  }
  if (dims[20] == -3) {
    if (!solver_component_mass_solve(debug, dims, awake_tree, nv,
        qfrc + qb, work_a, work_b)) return false;
    for (int i = 0; i < nv; ++i) {
      out_acc[qb + i] = work_a[i];
      out_acc[batch * nv + qb + i] = 0.0f;
    }
    return true;
  }
  // Dense M=L L^T: reproduce the smooth solve, using caller-owned vectors
  // rather than the component-only workspace layout.
  for (int i = 0; i < nv; ++i) {
    if (!solver_dof_awake(dims, awake_tree, i)) {
      work_a[i] = 0.0f;
      continue;
    }
    float value = qfrc[qb + i];
    for (int k = 0; k < i; ++k) value -= L[i * nv + k] * work_a[k];
    work_a[i] = value / L[i * nv + i];
  }
  for (int i = nv - 1; i >= 0; --i) {
    if (!solver_dof_awake(dims, awake_tree, i)) {
      work_b[i] = 0.0f;
      continue;
    }
    float value = work_a[i];
    for (int k = i + 1; k < nv; ++k) value -= L[k * nv + i] * work_b[k];
    work_b[i] = value / L[i * nv + i];
  }
  for (int i = 0; i < nv; ++i) {
    out_acc[qb + i] = work_b[i];
    out_acc[batch * nv + qb + i] = 0.0f;
  }
  return true;
}

inline bool solver_row_has_awake_contributor(constant int* dims,
    device const float* J, device const int* awake_tree, int row, int nv) {
  for (int dof = 0; dof < nv; ++dof) {
    if (ccj_get(J, row, dof, nv) != 0.0f
        && solver_dof_awake(dims, awake_tree, dof)) return true;
  }
  // Contacts and connect/weld rows retain structural endpoint identity even
  // when their current numerical Jacobian is zero.
  int meta = dims[24] + row * 10;
  if (dims[meta + 5] != 0) {
    for (int side = 0; side < 2; ++side) {
      int tree = dims[meta + 2 + side];
      if (tree >= 0 && tree < dims[22] && awake_tree[tree] != 0) return true;
    }
  }
  return false;
}

inline void solver_filter_sleeping_rows(constant int* dims,
    device const float* J, device const int* awake_tree,
    device int* enabled, int nv, int nr, int nrows) {
  for (int row = 0; row < nrows; ++row) {
    if (!enabled[row]) continue;
    if (!solver_row_has_awake_contributor(dims, J, awake_tree, row, nv)) {
      enabled[row] = false;
    }
  }
}

inline void solver_mask_sleeping_dofs(constant int* dims,
    device float* J, device const int* awake_tree, int nv, int nrows) {
  for (int dof = 0; dof < nv; ++dof) {
    if (solver_dof_awake(dims, awake_tree, dof)) continue;
    for (int row = 0; row < nrows; ++row) ccj_set(J, row, dof, 0.0f, nv);
  }
}

inline int build_solver_island_maps(constant int* dims,
    device float* dbg, device float* J, device const int* enabled,
    int nv, int nr, int nrows) {
  int ntree = dims[22];
  int dof_tree_offset = dims[23];
  int row_meta_offset = dims[24];
  int core_scratch = dims[25];
  int disable_island_bit = dims[26];
  device float* scratch = dbg + nr * nr + 7 * nr;
  device int* scratch_i = reinterpret_cast<device int*>(scratch);
  device int* dof_island = scratch_i + core_scratch;
  device int* row_island = dof_island + max(nv, 1);
  device int* parent = row_island + max(nr, 1);
  device int* used = parent + max(ntree, 1);
  device int* root_label = used + max(ntree, 1);
  device int* midpoint_blocked_body = root_label + max(ntree, 1);
  device int* midpoint_blocked_tree = midpoint_blocked_body + max(dims[11], 1);
  device const int* awake_tree = scratch_i + dims[27];
  for (int dof = 0; dof < nv; ++dof) dof_island[dof] = -1;
  for (int row = 0; row < nr; ++row) row_island[row] = -1;
  for (int body = 0; body < dims[11]; ++body) midpoint_blocked_body[body] = 0;
  for (int tree = 0; tree < ntree; ++tree) midpoint_blocked_tree[tree] = 0;
  if (ntree <= 0) return 0;
  for (int tree = 0; tree < ntree; ++tree) {
    parent[tree] = tree;
    used[tree] = 0;
    root_label[tree] = -1;
  }

  // With islands disabled, MuJoCo invokes the monolithic solver. Preserve
  // that one solver island while keeping the pinned dof_island=-1 output.
  if ((dims[6] & disable_island_bit) != 0) {
    bool any = false;
    for (int row = 0; row < nrows; ++row) {
      if (!enabled[row]) continue;
      int meta = row_meta_offset + row * 10;
      int body0 = dims[meta + 6], body1 = dims[meta + 7];
      int tendon_tree0 = dims[meta + 8], tendon_tree1 = dims[meta + 9];
      if (body0 >= 0 && body0 < dims[11]) midpoint_blocked_body[body0] = 1;
      if (body1 >= 0 && body1 < dims[11]) midpoint_blocked_body[body1] = 1;
      if (tendon_tree0 >= 0 && tendon_tree0 < ntree) midpoint_blocked_tree[tendon_tree0] = 1;
      if (tendon_tree1 >= 0 && tendon_tree1 < ntree) midpoint_blocked_tree[tendon_tree1] = 1;
      if (solver_row_has_awake_contributor(dims, J, awake_tree, row, nv)) {
        row_island[row] = 0;
        any = true;
      }
    }
    return any ? 1 : 0;
  }

  int previous_type = -2147483647;
  int previous_id = -1;
  int previous_row = -1;
  for (int row = 0; row < nrows; ++row) {
    if (!enabled[row]) continue;
    int meta = row_meta_offset + row * 10;
    int type = dims[meta], ident = dims[meta + 1];
    int body0 = dims[meta + 6], body1 = dims[meta + 7];
    int midpoint_tree0 = dims[meta + 8], midpoint_tree1 = dims[meta + 9];
    if (body0 >= 0 && body0 < dims[11]) midpoint_blocked_body[body0] = 1;
    if (body1 >= 0 && body1 < dims[11]) midpoint_blocked_body[body1] = 1;
    if (midpoint_tree0 >= 0 && midpoint_tree0 < ntree) midpoint_blocked_tree[midpoint_tree0] = 1;
    if (midpoint_tree1 >= 0 && midpoint_tree1 < ntree) midpoint_blocked_tree[midpoint_tree1] = 1;
    bool exempt = dims[meta + 4] != 0;
    bool shortcut = dims[meta + 5] != 0;
    int first_tree = -1;
    if (!exempt && type >= 0 && type == previous_type
        && ident == previous_id && previous_row >= 0) {
      first_tree = row_island[previous_row];
    } else if (shortcut) {
      int tree0 = dims[meta + 2], tree1 = dims[meta + 3];
      if (tree0 >= 0 && tree0 < ntree && awake_tree[tree0] == 0) tree0 = -1;
      if (tree1 >= 0 && tree1 < ntree && awake_tree[tree1] == 0) tree1 = -1;
      if (tree0 >= 0) first_tree = tree0;
      else if (tree1 >= 0) first_tree = tree1;
      solver_tree_union(parent, used, tree0, tree1, ntree);
    } else {
      for (int dof = 0; dof < nv; ++dof) {
        if (ccj_get(J, row, dof, nv) == 0.0f) continue;
        int tree = dims[dof_tree_offset + dof];
        if (tree < 0 || tree >= ntree) continue;
        used[tree] = 1;
        if (first_tree < 0) first_tree = tree;
        else solver_tree_union(parent, used, first_tree, tree, ntree);
      }
    }
    row_island[row] = first_tree;
    previous_type = type;
    previous_id = ident;
    previous_row = row;
  }

  int island_count = 0;
  for (int tree = 0; tree < ntree; ++tree) {
    if (used[tree] == 0) continue;
    int root = solver_tree_root(parent, tree, ntree);
    if (root_label[root] < 0) root_label[root] = island_count++;
  }
  for (int dof = 0; dof < nv; ++dof) {
    int tree = dims[dof_tree_offset + dof];
    if (tree >= 0 && tree < ntree && used[tree] != 0) {
      int root = solver_tree_root(parent, tree, ntree);
      dof_island[dof] = root_label[root];
    }
  }
  for (int row = 0; row < nrows; ++row) {
    int tree = row_island[row];
    if (tree >= 0 && tree < ntree) {
      int root = solver_tree_root(parent, tree, ntree);
      row_island[row] = root_label[root];
    }
  }
  return island_count;
}

inline float3 cross3(float3 a, float3 b) { return cross(a, b); }

inline float impedance_at(device const float* imp, int index, float pos, float margin) {
  float d0 = imp[index * 5];
  float d1 = imp[index * 5 + 1];
  float width = imp[index * 5 + 2];
  float mid = imp[index * 5 + 3];
  float power = imp[index * 5 + 4];
  if (d0 == d1 || width <= 1e-15f) return 0.5f * (d0 + d1);
  float x = abs((pos - margin) / width);
  if (x >= 1.0f) return d1;
  if (x <= 0.0f) return d0;
  float y = power == 1.0f ? x : (x <= mid ? pow(x, power) / pow(mid, power - 1.0f)
      : 1.0f - pow(1.0f - x, power) / pow(1.0f - mid, power - 1.0f));
  return d0 + y * (d1 - d0);
}

inline void reference_params(device const float* solref, device const float* solimp,
    int index, float pos, float margin, float vel, float diag, bool friction,
    float timestep, bool refsafe, thread float& compliance, thread float& aref) {
  float width = max(1e-15f, solimp[index * 5 + 1]);
  float impedance = impedance_at(solimp, index, pos, margin);
  float r0 = solref[index * 2];
  float r1 = solref[index * 2 + 1];
  if (refsafe && r0 > 0.0f) r0 = max(r0, 2.0f * timestep);
  float K = r0 > 0.0f ? 1.0f / max(1e-15f, width * width * r0 * r0 * r1 * r1)
                      : -r0 / max(1e-15f, width * width);
  float B = r1 > 0.0f ? 2.0f / max(1e-15f, width * r0) : -r1 / width;
  if (friction) K = 0.0f;
  compliance = max(1e-15f, (1.0f - impedance) * diag / impedance);
  aref = -B * vel - K * impedance * (pos - margin);
}

// Device-workspace overload used by the dynamically sized coupled row state.
inline void reference_params(device const float* solref, device const float* solimp,
    int index, float pos, float margin, float vel, float diag, bool friction,
    float timestep, bool refsafe, device float& compliance, device float& aref) {
  float width = max(1e-15f, solimp[index * 5 + 1]);
  float impedance = impedance_at(solimp, index, pos, margin);
  float r0 = solref[index * 2];
  float r1 = solref[index * 2 + 1];
  if (refsafe && r0 > 0.0f) r0 = max(r0, 2.0f * timestep);
  float K = r0 > 0.0f ? 1.0f / max(1e-15f, width * width * r0 * r0 * r1 * r1)
                      : -r0 / max(1e-15f, width * width);
  float B = r1 > 0.0f ? 2.0f / max(1e-15f, width * r0) : -r1 / width;
  if (friction) K = 0.0f;
  compliance = max(1e-15f, (1.0f - impedance) * diag / impedance);
  aref = -B * vel - K * impedance * (pos - margin);
}

inline float solve_contact_block(
    thread const float* A, thread const float* b, int n,
    thread float* solution) {
  float best_error = 1e30f;
  float best_objective = 1e30f;
  for (int i = 0; i < 4; ++i) solution[i] = 0.0f;
  for (int set = 0; set < (1 << n); ++set) {
    thread int ids[4];
    int nactive = 0;
    for (int i = 0; i < n; ++i) if (set & (1 << i)) ids[nactive++] = i;
    thread float mat[16], rhs[4], x[4], candidate[4];
    for (int i = 0; i < 16; ++i) mat[i] = 0.0f;
    for (int i = 0; i < 4; ++i) { rhs[i] = 0.0f; x[i] = 0.0f; candidate[i] = 0.0f; }
    for (int i = 0; i < nactive; ++i) {
      rhs[i] = -b[ids[i]];
      for (int j = 0; j < nactive; ++j)
        mat[i * 4 + j] = A[ids[i] * 4 + ids[j]];
    }
    bool valid = true;
    for (int p = 0; p < nactive; ++p) {
      int pivot_row = p;
      float pivot_abs = abs(mat[p * 4 + p]);
      for (int r = p + 1; r < nactive; ++r) {
        float value = abs(mat[r * 4 + p]);
        if (value > pivot_abs) { pivot_abs = value; pivot_row = r; }
      }
      if (!(pivot_abs > 1e-15f) || !isfinite(pivot_abs)) { valid = false; break; }
      if (pivot_row != p) {
        for (int j = 0; j < nactive; ++j) {
          float tmp = mat[p * 4 + j];
          mat[p * 4 + j] = mat[pivot_row * 4 + j];
          mat[pivot_row * 4 + j] = tmp;
        }
        float tmp = rhs[p];
        rhs[p] = rhs[pivot_row];
        rhs[pivot_row] = tmp;
      }
      float pivot = mat[p * 4 + p];
      for (int r = p + 1; r < nactive; ++r) {
        float factor = mat[r * 4 + p] / pivot;
        for (int j = p; j < nactive; ++j) mat[r * 4 + j] -= factor * mat[p * 4 + j];
        rhs[r] -= factor * rhs[p];
      }
    }
    if (!valid) continue;
    for (int r = nactive - 1; r >= 0; --r) {
      float value = rhs[r];
      for (int j = r + 1; j < nactive; ++j) value -= mat[r * 4 + j] * x[j];
      x[r] = value / mat[r * 4 + r];
      if (!isfinite(x[r])) valid = false;
    }
    if (!valid) continue;
    for (int i = 0; i < nactive; ++i) candidate[ids[i]] = x[i];
    float error = 0.0f;
    float scale = 1.0f;
    float objective = 0.0f;
    for (int i = 0; i < n; ++i) {
      float gradient = b[i];
      for (int j = 0; j < n; ++j) gradient += A[i * 4 + j] * candidate[j];
      scale += abs(b[i]);
      for (int j = 0; j < n; ++j) scale += abs(A[i * 4 + j] * candidate[j]);
      bool active = (set & (1 << i)) != 0;
      error = max(error, active ? max(0.0f, -candidate[i]) : max(0.0f, -gradient));
      objective += 0.5f * candidate[i] * (gradient + b[i]);
    }
    float scaled_error = error / scale;
    if (scaled_error < best_error ||
        (scaled_error == best_error && objective < best_objective)) {
      best_error = scaled_error;
      best_objective = objective;
      for (int i = 0; i < 4; ++i) solution[i] = candidate[i];
    }
  }
  return best_error;
}

inline float solve_contact_block_iterative(
    thread const float* A, thread const float* b, int n,
    thread float* solution) {
  thread float x[10], z[10], next[10];
  float lipschitz = 1e-15f;
  for (int i = 0; i < 10; ++i) x[i] = z[i] = next[i] = 0.0f;
  for (int i = 0; i < n; ++i) {
    x[i] = z[i] = max(0.0f, solution[i]);
    float row_sum = 0.0f;
    for (int j = 0; j < n; ++j) row_sum += abs(A[i * 10 + j]);
    lipschitz = max(lipschitz, row_sum);
  }
  float momentum = 1.0f;
  float residual = INFINITY;
  for (int iteration = 0; iteration < 1024; ++iteration) {
    for (int i = 0; i < n; ++i) {
      float gradient = b[i];
      for (int j = 0; j < n; ++j) gradient += A[i * 10 + j] * z[j];
      next[i] = max(0.0f, z[i] - gradient / lipschitz);
    }
    float next_momentum = 0.5f * (1.0f + sqrt(1.0f + 4.0f * momentum * momentum));
    float beta = (momentum - 1.0f) / next_momentum;
    for (int i = 0; i < n; ++i) {
      float previous = x[i];
      x[i] = next[i];
      z[i] = next[i] + beta * (next[i] - previous);
    }
    momentum = next_momentum;

    if ((iteration & 15) == 15 || iteration == 1023) {
      residual = 0.0f;
      for (int i = 0; i < n; ++i) {
        float gradient = b[i];
        float row_scale = abs(b[i]);
        for (int j = 0; j < n; ++j) {
          float value = A[i * 10 + j] * x[j];
          gradient += value;
          row_scale += abs(value);
        }
        float projected = max(0.0f, x[i] - gradient / max(A[i * 10 + i], 1e-15f));
        residual = max(residual, abs(projected - x[i]) * A[i * 10 + i] / max(1.0f, row_scale));
      }
      if (residual <= 5e-7f) break;
    }
  }
  for (int i = 0; i < n; ++i) solution[i] = x[i];
  return residual;
}

// Geometric contact detection and Jacobian generation for all 9 primitive pairs
inline void contact_cache_store(
    device float* records, int record, thread const ContactGeom& contact) {
  int base=record*25+12;
  records[base+0]=contact.dist;
  records[base+1]=contact.pos.x; records[base+2]=contact.pos.y;
  records[base+3]=contact.pos.z;
  records[base+4]=contact.normal.x; records[base+5]=contact.normal.y;
  records[base+6]=contact.normal.z;
  records[base+7]=contact.t1.x; records[base+8]=contact.t1.y;
  records[base+9]=contact.t1.z;
  records[base+10]=contact.t2.x; records[base+11]=contact.t2.y;
  records[base+12]=contact.t2.z;
}

// Generic contacts are produced in their own entry point. The dispatcher
// reached here intentionally has no plugin-SDF or mesh-SDF call edges.
// Reset pair counts for each unmasked world before any producer dispatch.
// This also handles SDF initpoints=0, where source output capacity is zero and
// the ordinary contact-producer block is skipped entirely.
kernel void clear_contact_pair_counts(
    device int* contact_count [[buffer(0)]],
    device const int* dims [[buffer(1)]],
    uint tid [[thread_position_in_grid]]) {
  int npairs=dims[1], batch=dims[3];
  if (npairs<=0) return;
  int world=int(tid)/npairs;
  if (world>=batch || !pair_contact_world_enabled(dims,world)) return;
  contact_count[tid]=0;
}

kernel void contact_produce_generic(
    device const float* geom_pos [[buffer(0)]],
    device const float* geom_quat [[buffer(1)]],
    device const float* geom_size [[buffer(2)]],
    device const int* geom_type [[buffer(3)]],
    device const float* geom_rbound [[buffer(4)]],
    device const int* pair_geoms [[buffer(5)]],
    device const float* pair_margin_gap [[buffer(6)]],
    device const int* dims [[buffer(7)]],
    device const int* logical_to_packed [[buffer(8)]],
    device const float* mesh_hull [[buffer(9)]],
    device const int* mesh_hull_info [[buffer(10)]],
    device int* contact_count [[buffer(11)]],
    device float* contact_records [[buffer(12)]],
    uint tid [[thread_position_in_grid]]) {
  int npairs=dims[1], ncontacts_max=dims[2], batch=dims[3];
  int world=int(tid)/max(npairs,1), pair_idx=int(tid)%max(npairs,1);
  if (npairs<=0 || world>=batch || pair_idx>=npairs) return;
  int count_index=world*npairs+pair_idx;
  if (!pair_contact_world_enabled(dims,world)) return;
  int a=pair_geoms[pair_idx*2], b=pair_geoms[pair_idx*2+1];
  int ta=geom_type[a], tb=geom_type[b];
  if (ta==8 || tb==8) return;
  contact_count[count_index]=0;
  bool prune=(dims[9]&1)!=0;
  if (prune && logical_to_packed[world*npairs+pair_idx]<0) return;
  int offsets_base=10;
  int offset=dims[offsets_base+pair_idx];
  int max_con=min(50,dims[offsets_base+pair_idx+1]-offset);
  if (max_con<=0) return;
  int go=world*dims[6];
  float3 pa=float3(geom_pos[(go+a)*3],geom_pos[(go+a)*3+1],geom_pos[(go+a)*3+2]);
  float3 pb=float3(geom_pos[(go+b)*3],geom_pos[(go+b)*3+1],geom_pos[(go+b)*3+2]);
  float4 qa=float4(geom_quat[(go+a)*4],geom_quat[(go+a)*4+1],
                   geom_quat[(go+a)*4+2],geom_quat[(go+a)*4+3]);
  float4 qb=float4(geom_quat[(go+b)*4],geom_quat[(go+b)*4+1],
                   geom_quat[(go+b)*4+2],geom_quat[(go+b)*4+3]);
  float3 sza=float3(geom_size[a*3],geom_size[a*3+1],geom_size[a*3+2]);
  float3 szb=float3(geom_size[b*3],geom_size[b*3+1],geom_size[b*3+2]);
  float rba=geom_rbound[a], rbb=geom_rbound[b];
  float margin=pair_margin_gap[pair_idx*2];
  float gap=pair_margin_gap[pair_idx*2+1];
  int disable_multiccd=dims[8];
  thread ContactGeom con[50];
  for (int k=0;k<50;++k) {
    con[k].dist=1e30f; con[k].pos=float3(0.0f);
    con[k].normal=float3(0.0f); con[k].t1=float3(0.0f);
    con[k].t2=float3(0.0f);
  }
  int ngeom=dims[6];
  int ncon=0;
  bool heightfield_mesh = (ta == 1 && tb == 7) || (ta == 7 && tb == 1);
  bool common_rigid_convex = common_ccd_production_uses_mjc_convex(ta, tb);
  int common_header = 9 * ngeom;
  bool common_ready = (heightfield_mesh || common_rigid_convex) &&
      mesh_hull_info[common_header] == 1128487732;
  if (common_ready) {
    int query = world * npairs + pair_idx;
    int output_base = mesh_hull_info[common_header + 5] +
                      query * mesh_hull_info[common_header + 6];
    int count = min(max_con, min(50, int(mesh_hull[output_base])));
    int error = int(mesh_hull[output_base + 1]);
    if (error == 0) {
      for (int k = 0; k < count; ++k) {
        int src = output_base + 2 + 16 * k;
        con[k].dist = mesh_hull[src + 0];
        con[k].normal = float3(mesh_hull[src + 1], mesh_hull[src + 2],
                               mesh_hull[src + 3]);
        con[k].pos = float3(mesh_hull[src + 4], mesh_hull[src + 5],
                            mesh_hull[src + 6]);
        con[k].t1 = float3(mesh_hull[src + 7], mesh_hull[src + 8],
                           mesh_hull[src + 9]);
        con[k].t2 = float3(mesh_hull[src + 10], mesh_hull[src + 11],
                           mesh_hull[src + 12]);
        if (heightfield_mesh && ta == 7) {
          con[k].normal = -con[k].normal;
          con[k].t1 = -con[k].t1;
          con[k].t2 = cross(con[k].normal, con[k].t1);
        }
      }
      ncon = count;
    }
  } else {
    ncon = collide_pair_without_sdf(ta, pa, qa, sza, rba, tb, pb, qb, szb, rbb,
                        margin, gap, disable_multiccd, con, a, b,
                        mesh_hull, mesh_hull_info, max_con);
  }
  ncon=clamp(ncon,0,max_con);
  contact_count[count_index]=ncon;
  for (int k=0;k<ncon;++k)
    contact_cache_store(contact_records,world*ncontacts_max+offset+k,con[k]);
}

struct SdfStagePairInput {
  int world; int pair_idx; int seed; int slot; int state_base; int ctrl_index;
  int geom1; int geom2; int type1; int type2; int swapped;
  int canonical1; int canonical2; int canonical_type1; int canonical_type2;
  float3 pos1; float3 pos2; float4 quat1; float4 quat2;
  float3 size1; float3 size2;
};

inline bool sdf_stage_decode(device const float* geom_pos,
    device const float* geom_quat, device const float* geom_size,
    device const int* geom_type, device const int* pair_geoms,
    device const int* dims, device const int* logical_to_packed,
    device const int* stage_dims, uint tid, int tile_count,
    thread SdfStagePairInput& input) {
  int npairs = dims[1], batch = dims[3];
  int seed_start = stage_dims[1], capacity = max(stage_dims[2], 1);
  int tile = max(tile_count, 1);
  int pair_world_tile = int(tid) / tile, tile_seed = int(tid) % tile;
  int world = pair_world_tile / max(npairs, 1);
  int pair_idx = pair_world_tile % max(npairs, 1);
  if (npairs <= 0 || world >= batch || pair_idx >= npairs ||
      tile_seed >= tile_count || !pair_contact_world_enabled(dims, world))
    return false;
  int ga = pair_geoms[pair_idx * 2], gb = pair_geoms[pair_idx * 2 + 1];
  int ta = geom_type[ga], tb = geom_type[gb];
  if ((ta != 8 && tb != 8) || ta == 7 || tb == 7) return false;
  bool prune = (dims[9] & 1) != 0;
  if (prune && logical_to_packed[world * npairs + pair_idx] < 0)
    return false;
  int swapped = ta > tb;
  int go = world * dims[6];
  input.world = world; input.pair_idx = pair_idx;
  input.seed = seed_start + tile_seed;
  input.slot = (world * npairs + pair_idx) * capacity + tile_seed;
  input.state_base = input.slot * SDF_STAGE_STATE_WORDS;
  input.ctrl_index = input.slot;
  input.geom1 = ga; input.geom2 = gb; input.type1 = ta; input.type2 = tb;
  input.swapped = swapped;
  input.canonical1 = swapped ? gb : ga;
  input.canonical2 = swapped ? ga : gb;
  input.canonical_type1 = swapped ? tb : ta;
  input.canonical_type2 = swapped ? ta : tb;
  float3 pa = float3(geom_pos[(go + ga) * 3],
      geom_pos[(go + ga) * 3 + 1], geom_pos[(go + ga) * 3 + 2]);
  float3 pb = float3(geom_pos[(go + gb) * 3],
      geom_pos[(go + gb) * 3 + 1], geom_pos[(go + gb) * 3 + 2]);
  float4 qa = float4(geom_quat[(go + ga) * 4],
      geom_quat[(go + ga) * 4 + 1], geom_quat[(go + ga) * 4 + 2],
      geom_quat[(go + ga) * 4 + 3]);
  float4 qb = float4(geom_quat[(go + gb) * 4],
      geom_quat[(go + gb) * 4 + 1], geom_quat[(go + gb) * 4 + 2],
      geom_quat[(go + gb) * 4 + 3]);
  float3 sza = float3(geom_size[ga * 3], geom_size[ga * 3 + 1],
                      geom_size[ga * 3 + 2]);
  float3 szb = float3(geom_size[gb * 3], geom_size[gb * 3 + 1],
                      geom_size[gb * 3 + 2]);
  input.pos1 = swapped ? pb : pa; input.pos2 = swapped ? pa : pb;
  input.quat1 = swapped ? qb : qa; input.quat2 = swapped ? qa : qb;
  input.size1 = swapped ? szb : sza;
  input.size2 = swapped ? sza : szb;
  return true;
}

// Analytic SDF stages keep each registered-function PSO bounded. The source
// seed order is retained by the host's sequential tile and iteration dispatch.
kernel void contact_sdf_seed_init(
    device const float* geom_pos [[buffer(0)]],
    device const float* geom_quat [[buffer(1)]],
    device const float* geom_size [[buffer(2)]],
    device const int* geom_type [[buffer(3)]],
    device const int* pair_geoms [[buffer(4)]],
    device const int* dims [[buffer(5)]],
    device const int* logical_to_packed [[buffer(6)]],
    device const float* mesh_hull [[buffer(7)]],
    device const int* mesh_hull_info [[buffer(8)]],
    device int* seed_valid [[buffer(9)]],
    device float* seed_state [[buffer(10)]],
    device int* seed_ctrl [[buffer(11)]],
    device const int* stage_dims [[buffer(12)]],
    uint tid [[thread_position_in_grid]]) {
  int npairs = dims[1], batch = dims[3];
  int tile_count = stage_dims[0], capacity = max(stage_dims[2], 1);
  int tile = max(tile_count, 1);
  int pair_world_tile = int(tid) / tile, tile_seed = int(tid) % tile;
  int world = pair_world_tile / max(npairs, 1);
  int pair_idx = pair_world_tile % max(npairs, 1);
  if (npairs <= 0 || world >= batch || pair_idx >= npairs ||
      tile_seed >= tile_count || !pair_contact_world_enabled(dims, world)) return;
  int slot = (world * npairs + pair_idx) * capacity + tile_seed;
  int state_base = slot * SDF_STAGE_STATE_WORDS;
  seed_valid[slot] = 0; seed_ctrl[slot] = 0;
  SdfStagePairInput in;
  if (!sdf_stage_decode(geom_pos, geom_quat, geom_size, geom_type,
      pair_geoms, dims, logical_to_packed, stage_dims, tid, tile_count, in)) return;
  if (!sdf_stage_seed_init(in.canonical_type1, in.pos1, in.quat1, in.size1,
      in.pos2, in.quat2, in.canonical1, in.canonical2, in.seed,
      mesh_hull, mesh_hull_info, seed_state, state_base)) return;
  seed_valid[slot] = 1; seed_ctrl[slot] = 1;
}

kernel void contact_sdf_phase_reset(
    device const float* geom_pos [[buffer(0)]],
    device const float* geom_quat [[buffer(1)]],
    device const float* geom_size [[buffer(2)]],
    device const int* geom_type [[buffer(3)]],
    device const int* pair_geoms [[buffer(4)]],
    device const int* dims [[buffer(5)]],
    device const int* logical_to_packed [[buffer(6)]],
    device int* seed_valid [[buffer(7)]],
    device float* seed_state [[buffer(8)]],
    device int* seed_ctrl [[buffer(9)]],
    device const int* stage_dims [[buffer(10)]],
    uint tid [[thread_position_in_grid]]) {
  int tile_count = stage_dims[0], npairs = dims[1], batch = dims[3];
  int capacity = max(stage_dims[2], 1), tile = max(tile_count, 1);
  int pw = int(tid) / tile, ts = int(tid) % tile;
  int world = pw / max(npairs, 1), pair = pw % max(npairs, 1);
  if (npairs <= 0 || world >= batch || pair >= npairs || ts >= tile_count ||
      !pair_contact_world_enabled(dims, world)) return;
  int slot = (world * npairs + pair) * capacity + ts;
  if (seed_valid[slot] == 0) return;
  if (((dims[9] & 1) != 0) && logical_to_packed[world * npairs + pair] < 0)
    return;
  seed_ctrl[slot] = 1;
  sdf_stage_store_wide(seed_state, slot * SDF_STAGE_STATE_WORDS + SDF_STAGE_DIST,
                       sw(SDF_MAXVAL));
}

kernel void contact_sdf_descent_prepare(
    device const float* geom_pos [[buffer(0)]],
    device const float* geom_quat [[buffer(1)]],
    device const float* geom_size [[buffer(2)]],
    device const int* geom_type [[buffer(3)]],
    device const int* pair_geoms [[buffer(4)]],
    device const int* dims [[buffer(5)]],
    device const int* logical_to_packed [[buffer(6)]],
    device const float* mesh_hull [[buffer(7)]],
    device const int* mesh_hull_info [[buffer(8)]],
    device int* seed_valid [[buffer(9)]],
    device float* seed_state [[buffer(10)]],
    device int* seed_ctrl [[buffer(11)]],
    device const int* stage_dims [[buffer(12)]],
    uint tid [[thread_position_in_grid]]) {
  int tile_count = stage_dims[0], npairs = dims[1], batch = dims[3];
  int capacity = max(stage_dims[2], 1), tile = max(tile_count, 1);
  int pw = int(tid) / tile, ts = int(tid) % tile;
  int world = pw / max(npairs, 1), pair = pw % max(npairs, 1);
  if (npairs <= 0 || world >= batch || pair >= npairs || ts >= tile_count ||
      !pair_contact_world_enabled(dims, world)) return;
  int slot = (world * npairs + pair) * capacity + ts;
  if (seed_valid[slot] == 0 || seed_ctrl[slot] == 0) return;
  if (((dims[9] & 1) != 0) && logical_to_packed[world * npairs + pair] < 0)
    return;
  SdfStagePairInput in;
  if (!sdf_stage_decode(geom_pos, geom_quat, geom_size, geom_type,
      pair_geoms, dims, logical_to_packed, stage_dims, tid, tile_count, in)) return;
  int state_base = slot * SDF_STAGE_STATE_WORDS;
  int iters = mesh_hull_info[in.canonical2 * 9 + 2];
  if (stage_dims[4] >= iters) return;
  SdfSide A, B; SdfTransformWide T21;
  if (!sdf_stage_load_sides_transform(in.canonical_type1, in.size1,
      in.canonical1, in.canonical2, mesh_hull, mesh_hull_info,
      seed_state, state_base, A, B, T21)) {
    seed_ctrl[slot] = 0; return;
  }
  sdf_stage_prepare_step(stage_dims[3], A, B, T21, mesh_hull,
                         seed_state, state_base, seed_ctrl, slot);
}

kernel void contact_sdf_line_search(
    device const float* geom_pos [[buffer(0)]],
    device const float* geom_quat [[buffer(1)]],
    device const float* geom_size [[buffer(2)]],
    device const int* geom_type [[buffer(3)]],
    device const int* pair_geoms [[buffer(4)]],
    device const int* dims [[buffer(5)]],
    device const int* logical_to_packed [[buffer(6)]],
    device const float* mesh_hull [[buffer(7)]],
    device const int* mesh_hull_info [[buffer(8)]],
    device int* seed_valid [[buffer(9)]],
    device float* seed_state [[buffer(10)]],
    device int* seed_ctrl [[buffer(11)]],
    device const int* stage_dims [[buffer(12)]],
    uint tid [[thread_position_in_grid]]) {
  int tile_count = stage_dims[0], npairs = dims[1], batch = dims[3];
  int capacity = max(stage_dims[2], 1), tile = max(tile_count, 1);
  int pw = int(tid) / tile, ts = int(tid) % tile;
  int world = pw / max(npairs, 1), pair = pw % max(npairs, 1);
  if (npairs <= 0 || world >= batch || pair >= npairs || ts >= tile_count ||
      !pair_contact_world_enabled(dims, world)) return;
  int slot = (world * npairs + pair) * capacity + ts;
  if (seed_valid[slot] == 0 || seed_ctrl[slot] == 0) return;
  if (((dims[9] & 1) != 0) && logical_to_packed[world * npairs + pair] < 0)
    return;
  SdfStagePairInput in;
  if (!sdf_stage_decode(geom_pos, geom_quat, geom_size, geom_type,
      pair_geoms, dims, logical_to_packed, stage_dims, tid, tile_count, in)) return;
  int state_base = slot * SDF_STAGE_STATE_WORDS;
  int iters = mesh_hull_info[in.canonical2 * 9 + 2];
  if (stage_dims[4] >= iters) return;
  SdfSide A, B; SdfTransformWide T21;
  if (!sdf_stage_load_sides_transform(in.canonical_type1, in.size1,
      in.canonical1, in.canonical2, mesh_hull, mesh_hull_info,
      seed_state, state_base, A, B, T21)) {
    seed_ctrl[slot] = 0; return;
  }
  sdf_stage_line_search(stage_dims[3], A, B, T21, mesh_hull,
                        seed_state, state_base, seed_ctrl, slot);
}

kernel void contact_sdf_publish_normal(
    device const float* geom_pos [[buffer(0)]],
    device const float* geom_quat [[buffer(1)]],
    device const float* geom_size [[buffer(2)]],
    device const int* geom_type [[buffer(3)]],
    device const int* pair_geoms [[buffer(4)]],
    device const int* dims [[buffer(5)]],
    device const int* logical_to_packed [[buffer(6)]],
    device const float* mesh_hull [[buffer(7)]],
    device const int* mesh_hull_info [[buffer(8)]],
    device int* seed_valid [[buffer(9)]],
    device float* seed_state [[buffer(10)]],
    device const int* stage_dims [[buffer(11)]],
    uint tid [[thread_position_in_grid]]) {
  int tile_count = stage_dims[0], npairs = dims[1], batch = dims[3];
  int capacity = max(stage_dims[2], 1), tile = max(tile_count, 1);
  int pw = int(tid) / tile, ts = int(tid) % tile;
  int world = pw / max(npairs, 1), pair = pw % max(npairs, 1);
  if (npairs <= 0 || world >= batch || pair >= npairs || ts >= tile_count ||
      !pair_contact_world_enabled(dims, world)) return;
  int slot = (world * npairs + pair) * capacity + ts;
  if (seed_valid[slot] == 0) return;
  if (((dims[9] & 1) != 0) && logical_to_packed[world * npairs + pair] < 0) {
    seed_valid[slot] = 0; return;
  }
  SdfWide dint = sdf_stage_load_wide(seed_state,
      slot * SDF_STAGE_STATE_WORDS + SDF_STAGE_DIST);
  if (sw_sign(dint) > 0) { seed_valid[slot] = 0; return; }
  SdfStagePairInput in;
  if (!sdf_stage_decode(geom_pos, geom_quat, geom_size, geom_type,
      pair_geoms, dims, logical_to_packed, stage_dims, tid, tile_count, in)) {
    seed_valid[slot] = 0; return;
  }
  int state_base = slot * SDF_STAGE_STATE_WORDS;
  SdfSide A, B; SdfTransformWide T21;
  if (!sdf_stage_load_sides_transform(in.canonical_type1, in.size1,
      in.canonical1, in.canonical2, mesh_hull, mesh_hull_info,
      seed_state, state_base, A, B, T21)) {
    seed_valid[slot] = 0; return;
  }
  SdfPointPair x = sdf_stage_load_point(seed_state, state_base + SDF_STAGE_X);
  SdfVectorPair grad = sdf_obj_pair_grad(1, A, B, T21, mesh_hull, x);
  SdfWide gl = sw_sqrt(sdf_vector_dot_wide(grad, grad));
  if (sw_sign(sw_sub(gl, sw_source_minval())) < 0) {
    seed_valid[slot] = 0; return;
  }
  sdf_stage_store_vector(seed_state, state_base + SDF_STAGE_NORMAL,
                         sdf_vector_normalize(grad));
}

kernel void contact_sdf_publish_contact(
    device const float* geom_pos [[buffer(0)]],
    device const float* geom_quat [[buffer(1)]],
    device const int* geom_type [[buffer(2)]],
    device const int* pair_geoms [[buffer(3)]],
    device const int* dims [[buffer(4)]],
    device const int* logical_to_packed [[buffer(5)]],
    device float* seed_records [[buffer(6)]],
    device const int* seed_valid [[buffer(7)]],
    device float* seed_state [[buffer(8)]],
    device const int* stage_dims [[buffer(9)]],
    uint tid [[thread_position_in_grid]]) {
  int tile_count = stage_dims[0], npairs = dims[1], batch = dims[3];
  int capacity = max(stage_dims[2], 1), tile = max(tile_count, 1);
  int pw = int(tid) / tile, ts = int(tid) % tile;
  int world = pw / max(npairs, 1), pair = pw % max(npairs, 1);
  if (npairs <= 0 || world >= batch || pair >= npairs || ts >= tile_count ||
      !pair_contact_world_enabled(dims, world)) return;
  int slot = (world * npairs + pair) * capacity + ts;
  if (seed_valid[slot] == 0) return;
  if (((dims[9] & 1) != 0) && logical_to_packed[world * npairs + pair] < 0)
    return;
  int ga = pair_geoms[pair * 2], gb = pair_geoms[pair * 2 + 1];
  int ta = geom_type[ga], tb = geom_type[gb], swapped = ta > tb;
  int canonical_sdf = swapped ? ga : gb;
  int go = world * dims[6];
  float3 p2 = float3(geom_pos[(go + canonical_sdf) * 3],
      geom_pos[(go + canonical_sdf) * 3 + 1],
      geom_pos[(go + canonical_sdf) * 3 + 2]);
  float4 q2 = float4(geom_quat[(go + canonical_sdf) * 4],
      geom_quat[(go + canonical_sdf) * 4 + 1],
      geom_quat[(go + canonical_sdf) * 4 + 2],
      geom_quat[(go + canonical_sdf) * 4 + 3]);
  int state_base = slot * SDF_STAGE_STATE_WORDS;
  sdf_stage_publish(swapped, p2, q2,
      sdf_stage_load_vector(seed_state, state_base + SDF_STAGE_NORMAL),
      sdf_stage_load_wide(seed_state, state_base + SDF_STAGE_DIST),
      seed_state, state_base, seed_records, slot);
}

// Serial source-order compaction/dedup per pair. The 32-seed device tile is
// reused, so raw initpoint budgets are not capped by temporary storage.
kernel void contact_finalize_sdf(
    device const int* geom_type [[buffer(0)]],
    device const int* pair_geoms [[buffer(1)]],
    device const int* dims [[buffer(2)]],
    device const int* logical_to_packed [[buffer(3)]],
    device const float* seed_records [[buffer(4)]],
    device const int* seed_valid [[buffer(5)]],
    device int* contact_count [[buffer(6)]],
    device float* contact_records [[buffer(7)]],
    device const int* stage_dims [[buffer(8)]],
    uint tid [[thread_position_in_grid]]) {
  int npairs=dims[1], ncontacts_max=dims[2], batch=dims[3];
  int world=int(tid)/max(npairs,1), pair_idx=int(tid)%max(npairs,1);
  if (npairs<=0 || world>=batch || pair_idx>=npairs) return;
  if (!pair_contact_world_enabled(dims,world)) return;
  int a=pair_geoms[pair_idx*2], b=pair_geoms[pair_idx*2+1];
  int ta=geom_type[a], tb=geom_type[b];
  if ((ta!=8 && tb!=8) || ta==7 || tb==7) return;
  bool prune=(dims[9]&1)!=0;
  if (prune && logical_to_packed[world*npairs+pair_idx]<0) return;
  int count_index=world*npairs+pair_idx;
  int tile_count=stage_dims[0], seed_start=stage_dims[1];
  int capacity=max(stage_dims[2],1);
  if (seed_start==0) contact_count[count_index]=0;
  if (contact_count[count_index]<0) return;
  int offsets_base=10;
  int offset=dims[offsets_base+pair_idx];
  int max_con=min(50,dims[offsets_base+pair_idx+1]-offset);
  if (max_con<=0) return;
  int ncon=contact_count[count_index];
  SdfPointPair accepted[50];
  for (int k=0;k<ncon;k++) {
    int outbase=(world*ncontacts_max+offset+k)*25;
    accepted[k]={
      float3(contact_records[outbase],contact_records[outbase+1],contact_records[outbase+2]),
      float3(contact_records[outbase+3],contact_records[outbase+4],contact_records[outbase+5]),
      float3(contact_records[outbase+6],contact_records[outbase+7],contact_records[outbase+8])};
  }
  for (int i=0;i<tile_count;i++) {
    int local=(world*npairs+pair_idx)*capacity+i;
    if (seed_valid[local]==0) continue;
    int inbase=local*25;
    SdfPointPair x={
      float3(seed_records[inbase],seed_records[inbase+1],seed_records[inbase+2]),
      float3(seed_records[inbase+3],seed_records[inbase+4],seed_records[inbase+5]),
      float3(seed_records[inbase+6],seed_records[inbase+7],seed_records[inbase+8])};
    bool known=false;
    for (int k=0;k<ncon;k++) {
      if (sw_sign(sw_sub(sdf_pair_distance(x,accepted[k]),
                         sw_source_minval()))<0) { known=true; break; }
    }
    if (known) continue;
    if (ncon>=max_con) {
      // Pinned mjc_SDF errors only when a unique accepted output exceeds 50.
      // Negative count is consumed as a fail-closed invalid row by contact_normal.
      contact_count[count_index]=-1;
      return;
    }
    int outbase=(world*ncontacts_max+offset+ncon)*25;
    for (int k=0;k<25;k++) contact_records[outbase+k]=seed_records[inbase+k];
    accepted[ncon]=x;
    ncon++;
  }
  contact_count[count_index]=ncon;
}

kernel void contact_produce_mesh_sdf(
    device const float* geom_pos [[buffer(0)]],
    device const float* geom_quat [[buffer(1)]],
    device const float* geom_size [[buffer(2)]],
    device const int* geom_type [[buffer(3)]],
    device const int* pair_geoms [[buffer(4)]],
    device const int* dims [[buffer(5)]],
    device const int* logical_to_packed [[buffer(6)]],
    device const float* mesh_hull [[buffer(7)]],
    device const int* mesh_hull_info [[buffer(8)]],
    device int* contact_count [[buffer(9)]],
    device float* contact_records [[buffer(10)]],
    device float* candidate_scratch [[buffer(11)]],
    uint tid [[thread_position_in_grid]]) {
  int npairs=dims[1], ncontacts_max=dims[2], batch=dims[3];
  int world=int(tid)/max(npairs,1), pair_idx=int(tid)%max(npairs,1);
  if (npairs<=0 || world>=batch || pair_idx>=npairs) return;
  if (!pair_contact_world_enabled(dims,world)) return;
  int a=pair_geoms[pair_idx*2], b=pair_geoms[pair_idx*2+1];
  int ta=geom_type[a], tb=geom_type[b];
  if (!((ta==7&&tb==8)||(ta==8&&tb==7))) return;
  int count_index=world*npairs+pair_idx;
  contact_count[count_index]=0;
  bool prune=(dims[9]&1)!=0;
  if (prune && logical_to_packed[world*npairs+pair_idx]<0) return;
  int offsets_base=10;
  int offset=dims[offsets_base+pair_idx];
  int max_con=min(50,dims[offsets_base+pair_idx+1]-offset);
  if (max_con<=0) return;
  int go=world*dims[6];
  float3 pa=float3(geom_pos[(go+a)*3],geom_pos[(go+a)*3+1],geom_pos[(go+a)*3+2]);
  float3 pb=float3(geom_pos[(go+b)*3],geom_pos[(go+b)*3+1],geom_pos[(go+b)*3+2]);
  float4 qa=float4(geom_quat[(go+a)*4],geom_quat[(go+a)*4+1],
                   geom_quat[(go+a)*4+2],geom_quat[(go+a)*4+3]);
  float4 qb=float4(geom_quat[(go+b)*4],geom_quat[(go+b)*4+1],
                   geom_quat[(go+b)*4+2],geom_quat[(go+b)*4+3]);
  float3 sza=float3(geom_size[a*3],geom_size[a*3+1],geom_size[a*3+2]);
  float3 szb=float3(geom_size[b*3],geom_size[b*3+1],geom_size[b*3+2]);
  int record_base=world*ncontacts_max+offset;
  int ncon=collide_pair_mesh_sdf_cached(ta,pa,qa,tb,pb,qb,a,b,
      max_con,record_base,contact_records,(world*npairs+pair_idx)*50,
      candidate_scratch,
      mesh_hull,mesh_hull_info);
  contact_count[count_index]=clamp(ncon,0,max_con);
}

kernel void contact_normal(
    device const int* geom_bodyid [[buffer(0)]],
    device const float* body_pos [[buffer(1)]],
    device const float* body_quat [[buffer(2)]],
    device const float* anchors [[buffer(3)]],
    device const float* axes [[buffer(4)]],
    device const float* qvel [[buffer(5)]],
    device const int* body_parentid [[buffer(6)]],
    device const int* body_jntadr [[buffer(7)]],
    device const int* body_jntnum [[buffer(8)]],
    device const int* jnt_type [[buffer(9)]],
    device const int* jnt_dofadr [[buffer(10)]],
    device const float* body_invweight0 [[buffer(11)]],
    device const int* pair_geoms [[buffer(12)]],
    device const float* pair_margin_gap [[buffer(13)]],
    device const float* pair_solref [[buffer(14)]],
    device const float* pair_solimp [[buffer(15)]],
    device const int* pair_condim [[buffer(16)]],
    device const float* pair_friction [[buffer(17)]],
    device const float* pair_solreffriction [[buffer(18)]],
    device const int* pair_contact_offsets_dims [[buffer(19)]],
    device float* row_data [[buffer(20)]],
    device float* frame [[buffer(21)]],
    device float* jacobian [[buffer(22)]],
    device const int* logical_pair_to_packed [[buffer(23)]],
    device const int* contact_count [[buffer(24)]],
    device const float* contact_records [[buffer(25)]],
    uint tid [[thread_position_in_grid]]) {
  // Fixed header [nv, npairs, ncontacts, batch, nbody, njnt, ngeom,
  // cone, disable_multiccd, prune], then npairs+1 slot offsets.
  int nv = pair_contact_offsets_dims[0];
  int npairs = pair_contact_offsets_dims[1];
  int ncontacts_max = pair_contact_offsets_dims[2];
  int batch = pair_contact_offsets_dims[3];
  int nbody = pair_contact_offsets_dims[4];
  int njnt = pair_contact_offsets_dims[5];
  int ngeom = pair_contact_offsets_dims[6];
  int offsets_base = 10;

  int world = int(tid) / max(npairs, 1);
  int pair_idx = int(tid) % max(npairs, 1);
  if (uint(world) >= uint(batch) || pair_idx >= npairs) return;
  if (!pair_contact_world_enabled(pair_contact_offsets_dims, world)) return;
  bool packed_rows = (pair_contact_offsets_dims[9] & 2) != 0;
  bool prune_pairs = (pair_contact_offsets_dims[9] & 1) != 0;
  device float* jacobian_record = jacobian;
  if (packed_rows) {
    device int* header = reinterpret_cast<device int*>(jacobian);
    int stride = header[9];
    if (header[0] != CCJ_MAGIC || header[1] != CCJ_VERSION
        || header[2] != CCJ_CSR || stride < CCJ_HEADER_WORDS
        || stride > 0x7fffffff / max(world + 1, 1)) {
      atomic_store_explicit(reinterpret_cast<device atomic_int*>(&header[10]), 1,
                            memory_order_relaxed);
      return;
    }
    jacobian_record += world * stride;
  }
  if (prune_pairs
      && logical_pair_to_packed[world * npairs + pair_idx] < 0) {
    int offset = pair_contact_offsets_dims[offsets_base + pair_idx];
    int max_con = pair_contact_offsets_dims[offsets_base + pair_idx + 1] - offset;
    for (int k = 0; k < max_con; ++k) {
      int slot = offset + k;
      if (slot >= ncontacts_max) break;
      int fb = (world * ncontacts_max + slot) * 12;
      frame[fb] = 1234.0f;
    }
    return;
  }

  int a = pair_geoms[pair_idx * 2 + 0];
  int b = pair_geoms[pair_idx * 2 + 1];
  int dim = pair_condim[pair_idx];
  float cone = float(pair_contact_offsets_dims[7]);
  float m = pair_margin_gap[pair_idx * 2 + 0];
  int offset = pair_contact_offsets_dims[offsets_base + pair_idx];
  int max_con = pair_contact_offsets_dims[offsets_base + pair_idx + 1] - offset;
  int slot_rows_base = offsets_base + npairs + 1;
  int raw_ncon=contact_count[world*npairs+pair_idx];
  if (raw_ncon<0) {
    int overflow_slot=offset;
    if (overflow_slot<ncontacts_max) {
      int rb=(world*ncontacts_max+overflow_slot)*36;
      row_data[rb]=1.0f; row_data[rb+1]=NAN;
      row_data[rb+2]=NAN; row_data[rb+3]=NAN;
      row_data[rb+4]=NAN; row_data[rb+5]=NAN;
    }
    return;
  }
  int ncon = clamp(raw_ncon, 0, min(50, max_con));

  int ba = geom_bodyid[a];
  int bb = geom_bodyid[b];
  int bo = world * nbody;
  int jo = world * njnt;
  float diag_approx = body_invweight0[ba * 2] + body_invweight0[bb * 2];

  float d0 = pair_solimp[pair_idx * 5 + 0];
  float d1 = pair_solimp[pair_idx * 5 + 1];
  float width = pair_solimp[pair_idx * 5 + 2];
  float mid = pair_solimp[pair_idx * 5 + 3];
  float power = pair_solimp[pair_idx * 5 + 4];

  float r0 = pair_solref[pair_idx * 2 + 0];
  float r1 = pair_solref[pair_idx * 2 + 1];
  float K, B;
  if (r0 > 0.0f) K = 1.0f / max(1e-15f, d1 * d1 * r0 * r0 * r1 * r1);
  else K = -r0 / max(1e-15f, d1 * d1);
  if (r1 > 0.0f) B = 2.0f / max(1e-15f, d1 * r0);
  else B = -r1 / max(1e-15f, d1);

  // For each detected contact:
  for (int k = 0; k < ncon; ++k) {
    int slot = offset + k;
    if (slot >= ncontacts_max) break;
    int out = world * ncontacts_max + slot;
    int jbase = out * 6 * nv;
    int rb = out * 6 * 6;
    int fb = out * 12;
    int row_start = pair_contact_offsets_dims[slot_rows_base + slot];

    int contact_base=(world*ncontacts_max+slot)*25+12;
    float d=contact_records[contact_base+0];
    float3 n=float3(contact_records[contact_base+4],
                     contact_records[contact_base+5],
                     contact_records[contact_base+6]);
    float3 t1=float3(contact_records[contact_base+7],
                      contact_records[contact_base+8],
                      contact_records[contact_base+9]);
    float3 t2=float3(contact_records[contact_base+10],
                      contact_records[contact_base+11],
                      contact_records[contact_base+12]);
    float3 point=float3(contact_records[contact_base+1],
                         contact_records[contact_base+2],
                         contact_records[contact_base+3]);

    // Zero out buffers
    for (int i = 0; i < 6 * 6; ++i) row_data[rb + i] = 0.0f;
    if (!packed_rows)
      for (int i = 0; i < 6 * nv; ++i) jacobian[jbase + i] = 0.0f;
    for (int i = 0; i < 12; ++i) frame[fb + i] = 0.0f;

    // Compute Kinematics Jacobian
    float3 rel = point;
    float axis_velocity[6] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    for (int side = 0; side < 2; ++side) {
      int body = side == 0 ? ba : bb;
      float sign = side == 0 ? -1.0f : 1.0f;
      while (body > 0 && body < nbody) {
        int ja = body_jntadr[body];
        int jn = body_jntnum[body];
        for (int jj = 0; jj < jn; ++jj) {
          int j = ja + jj;
          int da = jnt_dofadr[j];
            int typ = jnt_type[j];
            int nd = typ == 0 ? 6 : typ == 1 ? 3 : 1;
            for (int q = 0; q < nd; ++q) {
              int dof = da + q;
              if (dof < 0 || dof >= nv) continue;
              float3 col = float3(0.0f);
              float3 angular = float3(0.0f);
              if (typ == 2) {
                col = float3(axes[(jo + j) * 3], axes[(jo + j) * 3 + 1], axes[(jo + j) * 3 + 2]);
              } else if (typ == 3) {
                float3 axis = float3(axes[(jo + j) * 3], axes[(jo + j) * 3 + 1], axes[(jo + j) * 3 + 2]);
                angular = axis;
                col = cross(axis, rel - float3(anchors[(jo + j) * 3], anchors[(jo + j) * 3 + 1], anchors[(jo + j) * 3 + 2]));
              } else if (typ == 0 || typ == 1) {
              if (typ == 0 && q < 3) {
                col = float3(q == 0, q == 1, q == 2);
              } else {
                int qrot = typ == 0 ? q - 3 : q;
                float4 bq = float4(body_quat[(bo + body) * 4], body_quat[(bo + body) * 4 + 1], body_quat[(bo + body) * 4 + 2], body_quat[(bo + body) * 4 + 3]);
                float3 axis = rotate_q(bq, float3(qrot == 0, qrot == 1, qrot == 2));
                angular = axis;
                float3 pivot = typ == 0 ? float3(body_pos[(bo + body) * 3], body_pos[(bo + body) * 3 + 1], body_pos[(bo + body) * 3 + 2])
                                        : float3(anchors[(jo + j) * 3], anchors[(jo + j) * 3 + 1], anchors[(jo + j) * 3 + 2]);
                col = cross(axis, rel - pivot);
              }
            }
              float axis_j[6] = {
                sign * dot(n, col), sign * dot(t1, col), sign * dot(t2, col),
                sign * dot(n, angular), sign * dot(t1, angular),
                sign * dot(t2, angular)};
              for (int axis = 0; axis < 6; ++axis) {
                axis_velocity[axis] += axis_j[axis] * qvel[world * nv + dof];
                if (!packed_rows) {
                  jacobian[jbase + axis * nv + dof] += axis_j[axis];
                } else if (axis < dim) {
                  if (dim == 1) {
                    ccj_add(jacobian_record, row_start, dof, axis_j[0], nv);
                  } else if (cone > 0.5f) {
                    ccj_add(jacobian_record, row_start + axis, dof, axis_j[axis], nv);
                  }
                }
              }
              if (packed_rows && dim > 1 && cone < 0.5f) {
                for (int axis = 1; axis < dim; ++axis) {
                  float mu = pair_friction[pair_idx * 5 + axis - 1];
                  ccj_add(jacobian_record, row_start + 2 * (axis - 1), dof,
                          axis_j[0] + mu * axis_j[axis], nv);
                  ccj_add(jacobian_record, row_start + 2 * (axis - 1) + 1, dof,
                          axis_j[0] - mu * axis_j[axis], nv);
                }
              }
          }
        }
        body = body_parentid[body];
      }
    }

    float impedance;
    float x = width > 1e-15f ? abs((d - m) / width) : 1.0f;
    if (d0 == d1 || width <= 1e-15f) impedance = 0.5f * (d0 + d1);
    else if (x <= 0.0f) impedance = d0;
    else if (x >= 1.0f) impedance = d1;
    else {
      float y;
      if (power == 1.0f) y = x;
      else if (x <= mid) y = pow(x, power) / pow(mid, power - 1.0f);
      else y = 1.0f - pow(1.0f - x, power) / pow(1.0f - mid, power - 1.0f);
      impedance = d0 + y * (d1 - d0);
    }

    float friction_B = B;
    if (cone > 0.5f &&
        (pair_solreffriction[pair_idx * 2] != 0.0f || pair_solreffriction[pair_idx * 2 + 1] != 0.0f)) {
      float fr0 = pair_solreffriction[pair_idx * 2];
      float fr1 = pair_solreffriction[pair_idx * 2 + 1];
      friction_B = fr1 > 0.0f ? 2.0f / max(1e-15f, d1 * fr0) : -fr1 / max(1e-15f, d1);
    }

    for (int row = 0; row < 6; ++row) {
      float vel = 0.0f;
      if (dim == 3 && cone < 0.5f && row > 0 && row < 5) {
        int axis = (row - 1) / 2 + 1;
        float sign = ((row - 1) & 1) == 0 ? 1.0f : -1.0f;
        float mu = pair_friction[pair_idx * 5 + axis - 1];
        if (packed_rows) vel = axis_velocity[0] + sign * mu * axis_velocity[axis];
        else for (int i = 0; i < nv; ++i) {
          float edge_jac = jacobian[jbase + i] + sign * mu * jacobian[jbase + axis * nv + i];
          vel += edge_jac * qvel[world * nv + i];
        }
      } else {
        if (packed_rows) vel = axis_velocity[row];
        else for (int i = 0; i < nv; ++i) vel += jacobian[jbase + row * nv + i] * qvel[world * nv + i];
      }
      int r = rb + row * 6;
      row_data[r] = d < m ? 1.0f : 0.0f;
      row_data[r + 1] = d;
      row_data[r + 2] = vel;
      row_data[r + 3] = (row == 0 || (dim == 3 && cone < 0.5f))
          ? -B * vel - K * impedance * (d - m)
          : -((cone > 0.5f) ? friction_B : B) * vel;
      row_data[r + 4] = impedance;
      row_data[r + 5] = diag_approx;
    }

    frame[fb + 0] = n.x; frame[fb + 1] = n.y; frame[fb + 2] = n.z;
    frame[fb + 3] = t1.x; frame[fb + 4] = t1.y; frame[fb + 5] = t1.z;
    frame[fb + 6] = t2.x; frame[fb + 7] = t2.y; frame[fb + 8] = t2.z;
    frame[fb + 9] = point.x; frame[fb + 10] = point.y; frame[fb + 11] = point.z;
  }

  // Clear unpopulated slots for this pair
  for (int k = ncon; k < max_con; ++k) {
    int slot = offset + k;
    if (slot >= ncontacts_max) break;
    int out = world * ncontacts_max + slot;
    int jbase = out * 6 * nv;
    int rb = out * 6 * 6;
    int fb = out * 12;
    for (int i = 0; i < 6 * 6; ++i) row_data[rb + i] = 0.0f;
    if (!packed_rows)
      for (int i = 0; i < 6 * nv; ++i) jacobian[jbase + i] = 0.0f;
    for (int i = 0; i < 12; ++i) frame[fb + i] = 0.0f;
  }
}

// Coupled projected constraint solve
inline void project_lorentz(thread float* x, int dim) {
  float norm = 0.0f;
  for (int i = 1; i < dim; ++i) norm += x[i] * x[i];
  norm = sqrt(norm);
  if (norm <= x[0]) return;
  if (norm <= -x[0]) {
    for (int i = 0; i < dim; ++i) x[i] = 0.0f;
    return;
  }
  float head = 0.5f * (norm + x[0]);
  float scale = head / max(norm, 1e-20f);
  x[0] = head;
  for (int i = 1; i < dim; ++i) x[i] *= scale;
}

// Project the multiplier vector onto the product of scalar intervals and
// compiled elliptic friction cones.  The native CG/Newton paths use this
// projection in their globalization step; pyramidal and scalar rows retain
// their exact compiled box bounds.
inline float elliptic_projected_residual(thread const float* A,
    thread const float* g, thread const float* friction, int dim,
    thread const float* force);
inline float cert_block_residual(int start, int dim, thread const float* mu,
    thread const float* W, thread const float* R, thread const float* rhs,
    thread const float* lam, device const int* enabled,
    int nr, int total_nr);
inline float cert_block_residual(int start, int dim, thread const float* mu,
    device const float* W, device const float* R, device const float* rhs,
    device const float* lam, device const int* enabled,
    int nr, int total_nr);
inline float cert_retained_system_residual(device const float* W,
    device const float* R, device const float* rhs, device const float* ar,
    device const float* lo, device const float* hi, device const float* lam,
    device const int* enabled, device const int* elliptic_member,
    device const int* elliptic_start, device const int* elliptic_dim,
    device const float* elliptic_friction, int elliptic_count,
    int nr, int total_nr);

inline float primal_projected_residual(thread const float* W,
    thread const float* R, thread const float* rhs, thread const float* ar,
    thread const float* lo, thread const float* hi,
    device const int* enabled, device const int* elliptic_member,
    thread const float* lam, int nr, int n) {
  float residual = 0.0f;
  for (int row = 0; row < n; ++row) if (enabled[row] && !elliptic_member[row]) {
    float grad = -rhs[row];
    for (int col = 0; col < n; ++col) if (enabled[col])
      grad += W[row * nr + col] * lam[col];
    grad += R[row] * lam[row];
    float diag = max(1e-15f, W[row * nr + row] + R[row]);
    float proj = clamp(lam[row] - grad / diag, lo[row], hi[row]);
    float scale = abs(ar[row]) + abs(R[row] * lam[row]);
    for (int col = 0; col < n; ++col) if (enabled[col])
      scale += abs(W[row * nr + col] * lam[col]);
    residual = max(residual, abs(proj - lam[row]) * diag / max(1.0f, scale));
  }
  return residual;
}

inline void dual_gradient_thread(thread const float* W, thread const float* R,
    thread const float* rhs, device const int* enabled,
    thread const float* x, thread float* out, int nr, int n) {
  for (int row = 0; row < n; ++row) {
    float value = -rhs[row] + R[row] * x[row];
    if (enabled[row]) for (int col = 0; col < n; ++col)
      if (enabled[col]) value += W[row * nr + col] * x[col];
    out[row] = enabled[row] ? value : 0.0f;
  }
}

inline float dual_objective_thread(thread const float* W, thread const float* R,
    thread const float* rhs, device const int* enabled,
    thread const float* x, int nr, int n) {
  float result = 0.0f;
  for (int row = 0; row < n; ++row) if (enabled[row]) {
    float product = R[row] * x[row];
    for (int col = 0; col < n; ++col) if (enabled[col])
      product += W[row * nr + col] * x[col];
    result += 0.5f * x[row] * product - rhs[row] * x[row];
  }
  return result;
}

// Evaluate the pinned scalar/pyramidal primal objective in acceleration space.
// R is the inverse row stiffness in this assembled ABI.  Symmetric finite row
// bounds identify dry-friction loss rows; rows before n_eq_rows are bilateral
// equalities and all remaining non-friction rows are unilateral inequalities.
inline void primal_compensated_product_add(float a, float b,
    thread float& sum, thread float& correction);
inline device const float* primal_aref_low_view(device const float* debug,
    constant int* dims, int nv, int nr);
inline float primal_accel_eval_scalar(device const float* mass,
    device const float* L, bool sparse_mass,
    device const float* debug, constant int* dims,
    device const float* J, device const float* R, device const float* aref,
    device const float* lo, device const float* hi,
    device const float* q0, device const float* acc,
    device const int* enabled, device const int* elliptic_member,
    device float* force, device float* gradient,
    device float* row_hessian, device float* residual,
    device float* generalized_force, int nv, int nr, int n, int n_eq_rows,
    device const float* mass_delta, device const float* row_residual,
    device const float* row_residual_low,
    bool incremental_state) {
  device const float* aref_low = primal_aref_low_view(debug, dims, nv, nr);
  for (int i = 0; i < nv; ++i) {
    generalized_force[i] = 0.0f;
    gradient[i] = 0.0f;
  }
  float cost = 0.0f;
  for (int row = 0; row < n; ++row) {
    residual[row] = 0.0f;
    force[row] = 0.0f;
    row_hessian[row] = 0.0f;
    if (!enabled[row] || elliptic_member[row]) continue;
    float rsum = incremental_state ? row_residual[row] : -aref[row];
    float correction = incremental_state
        ? (row_residual_low == nullptr ? 0.0f : row_residual_low[row])
        : -aref_low[row];
    if (!incremental_state) {
      for (int dof = 0; dof < nv; ++dof)
        primal_compensated_product_add(ccj_get(J, row, dof, nv), acc[dof],
            rsum, correction);
    }
    float r = rsum + correction;
    residual[row] = r;
    float reg = max(R[row], 1e-15f);
    float D = 1.0f / reg;
    bool friction = lo[row] < 0.0f && hi[row] > 0.0f
        && isfinite(lo[row]) && isfinite(hi[row])
        && abs(lo[row] + hi[row]) <= 1e-5f * max(1.0f, abs(hi[row]));
    if (row < n_eq_rows) {
      force[row] = -D * r;
      row_hessian[row] = D;
      cost += 0.5f * D * r * r;
    } else if (friction) {
      float loss = hi[row];
      float bound = reg * loss;
      if (r <= -bound) {
        force[row] = loss;
        cost += loss * (-0.5f * bound - r);
      } else if (r >= bound) {
        force[row] = -loss;
        cost += loss * (-0.5f * bound + r);
      } else {
        force[row] = -D * r;
        row_hessian[row] = D;
        cost += 0.5f * D * r * r;
      }
    } else if (r < 0.0f) {
      force[row] = -D * r;
      row_hessian[row] = D;
      cost += 0.5f * D * r * r;
    }
  }
  if (incremental_state) {
    for (int dof = 0; dof < nv; ++dof) {
      gradient[dof] = mass_delta[dof];
      cost += 0.5f * (acc[dof] - q0[dof]) * mass_delta[dof];
    }
  } else if (sparse_mass) {
    solver_component_mass_apply_difference(debug, dims, nv, acc, q0,
        generalized_force, gradient);
    for (int i = 0; i < nv; ++i)
      cost += 0.5f * (acc[i] - q0[i]) * gradient[i];
  } else {
    for (int dof = 0; dof < nv; ++dof) {
      float displacement = acc[dof] - q0[dof];
      float mass_sum = 0.0f, mass_correction = 0.0f;
      for (int k = 0; k < nv; ++k) {
        float m = mass[dof * nv + k];
        primal_compensated_product_add(m, acc[k] - q0[k],
            mass_sum, mass_correction);
        cost += 0.5f * displacement * m * (acc[k] - q0[k]);
      }
      gradient[dof] = mass_sum + mass_correction;
    }
  }
  // The sparse mass-difference helper uses generalized_force as its
  // triangular-solve scratch (L^T*(acc-q0)); the corresponding mass product
  // is already in gradient. Clear the reused vector before accumulating J^T f.
  for (int dof = 0; dof < nv; ++dof) generalized_force[dof] = 0.0f;
  for (int dof = 0; dof < nv; ++dof) {
    float sum = 0.0f, correction = 0.0f;
    for (int row = 0; row < n; ++row)
      if (enabled[row] && force[row] != 0.0f)
        primal_compensated_product_add(ccj_get(J, row, dof, nv), force[row],
            sum, correction);
    generalized_force[dof] = sum + correction;
  }
  for (int dof = 0; dof < nv; ++dof) gradient[dof] -= generalized_force[dof];
  return cost;
}

// Acceleration-space cost, gradient force, and row curvature for elliptic
// contact blocks. The three cone zones and anisotropic tangent scaling follow
// MuJoCo's PrimalPrepare/PrimalEval formulas.
inline float primal_elliptic_cost_force_hessian(
    thread const float* residual, device const float* R,
    device const float* friction, int dim, thread float* force,
    thread float* hessian) {
  for (int i = 0; i < 36; ++i) hessian[i] = 0.0f;
  for (int i = 0; i < 6; ++i) force[i] = 0.0f;
  // `contact_friction[0]` is not always `con->mu`: MuJoCo adjusts the
  // contact master coefficient after it finalizes normal/tangent R.
  float mu = max(friction[0] * sqrt(max(R[1], 0.0f)
      / max(R[0], 1e-15f)), 0.0f);
  if (!(mu > 1e-12f)) {
    if (residual[0] < 0.0f) {
      float D = 1.0f / max(R[0], 1e-15f);
      force[0] = -D * residual[0];
      hessian[0] = D;
      return 0.5f * D * residual[0] * residual[0];
    }
    return 0.0f;
  }
  float tangent_sq = 0.0f;
  for (int j = 1; j < dim; ++j) {
    float f = max(friction[j - 1], 0.0f);
    float scaled = f * residual[j];
    tangent_sq += scaled * scaled;
  }
  float tangent = sqrt(max(tangent_sq, 0.0f));
  float normal = residual[0];
  float D[6];
  for (int j = 0; j < 6; ++j) D[j] = j < dim ? 1.0f / max(R[j], 1e-15f) : 0.0f;
  if (tangent <= 0.0f || mu * mu * normal + tangent <= 0.0f) {
    if (mu * mu * normal + tangent <= 0.0f || normal < 0.0f) {
      float cost = 0.0f;
      for (int j = 0; j < dim; ++j) {
        force[j] = -D[j] * residual[j];
        hessian[j * 6 + j] = D[j];
        cost += 0.5f * D[j] * residual[j] * residual[j];
      }
      return cost;
    }
    return 0.0f;
  }
  if (normal >= tangent) return 0.0f;

  float scale = D[0] / (1.0f + mu * mu);
  float gap = normal - tangent;
  force[0] = -scale * gap;
  float a[6];
  for (int j = 0; j < 6; ++j) a[j] = 0.0f;
  for (int j = 1; j < dim; ++j) {
    float f = max(friction[j - 1], 0.0f);
    a[j] = f * f * residual[j];
    force[j] = scale * gap * a[j] / tangent;
  }
  hessian[0] = scale;
  for (int j = 1; j < dim; ++j) {
    hessian[j] = hessian[j * 6] = -scale * a[j] / tangent;
    float fj = max(friction[j - 1], 0.0f);
    for (int k = 1; k < dim; ++k) {
      float second_t = ((j == k ? fj * fj : 0.0f) / tangent)
          - (a[j] * a[k]) / (tangent * tangent * tangent);
      hessian[j * 6 + k] = scale * a[j] * a[k] / (tangent * tangent)
          - scale * gap * second_t;
    }
  }
  return 0.5f * scale * gap * gap;
}

inline float primal_accel_eval_elliptic(device const float* mass,
    device const float* L, bool sparse_mass,
    device const float* debug, constant int* dims,
    device const float* J, device const float* R, device const float* aref,
    device const float* lo, device const float* hi,
    device const float* q0, device const float* acc,
    device const int* enabled, device const int* elliptic_member,
    device const int* elliptic_start, device const int* elliptic_dim,
    device const float* elliptic_friction, int elliptic_count,
    device float* force, device float* gradient, device float* row_hessian,
    device float* residual, device float* generalized_force,
    int nv, int nr, int n, int n_eq_rows,
    device const float* mass_delta, device const float* row_residual,
    device const float* row_residual_low,
    bool incremental_state) {
  device const float* aref_low = primal_aref_low_view(debug, dims, nv, nr);
  float cost = primal_accel_eval_scalar(mass, L, sparse_mass,
      debug, dims, J, R, aref, lo, hi, q0, acc,
      enabled, elliptic_member, force, gradient, row_hessian, residual,
      generalized_force, nv, nr, n, n_eq_rows, mass_delta, row_residual,
      row_residual_low, incremental_state);
  for (int block = 0; block < elliptic_count; ++block) {
    int start = elliptic_start[block], dim = elliptic_dim[block];
    if (start < 0 || dim < 2 || dim > 6 || start + dim > n || !enabled[start]) continue;
    thread float local_residual[6], local_force[6], local_hessian[36];
    for (int k = 0; k < dim; ++k) {
      float value;
      if (incremental_state) {
        value = row_residual[start + k]
            + (row_residual_low == nullptr ? 0.0f
                                           : row_residual_low[start + k]);
      } else {
        float sum = -aref[start + k];
        float correction = -aref_low[start + k];
        for (int dof = 0; dof < nv; ++dof)
          primal_compensated_product_add(
              ccj_get(J, (start + k), dof, nv), acc[dof], sum, correction);
        value = sum + correction;
      }
      residual[start + k] = local_residual[k] = value;
    }
    cost += primal_elliptic_cost_force_hessian(local_residual, R + start,
        elliptic_friction + block * 5, dim, local_force, local_hessian);
    for (int k = 0; k < dim; ++k) {
      force[start + k] = local_force[k];
      row_hessian[start + k] = 0.0f;
      if (local_force[k] != 0.0f) for (int dof = 0; dof < nv; ++dof)
        generalized_force[dof] = fma(
            ccj_get(J, (start + k), dof, nv), local_force[k], generalized_force[dof]);
    }
    // The scalar evaluator already subtracted its generalized force. Add the
    // elliptic block gradient here with the matching sign convention.
    for (int dof = 0; dof < nv; ++dof) {
      float block_force = 0.0f, block_force_correction = 0.0f;
      for (int k = 0; k < dim; ++k)
        primal_compensated_product_add(ccj_get(J, (start + k), dof, nv),
            local_force[k], block_force, block_force_correction);
      gradient[dof] -= block_force + block_force_correction;
    }
  }
  return cost;
}


inline void primal_mass_solve(device const float* L, device const float* rhs,
    device float* solution, device float* scratch, int nv) {
  for (int i = 0; i < nv; ++i) {
    float v = rhs[i];
    for (int k = 0; k < i; ++k)
      v = fma(-L[i * nv + k], scratch[k], v);
    scratch[i] = v / L[i * nv + i];
  }
  for (int i = nv - 1; i >= 0; --i) {
    float v = scratch[i];
    for (int k = i + 1; k < nv; ++k)
      v = fma(-L[k * nv + i], solution[k], v);
    solution[i] = v / L[i * nv + i];
  }
}

// Float32 CG/Newton systems can have a small physical RHS after the Hessian
// has been formed from much larger row terms. Keep product error while
// accumulating reductions so the true-residual check measures the original
// operator instead of cancellation in a long float sum.
inline void primal_compensated_product_add(float a, float b,
    thread float& sum, thread float& correction) {
  float product = a * b;
  float product_error = fma(a, b, -product);
  float next = sum + product;
  float addition_error = abs(sum) >= abs(product)
      ? (sum - next) + product
      : (product - next) + sum;
  correction += addition_error + product_error;
  sum = next;
}

inline void primal_compensated_product3_add(float a, float b, float c,
    thread float& sum, thread float& correction) {
  float ab = a * b;
  float ab_error = fma(a, b, -ab);
  float product = ab * c;
  float product_error = fma(ab, c, -product) + ab_error * c;
  float next = sum + product;
  float addition_error = abs(sum) >= abs(product)
      ? (sum - next) + product
      : (product - next) + sum;
  correction += addition_error + product_error;
  sum = next;
}

inline float primal_compensated_dot(device const float* a,
    device const float* b, int count) {
  float sum = 0.0f, correction = 0.0f;
  for (int i = 0; i < count; ++i)
    primal_compensated_product_add(a[i], b[i], sum, correction);
  return sum + correction;
}

// A two-float expansion carries sub-ULP Newton updates across the bounded
// float32 PCG and line-search stages.  The low word is part of the solver
// state, not a relaxed residual allowance: convergence checks evaluate the
// original operator on hi+lo before the public float32 result is emitted.
struct PrimalFloatPair {
  float hi;
  float lo;
};

inline PrimalFloatPair primal_pair_renormalize(float hi, float lo) {
  float sum = hi + lo;
  float virtual_lo = sum - hi;
  float error = (hi - (sum - virtual_lo)) + (lo - virtual_lo);
  return {sum, error};
}

inline PrimalFloatPair primal_pair_add(PrimalFloatPair a,
    PrimalFloatPair b) {
  float sum = a.hi + b.hi;
  float virtual_b = sum - a.hi;
  float error = (a.hi - (sum - virtual_b)) + (b.hi - virtual_b);
  error += a.lo + b.lo;
  return primal_pair_renormalize(sum, error);
}

inline PrimalFloatPair primal_pair_multiply(PrimalFloatPair a, float b) {
  float product = a.hi * b;
  float error = fma(a.hi, b, -product);
  float low_product = a.lo * b;
  float low_error = fma(a.lo, b, -low_product);
  return primal_pair_add(primal_pair_renormalize(product, error),
      {low_product, low_error});
}

inline PrimalFloatPair primal_pair_fma(float a, PrimalFloatPair b,
    PrimalFloatPair c) {
  float product = a * b.hi;
  float error = fma(a, b.hi, -product);
  float low_product = a * b.lo;
  float low_error = fma(a, b.lo, -low_product);
  PrimalFloatPair term = primal_pair_add(
      primal_pair_renormalize(product, error),
      {low_product, low_error});
  return primal_pair_add(term, c);
}

inline float primal_pair_dot(device const float* a_hi,
    device const float* a_lo, device const float* b_hi,
    device const float* b_lo, int count) {
  float sum = 0.0f, correction = 0.0f;
  for (int i = 0; i < count; ++i) {
    primal_compensated_product_add(a_hi[i], b_hi[i], sum, correction);
    primal_compensated_product_add(a_hi[i], b_lo[i], sum, correction);
    primal_compensated_product_add(a_lo[i], b_hi[i], sum, correction);
    primal_compensated_product_add(a_lo[i], b_lo[i], sum, correction);
  }
  return sum + correction;
}

inline PrimalFloatPair primal_pair_product(PrimalFloatPair a,
    PrimalFloatPair b) {
  float sum = 0.0f, correction = 0.0f;
  primal_compensated_product_add(a.hi, b.hi, sum, correction);
  primal_compensated_product_add(a.hi, b.lo, sum, correction);
  primal_compensated_product_add(a.lo, b.hi, sum, correction);
  primal_compensated_product_add(a.lo, b.lo, sum, correction);
  return primal_pair_renormalize(sum, correction);
}

inline PrimalFloatPair primal_pair_fma_pair(PrimalFloatPair a,
    PrimalFloatPair b, PrimalFloatPair c) {
  return primal_pair_add(primal_pair_product(a, b), c);
}

inline bool primal_pair_is_finite(PrimalFloatPair value) {
  return isfinite(value.hi) && isfinite(value.lo);
}

inline PrimalFloatPair primal_compensated_dot_pair(
    device const float* a, device const float* b, int count) {
  float sum = 0.0f, correction = 0.0f;
  for (int i = 0; i < count; ++i)
    primal_compensated_product_add(a[i], b[i], sum, correction);
  return primal_pair_renormalize(sum, correction);
}

// Hessian operators receive the canonical row-record pointer, not a dense
// row-major float plane. Resolve every coefficient through the validated
// dense/CSR accessor while retaining the compensated reduction.
inline PrimalFloatPair primal_ccj_compensated_dot_pair(
    device const float* J, int row, device const float* vector, int nv) {
  float sum = 0.0f, correction = 0.0f;
  for (int dof = 0; dof < nv; ++dof)
    primal_compensated_product_add(ccj_get(J, row, dof, nv), vector[dof],
        sum, correction);
  return primal_pair_renormalize(sum, correction);
}

inline PrimalFloatPair primal_pair_dot_expansion(device const float* a_hi,
    device const float* a_lo, device const float* b_hi,
    device const float* b_lo, int count) {
  float sum = 0.0f, correction = 0.0f;
  for (int i = 0; i < count; ++i) {
    primal_compensated_product_add(a_hi[i], b_hi[i], sum, correction);
    primal_compensated_product_add(a_hi[i], b_lo[i], sum, correction);
    primal_compensated_product_add(a_lo[i], b_hi[i], sum, correction);
    primal_compensated_product_add(a_lo[i], b_lo[i], sum, correction);
  }
  return primal_pair_renormalize(sum, correction);
}

inline PrimalFloatPair primal_pair_divide(PrimalFloatPair numerator,
    PrimalFloatPair denominator) {
  float q0 = numerator.hi / denominator.hi;
  PrimalFloatPair remainder = primal_pair_add(numerator,
      primal_pair_multiply(denominator, -q0));
  float q1 = (remainder.hi + remainder.lo) / denominator.hi;
  return primal_pair_renormalize(q0, q1);
}

inline bool primal_float_is_subnormal(float value) {
  uint bits = as_type<uint>(value);
  return (bits & 0x7f800000u) == 0u && (bits & 0x007fffffu) != 0u;
}

inline bool primal_float_bits_nonzero(float value) {
  return (as_type<uint>(value) & 0x7fffffffu) != 0u;
}

// Scale-safe norm of a represented two-word vector. Scale from hi+lo, not
// either storage word separately, so cancellation cannot square a small
// residual down to zero and produce a false convergence certificate.
inline float primal_pair_norm(device const float* hi, device const float* lo,
    int count) {
  float scale = 0.0f;
  for (int i = 0; i < count; ++i) {
    if (!isfinite(hi[i]) || !isfinite(lo[i])) return INFINITY;
    PrimalFloatPair value = primal_pair_add({hi[i], 0.0f}, {lo[i], 0.0f});
    if (!isfinite(value.hi) || !isfinite(value.lo)
        || primal_float_is_subnormal(value.hi)
        || primal_float_is_subnormal(value.lo)) return INFINITY;
    scale = max(scale, max(abs(value.hi), abs(value.lo)));
  }
  if (scale == 0.0f) return 0.0f;
  if (scale < 1.17549435e-38f) return INFINITY;
  float sum = 0.0f, correction = 0.0f;
  for (int i = 0; i < count; ++i) {
    PrimalFloatPair represented = primal_pair_add(
        {hi[i], 0.0f}, {lo[i], 0.0f});
    PrimalFloatPair value = primal_pair_add(
        {represented.hi / scale, 0.0f}, {represented.lo / scale, 0.0f});
    PrimalFloatPair square = primal_pair_product(value, value);
    primal_compensated_product_add(1.0f, square.hi, sum, correction);
    primal_compensated_product_add(1.0f, square.lo, sum, correction);
  }
  float normalized_sq = sum + correction;
  if (!(normalized_sq >= 0.0f) || !isfinite(normalized_sq)) return INFINITY;
  float norm = scale * sqrt(normalized_sq);
  return isfinite(norm) ? norm : INFINITY;
}

// Apply one residual-correction solve with the already-built Cholesky factor.
// The RHS and solution use two-word expansions so the final acceleration can
// retain the mass-equation residual that a float-only triangular solve drops.
inline void primal_cholesky_solve_pair(device const float* L, int nv,
    device const int* awake_tree, constant int* dims,
    device const float* rhs_hi, device const float* rhs_lo,
    device float* out_hi, device float* out_lo,
    device float* y_hi, device float* y_lo) {
  for (int i = 0; i < nv; ++i) {
    if (!solver_dof_awake(dims, awake_tree, i)) {
      y_hi[i] = 0.0f;
      y_lo[i] = 0.0f;
      continue;
    }
    PrimalFloatPair value = {rhs_hi[i], rhs_lo[i]};
    for (int k = 0; k < i; ++k)
      value = primal_pair_fma(-L[i * nv + k],
          {y_hi[k], y_lo[k]}, value);
    PrimalFloatPair solved = primal_pair_divide(value,
        {L[i * nv + i], 0.0f});
    y_hi[i] = solved.hi;
    y_lo[i] = solved.lo;
  }
  for (int i = nv - 1; i >= 0; --i) {
    if (!solver_dof_awake(dims, awake_tree, i)) {
      out_hi[i] = 0.0f;
      out_lo[i] = 0.0f;
      continue;
    }
    PrimalFloatPair value = {y_hi[i], y_lo[i]};
    for (int k = i + 1; k < nv; ++k)
      value = primal_pair_fma(-L[k * nv + i],
          {out_hi[k], out_lo[k]}, value);
    PrimalFloatPair solved = primal_pair_divide(value,
        {L[i * nv + i], 0.0f});
    out_hi[i] = solved.hi;
    out_lo[i] = solved.lo;
  }
}

inline bool primal_pair_positive(PrimalFloatPair value) {
  return value.hi > 0.0f || (value.hi == 0.0f && value.lo > 0.0f);
}

inline bool primal_pair_negative(PrimalFloatPair value) {
  return value.hi < 0.0f || (value.hi == 0.0f && value.lo < 0.0f);
}

// Round an expansion multiplied by an exact power of two without asking the
// hardware to materialize a subnormal intermediate.  In the subnormal output
// range the scaled value is first expressed in units of 2^-149 (which are
// ordinary, accurately representable floats), rounded to an integer with
// ties-to-even, then written directly as IEEE-754 bits.
inline float primal_pair_scale_pow2_rne(PrimalFloatPair value, int exponent) {
  if (!isfinite(value.hi) || !isfinite(value.lo)) return NAN;
  uint source_hi_bits = as_type<uint>(value.hi);
  uint source_lo_bits = as_type<uint>(value.lo);
  bool hi_nonzero = (source_hi_bits & 0x7fffffffu) != 0u;
  bool negative = hi_nonzero ? ((source_hi_bits & 0x80000000u) != 0u)
                             : ((source_lo_bits & 0x80000000u) != 0u);
  if (negative) {
    value.hi = -value.hi;
    value.lo = -value.lo;
  }
  uint hi_bits = as_type<uint>(value.hi);
  int hi_exp = int((hi_bits >> 23) & 0xffu);
  if (hi_exp == 0 || hi_exp == 0xff) return NAN;
  int result_exp = hi_exp - 127 + exponent;
  // A negative low word immediately below an exact power of two belongs to
  // the preceding binade. This is the only boundary where the high word's
  // exponent alone would select the wrong rounding unit.
  if ((hi_bits & 0x007fffffu) == 0u
      && primal_float_bits_nonzero(value.lo)
      && (as_type<uint>(value.lo) & 0x80000000u) != 0u) --result_exp;

  uint sign = negative ? 0x80000000u : 0u;
  if (result_exp >= -126) {
    if (result_exp > 127) return as_type<float>(sign | 0x7f800000u);
    int shift = 23 - result_exp;
    float units_hi = ldexp(value.hi, exponent + shift);
    float units_lo = ldexp(value.lo, exponent + shift);
    int rounded = int(units_hi);
    float correction = units_lo;
    if (correction > 0.5f || (correction == 0.5f && (rounded & 1))) {
      ++rounded;
    } else if (correction < -0.5f
        || (correction == -0.5f && (rounded & 1))) {
      --rounded;
    }
    if (rounded >= 0x01000000) {
      rounded >>= 1;
      ++result_exp;
      if (result_exp > 127) return as_type<float>(sign | 0x7f800000u);
    } else if (rounded < 0x00800000) {
      // This can only occur at the normal/subnormal boundary after a negative
      // correction. Re-enter the subnormal path with its fixed output quantum.
      result_exp = -127;
    }
    if (result_exp >= -126) {
      uint bits = sign | (uint(result_exp + 127) << 23)
          | (uint(rounded) - 0x00800000u);
      return as_type<float>(bits);
    }
  }

  float units_hi = ldexp(value.hi, exponent + 149);
  float units_lo = ldexp(value.lo, exponent + 149);
  float integral = floor(units_hi);
  int rounded = int(integral);
  // Keep both half-integer comparisons as expansions. Adding a tiny low word
  // to exactly +/-0.5f can round back to the tie and choose the wrong even
  // neighbor; it can also cross the negative half boundary and decrement.
  float fraction = units_hi - integral;
  float fraction_delta = (units_hi - integral) - 0.5f;
  PrimalFloatPair half_delta = primal_pair_add(
      {fraction_delta, 0.0f}, {units_lo, 0.0f});
  PrimalFloatPair lower_half_delta = primal_pair_add(
      primal_pair_add({fraction, 0.0f}, {0.5f, 0.0f}),
      {units_lo, 0.0f});
  bool above_half = primal_pair_positive(half_delta);
  bool below_half = primal_pair_negative(half_delta);
  bool below_negative_half = primal_pair_negative(lower_half_delta);
  bool at_negative_half = !primal_float_bits_nonzero(lower_half_delta.hi)
      && !primal_float_bits_nonzero(lower_half_delta.lo);
  // If scaling the low correction into subnormal integer units was flushed,
  // its sign still decides an exact high-word tie. The scale is positive.
  if (!above_half && !below_half && units_lo == 0.0f
      && primal_float_bits_nonzero(value.lo)) {
    above_half = value.lo > 0.0f;
    below_half = value.lo < 0.0f;
  }
  if (above_half || (!below_half && !above_half && (rounded & 1))) {
    ++rounded;
  } else if (below_negative_half || (at_negative_half && (rounded & 1))) {
    --rounded;
  }
  if (rounded <= 0) return as_type<float>(sign);
  if (rounded >= 0x00800000)
    return as_type<float>(sign | 0x00800000u);
  return as_type<float>(sign | uint(rounded));
}

inline float primal_float_scale_pow2_rne(float value, int exponent) {
  uint bits = as_type<uint>(value);
  if ((bits & 0x7fffffffu) == 0u) return value;
  uint exponent_field = (bits >> 23) & 0xffu;
  if (exponent_field == 0xffu) return value;
  if (exponent_field != 0u)
    return primal_pair_scale_pow2_rne({value, 0.0f}, exponent);
  // A subnormal word is fraction * 2^-149. Convert the integer fraction to a
  // normal mantissa before applying the requested exponent, so even a valid
  // low word is not flushed by an arithmetic multiply.
  uint fraction = bits & 0x007fffffu;
  float normalized = float(fraction) * 0x1p-23f;
  if ((bits & 0x80000000u) != 0u) normalized = -normalized;
  return primal_pair_scale_pow2_rne({normalized, 0.0f}, exponent - 126);
}

// Keep the RHS scale's exponent out of intermediate products. Multiplying a
// normal expansion by the scale mantissa preserves its low word; apply the
// final exponent independently to both expansion words. Collapsing the pair
// here would lose internal Newton precision even when the result is normal.
inline PrimalFloatPair primal_pair_multiply_scaled(PrimalFloatPair value,
    float scale) {
  uint bits = as_type<uint>(scale);
  int exponent_field = int((bits >> 23) & 0xffu);
  if (exponent_field == 0 || exponent_field == 0xff)
    return {NAN, NAN};
  // Use a [1, 2) mantissa so a minimum-normal solver word is not pushed into
  // the FTZ range before its exponent is applied. Near the top of float32,
  // use [0.5, 1) instead so a finite final result cannot overflow early.
  bool large_value = max(abs(value.hi), abs(value.lo))
      > 8.5070586659632215e37f;
  uint mantissa_bits = (bits & 0x807fffffu)
      | ((large_value ? 126u : 127u) << 23);
  float mantissa = as_type<float>(mantissa_bits);
  int exponent = exponent_field - (large_value ? 126 : 127);
  PrimalFloatPair product = primal_pair_multiply(value, mantissa);
  float high = primal_float_scale_pow2_rne(product.hi, exponent);
  float low = primal_float_scale_pow2_rne(product.lo, exponent);
  bool lost_high = primal_float_bits_nonzero(product.hi)
      && !primal_float_bits_nonzero(high);
  bool lost_low = primal_float_bits_nonzero(product.lo)
      && !primal_float_bits_nonzero(low);
  // If either expansion limb falls below the output lattice, it can still
  // decide a tie in the other limb. Round the complete pair on that lattice
  // before the lost correction can disappear.
  if (lost_high || lost_low) {
    float combined = primal_pair_scale_pow2_rne(product, exponent);
    if (!isfinite(combined)) return {NAN, NAN};
    return {combined, 0.0f};
  }
  if (!primal_float_bits_nonzero(high) && primal_float_bits_nonzero(low)) {
    high = low;
    low = 0.0f;
  }
  return {high, low};
}

inline void primal_rescale_trace_failure(device float* trace_out, int code_base,
    int dof, int subcase, PrimalFloatPair output) {
  if (trace_out == nullptr || trace_out[9] != 0.0f) return;
  // [7:9] is an opt-in failure witness: the rejected physical-unit output
  // pair and a code carrying the strict/roundoff site, active DOF, and
  // failure subcase. The normalized input pair and RHS scale remain in the
  // caller-owned PCG scratch and are copied by the test immediately after
  // this selected dispatch returns.
  trace_out[7] = isfinite(output.hi) ? output.hi : 0.0f;
  trace_out[8] = isfinite(output.lo) ? output.lo : 0.0f;
  trace_out[9] = float(code_base + 2 * dof + subcase);
}

inline bool primal_rescale_pair_vector_impl(device float* hi, device float* lo,
    device const int* awake_tree, constant int* dims,
    device const int* dof_island, int island, bool partitioned,
    float scale, int nv, device float* trace_out, int trace_code_base) {
  for (int dof = 0; dof < nv; ++dof) {
    if (!solver_dof_awake(dims, awake_tree, dof)
        || (partitioned && dof_island[dof] != island)) {
      hi[dof] = lo[dof] = 0.0f;
      continue;
    }
    PrimalFloatPair value = primal_pair_multiply_scaled(
        {hi[dof], lo[dof]}, scale);
    if (!isfinite(value.hi) || !isfinite(value.lo)) {
      primal_rescale_trace_failure(trace_out, trace_code_base, dof, 1,
          value);
      return false;
    }
    // A nonzero normalized component may correctly round to signed zero in
    // physical units. The scaled helper above first combines both limbs when
    // independent rounding would erase a representable min-subnormal, so a
    // zero pair here is the exact binary32 RNE result, not a flush-to-zero.
    hi[dof] = value.hi;
    lo[dof] = value.lo;
  }
  return true;
}

inline bool primal_rescale_pair_vector(device float* hi, device float* lo,
    device const int* awake_tree, constant int* dims,
    device const int* dof_island, int island, bool partitioned,
    float scale, int nv) {
  return primal_rescale_pair_vector_impl(hi, lo, awake_tree, dims,
      dof_island, island, partitioned, scale, nv, nullptr, 0);
}

inline bool primal_rescale_pair_vector_traced(device float* hi, device float* lo,
    device const int* awake_tree, constant int* dims,
    device const int* dof_island, int island, bool partitioned,
    float scale, int nv, device float* trace_out, int trace_code_base) {
  return primal_rescale_pair_vector_impl(hi, lo, awake_tree, dims,
      dof_island, island, partitioned, scale, nv, trace_out, trace_code_base);
}

inline float primal_pair_value(PrimalFloatPair value) {
  return value.hi + value.lo;
}

inline device const float* primal_aref_low_view(device const float* debug,
    constant int* dims, int nv, int nr) {
  int metadata = dims[24] + nr * 10;
  device const float* solver_scratch = debug + nr * nr + 7 * nr;
  device const float* pair_tail = solver_scratch + dims[metadata + 15];
  return pair_tail + 8 * max(nv, 1) + 2 * max(nr, 1);
}

inline bool primal_pair_less_float(PrimalFloatPair value, float scalar) {
  if (value.hi < scalar) return true;
  if (value.hi > scalar) return false;
  return value.lo < 0.0f;
}

inline bool primal_pair_less_equal_float(PrimalFloatPair value, float scalar) {
  if (value.hi < scalar) return true;
  if (value.hi > scalar) return false;
  return value.lo <= 0.0f;
}

inline float primal_pair_abs_value(PrimalFloatPair value) {
  float result = abs(value.hi + value.lo);
  return result;
}

inline bool primal_pair_less(PrimalFloatPair a, PrimalFloatPair b) {
  PrimalFloatPair d = primal_pair_add(a, primal_pair_multiply(b, -1.0f));
  return primal_pair_negative(d);
}

inline bool primal_pair_less_equal(PrimalFloatPair a, PrimalFloatPair b) {
  PrimalFloatPair d = primal_pair_add(a, primal_pair_multiply(b, -1.0f));
  return primal_pair_negative(d) || (d.hi == 0.0f && d.lo == 0.0f);
}

inline PrimalFloatPair primal_pair_abs(PrimalFloatPair value) {
  return primal_pair_negative(value)
      ? primal_pair_multiply(value, -1.0f) : value;
}

inline PrimalFloatPair primal_pair_sqrt(PrimalFloatPair value) {
  float root = sqrt(max(value.hi, 0.0f));
  if (!(root > 0.0f)) return {root, 0.0f};
  PrimalFloatPair square = primal_pair_product({root, 0.0f}, {root, 0.0f});
  PrimalFloatPair remainder = primal_pair_add(value,
      {-square.hi, -square.lo});
  PrimalFloatPair correction = primal_pair_divide(remainder,
      {2.0f * root, 0.0f});
  return primal_pair_renormalize(root, correction.hi + correction.lo);
}

// Evaluate the pinned elliptic constraint force on the represented
// high+low row residual. The ordinary cost path keeps a float32 force for
// its public high plane; this pair is used to refresh the primal gradient and
// prevents arithmetic rounding in the force formula from masquerading as a
// nonconverged Newton residual.
inline void primal_elliptic_force_pair(
    thread const float* residual_hi, thread const float* residual_low,
    device const float* R, device const float* friction, int dim,
    thread float* force_hi, thread float* force_low) {
  for (int j = 0; j < 6; ++j) force_hi[j] = force_low[j] = 0.0f;
  PrimalFloatPair mu = primal_pair_multiply(
      primal_pair_sqrt(primal_pair_divide(
          {max(R[1], 0.0f), 0.0f}, {max(R[0], 1e-15f), 0.0f})),
      friction[0]);
  PrimalFloatPair normal = {residual_hi[0], residual_low[0]};
  PrimalFloatPair mu_abs = primal_pair_abs(mu);
  if (mu_abs.hi < 1e-12f
      || (mu_abs.hi == 1e-12f && mu_abs.lo <= 0.0f)) {
    if (primal_pair_negative(normal)) {
      PrimalFloatPair f = primal_pair_multiply(normal, -1.0f);
      f = primal_pair_divide(f, {max(R[0], 1e-15f), 0.0f});
      force_hi[0] = f.hi;
      force_low[0] = f.lo;
    }
    return;
  }
  PrimalFloatPair tangent_sq = {0.0f, 0.0f};
  PrimalFloatPair a[6];
  for (int j = 0; j < 6; ++j) a[j] = {0.0f, 0.0f};
  for (int j = 1; j < dim; ++j) {
    PrimalFloatPair fj = {max(friction[j - 1], 0.0f), 0.0f};
    PrimalFloatPair rj = {residual_hi[j], residual_low[j]};
    PrimalFloatPair scaled = primal_pair_product(fj, rj);
    tangent_sq = primal_pair_add(tangent_sq,
        primal_pair_product(scaled, scaled));
    a[j] = primal_pair_product(primal_pair_product(fj, fj), rj);
  }
  PrimalFloatPair tangent = primal_pair_sqrt(tangent_sq);
  PrimalFloatPair mu_sq = primal_pair_product(mu, mu);
  PrimalFloatPair bottom_test = primal_pair_add(
      primal_pair_product(mu_sq, normal), tangent);
  if (primal_pair_less_equal_float(tangent, 0.0f)) {
    if (primal_pair_negative(normal)) {
      for (int j = 0; j < dim; ++j) {
        PrimalFloatPair f = primal_pair_divide(
            primal_pair_multiply({residual_hi[j], residual_low[j]}, -1.0f),
            {max(R[j], 1e-15f), 0.0f});
        force_hi[j] = f.hi;
        force_low[j] = f.lo;
      }
    }
    return;
  }
  if (primal_pair_less_equal_float(bottom_test, 0.0f)) {
    for (int j = 0; j < dim; ++j) {
      PrimalFloatPair f = primal_pair_divide(
          primal_pair_multiply({residual_hi[j], residual_low[j]}, -1.0f),
          {max(R[j], 1e-15f), 0.0f});
      force_hi[j] = f.hi;
      force_low[j] = f.lo;
    }
    return;
  }
  PrimalFloatPair gap = primal_pair_add(normal,
      primal_pair_multiply(tangent, -1.0f));
  if (!primal_pair_negative(gap)) return;
  PrimalFloatPair scale = primal_pair_divide(
      primal_pair_divide({1.0f, 0.0f}, {max(R[0], 1e-15f), 0.0f}),
      primal_pair_add({1.0f, 0.0f}, mu_sq));
  PrimalFloatPair f0 = primal_pair_multiply(
      primal_pair_product(scale, gap), -1.0f);
  force_hi[0] = f0.hi;
  force_low[0] = f0.lo;
  for (int j = 1; j < dim; ++j) {
    PrimalFloatPair fjgap = primal_pair_product(scale, gap);
    PrimalFloatPair force = primal_pair_divide(
        primal_pair_product(fjgap, a[j]), tangent);
    force_hi[j] = force.hi;
    force_low[j] = force.lo;
  }
}

// Source-ordered elliptic cone Hessian in two-word arithmetic. This is the
// second derivative assembled by MuJoCo's HessianCone from the normalized
// dual-cone variables N=mu*jar[0], U[j]=friction[j-1]*jar[j], and
// T=norm(U[1:]). The represented row residual may itself have a low word.
inline void primal_elliptic_hessian_pair(
    thread const float* residual_hi, thread const float* residual_low,
    device const float* R, device const float* friction, int dim,
    thread float* hessian_hi, thread float* hessian_low) {
  for (int i = 0; i < 36; ++i) hessian_hi[i] = hessian_low[i] = 0.0f;
  PrimalFloatPair mu = primal_pair_multiply(
      primal_pair_sqrt(primal_pair_divide(
          {max(R[1], 0.0f), 0.0f}, {max(R[0], 1e-15f), 0.0f})),
      friction[0]);
  PrimalFloatPair mu_abs = primal_pair_abs(mu);
  if (mu_abs.hi < 1e-12f
      || (mu_abs.hi == 1e-12f && mu_abs.lo <= 0.0f)) {
    if (primal_pair_negative({residual_hi[0], residual_low[0]})) {
      PrimalFloatPair inverse = primal_pair_divide({1.0f, 0.0f},
          {max(R[0], 1e-15f), 0.0f});
      hessian_hi[0] = inverse.hi;
      hessian_low[0] = inverse.lo;
    }
    return;
  }
  PrimalFloatPair tangent_sq = {0.0f, 0.0f};
  PrimalFloatPair a[6];
  for (int j = 0; j < 6; ++j) a[j] = {0.0f, 0.0f};
  for (int j = 1; j < dim; ++j) {
    PrimalFloatPair r = {residual_hi[j], residual_low[j]};
    PrimalFloatPair f = {max(friction[j - 1], 0.0f), 0.0f};
    PrimalFloatPair scaled = primal_pair_product(f, r);
    tangent_sq = primal_pair_add(tangent_sq,
        primal_pair_product(scaled, scaled));
    a[j] = primal_pair_product(primal_pair_product(f, f), r);
  }
  PrimalFloatPair tangent = primal_pair_sqrt(tangent_sq);
  PrimalFloatPair normal = {residual_hi[0], residual_low[0]};
  PrimalFloatPair mu_sq = primal_pair_product(mu, mu);
  PrimalFloatPair normal_minus_tangent = primal_pair_add(normal,
      primal_pair_multiply(tangent, -1.0f));
  if (primal_pair_less_equal_float(tangent, 0.0f)) {
    if (primal_pair_negative(normal)) {
      for (int j = 0; j < dim; ++j) {
        PrimalFloatPair inverse = primal_pair_divide({1.0f, 0.0f},
            {max(R[j], 1e-15f), 0.0f});
        hessian_hi[j * 6 + j] = inverse.hi;
        hessian_low[j * 6 + j] = inverse.lo;
      }
    }
    return;
  }
  PrimalFloatPair bottom_test = primal_pair_add(
      primal_pair_product(mu_sq, normal), tangent);
  if (primal_pair_less_equal_float(bottom_test, 0.0f)) {
    for (int j = 0; j < dim; ++j) {
      PrimalFloatPair inverse = primal_pair_divide({1.0f, 0.0f},
          {max(R[j], 1e-15f), 0.0f});
      hessian_hi[j * 6 + j] = inverse.hi;
      hessian_low[j * 6 + j] = inverse.lo;
    }
    return;
  }
  if (!primal_pair_negative(normal_minus_tangent)) return;

  PrimalFloatPair scale = primal_pair_divide(
      primal_pair_divide({1.0f, 0.0f}, {max(R[0], 1e-15f), 0.0f}),
      primal_pair_add({1.0f, 0.0f}, mu_sq));
  hessian_hi[0] = scale.hi;
  hessian_low[0] = scale.lo;
  PrimalFloatPair tangent_sq_den = primal_pair_product(tangent, tangent);
  PrimalFloatPair tangent_cube = primal_pair_product(tangent_sq_den, tangent);
  for (int j = 1; j < dim; ++j) {
    PrimalFloatPair fj = {max(friction[j - 1], 0.0f), 0.0f};
    PrimalFloatPair fj_sq = primal_pair_product(fj, fj);
    PrimalFloatPair h0j = primal_pair_multiply(
        primal_pair_divide(primal_pair_product(scale, a[j]), tangent), -1.0f);
    hessian_hi[j] = hessian_hi[j * 6] = h0j.hi;
    hessian_low[j] = hessian_low[j * 6] = h0j.lo;
    for (int k = 1; k < dim; ++k) {
      PrimalFloatPair ajak = primal_pair_product(a[j], a[k]);
      PrimalFloatPair second = primal_pair_divide(ajak, tangent_cube);
      PrimalFloatPair diagonal = j == k
          ? primal_pair_product(fj_sq, primal_pair_add({1.0f, 0.0f},
              primal_pair_multiply(primal_pair_divide(normal, tangent), -1.0f)))
          : PrimalFloatPair{0.0f, 0.0f};
      PrimalFloatPair hjk = primal_pair_product(scale,
          primal_pair_add(diagonal,
              primal_pair_product(normal, second)));
      // The normalized dual Hessian simplifies to this source-equivalent
      // form: scale * (delta*f_j^2*(1-N/(mu*T)) + r0*a_j*a_k/T^3).
      // Here N/(mu*T) = r0/T, and the off-diagonal term uses a_j*a_k.
      hessian_hi[j * 6 + k] = hjk.hi;
      hessian_low[j * 6 + k] = hjk.lo;
    }
  }
}

// Refresh the internal Newton state without rounding accepted sub-ULP updates
// out of acceleration, row residual, force, or gradient.  Public output is
// still emitted as float32; the pair state is used only while the bounded
// solver is active.
inline void primal_refresh_pair_state(device const float* mass,
    bool sparse_mass, device const float* debug, constant int* dims,
    device const float* J, device const float* R, device const float* aref,
    device const float* lo, device const float* hi,
    device const float* q0, device const float* q0_low,
    device const float* acc,
    device const float* acc_low,
    device const int* enabled, device const int* elliptic_member,
    device const int* elliptic_start, device const int* elliptic_dim,
    device const float* elliptic_friction, int elliptic_count,
    device float* force, device float* force_low,
    device float* gradient, device float* gradient_low,
    device float* row_hessian, device float* row_residual,
    device float* row_residual_low, int nv, int nr, int n, int n_eq_rows) {
  device const float* aref_low = primal_aref_low_view(debug, dims, nv, nr);
  for (int dof = 0; dof < nv; ++dof) {
    float value = 0.0f, correction = 0.0f;
    for (int other = 0; other < nv; ++other) {
      float entry = sparse_mass
          ? solver_component_mass_entry(debug, dims, nv, dof, other)
          : mass[dof * nv + other];
      primal_compensated_product_add(entry, acc[other], value, correction);
      primal_compensated_product_add(entry, acc_low[other], value, correction);
      primal_compensated_product_add(-entry, q0[other], value, correction);
      primal_compensated_product_add(-entry, q0_low[other], value, correction);
    }
    PrimalFloatPair g = primal_pair_renormalize(value, correction);
    gradient[dof] = g.hi;
    gradient_low[dof] = g.lo;
  }

  for (int row = 0; row < n; ++row) {
    force[row] = force_low[row] = row_hessian[row] = 0.0f;
    // The pair represents aref_hi + aref_low, so J*qacc - aref must
    // subtract both words.  Keeping the low word positive here doubles its
    // contribution and perturbs the refreshed gradient after each accepted
    // line-search update.
    float sum = -aref[row], correction = -aref_low[row];
    for (int dof = 0; dof < nv; ++dof) {
      primal_compensated_product_add(ccj_get(J, row, dof, nv), acc[dof],
          sum, correction);
      primal_compensated_product_add(ccj_get(J, row, dof, nv), acc_low[dof],
          sum, correction);
    }
    PrimalFloatPair r = primal_pair_renormalize(sum, correction);
    row_residual[row] = r.hi;
    row_residual_low[row] = r.lo;
    if (!enabled[row] || elliptic_member[row]) continue;
    float reg = max(R[row], 1e-15f);
    float D = 1.0f / reg;
    bool friction = lo[row] < 0.0f && hi[row] > 0.0f
        && isfinite(lo[row]) && isfinite(hi[row])
        && abs(lo[row] + hi[row]) <= 1e-5f * max(1.0f, abs(hi[row]));
    float residual_value = r.hi + r.lo;
    if (row < n_eq_rows) {
      PrimalFloatPair f = primal_pair_multiply(r, -D);
      force[row] = f.hi;
      force_low[row] = f.lo;
      row_hessian[row] = D;
    } else if (friction) {
      float bound = reg * hi[row];
      if (residual_value <= -bound) force[row] = hi[row];
      else if (residual_value >= bound) force[row] = -hi[row];
      else {
        PrimalFloatPair f = primal_pair_multiply(r, -D);
        force[row] = f.hi;
        force_low[row] = f.lo;
        row_hessian[row] = D;
      }
    } else if (residual_value < 0.0f) {
      PrimalFloatPair f = primal_pair_multiply(r, -D);
      force[row] = f.hi;
      force_low[row] = f.lo;
      row_hessian[row] = D;
    }
  }

  for (int block = 0; block < elliptic_count; ++block) {
    int start = elliptic_start[block], dim = elliptic_dim[block];
    if (start < 0 || dim < 2 || dim > 6 || start + dim > n
        || !enabled[start]) continue;
    thread float local_residual[6], local_residual_low[6];
    thread float local_force[6], local_force_low[6];
    for (int k = 0; k < dim; ++k) {
      local_residual[k] = row_residual[start + k];
      local_residual_low[k] = row_residual_low[start + k];
    }
    primal_elliptic_force_pair(local_residual, local_residual_low,
        R + start, elliptic_friction + block * 5, dim,
        local_force, local_force_low);
    for (int k = 0; k < dim; ++k) {
      force[start + k] = local_force[k];
      force_low[start + k] = local_force_low[k];
    }
  }

  for (int dof = 0; dof < nv; ++dof) {
    PrimalFloatPair g = {gradient[dof], gradient_low[dof]};
    for (int row = 0; row < n; ++row) if (enabled[row])
      g = primal_pair_fma(-ccj_get(J, row, dof, nv),
          {force[row], force_low[row]}, g);
    gradient[dof] = g.hi;
    gradient_low[dof] = g.lo;
  }
}

inline void primal_mass_difference_pair(device const float* mass,
    bool sparse_mass, device const float* debug, constant int* dims,
    device const float* q0, device const float* q0_low,
    device const float* acc,
    device const float* acc_low, device float* difference,
    device float* difference_low, int nv) {
  for (int dof = 0; dof < nv; ++dof) {
    float sum = 0.0f, correction = 0.0f;
    for (int other = 0; other < nv; ++other) {
      float entry = sparse_mass
          ? solver_component_mass_entry(debug, dims, nv, dof, other)
          : mass[dof * nv + other];
      primal_compensated_product_add(entry, acc[other], sum, correction);
      primal_compensated_product_add(entry, acc_low[other], sum, correction);
      primal_compensated_product_add(-entry, q0[other], sum, correction);
      primal_compensated_product_add(-entry, q0_low[other], sum, correction);
    }
    PrimalFloatPair value = primal_pair_renormalize(sum, correction);
    difference[dof] = value.hi;
    difference_low[dof] = value.lo;
  }
}


inline void primal_mass_product(device const float* mass, bool sparse_mass,
    device const float* debug, constant int* dims,
    device const float* vector, device float* product, int nv) {
  for (int i = 0; i < nv; ++i) {
    float value = 0.0f, correction = 0.0f;
    for (int j = 0; j < nv; ++j) {
      float entry = sparse_mass
          ? solver_component_mass_entry(debug, dims, nv, i, j)
          : mass[i * nv + j];
      primal_compensated_product_add(entry, vector[j], value, correction);
    }
    product[i] = value + correction;
  }
}

inline void primal_mass_product_pair(device const float* mass,
    bool sparse_mass, device const float* debug, constant int* dims,
    device const float* vector, device const float* vector_low,
    device float* product, device float* product_low, int nv) {
  for (int i = 0; i < nv; ++i) {
    float sum = 0.0f, correction = 0.0f;
    for (int j = 0; j < nv; ++j) {
      float entry = sparse_mass
          ? solver_component_mass_entry(debug, dims, nv, i, j)
          : mass[i * nv + j];
      primal_compensated_product_add(entry, vector[j], sum, correction);
      primal_compensated_product_add(entry, vector_low[j], sum, correction);
    }
    PrimalFloatPair result = primal_pair_renormalize(sum, correction);
    product[i] = result.hi;
    product_low[i] = result.lo;
  }
}

inline void primal_mass_difference(device const float* mass, bool sparse_mass,
    device const float* debug, constant int* dims,
    device const float* value, device const float* reference,
    device float* projected, device float* output, int nv) {
  if (sparse_mass) {
    solver_component_mass_apply_difference(debug, dims, nv, value,
        reference, projected, output);
  } else {
    for (int i = 0; i < nv; ++i) {
      float result = 0.0f;
      for (int j = 0; j < nv; ++j)
        result = fma(mass[i * nv + j], value[j] - reference[j], result);
      output[i] = result;
    }
  }
}

inline void primal_hessian_apply_elliptic(device const float* mass,
    bool sparse_mass, device const float* debug, constant int* dims,
    device const float* J, device const float* R, device const float* aref,
    device const float* row_hessian, device const int* enabled,
    device const int* elliptic_member, device const float* acc,
    device const float* row_residual,
    device const float* row_residual_low,
    device const int* elliptic_start, device const int* elliptic_dim,
    device const float* elliptic_friction, int elliptic_count,
    device const float* vector, device const float* vector_low,
    device float* product, device float* product_low,
    device const int* awake_tree,
    device const int* dof_island, int island, bool partitioned,
    int nv, int n, device float* product_scale) {
  // Accumulate the complete operator result per DOF.  Keeping M, scalar-row
  // and elliptic contributions in separate rounded writes loses the low bits
  // needed by the strict original-space PCG residual near convergence.
  for (int dof = 0; dof < nv; ++dof) {
    if (!solver_dof_awake(dims, awake_tree, dof)
        || (partitioned && dof_island[dof] != island)) {
      product[dof] = 0.0f;
      product_low[dof] = 0.0f;
      if (product_scale) product_scale[dof] = 0.0f;
      continue;
    }
    float value = 0.0f, correction = 0.0f;
    float absolute_scale = 0.0f;
    for (int other = 0; other < nv; ++other) {
      if (!solver_dof_awake(dims, awake_tree, other)
          || (partitioned && dof_island[other] != island)) continue;
      float mij = sparse_mass
          ? solver_component_mass_entry(debug, dims, nv, dof, other)
          : mass[dof * nv + other];
      primal_compensated_product_add(mij, vector[other], value, correction);
      primal_compensated_product_add(mij, vector_low[other], value, correction);
      if (product_scale)
        absolute_scale += abs(mij)
            * (abs(vector[other]) + abs(vector_low[other]));
    }
    for (int row = 0; row < n; ++row) {
      if (!enabled[row] || elliptic_member[row] || row_hessian[row] <= 0.0f)
        continue;
      PrimalFloatPair jd_hi = primal_ccj_compensated_dot_pair(
          J, row, vector, nv);
      PrimalFloatPair jd_low = primal_ccj_compensated_dot_pair(
          J, row, vector_low, nv);
      PrimalFloatPair jd = primal_pair_add(jd_hi, jd_low);
      PrimalFloatPair coefficient = primal_pair_product(
          {row_hessian[row], 0.0f}, {ccj_get(J, row, dof, nv), 0.0f});
      PrimalFloatPair contribution = primal_pair_product(coefficient, jd);
      primal_compensated_product_add(contribution.hi, 1.0f, value, correction);
      primal_compensated_product_add(contribution.lo, 1.0f, value, correction);
      if (product_scale) {
        float row_direction_scale = 0.0f;
        for (int other = 0; other < nv; ++other) {
          if (!solver_dof_awake(dims, awake_tree, other)
              || (partitioned && dof_island[other] != island)) continue;
          row_direction_scale += abs(ccj_get(J, row, other, nv))
              * (abs(vector[other]) + abs(vector_low[other]));
        }
        absolute_scale += abs(row_hessian[row])
            * abs(ccj_get(J, row, dof, nv)) * row_direction_scale;
      }
    }
    for (int block = 0; block < elliptic_count; ++block) {
      int start = elliptic_start[block], dim = elliptic_dim[block];
      if (start < 0 || dim < 2 || dim > 6 || start + dim > n
          || !enabled[start]) continue;
      thread float local_residual[6], local_residual_low[6];
      thread float local_hessian[36], local_hessian_low[36];
      thread float local_direction[6], local_direction_low[6];
      thread float local_hv[6], local_hv_low[6];
      thread float local_direction_scale[6];
      for (int k = 0; k < dim; ++k) {
        local_residual[k] = row_residual[start + k];
        local_residual_low[k] = row_residual_low[start + k];
        float dir_sum = 0.0f, dir_correction = 0.0f;
        for (int j = 0; j < nv; ++j) {
          primal_compensated_product_add(ccj_get(J, (start + k), j, nv),
              vector[j], dir_sum, dir_correction);
          primal_compensated_product_add(ccj_get(J, (start + k), j, nv),
              vector_low[j], dir_sum, dir_correction);
        }
        PrimalFloatPair dir = primal_pair_renormalize(dir_sum,
            dir_correction);
        local_direction[k] = dir.hi;
        local_direction_low[k] = dir.lo;
        if (product_scale) {
          float dir_scale = 0.0f;
          for (int j = 0; j < nv; ++j) {
            if (!solver_dof_awake(dims, awake_tree, j)
                || (partitioned && dof_island[j] != island)) continue;
            dir_scale += abs(ccj_get(J, start + k, j, nv))
                * (abs(vector[j]) + abs(vector_low[j]));
          }
          local_direction_scale[k] = dir_scale;
        }
      }
      primal_elliptic_hessian_pair(local_residual, local_residual_low,
          R + start, elliptic_friction + block * 5, dim,
          local_hessian, local_hessian_low);
      for (int a = 0; a < dim; ++a) {
        PrimalFloatPair hv_pair = {0.0f, 0.0f};
        for (int b = 0; b < dim; ++b) {
          PrimalFloatPair term = primal_pair_product(
              {local_hessian[a * 6 + b], local_hessian_low[a * 6 + b]},
              {local_direction[b], local_direction_low[b]});
          hv_pair = primal_pair_add(hv_pair, term);
        }
        local_hv[a] = hv_pair.hi;
        local_hv_low[a] = hv_pair.lo;
        primal_compensated_product_add(ccj_get(J, (start + a), dof, nv),
            local_hv[a], value, correction);
        primal_compensated_product_add(ccj_get(J, (start + a), dof, nv),
            local_hv_low[a], value, correction);
      }
      if (product_scale) {
        for (int a = 0; a < dim; ++a) {
          float hv_scale = 0.0f;
          for (int b = 0; b < dim; ++b)
            hv_scale += (abs(local_hessian[a * 6 + b])
                + abs(local_hessian_low[a * 6 + b]))
                * local_direction_scale[b];
          absolute_scale += abs(ccj_get(J, start + a, dof, nv)) * hv_scale;
        }
      }
    }
    PrimalFloatPair result = primal_pair_renormalize(value, correction);
    product[dof] = result.hi;
    product_low[dof] = result.lo;
    if (product_scale) product_scale[dof] = absolute_scale;
  }
}

// Keep the principal-mass preconditioner out of the caller's large PCG loop.
// Apple Metal's inlined nested solver path stopped updating its scratch-backed
// preconditioned vector on multi-contact elliptic systems; preserving this
// function boundary restores the ordinary per-iteration mass solve.
__attribute__((noinline)) bool primal_hessian_mass_precondition_elliptic(
    device const float* mass, device const float* L, bool sparse_mass,
    device const float* debug, constant int* dims,
    device const int* awake_tree, device const float* rhs,
    device float* solution, device float* inner_residual,
    device float* inner_preconditioned, device float* inner_direction,
    device float* inner_product, device float* temp,
    device const int* dof_island, int island, bool partitioned, int nv);

__attribute__((noinline)) bool primal_hessian_mass_precondition_pair_elliptic(
    device const float* mass, device const float* L, bool sparse_mass,
    device const float* debug, constant int* dims,
    device const int* awake_tree, device const float* rhs_hi,
    device const float* rhs_low, device float* solution_hi,
    device float* solution_low, device float* residual_work,
    device float* residual_hi, device float* residual_low,
    device float* correction_hi, device float* correction_work,
    device const int* dof_island, int island, bool partitioned, int nv);

// CG-specific paired preconditioning. Complete mass blocks use the ordinary
// residual-corrected pair solve above. When an awake/island mask cuts a mass
// block, solve the high word with the bounded restricted PCG, then form the
// represented residual against M*(x_hi+x_lo) and solve its high and low words
// separately with the same restricted operator. Keeping this outside the
// generic Newton helper preserves its established scratch contract.
__attribute__((noinline)) bool primal_cg_mass_precondition_pair_elliptic(
    device const float* mass, device const float* L, bool sparse_mass,
    device const float* debug, constant int* dims,
    device const int* awake_tree, device const float* rhs_hi,
    device const float* rhs_low, device float* solution_hi,
    device float* solution_low, device float* residual_work,
    device float* residual_hi, device float* residual_low,
    device float* correction_hi, device float* correction_work,
    device float* low_stash, device const int* dof_island, int island,
    bool partitioned, int nv);

// Use the already factored Gauss mass exactly when the selected awake island
// is a union of complete mass components.  Dense factors are checked for zero
// cross-cut entries before restricted triangular substitution; packed sparse
// factors use their compiled component map.  A cut component falls back to
// the source-preserving restricted principal-system iteration below.
__attribute__((noinline)) bool primal_mass_solve_complete_island(device const float* mass,
    device const float* L, bool sparse_mass, device const float* debug,
    constant int* dims, device const int* awake_tree,
    device const int* dof_island, int island, device const float* rhs,
    device float* solution, device float* forward, int nv) {
  if (sparse_mass) {
    int header = dims[24] + dims[9] * 10;
    int ncomponent = dims[header];
    int layout_offset = dims[header + 3];
    device const int* layout = reinterpret_cast<device const int*>(debug)
        + layout_offset;
    for (int component = 0; component < ncomponent; ++component) {
      int begin = layout[component], end = layout[component + 1];
      bool selected = false, unselected = false;
      for (int k = begin; k < end; ++k) {
        int dof = layout[ncomponent + 1 + k];
        bool in_island = solver_dof_awake(dims, awake_tree, dof)
            && dof_island[dof] == island;
        selected = selected || in_island;
        unselected = unselected || !in_island;
      }
      if (selected && unselected) return false;
    }
    for (int dof = 0; dof < nv; ++dof) solution[dof] = forward[dof] = 0.0f;
    for (int component = 0; component < ncomponent; ++component) {
      int begin = layout[component], width = layout[component + 1] - begin;
      if (width <= 0) continue;
      int dof_base = ncomponent + 1 + begin;
      int first_dof = layout[dof_base];
      if (!solver_dof_awake(dims, awake_tree, first_dof)
          || dof_island[first_dof] != island) continue;
      for (int i = 0; i < width; ++i) {
        int dof_i = layout[dof_base + i];
        float value = rhs[dof_i];
        for (int k = 0; k < i; ++k) {
          int dof_k = layout[dof_base + k];
          value = fma(-solver_component_factor_entry(debug, dims, nv,
              dof_i, dof_k), forward[dof_k], value);
        }
        float diagonal = solver_component_factor_entry(debug, dims, nv,
            dof_i, dof_i);
        if (!(diagonal > 0.0f) || !isfinite(diagonal)) return false;
        forward[dof_i] = value / diagonal;
      }
      for (int i = width - 1; i >= 0; --i) {
        int dof_i = layout[dof_base + i];
        float value = forward[dof_i];
        for (int k = i + 1; k < width; ++k) {
          int dof_k = layout[dof_base + k];
          value = fma(-solver_component_factor_entry(debug, dims, nv,
              dof_k, dof_i), solution[dof_k], value);
        }
        float diagonal = solver_component_factor_entry(debug, dims, nv,
            dof_i, dof_i);
        solution[dof_i] = value / diagonal;
      }
    }
    return true;
  }

  // A dense Cholesky factor represents the selected principal operator only
  // when no factor entry crosses the awake island boundary. Reject mixed
  // components rather than solving the full system and masking afterward.
  for (int i = 0; i < nv; ++i) for (int j = 0; j < i; ++j) {
    bool selected_i = solver_dof_awake(dims, awake_tree, i)
        && dof_island[i] == island;
    bool selected_j = solver_dof_awake(dims, awake_tree, j)
        && dof_island[j] == island;
    if (selected_i != selected_j && L[i * nv + j] != 0.0f) return false;
  }
  for (int dof = 0; dof < nv; ++dof) solution[dof] = forward[dof] = 0.0f;
  for (int i = 0; i < nv; ++i) {
    if (!solver_dof_awake(dims, awake_tree, i) || dof_island[i] != island)
      continue;
    float value = rhs[i];
    for (int k = 0; k < i; ++k)
      if (solver_dof_awake(dims, awake_tree, k) && dof_island[k] == island)
        value = fma(-L[i * nv + k], forward[k], value);
    float diagonal = L[i * nv + i];
    if (!(diagonal > 0.0f) || !isfinite(diagonal)) return false;
    forward[i] = value / diagonal;
  }
  for (int i = nv - 1; i >= 0; --i) {
    if (!solver_dof_awake(dims, awake_tree, i) || dof_island[i] != island)
      continue;
    float value = forward[i];
    for (int k = i + 1; k < nv; ++k)
      if (solver_dof_awake(dims, awake_tree, k) && dof_island[k] == island)
        value = fma(-L[k * nv + i], solution[k], value);
    solution[i] = value / L[i * nv + i];
  }
  return true;
}

// Optional three-word failure witness for the initial primal H solve. The
// caller passes the existing diagnostics row only while detail tracing is
// enabled, so the ordinary path has neither writes nor extra storage.
inline void primal_hessian_trace_failure(device float* trace_out, int code,
    float value, float limit) {
  if (trace_out == nullptr) return;
  // Keep the first failure within this initial solve. Test-owned observers
  // capture it before a recovery dispatch can reuse the diagnostics row.
  // Encode non-finite operands through
  // their distinct code, leaving all three witness words finite.
  if (trace_out[9] != 0.0f) return;
  trace_out[7] = isfinite(value) ? value : 0.0f;
  trace_out[8] = isfinite(limit) ? limit : 0.0f;
  trace_out[9] = float(code);
}

inline bool primal_hessian_solve_elliptic_legacy(device const float* mass,
    device const float* L, bool sparse_mass,
    device const float* debug, device float* pair_tail,
    device const float* J, device const float* R, device const float* aref,
    device const float* row_hessian, constant int* dims,
    device const int* awake_tree, device const int* enabled,
    device const int* elliptic_member, device const float* acc,
    device const float* row_residual,
    device const float* row_residual_low,
    device const int* elliptic_start, device const int* elliptic_dim,
    device const float* elliptic_friction, int elliptic_count,
    device const float* rhs, device const float* rhs_low,
    device float* solution, device float* factor,
    device float* temp, device float* inner_residual,
    device float* inner_preconditioned, device float* inner_direction,
    device const int* dof_island, int island, bool partitioned, int nv, int n,
    float tolerance, float scale) {
  // Matrix-free PCG avoids materializing the nv-by-nv primal Hessian. The
  // four vectors are slices of the preallocated solver scratch interval:
  // residual, preconditioned residual, search direction, and H*direction.
  device float* residual = factor;
  device float* preconditioned = factor + nv;
  device float* direction = factor + 2 * nv;
  device float* product = factor + 3 * nv;
  device float* solution_low = pair_tail + 4 * nv;
  device float* residual_low = pair_tail + 3 * nv;
  device float* direction_low = pair_tail + 5 * nv;
  device float* product_low = pair_tail + 6 * nv;

  int active_dof_count = 0;
  float rhs_component_scale = 0.0f;
  for (int dof = 0; dof < nv; ++dof) {
    solution[dof] = solution_low[dof] = 0.0f;
    bool active = solver_dof_awake(dims, awake_tree, dof)
        && (!partitioned || dof_island[dof] == island);
    residual[dof] = active ? rhs[dof] : 0.0f;
    residual_low[dof] = active ? rhs_low[dof] : 0.0f;
    active_dof_count += active ? 1 : 0;
    if (active) rhs_component_scale = max(rhs_component_scale,
        abs(rhs[dof]) + abs(rhs_low[dof]));
  }
  float rhs_norm_sq = primal_pair_dot(residual, residual_low,
      residual, residual_low, nv);
  if (!isfinite(rhs_norm_sq)) return false;
  if (rhs_norm_sq <= 1e-30f) return true;

  // The PCG search direction has an arbitrary common scale. Normalize its
  // preconditioned seed so the first rho and d^T H d stay representable when
  // a small regularizer makes H very stiff (for example J^T R^-1 J with
  // R near 1e-15 and a large finite aref). Apply this same scale to every
  // subsequent preconditioned residual: beta is unchanged and alpha*d gives
  // the same Newton update, while neither dot product has the spurious
  // O(rhs^4) magnitude of the unscaled direction.
  float direction_scale = 1.0f / max(1.0f, rhs_component_scale);
  if (!(direction_scale > 0.0f) || !isfinite(direction_scale)) return false;

  if (!primal_hessian_mass_precondition_pair_elliptic(mass, L, sparse_mass,
      debug, dims, awake_tree, residual, residual_low, preconditioned,
      product_low, inner_residual, inner_preconditioned, inner_direction,
      product, temp, dof_island, island, partitioned, nv)) return false;
  for (int dof = 0; dof < nv; ++dof) {
    PrimalFloatPair z = primal_pair_multiply(
        {preconditioned[dof], product_low[dof]}, direction_scale);
    preconditioned[dof] = direction[dof] = z.hi;
    product_low[dof] = direction_low[dof] = z.lo;
  }
  PrimalFloatPair rho = primal_pair_dot_expansion(residual, residual_low,
      preconditioned, direction_low, nv);
  if (!primal_pair_positive(rho) || !isfinite(primal_pair_value(rho)))
    return false;

  // Recompute the original-space residual when the recursive residual
  // predicts convergence, and at the fixed work limit. If the reliable check
  // disproves convergence, restart from that true residual while retaining
  // the remaining bounded work. Do not restart merely because n DOFs have
  // elapsed: that discards PCG conjugacy on ill-conditioned contact systems.
  // Acceptance uses the unchanged requested outer threshold, capped by the
  // original relative inner criterion. The direction normalization above
  // changes only the common scale of PCG's preconditioned vectors.
  float outer_absolute_residual = tolerance / max(scale, 1e-30f);
  float outer_residual_sq = outer_absolute_residual * outer_absolute_residual;
  float residual_tolerance_sq = min(1e-12f * rhs_norm_sq,
      outer_residual_sq);
  // `primal_compensated_product_add` retains each FMA product error and the
  // two-sum addition error. The remaining correction-stream addition loses
  // at most O(u^2) times the absolute term sum: the product/addition error is
  // O(u), then its un-compensated accumulation contributes another u. The
  // conservative factor 32 covers the two pair words, renormalization, and
  // final product accumulation per term (u=2^-23). For each output DOF, a
  // scalar row includes its two nv-term J*v projections; an elliptic block
  // includes at most six such projections and a 6x6 block product. Bound the
  // error separately from the actual absolute mass/row term magnitudes used
  // by H*x. Non-finite magnitude disables this fallback; a large row cannot
  // mask an inaccurate small active row.
  constexpr float float_epsilon_squared = 1.4210854715202004e-14f;
  int operator_term_count = 2 * nv + n * (2 * nv + 16)
      + elliptic_count * (12 * nv + 80);
  float pair_roundoff = 32.0f * float(max(operator_term_count, 1))
      * float_epsilon_squared;
  // A flushed subnormal operation has absolute error below FLT_MIN. The
  // term count bounds vector products/projections; charge up to 32 primitive
  // operations per counted term for product residuals, two-sum, correction
  // accumulation, and renormalization. If this absolute floor exceeds the
  // requested component allowance, the roundoff fallback is disabled below.
  float pair_underflow_floor = 32.0f * float(max(operator_term_count, 1))
      * 1.17549435e-38f;
  float component_residual_limit = sqrt(max(residual_tolerance_sq, 0.0f))
      / sqrt(float(max(active_dof_count, 1)));
  if (!isfinite(pair_roundoff) || !isfinite(pair_underflow_floor)
      || !isfinite(component_residual_limit)) return false;
  const int max_iterations = max(1, 4 * active_dof_count);
  for (int iteration = 0; iteration < max_iterations; ++iteration) {
    primal_hessian_apply_elliptic(mass, sparse_mass, debug, dims, J, R, aref,
        row_hessian, enabled, elliptic_member, acc, row_residual,
        row_residual_low, elliptic_start, elliptic_dim, elliptic_friction,
        elliptic_count, direction, direction_low, product, product_low,
        awake_tree, dof_island, island, partitioned, nv, n, nullptr);
    PrimalFloatPair denominator = primal_pair_dot_expansion(direction,
        direction_low, product, product_low, nv);
    if (!primal_pair_positive(denominator)
        || !isfinite(primal_pair_value(denominator))) return false;
    PrimalFloatPair alpha = primal_pair_divide(rho, denominator);
    if (!isfinite(primal_pair_value(alpha))) return false;
    for (int dof = 0; dof < nv; ++dof) {
      PrimalFloatPair x = primal_pair_add(
          primal_pair_product(alpha, {direction[dof], direction_low[dof]}),
          {solution[dof], solution_low[dof]});
      solution[dof] = x.hi;
      solution_low[dof] = x.lo;
      PrimalFloatPair r = primal_pair_add(
          primal_pair_product(primal_pair_multiply(alpha, -1.0f),
              {product[dof], product_low[dof]}),
          {residual[dof], residual_low[dof]});
      residual[dof] = r.hi;
      residual_low[dof] = r.lo;
    }
    float residual_norm_sq = primal_pair_dot(residual, residual_low,
        residual, residual_low, nv);
    if (!isfinite(residual_norm_sq)) return false;
    // Keep the Krylov recurrence across the active dimension.  Restarting
    // unconditionally at n DOFs discards conjugacy before a reliable
    // residual indicates convergence, and can take substantially more than
    // n iterations on ill-conditioned contact Hessians.  Check the original
    // operator residual when the recursive residual predicts convergence and
    // once at the fixed work limit; only restart early if that true check
    // disproves convergence.
    if (residual_norm_sq <= residual_tolerance_sq
        || iteration + 1 == max_iterations) {
      primal_hessian_apply_elliptic(mass, sparse_mass, debug, dims, J, R,
          aref, row_hessian, enabled, elliptic_member, acc, row_residual,
          row_residual_low, elliptic_start, elliptic_dim, elliptic_friction,
          elliptic_count, solution, solution_low, product, product_low,
          awake_tree, dof_island, island, partitioned, nv, n, inner_residual);
      for (int dof = 0; dof < nv; ++dof) {
        if (!solver_dof_awake(dims, awake_tree, dof)
            || (partitioned && dof_island[dof] != island)) {
          residual[dof] = 0.0f;
          residual_low[dof] = 0.0f;
          continue;
        }
        PrimalFloatPair r = primal_pair_add({rhs[dof], rhs_low[dof]},
            {-product[dof], -product_low[dof]});
        residual[dof] = r.hi;
        residual_low[dof] = r.lo;
      }
      float true_residual_sq = primal_pair_dot(residual, residual_low,
          residual, residual_low, nv);
      if (!isfinite(true_residual_sq)) return false;
      // A float32 sum of squares can underflow to zero even when the retained
      // two-word residual is nonzero. Check each active component against the
      // strict norm-derived allowance before accepting the squared norm.
      bool strict_component_residual = true;
      for (int dof = 0; dof < nv; ++dof) {
        if (!solver_dof_awake(dims, awake_tree, dof)
            || (partitioned && dof_island[dof] != island)) continue;
        float residual_magnitude = abs(residual[dof]) + abs(residual_low[dof]);
        if (!isfinite(residual_magnitude)
            || residual_magnitude > component_residual_limit) {
          strict_component_residual = false;
          break;
        }
      }
      if (strict_component_residual) return true;
      bool componentwise_pair_bound = pair_underflow_floor
          <= component_residual_limit;
      for (int dof = 0; dof < nv; ++dof) {
        if (!solver_dof_awake(dims, awake_tree, dof)
            || (partitioned && dof_island[dof] != island)) continue;
        float residual_magnitude = abs(residual[dof]) + abs(residual_low[dof]);
        float operator_magnitude = abs(rhs[dof]) + abs(rhs_low[dof])
            + inner_residual[dof];
        float component_bound = component_residual_limit
            + pair_roundoff * operator_magnitude + pair_underflow_floor;
        if (!isfinite(operator_magnitude) || !isfinite(component_bound)
            || !(residual_magnitude <= component_bound)) {
          componentwise_pair_bound = false;
          break;
        }
      }
      if (componentwise_pair_bound) return true;
      if (iteration + 1 >= max_iterations) return false;
      if (!primal_hessian_mass_precondition_pair_elliptic(mass, L, sparse_mass,
          debug, dims, awake_tree, residual, residual_low, preconditioned,
          product_low, inner_residual, inner_preconditioned, inner_direction,
          product, temp, dof_island, island, partitioned, nv)) return false;
      for (int dof = 0; dof < nv; ++dof) {
        PrimalFloatPair z = primal_pair_multiply(
            {preconditioned[dof], product_low[dof]}, direction_scale);
        preconditioned[dof] = direction[dof] = z.hi;
        product_low[dof] = direction_low[dof] = z.lo;
      }
      rho = primal_pair_dot_expansion(residual, residual_low,
          preconditioned, direction_low, nv);
      if (!primal_pair_positive(rho) || !isfinite(primal_pair_value(rho)))
        return false;
      continue;
    }

    if (!primal_hessian_mass_precondition_pair_elliptic(mass, L, sparse_mass,
        debug, dims, awake_tree, residual, residual_low, preconditioned,
        product_low, inner_residual, inner_preconditioned, inner_direction,
        product, temp, dof_island, island, partitioned, nv)) return false;
    for (int dof = 0; dof < nv; ++dof) {
      PrimalFloatPair z = primal_pair_multiply(
          {preconditioned[dof], product_low[dof]}, direction_scale);
      preconditioned[dof] = z.hi;
      product_low[dof] = z.lo;
    }
    PrimalFloatPair next_rho = primal_pair_dot_expansion(residual,
        residual_low, preconditioned, product_low, nv);
    if (!primal_pair_positive(next_rho)
        || !isfinite(primal_pair_value(next_rho))) {
      return false;
    }
    PrimalFloatPair beta = primal_pair_divide(next_rho, rho);
    for (int dof = 0; dof < nv; ++dof) {
      PrimalFloatPair p = primal_pair_add(
          primal_pair_product(beta,
              {direction[dof], direction_low[dof]}),
          {preconditioned[dof], product_low[dof]});
      direction[dof] = p.hi;
      direction_low[dof] = p.lo;
    }
    rho = next_rho;
  }
  return false;
}

inline bool primal_hessian_solve_elliptic_scaled(device const float* mass,
    device const float* L, bool sparse_mass,
    device const float* debug, device float* pair_tail,
    device const float* J, device const float* R, device const float* aref,
    device const float* row_hessian, constant int* dims,
    device const int* awake_tree, device const int* enabled,
    device const int* elliptic_member, device const float* acc,
    device const float* row_residual,
    device const float* row_residual_low,
    device const int* elliptic_start, device const int* elliptic_dim,
    device const float* elliptic_friction, int elliptic_count,
    device const float* rhs, device const float* rhs_low,
    device float* solution, device float* factor,
    device float* temp, device float* inner_residual,
    device float* inner_preconditioned, device float* inner_direction,
    device const int* dof_island, int island, bool partitioned, int nv, int n,
    float tolerance, float scale, device float* trace_out) {
  // Matrix-free PCG avoids materializing the nv-by-nv primal Hessian. The
  // four vectors are slices of the preallocated solver scratch interval:
  // residual, preconditioned residual, search direction, and H*direction.
  device float* residual = factor;
  device float* preconditioned = factor + nv;
  device float* direction = factor + 2 * nv;
  device float* product = factor + 3 * nv;
  device float* solution_low = pair_tail + 4 * nv;
  device float* residual_low = pair_tail + 3 * nv;
  device float* direction_low = pair_tail + 5 * nv;
  device float* product_low = pair_tail + 6 * nv;

  int active_dof_count = 0;
  float rhs_scale = 0.0f;
  for (int dof = 0; dof < nv; ++dof) {
    solution[dof] = solution_low[dof] = 0.0f;
    bool active = solver_dof_awake(dims, awake_tree, dof)
        && (!partitioned || dof_island[dof] == island);
    if (active) {
      if (!isfinite(rhs[dof]) || !isfinite(rhs_low[dof])) {
        primal_hessian_trace_failure(trace_out, 111, rhs[dof], rhs_low[dof]);
        return false;
      }
      if (primal_float_is_subnormal(rhs[dof])
          || primal_float_is_subnormal(rhs_low[dof])) {
        primal_hessian_trace_failure(trace_out, 112, rhs[dof], rhs_low[dof]);
        return false;
      }
      rhs_scale = max(rhs_scale, max(abs(rhs[dof]), abs(rhs_low[dof])));
    }
    active_dof_count += active ? 1 : 0;
  }
  if (active_dof_count == 0 || rhs_scale == 0.0f) return true;
  // Do not attempt to normalize subnormal values: the MSL execution mode may
  // flush them to zero. Failing this nonrepresentable system is safer than
  // certifying a zero RHS and silently skipping its Newton direction.
  if (!isfinite(rhs_scale) || rhs_scale < 1.17549435e-38f) {
    primal_hessian_trace_failure(trace_out, 113, rhs_scale,
        1.17549435e-38f);
    return false;
  }
  for (int dof = 0; dof < nv; ++dof) {
    bool active = solver_dof_awake(dims, awake_tree, dof)
        && (!partitioned || dof_island[dof] == island);
    float normalized_hi = active ? rhs[dof] / rhs_scale : 0.0f;
    float normalized_low = active ? rhs_low[dof] / rhs_scale : 0.0f;
    if (active && ((rhs[dof] != 0.0f && normalized_hi == 0.0f)
        || (rhs_low[dof] != 0.0f && normalized_low == 0.0f)
        || primal_float_is_subnormal(normalized_hi)
        || primal_float_is_subnormal(normalized_low))) {
      primal_hessian_trace_failure(trace_out, 114, normalized_hi,
          normalized_low);
      return false;
    }
    // The RHS ABI normally stores a normalized expansion, but a caller can
    // supply two normal words whose high+low nearly cancel (for example when
    // an RHS was externally rescaled). Canonicalize before the mass solve,
    // which solves the high word first and cannot recover a large low word
    // that cancels that high word.
    PrimalFloatPair represented_rhs = primal_pair_add(
        {normalized_hi, 0.0f}, {normalized_low, 0.0f});
    if (!isfinite(represented_rhs.hi) || !isfinite(represented_rhs.lo)
        || primal_float_is_subnormal(represented_rhs.hi)
        || primal_float_is_subnormal(represented_rhs.lo)) {
      primal_hessian_trace_failure(trace_out, 115, represented_rhs.hi,
          represented_rhs.lo);
      return false;
    }
    residual[dof] = represented_rhs.hi;
    residual_low[dof] = represented_rhs.lo;
  }
  float rhs_norm = primal_pair_norm(residual, residual_low, nv);
  // The represented RHS pair may cancel exactly even when its two words are
  // individually nonzero. That is the exact zero equation, so retain the
  // zero iterate rather than reporting a numerical failure.
  if (rhs_norm == 0.0f) return true;
  if (!(rhs_norm > 0.0f) || !isfinite(rhs_norm)) {
    primal_hessian_trace_failure(trace_out, 116, rhs_norm, rhs_scale);
    return false;
  }

  if (!primal_hessian_mass_precondition_pair_elliptic(mass, L, sparse_mass,
      debug, dims, awake_tree, residual, residual_low, preconditioned,
      product_low, inner_residual, inner_preconditioned, inner_direction,
      product, temp, dof_island, island, partitioned, nv)) {
    primal_hessian_trace_failure(trace_out, 201, rhs_norm, rhs_scale);
    return false;
  }
  float precondition_scale = 0.0f;
  for (int dof = 0; dof < nv; ++dof) {
    bool active = solver_dof_awake(dims, awake_tree, dof)
        && (!partitioned || dof_island[dof] == island);
    if (!active) {
      preconditioned[dof] = product_low[dof] = 0.0f;
      continue;
    }
    if (!isfinite(preconditioned[dof]) || !isfinite(product_low[dof])
        || primal_float_is_subnormal(preconditioned[dof])
        || primal_float_is_subnormal(product_low[dof])) {
      primal_hessian_trace_failure(trace_out, 202,
          preconditioned[dof], product_low[dof]);
      return false;
    }
    precondition_scale = max(precondition_scale,
        max(abs(preconditioned[dof]), abs(product_low[dof])));
  }
  if (!(precondition_scale >= 1.17549435e-38f)
      || !isfinite(precondition_scale)) {
    primal_hessian_trace_failure(trace_out, 203, precondition_scale,
        1.17549435e-38f);
    return false;
  }
  // PCG's search direction has an arbitrary common scale. Normalize the
  // first mass-preconditioned residual by its own finite component scale,
  // then use that same scale for every later preconditioned residual. This
  // keeps rho and d^T H d representable without changing beta or alpha*d.
  for (int dof = 0; dof < nv; ++dof) {
    PrimalFloatPair z = primal_pair_divide(
        {preconditioned[dof], product_low[dof]}, {precondition_scale, 0.0f});
    if (!isfinite(z.hi) || !isfinite(z.lo)
        || primal_float_is_subnormal(z.hi)
        || primal_float_is_subnormal(z.lo)) {
      primal_hessian_trace_failure(trace_out, 204, z.hi, z.lo);
      return false;
    }
    preconditioned[dof] = direction[dof] = z.hi;
    product_low[dof] = direction_low[dof] = z.lo;
  }
  PrimalFloatPair rho = primal_pair_dot_expansion(residual, residual_low,
      preconditioned, direction_low, nv);
  if (!primal_pair_positive(rho) || !isfinite(primal_pair_value(rho))) {
    primal_hessian_trace_failure(trace_out, 301, rho.hi, rho.lo);
    return false;
  }

  // Recompute the true operator residual in normalized RHS coordinates when
  // the recursive residual predicts convergence and at the fixed work
  // limit. Scaling the full equation by rhs_scale leaves the solution and
  // requested acceptance unchanged while making this check representable.
  // If the reliable check disproves convergence, restart from that true
  // residual within bounded work; do not discard conjugacy at n DOFs.
  // Preserve the historical relative H-solve accuracy while tightening it
  // enough that the resulting Newton gradient can satisfy the requested
  // source outer criterion in this bounded solve.
  float outer_absolute_residual = tolerance / max(scale, 1e-30f);
  float relative_residual_limit = 1e-6f * rhs_norm;
  if (!isfinite(outer_absolute_residual)
      || !isfinite(relative_residual_limit)) {
    primal_hessian_trace_failure(trace_out, 351, outer_absolute_residual,
        relative_residual_limit);
    return false;
  }
  // Express both bounds in the normalized RHS coordinates. Comparing norms
  // (rather than squared norms) avoids flushing a valid ~1e-20 allowance to
  // zero. The component allowance is no looser than the requested outer and
  // original relative-inner criteria.
  float outer_scaled_limit = outer_absolute_residual / rhs_scale;
  if (!isfinite(outer_scaled_limit)
      || outer_scaled_limit > relative_residual_limit)
    outer_scaled_limit = relative_residual_limit;
  float component_residual_limit = outer_scaled_limit
      / sqrt(float(max(active_dof_count, 1)));
  if (!(component_residual_limit >= 0.0f)
      || !isfinite(component_residual_limit)) {
    primal_hessian_trace_failure(trace_out, 352, outer_scaled_limit,
        component_residual_limit);
    return false;
  }
  // Retain the base's per-component arithmetic certificate in the same
  // normalized equation. The operator term magnitudes are computed from the
  // normalized solution by the final H*x call, so both allowances have units
  // of normalized RHS. Nonfinite bounds disable acceptance.
  constexpr float float_epsilon_squared = 1.4210854715202004e-14f;
  int operator_term_count = 2 * nv + n * (2 * nv + 16)
      + elliptic_count * (12 * nv + 80);
  float pair_roundoff = 32.0f * float(max(operator_term_count, 1))
      * float_epsilon_squared;
  float pair_underflow_floor = 32.0f * float(max(operator_term_count, 1))
      * 1.17549435e-38f;
  if (!isfinite(pair_roundoff) || !isfinite(pair_underflow_floor)) {
    primal_hessian_trace_failure(trace_out, 353, pair_roundoff,
        pair_underflow_floor);
    return false;
  }
  // A positive mathematical bound can underflow to zero after normalization.
  // This remains safe: the componentwise and global checks then accept only
  // an exactly zero represented residual (subnormal residual words are
  // rejected by primal_pair_norm). Never replace this with a positive floor.
  bool residual_limit_representable = outer_scaled_limit >= 0.0f
      && component_residual_limit >= 0.0f
      && isfinite(outer_scaled_limit) && isfinite(component_residual_limit);
  const int max_iterations = max(1, 4 * active_dof_count);
  for (int iteration = 0; iteration < max_iterations; ++iteration) {
    primal_hessian_apply_elliptic(mass, sparse_mass, debug, dims, J, R, aref,
        row_hessian, enabled, elliptic_member, acc, row_residual,
        row_residual_low, elliptic_start, elliptic_dim, elliptic_friction,
        elliptic_count, direction, direction_low, product, product_low,
        awake_tree, dof_island, island, partitioned, nv, n, nullptr);
    PrimalFloatPair denominator = primal_pair_dot_expansion(direction,
        direction_low, product, product_low, nv);
    if (!primal_pair_positive(denominator)
        || !isfinite(primal_pair_value(denominator))) {
      primal_hessian_trace_failure(trace_out, 401, denominator.hi,
          denominator.lo);
      return false;
    }
    PrimalFloatPair alpha = primal_pair_divide(rho, denominator);
    if (!isfinite(primal_pair_value(alpha))) {
      primal_hessian_trace_failure(trace_out, 402, alpha.hi, alpha.lo);
      return false;
    }
    for (int dof = 0; dof < nv; ++dof) {
      PrimalFloatPair x = primal_pair_add(
          primal_pair_product(alpha, {direction[dof], direction_low[dof]}),
          {solution[dof], solution_low[dof]});
      solution[dof] = x.hi;
      solution_low[dof] = x.lo;
      PrimalFloatPair r = primal_pair_add(
          primal_pair_product(primal_pair_multiply(alpha, -1.0f),
              {product[dof], product_low[dof]}),
          {residual[dof], residual_low[dof]});
      residual[dof] = r.hi;
      residual_low[dof] = r.lo;
    }
    float residual_norm = primal_pair_norm(residual, residual_low, nv);
    if (!isfinite(residual_norm)) {
      primal_hessian_trace_failure(trace_out, 501, residual_norm,
          outer_scaled_limit);
      return false;
    }
    // Keep the Krylov recurrence across the active dimension.  Restarting
    // unconditionally at n DOFs discards conjugacy before a reliable
    // residual indicates convergence, and can take substantially more than
    // n iterations on ill-conditioned contact Hessians.  Check the original
    // operator residual when the recursive residual predicts convergence and
    // once at the fixed work limit; only restart early if that true check
    // disproves convergence.
    if (residual_norm <= outer_scaled_limit
        || iteration + 1 == max_iterations) {
      primal_hessian_apply_elliptic(mass, sparse_mass, debug, dims, J, R,
          aref, row_hessian, enabled, elliptic_member, acc, row_residual,
          row_residual_low, elliptic_start, elliptic_dim, elliptic_friction,
          elliptic_count, solution, solution_low, product, product_low,
          awake_tree, dof_island, island, partitioned, nv, n, inner_residual);
      for (int dof = 0; dof < nv; ++dof) {
        if (!solver_dof_awake(dims, awake_tree, dof)
            || (partitioned && dof_island[dof] != island)) {
          residual[dof] = 0.0f;
          residual_low[dof] = 0.0f;
          continue;
        }
        PrimalFloatPair r = primal_pair_add(
            {rhs[dof] / rhs_scale, rhs_low[dof] / rhs_scale},
            {-product[dof], -product_low[dof]});
        residual[dof] = r.hi;
        residual_low[dof] = r.lo;
      }
      float true_residual_norm = primal_pair_norm(residual, residual_low, nv);
      if (!isfinite(true_residual_norm)) {
        primal_hessian_trace_failure(trace_out, 502, true_residual_norm,
            outer_scaled_limit);
        return false;
      }
      bool strict_component_residual = true;
      for (int dof = 0; dof < nv; ++dof) {
        if (!solver_dof_awake(dims, awake_tree, dof)
            || (partitioned && dof_island[dof] != island)) continue;
        float magnitude = abs(residual[dof]) + abs(residual_low[dof]);
        if (!isfinite(magnitude) || magnitude > component_residual_limit) {
          strict_component_residual = false;
          break;
        }
      }
      if (residual_limit_representable && strict_component_residual
          && true_residual_norm <= outer_scaled_limit) {
        bool rescaled = primal_rescale_pair_vector_traced(solution, solution_low,
            awake_tree, dims, dof_island, island, partitioned, rhs_scale, nv,
            trace_out, 60000);
        if (!rescaled) primal_hessian_trace_failure(trace_out, 601,
            rhs_scale, true_residual_norm);
        return rescaled;
      }
      bool componentwise_pair_bound =
          pair_underflow_floor <= component_residual_limit;
      for (int dof = 0; dof < nv; ++dof) {
        if (!solver_dof_awake(dims, awake_tree, dof)
            || (partitioned && dof_island[dof] != island)) continue;
        float residual_magnitude = abs(residual[dof]) + abs(residual_low[dof]);
        float operator_magnitude = abs(rhs[dof] / rhs_scale)
            + abs(rhs_low[dof] / rhs_scale) + inner_residual[dof];
        float component_bound = component_residual_limit
            + pair_roundoff * operator_magnitude + pair_underflow_floor;
        if (!isfinite(operator_magnitude) || !isfinite(component_bound)
            || !(residual_magnitude <= component_bound)) {
          componentwise_pair_bound = false;
          break;
        }
      }
      if (componentwise_pair_bound) {
        bool rescaled = primal_rescale_pair_vector_traced(solution, solution_low,
            awake_tree, dims, dof_island, island, partitioned, rhs_scale, nv,
            trace_out, 62000);
        if (!rescaled) primal_hessian_trace_failure(trace_out, 602,
            rhs_scale, true_residual_norm);
        return rescaled;
      }
      if (iteration + 1 >= max_iterations) {
        primal_hessian_trace_failure(trace_out, 503, true_residual_norm,
            outer_scaled_limit);
        return false;
      }
      if (!primal_hessian_mass_precondition_pair_elliptic(mass, L, sparse_mass,
          debug, dims, awake_tree, residual, residual_low, preconditioned,
          product_low, inner_residual, inner_preconditioned, inner_direction,
          product, temp, dof_island, island, partitioned, nv)) {
        primal_hessian_trace_failure(trace_out, 205, true_residual_norm,
            component_residual_limit);
        return false;
      }
      for (int dof = 0; dof < nv; ++dof)
        if (solver_dof_awake(dims, awake_tree, dof)
            && (!partitioned || dof_island[dof] == island)) {
          PrimalFloatPair z = primal_pair_divide(
              {preconditioned[dof], product_low[dof]},
              {precondition_scale, 0.0f});
          if (!isfinite(z.hi) || !isfinite(z.lo)
              || primal_float_is_subnormal(z.hi)
              || primal_float_is_subnormal(z.lo)) {
            primal_hessian_trace_failure(trace_out, 206, z.hi, z.lo);
            return false;
          }
          preconditioned[dof] = direction[dof] = z.hi;
          product_low[dof] = direction_low[dof] = z.lo;
        } else {
          preconditioned[dof] = direction[dof] = 0.0f;
          product_low[dof] = direction_low[dof] = 0.0f;
        }
      rho = primal_pair_dot_expansion(residual, residual_low,
          preconditioned, direction_low, nv);
      if (!primal_pair_positive(rho) || !isfinite(primal_pair_value(rho))) {
        primal_hessian_trace_failure(trace_out, 302, rho.hi, rho.lo);
        return false;
      }
      continue;
    }

    if (!primal_hessian_mass_precondition_pair_elliptic(mass, L, sparse_mass,
        debug, dims, awake_tree, residual, residual_low, preconditioned,
        product_low, inner_residual, inner_preconditioned, inner_direction,
        product, temp, dof_island, island, partitioned, nv)) {
      primal_hessian_trace_failure(trace_out, 207, residual_norm,
          outer_scaled_limit);
      return false;
    }
    for (int dof = 0; dof < nv; ++dof) {
      if (!solver_dof_awake(dims, awake_tree, dof)
          || (partitioned && dof_island[dof] != island)) {
        preconditioned[dof] = product_low[dof] = 0.0f;
        continue;
      }
      PrimalFloatPair z = primal_pair_divide(
          {preconditioned[dof], product_low[dof]},
          {precondition_scale, 0.0f});
      if (!isfinite(z.hi) || !isfinite(z.lo)
          || primal_float_is_subnormal(z.hi)
          || primal_float_is_subnormal(z.lo)) {
        primal_hessian_trace_failure(trace_out, 208, z.hi, z.lo);
        return false;
      }
      preconditioned[dof] = z.hi;
      product_low[dof] = z.lo;
    }
    PrimalFloatPair next_rho = primal_pair_dot_expansion(residual,
        residual_low, preconditioned, product_low, nv);
    if (!primal_pair_positive(next_rho)
        || !isfinite(primal_pair_value(next_rho))) {
      primal_hessian_trace_failure(trace_out, 303, next_rho.hi,
          next_rho.lo);
      return false;
    }
    PrimalFloatPair beta = primal_pair_divide(next_rho, rho);
    for (int dof = 0; dof < nv; ++dof) {
      PrimalFloatPair p = primal_pair_add(
          primal_pair_product(beta,
              {direction[dof], direction_low[dof]}),
          {preconditioned[dof], product_low[dof]});
      direction[dof] = p.hi;
      direction_low[dof] = p.lo;
    }
    rho = next_rho;
  }
  primal_hessian_trace_failure(trace_out, 504, rhs_norm,
      outer_scaled_limit);
  return false;
}

inline bool primal_hessian_solve_elliptic(device const float* mass,
    device const float* L, bool sparse_mass,
    device const float* debug, device float* pair_tail,
    device const float* J, device const float* R, device const float* aref,
    device const float* row_hessian, constant int* dims,
    device const int* awake_tree, device const int* enabled,
    device const int* elliptic_member, device const float* acc,
    device const float* row_residual,
    device const float* row_residual_low,
    device const int* elliptic_start, device const int* elliptic_dim,
    device const float* elliptic_friction, int elliptic_count,
    device const float* rhs, device const float* rhs_low,
    device float* solution, device float* factor,
    device float* temp, device float* inner_residual,
    device float* inner_preconditioned, device float* inner_direction,
    device const int* dof_island, int island, bool partitioned, int nv, int n,
    float tolerance, float scale, device float* trace_out) {
  int active = 0;
  float sum = 0.0f, correction = 0.0f, component_scale = 0.0f;
  bool canonical = true;
  bool rhs_all_active_zero = true;
  for (int dof = 0; dof < nv; ++dof) {
    bool use = solver_dof_awake(dims, awake_tree, dof)
        && (!partitioned || dof_island[dof] == island);
    if (!use) continue;
    ++active;
    float hi = rhs[dof], lo = rhs_low[dof];
    if (hi != 0.0f || lo != 0.0f) rhs_all_active_zero = false;
    if (!isfinite(hi) || !isfinite(lo)) canonical = false;
    PrimalFloatPair represented = primal_pair_add({hi, 0.0f}, {lo, 0.0f});
    if (represented.hi != hi || represented.lo != lo) canonical = false;
    component_scale = max(component_scale, abs(hi) + abs(lo));
    primal_compensated_product_add(hi, hi, sum, correction);
    primal_compensated_product_add(hi, lo, sum, correction);
    primal_compensated_product_add(lo, hi, sum, correction);
    primal_compensated_product_add(lo, lo, sum, correction);
  }
  float rhs_norm_sq = sum + correction;
  float direction_scale = 1.0f / max(1.0f, component_scale);
  float outer = tolerance / max(scale, 1e-30f);
  float outer_sq = outer * outer;
  float inner_sq = 1e-12f * rhs_norm_sq;
  float legacy_limit = min(inner_sq, outer_sq);
  // Keep the earlier solver recurrence on ordinary canonical
  // inputs whenever its float32 residual tests are representable. Scale the
  // equation only when the unscaled arithmetic would flush/overflow a norm,
  // threshold, or direction scale, or when the incoming expansion is not
  // normalized. This preserves that implementation path without masking a
  // nonrepresentable system with a tolerance floor.
  bool legacy_safe = canonical && active > 0 && isfinite(rhs_norm_sq)
      && rhs_norm_sq >= 0.0f
      && (rhs_norm_sq == 0.0f ? rhs_all_active_zero : rhs_norm_sq > 1e-30f)
      && isfinite(direction_scale)
      && direction_scale >= 1.17549435e-38f
      && isfinite(outer) && isfinite(outer_sq)
      && (outer_sq == 0.0f || outer_sq >= 1.17549435e-38f)
      && isfinite(inner_sq) && (inner_sq == 0.0f || inner_sq >= 1.17549435e-38f)
      && isfinite(legacy_limit)
      && (rhs_norm_sq == 0.0f || legacy_limit > 0.0f);
  if (legacy_safe) {
    return primal_hessian_solve_elliptic_legacy(mass, L, sparse_mass,
        debug, pair_tail, J, R, aref, row_hessian, dims, awake_tree, enabled,
        elliptic_member, acc, row_residual, row_residual_low, elliptic_start,
        elliptic_dim, elliptic_friction, elliptic_count, rhs, rhs_low,
        solution, factor, temp, inner_residual, inner_preconditioned,
        inner_direction, dof_island, island, partitioned, nv, n, tolerance,
        scale);
  }
  return primal_hessian_solve_elliptic_scaled(mass, L, sparse_mass, debug,
      pair_tail, J, R, aref, row_hessian, dims, awake_tree, enabled,
      elliptic_member, acc, row_residual, row_residual_low, elliptic_start,
      elliptic_dim, elliptic_friction, elliptic_count, rhs, rhs_low,
      solution, factor, temp, inner_residual, inner_preconditioned,
      inner_direction, dof_island, island, partitioned, nv, n, tolerance,
      scale, trace_out);
}

// Native regression entrypoint for the production bounded PCG solve. With
// n=0 the tested operator is exactly the supplied positive dense mass; the
// production normalization, preconditioner, true-residual, and output
// rescaling path is exercised at scalar and mixed component scales.
kernel void primal_hessian_scale_witness(
    device const float* mass [[buffer(0)]],
    device const float* factor [[buffer(1)]],
    device const float* rhs_pair [[buffer(2)]],
    device const float* tolerance_scale [[buffer(3)]],
    constant int* dims [[buffer(4)]],
    device const int* awake_tree [[buffer(5)]],
    device const int* dof_island [[buffer(6)]],
    device const int* dummy_int [[buffer(7)]],
    device const float* dummy_float [[buffer(8)]],
    device float* pcg_factor [[buffer(9)]],
    device float* pair_tail [[buffer(10)]],
    device float* temp [[buffer(11)]],
    device float* inner_residual [[buffer(12)]],
    device float* inner_preconditioned [[buffer(13)]],
    device float* inner_direction [[buffer(14)]],
    device float* result [[buffer(15)]],
    uint tid [[thread_position_in_grid]]) {
  if (tid != 0) return;
  int nv = dims[0];
  bool ok = primal_hessian_solve_elliptic(mass, factor, false, dummy_float,
      pair_tail, dummy_float, dummy_float, dummy_float, dummy_float, dims,
      awake_tree, dummy_int, dummy_int, dummy_float, dummy_float,
      dummy_float, dummy_int, dummy_int, dummy_float, 0, rhs_pair,
      rhs_pair + nv, result, pcg_factor, temp, inner_residual,
      inner_preconditioned, inner_direction, dof_island, 0, false, nv, 0,
      tolerance_scale[0], tolerance_scale[1], nullptr);
  for (int dof = 0; dof < nv; ++dof)
    result[nv + dof] = pair_tail[4 * nv + dof];
  result[2 * nv] = ok ? 1.0f : 0.0f;
}

// Hager-Zhang beta from paired raw gradients and paired M^-1 gradients.
// Every inner product includes both words; the active island restriction is
// applied before accumulation just as in the caller's source-space CG solve.
inline PrimalFloatPair primal_hager_zhang_beta(
    device const float* direction, device const float* direction_low,
    device const float* gradient, device const float* gradient_low,
    device const float* previous_gradient,
    device const float* previous_gradient_low,
    device const float* mgradient, device const float* mgradient_low,
    device const float* old_mgradient,
    device const float* old_mgradient_low,
    device const int* dof_island, int island, bool partitioned, int nv) {
  PrimalFloatPair d_dot_y = {0.0f, 0.0f};
  PrimalFloatPair y_dot_my = {0.0f, 0.0f};
  PrimalFloatPair y_dot_mg = {0.0f, 0.0f};
  PrimalFloatPair d_dot_g = {0.0f, 0.0f};
  PrimalFloatPair d_norm2 = {0.0f, 0.0f};
  PrimalFloatPair g_norm2 = {0.0f, 0.0f};
  for (int i = 0; i < nv; ++i) {
    if (partitioned && dof_island[i] != island) continue;
    PrimalFloatPair y = primal_pair_add(
        {gradient[i], gradient_low[i]},
        {-previous_gradient[i], -previous_gradient_low[i]});
    PrimalFloatPair my = primal_pair_add(
        {mgradient[i], mgradient_low[i]},
        {-old_mgradient[i], -old_mgradient_low[i]});
    PrimalFloatPair d = {direction[i], direction_low[i]};
    PrimalFloatPair g = {gradient[i], gradient_low[i]};
    PrimalFloatPair mg = {mgradient[i], mgradient_low[i]};
    d_dot_y = primal_pair_add(d_dot_y, primal_pair_product(d, y));
    y_dot_my = primal_pair_add(y_dot_my, primal_pair_product(y, my));
    y_dot_mg = primal_pair_add(y_dot_mg, primal_pair_product(y, mg));
    d_dot_g = primal_pair_add(d_dot_g, primal_pair_product(d, g));
    d_norm2 = primal_pair_add(d_norm2, primal_pair_product(d, d));
    g_norm2 = primal_pair_add(g_norm2, primal_pair_product(g, g));
  }
  if (primal_pair_less_float(d_dot_y, 1e-15f))
    return {0.0f, 0.0f};
  PrimalFloatPair hz_numerator = primal_pair_add(y_dot_mg,
      primal_pair_multiply(primal_pair_product(
          primal_pair_divide(y_dot_my, d_dot_y), d_dot_g), -2.0f));
  PrimalFloatPair beta_hz = primal_pair_divide(hz_numerator, d_dot_y);
  float eta_denom = max(1e-15f,
      sqrt(max(0.0f, primal_pair_value(d_norm2)))
      * min(0.01f, sqrt(max(0.0f, primal_pair_value(g_norm2)))));
  float eta_k = -1.0f / eta_denom;
  return primal_pair_less_float(beta_hz, eta_k)
      ? PrimalFloatPair{eta_k, 0.0f} : beta_hz;
}

inline PrimalFloatPair primal_hager_zhang_direction(
    PrimalFloatPair beta, PrimalFloatPair mgradient,
    PrimalFloatPair direction) {
  return primal_pair_add(primal_pair_multiply(mgradient, -1.0f),
      primal_pair_product(beta, direction));
}

inline float primal_accel_curvature_scalar(device const float* mass,
    device const float* J, device const float* row_hessian,
    device const int* enabled, device const float* direction,
    int nv, int nr, int n) {
  float value = 0.0f;
  for (int i = 0; i < nv; ++i) for (int j = 0; j < nv; ++j)
    value += direction[i] * mass[i * nv + j] * direction[j];
  for (int row = 0; row < n; ++row) if (enabled[row] && row_hessian[row] > 0.0f) {
    float jd = 0.0f;
    for (int i = 0; i < nv; ++i) jd += ccj_get(J, row, i, nv) * direction[i];
    value += row_hessian[row] * jd * jd;
  }
  return value;
}

inline float primal_accel_curvature_elliptic(device const float* mass,
    device const float* L, bool sparse_mass,
    device const float* debug, constant int* dims,
    device const float* J, device const float* R, device const float* aref,
    device const float* row_hessian, device const int* enabled,
    device const int* elliptic_member,
    device const float* row_residual,
    device const float* direction, device const float* direction_low,
    device const int* elliptic_start,
    device const int* elliptic_dim, device const float* elliptic_friction,
    int elliptic_count, int nv, int nr, int n) {
  float value = 0.0f, correction = 0.0f;
  if (sparse_mass) {
    int header = dims[24] + dims[9] * 10;
    int ncomponent = dims[header];
    int layout_offset = dims[header + 3];
    device const int* layout = reinterpret_cast<device const int*>(debug)
        + layout_offset;
    for (int component = 0; component < ncomponent; ++component) {
      int begin = layout[component];
      int width = layout[component + 1] - begin;
      int dof_base = ncomponent + 1 + begin;
      for (int j = 0; j < width; ++j) {
        float product = 0.0f, product_correction = 0.0f;
        float product_low = 0.0f, product_low_correction = 0.0f;
        for (int i = j; i < width; ++i)
          primal_compensated_product_add(
              solver_component_factor_entry(debug, dims, nv,
                  layout[dof_base + i], layout[dof_base + j]),
              direction[layout[dof_base + i]], product, product_correction);
        for (int i = j; i < width; ++i)
          primal_compensated_product_add(
              solver_component_factor_entry(debug, dims, nv,
                  layout[dof_base + i], layout[dof_base + j]),
              direction_low[layout[dof_base + i]], product_low,
              product_low_correction);
        product += product_correction;
        product_low += product_low_correction;
        primal_compensated_product_add(product, product, value, correction);
        primal_compensated_product_add(product, product_low, value, correction);
        primal_compensated_product_add(product_low, product, value, correction);
        primal_compensated_product_add(product_low, product_low, value, correction);
      }
    }
  } else {
    for (int i = 0; i < nv; ++i) for (int j = 0; j < nv; ++j) {
      primal_compensated_product3_add(direction[i], mass[i * nv + j],
          direction[j], value, correction);
      primal_compensated_product3_add(direction[i], mass[i * nv + j],
          direction_low[j], value, correction);
      primal_compensated_product3_add(direction_low[i], mass[i * nv + j],
          direction[j], value, correction);
      primal_compensated_product3_add(direction_low[i], mass[i * nv + j],
          direction_low[j], value, correction);
    }
  }
  for (int row = 0; row < n; ++row) {
    if (!enabled[row] || elliptic_member[row] || row_hessian[row] <= 0.0f) continue;
    PrimalFloatPair jd = primal_pair_add(
        primal_ccj_compensated_dot_pair(J, row, direction, nv),
        primal_ccj_compensated_dot_pair(J, row, direction_low, nv));
    PrimalFloatPair term = primal_pair_product(
        primal_pair_product({row_hessian[row], 0.0f}, jd), jd);
    primal_compensated_product_add(term.hi, 1.0f, value, correction);
    primal_compensated_product_add(term.lo, 1.0f, value, correction);
  }
  for (int block = 0; block < elliptic_count; ++block) {
    int start = elliptic_start[block], dim = elliptic_dim[block];
    if (start < 0 || dim < 2 || dim > 6 || start + dim > n || !enabled[start]) continue;
    thread float local_residual[6], local_force[6], local_hessian[36];
    float local_direction[6], local_direction_low[6];
    for (int k = 0; k < dim; ++k) {
      float residual = row_residual[start + k];
      PrimalFloatPair derivative = primal_ccj_compensated_dot_pair(
          J, start + k, direction, nv);
      derivative = primal_pair_add(derivative,
          primal_ccj_compensated_dot_pair(J, start + k, direction_low, nv));
      local_residual[k] = residual;
      local_direction[k] = derivative.hi;
      local_direction_low[k] = derivative.lo;
    }
    primal_elliptic_cost_force_hessian(local_residual, R + start,
        elliptic_friction + block * 5, dim, local_force, local_hessian);
    for (int i = 0; i < dim; ++i) for (int j = 0; j < dim; ++j)
      primal_compensated_product3_add(local_direction[i],
          local_hessian[i * 6 + j], local_direction[j], value, correction);
    for (int i = 0; i < dim; ++i) for (int j = 0; j < dim; ++j) {
      primal_compensated_product3_add(local_direction[i],
          local_hessian[i * 6 + j], local_direction_low[j], value, correction);
      primal_compensated_product3_add(local_direction_low[i],
          local_hessian[i * 6 + j], local_direction[j], value, correction);
      primal_compensated_product3_add(local_direction_low[i],
          local_hessian[i * 6 + j], local_direction_low[j], value, correction);
    }
  }
  return value + correction;
}

struct PrimalLinePoint {
  PrimalFloatPair alpha;
  float cost;
  float cost_low;
  float slope;
  float slope_low;
  float curvature;
  float curvature_low;
};

inline float primal_friction_cost(float residual, float friction, float bound,
    float inverse_R) {
  if (-bound < residual && residual < bound)
    return 0.5f * inverse_R * residual * residual;
  if (residual <= -bound) return friction * (-0.5f * bound - residual);
  return friction * (-0.5f * bound + residual);
}

inline float primal_friction_cost_delta(float start, float end,
    float alpha, float row_direction, float friction, float bound,
    float inverse_R) {
  int start_zone = (-bound < start && start < bound)
      ? 0 : (start <= -bound ? -1 : 1);
  int end_zone = (-bound < end && end < bound)
      ? 0 : (end <= -bound ? -1 : 1);
  if (start_zone == 0 && end_zone == 0)
    return 0.5f * inverse_R * (end - start) * (end + start);
  if (start_zone == -1 && end_zone == -1)
    return -friction * alpha * row_direction;
  if (start_zone == 1 && end_zone == 1)
    return friction * alpha * row_direction;
  // Match MuJoCo's source fallback only when the line crosses a Huber corner.
  return primal_friction_cost(end, friction, bound, inverse_R)
      - primal_friction_cost(start, friction, bound, inverse_R);
}

inline PrimalFloatPair primal_huber_cost_pair(PrimalFloatPair residual,
    float friction, float bound, float inverse_R) {
  if (primal_pair_less_float(residual, -bound))
    return primal_pair_renormalize(
        friction * (-0.5f * bound - residual.hi), -friction * residual.lo);
  if (primal_pair_less_equal_float(residual, bound))
    return primal_pair_multiply(primal_pair_product(residual, residual),
        0.5f * inverse_R);
  return primal_pair_renormalize(
      friction * (-0.5f * bound + residual.hi), friction * residual.lo);
}

inline PrimalFloatPair primal_huber_cost_delta_pair(
    PrimalFloatPair start, PrimalFloatPair end, PrimalFloatPair alpha,
    PrimalFloatPair row_direction, float friction, float bound,
    float inverse_R) {
  bool start_left = primal_pair_less_equal_float(start, -bound);
  bool start_right = primal_pair_less_float({bound - start.hi, -start.lo}, 0.0f);
  bool end_left = primal_pair_less_equal_float(end, -bound);
  bool end_right = primal_pair_less_float({bound - end.hi, -end.lo}, 0.0f);
  if (!start_left && !start_right && !end_left && !end_right) {
    PrimalFloatPair delta = primal_pair_product(row_direction, alpha);
    PrimalFloatPair midpoint_sum = primal_pair_add(start, end);
    return primal_pair_multiply(
        primal_pair_product(delta, midpoint_sum), 0.5f * inverse_R);
  }
  if (start_left && end_left)
    return primal_pair_multiply(primal_pair_product(row_direction, alpha),
        -friction);
  if (start_right && end_right)
    return primal_pair_multiply(primal_pair_product(row_direction, alpha),
        friction);
  return primal_pair_add(primal_huber_cost_pair(end, friction, bound, inverse_R),
      primal_pair_multiply(
          primal_huber_cost_pair(start, friction, bound, inverse_R), -1.0f));
}

inline PrimalFloatPair primal_unilateral_cost_delta_pair(
    PrimalFloatPair start, PrimalFloatPair end, float inverse_R) {
  bool old_active = primal_pair_less_float(start, 0.0f);
  bool new_active = primal_pair_less_float(end, 0.0f);
  if (old_active && new_active) {
    PrimalFloatPair delta = primal_pair_add(end,
        primal_pair_multiply(start, -1.0f));
    PrimalFloatPair sum = primal_pair_add(end, start);
    return primal_pair_multiply(primal_pair_product(delta, sum),
        0.5f * inverse_R);
  }
  if (!old_active && !new_active) return {0.0f, 0.0f};
  if (new_active)
    return primal_pair_multiply(primal_pair_product(end, end),
        0.5f * inverse_R);
  return primal_pair_multiply(primal_pair_product(start, start),
      -0.5f * inverse_R);
}

inline float primal_elliptic_cost_at(float alpha, float normal0,
    float normal_dir, float tangent_sq0, float tangent_cross,
    float tangent_sq_dir, float mu, float Dm,
    float bottom_constant, float bottom_linear, float bottom_quadratic) {
  float normal = normal0 + alpha * normal_dir;
  float tangent_sq = tangent_sq0
      + alpha * (2.0f * tangent_cross + alpha * tangent_sq_dir);
  if (tangent_sq <= 0.0f)
    return normal < 0.0f
        ? bottom_constant + alpha * alpha * bottom_quadratic
            + alpha * bottom_linear : 0.0f;
  float tangent = sqrt(tangent_sq);
  if (normal >= mu * tangent) return 0.0f;
  if (mu * normal + tangent <= 0.0f)
    return bottom_constant + alpha * alpha * bottom_quadratic
        + alpha * bottom_linear;
  float gap = normal - mu * tangent;
  return 0.5f * Dm * gap * gap;
}

inline float primal_elliptic_cost_delta(float alpha,
    thread const float* residual, thread const float* row_direction,
    device const float* R, device const float* friction, int dim) {
  if (dim < 2 || dim > 6) return 0.0f;
  float mu = max(friction[0]
      * sqrt(max(R[1], 0.0f) / max(R[0], 1e-15f)), 0.0f);
  // The source elliptic path reduces to a normal unilateral row at zero mu.
  if (!(mu > 1e-12f)) {
    float start = residual[0];
    float end = start + alpha * row_direction[0];
    float inverse_R = 1.0f / max(R[0], 1e-15f);
    float old_cost = start < 0.0f ? 0.5f * inverse_R * start * start : 0.0f;
    float new_cost = end < 0.0f ? 0.5f * inverse_R * end * end : 0.0f;
    return new_cost - old_cost;
  }

  float normal0 = mu * residual[0];
  float normal_dir = mu * row_direction[0];
  float tangent_sq0 = 0.0f;
  float tangent_cross = 0.0f;
  float tangent_sq_dir = 0.0f;
  float bottom_constant = 0.0f;
  float bottom_linear = 0.0f;
  float bottom_quadratic = 0.0f;
  for (int k = 1; k < dim; ++k) {
    float fk = max(friction[k - 1], 0.0f);
    float u = fk * residual[k];
    float v = fk * row_direction[k];
    tangent_sq0 += u * u;
    tangent_cross += u * v;
    tangent_sq_dir += v * v;
  }
  for (int k = 0; k < dim; ++k) {
    float inverse_R = 1.0f / max(R[k], 1e-15f);
    bottom_constant += 0.5f * inverse_R * residual[k] * residual[k];
    bottom_linear += inverse_R * residual[k] * row_direction[k];
    bottom_quadratic += 0.5f * inverse_R * row_direction[k] * row_direction[k];
  }
  float Dm = (1.0f / max(R[0], 1e-15f))
      / (mu * mu * (1.0f + mu * mu));
  float N0 = normal0;
  float T0 = sqrt(max(tangent_sq0, 0.0f));
  float N1 = normal0 + alpha * normal_dir;
  float Tsq1 = tangent_sq0
      + alpha * (2.0f * tangent_cross + alpha * tangent_sq_dir);
  int zone0;
  if (tangent_sq0 <= 0.0f) zone0 = N0 < 0.0f ? 2 : 1;
  else if (N0 >= mu * T0) zone0 = 1;
  else if (mu * N0 + T0 <= 0.0f) zone0 = 2;
  else zone0 = 3;
  int zone1;
  if (Tsq1 <= 0.0f) zone1 = N1 < 0.0f ? 2 : 1;
  else {
    float T1 = sqrt(Tsq1);
    if (N1 >= mu * T1) zone1 = 1;
    else if (mu * N1 + T1 <= 0.0f) zone1 = 2;
    else zone1 = 3;
  }
  if (zone0 == 1 && zone1 == 1) return 0.0f;
  if (zone0 == 2 && zone1 == 2)
    return alpha * alpha * bottom_quadratic + alpha * bottom_linear;
  if (zone0 == 3 && zone1 == 3) {
    float T1 = sqrt(max(Tsq1, 0.0f));
    float diff0 = N0 - mu * T0;
    float diff1 = N1 - mu * T1;
    return 0.5f * Dm * (diff1 - diff0) * (diff1 + diff0);
  }
  float end_cost = primal_elliptic_cost_at(alpha, normal0, normal_dir,
      tangent_sq0, tangent_cross, tangent_sq_dir, mu, Dm,
      bottom_constant, bottom_linear, bottom_quadratic);
  float start_cost = primal_elliptic_cost_at(0.0f, normal0, normal_dir,
      tangent_sq0, tangent_cross, tangent_sq_dir, mu, Dm,
      bottom_constant, bottom_linear, bottom_quadratic);
  return end_cost - start_cost;
}

// PrimalEval's line-point derivatives come from the fixed search polynomial
// and the elliptic zone at alpha, not from a rounded candidate acceleration.
inline void primal_elliptic_line_derivatives(float alpha,
    thread const float* residual, thread const float* row_direction,
    device const float* R, device const float* friction, int dim,
    thread float* derivative, thread float* second_derivative) {
  if (dim < 2 || dim > 6) return;
  float mu = max(friction[0]
      * sqrt(max(R[1], 0.0f) / max(R[0], 1e-15f)), 0.0f);
  float bottom_linear = 0.0f;
  float bottom_quadratic = 0.0f;
  for (int k = 0; k < dim; ++k) {
    float inverse_R = 1.0f / max(R[k], 1e-15f);
    bottom_linear += inverse_R * residual[k] * row_direction[k];
    bottom_quadratic += 0.5f * inverse_R
        * row_direction[k] * row_direction[k];
  }
  float normal0 = mu * residual[0];
  float normal_dir = mu * row_direction[0];
  float tangent_sq0 = 0.0f;
  float tangent_cross = 0.0f;
  float tangent_sq_dir = 0.0f;
  for (int k = 1; k < dim; ++k) {
    float fk = max(friction[k - 1], 0.0f);
    float u = fk * residual[k];
    float v = fk * row_direction[k];
    tangent_sq0 += u * u;
    tangent_cross += u * v;
    tangent_sq_dir += v * v;
  }
  float normal = normal0 + alpha * normal_dir;
  float tangent_sq = tangent_sq0
      + alpha * (2.0f * tangent_cross + alpha * tangent_sq_dir);
  if (tangent_sq <= 0.0f) {
    if (normal < 0.0f) {
      *derivative += bottom_linear + 2.0f * alpha * bottom_quadratic;
      *second_derivative += 2.0f * bottom_quadratic;
    }
    return;
  }
  float tangent = sqrt(tangent_sq);
  if (normal >= mu * tangent) return;
  if (mu * normal + tangent <= 0.0f) {
    *derivative += bottom_linear + 2.0f * alpha * bottom_quadratic;
    *second_derivative += 2.0f * bottom_quadratic;
    return;
  }
  float Dm = (1.0f / max(R[0], 1e-15f))
      / (mu * mu * (1.0f + mu * mu));
  float tangent_first = (tangent_cross + alpha * tangent_sq_dir) / tangent;
  float tangent_second = tangent_sq_dir / tangent
      - (tangent_cross + alpha * tangent_sq_dir)
          * tangent_first / (tangent * tangent);
  float gap = normal - mu * tangent;
  float gap_first = normal_dir - mu * tangent_first;
  float gap_second = -mu * tangent_second;
  *derivative += Dm * gap * gap_first;
  *second_derivative += Dm
      * (gap_first * gap_first + gap * gap_second);
}

inline PrimalFloatPair primal_elliptic_cost_at_pair(
    PrimalFloatPair alpha, PrimalFloatPair normal0,
    PrimalFloatPair normal_dir, PrimalFloatPair tangent_sq0,
    PrimalFloatPair tangent_cross, PrimalFloatPair tangent_sq_dir,
    PrimalFloatPair mu, PrimalFloatPair Dm,
    PrimalFloatPair bottom_constant, PrimalFloatPair bottom_linear,
    PrimalFloatPair bottom_quadratic) {
  PrimalFloatPair normal = primal_pair_add(normal0,
      primal_pair_product(alpha, normal_dir));
  PrimalFloatPair tangent_sq = primal_pair_add(tangent_sq0,
      primal_pair_product(alpha, primal_pair_add(
          primal_pair_multiply(tangent_cross, 2.0f),
          primal_pair_product(alpha, tangent_sq_dir))));
  if (primal_pair_less_equal_float(tangent_sq, 0.0f)) {
    if (primal_pair_negative(normal))
      return primal_pair_add(bottom_constant,
          primal_pair_add(primal_pair_product(alpha, bottom_linear),
              primal_pair_product(primal_pair_product(alpha, alpha),
                  bottom_quadratic)));
    return {0.0f, 0.0f};
  }
  PrimalFloatPair tangent = primal_pair_sqrt(tangent_sq);
  if (!primal_pair_less(normal, primal_pair_product(mu, tangent)))
    return {0.0f, 0.0f};
  if (primal_pair_less_equal_float(primal_pair_add(
      primal_pair_product(mu, normal), tangent), 0.0f))
    return primal_pair_add(bottom_constant,
        primal_pair_add(primal_pair_product(alpha, bottom_linear),
            primal_pair_product(primal_pair_product(alpha, alpha),
                bottom_quadratic)));
  PrimalFloatPair gap = primal_pair_add(normal,
      primal_pair_multiply(primal_pair_product(mu, tangent), -1.0f));
  return primal_pair_multiply(primal_pair_product(Dm,
      primal_pair_product(gap, gap)), 0.5f);
}

inline PrimalFloatPair primal_elliptic_line_eval_pair(
    PrimalFloatPair alpha,
    thread const PrimalFloatPair* residual,
    thread const PrimalFloatPair* direction,
    device const float* R, device const float* friction, int dim,
    thread PrimalFloatPair& derivative,
    thread PrimalFloatPair& second_derivative) {
  derivative = {0.0f, 0.0f};
  second_derivative = {0.0f, 0.0f};
  if (dim < 2 || dim > 6) return {0.0f, 0.0f};

  PrimalFloatPair mu = primal_pair_multiply(
      primal_pair_sqrt(primal_pair_divide({max(R[1], 0.0f), 0.0f},
          {max(R[0], 1e-15f), 0.0f})), friction[0]);
  PrimalFloatPair Dm = primal_pair_divide(
      primal_pair_divide({1.0f, 0.0f}, {max(R[0], 1e-15f), 0.0f}),
      primal_pair_product(primal_pair_product(mu, mu),
          primal_pair_add({1.0f, 0.0f},
              primal_pair_product(mu, mu))));
  PrimalFloatPair normal0 = primal_pair_product(mu, residual[0]);
  PrimalFloatPair normal_dir = primal_pair_product(mu, direction[0]);
  PrimalFloatPair tangent_sq0 = {0.0f, 0.0f};
  PrimalFloatPair tangent_cross = {0.0f, 0.0f};
  PrimalFloatPair tangent_sq_dir = {0.0f, 0.0f};
  PrimalFloatPair bottom_constant = {0.0f, 0.0f};
  PrimalFloatPair bottom_linear = {0.0f, 0.0f};
  PrimalFloatPair bottom_quadratic = {0.0f, 0.0f};
  for (int k = 0; k < dim; ++k) {
    PrimalFloatPair inverse_R = primal_pair_divide({1.0f, 0.0f},
        {max(R[k], 1e-15f), 0.0f});
    bottom_constant = primal_pair_add(bottom_constant,
        primal_pair_multiply(primal_pair_product(
            primal_pair_product(residual[k], residual[k]), inverse_R), 0.5f));
    bottom_linear = primal_pair_add(bottom_linear,
        primal_pair_product(primal_pair_product(residual[k], direction[k]),
            inverse_R));
    bottom_quadratic = primal_pair_add(bottom_quadratic,
        primal_pair_multiply(primal_pair_product(
            primal_pair_product(direction[k], direction[k]), inverse_R), 0.5f));
    if (k == 0) continue;
    PrimalFloatPair fk = {max(friction[k - 1], 0.0f), 0.0f};
    PrimalFloatPair u = primal_pair_product(fk, residual[k]);
    PrimalFloatPair v = primal_pair_product(fk, direction[k]);
    tangent_sq0 = primal_pair_add(tangent_sq0, primal_pair_product(u, u));
    tangent_cross = primal_pair_add(tangent_cross, primal_pair_product(u, v));
    tangent_sq_dir = primal_pair_add(tangent_sq_dir, primal_pair_product(v, v));
  }

  PrimalFloatPair alpha_normal = primal_pair_add(normal0,
      primal_pair_product(alpha, normal_dir));
  PrimalFloatPair alpha_tangent_sq = primal_pair_add(tangent_sq0,
      primal_pair_product(alpha, primal_pair_add(
          primal_pair_multiply(tangent_cross, 2.0f),
          primal_pair_product(alpha, tangent_sq_dir))));
  int zone0 = 0, zone_alpha = 0;
  if (primal_pair_less_equal_float(tangent_sq0, 0.0f))
    zone0 = primal_pair_negative(normal0) ? 2 : 1;
  else {
    PrimalFloatPair tangent0 = primal_pair_sqrt(tangent_sq0);
    if (!primal_pair_less(normal0, primal_pair_product(mu, tangent0))) zone0 = 1;
    else if (primal_pair_less_equal_float(primal_pair_add(
        primal_pair_product(mu, normal0), tangent0), 0.0f)) zone0 = 2;
    else zone0 = 3;
  }
  if (primal_pair_less_equal_float(alpha_tangent_sq, 0.0f))
    zone_alpha = primal_pair_negative(alpha_normal) ? 2 : 1;
  else {
    PrimalFloatPair tangent1 = primal_pair_sqrt(alpha_tangent_sq);
    if (!primal_pair_less(alpha_normal, primal_pair_product(mu, tangent1))) zone_alpha = 1;
    else if (primal_pair_less_equal_float(primal_pair_add(
        primal_pair_product(mu, alpha_normal), tangent1), 0.0f)) zone_alpha = 2;
    else zone_alpha = 3;
  }

  PrimalFloatPair cost_delta;
  if (zone0 == 1 && zone_alpha == 1) {
    cost_delta = {0.0f, 0.0f};
  } else if (zone0 == 2 && zone_alpha == 2) {
    cost_delta = primal_pair_add(
        primal_pair_product(alpha, bottom_linear),
        primal_pair_product(primal_pair_product(alpha, alpha),
            bottom_quadratic));
  } else if (zone0 == 3 && zone_alpha == 3) {
    PrimalFloatPair tangent0 = primal_pair_sqrt(tangent_sq0);
    PrimalFloatPair tangent1 = primal_pair_sqrt(alpha_tangent_sq);
    PrimalFloatPair diff0 = primal_pair_add(normal0,
        primal_pair_multiply(primal_pair_product(mu, tangent0), -1.0f));
    PrimalFloatPair diff1 = primal_pair_add(alpha_normal,
        primal_pair_multiply(primal_pair_product(mu, tangent1), -1.0f));
    cost_delta = primal_pair_multiply(primal_pair_product(Dm,
        primal_pair_product(primal_pair_add(diff1,
            primal_pair_multiply(diff0, -1.0f)),
            primal_pair_add(diff1, diff0))), 0.5f);
  } else {
    PrimalFloatPair end_cost = primal_elliptic_cost_at_pair(alpha,
        normal0, normal_dir, tangent_sq0, tangent_cross, tangent_sq_dir,
        mu, Dm, bottom_constant, bottom_linear, bottom_quadratic);
    PrimalFloatPair start_cost = primal_elliptic_cost_at_pair(
        {0.0f, 0.0f}, normal0, normal_dir, tangent_sq0, tangent_cross,
        tangent_sq_dir, mu, Dm, bottom_constant, bottom_linear,
        bottom_quadratic);
    cost_delta = primal_pair_add(end_cost,
        primal_pair_multiply(start_cost, -1.0f));
  }

  if (zone_alpha == 2) {
    derivative = primal_pair_add(bottom_linear,
        primal_pair_multiply(primal_pair_product(alpha,
            bottom_quadratic), 2.0f));
    second_derivative = primal_pair_multiply(bottom_quadratic, 2.0f);
  } else if (zone_alpha == 3) {
    PrimalFloatPair tangent = primal_pair_sqrt(alpha_tangent_sq);
    PrimalFloatPair cross_at_alpha = primal_pair_add(tangent_cross,
        primal_pair_product(alpha, tangent_sq_dir));
    PrimalFloatPair tangent_first = primal_pair_divide(cross_at_alpha, tangent);
    PrimalFloatPair tangent_second = primal_pair_add(
        primal_pair_divide(tangent_sq_dir, tangent),
        primal_pair_multiply(primal_pair_divide(
            primal_pair_product(cross_at_alpha, tangent_first),
            primal_pair_product(tangent, tangent)), -1.0f));
    PrimalFloatPair gap = primal_pair_add(alpha_normal,
        primal_pair_multiply(primal_pair_product(mu, tangent), -1.0f));
    PrimalFloatPair gap_first = primal_pair_add(normal_dir,
        primal_pair_multiply(primal_pair_product(mu, tangent_first), -1.0f));
    PrimalFloatPair gap_second = primal_pair_multiply(
        primal_pair_product(mu, tangent_second), -1.0f);
    derivative = primal_pair_product(Dm,
        primal_pair_product(gap, gap_first));
    second_derivative = primal_pair_product(Dm,
        primal_pair_add(primal_pair_product(gap_first, gap_first),
            primal_pair_product(gap, gap_second)));
  }
  return cost_delta;
}

// Compute the source's shifted primal line-search objective directly. The
// absolute objective can be large while a late CG step improves it by less
// than one float32 ulp; subtracting two absolute evaluations then loses the
// progress decision. This mirrors MuJoCo 3.10 PrimalEval's shared Gauss,
// equality, active-unilateral, and elliptic-bottom quadratic coefficients,
// plus Huber and elliptic zone-crossing cost differences.
inline PrimalFloatPair primal_accel_cost_delta_elliptic(
    PrimalFloatPair alpha,
    device const float* mass, bool sparse_mass, device const float* debug,
    constant int* dims, device const float* J, device const float* R,
    device const float* aref, device const float* lo, device const float* hi,
    device const float* q0, device const float* acc,
    device const float* acc_low, device const float* direction,
    device const float* direction_low, device const int* enabled,
    device const int* elliptic_member, device const int* elliptic_start,
    device const int* elliptic_dim, device const float* elliptic_friction,
    int elliptic_count, device float* mass_gradient,
    device float* projected, int nv, int nr, int n, int n_eq_rows,
    thread PrimalFloatPair& derivative,
    thread PrimalFloatPair& second_derivative,
    device const float* mass_delta, device const float* mass_delta_low,
    device const float* base_row_residual,
    device const float* base_row_residual_low) {
  if (mass_delta == nullptr) {
    if (sparse_mass) {
    solver_component_mass_apply_difference(debug, dims, nv, acc, q0,
        projected, mass_gradient);
    } else {
    for (int i = 0; i < nv; ++i) {
      float value = 0.0f;
      for (int j = 0; j < nv; ++j)
        value += mass[i * nv + j] * (acc[j] - q0[j]);
      mass_gradient[i] = value;
    }
    }
  }
  float gauss_linear = 0.0f, gauss_linear_correction = 0.0f;
  device const float* aref_low = primal_aref_low_view(debug, dims, nv, nr);
  float gauss_quadratic = 0.0f, gauss_quadratic_correction = 0.0f;
  for (int i = 0; i < nv; ++i) {
    float g = mass_delta == nullptr ? mass_gradient[i] : mass_delta[i];
    float glow = mass_delta == nullptr ? 0.0f : mass_delta_low[i];
    primal_compensated_product_add(g, direction[i], gauss_linear,
        gauss_linear_correction);
    primal_compensated_product_add(g, direction_low[i], gauss_linear,
        gauss_linear_correction);
    primal_compensated_product_add(glow, direction[i], gauss_linear,
        gauss_linear_correction);
    primal_compensated_product_add(glow, direction_low[i], gauss_linear,
        gauss_linear_correction);
    for (int j = 0; j < nv; ++j) {
      float mij = sparse_mass
          ? solver_component_mass_entry(debug, dims, nv, i, j)
          : mass[i * nv + j];
      primal_compensated_product3_add(direction[i], mij, direction[j],
          gauss_quadratic, gauss_quadratic_correction);
      primal_compensated_product3_add(direction[i], mij, direction_low[j],
          gauss_quadratic, gauss_quadratic_correction);
      primal_compensated_product3_add(direction_low[i], mij, direction[j],
          gauss_quadratic, gauss_quadratic_correction);
      primal_compensated_product3_add(direction_low[i], mij,
          direction_low[j], gauss_quadratic, gauss_quadratic_correction);
    }
  }
  PrimalFloatPair shifted_constant = {0.0f, 0.0f};
  PrimalFloatPair shifted_linear =
      primal_pair_renormalize(gauss_linear, gauss_linear_correction);
  PrimalFloatPair shifted_quadratic = primal_pair_multiply(
      primal_pair_renormalize(gauss_quadratic, gauss_quadratic_correction),
      0.5f);
  PrimalFloatPair delta_cost = {0.0f, 0.0f};
  PrimalFloatPair explicit_derivative = {0.0f, 0.0f};
  PrimalFloatPair explicit_second_derivative = {0.0f, 0.0f};

  for (int row = 0; row < n; ++row) {
    if (!enabled[row] || elliptic_member[row]) continue;
    PrimalFloatPair start = base_row_residual == nullptr
        ? PrimalFloatPair{-aref[row], -aref_low[row]}
        : PrimalFloatPair{base_row_residual[row],
            base_row_residual_low == nullptr ? 0.0f
                                             : base_row_residual_low[row]};
    float start_sum = 0.0f, start_correction = 0.0f;
    float direction_sum = 0.0f, direction_correction = 0.0f;
    for (int dof = 0; dof < nv; ++dof) {
      if (base_row_residual == nullptr)
        primal_compensated_product_add(ccj_get(J, row, dof, nv), acc[dof],
            start_sum, start_correction);
      if (base_row_residual == nullptr)
        primal_compensated_product_add(ccj_get(J, row, dof, nv), acc_low[dof],
            start_sum, start_correction);
      primal_compensated_product_add(ccj_get(J, row, dof, nv), direction[dof],
          direction_sum, direction_correction);
      primal_compensated_product_add(ccj_get(J, row, dof, nv), direction_low[dof],
          direction_sum, direction_correction);
    }
    if (base_row_residual == nullptr)
      start = primal_pair_add(start,
          primal_pair_renormalize(start_sum, start_correction));
    PrimalFloatPair row_direction =
        primal_pair_renormalize(direction_sum, direction_correction);
    PrimalFloatPair end = primal_pair_add(start,
        primal_pair_product(row_direction, alpha));
    float inverse_R = 1.0f / max(R[row], 1e-15f);
    if (row < n_eq_rows) {
      shifted_linear = primal_pair_add(shifted_linear,
          primal_pair_multiply(
              primal_pair_product(start, row_direction), inverse_R));
      shifted_quadratic = primal_pair_add(shifted_quadratic,
          primal_pair_multiply(
              primal_pair_product(row_direction, row_direction),
              0.5f * inverse_R));
    } else {
      bool friction_row = lo[row] < 0.0f && hi[row] > 0.0f
          && isfinite(lo[row]) && isfinite(hi[row])
          && abs(lo[row] + hi[row])
              <= 1e-5f * max(1.0f, abs(hi[row]));
      if (friction_row) {
        float loss = hi[row];
        float bound = R[row] * loss;
        delta_cost = primal_pair_add(delta_cost,
            primal_huber_cost_delta_pair(start, end, alpha,
                row_direction, loss, bound, inverse_R));
        if (!primal_pair_less_equal_float(end, -bound)
            && primal_pair_less_float(end, bound)) {
          explicit_derivative = primal_pair_add(explicit_derivative,
              primal_pair_multiply(
                  primal_pair_product(end, row_direction), inverse_R));
          explicit_second_derivative = primal_pair_add(
              explicit_second_derivative,
              primal_pair_multiply(
                  primal_pair_product(row_direction, row_direction),
                  inverse_R));
        } else if (primal_pair_less_equal_float(end, -bound)) {
          explicit_derivative = primal_pair_add(explicit_derivative,
              primal_pair_multiply(row_direction, -loss));
        } else {
          explicit_derivative = primal_pair_add(explicit_derivative,
              primal_pair_multiply(row_direction, loss));
        }
      } else {
        PrimalFloatPair old_cost = primal_pair_less_float(start, 0.0f)
            ? primal_pair_multiply(primal_pair_product(start, start),
                  0.5f * inverse_R)
            : PrimalFloatPair{0.0f, 0.0f};
        if (primal_pair_less_float(end, 0.0f)) {
          shifted_linear = primal_pair_add(shifted_linear,
              primal_pair_multiply(
                  primal_pair_product(start, row_direction), inverse_R));
          shifted_quadratic = primal_pair_add(shifted_quadratic,
              primal_pair_multiply(
                  primal_pair_product(row_direction, row_direction),
                  0.5f * inverse_R));
        } else {
          delta_cost = primal_pair_add(delta_cost,
              primal_pair_multiply(old_cost, -1.0f));
        }
      }
    }
  }

  for (int block = 0; block < elliptic_count; ++block) {
    int start_row = elliptic_start[block];
    int dim = elliptic_dim[block];
    if (start_row < 0 || dim < 2 || dim > 6 || start_row + dim > n
        || !enabled[start_row]) continue;
    thread PrimalFloatPair row_residual[6], row_direction[6];
    for (int k = 0; k < dim; ++k) {
      int row = start_row + k;
      PrimalFloatPair r = base_row_residual == nullptr
          ? PrimalFloatPair{-aref[row], -aref_low[row]}
          : PrimalFloatPair{base_row_residual[row],
              base_row_residual_low == nullptr ? 0.0f
                                                : base_row_residual_low[row]};
      PrimalFloatPair d = {0.0f, 0.0f};
      for (int dof = 0; dof < nv; ++dof) {
        if (base_row_residual == nullptr)
          r = primal_pair_fma(ccj_get(J, row, dof, nv),
              {acc[dof], acc_low[dof]}, r);
        d = primal_pair_fma(ccj_get(J, row, dof, nv),
            {direction[dof], direction_low[dof]}, d);
      }
      row_residual[k] = r;
      row_direction[k] = d;
    }
    PrimalFloatPair local_derivative, local_second;
    PrimalFloatPair local_delta = primal_elliptic_line_eval_pair(alpha,
        row_residual, row_direction, R + start_row,
        elliptic_friction + block * 5, dim,
        local_derivative, local_second);
    delta_cost = primal_pair_add(delta_cost, local_delta);
    explicit_derivative = primal_pair_add(explicit_derivative,
        local_derivative);
    explicit_second_derivative = primal_pair_add(explicit_second_derivative,
        local_second);
  }
  derivative = primal_pair_add(shifted_linear,
      primal_pair_add(primal_pair_product(shifted_quadratic,
          primal_pair_multiply(alpha, 2.0f)), explicit_derivative));
  second_derivative = primal_pair_add(
      primal_pair_multiply(shifted_quadratic, 2.0f),
      explicit_second_derivative);
  PrimalFloatPair cost = primal_pair_add(delta_cost, shifted_constant);
  cost = primal_pair_add(cost, primal_pair_product(shifted_linear, alpha));
  cost = primal_pair_add(cost,
      primal_pair_product(shifted_quadratic,
          primal_pair_product(alpha, alpha)));
  return cost;
}

// Solve the mass-preconditioner system on one independent solver island.
// Its restricted principal operator is applied directly; solving the full
// matrix and masking afterward would leak cross-island acceleration updates.
inline bool primal_mass_solve_island(device const float* mass,
    device const float* debug, constant int* dims,
    device const float* rhs, device float* solution,
    device float* residual, device float* preconditioned,
    device float* direction, device float* product,
    device const int* awake_tree, device const int* dof_island,
    int island, bool partitioned, int nv) {
  bool sparse_mass = dims[20] == -3;
  int active_dof_count = 0;
  for (int i = 0; i < nv; ++i) {
    bool active = solver_dof_awake(dims, awake_tree, i)
        && (!partitioned || dof_island[i] == island);
    solution[i] = 0.0f;
    residual[i] = active ? rhs[i] : 0.0f;
    active_dof_count += active ? 1 : 0;
  }
  float rhs_norm_sq = primal_compensated_dot(residual, residual, nv);
  if (!isfinite(rhs_norm_sq)) return false;
  if (rhs_norm_sq <= 1e-30f) return true;
  for (int i = 0; i < nv; ++i) {
    if (!solver_dof_awake(dims, awake_tree, i)
        || (partitioned && dof_island[i] != island)) {
      preconditioned[i] = direction[i] = 0.0f;
      continue;
    }
    float diagonal = sparse_mass
        ? solver_component_mass_entry(debug, dims, nv, i, i)
        : mass[i * nv + i];
    if (!(diagonal > 0.0f) || !isfinite(diagonal)) return false;
    preconditioned[i] = residual[i] / diagonal;
    direction[i] = preconditioned[i];
  }
  float rho = primal_compensated_dot(residual, preconditioned, nv);
  if (!(rho > 0.0f) || !isfinite(rho)) return false;
  int since_restart = 0;
  int max_iterations = max(1, 4 * active_dof_count);
  for (int iteration = 0; iteration < max_iterations; ++iteration) {
    for (int i = 0; i < nv; ++i) {
      if (!solver_dof_awake(dims, awake_tree, i)
          || (partitioned && dof_island[i] != island)) {
        product[i] = 0.0f;
        continue;
      }
      float value = 0.0f, correction = 0.0f;
      for (int j = 0; j < nv; ++j)
        if (solver_dof_awake(dims, awake_tree, j)
            && (!partitioned || dof_island[j] == island)) {
        float mij = sparse_mass
            ? solver_component_mass_entry(debug, dims, nv, i, j)
            : mass[i * nv + j];
        primal_compensated_product_add(mij, direction[j], value, correction);
      }
      product[i] = value + correction;
    }
    float denom = primal_compensated_dot(direction, product, nv);
    if (!(denom > 0.0f) || !isfinite(denom)) return false;
    float alpha = rho / denom;
    if (!isfinite(alpha)) return false;
    for (int i = 0; i < nv; ++i)
      if (solver_dof_awake(dims, awake_tree, i)
          && (!partitioned || dof_island[i] == island)) {
      solution[i] = fma(alpha, direction[i], solution[i]);
      residual[i] = fma(-alpha, product[i], residual[i]);
    }
    float residual_norm_sq = primal_compensated_dot(residual, residual, nv);
    if (!isfinite(residual_norm_sq)) return false;
    ++since_restart;
    if (residual_norm_sq <= 1e-12f * rhs_norm_sq
        || since_restart >= active_dof_count) {
      // Recompute the original principal-system residual before accepting a
      // preconditioner solve. If float32 recursive CG has stagnated, restart
      // from this true residual within the fixed 4*active_dof_count budget.
      for (int i = 0; i < nv; ++i) {
        if (!solver_dof_awake(dims, awake_tree, i)
            || (partitioned && dof_island[i] != island)) {
          product[i] = 0.0f;
          continue;
        }
        float value = 0.0f, correction = 0.0f;
        for (int j = 0; j < nv; ++j) {
          if (!solver_dof_awake(dims, awake_tree, j)
              || (partitioned && dof_island[j] != island)) continue;
          float mij = sparse_mass
              ? solver_component_mass_entry(debug, dims, nv, i, j)
              : mass[i * nv + j];
          primal_compensated_product_add(mij, solution[j], value, correction);
        }
        product[i] = value + correction;
      }
      for (int i = 0; i < nv; ++i) {
        if (!solver_dof_awake(dims, awake_tree, i)
            || (partitioned && dof_island[i] != island)) {
          residual[i] = 0.0f;
          continue;
        }
        residual[i] = rhs[i] - product[i];
      }
      float true_residual_sq = primal_compensated_dot(residual, residual, nv);
      if (!isfinite(true_residual_sq)) return false;
      if (true_residual_sq <= 1e-10f * rhs_norm_sq) return true;
      if (iteration + 1 >= max_iterations) return false;
      since_restart = 0;
      for (int i = 0; i < nv; ++i) {
        if (!solver_dof_awake(dims, awake_tree, i)
            || (partitioned && dof_island[i] != island)) {
          preconditioned[i] = direction[i] = 0.0f;
          continue;
        }
        float diagonal = sparse_mass
            ? solver_component_mass_entry(debug, dims, nv, i, i)
            : mass[i * nv + i];
        if (!(diagonal > 0.0f) || !isfinite(diagonal)) return false;
        preconditioned[i] = residual[i] / diagonal;
        direction[i] = preconditioned[i];
      }
      rho = primal_compensated_dot(residual, preconditioned, nv);
      if (!(rho > 0.0f) || !isfinite(rho)) return false;
      continue;
    }
    for (int i = 0; i < nv; ++i) {
      if (!solver_dof_awake(dims, awake_tree, i)
          || (partitioned && dof_island[i] != island)) {
        preconditioned[i] = 0.0f;
        continue;
      }
      float diagonal = sparse_mass
          ? solver_component_mass_entry(debug, dims, nv, i, i)
          : mass[i * nv + i];
      if (!(diagonal > 0.0f) || !isfinite(diagonal)) return false;
      preconditioned[i] = residual[i] / diagonal;
    }
    float next_rho = primal_compensated_dot(residual, preconditioned, nv);
    if (!(next_rho > 0.0f) || !isfinite(next_rho)) return false;
    float beta = next_rho / rho;
    for (int i = 0; i < nv; ++i)
      direction[i] = solver_dof_awake(dims, awake_tree, i)
          && (!partitioned || dof_island[i] == island)
          ? fma(beta, direction[i], preconditioned[i]) : 0.0f;
    rho = next_rho;
  }
  float residual_norm_sq = primal_compensated_dot(residual, residual, nv);
  return isfinite(residual_norm_sq)
      && residual_norm_sq <= 1e-10f * rhs_norm_sq;
}

// Apply the exact acceleration-space Gauss operator as the PCG
// preconditioner.  A solver island uses its restricted principal mass block;
// the unrestricted paths use the already factored dense or component mass.
// Inactive DOFs remain excluded from both the solve and its result.
__attribute__((noinline)) bool primal_hessian_mass_precondition_elliptic(
    device const float* mass, device const float* L, bool sparse_mass,
    device const float* debug, constant int* dims,
    device const int* awake_tree, device const float* rhs,
    device float* solution, device float* inner_residual,
    device float* inner_preconditioned, device float* inner_direction,
    device float* inner_product, device float* temp,
    device const int* dof_island, int island, bool partitioned, int nv) {
  bool has_inactive_dof = false;
  for (int dof = 0; dof < nv; ++dof)
    has_inactive_dof = has_inactive_dof
        || !solver_dof_awake(dims, awake_tree, dof);
  if (partitioned || (!sparse_mass && has_inactive_dof)) {
    if (partitioned && primal_mass_solve_complete_island(mass, L,
        sparse_mass, debug, dims, awake_tree, dof_island, island, rhs,
        solution, inner_residual, nv)) return true;
    return primal_mass_solve_island(mass, debug, dims, rhs, solution,
        inner_residual, inner_preconditioned, inner_direction, inner_product,
        awake_tree, dof_island, island, partitioned, nv);
  }
  bool solved = sparse_mass
      ? solver_component_mass_solve(debug, dims, awake_tree, nv, rhs,
          solution, temp)
      : true;
  if (!sparse_mass) primal_mass_solve(L, rhs, solution, temp, nv);
  return solved;
}

// Solve M^-1(rhs_hi+rhs_lo) as a two-word vector. The direct component or
// Cholesky path first computes the high solve, measures its original mass
// residual with compensated M*(hi+lo), then adds a source-space correction
// solve into the low word. If a partition cuts a mass component, preserve the
// existing restricted iterative preconditioner and its fixed budget.
__attribute__((noinline)) bool primal_hessian_mass_precondition_pair_elliptic(
    device const float* mass, device const float* L, bool sparse_mass,
    device const float* debug, constant int* dims,
    device const int* awake_tree, device const float* rhs_hi,
    device const float* rhs_low, device float* solution_hi,
    device float* solution_low, device float* residual_work,
    device float* residual_hi, device float* residual_low,
    device float* correction_hi, device float* correction_work,
    device const int* dof_island, int island, bool partitioned, int nv) {
  if (!primal_hessian_mass_precondition_elliptic(mass, L, sparse_mass,
      debug, dims, awake_tree, rhs_hi, solution_hi, residual_work,
      residual_hi, residual_low, correction_hi, correction_work,
      dof_island, island, partitioned, nv)) return false;

  bool has_inactive_dof = false;
  for (int dof = 0; dof < nv; ++dof)
    has_inactive_dof = has_inactive_dof
        || !solver_dof_awake(dims, awake_tree, dof);
  bool direct = !partitioned && (sparse_mass || !has_inactive_dof);
  if (partitioned) {
    // This call is also the exact component-boundary check. On success it
    // writes a harmless probe solution into correction_hi; that scratch is
    // overwritten by the actual residual correction below.
    direct = primal_mass_solve_complete_island(mass, L, sparse_mass, debug,
        dims, awake_tree, dof_island, island, rhs_hi, correction_hi,
        residual_work, nv);
  }
  if (!direct) {
    return primal_hessian_mass_precondition_elliptic(mass, L, sparse_mass,
        debug, dims, awake_tree, rhs_low, solution_low, residual_work,
        residual_hi, residual_low, correction_hi, correction_work,
        dof_island, island, partitioned, nv);
  }

  for (int dof = 0; dof < nv; ++dof) solution_low[dof] = 0.0f;
  for (int dof = 0; dof < nv; ++dof) {
    if (!solver_dof_awake(dims, awake_tree, dof)
        || (partitioned && dof_island[dof] != island)) {
      residual_hi[dof] = residual_low[dof] = 0.0f;
      continue;
    }
    float sum = 0.0f, correction = 0.0f;
    for (int other = 0; other < nv; ++other) {
      if (!solver_dof_awake(dims, awake_tree, other)
          || (partitioned && dof_island[other] != island)) continue;
      float mij = sparse_mass
          ? solver_component_mass_entry(debug, dims, nv, dof, other)
          : mass[dof * nv + other];
      primal_compensated_product_add(mij, solution_hi[other], sum, correction);
      primal_compensated_product_add(mij, solution_low[other], sum, correction);
    }
    PrimalFloatPair product = primal_pair_renormalize(sum, correction);
    PrimalFloatPair residual = primal_pair_add(
        {rhs_hi[dof], rhs_low[dof]}, {-product.hi, -product.lo});
    residual_hi[dof] = residual.hi;
    residual_low[dof] = residual.lo;
  }

  bool correction_ok = false;
  if (partitioned) {
    correction_ok = primal_mass_solve_complete_island(mass, L, sparse_mass,
        debug, dims, awake_tree, dof_island, island, residual_hi,
        correction_hi, residual_work, nv);
    if (!correction_ok) return false;
    correction_ok = primal_mass_solve_complete_island(mass, L, sparse_mass,
        debug, dims, awake_tree, dof_island, island, residual_low,
        solution_low, residual_work, nv);
    if (!correction_ok) return false;
  } else if (sparse_mass) {
    if (!solver_component_mass_solve(debug, dims, awake_tree, nv,
        residual_hi, correction_hi, correction_work)) return false;
    if (!solver_component_mass_solve(debug, dims, awake_tree, nv,
        residual_low, solution_low, correction_work)) return false;
  } else {
    primal_mass_solve(L, residual_hi, correction_hi, correction_work, nv);
    primal_mass_solve(L, residual_low, solution_low, correction_work, nv);
  }
  for (int dof = 0; dof < nv; ++dof) {
    if (!solver_dof_awake(dims, awake_tree, dof)
        || (partitioned && dof_island[dof] != island)) {
      solution_low[dof] = 0.0f;
      continue;
    }
    PrimalFloatPair correction = primal_pair_add(
        {correction_hi[dof], 0.0f}, {solution_low[dof], 0.0f});
    solution_low[dof] = correction.hi + correction.lo;
  }
  return true;
}

__attribute__((noinline)) bool primal_cg_mass_precondition_pair_elliptic(
    device const float* mass, device const float* L, bool sparse_mass,
    device const float* debug, constant int* dims,
    device const int* awake_tree, device const float* rhs_hi,
    device const float* rhs_low, device float* solution_hi,
    device float* solution_low, device float* residual_work,
    device float* residual_hi, device float* residual_low,
    device float* correction_hi, device float* correction_work,
    device float* low_stash, device const int* dof_island, int island,
    bool partitioned, int nv) {
  bool has_inactive_dof = false;
  for (int dof = 0; dof < nv; ++dof)
    has_inactive_dof = has_inactive_dof
        || !solver_dof_awake(dims, awake_tree, dof);
  bool direct = !partitioned && (sparse_mass || !has_inactive_dof);
  if (partitioned) {
    direct = primal_mass_solve_complete_island(mass, L, sparse_mass, debug,
        dims, awake_tree, dof_island, island, rhs_hi, correction_hi,
        residual_work, nv);
  }
  if (direct) {
    return primal_hessian_mass_precondition_pair_elliptic(mass, L,
        sparse_mass, debug, dims, awake_tree, rhs_hi, rhs_low, solution_hi,
        solution_low, residual_work, residual_hi, residual_low, correction_hi,
        correction_work, dof_island, island, partitioned, nv);
  }

  if (!primal_hessian_mass_precondition_elliptic(mass, L, sparse_mass,
      debug, dims, awake_tree, rhs_hi, solution_hi, residual_work,
      residual_hi, residual_low, correction_hi, correction_work,
      dof_island, island, partitioned, nv)) return false;
  for (int dof = 0; dof < nv; ++dof) solution_low[dof] = 0.0f;

  for (int dof = 0; dof < nv; ++dof) {
    if (!solver_dof_awake(dims, awake_tree, dof)
        || (partitioned && dof_island[dof] != island)) {
      residual_hi[dof] = residual_low[dof] = low_stash[dof] = 0.0f;
      continue;
    }
    float sum = 0.0f, correction = 0.0f;
    for (int other = 0; other < nv; ++other) {
      if (!solver_dof_awake(dims, awake_tree, other)
          || (partitioned && dof_island[other] != island)) continue;
      float mij = sparse_mass
          ? solver_component_mass_entry(debug, dims, nv, dof, other)
          : mass[dof * nv + other];
      primal_compensated_product_add(mij, solution_hi[other], sum, correction);
    }
    PrimalFloatPair product = primal_pair_renormalize(sum, correction);
    PrimalFloatPair residual = primal_pair_add(
        {rhs_hi[dof], rhs_low[dof]}, {-product.hi, -product.lo});
    residual_hi[dof] = residual.hi;
    residual_low[dof] = residual.lo;
    low_stash[dof] = 0.0f;
  }

  // Solve the high residual with the restricted principal operator. The
  // low residual is held aside because the iterative solve reuses that
  // vector as its preconditioned workspace.
  if (!primal_mass_solve_island(mass, debug, dims, residual_hi,
      correction_hi, residual_work, residual_low, correction_work,
      solution_low, awake_tree, dof_island, island, partitioned, nv))
    return false;
  for (int dof = 0; dof < nv; ++dof) {
    if (!solver_dof_awake(dims, awake_tree, dof)
        || (partitioned && dof_island[dof] != island)) {
      solution_hi[dof] = solution_low[dof] = 0.0f;
      continue;
    }
    PrimalFloatPair updated = primal_pair_add(
        {solution_hi[dof], 0.0f}, {correction_hi[dof], 0.0f});
    solution_hi[dof] = updated.hi;
    // Preserve the representational addition tail while the restricted
    // solve reuses solution_low as product scratch below.
    low_stash[dof] = updated.lo;
    solution_low[dof] = 0.0f;
  }

  // Include the saved low residual, plus the high solve's represented update,
  // in the final residual before solving the low correction.
  for (int dof = 0; dof < nv; ++dof) {
    if (!solver_dof_awake(dims, awake_tree, dof)
        || (partitioned && dof_island[dof] != island)) {
      residual_hi[dof] = 0.0f;
      continue;
    }
    float sum = 0.0f, correction = 0.0f;
    for (int other = 0; other < nv; ++other) {
      if (!solver_dof_awake(dims, awake_tree, other)
          || (partitioned && dof_island[other] != island)) continue;
      float mij = sparse_mass
          ? solver_component_mass_entry(debug, dims, nv, dof, other)
          : mass[dof * nv + other];
      primal_compensated_product_add(mij, solution_hi[other], sum, correction);
      primal_compensated_product_add(mij, low_stash[other], sum, correction);
    }
    PrimalFloatPair product = primal_pair_renormalize(sum, correction);
    PrimalFloatPair residual = primal_pair_add(
        {rhs_hi[dof], rhs_low[dof]}, {-product.hi, -product.lo});
    residual_hi[dof] = residual.hi;
    residual_low[dof] = residual.lo;
  }
  if (!primal_mass_solve_island(mass, debug, dims, residual_low,
      solution_low, residual_work, residual_hi, correction_work,
      correction_hi, awake_tree, dof_island, island, partitioned, nv))
    return false;
  for (int dof = 0; dof < nv; ++dof) {
    if (!solver_dof_awake(dims, awake_tree, dof)
        || (partitioned && dof_island[dof] != island)) {
      solution_low[dof] = 0.0f;
      continue;
    }
    PrimalFloatPair low_correction = primal_pair_add(
        {low_stash[dof], 0.0f}, {solution_low[dof], 0.0f});
    PrimalFloatPair updated = primal_pair_add(
        {solution_hi[dof], 0.0f}, low_correction);
    solution_hi[dof] = updated.hi;
    solution_low[dof] = updated.lo;
  }
  return true;
}

inline PrimalLinePoint primal_line_eval_elliptic(PrimalFloatPair alpha,
    device const float* mass, bool sparse_mass, device const float* debug,
    constant int* dims, device const float* J,
    device const float* R, device const float* aref, device const float* lo,
    device const float* hi, device const float* q0, device const float* acc,
    device const float* acc_low, device const float* direction,
    device const float* direction_low, device const int* enabled,
    device const int* elliptic_member, device const int* elliptic_start,
    device const int* elliptic_dim, device const float* elliptic_friction,
  int elliptic_count, device float* candidate,
    device float* gradient, device float* generalized_force,
    int nv, int nr, int n, int n_eq_rows,
    device const float* mass_delta, device const float* mass_delta_low,
    device const float* row_residual,
    device const float* row_residual_low) {
  PrimalFloatPair slope = {0.0f, 0.0f};
  PrimalFloatPair curvature = {0.0f, 0.0f};
  PrimalFloatPair cost_delta = primal_accel_cost_delta_elliptic(alpha, mass,
      sparse_mass, debug, dims, J, R, aref, lo, hi, q0, acc, acc_low,
      direction, direction_low,
      enabled, elliptic_member, elliptic_start, elliptic_dim,
      elliptic_friction, elliptic_count, gradient, generalized_force,
      nv, nr, n, n_eq_rows, slope, curvature,
      mass_delta, mass_delta_low, row_residual, row_residual_low);
  for (int i = 0; i < nv; ++i) {
    PrimalFloatPair next = primal_pair_add(
        primal_pair_product(alpha, {direction[i], direction_low[i]}),
        {acc[i], acc_low[i]});
    candidate[i] = next.hi;
  }
  if (!primal_pair_positive(curvature)
      || !isfinite(primal_pair_value(curvature)))
    curvature = {1e-15f, 0.0f};
  return {alpha, cost_delta.hi, cost_delta.lo,
      slope.hi, slope.lo, curvature.hi, curvature.lo};
}

// Mathematical witness kernel used by the opt-in precision regression.  It
// deliberately calls the same compensated Hessian and line-polynomial
// helpers as the production Newton path, on a small dense bilateral system
// whose exact quadratic coefficients have an independent double oracle.
kernel void primal_precision_witness(
    device const float* mass [[buffer(0)]],
    device const float* J [[buffer(1)]],
    device const float* R [[buffer(2)]],
    device const float* aref [[buffer(3)]],
    device const float* q0 [[buffer(4)]],
    device const float* acc [[buffer(5)]],
    device const float* acc_low [[buffer(6)]],
    device const float* direction [[buffer(7)]],
    device const float* direction_low [[buffer(8)]],
    device const float* row_hessian [[buffer(9)]],
    device const int* enabled [[buffer(10)]],
    device const int* elliptic_member [[buffer(11)]],
    device const int* elliptic_start [[buffer(12)]],
    device const int* elliptic_dim [[buffer(13)]],
    device const float* elliptic_friction [[buffer(14)]],
    device const float* mass_delta [[buffer(15)]],
    device const float* mass_delta_low [[buffer(16)]],
    device const float* row_residual [[buffer(17)]],
    device const float* row_residual_low [[buffer(18)]],
    constant int* dims [[buffer(19)]],
    device float* output [[buffer(20)]],
    device float* scratch [[buffer(21)]],
    uint tid [[thread_position_in_grid]]) {
  if (tid != 0) return;
  PrimalFloatPair derivative = {0.0f, 0.0f};
  PrimalFloatPair second_derivative = {0.0f, 0.0f};
  PrimalFloatPair cost = primal_accel_cost_delta_elliptic({0.375f, 0.0f}, mass,
      false, mass, dims, J, R, aref, R, R, q0, acc, acc_low,
      direction, direction_low, enabled, elliptic_member, elliptic_start,
      elliptic_dim, elliptic_friction, 0, scratch, scratch + 2,
      2, 1, 1, 1,
      derivative, second_derivative, mass_delta, mass_delta_low,
      row_residual, row_residual_low);
  device float* product = output + 6;
  device float* product_low = output + 8;
  device const int* island_ids = enabled;
  device const int* awake = enabled;
  primal_hessian_apply_elliptic(mass, false, mass, dims, J, R, aref,
      row_hessian, enabled, elliptic_member, acc, row_residual,
      row_residual_low, elliptic_start, elliptic_dim, elliptic_friction, 0,
      direction, direction_low, product, product_low, awake, island_ids,
      0, false, 2, 1, nullptr);
  output[0] = cost.hi;
  output[1] = cost.lo;
  output[2] = derivative.hi;
  output[3] = derivative.lo;
  output[4] = second_derivative.hi;
  output[5] = second_derivative.lo;
}

// Verify the matrix-free paired H*v path against its native basis columns.
// This is an opt-in numerical witness, separate from the production solver.
kernel void primal_hessian_operator_pair_witness(
    device const float* mass [[buffer(0)]],
    device const float* J [[buffer(1)]],
    device const float* R [[buffer(2)]],
    device const float* aref [[buffer(3)]],
    device const float* row_hessian [[buffer(4)]],
    device const int* enabled [[buffer(5)]],
    device const int* elliptic_member [[buffer(6)]],
    device const float* acc [[buffer(7)]],
    device const float* row_residual [[buffer(8)]],
    device const float* row_residual_low [[buffer(9)]],
    device const int* elliptic_start [[buffer(10)]],
    device const int* elliptic_dim [[buffer(11)]],
    device const float* elliptic_friction [[buffer(12)]],
    constant int* elliptic_count [[buffer(13)]],
    device const float* vector [[buffer(14)]],
    device const float* vector_low [[buffer(15)]],
    device const int* awake_tree [[buffer(16)]],
    device const int* dof_island [[buffer(17)]],
    constant int* dims [[buffer(18)]],
    device float* output [[buffer(19)]],
    device float* scratch [[buffer(20)]],
    uint tid [[thread_position_in_grid]]) {
  if (tid != 0) return;
  int nv = dims[0], n = dims[1];
  device float* basis = scratch;
  device float* basis_low = scratch + nv;
  device float* product = scratch + 2 * nv;
  device float* product_low = scratch + 3 * nv;
  int high_matrix = 0;
  int low_matrix = nv * nv;
  int product_high = 2 * nv * nv;
  int product_low_offset = product_high + nv;
  for (int column = 0; column < nv; ++column) {
    for (int i = 0; i < nv; ++i) basis[i] = basis_low[i] = 0.0f;
    basis[column] = 1.0f;
    primal_hessian_apply_elliptic(mass, false, mass, dims, J, R, aref,
        row_hessian, enabled, elliptic_member, acc, row_residual,
        row_residual_low, elliptic_start, elliptic_dim, elliptic_friction,
        elliptic_count[0], basis, basis_low, product, product_low, awake_tree,
        dof_island, 0, false, nv, n, nullptr);
    for (int row = 0; row < nv; ++row) {
      output[high_matrix + row * nv + column] = product[row];
      output[low_matrix + row * nv + column] = product_low[row];
    }
  }
  primal_hessian_apply_elliptic(mass, false, mass, dims, J, R, aref,
      row_hessian, enabled, elliptic_member, acc, row_residual,
      row_residual_low, elliptic_start, elliptic_dim, elliptic_friction,
      elliptic_count[0], vector, vector_low, product, product_low, awake_tree,
      dof_island, 0, false, nv, n, nullptr);
  for (int i = 0; i < nv; ++i) {
    output[product_high + i] = product[i];
    output[product_low_offset + i] = product_low[i];
  }
}

// Native witness for the exact paired Hager-Zhang beta and update used by
// the production island CG recurrence. The test supplies an island map with
// at least one excluded DOF to exercise the principal solve mask.
kernel void primal_hager_zhang_witness(
    device const float* direction [[buffer(0)]],
    device const float* direction_low [[buffer(1)]],
    device const float* gradient [[buffer(2)]],
    device const float* gradient_low [[buffer(3)]],
    device const float* previous_gradient [[buffer(4)]],
    device const float* previous_gradient_low [[buffer(5)]],
    device const float* mgradient [[buffer(6)]],
    device const float* mgradient_low [[buffer(7)]],
    device const float* old_mgradient [[buffer(8)]],
    device const float* old_mgradient_low [[buffer(9)]],
    device const int* dof_island [[buffer(10)]],
    device float* output [[buffer(11)]],
    uint tid [[thread_position_in_grid]]) {
  if (tid != 0) return;
  constexpr int nv = 5;
  PrimalFloatPair beta = primal_hager_zhang_beta(direction, direction_low,
      gradient, gradient_low, previous_gradient, previous_gradient_low,
      mgradient, mgradient_low, old_mgradient, old_mgradient_low,
      dof_island, 0, true, nv);
  output[0] = beta.hi;
  output[1] = beta.lo;
  for (int i = 0; i < nv; ++i) {
    if (dof_island[i] != 0) {
      output[2 + i] = 0.0f;
      output[7 + i] = 0.0f;
      continue;
    }
    PrimalFloatPair next = primal_hager_zhang_direction(beta,
        {mgradient[i], mgradient_low[i]},
        {direction[i], direction_low[i]});
    output[2 + i] = next.hi;
    output[7 + i] = next.lo;
  }
}

// Pinned MuJoCo CG operates on qacc, preconditions with M^{-1}, uses a
// safeguarded primal line search, and updates directions with Hager-Zhang.
// Elliptic contact groups contribute the source-derived cone objective,
// curvature, and Newton Hessian through the helpers above.
inline int solve_primal_accel_scalar(device const float* mass,
    device const float* debug, device const float* J,
    device const float* R, device const float* aref,
    device const float* lo, device const float* hi,
    device const float* L, device const float* q0,
    device const float* q0_low,
    constant int* dims, device const int* awake_tree,
    device const int* enabled, device const int* elliptic_member,
    device const int* elliptic_start, device const int* elliptic_dim,
    device const float* elliptic_friction, int elliptic_count,
    device float* lambda,
    int nv, int nr, int n, int n_eq_rows, int solver_type, int maxiter,
    int ls_iterations, float ls_tolerance, float tolerance,
    float scale, bool warmstart_enabled, device float* scratch,
    device float* pair_tail, device float* residual_out,
    device const int* dof_island,
    int island, bool partitioned) {
  bool sparse_mass = dims[20] == -3;
  device float* acc = scratch;
  device float* gradient = scratch + nv;
  device float* old_gradient = scratch + 2 * nv;
  device float* previous_gradient = scratch + 3 * nv;
  device float* mgradient = scratch + 4 * nv;
  device float* old_mgradient = scratch + 5 * nv;
  device float* direction = scratch + 6 * nv;
  device float* candidate = scratch + 7 * nv;
  device float* mass_scratch = scratch + 8 * nv;
  device float* generalized_force = scratch + 9 * nv;
  device float* rhs_dof = scratch + 10 * nv;
  device float* solved = scratch + 11 * nv;
  device float* force = scratch + 12 * nv;
  device float* candidate_force = force + n;
  device float* row_hessian = candidate_force + n;
  device float* candidate_hessian = row_hessian + n;
  device float* residual = candidate_hessian + n;
  device float* hessian = residual + n;
  int solver_metadata = dims[24] + dims[9] * 10;
  device float* high_low = pair_tail;
  device float* acc_low = high_low;
  device float* gradient_low = acc_low + nv;
  device float* direction_low = gradient_low + nv;
  // The CG outer recurrence needs both words of the previous raw gradient
  // and previous preconditioned gradient. These vectors use otherwise idle
  // high/low workspace slices; Newton's PCG owns them only during its separate
  // Hessian solve, and CG never dispatches that solver.
  device float* previous_gradient_low = high_low + 3 * nv;
  device float* mgradient_low = high_low + 4 * nv;
  device float* old_mgradient_low = high_low + 5 * nv;
  // This slice aliases PCG's product-low vector only across disjoint phases;
  // the line search needs it after each Hessian solve has completed.
  device float* mass_delta_low = high_low + 6 * nv;
  device float* row_residual_low = high_low + 8 * max(nv, 1);
    device float* row_force_low = row_residual_low + nr;
    device const float* qacc_warmstart = debug + dims[solver_metadata + 5];
  bool trace_primal_detail = (dims[7] & 64) != 0;
  bool trace_second_primal_point = (dims[7] & 128) != 0;
  bool trace_primal_gradient_components = (dims[7] & 256) != 0;
  float warm_cost_delta = 0.0f;
  for (int i = 0; i < nv; ++i) {
    acc[i] = q0[i];
    acc_low[i] = q0_low[i];
    gradient_low[i] = direction_low[i] = 0.0f;
    gradient[i] = old_gradient[i] = previous_gradient[i] = 0.0f;
    mgradient[i] = old_mgradient[i] = 0.0f;
    if (solver_type == 1) {
      previous_gradient_low[i] = 0.0f;
      mgradient_low[i] = old_mgradient_low[i] = 0.0f;
    }
    direction[i] = candidate[i] = mass_scratch[i] = 0.0f;
    generalized_force[i] = rhs_dof[i] = solved[i] = 0.0f;
  }
  for (int row = 0; row < nr; ++row)
    row_residual_low[row] = row_force_low[row] = 0.0f;
  // Pinned warmstart selection compares qacc_warmstart against the cold
  // qacc_smooth point. Retained row multipliers are not an independent input
  // to CG/Newton and must not perturb that cost comparison.
  float cold_cost = primal_accel_eval_elliptic(mass, L, sparse_mass,
      debug, dims, J, R, aref, lo, hi,
      q0, acc, enabled, elliptic_member, elliptic_start, elliptic_dim,
      elliptic_friction, elliptic_count, force, gradient, row_hessian,
      residual, generalized_force, nv, nr, n, n_eq_rows,
      nullptr, nullptr, row_residual_low, false);
  float cold_start_cost = cold_cost;
  for (int i = 0; i < nv; ++i)
    if (partitioned && dof_island[i] != island) gradient[i] = 0.0f;
  // Compare the full acceleration-space constraint + Gauss cost against the
  // cold qacc_smooth point.
  if (warmstart_enabled) {
    for (int i = 0; i < nv; ++i)
      candidate[i] = solver_dof_awake(dims, awake_tree, i)
          && (!partitioned || dof_island[i] == island)
          ? qacc_warmstart[i] : q0[i];
    float warm_cost = primal_accel_eval_elliptic(mass, L, sparse_mass,
        debug, dims, J, R, aref, lo, hi,
        q0, candidate, enabled, elliptic_member, elliptic_start, elliptic_dim,
        elliptic_friction, elliptic_count, candidate_force, old_gradient,
        candidate_hessian, residual, generalized_force, nv, nr, n, n_eq_rows,
        nullptr, nullptr, row_residual_low, false);
    if (isfinite(warm_cost)) warm_cost_delta = warm_cost - cold_start_cost;
    if (isfinite(warm_cost) && warm_cost <= cold_cost) {
      for (int i = 0; i < nv; ++i) {
        acc[i] = candidate[i];
        acc_low[i] = 0.0f;
      }
      cold_cost = warm_cost;
    }
  }
  float cost = primal_accel_eval_elliptic(mass, L, sparse_mass,
      debug, dims, J, R, aref, lo, hi,
      q0, acc, enabled, elliptic_member, elliptic_start, elliptic_dim,
      elliptic_friction, elliptic_count, force, gradient, row_hessian,
      residual, generalized_force, nv, nr, n, n_eq_rows,
      nullptr, nullptr, row_residual_low, false);
  primal_refresh_pair_state(mass, sparse_mass, debug, dims, J, R, aref,
      lo, hi, q0, q0_low, acc, acc_low, enabled, elliptic_member,
      elliptic_start, elliptic_dim, elliptic_friction, elliptic_count,
      force, row_force_low, gradient, gradient_low, row_hessian,
      residual, row_residual_low, nv, nr, n, n_eq_rows);
  primal_mass_difference(mass, sparse_mass, debug, dims, acc, q0,
      solved, old_gradient, nv);
  for (int i = 0; i < nv; ++i)
    if (partitioned && dof_island[i] != island)
      gradient[i] = gradient_low[i] = 0.0f;
  // A partitioned solve reuses the canonical lambda backing for each island.
  // Preserve forces already solved for earlier islands; disabled rows are
  // zeroed once by the caller before the island loop.
  for (int row = 0; row < n; ++row)
    if (enabled[row]) lambda[row] = force[row];
  if (trace_primal_detail && island == 0) {
    // Preserve the primary call's input context before its initial Newton H
    // solve can return early. History[2:7] is
    // [scale,tolerance,cold cost,warm-candidate cost delta,initial scaled
    // gradient norm]; history[7:10] is the initial H-solve witness until an
    // accepted point replaces those last three words with its legacy detail.
    float initial_grad_norm_sq = primal_pair_dot(gradient, gradient_low,
        gradient, gradient_low, nv);
    residual_out[2] = scale;
    residual_out[3] = tolerance;
    residual_out[4] = cold_start_cost;
    residual_out[5] = warm_cost_delta;
    residual_out[6] = scale * sqrt(max(0.0f, initial_grad_norm_sq));
    residual_out[7] = residual_out[8] = residual_out[9] = 0.0f;
  }
  if (solver_type == 2) {
    if (!primal_hessian_solve_elliptic(mass, L, sparse_mass, debug, high_low,
        J, R, aref, row_hessian,
        dims, awake_tree, enabled, elliptic_member, acc, residual,
        row_residual_low,
        elliptic_start, elliptic_dim,
        elliptic_friction, elliptic_count, gradient, gradient_low,
        mgradient, hessian,
        mass_scratch, rhs_dof, solved, generalized_force,
        dof_island, island, partitioned, nv, n, tolerance, scale,
        trace_primal_detail && island == 0 ? residual_out : nullptr)) return -1;
    device float* pcg_solution_low = high_low + 4 * nv;
    for (int i = 0; i < nv; ++i)
      direction_low[i] = -pcg_solution_low[i];
  } else {
    if (solver_type == 1) {
      // The first Hager-Zhang direction is M^-1 applied to the entire
      // represented gradient. Solving its high and low words independently
      // leaves the low word without the residual of M*y=g_hi+g_lo and can
      // destroy conjugacy before the first accepted line point.
      if (!primal_cg_mass_precondition_pair_elliptic(
          mass, L, sparse_mass, debug, dims, awake_tree,
          gradient, gradient_low, mgradient, mgradient_low,
          mass_scratch, candidate, rhs_dof, solved, generalized_force,
          old_gradient, dof_island, island, partitioned, nv)) return -1;
    } else if (partitioned) {
      if (!primal_mass_solve_island(mass, debug, dims, gradient, mgradient,
          rhs_dof, solved, generalized_force, mass_scratch,
          awake_tree, dof_island, island, true, nv)) return -1;
    } else if (sparse_mass) {
      if (!solver_component_mass_solve(debug, dims, awake_tree, nv, gradient,
          mgradient, mass_scratch)) return 0;
    } else {
      primal_mass_solve(L, gradient, mgradient, mass_scratch, nv);
    }
  }
  for (int i = 0; i < nv; ++i) {
    if (partitioned && dof_island[i] != island) {
      gradient[i] = mgradient[i] = 0.0f;
      if (solver_type == 1) gradient_low[i] = mgradient_low[i] = 0.0f;
    }
    direction[i] = -mgradient[i];
    if (solver_type == 1) direction_low[i] = -mgradient_low[i];
  }
  // PCG temporarily uses the mass-delta low-word slice for H*direction.
  // Restore the source Gauss residual after its return before evaluating the
  // line polynomial; otherwise the line search consumes the final Hessian
  // product's low word as M*(qacc-q0)'s low word.
  primal_mass_difference_pair(mass, sparse_mass, debug, dims, q0, q0_low, acc,
      acc_low, old_gradient, mass_delta_low, nv);
  int used = 0;
  bool solve_failed = false;
  float grad_norm_sq = primal_pair_dot(gradient, gradient_low,
      gradient, gradient_low, nv);
  float last_residual = scale * sqrt(max(0.0f, grad_norm_sq));
  for (int iter = 0; iter < maxiter; ++iter) {
    primal_mass_difference_pair(mass, sparse_mass, debug, dims, q0, q0_low, acc,
        acc_low, old_gradient, mass_delta_low, nv);
    for (int i = 0; i < nv; ++i) {
      previous_gradient[i] = gradient[i];
      if (solver_type == 1) previous_gradient_low[i] = gradient_low[i];
    }
    PrimalLinePoint p0 = primal_line_eval_elliptic({0.0f, 0.0f}, mass,
        sparse_mass, debug, dims, J, R, aref, lo, hi, q0, acc, acc_low,
        direction, direction_low, enabled, elliptic_member, elliptic_start,
        elliptic_dim, elliptic_friction, elliptic_count, candidate,
        rhs_dof, solved, nv, nr, n, n_eq_rows, old_gradient,
        mass_delta_low, residual, row_residual_low);
    PrimalFloatPair slope0 = {p0.slope, p0.slope_low};
    PrimalFloatPair curvature_pair = {p0.curvature, p0.curvature_low};
    float slope0_value = primal_pair_value(slope0);
    float curvature = primal_pair_value(curvature_pair);
    float direction_norm_sq = primal_pair_dot(direction, direction_low,
        direction, direction_low, nv);
    if (!primal_pair_negative(slope0) || !(curvature > 1e-20f)
        || !isfinite(slope0_value) || !isfinite(curvature)) break;
    float gtol = tolerance * ls_tolerance * sqrt(direction_norm_sq)
        / max(scale, 1e-15f);
    // PrimalSearch counts its alpha=0 evaluation against LSiter before it
    // unconditionally probes the first Newton point.
    int ls_count = 1;
    p0.alpha = {0.0f, 0.0f};
    PrimalLinePoint p1 = primal_line_eval_elliptic(
        primal_pair_add(p0.alpha, primal_pair_multiply(
            primal_pair_divide({p0.slope, p0.slope_low},
                {p0.curvature, p0.curvature_low}), -1.0f)), mass, sparse_mass,
        debug, dims, J, R, aref, lo, hi,
        q0, acc, acc_low, direction, direction_low, enabled, elliptic_member, elliptic_start,
        elliptic_dim, elliptic_friction, elliptic_count, candidate,
        rhs_dof, solved, nv, nr, n, n_eq_rows, old_gradient, mass_delta_low, residual, row_residual_low);
    ++ls_count;
    PrimalFloatPair accepted_alpha = {0.0f, 0.0f};
    PrimalFloatPair accepted_delta_cost = {0.0f, 0.0f};
    bool line_done = false;
    if (primal_pair_less_float(primal_pair_abs(
            {p1.slope, p1.slope_low}), gtol)) {
      accepted_alpha = p1.alpha;
      accepted_delta_cost = {p1.cost, p1.cost_low};
      line_done = true;
    }
    int search_direction = primal_pair_negative(
        {p1.slope, p1.slope_low}) ? 1 : -1;
    PrimalLinePoint p2 = p0;
    while (!line_done && (search_direction > 0
            ? primal_pair_less_equal_float({p1.slope, p1.slope_low}, -gtol)
            : primal_pair_less_equal_float({-p1.slope, -p1.slope_low}, -gtol))
        && ls_count < ls_iterations) {
      p2 = p1;
      p1 = primal_line_eval_elliptic(
          primal_pair_add(p1.alpha, primal_pair_multiply(
              primal_pair_divide({p1.slope, p1.slope_low},
                  {p1.curvature, p1.curvature_low}), -1.0f)), mass, sparse_mass,
          debug, dims, J, R, aref,
          lo, hi, q0, acc, acc_low, direction, direction_low, enabled, elliptic_member,
          elliptic_start, elliptic_dim, elliptic_friction, elliptic_count,
          candidate, rhs_dof, solved, nv, nr, n, n_eq_rows,
          old_gradient, mass_delta_low, residual, row_residual_low);
      ++ls_count;
      if (primal_pair_less_float(primal_pair_abs(
              {p1.slope, p1.slope_low}), gtol)) {
        accepted_alpha = p1.alpha;
        accepted_delta_cost = {p1.cost, p1.cost_low};
        line_done = true;
      }
    }
    if (!line_done && ls_count < ls_iterations) {
      PrimalLinePoint p2next = p1;
      PrimalLinePoint p1next = primal_line_eval_elliptic(
          primal_pair_add(p1.alpha, primal_pair_multiply(
              primal_pair_divide({p1.slope, p1.slope_low},
                  {p1.curvature, p1.curvature_low}), -1.0f)), mass, sparse_mass,
          debug, dims, J, R, aref,
          lo, hi, q0, acc, acc_low, direction, direction_low, enabled, elliptic_member,
          elliptic_start, elliptic_dim, elliptic_friction, elliptic_count,
          candidate, rhs_dof, solved, nv, nr, n, n_eq_rows,
          old_gradient, mass_delta_low, residual, row_residual_low);
      ++ls_count;
      while (ls_count < ls_iterations) {
        PrimalLinePoint pmid = primal_line_eval_elliptic(
            primal_pair_multiply(primal_pair_add(p1.alpha, p2.alpha), 0.5f),
            mass, sparse_mass,
            debug, dims, J, R, aref, lo, hi,
            q0, acc, acc_low, direction, direction_low, enabled, elliptic_member, elliptic_start,
            elliptic_dim, elliptic_friction, elliptic_count, candidate,
            rhs_dof, solved, nv, nr, n, n_eq_rows,
            old_gradient, mass_delta_low, residual, row_residual_low);
        ++ls_count;
        PrimalLinePoint candidates[3] = {p1next, p2next, pmid};
        int best = -1;
        for (int k = 0; k < 3; ++k) {
          if (primal_pair_less_float(primal_pair_abs(
                  {candidates[k].slope, candidates[k].slope_low}), gtol)
              && (best < 0 || primal_pair_less(
                  {candidates[k].cost, candidates[k].cost_low},
                  {candidates[best].cost, candidates[best].cost_low}))) best = k;
        }
        if (best >= 0) {
          accepted_alpha = candidates[best].alpha;
          accepted_delta_cost = {candidates[best].cost,
              candidates[best].cost_low};
          line_done = true;
          break;
        }
        int b1 = 0, b2 = 0;
        for (int k = 0; k < 3; ++k) {
          PrimalFloatPair p1s = {p1.slope, p1.slope_low};
          PrimalFloatPair cands = {candidates[k].slope,
              candidates[k].slope_low};
          if (primal_pair_negative(p1s) && primal_pair_negative(cands)
              && primal_pair_less(p1s, cands)) {
            p1 = candidates[k];
            b1 = 1;
          } else if (!primal_pair_negative(p1s) && !primal_pair_negative(cands)
              && primal_pair_less(cands, p1s)) {
            p1 = candidates[k];
            b1 = 1;
          }
        }
        if (b1) {
          p1next = primal_line_eval_elliptic(
              primal_pair_add(p1.alpha, primal_pair_multiply(
                  primal_pair_divide({p1.slope, p1.slope_low},
                      {p1.curvature, p1.curvature_low}), -1.0f)),
              mass, sparse_mass,
              debug, dims, J, R, aref,
              lo, hi, q0, acc, acc_low, direction, direction_low, enabled, elliptic_member,
              elliptic_start, elliptic_dim, elliptic_friction, elliptic_count,
              candidate, rhs_dof, solved, nv, nr, n, n_eq_rows,
              old_gradient, mass_delta_low, residual, row_residual_low);
          ++ls_count;
        }
        for (int k = 0; k < 3; ++k) {
          PrimalFloatPair p2s = {p2.slope, p2.slope_low};
          PrimalFloatPair cands = {candidates[k].slope,
              candidates[k].slope_low};
          if (primal_pair_negative(p2s) && primal_pair_negative(cands)
              && primal_pair_less(p2s, cands)) {
            p2 = candidates[k];
            b2 = 1;
          } else if (!primal_pair_negative(p2s) && !primal_pair_negative(cands)
              && primal_pair_less(cands, p2s)) {
            p2 = candidates[k];
            b2 = 1;
          }
        }
        if (b2) {
          p2next = primal_line_eval_elliptic(
              primal_pair_add(p2.alpha, primal_pair_multiply(
                  primal_pair_divide({p2.slope, p2.slope_low},
                      {p2.curvature, p2.curvature_low}), -1.0f)),
              mass, sparse_mass,
              debug, dims, J, R, aref,
              lo, hi, q0, acc, acc_low, direction, direction_low, enabled, elliptic_member,
              elliptic_start, elliptic_dim, elliptic_friction, elliptic_count,
              candidate, rhs_dof, solved, nv, nr, n, n_eq_rows,
              old_gradient, mass_delta_low, residual, row_residual_low);
          ++ls_count;
        }
        if (!b1 && !b2) {
          // MuJoCo 3.10 returns the midpoint even when its line-search
          // objective did not improve; PrimalSearch records that condition
          // separately, while the outer iteration still applies the point.
          accepted_alpha = pmid.alpha;
          accepted_delta_cost = {pmid.cost, pmid.cost_low};
          line_done = true;
        }
      }
      if (!line_done) {
        if (primal_pair_less_equal({p1.cost, p1.cost_low},
                {p2.cost, p2.cost_low})
            && primal_pair_negative({p1.cost, p1.cost_low})) {
          accepted_alpha = p1.alpha;
          accepted_delta_cost = {p1.cost, p1.cost_low};
        } else if (primal_pair_less_equal({p2.cost, p2.cost_low},
                {p1.cost, p1.cost_low})
            && primal_pair_negative({p2.cost, p2.cost_low})) {
          accepted_alpha = p2.alpha;
          accepted_delta_cost = {p2.cost, p2.cost_low};
        }
        line_done = true;
      }
    } else if (!line_done) {
      // Source PrimalSearch returns the last one-sided Newton point when it
      // exhausts its line-search budget before it can bracket a minimum.
      accepted_alpha = p1.alpha;
      accepted_delta_cost = {p1.cost, p1.cost_low};
      line_done = true;
    }
    if (!line_done || (!primal_pair_positive(accepted_alpha)
        && !primal_pair_negative(accepted_alpha))
        || !primal_pair_is_finite(accepted_alpha)) break;
    // Match MuJoCo 3.10's accepted-state update: retain Ma-M*q0 and J*qacc-
    // aref as independent accumulated quantities instead of rebuilding them
    // from rounded qacc on the next iteration.
    for (int row = 0; row < n; ++row) {
      float row_direction = 0.0f, row_direction_correction = 0.0f;
      for (int dof = 0; dof < nv; ++dof) {
        primal_compensated_product_add(ccj_get(J, row, dof, nv), direction[dof],
            row_direction, row_direction_correction);
        primal_compensated_product_add(ccj_get(J, row, dof, nv),
            direction_low[dof], row_direction, row_direction_correction);
      }
      PrimalFloatPair updated = primal_pair_fma_pair(accepted_alpha,
          primal_pair_renormalize(row_direction, row_direction_correction),
          {residual[row], row_residual_low[row]});
      residual[row] = updated.hi;
      row_residual_low[row] = updated.lo;
    }
    primal_mass_product_pair(mass, sparse_mass, debug, dims, direction,
        direction_low, rhs_dof, candidate, nv);
    for (int i = 0; i < nv; ++i) {
      PrimalFloatPair updated_mass_delta = primal_pair_fma_pair(accepted_alpha,
          {rhs_dof[i], candidate[i]},
          {old_gradient[i], mass_delta_low[i]});
      old_gradient[i] = updated_mass_delta.hi;
      mass_delta_low[i] = updated_mass_delta.lo;
      if (!partitioned || dof_island[i] == island) {
        PrimalFloatPair updated = primal_pair_fma_pair(accepted_alpha,
            {direction[i], direction_low[i]}, {acc[i], acc_low[i]});
        acc[i] = updated.hi;
        acc_low[i] = updated.lo;
      }
      old_mgradient[i] = mgradient[i];
      if (solver_type == 1) old_mgradient_low[i] = mgradient_low[i];
    }
    // Preserve the accepted row residual in its own row-sized vector while
    // evaluation rewrites `residual` with the next force/Hessian state.
    // Passing residual as both the incremental input and mutable output makes
    // the scalar evaluator clear each input row before reading it.
    for (int row = 0; row < n; ++row)
      candidate_hessian[row] = residual[row];
    float accepted_cost = primal_accel_eval_elliptic(mass, L, sparse_mass,
        debug, dims, J, R, aref,
        lo, hi, q0, acc, enabled, elliptic_member, elliptic_start,
        elliptic_dim, elliptic_friction, elliptic_count, candidate_force,
        gradient, row_hessian, residual, generalized_force, nv, nr, n,
        n_eq_rows, old_gradient, candidate_hessian, row_residual_low, true);
    primal_refresh_pair_state(mass, sparse_mass, debug, dims, J, R, aref,
        lo, hi, q0, q0_low, acc, acc_low, enabled, elliptic_member,
        elliptic_start, elliptic_dim, elliptic_friction, elliptic_count,
        candidate_force, row_force_low, gradient, gradient_low, row_hessian,
        residual, row_residual_low, nv, nr, n, n_eq_rows);
    for (int i = 0; i < nv; ++i)
      if (partitioned && dof_island[i] != island)
        gradient[i] = gradient_low[i] = 0.0f;
    PrimalFloatPair improvement = primal_pair_multiply(
        accepted_delta_cost, -1.0f);
    cost = accepted_cost;
    for (int row = 0; row < n; ++row)
      if (enabled[row]) lambda[row] = candidate_force[row];
    // MuJoCo 3.10 checks improvement and the refreshed raw gradient before
    // preparing the next search direction. In particular, do not make an
    // otherwise converged accepted point fail because the next Newton
    // Hessian solve cannot certify a direction that the outer loop will not
    // use. Count the accepted line-search iteration before this check, as the
    // pinned driver does.
    ++used;
    grad_norm_sq = primal_pair_dot(gradient, gradient_low,
        gradient, gradient_low, nv);
    last_residual = scale * sqrt(max(0.0f, grad_norm_sq));
    int detail_iteration = trace_second_primal_point ? 2 : 1;
    if (trace_primal_detail && island == 0 && used == detail_iteration) {
      if (trace_primal_gradient_components) {
        // The mixed contact fixture has two generalized coordinates. Keep
        // the first three high and low residual-gradient components in the
        // existing eight history words so the CPU/GPU comparison can locate
        // cancellation before the norm reduction.
        for (int i = 0; i < 3; ++i) {
          residual_out[2 + i] = i < nv ? gradient[i] : 0.0f;
          residual_out[5 + i] = i < nv ? gradient_low[i] : 0.0f;
        }
      } else {
        residual_out[7] = accepted_alpha.hi;
        residual_out[8] = scale * primal_pair_value(improvement);
        residual_out[9] = last_residual;
      }
    }
    if ((primal_pair_positive(improvement)
            && scale * primal_pair_value(improvement) < tolerance)
        || last_residual < tolerance) break;
    if (solver_type == 2) {
      if (!primal_hessian_solve_elliptic(mass, L, sparse_mass, debug, high_low,
          J, R, aref, row_hessian,
          dims, awake_tree, enabled, elliptic_member, acc, residual,
          row_residual_low,
          elliptic_start, elliptic_dim,
          elliptic_friction, elliptic_count, gradient, gradient_low,
          mgradient, hessian,
          mass_scratch, rhs_dof, solved, generalized_force,
          dof_island, island, partitioned, nv, n, tolerance, scale,
          nullptr)) {
        solve_failed = true;
        break;
      }
      device float* pcg_solution_low = high_low + 4 * nv;
      for (int i = 0; i < nv; ++i)
        direction_low[i] = -pcg_solution_low[i];
    } else {
      if (!primal_cg_mass_precondition_pair_elliptic(
          mass, L, sparse_mass, debug, dims, awake_tree,
          gradient, gradient_low, mgradient, mgradient_low,
          mass_scratch, candidate, rhs_dof, solved, generalized_force,
          old_gradient, dof_island, island, partitioned, nv)) {
        solve_failed = true;
        break;
      }
    }
    for (int i = 0; i < nv; ++i)
      if (partitioned && dof_island[i] != island) {
        gradient_low[i] = 0.0f;
        mgradient[i] = mgradient_low[i] = 0.0f;
        gradient[i] = 0.0f;
      }
    if (solver_type == 2) {
      for (int i = 0; i < nv; ++i) {
        if (partitioned && dof_island[i] != island) {
          gradient[i] = mgradient[i] = 0.0f;
        }
        direction[i] = -mgradient[i];
      }
      continue;
    }
    // Keep the Hager-Zhang recurrence paired through the directional update.
    PrimalFloatPair beta = primal_hager_zhang_beta(direction, direction_low,
        gradient, gradient_low, previous_gradient, previous_gradient_low,
        mgradient, mgradient_low, old_mgradient, old_mgradient_low,
        dof_island, island, partitioned, nv);
    for (int i = 0; i < nv; ++i) {
      if (partitioned && dof_island[i] != island) {
        direction[i] = direction_low[i] = 0.0f;
        continue;
      }
      PrimalFloatPair next_direction = primal_hager_zhang_direction(beta,
          {mgradient[i], mgradient_low[i]},
          {direction[i], direction_low[i]});
      direction[i] = next_direction.hi;
      direction_low[i] = next_direction.lo;
    }
  }
  if (solve_failed) return -1;
  residual_out[0] = last_residual;
  return used;
}

inline void seed_pgs_warmstart_qacc(device const float* J, int nr,
    device const float* qacc_warmstart, thread const float* R,
    thread const float* aref, thread const float* lo, thread const float* hi,
    device const int* enabled, thread float* lambda, int nv, int n,
    int n_eq_rows, int elliptic_count, device const int* elliptic_start,
    device const int* elliptic_dim, device const float* elliptic_friction) {
  for (int row = 0; row < n; ++row) {
    lambda[row] = 0.0f;
    if (!enabled[row]) continue;
    float jar = -aref[row];
    for (int dof = 0; dof < nv; ++dof)
      jar += ccj_get(J, row, dof, nv) * qacc_warmstart[dof];
    float D = 1.0f / max(R[row], 1e-15f);
    bool dry_friction = lo[row] < 0.0f && hi[row] > 0.0f
        && isfinite(lo[row]) && isfinite(hi[row])
        && abs(lo[row] + hi[row]) <= 1e-5f * max(1.0f, abs(hi[row]));
    if (row < n_eq_rows) {
      lambda[row] = -D * jar;
    } else if (dry_friction) {
      float loss = hi[row];
      float threshold = R[row] * loss;
      lambda[row] = jar <= -threshold ? loss
                  : (jar >= threshold ? -loss : -D * jar);
    } else if (jar < 0.0f) {
      lambda[row] = -D * jar;
    }
  }
  for (int block = 0; block < elliptic_count; ++block) {
    int start = elliptic_start[block];
    int dim = elliptic_dim[block];
    // mj_makeConstraint sets contact.mu from the normal/tangent R ratio;
    // mj_constraintUpdate uses that regularized coefficient for warmstarts.
    // Physical tangential friction below remains unchanged.
    float mu = max(elliptic_friction[block * 5], 0.0f)
        * sqrt(max(R[start + 1], 1e-15f) / max(R[start], 1e-15f));
    if (!(mu > 1e-12f)) {
      for (int j = 0; j < dim; ++j) lambda[start + j] = 0.0f;
      continue;
    }
    thread float jar[6], U[6];
    for (int j = 0; j < 6; ++j) jar[j] = U[j] = 0.0f;
    for (int j = 0; j < dim; ++j) {
      jar[j] = -aref[start + j];
      for (int dof = 0; dof < nv; ++dof)
        jar[j] += ccj_get(J, (start + j), dof, nv) * qacc_warmstart[dof];
    }
    U[0] = jar[0] * mu;
    float tangent_sq = 0.0f;
    for (int j = 1; j < dim; ++j) {
      float friction = max(elliptic_friction[block * 5 + j - 1], 0.0f);
      U[j] = jar[j] * friction;
      tangent_sq += U[j] * U[j];
    }
    float N = U[0], T = sqrt(max(tangent_sq, 0.0f));
    if (N >= mu * T || (T <= 0.0f && N >= 0.0f)) {
      for (int j = 0; j < dim; ++j) lambda[start + j] = 0.0f;
    } else if (mu * N + T <= 0.0f || (T <= 0.0f && N < 0.0f)) {
      for (int j = 0; j < dim; ++j)
        lambda[start + j] = -jar[j] / max(R[start + j], 1e-15f);
    } else {
      float Dm = (1.0f / max(R[start], 1e-15f))
          / (mu * mu * (1.0f + mu * mu));
      float NmT = N - mu * T;
      lambda[start] = -Dm * NmT * mu;
      for (int j = 1; j < dim; ++j) {
        float friction = max(elliptic_friction[block * 5 + j - 1], 0.0f);
        lambda[start + j] = T > 1e-15f
            ? -lambda[start] / T * U[j] * friction : 0.0f;
      }
    }
  }
}

// Block solver keeps assembled row data in persistent device workspace rather
// than copying it into thread-local arrays. Preserve that address space in a
// matching overload so larger models use the same pinned warmstart mapping.
inline void seed_pgs_warmstart_qacc(device const float* J, int nr,
    device const float* qacc_warmstart, device const float* R,
    device const float* aref, device const float* lo, device const float* hi,
    device const int* enabled, thread float* lambda, int nv, int n,
    int n_eq_rows, int elliptic_count, device const int* elliptic_start,
    device const int* elliptic_dim, device const float* elliptic_friction) {
  for (int row = 0; row < n; ++row) {
    lambda[row] = 0.0f;
    if (!enabled[row]) continue;
    float jar = -aref[row];
    for (int dof = 0; dof < nv; ++dof)
      jar += ccj_get(J, row, dof, nv) * qacc_warmstart[dof];
    float D = 1.0f / max(R[row], 1e-15f);
    bool dry_friction = lo[row] < 0.0f && hi[row] > 0.0f
        && isfinite(lo[row]) && isfinite(hi[row])
        && abs(lo[row] + hi[row]) <= 1e-5f * max(1.0f, abs(hi[row]));
    if (row < n_eq_rows) {
      lambda[row] = -D * jar;
    } else if (dry_friction) {
      float loss = hi[row];
      float threshold = R[row] * loss;
      lambda[row] = jar <= -threshold ? loss
                  : (jar >= threshold ? -loss : -D * jar);
    } else if (jar < 0.0f) {
      lambda[row] = -D * jar;
    }
  }
  for (int block = 0; block < elliptic_count; ++block) {
    int start = elliptic_start[block];
    int dim = elliptic_dim[block];
    // mj_makeConstraint sets contact.mu from the normal/tangent R ratio;
    // mj_constraintUpdate uses that regularized coefficient for warmstarts.
    // Physical tangential friction below remains unchanged.
    float mu = max(elliptic_friction[block * 5], 0.0f)
        * sqrt(max(R[start + 1], 1e-15f) / max(R[start], 1e-15f));
    if (!(mu > 1e-12f)) {
      for (int j = 0; j < dim; ++j) lambda[start + j] = 0.0f;
      continue;
    }
    thread float jar[6], U[6];
    for (int j = 0; j < 6; ++j) jar[j] = U[j] = 0.0f;
    for (int j = 0; j < dim; ++j) {
      jar[j] = -aref[start + j];
      for (int dof = 0; dof < nv; ++dof)
        jar[j] += ccj_get(J, (start + j), dof, nv) * qacc_warmstart[dof];
    }
    U[0] = jar[0] * mu;
    float tangent_sq = 0.0f;
    for (int j = 1; j < dim; ++j) {
      float friction = max(elliptic_friction[block * 5 + j - 1], 0.0f);
      U[j] = jar[j] * friction;
      tangent_sq += U[j] * U[j];
    }
    float N = U[0], T = sqrt(max(tangent_sq, 0.0f));
    if (N >= mu * T || (T <= 0.0f && N >= 0.0f)) {
      for (int j = 0; j < dim; ++j) lambda[start + j] = 0.0f;
    } else if (mu * N + T <= 0.0f || (T <= 0.0f && N < 0.0f)) {
      for (int j = 0; j < dim; ++j)
        lambda[start + j] = -jar[j] / max(R[start + j], 1e-15f);
    } else {
      float Dm = (1.0f / max(R[start], 1e-15f))
          / (mu * mu * (1.0f + mu * mu));
      float NmT = N - mu * T;
      lambda[start] = -Dm * NmT * mu;
      for (int j = 1; j < dim; ++j) {
        float friction = max(elliptic_friction[block * 5 + j - 1], 0.0f);
        lambda[start + j] = T > 1e-15f
            ? -lambda[start] / T * U[j] * friction : 0.0f;
      }
    }
  }
}

inline void seed_pgs_warmstart_qacc(device const float* J, int nr,
    device const float* qacc_warmstart, device const float* R,
    device const float* aref, device const float* lo, device const float* hi,
    device const int* enabled, device float* lambda, int nv, int n,
    int n_eq_rows, int elliptic_count, device const int* elliptic_start,
    device const int* elliptic_dim, device const float* elliptic_friction) {
  for (int row = 0; row < n; ++row) {
    lambda[row] = 0.0f;
    if (!enabled[row]) continue;
    float jar = -aref[row];
    for (int dof = 0; dof < nv; ++dof)
      jar += ccj_get(J, row, dof, nv) * qacc_warmstart[dof];
    float D = 1.0f / max(R[row], 1e-15f);
    bool dry_friction = lo[row] < 0.0f && hi[row] > 0.0f
        && isfinite(lo[row]) && isfinite(hi[row])
        && abs(lo[row] + hi[row]) <= 1e-5f * max(1.0f, abs(hi[row]));
    if (row < n_eq_rows) {
      lambda[row] = -D * jar;
    } else if (dry_friction) {
      float loss = hi[row];
      float threshold = R[row] * loss;
      lambda[row] = jar <= -threshold ? loss
                  : (jar >= threshold ? -loss : -D * jar);
    } else if (jar < 0.0f) {
      lambda[row] = -D * jar;
    }
  }
  for (int block = 0; block < elliptic_count; ++block) {
    int start = elliptic_start[block];
    int dim = elliptic_dim[block];
    // mj_makeConstraint sets contact.mu from the normal/tangent R ratio;
    // mj_constraintUpdate uses that regularized coefficient for warmstarts.
    // Physical tangential friction below remains unchanged.
    float mu = max(elliptic_friction[block * 5], 0.0f)
        * sqrt(max(R[start + 1], 1e-15f) / max(R[start], 1e-15f));
    if (!(mu > 1e-12f)) {
      for (int j = 0; j < dim; ++j) lambda[start + j] = 0.0f;
      continue;
    }
    thread float jar[6], U[6];
    for (int j = 0; j < 6; ++j) jar[j] = U[j] = 0.0f;
    for (int j = 0; j < dim; ++j) {
      jar[j] = -aref[start + j];
      for (int dof = 0; dof < nv; ++dof)
        jar[j] += ccj_get(J, (start + j), dof, nv) * qacc_warmstart[dof];
    }
    U[0] = jar[0] * mu;
    float tangent_sq = 0.0f;
    for (int j = 1; j < dim; ++j) {
      float friction = max(elliptic_friction[block * 5 + j - 1], 0.0f);
      U[j] = jar[j] * friction;
      tangent_sq += U[j] * U[j];
    }
    float N = U[0], T = sqrt(max(tangent_sq, 0.0f));
    if (N >= mu * T || (T <= 0.0f && N >= 0.0f)) {
      for (int j = 0; j < dim; ++j) lambda[start + j] = 0.0f;
    } else if (mu * N + T <= 0.0f || (T <= 0.0f && N < 0.0f)) {
      for (int j = 0; j < dim; ++j)
        lambda[start + j] = -jar[j] / max(R[start + j], 1e-15f);
    } else {
      float Dm = (1.0f / max(R[start], 1e-15f))
          / (mu * mu * (1.0f + mu * mu));
      float NmT = N - mu * T;
      lambda[start] = -Dm * NmT * mu;
      for (int j = 1; j < dim; ++j) {
        float friction = max(elliptic_friction[block * 5 + j - 1], 0.0f);
        lambda[start + j] = T > 1e-15f
            ? -lambda[start] / T * U[j] * friction : 0.0f;
      }
    }
  }
}

// Native primal CG and truncated-Newton iterations for the convex coupled
// quadratic.  Both share only the feasibility projection and exact configured
// outer budget: CG updates a Polak-Ribiere+ direction, while Newton computes a
// matrix-free Hessian Newton direction with a CG linear solve.  Armijo search
// uses MuJoCo's configured ls_iterations/ls_tolerance values.
inline float elliptic_projected_residual(thread const float* A,
    thread const float* g, thread const float* friction, int dim,
    thread const float* force) {
  thread float scale[6], H[36], linear[6], y[6], projected[6];
  for (int i = 0; i < 6; ++i) {
    scale[i] = i == 0 ? 1.0f : max(friction[i - 1], 0.0f);
    y[i] = projected[i] = 0.0f;
    linear[i] = 0.0f;
  }
  for (int i = 0; i < 36; ++i) H[i] = 0.0f;
  y[0] = force[0];
  for (int i = 1; i < dim; ++i) {
    y[i] = scale[i] > 1e-12f ? force[i] / scale[i] : 0.0f;
  }
  for (int i = 0; i < dim; ++i) {
    linear[i] = scale[i] * g[i];
    for (int j = 0; j < dim; ++j) H[i * 6 + j] = scale[i] * A[i * 6 + j] * scale[j];
  }
  float lipschitz = 1e-15f;
  for (int i = 0; i < dim; ++i) {
    float row_sum = 0.0f;
    for (int j = 0; j < dim; ++j) row_sum += abs(H[i * 6 + j]);
    lipschitz = max(lipschitz, row_sum);
  }
  // Measure stationarity at the retained global force. A convergence check
  // must not run a second local optimizer whose result is thrown away, since
  // coupled contacts can change this block's optimum.
  float residual = 0.0f, scale_ref = 1.0f;
  for (int i = 0; i < dim; ++i) {
    float gradient = linear[i];
    for (int j = 0; j < dim; ++j) gradient += H[i * 6 + j] * y[j];
    projected[i] = y[i] - gradient / lipschitz;
    scale_ref += abs(linear[i]);
    for (int j = 0; j < dim; ++j) scale_ref += abs(H[i * 6 + j] * y[j]);
  }
  project_lorentz(projected, dim);
  for (int i = 0; i < dim; ++i)
    residual = max(residual, abs(y[i] - projected[i]) * lipschitz / scale_ref);
  return residual;
}

// Cone-aware row residual for re-certification loops (R04): elliptic
// member rows use the Lorentz certificate over their block; all other
// rows use the box-projected row residual with identical scaling.
inline float cert_row_residual(int row,
    thread const float* W, thread const float* R, thread const float* rhs,
    thread const float* ar, thread const float* lam,
    thread const float* lo, thread const float* hi,
    device const int* enabled, int nr, int total_nr,
    int eblock, int edim, thread const float* emu) {
  if (eblock >= 0) {
    return 0.0f;  // aggregated per block by the caller (see below)
  }
  float grad = -rhs[row];
  for (int col = 0; col < total_nr; ++col) if (enabled[col]) grad += W[row * nr + col] * lam[col];
  grad += R[row] * lam[row];
  float diag = max(1e-15f, W[row * nr + row] + R[row]);
  float proj = clamp(lam[row] - grad / diag, lo[row], hi[row]);
  float row_scale = abs(ar[row]) + abs(R[row] * lam[row]);
  for (int col = 0; col < total_nr; ++col) if (enabled[col]) row_scale += abs(W[row * nr + col] * lam[col]);
  row_scale = max(1.0f, row_scale);
  return abs(proj - lam[row]) * diag / row_scale;
}

// Per-block cone residual (caller aggregates the max over blocks).
inline float cert_block_residual(int start, int dim, thread const float* mu,
    thread const float* W, thread const float* R, thread const float* rhs,
    thread const float* lam, device const int* enabled,
    int nr, int total_nr) {
  thread float A[36], g[6], frc[6], mulo[5];
  for (int i = 0; i < 36; ++i) A[i] = 0.0f;
  for (int i = 0; i < 6; ++i) { g[i] = 0.0f; frc[i] = 0.0f; }
  for (int i = 0; i < 5; ++i) mulo[i] = mu[i];
  for (int i = 0; i < dim; ++i) {
    int ri = start + i;
    frc[i] = lam[ri];
    g[i] = -rhs[ri];
    for (int col = 0; col < total_nr; ++col)
      if (enabled[col] && (col < start || col >= start + dim))
        g[i] += W[ri * nr + col] * lam[col];
    for (int j = 0; j < dim; ++j) {
      int cj = start + j;
      A[i * 6 + j] = W[ri * nr + cj] + (i == j ? R[ri] : 0.0f);
    }
  }
  return elliptic_projected_residual(A, g, mulo, dim, frc);
}

// ---- No-slip helpers (pinned solNoSlip/mju_QCQP family, R04) ----
// QCQP in 2 dimensions: min 0.5*x'*A*x + x'*b s.t. sum (xi/di)^2 <= r^2.
// Returns 0 if unconstrained, 1 if constrained.
inline int nsl_qcqp2(thread float* res, thread const float* Ain, thread const float* bin,
    thread const float* d, float r) {
  float b1 = bin[0]*d[0];
  float b2 = bin[1]*d[1];
  float A11 = Ain[0]*d[0]*d[0];
  float A22 = Ain[3]*d[1]*d[1];
  float A12 = Ain[1]*d[0]*d[1];
  float la = 0.0f;
  float v1 = 0.0f, v2 = 0.0f;
  for (int iter = 0; iter < 20; ++iter) {
    float det = (A11+la)*(A22+la) - A12*A12;
    if (det < 1e-10f) {
      res[0] = 0.0f;
      res[1] = 0.0f;
      return 0;
    }
    float detinv = 1.0f/det;
    float P11 = (A22+la)*detinv;
    float P22 = (A11+la)*detinv;
    float P12 = -A12*detinv;
    v1 = -P11*b1 - P12*b2;
    v2 = -P12*b1 - P22*b2;
    float val = v1*v1 + v2*v2 - r*r;
    if (val < 1e-10f) break;
    float deriv = -2.0f*(P11*v1*v1 + 2.0f*P12*v1*v2 + P22*v2*v2);
    float delta = -val/deriv;
    if (delta < 1e-10f) break;
    la += delta;
  }
  res[0] = v1*d[0];
  res[1] = v2*d[1];
  return (la != 0.0f);
}

// QCQP in 3 dimensions (same contract).
inline int nsl_qcqp3(thread float* res, thread const float* Ain, thread const float* bin,
    thread const float* d, float r) {
  float b1 = bin[0]*d[0];
  float b2 = bin[1]*d[1];
  float b3 = bin[2]*d[2];
  float A11 = Ain[0]*d[0]*d[0];
  float A22 = Ain[4]*d[1]*d[1];
  float A33 = Ain[8]*d[2]*d[2];
  float A12 = Ain[1]*d[0]*d[1];
  float A13 = Ain[2]*d[0]*d[2];
  float A23 = Ain[5]*d[1]*d[2];
  float la = 0.0f;
  float v1 = 0.0f, v2 = 0.0f, v3 = 0.0f;
  for (int iter = 0; iter < 20; ++iter) {
    float P11 = (A22+la)*(A33+la) - A23*A23;
    float P22 = (A11+la)*(A33+la) - A13*A13;
    float P33 = (A11+la)*(A22+la) - A12*A12;
    float P12 = A13*A23 - A12*(A33+la);
    float P13 = A12*A23 - A13*(A22+la);
    float P23 = A12*A13 - A23*(A11+la);
    float det = (A11+la)*P11 + A12*P12 + A13*P13;
    if (det < 1e-10f) {
      res[0] = 0.0f;
      res[1] = 0.0f;
      res[2] = 0.0f;
      return 0;
    }
    float detinv = 1.0f/det;
    P11 *= detinv;
    P22 *= detinv;
    P33 *= detinv;
    P12 *= detinv;
    P13 *= detinv;
    P23 *= detinv;
    v1 = -P11*b1 - P12*b2 - P13*b3;
    v2 = -P12*b1 - P22*b2 - P23*b3;
    v3 = -P13*b1 - P23*b2 - P33*b3;
    float val = v1*v1 + v2*v2 + v3*v3 - r*r;
    if (val < 1e-10f) break;
    float deriv = -2.0f*(P11*v1*v1 + P22*v2*v2 + P33*v3*v3)
            -4.0f*(P12*v1*v2 + P13*v1*v3 + P23*v2*v3);
    float delta = -val/deriv;
    if (delta < 1e-10f) break;
    la += delta;
  }
  res[0] = v1*d[0];
  res[1] = v2*d[1];
  res[2] = v3*d[2];
  return (la != 0.0f);
}

// QCQP in n<=5 dimensions via Cholesky Newton (same contract). The pinned
// mju_cholFactor/mju_cholSolve steps are inlined (rank threshold 1e-10).
inline int nsl_qcqpn(thread float* res, thread const float* Ain, thread const float* bin,
    thread const float* d, float r, int n) {
  thread float A[25], Ala[25], b[5], tmp[5];
  for (int i = 0; i < 25; ++i) A[i] = 0.0f;
  for (int i = 0; i < 5; ++i) { b[i] = 0.0f; tmp[i] = 0.0f; res[i] = 0.0f; }
  if (n > 5) return 0;
  for (int i = 0; i < n; ++i) {
    b[i] = bin[i] * d[i];
    for (int j = 0; j < n; ++j) A[j+i*n] = Ain[j+i*n] * d[i] * d[j];
  }
  float la = 0.0f;
  for (int iter = 0; iter < 20; ++iter) {
    for (int i = 0; i < n*n; ++i) Ala[i] = A[i];
    for (int i = 0; i < n; ++i) Ala[i*(n+1)] += la;
    int rank = 0;
    for (int i = 0; i < n; ++i) {
      for (int j = 0; j <= i; ++j) {
        float v = Ala[i*n+j];
        for (int k = 0; k < j; ++k) v -= Ala[i*n+k] * Ala[j*n+k];
        if (i == j) {
          if (!(v > 1e-10f)) break;
          Ala[i*n+j] = sqrt(v);
          rank++;
        } else {
          Ala[i*n+j] = v / Ala[j*n+j];
        }
      }
      if (rank <= i) break;
    }
    if (rank < n) {
      for (int i = 0; i < n; ++i) res[i] = 0.0f;
      return 0;
    }
    // res = -Ala \ b.
    for (int i = 0; i < n; ++i) {
      float v = b[i];
      for (int k = 0; k < i; ++k) v -= Ala[i*n+k] * tmp[k];
      tmp[i] = v / Ala[i*n+i];
    }
    for (int i = n - 1; i >= 0; --i) {
      float v = tmp[i];
      for (int k = i + 1; k < n; ++k) v -= Ala[k*n+i] * res[k];
      res[i] = v / Ala[i*n+i];
    }
    for (int i = 0; i < n; ++i) res[i] = -res[i];
    float val = 0.0f;
    for (int i = 0; i < n; ++i) val += res[i]*res[i];
    val -= r*r;
    if (val < 1e-10f) break;
    // deriv = -2 * res' * Ala^-1 * res.
    for (int i = 0; i < n; ++i) {
      float v = res[i];
      for (int k = 0; k < i; ++k) v -= Ala[i*n+k] * tmp[k];
      tmp[i] = v / Ala[i*n+i];
    }
    for (int i = n - 1; i >= 0; --i) {
      float v = tmp[i];
      for (int k = i + 1; k < n; ++k) v -= Ala[k*n+i] * tmp[k];
      tmp[i] = v / Ala[i*n+i];
    }
    float deriv = 0.0f;
    for (int i = 0; i < n; ++i) deriv += res[i] * tmp[i];
    deriv *= -2.0f;
    float delta = -val/deriv;
    if (delta < 1e-10f) break;
    la += delta;
  }
  for (int i = 0; i < n; ++i) res[i] = res[i] * d[i];
  return (la != 0.0f);
}

inline uint pgs_pcg32_next(thread ulong& state, thread ulong& inc) {
  ulong oldstate = state;
  state = oldstate * 6364136223846793005ul + (inc | 1ul);
  uint xorshifted = uint(((oldstate >> 18u) ^ oldstate) >> 27u);
  uint rot = uint(oldstate >> 59u);
  return (xorshifted >> rot) | (xorshifted << ((-rot) & 31u));
}

// The permutation shares a float32 scratch allocation with solver vectors.
// Store row indices as their raw uint bit pattern, never as float values:
// integer-valued floats stop representing every row above 2^24.
inline float pgs_order_encode(int row) { return as_type<float>(uint(row)); }
inline int pgs_order_decode(float encoded) { return int(as_type<uint>(encoded)); }

inline int pgs_build_block_order(device const int* enabled,
    device const int* elliptic_member, int n,
    device const int* elliptic_start, int elliptic_count,
    device float* order) {
  int count = 0;
  for (int row = 0; row < n; ++row) {
    if (!enabled[row]) continue;
    if (!elliptic_member[row]) {
      order[count++] = pgs_order_encode(row);
      continue;
    }
    for (int block = 0; block < elliptic_count; ++block) {
      if (elliptic_start[block] == row) {
        order[count++] = pgs_order_encode(row);
        break;
      }
    }
  }
  return count;
}

inline int solve_pgs_thread(thread const float* W, thread const float* R,
    thread const float* rhs, thread const float* ar,
    thread const float* lo, thread const float* hi,
    device const int* enabled, device const int* elliptic_member,
    thread float* lam, int nr, int n, int maxiter, float tolerance,
    float improvement_scale,
    int elliptic_count, device const int* elliptic_start,
    device const int* elliptic_dim, device const float* elliptic_friction,
    device float* order, thread float* final_residual) {
  int used = 0;
  float residual = 0.0f;
  int nblocks = pgs_build_block_order(enabled, elliptic_member, n,
      elliptic_start, elliptic_count, order);
  thread ulong rng_state = 0ul;
  thread ulong rng_inc = 1ul;
  pgs_pcg32_next(rng_state, rng_inc);
  for (int iter = 0; iter < maxiter; ++iter) {
    float improvement = 0.0f;
    for (int i = nblocks - 1; i > 0; --i) {
      int j = int(pgs_pcg32_next(rng_state, rng_inc) % uint(i + 1));
      float tmp = order[i]; order[i] = order[j]; order[j] = tmp;
    }
    for (int visit = 0; visit < nblocks; ++visit) {
      int row = pgs_order_decode(order[visit]);
      if (!elliptic_member[row]) {
        float res = -rhs[row];
        for (int col = 0; col < n; ++col)
          if (enabled[col]) res += W[row * nr + col] * lam[col];
        res += R[row] * lam[row];
        float diag = max(1e-15f, W[row * nr + row] + R[row]);
        float old_force = lam[row];
        float next_force = clamp(old_force - res / diag, lo[row], hi[row]);
        float delta = next_force - old_force;
        float change = 0.5f * delta * delta * diag + delta * res;
        if (change > 1e-10f) { next_force = old_force; change = 0.0f; }
        lam[row] = next_force;
        improvement -= change;
        continue;
      }
      int block = -1;
      for (int b = 0; b < elliptic_count; ++b)
        if (elliptic_start[b] == row) { block = b; break; }
      if (block < 0) continue;
      int start = elliptic_start[block], dim = elliptic_dim[block];
      thread float A[36], old[6], res[6], Ac[25], bc[5], qv[6], mu[5];
      for (int i = 0; i < 36; ++i) A[i] = 0.0f;
      for (int i = 0; i < 25; ++i) Ac[i] = 0.0f;
      for (int i = 0; i < 6; ++i) { old[i] = res[i] = qv[i] = 0.0f; }
      for (int i = 0; i < 5; ++i) { bc[i] = 0.0f; mu[i] = elliptic_friction[block * 5 + i]; }
      for (int i = 0; i < dim; ++i) {
        int row = start + i;
        old[i] = lam[row];
        res[i] = -rhs[row];
        for (int col = 0; col < n; ++col) if (enabled[col])
          res[i] += W[row * nr + col] * lam[col];
        // The block Hessian uses the regularized Delassus operator
        // A = W + diag(R); its residual must use that same operator.
        res[i] += R[row] * lam[row];
        for (int j = 0; j < dim; ++j) {
          int col = start + j;
          A[i * dim + j] = W[row * nr + col] + (i == j ? R[row] : 0.0f);
        }
      }
      if (old[0] < 1e-15f) {
        lam[start] = max(0.0f, old[0] - res[0] / max(A[0], 1e-15f));
        for (int i = 1; i < dim; ++i) lam[start + i] = 0.0f;
      } else {
        float denom = 0.0f, numerator = 0.0f;
        for (int i = 0; i < dim; ++i) {
          float av = 0.0f;
          for (int j = 0; j < dim; ++j) av += A[i * dim + j] * old[j];
          denom += old[i] * av;
          numerator += old[i] * res[i];
        }
        if (denom > 1e-15f) {
          float alpha = -numerator / denom;
          if (old[0] + alpha * old[0] < 0.0f) alpha = -1.0f;
          for (int i = 0; i < dim; ++i) lam[start + i] = old[i] + alpha * old[i];
        }
      }
      // Pinned PGS performs the friction QCQP after either the normal ray
      // step or the zero-normal projected step. In particular, a cold block
      // can acquire a positive normal impulse and tangential impulse on its
      // first sweep.
      int nf = dim - 1;
      for (int i = 0; i < nf; ++i) {
        bc[i] = res[i + 1] + A[(i + 1) * dim] * (lam[start] - old[0]);
        for (int j = 0; j < nf; ++j) {
          Ac[i * nf + j] = A[(i + 1) * dim + j + 1];
          bc[i] -= Ac[i * nf + j] * old[j + 1];
        }
      }
      if (lam[start] < 1e-15f) {
        for (int i = 1; i < dim; ++i) lam[start + i] = 0.0f;
      } else {
        int active = 0;
        if (nf == 2) active = nsl_qcqp2(qv, Ac, bc, mu, lam[start]);
        else if (nf == 3) active = nsl_qcqp3(qv, Ac, bc, mu, lam[start]);
        else active = nsl_qcqpn(qv, Ac, bc, mu, lam[start], nf);
        if (active) {
          float norm = 0.0f;
          for (int i = 0; i < nf; ++i) norm += qv[i] * qv[i] / max(mu[i] * mu[i], 1e-30f);
          float scale = sqrt(lam[start] * lam[start] / max(norm, 1e-15f));
          for (int i = 0; i < nf; ++i) qv[i] *= scale;
        }
        for (int i = 0; i < nf; ++i) lam[start + i + 1] = qv[i];
      }
      float change = 0.0f;
      for (int i = 0; i < dim; ++i) {
        float delta_i = lam[start + i] - old[i];
        float product = 0.0f;
        for (int j = 0; j < dim; ++j)
          product += A[i * dim + j] * (lam[start + j] - old[j]);
        change += 0.5f * delta_i * product + delta_i * res[i];
      }
      if (change > 1e-10f) {
        for (int i = 0; i < dim; ++i) lam[start + i] = old[i];
        change = 0.0f;
      }
      improvement -= change;
    }
    residual = 0.0f;
    for (int row = 0; row < n; ++row) if (enabled[row] && !elliptic_member[row]) {
      float grad = -rhs[row];
      for (int col = 0; col < n; ++col) if (enabled[col]) grad += W[row * nr + col] * lam[col];
      grad += R[row] * lam[row];
      float diag = max(1e-15f, W[row * nr + row] + R[row]);
      float projected = clamp(lam[row] - grad / diag, lo[row], hi[row]);
      float scale = abs(ar[row]) + abs(R[row] * lam[row]);
      for (int col = 0; col < n; ++col) if (enabled[col]) scale += abs(W[row * nr + col] * lam[col]);
      residual = max(residual, abs(projected - lam[row]) * diag / max(1.0f, scale));
    }
    for (int block = 0; block < elliptic_count; ++block) {
      int start = elliptic_start[block], dim = elliptic_dim[block];
      thread float A[36], g[6], force[6], mu[5];
      for (int i = 0; i < 36; ++i) A[i] = 0.0f;
      for (int i = 0; i < 6; ++i) { g[i] = 0.0f; force[i] = 0.0f; }
      for (int i = 0; i < 5; ++i) mu[i] = elliptic_friction[block * 5 + i];
      for (int i = 0; i < dim; ++i) {
        int ri = start + i;
        force[i] = lam[ri];
        g[i] = -rhs[ri];
        for (int col = 0; col < n; ++col)
          if (enabled[col] && (col < start || col >= start + dim)) g[i] += W[ri * nr + col] * lam[col];
        for (int j = 0; j < dim; ++j) {
          int rj = start + j;
          A[i * 6 + j] = W[ri * nr + rj] + (i == j ? R[ri] : 0.0f);
        }
      }
      residual = max(residual, elliptic_projected_residual(A, g, mu, dim, force));
    }
    ++used;
    improvement *= improvement_scale;
    if (improvement < tolerance) break;
  }
  *final_residual = residual;
  return used;
}

// Rebuild one Delassus entry as a compensated pair for PGS. The retained W
// plane stays the public/high workspace; component mode also has a low half
// for M^-1 J^T in `mass`. Dense mode at least preserves dot-product error.
inline PrimalFloatPair pgs_delassus_pair(device const float* J,
    device const float* Z, device const float* mass, constant int* dims,
    int batch, int world, int nr, int nv, int row, int col) {
  PrimalFloatPair value = {0.0f, 0.0f};
  for (int k = 0; k < nv; ++k) {
    float z_low = 0.0f;
    if (dims[20] == -2)
      z_low = mass[component_pair_index(batch, world, col, k,
                                        nr, nv, true)];
    value = primal_pair_fma(ccj_get(J, row, k, nv),
        {Z[col * nv + k], z_low}, value);
  }
  return value;
}

inline int solve_pgs_device(device const float* W, device const float* R,
    device const float* rhs, device const float* rhs_low,
    device const float* ar,
    device const float* lo, device const float* hi,
    device const int* enabled, device const int* elliptic_member,
    device float* lam, device float* lam_low,
    device const float* J, device const float* Z, device const float* mass,
    constant int* dims, int batch, int world, int nv,
    int nr, int n, int maxiter, float tolerance,
    float improvement_scale,
    int elliptic_count, device const int* elliptic_start,
    device const int* elliptic_dim, device const float* elliptic_friction,
    device float* order, thread float* final_residual) {
  int used = 0;
  float residual = 0.0f;
  int nblocks = pgs_build_block_order(enabled, elliptic_member, n,
      elliptic_start, elliptic_count, order);
  thread ulong rng_state = 0ul;
  thread ulong rng_inc = 1ul;
  pgs_pcg32_next(rng_state, rng_inc);
  for (int iter = 0; iter < maxiter; ++iter) {
    float improvement = 0.0f;
    for (int i = nblocks - 1; i > 0; --i) {
      int j = int(pgs_pcg32_next(rng_state, rng_inc) % uint(i + 1));
      float tmp = order[i]; order[i] = order[j]; order[j] = tmp;
    }
    for (int visit = 0; visit < nblocks; ++visit) {
      int row = pgs_order_decode(order[visit]);
      if (!elliptic_member[row]) {
        PrimalFloatPair res = {-rhs[row], -rhs_low[row]};
        for (int col = 0; col < n; ++col) if (enabled[col]) {
          PrimalFloatPair wij = pgs_delassus_pair(
              J, Z, mass, dims, batch, world, nr, nv, row, col);
          res = primal_pair_add(res,
              primal_pair_product(wij, {lam[col], lam_low[col]}));
        }
        res = primal_pair_fma(R[row], {lam[row], lam_low[row]}, res);
        PrimalFloatPair diag_pair = primal_pair_add(
            pgs_delassus_pair(J, Z, mass, dims, batch, world,
                              nr, nv, row, row), {R[row], 0.0f});
        if (!(diag_pair.hi > 1e-15f)) diag_pair = {1e-15f, 0.0f};
        PrimalFloatPair old_force = {lam[row], lam_low[row]};
        PrimalFloatPair next_force = primal_pair_add(old_force,
            primal_pair_multiply(primal_pair_divide(res, diag_pair), -1.0f));
        float next_value = primal_pair_value(next_force);
        if (next_value < lo[row]) next_force = {lo[row], 0.0f};
        else if (next_value > hi[row]) next_force = {hi[row], 0.0f};
        PrimalFloatPair delta = primal_pair_add(next_force,
            primal_pair_multiply(old_force, -1.0f));
        PrimalFloatPair change = primal_pair_add(
            primal_pair_multiply(primal_pair_product(delta, delta),
                                 0.5f * diag_pair.hi),
            primal_pair_product(delta, res));
        if (primal_pair_value(change) > 1e-10f) {
          next_force = old_force;
          change = {0.0f, 0.0f};
        }
        lam[row] = next_force.hi;
        lam_low[row] = next_force.lo;
        improvement -= primal_pair_value(change);
        continue;
      }
      int block = -1;
      for (int b = 0; b < elliptic_count; ++b)
        if (elliptic_start[b] == row) { block = b; break; }
      if (block < 0) continue;
      int start = elliptic_start[block], dim = elliptic_dim[block];
      thread float A[36], old[6], old_low[6], res[6], Ac[25], bc[5], qv[6], mu[5];
      thread PrimalFloatPair A_pair[36], old_pair[6], res_pair_local[6];
      for (int i = 0; i < 36; ++i) { A[i] = 0.0f; A_pair[i] = {0.0f, 0.0f}; }
      for (int i = 0; i < 25; ++i) Ac[i] = 0.0f;
      for (int i = 0; i < 6; ++i) { old[i] = old_low[i] = res[i] = qv[i] = 0.0f; old_pair[i] = res_pair_local[i] = {0.0f, 0.0f}; }
      for (int i = 0; i < 5; ++i) { bc[i] = 0.0f; mu[i] = elliptic_friction[block * 5 + i]; }
      for (int i = 0; i < dim; ++i) {
        int row = start + i;
        old_low[i] = lam_low[row];
        old_pair[i] = {lam[row], lam_low[row]};
        old[i] = primal_pair_value(old_pair[i]);
        PrimalFloatPair residual_pair = {-rhs[row], -rhs_low[row]};
        for (int col = 0; col < n; ++col) if (enabled[col])
          residual_pair = primal_pair_add(residual_pair,
              primal_pair_product(pgs_delassus_pair(
                  J, Z, mass, dims, batch, world, nr, nv, row, col),
                  {lam[col], lam_low[col]}));
        residual_pair = primal_pair_fma(R[row],
            {lam[row], lam_low[row]}, residual_pair);
        res_pair_local[i] = residual_pair;
        res[i] = primal_pair_value(residual_pair);
        for (int j = 0; j < dim; ++j) {
          int col = start + j;
          A_pair[i * dim + j] = primal_pair_add(
              pgs_delassus_pair(J, Z, mass, dims, batch, world,
                                nr, nv, row, col),
              {i == j ? R[row] : 0.0f, 0.0f});
          A[i * dim + j] = primal_pair_value(A_pair[i * dim + j]);
        }
      }
      if (old[0] < 1e-15f) {
        lam[start] = max(0.0f, old[0] - res[0] / max(A[0], 1e-15f));
        for (int i = 1; i < dim; ++i) lam[start + i] = 0.0f;
      } else {
        float denom = 0.0f, numerator = 0.0f;
        for (int i = 0; i < dim; ++i) {
          float av = 0.0f;
          for (int j = 0; j < dim; ++j) av += A[i * dim + j] * old[j];
          denom += old[i] * av;
          numerator += old[i] * res[i];
        }
        if (denom > 1e-15f) {
          float alpha = -numerator / denom;
          if (old[0] + alpha * old[0] < 0.0f) alpha = -1.0f;
          for (int i = 0; i < dim; ++i) lam[start + i] = old[i] + alpha * old[i];
        }
      }
      int nf = dim - 1;
      for (int i = 0; i < nf; ++i) {
        bc[i] = res[i + 1] + A[(i + 1) * dim] * (lam[start] - old[0]);
        for (int j = 0; j < nf; ++j) {
          Ac[i * nf + j] = A[(i + 1) * dim + j + 1];
          bc[i] -= Ac[i * nf + j] * old[j + 1];
        }
      }
      if (lam[start] < 1e-15f) {
        for (int i = 1; i < dim; ++i) lam[start + i] = 0.0f;
      } else {
        int active = 0;
        if (nf == 2) active = nsl_qcqp2(qv, Ac, bc, mu, lam[start]);
        else if (nf == 3) active = nsl_qcqp3(qv, Ac, bc, mu, lam[start]);
        else active = nsl_qcqpn(qv, Ac, bc, mu, lam[start], nf);
        if (active) {
          float norm = 0.0f;
          for (int i = 0; i < nf; ++i) norm += qv[i] * qv[i] / max(mu[i] * mu[i], 1e-30f);
          float scale = sqrt(lam[start] * lam[start] / max(norm, 1e-15f));
          for (int i = 0; i < nf; ++i) qv[i] *= scale;
        }
        for (int i = 0; i < nf; ++i) lam[start + i + 1] = qv[i];
      }
      PrimalFloatPair change_pair = {0.0f, 0.0f};
      for (int i = 0; i < dim; ++i) {
        PrimalFloatPair delta_i = primal_pair_add(
            {lam[start + i], 0.0f},
            primal_pair_multiply(old_pair[i], -1.0f));
        PrimalFloatPair product = {0.0f, 0.0f};
        for (int j = 0; j < dim; ++j) {
          PrimalFloatPair delta_j = primal_pair_add(
              {lam[start + j], 0.0f},
              primal_pair_multiply(old_pair[j], -1.0f));
          product = primal_pair_add(product,
              primal_pair_product(A_pair[i * dim + j], delta_j));
        }
        change_pair = primal_pair_add(change_pair,
            primal_pair_add(
                primal_pair_multiply(primal_pair_product(delta_i, product), 0.5f),
                primal_pair_product(delta_i, res_pair_local[i])));
      }
      float change = primal_pair_value(change_pair);
      if (change > 1e-10f) {
        for (int i = 0; i < dim; ++i) {
          lam[start + i] = old[i];
          lam_low[start + i] = old_low[i];
        }
        change = 0.0f;
      } else {
        // The cone optimizer returns float coordinates; clear the old residual
        // word after accepting a projected point so the pair stays coherent.
        for (int i = 0; i < dim; ++i) lam_low[start + i] = 0.0f;
      }
      improvement -= change;
    }
    residual = 0.0f;
    for (int row = 0; row < n; ++row) if (enabled[row] && !elliptic_member[row]) {
      float grad = -rhs[row];
      for (int col = 0; col < n; ++col) if (enabled[col]) grad += W[row * nr + col] * lam[col];
      grad += R[row] * lam[row];
      float diag = max(1e-15f, W[row * nr + row] + R[row]);
      float projected = clamp(lam[row] - grad / diag, lo[row], hi[row]);
      float scale = abs(ar[row]) + abs(R[row] * lam[row]);
      for (int col = 0; col < n; ++col) if (enabled[col]) scale += abs(W[row * nr + col] * lam[col]);
      residual = max(residual, abs(projected - lam[row]) * diag / max(1.0f, scale));
    }
    for (int block = 0; block < elliptic_count; ++block) {
      int start = elliptic_start[block], dim = elliptic_dim[block];
      thread float A[36], g[6], force[6], mu[5];
      for (int i = 0; i < 36; ++i) A[i] = 0.0f;
      for (int i = 0; i < 6; ++i) { g[i] = 0.0f; force[i] = 0.0f; }
      for (int i = 0; i < 5; ++i) mu[i] = elliptic_friction[block * 5 + i];
      for (int i = 0; i < dim; ++i) {
        int ri = start + i;
        force[i] = lam[ri];
        g[i] = -rhs[ri];
        for (int col = 0; col < n; ++col)
          if (enabled[col] && (col < start || col >= start + dim)) g[i] += W[ri * nr + col] * lam[col];
        for (int j = 0; j < dim; ++j) {
          int rj = start + j;
          A[i * 6 + j] = W[ri * nr + rj] + (i == j ? R[ri] : 0.0f);
        }
      }
      residual = max(residual, elliptic_projected_residual(A, g, mu, dim, force));
    }
    ++used;
    improvement *= improvement_scale;
    if (improvement < tolerance) break;
  }
  *final_residual = residual;
  return used;
}

kernel void solve_coupled_constraints(
    device const float* mass [[buffer(0)]],
    device const float* qfrc [[buffer(1)]],
    device const float* qpos [[buffer(2)]],
    device const float* qvel [[buffer(3)]],
    device const int* eq_active [[buffer(4)]],
    device const int* joint_qadr [[buffer(5)]],
    device const float* qpos0 [[buffer(6)]],
    device const int* joint_dadr [[buffer(7)]],
    device const uchar* joint_limited [[buffer(8)]],
    device const float* joint_limit_params [[buffer(9)]],
    device const float* joint_sol_params [[buffer(10)]],
    device const float* frictionloss [[buffer(11)]],
    device const float* invweight [[buffer(12)]],
    device const float* dof_sol_params [[buffer(13)]],
    device const int* eq_obj [[buffer(14)]],
    device const float* eq_data [[buffer(15)]],
    device const float* eq_sol_params [[buffer(16)]],
    device const float* contact_jacobian [[buffer(17)]],
    device const float* contact_row_data [[buffer(18)]],
    device const float* contact_friction [[buffer(19)]],
    device const int* contact_condim [[buffer(20)]],
    constant int* dims [[buffer(21)]],
    constant float* params [[buffer(22)]],
    device float* out_force [[buffer(23)]],
    device float* out_acc [[buffer(24)]],
    device int* out_status [[buffer(25)]],
    device float* out_diagnostics [[buffer(26)]],
    device float* out_contact_force [[buffer(27)]],
    device const int* packed_slot_to_logical [[buffer(28)]],
    device float* workspace_J [[buffer(29)]],
    device float* workspace_debug [[buffer(30)]],
    uint world [[thread_position_in_grid]]) {
  int nq = dims[0];
  int nv = dims[1];
  int nj = dims[2];
  int neq = dims[3];
  int ncontacts_max = dims[4];
  int batch = dims[5];
  int flags = dims[6];
  bool cached_velocity_stage = (dims[7] & 2) != 0;
  // Bit 8 reuses a just-assembled canonical row set after a host/device
  // impedance update. It does not imply cached-velocity/Jdot semantics.
  bool preassembled_rows_stage = (dims[7] & 8) != 0;
  bool reuse_rows_stage = cached_velocity_stage || preassembled_rows_stage;
  bool assembly_only_stage = dims[20] == -1;
  bool provided_qacc_smooth = (dims[7] & 4) != 0;
  // PGS may receive qfrc as two contiguous world-major planes (high then
  // low). This is independent of provided_qacc_smooth: without that flag the
  // pair is smooth force; with it, the pair is smooth acceleration.
  bool paired_qfrc_input = (dims[7] & 16) != 0;
  bool refsafe = (dims[7] & 1) != 0;
  bool trace_solver_counts = (dims[7] & 32) != 0;
  bool trace_primal_details = (dims[7] & 64) != 0;
  int maxiter = dims[8];
  int solver_type = dims[19];
  int nr = dims[9];
  int cone_type = dims[10];
  int n_eq_rows = nr > 0 ? dims[14] : 0;
  if (n_eq_rows < 0) n_eq_rows = 0;
  if (n_eq_rows > nr) n_eq_rows = nr;
  int ten_base = nr > 0 ? dims[16] : 0;
  if (ten_base < 0) ten_base = 0;
  if (ten_base > nr) ten_base = nr;
  if (world >= uint(batch)) return;
  if (!cc_solver_world_enabled(dims, int(world))) return;

  int jmetadata = dims[24] + nr * 10;
  int jstride = dims[jmetadata + 17];
  device float* J_world = ccj_world(workspace_J, int(world), nr, nv, dims);
  device int* jheader = reinterpret_cast<device int*>(J_world);
  if (jstride < 12 || !ccj_header_valid(jheader, nr, nv)
      || jheader[9] != jstride || jheader[10] != 0) {
    out_status[world] = max(out_status[world], 2);
    if (dims[20] != -1) {
      int base = int(world) * nv;
      for (int i = 0; i < nv; ++i) {
        out_force[base + i] = 0.0f;
        out_acc[base + i] = 0.0f;
        out_acc[batch * nv + base + i] = 0.0f;
      }
    }
    return;
  }

  int mb = world * nv * nv;  int qb = world * nv;
  int pb = world * nq;
  int incoming_status = out_status[world];
  if (!assembly_only_stage) {
    out_diagnostics[world * 10] = 0.0f;
    out_diagnostics[world * 10 + 1] = 0.0f;
    for (int k = 0; k < 8; ++k)
      out_diagnostics[world * 10 + 2 + k] = trace_solver_counts ? -1.0f : 0.0f;
    for (int i = 0; i < nv; ++i) {
      out_force[qb + i] = 0.0f;
      out_acc[qb + i] = 0.0f;
      out_acc[batch * nv + qb + i] = 0.0f;
    }
  }
  if (incoming_status != 0) return;
  if (!assembly_only_stage) out_status[world] = 0;
  if (!reuse_rows_stage && workspace_debug && nr > 0) {
    device float* dbg = workspace_debug + world * max(dims[21], nr * nr + 7 * nr);
    for (int i = 0; i < nr * nr; ++i) dbg[i] = 0.0f;
    for (int r = n_eq_rows; r < nr; ++r) {
      // Tendon-owned rows are preassembled by tendon_constraint_rows; keep them.
      if (dbg[nr * nr + 6 * nr + r] > 0.5f) continue;
      dbg[nr * nr + r] = 0.0f;
      dbg[nr * nr + nr + r] = 0.0f;
    }
    // Zero the residual vector region only; the retained-multiplier region
    // [3*nr, 4*nr) persists across steps for warm starts (cleared by reset,
    // restore-clear and clear_warmstart on the host).
    // Stage -1 leaves this RHS slice available as a temporary row-impedance
    // channel written by the canonical row producers. The solver-only
    // preassembled stage consumes that channel before reusing RHS storage.
    if (dims[20] != -1)
      for (int i = 0; i < nr; ++i) dbg[nr * nr + 2 * nr + i] = 0.0f;
  }
  if (nv == 0) return;
  device float* dbg_pre = workspace_debug + world * max(dims[21], nr * nr + 7 * nr);
  device float* solver_scratch = dbg_pre + nr * nr + 7 * nr;
  int metadata_header = dims[24] + dims[9] * 10;
  int row_workspace = dims[metadata_header + 8];
  device float* Z = solver_scratch + dims[metadata_header + 10];
  device int* row_meta = reinterpret_cast<device int*>(solver_scratch)
      + row_workspace;
  device int* enabled = row_meta;
  device int* elliptic_member = enabled + nr;
  device int* contact_block_start = elliptic_member + nr;
  device int* contact_block_size = contact_block_start + nr;
  device int* contact_slot_id = contact_block_size + nr;
  device int* elliptic_start = contact_slot_id + nr;
  device int* elliptic_dim = elliptic_start + nr;
  device float* elliptic_friction = reinterpret_cast<device float*>(elliptic_dim + nr);
  for (int r = n_eq_rows; !reuse_rows_stage && jheader[2] == CCJ_DENSE && r < nr; ++r) {
    // Tendon-owned rows are preassembled by tendon_constraint_rows; keep them.
    if (nr > 0 && dbg_pre[nr * nr + 6 * nr + r] > 0.5f) continue;
    ccj_zero_row(J_world, r, nv);
  }

  device float* L = solver_scratch;
  device float* y = L + nv * nv;
  device float* x = y + nv;
  device const float* q0 = out_acc + qb;
  // These are the canonical retained row views already backed by the
  // dynamically sized per-world workspace. Keeping them here avoids the
  // former fixed [96,96] thread-local Delassus matrix and [96] row vectors.
  device float* W = dbg_pre;
  device float* R = dbg_pre + nr * nr;
  device float* ar = R + nr;
  device float* rhs = ar + nr;
  device float* lam = rhs + nr;
  device float* lo = lam + nr;
  device float* hi = lo + nr;
  device float* pgs_pair_tail = solver_scratch
      + dims[metadata_header + 15];
  device float* pgs_rhs_low = pgs_pair_tail + 8 * max(nv, 1);
  device float* pgs_lambda_low = pgs_rhs_low + max(nr, 1);
  device const float* aref_low = pgs_lambda_low + max(nr, 1);
  // Row R/aref/bounds and retained lambda share the public dynamic prefix.
  // Reset only activation here: assembly already wrote source row data into
  // that prefix, and an aliasing copy from a just-cleared local array would
  // erase it before the solve.
  bool warm_ok = (nr > 0) && ((flags & 512) == 0);
  for (int i = 0; i < nr; ++i) {
    enabled[i] = 0;
    elliptic_member[i] = 0;
  }
  for (int i = 0; i < nr * nr; ++i) W[i] = 0.0f;
  // Warm start: seed multipliers from the retained solution unless the
  // WARMSTART disable bit (512) is set. Bad/foreign warm vectors are
  // rejected by the cost check after Delassus assembly below.
  for (int i = 0; i < nr; ++i) {
    if (reuse_rows_stage && workspace_debug) {
      device float* cached = workspace_debug + world * max(dims[21], nr * nr + 7 * nr);
      R[i] = cached[nr * nr + i];
      ar[i] = cached[nr * nr + nr + i];
      lo[i] = cached[nr * nr + 4 * nr + i];
      hi[i] = cached[nr * nr + 5 * nr + i];
      enabled[i] = cached[nr * nr + 6 * nr + i] > 0.5f;
    } else {
      enabled[i] = false;
    }
    if (!warm_ok && !assembly_only_stage) lam[i] = 0.0f;
  }
  if (warm_ok && workspace_debug) {
    device float* dbg0 = workspace_debug + world * max(dims[21], nr * nr + 7 * nr);
    for (int i = 0; i < nr; ++i) lam[i] = dbg0[nr * nr + 3 * nr + i];
  }
  // Equality rows [0, n_eq_rows) are preassembled on Metal by equality_assembly
  // (joint/connect/weld with version-pinned residuals, Jacobians, Jdot, impedance).
  // Treat them as bilateral; inactive rows were written as zero J/R/ar and
  // naturally yield zero multipliers without affecting coupled rows.
  if (!reuse_rows_stage && workspace_debug && nr > 0) {
    device float* dbg = workspace_debug + world * max(dims[21], nr * nr + 7 * nr);
    for (int r = 0; r < n_eq_rows && r < nr; ++r) {
      if (dbg[nr * nr + 6 * nr + r] > 0.5f) {
        // Tendon-owned equality row (milestone 008): fully preassembled by
        // tendon_constraint_rows, including bilateral bounds.
        R[r] = dbg[nr * nr + r];
        ar[r] = dbg[nr * nr + nr + r];
        lo[r] = dbg[nr * nr + 4 * nr + r];
        hi[r] = dbg[nr * nr + 5 * nr + r];
        enabled[r] = true;
        continue;
      }
      R[r] = dbg[nr * nr + r];
      ar[r] = dbg[nr * nr + nr + r];
      // The equality producer publishes bilateral bounds only for an
      // actually active, nondegenerate equality. Preserve its zero bounds
      // for disabled/degenerate rows; inferring bounds here makes an
      // inactive row appear active in captured POS metadata.
      lo[r] = dbg[nr * nr + 4 * nr + r];
      hi[r] = dbg[nr * nr + 5 * nr + r];
      enabled[r] = (R[r] > 0.0f || abs(ar[r]) > 0.0f);
    }
    // Tendon limit/friction rows in the reserved ten region.
    for (int r = ten_base; r < nr; ++r) {
      if (dbg[nr * nr + 6 * nr + r] > 0.5f) {
        R[r] = dbg[nr * nr + r];
        ar[r] = dbg[nr * nr + nr + r];
        lo[r] = dbg[nr * nr + 4 * nr + r];
        hi[r] = dbg[nr * nr + 5 * nr + r];
        enabled[r] = true;
      }
    }
  }

  int base_contact = n_eq_rows + nv + 2 * nj;
  if (nr > 0) {
    int ten_rows = dims[17];
    if (ten_rows < 0) ten_rows = 0;
    if (ten_rows > nr) ten_rows = nr;
    base_contact += ten_rows;
  }
  device float* out_joint_force = out_contact_force
      + max(batch * ncontacts_max * 11, 1)
      + world * max(base_contact, 1);
  int nactive_slots = 0;
  while (nactive_slots < ncontacts_max
         && packed_slot_to_logical[world * ncontacts_max + nactive_slots] >= 0)
    ++nactive_slots;

  // 1. Joint constraints (if constraint disable flag not set: bit 0 mjDSBL_CONSTRAINT)
  if (!reuse_rows_stage && (flags & 1) == 0) {
    // Frictionloss (bit 2 mjDSBL_FRICTIONLOSS)
    for (int d = 0; d < nv; ++d) {
      int row = n_eq_rows + d;
      float loss = frictionloss[d];
      if ((flags & 4) != 0 || loss <= 0.0f) continue;
      ccj_set(J_world, row, d, 1.0f, nv);
          reference_params(dof_sol_params + d * 7, dof_sol_params + d * 7 + 2, 0, 0.0f, 0.0f, qvel[qb + d], invweight[d], true, params[0], refsafe, R[row], ar[row]);
          if (dims[20] == -1)
            dbg_pre[nr * nr + 2 * nr + row] = impedance_at(
                dof_sol_params + d * 7 + 2, 0, 0.0f, 0.0f);
      lo[row] = -loss;
      hi[row] = loss;
      enabled[row] = true;
    }

    // Joint Limits (bit 3 mjDSBL_LIMIT). joint_limited packs bit0=limited,
    // bits[2:1]=joint type (milestone 009 ball support).
    for (int j = 0; j < nj; ++j) {
      int limpack = joint_limited[j];
      if ((limpack & 1) == 0) continue;
      int d = joint_dadr[j];
      int q = joint_qadr[j];
      float margin = joint_limit_params[j * 3 + 2];
      if (((limpack >> 1) & 3) == 1) {
        // Ball limit (pinned mj_instantiateLimit): single row on the rotation
        // angle with Jacobian -axis over the 3 DOFs. Uses the first reserved
        // row of the pair; the second stays disabled.
        float4 quat = float4(qpos[pb+q], qpos[pb+q+1], qpos[pb+q+2], qpos[pb+q+3]);
        float nq4 = length(quat);
        quat = nq4 > 1e-30f ? quat / nq4 : float4(1, 0, 0, 0);
        float3 vv = quat.yzw;
        float s = length(vv);
        float speed = 2.0f * atan2(s, quat.x);
        if (speed > 3.14159265358979f) speed -= 2.0f * 3.14159265358979f;
        float3 aa = s > 1e-30f ? vv * (speed / s) : float3(0.0f);
        float value = length(aa);
        // Pinned mju_normalize3 normalizes angleAxis in place: the Jacobian
        // uses the UNIT axis, value is the pre-normalized angle.
        float3 naxis = value > 1e-30f ? aa / value : float3(0.0f);
        float rmax = max(joint_limit_params[j * 3], joint_limit_params[j * 3 + 1]);
        float dist = rmax - value;
        int rowb = n_eq_rows + nv + 2 * j;
        if ((flags & 8) == 0 && dist < margin) {
          for (int k = 0; k < 3; ++k) {
            int dof = d + k;
            if (dof >= 0 && dof < nv) ccj_set(J_world, rowb, dof, -naxis[k], nv);
          }
          float vel = 0.0f;
          for (int k = 0; k < 3; ++k) {
            int dof = d + k;
            if (dof >= 0 && dof < nv) vel += -naxis[k] * qvel[qb + dof];
          }
          reference_params(joint_sol_params + j * 7, joint_sol_params + j * 7 + 2, 0, dist, margin, vel, invweight[d], false, params[0], refsafe, R[rowb], ar[rowb]);
          if (dims[20] == -1)
            dbg_pre[nr * nr + 2 * nr + rowb] = impedance_at(
                joint_sol_params + j * 7 + 2, 0, dist, margin);
          lo[rowb] = 0.0f;
          hi[rowb] = INFINITY;
          enabled[rowb] = true;
        }
        continue;
      }
      // Lower limit
      int row0 = n_eq_rows + nv + 2 * j;
      float dist0 = qpos[pb + q] - joint_limit_params[j * 3 + 0];
      if ((flags & 8) == 0 && dist0 < margin) {
        ccj_set(J_world, row0, d, 1.0f, nv);
        reference_params(joint_sol_params + j * 7, joint_sol_params + j * 7 + 2, 0, dist0, margin, qvel[qb + d], invweight[d], false, params[0], refsafe, R[row0], ar[row0]);
        if (dims[20] == -1)
          dbg_pre[nr * nr + 2 * nr + row0] = impedance_at(
              joint_sol_params + j * 7 + 2, 0, dist0, margin);
        lo[row0] = 0.0f;
        hi[row0] = INFINITY;
        enabled[row0] = true;
      }
      // Upper limit
      int row1 = n_eq_rows + nv + 2 * j + 1;
      float dist1 = joint_limit_params[j * 3 + 1] - qpos[pb + q];
      if ((flags & 8) == 0 && dist1 < margin) {
        ccj_set(J_world, row1, d, -1.0f, nv);
        reference_params(joint_sol_params + j * 7, joint_sol_params + j * 7 + 2, 0, dist1, margin, -qvel[qb + d], invweight[d], false, params[0], refsafe, R[row1], ar[row1]);
        if (dims[20] == -1)
          dbg_pre[nr * nr + 2 * nr + row1] = impedance_at(
              joint_sol_params + j * 7 + 2, 0, dist1, margin);
        lo[row1] = 0.0f;
        hi[row1] = INFINITY;
        enabled[row1] = true;
      }
    }
  }

  // 2. Contacts (if not disabled by mjDSBL_CONSTRAINT bit 0 or mjDSBL_CONTACT bit 4)
  thread int contact_block_count = 0;
  thread int elliptic_count = 0;

  if (reuse_rows_stage) {
    // Cached position stages reuse canonical J/R/bounds/activity. Rebuild only
    // the small cone grouping metadata required by the selected solver.
    for (int packed_s = 0; packed_s < nactive_slots; ++packed_s) {
      int s = packed_slot_to_logical[world * ncontacts_max + packed_s];
      if (s < 0 || s >= ncontacts_max) continue;
      int cdim = contact_condim[s * 3 + 0];
      int row_offset = contact_condim[s * 3 + 1];
      int row_start = base_contact + row_offset;
      int crbase = (world * ncontacts_max + s) * 36;
      if (contact_row_data[crbase] <= 0.5f) continue;
      if (cone_type == 1 && cdim > 1) {
        if (elliptic_count < nr) {
          elliptic_start[elliptic_count] = row_start;
          elliptic_dim[elliptic_count] = cdim;
          for (int k = 0; k < 5; ++k)
            elliptic_friction[elliptic_count * 5 + k] = contact_friction[s * 5 + k];
          ++elliptic_count;
        }
        for (int k = 0; k < cdim; ++k)
          if (row_start + k < nr) elliptic_member[row_start + k] = true;
      } else if (cdim > 1 && contact_block_count < nr) {
        contact_block_start[contact_block_count] = row_start;
        contact_block_size[contact_block_count++] = 2 * (cdim - 1);
      }
    }
  } else if ((flags & 1) == 0 && (flags & 16) == 0) {
    for (int packed_s = 0; packed_s < nactive_slots; ++packed_s) {
      int s = packed_slot_to_logical[world * ncontacts_max + packed_s];
      if (s < 0 || s >= ncontacts_max) continue;
      int cdim = contact_condim[s * 3 + 0];
      int row_offset = contact_condim[s * 3 + 1];
      int row_start = base_contact + row_offset;
      int cb = world * ncontacts_max + s;
      int cjbase = cb * 6 * nv;
      int crbase = cb * 6 * 6;

      if (cdim == 1) {
        int r = crbase;
        if (contact_row_data[r] > 0.5f) {
          if (row_start + 1 > nr) { out_status[world] = 2; return; }
          int row = row_start;
          if (jheader[2] == CCJ_DENSE)
            for (int i = 0; i < nv; ++i) ccj_set(J_world, row, i, contact_jacobian[cjbase + i], nv);
          float imp = clamp(contact_row_data[r + 4], 1e-6f, 0.999999f);
          float diag_approx = max(contact_row_data[r + 5], 1e-15f);
          R[row] = max(1e-15f, (1.0f - imp) * diag_approx / imp);
          if (dims[20] == -1) dbg_pre[nr * nr + 2 * nr + row] = imp;
          ar[row] = contact_row_data[r + 3];
          lo[row] = 0.0f;
          hi[row] = INFINITY;
          enabled[row] = true;
        }
      } else if (contact_row_data[crbase] > 0.5f) {
        int edge_count = 2 * (cdim - 1);
        int block_rows = cone_type == 0 ? edge_count : cdim;
        if (row_start + block_rows > nr) { out_status[world] = 2; return; }
        float imp = clamp(contact_row_data[crbase + 4], 1e-6f, 0.999999f);
        float diag_approx = max(contact_row_data[crbase + 5], 1e-15f);
        float normal_R = max(1e-15f, (1.0f - imp) * diag_approx / imp);
        if (cone_type == 1) {
          if (elliptic_count < nr) {
            elliptic_start[elliptic_count] = row_start;
            elliptic_dim[elliptic_count] = cdim;
            for (int k = 0; k < 5; ++k) elliptic_friction[elliptic_count * 5 + k] = contact_friction[s * 5 + k];
            ++elliptic_count;
          }
          for (int k = 0; k < cdim; ++k) {
            int row = row_start + k;
            elliptic_member[row] = true;
            int r = crbase + k * 6;
            if (jheader[2] == CCJ_DENSE)
              for (int i = 0; i < nv; ++i) ccj_set(J_world, row, i, contact_jacobian[cjbase + k * nv + i], nv);
            if (k == 0) {
              R[row] = normal_R;
              if (dims[20] == -1) dbg_pre[nr * nr + 2 * nr + row] = imp;
              lo[row] = 0.0f;
              hi[row] = INFINITY;
            } else {
              float mu = contact_friction[s * 5 + k - 1];
              float mu0 = contact_friction[s * 5];
              float tangent_R = normal_R / max(params[1], 1e-15f);
              R[row] = tangent_R * mu0 * mu0 / max(mu * mu, 1e-12f);
              if (dims[20] == -1) dbg_pre[nr * nr + 2 * nr + row] = imp;
              lo[row] = -INFINITY;
              hi[row] = INFINITY;
            }
            ar[row] = contact_row_data[r + 3];
            enabled[row] = true;
          }
        } else {
          if (cdim > 1 && contact_block_count < nr) {
            contact_block_start[contact_block_count] = row_start;
            contact_block_size[contact_block_count++] = 2 * (cdim - 1);
          }
          float mu0 = contact_friction[s * 5];
          float mu_master = mu0 / sqrt(max(params[1], 1e-15f));
          // MuJoCo first forms the first edge diagonal from D0 = tran *
          // (1 + mu0^2), then sets Rpy = 2 * mu^2 * R[i] with
          // mu = mu0 / sqrt(impratio).
          float first_edge_R = normal_R * (1.0f + mu0 * mu0);
          float pyramid_R = max(1e-15f,
              2.0f * mu_master * mu_master * first_edge_R);
          for (int axis = 0; axis < cdim - 1; ++axis) {
            float mu = contact_friction[s * 5 + axis];
            for (int side = 0; side < 2; ++side) {
              int k = axis * 2 + side;
              int row = row_start + k;
              float sign = side == 0 ? 1.0f : -1.0f;
              int r0 = crbase;
              int rt = crbase + (axis + 1) * 6;
              if (jheader[2] == CCJ_DENSE)
                for (int i = 0; i < nv; ++i) {
                  ccj_set(J_world, row, i, contact_jacobian[cjbase + i] + sign * mu * contact_jacobian[cjbase + (axis + 1) * nv + i], nv);
                }
              R[row] = pyramid_R;
              if (dims[20] == -1) dbg_pre[nr * nr + 2 * nr + row] = imp;
              // For condim=3 the contact producer has already formed the
              // pyramid edge velocity and reference (including the single
              // penetration term) in each edge slot. Higher condims retain
              // normal/tangent source rows here, so combine their velocity
              // references and add the normal position term only once.
              ar[row] = cdim == 3
                  ? contact_row_data[crbase + (k + 1) * 6 + 3]
                  : contact_row_data[r0 + 3] + sign * mu * contact_row_data[rt + 3];
              lo[row] = 0.0f;
              hi[row] = INFINITY;
              enabled[row] = true;
            }
          }
        }
      }
    }
  }

  // Flex contacts already arrive in canonical descriptor row order from the
  // flex row producer. Import their complete per-row coefficients, then add
  // the same cone grouping metadata used by rigid contacts. Flex slot metadata
  // is appended to contact_condim/contact_friction after the rigid slots.
  int flex_slot_count = dims[metadata_header + 11];
  int flex_row_offset = dims[metadata_header + 12];
  int flex_row_count = dims[metadata_header + 13];
  if (flex_slot_count > 0 && flex_row_count > 0
      && (reuse_rows_stage || ((flags & 1) == 0 && (flags & 16) == 0))) {
    for (int flex_slot = 0; flex_slot < flex_slot_count; ++flex_slot) {
      int slot = ncontacts_max + flex_slot;
      int cdim = contact_condim[slot * 3];
      int row_offset = contact_condim[slot * 3 + 1];
      int row_start = base_contact + row_offset;
      int row_span = cdim == 1 ? 1 : (cone_type == 0 ? 2 * (cdim - 1) : cdim);
      if (cdim < 1 || cdim > 6 || row_start < base_contact
          || row_start + row_span > nr
          || row_offset < flex_row_offset
          || row_offset + row_span > flex_row_offset + flex_row_count) {
        out_status[world] = 2;
        return;
      }
      bool slot_active = false;
      for (int k = 0; k < row_span; ++k) {
        int row = row_start + k;
        bool row_active = dbg_pre[nr * nr + 6 * nr + row] > 0.5f;
        if (row_active) {
          R[row] = dbg_pre[nr * nr + row];
          ar[row] = dbg_pre[nr * nr + nr + row];
          lo[row] = dbg_pre[nr * nr + 4 * nr + row];
          hi[row] = dbg_pre[nr * nr + 5 * nr + row];
          enabled[row] = true;
          slot_active = true;
        }
      }
      if (!slot_active || cdim == 1) continue;
      if (cone_type == 1) {
        if (elliptic_count >= nr) { out_status[world] = 2; return; }
        elliptic_start[elliptic_count] = row_start;
        elliptic_dim[elliptic_count] = cdim;
        for (int k = 0; k < 5; ++k)
          elliptic_friction[elliptic_count * 5 + k] = contact_friction[slot * 5 + k];
        ++elliptic_count;
        for (int k = 0; k < cdim; ++k) elliptic_member[row_start + k] = true;
      } else {
        if (contact_block_count >= nr) { out_status[world] = 2; return; }
        contact_block_start[contact_block_count] = row_start;
        contact_block_size[contact_block_count] = row_span;
        ++contact_block_count;
      }
    }
  }

  int total_nr = nr;


  // Sleeping kinematic trees have no current mass factor after mj_forwardSkip
  // and are omitted from the solve. Keep identity pivots in their slots so
  // the fixed dense ABI remains valid without touching the singular zero block.
  device const int* solver_awake_tree = reinterpret_cast<device const int*>(solver_scratch)
      + dims[27];
  solver_mask_sleeping_dofs(dims, J_world, solver_awake_tree, nv, total_nr);
  solver_filter_sleeping_rows(dims, J_world, solver_awake_tree, enabled,
      nv, nr, total_nr);

  // The sparse-component route splits assembly from the constraint solve.
  // Persist the canonical row state in the debug prefix, then return so the
  // caller can apply component-wise M^-1 to qfrc and the live J rows.
  if (dims[20] == -1) {
    for (int row = 0; row < nr; ++row) {
      dbg_pre[nr * nr + row] = R[row];
      dbg_pre[nr * nr + nr + row] = ar[row];
      // During assembly-only dispatch, the ordinary RHS slice temporarily
      // carries exact source impedance for DIAGEXACT. Inactive rows are
      // canonical zero, preventing stale values from prior assemblies.
      if (!enabled[row]) dbg_pre[nr * nr + 2 * nr + row] = 0.0f;
      dbg_pre[nr * nr + 4 * nr + row] = lo[row];
      dbg_pre[nr * nr + 5 * nr + row] = hi[row];
      dbg_pre[nr * nr + 6 * nr + row] = enabled[row] ? 1.0f : 0.0f;
    }
    return;
  }

  if (dims[20] == -2) {
    // Component result is a pair of row-major solution planes. This path
    // never reads or reconstructs the global dense mass matrix.
    for (int i = 0; i < nv; ++i) {
      int smooth_hi = component_pair_index(batch, world, nr, i, nr, nv, false);
      int smooth_lo = component_pair_index(batch, world, nr, i, nr, nv, true);
      out_acc[qb + i] = provided_qacc_smooth ? qfrc[qb + i] : mass[smooth_hi];
      out_acc[batch * nv + qb + i] = provided_qacc_smooth
          ? qfrc[batch * nv + qb + i] : mass[smooth_lo];
    }
    for (int row = 0; row < total_nr; ++row) if (enabled[row])
      for (int i = 0; i < nv; ++i)
        Z[row * nv + i] = mass[
            component_pair_index(batch, world, row, i, nr, nv, false)];
  } else if (dims[20] == -3) {
    if (!solver_factor_component_mass(mass, dbg_pre, dims,
        solver_awake_tree, world, nv)) { out_status[world] = 2; return; }
    device float* component_solution = solver_scratch;
    device float* component_work = solver_scratch + nv;
    if (provided_qacc_smooth) {
      for (int i = 0; i < nv; ++i) component_solution[i] = qfrc[qb + i];
    } else if (!solver_component_mass_solve(dbg_pre, dims, solver_awake_tree, nv,
        qfrc + qb, component_solution, component_work)) {
        out_status[world] = 2;
        return;
    }
    for (int i = 0; i < nv; ++i) out_acc[qb + i] = component_solution[i];
    device float* component_row_rhs = solver_scratch
        + dims[metadata_header + 15];
    for (int row = 0; row < total_nr; ++row) if (enabled[row]) {
      for (int dof = 0; dof < nv; ++dof)
        component_row_rhs[dof] = ccj_get(J_world, row, dof, nv);
      if (!solver_component_mass_solve(dbg_pre, dims, solver_awake_tree, nv,
          component_row_rhs, component_solution, component_work)) {
        out_status[world] = 2;
        return;
      }
      for (int i = 0; i < nv; ++i) Z[row * nv + i] = component_solution[i];
    }
  } else {
    // 3. Dense Cholesky factorization of the awake principal mass matrix.
    device const float* mass_world = dims[20] == -3 ? mass : mass + mb;
    for (int i = 0; i < nv; ++i) for (int j = 0; j < nv; ++j) L[i * nv + j] = 0.0f;
    for (int i = 0; i < nv; ++i) for (int j = 0; j <= i; ++j) {
      if (!solver_dof_awake(dims, solver_awake_tree, i)) {
        L[i * nv + j] = (i == j) ? 1.0f : 0.0f;
        continue;
      }
      if (!solver_dof_awake(dims, solver_awake_tree, j)) {
        L[i * nv + j] = 0.0f;
        continue;
      }
      float v = solver_mass_entry(mass_world, dbg_pre, dims,
          solver_awake_tree, world, nv, i, j);
      for (int k = 0; k < j; ++k) v -= L[i * nv + k] * L[j * nv + k];
      if (i == j) {
        if (!(v > 1e-12f) || !isfinite(v)) { out_status[world] = 2; return; }
        L[i * nv + j] = sqrt(v);
      } else {
        L[i * nv + j] = v / L[j * nv + j];
      }
    }

    // 4. Use the already-computed smooth acceleration when this is the
    // split VEL->ACC->CONSTRAINT path; ordinary forwards solve M q0 = qfrc.
    if (provided_qacc_smooth) {
      for (int i = 0; i < nv; ++i) {
        out_acc[qb + i] = qfrc[qb + i];
        out_acc[batch * nv + qb + i] = paired_qfrc_input
            ? qfrc[batch * nv + qb + i] : 0.0f;
      }
    } else if (paired_qfrc_input) {
      device float* pair_tail = solver_scratch + dims[metadata_header + 15];
      primal_cholesky_solve_pair(L, nv, solver_awake_tree, dims,
          qfrc + qb, qfrc + batch * nv + qb,
          out_acc + qb, out_acc + batch * nv + qb,
          y, pair_tail + 5 * max(nv, 1));
    } else {
      for (int i = 0; i < nv; ++i) {
        float v = solver_dof_awake(dims, solver_awake_tree, i) ? qfrc[qb + i] : 0.0f;
        for (int k = 0; k < i; ++k) v -= L[i * nv + k] * y[k];
        y[i] = v / L[i * nv + i];
      }
      for (int i = nv - 1; i >= 0; --i) {
        float v = y[i];
        for (int k = i + 1; k < nv; ++k) v -= L[k * nv + i] * x[k];
        x[i] = v / L[i * nv + i];
      }
      for (int i = 0; i < nv; ++i) out_acc[qb + i] = x[i];
    }

    // 5. Solve M Z_r = J_r^T for all active rows
    for (int row = 0; row < total_nr; ++row) if (enabled[row]) {
      for (int i = 0; i < nv; ++i) {
        float v = solver_dof_awake(dims, solver_awake_tree, i)
            ? ccj_get(J_world, row, i, nv) : 0.0f;
        for (int k = 0; k < i; ++k) v -= L[i * nv + k] * y[k];
        y[i] = v / L[i * nv + i];
      }
      for (int i = nv - 1; i >= 0; --i) {
        float v = y[i];
        for (int k = i + 1; k < nv; ++k) v -= L[k * nv + i] * x[k];
        x[i] = v / L[i * nv + i];
      }
      for (int i = 0; i < nv; ++i) Z[row * nv + i] = x[i];
    }
  }

  // 6. Form Delassus matrix W = J M^-1 J^T
  for (int a = 0; a < total_nr; ++a) if (enabled[a]) {
    for (int b = 0; b < total_nr; ++b) if (enabled[b]) {
      float v = 0.0f;
      for (int k = 0; k < nv; ++k) {
        v += ccj_get(J_world, a, k, nv) * Z[b * nv + k];
        if (dims[20] == -2)
          v += ccj_get(J_world, a, k, nv) * mass[
              component_pair_index(batch, world, b, k, nr, nv, true)];
      }
      W[a * nr + b] = v;
    }
  }

  // 7. Form RHS: ar - J q0
  if (solver_type == 0) {
    for (int row = 0; row < nr; ++row) {
      pgs_rhs_low[row] = 0.0f;
      pgs_lambda_low[row] = 0.0f;
    }
  }
  for (int row = 0; row < total_nr; ++row) {
    if (enabled[row]) {
      PrimalFloatPair ja = {0.0f, 0.0f};
      for (int k = 0; k < nv; ++k)
        ja = primal_pair_fma(ccj_get(J_world, row, k, nv),
            {out_acc[qb + k], out_acc[batch * nv + qb + k]}, ja);
      PrimalFloatPair rhs_pair = primal_pair_add(
          {ar[row], aref_low[row]}, primal_pair_multiply(ja, -1.0f));
      rhs[row] = rhs_pair.hi;
      if (solver_type == 0) pgs_rhs_low[row] = rhs_pair.lo;
    } else {
      rhs[row] = 0.0f;
      if (solver_type == 0) pgs_rhs_low[row] = 0.0f;
    }
  }

  // Pinned mj_warmstart maps the owned qacc_warmstart state through
  // mj_constraintUpdate before PGS's efc_AR cost check. It is not equivalent
  // to carrying an arbitrary previous multiplier vector across a row change.
  int solver_island_count = build_solver_island_maps(
      dims, dbg_pre, J_world, enabled, nv, nr, total_nr);
  device float* island_scratch = dbg_pre + nr * nr + 7 * nr;
  device int* island_scratch_i = reinterpret_cast<device int*>(island_scratch);
  device int* solver_row_island = island_scratch_i + dims[25] + max(nv, 1);
  if (solver_type == 0 && warm_ok && total_nr > 0) {
    int solver_metadata = dims[24] + dims[9] * 10;
    device const float* warm_acc = workspace_debug
        + world * max(dims[21], nr * nr + 7 * nr)
        + dims[solver_metadata + 5];
    seed_pgs_warmstart_qacc(J_world, nr, warm_acc, R, ar, lo, hi,
        enabled, lam, nv, total_nr, n_eq_rows, elliptic_count,
        elliptic_start, elliptic_dim, elliptic_friction);
  }

  // 7b. Warmstart cost check (pinned PGS rule): start from the retained
  // multipliers only when they improve on zero for the CURRENT Delassus
  // system; otherwise (released contacts, new manifolds, foreign vectors)
  // fall back to a cold start. Disabled rows never contribute.
  // Pinned: efc_b = J*qacc_smooth - aref = -rhs, cost = f.b + 0.5 f.A.f.
  if (solver_type == 0 && warm_ok && total_nr > 0) {
    PrimalFloatPair wcost = {0.0f, 0.0f};
    for (int row = 0; row < total_nr; ++row) if (enabled[row]) {
      wcost = primal_pair_add(wcost,
          primal_pair_multiply(primal_pair_product(
              {lam[row], pgs_lambda_low[row]},
              {rhs[row], pgs_rhs_low[row]}), -1.0f));
      for (int col = 0; col < total_nr; ++col) if (enabled[col]) {
        PrimalFloatPair a = primal_pair_add(
            pgs_delassus_pair(J_world, Z, mass, dims, batch, world,
                              nr, nv, row, col),
            {row == col ? R[row] : 0.0f, 0.0f});
        PrimalFloatPair term = primal_pair_product(
            primal_pair_product({lam[row], pgs_lambda_low[row]}, a),
            {lam[col], pgs_lambda_low[col]});
        wcost = primal_pair_add(wcost, primal_pair_multiply(term, 0.5f));
      }
    }
    if (primal_pair_value(wcost) > 0.0f || !isfinite(wcost.hi)) {
      for (int row = 0; row < total_nr; ++row) {
        lam[row] = 0.0f;
        pgs_lambda_low[row] = 0.0f;
      }
    }
  } else if (warm_ok && total_nr > 0) {
    // CG/Newton do not own the PGS low-word tail. Keep their pinned warm-start
    // cost check on the ordinary W/R high planes instead of reading scratch
    // that belongs to their live primal workspace.
    float wcost = 0.0f;
    for (int row = 0; row < total_nr; ++row) if (enabled[row]) {
      wcost -= lam[row] * rhs[row];
      for (int col = 0; col < total_nr; ++col) if (enabled[col]) {
        float a = W[row * nr + col] + (row == col ? R[row] : 0.0f);
        wcost += 0.5f * lam[row] * a * lam[col];
      }
    }
    if (wcost > 0.0f || !isfinite(wcost))
      for (int row = 0; row < total_nr; ++row) lam[row] = 0.0f;
  }

  // 8. Projected Gauss-Seidel on coupled W + diag(R)
  // params[2] remains the public float32 certification floor. Pinned 3.10
  // outer stopping uses the model-requested tolerance, carried separately in
  // params[6]; the certification floor must never relax that criterion.
  float tol = params[6];
  bool converged = false;
  bool primal_acceleration_output = false;
  float max_res = 0.0f;
  // Convergence history: 7 coarse main-sweep samples + final main-sweep
  // residual (refinement afterwards is a separate exact phase; the retained
  // diagnostics[0] is post-refinement). Unfilled slots repeat the final.
  thread float hist[8];
  thread int primal_island_iterations[8];
  for (int k = 0; k < 8; ++k) primal_island_iterations[k] = -1;
  int hsamp = 0;
  int hstep = max(1, maxiter / 7);
  for (int k = 0; k < 8; ++k) hist[k] = 0.0f;
  // Report the residual of the retained warm start even when iterations is
  // zero.  Exhausting a finite configured budget is diagnostic information,
  // not a numerical world failure in pinned MuJoCo semantics.
  for (int row = 0; row < total_nr; ++row) if (enabled[row] && !elliptic_member[row]) {
    float grad = -rhs[row];
    for (int col = 0; col < total_nr; ++col) if (enabled[col])
      grad += W[row * nr + col] * lam[col];
    grad += R[row] * lam[row];
    float diag = max(1e-15f, W[row * nr + row] + R[row]);
    float proj = clamp(lam[row] - grad / diag, lo[row], hi[row]);
    float scale = abs(ar[row]) + abs(R[row] * lam[row]);
    for (int col = 0; col < total_nr; ++col) if (enabled[col])
      scale += abs(W[row * nr + col] * lam[col]);
    max_res = max(max_res, abs(proj - lam[row]) * diag / max(1.0f, scale));
  }
  for (int b = 0; b < elliptic_count; ++b) {
    thread float mu[5];
    for (int k = 0; k < 5; ++k) mu[k] = elliptic_friction[b * 5 + k];
    max_res = max(max_res, cert_block_residual(elliptic_start[b], elliptic_dim[b],
        mu, W, R, rhs, lam, enabled, nr, total_nr));
  }
  if (solver_type == 0) {
    int used = 0;
    float pgs_residual = 0.0f;
    float pgs_mean_inertia = params[4];
    // Preserve MuJoCo's scale for every valid positive mean inertia; small
    // physical inertias are not replaced by an arbitrary unit scale.
    float pgs_improvement_scale = 1.0f
        / (pgs_mean_inertia * float(max(1, nv)));
    for (int island = 0; island < solver_island_count; ++island) {
      for (int row = 0; row < total_nr; ++row)
        enabled[row] = solver_row_island[row] == island;
      float island_residual = 0.0f;
      used += solve_pgs_device(W, R, rhs, pgs_rhs_low, ar, lo, hi, enabled,
          elliptic_member, lam, pgs_lambda_low, J_world, Z, mass, dims,
          batch, world, nv, nr, total_nr, maxiter, tol,
          pgs_improvement_scale, elliptic_count,
          elliptic_start, elliptic_dim, elliptic_friction,
          solver_scratch + dims[metadata_header + 9], &island_residual);
      pgs_residual = max(pgs_residual, island_residual);
    }
    for (int row = 0; row < total_nr; ++row)
      enabled[row] = solver_row_island[row] >= 0;
    max_res = pgs_residual;
    converged = max_res <= tol;
    out_diagnostics[world * 10] = max_res;
    out_diagnostics[world * 10 + 1] = float(used);
    for (int k = 0; k < 8; ++k) hist[k] = max_res;
    hsamp = 7;
  } else if (solver_type == 1 || solver_type == 2) {
    device float* primal_R = dbg_pre + nr * nr;
    device float* primal_aref = dbg_pre + nr * nr + nr;
    device float* primal_lambda = dbg_pre + nr * nr + 3 * nr;
    device float* primal_lo = dbg_pre + nr * nr + 4 * nr;
    device float* primal_hi = dbg_pre + nr * nr + 5 * nr;
    for (int row = 0; row < total_nr; ++row) {
      primal_R[row] = R[row];
      primal_aref[row] = ar[row];
      primal_lambda[row] = lam[row];
      primal_lo[row] = lo[row];
      primal_hi[row] = hi[row];
    }
    int used = 0;
    float primal_residual = 0.0f;
    bool partitioned = (dims[6] & dims[26]) == 0;
    device int* dof_island = island_scratch_i + dims[25];
    device float* primal_scratch = solver_scratch
        + dims[dims[24] + dims[9] * 10 + 9];
    device float* high_low_scratch = solver_scratch
        + dims[metadata_header + 15];
    device float* qacc_smooth_low = high_low_scratch + 7 * max(nv, 1);
    for (int i = 0; i < nv; ++i)
      qacc_smooth_low[i] = out_acc[batch * nv + qb + i];
    for (int row = 0; row < total_nr; ++row) primal_lambda[row] = 0.0f;
    // Preserve qacc_smooth across island solves. out_acc is the published
    // qacc destination and is updated island-by-island; out_force is not
    // populated until final J^T lambda reconstruction.
    for (int i = 0; i < nv; ++i) out_force[qb + i] = out_acc[qb + i];
    device const float* qacc_baseline = out_force + qb;
    for (int island = 0; island < solver_island_count; ++island) {
      for (int row = 0; row < total_nr; ++row)
        enabled[row] = solver_row_island[row] == island;
      // MuJoCo scales each island's primal residuals by the reciprocal of
      // that island's compiled mass diagonal sum.  The global mean-inertia
      // statistic is only used for the monolithic solver entry point.
      float island_inertia = 0.0f;
      for (int i = 0; i < nv; ++i) if (dof_island[i] == island) {
        island_inertia += dims[20] == -3
            ? solver_component_mass_entry(dbg_pre, dims, nv, i, i)
            : mass[mb + i * nv + i];
      }
      int island_used = solve_primal_accel_scalar(
          dims[20] == -3 ? mass : mass + mb, dbg_pre, J_world, primal_R,
          primal_aref, primal_lo, primal_hi, L, qacc_baseline,
          qacc_smooth_low,
          dims, solver_awake_tree, enabled,
          elliptic_member, elliptic_start, elliptic_dim, elliptic_friction,
          elliptic_count, primal_lambda, nv, nr, total_nr, n_eq_rows, solver_type, maxiter,
          dims[metadata_header + 14], params[5], tol,
          1.0f / max(island_inertia, 1e-15f),
          (flags & 512) == 0, primal_scratch, high_low_scratch,
          out_diagnostics + world * 10,
          dof_island, island, partitioned);
      if (island_used < 0) {
        for (int i = 0; i < nv; ++i) {
          out_force[qb + i] = 0.0f;
          out_acc[batch * nv + qb + i] = 0.0f;
        }
        out_status[world] = 2;
        return;
      }
      used += island_used;
      if (trace_solver_counts && island < 8)
        primal_island_iterations[island] = island_used;
      primal_residual = max(primal_residual, out_diagnostics[world * 10]);
      device const float* primal_acc = primal_scratch;
      for (int i = 0; i < nv; ++i) {
        if (!partitioned || dof_island[i] == island) {
          out_acc[qb + i] = primal_acc[i];
          out_acc[batch * nv + qb + i] = high_low_scratch[i];
        }
      }
      for (int row = 0; row < total_nr; ++row)
        if (solver_row_island[row] == island) lam[row] = primal_lambda[row];
    }
    for (int row = 0; row < total_nr; ++row)
      enabled[row] = solver_row_island[row] >= 0;
    primal_acceleration_output = true;
    max_res = primal_residual;
    converged = max_res <= tol;
    out_diagnostics[world * 10] = max_res;
    out_diagnostics[world * 10 + 1] = float(used);
    if (!trace_solver_counts) {
      for (int k = 0; k < 8; ++k) hist[k] = max_res;
      hsamp = 7;
    }
  } else if (elliptic_count > 0) {
    // Elliptic contacts are cone blocks in scaled coordinates. Solve the full
    // coupled quadratic with projected FISTA so interactions between multiple
    // manifold points are handled globally, rather than by slowly converging
    // local contact updates. Scalar joint/pyramid rows keep their own bounds.
    device float* cone_scratch = solver_scratch
        + dims[dims[24] + dims[9] * 10 + 9];
    device float* dscale = cone_scratch;
    device float* z = dscale + nr;
    device float* extrapolated = z + nr;
    device float* candidate = extrapolated + nr;
    device float* gradient_z = candidate + nr;
    device float* power_vector = gradient_z + nr;
    device float* power_product = power_vector + nr;
    for (int row = 0; row < total_nr; ++row) {
      dscale[row] = 1.0f;
      z[row] = extrapolated[row] = candidate[row] = 0.0f;
      gradient_z[row] = 0.0f;
    }
    for (int block = 0; block < elliptic_count; ++block) {
      int start = elliptic_start[block];
      int dim = elliptic_dim[block];
      for (int k = 1; k < dim; ++k)
        dscale[start + k] = max(elliptic_friction[block * 5 + k - 1], 0.0f);
    }
    float lipschitz = 1e-15f;
    int enabled_count = 0;
    for (int row = 0; row < total_nr; ++row) if (enabled[row]) {
      ++enabled_count;
      float row_sum = 0.0f;
      for (int col = 0; col < total_nr; ++col) if (enabled[col]) {
        float value = dscale[row] * W[row * nr + col] * dscale[col];
        if (row == col) value += dscale[row] * R[row] * dscale[col];
        row_sum += abs(value);
      }
      lipschitz = max(lipschitz, row_sum);
      float scale = dscale[row];
      z[row] = scale > 1e-12f ? lam[row] / scale : 0.0f;
      extrapolated[row] = z[row];
    }
    float gershgorin_bound = lipschitz;
    // The row-sum bound is safe but can be overly conservative for manifold
    // blocks. Estimate the dominant eigenvalue of the symmetric PSD scaled
    // Delassus matrix and retain a 10% safety margin for the global step.
    float inv_norm = rsqrt(float(max(enabled_count, 1)));
    for (int row = 0; row < total_nr; ++row)
      power_vector[row] = enabled[row] ? inv_norm : 0.0f;
    for (int iteration = 0; iteration < 24; ++iteration) {
      float norm_sq = 0.0f;
      for (int row = 0; row < total_nr; ++row) {
        float value = 0.0f;
        if (enabled[row]) {
          for (int col = 0; col < total_nr; ++col) if (enabled[col]) {
            float entry = dscale[row] * W[row * nr + col] * dscale[col];
            if (row == col) entry += dscale[row] * R[row] * dscale[col];
            value += entry * power_vector[col];
          }
        }
        power_product[row] = value;
        norm_sq += value * value;
      }
      float inv_power_norm = rsqrt(max(norm_sq, 1e-30f));
      for (int row = 0; row < total_nr; ++row)
        power_vector[row] = power_product[row] * inv_power_norm;
    }
    float rayleigh = 0.0f;
    for (int row = 0; row < total_nr; ++row) if (enabled[row]) {
      float value = 0.0f;
      for (int col = 0; col < total_nr; ++col) if (enabled[col]) {
        float entry = dscale[row] * W[row * nr + col] * dscale[col];
        if (row == col) entry += dscale[row] * R[row] * dscale[col];
        value += entry * power_vector[col];
      }
      rayleigh += power_vector[row] * value;
    }
    lipschitz = min(gershgorin_bound, max(1e-15f, rayleigh * 1.1f));
    float momentum = 1.0f;
    for (int it = 0; it < maxiter; ++it) {
      for (int row = 0; row < total_nr; ++row)
        lam[row] = enabled[row] ? dscale[row] * extrapolated[row] : 0.0f;
      for (int row = 0; row < total_nr; ++row) if (enabled[row]) {
        float gradient = -rhs[row];
        for (int col = 0; col < total_nr; ++col) if (enabled[col])
          gradient += W[row * nr + col] * lam[col];
        gradient += R[row] * lam[row];
        gradient_z[row] = dscale[row] * gradient;
      } else {
        gradient_z[row] = 0.0f;
      }
      // Backtracking makes the power-iteration step estimate safe even when
      // the transformed Delassus spectrum is clustered. The quadratic model
      // check guarantees that the accepted projected-gradient step majorizes
      // the actual coupled objective at this extrapolated point.
      bool step_accepted = false;
      for (int backtrack = 0; backtrack < 12; ++backtrack) {
        for (int row = 0; row < total_nr; ++row)
          candidate[row] = enabled[row]
              ? extrapolated[row] - gradient_z[row] / lipschitz : 0.0f;
        for (int block = 0; block < elliptic_count; ++block) {
          int start = elliptic_start[block];
          int dim = elliptic_dim[block];
          thread float cone_value[6];
          for (int k = 0; k < 6; ++k) cone_value[k] = k < dim ? candidate[start + k] : 0.0f;
          project_lorentz(cone_value, dim);
          for (int k = 0; k < dim; ++k) candidate[start + k] = cone_value[k];
        }
        for (int row = 0; row < total_nr; ++row)
          if (enabled[row] && !elliptic_member[row])
            candidate[row] = clamp(candidate[row], lo[row], hi[row]);

        float objective_base = 0.0f, objective_candidate = 0.0f;
        float linear_delta = 0.0f, delta_norm_sq = 0.0f;
        for (int row = 0; row < total_nr; ++row) if (enabled[row]) {
          float base_lambda = dscale[row] * extrapolated[row];
          float candidate_lambda = dscale[row] * candidate[row];
          float base_product = 0.0f;
          float candidate_product = 0.0f;
          for (int col = 0; col < total_nr; ++col) if (enabled[col]) {
            float base_col = dscale[col] * extrapolated[col];
            float candidate_col = dscale[col] * candidate[col];
            base_product += W[row * nr + col] * base_col;
            candidate_product += W[row * nr + col] * candidate_col;
          }
          base_product += R[row] * base_lambda;
          candidate_product += R[row] * candidate_lambda;
          objective_base += 0.5f * base_lambda * base_product - rhs[row] * base_lambda;
          objective_candidate += 0.5f * candidate_lambda * candidate_product - rhs[row] * candidate_lambda;
          float delta = candidate[row] - extrapolated[row];
          linear_delta += gradient_z[row] * delta;
          delta_norm_sq += delta * delta;
        }
        float majorizer = objective_base + linear_delta + 0.5f * lipschitz * delta_norm_sq;
        if (objective_candidate <= majorizer + 1e-6f * max(1.0f, abs(objective_candidate))) {
          step_accepted = true;
          break;
        }
        lipschitz *= 2.0f;
      }
      if (!step_accepted) { out_status[world] = 2; return; }

      float next_momentum = 0.5f * (1.0f + sqrt(1.0f + 4.0f * momentum * momentum));
      float beta = (momentum - 1.0f) / next_momentum;
      float restart_dot = 0.0f;
      for (int row = 0; row < total_nr; ++row) if (enabled[row])
        restart_dot += (candidate[row] - z[row]) * (extrapolated[row] - candidate[row]);
      if (restart_dot > 0.0f) {
        next_momentum = 1.0f;
        beta = 0.0f;
      }
      for (int row = 0; row < total_nr; ++row) if (enabled[row]) {
        float previous = z[row];
        z[row] = candidate[row];
        extrapolated[row] = candidate[row] + beta * (candidate[row] - previous);
        lam[row] = dscale[row] * z[row];
      }
      momentum = next_momentum;

      max_res = 0.0f;
      for (int row = 0; row < total_nr; ++row) if (enabled[row]) {
        if (elliptic_member[row]) {
          int block = -1;
          for (int b = 0; b < elliptic_count; ++b) if (elliptic_start[b] == row) block = b;
          if (block < 0) continue;
          int dim = elliptic_dim[block];
          thread float A[36], g[6], mu[5], force[6];
          for (int i = 0; i < 36; ++i) A[i] = 0.0f;
          for (int i = 0; i < 6; ++i) { g[i] = 0.0f; force[i] = 0.0f; }
          for (int i = 0; i < 5; ++i) mu[i] = elliptic_friction[block * 5 + i];
          for (int i = 0; i < dim; ++i) {
            int ri = row + i;
            force[i] = lam[ri];
            g[i] = -rhs[ri];
            for (int col = 0; col < total_nr; ++col)
              if (enabled[col] && (col < row || col >= row + dim))
                g[i] += W[ri * nr + col] * lam[col];
            for (int j = 0; j < dim; ++j) {
              int rj = row + j;
              A[i * 6 + j] = W[ri * nr + rj] + (i == j ? R[ri] : 0.0f);
            }
          }
          max_res = max(max_res, elliptic_projected_residual(A, g, mu, dim, force));
          continue;
        }
        float grad = -rhs[row];
        for (int col = 0; col < total_nr; ++col) if (enabled[col]) grad += W[row * nr + col] * lam[col];
        grad += R[row] * lam[row];
        float diag = max(1e-15f, W[row * nr + row] + R[row]);
        float proj = clamp(lam[row] - grad / diag, lo[row], hi[row]);
        float row_scale = abs(ar[row]) + abs(R[row] * lam[row]);
        for (int col = 0; col < total_nr; ++col) if (enabled[col]) row_scale += abs(W[row * nr + col] * lam[col]);
        row_scale = max(1.0f, row_scale);
        max_res = max(max_res, abs(proj - lam[row]) * diag / row_scale);
      }
      out_diagnostics[world * 10] = max_res;
      out_diagnostics[world * 10 + 1] = float(it + 1);
      if (hsamp < 7 && ((it + 1) % hstep == 0 || it + 1 == maxiter)) {
        hist[hsamp++] = max_res;
      }
      if (max_res <= tol) { converged = true; break; }
    }
  } else {
    // F4: redundant-contact systems certify slowly in float32, so a
    // budgeted PGS sweep can end close (<=1e-3) but uncertified while still
    // improving (box face plants freeze without this): grant another budget
    // window up to a hard cap. Acceptance still requires max_res <= tol;
    // true divergence keeps status 3 via the unchanged failure path below.
    // Loose-but-certified solutions are tightened by block refinement in
    // section 9; already-tight ones skip it bit-identically.
    int pgs_cap = maxiter;
    // G3: seed the progress comparison from the actual starting residual
    // (lam holds the retained warm vector, or zero when cold), not
    // from +inf, so the first extension must prove improvement over the
    // actual starting point.
    float win_start = 0.0f;
    for (int row = 0; row < total_nr; ++row) if (enabled[row]) {
      float grad = -rhs[row];
      for (int col = 0; col < total_nr; ++col) if (enabled[col])
        grad += W[row * nr + col] * lam[col];
      grad += R[row] * lam[row];
      float diag = max(1e-15f, W[row * nr + row] + R[row]);
      float proj = clamp(lam[row] - grad / diag, lo[row], hi[row]);
      float row_scale = abs(ar[row]) + abs(R[row] * lam[row]);
      for (int col = 0; col < total_nr; ++col) if (enabled[col])
        row_scale += abs(W[row * nr + col] * lam[col]);
      row_scale = max(1.0f, row_scale);
      win_start = max(win_start, abs(proj - lam[row]) * diag / row_scale);
    }
    for (int it = 0; it < pgs_cap; ++it) {
      for (int row = 0; row < total_nr; ++row) if (enabled[row]) {
        float diag = max(1e-15f, W[row * nr + row] + R[row]);
        float v = rhs[row];
        for (int col = 0; col < total_nr; ++col) if (enabled[col] && col != row)
          v -= W[row * nr + col] * lam[col];
        lam[row] = clamp(v / diag, lo[row], hi[row]);
      }
      max_res = 0.0f;
      for (int row = 0; row < total_nr; ++row) if (enabled[row]) {
        float grad = -rhs[row];
        for (int col = 0; col < total_nr; ++col) if (enabled[col]) grad += W[row * nr + col] * lam[col];
        grad += R[row] * lam[row];
        float diag = max(1e-15f, W[row * nr + row] + R[row]);
        float proj = clamp(lam[row] - grad / diag, lo[row], hi[row]);
        float row_scale = abs(ar[row]) + abs(R[row] * lam[row]);
        for (int col = 0; col < total_nr; ++col) if (enabled[col]) row_scale += abs(W[row * nr + col] * lam[col]);
        row_scale = max(1.0f, row_scale);
        max_res = max(max_res, abs(proj - lam[row]) * diag / row_scale);
      }
      out_diagnostics[world * 10] = max_res;
      out_diagnostics[world * 10 + 1] = float(it + 1);
      if (hsamp < 7 && (it + 1) % hstep == 0) hist[hsamp++] = max_res;
      if (max_res <= tol) {
        converged = true;
        break;
      }
    }
  }

  // Complete the convergence history: unfilled sample slots repeat the
  // final main-sweep residual; slot 7 always holds it.
  if (!trace_solver_counts) {
    while (hsamp < 7) hist[hsamp++] = max_res;
    hist[7] = max_res;
    for (int k = 0; k < 8; ++k)
      out_diagnostics[world * 10 + 2 + k] = hist[k];
  }

  // The configured outer solver budget is authoritative; no hidden refinement.

  // 9b. No-slip post-pass (pinned solNoSlip): exact friction subproblem
  // solves over dry-friction rows and contact friction blocks. Gated on
  // noslip_iterations (dims[18]); zero iterations skip bit-identically.
  // Improvement accounting and the noslip_tolerance stop rule (params[3])
  // mirror the pinned stage; iteration counts accumulate into diagnostics.
  int noslip_iters = dims[18];
  float noslip_tol = params[3];
  float mean_inertia = params[4];
  float noslip_scale = 1.0f
      / (mean_inertia * float(max(1, nv)));
  if (noslip_iters > 0 && total_nr > 0) {
    int ns_done = 0;
    for (int nsit = 0; nsit < noslip_iters; ++nsit) {
      float improvement = 0.0f;
      // At iteration 0, account for regularizer removal (pinned engine_solver.c:674-678)
      if (nsit == 0) {
        for (int row = 0; row < total_nr; ++row) {
          if (enabled[row]) {
            improvement += 0.5f * lam[row] * lam[row] * R[row];
          }
        }
      }

      // Dry friction rows: symmetric finite bounds identify joint and
      // tendon frictionloss rows (bilateral equalities use infinite
      // bounds; limits use [0, inf)). Regularizer R is excluded.
      for (int row = 0; row < total_nr; ++row) if (enabled[row]) {
        float lorb = lo[row], hib = hi[row];
        if (!(lorb < 0.0f && hib == -lorb && hib < INFINITY)) continue;
        float diag = max(1e-10f, W[row * nr + row]);
        float res = -rhs[row];
        for (int col = 0; col < total_nr; ++col) if (enabled[col]) res += W[row * nr + col] * lam[col];
        float old = lam[row];
        float v = clamp(old - res / diag, lorb, hib);
        float delta = v - old;
        float change = 0.5f * delta * delta * diag + delta * res;
        if (change > 1e-10f) {
          v = old;
          change = 0.0f;
        }
        lam[row] = v;
        pgs_lambda_low[row] = 0.0f;
        improvement -= change;
      }
      // Contact friction blocks.
      for (int b = 0; b < contact_block_count; ++b) {
        int row_start = contact_block_start[b];
        int block_size = contact_block_size[b];
        // Pyramidal edge pairs: exact 2D solve preserving the pair sum (R excluded).
        for (int k = 0; k + 1 < block_size; k += 2) {
          int j0 = row_start + k, j1 = row_start + k + 1;
          if (!(enabled[j0] && enabled[j1])) continue;
          float r0 = -rhs[j0], r1 = -rhs[j1];
          for (int col = 0; col < total_nr; ++col) if (enabled[col]) {
            r0 += W[j0 * nr + col] * lam[col];
            r1 += W[j1 * nr + col] * lam[col];
          }
          float old0 = lam[j0], old1 = lam[j1];
          float A00 = W[j0 * nr + j0], A11 = W[j1 * nr + j1];
          float A01 = W[j0 * nr + j1];
          float bc0 = r0 - (A00 * old0 + A01 * old1);
          float bc1 = r1 - (A01 * old0 + A11 * old1);
          float mid = 0.5f * (old0 + old1);
          float y = 0.5f * (old0 - old1);
          float K1 = A00 + A11 - 2.0f * A01;
          float K0 = mid * (A00 - A11) + bc0 - bc1;
          float ny = y;
          if (K1 < 1e-15f) {
            ny = y;
          } else {
            ny = -K0 / K1;
            ny = ny < -mid ? -mid : (ny > mid ? mid : ny);
          }
          float f0 = mid + ny, f1 = mid - ny;
          // costChange with unregularized residual r0, r1.
          float d0 = f0 - old0, d1 = f1 - old1;
          float change = 0.5f * (d0 * (A00 * d0 + A01 * d1) + d1 * (A01 * d0 + A11 * d1))
              + d0 * r0 + d1 * r1;
          if (change > 1e-10f) {
            f0 = old0;
            f1 = old1;
            change = 0.0f;
          }
          lam[j0] = f0;
          lam[j1] = f1;
          pgs_lambda_low[j0] = 0.0f;
          pgs_lambda_low[j1] = 0.0f;
          improvement -= change;
        }
      }
      // Elliptic friction blocks: QCQP over friction rows given normal (R excluded).
      for (int b = 0; b < elliptic_count; ++b) {
        int start = elliptic_start[b];
        int dim = elliptic_dim[b];
        if (!enabled[start]) continue;
        float fn = lam[start];
        if (fn < 1e-15f) {
          for (int k = 1; k < dim; ++k) {
            lam[start + k] = 0.0f;
            pgs_lambda_low[start + k] = 0.0f;
          }
          continue;
        }
        int nf = dim - 1;
        thread float Ac[36], bc[6], oldf[6], muf[5], res_arr[6];
        for (int i = 0; i < 36; ++i) Ac[i] = 0.0f;
        for (int i = 0; i < 6; ++i) { bc[i] = 0.0f; oldf[i] = 0.0f; res_arr[i] = 0.0f; }
        for (int i = 0; i < 5; ++i) muf[i] = elliptic_friction[b * 5 + i];
        for (int i = 0; i < nf; ++i) {
          int ri = start + 1 + i;
          oldf[i] = primal_pair_value({lam[ri], pgs_lambda_low[ri]});
          float v = -rhs[ri];
          for (int col = 0; col < total_nr; ++col) if (enabled[col]) v += W[ri * nr + col] * lam[col];
          res_arr[i] = v;
          bc[i] = v;
          for (int j = 0; j < nf; ++j) {
            Ac[i * nf + j] = W[ri * nr + start + 1 + j];
            bc[i] -= Ac[i * nf + j] * lam[start + 1 + j];
          }
        }
        thread float qv[6];
        for (int i = 0; i < 6; ++i) qv[i] = 0.0f;
        int active = 0;
        if (nf == 2) active = nsl_qcqp2(qv, Ac, bc, muf, fn);
        else if (nf == 3) active = nsl_qcqp3(qv, Ac, bc, muf, fn);
        else active = nsl_qcqpn(qv, Ac, bc, muf, fn, nf);
        if (active) {
          float s = 0.0f;
          for (int j = 0; j < nf; ++j) s += qv[j] * qv[j] / max(muf[j] * muf[j], 1e-30f);
          s = sqrt(fn * fn / max(1e-15f, s));
          for (int j = 0; j < nf; ++j) qv[j] *= s;
        }
        for (int i = 0; i < nf; ++i) {
          lam[start + 1 + i] = qv[i];
          pgs_lambda_low[start + 1 + i] = 0.0f;
        }
        // costChange uses unregularized residual res_arr, matching pinned engine_solver.c.
        float change = 0.0f;
        for (int i = 0; i < nf; ++i) {
          float delta = lam[start + 1 + i] - oldf[i];
          float ad = 0.0f;
          for (int j = 0; j < nf; ++j) ad += Ac[i * nf + j] * delta;
          change += 0.5f * delta * ad + delta * res_arr[i];
        }
        if (change > 1e-10f) {
          for (int i = 0; i < nf; ++i) {
            lam[start + 1 + i] = oldf[i];
            pgs_lambda_low[start + 1 + i] = 0.0f;
          }
          change = 0.0f;
        }
        improvement -= change;
      }
      ns_done++;
      improvement *= noslip_scale;
      if (improvement < noslip_tol) break;
    }
    out_diagnostics[world * 10 + 1] += float(ns_done);
    if (trace_solver_counts) {
      out_diagnostics[world * 10 + 2] =
          out_diagnostics[world * 10 + 1] - float(ns_done);
      out_diagnostics[world * 10 + 3] = float(ns_done);
    }
    // Note: No-slip solves the unregularized friction subproblem and deliberately
    // deviates from the regularized QP optimum. Do not overwrite `converged` with
    // a regularized KKT certificate here (pinned semantics, R04).
    bool lam_finite = true;
    for (int r = 0; r < total_nr; ++r) {
      if (enabled[r] && !isfinite(lam[r])) { lam_finite = false; break; }
    }
    if (!lam_finite) converged = false;
  }

  // MuJoCo applies no-slip after every dual solver.  For CG/Newton the
  // retained primal iterate predates that multiplier update, so restore the
  // smooth baseline; final force reconstruction below then performs the
  // equivalent dual-finish operation on the updated multipliers.
  if (primal_acceleration_output && noslip_iters > 0 && total_nr > 0) {
    device float* restore_a = y;
    device float* restore_b = x;
    if (dims[20] == -3) {
      device float* pair_tail = solver_scratch
          + dims[metadata_header + 15];
      restore_a = pair_tail + 4 * max(nv, 1);
      restore_b = pair_tail + 5 * max(nv, 1);
    }
    if (!solver_restore_smooth_acceleration(out_acc, qfrc, mass, dbg_pre,
        L, restore_a, restore_b, dims, solver_awake_tree, batch, world, nv,
        nr, qb, provided_qacc_smooth, paired_qfrc_input)) {
      out_status[world] = 2;
      for (int i = 0; i < nv; ++i) {
        out_force[qb + i] = 0.0f;
        out_acc[qb + i] = 0.0f;
        out_acc[batch * nv + qb + i] = 0.0f;
      }
      return;
    }
    primal_acceleration_output = false;
  }

  // solver_diagnostics[0] certifies the multipliers that will be retained
  // and published below.  The optimizer's scaled-gradient/cost-improvement
  // metric remains in solver_history and is not a dual-feasibility claim.
  out_diagnostics[world * 10] = cert_retained_system_residual(W, R, rhs, ar,
      lo, hi, lam, enabled, elliptic_member, elliptic_start, elliptic_dim,
      elliptic_friction, elliptic_count, nr, total_nr);

  // G2: nonfinite residuals fail explicitly (NaN never satisfies `> tol`,
  // so without this guard a nonfinite solve would report success).
  bool finite_solution = isfinite(max_res);
  for (int r = 0; r < total_nr; ++r)
    if (enabled[r] && !isfinite(lam[r])) finite_solution = false;
  if (!finite_solution) out_status[world] = 3;

  // 10. Reconstruct forces and acceleration
  for (int i = 0; i < nv; ++i) {
    out_force[qb + i] = 0.0f;
    pgs_pair_tail[i] = 0.0f;
  }
  for (int row = 0; row < total_nr; ++row) if (enabled[row]) {
    float lambda_low = solver_type == 0 ? pgs_lambda_low[row] : 0.0f;
    for (int i = 0; i < nv; ++i) {
      PrimalFloatPair force = primal_pair_fma(ccj_get(J_world, row, i, nv),
          {lam[row], lambda_low}, {out_force[qb + i], pgs_pair_tail[i]});
      out_force[qb + i] = force.hi;
      pgs_pair_tail[i] = force.lo;
    }
  }
  for (int row = 0; row < total_nr; ++row) if (enabled[row]) {
    if (!primal_acceleration_output) {
      for (int i = 0; i < nv; ++i) {
        PrimalFloatPair acceleration = primal_pair_fma(
            Z[row * nv + i],
            {lam[row], solver_type == 0 ? pgs_lambda_low[row] : 0.0f},
            {out_acc[qb + i], out_acc[batch * nv + qb + i]});
        if (dims[20] == -2) {
          // Component PGS retains the mass-solve residual correction for each
          // J^T row in the low half of the borrowed result pair. Carry that
          // half through the final lambda accumulation as well as Delassus W;
          // otherwise qacc and W describe different solved operators.
          float z_low = mass[component_pair_index(
              batch, world, row, i, nr, nv, true)];
          acceleration = primal_pair_fma(
              z_low,
              {lam[row], solver_type == 0 ? pgs_lambda_low[row] : 0.0f},
              acceleration);
        }
        out_acc[qb + i] = acceleration.hi;
        out_acc[batch * nv + qb + i] = acceleration.lo;
      }
    }
  }

  // Dense PGS forms qacc by summing independently rounded M^-1 J^T columns.
  // Recover the residual of that accepted iterate against the original mass
  // equation, then apply one bounded Cholesky correction in paired precision.
  // This carries the solve residual into qacc_low for downstream finite
  // differences; it does not change PGS multipliers or its iteration budget.
  if (solver_type == 0 && dims[20] != -2 && dims[20] != -3) {
    device float* pair_tail = solver_scratch + dims[metadata_header + 15];
    device float* constraint_force_low = pair_tail;
    device float* residual_hi = pair_tail + nv;
    device float* residual_low = pair_tail + 2 * nv;
    device float* correction_hi = pair_tail + 3 * nv;
    device float* correction_low = pair_tail + 4 * nv;
    device float* solve_y_low = pair_tail + 5 * nv;
    device float* solve_y_hi = y;
    device const float* mass_world = mass + mb;
    for (int i = 0; i < nv; ++i) {
      PrimalFloatPair force = {0.0f, 0.0f};
      for (int row = 0; row < total_nr; ++row) if (enabled[row])
        force = primal_pair_fma(ccj_get(J_world, row, i, nv),
            {lam[row], pgs_lambda_low[row]}, force);
      out_force[qb + i] = force.hi;
      constraint_force_low[i] = force.lo;
    }
    for (int i = 0; i < nv; ++i) {
      if (!solver_dof_awake(dims, solver_awake_tree, i)) {
        residual_hi[i] = 0.0f;
        residual_low[i] = 0.0f;
        continue;
      }
      PrimalFloatPair rhs = {0.0f, 0.0f};
      if (provided_qacc_smooth) {
        for (int k = 0; k < nv; ++k)
          rhs = primal_pair_fma(solver_mass_entry(mass_world, dbg_pre,
              dims, solver_awake_tree, world, nv, i, k),
              {qfrc[qb + k], paired_qfrc_input
                  ? qfrc[batch * nv + qb + k] : 0.0f}, rhs);
      } else {
        rhs = {qfrc[qb + i], paired_qfrc_input
            ? qfrc[batch * nv + qb + i] : 0.0f};
      }
      rhs = primal_pair_add(rhs,
          {out_force[qb + i], constraint_force_low[i]});
      PrimalFloatPair mass_product = {0.0f, 0.0f};
      for (int k = 0; k < nv; ++k)
        mass_product = primal_pair_fma(solver_mass_entry(mass_world,
            dbg_pre, dims, solver_awake_tree, world, nv, i, k),
            {out_acc[qb + k], out_acc[batch * nv + qb + k]}, mass_product);
      PrimalFloatPair residual = primal_pair_add(rhs,
          primal_pair_multiply(mass_product, -1.0f));
      residual_hi[i] = residual.hi;
      residual_low[i] = residual.lo;
    }
    primal_cholesky_solve_pair(L, nv, solver_awake_tree, dims,
        residual_hi, residual_low, correction_hi, correction_low,
        solve_y_hi, solve_y_low);
    for (int i = 0; i < nv; ++i) {
      PrimalFloatPair corrected = primal_pair_add(
          {out_acc[qb + i], out_acc[batch * nv + qb + i]},
          {correction_hi[i], correction_low[i]});
      out_acc[qb + i] = corrected.hi;
      out_acc[batch * nv + qb + i] = corrected.lo;
    }
  }

  // 11. Write joint forces
  for (int row = 0; row < base_contact; ++row) {
    // out_joint_force already points at this world's joint-force slice.
    out_joint_force[row] = enabled[row] ? lam[row] : 0.0f;
  }

  // 12. Write contact forces
  for (int s = 0; s < ncontacts_max; ++s) {
    int cdim = contact_condim[s * 3 + 0];
    int row_offset = contact_condim[s * 3 + 1];
    int row_start = base_contact + row_offset;
    int ofb = (world * ncontacts_max + s) * 11;
    for (int k = 0; k < 11; ++k) out_contact_force[ofb + k] = 0.0f;
    if (cdim == 0 || !enabled[row_start]) continue;
    if (cdim == 1) {
      out_contact_force[ofb] = lam[row_start];
    } else if (cone_type == 1) {
      for (int k = 0; k < cdim; ++k) out_contact_force[ofb + k] = enabled[row_start + k] ? lam[row_start + k] : 0.0f;
    } else {
      int edge_count = 2 * (cdim - 1);
      float normal = 0.0f;
      for (int k = 0; k < edge_count; ++k) {
        float value = lam[row_start + k];
        normal += value;
        out_contact_force[ofb + 1 + k] = value;
      }
      out_contact_force[ofb] = normal;
    }
  }


  // Publish per-island primal counts after all solver/history writers. The
  // no-slip split owns slots 2-3 when that stage ran; otherwise these slots
  // carry the retained per-island counts captured above.
  if (trace_solver_counts && !trace_primal_details
      && (solver_type == 1 || solver_type == 2)
      && !(noslip_iters > 0 && total_nr > 0)) {
    for (int k = 0; k < 8; ++k)
      out_diagnostics[world * 10 + 2 + k] = float(primal_island_iterations[k]);
  }

  // 13. Write debug matrices and vectors if workspace_debug is provided
  if (workspace_debug && nr > 0) {
    device float* dbg = workspace_debug + world * max(dims[21], nr * nr + 7 * nr);
    for (int a = 0; a < nr; ++a) {
      for (int b = 0; b < nr; ++b) {
        dbg[a * nr + b] = (a < total_nr && b < total_nr) ? W[a * nr + b] : 0.0f;
      }
      dbg[nr * nr + a] = a < total_nr ? R[a] : 0.0f;
      dbg[nr * nr + nr + a] = a < total_nr ? ar[a] : 0.0f;
      dbg[nr * nr + 2 * nr + a] = a < total_nr ? rhs[a] : 0.0f;
      dbg[nr * nr + 3 * nr + a] = (a < total_nr && enabled[a]) ? lam[a] : 0.0f;
      // Retain joint/contact row bounds for host-side KKT residual checks.
      // Equality and tendon-owned rows keep their preassembled bounds.
      if (a < total_nr && a >= n_eq_rows && dbg[nr * nr + 6 * nr + a] <= 0.5f) {
        dbg[nr * nr + 4 * nr + a] = lo[a];
        dbg[nr * nr + 5 * nr + a] = hi[a];
      }
    }
  }
}

// Pack canonical active constraint Jacobians followed by qfrc_smooth as
// fixed-capacity component-solver right hand sides. The final row is reserved
// even for models without any constraints.
kernel void pack_component_mass_rhs(
    device const float* workspace_J [[buffer(0)]],
    device const float* qfrc [[buffer(1)]],
    device const float* workspace_debug [[buffer(2)]],
    device float* rhs [[buffer(3)]],
    constant int* dims [[buffer(4)]],
    uint world [[thread_position_in_grid]]) {
  int nv = dims[1];
  int nr = dims[9];
  int batch = dims[5];
  if (int(world) >= batch) return;
  if (!cc_solver_world_enabled(dims, int(world))) return;
  int debug_stride = dims[21];
  bool qacc_smooth_ready = (dims[7] & 4) != 0;
  int debug_base = int(world) * debug_stride;
  device const float* J_world = ccj_world(workspace_J, int(world), nr, nv, dims);
  int rhs_base = int(world) * (nr + 1) * nv;
  for (int row = 0; row < nr; ++row) {
    bool active = workspace_debug[debug_base + nr * nr + 6 * nr + row] > 0.5f;
    for (int dof = 0; dof < nv; ++dof)
      rhs[rhs_base + row * nv + dof] = active
          ? ccj_get(J_world, row, dof, nv) : 0.0f;
  }
  int qfrc_base = int(world) * nv;
  for (int dof = 0; dof < nv; ++dof)
    rhs[rhs_base + nr * nv + dof] = qacc_smooth_ready
        ? 0.0f : qfrc[qfrc_base + dof];
}

// Tendon constraint rows for MuJoCo 3.10.0 (milestone 008).
// Pinned sources: engine/engine_core_constraint.c (tendon-limit loop with
// side-scaled Jacobian, friction-tendon rows, JOINT/TENDON cubic equality with
// tendon_length0 reference and tendon_invweight0 diagonal).
// Assembles tendon equality rows (eq region, cubic coupling), tendon
// frictionloss rows and tendon limit rows (reserved ten region) into
// workspace_J, and writes R/ar/lo/hi/enabled into the extended debug regions
// [4*nr, 7*nr). Runs after equality_assembly (which reserves zeros for tendon
// equalities) and before solve_coupled_constraints (which reads rows flagged
// in ten_en instead of computing them).
kernel void tendon_constraint_rows(
    device const float* qpos [[buffer(0)]],
    device const float* qvel [[buffer(1)]],
    device const float* ten_J_spatial [[buffer(2)]],
    device const float* ten_length_spatial [[buffer(3)]],
    device const float* ten_length_map [[buffer(4)]],
    device const float* ten_moment_map [[buffer(5)]],
    device const int* ten_limited [[buffer(6)]],
    device const float* ten_range [[buffer(7)]],
    device const float* ten_margin [[buffer(8)]],
    device const float* ten_length0 [[buffer(9)]],
    device const float* ten_invweight0 [[buffer(10)]],
    device const float* ten_solref_lim [[buffer(11)]],
    device const float* ten_solimp_lim [[buffer(12)]],
    device const float* ten_frictionloss [[buffer(13)]],
    device const float* ten_solref_fri [[buffer(14)]],
    device const float* ten_solimp_fri [[buffer(15)]],
    device const int* eq_type [[buffer(16)]],
    device const int* eq_obj [[buffer(17)]],
    device const float* eq_data [[buffer(18)]],
    device const float* eq_sol_params [[buffer(19)]],
    device const int* eq_rowadr [[buffer(20)]],
    device const int* eq_active [[buffer(21)]],
    constant int* dims [[buffer(22)]],
    constant float* sparams [[buffer(23)]],
    device float* workspace_J [[buffer(24)]],
    device float* workspace_debug [[buffer(25)]],
    uint world [[thread_position_in_grid]]) {
  int nq=dims[0], nv=dims[1], nt=dims[15], neq=dims[3], batch=dims[5];
  int flags=dims[6];
  bool refsafe=(dims[7]&1)!=0;
  int nr=dims[9];
  int ten_base=dims[16];
  float timestep=sparams[0];
  if (uint(world)>=uint(batch)) return;
  if (!cc_solver_world_enabled(dims, int(world))) return;
  if (nr<=0||nv<=0) return;
  device float* Jw = ccj_world(workspace_J, int(world), nr, nv, dims);
  device float* dbg = workspace_debug + uint(world)*uint(max(dims[21], nr*nr+7*nr));
  uint vbase=uint(world)*uint(max(nv,1)), qbase=uint(world)*uint(max(nq,1));
  uint tbase=uint(world)*uint(max(nt,1));
  // This kernel owns only the reserved tendon interval [ten_base,
  // ten_base + tendon_row_count), plus the single row of each tendon
  // equality.  Contact and flex rows are produced by later, independent
  // writers; clearing metadata across all nr here erased their cached-row
  // ownership and made VEL-only refresh silently drop valid equalities.
  int ten_stop = min(nr, ten_base + max(dims[17], 0));
  for (int r=max(0, ten_base);r<ten_stop;++r) {
    for (int d=0; d<nv; ++d) ccj_set(Jw, r, d, 0.0f, nv);
    dbg[nr*nr+r]=0.0f;
    dbg[nr*nr+nr+r]=0.0f;
    dbg[nr*nr+4*nr+r]=0.0f;
    dbg[nr*nr+5*nr+r]=0.0f;
    dbg[nr*nr+6*nr+r]=0.0f;
    dbg[nr*nr+2*nr+r]=0.0f;
  }
  // Tendon equalities live in the equality prefix, so clear only those
  // producer-owned rows while preserving activity and bounds emitted by the
  // joint/connect/weld equality producer earlier in this same assembly.
  for (int e=0;e<neq;++e) {
    if (eq_type[e] == 3) {
      int row = eq_rowadr[e];
      if (row >= 0 && row < nr) {
        for (int d=0; d<nv; ++d) ccj_set(Jw, row, d, 0.0f, nv);
        dbg[nr*nr+row]=0.0f;
        dbg[nr*nr+nr+row]=0.0f;
        dbg[nr*nr+2*nr+row]=0.0f;
        dbg[nr*nr+4*nr+row]=0.0f;
        dbg[nr*nr+5*nr+row]=0.0f;
        dbg[nr*nr+6*nr+row]=0.0f;
      }
    }
  }
  if (nt<=0) return;
  if ((flags&1)!=0) return;
  // Per-tendon combined length and dense Jacobian row.  Evaluate each
  // component directly from the immutable moment/spatial maps: nv is not
  // bounded by a thread-local scratch vector on the scalable path.
  for (int t=0;t<nt;++t) {
    float L=0.0f;
    for (int q=0;q<nq;++q) L+=ten_length_map[t*max(nq,1)+q]*qpos[qbase+uint(q)];
    L+=ten_length_spatial[tbase+uint(t)];
    float vel=0.0f;
    for (int d=0;d<nv;++d) {
      float jd=ten_moment_map[t*max(nv,1)+d]+
        ten_J_spatial[(tbase+uint(t))*uint(max(nv,1))+uint(d)];
      vel+=jd*qvel[vbase+uint(d)];
    }
    // Friction loss row (pinned: J=ten_J, margin 0, bound loss).
    if ((flags&4)==0 && ten_frictionloss[t]>0.0f) {
      int slot=0;
      for (int u=0;u<t;++u) if (ten_frictionloss[u]>0.0f) slot++;
      int row=ten_base+slot;
      if (row>=0&&row<nr) {
        for (int d=0;d<nv;++d)
          ccj_set(Jw, row, d, ten_moment_map[t*max(nv,1)+d]+
            ten_J_spatial[(tbase+uint(t))*uint(max(nv,1))+uint(d)], nv);
        float R, ar;
        reference_params(ten_solref_fri+t*2, ten_solimp_fri+t*5, 0, 0.0f, 0.0f,
          vel, ten_invweight0[t], true, timestep, refsafe, R, ar);
        dbg[nr*nr+2*nr+row]=impedance_at(ten_solimp_fri+t*5, 0, 0.0f, 0.0f);
        dbg[nr*nr+row]=R; dbg[nr*nr+nr+row]=ar;
        dbg[nr*nr+4*nr+row]=-ten_frictionloss[t];
        dbg[nr*nr+5*nr+row]=ten_frictionloss[t];
        dbg[nr*nr+6*nr+row]=1.0f;
      }
    }
    // Limit rows (pinned: J scaled by -side, dist=side*(range-value)).
    if ((flags&8)==0 && ten_limited[t]!=0) {
      int nfric=0;
      for (int u=0;u<nt;++u) if (ten_frictionloss[u]>0.0f) nfric++;
      int pre=0;
      for (int u=0;u<t;++u) if (ten_limited[u]!=0) pre++;
      for (int s=0;s<2;++s) {
        float side=s==0?-1.0f:1.0f;
        float rangev=s==0?ten_range[2*t]:ten_range[2*t+1];
        float dist=side*(rangev-L);
        if (dist<ten_margin[t]) {
          int row=ten_base+nfric+2*pre+s;
          if (row>=0&&row<nr) {
            for (int d=0;d<nv;++d)
              ccj_set(Jw, row, d, -side*(ten_moment_map[t*max(nv,1)+d]+
                ten_J_spatial[(tbase+uint(t))*uint(max(nv,1))+uint(d)]), nv);
            float R, ar;
            reference_params(ten_solref_lim+t*2, ten_solimp_lim+t*5, 0, dist,
              ten_margin[t], -side*vel, ten_invweight0[t], false, timestep, refsafe, R, ar);
            dbg[nr*nr+2*nr+row]=impedance_at(
                ten_solimp_lim+t*5, 0, dist, ten_margin[t]);
            dbg[nr*nr+row]=R; dbg[nr*nr+nr+row]=ar;
            dbg[nr*nr+4*nr+row]=0.0f;
            dbg[nr*nr+5*nr+row]=INFINITY;
            dbg[nr*nr+6*nr+row]=1.0f;
          }
        }
      }
    }
  }
  // Tendon equality rows (pinned cubic coupling, margin 0).
  if ((flags&2)!=0) return;
  for (int e=0;e<neq;++e) {
    if (eq_type[e]!=3) continue;
    if (eq_active[uint(world)*uint(max(neq,1))+uint(e)]==0) continue;
    int t1=eq_obj[2*e], t2=eq_obj[2*e+1];
    if (t1<0||t1>=nt) continue;
    int row=eq_rowadr[e];
    if (row<0||row>=nr) continue;
    // combined length/J for t1
    float L1=0.0f;
    for (int q=0;q<nq;++q) L1+=ten_length_map[t1*max(nq,1)+q]*qpos[qbase+uint(q)];
    L1+=ten_length_spatial[tbase+uint(t1)];
    float pos=L1-ten_length0[t1]-eq_data[e*11];
    float vel=0.0f;
    for (int d=0;d<nv;++d) {
      float j1=ten_moment_map[t1*max(nv,1)+d]+
        ten_J_spatial[(tbase+uint(t1))*uint(max(nv,1))+uint(d)];
      vel+=j1*qvel[vbase+uint(d)];
    }
    float diag=ten_invweight0[t1];
    for (int d=0;d<nv;++d)
      ccj_set(Jw, row, d, ten_moment_map[t1*max(nv,1)+d]+
        ten_J_spatial[(tbase+uint(t1))*uint(max(nv,1))+uint(d)], nv);
    if (t2>=0&&t2<nt) {
      float L2=0.0f;
      for (int q=0;q<nq;++q) L2+=ten_length_map[t2*max(nq,1)+q]*qpos[qbase+uint(q)];
      L2+=ten_length_spatial[tbase+uint(t2)];
      float dif=L2-ten_length0[t2];
      float poly=0.0f, deriv=0.0f, power=dif;
      for (int k=0;k<4;++k) {
        poly+=eq_data[e*11+k+1]*power;
        deriv+=float(k+1)*eq_data[e*11+k+1]*pow(dif,float(k));
        power*=dif;
      }
      pos-=poly;
      float vel2=0.0f;
      for (int d=0;d<nv;++d) {
        float j1=ten_moment_map[t1*max(nv,1)+d]+
          ten_J_spatial[(tbase+uint(t1))*uint(max(nv,1))+uint(d)];
        float j2=ten_moment_map[t2*max(nv,1)+d]+
          ten_J_spatial[(tbase+uint(t2))*uint(max(nv,1))+uint(d)];
        vel2+=j2*qvel[vbase+uint(d)];
        ccj_set(Jw, row, d, j1-deriv*j2, nv);
      }
      vel-=deriv*vel2;
      diag+=ten_invweight0[t2];
    }
    float d_width=max(1e-15f, eq_sol_params[e*14+3]);
    float imp=eq_impedance(eq_sol_params, e, pos, 0.0f);
    EqFloatPair imp_pair=eq_impedance_pair(eq_sol_params, e, pos, 0.0f);
    float r0=eq_sol_params[e*14], r1=eq_sol_params[e*14+1];
    if (refsafe&&r0>0.0f) r0=max(r0,2.0f*timestep);
    float K=r0>0.0f?1.0f/max(1e-15f,d_width*d_width*r0*r0*r1*r1):-r0/max(1e-15f,d_width*d_width);
    float B=r1>0.0f?2.0f/max(1e-15f,d_width*r0):-r1/d_width;
    dbg[nr*nr+row]=max(1e-15f,(1.0f-imp)*diag/imp);
    EqFloatPair aref_pair=eq_aref_pair(eq_sol_params, e,
        {pos, 0.0f}, {vel, 0.0f}, imp_pair, 0.0f, refsafe, timestep);
    dbg[nr*nr+nr+row]=aref_pair.hi;
    dbg[nr*nr+2*nr+row]=imp;
    int solver_metadata=dims[24]+nr*10;
    device float* solver_scratch=dbg+nr*nr+7*nr;
    device float* pair_tail=solver_scratch+dims[solver_metadata+15];
    device float* aref_low=pair_tail+7*max(nv,1)+2*max(nr,1);
    aref_low[row]=aref_pair.lo;
    dbg[nr*nr+4*nr+row]=-INFINITY;
    dbg[nr*nr+5*nr+row]=INFINITY;
    dbg[nr*nr+6*nr+row]=1.0f;
  }
}

// Device-memory overload of cert_block_residual for the scalable block solver.
inline float cert_block_residual(int start, int dim, thread const float* mu,
    device const float* W, device const float* R, device const float* rhs,
    device const float* lam, device const int* enabled,
    int nr, int total_nr) {
  thread float A[36], g[6], frc[6], mulo[5];
  for (int i = 0; i < 36; ++i) A[i] = 0.0f;
  for (int i = 0; i < 6; ++i) { g[i] = 0.0f; frc[i] = 0.0f; }
  for (int i = 0; i < 5; ++i) mulo[i] = mu[i];
  for (int i = 0; i < dim; ++i) {
    int ri = start + i;
    frc[i] = lam[ri];
    g[i] = -rhs[ri];
    for (int col = 0; col < total_nr; ++col)
      if (enabled[col] && (col < start || col >= start + dim))
        g[i] += W[ri * nr + col] * lam[col];
    for (int j = 0; j < dim; ++j) {
      int cj = start + j;
      A[i * 6 + j] = W[ri * nr + cj] + (i == j ? R[ri] : 0.0f);
    }
  }
  return elliptic_projected_residual(A, g, mulo, dim, frc);
}

// Re-certify the multiplier retained by the public solve against the exact
// assembled row system. Primal CG/Newton stop on their scaled objective
// gradient, while PGS stops on cost improvement; neither quantity certifies
// the retained dual force. This diagnostic must not affect solver stopping.
inline float cert_retained_system_residual(device const float* W,
    device const float* R, device const float* rhs, device const float* ar,
    device const float* lo, device const float* hi, device const float* lam,
    device const int* enabled, device const int* elliptic_member,
    device const int* elliptic_start, device const int* elliptic_dim,
    device const float* elliptic_friction, int elliptic_count,
    int nr, int total_nr) {
  float residual = 0.0f;
  for (int row = 0; row < total_nr; ++row) {
    if (!enabled[row] || elliptic_member[row]) continue;
    float grad = -rhs[row];
    for (int col = 0; col < total_nr; ++col)
      if (enabled[col]) grad += W[row * nr + col] * lam[col];
    grad += R[row] * lam[row];
    float diag = max(1e-15f, W[row * nr + row] + R[row]);
    float projected = clamp(lam[row] - grad / diag, lo[row], hi[row]);
    float scale = abs(ar[row]) + abs(R[row] * lam[row]);
    for (int col = 0; col < total_nr; ++col)
      if (enabled[col]) scale += abs(W[row * nr + col] * lam[col]);
    residual = max(residual,
        abs(projected - lam[row]) * diag / max(1.0f, scale));
  }
  for (int block = 0; block < elliptic_count; ++block) {
    int start = elliptic_start[block];
    int dim = elliptic_dim[block];
    if (start < 0 || start >= total_nr || !enabled[start]) continue;
    thread float mu[5];
    for (int i = 0; i < 5; ++i) mu[i] = elliptic_friction[block * 5 + i];
    residual = max(residual, cert_block_residual(start, dim, mu, W, R, rhs,
        lam, enabled, nr, total_nr));
  }
  return residual;
}

// Scalable block-row coupled constraint solver (milestone 017).
// Uses device memory for Delassus matrix storage and candidate compaction,
// scaling beyond 96 rows up to 256 rows and 64 DOFs without thread stack overflow.
kernel void solve_coupled_constraints_block(
    device const float* mass [[buffer(0)]],
    device const float* qfrc [[buffer(1)]],
    device const float* qpos [[buffer(2)]],
    device const float* qvel [[buffer(3)]],
    device const int* eq_active [[buffer(4)]],
    device const int* joint_qadr [[buffer(5)]],
    device const float* qpos0 [[buffer(6)]],
    device const int* joint_dadr [[buffer(7)]],
    device const uchar* joint_limited [[buffer(8)]],
    device const float* joint_limit_params [[buffer(9)]],
    device const float* joint_sol_params [[buffer(10)]],
    device const float* frictionloss [[buffer(11)]],
    device const float* invweight [[buffer(12)]],
    device const float* dof_sol_params [[buffer(13)]],
    device const int* eq_obj [[buffer(14)]],
    device const float* eq_data [[buffer(15)]],
    device const float* eq_sol_params [[buffer(16)]],
    device const float* contact_jacobian [[buffer(17)]],
    device const float* contact_row_data [[buffer(18)]],
    device const float* contact_friction [[buffer(19)]],
    device const int* contact_condim [[buffer(20)]],
    constant int* dims [[buffer(21)]],
    constant float* params [[buffer(22)]],
    device float* out_force [[buffer(23)]],
    device float* out_acc [[buffer(24)]],
    device int* out_status [[buffer(25)]],
    device float* out_diagnostics [[buffer(26)]],
    device float* out_contact_force [[buffer(27)]],
    device const int* packed_slot_to_logical [[buffer(28)]],
    device float* workspace_J [[buffer(29)]],
    device float* workspace_debug [[buffer(30)]],
    uint world [[thread_position_in_grid]]) {
  int nq = dims[0];
  int nv = dims[1];
  int nj = dims[2];
  int neq = dims[3];
  int ncontacts_max = dims[4];
  int batch = dims[5];
  int flags = dims[6];
  bool cached_velocity_stage = (dims[7] & 2) != 0;
  // Bit 8 reuses a just-assembled canonical row set after a host/device
  // impedance update. It does not imply cached-velocity/Jdot semantics.
  bool preassembled_rows_stage = (dims[7] & 8) != 0;
  bool reuse_rows_stage = cached_velocity_stage || preassembled_rows_stage;
  bool assembly_only_stage = dims[20] == -1;
  bool provided_qacc_smooth = (dims[7] & 4) != 0;
  bool paired_qfrc_input = (dims[7] & 16) != 0;
  bool refsafe = (dims[7] & 1) != 0;
  bool trace_solver_counts = (dims[7] & 32) != 0;
  bool trace_primal_details = (dims[7] & 64) != 0;
  int maxiter = dims[8];
  int solver_type = dims[19];
  int nr = dims[9];
  int cone_type = dims[10];
  int n_eq_rows = nr > 0 ? dims[14] : 0;
  if (n_eq_rows < 0) n_eq_rows = 0;
  if (n_eq_rows > nr) n_eq_rows = nr;
  int ten_base = nr > 0 ? dims[16] : 0;
  if (ten_base < 0) ten_base = 0;
  if (ten_base > nr) ten_base = nr;
  if (world >= uint(batch)) return;
  if (!cc_solver_world_enabled(dims, int(world))) return;

  int jmetadata = dims[24] + nr * 10;
  int jstride = dims[jmetadata + 17];
  device float* J_world = ccj_world(workspace_J, int(world), nr, nv, dims);
  device int* jheader = reinterpret_cast<device int*>(J_world);
  if (jstride < 12 || !ccj_header_valid(jheader, nr, nv)
      || jheader[9] != jstride || jheader[10] != 0) {
    out_status[world] = max(out_status[world], 2);
    if (dims[20] != -1) {
      int base = int(world) * nv;
      for (int i = 0; i < nv; ++i) {
        out_force[base + i] = 0.0f;
        out_acc[base + i] = 0.0f;
        out_acc[batch * nv + base + i] = 0.0f;
      }
    }
    return;
  }

  int mb = world * nv * nv;
  int qb = world * nv;
  int pb = world * nq;
  int incoming_status = out_status[world];
  if (!assembly_only_stage) {
    out_diagnostics[world * 10] = 0.0f;
    out_diagnostics[world * 10 + 1] = 0.0f;
    for (int k = 0; k < 8; ++k)
      out_diagnostics[world * 10 + 2 + k] = trace_solver_counts ? -1.0f : 0.0f;
    for (int i = 0; i < nv; ++i) {
      out_force[qb + i] = 0.0f;
      out_acc[qb + i] = 0.0f;
      out_acc[batch * nv + qb + i] = 0.0f;
    }
  }
  if (incoming_status != 0) return;
  if (!assembly_only_stage) out_status[world] = 0;

  if (nv == 0) return;
  device float* dbg = workspace_debug + world * max(dims[21], nr * nr + 7 * nr);
  device float* W = dbg;
  device float* R = dbg + nr * nr;
  device float* ar = dbg + nr * nr + nr;
  device float* rhs = dbg + nr * nr + 2 * nr;
  device float* lam = dbg + nr * nr + 3 * nr;
  device float* lo = dbg + nr * nr + 4 * nr;
  device float* hi = dbg + nr * nr + 5 * nr;
  device float* meta = dbg + nr * nr + 6 * nr;
  device float* solver_scratch = dbg + nr * nr + 7 * nr;
  int metadata_header = dims[24] + dims[9] * 10;
  device float* Z = solver_scratch + dims[metadata_header + 10];
  device float* pgs_pair_tail = solver_scratch
      + dims[metadata_header + 15];
  device float* pgs_rhs_low = pgs_pair_tail + 8 * max(nv, 1);
  device float* pgs_lambda_low = pgs_rhs_low + max(nr, 1);
  device const float* aref_low = pgs_lambda_low + max(nr, 1);
  int row_workspace = dims[metadata_header + 8];
  device int* row_meta = reinterpret_cast<device int*>(solver_scratch)
      + row_workspace;
  device int* enabled = row_meta;
  device int* elliptic_member = enabled + nr;
  device int* contact_block_start = elliptic_member + nr;
  device int* contact_block_size = contact_block_start + nr;
  device int* contact_slot_id = contact_block_size + nr;
  device int* elliptic_start = contact_slot_id + nr;
  device int* elliptic_dim = elliptic_start + nr;
  device float* elliptic_friction = reinterpret_cast<device float*>(elliptic_dim + nr);
  device float* L = solver_scratch;
  device float* y = L + nv * nv;
  device float* x = y + nv;
  device const float* q0 = out_acc + qb;

  for (int i = 0; i < nr * nr; ++i) W[i] = 0.0f;
  if (dims[20] != -1)
    for (int i = 0; i < nr; ++i) rhs[i] = 0.0f;

  for (int i = 0; i < nr; ++i) {
    enabled[i] = reuse_rows_stage ? (meta[i] > 0.5f) : false;
    elliptic_member[i] = 0;
  }

  bool warm_ok = (nr > 0) && ((flags & 512) == 0);
  if (!warm_ok && !assembly_only_stage) {
    for (int i = 0; i < nr; ++i) lam[i] = 0.0f;
  }

  for (int r = n_eq_rows; !reuse_rows_stage && jheader[2] == CCJ_DENSE && r < nr; ++r) {
    if (meta[r] > 0.5f) continue;
    R[r] = 0.0f;
    ar[r] = 0.0f;
    lo[r] = 0.0f;
    hi[r] = 0.0f;
    ccj_zero_row(J_world, r, nv);
  }
  for (int r = 0; !reuse_rows_stage && r < n_eq_rows && r < nr; ++r) {
    if (meta[r] > 0.5f) {
      enabled[r] = true;
      continue;
    }
    // Preserve the equality producer's row bounds. Inactive equalities are
    // published with zero R/aref and zero bounds; rewriting them as
    // bilateral here corrupts the cached-position activity contract.
    lo[r] = dbg[nr * nr + 4 * nr + r];
    hi[r] = dbg[nr * nr + 5 * nr + r];
    enabled[r] = (R[r] > 0.0f || abs(ar[r]) > 0.0f);
  }
  for (int r = ten_base; !reuse_rows_stage && r < nr; ++r) {
    if (meta[r] > 0.5f) enabled[r] = true;
  }

  int base_contact = n_eq_rows + nv + 2 * nj;
  if (nr > 0) {
    int ten_rows = dims[17];
    if (ten_rows < 0) ten_rows = 0;
    if (ten_rows > nr) ten_rows = nr;
    base_contact += ten_rows;
  }
  device float* out_joint_force = out_contact_force
      + max(batch * ncontacts_max * 11, 1)
      + world * max(base_contact, 1);
  int nactive_slots = 0;
  while (nactive_slots < ncontacts_max
         && packed_slot_to_logical[world * ncontacts_max + nactive_slots] >= 0)
    ++nactive_slots;

  // 1. Joint constraints
  if (!reuse_rows_stage && (flags & 1) == 0) {
    // Frictionloss
    for (int d = 0; d < nv; ++d) {
      int row = n_eq_rows + d;
      float loss = frictionloss[d];
      if ((flags & 4) != 0 || loss <= 0.0f) continue;
      ccj_set(J_world, row, d, 1.0f, nv);
      float comp = 0.0f, aref = 0.0f;
      reference_params(dof_sol_params + d * 7, dof_sol_params + d * 7 + 2, 0, 0.0f, 0.0f, qvel[qb + d], invweight[d], true, params[0], refsafe, comp, aref);
      R[row] = comp;
      if (dims[20] == -1)
        rhs[row] = impedance_at(dof_sol_params + d * 7 + 2, 0, 0.0f, 0.0f);
      ar[row] = aref;
      lo[row] = -loss;
      hi[row] = loss;
      enabled[row] = true;
    }

    // Joint Limits
    for (int j = 0; j < nj; ++j) {
      int limpack = joint_limited[j];
      if ((limpack & 1) == 0) continue;
      int d = joint_dadr[j];
      int q = joint_qadr[j];
      float margin = joint_limit_params[j * 3 + 2];
      if (((limpack >> 1) & 3) == 1) {
        // Ball limit
        float4 quat = float4(qpos[pb+q], qpos[pb+q+1], qpos[pb+q+2], qpos[pb+q+3]);
        float nq4 = length(quat);
        quat = nq4 > 1e-30f ? quat / nq4 : float4(1, 0, 0, 0);
        float3 vv = quat.yzw;
        float s = length(vv);
        float speed = 2.0f * atan2(s, quat.x);
        if (speed > 3.14159265358979f) speed -= 2.0f * 3.14159265358979f;
        float3 aa = s > 1e-30f ? vv * (speed / s) : float3(0.0f);
        float value = length(aa);
        float3 naxis = value > 1e-30f ? aa / value : float3(0.0f);
        float rmax = max(joint_limit_params[j * 3], joint_limit_params[j * 3 + 1]);
        float dist = rmax - value;
        int rowb = n_eq_rows + nv + 2 * j;
        if ((flags & 8) == 0 && dist < margin) {
          for (int k = 0; k < 3; ++k) {
            int dof = d + k;
            if (dof >= 0 && dof < nv) ccj_set(J_world, rowb, dof, -naxis[k], nv);
          }
          float vel = 0.0f;
          for (int k = 0; k < 3; ++k) {
            int dof = d + k;
            if (dof >= 0 && dof < nv) vel += -naxis[k] * qvel[qb + dof];
          }
          float comp = 0.0f, aref = 0.0f;
          reference_params(joint_sol_params + j * 7, joint_sol_params + j * 7 + 2, 0, dist, margin, vel, invweight[d], false, params[0], refsafe, comp, aref);
          R[rowb] = comp;
          if (dims[20] == -1)
            rhs[rowb] = impedance_at(joint_sol_params + j * 7 + 2, 0, dist, margin);
          ar[rowb] = aref;
          lo[rowb] = 0.0f;
          hi[rowb] = INFINITY;
          enabled[rowb] = true;
        }
      } else {
        // Hinge / Slide
        int row0 = n_eq_rows + nv + 2 * j;
        float dist0 = qpos[pb + q] - joint_limit_params[j * 3 + 0];
        if ((flags & 8) == 0 && dist0 < margin) {
          ccj_set(J_world, row0, d, 1.0f, nv);
          float comp = 0.0f, aref = 0.0f;
          reference_params(joint_sol_params + j * 7, joint_sol_params + j * 7 + 2, 0, dist0, margin, qvel[qb + d], invweight[d], false, params[0], refsafe, comp, aref);
          R[row0] = comp;
          if (dims[20] == -1)
            rhs[row0] = impedance_at(joint_sol_params + j * 7 + 2, 0, dist0, margin);
          ar[row0] = aref;
          lo[row0] = 0.0f;
          hi[row0] = INFINITY;
          enabled[row0] = true;
        }
        int row1 = n_eq_rows + nv + 2 * j + 1;
        float dist1 = joint_limit_params[j * 3 + 1] - qpos[pb + q];
        if ((flags & 8) == 0 && dist1 < margin) {
          ccj_set(J_world, row1, d, -1.0f, nv);
          float comp = 0.0f, aref = 0.0f;
          reference_params(joint_sol_params + j * 7, joint_sol_params + j * 7 + 2, 0, dist1, margin, -qvel[qb + d], invweight[d], false, params[0], refsafe, comp, aref);
          R[row1] = comp;
          if (dims[20] == -1)
            rhs[row1] = impedance_at(joint_sol_params + j * 7 + 2, 0, dist1, margin);
          ar[row1] = aref;
          lo[row1] = 0.0f;
          hi[row1] = INFINITY;
          enabled[row1] = true;
        }
      }
    }
  }

  // 2. Contacts with Candidate Compaction
  thread int contact_block_count = 0;
  thread int elliptic_count = 0;

  if (reuse_rows_stage) {
    for (int packed_s = 0; packed_s < nactive_slots; ++packed_s) {
      int s = packed_slot_to_logical[world * ncontacts_max + packed_s];
      if (s < 0 || s >= ncontacts_max) continue;
      int cdim = contact_condim[s * 3 + 0];
      int row_offset = contact_condim[s * 3 + 1];
      int row_start = base_contact + row_offset;
      int crbase = (world * ncontacts_max + s) * 36;
      if (contact_row_data[crbase] <= 0.5f) continue;
      if (cone_type == 1 && cdim > 1) {
        if (elliptic_count < nr) {
          elliptic_start[elliptic_count] = row_start;
          elliptic_dim[elliptic_count] = cdim;
          for (int k = 0; k < 5; ++k)
            elliptic_friction[elliptic_count * 5 + k] = contact_friction[s * 5 + k];
          ++elliptic_count;
        }
        for (int k = 0; k < cdim; ++k)
          if (row_start + k < nr) elliptic_member[row_start + k] = true;
      } else if (cdim > 1 && contact_block_count < nr) {
        contact_block_start[contact_block_count] = row_start;
        contact_block_size[contact_block_count] = 2 * (cdim - 1);
        contact_slot_id[contact_block_count] = s;
        ++contact_block_count;
      }
    }
  } else if ((flags & 1) == 0 && (flags & 16) == 0) {
    for (int packed_s = 0; packed_s < nactive_slots; ++packed_s) {
      int s = packed_slot_to_logical[world * ncontacts_max + packed_s];
      if (s < 0 || s >= ncontacts_max) continue;
      int cdim = contact_condim[s * 3 + 0];
      int row_offset = contact_condim[s * 3 + 1];
      int row_start = base_contact + row_offset;
      int cb = world * ncontacts_max + s;
      int cjbase = cb * 6 * nv;
      int crbase = cb * 6 * 6;

      if (cdim == 1) {
        int r = crbase;
        if (contact_row_data[r] > 0.5f) {
          if (row_start + 1 > nr) { out_status[world] = 2; return; }
          int row = row_start;
          if (jheader[2] == CCJ_DENSE)
            for (int i = 0; i < nv; ++i) ccj_set(J_world, row, i, contact_jacobian[cjbase + i], nv);
          float imp = clamp(contact_row_data[r + 4], 1e-6f, 0.999999f);
          float diag_approx = max(contact_row_data[r + 5], 1e-15f);
          R[row] = max(1e-15f, (1.0f - imp) * diag_approx / imp);
          if (dims[20] == -1) rhs[row] = imp;
          ar[row] = contact_row_data[r + 3];
          lo[row] = 0.0f;
          hi[row] = INFINITY;
          enabled[row] = true;
        } else {
          lam[row_start] = 0.0f;
        }
      } else if (contact_row_data[crbase] > 0.5f) {
        int edge_count = 2 * (cdim - 1);
        int block_rows = cone_type == 0 ? edge_count : cdim;
        if (row_start + block_rows > nr) { out_status[world] = 2; return; }
        float imp = clamp(contact_row_data[crbase + 4], 1e-6f, 0.999999f);
        float diag_approx = max(contact_row_data[crbase + 5], 1e-15f);
        float normal_R = max(1e-15f, (1.0f - imp) * diag_approx / imp);
        if (cone_type == 1) {
          if (elliptic_count < nr) {
            elliptic_start[elliptic_count] = row_start;
            elliptic_dim[elliptic_count] = cdim;
            for (int k = 0; k < 5; ++k) elliptic_friction[elliptic_count * 5 + k] = contact_friction[s * 5 + k];
            ++elliptic_count;
          }
          for (int k = 0; k < cdim; ++k) {
            int row = row_start + k;
            elliptic_member[row] = true;
            int r = crbase + k * 6;
            if (jheader[2] == CCJ_DENSE)
              for (int i = 0; i < nv; ++i) ccj_set(J_world, row, i, contact_jacobian[cjbase + k * nv + i], nv);
            if (k == 0) {
              R[row] = normal_R;
              if (dims[20] == -1) rhs[row] = imp;
              lo[row] = 0.0f;
              hi[row] = INFINITY;
            } else {
              float mu = contact_friction[s * 5 + k - 1];
              float mu0 = contact_friction[s * 5];
              float tangent_R = normal_R / max(params[1], 1e-15f);
              R[row] = tangent_R * mu0 * mu0 / max(mu * mu, 1e-12f);
              if (dims[20] == -1) rhs[row] = imp;
              lo[row] = -INFINITY;
              hi[row] = INFINITY;
            }
            ar[row] = contact_row_data[r + 3];
            enabled[row] = true;
          }
        } else {
          int edge_cnt = 2 * (cdim - 1);
          if (cdim > 1 && contact_block_count < nr) {
            contact_block_start[contact_block_count] = row_start;
            contact_block_size[contact_block_count] = edge_cnt;
            contact_slot_id[contact_block_count] = s;
            ++contact_block_count;
          }
          float mu0 = contact_friction[s * 5];
          float mu_master = mu0 / sqrt(max(params[1], 1e-15f));
          float first_edge_R = normal_R * (1.0f + mu0 * mu0);
          float pyramid_R = max(1e-15f,
              2.0f * mu_master * mu_master * first_edge_R);
          for (int axis = 0; axis < cdim - 1; ++axis) {
            float mu = contact_friction[s * 5 + axis];
            for (int side = 0; side < 2; ++side) {
              int k = axis * 2 + side;
              int row = row_start + k;
              float sign = side == 0 ? 1.0f : -1.0f;
              int r0 = crbase;
              int rt = crbase + (axis + 1) * 6;
              if (jheader[2] == CCJ_DENSE)
                for (int i = 0; i < nv; ++i) {
                  ccj_set(J_world, row, i, contact_jacobian[cjbase + i] + sign * mu * contact_jacobian[cjbase + (axis + 1) * nv + i], nv);
                }
              R[row] = pyramid_R;
              if (dims[20] == -1) rhs[row] = imp;
              ar[row] = cdim == 3
                  ? contact_row_data[crbase + (k + 1) * 6 + 3]
                  : contact_row_data[r0 + 3] + sign * mu * contact_row_data[rt + 3];
              lo[row] = 0.0f;
              hi[row] = INFINITY;
              enabled[row] = true;
            }
          }
        }
      } else {
        int edge_cnt = (cone_type == 1) ? cdim : 2 * (cdim - 1);
        for (int k = 0; k < edge_cnt; ++k) {
          if (row_start + k < nr) lam[row_start + k] = 0.0f;
        }
      }
    }
  }

  // Canonical flex rows are already assembled in workspace_J and debug.
  // Add their per-slot cone grouping after rigid candidates without rebuilding
  // their contact Jacobians or shifting the fixed descriptor row spans.
  int flex_slot_count = dims[metadata_header + 11];
  int flex_row_offset = dims[metadata_header + 12];
  int flex_row_count = dims[metadata_header + 13];
  if (flex_slot_count > 0 && flex_row_count > 0
      && (reuse_rows_stage || ((flags & 1) == 0 && (flags & 16) == 0))) {
    for (int flex_slot = 0; flex_slot < flex_slot_count; ++flex_slot) {
      int slot = ncontacts_max + flex_slot;
      int cdim = contact_condim[slot * 3];
      int row_offset = contact_condim[slot * 3 + 1];
      int row_start = base_contact + row_offset;
      int row_span = cdim == 1 ? 1 : (cone_type == 0 ? 2 * (cdim - 1) : cdim);
      if (cdim < 1 || cdim > 6 || row_start < base_contact
          || row_start + row_span > nr
          || row_offset < flex_row_offset
          || row_offset + row_span > flex_row_offset + flex_row_count) {
        out_status[world] = 2;
        return;
      }
      bool slot_active = false;
      for (int k = 0; k < row_span; ++k) {
        int row = row_start + k;
        bool row_active = meta[row] > 0.5f;
        if (row_active) {
          R[row] = R[row];
          ar[row] = ar[row];
          lo[row] = lo[row];
          hi[row] = hi[row];
          enabled[row] = true;
          slot_active = true;
        }
      }
      if (!slot_active || cdim == 1) continue;
      if (cone_type == 1) {
        if (elliptic_count >= nr) { out_status[world] = 2; return; }
        elliptic_start[elliptic_count] = row_start;
        elliptic_dim[elliptic_count] = cdim;
        for (int k = 0; k < 5; ++k)
          elliptic_friction[elliptic_count * 5 + k] = contact_friction[slot * 5 + k];
        ++elliptic_count;
        for (int k = 0; k < cdim; ++k) elliptic_member[row_start + k] = true;
      } else {
        if (contact_block_count >= nr) { out_status[world] = 2; return; }
        contact_block_start[contact_block_count] = row_start;
        contact_block_size[contact_block_count] = row_span;
        contact_slot_id[contact_block_count] = slot;
        ++contact_block_count;
      }
    }
  }

  device const int* solver_awake_tree = reinterpret_cast<device const int*>(solver_scratch)
      + dims[27];
  solver_mask_sleeping_dofs(dims, J_world, solver_awake_tree, nv, nr);
  solver_filter_sleeping_rows(dims, J_world, solver_awake_tree, enabled,
      nv, nr, nr);

  if (dims[20] == -1) {
    for (int row = 0; row < nr; ++row) {
      if (!enabled[row]) dbg[nr * nr + 2 * nr + row] = 0.0f;
      dbg[nr * nr + row] = R[row];
      dbg[nr * nr + nr + row] = ar[row];
      dbg[nr * nr + 4 * nr + row] = lo[row];
      dbg[nr * nr + 5 * nr + row] = hi[row];
      dbg[nr * nr + 6 * nr + row] = enabled[row] ? 1.0f : 0.0f;
    }
    return;
  }

  // 3. Dense Cholesky of the awake principal mass matrix.
  if (dims[20] == -2) {
    for (int i = 0; i < nv; ++i) {
      int smooth_hi = component_pair_index(batch, world, nr, i, nr, nv, false);
      int smooth_lo = component_pair_index(batch, world, nr, i, nr, nv, true);
      out_acc[qb + i] = provided_qacc_smooth ? qfrc[qb + i] : mass[smooth_hi];
      out_acc[batch * nv + qb + i] = provided_qacc_smooth
          ? qfrc[batch * nv + qb + i] : mass[smooth_lo];
    }
  } else if (dims[20] == -3) {
    if (!solver_factor_component_mass(mass, dbg, dims, solver_awake_tree,
        world, nv)) { out_status[world] = 2; return; }
    device float* component_solution = solver_scratch;
    device float* component_work = solver_scratch + nv;
    if (provided_qacc_smooth) {
      for (int i = 0; i < nv; ++i) component_solution[i] = qfrc[qb + i];
    } else if (!solver_component_mass_solve(dbg, dims, solver_awake_tree, nv,
        qfrc + qb, component_solution, component_work)) {
        out_status[world] = 2;
        return;
    }
    for (int i = 0; i < nv; ++i) {
      out_acc[qb + i] = component_solution[i];
    }
  } else {
  device const float* mass_world = mass + mb;
  for (int i = 0; i < nv; ++i) for (int j = 0; j < nv; ++j) L[i * nv + j] = 0.0f;
  for (int i = 0; i < nv; ++i) for (int j = 0; j <= i; ++j) {
    if (!solver_dof_awake(dims, solver_awake_tree, i)) {
      L[i * nv + j] = (i == j) ? 1.0f : 0.0f;
      continue;
    }
    if (!solver_dof_awake(dims, solver_awake_tree, j)) {
      L[i * nv + j] = 0.0f;
      continue;
    }
      float v = solver_mass_entry(mass_world, dbg, dims,
          solver_awake_tree, world, nv, i, j);
    for (int k = 0; k < j; ++k) v -= L[i * nv + k] * L[j * nv + k];
    if (i == j) {
      if (!(v > 1e-12f) || !isfinite(v)) { out_status[world] = 2; return; }
      L[i * nv + j] = sqrt(v);
    } else {
      L[i * nv + j] = v / L[j * nv + j];
    }
  }

  // 4. Smooth ACC may already have computed M^-1 qfrc. Avoid factoring or
  // solving that RHS twice in the VEL->ACC->CONSTRAINT split sequence.
  if (provided_qacc_smooth) {
    for (int i = 0; i < nv; ++i) {
      out_acc[qb + i] = qfrc[qb + i];
      out_acc[batch * nv + qb + i] = paired_qfrc_input
          ? qfrc[batch * nv + qb + i] : 0.0f;
    }
  } else if (paired_qfrc_input) {
    // Dense PGS solves its smooth force RHS in paired precision. Reuse the
    // row-pair solve work plane before Delassus construction takes ownership.
    device float* pair_tail = solver_scratch + dims[metadata_header + 15];
    device float* solve_y_low = pair_tail + 5 * max(nv, 1);
    primal_cholesky_solve_pair(L, nv, solver_awake_tree, dims,
        qfrc + qb, qfrc + batch * nv + qb,
        out_acc + qb, out_acc + batch * nv + qb,
        y, solve_y_low);
  } else {
    for (int i = 0; i < nv; ++i) {
      float v = solver_dof_awake(dims, solver_awake_tree, i) ? qfrc[qb + i] : 0.0f;
      for (int k = 0; k < i; ++k) v -= L[i * nv + k] * y[k];
      y[i] = v / L[i * nv + i];
    }
    for (int i = nv - 1; i >= 0; --i) {
      float v = y[i];
      for (int k = i + 1; k < nv; ++k) v -= L[k * nv + i] * x[k];
      x[i] = v / L[i * nv + i];
    }
    for (int i = 0; i < nv; ++i) out_acc[qb + i] = x[i];
  }
  }

  // 5 & 6. Assemble Delassus matrix W = J M^-1 J^T into device memory W
  for (int b = 0; b < nr; ++b) {
    if (!enabled[b]) continue;
    device float* z_b = solver_scratch + dims[metadata_header + 10] + b * nv;
    if (dims[20] == -2) {
      for (int i = 0; i < nv; ++i)
        z_b[i] = mass[component_pair_index(
            batch, world, b, i, nr, nv, false)];
    } else if (dims[20] == -3) {
      device float* component_solution = solver_scratch;
      device float* component_work = solver_scratch + nv;
      device float* component_row_rhs = solver_scratch
          + dims[metadata_header + 15];
      for (int dof = 0; dof < nv; ++dof)
        component_row_rhs[dof] = ccj_get(J_world, b, dof, nv);
      if (!solver_component_mass_solve(dbg, dims, solver_awake_tree, nv,
          component_row_rhs, component_solution, component_work)) {
        out_status[world] = 2;
        return;
      }
      for (int i = 0; i < nv; ++i) z_b[i] = component_solution[i];
    } else {
      for (int i = 0; i < nv; ++i) {
        float v = solver_dof_awake(dims, solver_awake_tree, i)
            ? ccj_get(J_world, b, i, nv) : 0.0f;
        for (int k = 0; k < i; ++k) v -= L[i * nv + k] * y[k];
        y[i] = v / L[i * nv + i];
      }
      for (int i = nv - 1; i >= 0; --i) {
        float v = y[i];
        for (int k = i + 1; k < nv; ++k) v -= L[k * nv + i] * z_b[k];
        z_b[i] = v / L[i * nv + i];
      }
    }
    for (int a = 0; a < nr; ++a) {
      if (!enabled[a]) continue;
      float v = 0.0f;
      for (int k = 0; k < nv; ++k) {
        v += ccj_get(J_world, a, k, nv) * z_b[k];
        if (dims[20] == -2)
          v += ccj_get(J_world, a, k, nv) * mass[
              component_pair_index(batch, world, b, k, nr, nv, true)];
      }
      W[a * nr + b] = v;
    }
  }

  // 7. Form RHS
  if (solver_type == 0) {
    for (int row = 0; row < nr; ++row) {
      pgs_rhs_low[row] = 0.0f;
      pgs_lambda_low[row] = 0.0f;
    }
  }
  for (int row = 0; row < nr; ++row) {
    if (enabled[row]) {
      PrimalFloatPair ja = {0.0f, 0.0f};
      for (int k = 0; k < nv; ++k)
        ja = primal_pair_fma(ccj_get(J_world, row, k, nv),
            {q0[k], out_acc[batch * nv + qb + k]}, ja);
      PrimalFloatPair rhs_pair = primal_pair_add(
          {ar[row], aref_low[row]}, primal_pair_multiply(ja, -1.0f));
      rhs[row] = rhs_pair.hi;
      if (solver_type == 0) pgs_rhs_low[row] = rhs_pair.lo;
    } else {
      rhs[row] = 0.0f;
      if (solver_type == 0) pgs_rhs_low[row] = 0.0f;
    }
  }

  int solver_island_count = build_solver_island_maps(
      dims, dbg, J_world, enabled, nv, nr, nr);
  device float* island_scratch = dbg + nr * nr + 7 * nr;
  device int* island_scratch_i = reinterpret_cast<device int*>(island_scratch);
  device int* solver_row_island = island_scratch_i + dims[25] + max(nv, 1);

  // The sparse/block world solver uses the same qacc_warmstart-to-force
  // mapping and PGS cost test as the dense path.
  if (solver_type == 0 && warm_ok && nr > 0) {
    int solver_metadata = dims[24] + dims[9] * 10;
    device const float* warm_acc = workspace_debug
        + world * max(dims[21], nr * nr + 7 * nr)
        + dims[solver_metadata + 5];
    seed_pgs_warmstart_qacc(J_world, nr, warm_acc, R, ar, lo, hi,
        enabled, lam, nv, nr, n_eq_rows, elliptic_count,
        elliptic_start, elliptic_dim, elliptic_friction);
  }

  // 7b. Warmstart cost check
  if (solver_type == 0 && warm_ok && nr > 0) {
    PrimalFloatPair wcost = {0.0f, 0.0f};
    for (int row = 0; row < nr; ++row) if (enabled[row]) {
      wcost = primal_pair_add(wcost,
          primal_pair_multiply(primal_pair_product(
              {lam[row], pgs_lambda_low[row]},
              {rhs[row], pgs_rhs_low[row]}), -1.0f));
      for (int col = 0; col < nr; ++col) if (enabled[col]) {
        PrimalFloatPair a = primal_pair_add(
            pgs_delassus_pair(J_world, Z, mass, dims, batch,
                              int(world), nr, nv, row, col),
            {row == col ? R[row] : 0.0f, 0.0f});
        PrimalFloatPair term = primal_pair_product(
            primal_pair_product({lam[row], pgs_lambda_low[row]}, a),
            {lam[col], pgs_lambda_low[col]});
        wcost = primal_pair_add(wcost, primal_pair_multiply(term, 0.5f));
      }
    }
    if (primal_pair_value(wcost) > 0.0f || !isfinite(wcost.hi)) {
      for (int row = 0; row < nr; ++row) {
        lam[row] = 0.0f;
        pgs_lambda_low[row] = 0.0f;
      }
    }
  } else if (warm_ok && nr > 0) {
    // The component primal route reuses the same shader but not PGS scratch.
    float wcost = 0.0f;
    for (int row = 0; row < nr; ++row) if (enabled[row]) {
      wcost -= lam[row] * rhs[row];
      for (int col = 0; col < nr; ++col) if (enabled[col]) {
        float a = W[row * nr + col] + (row == col ? R[row] : 0.0f);
        wcost += 0.5f * lam[row] * a * lam[col];
      }
    }
    if (wcost > 0.0f || !isfinite(wcost))
      for (int row = 0; row < nr; ++row) lam[row] = 0.0f;
  }

  // 8. Projected Gauss-Seidel solve
  // Keep cached VEL solves on the same requested-tolerance stopping contract.
  float tol = params[6];
  bool converged = false;
  bool primal_acceleration_output = false;
  float max_res = 0.0f;
  thread float hist[8];
  thread int primal_island_iterations[8];
  for (int k = 0; k < 8; ++k) primal_island_iterations[k] = -1;
  int hsamp = 0;
  int hstep = max(1, maxiter / 7);
  for (int k = 0; k < 8; ++k) hist[k] = 0.0f;
  // The zero-budget result is the validated retained warm start.  Record its
  // projected residual instead of converting non-convergence into failure.
  for (int row = 0; row < nr; ++row) if (enabled[row] && !elliptic_member[row]) {
    float grad = -rhs[row];
    for (int col = 0; col < nr; ++col) if (enabled[col])
      grad += W[row * nr + col] * lam[col];
    grad += R[row] * lam[row];
    float diag = max(1e-15f, W[row * nr + row] + R[row]);
    float proj = clamp(lam[row] - grad / diag, lo[row], hi[row]);
    float scale = abs(ar[row]) + abs(R[row] * lam[row]);
    for (int col = 0; col < nr; ++col) if (enabled[col])
      scale += abs(W[row * nr + col] * lam[col]);
    max_res = max(max_res, abs(proj - lam[row]) * diag / max(1.0f, scale));
  }
  for (int b = 0; b < elliptic_count; ++b) {
    int start = elliptic_start[b], dim = elliptic_dim[b];
    thread float A[36], g[6], force[6], mu[5];
    for (int i = 0; i < 36; ++i) A[i] = 0.0f;
    for (int i = 0; i < 6; ++i) { g[i] = 0.0f; force[i] = 0.0f; }
    for (int k = 0; k < 5; ++k) mu[k] = elliptic_friction[b * 5 + k];
    for (int i = 0; i < dim; ++i) {
      int ri = start + i;
      force[i] = lam[ri];
      g[i] = -rhs[ri];
      for (int col = 0; col < nr; ++col)
        if (enabled[col] && (col < start || col >= start + dim))
          g[i] += W[ri * nr + col] * lam[col];
      for (int j = 0; j < dim; ++j) {
        int rj = start + j;
        A[i * 6 + j] = W[ri * nr + rj] + (i == j ? R[ri] : 0.0f);
      }
    }
    max_res = max(max_res, elliptic_projected_residual(A, g, mu, dim, force));
  }

  if (solver_type == 0) {
    int used = 0;
    float pgs_residual = 0.0f;
    float pgs_mean_inertia = params[4];
    float pgs_improvement_scale = 1.0f
        / (pgs_mean_inertia * float(max(1, nv)));
    for (int island = 0; island < solver_island_count; ++island) {
      for (int row = 0; row < nr; ++row)
        enabled[row] = solver_row_island[row] == island;
      float island_residual = 0.0f;
      used += solve_pgs_device(W, R, rhs, pgs_rhs_low, ar, lo, hi, enabled,
          elliptic_member, lam, pgs_lambda_low, J_world, Z, mass, dims,
          batch, int(world), nv, nr, nr, maxiter, tol,
          pgs_improvement_scale, elliptic_count,
          elliptic_start, elliptic_dim, elliptic_friction,
          solver_scratch + dims[metadata_header + 9], &island_residual);
      pgs_residual = max(pgs_residual, island_residual);
    }
    for (int row = 0; row < nr; ++row)
      enabled[row] = solver_row_island[row] >= 0;
    max_res = pgs_residual;
    converged = max_res <= tol;
    out_diagnostics[world * 10] = max_res;
    out_diagnostics[world * 10 + 1] = float(used);
    for (int k = 0; k < 8; ++k) hist[k] = max_res;
    hsamp = 7;
  } else if (solver_type == 1 || solver_type == 2) {
    int used = 0;
    float primal_residual = 0.0f;
    bool partitioned = (dims[6] & dims[26]) == 0;
    // The block ABI is W, R, aref, rhs, lambda, lo, hi, row-meta.
    // Keep primal force output in the canonical lambda slice; +6*nr is
    // activation/row metadata and must not be used as solver scratch.
    device float* primal_lambda = lam;
    device int* dof_island = island_scratch_i + dims[25];
    device float* primal_scratch = solver_scratch + dims[metadata_header + 9];
    device float* high_low_scratch = solver_scratch
        + dims[metadata_header + 15];
    device float* qacc_smooth_low = high_low_scratch + 7 * max(nv, 1);
    for (int i = 0; i < nv; ++i)
      qacc_smooth_low[i] = out_acc[batch * nv + qb + i];
    for (int row = 0; row < nr; ++row) primal_lambda[row] = 0.0f;
    // out_acc is mutated after each island, so retain one immutable
    // qacc_smooth baseline in the not-yet-written generalized-force output.
    for (int i = 0; i < nv; ++i) out_force[qb + i] = out_acc[qb + i];
    device const float* qacc_baseline = out_force + qb;
    for (int island = 0; island < solver_island_count; ++island) {
      for (int row = 0; row < nr; ++row)
        enabled[row] = solver_row_island[row] == island;
      float island_inertia = 0.0f;
      for (int i = 0; i < nv; ++i) if (dof_island[i] == island) {
        island_inertia += dims[20] == -3
            ? solver_component_mass_entry(dbg, dims, nv, i, i)
            : mass[i * nv + i];
      }
      int island_used = solve_primal_accel_scalar(
          dims[20] == -3 ? mass : mass + mb, dbg, J_world, R, ar, lo, hi,
          L, qacc_baseline, qacc_smooth_low, dims, solver_awake_tree, enabled,
          elliptic_member, elliptic_start, elliptic_dim,
          elliptic_friction, elliptic_count, primal_lambda, nv, nr, nr,
          n_eq_rows, solver_type, maxiter, dims[metadata_header + 14],
          params[5], tol,
          1.0f / max(island_inertia, 1e-15f),
          (flags & 512) == 0, primal_scratch, high_low_scratch,
          out_diagnostics + world * 10, dof_island, island, partitioned);
      if (island_used < 0) {
        for (int i = 0; i < nv; ++i) {
          out_force[qb + i] = 0.0f;
          out_acc[batch * nv + qb + i] = 0.0f;
        }
        out_status[world] = 2;
        return;
      }
      used += island_used;
      if (trace_solver_counts && island < 8)
        primal_island_iterations[island] = island_used;
      primal_residual = max(primal_residual, out_diagnostics[world * 10]);
      device const float* primal_acc = primal_scratch;
      for (int i = 0; i < nv; ++i)
        if (!partitioned || dof_island[i] == island) {
          out_acc[qb + i] = primal_acc[i];
          out_acc[batch * nv + qb + i] = high_low_scratch[i];
        }
      for (int row = 0; row < nr; ++row)
        if (solver_row_island[row] == island) lam[row] = primal_lambda[row];
    }
    for (int row = 0; row < nr; ++row)
      enabled[row] = solver_row_island[row] >= 0;
    primal_acceleration_output = true;
    max_res = primal_residual;
    converged = max_res <= tol;
    out_diagnostics[world * 10] = max_res;
    out_diagnostics[world * 10 + 1] = float(used);
    if (!trace_solver_counts) {
      for (int k = 0; k < 8; ++k) hist[k] = max_res;
      hsamp = 7;
    }
  } else if (elliptic_count > 0) {
    device float* cone_scratch = solver_scratch + dims[metadata_header + 9];
    device float* dscale = cone_scratch;
    device float* z = dscale + nr;
    device float* extrapolated = z + nr;
    device float* candidate = extrapolated + nr;
    device float* gradient_z = candidate + nr;
    device float* power_vector = gradient_z + nr;
    device float* power_product = power_vector + nr;
    for (int row = 0; row < nr; ++row) {
      dscale[row] = 1.0f;
      z[row] = extrapolated[row] = candidate[row] = 0.0f;
      gradient_z[row] = 0.0f;
    }
    for (int block = 0; block < elliptic_count; ++block) {
      int start = elliptic_start[block];
      int dim = elliptic_dim[block];
      for (int k = 1; k < dim; ++k)
        dscale[start + k] = max(elliptic_friction[block * 5 + k - 1], 0.0f);
    }
    float lipschitz = 1e-15f;
    int enabled_count = 0;
    for (int row = 0; row < nr; ++row) if (enabled[row]) {
      ++enabled_count;
      float row_sum = 0.0f;
      for (int col = 0; col < nr; ++col) if (enabled[col]) {
        float value = dscale[row] * W[row * nr + col] * dscale[col];
        if (row == col) value += dscale[row] * R[row] * dscale[col];
        row_sum += abs(value);
      }
      lipschitz = max(lipschitz, row_sum);
      float scale = dscale[row];
      z[row] = scale > 1e-12f ? lam[row] / scale : 0.0f;
      extrapolated[row] = z[row];
    }
    float gershgorin_bound = lipschitz;
    float inv_norm = rsqrt(float(max(enabled_count, 1)));
    for (int row = 0; row < nr; ++row)
      power_vector[row] = enabled[row] ? inv_norm : 0.0f;
    for (int iteration = 0; iteration < 24; ++iteration) {
      float norm_sq = 0.0f;
      for (int row = 0; row < nr; ++row) {
        float value = 0.0f;
        if (enabled[row]) {
          for (int col = 0; col < nr; ++col) if (enabled[col]) {
            float entry = dscale[row] * W[row * nr + col] * dscale[col];
            if (row == col) entry += dscale[row] * R[row] * dscale[col];
            value += entry * power_vector[col];
          }
        }
        power_product[row] = value;
        norm_sq += value * value;
      }
      float inv_power_norm = rsqrt(max(norm_sq, 1e-30f));
      for (int row = 0; row < nr; ++row)
        power_vector[row] = power_product[row] * inv_power_norm;
    }
    float rayleigh = 0.0f;
    for (int row = 0; row < nr; ++row) if (enabled[row]) {
      float value = 0.0f;
      for (int col = 0; col < nr; ++col) if (enabled[col]) {
        float entry = dscale[row] * W[row * nr + col] * dscale[col];
        if (row == col) entry += dscale[row] * R[row] * dscale[col];
        value += entry * power_vector[col];
      }
      rayleigh += power_vector[row] * value;
    }
    lipschitz = min(gershgorin_bound, max(1e-15f, rayleigh * 1.1f));
    float momentum = 1.0f;
    for (int it = 0; it < maxiter; ++it) {
      for (int row = 0; row < nr; ++row)
        lam[row] = enabled[row] ? dscale[row] * extrapolated[row] : 0.0f;
      for (int row = 0; row < nr; ++row) if (enabled[row]) {
        float gradient = -rhs[row];
        for (int col = 0; col < nr; ++col) if (enabled[col])
          gradient += W[row * nr + col] * lam[col];
        gradient += R[row] * lam[row];
        gradient_z[row] = dscale[row] * gradient;
      } else {
        gradient_z[row] = 0.0f;
      }
      bool step_accepted = false;
      for (int backtrack = 0; backtrack < 12; ++backtrack) {
        for (int row = 0; row < nr; ++row)
          candidate[row] = enabled[row]
              ? extrapolated[row] - gradient_z[row] / lipschitz : 0.0f;
        for (int block = 0; block < elliptic_count; ++block) {
          int start = elliptic_start[block];
          int dim = elliptic_dim[block];
          thread float cone_value[6];
          for (int k = 0; k < 6; ++k) cone_value[k] = k < dim ? candidate[start + k] : 0.0f;
          project_lorentz(cone_value, dim);
          for (int k = 0; k < dim; ++k) candidate[start + k] = cone_value[k];
        }
        for (int row = 0; row < nr; ++row)
          if (enabled[row] && !elliptic_member[row])
            candidate[row] = clamp(candidate[row], lo[row], hi[row]);

        float objective_base = 0.0f, objective_candidate = 0.0f;
        float linear_delta = 0.0f, delta_norm_sq = 0.0f;
        for (int row = 0; row < nr; ++row) if (enabled[row]) {
          float base_lambda = dscale[row] * extrapolated[row];
          float candidate_lambda = dscale[row] * candidate[row];
          float base_product = 0.0f;
          float candidate_product = 0.0f;
          for (int col = 0; col < nr; ++col) if (enabled[col]) {
            float base_col = dscale[col] * extrapolated[col];
            float candidate_col = dscale[col] * candidate[col];
            base_product += W[row * nr + col] * base_col;
            candidate_product += W[row * nr + col] * candidate_col;
          }
          base_product += R[row] * base_lambda;
          candidate_product += R[row] * candidate_lambda;
          objective_base += 0.5f * base_lambda * base_product - rhs[row] * base_lambda;
          objective_candidate += 0.5f * candidate_lambda * candidate_product - rhs[row] * candidate_lambda;
          float delta = candidate[row] - extrapolated[row];
          linear_delta += gradient_z[row] * delta;
          delta_norm_sq += delta * delta;
        }
        float majorizer = objective_base + linear_delta + 0.5f * lipschitz * delta_norm_sq;
        if (objective_candidate <= majorizer + 1e-6f * max(1.0f, abs(objective_candidate))) {
          step_accepted = true;
          break;
        }
        lipschitz *= 2.0f;
      }
      if (!step_accepted) { out_status[world] = 2; return; }

      float next_momentum = 0.5f * (1.0f + sqrt(1.0f + 4.0f * momentum * momentum));
      float beta = (momentum - 1.0f) / next_momentum;
      float restart_dot = 0.0f;
      for (int row = 0; row < nr; ++row) if (enabled[row])
        restart_dot += (candidate[row] - z[row]) * (extrapolated[row] - candidate[row]);
      if (restart_dot > 0.0f) {
        next_momentum = 1.0f;
        beta = 0.0f;
      }
      for (int row = 0; row < nr; ++row) if (enabled[row]) {
        float previous = z[row];
        z[row] = candidate[row];
        extrapolated[row] = candidate[row] + beta * (candidate[row] - previous);
        lam[row] = dscale[row] * z[row];
      }
      momentum = next_momentum;

      max_res = 0.0f;
      for (int row = 0; row < nr; ++row) if (enabled[row]) {
        if (elliptic_member[row]) {
          int block = -1;
          for (int b = 0; b < elliptic_count; ++b) if (elliptic_start[b] == row) block = b;
          if (block < 0) continue;
          int dim = elliptic_dim[block];
          thread float A[36], g[6], mu[5], force[6];
          for (int i = 0; i < 36; ++i) A[i] = 0.0f;
          for (int i = 0; i < 6; ++i) { g[i] = 0.0f; force[i] = 0.0f; }
          for (int i = 0; i < 5; ++i) mu[i] = elliptic_friction[block * 5 + i];
          for (int i = 0; i < dim; ++i) {
            int ri = row + i;
            force[i] = lam[ri];
            g[i] = -rhs[ri];
            for (int col = 0; col < nr; ++col)
              if (enabled[col] && (col < row || col >= row + dim))
                g[i] += W[ri * nr + col] * lam[col];
            for (int j = 0; j < dim; ++j) {
              int rj = row + j;
              A[i * 6 + j] = W[ri * nr + rj] + (i == j ? R[ri] : 0.0f);
            }
          }
          max_res = max(max_res, elliptic_projected_residual(A, g, mu, dim, force));
          continue;
        }
        float grad = -rhs[row];
        for (int col = 0; col < nr; ++col) if (enabled[col]) grad += W[row * nr + col] * lam[col];
        grad += R[row] * lam[row];
        float diag = max(1e-15f, W[row * nr + row] + R[row]);
        float proj = clamp(lam[row] - grad / diag, lo[row], hi[row]);
        float row_scale = abs(ar[row]) + abs(R[row] * lam[row]);
        for (int col = 0; col < nr; ++col) if (enabled[col]) row_scale += abs(W[row * nr + col] * lam[col]);
        row_scale = max(1.0f, row_scale);
        max_res = max(max_res, abs(proj - lam[row]) * diag / row_scale);
      }
      out_diagnostics[world * 10] = max_res;
      out_diagnostics[world * 10 + 1] = float(it + 1);
      if (hsamp < 7 && ((it + 1) % hstep == 0 || it + 1 == maxiter)) {
        hist[hsamp++] = max_res;
      }
      if (max_res <= tol) { converged = true; break; }
    }
  } else {
    int pgs_cap = maxiter;
    for (int it = 0; it < pgs_cap; ++it) {
      for (int row = 0; row < nr; ++row) if (enabled[row]) {
        float diag = max(1e-15f, W[row * nr + row] + R[row]);
        float v = rhs[row];
        for (int col = 0; col < nr; ++col) if (enabled[col] && col != row)
          v -= W[row * nr + col] * lam[col];
        lam[row] = clamp(v / diag, lo[row], hi[row]);
      }
      max_res = 0.0f;
      for (int row = 0; row < nr; ++row) if (enabled[row]) {
        float grad = -rhs[row];
        for (int col = 0; col < nr; ++col) if (enabled[col]) grad += W[row * nr + col] * lam[col];
        grad += R[row] * lam[row];
        float diag = max(1e-15f, W[row * nr + row] + R[row]);
        float proj = clamp(lam[row] - grad / diag, lo[row], hi[row]);
        float row_scale = abs(ar[row]) + abs(R[row] * lam[row]);
        for (int col = 0; col < nr; ++col) if (enabled[col]) row_scale += abs(W[row * nr + col] * lam[col]);
        row_scale = max(1.0f, row_scale);
        max_res = max(max_res, abs(proj - lam[row]) * diag / row_scale);
      }
      out_diagnostics[world * 10] = max_res;
      out_diagnostics[world * 10 + 1] = float(it + 1);
      if (hsamp < 7 && ((it + 1) % hstep == 0 || it + 1 == maxiter)) {
        hist[hsamp++] = max_res;
      }
      if (max_res <= tol) { converged = true; break; }
    }

  }

  out_diagnostics[world * 10] = max_res;

  // 9b. No-slip post-pass
  int noslip_iters = dims[18];
  float noslip_tol = params[3];
  float mean_inertia = params[4];
  float noslip_scale = 1.0f
      / (mean_inertia * float(max(1, nv)));
  if (noslip_iters > 0 && nr > 0) {
    int ns_done = 0;
    for (int nsit = 0; nsit < noslip_iters; ++nsit) {
      float improvement = 0.0f;
      if (nsit == 0) {
        for (int row = 0; row < nr; ++row) {
          if (enabled[row]) {
            improvement += 0.5f * lam[row] * lam[row] * R[row];
          }
        }
      }
      // Dry friction rows
      for (int row = 0; row < nr; ++row) if (enabled[row]) {
        float lorb = lo[row], hib = hi[row];
        if (!(lorb < 0.0f && hib == -lorb && hib < INFINITY)) continue;
        float diag = max(1e-10f, W[row * nr + row]);
        float res = -rhs[row];
        for (int col = 0; col < nr; ++col) if (enabled[col]) res += W[row * nr + col] * lam[col];
        float old = lam[row];
        float v = clamp(old - res / diag, lorb, hib);
        float delta = v - old;
        float change = 0.5f * delta * delta * diag + delta * res;
        if (change > 1e-10f) { v = old; change = 0.0f; }
        lam[row] = v;
        improvement -= change;
      }
      // Contact friction blocks
      for (int b = 0; b < contact_block_count; ++b) {
        int row_start = contact_block_start[b];
        int block_size = contact_block_size[b];
        for (int k = 0; k + 1 < block_size; k += 2) {
          int j0 = row_start + k, j1 = row_start + k + 1;
          if (!(enabled[j0] && enabled[j1])) continue;
          float r0 = -rhs[j0], r1 = -rhs[j1];
          for (int col = 0; col < nr; ++col) if (enabled[col]) {
            r0 += W[j0 * nr + col] * lam[col];
            r1 += W[j1 * nr + col] * lam[col];
          }
          float old0 = lam[j0], old1 = lam[j1];
          float A00 = W[j0 * nr + j0], A11 = W[j1 * nr + j1];
          float A01 = W[j0 * nr + j1];
          float bc0 = r0 - (A00 * old0 + A01 * old1);
          float bc1 = r1 - (A01 * old0 + A11 * old1);
          float mid = 0.5f * (old0 + old1);
          float y_val = 0.5f * (old0 - old1);
          float K1 = A00 + A11 - 2.0f * A01;
          float K0 = mid * (A00 - A11) + bc0 - bc1;
          float ny = y_val;
          if (K1 < 1e-15f) {
            ny = y_val;
          } else {
            ny = -K0 / K1;
            ny = ny < -mid ? -mid : (ny > mid ? mid : ny);
          }
          float f0 = mid + ny, f1 = mid - ny;
          float d0 = f0 - old0, d1 = f1 - old1;
          float change = 0.5f * (d0 * (A00 * d0 + A01 * d1) + d1 * (A01 * d0 + A11 * d1))
              + d0 * r0 + d1 * r1;
          if (change > 1e-10f) { f0 = old0; f1 = old1; change = 0.0f; }
          lam[j0] = f0;
          lam[j1] = f1;
          improvement -= change;
        }
      }
      ns_done++;
      improvement *= noslip_scale;
      if (improvement < noslip_tol) break;
    }
    out_diagnostics[world * 10 + 1] += float(ns_done);
    if (trace_solver_counts) {
      out_diagnostics[world * 10 + 2] =
          out_diagnostics[world * 10 + 1] - float(ns_done);
      out_diagnostics[world * 10 + 3] = float(ns_done);
    }
  }

  // As in the dense path, primal acceleration is stale after any configured
  // no-slip pass. Restore the smooth input before the final dual-force solve.
  if (primal_acceleration_output && noslip_iters > 0 && nr > 0) {
    device float* restore_a = y;
    device float* restore_b = x;
    if (dims[20] == -3) {
      device float* pair_tail = solver_scratch
          + dims[metadata_header + 15];
      restore_a = pair_tail + 4 * max(nv, 1);
      restore_b = pair_tail + 5 * max(nv, 1);
    }
    if (!solver_restore_smooth_acceleration(out_acc, qfrc, mass, dbg,
        L, restore_a, restore_b, dims, solver_awake_tree, batch, int(world),
        nv, nr, qb, provided_qacc_smooth, paired_qfrc_input)) {
      out_status[world] = 2;
      for (int i = 0; i < nv; ++i) {
        out_force[qb + i] = 0.0f;
        out_acc[qb + i] = 0.0f;
        out_acc[batch * nv + qb + i] = 0.0f;
      }
      return;
    }
    primal_acceleration_output = false;
  }

  // solver_diagnostics[0] certifies the multipliers that will be retained
  // and published below. The optimizer metric sampled in solver_history is
  // deliberately kept distinct from this final dual-system residual.
  out_diagnostics[world * 10] = cert_retained_system_residual(W, R, rhs, ar,
      lo, hi, lam, enabled, elliptic_member, elliptic_start, elliptic_dim,
      elliptic_friction, elliptic_count, nr, nr);

  bool finite_solution = isfinite(max_res);
  for (int r = 0; r < nr; ++r)
    if (enabled[r] && !isfinite(lam[r])) finite_solution = false;
  if (!finite_solution) out_status[world] = 3;

  // 10. Reconstruct forces and acceleration
  for (int i = 0; i < nv; ++i) {
    float total = 0.0f;
    for (int row = 0; row < nr; ++row)
      if (enabled[row]) total += ccj_get(J_world, row, i, nv) * lam[row];
    out_force[qb + i] = total;
  }

  // Acceleration-space CG retains its finite accepted primal iterate.
  // Other solver branches reconstruct acceleration from the constraint force.
  if (dims[20] == -2) {
    for (int i = 0; i < nv; ++i) {
      PrimalFloatPair acceleration = {
          provided_qacc_smooth
              ? qfrc[qb + i]
              : mass[component_pair_index(batch, world, nr, i, nr, nv, false)],
          provided_qacc_smooth
              ? qfrc[batch * nv + qb + i]
              : mass[component_pair_index(batch, world, nr, i, nr, nv, true)]};
      for (int row = 0; row < nr; ++row) if (enabled[row])
        acceleration = primal_pair_fma(
            mass[component_pair_index(batch, world, row, i, nr, nv, false)],
            {lam[row], solver_type == 0 ? pgs_lambda_low[row] : 0.0f},
            acceleration);
      for (int row = 0; row < nr; ++row) if (enabled[row])
        acceleration = primal_pair_fma(
            mass[component_pair_index(batch, world, row, i, nr, nv, true)],
            {lam[row], solver_type == 0 ? pgs_lambda_low[row] : 0.0f},
            acceleration);
      out_acc[qb + i] = acceleration.hi;
      out_acc[batch * nv + qb + i] = acceleration.lo;
    }
  } else if (primal_acceleration_output) {
    // `out_acc` was written directly from the reusable primal scratch above.
  } else if (dims[20] == -3) {
    // The component-factor route has no dense L matrix. Use its compiled
    // per-row M^-1 J^T values to finish from the restored smooth baseline.
    for (int i = 0; i < nv; ++i) {
      PrimalFloatPair acceleration = {
          out_acc[qb + i], out_acc[batch * nv + qb + i]};
      for (int row = 0; row < nr; ++row) if (enabled[row])
        acceleration = primal_pair_fma(Z[row * nv + i],
            {lam[row], solver_type == 0 ? pgs_lambda_low[row] : 0.0f},
            acceleration);
      out_acc[qb + i] = acceleration.hi;
      out_acc[batch * nv + qb + i] = acceleration.lo;
    }
  } else {
  // Solve M delta_a = f_tot using Cholesky factor L
  for (int i = 0; i < nv; ++i) {
    float v = out_force[qb + i];
    for (int k = 0; k < i; ++k) v -= L[i * nv + k] * y[k];
    y[i] = v / L[i * nv + i];
  }
  for (int i = nv - 1; i >= 0; --i) {
    float v = y[i];
    for (int k = i + 1; k < nv; ++k) v -= L[k * nv + i] * x[k];
    x[i] = v / L[i * nv + i];
  }
  for (int i = 0; i < nv; ++i) {
    PrimalFloatPair acceleration = primal_pair_renormalize(q0[i], x[i]);
    out_acc[qb + i] = acceleration.hi;
    out_acc[batch * nv + qb + i] = acceleration.lo;
  }
  }

  // 11. Write joint forces
  for (int row = 0; row < base_contact; ++row) {
    // out_joint_force already points at this world's joint-force slice.
    out_joint_force[row] = enabled[row] ? lam[row] : 0.0f;
  }

  // 12. Write contact forces
  for (int s = 0; s < ncontacts_max; ++s) {
    int cdim = contact_condim[s * 3 + 0];
    int row_offset = contact_condim[s * 3 + 1];
    int row_start = base_contact + row_offset;
    int ofb = (world * ncontacts_max + s) * 11;
    for (int k = 0; k < 11; ++k) out_contact_force[ofb + k] = 0.0f;
    if (cdim == 0 || !enabled[row_start]) continue;
    if (cdim == 1) {
      out_contact_force[ofb] = lam[row_start];
    } else if (cone_type == 1) {
      for (int k = 0; k < cdim; ++k) out_contact_force[ofb + k] = enabled[row_start + k] ? lam[row_start + k] : 0.0f;
    } else {
      int edge_count = 2 * (cdim - 1);
      float normal = 0.0f;
      for (int k = 0; k < edge_count; ++k) {
        float value = lam[row_start + k];
        normal += value;
        out_contact_force[ofb + 1 + k] = value;
      }
      out_contact_force[ofb] = normal;
    }
  }
  // Publish per-island primal counts only after every diagnostic/history
  // writer. If no-slip ran, its main/no-slip split remains authoritative.
  if (trace_solver_counts && !trace_primal_details
      && (solver_type == 1 || solver_type == 2)
      && !(noslip_iters > 0 && nr > 0)) {
    for (int k = 0; k < 8; ++k)
      out_diagnostics[world * 10 + 2 + k] = float(primal_island_iterations[k]);
  }
}

// Candidate narrowphase status is produced by the flex detector on device.
// A positive code means the fixed-capacity algorithm could not certify this
// candidate; preserve the first-stage failure in the public per-world status
// without reading a candidate mask back to the host.
kernel void merge_flex_narrowphase_status(
    device const int* candidate_status [[buffer(0)]],
    device int* world_status [[buffer(1)]],
    constant int* dims [[buffer(2)]],
    uint world [[thread_position_in_grid]]) {
  int batch = dims[5];
  if (world >= uint(batch)) return;
  if (!cc_solver_world_enabled(dims, int(world))) return;
  int nr = dims[9];
  int header = dims[24] + nr * 10;
  int slots = dims[header + 11];
  if (slots <= 0) return;
  int status = world_status[world];
  for (int slot = 0; slot < slots; ++slot) {
    if (candidate_status[int(world) * slots + slot] != 0) {
      // Raw per-slot narrowphase codes remain available in flex diagnostics;
      // world status uses one stable general solver-failure category.
      status = max(status, 4);
    }
  }
  world_status[world] = status;
}

// An SDF producer marks source-equivalent contact-buffer overflow with a
// negative pair count.  Preserve that fail-closed signal in public world
// status after the normal per-world status reset and before the solver reads
// assembled rows.  The candidate world mask is authoritative here: masked
// worlds may intentionally retain old pair counts and must not poison an
// active world or a later re-enabled pass.
kernel void merge_sdf_narrowphase_status(
    device const int* contact_count [[buffer(0)]],
    device int* world_status [[buffer(1)]],
    device const int* candidate_world_mask [[buffer(2)]],
    constant int* dims [[buffer(3)]],
    uint world [[thread_position_in_grid]]) {
  // Reuse the admitted pair_contact_offsets_dims header: nv, npairs,
  // ncontacts_max, batch, ...; this status-only consumer owns no side tensor.
  int batch = dims[3];
  int npairs = dims[1];
  if (world >= uint(batch) || candidate_world_mask[world] == 0) return;
  for (int pair_idx = 0; pair_idx < npairs; ++pair_idx) {
    if (contact_count[int(world) * npairs + pair_idx] < 0) {
      world_status[world] = max(world_status[world], 4);
      return;
    }
  }
}

// Refresh only the affine reference term for velocity stages that reuse a
// previously assembled position context. Canonical J/R/bounds/activity stay
// untouched; the caller separately supplies any equality Jdot-v correction.
kernel void capture_cached_position_context(
    device const float* J [[buffer(0)]],
    device const float* qvel [[buffer(1)]],
    device const float* velocity_scale [[buffer(2)]],
    device const float* workspace_debug [[buffer(3)]],
    device float* position_context [[buffer(4)]],
    constant int* dims [[buffer(5)]],
    uint tid [[thread_position_in_grid]]) {
  int nv = dims[1];
  int batch = dims[5];
  int nr = dims[9];
  int stride = dims[21];
  if (nv < 0 || nr <= 0 || batch <= 0 || stride < nr * nr + 7 * nr) return;
  uint total = uint(batch) * uint(nr);
  if (tid >= total) return;
  int world = int(tid / uint(nr));
  int row = int(tid % uint(nr));
  if (!cc_solver_world_enabled(dims, world)) return;
  device const float* J_world = ccj_world(J, world, nr, nv, dims);
  float jv = 0.0f;
  for (int dof = 0; dof < nv; ++dof)
    jv += ccj_get(J_world, row, dof, nv) * qvel[world * nv + dof];
  float B = velocity_scale[row];
  int context_base = (world * nr + row) * 5;
  int debug_base = world * stride;
  float ar = workspace_debug[debug_base + nr * nr + nr + row];
  // Encode the source-produced position residual into the standard five-word
  // context: -B*(J*qvel_new) - K*imp*(pos-margin), with K=imp=1.
  position_context[context_base] = 1.0f;
  position_context[context_base + 1] = B;
  position_context[context_base + 2] = 1.0f;
  position_context[context_base + 3] = -(ar + B * jv);
  position_context[context_base + 4] = 0.0f;
}

inline bool cc_refresh_copy_world_enabled(constant int* dims, int world) {
  return world >= 0 && world < dims[0] && dims[6 + world] != 0;
}

kernel void copy_selected_world_float(
    device const float* source [[buffer(0)]],
    device float* destination [[buffer(1)]],
    constant int* dims [[buffer(2)]],
    uint tid [[thread_position_in_grid]]) {
  int batch = dims[0], width = dims[1];
  int source_stride = dims[2], destination_stride = dims[3];
  int source_offset = dims[4], destination_offset = dims[5];
  if (batch <= 0 || width <= 0 || source_stride < source_offset + width
      || destination_stride < destination_offset + width) return;
  uint total = uint(batch) * uint(width);
  if (tid >= total) return;
  int world = int(tid / uint(width));
  if (!cc_refresh_copy_world_enabled(dims, world)) return;
  int lane = int(tid % uint(width));
  destination[world * destination_stride + destination_offset + lane] =
      source[world * source_stride + source_offset + lane];
}

kernel void copy_selected_world_int(
    device const int* source [[buffer(0)]],
    device int* destination [[buffer(1)]],
    constant int* dims [[buffer(2)]],
    uint tid [[thread_position_in_grid]]) {
  int batch = dims[0], width = dims[1];
  int source_stride = dims[2], destination_stride = dims[3];
  int source_offset = dims[4], destination_offset = dims[5];
  if (batch <= 0 || width <= 0 || source_stride < source_offset + width
      || destination_stride < destination_offset + width) return;
  uint total = uint(batch) * uint(width);
  if (tid >= total) return;
  int world = int(tid / uint(width));
  if (!cc_refresh_copy_world_enabled(dims, world)) return;
  int lane = int(tid % uint(width));
  destination[world * destination_stride + destination_offset + lane] =
      source[world * source_stride + source_offset + lane];
}

kernel void add_selected_world_float(
    device const float* source [[buffer(0)]],
    device float* destination [[buffer(1)]],
    constant int* dims [[buffer(2)]],
    uint tid [[thread_position_in_grid]]) {
  int batch = dims[0], width = dims[1];
  int source_stride = dims[2], destination_stride = dims[3];
  int source_offset = dims[4], destination_offset = dims[5];
  if (batch <= 0 || width <= 0 || source_stride < source_offset + width
      || destination_stride < destination_offset + width) return;
  uint total = uint(batch) * uint(width);
  if (tid >= total) return;
  int world = int(tid / uint(width));
  if (!cc_refresh_copy_world_enabled(dims, world)) return;
  int lane = int(tid % uint(width));
  int index = world * destination_stride + destination_offset + lane;
  destination[index] +=
      source[world * source_stride + source_offset + lane];
}

kernel void clear_selected_world_int(
    device int* destination [[buffer(0)]],
    constant int* dims [[buffer(1)]],
    uint tid [[thread_position_in_grid]]) {
  int batch = dims[0], width = dims[1];
  int destination_stride = dims[3], destination_offset = dims[5];
  if (batch <= 0 || width <= 0
      || destination_stride < destination_offset + width) return;
  uint total = uint(batch) * uint(width);
  if (tid >= total) return;
  int world = int(tid / uint(width));
  if (!cc_refresh_copy_world_enabled(dims, world)) return;
  int lane = int(tid % uint(width));
  destination[world * destination_stride + destination_offset + lane] = 0;
}

kernel void clear_selected_world_float(
    device float* destination [[buffer(0)]],
    constant int* dims [[buffer(1)]],
    uint tid [[thread_position_in_grid]]) {
  int batch = dims[0], width = dims[1];
  int destination_stride = dims[3], destination_offset = dims[5];
  if (batch <= 0 || width <= 0
      || destination_stride < destination_offset + width) return;
  uint total = uint(batch) * uint(width);
  if (tid >= total) return;
  int world = int(tid / uint(width));
  if (!cc_refresh_copy_world_enabled(dims, world)) return;
  int lane = int(tid % uint(width));
  destination[world * destination_stride + destination_offset + lane] = 0.0f;
}

kernel void fill_selected_world_int_one(
    device int* destination [[buffer(0)]],
    constant int* dims [[buffer(1)]],
    uint tid [[thread_position_in_grid]]) {
  int batch = dims[0], width = dims[1];
  int destination_stride = dims[3], destination_offset = dims[5];
  if (batch <= 0 || width <= 0
      || destination_stride < destination_offset + width) return;
  uint total = uint(batch) * uint(width);
  if (tid >= total) return;
  int world = int(tid / uint(width));
  if (!cc_refresh_copy_world_enabled(dims, world)) return;
  int lane = int(tid % uint(width));
  destination[world * destination_stride + destination_offset + lane] = 1;
}

kernel void refresh_cached_constraint_aref(
    device const float* J [[buffer(0)]],
    device const float* qvel [[buffer(1)]],
    device const float* position_context [[buffer(2)]],
    device const float* surface_velocity [[buffer(3)]],
    device const float* extra_aref [[buffer(4)]],
    device float* workspace_debug [[buffer(5)]],
    constant int* dims [[buffer(6)]],
    uint tid [[thread_position_in_grid]]) {
  int nv = dims[1];
  int batch = dims[5];
  int nr = dims[9];
  int stride = dims[21];
  if (nv < 0 || nr <= 0 || batch <= 0 || stride < nr * nr + 7 * nr) return;
  uint total = uint(batch) * uint(nr);
  if (tid >= total) return;
  int world = int(tid / uint(nr));
  int row = int(tid % uint(nr));
  if (!cc_solver_world_enabled(dims, world)) return;
  int context_base = (world * nr + row) * 5;
  int debug_base = world * stride;
  int active_offset = nr * nr + 6 * nr + row;
  int aref_offset = nr * nr + nr + row;
  if (workspace_debug[debug_base + active_offset] <= 0.5f) {
    workspace_debug[debug_base + aref_offset] = 0.0f;
    return;
  }
  float velocity = surface_velocity[world * nr + row];
  device const float* J_world = ccj_world(J, world, nr, nv, dims);
  for (int dof = 0; dof < nv; ++dof)
    velocity += ccj_get(J_world, row, dof, nv) * qvel[world * nv + dof];
  float K = position_context[context_base];
  float B = position_context[context_base + 1];
  float impedance = position_context[context_base + 2];
  float pos = position_context[context_base + 3];
  float margin = position_context[context_base + 4];
  workspace_debug[debug_base + aref_offset] =
      -B * velocity - K * impedance * (pos - margin)
      + extra_aref[world * nr + row];
}

// Transport the captured two-float reference to the current cached velocity.
// The high-word ABI is written by refresh_cached_constraint_aref above. This
// kernel updates its residual using the exact velocity delta and a separately
// lowered residual for B, without rebuilding any row or allocating scratch.
kernel void refresh_cached_constraint_aref_low(
    device const float* J [[buffer(0)]],
    device const float* qvel [[buffer(1)]],
    device const float* old_qvel [[buffer(2)]],
    device const float* position_context [[buffer(3)]],
    device const float* velocity_scale_low [[buffer(4)]],
    device const float* old_extra_aref [[buffer(5)]],
    device const float* new_extra_aref [[buffer(6)]],
    device const float* position_cache_rows [[buffer(7)]],
    device const float* position_cache_aref_low [[buffer(8)]],
    device float* workspace_debug [[buffer(9)]],
    constant int* dims [[buffer(10)]],
    uint tid [[thread_position_in_grid]]) {
  int nv = dims[1];
  int batch = dims[5];
  int nr = dims[9];
  int stride = dims[21];
  if (nv < 0 || nr <= 0 || batch <= 0
      || stride < nr * nr + 7 * nr) return;
  uint total = uint(batch) * uint(nr);
  if (tid >= total) return;
  int world = int(tid / uint(nr));
  int row = int(tid % uint(nr));
  if (!cc_solver_world_enabled(dims, world)) return;
  int row_index = world * nr + row;
  int context_base = row_index * 5;
  int debug_base = world * stride;
  int active_offset = nr * nr + 6 * nr + row;
  int aref_offset = nr * nr + nr + row;
  int cache_row_base = world * 7 * nr;
  int solver_metadata = dims[24] + nr * 10;
  device float* solver_scratch = workspace_debug + debug_base + nr * nr + 7 * nr;
  device float* pair_tail = solver_scratch + dims[solver_metadata + 15];
  device float* aref_low = pair_tail + 8 * max(nv, 1) + 2 * max(nr, 1);
  if (workspace_debug[debug_base + active_offset] <= 0.5f) {
    aref_low[row] = 0.0f;
    return;
  }
  PrimalFloatPair velocity_delta = {0.0f, 0.0f};
  device const float* J_world = ccj_world(J, world, nr, nv, dims);
  for (int dof = 0; dof < nv; ++dof) {
    float j = ccj_get(J_world, row, dof, nv);
    velocity_delta = primal_pair_add(velocity_delta,
        primal_pair_product({j, 0.0f}, {qvel[world * nv + dof], 0.0f}));
    velocity_delta = primal_pair_add(velocity_delta,
        primal_pair_product({-j, 0.0f}, {old_qvel[world * nv + dof], 0.0f}));
  }
  PrimalFloatPair coeff = {
      position_context[context_base + 1], velocity_scale_low[row]};
  PrimalFloatPair reference_delta = primal_pair_multiply(
      primal_pair_product(coeff, velocity_delta), -1.0f);
  float extra_delta = new_extra_aref[row_index] - old_extra_aref[row_index];
  PrimalFloatPair old_reference = primal_pair_add(
      {position_cache_rows[cache_row_base + nr + row],
       position_cache_aref_low[row_index]},
      primal_pair_add(reference_delta, {extra_delta, 0.0f}));
  float new_high = workspace_debug[debug_base + aref_offset];
  PrimalFloatPair remainder = primal_pair_add(old_reference,
      {-new_high, 0.0f});
  aref_low[row] = remainder.hi + remainder.lo;
}

// Compute pinned dense mj_jacDot(qpos, qvel) * qvel for connect/weld rows
// using retained position arrays and current smooth cdof/cdof_dot/cvel. This
// velocity-only producer never assembles a Jacobian or a position residual.
inline void cached_eq_jacdot_point(
    int world, int body, float3 point, int nv, int nbody,
    int rootid, int weldid, int dofadr, int dofnum,
    device const int* body_rootid, device const int* body_weldid,
    device const int* body_dofadr, device const int* body_dofnum,
    device const int* dof_parentid, device const int* dof_bodyid,
    device const int* dof_jntid, device const int* jnt_type,
    device const int* jnt_dofadr, device const float* cvel,
    device const float* root_com, device const float* cdof,
    device const float* cdof_dot, device const float* qvel,
    thread float3& jdv, thread float3& jrdv) {
  jdv = float3(0.0f);
  jrdv = float3(0.0f);
  if (body < 0 || body >= nbody) return;
  float3 com = float3(root_com[(world*nbody + rootid)*3],
                      root_com[(world*nbody + rootid)*3 + 1],
                      root_com[(world*nbody + rootid)*3 + 2]);
  float3 offset = point - com;
  float3 omega = float3(cvel[(world*nbody + body)*6],
                        cvel[(world*nbody + body)*6 + 1],
                        cvel[(world*nbody + body)*6 + 2]);
  float3 vcom = float3(cvel[(world*nbody + body)*6 + 3],
                       cvel[(world*nbody + body)*6 + 4],
                       cvel[(world*nbody + body)*6 + 5]);
  float3 pvel = vcom + cross(omega, offset);
  int dof = body_dofadr[weldid] + body_dofnum[weldid] - 1;
  while (dof >= 0) {
    int base = (world*nv + dof)*6;
    float3 motion_w = float3(cdof[base], cdof[base+1], cdof[base+2]);
    float3 motion_v = float3(cdof[base+3], cdof[base+4], cdof[base+5]);
    float3 derivative_w = float3(cdof_dot[base], cdof_dot[base+1], cdof_dot[base+2]);
    float3 derivative_v = float3(cdof_dot[base+3], cdof_dot[base+4], cdof_dot[base+5]);
    int joint = dof_jntid[dof];
    int jt = jnt_type[joint];
    bool quaternion_dof = jt == 1 || (jt == 0 && dof >= jnt_dofadr[joint] + 3);
    if (quaternion_dof) {
      int body_dof = dof_bodyid[dof];
      float3 body_w = float3(cvel[(world*nbody + body_dof)*6],
                             cvel[(world*nbody + body_dof)*6 + 1],
                             cvel[(world*nbody + body_dof)*6 + 2]);
      float3 body_v = float3(cvel[(world*nbody + body_dof)*6 + 3],
                             cvel[(world*nbody + body_dof)*6 + 4],
                             cvel[(world*nbody + body_dof)*6 + 5]);
      derivative_w = cross(body_w, motion_w);
      derivative_v = cross(body_w, motion_v) + cross(body_v, motion_w);
    }
    float3 point_derivative = derivative_v + cross(derivative_w, offset)
                              + cross(motion_w, pvel);
    float speed = qvel[world*nv + dof];
    jdv += speed * point_derivative;
    jrdv += speed * derivative_w;
    dof = dof_parentid[dof];
  }
}

kernel void refresh_cached_equality_jdot(
    device const float* qvel [[buffer(0)]],
    device const float* cvel [[buffer(1)]],
    device const float* cdof [[buffer(2)]],
    device const float* cdof_dot [[buffer(3)]],
    device const float* body_pos [[buffer(4)]],
    device const float* body_quat [[buffer(5)]],
    device const float* root_com [[buffer(6)]],
    device const float* site_pos [[buffer(7)]],
    device const float* site_quat [[buffer(8)]],
    device const int* eq_type [[buffer(9)]],
    device const int* eq_objtype [[buffer(10)]],
    device const int* eq_obj [[buffer(11)]],
    device const int* eq_rowadr [[buffer(12)]],
    device const float* eq_data [[buffer(13)]],
    device const int* site_bodyid [[buffer(14)]],
    device const int* body_rootid [[buffer(15)]],
    device const int* body_weldid [[buffer(16)]],
    device const int* body_dofadr [[buffer(17)]],
    device const int* body_dofnum [[buffer(18)]],
    device const int* dof_parentid [[buffer(19)]],
    device const int* dof_bodyid [[buffer(20)]],
    device const int* dof_jntid [[buffer(21)]],
    device const int* jnt_type [[buffer(22)]],
    device const int* jnt_dofadr [[buffer(23)]],
    device float* extra_aref [[buffer(24)]],
    device const float* workspace_J [[buffer(25)]],
    device const float* body_invweight0 [[buffer(26)]],
    device const int* eq_active [[buffer(27)]],
    constant int* dims [[buffer(28)]],
    uint world [[thread_position_in_grid]]) {
  int nv = dims[1], neq = dims[3], batch = dims[5], nr = dims[9];
  int nbody = dims[11], njnt = dims[12], nsite = dims[13];
  int disableflags = dims[6];
  if (world >= uint(batch)) return;
  if (!cc_solver_world_enabled(dims, int(world))) return;
  int outbase = int(world)*nr;
  for (int r=0; r<nr; ++r) extra_aref[outbase+r] = 0.0f;
  if (nv <= 0 || neq <= 0) return;
  for (int e=0; e<neq; ++e) {
    int typ = eq_type[e];
    if (typ != 0 && typ != 1) continue;
    if ((disableflags & 3) != 0 || eq_active[int(world) * neq + e] == 0) continue;
    int row = eq_rowadr[e];
    int span = typ == 0 ? 3 : 6;
    if (row < 0 || row + span > nr) continue;
    int objtype = eq_objtype[e];
    int obj1 = eq_obj[e*2], obj2 = eq_obj[e*2+1];
    int body1 = obj1, body2 = obj2;
    float3 point1 = float3(0.0f), point2 = float3(0.0f);
    if (objtype == 6) {
      if (obj1 < 0 || obj1 >= nsite || obj2 < 0 || obj2 >= nsite) continue;
      body1 = site_bodyid[obj1]; body2 = site_bodyid[obj2];
      point1 = float3(site_pos[(world*nsite+obj1)*3], site_pos[(world*nsite+obj1)*3+1], site_pos[(world*nsite+obj1)*3+2]);
      point2 = float3(site_pos[(world*nsite+obj2)*3], site_pos[(world*nsite+obj2)*3+1], site_pos[(world*nsite+obj2)*3+2]);
    } else {
      if (body1 < 0 || body1 >= nbody || body2 < 0 || body2 >= nbody) continue;
      int local1 = (typ == 0 ? 0 : 3), local2 = (typ == 0 ? 3 : 0);
      float3 l1 = float3(eq_data[e*11+local1], eq_data[e*11+local1+1], eq_data[e*11+local1+2]);
      float3 l2 = float3(eq_data[e*11+local2], eq_data[e*11+local2+1], eq_data[e*11+local2+2]);
      float4 q1 = float4(body_quat[(world*nbody+body1)*4], body_quat[(world*nbody+body1)*4+1], body_quat[(world*nbody+body1)*4+2], body_quat[(world*nbody+body1)*4+3]);
      float4 q2 = float4(body_quat[(world*nbody+body2)*4], body_quat[(world*nbody+body2)*4+1], body_quat[(world*nbody+body2)*4+2], body_quat[(world*nbody+body2)*4+3]);
      point1 = float3(body_pos[(world*nbody+body1)*3], body_pos[(world*nbody+body1)*3+1], body_pos[(world*nbody+body1)*3+2]) + eq_qrot(q1,l1);
      point2 = float3(body_pos[(world*nbody+body2)*3], body_pos[(world*nbody+body2)*3+1], body_pos[(world*nbody+body2)*3+2]) + eq_qrot(q2,l2);
    }
    int root1 = body_rootid[body1], root2 = body_rootid[body2];
    int weld1 = body_weldid[body1], weld2 = body_weldid[body2];
    float3 jdv1, jdv2, jrdv1, jrdv2;
    cached_eq_jacdot_point(int(world), body1, point1, nv, nbody, root1, weld1,
        body_dofadr[weld1], body_dofnum[weld1], body_rootid, body_weldid,
        body_dofadr, body_dofnum, dof_parentid, dof_bodyid, dof_jntid,
        jnt_type, jnt_dofadr, cvel, root_com, cdof, cdof_dot, qvel, jdv1, jrdv1);
    cached_eq_jacdot_point(int(world), body2, point2, nv, nbody, root2, weld2,
        body_dofadr[weld2], body_dofnum[weld2], body_rootid, body_weldid,
        body_dofadr, body_dofnum, dof_parentid, dof_bodyid, dof_jntid,
        jnt_type, jnt_dofadr, cvel, root_com, cdof, cdof_dot, qvel, jdv2, jrdv2);
    float3 transl = jdv1 - jdv2;
    float invt1 = body_invweight0[body1*2];
    float invt2 = body_invweight0[body2*2];
    float invr1 = body_invweight0[body1*2+1];
    float invr2 = body_invweight0[body2*2+1];
    float jmax = 0.0f;
    for (int k=0; k<span; ++k)
      for (int dof=0; dof<nv; ++dof)
        jmax = max(jmax, abs(ccj_get(ccj_world(workspace_J, int(world), nr, nv, dims), row + k, dof, nv)));
    bool degenerate = (jmax == 0.0f && invt1 + invt2 == 0.0f
                       && (typ != 1 || invr1 + invr2 == 0.0f));
    if (degenerate) transl = float3(0.0f);
    extra_aref[outbase+row] = -transl.x;
    extra_aref[outbase+row+1] = -transl.y;
    extra_aref[outbase+row+2] = -transl.z;
    if (typ != 1) continue;
    float4 q0base, q0r, q1;
    if (objtype == 6) {
      q0base = float4(site_quat[(world*nsite+obj1)*4], site_quat[(world*nsite+obj1)*4+1], site_quat[(world*nsite+obj1)*4+2], site_quat[(world*nsite+obj1)*4+3]);
      q1 = float4(site_quat[(world*nsite+obj2)*4], site_quat[(world*nsite+obj2)*4+1], site_quat[(world*nsite+obj2)*4+2], site_quat[(world*nsite+obj2)*4+3]);
      q0r = q0base;
    } else {
      float4 bq0 = float4(body_quat[(world*nbody+body1)*4], body_quat[(world*nbody+body1)*4+1], body_quat[(world*nbody+body1)*4+2], body_quat[(world*nbody+body1)*4+3]);
      float4 bq1 = float4(body_quat[(world*nbody+body2)*4], body_quat[(world*nbody+body2)*4+1], body_quat[(world*nbody+body2)*4+2], body_quat[(world*nbody+body2)*4+3]);
      float4 rel = float4(eq_data[e*11+6], eq_data[e*11+7], eq_data[e*11+8], eq_data[e*11+9]);
      q0base = bq0; q0r = eq_qmul(bq0, rel); q1 = bq1;
    }
    float3 omega1 = float3(cvel[(world*nbody+body1)*6], cvel[(world*nbody+body1)*6+1], cvel[(world*nbody+body1)*6+2]);
    float3 omega2 = float3(cvel[(world*nbody+body2)*6], cvel[(world*nbody+body2)*6+1], cvel[(world*nbody+body2)*6+2]);
    float3 domega = omega1 - omega2;
    float4 qdot0 = eq_qderiv(q0base, omega1);
    float4 qdot0r = qdot0;
    if (objtype != 6) {
      float4 rel = float4(eq_data[e*11+6], eq_data[e*11+7], eq_data[e*11+8], eq_data[e*11+9]);
      qdot0r = eq_qmul(qdot0, rel);
    }
    float4 negq1 = eq_qneg(q1);
    float4 negqdot1 = eq_qneg(eq_qderiv(q1, omega2));
    float4 t1 = eq_qmul(eq_qmul_axis(negqdot1, domega), q0r);
    float4 t2 = eq_qmul(eq_qmul_axis(negq1, jrdv1-jrdv2), q0r);
    float4 t3 = eq_qmul(eq_qmul_axis(negq1, domega), qdot0r);
    float scale = 0.5f * eq_data[e*11+10];
    extra_aref[outbase+row+3] = degenerate ? 0.0f : -scale * (t1.y + t2.y + t3.y);
    extra_aref[outbase+row+4] = degenerate ? 0.0f : -scale * (t1.z + t2.z + t3.z);
    extra_aref[outbase+row+5] = degenerate ? 0.0f : -scale * (t1.w + t2.w + t3.w);
  }
}
// Opt-in test witness for the real paired CG mass preconditioner. This kernel
// is intentionally separate from production dispatch and receives the same
// dense mass/factor, awake-tree map, and island selector as the helper.
kernel void primal_cg_mass_precondition_pair_witness(
    device const float* input [[buffer(0)]],
    constant int* dims [[buffer(1)]],
    device const int* awake_tree [[buffer(2)]],
    device const int* dof_island [[buffer(3)]],
    device float* output [[buffer(4)]],
    uint tid [[thread_position_in_grid]]) {
  if (tid != 0) return;
  constexpr int nv = 3;
  device const float* mass = input;
  device const float* factor = input + 9;
  device const float* rhs_hi = input + 18;
  device const float* rhs_low = input + 21;
  device float* solution_hi = output;
  device float* solution_low = output + 3;
  device float* residual_work = output + 6;
  device float* residual_hi = output + 9;
  device float* residual_low = output + 12;
  device float* correction_hi = output + 15;
  device float* correction_work = output + 18;
  device float* low_stash = output + 21;
  bool ok = primal_cg_mass_precondition_pair_elliptic(
      mass, factor, false, input, dims, awake_tree, rhs_hi, rhs_low,
      solution_hi, solution_low, residual_work, residual_hi, residual_low,
      correction_hi, correction_work, low_stash, dof_island, 0, true, nv);
  output[24] = ok ? 1.0f : 0.0f;

  // Return the independently recomputed represented operator residual for
  // the selected principal block: r = rhs - M_selected*(x_hi+x_lo).
  for (int i = 0; i < nv; ++i) {
    PrimalFloatPair product = {0.0f, 0.0f};
    bool selected_i = solver_dof_awake(dims, awake_tree, i)
        && dof_island[i] == 0;
    if (selected_i) for (int j = 0; j < nv; ++j) {
      bool selected_j = solver_dof_awake(dims, awake_tree, j)
          && dof_island[j] == 0;
      if (!selected_j) continue;
      PrimalFloatPair x = primal_pair_add(
          {solution_hi[j], 0.0f}, {solution_low[j], 0.0f});
      product = primal_pair_add(product,
          primal_pair_product({mass[i * nv + j], 0.0f}, x));
    }
    PrimalFloatPair rhs = primal_pair_add(
        {rhs_hi[i], 0.0f}, {rhs_low[i], 0.0f});
    PrimalFloatPair r = selected_i
        ? primal_pair_add(rhs, {-product.hi, -product.lo}) : rhs;
    output[25 + i] = r.hi;
    output[28 + i] = r.lo;
  }
}

// Native witness for the accepted Newton line parameter. This compares the
// source-relevant two-word quotient/update with collapsing alpha to one float
// before applying the same represented direction and accumulator.
kernel void primal_line_alpha_witness(
    device const float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    uint tid [[thread_position_in_grid]]) {
  if (tid != 0) return;
  PrimalFloatPair slope = {input[0], input[1]};
  PrimalFloatPair curvature = {input[2], input[3]};
  PrimalFloatPair direction = {input[4], input[5]};
  PrimalFloatPair accumulator = {input[6], input[7]};
  PrimalFloatPair quotient = primal_pair_multiply(
      primal_pair_divide(slope, curvature), -1.0f);
  float collapsed_alpha = primal_pair_value(quotient);
  PrimalFloatPair paired_update = primal_pair_add(
      primal_pair_product(quotient, direction), accumulator);
  PrimalFloatPair collapsed_update = primal_pair_fma(
      collapsed_alpha, direction, accumulator);
  output[0] = quotient.hi;
  output[1] = quotient.lo;
  output[2] = collapsed_alpha;
  output[3] = paired_update.hi;
  output[4] = paired_update.lo;
  output[5] = collapsed_update.hi;
  output[6] = collapsed_update.lo;
}

kernel void primal_elliptic_line_pair_witness(
    device const float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    uint tid [[thread_position_in_grid]]) {
  if (tid != 0) return;
  thread PrimalFloatPair residual[6], direction[6];
  thread PrimalFloatPair derivative, second_derivative;
  for (int i = 0; i < 6; ++i) {
    residual[i] = {input[i], input[6 + i]};
    direction[i] = {input[12 + i], input[18 + i]};
  }
  int dim = int(input[37]);
  PrimalFloatPair alpha = {input[35], input[36]};
  PrimalFloatPair delta = primal_elliptic_line_eval_pair(alpha,
      residual, direction, input + 24, input + 30, dim,
      derivative, second_derivative);
  output[0] = delta.hi;
  output[1] = delta.lo;
  output[2] = derivative.hi;
  output[3] = derivative.lo;
  output[4] = second_derivative.hi;
  output[5] = second_derivative.lo;
}

// Diagnostic for the source elliptic cone Hessian in represented two-word
// row residuals. This shares the exact helper used by Newton's production
// Hessian application.
kernel void primal_elliptic_hessian_pair_witness(
    device const float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    uint tid [[thread_position_in_grid]]) {
  if (tid != 0) return;
  thread float residual_hi[6], residual_low[6];
  thread float hessian_hi[36], hessian_low[36];
  for (int i = 0; i < 6; ++i) {
    residual_hi[i] = input[i];
    residual_low[i] = input[6 + i];
  }
  primal_elliptic_hessian_pair(residual_hi, residual_low,
      input + 12, input + 18, int(input[23]), hessian_hi, hessian_low);
  for (int i = 0; i < 36; ++i) {
    output[i] = hessian_hi[i];
    output[36 + i] = hessian_low[i];
  }
}

kernel void primal_elliptic_force_pair_witness(
    device const float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    uint tid [[thread_position_in_grid]]) {
  if (tid != 0) return;
  thread float residual_hi[6], residual_low[6];
  thread float force_hi[6], force_low[6];
  for (int i = 0; i < 6; ++i) {
    residual_hi[i] = input[i];
    residual_low[i] = input[6 + i];
  }
  primal_elliptic_force_pair(residual_hi, residual_low,
      input + 12, input + 18, int(input[23]), force_hi, force_low);
  for (int i = 0; i < 6; ++i) {
    output[i] = force_hi[i];
    output[6 + i] = force_low[i];
  }
}

// Exercise the exact production rescale path on a captured six-DOF vector.
kernel void primal_tiny_rhs_rescale_witness(
    device const float* input [[buffer(0)]],
    constant int* dims [[buffer(1)]],
    device const int* awake [[buffer(2)]],
    device float* output [[buffer(3)]],
    uint tid [[thread_position_in_grid]]) {
  if (tid != 0) return;
  for (int i = 0; i < 6; ++i) {
    output[1 + i] = input[i];
    output[7 + i] = input[6 + i];
  }
  bool ok = primal_rescale_pair_vector(output + 1, output + 7, awake, dims,
      nullptr, -1, false, input[12], 6);
  output[0] = ok ? 1.0f : 0.0f;
}
