#include <metal_stdlib>
using namespace metal;

inline int solver_island_root(device int* parent, uint base, int tree,
                              int ntree) {
  if (tree < 0 || tree >= ntree) return -1;
  int root = tree;
  for (int step = 0; step < ntree; ++step) {
    int next = parent[base + uint(root)];
    if (next == root) break;
    root = next;
  }
  int node = tree;
  for (int step = 0; step < ntree; ++step) {
    int next = parent[base + uint(node)];
    if (next == node) break;
    parent[base + uint(node)] = root;
    node = next;
  }
  return root;
}

inline void solver_island_union(device int* parent, device int* used,
                                uint base, int a, int b, int ntree) {
  if (a < 0 && b < 0) return;
  if (a < 0) a = b;
  if (b < 0) b = a;
  if (a < 0 || b < 0 || a >= ntree || b >= ntree) return;
  used[base + uint(a)] = 1;
  used[base + uint(b)] = 1;
  int ra = solver_island_root(parent, base, a, ntree);
  int rb = solver_island_root(parent, base, b, ntree);
  if (ra != rb) {
    int low = min(ra, rb), high = max(ra, rb);
    parent[base + uint(high)] = low;
  }
}

kernel void build_solver_island_partition(
    device const int* dof_tree [[buffer(0)]],
    device const float* jacobian [[buffer(1)]],
    device const float* row_activity [[buffer(2)]],
    device const int* row_tree_links [[buffer(3)]],
    device const int* row_group [[buffer(4)]],
    device const int* row_group_exempt [[buffer(5)]],
    constant int* dims [[buffer(6)]],
    device int* dof_island [[buffer(7)]],
    device int* row_island [[buffer(8)]],
    device int* row_order [[buffer(9)]],
    device int* row_inverse [[buffer(10)]],
    device int* island_offsets [[buffer(11)]],
    device int* island_count [[buffer(12)]],
    device int* parent [[buffer(13)]],
    device int* used_tree [[buffer(14)]],
    device int* root_label [[buffer(15)]],
    device int* row_first_tree [[buffer(16)]],
    uint world [[thread_position_in_grid]]) {
  int nr = dims[0], nv = dims[1], ntree = dims[2], batch = dims[3];
  int activity_stride = dims[4], activity_offset = dims[5];
  if (world >= uint(batch)) return;
  uint tbase = world * uint(ntree);
  uint rbase = world * uint(nr);
  uint jbase = world * uint(nr) * uint(nv);
  uint abase = world * uint(activity_stride) + uint(activity_offset);
  for (int tree = 0; tree < ntree; ++tree) {
    parent[tbase + uint(tree)] = tree;
    used_tree[tbase + uint(tree)] = 0;
    root_label[tbase + uint(tree)] = -1;
  }
  for (int row = 0; row < nr; ++row) {
    row_island[rbase + uint(row)] = -1;
    row_order[rbase + uint(row)] = -1;
    row_inverse[rbase + uint(row)] = -1;
    row_first_tree[rbase + uint(row)] = -1;
  }

  int previous_type = -2147483647;
  int previous_id = -1;
  int previous_first = -1;
  for (int row = 0; row < nr; ++row) {
    if (row_activity[abase + uint(row)] <= 0.5f) continue;
    int type = row_group[uint(row) * 2u];
    int ident = row_group[uint(row) * 2u + 1u];
    bool exempt = row_group_exempt[uint(row)] != 0;
    int first = -1;
    if (!exempt && type >= 0 && type == previous_type
        && ident == previous_id && previous_first >= 0) {
      first = previous_first;
    } else {
      int link0 = row_tree_links[uint(row) * 2u];
      int link1 = row_tree_links[uint(row) * 2u + 1u];
      if (link0 >= 0) first = link0;
      else if (link1 >= 0) first = link1;
      if (link0 >= 0 || link1 >= 0) {
        solver_island_union(parent, used_tree, tbase, link0, link1, ntree);
      }
      for (int dof = 0; dof < nv; ++dof) {
        if (jacobian[jbase + uint(row) * uint(nv) + uint(dof)] == 0.0f) continue;
        int tree = dof_tree[uint(dof)];
        if (tree < 0 || tree >= ntree) continue;
        used_tree[tbase + uint(tree)] = 1;
        if (first < 0) first = tree;
        else solver_island_union(parent, used_tree, tbase, first, tree, ntree);
      }
      if (link0 >= 0 && first >= 0 && link0 != first)
        solver_island_union(parent, used_tree, tbase, first, link0, ntree);
      if (link1 >= 0 && first >= 0 && link1 != first)
        solver_island_union(parent, used_tree, tbase, first, link1, ntree);
    }
    row_first_tree[rbase + uint(row)] = first;
    previous_type = type;
    previous_id = ident;
    previous_first = first;
  }

  int n_island = 0;
  for (int tree = 0; tree < ntree; ++tree) {
    if (used_tree[tbase + uint(tree)] == 0) continue;
    int root = solver_island_root(parent, tbase, tree, ntree);
    if (root_label[tbase + uint(root)] < 0)
      root_label[tbase + uint(root)] = n_island++;
  }
  for (int tree = 0; tree < ntree; ++tree) {
    int label = -1;
    if (used_tree[tbase + uint(tree)] != 0) {
      int root = solver_island_root(parent, tbase, tree, ntree);
      label = root_label[tbase + uint(root)];
    }
    root_label[tbase + uint(tree)] = label;
  }
  for (int dof = 0; dof < nv; ++dof) {
    int tree = dof_tree[uint(dof)];
    dof_island[world * uint(nv) + uint(dof)] =
        (tree >= 0 && tree < ntree) ? root_label[tbase + uint(tree)] : -1;
  }
  for (int row = 0; row < nr; ++row) {
    if (row_activity[abase + uint(row)] <= 0.5f) continue;
    int tree = row_first_tree[rbase + uint(row)];
    if (tree >= 0 && tree < ntree)
      row_island[rbase + uint(row)] = root_label[tbase + uint(tree)];
  }

  int cursor = 0;
  for (int island = 0; island < n_island; ++island) {
    island_offsets[world * uint(ntree + 1) + uint(island)] = cursor;
    for (int row = 0; row < nr; ++row) {
      if (row_island[rbase + uint(row)] != island) continue;
      row_order[rbase + uint(cursor)] = row;
      row_inverse[rbase + uint(row)] = cursor;
      ++cursor;
    }
  }
  for (int island = n_island; island <= ntree; ++island)
    island_offsets[world * uint(ntree + 1) + uint(island)] = cursor;
  island_count[world] = n_island;
}
