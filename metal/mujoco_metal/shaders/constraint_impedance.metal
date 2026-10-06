#include <metal_stdlib>
using namespace metal;

constant int IMP_J_MAGIC = 0x4d4a4353;
constant int IMP_J_VERSION = 1;

inline float impedance_j_value(device const float* storage, int world,
                               int row, int dof, int nr, int nv,
                               bool packed) {
  if (!packed) return storage[(world * nr + row) * nv + dof];
  device const float* record = storage;
  device const int* h = reinterpret_cast<device const int*>(record);
  if (h[0] != IMP_J_MAGIC || h[1] != IMP_J_VERSION
      || h[3] != nr || h[4] != nv || row < 0 || row >= nr
      || dof < 0 || dof >= nv || h[9] < h[8]) return 0.0f;
  record = storage + world * h[9];
  h = reinterpret_cast<device const int*>(record);
  if (h[0] != IMP_J_MAGIC || h[1] != IMP_J_VERSION
      || h[3] != nr || h[4] != nv) return 0.0f;
  if (h[2] == 0) return record[h[8] + row * nv + dof];
  if (h[2] != 1) return 0.0f;
  device const int* rowptr = reinterpret_cast<device const int*>(record) + h[6];
  device const int* columns = reinterpret_cast<device const int*>(record) + h[7];
  int lo = rowptr[row], hi = rowptr[row + 1];
  while (lo < hi) {
    int mid = (lo + hi) >> 1;
    if (columns[mid] < dof) lo = mid + 1;
    else hi = mid;
  }
  if (lo >= rowptr[row + 1] || columns[lo] != dof) return 0.0f;
  return record[h[8] + lo];
}

kernel void mask_active_rows(
    device const float* source_active [[buffer(0)]],
    device const int* world_status [[buffer(1)]],
    device int* active_rows [[buffer(2)]],
    constant int* dims [[buffer(3)]],
    uint2 tid [[thread_position_in_grid]]) {
  int batch=dims[0], rows=dims[1];
  int world=int(tid.x), row=int(tid.y);
  if (world>=batch || row>=rows) return;
  int index=world*rows+row;
  active_rows[index]=(world_status[world]==0 && source_active[index]!=0.0f)
      ? 1 : 0;
}

// Build exact mass-solve RHS from canonical rows. The last component RHS is
// reserved for the smooth force by the shared factor ABI; DIAGEXACT writes it
// as zero because only J^T rows contribute to diag(J M^-1 J^T).
kernel void pack_component_inverse_rhs(
    device const float* jacobian [[buffer(0)]],
    device const int* active_rows [[buffer(1)]],
    device float* rhs [[buffer(2)]],
    constant int* dims [[buffer(3)]],
    uint3 gid [[thread_position_in_grid]]) {
  int world = int(gid.x), row = int(gid.y), dof = int(gid.z);
  int batch = dims[0], nr = dims[1], nv = dims[2];
  if (world >= batch || dof >= nv || row > nr) return;
  int out = (world * (nr + 1) + row) * nv + dof;
  if (row == nr || active_rows[world * nr + row] == 0) {
    rhs[out] = 0.0f;
    return;
  }
  rhs[out] = impedance_j_value(jacobian, world, row, dof, nr, nv,
                                dims[4] != 0);
}

// Dense factored solves consume [B,nv,nrhs]; transpose the canonical row
// Jacobian into columns without an intermediate host or dense matrix.
kernel void pack_dense_inverse_rhs(
    device const float* jacobian [[buffer(0)]],
    device const int* active_rows [[buffer(1)]],
    device float* rhs [[buffer(2)]],
    constant int* dims [[buffer(3)]],
    uint3 gid [[thread_position_in_grid]]) {
  int world = int(gid.x), row = int(gid.y), dof = int(gid.z);
  int batch = dims[0], nr = dims[1], nv = dims[2];
  if (world >= batch || row >= nr || dof >= nv) return;
  int out = (world * nv + dof) * nr + row;
  rhs[out] = active_rows[world * nr + row] != 0
      ? impedance_j_value(jacobian, world, row, dof, nr, nv,
                          dims[4] != 0) : 0.0f;
}

// MuJoCo 3.10.0 engine_core_constraint.c:mj_projectConstraint and
// mj_makeImpedance. Each dispatch uses the fixed canonical [B,nr,nv] ABI.
kernel void exact_constraint_diagonal(
    device const float* jacobian [[buffer(0)]],
    device const float* inverse_jacobian [[buffer(1)]],
    device const int* active_rows [[buffer(2)]],
    device float* diagonal [[buffer(3)]],
    constant int* dims [[buffer(4)]],
    uint2 gid [[thread_position_in_grid]]) {
  int world = int(gid.x), row = int(gid.y);
  int batch = dims[0], nr = dims[1], nv = dims[2];
  if (world >= batch || row >= nr) return;
  int out = world * nr + row;
  if (active_rows[out] == 0) {
    diagonal[out] = 0.0f;
    return;
  }
  float sum = 0.0f;
  if (dims[3] == 0) {
    int base = out * nv;
    for (int dof = 0; dof < nv; ++dof)
      sum += impedance_j_value(jacobian, world, row, dof, nr, nv,
                               dims[5] != 0)
          * inverse_jacobian[base + dof];
  } else {
    int rhs_rows = dims[4];
    int base = world * nv * nr + row;
    int jacobian_base = out * nv;
    for (int dof = 0; dof < nv; ++dof)
      sum += impedance_j_value(jacobian, world, row, dof, nr, nv,
                               dims[5] != 0)
          * inverse_jacobian[world * nv * nr + dof * nr + row];
  }
  if (dims[3] == 0 && dims[4] > nr) {
    // Component factors return canonical row-major RHS with one extra smooth
    // force row; the unused tail is intentionally excluded from diag(JM^-1J').
    int compact_base = out * nv;
    int padded_base = world * dims[4] * nv + row * nv;
    sum = 0.0f;
    for (int dof = 0; dof < nv; ++dof)
      sum += impedance_j_value(jacobian, world, row, dof, nr, nv,
                               dims[5] != 0)
          * inverse_jacobian[padded_base + dof];
  }
  diagonal[out] = sum;
}

kernel void recompute_row_impedance(
    device const float* exact_diagonal [[buffer(0)]],
    device const float* row_impedance [[buffer(1)]],
    device const int* active_rows [[buffer(2)]],
    device float* R [[buffer(3)]],
    constant int* dims [[buffer(4)]],
    uint2 gid [[thread_position_in_grid]]) {
  int world = int(gid.x), row = int(gid.y);
  int batch = dims[0], nr = dims[1];
  if (world >= batch || row >= nr) return;
  int index = world * nr + row;
  if (active_rows[index] == 0) {
    R[index] = 0.0f;
    return;
  }
  float imp = row_impedance[index];
  float diag = exact_diagonal[index];
  R[index] = max(1.0e-15f, (1.0f - imp) * diag / imp);
}

// cone_groups packs the static tuple (row_start, condim, kind), with kind 1
// elliptic, 2 pyramidal. Runtime activity comes from the canonical row mask;
// friction uses MuJoCo's five-value contact ordering.
kernel void scale_contact_cones(
    device float* R [[buffer(0)]],
    device const int* active_rows [[buffer(1)]],
    device const int* cone_groups [[buffer(2)]],
    device const float* friction [[buffer(3)]],
    device float* contact_mu [[buffer(4)]],
    constant int* dims [[buffer(5)]],
    uint2 gid [[thread_position_in_grid]]) {
  int world = int(gid.x), group = int(gid.y);
  int batch = dims[0], nr = dims[1], groups = dims[2];
  if (world >= batch || group >= groups) return;
  int meta = group * 3;
  int mu_index = world * groups + group;
  int start = cone_groups[meta];
  int condim = cone_groups[meta + 1];
  int kind = cone_groups[meta + 2];
  contact_mu[mu_index] = 0.0f;
  if (start < 0 || start >= nr ||
      (condim != 3 && condim != 4 && condim != 6) ||
      (kind != 1 && kind != 2)) return;
  bool active = active_rows[world * nr + start] != 0;
  if (!active) return;
  int nrows = kind == 1 ? condim : 2 * (condim - 1);
  if (start + nrows > nr) return;
  int base = world * nr + start;
  int fbase = group * 5;
  float normal = R[base];
  float R1 = normal / max(as_type<float>(dims[3]), 1.0e-15f);
  float mu0 = max(friction[fbase], 1.0e-5f);
  float mu = mu0 * sqrt(R1 / max(normal, 1.0e-15f));
  contact_mu[mu_index] = mu;
  if (kind == 1) {
    R[base + 1] = R1;
    for (int j = 1; j < condim - 1; ++j) {
      float fj = max(friction[fbase + j], 1.0e-5f);
      R[base + j + 1] = R1 * mu0 * mu0 / (fj * fj);
    }
  } else {
    float Rpy = 2.0f * mu * mu * normal;
    for (int row = 0; row < nrows; ++row) R[base + row] = Rpy;
  }
}

kernel void adjusted_constraint_diagonal(
    device const float* R [[buffer(0)]],
    device const float* row_impedance [[buffer(1)]],
    device const int* active_rows [[buffer(2)]],
    device float* diagonal_adjusted [[buffer(3)]],
    constant int* dims [[buffer(4)]],
    uint2 gid [[thread_position_in_grid]]) {
  int world = int(gid.x), row = int(gid.y);
  int batch = dims[0], nr = dims[1];
  if (world >= batch || row >= nr) return;
  int index = world * nr + row;
  if (active_rows[index] == 0) {
    diagonal_adjusted[index] = 0.0f;
    return;
  }
  float imp = row_impedance[index];
  diagonal_adjusted[index] = R[index] * imp / (1.0f - imp);
}
