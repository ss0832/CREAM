// =============================================================================
// cell_pass0b_sort.wgsl  —  Pass 0b: count atoms per cell (atomicAdd)
//
// Part of the Prefix Sum cell list construction (Q9, replaces bitonic sort).
// common.wgsl is prepended via concat!. Bindings 0-4 declared there.
//
// Input:
//   binding 5 — cell_ids:     array<u32>      (read)  from pass0a
// Output:
//   binding 6 — cell_counts:  array<atomic<u32>>  (read_write)  initialized to 0
//
// Each thread atomically increments the count for its atom's cell.
// After dispatch, cell_counts[c] = number of atoms in cell c.
// The engine reads back cell_counts to CPU, computes prefix sum → cell_start,
// then uploads cell_start before pass0c.
// =============================================================================

@group(0) @binding(5) var<storage, read>            cell_ids:    array<u32>;
@group(0) @binding(6) var<storage, read_write>      cell_counts: array<atomic<u32>>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    // Chunked dispatch support (N > 4_194_240):
    //   params.i_offset — first atom index for this chunk.
    //   params.i_count  — number of atoms in this chunk (≤ 65535 × 64).
    let i = params.i_offset + gid.x;
    if gid.x >= params.i_count || i >= params.n_atoms { return; }

    atomicAdd(&cell_counts[cell_ids[i]], 1u);
}
