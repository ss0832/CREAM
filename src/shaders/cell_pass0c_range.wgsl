// =============================================================================
// cell_pass0c_range.wgsl  —  Pass 0c: scatter atoms into sorted_atoms[]
//
// Part of the Prefix Sum cell list construction (Q9).
// common.wgsl is prepended via concat!. Bindings 0-4 declared there.
//
// Input:
//   binding 5 — cell_ids:      array<u32>           (read)  from pass0a
//   binding 6 — write_offsets: array<atomic<u32>>   (read_write)
//               Initialized by CPU to cell_start[0..n_cells] before dispatch.
//               After pass0c, write_offsets[c] == cell_start[c+1].
// Output:
//   binding 7 — sorted_atoms:  array<u32>           (write)
//               sorted_atoms[cell_start[c]..cell_start[c+1]] = atoms in cell c.
//               Order within a cell is arbitrary but deterministic per frame.
//
// Chunked dispatch (N > 4_194_240 support):
//   params.i_offset — first atom index for this chunk.
//   params.i_count  — number of atoms in this chunk (≤ 65535 × 64).
//   For a full (non-chunked) dispatch: i_offset = 0, i_count = n_atoms.
//   The Rust engine loops over ceil(N / MAX_DISPATCH_ATOMS) chunks.
//
// Algorithm: conflict-free scatter using per-cell atomic write cursor.
//   slot = atomicAdd(&write_offsets[cell], 1)
//   sorted_atoms[slot] = i
// =============================================================================

@group(0) @binding(5) var<storage, read>            cell_ids:      array<u32>;
@group(0) @binding(6) var<storage, read_write>      write_offsets: array<atomic<u32>>;
@group(0) @binding(7) var<storage, read_write>      sorted_atoms:  array<u32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    // Chunked: atom index i = i_offset + gid.x
    let i = params.i_offset + gid.x;
    if gid.x >= params.i_count || i >= params.n_atoms { return; }

    let c    = cell_ids[i];
    let slot = atomicAdd(&write_offsets[c], 1u);
    sorted_atoms[slot] = i;
}
