// =============================================================================
// cell_pass0d_reorder.wgsl  —  Pass 0d: reorder positions and types by Morton order
//
// common.wgsl is prepended via concat!.  Bindings 0–4 declared there.
//
// After pass0c, sorted_atoms[k] = original atom index j for Morton slot k.
// This pass physically reorders the position and type arrays so that
//   reordered_positions[k].xyz = positions[ sorted_atoms[k] ].xyz
//   reordered_positions[k].w   = f32(cell_ids[ sorted_atoms[k] ])  — Morton cell ID
//   reordered_types[k]         = atom_types[ sorted_atoms[k] ]
//
// The W component carries the authoritative Morton cell ID assigned by pass0a.
// Pass1/pass2 read it back via morton3_x/y/z to obtain guaranteed-consistent
// home-cell coordinates, eliminating float-rounding mismatches when the
// fractional-coordinate dot product is recomputed in a different shader.
//
// After this pass, CellList passes 1 and 2 bind reordered_positions /
// reordered_types to common.wgsl bindings 0 / 1 respectively.  The inner-loop
// reads "positions[c_s + si]" and "atom_types[c_s + si]" then become sequential
// (coalesced) global-memory accesses, removing the per-pair sorted_atoms[si]
// indirection that was the main bottleneck.
//
// Bindings:
//   binding 5 — sorted_atoms:          array<u32>        (read)
//   binding 6 — reordered_positions:   array<vec4<f32>>  (write, W = cell ID)
//   binding 7 — reordered_types:       array<u32>        (write)
//   binding 8 — cell_ids:              array<u32>        (read, from pass0a)
//
// Note: bindings 3 (potential_tables) and 4 (tbl) are declared in common.wgsl
// but are unused here; WebGPU allows omitting unused bindings from the layout.
//
// Chunked dispatch (N > 4_194_240 support):
//   params.i_offset — first Morton slot k for this chunk.
//   params.i_count  — number of slots in this chunk (≤ 65535 × 64).
//   For a full (non-chunked) dispatch: i_offset = 0, i_count = n_atoms.
// =============================================================================

@group(0) @binding(5) var<storage, read>       sorted_atoms:        array<u32>;
@group(0) @binding(6) var<storage, read_write> reordered_positions: array<vec4<f32>>;
@group(0) @binding(7) var<storage, read_write> reordered_types:     array<u32>;
@group(0) @binding(8) var<storage, read>       cell_ids:            array<u32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    // Chunked: Morton slot k = i_offset + gid.x
    let k = params.i_offset + gid.x;
    if gid.x >= params.i_count || k >= params.n_atoms { return; }
    let j                   = sorted_atoms[k];
    reordered_positions[k]  = vec4<f32>(positions[j].xyz, f32(cell_ids[j]));
    reordered_types[k]      = atom_types[j];
}
