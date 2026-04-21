// =============================================================================
// cell_pass0a_assign.wgsl  —  Pass 0a: assign atoms to Morton-code cell IDs
//
// common.wgsl is prepended via concat!. Bindings 0-4 declared there.
//
// Fractional coordinates → (cx, cy, cz) in the padded grid → Morton code.
//
// Using the padded grid (nx_pad × ny_pad × nz_pad, each a power-of-two) ensures
// that Morton codes form a contiguous range [0, nx_pad*ny_pad*nz_pad), with no
// gaps caused by non-power-of-two cell counts.
//
// Atoms with cx >= n_cells_x etc. (in the padding zone) cannot exist because
// fractional coordinates are always in [0,1) and we clamp to n_cells-1.
// Padded cells beyond the real grid simply have count 0 in cell_counts_buf.
//
// Output:
//   binding 5 — cell_ids: array<u32>  — Morton code for each atom
// =============================================================================

@group(0) @binding(5) var<storage, read_write> cell_ids: array<u32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    // Chunked dispatch support (N > 4_194_240):
    //   params.i_offset — first atom index for this chunk.
    //   params.i_count  — number of atoms in this chunk (≤ 65535 × 64).
    //   For a full (non-chunked) dispatch: i_offset = 0, i_count = n_atoms.
    let i = params.i_offset + gid.x;
    if gid.x >= params.i_count || i >= params.n_atoms { return; }

    let pos = positions[i].xyz;

    // Fractional coordinates: s = pos @ H⁻¹
    let sx = dot(pos, params.hinv_col0.xyz);
    let sy = dot(pos, params.hinv_col1.xyz);
    let sz = dot(pos, params.hinv_col2.xyz);

    // Wrap fractional coord to [0, 1)
    let fx = sx - floor(sx);
    let fy = sy - floor(sy);
    let fz = sz - floor(sz);

    // Cell indices in real (non-padded) grid — clamp to avoid OOB on fx==1.0
    let cx = min(u32(fx * f32(params.n_cells_x)), params.n_cells_x - 1u);
    let cy = min(u32(fy * f32(params.n_cells_y)), params.n_cells_y - 1u);
    let cz = min(u32(fz * f32(params.n_cells_z)), params.n_cells_z - 1u);

    // Morton code uses padded grid — cx/cy/cz are always < n_cells_{x,y,z} <= n_cells_{x,y,z}_pad
    cell_ids[i] = morton3(cx, cy, cz);
}
