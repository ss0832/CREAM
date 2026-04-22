// =============================================================================
// eam_pass1_neighlist.wgsl  —  Pass 1 (density) — CSR neighbour-list driven
//
// Replaces the GPU cell-list pipeline for single-point EAM calculations.
// The neighbour list is built on CPU (bit-for-bit identical f32 min-image
// arithmetic to this shader) and uploaded as two buffers:
//     nl_starts[N+1]  — CSR row pointers into nl_list
//     nl_list[total]  — flat j-indices, i's neighbours are
//                        nl_list[nl_starts[i] .. nl_starts[i+1]]
//
// common.wgsl is prepended via concat!.  Bindings 0–4 declared there
// (positions/atom_types/params/potential_tables/tbl).
//
// Bindings:
//   binding 5  — densities  (write, original atom order)
//   binding 6  — nl_starts  (read, length N+1)
//   binding 7  — nl_list    (read, flat list)
//   binding 10 — dbg        (atomic<u32>*32, diagnostics)
//
// Atom ordering:
//   Unlike the cell-list shaders, there is NO Morton reordering here.  Thread
//   `gid.x` processes atom index `i = params.i_offset + gid.x` directly, and
//   positions[i] / atom_types[i] are the original user-supplied buffers.
//
// Diagnostics (same dbg slots as the cell-list shaders so tooling still works):
//   [5] nan_rho_count
//   [6] neighbour_count   — pairs visited (before cutoff test)
//   [7] cutoff_hit_count  — pairs passing cutoff
// =============================================================================

override ENABLE_DEBUG: bool = false;

@group(0) @binding(5)  var<storage, read_write> densities: array<f32>;
@group(0) @binding(6)  var<storage, read>       nl_starts: array<u32>;
@group(0) @binding(7)  var<storage, read>       nl_list:   array<u32>;
@group(0) @binding(10) var<storage, read_write> dbg:       array<atomic<u32>, 32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    // Chunked-dispatch support (matches CellList / AllPairs conventions).
    let i = params.i_offset + gid.x;
    if gid.x >= params.i_count || i >= params.n_atoms { return; }

    let pos_i = positions[i].xyz;

    // Neighbour range for atom i
    let s = nl_starts[i];
    let e = nl_starts[i + 1u];

    // Neumaier-compensated accumulator
    var rho: f32 = 0.0;
    var kc:  f32 = 0.0;

    for (var idx: u32 = s; idx < e; idx = idx + 1u) {
        let j    = nl_list[idx];
        let dv   = min_image(positions[j].xyz - pos_i);
        let r_sq = dot(dv, dv);

        if ENABLE_DEBUG { atomicAdd(&dbg[6], 1u); }

        // CPU-side filter already applied |d| < cutoff, but re-check in f32
        // to be robust against rounding at the exact boundary.  MIN_R_SQ
        // guards against degenerate self/overlap pairs.
        if r_sq < params.cutoff_sq && r_sq > MIN_R_SQ {
            if ENABLE_DEBUG { atomicAdd(&dbg[7], 1u); }
            let r    = sqrt(r_sq);
            let _val = lookup_by_r(tbl.rho_offset, atom_types[j], r);
            let t    = rho + _val;
            if abs(rho) >= abs(_val) { kc += (rho - t) + _val; }
            else                     { kc += (_val - t) + rho; }
            rho = t;
        }
    }

    if ENABLE_DEBUG && (!(rho == rho) || rho > 1e30 || rho < -1e30) {
        atomicAdd(&dbg[5], 1u);
    }

    densities[i] = rho + kc;
}
