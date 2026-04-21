// =============================================================================
// eam_pass1_density.wgsl  ——  Pass 1: electron density accumulation
//
// common.wgsl is prepended via concat!; bindings 0-4 are declared there.
//
// ρᵢ = Σⱼ≠ᵢ f_β(rᵢⱼ)
//
// Flash Attention–style tiling (Q6):
//   j-loop is split into tiles of 64 atoms.  Thread `lid` collaboratively
//   loads tile_pos/tile_type into workgroup shared memory, then all 64 threads
//   compute against the same 64 loaded atoms.
//
//   With N atoms, this reduces global `positions[]` reads from N²/wg to N reads
//   total (one coalesced load per tile per workgroup).  On HBM-limited GPUs
//   (discrete, high atom-count) this gives a direct throughput improvement.
//
//   Workgroup shared memory used:
//     tile_pos  : 64 × 16 B = 1 024 B
//     tile_type : 64 ×  4 B =   256 B
//     Total     :           = 1 280 B  (< 16 KiB WebGPU minimum)
//
//   NOTE: all 64 threads must reach every workgroupBarrier().
//   Inactive threads (i >= n_atoms) still participate in tile loads; their
//   outputs are suppressed by the `if is_active` guard on the final write.
//
// Kahan compensated summation suppresses f32 round-off accumulation.
// =============================================================================

var<workgroup> tile_pos  : array<vec4<f32>, 64>;
var<workgroup> tile_type : array<u32,       64>;

@group(0) @binding(5) var<storage, read_write> densities: array<f32>;

@compute @workgroup_size(64)
fn main(
    @builtin(global_invocation_id)   gid: vec3<u32>,
    @builtin(local_invocation_index) lid: u32,
) {
    // In chunked mode i_offset > 0; in full-dispatch mode i_offset = 0.
    let i = params.i_offset + gid.x;
    let is_active = (gid.x < params.i_count) && (i < params.n_atoms);

    // Guard OOB read for inactive threads (read atom 0, result discarded)
    let si   = select(0u, i, is_active);
    let pos_i = positions[si].xyz;

    // Kahan compensated sum for ρᵢ
    var rho: f32 = 0.0;
    var kc:  f32 = 0.0;

    let n_tiles = (params.n_atoms + 63u) / 64u;

    for (var t = 0u; t < n_tiles; t++) {
        let tile_base = t * 64u;

        // ── Collaborative tile load (all 64 threads) ─────────────────────────
        let j_load = tile_base + lid;
        // Clamp OOB to last valid atom; inner loop re-checks j_global < n_atoms.
        let j_safe = min(j_load, params.n_atoms - 1u);
        tile_pos[lid]  = positions[j_safe];
        tile_type[lid] = atom_types[j_safe];
        workgroupBarrier();

        // ── Accumulate contributions from this tile ───────────────────────────
        if is_active {
            let tile_size = min(64u, params.n_atoms - tile_base);
            for (var tj = 0u; tj < tile_size; tj++) {
                let j = tile_base + tj;
                if j == i { continue; }

                let dv   = min_image(tile_pos[tj].xyz - pos_i);
                let r_sq = dot(dv, dv);

                if r_sq < params.cutoff_sq && r_sq > MIN_R_SQ {
                    let r  = sqrt(r_sq);
                    let y  = lookup_by_r(tbl.rho_offset, tile_type[tj], r) - kc;
                    let s  = rho + y;
                    kc     = (s - rho) - y;
                    rho    = s;
                }
            }
        }

        // Barrier before next tile overwrites shared memory
        workgroupBarrier();
    }

    if is_active {
        densities[i] = rho;
    }
}
