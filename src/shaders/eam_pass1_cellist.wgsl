// =============================================================================
// eam_pass1_cellist.wgsl  —  Pass 1 (density) — reordered CellList
//
// common.wgsl is prepended via concat!.  Bindings 0–4 declared there.
//
// After pass0d, binding 0 holds reordered_positions and binding 1 holds
// reordered_types (both in Morton order).  Thread k = i_offset + gid.x
// processes the k-th Morton atom.  densities[k] is written in Morton order
// and consumed directly by pass2_cellist.
//
// The W component of reordered_positions[k] carries the authoritative Morton
// cell ID assigned by pass0a and stored by pass0d.  This is decoded via
// morton3_x/y/z to obtain home-cell coordinates, avoiding any float-rounding
// inconsistency from recomputing fractional coordinates in a different shader.
//
// Chunked dispatch (beta.10):
//   params.i_offset — first Morton index for this chunk (multiple of 64).
//   params.i_count  — number of atoms in this chunk (≤ 65535 × 64).
//   For a full (non-chunked) dispatch: i_offset = 0, i_count = n_atoms.
//   The Rust engine loops over ceil(N / MAX_DISPATCH_ATOMS) chunks, patching
//   params_buf via copy_buffer_to_buffer before each dispatch.
//
// Bindings:
//   binding 5 — densities:  array<f32>   (write, Morton order)
//   binding 6 — cell_start: array<u32>   (read, prefix-sum over Morton cells)
//
// Algorithm — workgroup (WG) bounding-box cooperative tile loading:
//   1. Each active thread reads its home cell from positions[k].w (cell ID).
//   2. Workgroup atomics reduce to the WG-wide bounding box (bb_min / bb_max).
//   3. All 64 threads iterate the padded bounding box (bb ± 1 cell).
//      Every thread sees the same cells → barriers are executed uniformly.
//   4. For each cell, threads cooperatively load atoms into var<workgroup>
//      tile_pos / tile_type (64-atom chunks).  Reads are sequential (coalesced).
//   5. Each active thread accumulates ρᵢ against the tile atoms within cutoff.
//
// Self-exclusion: atom_k == k (both are Morton indices after reordering).
// Kahan compensated summation suppresses f32 round-off.
// =============================================================================

// ── Debug mode toggle ──────────────────────────────────────────────────────────
// Set to true at pipeline creation time (PipelineDescriptor::constants) to
// enable the diagnostic atomic counters in binding 10.  When false the entire
// debug path is dead-code-eliminated by the GPU compiler and has zero runtime
// cost.  Example (wgpu): pipeline_desc.constants = [("ENABLE_DEBUG", 1.0)];
override ENABLE_DEBUG: bool = false;

// ── Workgroup shared memory ────────────────────────────────────────────────────
var<workgroup> tile_pos:    array<vec4<f32>, 64>;
var<workgroup> tile_type:   array<u32,       64>;

// WG bounding-box accumulators (one thread initialises, all update).
var<workgroup> wg_bb_min_x: atomic<i32>;
var<workgroup> wg_bb_min_y: atomic<i32>;
var<workgroup> wg_bb_min_z: atomic<i32>;
var<workgroup> wg_bb_max_x: atomic<i32>;
var<workgroup> wg_bb_max_y: atomic<i32>;
var<workgroup> wg_bb_max_z: atomic<i32>;

@group(0) @binding(5) var<storage, read_write> densities:  array<f32>;
@group(0) @binding(6) var<storage, read>       cell_start: array<u32>;

// ── Diagnostic counters (atomic<u32> × 32) ─────────────────────────────────────
// Layout documented in engine.rs::CellListBuffers::debug_flags_buf.
// Slots used by this pass:
//   [0] pass1_cid_oob_count
//   [1] pass1_cs_oob_count
//   [2] pass1_cs_inverted_count
//   [3] pass1_bb_empty_count       (incremented ONCE per WG that sees bb_x0>bb_x1)
//   [4] pass1_atom_k_oob_count
//   [5] pass1_nan_rho_count
//   [6] pass1_neighbor_count       (lid==0 only — per-thread-0 pair visits)
//   [7] pass1_cutoff_hit_count     (lid==0 only — per-thread-0 cutoff hits)
//  [17] pass1_all_pair_visits      (ALL active threads — total pair work)
//  [18] pass1_all_cutoff_hits      (ALL active threads — total cutoff-passing pairs)
//  [19] pass1_real_cell_visits     (one per WG per iterated cell with csz>0 — tells us
//                                   how many non-empty cells the 27-stencil walked)
//  [20] pass1_padding_cell_visits  (one per WG per iterated cell with csz==0 — tells us
//                                   how often the stencil hit padding slots, only >0
//                                   when n_cells != n_pad)
@group(0) @binding(10) var<storage, read_write> dbg: array<atomic<u32>, 32>;

@compute @workgroup_size(64)
fn main(
    @builtin(global_invocation_id)   gid: vec3<u32>,
    @builtin(local_invocation_index) lid: u32,
) {
    // k is the Morton index of the atom processed by this thread.
    // In chunked mode i_offset > 0; in full-dispatch mode i_offset = 0.
    let k         = params.i_offset + gid.x;
    let is_active = (gid.x < params.i_count) && (k < params.n_atoms);
    let ncx = i32(params.n_cells_x);
    let ncy = i32(params.n_cells_y);
    let ncz = i32(params.n_cells_z);

    // ── Home-cell coordinates for this thread ─────────────────────────────────
    // Read the authoritative cell ID from positions[k].w (set by pass0d from
    // pass0a's cell_ids).  This avoids recomputing fractional coordinates,
    // which can round differently across shader compilations (triclinic bug).
    var pos_i = vec3<f32>(0.0);
    var hcx   = 0i;
    var hcy   = 0i;
    var hcz   = 0i;

    if is_active {
        let p = positions[k];       // positions == reordered_positions (binding 0)
        pos_i = p.xyz;
        let cid = u32(p.w);        // authoritative Morton cell ID from pass0a
        hcx = i32(morton3_x(cid));
        hcy = i32(morton3_y(cid));
        hcz = i32(morton3_z(cid));
    }

    // ── WG bounding-box reduction ─────────────────────────────────────────────
    // Neutral initialisation: min <- INT_MAX, max <- INT_MIN+1.
    // All threads write the same value so the atomic races are benign, but every
    // thread's store is visible to itself without relying on workgroupBarrier to
    // flush a single writer — avoiding a DX12/NVIDIA RTX barrier-flushing bug
    // where threads 1..63 could read the pre-barrier zero instead of INT_MAX.
    // If the entire WG is inactive the outer loop bound is empty (bb_x0 > bb_x1).
    atomicStore(&wg_bb_min_x,  2147483647i);
    atomicStore(&wg_bb_min_y,  2147483647i);
    atomicStore(&wg_bb_min_z,  2147483647i);
    atomicStore(&wg_bb_max_x, -2147483647i);
    atomicStore(&wg_bb_max_y, -2147483647i);
    atomicStore(&wg_bb_max_z, -2147483647i);
    workgroupBarrier();

    if is_active {
        atomicMin(&wg_bb_min_x, hcx);
        atomicMin(&wg_bb_min_y, hcy);
        atomicMin(&wg_bb_min_z, hcz);
        atomicMax(&wg_bb_max_x, hcx);
        atomicMax(&wg_bb_max_y, hcy);
        atomicMax(&wg_bb_max_z, hcz);
    }
    workgroupBarrier();

    // Padded bounding box: extend each edge by one cell to cover all 27-neighbour
    // cells of every atom in the WG.  All threads use the SAME bounds.
    // ±1 is sufficient because home-cell IDs now come directly from pass0a
    // (no float recomputation).
    //
    // ── Iteration-range selection (two-branch) ────────────────────────────────
    // Branch A: POWER-OF-TWO grid (ncx == nx_pad AND ncy == ny_pad AND ncz == nz_pad)
    //    Use the classic WG-wide bounding-box + periodic wrap stencil.  This
    //    is the original performant path: iterate `[bb_min-1, bb_max+1]` with
    //    `% ncell` wrap.  On power-of-two grids the Morton encoding and the
    //    grid wrap commute cleanly, so two cx values that would alias under
    //    `% ncell` also alias under `morton3_x`, and the only required cap is
    //    `bb_x0 + ncx - 1` (= up to ncx distinct iterations).
    //
    // Branch B: NON-POWER-OF-TWO grid (any of {ncx, ncy, ncz} < {nx_pad, ny_pad, nz_pad})
    //    Iterate ALL real cells `0..ncell-1` on each axis.  Guaranteed correct
    //    at the cost of `ncx*ncy*ncz / 27` extra cells per WG.  Use this only
    //    when a user-specified `cell_size` does not divide the box dimensions
    //    into a power-of-two count.  For production, choose `cell_size` so
    //    that all three axes give power-of-two counts (the common case).
    //
    // HISTORY: Step-3 v1 tried a unified `cover` logic that triggered full-
    // axis iteration when `bb_max-bb_min+2 >= ncx`, but a residual off-by-one
    // left ~1-pair/atom error on WG 0 of non-p2 grids.  v2 dropped the WG BB
    // entirely (always full-grid) which was correct but slow.  v3 (this)
    // restores the fast path for power-of-2 grids and keeps v2 as fallback.
    let is_pow2 = (params.n_cells_x == params.n_cells_x_pad)
               && (params.n_cells_y == params.n_cells_y_pad)
               && (params.n_cells_z == params.n_cells_z_pad);

    let bb_x_min = atomicLoad(&wg_bb_min_x);
    let bb_y_min = atomicLoad(&wg_bb_min_y);
    let bb_z_min = atomicLoad(&wg_bb_min_z);
    let bb_x_max = atomicLoad(&wg_bb_max_x);
    let bb_y_max = atomicLoad(&wg_bb_max_y);
    let bb_z_max = atomicLoad(&wg_bb_max_z);

    // For the power-of-2 branch, apply the original cap:
    //   range = [bb_min-1, min(bb_max+1, bb_min-1+ncx-1)]
    // For the non-p2 branch, iterate the full real grid.
    let lo_x = select(0i,            bb_x_min - 1i,                                is_pow2);
    let hi_x = select(ncx - 1i,      min(bb_x_max + 1i, bb_x_min - 1i + ncx - 1i), is_pow2);
    let lo_y = select(0i,            bb_y_min - 1i,                                is_pow2);
    let hi_y = select(ncy - 1i,      min(bb_y_max + 1i, bb_y_min - 1i + ncy - 1i), is_pow2);
    let lo_z = select(0i,            bb_z_min - 1i,                                is_pow2);
    let hi_z = select(ncz - 1i,      min(bb_z_max + 1i, bb_z_min - 1i + ncz - 1i), is_pow2);

    // DIAG: flag WGs whose bb is empty (all-inactive WGs hit this legitimately;
    // a non-empty chunk hitting this would indicate the atomic reduction failed).
    if ENABLE_DEBUG && lid == 0u && (lo_x > hi_x || lo_y > hi_y || lo_z > hi_z) {
        atomicAdd(&dbg[3], 1u);
    }

    // DIAG: Morton capacity upper bound (exclusive) for this cell list.
    // cell_ids_buf is n_morton entries long; cell_start_buf is n_morton+1.
    // Computed from padded grid: morton3(nx_pad-1, ny_pad-1, nz_pad-1)+1.
    let n_morton_cap = morton3(params.n_cells_x_pad - 1u,
                               params.n_cells_y_pad - 1u,
                               params.n_cells_z_pad - 1u) + 1u;

    // ── Kahan compensated accumulator for rho_i ───────────────────────────────
    var rho: f32 = 0.0;
    var kc:  f32 = 0.0;

    // ── Iterate over unique real-grid cells in the stencil ────────────────────
    // When cover_{xyz} is true the loop bound is already in [0, ncell-1] so
    // the wrap is a no-op; when false the loop bound is a contiguous window
    // of width < ncell so each `% ncell` result is distinct.  In either case
    // every real cell is visited AT MOST ONCE per WG.  Uniform control flow:
    // all threads see the same (cx, cy, cz) sequence.
    for (var cx: i32 = lo_x; cx <= hi_x; cx++) {
        for (var cy: i32 = lo_y; cy <= hi_y; cy++) {
            for (var cz: i32 = lo_z; cz <= hi_z; cz++) {
                // Periodic wrap (no-op in "cover" case).
                let nx  = u32(((cx % ncx) + ncx) % ncx);
                let ny  = u32(((cy % ncy) + ncy) % ncy);
                let nz  = u32(((cz % ncz) + ncz) % ncz);
                let cid = morton3(nx, ny, nz);

                // DIAG: cid should always be < n_morton_cap (otherwise we'd
                // OOB-read cell_start/cell_counts).  Only lid==0 writes to
                // keep the atomic cheap — one increment per WG per occurrence.
                if ENABLE_DEBUG && lid == 0u && cid >= n_morton_cap {
                    atomicAdd(&dbg[0], 1u);
                }

                let c_s = cell_start[cid];
                let c_e = cell_start[cid + 1u];

                // DIAG: cell_start must be monotone and bounded.
                if ENABLE_DEBUG && lid == 0u && c_e > params.n_atoms {
                    atomicAdd(&dbg[1], 1u);
                }
                if ENABLE_DEBUG && lid == 0u && c_e < c_s {
                    atomicAdd(&dbg[2], 1u);
                }

                let csz = c_e - c_s;

                // DIAG: per-WG cell-visit counters.
                // lid==0 increments once per iterated cell — sums across all
                // active WGs.  Split by whether the cell has atoms (real) or
                // is a Morton padding slot (csz==0, only possible when
                // n_cells != n_pad).  Comparing these across N will reveal
                // whether the 27-stencil walks too many / too few cells.
                if ENABLE_DEBUG && lid == 0u {
                    if csz > 0u {
                        atomicAdd(&dbg[19], 1u);
                    } else {
                        atomicAdd(&dbg[20], 1u);
                    }
                }

                // ── Cooperative tile load: 64-atom chunks ─────────────────────
                // n_tiles is uniform across the WG (all threads read same csz).
                let n_tiles = (csz + 63u) / 64u;
                for (var tile: u32 = 0u; tile < n_tiles; tile++) {
                    let base = tile * 64u;
                    let tsz  = min(64u, csz - base);
                    let src  = c_s + base + lid;

                    if lid < tsz {
                        tile_pos[lid]  = positions[src];   // coalesced sequential read
                        tile_type[lid] = atom_types[src];  // coalesced sequential read
                    }
                    workgroupBarrier();

                    if is_active {
                        for (var tj: u32 = 0u; tj < tsz; tj++) {
                            let atom_k = c_s + base + tj;

                            // DIAG: atom_k must fit in reordered arrays.
                            if ENABLE_DEBUG && atom_k >= params.n_atoms && lid == 0u {
                                atomicAdd(&dbg[4], 1u);
                            }

                            if atom_k == k { continue; }   // self-exclusion (Morton idx)
                            let dv   = min_image(tile_pos[tj].xyz - pos_i);
                            let r_sq = dot(dv, dv);
                            // DIAG: pair-visit throughput counter (one per tj per active thread).
                            // Cheap — one atomicAdd per pair but collapsed to WG-wide totals.
                            if ENABLE_DEBUG && lid == 0u {
                                atomicAdd(&dbg[6], 1u);
                            }
                            // DIAG: all-threads pair visit counter.
                            // Because `is_active` gates the entire loop, only
                            // active threads increment here — so this counts
                            // the total number of (atom, candidate-pair)
                            // visits, which must equal
                            //   sum over active atoms of {pairs scanned in 27-stencil} .
                            if ENABLE_DEBUG {
                                atomicAdd(&dbg[17], 1u);
                            }
                            if r_sq < params.cutoff_sq && r_sq > MIN_R_SQ {
                                if ENABLE_DEBUG && lid == 0u {
                                    atomicAdd(&dbg[7], 1u);
                                }
                                if ENABLE_DEBUG {
                                    atomicAdd(&dbg[18], 1u);
                                }
                                let r = sqrt(r_sq);
                                let y = lookup_by_r(tbl.rho_offset, tile_type[tj], r) - kc;
                                let t = rho + y;
                                kc    = (t - rho) - y;
                                rho   = t;
                            }
                        }
                    }
                    workgroupBarrier();
                }
            }
        }
    }

    if is_active {
        // DIAG: catch NaN/Inf rho before it poisons pass2.
        if ENABLE_DEBUG && (!(rho == rho) || rho > 1e30 || rho < -1e30) {
            atomicAdd(&dbg[5], 1u);
        }
        densities[k] = rho;   // Morton order: consumed by pass2_cellist as densities_in[k]
    }
}
