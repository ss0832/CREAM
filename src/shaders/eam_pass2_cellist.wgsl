// =============================================================================
// eam_pass2_cellist.wgsl  —  Pass 2 (forces + energy + virial) — CellList
//
// common.wgsl is prepended via concat!.  Bindings 0–4 declared there.
//
// After pass0d, binding 0 = reordered_positions, binding 1 = reordered_types
// (both Morton order).  densities_in[k] holds ρ for Morton atom k (from pass1).
// The W component of reordered_positions carries the Morton cell ID from pass0a.
//
// Chunked dispatch (beta.10):
//   params.i_offset — first Morton index for this chunk (multiple of 64).
//   params.i_count  — number of atoms in this chunk.
//   k = params.i_offset + gid.x.  Inactive threads (gid.x ≥ i_count or
//   k ≥ n_atoms) contribute zero to the energy / virial reductions and skip
//   all writes.  wg_energy_out is indexed as (i_offset / 64) + wgid.x so each
//   chunk writes to a disjoint region; wg_virial_out uses the same slot index
//   times 6.  i_offset is always a multiple of 64 (guaranteed by Rust).
//
// Bindings:
//   binding 5  — densities_in:  array<f32>        (read, Morton order)
//   binding 6  — forces_out:    array<vec4<f32>>  (write, original order via scatter)
//   binding 7  — wg_energy_out: array<f32>        (write, ceil(N/64) partial sums)
//   binding 8  — sorted_atoms:  array<u32>        (read, k -> original index j)
//   binding 9  — cell_start:    array<u32>        (read)
//   binding 10 — dbg:           array<atomic<u32>,32> (diagnostic counters)
//   binding 11 — wg_virial_out: array<f32>        (write, 6 × ceil(N/64) partials)
//
// Algorithm:
//   Same WG bounding-box cooperative tile loading as pass1, extended with
//   tile_rho (densities for neighbour atoms).
//   Forces are scatter-written: forces_out[ sorted_atoms[k] ] = force, restoring
//   original atom ordering so that the Verlet integrator and CPU readback are
//   unaffected by the Morton reordering.
//   Energy + virial are reduced via the same 6-barrier workgroup tree:
//   one add/stride for energy plus six adds/stride for the virial (xx,yy,zz,
//   yz,xz,xy).  No new barriers are introduced.
//
// Virial half-pair factor:
//   Each physical pair is visited from both atoms' sides (full-pair walk).
//   Per visit the shader adds `0.5 * dv_α * contrib_β` (Voigt order) — the
//   two visits sum to `(r_j - r_i)_α · F_ij_β`, matching the CPU half-pair
//   reference in cpu_engine.rs.
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
var<workgroup> tile_rho:    array<f32,       64>;
var<workgroup> wg_energy:   array<f32,       64>;

// Virial reduction scratch — one workgroup-shared array per Voigt component.
// 6 × 64 × 4 B = 1.5 KB of shared memory; combined with the existing
// tile_pos + tile_type + tile_rho + wg_energy (~1.8 KB) this leaves
// ample headroom under the 16 KB WebGPU per-WG shared-memory floor.
var<workgroup> wg_vxx: array<f32, 64>;
var<workgroup> wg_vyy: array<f32, 64>;
var<workgroup> wg_vzz: array<f32, 64>;
var<workgroup> wg_vyz: array<f32, 64>;
var<workgroup> wg_vxz: array<f32, 64>;
var<workgroup> wg_vxy: array<f32, 64>;

var<workgroup> wg_bb_min_x: atomic<i32>;
var<workgroup> wg_bb_min_y: atomic<i32>;
var<workgroup> wg_bb_min_z: atomic<i32>;
var<workgroup> wg_bb_max_x: atomic<i32>;
var<workgroup> wg_bb_max_y: atomic<i32>;
var<workgroup> wg_bb_max_z: atomic<i32>;

@group(0) @binding(5) var<storage, read>       densities_in:  array<f32>;
@group(0) @binding(6) var<storage, read_write> forces_out:    array<vec4<f32>>;
@group(0) @binding(7) var<storage, read_write> wg_energy_out: array<f32>;
@group(0) @binding(8) var<storage, read>       sorted_atoms:  array<u32>;
@group(0) @binding(9) var<storage, read>       cell_start:    array<u32>;

// ── Diagnostic counters (atomic<u32> × 32) ─────────────────────────────────────
// Layout documented in engine.rs::CellListBuffers::debug_flags_buf.
// Slots used by this pass:
//   [ 8] pass2_cid_oob_count
//   [ 9] pass2_cs_oob_count
//   [10] pass2_cs_inverted_count
//   [11] pass2_bb_empty_count
//   [12] pass2_atom_k_oob_count
//   [13] pass2_nan_force_count
//   [14] pass2_neighbor_count
//   [15] pass2_cutoff_hit_count
//   [16] pass2_sorted_atom_oob
@group(0) @binding(10) var<storage, read_write> dbg: array<atomic<u32>, 32>;
@group(0) @binding(11) var<storage, read_write> wg_virial_out: array<f32>;

@compute @workgroup_size(64)
fn main(
    @builtin(global_invocation_id)   gid:  vec3<u32>,
    @builtin(local_invocation_index) lid:  u32,
    @builtin(workgroup_id)           wgid: vec3<u32>,
) {
    // k is the Morton index of the atom processed by this thread.
    // In chunked mode i_offset > 0; in full-dispatch mode i_offset = 0.
    let k         = params.i_offset + gid.x;
    let is_active = (gid.x < params.i_count) && (k < params.n_atoms);
    let ncx = i32(params.n_cells_x);
    let ncy = i32(params.n_cells_y);
    let ncz = i32(params.n_cells_z);

    var force:       vec3<f32> = vec3<f32>(0.0);
    var pair_energy: f32       = 0.0;
    var F_i:         f32       = 0.0;

    // ── Home-cell coordinates ─────────────────────────────────────────────────
    // Read the authoritative cell ID from positions[k].w (set by pass0d from
    // pass0a's cell_ids).  This avoids recomputing fractional coordinates.
    var pos_i  = vec3<f32>(0.0);
    var type_i = 0u;
    var dF_i   = 0.0f;
    var hcx    = 0i;
    var hcy    = 0i;
    var hcz    = 0i;

    if is_active {
        let p  = positions[k];       // reordered_positions
        pos_i  = p.xyz;
        type_i = atom_types[k];       // reordered_types
        let rho_i = densities_in[k];  // Morton-order density from pass1
        dF_i  = lookup_by_rho(tbl.dembed_offset, type_i, rho_i);
        F_i   = lookup_by_rho(tbl.embed_offset,  type_i, rho_i);

        let cid = u32(p.w);          // authoritative Morton cell ID from pass0a
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
    // ── Iteration-range selection (matches pass1_cellist v3) ──────────────
    // Power-of-2 grid: use WG-wide BB + periodic wrap (fast path).
    // Non-power-of-2 grid: iterate all real cells (correct fallback).
    // See pass1_cellist for the full rationale and history.
    let is_pow2 = (params.n_cells_x == params.n_cells_x_pad)
               && (params.n_cells_y == params.n_cells_y_pad)
               && (params.n_cells_z == params.n_cells_z_pad);

    let bb_x_min = atomicLoad(&wg_bb_min_x);
    let bb_y_min = atomicLoad(&wg_bb_min_y);
    let bb_z_min = atomicLoad(&wg_bb_min_z);
    let bb_x_max = atomicLoad(&wg_bb_max_x);
    let bb_y_max = atomicLoad(&wg_bb_max_y);
    let bb_z_max = atomicLoad(&wg_bb_max_z);

    let lo_x = select(0i,       bb_x_min - 1i,                                is_pow2);
    let hi_x = select(ncx - 1i, min(bb_x_max + 1i, bb_x_min - 1i + ncx - 1i), is_pow2);
    let lo_y = select(0i,       bb_y_min - 1i,                                is_pow2);
    let hi_y = select(ncy - 1i, min(bb_y_max + 1i, bb_y_min - 1i + ncy - 1i), is_pow2);
    let lo_z = select(0i,       bb_z_min - 1i,                                is_pow2);
    let hi_z = select(ncz - 1i, min(bb_z_max + 1i, bb_z_min - 1i + ncz - 1i), is_pow2);

    if ENABLE_DEBUG && lid == 0u && (lo_x > hi_x || lo_y > hi_y || lo_z > hi_z) {
        atomicAdd(&dbg[11], 1u);
    }
    let n_morton_cap = morton3(params.n_cells_x_pad - 1u,
                               params.n_cells_y_pad - 1u,
                               params.n_cells_z_pad - 1u) + 1u;

    // ── Neumaier compensated force / energy accumulators ─────────────────────────
    var kc_fx: f32 = 0.0;
    var kc_fy: f32 = 0.0;
    var kc_fz: f32 = 0.0;
    var kc_e:  f32 = 0.0;

    // ── Neumaier compensated per-thread virial accumulators (Voigt order) ────────
    var vxx: f32 = 0.0;   var kc_vxx: f32 = 0.0;
    var vyy: f32 = 0.0;   var kc_vyy: f32 = 0.0;
    var vzz: f32 = 0.0;   var kc_vzz: f32 = 0.0;
    var vyz: f32 = 0.0;   var kc_vyz: f32 = 0.0;
    var vxz: f32 = 0.0;   var kc_vxz: f32 = 0.0;
    var vxy: f32 = 0.0;   var kc_vxy: f32 = 0.0;

    // ── Iterate over unique real-grid cells in the stencil ────────────────────
    for (var cx: i32 = lo_x; cx <= hi_x; cx++) {
        for (var cy: i32 = lo_y; cy <= hi_y; cy++) {
            for (var cz: i32 = lo_z; cz <= hi_z; cz++) {
                let nx  = u32(((cx % ncx) + ncx) % ncx);
                let ny  = u32(((cy % ncy) + ncy) % ncy);
                let nz  = u32(((cz % ncz) + ncz) % ncz);
                let cid = morton3(nx, ny, nz);

                if ENABLE_DEBUG && lid == 0u && cid >= n_morton_cap {
                    atomicAdd(&dbg[8], 1u);
                }

                let c_s = cell_start[cid];
                let c_e = cell_start[cid + 1u];

                if ENABLE_DEBUG && lid == 0u && c_e > params.n_atoms {
                    atomicAdd(&dbg[9], 1u);
                }
                if ENABLE_DEBUG && lid == 0u && c_e < c_s {
                    atomicAdd(&dbg[10], 1u);
                }

                let csz = c_e - c_s;

                // DIAG: per-WG cell-visit counters for pass2.
                // [23] real cells visited, [24] padding cells.  Should match
                // pass1's [19]/[20] (same 27-stencil walk).
                if ENABLE_DEBUG && lid == 0u {
                    if csz > 0u {
                        atomicAdd(&dbg[23], 1u);
                    } else {
                        atomicAdd(&dbg[24], 1u);
                    }
                }

                // ── Cooperative tile load: 64-atom chunks ─────────────────────
                let n_tiles = (csz + 63u) / 64u;
                for (var tile: u32 = 0u; tile < n_tiles; tile++) {
                    let base = tile * 64u;
                    let tsz  = min(64u, csz - base);
                    let src  = c_s + base + lid;

                    if lid < tsz {
                        tile_pos[lid]  = positions[src];      // coalesced
                        tile_type[lid] = atom_types[src];     // coalesced
                        tile_rho[lid]  = densities_in[src];   // coalesced Morton
                    }
                    workgroupBarrier();

                    if is_active {
                        for (var tj: u32 = 0u; tj < tsz; tj++) {
                            let atom_k = c_s + base + tj;

                            if ENABLE_DEBUG && atom_k >= params.n_atoms && lid == 0u {
                                atomicAdd(&dbg[12], 1u);
                            }

                            if atom_k == k { continue; }   // self-exclusion

                            let dv   = min_image(tile_pos[tj].xyz - pos_i);
                            let r_sq = dot(dv, dv);

                            if ENABLE_DEBUG && lid == 0u {
                                atomicAdd(&dbg[14], 1u);
                            }
                            // DIAG: all-threads pair visit counter
                            // for pass2 (slot 21).  Compare with pass1's [17]
                            // — they should match (same 27-stencil walk).
                            if ENABLE_DEBUG {
                                atomicAdd(&dbg[21], 1u);
                            }

                            if r_sq < params.cutoff_sq && r_sq > MIN_R_SQ {
                                if ENABLE_DEBUG && lid == 0u {
                                    atomicAdd(&dbg[15], 1u);
                                }
                                if ENABLE_DEBUG {
                                    atomicAdd(&dbg[22], 1u);
                                }
                                let r      = sqrt(r_sq);
                                let r_inv  = 1.0 / r;
                                let r_hat  = dv * r_inv;
                                let t_j    = tile_type[tj];
                                let pidx   = pair_idx(type_i, t_j);

                                let dF_j        = lookup_by_rho(tbl.dembed_offset,  t_j,    tile_rho[tj]);
                                let df_beta_dr  = lookup_by_r(tbl.drho_tab_offset,  t_j,    r);
                                let df_alpha_dr = lookup_by_r(tbl.drho_tab_offset,  type_i, r);
                                let dphi_dr     = lookup_by_r(tbl.dpair_offset,     pidx,   r);
                                let phi         = lookup_by_r(tbl.pair_offset,      pidx,   r);

                                let coeff   = dF_i * df_beta_dr + dF_j * df_alpha_dr + dphi_dr;
                                let contrib = coeff * r_hat;

                                { let _v = contrib.x;   let t2 = force.x + _v;       if abs(force.x)      >= abs(_v) { kc_fx  += (force.x      - t2) + _v; } else { kc_fx  += (_v - t2) + force.x;      } force.x      = t2; }
                                { let _v = contrib.y;   let t2 = force.y + _v;       if abs(force.y)      >= abs(_v) { kc_fy  += (force.y      - t2) + _v; } else { kc_fy  += (_v - t2) + force.y;      } force.y      = t2; }
                                { let _v = contrib.z;   let t2 = force.z + _v;       if abs(force.z)      >= abs(_v) { kc_fz  += (force.z      - t2) + _v; } else { kc_fz  += (_v - t2) + force.z;      } force.z      = t2; }
                                { let _v = 0.5 * phi;   let t2 = pair_energy + _v;   if abs(pair_energy)  >= abs(_v) { kc_e   += (pair_energy  - t2) + _v; } else { kc_e   += (_v - t2) + pair_energy;  } pair_energy  = t2; }

                                // Virial — half-pair factor 0.5 absorbed here.
                                // Voigt order: xx, yy, zz, yz, xz, xy.
                                let wxx_c = -0.5 * dv.x * contrib.x;
                                let wyy_c = -0.5 * dv.y * contrib.y;
                                let wzz_c = -0.5 * dv.z * contrib.z;
                                let wyz_c = -0.5 * dv.y * contrib.z;
                                let wxz_c = -0.5 * dv.x * contrib.z;
                                let wxy_c = -0.5 * dv.x * contrib.y;
                                { let _v = wxx_c; let t2 = vxx + _v; if abs(vxx) >= abs(_v) { kc_vxx += (vxx - t2) + _v; } else { kc_vxx += (_v - t2) + vxx; } vxx = t2; }
                                { let _v = wyy_c; let t2 = vyy + _v; if abs(vyy) >= abs(_v) { kc_vyy += (vyy - t2) + _v; } else { kc_vyy += (_v - t2) + vyy; } vyy = t2; }
                                { let _v = wzz_c; let t2 = vzz + _v; if abs(vzz) >= abs(_v) { kc_vzz += (vzz - t2) + _v; } else { kc_vzz += (_v - t2) + vzz; } vzz = t2; }
                                { let _v = wyz_c; let t2 = vyz + _v; if abs(vyz) >= abs(_v) { kc_vyz += (vyz - t2) + _v; } else { kc_vyz += (_v - t2) + vyz; } vyz = t2; }
                                { let _v = wxz_c; let t2 = vxz + _v; if abs(vxz) >= abs(_v) { kc_vxz += (vxz - t2) + _v; } else { kc_vxz += (_v - t2) + vxz; } vxz = t2; }
                                { let _v = wxy_c; let t2 = vxy + _v; if abs(vxy) >= abs(_v) { kc_vxy += (vxy - t2) + _v; } else { kc_vxy += (_v - t2) + vxy; } vxy = t2; }
                            }
                        }
                    }
                    workgroupBarrier();
                }
            }
        }
    }

    // ── Scatter forces back to original atom order ────────────────────────────
    // sorted_atoms[k] maps Morton index k -> original index j.
    // forces_out[j] is in the same order as pos_buf, consumed by Verlet / CPU.
    if is_active {
        let j = sorted_atoms[k];
        // DIAG: sorted_atoms[k] must be < n_atoms; OOB here = corrupted permutation.
        if ENABLE_DEBUG && j >= params.n_atoms {
            atomicAdd(&dbg[16], 1u);
        }
        // DIAG: NaN/Inf forces from bad density or bad positions.
        let is_bad = !(force.x == force.x) || !(force.y == force.y) || !(force.z == force.z)
                     || abs(force.x) > 1e20 || abs(force.y) > 1e20 || abs(force.z) > 1e20;
        if ENABLE_DEBUG && is_bad {
            atomicAdd(&dbg[13], 1u);
        }
        forces_out[j] = vec4<f32>(force.x + kc_fx, force.y + kc_fy, force.z + kc_fz, 0.0);
    }

    // ── Workgroup energy + virial tree-reduction (same 6 barriers) ────────────
    wg_energy[lid] = select(0.0, F_i + pair_energy + kc_e,   is_active);
    wg_vxx[lid]    = select(0.0, vxx + kc_vxx, is_active);
    wg_vyy[lid]    = select(0.0, vyy + kc_vyy, is_active);
    wg_vzz[lid]    = select(0.0, vzz + kc_vzz, is_active);
    wg_vyz[lid]    = select(0.0, vyz + kc_vyz, is_active);
    wg_vxz[lid]    = select(0.0, vxz + kc_vxz, is_active);
    wg_vxy[lid]    = select(0.0, vxy + kc_vxy, is_active);
    workgroupBarrier();

    for (var stride = 32u; stride > 0u; stride >>= 1u) {
        if lid < stride {
            wg_energy[lid] += wg_energy[lid + stride];
            wg_vxx[lid]    += wg_vxx[lid + stride];
            wg_vyy[lid]    += wg_vyy[lid + stride];
            wg_vzz[lid]    += wg_vzz[lid + stride];
            wg_vyz[lid]    += wg_vyz[lid + stride];
            wg_vxz[lid]    += wg_vxz[lid + stride];
            wg_vxy[lid]    += wg_vxy[lid + stride];
        }
        workgroupBarrier();
    }

    if lid == 0u {
        let slot = (params.i_offset / 64u) + wgid.x;
        wg_energy_out[slot] = wg_energy[0];
        let vbase = 6u * slot;
        wg_virial_out[vbase + 0u] = wg_vxx[0];
        wg_virial_out[vbase + 1u] = wg_vyy[0];
        wg_virial_out[vbase + 2u] = wg_vzz[0];
        wg_virial_out[vbase + 3u] = wg_vyz[0];
        wg_virial_out[vbase + 4u] = wg_vxz[0];
        wg_virial_out[vbase + 5u] = wg_vxy[0];
    }
}
