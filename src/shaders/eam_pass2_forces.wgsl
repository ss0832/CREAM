// =============================================================================
// eam_pass2_forces.wgsl  —  Pass 2: forces + workgroup energy & virial reduction
//
// common.wgsl is prepended via concat!.
// Requires densities[] from Pass 1 (submit barrier between passes).
//
// Bindings:
//   binding  7 = wg_energy_out  (ceil(N/64) f32 — per-WG energy partial)
//   binding 11 = wg_virial_out  (6×ceil(N/64) f32 — per-WG virial partial,
//                                Voigt order [xx, yy, zz, yz, xz, xy])
//
// Flash Attention–style tiling (Q6):  j-loop tiled at 64 atoms.
// Kahan compensated summation on:
//   - force components (3)
//   - pair energy (1)
//   - virial components (6)
//
// Virial contribution per pair visit:
//   Each physical pair (i, j) is visited twice (once from each atom's side).
//   From i's side  : dv = r_j - r_i,  contrib = F_ij  (force added to atom i).
//   From j's side  : dv' = -dv,       contrib' = -contrib  (Newton 3rd).
//   Either visit contributes `0.5 · dv_α · contrib_β` → the two visits sum to
//   `(r_j - r_i)_α · F_ij_β`, matching the CPU half-pair reference in
//   cpu_engine.rs which uses `dx · fx` once per (i<j) pair.
// =============================================================================

var<workgroup> tile_pos   : array<vec4<f32>, 64>;
var<workgroup> tile_type  : array<u32,       64>;
var<workgroup> tile_rho   : array<f32,       64>;
var<workgroup> wg_energy  : array<f32,       64>;  // workgroup reduction scratch

// Virial reduction scratch — one shared array per Voigt component.
var<workgroup> wg_vxx : array<f32, 64>;
var<workgroup> wg_vyy : array<f32, 64>;
var<workgroup> wg_vzz : array<f32, 64>;
var<workgroup> wg_vyz : array<f32, 64>;
var<workgroup> wg_vxz : array<f32, 64>;
var<workgroup> wg_vxy : array<f32, 64>;

@group(0) @binding(5)  var<storage, read>       densities_in  : array<f32>;
@group(0) @binding(6)  var<storage, read_write> forces_out    : array<vec4<f32>>;
@group(0) @binding(7)  var<storage, read_write> wg_energy_out : array<f32>;   // ceil(N/64) elements
@group(0) @binding(11) var<storage, read_write> wg_virial_out : array<f32>;   // 6 × ceil(N/64) elements

@compute @workgroup_size(64)
fn main(
    @builtin(global_invocation_id)   gid:  vec3<u32>,
    @builtin(local_invocation_index) lid:  u32,
    @builtin(workgroup_id)           wgid: vec3<u32>,
) {
    // In chunked mode i_offset > 0; in full-dispatch mode i_offset = 0.
    let i = params.i_offset + gid.x;
    let is_active = (gid.x < params.i_count) && (i < params.n_atoms);

    let si     = select(0u, i, is_active);
    let pos_i  = positions[si].xyz;
    let type_i = atom_types[si];
    let rho_i  = densities_in[si];

    let dF_i = lookup_by_rho(tbl.dembed_offset, type_i, rho_i);
    let F_i  = lookup_by_rho(tbl.embed_offset,  type_i, rho_i);

    var force:       vec3<f32> = vec3<f32>(0.0, 0.0, 0.0);
    var pair_energy: f32       = 0.0;

    // Kahan compensators — forces & energy.
    var kc_fx: f32 = 0.0;
    var kc_fy: f32 = 0.0;
    var kc_fz: f32 = 0.0;
    var kc_e:  f32 = 0.0;

    // Per-thread virial accumulators (Voigt order: xx, yy, zz, yz, xz, xy).
    var vxx: f32 = 0.0;   var kc_vxx: f32 = 0.0;
    var vyy: f32 = 0.0;   var kc_vyy: f32 = 0.0;
    var vzz: f32 = 0.0;   var kc_vzz: f32 = 0.0;
    var vyz: f32 = 0.0;   var kc_vyz: f32 = 0.0;
    var vxz: f32 = 0.0;   var kc_vxz: f32 = 0.0;
    var vxy: f32 = 0.0;   var kc_vxy: f32 = 0.0;

    let n_tiles = (params.n_atoms + 63u) / 64u;

    for (var t = 0u; t < n_tiles; t++) {
        let tile_base = t * 64u;

        let j_load = tile_base + lid;
        let j_safe = min(j_load, params.n_atoms - 1u);
        tile_pos[lid]  = positions[j_safe];
        tile_type[lid] = atom_types[j_safe];
        tile_rho[lid]  = densities_in[j_safe];
        workgroupBarrier();

        if is_active {
            let tile_size = min(64u, params.n_atoms - tile_base);
            for (var tj = 0u; tj < tile_size; tj++) {
                let j = tile_base + tj;
                if j == i { continue; }

                let dv   = min_image(tile_pos[tj].xyz - pos_i);
                let r_sq = dot(dv, dv);

                if r_sq < params.cutoff_sq && r_sq > MIN_R_SQ {
                    let r     = sqrt(r_sq);
                    let r_inv = 1.0 / r;
                    let r_hat = dv * r_inv;
                    let tj_type = tile_type[tj];
                    let pidx    = pair_idx(type_i, tj_type);

                    let rho_j       = tile_rho[tj];
                    let dF_j        = lookup_by_rho(tbl.dembed_offset,   tj_type, rho_j);
                    let df_beta_dr  = lookup_by_r(tbl.drho_tab_offset, tj_type, r);
                    let df_alpha_dr = lookup_by_r(tbl.drho_tab_offset, type_i,  r);
                    let dphi_dr     = lookup_by_r(tbl.dpair_offset,    pidx,    r);
                    let phi         = lookup_by_r(tbl.pair_offset,     pidx,    r);

                    let coeff   = dF_i * df_beta_dr + dF_j * df_alpha_dr + dphi_dr;
                    let contrib = coeff * r_hat;

                    // Kahan-compensated force / energy.
                    { let y = contrib.x - kc_fx; let t2 = force.x + y; kc_fx = (t2 - force.x) - y; force.x = t2; }
                    { let y = contrib.y - kc_fy; let t2 = force.y + y; kc_fy = (t2 - force.y) - y; force.y = t2; }
                    { let y = contrib.z - kc_fz; let t2 = force.z + y; kc_fz = (t2 - force.z) - y; force.z = t2; }
                    { let y = (0.5 * phi) - kc_e; let t2 = pair_energy + y; kc_e = (t2 - pair_energy) - y; pair_energy = t2; }

                    // Virial: half-pair factor 0.5 absorbed once here.
                    // Voigt order matches cpu_engine.rs: xx, yy, zz, yz, xz, xy.
                    let wxx = -0.5 * dv.x * contrib.x;
                    let wyy = -0.5 * dv.y * contrib.y;
                    let wzz = -0.5 * dv.z * contrib.z;
                    let wyz = -0.5 * dv.y * contrib.z;
                    let wxz = -0.5 * dv.x * contrib.z;
                    let wxy = -0.5 * dv.x * contrib.y;
                    { let y = wxx - kc_vxx; let t2 = vxx + y; kc_vxx = (t2 - vxx) - y; vxx = t2; }
                    { let y = wyy - kc_vyy; let t2 = vyy + y; kc_vyy = (t2 - vyy) - y; vyy = t2; }
                    { let y = wzz - kc_vzz; let t2 = vzz + y; kc_vzz = (t2 - vzz) - y; vzz = t2; }
                    { let y = wyz - kc_vyz; let t2 = vyz + y; kc_vyz = (t2 - vyz) - y; vyz = t2; }
                    { let y = wxz - kc_vxz; let t2 = vxz + y; kc_vxz = (t2 - vxz) - y; vxz = t2; }
                    { let y = wxy - kc_vxy; let t2 = vxy + y; kc_vxy = (t2 - vxy) - y; vxy = t2; }
                }
            }
        }

        workgroupBarrier();
    }

    if is_active {
        forces_out[i] = vec4<f32>(force, 0.0);
    }

    // ── Workgroup energy + virial tree-reduction (shared 6-barrier schedule) ──
    // Inactive threads contribute 0 to all reductions.
    wg_energy[lid] = select(0.0, F_i + pair_energy, is_active);
    wg_vxx[lid]    = select(0.0, vxx, is_active);
    wg_vyy[lid]    = select(0.0, vyy, is_active);
    wg_vzz[lid]    = select(0.0, vzz, is_active);
    wg_vyz[lid]    = select(0.0, vyz, is_active);
    wg_vxz[lid]    = select(0.0, vxz, is_active);
    wg_vxy[lid]    = select(0.0, vxy, is_active);
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

    // Thread 0 of each workgroup writes the partial sums.
    //   energy: one f32 at slot (i_offset/64) + wgid.x
    //   virial: six f32 at slot 6 * ((i_offset/64) + wgid.x) + c
    // In chunked mode i_offset is always a multiple of 64 (Rust guarantees this).
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
