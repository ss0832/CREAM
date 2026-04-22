// =============================================================================
// eam_pass2_neighlist.wgsl  —  Pass 2 (forces + energy + virial) — CSR NL
//
// Replaces the GPU cell-list pipeline for single-point EAM calculations.
// Same CSR neighbour-list format as pass1_neighlist.
//
// common.wgsl is prepended via concat!.  Bindings 0–4 declared there.
//
// Bindings:
//   binding 5  — densities_in  (read, from pass1)
//   binding 6  — forces_out    (write, original atom order — NO scatter)
//   binding 7  — wg_energy_out (write, one f32 per WG)
//   binding 8  — nl_starts     (read, length N+1)
//   binding 9  — nl_list       (read, flat list)
//   binding 10 — dbg           (atomic<u32>*32)
//   binding 11 — wg_virial_out (write, 6 f32 per WG — Voigt partial virial)
//
// Atom ordering: forces_out[i] is written directly (no sorted_atoms scatter),
// because there is no Morton reordering in the neighbour-list pipeline.
//
// Energy + virial reduction:
//   The canonical 6-barrier tree-reduction over a 64-entry workgroup-shared
//   array is retained and extended: the same 6 barriers now reduce 1 energy
//   + 6 virial components in lock-step, no new barriers introduced.
//   wg_virial_out is indexed as `6 * ((i_offset/64) + wgid.x) + c` so each
//   chunk writes to a disjoint region.
//
// Virial half-pair factor:
//   CSR contains both (i,j) and (j,i) entries, so each physical pair is
//   visited twice.  Per visit we accumulate `0.5 * dv_α * contrib_β` —
//   the sum over the two visits equals `(r_j - r_i)_α · F_ij_β`,
//   matching the CPU half-pair reference exactly.
//
// Diagnostics:
//   [13] nan_force_count
//   [14] pair_visit_count
//   [15] cutoff_hit_count
// =============================================================================

override ENABLE_DEBUG: bool = false;

var<workgroup> wg_energy: array<f32, 64>;
var<workgroup> wg_vxx   : array<f32, 64>;
var<workgroup> wg_vyy   : array<f32, 64>;
var<workgroup> wg_vzz   : array<f32, 64>;
var<workgroup> wg_vyz   : array<f32, 64>;
var<workgroup> wg_vxz   : array<f32, 64>;
var<workgroup> wg_vxy   : array<f32, 64>;

@group(0) @binding(5)  var<storage, read>       densities_in:  array<f32>;
@group(0) @binding(6)  var<storage, read_write> forces_out:    array<vec4<f32>>;
@group(0) @binding(7)  var<storage, read_write> wg_energy_out: array<f32>;
@group(0) @binding(8)  var<storage, read>       nl_starts:     array<u32>;
@group(0) @binding(9)  var<storage, read>       nl_list:       array<u32>;
@group(0) @binding(10) var<storage, read_write> dbg:           array<atomic<u32>, 32>;
@group(0) @binding(11) var<storage, read_write> wg_virial_out: array<f32>;

@compute @workgroup_size(64)
fn main(
    @builtin(global_invocation_id)   gid:  vec3<u32>,
    @builtin(local_invocation_index) lid:  u32,
    @builtin(workgroup_id)           wgid: vec3<u32>,
) {
    let i         = params.i_offset + gid.x;
    let is_active = (gid.x < params.i_count) && (i < params.n_atoms);

    var force:       vec3<f32> = vec3<f32>(0.0);
    var pair_energy: f32       = 0.0;
    var F_i:         f32       = 0.0;

    // Per-thread virial accumulators — Voigt order (xx, yy, zz, yz, xz, xy).
    var vxx: f32 = 0.0;  var kc_vxx: f32 = 0.0;
    var vyy: f32 = 0.0;  var kc_vyy: f32 = 0.0;
    var vzz: f32 = 0.0;  var kc_vzz: f32 = 0.0;
    var vyz: f32 = 0.0;  var kc_vyz: f32 = 0.0;
    var vxz: f32 = 0.0;  var kc_vxz: f32 = 0.0;
    var vxy: f32 = 0.0;  var kc_vxy: f32 = 0.0;

    if is_active {
        let pos_i  = positions[i].xyz;
        let type_i = atom_types[i];
        let rho_i  = densities_in[i];
        let dF_i   = lookup_by_rho(tbl.dembed_offset, type_i, rho_i);
        F_i        = lookup_by_rho(tbl.embed_offset,  type_i, rho_i);

        let s = nl_starts[i];
        let e = nl_starts[i + 1u];

        var kc_fx: f32 = 0.0;
        var kc_fy: f32 = 0.0;
        var kc_fz: f32 = 0.0;
        var kc_e:  f32 = 0.0;

        for (var idx: u32 = s; idx < e; idx = idx + 1u) {
            let j    = nl_list[idx];
            let p_j  = positions[j].xyz;
            let dv   = min_image(p_j - pos_i);
            let r_sq = dot(dv, dv);

            if ENABLE_DEBUG { atomicAdd(&dbg[14], 1u); }

            if r_sq < params.cutoff_sq && r_sq > MIN_R_SQ {
                if ENABLE_DEBUG { atomicAdd(&dbg[15], 1u); }
                let r     = sqrt(r_sq);
                let r_inv = 1.0 / r;
                let r_hat = dv * r_inv;
                let t_j   = atom_types[j];
                let pidx  = pair_idx(type_i, t_j);

                let dF_j        = lookup_by_rho(tbl.dembed_offset, t_j,    densities_in[j]);
                let df_beta_dr  = lookup_by_r(tbl.drho_tab_offset, t_j,    r);
                let df_alpha_dr = lookup_by_r(tbl.drho_tab_offset, type_i, r);
                let dphi_dr     = lookup_by_r(tbl.dpair_offset,    pidx,   r);
                let phi         = lookup_by_r(tbl.pair_offset,     pidx,   r);

                let coeff   = dF_i * df_beta_dr + dF_j * df_alpha_dr + dphi_dr;
                let contrib = coeff * r_hat;

                { let _v = contrib.x;   let t2 = force.x + _v;       if abs(force.x)      >= abs(_v) { kc_fx  += (force.x      - t2) + _v; } else { kc_fx  += (_v - t2) + force.x;      } force.x      = t2; }
                { let _v = contrib.y;   let t2 = force.y + _v;       if abs(force.y)      >= abs(_v) { kc_fy  += (force.y      - t2) + _v; } else { kc_fy  += (_v - t2) + force.y;      } force.y      = t2; }
                { let _v = contrib.z;   let t2 = force.z + _v;       if abs(force.z)      >= abs(_v) { kc_fz  += (force.z      - t2) + _v; } else { kc_fz  += (_v - t2) + force.z;      } force.z      = t2; }
                { let _v = 0.5 * phi;   let t2 = pair_energy + _v;   if abs(pair_energy)  >= abs(_v) { kc_e   += (pair_energy  - t2) + _v; } else { kc_e   += (_v - t2) + pair_energy;  } pair_energy  = t2; }

                // Virial — half-pair factor 0.5 absorbed here.
                let wxx = -0.5 * dv.x * contrib.x;
                let wyy = -0.5 * dv.y * contrib.y;
                let wzz = -0.5 * dv.z * contrib.z;
                let wyz = -0.5 * dv.y * contrib.z;
                let wxz = -0.5 * dv.x * contrib.z;
                let wxy = -0.5 * dv.x * contrib.y;
                { let _v = wxx; let t2 = vxx + _v; if abs(vxx) >= abs(_v) { kc_vxx += (vxx - t2) + _v; } else { kc_vxx += (_v - t2) + vxx; } vxx = t2; }
                { let _v = wyy; let t2 = vyy + _v; if abs(vyy) >= abs(_v) { kc_vyy += (vyy - t2) + _v; } else { kc_vyy += (_v - t2) + vyy; } vyy = t2; }
                { let _v = wzz; let t2 = vzz + _v; if abs(vzz) >= abs(_v) { kc_vzz += (vzz - t2) + _v; } else { kc_vzz += (_v - t2) + vzz; } vzz = t2; }
                { let _v = wyz; let t2 = vyz + _v; if abs(vyz) >= abs(_v) { kc_vyz += (vyz - t2) + _v; } else { kc_vyz += (_v - t2) + vyz; } vyz = t2; }
                { let _v = wxz; let t2 = vxz + _v; if abs(vxz) >= abs(_v) { kc_vxz += (vxz - t2) + _v; } else { kc_vxz += (_v - t2) + vxz; } vxz = t2; }
                { let _v = wxy; let t2 = vxy + _v; if abs(vxy) >= abs(_v) { kc_vxy += (vxy - t2) + _v; } else { kc_vxy += (_v - t2) + vxy; } vxy = t2; }
            }
        }

        // Direct write — no sorted_atoms scatter because atom order is preserved.
        let is_bad = !(force.x == force.x) || !(force.y == force.y) || !(force.z == force.z)
                     || abs(force.x) > 1e20 || abs(force.y) > 1e20 || abs(force.z) > 1e20;
        if ENABLE_DEBUG && is_bad { atomicAdd(&dbg[13], 1u); }

        // Apply Neumaier compensation before final output.
        pair_energy += kc_e;
        forces_out[i] = vec4<f32>(force.x + kc_fx, force.y + kc_fy, force.z + kc_fz, 0.0);
    }

    // ── Per-workgroup energy + virial tree-reduction (shared 6-barrier tree) ──
    wg_energy[lid] = select(0.0, F_i + pair_energy,   is_active);
    wg_vxx[lid]    = select(0.0, vxx + kc_vxx, is_active);
    wg_vyy[lid]    = select(0.0, vyy + kc_vyy, is_active);
    wg_vzz[lid]    = select(0.0, vzz + kc_vzz, is_active);
    wg_vyz[lid]    = select(0.0, vyz + kc_vyz, is_active);
    wg_vxz[lid]    = select(0.0, vxz + kc_vxz, is_active);
    wg_vxy[lid]    = select(0.0, vxy + kc_vxy, is_active);
    workgroupBarrier();

    for (var stride: u32 = 32u; stride > 0u; stride = stride >> 1u) {
        if lid < stride {
            wg_energy[lid] = wg_energy[lid] + wg_energy[lid + stride];
            wg_vxx[lid]    = wg_vxx[lid]    + wg_vxx[lid + stride];
            wg_vyy[lid]    = wg_vyy[lid]    + wg_vyy[lid + stride];
            wg_vzz[lid]    = wg_vzz[lid]    + wg_vzz[lid + stride];
            wg_vyz[lid]    = wg_vyz[lid]    + wg_vyz[lid + stride];
            wg_vxz[lid]    = wg_vxz[lid]    + wg_vxz[lid + stride];
            wg_vxy[lid]    = wg_vxy[lid]    + wg_vxy[lid + stride];
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
