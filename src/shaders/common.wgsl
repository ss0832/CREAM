// =============================================================================
// common.wgsl  ——  Common definitions prepended to all pass shaders via concat!
//
// Binding layout (group 0, shared by all passes):
//   binding 0  positions        : array<vec4<f32>>   xyz + w=0 padding
//                                  (vec4 required by WebGPU's 16-byte alignment)
//   binding 1  atom_types       : array<u32>
//   binding 2  params           : SimParams  (uniform)
//   binding 3  potential_tables : array<f32>  — single flat buffer packing
//                                  all EAM tables (see `TableLayout`)
//   binding 4  tbl              : TableLayout (uniform)
// Pass-specific bindings are declared from binding 5 onward in each shader.
// =============================================================================

struct SimParams {
    n_atoms:       u32,
    n_elem:        u32,
    cutoff_sq:     f32,
    min_dist_sq:   f32,
    h0:            vec4<f32>,
    h1:            vec4<f32>,
    h2:            vec4<f32>,
    hinv_col0:     vec4<f32>,
    hinv_col1:     vec4<f32>,
    hinv_col2:     vec4<f32>,
    use_cell_list: u32,
    use_pbc:       u32,
    _pad0:         u32,
    _pad1:         u32,
    n_cells_x:     u32,
    n_cells_y:     u32,
    n_cells_z:     u32,
    n_cells_total: u32,
    cell_size:     f32,
    n_cells_x_pad: u32,
    n_cells_y_pad: u32,
    n_cells_z_pad: u32,
    // AllPairs chunking: first atom index and count for this chunk.
    // For a full (non-chunked) dispatch: i_offset = 0, i_count = n_atoms.
    i_offset:      u32,
    i_count:       u32,
    _pad2:         u32,
    _pad3:         u32,
}
// Total: 176 bytes (multiple of 16). MUST match GpuSimParams in src/engine.rs.

struct TableLayout {
    nr:              u32,
    nrho:            u32,
    n_elem:          u32,
    n_pairs:         u32,
    dr_inv:          f32,
    drho_inv:        f32,
    rho_offset:      u32,
    embed_offset:    u32,
    dembed_offset:   u32,
    pair_offset:     u32,
    drho_tab_offset: u32,
    dpair_offset:    u32,
    _pad0:           u32,
    _pad1:           u32,
    _pad2:           u32,
    _pad3:           u32,
}

@group(0) @binding(0) var<storage, read> positions        : array<vec4<f32>>;
@group(0) @binding(1) var<storage, read> atom_types       : array<u32>;
@group(0) @binding(2) var<uniform>       params           : SimParams;
@group(0) @binding(3) var<storage, read> potential_tables : array<f32>;
@group(0) @binding(4) var<uniform>       tbl              : TableLayout;

const MIN_R_SQ: f32 = 1e-4;

// ── Triclinic minimum image convention ────────────────────────────────────────
fn min_image(d: vec3<f32>) -> vec3<f32> {
    if params.use_pbc == 0u { return d; }
    let sx = dot(d, params.hinv_col0.xyz);
    let sy = dot(d, params.hinv_col1.xyz);
    let sz = dot(d, params.hinv_col2.xyz);
    let fx = sx - round(sx);
    let fy = sy - round(sy);
    let fz = sz - round(sz);
    return fx * params.h0.xyz + fy * params.h1.xyz + fz * params.h2.xyz;
}

// ── Table lookup ───────────────────────────────────────────────────────────────
//
// _lerp: linear interpolation inside a contiguous table sub-section.
//
// Zero-division guard:  n < 2 would make (n - 2u) wrap to 0xFFFFFFFE;
// we clamp it to 0 via max(..., 0u) so the read stays at index 0.
// idx_f is clamped to [0, n-1] before the cast so a negative or
// out-of-range query saturates at the table boundary rather than
// producing an out-of-bounds GPU access or NaN.
fn _lerp(base: u32, n: u32, idx_f: f32) -> f32 {
    // Clamp idx_f to the valid range [0, n-1] to avoid OOB reads and NaN.
    let safe_n    = max(n, 2u);                          // need at least 2 entries for lerp
    let idx_clamp = clamp(idx_f, 0.0, f32(safe_n - 1u));
    let idx       = u32(idx_clamp);
    let frac      = idx_clamp - f32(idx);
    let i0        = min(idx, safe_n - 2u);
    let v0        = potential_tables[base + i0];
    let v1        = potential_tables[base + i0 + 1u];
    return v0 + frac * (v1 - v0);
}

// r-axis lookup: f_α(r), φ_αβ(r), and derivatives.
fn lookup_by_r(section: u32, idx: u32, r: f32) -> f32 {
    // Guard: dr_inv == 0 (degenerate table) → return first table entry.
    return _lerp(section + idx * tbl.nr, tbl.nr, r * tbl.dr_inv);
}

// ρ-axis lookup: F_α(ρ), F′_α(ρ).
fn lookup_by_rho(section: u32, idx: u32, rho: f32) -> f32 {
    // Guard: drho_inv == 0 → return first table entry.
    return _lerp(section + idx * tbl.nrho, tbl.nrho, rho * tbl.drho_inv);
}

// Upper-triangular pair index — symmetric for (a,b) and (b,a).
// NOTE: when lo == 0u, `lo - 1u` wraps to 0xFFFFFFFF (u32 unsigned wrap),
// but `lo * (lo - 1u) = 0 * 0xFFFFFFFF = 0` so the result is always correct.
fn pair_idx(a: u32, b: u32) -> u32 {
    let lo = min(a, b);
    let hi = max(a, b);
    return lo * tbl.n_elem - lo * (lo - 1u) / 2u + (hi - lo);
}

// ── Morton Code helpers ───────────────────────────────────────────────────────
fn spread_bits(v: u32) -> u32 {
    var x: u32 = v & 0x000003ffu;
    x = (x | (x << 16u)) & 0x030000ffu;
    x = (x | (x <<  8u)) & 0x0300f00fu;
    x = (x | (x <<  4u)) & 0x030c30c3u;
    x = (x | (x <<  2u)) & 0x09249249u;
    return x;
}
fn morton3(cx: u32, cy: u32, cz: u32) -> u32 {
    return spread_bits(cx) | (spread_bits(cy) << 1u) | (spread_bits(cz) << 2u);
}
fn compact_bits(w: u32) -> u32 {
    var x: u32 = w & 0x09249249u;
    x = (x | (x >>  2u)) & 0x030c30c3u;
    x = (x | (x >>  4u)) & 0x0300f00fu;
    x = (x | (x >>  8u)) & 0x030000ffu;
    x = (x | (x >> 16u)) & 0x000003ffu;
    return x;
}
fn morton3_x(m: u32) -> u32 { return compact_bits(m      ); }
fn morton3_y(m: u32) -> u32 { return compact_bits(m >> 1u); }
fn morton3_z(m: u32) -> u32 { return compact_bits(m >> 2u); }
