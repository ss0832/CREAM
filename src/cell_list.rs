//! Shared CPU-side cell list data structure and helpers.
//!
//! Extracted from `cpu_engine.rs` so that `neighbor_list.rs` can reuse the
//! same Morton-ordered cell list for O(N) neighbour construction. Keeping a
//! single implementation avoids drift between the two call sites and ensures
//! that the CPU physics engine and the neighbour list builder agree on cell
//! assignment (crucial for reproducibility and debugging).
//!
//! # What lives here
//! * [`CellListData`] — Morton-ordered spatial index over atoms.
//! * Morton code helpers ([`morton_encode`], [`morton_decode_x`] etc.) —
//!   mirror the WGSL implementations in `src/shaders/common.wgsl`.
//! * Small f32 linear-algebra helpers ([`mat3_inv_f32`], [`min_image_mat_f32`])
//!   that both the cell list and the neighbour list need.
//!
//! Everything is `pub(crate)` so downstream users can't depend on these
//! details.

// ── Morton code (Z-order curve) helpers ───────────────────────────────────────
//
// Mirrors the WGSL implementation in `src/shaders/common.wgsl` so that the CPU
// cell list uses the same spatial ordering as the GPU cell list.
//
// `morton_spread(v)`  — inserts two zero bits between each of the 10 low bits
//                       of v.  Accepts values 0..1023 (10-bit input → 30-bit
//                       output).
// `morton_compact(w)` — inverse: extracts every 3rd bit starting at position 0.
// `morton_encode(cx, cy, cz)` — 3D coords → 30-bit Morton code.
// `morton_decode_{x,y,z}(m)`  — Morton code → single axis component.
//
// Reference: Warren & Salmon (1993); Pharr & Humphreys (2004) §7.3.

#[inline]
fn morton_spread(v: u32) -> u32 {
    let mut x = v & 0x0000_03ff; // keep only 10 bits
    x = (x | (x << 16)) & 0x0300_00ff;
    x = (x | (x << 8)) & 0x0300_f00f;
    x = (x | (x << 4)) & 0x030c_30c3;
    x = (x | (x << 2)) & 0x0924_9249;
    x
}

#[allow(dead_code)] // mirrors GPU shader compact_bits; kept for parity / future use
#[inline]
fn morton_compact(w: u32) -> u32 {
    let mut x = w & 0x0924_9249;
    x = (x | (x >> 2)) & 0x030c_30c3;
    x = (x | (x >> 4)) & 0x0300_f00f;
    x = (x | (x >> 8)) & 0x0300_00ff;
    x = (x | (x >> 16)) & 0x0000_03ff;
    x
}

/// Encode 3D cell coordinates → 30-bit Morton (Z-order) code.
/// Each coordinate must fit in 10 bits (< 1024).
#[inline]
pub(crate) fn morton_encode(cx: u32, cy: u32, cz: u32) -> u32 {
    morton_spread(cx) | (morton_spread(cy) << 1) | (morton_spread(cz) << 2)
}

/// Decode Morton code → x component (bits 0, 3, 6, …).
#[allow(dead_code)] // mirrors GPU shader morton3_x; kept for parity / future use
#[inline]
pub(crate) fn morton_decode_x(m: u32) -> u32 {
    morton_compact(m)
}

/// Decode Morton code → y component (bits 1, 4, 7, …).
#[allow(dead_code)] // mirrors GPU shader morton3_y
#[inline]
pub(crate) fn morton_decode_y(m: u32) -> u32 {
    morton_compact(m >> 1)
}

/// Decode Morton code → z component (bits 2, 5, 8, …).
#[allow(dead_code)] // mirrors GPU shader morton3_z
#[inline]
pub(crate) fn morton_decode_z(m: u32) -> u32 {
    morton_compact(m >> 2)
}

// ── Small f32 linear-algebra helpers ──────────────────────────────────────────
//
// Both `cpu_engine.rs` and `neighbor_list.rs` need these.  They are kept here
// (not in a separate math module) because they are short and only meaningful
// together with the cell list.

/// Invert a 3×3 f32 matrix.  Returns `None` when the determinant is below
/// `1e-30` (effectively singular for our purposes).
#[inline]
pub(crate) fn mat3_inv_f32(h: &[[f32; 3]; 3]) -> Option<[[f32; 3]; 3]> {
    let [[h00, h01, h02], [h10, h11, h12], [h20, h21, h22]] = *h;
    let det = h00 * (h11 * h22 - h12 * h21) - h01 * (h10 * h22 - h12 * h20)
        + h02 * (h10 * h21 - h11 * h20);
    if det.abs() < 1e-30 {
        return None;
    }
    let d = 1.0 / det;
    Some([
        [
            (h11 * h22 - h12 * h21) * d,
            -(h01 * h22 - h02 * h21) * d,
            (h01 * h12 - h02 * h11) * d,
        ],
        [
            -(h10 * h22 - h12 * h20) * d,
            (h00 * h22 - h02 * h20) * d,
            -(h00 * h12 - h02 * h10) * d,
        ],
        [
            (h10 * h21 - h11 * h20) * d,
            -(h00 * h21 - h01 * h20) * d,
            (h00 * h11 - h01 * h10) * d,
        ],
    ])
}

/// Fold the raw value `(s * n).floor() as i32` to a valid cell index in
/// `[0, n)` for a **pre-wrapped** fractional coordinate.
///
/// # Why this isn't just `rem_euclid`
///
/// `compute_cell_list_sync` wraps every atom into `s ∈ [0, 1)` before
/// calling `CellListData::build`, but the wrap step does
/// `sf = s - s.floor();  x = sf · H` and the cell-assignment step does
/// `s_new = x · H⁻¹;  cx = floor(s_new · n).rem_euclid(n)`.  In f32 the
/// round-trip `sf → x → s_new` is **not** idempotent: for any `sf` close
/// enough to 1 (roughly `sf ≥ 1 − 2⁻²²`) the f32 product/quotient can
/// round `s_new` to exactly `1.0`, making `floor(s_new · n) = n`.  Then
/// `rem_euclid(n, n) = 0`, sending the atom to the **opposite** end of
/// the box while its Cartesian position is still at `x ≈ L`.  The ortho
/// fast-path in `for_each_forward_neighbor` derives its shift from the
/// cell-index delta, so a single mis-assigned atom corrupts every pair
/// it participates in.  Observed at rep ≥ 35 cubic in `bench_gpu` as
/// max |Δf| ≈ 7.6e-2 eV/Å.
///
/// # The fix
///
/// The artifact has one and only one signature: `raw == n` exactly.  We
/// fold that to `n − 1` (physically the correct cell for an atom at
/// `s ≈ 1`) and otherwise use the standard `rem_euclid` that handles
/// genuine periodic images (`raw ∈ [-1, n-1]` maps to `[0, n-1]`,
/// `raw ≥ n + 1` maps normally, etc.).  No-op in every case except the
/// boundary rounding artifact itself.
#[inline]
pub(crate) fn fold_cell_index_pbc(raw: i32, n: i32) -> i32 {
    debug_assert!(n >= 1);
    if raw == n {
        // f32 round-trip artifact: atom at s ≈ 1 wrapped up to s_new == 1.0
        // exactly.  Correct cell is n-1, NOT 0.
        n - 1
    } else {
        raw.rem_euclid(n)
    }
}

/// Triclinic minimum-image displacement in f32, matching WGSL `common.wgsl`
/// (uses `round()` = IEEE round-to-nearest-ties-to-even in the shader; Rust's
/// `f32::round` is ties-away-from-zero — the difference only matters at the
/// exact ±0.5 boundary, which the `skin` margin in the NL builder handles).
#[inline]
pub(crate) fn min_image_mat_f32(d: [f32; 3], h: &[[f32; 3]; 3], h_inv: &[[f32; 3]; 3]) -> [f32; 3] {
    let s = [
        d[0] * h_inv[0][0] + d[1] * h_inv[1][0] + d[2] * h_inv[2][0],
        d[0] * h_inv[0][1] + d[1] * h_inv[1][1] + d[2] * h_inv[2][1],
        d[0] * h_inv[0][2] + d[1] * h_inv[1][2] + d[2] * h_inv[2][2],
    ];
    let sf = [
        s[0] - s[0].round(),
        s[1] - s[1].round(),
        s[2] - s[2].round(),
    ];
    [
        sf[0] * h[0][0] + sf[1] * h[1][0] + sf[2] * h[2][0],
        sf[0] * h[0][1] + sf[1] * h[1][1] + sf[2] * h[2][1],
        sf[0] * h[0][2] + sf[1] * h[1][2] + sf[2] * h[2][2],
    ]
}

// ── Cell list ─────────────────────────────────────────────────────────────────
//
// Supports both **orthorhombic and triclinic** PBC boxes and finite (non-PBC)
// systems.
//
// ## PBC (orthorhombic + triclinic): fractional-coordinate grid
//
// Cell assignment uses fractional coordinates s = x @ H⁻¹, so the grid is
// uniform in fractional space regardless of cell shape.  Grid dimensions are
// set from perpendicular heights so that one cell height ≥ cutoff in each
// direction, guaranteeing the 27-cell stencil captures all neighbours.
//
// ## Non-PBC: Cartesian bounding-box grid
//
// A Cartesian grid over the atom positions + cutoff/2 margin.

/// Morton-ordered cell list used by both `cpu_engine` (physics) and
/// `neighbor_list` (GPU-NL builder).  All fields are `pub(crate)` so callers
/// can read them directly without a heavyweight getter layer.
pub(crate) struct CellListData {
    /// Number of cells along each axis (actual grid dimensions).
    pub(crate) n: [usize; 3],
    /// Next power-of-two padding for each axis, used as the Morton code grid.
    /// Mirrors `n_cells_x_pad / n_cells_y_pad / n_cells_z_pad` in the GPU
    /// `SimParams` uniform — kept for parity and to enable future serialisation.
    #[allow(dead_code)]
    pub(crate) n_pad: [usize; 3],
    /// True size of the Morton code space:
    ///   `morton_encode(n_pad[0]−1, n_pad[1]−1, n_pad[2]−1) + 1`
    ///
    /// **Not** equal to `n_pad[0] * n_pad[1] * n_pad[2]` for non-cubic grids.
    /// Example: n_pad = [2, 2, 8] → product = 32, but morton3(1,1,7) = 295,
    /// so `n_morton` = 296.  Using the product would cause OOB in `cell_start`.
    /// Stored for diagnostics and future serialisation; `cell_start.len() == n_morton + 1`.
    #[allow(dead_code)]
    pub(crate) n_morton: usize,
    /// Whether the PBC cell is orthorhombic (diagonal H matrix).
    ///
    /// `true`  → stencil-based shift is safe (atoms in `[0,L)³` have fractional
    ///            coords in `[0,1)`, so cell index uniquely determines the image).
    /// `false` → per-pair `min_image_mat_f32` fallback used in neighbour iterators.
    pub(crate) is_ortho: bool,
    /// PBC: `Some(H⁻¹)` used for fractional-coordinate cell assignment.
    /// Non-PBC: `None` (Cartesian box_lo / cell_size used instead).
    pub(crate) h_inv: Option<[[f32; 3]; 3]>,
    /// Non-PBC only: lower corner of the Cartesian bounding box.
    pub(crate) box_lo: [f32; 3],
    /// Non-PBC only: Cartesian cell dimensions (Å).
    pub(crate) cell_size: [f32; 3],
    /// `cell_start[m]` = first index in `sorted` belonging to Morton bin `m`.
    /// Length = `n_morton + 1` (sentinel at the end).
    /// Indexed by `morton_encode(cx, cy, cz)`, so spatially adjacent cells
    /// have nearby indices — improving cache locality during neighbour traversal.
    pub(crate) cell_start: Vec<usize>,
    /// Atom indices sorted by Morton code (Z-order) of their cell.
    pub(crate) sorted: Vec<usize>,
}

impl CellListData {
    /// Build a cell list.  Returns `None` only for zero-atom inputs or a
    /// singular cell matrix (degenerate geometry).
    pub(crate) fn build(
        positions: &[[f32; 4]],
        cell: Option<[[f32; 3]; 3]>,
        cutoff: f32,
    ) -> Option<Self> {
        let n_atoms = positions.len();
        if n_atoms == 0 {
            return None;
        }

        match cell {
            // ── PBC case (orthorhombic or triclinic) ────────────────────
            Some(h) => {
                let h_inv = mat3_inv_f32(&h)?; // None if singular

                // Perpendicular heights hₖ = |det H| / |aⱼ × aₗ|.
                // Each cell in direction k has height hₖ / nₖ ≥ cutoff,
                // ensuring the 27-stencil covers all neighbours.
                let cross = |u: [f32; 3], v: [f32; 3]| -> [f32; 3] {
                    [
                        u[1] * v[2] - u[2] * v[1],
                        u[2] * v[0] - u[0] * v[2],
                        u[0] * v[1] - u[1] * v[0],
                    ]
                };
                let norm = |w: [f32; 3]| (w[0] * w[0] + w[1] * w[1] + w[2] * w[2]).sqrt();
                let a = h[0];
                let b = h[1];
                let c = h[2];
                let vol = (a[0] * (b[1] * c[2] - b[2] * c[1]) - a[1] * (b[0] * c[2] - b[2] * c[0])
                    + a[2] * (b[0] * c[1] - b[1] * c[0]))
                    .abs();
                let n = [
                    ((vol / norm(cross(b, c)) / cutoff).floor() as usize).max(1),
                    ((vol / norm(cross(a, c)) / cutoff).floor() as usize).max(1),
                    ((vol / norm(cross(a, b)) / cutoff).floor() as usize).max(1),
                ];

                // Auto-power-of-two grid rounding (mirrors GPU v4 behaviour).
                //
                // # Why
                // The GPU path (`engine.rs::n_cells_from_dspacing`) rounds each
                // axis DOWN to a power of two so the BB-stencil fast path in
                // the CellList shaders is usable.  Before this change the CPU
                // cell list used the raw `floor(L/cutoff)` count, producing a
                // DIFFERENT grid from the GPU on the same geometry.  That by
                // itself is allowed (both are mathematically correct cell
                // lists), but the CPU grid had a subtle numerical weakness:
                // when `cell_width ≈ cutoff + ε` (2% skin), f32 roundoff in
                // `cell_coords_of` could place an atom in the wrong cell, and
                // the 27-stencil on the resulting layout missed ~2% of
                // near-cutoff pairs.  The symptom was per-atom force errors
                // of ~0.08 eV/Å reported as MISMATCH in `bench_gpu` at
                // rep ≥ 35 (Cubic N=171500, Tetra N=4e6, Ortho N=1.69e6).
                //
                // # Fix
                // Round down to the nearest power of two.  Rounding DOWN
                // only makes cells LARGER, preserving `cell_width ≥ cutoff`,
                // and it aligns CPU with GPU so test comparisons are direct.
                //
                // # Opt-out
                // Honour `CREAM_DISABLE_AUTO_P2=1` (same env var as GPU) so
                // regression tests can restore the raw non-p2 grid.
                let auto_p2 = !std::env::var("CREAM_DISABLE_AUTO_P2")
                    .ok()
                    .is_some_and(|v| v == "1" || v.eq_ignore_ascii_case("true"));
                let round_down_p2 = |k: usize| -> usize {
                    if k == 0 {
                        1
                    } else {
                        1usize << (usize::BITS - 1 - k.leading_zeros()) as usize
                    }
                };
                let n = if auto_p2 {
                    [
                        round_down_p2(n[0]),
                        round_down_p2(n[1]),
                        round_down_p2(n[2]),
                    ]
                } else {
                    n
                };

                // Morton-padded grid dimensions: round each axis up to the next
                // power of two so spread_bits produces a clean interleaved code.
                let n_pad = [
                    n[0].next_power_of_two(),
                    n[1].next_power_of_two(),
                    n[2].next_power_of_two(),
                ];

                // True Morton space size — the maximum code any valid cell can
                // produce, plus one.  For non-cubic grids this exceeds
                // n_pad[0]*n_pad[1]*n_pad[2]: e.g. n_pad=[2,2,8] gives
                // morton3(1,1,7)=295, so n_morton=296, not 32.
                let n_morton = morton_encode(
                    (n_pad[0] - 1) as u32,
                    (n_pad[1] - 1) as u32,
                    (n_pad[2] - 1) as u32,
                ) as usize
                    + 1;

                // Cell assignment via fractional coordinates → Morton code.
                // s[k] = Σⱼ x[j] · H⁻¹[j][k]   (same convention as min_image_mat_f32)
                //
                // For the `n → n-1` boundary artifact handled by
                // `fold_cell_index_pbc`, see that function's docstring.
                let cell_of = |p: &[f32; 4]| -> usize {
                    let s0 = p[0] * h_inv[0][0] + p[1] * h_inv[1][0] + p[2] * h_inv[2][0];
                    let s1 = p[0] * h_inv[0][1] + p[1] * h_inv[1][1] + p[2] * h_inv[2][1];
                    let s2 = p[0] * h_inv[0][2] + p[1] * h_inv[1][2] + p[2] * h_inv[2][2];
                    let cx =
                        fold_cell_index_pbc((s0 * n[0] as f32).floor() as i32, n[0] as i32) as u32;
                    let cy =
                        fold_cell_index_pbc((s1 * n[1] as f32).floor() as i32, n[1] as i32) as u32;
                    let cz =
                        fold_cell_index_pbc((s2 * n[2] as f32).floor() as i32, n[2] as i32) as u32;
                    morton_encode(cx, cy, cz) as usize
                };

                let (cell_start, sorted) =
                    Self::counting_sort_morton(positions, n_morton, &cell_of);

                // Orthorhombic iff all off-diagonal elements of H are negligible.
                // When true, stencil-based integer shifts are exact (atoms in [0,L)³
                // have fractional coordinates in [0,1)).  When false, per-pair
                // min_image_mat_f32 is used instead.
                let is_ortho = h[0][1].abs() < 1e-6
                    && h[0][2].abs() < 1e-6
                    && h[1][0].abs() < 1e-6
                    && h[1][2].abs() < 1e-6
                    && h[2][0].abs() < 1e-6
                    && h[2][1].abs() < 1e-6;

                Some(CellListData {
                    n,
                    n_pad,
                    n_morton,
                    is_ortho,
                    h_inv: Some(h_inv),
                    box_lo: [0.0; 3],
                    cell_size: [0.0; 3],
                    cell_start,
                    sorted,
                })
            }

            // ── Non-PBC case: Cartesian bounding box ────────────────────
            None => {
                let mut lo = [f32::INFINITY; 3];
                let mut hi = [f32::NEG_INFINITY; 3];
                for p in positions {
                    for k in 0..3 {
                        lo[k] = lo[k].min(p[k]);
                        hi[k] = hi[k].max(p[k]);
                    }
                }
                let margin = cutoff * 0.5;
                let box_lo = [lo[0] - margin, lo[1] - margin, lo[2] - margin];
                let box_size = [
                    hi[0] - lo[0] + 2.0 * margin,
                    hi[1] - lo[1] + 2.0 * margin,
                    hi[2] - lo[2] + 2.0 * margin,
                ];
                let n = [
                    ((box_size[0] / cutoff).floor() as usize).max(1),
                    ((box_size[1] / cutoff).floor() as usize).max(1),
                    ((box_size[2] / cutoff).floor() as usize).max(1),
                ];
                let cell_size = [
                    box_size[0] / n[0] as f32,
                    box_size[1] / n[1] as f32,
                    box_size[2] / n[2] as f32,
                ];
                // Morton-padded grid dimensions.
                let n_pad = [
                    n[0].next_power_of_two(),
                    n[1].next_power_of_two(),
                    n[2].next_power_of_two(),
                ];
                // True Morton space size (see PBC branch for explanation).
                let n_morton = morton_encode(
                    (n_pad[0] - 1) as u32,
                    (n_pad[1] - 1) as u32,
                    (n_pad[2] - 1) as u32,
                ) as usize
                    + 1;
                let cell_of = |p: &[f32; 4]| -> usize {
                    let cx =
                        (((p[0] - box_lo[0]) / cell_size[0]).floor() as usize).min(n[0] - 1) as u32;
                    let cy =
                        (((p[1] - box_lo[1]) / cell_size[1]).floor() as usize).min(n[1] - 1) as u32;
                    let cz =
                        (((p[2] - box_lo[2]) / cell_size[2]).floor() as usize).min(n[2] - 1) as u32;
                    morton_encode(cx, cy, cz) as usize
                };
                let (cell_start, sorted) =
                    Self::counting_sort_morton(positions, n_morton, &cell_of);
                Some(CellListData {
                    n,
                    n_pad,
                    n_morton,
                    is_ortho: false, // non-PBC: field unused (pbc is None in iterators)
                    h_inv: None,
                    box_lo,
                    cell_size,
                    cell_start,
                    sorted,
                })
            }
        }
    }

    /// Sort atoms into Morton-ordered bins via **parallel unstable sort**.
    ///
    /// Uses `rayon::par_sort_unstable_by_key` (parallel PDQsort, in-place).
    /// Compared with sequential counting sort:
    /// - **No** auxiliary fill-pointer bookkeeping pass.
    /// - **Parallel** key computation + sort → better utilisation on multi-core.
    /// - Atoms within the same bin may be reordered between runs; intra-bin
    ///   order is irrelevant because neighbour traversal is order-independent.
    ///
    /// `cell_of` must be `Sync` (each element is computed in parallel).
    #[cfg(not(miri))]
    fn counting_sort_morton(
        positions: &[[f32; 4]],
        n_morton: usize,
        cell_of: &(impl Fn(&[f32; 4]) -> usize + Sync),
    ) -> (Vec<usize>, Vec<usize>) {
        use rayon::prelude::*;
        let n_atoms = positions.len();

        // Build (Morton code, original index) pairs in parallel.
        let mut keyed: Vec<(usize, usize)> = (0..n_atoms)
            .into_par_iter()
            .map(|idx| {
                let c = cell_of(&positions[idx]);
                debug_assert!(c < n_morton, "Morton code {c} out of range [0, {n_morton})");
                (c, idx)
            })
            .collect();

        // Parallel unstable sort — in-place PDQsort, no merge buffer.
        // Intra-bin order does not affect physics, so unstable is safe.
        keyed.par_sort_unstable_by_key(|&(code, _)| code);

        // Extract sorted atom indices.
        let sorted: Vec<usize> = keyed.iter().map(|&(_, idx)| idx).collect();

        // Rebuild cell_start prefix-sum from the already-sorted key sequence
        // (one pass over keyed — cheaper than re-computing cell_of).
        let mut cell_start = vec![0usize; n_morton + 1];
        for &(code, _) in &keyed {
            cell_start[code + 1] += 1;
        }
        for i in 0..n_morton {
            cell_start[i + 1] += cell_start[i];
        }
        (cell_start, sorted)
    }

    /// Sequential fallback for Miri (rayon unavailable under Miri).
    #[cfg(miri)]
    fn counting_sort_morton(
        positions: &[[f32; 4]],
        n_morton: usize,
        cell_of: &(impl Fn(&[f32; 4]) -> usize + Sync),
    ) -> (Vec<usize>, Vec<usize>) {
        let n_atoms = positions.len();
        let mut counts = vec![0usize; n_morton];
        let mut atom_cell = vec![0usize; n_atoms];
        for (idx, p) in positions.iter().enumerate() {
            let c = cell_of(p);
            debug_assert!(c < n_morton, "Morton code {c} out of range [0, {n_morton})");
            atom_cell[idx] = c;
            counts[c] += 1;
        }
        let mut cell_start = vec![0usize; n_morton + 1];
        for c in 0..n_morton {
            cell_start[c + 1] = cell_start[c] + counts[c];
        }
        let mut sorted = vec![0usize; n_atoms];
        let mut fill_ptr = cell_start[..n_morton].to_vec();
        for (idx, &c) in atom_cell.iter().enumerate() {
            sorted[fill_ptr[c]] = idx;
            fill_ptr[c] += 1;
        }
        (cell_start, sorted)
    }

    /// Cell-grid coordinates `(cx, cy, cz)` for atom at position `pi`.
    ///
    /// PBC: fractional-coordinate grid (same convention as [`CellListData::build`]).
    /// Non-PBC: Cartesian bounding-box grid, clamped to `[0, n-1]`.
    ///
    /// For the `n → n-1` boundary artifact handled in the PBC branch, see
    /// [`fold_cell_index_pbc`].
    #[inline]
    pub(crate) fn cell_coords_of(&self, pi: &[f32; 4]) -> (i32, i32, i32) {
        let [n0, n1, n2] = self.n;
        if let Some(ref hi) = self.h_inv {
            let s0 = pi[0] * hi[0][0] + pi[1] * hi[1][0] + pi[2] * hi[2][0];
            let s1 = pi[0] * hi[0][1] + pi[1] * hi[1][1] + pi[2] * hi[2][1];
            let s2 = pi[0] * hi[0][2] + pi[1] * hi[1][2] + pi[2] * hi[2][2];
            (
                fold_cell_index_pbc((s0 * n0 as f32).floor() as i32, n0 as i32),
                fold_cell_index_pbc((s1 * n1 as f32).floor() as i32, n1 as i32),
                fold_cell_index_pbc((s2 * n2 as f32).floor() as i32, n2 as i32),
            )
        } else {
            (
                (((pi[0] - self.box_lo[0]) / self.cell_size[0]).floor() as i32)
                    .clamp(0, n0 as i32 - 1),
                (((pi[1] - self.box_lo[1]) / self.cell_size[1]).floor() as i32)
                    .clamp(0, n1 as i32 - 1),
                (((pi[2] - self.box_lo[2]) / self.cell_size[2]).floor() as i32)
                    .clamp(0, n2 as i32 - 1),
            )
        }
    }

    /// Compute the Cartesian PBC shift for a neighbour cell at offset `(dcx, dcy, dcz)`.
    ///
    /// When the raw cell index `cx0 + dcx` wraps around the periodic boundary,
    /// the crossing count `k = (raw - wrapped) / n` is either −1, 0, or +1.
    /// The Cartesian shift is then `k_x·H[0] + k_y·H[1] + k_z·H[2]`, replacing
    /// the per-pair `round()` calls in `min_image_mat_f32`.
    ///
    /// Returns `(nx_wrapped, ny_wrapped, nz_wrapped, shift)`.
    ///
    /// # Safety argument
    /// The 2×cutoff cell-height guarantee ensures each `dcx/dcy/dcz ∈ {-1,0,1}`
    /// maps to exactly one periodic image, so the integer `k` is exact.
    #[allow(clippy::too_many_arguments)] // (cx0,cy0,cz0) + (dcx,dcy,dcz) + h are all physically distinct
    #[inline]
    pub(crate) fn pbc_cell_and_shift(
        &self,
        cx0: i32,
        cy0: i32,
        cz0: i32,
        dcx: i32,
        dcy: i32,
        dcz: i32,
        h: &[[f32; 3]; 3],
    ) -> (u32, u32, u32, [f32; 3]) {
        let [n0, n1, n2] = self.n;
        let nx_raw = cx0 + dcx;
        let ny_raw = cy0 + dcy;
        let nz_raw = cz0 + dcz;
        let nx_w = nx_raw.rem_euclid(n0 as i32);
        let ny_w = ny_raw.rem_euclid(n1 as i32);
        let nz_w = nz_raw.rem_euclid(n2 as i32);
        // kx/ky/kz: number of full box lengths crossed (-1, 0, or +1).
        let kx = (nx_raw - nx_w) / n0 as i32;
        let ky = (ny_raw - ny_w) / n1 as i32;
        let kz = (nz_raw - nz_w) / n2 as i32;
        let shift = [
            kx as f32 * h[0][0] + ky as f32 * h[1][0] + kz as f32 * h[2][0],
            kx as f32 * h[0][1] + ky as f32 * h[1][1] + kz as f32 * h[2][1],
            kx as f32 * h[0][2] + ky as f32 * h[1][2] + kz as f32 * h[2][2],
        ];
        (nx_w as u32, ny_w as u32, nz_w as u32, shift)
    }
}

// ────────────────────────────────────────────────────────────────────────────
#[cfg(test)]
mod tests {
    use super::*;

    /// `morton_encode` / decode round-trip for a representative set of coords.
    #[test]
    fn morton_encode_decode_roundtrip() {
        for cx in 0..8u32 {
            for cy in 0..8u32 {
                for cz in 0..8u32 {
                    let m = morton_encode(cx, cy, cz);
                    assert_eq!(morton_decode_x(m), cx);
                    assert_eq!(morton_decode_y(m), cy);
                    assert_eq!(morton_decode_z(m), cz);
                }
            }
        }
    }

    #[test]
    fn morton_sort_order_is_monotone_within_bins() {
        // All 64 Morton codes for a 4×4×4 grid should be in [0, 64).
        let mut codes: Vec<u32> = (0..4)
            .flat_map(|cx| {
                (0..4).flat_map(move |cy| (0..4).map(move |cz| morton_encode(cx, cy, cz)))
            })
            .collect();
        codes.sort_unstable();
        for (i, c) in codes.iter().enumerate() {
            assert_eq!(*c, i as u32);
        }
    }

    #[test]
    fn cell_list_build_pbc_basic() {
        // 3×3×3 atoms in a 9 Å cube, cutoff 3 Å.
        //
        // Raw grid would be `floor(9/3) = 3` per axis.  Under auto-p2
        // rounding (v4 default) this becomes 2 per axis — cell_width = 4.5 Å
        // which still satisfies the cutoff.  The sort MUST still place all
        // 27 atoms (no cells are dropped) and the terminal sentinel equals
        // the atom count.
        let positions: Vec<[f32; 4]> = (0..3)
            .flat_map(|x| {
                (0..3).flat_map(move |y| {
                    (0..3).map(move |z| [x as f32 * 3.0, y as f32 * 3.0, z as f32 * 3.0, 0.0])
                })
            })
            .collect();
        let cell = [[9.0, 0.0, 0.0], [0.0, 9.0, 0.0], [0.0, 0.0, 9.0]];
        let cl = CellListData::build(&positions, Some(cell), 3.0).unwrap();
        // Under auto-p2, expect n = [2, 2, 2]; under raw path, n = [3, 3, 3].
        // Accept both so this test works with or without
        // CREAM_DISABLE_AUTO_P2.
        assert!(
            cl.n == [3, 3, 3] || cl.n == [2, 2, 2],
            "unexpected n: {:?}",
            cl.n
        );
        assert_eq!(cl.sorted.len(), 27);
        // Terminal sentinel must equal atom count.
        assert_eq!(*cl.cell_start.last().unwrap(), 27);
    }

    #[test]
    fn cell_list_build_non_pbc_basic() {
        let positions: Vec<[f32; 4]> = vec![
            [0.0, 0.0, 0.0, 0.0],
            [2.0, 0.0, 0.0, 0.0],
            [0.0, 2.0, 0.0, 0.0],
        ];
        let cl = CellListData::build(&positions, None, 2.0).unwrap();
        assert_eq!(cl.sorted.len(), 3);
        assert_eq!(*cl.cell_start.last().unwrap(), 3);
    }

    /// `fold_cell_index_pbc` must fold the `raw == n` boundary artifact to
    /// `n-1` and otherwise behave like `rem_euclid`.  The `raw == n` case is
    /// the specific f32 round-trip artifact observed at rep ≥ 35 cubic:
    /// an atom wrapped to `s ≈ 1 − ulp` had its `s * H⁻¹` round back to
    /// exactly `1.0`, making `floor(s · n) == n` and `rem_euclid(n, n) == 0`.
    #[test]
    fn fold_cell_index_pbc_boundary_artifact() {
        // The specific artifact: raw == n folds to n-1, NOT 0.
        assert_eq!(fold_cell_index_pbc(16, 16), 15);
        assert_eq!(fold_cell_index_pbc(32, 32), 31);
        assert_eq!(fold_cell_index_pbc(1, 1), 0); // n=1 degenerate: only cell is 0

        // In-range values pass through unchanged.
        for n in [2, 4, 8, 16, 32] {
            for raw in 0..n {
                assert_eq!(fold_cell_index_pbc(raw, n), raw);
            }
        }

        // Genuine periodic-image wrapping still works via rem_euclid.
        assert_eq!(fold_cell_index_pbc(-1, 16), 15); // one period behind start
        assert_eq!(fold_cell_index_pbc(-16, 16), 0); // exactly one period behind → 0
        assert_eq!(fold_cell_index_pbc(17, 16), 1); // one past boundary (NOT the artifact)
        assert_eq!(fold_cell_index_pbc(32 + 5, 16), 5); // two periods + 5
    }

    /// Regression test for the rep ≥ 35 cubic `bench_gpu` MISMATCH.
    ///
    /// Constructs an atom whose wrapped fractional coord lands exactly on
    /// `1.0 − 2⁻²²` (the threshold at which the pre-fix code returned the
    /// wrong cell index under f32 round-trip).  After the fix,
    /// `cell_coords_of` must return `n − 1` for that atom, matching its
    /// physical position at the far end of the box.
    #[test]
    fn cell_coords_of_handles_f32_boundary_roundtrip() {
        // n=16 grid on a Cubic box of side L = 35 × 3.615 (matches bench_gpu
        // rep=35).  This was the failing configuration.
        let l = 35.0_f32 * 3.615_f32;
        let cell = [[l, 0.0, 0.0], [0.0, l, 0.0], [0.0, 0.0, l]];
        // Build a minimal valid CellListData (one placeholder atom).
        let positions = vec![[0.0_f32, 0.0, 0.0, 0.0]];
        let cl = CellListData::build(&positions, Some(cell), 4.95).unwrap();
        assert_eq!(cl.n, [16, 16, 16]);

        // An atom at the worst-case boundary: physically right at the
        // far edge (x just below L after wrap).  We directly feed the
        // problematic f32 value `L - 2⁻¹⁶ · L` ≈ `L · (1 − 1.5e-5)`
        // which is the f32 representable that lies one ulp below L.
        let l_minus_1ulp = f32::from_bits(l.to_bits() - 1);
        let pi = [l_minus_1ulp, l_minus_1ulp, l_minus_1ulp, 0.0];
        let (cx, cy, cz) = cl.cell_coords_of(&pi);
        assert_eq!(
            (cx, cy, cz),
            (15, 15, 15),
            "boundary atom at x = L − 1 ulp must land in cell n-1, not cell 0"
        );
    }
}
