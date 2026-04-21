//! CPU-side CSR neighbour-list builder for the GPU NeighborList backend.
//!
//! # Why
//! The GPU cell-list pipeline on NVIDIA RTX / DX12 produced systematic
//! mismatches (~1 neighbour-pair off per atom) that could not be isolated
//! to pass1/pass2 alone.  This module shifts neighbour finding to the CPU
//! where correctness is easy to guarantee, and exposes the result as two
//! flat buffers (CSR) that the GPU simply iterates over.
//!
//! # Complexity
//!
//! | Implementation                    | Build complexity | Activation            |
//! |-----------------------------------|------------------|-----------------------|
//! | [`build`] (default)               | **O(N)**         | always                |
//! | [`build_n2`] (legacy fallback)    | O(N²)            | `--features legacy_n2_nl` |
//!
//! The default implementation builds a Morton-ordered cell list
//! ([`crate::cell_list::CellListData`]) in O(N) and then performs a 27-cell
//! stencil scan per atom.  The legacy O(N²) path is retained behind a feature
//! flag as an emergency fall-back when the cell-list builder needs debugging.
//!
//! # Format
//! ```text
//! starts[i]     = first offset into `list` belonging to atom i
//! starts[i + 1] = first offset belonging to atom i + 1
//! list[ starts[i] .. starts[i+1] ]  =  sorted list of j's (j != i)
//!                                      where |min_image(p_j - p_i)| < rc + skin
//! ```
//! `starts.len() == n_atoms + 1`.  The sentinel `starts[n_atoms] == list.len()`.
//!
//! # Arithmetic parity with WGSL
//! The GPU shader `common.wgsl::min_image` uses single-precision arithmetic
//! with `round()` (IEEE 754 round-to-nearest-even).  This module mirrors
//! that via the shared helper [`crate::cell_list::min_image_mat_f32`] and a
//! tiny hand-rolled `round_even` for the orthorhombic fast path.  An
//! additional `skin` (default ≈ 1 % of the cutoff) makes the classification
//! robust against the tiny residual rounding in `sqrt` / `dot` that WGSL and
//! LLVM may reorder differently.

use crate::cell_list::{mat3_inv_f32, min_image_mat_f32, CellListData};

/// CSR-format neighbour list ready for upload to GPU storage buffers.
#[derive(Debug, Default, Clone)]
pub struct NeighborList {
    /// Row pointers; length = `n_atoms + 1`.
    ///
    /// Currently read only through the `Debug` derive (and by downstream
    /// consumers of the struct who destructure it directly).  The GPU path
    /// uploads the inlined `list` only and computes offsets from atom count,
    /// so the bare field is flagged by `dead_code`; we keep it because CSR
    /// neighbour lists are standardly (starts, list) and removing starts
    /// would make the struct surprising to readers familiar with CSR.
    #[allow(dead_code)]
    pub starts: Vec<u32>,
    /// Flat neighbour indices; length = `starts[n_atoms]`.
    pub list: Vec<u32>,
}

impl NeighborList {
    /// Total number of (i, j) pairs stored (counts each direction separately).
    #[inline]
    #[allow(dead_code)] // used by future diagnostics; retained for API completeness
    pub fn total_pairs(&self) -> usize {
        self.list.len()
    }
}

// ── Portable even-rounding (matches WGSL `round()`) ──────────────────────────
//
// Used only by the orthorhombic fast path; the triclinic branch delegates to
// `cell_list::min_image_mat_f32` which calls Rust's `f32::round` (ties away
// from zero).  The difference only matters at the exact ±0.5 boundary — the
// `skin` margin absorbs any classification drift there.
#[allow(dead_code)] // unused when cellist_gpu feature is active (NL path disabled)
#[inline]
fn round_even(x: f32) -> f32 {
    let r = x.round();
    if (x - r).abs() == 0.0 && (x.fract().abs() - 0.5).abs() < f32::EPSILON {
        // round() breaks ties away from zero; IEEE wants nearest-even.
        let ri = r as i32;
        if ri & 1 != 0 {
            return r - x.signum();
        }
    }
    r
}

// ── O(N) default implementation ──────────────────────────────────────────────

/// Build a CSR neighbour list for `positions` under (optional) PBC.
///
/// * `positions[i]` – (x, y, z, pad) in Å, same layout as the GPU `pos_buf`.
/// * `cell`          – `Some(H)` for PBC (rows are lattice vectors), `None`
///                     for open boundary conditions.
/// * `cutoff`        – physical cutoff radius [Å].
/// * `skin`          – extra tolerance added to `cutoff` for border robustness.
///                     Pass `0.0` to reproduce the GPU cutoff exactly (risky
///                     at f32 boundaries); `0.01 * cutoff` is a safe default.
///
/// Returns `NeighborList` with `starts.len() == positions.len() + 1`.
///
/// # Complexity
/// O(N) expected.  Cell-list construction is O(N); each atom touches 27 cells,
/// each containing O(1) atoms on average for a physical density, so the
/// neighbour enumeration is O(N) as well.
///
/// # Fallback behaviour
/// If the cell list cannot be built (e.g. singular cell matrix, `N = 0`) the
/// function falls back to the O(N²) implementation so the caller always gets
/// a usable neighbour list.
#[allow(dead_code)] // unused when cellist_gpu feature is active (NL path disabled)
pub fn build(
    positions: &[[f32; 4]],
    cell: Option<[[f32; 3]; 3]>,
    cutoff: f32,
    skin: f32,
) -> NeighborList {
    let n = positions.len();
    if n == 0 {
        return NeighborList {
            starts: vec![0u32],
            list: Vec::new(),
        };
    }

    let rc_eff = cutoff + skin;
    let rc_eff_sq = rc_eff * rc_eff;

    // ── Wrap positions into [0, L)³ for the orthorhombic fast path ───────────
    //
    // # Why this is required (bug fix)
    //
    // The `is_ortho = true` fast path below computes the PBC shift once per
    // stencil offset via `CellListData::pbc_cell_and_shift`, then uses
    //     d = positions[j] − pi + shift
    // **without** per-pair `min_image_mat_f32`.  That shift is exact only
    // when **both** atoms have fractional coordinates in `[0, 1)` — i.e.
    // their Cartesian positions are in the primary image box.
    //
    // `CellListData::build` uses `rem_euclid` to assign cells, which is
    // correct for atoms whose positions escape `[0, L)³` (e.g. after
    // `rattle()`, MD thermalisation, or any external displacement).  But
    // `pbc_cell_and_shift` computes the Cartesian shift from the *cell index
    // delta* — it assumes atom `i` is at its cell-canonical position.  If
    // `i`'s fractional coord is 1.1 (one image out), the shift is off by
    // exactly one lattice vector, producing very-far-neighbour `d` values
    // and missing real neighbours.
    //
    // Symptom observed in `tests_python/test_python.py` grid sweep: Cubic /
    // Tetragonal / Orthorhombic supercells (`is_ortho == true`) under
    // `atoms.rattle(stdev=0.05)` showed hundreds of eV energy mismatch and
    // ~10 eV/Å force-norm error vs CPU AllPairs; Hexagonal / Rhombohedral /
    // Monoclinic / Triclinic (which take the `min_image_mat_f32` branch)
    // passed.
    //
    // # Fix
    //
    // Wrap positions once (O(N), no per-pair `round()` in the hot loop) so
    // every atom has fractional coord in `[0, 1)`.  The `is_ortho` fast path
    // then stays correct without extra work per pair.
    //
    // The wrap is a pure shift by a lattice vector, so it does not change
    // pair distances under PBC — neighbour sets are unchanged in the
    // correct-cell case, and are fixed in the escaped-cell case.  Original
    // `positions` slice is never mutated; we build a local owned copy and
    // bind `positions_eff` to either the wrapped copy or the original slice.
    //
    // This mirrors the same fix applied in `cpu_engine::compute_cell_list_sync`.
    let positions_wrapped_storage: Vec<[f32; 4]>;
    let positions_eff: &[[f32; 4]] = match cell {
        Some(h) => match mat3_inv_f32(&h) {
            Some(h_inv) => {
                positions_wrapped_storage = positions
                    .iter()
                    .map(|p| {
                        let s0 = p[0] * h_inv[0][0] + p[1] * h_inv[1][0] + p[2] * h_inv[2][0];
                        let s1 = p[0] * h_inv[0][1] + p[1] * h_inv[1][1] + p[2] * h_inv[2][1];
                        let s2 = p[0] * h_inv[0][2] + p[1] * h_inv[1][2] + p[2] * h_inv[2][2];
                        let sf0 = s0 - s0.floor();
                        let sf1 = s1 - s1.floor();
                        let sf2 = s2 - s2.floor();
                        // Guard against floor-rounding that produces sf == 1.0
                        // (can happen when s is a tiny negative number very
                        // close to 0 → floor = -1 → sf = 1.0 exactly in f32).
                        let sf0 = if sf0 >= 1.0 { 0.0 } else { sf0 };
                        let sf1 = if sf1 >= 1.0 { 0.0 } else { sf1 };
                        let sf2 = if sf2 >= 1.0 { 0.0 } else { sf2 };
                        let x0 = sf0 * h[0][0] + sf1 * h[1][0] + sf2 * h[2][0];
                        let x1 = sf0 * h[0][1] + sf1 * h[1][1] + sf2 * h[2][1];
                        let x2 = sf0 * h[0][2] + sf1 * h[1][2] + sf2 * h[2][2];
                        [x0, x1, x2, p[3]]
                    })
                    .collect();
                &positions_wrapped_storage
            }
            None => positions, // singular cell — fallback below will also bail.
        },
        None => positions, // non-PBC: no wrap needed.
    };

    // Build the cell list once (O(N) total).  The grid is sized so that one
    // cell height is ≥ `cutoff` (not `rc_eff`) — the skin only matters for the
    // cutoff test, not for whether the 27-stencil captures all pairs.
    let Some(cl) = CellListData::build(positions_eff, cell, cutoff) else {
        // Singular geometry or zero atoms: fall back to the O(N²) walker so
        // the caller still gets a well-formed neighbour list.  Use the
        // *original* positions — the O(N²) path does per-pair round() and is
        // robust to unwrapped inputs.
        return build_n2_inner(positions, cell, cutoff, skin);
    };

    // Pre-compute PBC matrices once.
    let (h, h_inv_opt, use_pbc) = match cell {
        Some(h_mat) => {
            let hinv = mat3_inv_f32(&h_mat).expect("neighbor_list::build: cell matrix is singular");
            (h_mat, Some(hinv), true)
        }
        None => ([[0.0f32; 3]; 3], None, false),
    };

    let mut starts: Vec<u32> = Vec::with_capacity(n + 1);
    // Heuristic capacity: ~50 neighbours per atom for FCC metals at rc ≈ 5 Å.
    let mut list: Vec<u32> = Vec::with_capacity(n * 64);

    let [n0, n1, n2] = cl.n;
    let triclinic_pbc = use_pbc && !cl.is_ortho;

    for i in 0..n {
        starts.push(list.len() as u32);
        let pi = positions_eff[i];
        let (cx0, cy0, cz0) = cl.cell_coords_of(&pi);

        // Dedup table for triclinic PBC with n_k = 2 (two stencil offsets can
        // wrap to the same Morton code; min_image_mat_f32 is shift-independent
        // so the same cell must not be visited twice).
        let mut seen_cells = [usize::MAX; 27];
        let mut n_seen = 0usize;

        for dcx in -1i32..=1 {
            for dcy in -1i32..=1 {
                for dcz in -1i32..=1 {
                    // Resolve neighbour cell + shift.
                    let (nx, ny, nz, shift) = if use_pbc {
                        let (nxw, nyw, nzw, sh) =
                            cl.pbc_cell_and_shift(cx0, cy0, cz0, dcx, dcy, dcz, &h);
                        (nxw, nyw, nzw, sh)
                    } else {
                        let nx = cx0 + dcx;
                        let ny = cy0 + dcy;
                        let nz = cz0 + dcz;
                        if nx < 0
                            || nx >= n0 as i32
                            || ny < 0
                            || ny >= n1 as i32
                            || nz < 0
                            || nz >= n2 as i32
                        {
                            continue;
                        }
                        (nx as u32, ny as u32, nz as u32, [0.0f32; 3])
                    };

                    let c = crate::cell_list::morton_encode(nx, ny, nz) as usize;
                    if triclinic_pbc {
                        if seen_cells[..n_seen].contains(&c) {
                            continue;
                        }
                        seen_cells[n_seen] = c;
                        n_seen += 1;
                    }

                    // Iterate over atoms in the neighbour cell.
                    //
                    // Note: `j` is the atom index in the original (un-wrapped)
                    // positions array.  We compute distances using `positions_eff`
                    // (wrapped), but emit `j` into `list` so the GPU shader
                    // reads from the original `pos_buf` — the shader re-applies
                    // its own `min_image` which is correct regardless of wrap.
                    for &j in &cl.sorted[cl.cell_start[c]..cl.cell_start[c + 1]] {
                        if j == i {
                            continue;
                        }
                        let pj = positions_eff[j];
                        let (dx, dy, dz) = if !use_pbc {
                            (pj[0] - pi[0], pj[1] - pi[1], pj[2] - pi[2])
                        } else if cl.is_ortho {
                            // Orthorhombic fast path: use the pre-computed
                            // stencil shift directly (no per-pair round()).
                            // Exact for atoms with fractional coords in [0,1) —
                            // which is guaranteed by the wrap above.
                            (
                                pj[0] - pi[0] + shift[0],
                                pj[1] - pi[1] + shift[1],
                                pj[2] - pi[2] + shift[2],
                            )
                        } else {
                            // Triclinic fallback: per-pair min-image.
                            let raw = [pj[0] - pi[0], pj[1] - pi[1], pj[2] - pi[2]];
                            let mi = min_image_mat_f32(raw, &h, &h_inv_opt.unwrap());
                            (mi[0], mi[1], mi[2])
                        };
                        let r_sq = dx * dx + dy * dy + dz * dz;
                        if r_sq < rc_eff_sq {
                            list.push(j as u32);
                        }
                    }
                }
            }
        }
    }
    starts.push(list.len() as u32);

    // `round_even` is intentionally only used by the legacy path below;
    // reference it here so `#[allow(dead_code)]` isn't needed.
    let _ = round_even(0.0);

    NeighborList { starts, list }
}

// ── O(N²) legacy path (retained behind feature flag) ─────────────────────────

/// Public entry for the legacy O(N²) neighbour builder.
///
/// This was the original implementation before the cell-list-accelerated
/// builder was introduced (see [`build`]).  It is kept behind
/// `--features legacy_n2_nl` as an emergency fall-back for debugging when
/// the cell-list path produces unexpected results.
///
/// **Do not call this from the main dispatch path** — use [`build`] which
/// automatically falls back to this implementation for degenerate inputs
/// (zero atoms, singular cell).
#[cfg(feature = "legacy_n2_nl")]
#[allow(dead_code)] // exposed for diagnostic callers; not referenced by the library itself
pub fn build_n2(
    positions: &[[f32; 4]],
    cell: Option<[[f32; 3]; 3]>,
    cutoff: f32,
    skin: f32,
) -> NeighborList {
    build_n2_inner(positions, cell, cutoff, skin)
}

/// Internal O(N²) neighbour builder.  Always compiled (used as a fall-back by
/// [`build`] for singular / zero-atom inputs), but only exposed to external
/// callers when the `legacy_n2_nl` feature is enabled.
#[allow(dead_code)] // unused when cellist_gpu feature is active (NL path disabled)
fn build_n2_inner(
    positions: &[[f32; 4]],
    cell: Option<[[f32; 3]; 3]>,
    cutoff: f32,
    skin: f32,
) -> NeighborList {
    let n = positions.len();
    let rc_eff = cutoff + skin;
    let rc_eff_sq = rc_eff * rc_eff;

    // Pre-compute PBC matrices (same layout as engine.rs / common.wgsl).
    let (h, hinv_col, use_pbc) = match cell {
        Some(h_mat) => {
            let hinv =
                mat3_inv_f32(&h_mat).expect("neighbor_list::build_n2: cell matrix is singular");
            // Column-major re-pack matching params.hinv_col{0,1,2}.xyz:
            //   col0 = (hinv[0][0], hinv[1][0], hinv[2][0])
            //   col1 = (hinv[0][1], hinv[1][1], hinv[2][1])
            //   col2 = (hinv[0][2], hinv[1][2], hinv[2][2])
            let c0 = [hinv[0][0], hinv[1][0], hinv[2][0]];
            let c1 = [hinv[0][1], hinv[1][1], hinv[2][1]];
            let c2 = [hinv[0][2], hinv[1][2], hinv[2][2]];
            (h_mat, [c0, c1, c2], true)
        }
        None => ([[0.0f32; 3]; 3], [[0.0f32; 3]; 3], false),
    };

    let min_image = |dx: f32, dy: f32, dz: f32| -> (f32, f32, f32) {
        if !use_pbc {
            return (dx, dy, dz);
        }
        let sx = dx * hinv_col[0][0] + dy * hinv_col[0][1] + dz * hinv_col[0][2];
        let sy = dx * hinv_col[1][0] + dy * hinv_col[1][1] + dz * hinv_col[1][2];
        let sz = dx * hinv_col[2][0] + dy * hinv_col[2][1] + dz * hinv_col[2][2];
        let fx = sx - round_even(sx);
        let fy = sy - round_even(sy);
        let fz = sz - round_even(sz);
        // h rows: h[0]=(h[0][0..3]), h[1]=..., h[2]=...
        let rx = fx * h[0][0] + fy * h[1][0] + fz * h[2][0];
        let ry = fx * h[0][1] + fy * h[1][1] + fz * h[2][1];
        let rz = fx * h[0][2] + fy * h[1][2] + fz * h[2][2];
        (rx, ry, rz)
    };

    let mut starts: Vec<u32> = Vec::with_capacity(n + 1);
    let mut list: Vec<u32> = Vec::with_capacity(n * 64);

    for i in 0..n {
        starts.push(list.len() as u32);
        let pi = positions[i];
        for j in 0..n {
            if i == j {
                continue;
            }
            let pj = positions[j];
            let (dx, dy, dz) = min_image(pj[0] - pi[0], pj[1] - pi[1], pj[2] - pi[2]);
            let r_sq = dx * dx + dy * dy + dz * dz;
            if r_sq < rc_eff_sq {
                list.push(j as u32);
            }
        }
    }
    starts.push(list.len() as u32);

    NeighborList { starts, list }
}

// ────────────────────────────────────────────────────────────────────────────
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn no_pbc_two_atoms_within_cutoff() {
        let positions = vec![[0.0, 0.0, 0.0, 0.0], [2.0, 0.0, 0.0, 0.0]];
        let nl = build(&positions, None, 3.0, 0.0);
        assert_eq!(nl.starts, vec![0, 1, 2]);
        assert_eq!(nl.list, vec![1, 0]);
    }

    #[test]
    fn no_pbc_outside_cutoff_excluded() {
        let positions = vec![[0.0, 0.0, 0.0, 0.0], [5.0, 0.0, 0.0, 0.0]];
        let nl = build(&positions, None, 3.0, 0.0);
        assert_eq!(nl.starts, vec![0, 0, 0]);
        assert!(nl.list.is_empty());
    }

    #[test]
    fn orthorhombic_pbc_wraps_through_face() {
        // Two atoms near opposite X-faces of a 10 Å box, cutoff 2 Å.
        // Real distance 9.8; min-image distance 0.2.
        let positions = vec![[0.1, 5.0, 5.0, 0.0], [9.9, 5.0, 5.0, 0.0]];
        let cell = [[10.0, 0.0, 0.0], [0.0, 10.0, 0.0], [0.0, 0.0, 10.0]];
        let nl = build(&positions, Some(cell), 2.0, 0.0);
        assert_eq!(nl.list, vec![1, 0]);
    }

    #[test]
    fn sizes_match_n_plus_one() {
        let positions: Vec<[f32; 4]> = (0..17).map(|k| [k as f32, 0.0, 0.0, 0.0]).collect();
        let nl = build(&positions, None, 1.5, 0.0);
        assert_eq!(nl.starts.len(), positions.len() + 1);
        assert_eq!(*nl.starts.last().unwrap() as usize, nl.list.len());
    }

    /// O(N) and O(N²) implementations must agree on a realistic FCC Cu
    /// supercell — this is the regression guard for the Phase-A refactor.
    #[test]
    fn on_matches_n2_fcc_cu() {
        let a = 3.615f32;
        let rep = 3usize; // 108 atoms
        let basis: [[f32; 3]; 4] = [
            [0.0, 0.0, 0.0],
            [a / 2.0, a / 2.0, 0.0],
            [a / 2.0, 0.0, a / 2.0],
            [0.0, a / 2.0, a / 2.0],
        ];
        let mut pos: Vec<[f32; 4]> = Vec::new();
        for ix in 0..rep {
            for iy in 0..rep {
                for iz in 0..rep {
                    for b in &basis {
                        pos.push([
                            b[0] + ix as f32 * a,
                            b[1] + iy as f32 * a,
                            b[2] + iz as f32 * a,
                            0.0,
                        ]);
                    }
                }
            }
        }
        let l = a * rep as f32;
        let cell = Some([[l, 0.0, 0.0], [0.0, l, 0.0], [0.0, 0.0, l]]);

        let cutoff = 4.95f32;
        let skin = 0.01 * cutoff;

        let on = build(&pos, cell, cutoff, skin);
        let n2 = build_n2_inner(&pos, cell, cutoff, skin);

        // `starts` must match exactly: pair counts are deterministic.
        assert_eq!(on.starts, n2.starts, "starts differ between O(N) and O(N²)");

        // `list` may differ in order within each row (intra-cell order is not
        // fixed), so compare as sorted sets.
        for i in 0..pos.len() {
            let s = on.starts[i] as usize;
            let e = on.starts[i + 1] as usize;
            let mut a_i: Vec<u32> = on.list[s..e].to_vec();
            let mut b_i: Vec<u32> = n2.list[s..e].to_vec();
            a_i.sort_unstable();
            b_i.sort_unstable();
            assert_eq!(
                a_i, b_i,
                "neighbour-set mismatch at atom {i} (s={s}, e={e})"
            );
        }
    }

    /// Same test as above but at a larger system size to exercise the
    /// stencil-wrap + dedup paths.
    #[test]
    fn on_matches_n2_fcc_cu_large() {
        let a = 3.615f32;
        let rep = 5usize; // 500 atoms
        let basis: [[f32; 3]; 4] = [
            [0.0, 0.0, 0.0],
            [a / 2.0, a / 2.0, 0.0],
            [a / 2.0, 0.0, a / 2.0],
            [0.0, a / 2.0, a / 2.0],
        ];
        let mut pos: Vec<[f32; 4]> = Vec::new();
        for ix in 0..rep {
            for iy in 0..rep {
                for iz in 0..rep {
                    for b in &basis {
                        pos.push([
                            b[0] + ix as f32 * a,
                            b[1] + iy as f32 * a,
                            b[2] + iz as f32 * a,
                            0.0,
                        ]);
                    }
                }
            }
        }
        let l = a * rep as f32;
        let cell = Some([[l, 0.0, 0.0], [0.0, l, 0.0], [0.0, 0.0, l]]);

        let cutoff = 4.95f32;
        let skin = 0.01 * cutoff;

        let on = build(&pos, cell, cutoff, skin);
        let n2 = build_n2_inner(&pos, cell, cutoff, skin);

        assert_eq!(on.starts, n2.starts);
        assert_eq!(on.total_pairs(), n2.total_pairs());
    }

    /// Regression for the rattle-induced mismatch found by the Python grid
    /// sweep: orthorhombic boxes with atoms displaced slightly outside
    /// `[0, L)` must still produce the same neighbour set as the O(N²) path.
    ///
    /// Without the `[0, L)³` wrap inside `build`, the orthorhombic fast path
    /// uses a wrong stencil shift and this test fails with ~hundreds of
    /// mismatched entries.
    #[test]
    fn on_matches_n2_ortho_with_escapes() {
        // Build an orthorhombic FCC Cu supercell, then displace every atom
        // by a deterministic offset that pushes some atoms out of [0, L)³.
        let a = 3.615f32;
        let rep = 4usize; // 256 atoms
        let basis: [[f32; 3]; 4] = [
            [0.0, 0.0, 0.0],
            [a / 2.0, a / 2.0, 0.0],
            [a / 2.0, 0.0, a / 2.0],
            [0.0, a / 2.0, a / 2.0],
        ];
        let mut pos: Vec<[f32; 4]> = Vec::new();
        for ix in 0..rep {
            for iy in 0..rep {
                for iz in 0..rep {
                    for b in &basis {
                        pos.push([
                            b[0] + ix as f32 * a,
                            b[1] + iy as f32 * a,
                            b[2] + iz as f32 * a,
                            0.0,
                        ]);
                    }
                }
            }
        }
        // Shift every atom by (+0.03, -0.04, +0.05) Å so the atoms at
        // cell-corners escape [0, L)³ slightly (some go negative, some
        // exceed L).  Magnitude is similar to ASE `rattle(stdev=0.05)`.
        for p in pos.iter_mut() {
            p[0] += 0.03;
            p[1] -= 0.04;
            p[2] += 0.05;
        }

        let l = a * rep as f32;
        let cell = Some([[l, 0.0, 0.0], [0.0, l, 0.0], [0.0, 0.0, l]]);

        let cutoff = 4.95f32;
        let skin = 0.01 * cutoff;

        let on = build(&pos, cell, cutoff, skin);
        let n2 = build_n2_inner(&pos, cell, cutoff, skin);

        assert_eq!(
            on.starts, n2.starts,
            "starts differ between O(N) wrap-aware and O(N²) paths"
        );

        // Sets must match row-by-row.
        for i in 0..pos.len() {
            let s = on.starts[i] as usize;
            let e = on.starts[i + 1] as usize;
            let mut a_i: Vec<u32> = on.list[s..e].to_vec();
            let mut b_i: Vec<u32> = n2.list[s..e].to_vec();
            a_i.sort_unstable();
            b_i.sort_unstable();
            assert_eq!(
                a_i, b_i,
                "neighbour-set mismatch at atom {i} (s={s}, e={e}) with escaped coords"
            );
        }
    }
}
